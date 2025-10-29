import asyncio
import base64
import json
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import httpx
from astrbot.api import logger


class _State:
    def __init__(self):
        self.api_key_index = 0
        self._lock = asyncio.Lock()

    async def get_next_key(self, keys: List[str]) -> str:
        async with self._lock:
            if not keys:
                raise ValueError("API key list is empty")
            key = keys[self.api_key_index % len(keys)]
            return key

    async def rotate(self, keys: List[str]):
        async with self._lock:
            if keys:
                self.api_key_index = (self.api_key_index + 1) % len(keys)


_state = _State()


async def _save_bytes(content: bytes, suffix: str = "png") -> str:
    # 优化：使用更高效的文件写入方式
    try:
        plugin_root = Path(__file__).parent.parent
        images_dir = plugin_root / "images"
        
        # 确保目录存在，包括所有父目录
        images_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"图片保存目录: {images_dir}, 存在: {images_dir.exists()}")
        
        # 优化：使用更高效的文件名生成
        timestamp = int(time.time() * 1000)
        uid = uuid.uuid4().hex[:8]
        file_path = images_dir / f"gemini_{timestamp}_{uid}.{suffix}"
        
        # 优化：直接写入文件，避免额外的中间步骤
        file_path.write_bytes(content)
        
        # 验证文件是否成功写入
        if file_path.exists() and file_path.stat().st_size > 0:
            logger.info(f"图片保存成功: {file_path}, 大小: {file_path.stat().st_size} 字节")
            return str(file_path)
        else:
            logger.error(f"图片保存失败: 文件不存在或为空 - {file_path}")
            raise RuntimeError(f"图片保存失败: 文件不存在或为空 - {file_path}")
    except Exception as e:
        logger.error(f"保存图片文件失败: {e}")
        raise


async def _decode_and_save_base64(data_b64: str, mime: Optional[str]) -> str:
    # 优化：使用更高效的base64解码和文件保存
    try:
        # strip data URL if present
        if data_b64.startswith("data:"):
            try:
                header, b64 = data_b64.split(",", 1)
                data_b64 = b64
            except Exception:
                pass
        
        # 优化：直接解码，避免不必要的中间变量
        raw = base64.b64decode(data_b64)
        
        # 检查解码后的数据是否为空
        if not raw:
            raise ValueError("Base64解码结果为空")
        
        # 优化：使用更高效的后缀名判断
        suffix = "png"
        if mime:
            mime_lower = mime.lower()
            if "jpeg" in mime_lower or "jpg" in mime_lower:
                suffix = "jpg"
            elif "webp" in mime_lower:
                suffix = "webp"
            elif "png" in mime_lower:
                suffix = "png"
        
        file_path = await _save_bytes(raw, suffix)
        logger.info(f"Base64图片解码保存成功: {file_path}")
        return file_path
    except Exception as e:
        logger.error(f"解码和保存base64图片失败: {e}")
        raise


def _build_url(api_base: str, path: str, api_key: str, model: str, append_key_query: bool, extra_query: Optional[Dict[str, str]] = None) -> str:
    base = api_base.rstrip("/")
    # 支持 {model} 占位符
    if "{model}" in path:
        path = path.replace("{model}", model)
    path = path if path.startswith("/") else f"/{path}"
    # 追加查询参数
    query_items = []
    if append_key_query and api_key:
        query_items.append(("key", api_key))
    if extra_query:
        for k, v in extra_query.items():
            if v is not None:
                query_items.append((k, str(v)))
    if query_items:
        sep = "&" if "?" in base + path else "?"
        qs = "&".join([f"{k}={v}" for k, v in query_items])
        return f"{base}{path}{sep}{qs}"
    else:
        return f"{base}{path}"


async def generate_or_edit_image_gemini(
    prompt: str,
    api_keys: List[str],
    model: str,
    api_base: str,
    endpoint_path: str,
    input_images_b64: Optional[List[str]] = None,
    max_retry_attempts: int = 3,
    timeout_seconds: int = 60,
    temperature: Optional[float] = None,
) -> Tuple[Optional[str], Optional[str]]:
    """
    使用 gcli2api 的 generateContent 接口生图/改图（通过 parts 注入图片）。

    返回 (image_url, image_path)。image_url 可能为 None（当只返回内联 base64 时）。
    """
    if isinstance(api_keys, str):
        api_keys = [api_keys]

    if not api_keys:
        logger.error("未提供 API 密码/口令")
        return None, None

    # 允许传入参考图片进行编辑或条件生成
    input_images_b64 = input_images_b64 or []

    for key_attempt in range(len(api_keys)):
        current_key = await _state.get_next_key(api_keys)

        for attempt in range(max_retry_attempts):
            if attempt > 0:
                # 指数退避
                await asyncio.sleep(min(2 ** attempt, 10))

            # 允许通过 ?key= 传参，并附带头部，适配 gcli2api 的灵活鉴权
            use_query_key = True if current_key else False
            url = _build_url(api_base, endpoint_path, current_key, model, use_query_key, None)
            headers = {"Content-Type": "application/json"}
            if current_key:
                headers["x-goog-api-key"] = current_key
                headers["Authorization"] = f"Bearer {current_key}"

            # 构造 generateContent 风格负载
            parts = []
            parts.append({"text": prompt})
            for b64 in input_images_b64:
                mime_type = "image/png"
                if b64.startswith("data:"):
                    try:
                        header, b64data = b64.split(",", 1)
                        if header.startswith("data:") and ";base64" in header:
                            mime_type = header[5: header.find(";")]
                        b64 = b64data
                    except Exception:
                        pass
                parts.append({
                    "inlineData": {
                        "mimeType": mime_type,
                        "data": b64
                    }
                })

            payload: Dict = {
                "contents": [
                    {
                        "role": "user",
                        "parts": parts
                    }
                ]
            }
            # 附加温度（仅传入 temperature，不包含 topP 等）
            if temperature is not None:
                gen = payload.get("generationConfig", {})
                gen2 = payload.get("generation_config", {})
                gen["temperature"] = temperature
                gen2["temperature"] = temperature
                payload["generationConfig"] = gen
                payload["generation_config"] = gen2

            try:
                async with httpx.AsyncClient(timeout=timeout_seconds) as client:
                    resp = await client.post(url, headers=headers, json=payload)
                    if resp.status_code == 429:
                        logger.warning("Gemini API 限流，稍后重试")
                        continue

                    if resp.status_code >= 500:
                        logger.warning(f"Gemini API 服务端错误 {resp.status_code}")
                        continue

                    if resp.status_code != 200:
                        # 4xx 明确错误不再重试当前密钥
                        try:
                            err = resp.json()
                        except Exception:
                            err = {"text": resp.text}
                        logger.error(f"Gemini API 调用失败 {resp.status_code}: {err}")
                        break

                    data = resp.json()

                    # 解析 generateContent 返回结构
                    image_path = None
                    image_url = None

                    if isinstance(data, dict):
                        if data.get("error"):
                            logger.error(f"Gemini API 返回错误: {data['error']}")
                            return None, None
                        cands = data.get("candidates") or []
                        if cands:
                            parts = (cands[0].get("content") or {}).get("parts") or []
                            for p in parts:
                                inline = p.get("inline_data") or p.get("inlineData")
                                if inline and inline.get("data"):
                                    image_path = await _decode_and_save_base64(
                                        inline.get("data"), inline.get("mime_type") or inline.get("mimeType")
                                    )
                                    break

                    if image_path:
                        return image_url, image_path

                    logger.error("Gemini API 响应未包含可解析的图片数据")
                    return None, None

            except (httpx.ConnectError, httpx.ReadTimeout) as e:
                logger.error(f"网络错误: {e}")
                continue
            except Exception as e:
                logger.error(f"调用 Gemini API 异常: {e}")
                continue

        # 尝试下一个密钥
        await _state.rotate(api_keys)

    logger.error("所有 API 密钥与重试次数用尽，生成失败")
    return None, None


async def _parse_generate_content_json_for_image(data: dict) -> Tuple[Optional[str], Optional[str]]:
    """从 generateContent 风格响应解析 inlineData 图片，返回 (url, path)"""
    image_path = None
    image_url = None
    try:
        # 优化：使用更高效的路径查找
        candidates = data.get("candidates", [])
        if not candidates:
            return image_url, image_path

        # 只处理第一个候选，因为通常只有一个结果
        candidate = candidates[0]
        content = candidate.get("content", {})
        parts = content.get("parts", [])
        
        for part in parts:
            # 优化：统一检查两种可能的字段名
            inline = part.get("inline_data") or part.get("inlineData")
            if inline and inline.get("data"):
                image_path = await _decode_and_save_base64(
                    inline.get("data"), inline.get("mime_type") or inline.get("mimeType")
                )
                # 优化：找到图片后立即返回，避免不必要的循环
                break
    except Exception as e:
        logger.warning(f"解析 generateContent 响应失败: {e}")
    return image_url, image_path


async def generate_or_edit_image_gemini_stream(
    prompt: str,
    api_keys: List[str],
    model: str,
    api_base: str,
    endpoint_path: str,
    input_images_b64: Optional[List[str]] = None,
    max_retry_attempts: int = 3,
    timeout_seconds: int = 60,
    temperature: Optional[float] = None,
) -> Tuple[Optional[str], Optional[str]]:
    """
    调用流式接口（streamGenerateContent）。收到第一帧图片即返回。
    失败时返回 (None, None)。
    """
    if isinstance(api_keys, str):
        api_keys = [api_keys]
    if not api_keys:
        logger.error("未提供 API 密钥/口令")
        return None, None

    input_images_b64 = input_images_b64 or []

    for key_attempt in range(len(api_keys)):
        current_key = await _state.get_next_key(api_keys)

        for attempt in range(max_retry_attempts):
            if attempt > 0:
                await asyncio.sleep(min(2 ** attempt, 10))

            use_query_key = True if current_key else False
            url = _build_url(api_base, endpoint_path, current_key, model, use_query_key, {"alt": "sse"})
            headers = {"Content-Type": "application/json"}
            if current_key:
                headers["x-goog-api-key"] = current_key
                headers["Authorization"] = f"Bearer {current_key}"

            # 构造 generateContent 风格负载
            parts = [{"text": prompt}]
            for b64 in input_images_b64:
                mime_type = "image/png"
                if b64.startswith("data:"):
                    try:
                        header, b64data = b64.split(",", 1)
                        if header.startswith("data:") and ";base64" in header:
                            mime_type = header[5: header.find(";")]
                        b64 = b64data
                    except Exception:
                        pass
                parts.append({"inlineData": {"mimeType": mime_type, "data": b64}})

            payload: Dict = {
                "contents": [{"role": "user", "parts": parts}]
            }
            if temperature is not None:
                gen = payload.get("generationConfig", {})
                gen2 = payload.get("generation_config", {})
                gen["temperature"] = temperature
                gen2["temperature"] = temperature
                payload["generationConfig"] = gen
                payload["generation_config"] = gen2

            try:
                # 优化：减少超时时间，提高响应速度
                async with httpx.AsyncClient(timeout=min(timeout_seconds, 30)) as client:
                    async with client.stream("POST", url, headers=headers, json=payload) as resp:
                        # 非 200 直接解析一次错误文本
                        if resp.status_code != 200:
                            try:
                                err_text = await resp.aread()
                                logger.error(f"流式接口状态 {resp.status_code}: {err_text[:200]}")
                            except Exception:
                                logger.error(f"流式接口状态 {resp.status_code}")
                            break

                        ctype = resp.headers.get("content-type", "")
                        # SSE: text/event-stream; charset=utf-8
                        if "text/event-stream" in ctype:
                            # 优化：增加缓冲区大小，提高读取效率
                            async for line in resp.aiter_lines():
                                if not line:
                                    continue
                                if line.startswith(":"):
                                    # SSE 注释行
                                    continue
                                if line.startswith("data:"):
                                    data_str = line[5:].strip()
                                    if data_str in ("[DONE]", "DONE"):
                                        break
                                    try:
                                        data_json = json.loads(data_str)
                                    except Exception:
                                        continue
                                    # 错误帧
                                    if isinstance(data_json, dict) and data_json.get("error"):
                                        logger.error(f"流式错误帧: {data_json.get('error')}")
                                        break
                                    # 优化：一旦找到图片数据立即返回
                                    image_url, image_path = await _parse_generate_content_json_for_image(data_json)
                                    if image_path:
                                        logger.info(f"流式接口成功返回图片: {image_path}")
                                        return image_url, image_path
                        else:
                            # 非 SSE：尝试按 chunk/换行分割 JSON
                            buf = b""
                            # 优化：增加缓冲区大小
                            async for chunk in resp.aiter_bytes(chunk_size=8192):
                                buf += chunk
                                # 尝试按换行切分
                                while b"\n" in buf:
                                    line, buf = buf.split(b"\n", 1)
                                    line = line.strip()
                                    if not line:
                                        continue
                                    try:
                                        data_json = json.loads(line.decode("utf-8", errors="ignore"))
                                    except Exception:
                                        continue
                                    if isinstance(data_json, dict) and data_json.get("error"):
                                        logger.error(f"流式分块错误: {data_json.get('error')}")
                                        break
                                    # 优化：一旦找到图片数据立即返回
                                    image_url, image_path = await _parse_generate_content_json_for_image(data_json)
                                    if image_path:
                                        logger.info(f"流式接口成功返回图片: {image_path}")
                                        return image_url, image_path
            except (httpx.ConnectError, httpx.ReadTimeout) as e:
                logger.error(f"流式网络错误: {e}")
                continue
            except Exception as e:
                logger.error(f"流式调用异常: {e}")
                continue

        await _state.rotate(api_keys)

    logger.error("流式接口未返回图片数据")
    return None, None