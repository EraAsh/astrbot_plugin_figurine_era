import asyncio
import base64
import functools
import io
import json
import random
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

import aiohttp
from PIL import Image as PILImage

from astrbot import logger
from astrbot.api.event import filter
from astrbot.api.star import Context, Star, register, StarTools
from astrbot.core import AstrBotConfig
from astrbot.core.message.components import At, Image, Reply, Plain
from astrbot.core.platform.astr_message_event import AstrMessageEvent


@register(
    "figurine_era",
    "EraAsh",
    "融合gcli2api和shoubanhua的图片生成插件",
    "1.0.3",
    "https://github.com/EraAsh/astrbot_plugin_figurine_era",
)
class FigurineProPlugin(Star):
    class ImageWorkflow:
        def __init__(self, proxy_url: str | None = None):
            if proxy_url: logger.info(f"ImageWorkflow 使用代理: {proxy_url}")
            self.session = aiohttp.ClientSession()
            self.proxy = proxy_url

        # --- 【修改】增加了详细的错误日志记录 ---
        async def _download_image(self, url: str) -> bytes | None:
            logger.info(f"正在尝试下载图片: {url}")
            try:
                async with self.session.get(url, proxy=self.proxy, timeout=30) as resp:
                    resp.raise_for_status()  # 这会对4xx/5xx状态码抛出异常
                    return await resp.read()
            except aiohttp.ClientResponseError as e:
                # 捕获特定的HTTP错误
                logger.error(f"图片下载失败: HTTP状态码 {e.status}, URL: {url}, 原因: {e.message}")
                return None
            except asyncio.TimeoutError:
                logger.error(f"图片下载失败: 请求超时 (30s), URL: {url}")
                return None
            except Exception as e:
                # 捕获其他所有异常 (如网络、SSL等)
                logger.error(f"图片下载失败: 发生未知错误, URL: {url}, 错误类型: {type(e).__name__}, 错误: {e}", exc_info=True)
                return None
        # --- 修改结束 ---

        async def _get_avatar(self, user_id: str) -> bytes | None:
            if not user_id.isdigit(): logger.warning(f"无法获取非 QQ 平台或无效 QQ 号 {user_id} 的头像。"); return None
            avatar_url = f"https://q1.qlogo.cn/g?b=qq&nk={user_id}&s=640"
            return await self._download_image(avatar_url)

        def _extract_first_frame_sync(self, raw: bytes) -> bytes:
            img_io = io.BytesIO(raw)
            try:
                with PILImage.open(img_io) as img:
                    if getattr(img, "is_animated", False):
                        logger.info("检测到动图, 将抽取第一帧进行生成")
                        img.seek(0)
                        first_frame = img.convert("RGBA")
                        out_io = io.BytesIO()
                        first_frame.save(out_io, format="PNG")
                        return out_io.getvalue()
            except Exception as e:
                logger.warning(f"抽取图片帧时发生错误, 将返回原始数据: {e}", exc_info=True)
            return raw

        async def _load_bytes(self, src: str) -> bytes | None:
            raw: bytes | None = None
            loop = asyncio.get_running_loop()
            if Path(src).is_file():
                raw = await loop.run_in_executor(None, Path(src).read_bytes)
            elif src.startswith("http"):
                raw = await self._download_image(src)
            elif src.startswith("base64://"):
                raw = await loop.run_in_executor(None, base64.b64decode, src[9:])
            if not raw: return None
            return await loop.run_in_executor(None, self._extract_first_frame_sync, raw)

        async def get_images(self, event: AstrMessageEvent) -> List[bytes]:
            img_bytes_list: List[bytes] = []
            at_user_ids: List[str] = []

            for seg in event.message_obj.message:
                if isinstance(seg, Reply) and seg.chain:
                    for s_chain in seg.chain:
                        if isinstance(s_chain, Image):
                            if s_chain.url and (img := await self._load_bytes(s_chain.url)):
                                img_bytes_list.append(img)
                            elif s_chain.file and (img := await self._load_bytes(s_chain.file)):
                                img_bytes_list.append(img)

            for seg in event.message_obj.message:
                if isinstance(seg, Image):
                    if seg.url and (img := await self._load_bytes(seg.url)):
                        img_bytes_list.append(img)
                    elif seg.file and (img := await self._load_bytes(seg.file)):
                        img_bytes_list.append(img)
                elif isinstance(seg, At):
                    at_user_ids.append(str(seg.qq))

            if img_bytes_list:
                return img_bytes_list

            if at_user_ids:
                for user_id in at_user_ids:
                    if avatar := await self._get_avatar(user_id):
                        img_bytes_list.append(avatar)
                return img_bytes_list

            if avatar := await self._get_avatar(event.get_sender_id()):
                img_bytes_list.append(avatar)

            return img_bytes_list

        async def terminate(self):
            if self.session and not self.session.closed: await self.session.close()

    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.conf = config
        
        # gcli2api 配置
        self.api_base = (config.get("gcli2api_base_url") or "http://127.0.0.1:7861").strip()
        self._GEN_PATH = "/v1beta/models/{model}:generateContent"
        self._STREAM_GEN_PATH = "/v1beta/models/{model}:streamGenerateContent"
        self.model_name = (config.get("model_name") or "gemini-2.5-flash-image").strip()
        self.gcli2api_api_password = (config.get("gcli2api_api_password") or "pwd").strip()
        self.max_retry_attempts = int(config.get("max_retry_attempts", 3))
        try:
            self.temperature = float(config.get("temperature", 1.0))
        except Exception:
            self.temperature = 1.0

        self.plugin_data_dir = StarTools.get_data_dir()
        self.user_counts_file = self.plugin_data_dir / "user_counts.json"
        self.user_counts: Dict[str, int] = {}
        self.group_counts_file = self.plugin_data_dir / "group_counts.json"
        self.group_counts: Dict[str, int] = {}
        self.user_checkin_file = self.plugin_data_dir / "user_checkin.json"
        self.user_checkin_data: Dict[str, str] = {}
        self.prompt_map: Dict[str, str] = {}
        self.iwf: Optional[FigurineProPlugin.ImageWorkflow] = None

    async def initialize(self):
        use_proxy = self.conf.get("use_proxy", False)
        proxy_url = self.conf.get("proxy_url") if use_proxy else None
        self.iwf = self.ImageWorkflow(proxy_url)
        await self._load_prompt_map()
        await self._load_user_counts()
        await self._load_group_counts()
        await self._load_user_checkin_data()
        logger.info("Figurine Era 插件已加载")
        if not self.gcli2api_api_password:
            logger.warning("Figurine Era: 未配置 gcli2api API 密码，插件可能无法工作")

    async def _load_prompt_map(self):
        logger.info("正在加载 prompts...")
        self.prompt_map.clear()
        prompt_list = self.conf.get("prompt_list", [])
        for item in prompt_list:
            try:
                if ":" in item:
                    key, value = item.split(":", 1)
                    self.prompt_map[key.strip()] = value.strip()
                else:
                    logger.warning(f"跳过格式错误的 prompt (缺少冒号): {item}")
            except ValueError:
                logger.warning(f"跳过格式错误的 prompt: {item}")
        logger.info(f"加载了 {len(self.prompt_map)} 个 prompts。")

    @filter.event_message_type(filter.EventMessageType.ALL, priority=1)
    async def on_figurine_request(self, event: AstrMessageEvent):
        if self.conf.get("prefix", True) and not event.is_at_or_wake_command:
            return
        text = event.message_str.strip()
        if not text: return
        cmd = text.split()[0].strip()
        bnn_command = self.conf.get("extra_prefix", "bnn")
        user_prompt = ""
        is_bnn = False
        if cmd == bnn_command:
            user_prompt = text.removeprefix(cmd).strip()
            is_bnn = True
            if not user_prompt: return
        elif cmd in self.prompt_map:
            user_prompt = self.prompt_map.get(cmd)
        else:
            return
        sender_id = event.get_sender_id()
        group_id = event.get_group_id()
        is_master = self.is_global_admin(event)
        if not is_master:
            if sender_id in self.conf.get("user_blacklist", []): return
            if group_id and group_id in self.conf.get("group_blacklist", []): return
            if self.conf.get("user_whitelist", []) and sender_id not in self.conf.get("user_whitelist", []): return
            if group_id and self.conf.get("group_whitelist", []) and group_id not in self.conf.get("group_whitelist", []): return
            user_count = self._get_user_count(sender_id)
            group_count = self._get_group_count(group_id) if group_id else 0
            user_limit_on = self.conf.get("enable_user_limit", True)
            group_limit_on = self.conf.get("enable_group_limit", False) and group_id
            has_group_count = not group_limit_on or group_count > 0
            has_user_count = not user_limit_on or user_count > 0
            if group_id:
                if not has_group_count and not has_user_count:
                    yield event.plain_result("❌ 本群次数与您的个人次数均已用尽。"); return
            elif not has_user_count:
                yield event.plain_result("❌ 您的使用次数已用完。"); return
        if not self.iwf or not (img_bytes_list := await self.iwf.get_images(event)):
            if not is_bnn:
                yield event.plain_result("请发送或引用一张图片。"); return
        images_to_process = []
        display_cmd = cmd
        if is_bnn:
            MAX_IMAGES = 5
            original_count = len(img_bytes_list)
            if original_count > MAX_IMAGES:
                images_to_process = img_bytes_list[:MAX_IMAGES]
                yield event.plain_result(f"🎨 检测到 {original_count} 张图片，已选取前 {MAX_IMAGES} 张…")
            else:
                images_to_process = img_bytes_list
            display_cmd = user_prompt[:10] + '...' if len(user_prompt) > 10 else user_prompt
            yield event.plain_result(f"🎨 检测到 {len(images_to_process)} 张图片，正在生成 [{display_cmd}]...")
        else:
            if not img_bytes_list:
                 yield event.plain_result("请发送或引用一张图片。"); return
            images_to_process = [img_bytes_list[0]]
            yield event.plain_result(f"🎨 收到请求，正在生成 [{cmd}]...")
        start_time = datetime.now()
        res = await self._call_api(images_to_process, user_prompt)
        elapsed = (datetime.now() - start_time).total_seconds()
        if isinstance(res, bytes):
            if not is_master:
                if self.conf.get("enable_group_limit", False) and group_id and self._get_group_count(group_id) > 0:
                    await self._decrease_group_count(group_id)
                elif self.conf.get("enable_user_limit", True) and self._get_user_count(sender_id) > 0:
                    await self._decrease_user_count(sender_id)
            caption_parts = [f"✅ 生成成功 ({elapsed:.2f}s)", f"预设: {display_cmd}"]
            if is_master:
                caption_parts.append("剩余次数: ∞")
            else:
                if self.conf.get("enable_user_limit", True): caption_parts.append(f"个人剩余: {self._get_user_count(sender_id)}")
                if self.conf.get("enable_group_limit", False) and group_id: caption_parts.append(f"本群剩余: {self._get_group_count(group_id)}")
            yield event.chain_result([Image.fromBytes(res), Plain(" | ".join(caption_parts))])
        else:
            yield event.plain_result(f"❌ 生成失败 ({elapsed:.2f}s)\n原因: {res}")
        event.stop_event()

    # The rest of the file remains the same...
    # ... (omitted for brevity, please keep the rest of your original file from here)
    @filter.command("lm添加", aliases={"lma"}, prefix_optional=True)
    async def add_lm_prompt(self, event: AstrMessageEvent):
        if not self.is_global_admin(event): return
        # 提取命令后的内容
        text = event.message_str.strip()
        parts = text.split(None, 1)
        raw = parts[1].strip() if len(parts) > 1 else ""
        if ":" not in raw:
            yield event.plain_result('格式错误, 正确示例:\n#lm添加 姿势表:为这幅图创建一个姿势表, 摆出各种姿势')
            return

        key, new_value = map(str.strip, raw.split(":", 1))
        prompt_list = self.conf.get("prompt_list", [])
        found = False
        for idx, item in enumerate(prompt_list):
            if item.strip().startswith(key + ":"):
                prompt_list[idx] = f"{key}:{new_value}"
                found = True
                break
        if not found: prompt_list.append(f"{key}:{new_value}")

        await self.conf.set("prompt_list", prompt_list)
        await self._load_prompt_map()
        yield event.plain_result(f"已保存LM生图提示语:\n{key}:{new_value}")

    @filter.command("生图", prefix_optional=True)
    async def cmd_generate(self, event: AstrMessageEvent, *, prompt: str):
        """生图：/生图 <提示词>"""
        sender_id = event.get_sender_id()
        group_id = event.get_group_id()
        is_master = self.is_global_admin(event)
        if not is_master:
            if sender_id in self.conf.get("user_blacklist", []): return
            if group_id and group_id in self.conf.get("group_blacklist", []): return
            if self.conf.get("user_whitelist", []) and sender_id not in self.conf.get("user_whitelist", []): return
            if group_id and self.conf.get("group_whitelist", []) and group_id not in self.conf.get("group_whitelist", []): return
            user_count = self._get_user_count(sender_id)
            group_count = self._get_group_count(group_id) if group_id else 0
            user_limit_on = self.conf.get("enable_user_limit", True)
            group_limit_on = self.conf.get("enable_group_limit", False) and group_id
            has_group_count = not group_limit_on or group_count > 0
            has_user_count = not user_limit_on or user_count > 0
            if group_id:
                if not has_group_count and not has_user_count:
                    yield event.plain_result("❌ 本群次数与您的个人次数均已用尽。"); return
            elif not has_user_count:
                yield event.plain_result("❌ 您的使用次数已用完。"); return
        
        yield event.plain_result(f"🎨 正在生成图片...")
        start_time = datetime.now()
        res = await self._call_api([], prompt)
        elapsed = (datetime.now() - start_time).total_seconds()
        if isinstance(res, bytes):
            if not is_master:
                if self.conf.get("enable_group_limit", False) and group_id and self._get_group_count(group_id) > 0:
                    await self._decrease_group_count(group_id)
                elif self.conf.get("enable_user_limit", True) and self._get_user_count(sender_id) > 0:
                    await self._decrease_user_count(sender_id)
            caption_parts = [f"✅ 生成成功 ({elapsed:.2f}s)", f"提示词: {prompt[:20]}..." if len(prompt) > 20 else f"提示词: {prompt}"]
            if is_master:
                caption_parts.append("剩余次数: ∞")
            else:
                if self.conf.get("enable_user_limit", True): caption_parts.append(f"个人剩余: {self._get_user_count(sender_id)}")
                if self.conf.get("enable_group_limit", False) and group_id: caption_parts.append(f"本群剩余: {self._get_group_count(group_id)}")
            yield event.chain_result([Image.fromBytes(res), Plain(" | ".join(caption_parts))])
        else:
            yield event.plain_result(f"❌ 生成失败 ({elapsed:.2f}s)\n原因: {res}")
        event.stop_event()

    @filter.command("改图", prefix_optional=True)
    async def cmd_edit(self, event: AstrMessageEvent, *, prompt: str):
        """改图（需携带/引用图片）：/改图 <提示词>"""
        sender_id = event.get_sender_id()
        group_id = event.get_group_id()
        is_master = self.is_global_admin(event)
        if not is_master:
            if sender_id in self.conf.get("user_blacklist", []): return
            if group_id and group_id in self.conf.get("group_blacklist", []): return
            if self.conf.get("user_whitelist", []) and sender_id not in self.conf.get("user_whitelist", []): return
            if group_id and self.conf.get("group_whitelist", []) and group_id not in self.conf.get("group_whitelist", []): return
            user_count = self._get_user_count(sender_id)
            group_count = self._get_group_count(group_id) if group_id else 0
            user_limit_on = self.conf.get("enable_user_limit", True)
            group_limit_on = self.conf.get("enable_group_limit", False) and group_id
            has_group_count = not group_limit_on or group_count > 0
            has_user_count = not user_limit_on or user_count > 0
            if group_id:
                if not has_group_count and not has_user_count:
                    yield event.plain_result("❌ 本群次数与您的个人次数均已用尽。"); return
            elif not has_user_count:
                yield event.plain_result("❌ 您的使用次数已用完。"); return
        
        # 检查是否包含图片
        if not self.iwf or not (img_bytes_list := await self.iwf.get_images(event)):
            yield event.plain_result("请先携带或引用一张图片后，再使用：/改图 <提示词>"); return
        
        yield event.plain_result(f"🎨 正在修改图片...")
        start_time = datetime.now()
        res = await self._call_api(img_bytes_list, prompt)
        elapsed = (datetime.now() - start_time).total_seconds()
        if isinstance(res, bytes):
            if not is_master:
                if self.conf.get("enable_group_limit", False) and group_id and self._get_group_count(group_id) > 0:
                    await self._decrease_group_count(group_id)
                elif self.conf.get("enable_user_limit", True) and self._get_user_count(sender_id) > 0:
                    await self._decrease_user_count(sender_id)
            caption_parts = [f"✅ 修改成功 ({elapsed:.2f}s)", f"提示词: {prompt[:20]}..." if len(prompt) > 20 else f"提示词: {prompt}"]
            if is_master:
                caption_parts.append("剩余次数: ∞")
            else:
                if self.conf.get("enable_user_limit", True): caption_parts.append(f"个人剩余: {self._get_user_count(sender_id)}")
                if self.conf.get("enable_group_limit", False) and group_id: caption_parts.append(f"本群剩余: {self._get_group_count(group_id)}")
            yield event.chain_result([Image.fromBytes(res), Plain(" | ".join(caption_parts))])
        else:
            yield event.plain_result(f"❌ 修改失败 ({elapsed:.2f}s)\n原因: {res}")
        event.stop_event()

    @filter.command("lm帮助", aliases={"lmh", "手办化帮助"}, prefix_optional=True)
    async def on_prompt_help(self, event: AstrMessageEvent):
        # 提取命令后的关键词
        text = event.message_str.strip()
        parts = text.split(None, 1)
        keyword = parts[1].strip() if len(parts) > 1 else ""
        if not keyword:
            msg = "可用的生图提示词: \n"
            msg += "、".join(self.prompt_map.keys())
            yield event.plain_result(msg)
            return

        prompt = self.prompt_map.get(keyword)
        if not prompt:
            yield event.plain_result("未找到此提示词")
            return
        yield event.plain_result(f"{keyword}:\n{prompt}")

    def is_global_admin(self, event: AstrMessageEvent) -> bool:
        admin_ids = self.context.get_config().get("admins_id", [])
        return event.get_sender_id() in admin_ids

    async def _load_user_counts(self):
        if not self.user_counts_file.exists(): self.user_counts = {}; return
        loop = asyncio.get_running_loop()
        try:
            content = await loop.run_in_executor(None, self.user_counts_file.read_text, "utf-8")
            data = await loop.run_in_executor(None, json.loads, content)
            if isinstance(data, dict): self.user_counts = {str(k): v for k, v in data.items()}
        except Exception as e:
            logger.error(f"加载用户次数文件时发生错误: {e}", exc_info=True); self.user_counts = {}

    async def _save_user_counts(self):
        loop = asyncio.get_running_loop()
        try:
            json_data = await loop.run_in_executor(None, functools.partial(json.dumps, self.user_counts, ensure_ascii=False, indent=4))
            await loop.run_in_executor(None, self.user_counts_file.write_text, json_data, "utf-8")
        except Exception as e:
            logger.error(f"保存用户次数文件时发生错误: {e}", exc_info=True)

    def _get_user_count(self, user_id: str) -> int:
        return self.user_counts.get(str(user_id), 0)

    async def _decrease_user_count(self, user_id: str):
        user_id_str = str(user_id)
        count = self._get_user_count(user_id_str)
        if count > 0: self.user_counts[user_id_str] = count - 1; await self._save_user_counts()

    async def _load_group_counts(self):
        if not self.group_counts_file.exists(): self.group_counts = {}; return
        loop = asyncio.get_running_loop()
        try:
            content = await loop.run_in_executor(None, self.group_counts_file.read_text, "utf-8")
            data = await loop.run_in_executor(None, json.loads, content)
            if isinstance(data, dict): self.group_counts = {str(k): v for k, v in data.items()}
        except Exception as e:
            logger.error(f"加载群组次数文件时发生错误: {e}", exc_info=True); self.group_counts = {}

    async def _save_group_counts(self):
        loop = asyncio.get_running_loop()
        try:
            json_data = await loop.run_in_executor(None, functools.partial(json.dumps, self.group_counts, ensure_ascii=False, indent=4))
            await loop.run_in_executor(None, self.group_counts_file.write_text, json_data, "utf-8")
        except Exception as e:
            logger.error(f"保存群组次数文件时发生错误: {e}", exc_info=True)

    def _get_group_count(self, group_id: str) -> int:
        return self.group_counts.get(str(group_id), 0)

    async def _decrease_group_count(self, group_id: str):
        group_id_str = str(group_id)
        count = self._get_group_count(group_id_str)
        if count > 0: self.group_counts[group_id_str] = count - 1; await self._save_group_counts()

    async def _load_user_checkin_data(self):
        if not self.user_checkin_file.exists(): self.user_checkin_data = {}; return
        loop = asyncio.get_running_loop()
        try:
            content = await loop.run_in_executor(None, self.user_checkin_file.read_text, "utf-8")
            data = await loop.run_in_executor(None, json.loads, content)
            if isinstance(data, dict): self.user_checkin_data = {str(k): v for k, v in data.items()}
        except Exception as e:
            logger.error(f"加载用户签到文件时发生错误: {e}", exc_info=True); self.user_checkin_data = {}

    async def _save_user_checkin_data(self):
        loop = asyncio.get_running_loop()
        try:
            json_data = await loop.run_in_executor(None, functools.partial(json.dumps, self.user_checkin_data, ensure_ascii=False, indent=4))
            await loop.run_in_executor(None, self.user_checkin_file.write_text, json_data, "utf-8")
        except Exception as e:
            logger.error(f"保存用户签到文件时发生错误: {e}", exc_info=True)

    @filter.command("手办化签到", prefix_optional=True)
    async def on_checkin(self, event: AstrMessageEvent):
        if not self.conf.get("enable_checkin", False):
            yield event.plain_result("📅 本机器人未开启签到功能。")
            return
        user_id = event.get_sender_id()
        today_str = datetime.now().strftime("%Y-%m-%d")
        if self.user_checkin_data.get(user_id) == today_str:
            yield event.plain_result(f"您今天已经签到过了。\n剩余次数: {self._get_user_count(user_id)}")
            return
        reward = 0
        if str(self.conf.get("enable_random_checkin", False)).lower() == 'true':
            max_reward = max(1, int(self.conf.get("checkin_random_reward_max", 5)))
            reward = random.randint(1, max_reward)
        else:
            reward = int(self.conf.get("checkin_fixed_reward", 3))
        current_count = self._get_user_count(user_id)
        new_count = current_count + reward
        self.user_counts[user_id] = new_count
        await self._save_user_counts()
        self.user_checkin_data[user_id] = today_str
        await self._save_user_checkin_data()
        yield event.plain_result(f"🎉 签到成功！获得 {reward} 次，当前剩余: {new_count} 次。")

    @filter.command("手办化开启签到", prefix_optional=True)
    async def on_enable_checkin(self, event: AstrMessageEvent):
        if not self.is_global_admin(event): return
        await self.conf.set("enable_checkin", True)
        yield event.plain_result("✅ 签到功能已开启")

    @filter.command("手办化关闭签到", prefix_optional=True)
    async def on_disable_checkin(self, event: AstrMessageEvent):
        if not self.is_global_admin(event): return
        await self.conf.set("enable_checkin", False)
        yield event.plain_result("✅ 签到功能已关闭")

    @filter.command("手办化增加用户次数", prefix_optional=True)
    async def on_add_user_counts(self, event: AstrMessageEvent):
        if not self.is_global_admin(event): return
        cmd_text = event.message_str.strip()
        at_seg = next((s for s in event.message_obj.message if isinstance(s, At)), None)
        target_qq, count = None, 0
        if at_seg:
            target_qq = str(at_seg.qq)
            match = re.search(r"(\d+)\s*$", cmd_text)
            if match: count = int(match.group(1))
        else:
            match = re.search(r"(\d+)\s+(\d+)", cmd_text)
            if match: target_qq, count = match.group(1), int(match.group(2))
        if not target_qq or count <= 0:
            yield event.plain_result('格式错误:\n#手办化增加用户次数 @用户 <次数>\n或 #手办化增加用户次数 <QQ号> <次数>')
            return
        current_count = self._get_user_count(target_qq)
        self.user_counts[str(target_qq)] = current_count + count
        await self._save_user_counts()
        yield event.plain_result(f"✅ 已为用户 {target_qq} 增加 {count} 次，TA当前剩余 {current_count + count} 次。")

    @filter.command("手办化增加群组次数", prefix_optional=True)
    async def on_add_group_counts(self, event: AstrMessageEvent):
        if not self.is_global_admin(event): return
        match = re.search(r"(\d+)\s+(\d+)", event.message_str.strip())
        if not match:
            yield event.plain_result('格式错误: #手办化增加群组次数 <群号> <次数>')
            return
        target_group, count = match.group(1), int(match.group(2))
        current_count = self._get_group_count(target_group)
        self.group_counts[str(target_group)] = current_count + count
        await self._save_group_counts()
        yield event.plain_result(f"✅ 已为群组 {target_group} 增加 {count} 次，该群当前剩余 {current_count + count} 次。")

    @filter.command("手办化查询次数", prefix_optional=True)
    async def on_query_counts(self, event: AstrMessageEvent):
        user_id_to_query = event.get_sender_id()
        if self.is_global_admin(event):
            at_seg = next((s for s in event.message_obj.message if isinstance(s, At)), None)
            if at_seg: user_id_to_query = str(at_seg.qq)
            else:
                match = re.search(r"(\d+)", event.message_str)
                if match: user_id_to_query = match.group(1)
        user_count = self._get_user_count(user_id_to_query)
        reply_msg = f"用户 {user_id_to_query} 个人剩余次数为: {user_count}"
        if user_id_to_query == event.get_sender_id(): reply_msg = f"您好，您当前个人剩余次数为: {user_count}"
        if group_id := event.get_group_id(): reply_msg += f"\n本群共享剩余次数为: {self._get_group_count(group_id)}"
        yield event.plain_result(reply_msg)

    @filter.command("手办化添加key", prefix_optional=True)
    async def on_add_key(self, event: AstrMessageEvent):
        if not self.is_global_admin(event): return
        new_keys = event.message_str.strip().split()
        if not new_keys: yield event.plain_result("格式错误，请提供要添加的Key。"); return
        api_keys = self.conf.get("api_keys", [])
        added_keys = [key for key in new_keys if key not in api_keys]
        api_keys.extend(added_keys)
        await self.conf.set("api_keys", api_keys)
        yield event.plain_result(f"✅ 操作完成，新增 {len(added_keys)} 个Key，当前共 {len(api_keys)} 个。")

    @filter.command("手办化key列表", prefix_optional=True)
    async def on_list_keys(self, event: AstrMessageEvent):
        if not self.is_global_admin(event): return
        api_keys = self.conf.get("api_keys", [])
        if not api_keys: yield event.plain_result("📝 暂未配置任何 API Key。"); return
        key_list_str = "\n".join(f"{i + 1}. {key[:8]}...{key[-4:]}" for i, key in enumerate(api_keys))
        yield event.plain_result(f"🔑 API Key 列表:\n{key_list_str}")

    @filter.command("手办化删除key", prefix_optional=True)
    async def on_delete_key(self, event: AstrMessageEvent):
        if not self.is_global_admin(event): return
        param = event.message_str.strip()
        api_keys = self.conf.get("api_keys", [])
        if param.lower() == "all":
            await self.conf.set("api_keys", [])
            yield event.plain_result(f"✅ 已删除全部 {len(api_keys)} 个 Key。")
        elif param.isdigit() and 1 <= int(param) <= len(api_keys):
            removed_key = api_keys.pop(int(param) - 1)
            await self.conf.set("api_keys", api_keys)
            yield event.plain_result(f"✅ 已删除 Key: {removed_key[:8]}...")
        else:
            yield event.plain_result("格式错误，请使用 #手办化删除key <序号|all>")

    async def _call_api(self, image_bytes_list: List[bytes], prompt: str) -> bytes | str:
        """使用 gcli2api 调用 Gemini API"""
        from .utils.gemini_images_api import generate_or_edit_image_gemini, generate_or_edit_image_gemini_stream
        
        if not self.gcli2api_api_password:
            return "gcli2api API 密码未配置"

        input_images_b64 = [base64.b64encode(img).decode("utf-8") for img in image_bytes_list]

        try:
            # 优先使用流式接口
            _, image_path = await generate_or_edit_image_gemini_stream(
                prompt=prompt,
                api_keys=[self.gcli2api_api_password],
                model=self.model_name,
                api_base=self.api_base,
                endpoint_path=self._STREAM_GEN_PATH,
                input_images_b64=input_images_b64,
                max_retry_attempts=self.max_retry_attempts,
                temperature=self.temperature,
            )
            # 流式失败则回退非流式
            if not image_path:
                logger.warning("流式接口未返回图片，尝试使用非流式接口...")
                _, image_path = await generate_or_edit_image_gemini(
                    prompt=prompt,
                    api_keys=[self.gcli2api_api_password],
                    model=self.model_name,
                    api_base=self.api_base,
                    endpoint_path=self._GEN_PATH,
                    input_images_b64=input_images_b64,
                    max_retry_attempts=self.max_retry_attempts,
                    temperature=self.temperature,
                )

            if image_path and Path(image_path).exists():
                return await asyncio.get_running_loop().run_in_executor(None, Path(image_path).read_bytes)
            else:
                return "生成了图片但未能找到文件"

        except Exception as e:
            logger.error(f"调用 gcli2api 时发生未知错误: {e}", exc_info=True)
            return f"发生未知错误: {e}"

    async def terminate(self):
        if self.iwf: await self.iwf.terminate()
        logger.info("[FigurinePro] 插件已终止")