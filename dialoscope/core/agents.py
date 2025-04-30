import abc
import os
from typing import List, Dict, Any

import openai

# 默认模型配置
DEFAULT_MODEL = "gpt-4o"
DEFAULT_MAX_TOKENS = 1024
DEFAULT_TEMPERATURE = 0.7

class Agent(abc.ABC):
    """AI 辩论角色的基类"""

    def __init__(self, role: str, model: str = DEFAULT_MODEL, **kwargs):
        self.role = role
        self.model = model
        # 从环境变量初始化 OpenAI 客户端
        try:
            self.client = openai.OpenAI(
                api_key=os.environ["OPENAI_API_KEY"],
                base_url=os.environ.get("OPENAI_BASE_URL") # base_url 是可选的
            )
        except KeyError as e:
            raise ValueError(f"错误：缺少环境变量 {e}。请确保设置了 OPENAI_API_KEY。") from e
        except Exception as e:
            raise RuntimeError(f"初始化 OpenAI 客户端时出错: {e}") from e

    def _format_history_for_llm(self, history: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """将内部历史记录格式化为 OpenAI API 需要的格式"""
        messages = []
        for entry in history:
            role_map = {
                "正方": "user", # 将正反方视为 user，让 LLM 扮演下一个角色
                "反方": "user",
                "评审": "assistant", # 评审的总结视为 assistant 回复
                "系统": "system"
            }
            # 如果 history 中的角色不在映射中，暂时跳过或用默认值
            api_role = role_map.get(entry["role"], "user")
            # 确保 content 是字符串
            content = str(entry.get("content", ""))
            messages.append({"role": api_role, "content": content})
        return messages

    @abc.abstractmethod
    def generate_response(
        self, 
        proposition: str, 
        history: List[Dict[str, Any]], 
        current_round: int | None = None, 
        total_rounds: int | None = None
        ) -> str:
        """
        根据辩论历史生成回应。
        Args:
            proposition: 当前辩题。
            history: 包含过去发言的列表。
            current_round: 当前轮数 (可选, 主要给 Judge 用)。
            total_rounds: 总轮数 (可选, 主要给 Judge 用)。
        Returns:
            当前角色的回应字符串。
        """
        pass

    def _call_llm(self, messages: List[Dict[str, str]], max_tokens: int = DEFAULT_MAX_TOKENS, temperature: float = DEFAULT_TEMPERATURE) -> str:
        """调用 LLM API"""
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            response = completion.choices[0].message.content
            if response is None:
                return "(模型未返回有效内容)"
            return response.strip()
        except openai.APIError as e:
            print(f"OpenAI API 返回错误: {e}")
            return f"(调用 API 时出错: {e})"
        except Exception as e:
            print(f"调用 LLM 时发生未知错误: {e}")
            return f"(内部错误: {e})"


class ProponentAgent(Agent):
    """正方 AI"""
    def __init__(self, **kwargs):
        super().__init__(role="正方", **kwargs)

    def generate_response(self, proposition: str, history: List[Dict[str, Any]], **kwargs) -> str:
        system_prompt = f"你现在是一个辩论中的正方。辩题是：'{proposition}'。你的任务是提出强有力的论点来支持这个命题，并反驳反方的观点。请根据当前的辩论历史，给出你的下一轮发言。请直接陈述你的观点和论据，不要说 '我是正方' 或进行角色扮演的确认。"
        messages = self._format_history_for_llm(history)
        messages.insert(0, {"role": "system", "content": system_prompt})
        
        # 添加用户提示，指示该 LLM 发言
        messages.append({"role": "user", "content": "现在轮到你作为正方发言了，请继续辩论。"})

        return self._call_llm(messages)


class OpponentAgent(Agent):
    """反方 AI"""
    def __init__(self, **kwargs):
        super().__init__(role="反方", **kwargs)

    def generate_response(self, proposition: str, history: List[Dict[str, Any]], **kwargs) -> str:
        system_prompt = f"你现在是一个辩论中的反方。辩题是：'{proposition}'。你的任务是提出强有力的论点来反驳这个命题，并对正方的论点进行质疑。请根据当前的辩论历史，给出你的下一轮发言。请直接陈述你的观点和论据，不要说 '我是反方' 或进行角色扮演的确认。"
        messages = self._format_history_for_llm(history)
        messages.insert(0, {"role": "system", "content": system_prompt})
        
        # 添加用户提示，指示该 LLM 发言
        messages.append({"role": "user", "content": "现在轮到你作为反方发言了，请继续辩论。"})

        return self._call_llm(messages)


class JudgeAgent(Agent):
    """评审 AI"""
    def __init__(self, **kwargs):
        super().__init__(role="评审", **kwargs)

    def generate_response(self, proposition: str, history: List[Dict[str, Any]], current_round: int | None = None, total_rounds: int | None = None) -> str:
        messages = self._format_history_for_llm(history)

        is_final_round = current_round is not None and total_rounds is not None and current_round == total_rounds

        if is_final_round:
            system_prompt = f"你现在是一个辩论的评审。辩题是：'{proposition}'。辩论现已结束。你的任务是基于完整的辩论历史，给出一个中立、全面的最终总结报告。报告应包括：双方核心论点梳理、关键交锋点分析、论证质量评估（优点与不足）、可能存在的逻辑谬误或未覆盖的视角。请直接给出总结报告，不要说 '我是评审'。"
            user_instruction = "请根据以上辩论给出最终总结报告。"
        else:
            system_prompt = f"你现在是一个辩论的评审。辩题是：'{proposition}'。当前辩论正在进行中。你的任务是基于到目前为止的辩论历史，进行一次简短的中立总结，提炼双方的要点和分歧，并可能提出引导性问题或建议下一轮的讨论焦点，以促进辩论深入。请直接给出你的总结和引导，不要说 '我是评审'。"
            user_instruction = f"请对第 {current_round} 轮辩论进行总结，并引导下一轮（总共 {total_rounds} 轮）。"
        
        messages.insert(0, {"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_instruction})

        # 对于评审，可能需要更长的输出来做总结
        return self._call_llm(messages, max_tokens=DEFAULT_MAX_TOKENS + 512) # 增加 token 限制

    # generate_final_report 方法不再需要，其逻辑已合并到 generate_response 