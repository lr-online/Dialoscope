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
                "系统": "system",
                "评审 (最终报告)": "assistant" # 最终报告也视为 assistant
            }
            # 如果 history 中的角色不在映射中，暂时跳过或用默认值
            api_role = role_map.get(entry["role"], "user")
            # 确保 content 是字符串
            content = str(entry.get("content", ""))
            # 避免添加空的 content
            if content:
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
        system_prompt = f"""
你现在是一个结构化辩论中的 **正方** 辩手。
**辩题是：'{proposition}'**

**你的任务:**
1.  **坚定支持** 正方立场，提出强有力的、逻辑清晰的论点和论据。
2.  **直接回应** 反方上一轮提出的具体论点，进行有力的反驳和质疑。指出其逻辑漏洞、证据不足或与现实不符之处。
3.  **深化论证**: 在之前自己论点的基础上进一步展开，或引入新的角度来支持命题。
4.  保持 **专业、客观、理性** 的辩论风格。

**输出要求:**
*   直接陈述你的观点和论据。
*   **不要** 说 "我是正方" 或进行角色扮演的确认。
*   如果合适，可以使用 Markdown 格式化你的回答（例如列表、重点）。
"""
        messages = self._format_history_for_llm(history)
        messages.insert(0, {"role": "system", "content": system_prompt.strip()})

        # 用户提示，指示该 LLM 发言
        messages.append({"role": "user", "content": "现在轮到你作为 **正方** 发言了。请严格按照要求，分析反方上一轮的发言并进行反驳，同时加强你方论点。"}) # Updated user prompt

        return self._call_llm(messages)


class OpponentAgent(Agent):
    """反方 AI"""
    def __init__(self, **kwargs):
        super().__init__(role="反方", **kwargs)

    def generate_response(self, proposition: str, history: List[Dict[str, Any]], **kwargs) -> str:
        system_prompt = f"""
你现在是一个结构化辩论中的 **反方** 辩手。
**辩题是：'{proposition}'**

**你的任务:**
1.  **坚定反对** 正方立场，提出强有力的、逻辑清晰的论点和论据来反驳命题。
2.  **直接回应** 正方上一轮提出的具体论点，进行有力的反驳和质疑。指出其逻辑漏洞、证据不足或与现实不符之处。
3.  **深化论证**: 在之前自己论点的基础上进一步展开，或引入新的角度来反驳命题。
4.  保持 **专业、客观、理性** 的辩论风格。

**输出要求:**
*   直接陈述你的观点和论据。
*   **不要** 说 "我是反方" 或进行角色扮演的确认。
*   如果合适，可以使用 Markdown 格式化你的回答（例如列表、重点）。
"""
        messages = self._format_history_for_llm(history)
        messages.insert(0, {"role": "system", "content": system_prompt.strip()})

        # 用户提示，指示该 LLM 发言
        messages.append({"role": "user", "content": "现在轮到你作为 **反方** 发言了。请严格按照要求，分析正方上一轮的发言并进行反驳，同时加强你方论点。"}) # Updated user prompt

        return self._call_llm(messages)


class JudgeAgent(Agent):
    """评审 AI"""
    def __init__(self, **kwargs):
        super().__init__(role="评审", **kwargs)

    def generate_response(self, proposition: str, history: List[Dict[str, Any]], current_round: int | None = None, total_rounds: int | None = None) -> str:
        messages = self._format_history_for_llm(history)
        is_final_round = current_round is not None and total_rounds is not None and current_round == total_rounds

        if is_final_round:
            system_prompt = f"""
你现在是一个 **中立且客观** 的辩论评审员。
**辩题是：'{proposition}'**
辩论现已 **结束** (共 {total_rounds} 轮)。

**你的任务:**
基于 **完整** 的辩论历史，给出一个 **结构化** 的最终总结报告。报告必须包含以下部分 (请使用 Markdown 标题):

1.  `### 双方核心论点总结`
    *   简明扼要地分别概括正反双方的主要论证思路和核心证据。
2.  `### 关键交锋点分析`
    *   识别并分析几轮辩论中最主要的矛盾点和双方争论的焦点。
3.  `### 论证质量评估`
    *   分别评估正反双方论证的优点（如逻辑性、证据充分性、回应有效性）和不足（如逻辑漏洞、回避问题、证据不足）。
4.  `### 待解决问题与启发`
    *   指出辩论结束后仍未完全解决的关键问题，或辩论过程带来的新思考角度/启发。

**输出要求:**
*   保持 **绝对中立**，不对任何一方有偏好。
*   语言精炼、客观。
*   **不要** 说 "我是评审" 或进行角色扮演的确认。
*   **必须** 使用 Markdown 格式化报告，特别是标题。
"""
            user_instruction = "请根据以上整场辩论的详细记录，严格按照要求生成结构化的最终总结报告。"
            max_tokens_multiplier = 1.5 # 最终报告需要更长
        else:
            system_prompt = f"""
你现在是一个 **中立且客观** 的辩论评审员。
**辩题是：'{proposition}'**
当前辩论正在进行中，刚刚结束第 {current_round} 轮 (共 {total_rounds} 轮)。

**你的任务:**
1.  **精炼总结** 刚刚结束的 **第 {current_round} 轮** 双方的主要论点和 **最关键的分歧**。
2.  **提出引导**: 基于本轮总结，提出 **1-2个清晰、具体的问题或讨论焦点**，引导双方在 **下一轮** 进行更深入、更有针对性的辩论。避免空泛的引导。

**输出要求:**
*   保持 **绝对中立**。
*   总结要 **简洁**，聚焦本轮核心内容。
*   引导性问题要 **明确**，能推动辩论进展。
*   **不要** 说 "我是评审" 或进行角色扮演的确认。
*   可以使用 Markdown 格式化回答。
"""
            user_instruction = f"请严格按照要求，对第 {current_round} 轮辩论进行简洁总结，并给出对下一轮的具体引导问题或焦点。"
            max_tokens_multiplier = 1.0 # 轮间总结不需要太长

        messages.insert(0, {"role": "system", "content": system_prompt.strip()}) # 使用 strip() 清理前后空白
        messages.append({"role": "user", "content": user_instruction})

        # 根据是否最终轮调整 token 限制
        adjusted_max_tokens = int(DEFAULT_MAX_TOKENS * max_tokens_multiplier)
        return self._call_llm(messages, max_tokens=adjusted_max_tokens) 