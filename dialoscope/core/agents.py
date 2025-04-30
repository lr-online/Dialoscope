import abc
import os
from typing import List, Dict, Any, Iterator

from dialoscope.llm_clients import LLMClient

# 默认模型配置
DEFAULT_MODEL = "gpt-4o"
DEFAULT_MAX_TOKENS = 1024
DEFAULT_TEMPERATURE = 0.7

class Agent(abc.ABC):
    """AI 辩论角色的基类"""

    def __init__(self, role: str, llm_client: LLMClient):
        self.role = role
        self.llm_client = llm_client

    def _format_history_for_llm(self, history: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """将内部历史记录格式化为 LLM API 需要的格式"""
        messages = []
        for entry in history:
            role_map = {
                "正方": "user",
                "反方": "user",
                "评审": "assistant",
                "系统": "system",
                "评审 (最终报告)": "assistant"
            }
            api_role = role_map.get(entry["role"], "user")
            content = str(entry.get("content", ""))
            if content:
                # Basic check to ensure role is valid for most APIs (user, assistant, system)
                if api_role not in ["user", "assistant", "system"]:
                    print(f"Warning: Mapping role '{entry['role']}' to '{api_role}', which might not be standard.")
                messages.append({"role": api_role, "content": content})
        return messages

    @abc.abstractmethod
    def _create_llm_messages(
        self,
        proposition: str,
        history: List[Dict[str, Any]],
        current_round: int | None = None,
        total_rounds: int | None = None
        ) -> List[Dict[str, str]]:
        """
        为特定角色和情境创建发送给 LLM 的消息列表。
        子类必须实现此方法来定义它们的提示和逻辑。
        """
        pass

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
        # 1. Create messages using the subclass's logic
        messages = self._create_llm_messages(proposition, history, current_round, total_rounds)

        # 2. Call the LLM client's generation method
        try:
            response = self.llm_client.generate_response(messages)
            # Basic check for empty response
            if not response:
                return "(模型未返回有效内容)"
            return response.strip()
        except Exception as e:
            # Catch potential errors from the client's generate_response
            print(f"Error during LLM response generation for {self.role}: {e}")
            return f"(调用 {type(self.llm_client).__name__} 时出错: {e})"

    def generate_response_stream(
        self,
        proposition: str,
        history: List[Dict[str, Any]],
        current_round: int | None = None,
        total_rounds: int | None = None
        ) -> Iterator[str]:
        """
        Generates response chunks as a stream.
        """
        # 1. Create messages using the subclass's logic
        messages = self._create_llm_messages(proposition, history, current_round, total_rounds)

        # 2. Call the LLM client's streaming generation method and yield chunks
        try:
            # Yield each chunk as it arrives from the client
            yield from self.llm_client.generate_response_stream(messages)
        except Exception as e:
            # Catch potential errors during stream generation
            print(f"Error during LLM stream generation for {self.role}: {e}")
            # Yield an error message chunk
            yield f"(调用 {type(self.llm_client).__name__} 流式接口时出错: {e})"


class ProponentAgent(Agent):
    """正方 AI"""
    def __init__(self, llm_client: LLMClient):
        super().__init__(role="正方", llm_client=llm_client)

    def _create_llm_messages(self, proposition: str, history: List[Dict[str, Any]], current_round: int | None = None, total_rounds: int | None = None) -> List[Dict[str, str]]:
        messages = self._format_history_for_llm(history)
        is_first_round = len(history) <= 1 # Check if it's the opening statement

        if is_first_round:
            system_prompt = f"""
你现在是一个结构化辩论中的 **正方** 辩手。
**辩题是：'{proposition}'**

**你的任务 (开场陈述):**
1.  **清晰阐述** 你支持 **'{proposition}'** 的核心立场。
2.  提出 **2-3个强有力、逻辑清晰的初始论点** 来支持你的立场，可以简要提及论据方向。
3.  为接下来的辩论奠定基础。
4.  保持 **专业、客观、理性** 的辩论风格。

**输出要求:**
*   直接陈述你的立场和初始论点。
*   **不要** 进行角色扮演的确认 (例如，不要说 "我是正方")。
*   如果合适，可以使用 Markdown 格式化你的回答。
"""
            user_instruction = f"请针对辩题 '{proposition}'，作为正方发表你的开场陈述。"
        else:
            system_prompt = f"""
你现在是一个结构化辩论中的 **正方** 辩手。
**辩题是：'{proposition}'**
这是第 {current_round} 轮辩论 (共 {total_rounds} 轮)。请回顾之前的辩论历史。

**你的任务 (回应与深化):**
1.  **直接回应** 反方在 **上一轮** 中提出的 **具体论点**，进行有力的反驳和质疑。指出其逻辑漏洞、证据不足或与现实不符之处。
2.  在之前自己论点的基础上 **深化论证**，或引入新的角度来支持命题。
3.  **重申并加强** 你的核心立场。
4.  保持 **专业、客观、理性** 的辩论风格。

**输出要求:**
*   直接陈述你的观点和论据。
*   **不要** 进行角色扮演的确认。
*   清晰地指明你在回应反方的哪些观点。
*   如果合适，可以使用 Markdown 格式化你的回答。
"""
            user_instruction = f"现在轮到你作为 **正方** 发言 (第 {current_round}/{total_rounds} 轮)。请严格按照要求，分析并反驳反方上一轮的发言，同时加强并深化你方论点。"

        messages.insert(0, {"role": "system", "content": system_prompt.strip()})
        return messages


class OpponentAgent(Agent):
    """反方 AI"""
    def __init__(self, llm_client: LLMClient):
        super().__init__(role="反方", llm_client=llm_client)

    def _create_llm_messages(self, proposition: str, history: List[Dict[str, Any]], current_round: int | None = None, total_rounds: int | None = None) -> List[Dict[str, str]]:
        messages = self._format_history_for_llm(history)
        is_first_round = len(history) <= 1 # Check if it's the opening statement

        if is_first_round:
            system_prompt = f"""
你现在是一个结构化辩论中的 **反方** 辩手。
**辩题是：'{proposition}'**

**你的任务 (开场陈述):**
1.  **清晰阐述** 你反对 **'{proposition}'** 的核心立场。
2.  提出 **2-3个强有力、逻辑清晰的初始论点** 来反驳你的立场，可以简要提及论据方向。
3.  为接下来的辩论奠定基础。
4.  保持 **专业、客观、理性** 的辩论风格。

**输出要求:**
*   直接陈述你的立场和初始论点。
*   **不要** 进行角色扮演的确认 (例如，不要说 "我是反方")。
*   如果合适，可以使用 Markdown 格式化你的回答。
"""
            user_instruction = f"请针对辩题 '{proposition}'，作为反方发表你的开场陈述。"
        else:
            system_prompt = f"""
你现在是一个结构化辩论中的 **反方** 辩手。
**辩题是：'{proposition}'**
这是第 {current_round} 轮辩论 (共 {total_rounds} 轮)。请回顾之前的辩论历史。

**你的任务 (回应与深化):**
1.  **直接回应** 正方在 **上一轮** 中提出的 **具体论点**，进行有力的反驳和质疑。指出其逻辑漏洞、证据不足或与现实不符之处。
2.  在之前自己论点的基础上 **深化论证**，或引入新的角度来反驳命题。
3.  **重申并加强** 你的核心立场。
4.  保持 **专业、客观、理性** 的辩论风格。

**输出要求:**
*   直接陈述你的观点和论据。
*   **不要** 进行角色扮演的确认。
*   清晰地指明你在回应正方的哪些观点。
*   如果合适，可以使用 Markdown 格式化你的回答。
"""
            user_instruction = f"现在轮到你作为 **反方** 发言 (第 {current_round}/{total_rounds} 轮)。请严格按照要求，分析并反驳正方上一轮的发言，同时加强并深化你方论点。"

        messages.insert(0, {"role": "system", "content": system_prompt.strip()})
        return messages


class JudgeAgent(Agent):
    """评审 AI"""
    def __init__(self, llm_client: LLMClient):
        super().__init__(role="评审", llm_client=llm_client)

    def _create_llm_messages(self, proposition: str, history: List[Dict[str, Any]], current_round: int | None = None, total_rounds: int | None = None) -> List[Dict[str, str]]:
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
        else:
            if current_round is None or total_rounds is None:
                print("Warning: Missing round information for intermediate judge summary. Generating generic prompt.")
                current_round_str = "当前"
                total_rounds_str = "未知"
            else:
                current_round_str = str(current_round)
                total_rounds_str = str(total_rounds)

            system_prompt = f"""
你现在是一个 **中立且客观** 的辩论评审员。
**辩题是：'{proposition}'**
当前辩论正在进行中，刚刚结束第 {current_round_str} 轮 (共 {total_rounds_str} 轮)。

**你的任务:**
1.  **精炼总结** 刚刚结束的 **第 {current_round_str} 轮** 双方的主要论点和 **最关键的分歧**。
2.  **提出引导**: 基于本轮总结，提出 **1-2个清晰、具体的问题或讨论焦点**，引导双方在 **下一轮** 进行更深入、更有针对性的辩论。避免空泛的引导。

**输出要求:**
*   保持 **绝对中立**。
*   总结要 **简洁**，聚焦本轮核心内容。
*   引导性问题要 **明确**，能推动辩论进展。
*   **不要** 说 "我是评审" 或进行角色扮演的确认。
*   可以使用 Markdown 格式化回答。
"""
            user_instruction = f"请严格按照要求，对第 {current_round_str} 轮辩论进行简洁总结，并给出对下一轮的具体引导问题或焦点。"

        messages.insert(0, {"role": "system", "content": system_prompt.strip()})
        messages.append({"role": "user", "content": user_instruction})

        return messages

    def generate_response(self, proposition: str, history: List[Dict[str, Any]], current_round: int | None = None, total_rounds: int | None = None) -> str:
        messages = self._create_llm_messages(proposition, history, current_round, total_rounds)
        return self.llm_client.generate_response(messages) 