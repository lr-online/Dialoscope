from typing import List, Dict, Any
import datetime
import json
import os

from .agents import ProponentAgent, OpponentAgent, JudgeAgent, DEFAULT_MODEL

class DebateManager:
    """管理整个辩论流程"""

    def __init__(self, proposition: str, rounds: int = 3, log_dir: str = "logs", model: str = DEFAULT_MODEL):
        self.proposition = proposition
        self.rounds = rounds
        self.history: List[Dict[str, Any]] = []
        self.log_dir = log_dir
        self.model = model
        os.makedirs(self.log_dir, exist_ok=True)

        # 初始化 AI 角色，传入模型名称
        try:
            # TODO: 后续可以添加更灵活的 Agent 配置方式
            self.proponent = ProponentAgent(model=self.model)
            self.opponent = OpponentAgent(model=self.model)
            self.judge = JudgeAgent(model=self.model)
        except (ValueError, RuntimeError) as e:
            # 在初始化 Agent 时捕获环境错误等
            print(f"初始化 Agent 时出错: {e}")
            # 向上抛出异常，让调用者（如 CLI）处理
            raise

    def _add_to_history(self, role: str, content: str):
        """将发言添加到历史记录"""
        entry = {"role": role, "content": content, "timestamp": datetime.datetime.now().isoformat()}
        self.history.append(entry)
        # 打印时也包含角色信息
        print(f"\n--- {role} ---\n{content}")

    def run_debate(self):
        """执行完整的辩论流程"""
        print(f"\n辩题: {self.proposition}")
        print(f"计划轮数: {self.rounds}")
        print(f"使用模型: {self.model}\n")
        self._add_to_history("系统", f"辩论开始，辩题：{self.proposition}")

        for i in range(self.rounds):
            current_round_num = i + 1
            print(f"\n===== 第 {current_round_num} 轮 =====")

            try:
                # 正方发言
                pro_response = self.proponent.generate_response(self.proposition, self.history)
                self._add_to_history(self.proponent.role, pro_response)

                # 反方发言
                opp_response = self.opponent.generate_response(self.proposition, self.history)
                self._add_to_history(self.opponent.role, opp_response)

                # 评审发言 (总结本轮 或 生成最终报告)
                judge_response = self.judge.generate_response(
                    proposition=self.proposition,
                    history=self.history,
                    current_round=current_round_num,
                    total_rounds=self.rounds
                )

                # 判断是否是最终轮的报告来确定角色名并打印结束信息
                if current_round_num == self.rounds:
                    print("\n===== 辩论结束 =====")
                    judge_role_name = self.judge.role + " (最终报告)"
                else:
                    # 可以选择性地给轮间总结加上轮次标记，或保持简洁
                    judge_role_name = self.judge.role # f" (第 {current_round_num} 轮总结)" 

                self._add_to_history(judge_role_name, judge_response)

            except Exception as e:
                # 捕获辩论进行中可能发生的错误 (例如 API 调用失败)
                print(f"\n错误：在第 {current_round_num} 轮辩论中发生错误: {e}")
                print("辩论提前终止。")
                # 可以在这里决定是否保存部分日志等
                break # 终止辩论循环

        # 循环结束后保存日志
        self.save_log()

    def save_log(self):
        """保存辩论日志到文件"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        # 在文件名中包含辩题的部分信息可能有助于识别
        safe_proposition = "".join(c if c.isalnum() else '_' for c in self.proposition[:30])
        filename = f"debate_{safe_proposition}_{timestamp}.json"
        filepath = os.path.join(self.log_dir, filename)
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self.history, f, ensure_ascii=False, indent=4)
            print(f"\n辩论日志已保存到: {filepath}")
        except Exception as e:
            print(f"\n错误：无法保存日志到 {filepath}. 原因: {e}") 