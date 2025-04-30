from typing import List, Dict, Any, Callable
import datetime
import json
import os

from .agents import ProponentAgent, OpponentAgent, JudgeAgent, Agent # Import Agent base class

class DebateManager:
    """管理整个辩论流程"""

    def __init__(self,
                 proposition: str,
                 proponent: ProponentAgent, # Accept ProponentAgent instance
                 opponent: OpponentAgent,   # Accept OpponentAgent instance
                 judge: JudgeAgent,       # Accept JudgeAgent instance
                 rounds: int = 3,
                 log_dir: str = "logs",
                 printer: Callable[[str, str], None] | None = None
                 ):
        self.proposition = proposition
        self.rounds = rounds
        self.history: List[Dict[str, Any]] = []
        self.log_dir = log_dir
        self.printer = printer
        os.makedirs(self.log_dir, exist_ok=True)

        # Store the provided agent instances
        self.proponent = proponent
        self.opponent = opponent
        self.judge = judge

        # Removed internal agent initialization and model parameter
        # Removed error handling for agent initialization here, should be done in main.py

    def _add_to_history(self, role: str, content: str):
        """将发言添加到历史记录，并使用 printer 输出（如果提供）"""
        entry = {"role": role, "content": content, "timestamp": datetime.datetime.now().isoformat()}
        self.history.append(entry)
        if self.printer:
            self.printer(role, content)
        else:
            print(f"\n--- {role} ---\n{content}")

    def run_debate(self):
        """执行完整的辩论流程"""
        # Pass the agent's role to _add_to_history for clarity
        initial_message = f"辩论开始，辩题：{self.proposition}\n计划轮数: {self.rounds}\nLLM 客户端: {type(self.proponent.llm_client).__name__}"
        self._add_to_history("系统", initial_message)

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

                # 评审发言
                judge_response = self.judge.generate_response(
                    proposition=self.proposition,
                    history=self.history,
                    current_round=current_round_num,
                    total_rounds=self.rounds
                )

                if current_round_num == self.rounds:
                    judge_role_name = self.judge.role + " (最终报告)"
                    print("\n===== 辩论结束 =====")
                else:
                    judge_role_name = self.judge.role

                self._add_to_history(judge_role_name, judge_response)

            except Exception as e:
                error_message = f"在第 {current_round_num} 轮辩论中发生错误: {e}"
                if self.printer:
                    self.printer("错误", error_message)
                else:
                    print(f"\n错误：{error_message}")
                print("辩论提前终止。")
                break

        self.save_log()

    def save_log(self):
        """保存辩论日志到文件"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_proposition = "".join(c if c.isalnum() else '_' for c in self.proposition[:30])
        filename = f"debate_{safe_proposition}_{timestamp}.json"
        filepath = os.path.join(self.log_dir, filename)
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self.history, f, ensure_ascii=False, indent=4)
            log_message = f"辩论日志已保存到: {filepath}"
            if self.printer:
                self.printer("系统", log_message)
            else:
                print(f"\n{log_message}")
        except Exception as e:
            error_message = f"无法保存日志到 {filepath}. 原因: {e}"
            if self.printer:
                self.printer("错误", error_message)
            else:
                print(f"\n错误：{error_message}") 