from typing import List, Dict, Any, Callable
import datetime
import json
import os

from .agents import ProponentAgent, OpponentAgent, JudgeAgent, DEFAULT_MODEL

class DebateManager:
    """管理整个辩论流程"""

    def __init__(self,
                 proposition: str,
                 rounds: int = 3,
                 log_dir: str = "logs",
                 model: str = DEFAULT_MODEL,
                 printer: Callable[[str, str], None] | None = None # 添加 printer 参数
                 ):
        self.proposition = proposition
        self.rounds = rounds
        self.history: List[Dict[str, Any]] = []
        self.log_dir = log_dir
        self.model = model
        self.printer = printer # 存储 printer
        os.makedirs(self.log_dir, exist_ok=True)

        # 初始化 AI 角色，传入模型名称
        try:
            # TODO: 后续可以添加更灵活的 Agent 配置方式
            self.proponent = ProponentAgent(model=self.model)
            self.opponent = OpponentAgent(model=self.model)
            self.judge = JudgeAgent(model=self.model)
        except (ValueError, RuntimeError) as e:
            # 初始化错误通过 printer 输出（如果提供）
            error_message = f"初始化 Agent 时出错: {e}"
            if self.printer:
                self.printer("错误", error_message)
            else:
                print(error_message)
            raise

    def _add_to_history(self, role: str, content: str):
        """将发言添加到历史记录，并使用 printer 输出（如果提供）"""
        entry = {"role": role, "content": content, "timestamp": datetime.datetime.now().isoformat()}
        self.history.append(entry)
        # 如果提供了 printer 函数，则调用它来输出
        if self.printer:
            self.printer(role, content)
        else:
            # 保留基本的 print 作为后备
            print(f"\n--- {role} ---\n{content}")

    def run_debate(self):
        """执行完整的辩论流程"""
        # 初始系统消息通过 _add_to_history 发送 (现在会通过 printer 输出)
        self._add_to_history("系统", f"辩论开始，辩题：{self.proposition}\n计划轮数: {self.rounds}\n使用模型: {self.model}")

        for i in range(self.rounds):
            current_round_num = i + 1
            # 打印轮次信息 - 可以考虑也通过 printer 发送一个 "系统" 消息
            # self._add_to_history("系统", f"===== 第 {current_round_num} 轮 =====")
            # 或者，如果 CLI 需要更明确的控制，让 CLI 打印轮次信息
            print(f"\n===== 第 {current_round_num} 轮 =====") # 暂时保留这个，让 CLI 结构更清晰

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

                # 判断是否是最终轮的报告来确定角色名
                if current_round_num == self.rounds:
                    judge_role_name = self.judge.role + " (最终报告)"
                    # 打印结束信息 - 同上，暂时保留
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
            # 日志保存信息可以通过 printer 或 print 输出
            log_message = f"辩论日志已保存到: {filepath}"
            if self.printer:
                # 可以定义一个 '系统' 或 '日志' 角色
                 self.printer("系统", log_message) 
            else:
                 print(f"\n{log_message}")
        except Exception as e:
            error_message = f"无法保存日志到 {filepath}. 原因: {e}"
            if self.printer:
                self.printer("错误", error_message)
            else:
                print(f"\n错误：{error_message}") 