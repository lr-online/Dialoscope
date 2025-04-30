from typing import List, Dict, Any, Callable, Iterator
import datetime
import json
import os

from .agents import Agent # Keep Agent base class

# Type hint for the printer functions passed from CLI
# Define a more specific type hint if possible, or use Callable
PrinterType = Callable[..., Any] # Generic callable for now

class DebateManager:
    """管理整个辩论流程"""

    def __init__(self,
                 proposition: str,
                 proponent: Agent, # Use base Agent type hint
                 opponent: Agent,
                 judge: Agent,
                 rounds: int = 3,
                 log_dir: str = "logs",
                 printer: PrinterType | None = None, # The non-stream printer
                 stream_printer: PrinterType | None = None # The stream printer
                 ):
        self.proposition = proposition
        self.rounds = rounds
        self.history: List[Dict[str, Any]] = []
        self.log_dir = log_dir
        # Store both printers
        self.printer = printer # For non-stream messages (system, errors)
        self.stream_printer = stream_printer # For agent responses
        os.makedirs(self.log_dir, exist_ok=True)

        self.proponent = proponent
        self.opponent = opponent
        self.judge = judge

        # Removed internal agent initialization and model parameter
        # Removed error handling for agent initialization here, should be done in main.py

    def _add_message_to_history(self, role: str, content: str):
        """将完整的发言添加到历史记录。"""
        entry = {"role": role, "content": content, "timestamp": datetime.datetime.now().isoformat()}
        self.history.append(entry)

    def _print_non_stream_message(self, role: str, content: str):
        if self.printer:
            self.printer(role, content)
        else:
            print(f"\n--- {role} ---\n{content}")

    def _handle_agent_turn(self, agent: Agent, role_name: str):
        """Handles getting stream, printing it, and adding full message to history."""
        full_content = ""
        try:
            # Get the stream from the agent
            response_stream = agent.generate_response_stream(
                self.proposition,
                self.history,
                # Pass round info only if it's the judge (or modify agent prompts further)
                # For now, pass it always, agents will ignore if not needed
                current_round=self.current_round, 
                total_rounds=self.rounds
            )

            # Print the stream using the stream_printer (if available)
            # and accumulate the full content
            if self.stream_printer:
                full_content = self.stream_printer(role_name, response_stream)
            else:
                # Fallback: consume stream and print full content at the end
                temp_content_list = []
                for chunk in response_stream:
                    print(chunk, end='', flush=True) # Basic stream print fallback
                    temp_content_list.append(chunk)
                full_content = "".join(temp_content_list)
                print() # Newline after fallback print
                # Optionally print the role header here if not using stream_printer
                # print(f"\n--- {role_name} ---")

        except Exception as e:
            error_message = f"生成 {role_name} 回应时出错: {e}"
            self._print_non_stream_message("错误", error_message)
            full_content = f"[错误: {error_message}]" # Store error in history
            # Decide if we should re-raise or just store error and continue
            # raise e # Re-raise to stop the debate? Or just log it?

        # Add the fully accumulated (or error) message to history
        self._add_message_to_history(role_name, full_content)

    def run_debate(self):
        """执行完整的辩论流程 (使用流式响应)"""
        # Store current round for _handle_agent_turn
        self.current_round = 0 

        initial_message = f"辩论开始，辩题：{self.proposition}\n计划轮数: {self.rounds}\n"
        # Print initial message and add to history
        self._add_message_to_history("系统", initial_message)

        for i in range(self.rounds):
            self.current_round = i + 1 # Update current round
            # Use non-stream printer for round separator
            self._print_non_stream_message("系统", f"===== 第 {self.current_round} 轮 =====") 

            try:
                # 正方发言
                self._handle_agent_turn(self.proponent, self.proponent.role)

                # 反方发言
                self._handle_agent_turn(self.opponent, self.opponent.role)

                # 评审发言
                judge_role_name = self.judge.role
                if self.current_round == self.rounds:
                    judge_role_name += " (最终报告)"
                    # Print end marker using non-stream printer
                    self._print_non_stream_message("系统", "===== 辩论结束 =====") 
                
                self._handle_agent_turn(self.judge, judge_role_name)

            except Exception as e:
                # Catch errors not caught within _handle_agent_turn (e.g., if we re-raise)
                error_message = f"在第 {self.current_round} 轮辩论中发生严重错误: {e}"
                self._print_non_stream_message("错误", error_message)
                print("辩论提前终止。")
                break # Stop the debate loop

        self.save_log()

    def save_log(self):
        """保存辩论日志到文件"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_proposition = "".join(c if c.isalnum() else '_' for c in self.proposition[:30])
        filename = f"{timestamp}_{safe_proposition}.json"
        filepath = os.path.join(self.log_dir, filename)
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self.history, f, ensure_ascii=False, indent=4)
            log_message = f"辩论日志已保存到: {filepath}"
            self._print_non_stream_message("系统", log_message)
        except Exception as e:
            error_message = f"无法保存日志到 {filepath}. 原因: {e}"
            self._print_non_stream_message("错误", error_message) 