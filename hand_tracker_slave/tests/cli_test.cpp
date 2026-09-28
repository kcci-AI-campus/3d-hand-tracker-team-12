#include "cli.hpp"
#include <sstream>
void Check(bool ok) { if (!ok) throw std::runtime_error("CLI regression"); }
int main() {
  try {
    std::ostringstream output;
    std::istringstream defaults(" 192.168.1.20 \n\n");
    auto d = PromptDestination(defaults, output);
    Check(d.first == "192.168.1.20" && d.second == 5001);
    std::istringstream retry("invalid\n0.0.0.0\n239.1.2.3\n127.0.0.1\n70000\n5002x\n5002\n");
    d = PromptDestination(retry, output);
    Check(d.first == "127.0.0.1" && d.second == 5002);
    bool cancelled = false;
    try { std::istringstream eof(""); PromptDestination(eof, output); }
    catch (const std::runtime_error&) { cancelled = true; }
    Check(cancelled);
    uv_stream::Sender disabled("",5001);
    Check(disabled.Sent() == 0 && disabled.Failed() == 0 && disabled.LastError() == 0);
    std::cout << "CLI validation/default/retry/EOF OK\n";
  } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
