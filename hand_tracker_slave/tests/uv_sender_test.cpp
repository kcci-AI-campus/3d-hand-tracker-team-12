#include "uv_sender.hpp"
#include <vector>
struct Point { float x, y; };
struct Hand { std::array<Point, 21> points; float right_probability, presence; };
void Check(bool ok) { if (!ok) throw std::runtime_error("UV protocol regression"); }
int main(int argc, char** argv) {
  try {
    std::vector<Hand> hands;
    auto uv = uv_stream::CoordinatesFrom(hands, 320, 240, false);
    for (float x : uv) Check(x == -1.f);
    uv_stream::Frame frame{1, 0x01020304u, 45678u, 1234u, uv};
    auto bytes = uv_stream::Encode(frame);
    Check(bytes.size() == 356 && bytes[0] == 'H' && bytes[1] == 'U' && bytes[2] == 'V' && bytes[3] == '2');
    Check(bytes[4] == 2 && bytes[5] == 1 && bytes[6] == 0 && bytes[7] == 0);
    Check(bytes[8] == 1 && bytes[9] == 2 && bytes[10] == 3 && bytes[11] == 4);            // big-endian sequence
    Check(bytes[20] == 0xbf && bytes[21] == 0x80 && bytes[22] == 0 && bytes[23] == 0);     // -1.0f
    auto decoded = uv_stream::Decode(bytes.data(), bytes.size());
    Check(decoded && decoded->camera == 1 && decoded->sequence == 0x01020304u && decoded->capture_to_send_us == 45678u &&
          decoded->infer_us == 1234u && decoded->uv == uv);
    Check(!uv_stream::Decode(bytes.data(), 336));                                           // legacy UV42 size
    bytes[4] = 3;
    Check(!uv_stream::Decode(bytes.data(), bytes.size()));                                  // unknown version
    Hand left{}, right{};
    left.points.fill({80,120}); left.right_probability = .9f; left.presence = .99f;
    right.points.fill({240,60}); right.right_probability = .1f; right.presence = .99f;
    hands = {right, left}; // Detection order must not change the slots.
    uv = uv_stream::CoordinatesFrom(hands, 320, 240, false);
    Check(uv[0] == .25f && uv[1] == .5f && uv[42] == .75f && uv[43] == .25f);
    auto mirrored = uv_stream::CoordinatesFrom(hands, 320, 240, true);
    Check(mirrored[0] == .75f && mirrored[42] == .25f);
    Hand duplicate = left; duplicate.presence = .5f; duplicate.points.fill({0,0});
    hands.push_back(duplicate);
    Check(uv_stream::CoordinatesFrom(hands,320,240,false) == uv);
    hands[1].points[0] = {std::numeric_limits<float>::quiet_NaN(), 0};
    hands[1].points[1] = {-2, 0};
    const auto invalid = uv_stream::CoordinatesFrom(hands,320,240,false);
    Check(invalid[0] == -1 && invalid[1] == -1 && invalid[2] == -1 && invalid[3] == -1);
    if (argc == 3) {
      uv_stream::Sender sender(argv[1], std::stoi(argv[2])); sender.Send(uv_stream::Frame{0, 0, 0, 0, uv});
      Check(sender.Sent() == 1 && sender.Failed() == 0 && sender.LastError() == 0);
      Check(sender.Status().find("Bytes: 356") != std::string::npos);
    }
    std::cout << "UV mapping/serialization OK\n";
  } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
