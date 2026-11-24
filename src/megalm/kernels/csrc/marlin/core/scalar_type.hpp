// Minimal stand-in for vLLM's core/scalar_type.hpp.  ScalarTypeId is used only
// as a non-type template parameter type, so an int alias matches vLLM's
// semantics and avoids nvcc's const-class-NTTP restrictions.
#pragma once
namespace vllm {
using ScalarTypeId = int;
struct ScalarType {
  int value_;
  int bits_;
  constexpr ScalarTypeId id() const { return value_; }
  constexpr int size_bits() const { return bits_; }
  static constexpr ScalarType from_id(ScalarTypeId id) {
    int bits = 16;
    switch (id) {
      case 3: case 6: case 8: case 9: bits = 4; break;
      case 4: case 5: case 7: case 10: case 11: bits = 8; break;
      default: break;
    }
    return ScalarType{id, bits};
  }
};
struct IdObj {
  int value;
  constexpr int id() const { return value; }
};
inline constexpr IdObj kFloat16{1};
inline constexpr IdObj kBFloat16{2};
inline constexpr IdObj kFE2M1f{3};
inline constexpr IdObj kFE4M3fn{4};
inline constexpr IdObj kFE8M0fnu{5};
inline constexpr IdObj kS4{6};
inline constexpr IdObj kS8{7};
inline constexpr IdObj kU4{8};
inline constexpr IdObj kU4B8{9};
inline constexpr IdObj kU8{10};
inline constexpr IdObj kU8B128{11};
constexpr bool operator==(const ScalarType& a, ScalarTypeId b) {
  return a.value_ == b;
}
constexpr bool operator==(const ScalarType& a, const IdObj& b) {
  return a.value_ == b.value;
}
}  // namespace vllm
