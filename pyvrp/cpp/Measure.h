#ifndef PYVRP_MEASURE_H
#define PYVRP_MEASURE_H

#include <cassert>
#include <cmath>
#include <compare>
#include <cstdint>
#include <functional>
#include <iostream>
#include <limits>
#include <type_traits>

namespace pyvrp
{
using Value = int64_t;

enum class MeasureType
{
    COORD,
    DIST,
    DURATION,
    COST,
    LOAD,
};

// Forward declaration so we can define the relevant type aliases early.
template <MeasureType _> class Measure;

// Type aliases. These are used throughout the program.
using Coordinate = Measure<MeasureType::COORD>;
using Cost = Measure<MeasureType::COST>;
using Distance = Measure<MeasureType::DIST>;
using Duration = Measure<MeasureType::DURATION>;
using Load = Measure<MeasureType::LOAD>;

//
//                 EVERYTHING BELOW THIS IS IMPLEMENTATION
//

/**
 * The measure class is a thin wrapper around an underlying ``Value``. The
 * measure forms a strong type that is only explicitly convertible into other
 * arithmetic or measure types.
 */
template <MeasureType _> class Measure
{
    Value value;

public:
    // Default construction initialises to 0.
    Measure() = default;

    // This constructor takes any arithmetic type (generally useful) and casts
    // its value to Value.
    template <typename T,
              std::enable_if_t<std::is_arithmetic_v<T>, bool> = true>
    Measure(T const value) : value(static_cast<Value>(value))
    {
    }

    // Explicit conversions of the underlying value to other arithmetic types.
    template <typename T,
              std::enable_if_t<std::is_arithmetic_v<T>, bool> = true>
    explicit operator T() const
    {
        return static_cast<T>(value);
    }

    // Explicit conversions to other measures.
    template <MeasureType Other> explicit operator Measure<Other>() const
    {
        return value;
    }

    // Retreives the underlying value.
    [[nodiscard]] Value get() const;

    // In-place unary operators.
    Measure &operator+=(Measure const &rhs);
    Measure &operator-=(Measure const &rhs);
    Measure &operator*=(Measure const &rhs);
    Measure &operator/=(Measure const &rhs);

    // Comparison operators.
    [[nodiscard]] bool operator==(Measure const &other) const;
    [[nodiscard]] std::strong_ordering operator<=>(Measure const &other) const;
};

// Retreives the underlying value.
template <MeasureType Type> Value Measure<Type>::get() const { return value; }

// In-place unary operators.
template <MeasureType Type>
Measure<Type> &Measure<Type>::operator+=(Measure<Type> const &rhs)
{
    [[maybe_unused]] Value res = 0;
    assert(!__builtin_add_overflow(this->value, rhs.value, &res));

    this->value += rhs.value;
    return *this;
}

template <MeasureType Type>
Measure<Type> &Measure<Type>::operator-=(Measure<Type> const &rhs)
{
    [[maybe_unused]] Value res = 0;
    assert(!__builtin_sub_overflow(this->value, rhs.value, &res));

    this->value -= rhs.value;
    return *this;
}

template <MeasureType Type>
Measure<Type> &Measure<Type>::operator*=(Measure<Type> const &rhs)
{
    [[maybe_unused]] Value res = 0;
    assert(!__builtin_mul_overflow(this->value, rhs.value, &res));

    this->value *= rhs.value;
    return *this;
}

template <MeasureType Type>
Measure<Type> &Measure<Type>::operator/=(Measure<Type> const &rhs)
{
    this->value /= rhs.value;
    return *this;
}

// Comparison operators.
template <MeasureType Type>
bool Measure<Type>::operator==(Measure<Type> const &other) const
{
    return value == other.value;
}

template <MeasureType Type>
std::strong_ordering
Measure<Type>::operator<=>(Measure<Type> const &other) const
{
    return value <=> other.value;
}

// Free-standing binary operators.
template <MeasureType Type>
Measure<Type> operator+(Measure<Type> const lhs, Measure<Type> const rhs)
{
    [[maybe_unused]] Value res = 0;
    assert(!__builtin_add_overflow(lhs.get(), rhs.get(), &res));

    return lhs.get() + rhs.get();
}

template <MeasureType Type> Measure<Type> operator+(Measure<Type> const lhs)
{
    return +lhs.get();
}

template <MeasureType Type>
Measure<Type> operator-(Measure<Type> const lhs, Measure<Type> const rhs)
{
    [[maybe_unused]] Value res = 0;
    assert(!__builtin_sub_overflow(lhs.get(), rhs.get(), &res));

    return lhs.get() - rhs.get();
}

template <MeasureType Type> Measure<Type> operator-(Measure<Type> const lhs)
{
    return -lhs.get();
}

template <MeasureType Type>
Measure<Type> operator*(Measure<Type> const lhs, Measure<Type> const rhs)
{
    [[maybe_unused]] Value res = 0;
    assert(!__builtin_mul_overflow(lhs.get(), rhs.get(), &res));

    return lhs.get() * rhs.get();
}

template <MeasureType Type>
Measure<Type> operator/(Measure<Type> const lhs, Measure<Type> const rhs)
{
    return lhs.get() / rhs.get();
}
}  // namespace pyvrp

// For printing.
template <pyvrp::MeasureType Type>
std::ostream &operator<<(std::ostream &out, pyvrp::Measure<Type> const measure)
{
    return out << measure.get();
}

// Specialisations for hashing and numerical limits.

template <pyvrp::MeasureType Type> struct std::hash<pyvrp::Measure<Type>>
{
    size_t operator()(pyvrp::Measure<Type> const measure) const
    {
        return std::hash<pyvrp::Value>()(measure.get());
    }
};

template <pyvrp::MeasureType Type>
class std::numeric_limits<pyvrp::Measure<Type>>
{
public:
    static pyvrp::Measure<Type> max()
    {
        return std::numeric_limits<pyvrp::Value>::max();
    }

    static pyvrp::Measure<Type> min()
    {
        return std::numeric_limits<pyvrp::Value>::min();
    }
};

#endif  // PYVRP_MEASURE_H
