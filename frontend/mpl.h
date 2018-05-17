#ifndef mpl_h_included
#define mpl_h_included

#include <tinympl/detection.hpp>
#include <tinympl/reverse.hpp>
#include <tinympl/tuple_as_sequence.hpp>

namespace detail {

template <typename T>
using _has_member_type_Permissions_archetype = typename T::Permissions;

} // end namespace detail

template < class T >                                                              
struct has_member_type_Permissions
  : tinympl::is_detected<detail::_has_member_type_Permissions_archetype, T>
{ };

template <int N, class... Args>
struct reverse_tuple {
  using type_t = typename tinympl::reverse<std::tuple<Args...>>::type;
};


#endif

