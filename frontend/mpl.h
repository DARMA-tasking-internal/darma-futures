#ifndef mpl_h_included
#define mpl_h_included

template < class T >                                                              
class HasMemberType_Permissions
{                                                                                 
 private:                                                                          
  using Yes = char[2];                                                          
  using  No = char[1];                                                          

  template < class U >                                                          
  static Yes& test ( typename U::Permissions* );                                        
  template < typename U >                                                       
  static No& test ( U* );                                                      

public:                                                                           
  static constexpr bool RESULT = sizeof(test<T>(nullptr)) == sizeof(Yes); 
};                                                                                

template < class T >                                                              
struct has_member_type_Permissions
: public std::integral_constant<bool, HasMemberType_Permissions<T>::RESULT>            
{ };  

template <int N, class T, class... Args>
struct reverse_tuple {
  using type_t = typename reverse_tuple<N-1,Args...,T>::type_t;
};

template <class T, class... Args>
struct reverse_tuple<1, T, Args...> {
  using type_t = std::tuple<Args...,T>;
};

#endif

