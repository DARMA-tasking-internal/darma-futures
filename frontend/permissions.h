
class Modify;
class ReadOnly;
class Idempotent;
class None;

template <class T, class U>
struct min_permissions {
  using type_t = T; //default equal
};

template <>
struct min_permissions<ReadOnly,None> {
  using type_t = None; 
};

template <>
struct min_permissions<None,ReadOnly> {
  using type_t = None;
};

template <>
struct min_permissions<Idempotent,None> {
  using type_t = None;
};

template <>
struct min_permissions<None,Idempotent> {
  using type_t = None;
};

template <>
struct min_permissions<Modify,None> {
  using type_t = None;
};

template <>
struct min_permissions<None,Modify> {
  using type_t = None;
};

template <>
struct min_permissions<Idempotent,ReadOnly> {
  using type_t = ReadOnly;
};

template <>
struct min_permissions<ReadOnly,Idempotent> {
  using type_t = ReadOnly;
};

template <>
struct min_permissions<Modify,ReadOnly> {
  using type_t = ReadOnly;
};

template <>
struct min_permissions<ReadOnly,Modify> {
  using type_t = ReadOnly;
};

template <>
struct min_permissions<Idempotent,Modify> {
  using type_t = Idempotent;
};

template <>
struct min_permissions<Modify,Idempotent> {
  using type_t = Idempotent;
};

