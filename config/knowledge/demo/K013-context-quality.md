# K013 VehicleMind 上下文质量

VehicleMind 把观测字段区分为 KNOWN、UNKNOWN、MISSING、INVALID 和 STALE。过期或无效观测不得被回答为当前真实车况；知识库中的静态说明也不能补造实时传感器读数。
