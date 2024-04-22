builtin.module {

func.func @forward() {
  %str = torch.constant.str "TEST_VALUE"
  torch.operator "torch.expand_custom_op_pass_test.print_string_attr"(%str) : (!torch.str) -> ()
  return  
}

}
