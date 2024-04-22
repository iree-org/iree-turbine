builtin.module {

func.func @forward() {
  %i = torch.constant.int 1000
  torch.operator "torch.expand_custom_op_pass_test.int_arg"(%i) : (!torch.int) -> ()
  return
}

}
