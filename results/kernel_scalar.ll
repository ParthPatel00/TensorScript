; ModuleID = 'tensorscript'
source_filename = "tensorscript"

; Function Attrs: nounwind
define void @kernel_0(ptr noalias %inputs, i32 %0, ptr noalias %out, i64 %n) #0 {
entry:
  %empty = icmp eq i64 %n, 0
  br i1 %empty, label %exit, label %preheader

preheader:                                        ; preds = %entry
  %slot = getelementptr ptr, ptr %inputs, i32 0
  %base0 = load ptr, ptr %slot, align 8
  %slot1 = getelementptr ptr, ptr %inputs, i32 1
  %base1 = load ptr, ptr %slot1, align 8
  br label %loop

loop:                                             ; preds = %loop, %preheader
  %i = phi i64 [ 0, %preheader ], [ %i_next, %loop ]
  %gep_in0 = getelementptr float, ptr %base0, i64 %i
  %in0 = load float, ptr %gep_in0, align 4
  %gep_in1 = getelementptr float, ptr %base1, i64 %i
  %in1 = load float, ptr %gep_in1, align 4
  %add = fadd float %in0, %in1
  %relu = call float @llvm.maxnum.f32(float %add, float 0.000000e+00)
  %mul = fmul float %relu, %in1
  %neg_x = fneg float %mul
  %exp_neg = call float @llvm.exp.f32(float %neg_x)
  %denom = fadd float 1.000000e+00, %exp_neg
  %sigmoid = fdiv float 1.000000e+00, %denom
  %two_x = fmul float 2.000000e+00, %sigmoid
  %exp2x = call float @llvm.exp.f32(float %two_x)
  %tanh_num = fsub float %exp2x, 1.000000e+00
  %tanh_den = fadd float %exp2x, 1.000000e+00
  %tanh = fdiv float %tanh_num, %tanh_den
  %gep_out = getelementptr float, ptr %out, i64 %i
  store float %tanh, ptr %gep_out, align 4
  %i_next = add i64 %i, 1
  %done = icmp eq i64 %i_next, %n
  br i1 %done, label %exit, label %loop, !llvm.loop !0

exit:                                             ; preds = %loop, %entry
  ret void
}

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.maxnum.f32(float, float) #1

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.exp.f32(float) #1

attributes #0 = { nounwind }
attributes #1 = { nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }

!0 = !{!1}
!1 = !{!"llvm.loop.vectorize.enable", i1 true}
