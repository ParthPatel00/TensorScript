; ModuleID = 'fuse'
source_filename = "fuse"

; Function Attrs: nofree norecurse nosync nounwind memory(read, argmem: readwrite, inaccessiblemem: none, target_mem0: none, target_mem1: none)
define void @kernel_0(ptr noalias readonly captures(none) %inputs, i32 %0, ptr noalias writeonly captures(none) %out, i64 %n) local_unnamed_addr #0 {
entry:
  %empty = icmp eq i64 %n, 0
  br i1 %empty, label %exit, label %preheader

preheader:                                        ; preds = %entry
  %base0 = load ptr, ptr %inputs, align 8
  %slot1 = getelementptr i8, ptr %inputs, i64 8
  %base1 = load ptr, ptr %slot1, align 8
  br label %loop

loop:                                             ; preds = %loop, %preheader
  %i = phi i64 [ 0, %preheader ], [ %i_next, %loop ]
  %gep_in0 = getelementptr float, ptr %base0, i64 %i
  %in0 = load float, ptr %gep_in0, align 4
  %gep_in1 = getelementptr float, ptr %base1, i64 %i
  %in1 = load float, ptr %gep_in1, align 4
  %add = fadd float %in0, %in1
  %relu = tail call float @llvm.maxnum.f32(float %add, float 0.000000e+00)
  %mul = fmul float %in1, %relu
  %gep_out = getelementptr float, ptr %out, i64 %i
  store float %mul, ptr %gep_out, align 4
  %i_next = add nuw i64 %i, 1
  %done = icmp eq i64 %i_next, %n
  br i1 %done, label %exit, label %loop, !llvm.loop !0

exit:                                             ; preds = %loop, %entry
  ret void
}

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.maxnum.f32(float, float) #1

; Function Attrs: nofree norecurse nosync nounwind memory(read, argmem: readwrite, inaccessiblemem: none, target_mem0: none, target_mem1: none)
define void @kernel_1(ptr noalias readonly captures(none) %inputs, i32 %0, ptr noalias writeonly captures(none) %out, i64 %n) local_unnamed_addr #0 {
entry:
  %empty = icmp eq i64 %n, 0
  br i1 %empty, label %exit, label %preheader

preheader:                                        ; preds = %entry
  %base0 = load ptr, ptr %inputs, align 8
  br label %loop

loop:                                             ; preds = %loop, %preheader
  %i = phi i64 [ 0, %preheader ], [ %i_next, %loop ]
  %gep_in0 = getelementptr float, ptr %base0, i64 %i
  %in0 = load float, ptr %gep_in0, align 4
  %neg = fneg float %in0
  %gep_out = getelementptr float, ptr %out, i64 %i
  store float %neg, ptr %gep_out, align 4
  %i_next = add nuw i64 %i, 1
  %done = icmp eq i64 %i_next, %n
  br i1 %done, label %exit, label %loop, !llvm.loop !0

exit:                                             ; preds = %loop, %entry
  ret void
}

; Function Attrs: nofree norecurse nosync nounwind memory(read, argmem: readwrite, inaccessiblemem: none, target_mem0: none, target_mem1: none)
define void @kernel_2(ptr noalias readonly captures(none) %inputs, i32 %0, ptr noalias writeonly captures(none) %out, i64 %n) local_unnamed_addr #0 {
entry:
  %empty = icmp eq i64 %n, 0
  br i1 %empty, label %exit, label %preheader

preheader:                                        ; preds = %entry
  %base0 = load ptr, ptr %inputs, align 8
  br label %loop

loop:                                             ; preds = %loop, %preheader
  %i = phi i64 [ 0, %preheader ], [ %i_next, %loop ]
  %gep_in0 = getelementptr float, ptr %base0, i64 %i
  %in0 = load float, ptr %gep_in0, align 4
  %sqrt = tail call float @llvm.sqrt.f32(float %in0)
  %gep_out = getelementptr float, ptr %out, i64 %i
  store float %sqrt, ptr %gep_out, align 4
  %i_next = add nuw i64 %i, 1
  %done = icmp eq i64 %i_next, %n
  br i1 %done, label %exit, label %loop, !llvm.loop !0

exit:                                             ; preds = %loop, %entry
  ret void
}

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.sqrt.f32(float) #1

attributes #0 = { nofree norecurse nosync nounwind memory(read, argmem: readwrite, inaccessiblemem: none, target_mem0: none, target_mem1: none) }
attributes #1 = { mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }

!0 = !{!1}
!1 = !{!"llvm.loop.vectorize.enable", i1 true}
