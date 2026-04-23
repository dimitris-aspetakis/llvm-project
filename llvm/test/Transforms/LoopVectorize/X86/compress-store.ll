; RUN: opt -mattr=+avx512f -passes=loop-vectorize \
; RUN:   -force-vector-width=16 -force-vector-interleave=1 \
; RUN:   -S < %s | FileCheck %s

; Tests that conditional-store-to-packed-array loops are vectorized to
; llvm.masked.compressstore + llvm.vector.reduce.add. Recognition is SCEV-
; driven: the write-index phi is identified by its SCEVConditionalAddRecExpr.
;
;   int j = 0;
;   for (int i = 0; i < N; ++i)
;     if (cond[i])
;       c[j++] = a[i];

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Canonical split-CFG form (pre-simplifycfg): header phi, gate, take block,
; merge PHI in a separate block. i32 index, i32 trip count, step 1.
; CHECK-LABEL: @compress_store_i32(
; CHECK:       vector.body:
; CHECK:         [[COMPRESS_IDX:%compress\.idx.*]] = phi i32
; CHECK:         [[MASK:%.+]] = icmp
; CHECK:         [[WIDE_LOAD:%.+]] = call <16 x i32> @llvm.masked.load
; CHECK:         call void @llvm.masked.compressstore.v16i32(<16 x i32> [[WIDE_LOAD]], ptr{{.*}}, <16 x i1> [[MASK]])
; CHECK:         [[POPCOUNT:%.+]] = call i{{[0-9]+}} @llvm.vector.reduce.add.v{{[0-9]+}}i8(
; CHECK:         [[IDX_NEXT:%compress\.idx\.next.*]] = add{{.*}} i32 [[COMPRESS_IDX]], {{%.+}}
define void @compress_store_i32(ptr noalias %a, ptr noalias %cond_arr,
                                 ptr noalias %c, i32 %N) {
entry:
  %cmp_entry = icmp sgt i32 %N, 0
  br i1 %cmp_entry, label %for.body.preheader, label %for.end

for.body.preheader:
  br label %for.body

for.body:
  %i = phi i32 [ 0, %for.body.preheader ], [ %i.next, %for.inc ]
  %j = phi i32 [ 0, %for.body.preheader ], [ %j.next, %for.inc ]
  %cond_ptr = getelementptr inbounds i8, ptr %cond_arr, i32 %i
  %cond_byte = load i8, ptr %cond_ptr, align 1
  %cond = icmp ne i8 %cond_byte, 0
  br i1 %cond, label %if.then, label %if.end

if.then:
  %a_ptr = getelementptr inbounds i32, ptr %a, i32 %i
  %val = load i32, ptr %a_ptr, align 4
  %c_ptr = getelementptr inbounds i32, ptr %c, i32 %j
  store i32 %val, ptr %c_ptr, align 4
  %j.inc = add i32 %j, 1
  br label %if.end

if.end:
  %j.next = phi i32 [ %j, %for.body ], [ %j.inc, %if.then ]
  br label %for.inc

for.inc:
  %i.next = add nuw nsw i32 %i, 1
  %exitcond = icmp eq i32 %i.next, %N
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}

; 64-bit trip count with i32 write index. GEP index is sext'd; the recognizer
; must peel that to match the header phi.
; CHECK-LABEL: @compress_store_i32_i64tc(
; CHECK:       vector.body:
; CHECK:         [[COMPRESS_IDX2:%compress\.idx.*]] = phi i32
; CHECK:         call void @llvm.masked.compressstore.v16i32(
; CHECK:         call i{{[0-9]+}} @llvm.vector.reduce.add.v{{[0-9]+}}i8(
define void @compress_store_i32_i64tc(ptr noalias %a, ptr noalias %cond_arr,
                                       ptr noalias %c, i64 %N) {
entry:
  %cmp_entry = icmp sgt i64 %N, 0
  br i1 %cmp_entry, label %for.body.preheader, label %for.end

for.body.preheader:
  br label %for.body

for.body:
  %i = phi i64 [ 0, %for.body.preheader ], [ %i.next, %for.inc ]
  %j = phi i32 [ 0, %for.body.preheader ], [ %j.next, %for.inc ]
  %cond_ptr = getelementptr inbounds i8, ptr %cond_arr, i64 %i
  %cond_byte = load i8, ptr %cond_ptr, align 1
  %cond = icmp ne i8 %cond_byte, 0
  br i1 %cond, label %if.then, label %if.end

if.then:
  %a_ptr = getelementptr inbounds i32, ptr %a, i64 %i
  %val = load i32, ptr %a_ptr, align 4
  %c_ptr = getelementptr inbounds i32, ptr %c, i32 %j
  store i32 %val, ptr %c_ptr, align 4
  %j.inc = add i32 %j, 1
  br label %if.end

if.end:
  %j.next = phi i32 [ %j, %for.body ], [ %j.inc, %if.then ]
  br label %for.inc

for.inc:
  %i.next = add nuw nsw i64 %i, 1
  %exitcond = icmp eq i64 %i.next, %N
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}

; Non-zero start for the write index. SCEV produces
; {%start,+[%cond],1}<%for.body> and the recipe's initial scalar phi takes
; %start as the preheader value.
; CHECK-LABEL: @compress_store_nonzero_start(
; CHECK:       vector.body:
; CHECK:         [[COMPRESS_IDX3:%compress\.idx.*]] = phi i32{{.*}}[ %start, {{.*}}]
; CHECK:         call void @llvm.masked.compressstore.v16i32(
; CHECK:         call i{{[0-9]+}} @llvm.vector.reduce.add.v{{[0-9]+}}i8(
define void @compress_store_nonzero_start(ptr noalias %a, ptr noalias %cond_arr,
                                           ptr noalias %c, i32 %N,
                                           i32 %start) {
entry:
  %cmp_entry = icmp sgt i32 %N, 0
  br i1 %cmp_entry, label %for.body.preheader, label %for.end

for.body.preheader:
  br label %for.body

for.body:
  %i = phi i32 [ 0, %for.body.preheader ], [ %i.next, %for.inc ]
  %j = phi i32 [ %start, %for.body.preheader ], [ %j.next, %for.inc ]
  %cond_ptr = getelementptr inbounds i8, ptr %cond_arr, i32 %i
  %cond_byte = load i8, ptr %cond_ptr, align 1
  %cond = icmp ne i8 %cond_byte, 0
  br i1 %cond, label %if.then, label %if.end

if.then:
  %a_ptr = getelementptr inbounds i32, ptr %a, i32 %i
  %val = load i32, ptr %a_ptr, align 4
  %c_ptr = getelementptr inbounds i32, ptr %c, i32 %j
  store i32 %val, ptr %c_ptr, align 4
  %j.inc = add i32 %j, 1
  br label %if.end

if.end:
  %j.next = phi i32 [ %j, %for.body ], [ %j.inc, %if.then ]
  br label %for.inc

for.inc:
  %i.next = add nuw nsw i32 %i, 1
  %exitcond = icmp eq i32 %i.next, %N
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}

; Negative: step is 2, not 1. The SCEV recognizer's MVP requires a constant
; step of 1 so this must not vectorize to compressstore.
; CHECK-LABEL: @compress_store_step2_no_match(
; CHECK-NOT:     call void @llvm.masked.compressstore
define void @compress_store_step2_no_match(ptr noalias %a, ptr noalias %cond_arr,
                                            ptr noalias %c, i32 %N) {
entry:
  %cmp_entry = icmp sgt i32 %N, 0
  br i1 %cmp_entry, label %for.body.preheader, label %for.end

for.body.preheader:
  br label %for.body

for.body:
  %i = phi i32 [ 0, %for.body.preheader ], [ %i.next, %for.inc ]
  %j = phi i32 [ 0, %for.body.preheader ], [ %j.next, %for.inc ]
  %cond_ptr = getelementptr inbounds i8, ptr %cond_arr, i32 %i
  %cond_byte = load i8, ptr %cond_ptr, align 1
  %cond = icmp ne i8 %cond_byte, 0
  br i1 %cond, label %if.then, label %if.end

if.then:
  %a_ptr = getelementptr inbounds i32, ptr %a, i32 %i
  %val = load i32, ptr %a_ptr, align 4
  %c_ptr = getelementptr inbounds i32, ptr %c, i32 %j
  store i32 %val, ptr %c_ptr, align 4
  %j.inc = add i32 %j, 2
  br label %if.end

if.end:
  %j.next = phi i32 [ %j, %for.body ], [ %j.inc, %if.then ]
  br label %for.inc

for.inc:
  %i.next = add nuw nsw i32 %i, 1
  %exitcond = icmp eq i32 %i.next, %N
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}

; Negative: both arms of the merge phi update the index (neither is the
; unchanged case). SCEV leaves j as SCEVUnknown so recognition bails.
; CHECK-LABEL: @compress_store_both_advance_no_match(
; CHECK-NOT:     call void @llvm.masked.compressstore
define void @compress_store_both_advance_no_match(ptr noalias %a,
                                                   ptr noalias %cond_arr,
                                                   ptr noalias %c, i32 %N) {
entry:
  %cmp_entry = icmp sgt i32 %N, 0
  br i1 %cmp_entry, label %for.body.preheader, label %for.end

for.body.preheader:
  br label %for.body

for.body:
  %i = phi i32 [ 0, %for.body.preheader ], [ %i.next, %for.inc ]
  %j = phi i32 [ 0, %for.body.preheader ], [ %j.next, %for.inc ]
  %cond_ptr = getelementptr inbounds i8, ptr %cond_arr, i32 %i
  %cond_byte = load i8, ptr %cond_ptr, align 1
  %cond = icmp ne i8 %cond_byte, 0
  br i1 %cond, label %if.then, label %if.else

if.then:
  %a_ptr = getelementptr inbounds i32, ptr %a, i32 %i
  %val = load i32, ptr %a_ptr, align 4
  %c_ptr = getelementptr inbounds i32, ptr %c, i32 %j
  store i32 %val, ptr %c_ptr, align 4
  %j.inc1 = add i32 %j, 1
  br label %if.end

if.else:
  %j.inc2 = add i32 %j, 3
  br label %if.end

if.end:
  %j.next = phi i32 [ %j.inc1, %if.then ], [ %j.inc2, %if.else ]
  br label %for.inc

for.inc:
  %i.next = add nuw nsw i32 %i, 1
  %exitcond = icmp eq i32 %i.next, %N
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}

; Negative: merge phi has three incoming values (from a switch). The SCEV
; producer declines to recognize the conditional IV, and even if it did the
; recognizer requires getNumIncomingValues() == 2 on the merge phi.
; CHECK-LABEL: @compress_store_three_preds_no_match(
; CHECK-NOT:     call void @llvm.masked.compressstore
define void @compress_store_three_preds_no_match(ptr noalias %a,
                                                  ptr noalias %cond_arr,
                                                  ptr noalias %c, i32 %N) {
entry:
  %cmp_entry = icmp sgt i32 %N, 0
  br i1 %cmp_entry, label %for.body.preheader, label %for.end

for.body.preheader:
  br label %for.body

for.body:
  %i = phi i32 [ 0, %for.body.preheader ], [ %i.next, %for.inc ]
  %j = phi i32 [ 0, %for.body.preheader ], [ %j.next, %for.inc ]
  switch i32 %i, label %skip [
    i32 0, label %take1
    i32 1, label %take2
  ]

take1:
  %a1 = getelementptr inbounds i32, ptr %a, i32 %i
  %v1 = load i32, ptr %a1, align 4
  %c1 = getelementptr inbounds i32, ptr %c, i32 %j
  store i32 %v1, ptr %c1, align 4
  %j.inc1 = add i32 %j, 1
  br label %merge

take2:
  %a2 = getelementptr inbounds i32, ptr %a, i32 %i
  %v2 = load i32, ptr %a2, align 4
  %c2 = getelementptr inbounds i32, ptr %c, i32 %j
  store i32 %v2, ptr %c2, align 4
  %j.inc2 = add i32 %j, 1
  br label %merge

skip:
  br label %merge

merge:
  %j.next = phi i32 [ %j.inc1, %take1 ], [ %j.inc2, %take2 ], [ %j, %skip ]
  br label %for.inc

for.inc:
  %i.next = add nuw nsw i32 %i, 1
  %exitcond = icmp eq i32 %i.next, %N
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}
