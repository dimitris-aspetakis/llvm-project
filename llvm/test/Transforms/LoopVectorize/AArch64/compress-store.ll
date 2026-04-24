; RUN: opt -mtriple=aarch64-linux-gnu -mattr=+sve2 -passes=loop-vectorize -S \
; RUN:   < %s | FileCheck %s
; RUN: opt -mtriple=aarch64-linux-gnu -mattr=+sve2 -passes=loop-vectorize \
; RUN:   -force-vector-interleave=2 -S < %s | FileCheck %s --check-prefix=FORCE-IC

; Tests that the compress-store autovectorization fires on AArch64 with SVE2
; at scalable VFs. The recipe must produce a `llvm.masked.compressstore` on
; a `<vscale x N x i32>` value plus a `llvm.vector.reduce.add` over a zext'd
; predicate to count active lanes (which lowers to COMPACT + ST1 + CNTP).

; Canonical split-CFG shape, i32 index, i32 trip count, step 1.
; CHECK-LABEL: @compress_store_i32(
; CHECK:       vector.body:
; CHECK:         [[COMPRESS_IDX:%compress\.idx.*]] = phi i32
; CHECK:         call void @llvm.masked.compressstore.nxv{{[0-9]+}}i32(<vscale x {{[0-9]+}} x i32>
; CHECK:         [[POPCOUNT:%.+]] = call i{{[0-9]+}} @llvm.vector.reduce.add.nxv{{[0-9]+}}i8(
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

; i64 trip count with i32 write index. The GEP index for the input load is
; sext'd; the SCEV recognizer must peel that to match the header phi.
; CHECK-LABEL: @compress_store_i32_i64tc(
; CHECK:       vector.body:
; CHECK:         call void @llvm.masked.compressstore.nxv{{[0-9]+}}i32(
; CHECK:         call i{{[0-9]+}} @llvm.vector.reduce.add.nxv{{[0-9]+}}i8(
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

; Non-zero write-index start: SCEV produces {%start,+[%cond],1}<%for.body>
; and the recipe's initial scalar phi takes %start as the preheader value.
; CHECK-LABEL: @compress_store_nonzero_start(
; CHECK:       vector.body:
; CHECK:         [[COMPRESS_IDX3:%compress\.idx.*]] = phi i32{{.*}}[ %start, {{.*}}]
; CHECK:         call void @llvm.masked.compressstore.nxv{{[0-9]+}}i32(
; CHECK:         call i{{[0-9]+}} @llvm.vector.reduce.add.nxv{{[0-9]+}}i8(
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

; Threshold-comparison predicate (`in[i] > T`) reproduces the
; UnrollState::unrollHeaderPHIByUF crash on AArch64 SVE2 when the cost
; model picks IC > 1. The interleaving gate in selectInterleaveCount
; clamps IC to 1, so the vector loop body must contain only one masked
; compressstore. Forcing -force-vector-interleave=2 must NOT bypass the
; clamp; we re-CHECK that the second RUN line still produces a single
; compressstore (FORCE-IC prefix).
;
; CHECK-LABEL: @compress_store_threshold_filter(
; CHECK:       vector.body:
; CHECK:         call void @llvm.masked.compressstore.nxv{{[0-9]+}}i32(
; CHECK-NOT:     call void @llvm.masked.compressstore.nxv{{[0-9]+}}i32(
;
; FORCE-IC-LABEL: @compress_store_threshold_filter(
; FORCE-IC:       vector.body:
; FORCE-IC:         call void @llvm.masked.compressstore.nxv{{[0-9]+}}i32(
; FORCE-IC-NOT:     call void @llvm.masked.compressstore.nxv{{[0-9]+}}i32(
define i32 @compress_store_threshold_filter(ptr noalias %in, ptr noalias %out,
                                              i32 %N, i32 %T) {
entry:
  %cmp_entry = icmp sgt i32 %N, 0
  br i1 %cmp_entry, label %for.body.preheader, label %for.end

for.body.preheader:
  br label %for.body

for.body:
  %i = phi i32 [ 0, %for.body.preheader ], [ %i.next, %for.inc ]
  %j = phi i32 [ 0, %for.body.preheader ], [ %j.next, %for.inc ]
  %in_ptr = getelementptr inbounds i32, ptr %in, i32 %i
  %val = load i32, ptr %in_ptr, align 4
  %cond = icmp sgt i32 %val, %T
  br i1 %cond, label %if.then, label %if.end

if.then:
  %out_ptr = getelementptr inbounds i32, ptr %out, i32 %j
  store i32 %val, ptr %out_ptr, align 4
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
  %j.lcssa = phi i32 [ 0, %entry ], [ %j.next, %for.inc ]
  ret i32 %j.lcssa
}
