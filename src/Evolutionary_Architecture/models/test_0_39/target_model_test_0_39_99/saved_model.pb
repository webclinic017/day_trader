ЦЙ&
░Б
D
AddV2
x"T
y"T
z"T"
Ttype:
2	АР
B
AssignVariableOp
resource
value"dtype"
dtypetypeИ
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
н
GatherV2
params"Tparams
indices"Tindices
axis"Taxis
output"Tparams"

batch_dimsint "
Tparamstype"
Tindicestype:
2	"
Taxistype:
2	
.
Identity

input"T
output"T"	
Ttype
:
Less
x"T
y"T
z
"
Ttype:
2	
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(И
?
Mul
x"T
y"T
z"T"
Ttype:
2	Р

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
Н
Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
@
ReadVariableOp
resource
value"dtype"
dtypetypeИ
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
9
Softmax
logits"T
softmax"T"
Ttype:
2
[
Split
	split_dim

value"T
output"T*	num_split"
	num_splitint(0"	
Ttype
╛
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring И
@
StaticRegexFullMatch	
input

output
"
patternstring
Ў
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
л
TensorListFromTensor
tensor"element_dtype
element_shape"
shape_type*
output_handleКщelement_dtype"
element_dtypetype"

shape_typetype:
2	
Ъ
TensorListReserve
element_shape"
shape_type
num_elements#
handleКщelement_dtype"
element_dtypetype"

shape_typetype:
2	
И
TensorListStack
input_handle
element_shape
tensor"element_dtype"
element_dtypetype" 
num_elementsint         
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
Ц
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 И
Ф
While

input2T
output2T"
T
list(type)("
condfunc"
bodyfunc" 
output_shapeslist(shape)
 "
parallel_iterationsint
И"serve*2.6.02v2.6.0-rc2-32-g919f693420e8н╢$
y
dense_9/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	╬*
shared_namedense_9/kernel
r
"dense_9/kernel/Read/ReadVariableOpReadVariableOpdense_9/kernel*
_output_shapes
:	╬*
dtype0
p
dense_9/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_9/bias
i
 dense_9/bias/Read/ReadVariableOpReadVariableOpdense_9/bias*
_output_shapes
:*
dtype0
У
lstm_18/lstm_cell_18/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	]а*,
shared_namelstm_18/lstm_cell_18/kernel
М
/lstm_18/lstm_cell_18/kernel/Read/ReadVariableOpReadVariableOplstm_18/lstm_cell_18/kernel*
_output_shapes
:	]а*
dtype0
з
%lstm_18/lstm_cell_18/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	hа*6
shared_name'%lstm_18/lstm_cell_18/recurrent_kernel
а
9lstm_18/lstm_cell_18/recurrent_kernel/Read/ReadVariableOpReadVariableOp%lstm_18/lstm_cell_18/recurrent_kernel*
_output_shapes
:	hа*
dtype0
Л
lstm_18/lstm_cell_18/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:а**
shared_namelstm_18/lstm_cell_18/bias
Д
-lstm_18/lstm_cell_18/bias/Read/ReadVariableOpReadVariableOplstm_18/lstm_cell_18/bias*
_output_shapes	
:а*
dtype0
У
lstm_19/lstm_cell_19/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	h╕
*,
shared_namelstm_19/lstm_cell_19/kernel
М
/lstm_19/lstm_cell_19/kernel/Read/ReadVariableOpReadVariableOplstm_19/lstm_cell_19/kernel*
_output_shapes
:	h╕
*
dtype0
и
%lstm_19/lstm_cell_19/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
╬╕
*6
shared_name'%lstm_19/lstm_cell_19/recurrent_kernel
б
9lstm_19/lstm_cell_19/recurrent_kernel/Read/ReadVariableOpReadVariableOp%lstm_19/lstm_cell_19/recurrent_kernel* 
_output_shapes
:
╬╕
*
dtype0
Л
lstm_19/lstm_cell_19/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:╕
**
shared_namelstm_19/lstm_cell_19/bias
Д
-lstm_19/lstm_cell_19/bias/Read/ReadVariableOpReadVariableOplstm_19/lstm_cell_19/bias*
_output_shapes	
:╕
*
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0

NoOpNoOp
е"
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*р!
value╓!B╙! B╠!
А
layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
	optimizer
trainable_variables
	variables
	regularization_losses

	keras_api

signatures
l
cell

state_spec
trainable_variables
	variables
regularization_losses
	keras_api
R
trainable_variables
	variables
regularization_losses
	keras_api
l
cell

state_spec
trainable_variables
	variables
regularization_losses
	keras_api
R
trainable_variables
	variables
regularization_losses
	keras_api
h

 kernel
!bias
"trainable_variables
#	variables
$regularization_losses
%	keras_api
 
8
&0
'1
(2
)3
*4
+5
 6
!7
8
&0
'1
(2
)3
*4
+5
 6
!7
 
н
trainable_variables
,layer_metrics
-layer_regularization_losses
	variables

.layers
/non_trainable_variables
	regularization_losses
0metrics
 
О
1
state_size

&kernel
'recurrent_kernel
(bias
2trainable_variables
3	variables
4regularization_losses
5	keras_api
 

&0
'1
(2

&0
'1
(2
 
╣
trainable_variables
6layer_metrics
7layer_regularization_losses
	variables

8layers
9non_trainable_variables
regularization_losses

:states
;metrics
 
 
 
н
trainable_variables
<layer_metrics
=layer_regularization_losses
	variables

>layers
?non_trainable_variables
regularization_losses
@metrics
О
A
state_size

)kernel
*recurrent_kernel
+bias
Btrainable_variables
C	variables
Dregularization_losses
E	keras_api
 

)0
*1
+2

)0
*1
+2
 
╣
trainable_variables
Flayer_metrics
Glayer_regularization_losses
	variables

Hlayers
Inon_trainable_variables
regularization_losses

Jstates
Kmetrics
 
 
 
н
trainable_variables
Llayer_metrics
Mlayer_regularization_losses
	variables

Nlayers
Onon_trainable_variables
regularization_losses
Pmetrics
ZX
VARIABLE_VALUEdense_9/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_9/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

 0
!1

 0
!1
 
н
"trainable_variables
Qlayer_metrics
Rlayer_regularization_losses
#	variables

Slayers
Tnon_trainable_variables
$regularization_losses
Umetrics
a_
VARIABLE_VALUElstm_18/lstm_cell_18/kernel0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUE
ki
VARIABLE_VALUE%lstm_18/lstm_cell_18/recurrent_kernel0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUElstm_18/lstm_cell_18/bias0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUElstm_19/lstm_cell_19/kernel0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUE
ki
VARIABLE_VALUE%lstm_19/lstm_cell_19/recurrent_kernel0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUElstm_19/lstm_cell_19/bias0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUE
 
 
#
0
1
2
3
4
 

V0
W1
 

&0
'1
(2

&0
'1
(2
 
н
2trainable_variables
Xlayer_metrics
Ylayer_regularization_losses
3	variables

Zlayers
[non_trainable_variables
4regularization_losses
\metrics
 
 

0
 
 
 
 
 
 
 
 
 

)0
*1
+2

)0
*1
+2
 
н
Btrainable_variables
]layer_metrics
^layer_regularization_losses
C	variables

_layers
`non_trainable_variables
Dregularization_losses
ametrics
 
 

0
 
 
 
 
 
 
 
 
 
 
 
 
 
4
	btotal
	ccount
d	variables
e	keras_api
D
	ftotal
	gcount
h
_fn_kwargs
i	variables
j	keras_api
 
 
 
 
 
 
 
 
 
 
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

b0
c1

d	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

f0
g1

i	variables
И
serving_default_lstm_18_inputPlaceholder*+
_output_shapes
:         ]*
dtype0* 
shape:         ]
к
StatefulPartitionedCallStatefulPartitionedCallserving_default_lstm_18_inputlstm_18/lstm_cell_18/kernel%lstm_18/lstm_cell_18/recurrent_kernellstm_18/lstm_cell_18/biaslstm_19/lstm_cell_19/kernel%lstm_19/lstm_cell_19/recurrent_kernellstm_19/lstm_cell_19/biasdense_9/kerneldense_9/bias*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         **
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8В */
f*R(
&__inference_signature_wrapper_32588583
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
Х
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename"dense_9/kernel/Read/ReadVariableOp dense_9/bias/Read/ReadVariableOp/lstm_18/lstm_cell_18/kernel/Read/ReadVariableOp9lstm_18/lstm_cell_18/recurrent_kernel/Read/ReadVariableOp-lstm_18/lstm_cell_18/bias/Read/ReadVariableOp/lstm_19/lstm_cell_19/kernel/Read/ReadVariableOp9lstm_19/lstm_cell_19/recurrent_kernel/Read/ReadVariableOp-lstm_19/lstm_cell_19/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOpConst*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В **
f%R#
!__inference__traced_save_32590938
а
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_9/kerneldense_9/biaslstm_18/lstm_cell_18/kernel%lstm_18/lstm_cell_18/recurrent_kernellstm_18/lstm_cell_18/biaslstm_19/lstm_cell_19/kernel%lstm_19/lstm_cell_19/recurrent_kernellstm_19/lstm_cell_19/biastotalcounttotal_1count_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *-
f(R&
$__inference__traced_restore_32590984╪ы#
╣F
Н
E__inference_lstm_18_layer_call_and_return_conditional_losses_32586524

inputs(
lstm_cell_18_32586442:	]а(
lstm_cell_18_32586444:	hа$
lstm_cell_18_32586446:	а
identityИв$lstm_cell_18/StatefulPartitionedCallвwhileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :h2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :h2
zeros/packed/1Г
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:         h2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :h2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :h2
zeros_1/packed/1Й
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:         h2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permГ
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :                  ]2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ю
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1Е
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         2
TensorArrayV2/element_shape▓
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2┐
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ]   27
5TensorArrayUnstack/TensorListFromTensor/element_shape°
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2№
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         ]*
shrink_axis_mask2
strided_slice_2е
$lstm_cell_18/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_18_32586442lstm_cell_18_32586444lstm_cell_18_32586446*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:         h:         h:         h*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_lstm_cell_18_layer_call_and_return_conditional_losses_325864412&
$lstm_cell_18/StatefulPartitionedCallП
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    h   2
TensorArrayV2_1/element_shape╕
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter╩
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_18_32586442lstm_cell_18_32586444lstm_cell_18_32586446*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         h:         h: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_32586455*
condR
while_cond_32586454*K
output_shapes:
8: : : : :         h:         h: : : : : *
parallel_iterations 2
while╡
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    h   22
0TensorArrayV2Stack/TensorListStack/element_shapeё
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :                  h*
element_dtype02$
"TensorArrayV2Stack/TensorListStackБ
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2Ъ
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         h*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permо
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :                  h2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimew
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :                  h2

Identity}
NoOpNoOp%^lstm_cell_18/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  ]: : : 2L
$lstm_cell_18/StatefulPartitionedCall$lstm_cell_18/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :                  ]
 
_user_specified_nameinputs
Ъ?
╥
while_body_32587699
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_18_matmul_readvariableop_resource_0:	]аH
5while_lstm_cell_18_matmul_1_readvariableop_resource_0:	hаC
4while_lstm_cell_18_biasadd_readvariableop_resource_0:	а
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_18_matmul_readvariableop_resource:	]аF
3while_lstm_cell_18_matmul_1_readvariableop_resource:	hаA
2while_lstm_cell_18_biasadd_readvariableop_resource:	аИв)while/lstm_cell_18/BiasAdd/ReadVariableOpв(while/lstm_cell_18/MatMul/ReadVariableOpв*while/lstm_cell_18/MatMul_1/ReadVariableOp├
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ]   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape╙
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         ]*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem╔
(while/lstm_cell_18/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_18_matmul_readvariableop_resource_0*
_output_shapes
:	]а*
dtype02*
(while/lstm_cell_18/MatMul/ReadVariableOp╫
while/lstm_cell_18/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_18/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         а2
while/lstm_cell_18/MatMul╧
*while/lstm_cell_18/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_18_matmul_1_readvariableop_resource_0*
_output_shapes
:	hа*
dtype02,
*while/lstm_cell_18/MatMul_1/ReadVariableOp└
while/lstm_cell_18/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_18/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         а2
while/lstm_cell_18/MatMul_1╕
while/lstm_cell_18/addAddV2#while/lstm_cell_18/MatMul:product:0%while/lstm_cell_18/MatMul_1:product:0*
T0*(
_output_shapes
:         а2
while/lstm_cell_18/add╚
)while/lstm_cell_18/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_18_biasadd_readvariableop_resource_0*
_output_shapes	
:а*
dtype02+
)while/lstm_cell_18/BiasAdd/ReadVariableOp┼
while/lstm_cell_18/BiasAddBiasAddwhile/lstm_cell_18/add:z:01while/lstm_cell_18/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         а2
while/lstm_cell_18/BiasAddК
"while/lstm_cell_18/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_18/split/split_dimЛ
while/lstm_cell_18/splitSplit+while/lstm_cell_18/split/split_dim:output:0#while/lstm_cell_18/BiasAdd:output:0*
T0*`
_output_shapesN
L:         h:         h:         h:         h*
	num_split2
while/lstm_cell_18/splitШ
while/lstm_cell_18/SigmoidSigmoid!while/lstm_cell_18/split:output:0*
T0*'
_output_shapes
:         h2
while/lstm_cell_18/SigmoidЬ
while/lstm_cell_18/Sigmoid_1Sigmoid!while/lstm_cell_18/split:output:1*
T0*'
_output_shapes
:         h2
while/lstm_cell_18/Sigmoid_1а
while/lstm_cell_18/mulMul while/lstm_cell_18/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:         h2
while/lstm_cell_18/mulП
while/lstm_cell_18/ReluRelu!while/lstm_cell_18/split:output:2*
T0*'
_output_shapes
:         h2
while/lstm_cell_18/Relu┤
while/lstm_cell_18/mul_1Mulwhile/lstm_cell_18/Sigmoid:y:0%while/lstm_cell_18/Relu:activations:0*
T0*'
_output_shapes
:         h2
while/lstm_cell_18/mul_1й
while/lstm_cell_18/add_1AddV2while/lstm_cell_18/mul:z:0while/lstm_cell_18/mul_1:z:0*
T0*'
_output_shapes
:         h2
while/lstm_cell_18/add_1Ь
while/lstm_cell_18/Sigmoid_2Sigmoid!while/lstm_cell_18/split:output:3*
T0*'
_output_shapes
:         h2
while/lstm_cell_18/Sigmoid_2О
while/lstm_cell_18/Relu_1Reluwhile/lstm_cell_18/add_1:z:0*
T0*'
_output_shapes
:         h2
while/lstm_cell_18/Relu_1╕
while/lstm_cell_18/mul_2Mul while/lstm_cell_18/Sigmoid_2:y:0'while/lstm_cell_18/Relu_1:activations:0*
T0*'
_output_shapes
:         h2
while/lstm_cell_18/mul_2р
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_18/mul_2:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1k
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity~
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_1m
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_2Ъ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3Н
while/Identity_4Identitywhile/lstm_cell_18/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:         h2
while/Identity_4Н
while/Identity_5Identitywhile/lstm_cell_18/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:         h2
while/Identity_5▐

while/NoOpNoOp*^while/lstm_cell_18/BiasAdd/ReadVariableOp)^while/lstm_cell_18/MatMul/ReadVariableOp+^while/lstm_cell_18/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_18_biasadd_readvariableop_resource4while_lstm_cell_18_biasadd_readvariableop_resource_0"l
3while_lstm_cell_18_matmul_1_readvariableop_resource5while_lstm_cell_18_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_18_matmul_readvariableop_resource3while_lstm_cell_18_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         h:         h: : : : : 2V
)while/lstm_cell_18/BiasAdd/ReadVariableOp)while/lstm_cell_18/BiasAdd/ReadVariableOp2T
(while/lstm_cell_18/MatMul/ReadVariableOp(while/lstm_cell_18/MatMul/ReadVariableOp2X
*while/lstm_cell_18/MatMul_1/ReadVariableOp*while/lstm_cell_18/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         h:-)
'
_output_shapes
:         h:

_output_shapes
: :

_output_shapes
: 
▀
═
while_cond_32589359
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_32589359___redundant_placeholder06
2while_while_cond_32589359___redundant_placeholder16
2while_while_cond_32589359___redundant_placeholder26
2while_while_cond_32589359___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :         h:         h: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         h:-)
'
_output_shapes
:         h:

_output_shapes
: :

_output_shapes
:
к

╤
/__inference_sequential_9_layer_call_fn_32588510
lstm_18_input
unknown:	]а
	unknown_0:	hа
	unknown_1:	а
	unknown_2:	h╕

	unknown_3:
╬╕

	unknown_4:	╕

	unknown_5:	╬
	unknown_6:
identityИвStatefulPartitionedCall╙
StatefulPartitionedCallStatefulPartitionedCalllstm_18_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         **
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_sequential_9_layer_call_and_return_conditional_losses_325884702
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:         2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':         ]: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
+
_output_shapes
:         ]
'
_user_specified_namelstm_18_input
у
═
while_cond_32587863
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_32587863___redundant_placeholder06
2while_while_cond_32587863___redundant_placeholder16
2while_while_cond_32587863___redundant_placeholder26
2while_while_cond_32587863___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :         ╬:         ╬: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:         ╬:.*
(
_output_shapes
:         ╬:

_output_shapes
: :

_output_shapes
:
Й
f
H__inference_dropout_19_layer_call_and_return_conditional_losses_32590621

inputs

identity_1_
IdentityIdentityinputs*
T0*,
_output_shapes
:         ╬2

Identityn

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:         ╬2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ╬:T P
,
_output_shapes
:         ╬
 
_user_specified_nameinputs
д
╡
J__inference_sequential_9_layer_call_and_return_conditional_losses_32588470

inputs#
lstm_18_32588448:	]а#
lstm_18_32588450:	hа
lstm_18_32588452:	а#
lstm_19_32588456:	h╕
$
lstm_19_32588458:
╬╕

lstm_19_32588460:	╕
#
dense_9_32588464:	╬
dense_9_32588466:
identityИвdense_9/StatefulPartitionedCallв"dropout_18/StatefulPartitionedCallв"dropout_19/StatefulPartitionedCallвlstm_18/StatefulPartitionedCallвlstm_19/StatefulPartitionedCallн
lstm_18/StatefulPartitionedCallStatefulPartitionedCallinputslstm_18_32588448lstm_18_32588450lstm_18_32588452*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         h*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_lstm_18_layer_call_and_return_conditional_losses_325884132!
lstm_18/StatefulPartitionedCallЪ
"dropout_18/StatefulPartitionedCallStatefulPartitionedCall(lstm_18/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         h* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_dropout_18_layer_call_and_return_conditional_losses_325882462$
"dropout_18/StatefulPartitionedCall╙
lstm_19/StatefulPartitionedCallStatefulPartitionedCall+dropout_18/StatefulPartitionedCall:output:0lstm_19_32588456lstm_19_32588458lstm_19_32588460*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ╬*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_lstm_19_layer_call_and_return_conditional_losses_325882172!
lstm_19/StatefulPartitionedCall└
"dropout_19/StatefulPartitionedCallStatefulPartitionedCall(lstm_19/StatefulPartitionedCall:output:0#^dropout_18/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ╬* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_dropout_19_layer_call_and_return_conditional_losses_325880502$
"dropout_19/StatefulPartitionedCall╛
dense_9/StatefulPartitionedCallStatefulPartitionedCall+dropout_19/StatefulPartitionedCall:output:0dense_9_32588464dense_9_32588466*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_dense_9_layer_call_and_return_conditional_losses_325879942!
dense_9/StatefulPartitionedCallЗ
IdentityIdentity(dense_9/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:         2

Identity■
NoOpNoOp ^dense_9/StatefulPartitionedCall#^dropout_18/StatefulPartitionedCall#^dropout_19/StatefulPartitionedCall ^lstm_18/StatefulPartitionedCall ^lstm_19/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':         ]: : : : : : : : 2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall2H
"dropout_18/StatefulPartitionedCall"dropout_18/StatefulPartitionedCall2H
"dropout_19/StatefulPartitionedCall"dropout_19/StatefulPartitionedCall2B
lstm_18/StatefulPartitionedCalllstm_18/StatefulPartitionedCall2B
lstm_19/StatefulPartitionedCalllstm_19/StatefulPartitionedCall:S O
+
_output_shapes
:         ]
 
_user_specified_nameinputs
Ъ?
╥
while_body_32589813
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_18_matmul_readvariableop_resource_0:	]аH
5while_lstm_cell_18_matmul_1_readvariableop_resource_0:	hаC
4while_lstm_cell_18_biasadd_readvariableop_resource_0:	а
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_18_matmul_readvariableop_resource:	]аF
3while_lstm_cell_18_matmul_1_readvariableop_resource:	hаA
2while_lstm_cell_18_biasadd_readvariableop_resource:	аИв)while/lstm_cell_18/BiasAdd/ReadVariableOpв(while/lstm_cell_18/MatMul/ReadVariableOpв*while/lstm_cell_18/MatMul_1/ReadVariableOp├
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ]   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape╙
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         ]*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem╔
(while/lstm_cell_18/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_18_matmul_readvariableop_resource_0*
_output_shapes
:	]а*
dtype02*
(while/lstm_cell_18/MatMul/ReadVariableOp╫
while/lstm_cell_18/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_18/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         а2
while/lstm_cell_18/MatMul╧
*while/lstm_cell_18/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_18_matmul_1_readvariableop_resource_0*
_output_shapes
:	hа*
dtype02,
*while/lstm_cell_18/MatMul_1/ReadVariableOp└
while/lstm_cell_18/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_18/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         а2
while/lstm_cell_18/MatMul_1╕
while/lstm_cell_18/addAddV2#while/lstm_cell_18/MatMul:product:0%while/lstm_cell_18/MatMul_1:product:0*
T0*(
_output_shapes
:         а2
while/lstm_cell_18/add╚
)while/lstm_cell_18/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_18_biasadd_readvariableop_resource_0*
_output_shapes	
:а*
dtype02+
)while/lstm_cell_18/BiasAdd/ReadVariableOp┼
while/lstm_cell_18/BiasAddBiasAddwhile/lstm_cell_18/add:z:01while/lstm_cell_18/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         а2
while/lstm_cell_18/BiasAddК
"while/lstm_cell_18/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_18/split/split_dimЛ
while/lstm_cell_18/splitSplit+while/lstm_cell_18/split/split_dim:output:0#while/lstm_cell_18/BiasAdd:output:0*
T0*`
_output_shapesN
L:         h:         h:         h:         h*
	num_split2
while/lstm_cell_18/splitШ
while/lstm_cell_18/SigmoidSigmoid!while/lstm_cell_18/split:output:0*
T0*'
_output_shapes
:         h2
while/lstm_cell_18/SigmoidЬ
while/lstm_cell_18/Sigmoid_1Sigmoid!while/lstm_cell_18/split:output:1*
T0*'
_output_shapes
:         h2
while/lstm_cell_18/Sigmoid_1а
while/lstm_cell_18/mulMul while/lstm_cell_18/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:         h2
while/lstm_cell_18/mulП
while/lstm_cell_18/ReluRelu!while/lstm_cell_18/split:output:2*
T0*'
_output_shapes
:         h2
while/lstm_cell_18/Relu┤
while/lstm_cell_18/mul_1Mulwhile/lstm_cell_18/Sigmoid:y:0%while/lstm_cell_18/Relu:activations:0*
T0*'
_output_shapes
:         h2
while/lstm_cell_18/mul_1й
while/lstm_cell_18/add_1AddV2while/lstm_cell_18/mul:z:0while/lstm_cell_18/mul_1:z:0*
T0*'
_output_shapes
:         h2
while/lstm_cell_18/add_1Ь
while/lstm_cell_18/Sigmoid_2Sigmoid!while/lstm_cell_18/split:output:3*
T0*'
_output_shapes
:         h2
while/lstm_cell_18/Sigmoid_2О
while/lstm_cell_18/Relu_1Reluwhile/lstm_cell_18/add_1:z:0*
T0*'
_output_shapes
:         h2
while/lstm_cell_18/Relu_1╕
while/lstm_cell_18/mul_2Mul while/lstm_cell_18/Sigmoid_2:y:0'while/lstm_cell_18/Relu_1:activations:0*
T0*'
_output_shapes
:         h2
while/lstm_cell_18/mul_2р
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_18/mul_2:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1k
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity~
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_1m
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_2Ъ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3Н
while/Identity_4Identitywhile/lstm_cell_18/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:         h2
while/Identity_4Н
while/Identity_5Identitywhile/lstm_cell_18/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:         h2
while/Identity_5▐

while/NoOpNoOp*^while/lstm_cell_18/BiasAdd/ReadVariableOp)^while/lstm_cell_18/MatMul/ReadVariableOp+^while/lstm_cell_18/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_18_biasadd_readvariableop_resource4while_lstm_cell_18_biasadd_readvariableop_resource_0"l
3while_lstm_cell_18_matmul_1_readvariableop_resource5while_lstm_cell_18_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_18_matmul_readvariableop_resource3while_lstm_cell_18_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         h:         h: : : : : 2V
)while/lstm_cell_18/BiasAdd/ReadVariableOp)while/lstm_cell_18/BiasAdd/ReadVariableOp2T
(while/lstm_cell_18/MatMul/ReadVariableOp(while/lstm_cell_18/MatMul/ReadVariableOp2X
*while/lstm_cell_18/MatMul_1/ReadVariableOp*while/lstm_cell_18/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         h:-)
'
_output_shapes
:         h:

_output_shapes
: :

_output_shapes
: 
¤
И
J__inference_lstm_cell_18_layer_call_and_return_conditional_losses_32590747

inputs
states_0
states_11
matmul_readvariableop_resource:	]а3
 matmul_1_readvariableop_resource:	hа.
biasadd_readvariableop_resource:	а
identity

identity_1

identity_2ИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpвMatMul_1/ReadVariableOpО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	]а*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         а2
MatMulФ
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	hа*
dtype02
MatMul_1/ReadVariableOp|
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         а2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:         а2
addН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:а*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         а2	
BiasAddd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim┐
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:         h:         h:         h:         h*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:         h2	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:         h2
	Sigmoid_1\
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:         h2
mulV
ReluRelusplit:output:2*
T0*'
_output_shapes
:         h2
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:         h2
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:         h2
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:         h2
	Sigmoid_2U
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:         h2
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:         h2
mul_2d
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:         h2

Identityh

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:         h2

Identity_1h

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:         h2

Identity_2Щ
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:         ]:         h:         h: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:         ]
 
_user_specified_nameinputs:QM
'
_output_shapes
:         h
"
_user_specified_name
states/0:QM
'
_output_shapes
:         h
"
_user_specified_name
states/1
Д\
Ю
E__inference_lstm_19_layer_call_and_return_conditional_losses_32590421

inputs>
+lstm_cell_19_matmul_readvariableop_resource:	h╕
A
-lstm_cell_19_matmul_1_readvariableop_resource:
╬╕
;
,lstm_cell_19_biasadd_readvariableop_resource:	╕

identityИв#lstm_cell_19/BiasAdd/ReadVariableOpв"lstm_cell_19/MatMul/ReadVariableOpв$lstm_cell_19/MatMul_1/ReadVariableOpвwhileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice]
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :╬2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessc
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :╬2
zeros/packed/1Г
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:         ╬2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :╬2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :╬2
zeros_1/packed/1Й
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const~
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:         ╬2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:         h2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ю
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1Е
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         2
TensorArrayV2/element_shape▓
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2┐
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    h   27
5TensorArrayUnstack/TensorListFromTensor/element_shape°
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2№
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         h*
shrink_axis_mask2
strided_slice_2╡
"lstm_cell_19/MatMul/ReadVariableOpReadVariableOp+lstm_cell_19_matmul_readvariableop_resource*
_output_shapes
:	h╕
*
dtype02$
"lstm_cell_19/MatMul/ReadVariableOpн
lstm_cell_19/MatMulMatMulstrided_slice_2:output:0*lstm_cell_19/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╕
2
lstm_cell_19/MatMul╝
$lstm_cell_19/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_19_matmul_1_readvariableop_resource* 
_output_shapes
:
╬╕
*
dtype02&
$lstm_cell_19/MatMul_1/ReadVariableOpй
lstm_cell_19/MatMul_1MatMulzeros:output:0,lstm_cell_19/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╕
2
lstm_cell_19/MatMul_1а
lstm_cell_19/addAddV2lstm_cell_19/MatMul:product:0lstm_cell_19/MatMul_1:product:0*
T0*(
_output_shapes
:         ╕
2
lstm_cell_19/add┤
#lstm_cell_19/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_19_biasadd_readvariableop_resource*
_output_shapes	
:╕
*
dtype02%
#lstm_cell_19/BiasAdd/ReadVariableOpн
lstm_cell_19/BiasAddBiasAddlstm_cell_19/add:z:0+lstm_cell_19/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╕
2
lstm_cell_19/BiasAdd~
lstm_cell_19/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_19/split/split_dimў
lstm_cell_19/splitSplit%lstm_cell_19/split/split_dim:output:0lstm_cell_19/BiasAdd:output:0*
T0*d
_output_shapesR
P:         ╬:         ╬:         ╬:         ╬*
	num_split2
lstm_cell_19/splitЗ
lstm_cell_19/SigmoidSigmoidlstm_cell_19/split:output:0*
T0*(
_output_shapes
:         ╬2
lstm_cell_19/SigmoidЛ
lstm_cell_19/Sigmoid_1Sigmoidlstm_cell_19/split:output:1*
T0*(
_output_shapes
:         ╬2
lstm_cell_19/Sigmoid_1М
lstm_cell_19/mulMullstm_cell_19/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:         ╬2
lstm_cell_19/mul~
lstm_cell_19/ReluRelulstm_cell_19/split:output:2*
T0*(
_output_shapes
:         ╬2
lstm_cell_19/ReluЭ
lstm_cell_19/mul_1Mullstm_cell_19/Sigmoid:y:0lstm_cell_19/Relu:activations:0*
T0*(
_output_shapes
:         ╬2
lstm_cell_19/mul_1Т
lstm_cell_19/add_1AddV2lstm_cell_19/mul:z:0lstm_cell_19/mul_1:z:0*
T0*(
_output_shapes
:         ╬2
lstm_cell_19/add_1Л
lstm_cell_19/Sigmoid_2Sigmoidlstm_cell_19/split:output:3*
T0*(
_output_shapes
:         ╬2
lstm_cell_19/Sigmoid_2}
lstm_cell_19/Relu_1Relulstm_cell_19/add_1:z:0*
T0*(
_output_shapes
:         ╬2
lstm_cell_19/Relu_1б
lstm_cell_19/mul_2Mullstm_cell_19/Sigmoid_2:y:0!lstm_cell_19/Relu_1:activations:0*
T0*(
_output_shapes
:         ╬2
lstm_cell_19/mul_2П
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    N  2
TensorArrayV2_1/element_shape╕
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterУ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_19_matmul_readvariableop_resource-lstm_cell_19_matmul_1_readvariableop_resource,lstm_cell_19_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :         ╬:         ╬: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_32590337*
condR
while_cond_32590336*M
output_shapes<
:: : : : :         ╬:         ╬: : : : : *
parallel_iterations 2
while╡
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    N  22
0TensorArrayV2Stack/TensorListStack/element_shapeщ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:         ╬*
element_dtype02$
"TensorArrayV2Stack/TensorListStackБ
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2Ы
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         ╬*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permж
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:         ╬2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimeo
IdentityIdentitytranspose_1:y:0^NoOp*
T0*,
_output_shapes
:         ╬2

Identity╚
NoOpNoOp$^lstm_cell_19/BiasAdd/ReadVariableOp#^lstm_cell_19/MatMul/ReadVariableOp%^lstm_cell_19/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         h: : : 2J
#lstm_cell_19/BiasAdd/ReadVariableOp#lstm_cell_19/BiasAdd/ReadVariableOp2H
"lstm_cell_19/MatMul/ReadVariableOp"lstm_cell_19/MatMul/ReadVariableOp2L
$lstm_cell_19/MatMul_1/ReadVariableOp$lstm_cell_19/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:         h
 
_user_specified_nameinputs
▀
═
while_cond_32587698
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_32587698___redundant_placeholder06
2while_while_cond_32587698___redundant_placeholder16
2while_while_cond_32587698___redundant_placeholder26
2while_while_cond_32587698___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :         h:         h: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         h:-)
'
_output_shapes
:         h:

_output_shapes
: :

_output_shapes
:
Д\
Ю
E__inference_lstm_19_layer_call_and_return_conditional_losses_32588217

inputs>
+lstm_cell_19_matmul_readvariableop_resource:	h╕
A
-lstm_cell_19_matmul_1_readvariableop_resource:
╬╕
;
,lstm_cell_19_biasadd_readvariableop_resource:	╕

identityИв#lstm_cell_19/BiasAdd/ReadVariableOpв"lstm_cell_19/MatMul/ReadVariableOpв$lstm_cell_19/MatMul_1/ReadVariableOpвwhileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice]
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :╬2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessc
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :╬2
zeros/packed/1Г
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:         ╬2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :╬2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :╬2
zeros_1/packed/1Й
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const~
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:         ╬2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:         h2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ю
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1Е
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         2
TensorArrayV2/element_shape▓
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2┐
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    h   27
5TensorArrayUnstack/TensorListFromTensor/element_shape°
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2№
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         h*
shrink_axis_mask2
strided_slice_2╡
"lstm_cell_19/MatMul/ReadVariableOpReadVariableOp+lstm_cell_19_matmul_readvariableop_resource*
_output_shapes
:	h╕
*
dtype02$
"lstm_cell_19/MatMul/ReadVariableOpн
lstm_cell_19/MatMulMatMulstrided_slice_2:output:0*lstm_cell_19/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╕
2
lstm_cell_19/MatMul╝
$lstm_cell_19/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_19_matmul_1_readvariableop_resource* 
_output_shapes
:
╬╕
*
dtype02&
$lstm_cell_19/MatMul_1/ReadVariableOpй
lstm_cell_19/MatMul_1MatMulzeros:output:0,lstm_cell_19/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╕
2
lstm_cell_19/MatMul_1а
lstm_cell_19/addAddV2lstm_cell_19/MatMul:product:0lstm_cell_19/MatMul_1:product:0*
T0*(
_output_shapes
:         ╕
2
lstm_cell_19/add┤
#lstm_cell_19/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_19_biasadd_readvariableop_resource*
_output_shapes	
:╕
*
dtype02%
#lstm_cell_19/BiasAdd/ReadVariableOpн
lstm_cell_19/BiasAddBiasAddlstm_cell_19/add:z:0+lstm_cell_19/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╕
2
lstm_cell_19/BiasAdd~
lstm_cell_19/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_19/split/split_dimў
lstm_cell_19/splitSplit%lstm_cell_19/split/split_dim:output:0lstm_cell_19/BiasAdd:output:0*
T0*d
_output_shapesR
P:         ╬:         ╬:         ╬:         ╬*
	num_split2
lstm_cell_19/splitЗ
lstm_cell_19/SigmoidSigmoidlstm_cell_19/split:output:0*
T0*(
_output_shapes
:         ╬2
lstm_cell_19/SigmoidЛ
lstm_cell_19/Sigmoid_1Sigmoidlstm_cell_19/split:output:1*
T0*(
_output_shapes
:         ╬2
lstm_cell_19/Sigmoid_1М
lstm_cell_19/mulMullstm_cell_19/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:         ╬2
lstm_cell_19/mul~
lstm_cell_19/ReluRelulstm_cell_19/split:output:2*
T0*(
_output_shapes
:         ╬2
lstm_cell_19/ReluЭ
lstm_cell_19/mul_1Mullstm_cell_19/Sigmoid:y:0lstm_cell_19/Relu:activations:0*
T0*(
_output_shapes
:         ╬2
lstm_cell_19/mul_1Т
lstm_cell_19/add_1AddV2lstm_cell_19/mul:z:0lstm_cell_19/mul_1:z:0*
T0*(
_output_shapes
:         ╬2
lstm_cell_19/add_1Л
lstm_cell_19/Sigmoid_2Sigmoidlstm_cell_19/split:output:3*
T0*(
_output_shapes
:         ╬2
lstm_cell_19/Sigmoid_2}
lstm_cell_19/Relu_1Relulstm_cell_19/add_1:z:0*
T0*(
_output_shapes
:         ╬2
lstm_cell_19/Relu_1б
lstm_cell_19/mul_2Mullstm_cell_19/Sigmoid_2:y:0!lstm_cell_19/Relu_1:activations:0*
T0*(
_output_shapes
:         ╬2
lstm_cell_19/mul_2П
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    N  2
TensorArrayV2_1/element_shape╕
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterУ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_19_matmul_readvariableop_resource-lstm_cell_19_matmul_1_readvariableop_resource,lstm_cell_19_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :         ╬:         ╬: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_32588133*
condR
while_cond_32588132*M
output_shapes<
:: : : : :         ╬:         ╬: : : : : *
parallel_iterations 2
while╡
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    N  22
0TensorArrayV2Stack/TensorListStack/element_shapeщ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:         ╬*
element_dtype02$
"TensorArrayV2Stack/TensorListStackБ
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2Ы
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         ╬*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permж
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:         ╬2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimeo
IdentityIdentitytranspose_1:y:0^NoOp*
T0*,
_output_shapes
:         ╬2

Identity╚
NoOpNoOp$^lstm_cell_19/BiasAdd/ReadVariableOp#^lstm_cell_19/MatMul/ReadVariableOp%^lstm_cell_19/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         h: : : 2J
#lstm_cell_19/BiasAdd/ReadVariableOp#lstm_cell_19/BiasAdd/ReadVariableOp2H
"lstm_cell_19/MatMul/ReadVariableOp"lstm_cell_19/MatMul/ReadVariableOp2L
$lstm_cell_19/MatMul_1/ReadVariableOp$lstm_cell_19/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:         h
 
_user_specified_nameinputs
у
═
while_cond_32590336
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_32590336___redundant_placeholder06
2while_while_cond_32590336___redundant_placeholder16
2while_while_cond_32590336___redundant_placeholder26
2while_while_cond_32590336___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :         ╬:         ╬: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:         ╬:.*
(
_output_shapes
:         ╬:

_output_shapes
: :

_output_shapes
:
╨

э
lstm_18_while_cond_32588976,
(lstm_18_while_lstm_18_while_loop_counter2
.lstm_18_while_lstm_18_while_maximum_iterations
lstm_18_while_placeholder
lstm_18_while_placeholder_1
lstm_18_while_placeholder_2
lstm_18_while_placeholder_3.
*lstm_18_while_less_lstm_18_strided_slice_1F
Blstm_18_while_lstm_18_while_cond_32588976___redundant_placeholder0F
Blstm_18_while_lstm_18_while_cond_32588976___redundant_placeholder1F
Blstm_18_while_lstm_18_while_cond_32588976___redundant_placeholder2F
Blstm_18_while_lstm_18_while_cond_32588976___redundant_placeholder3
lstm_18_while_identity
Ш
lstm_18/while/LessLesslstm_18_while_placeholder*lstm_18_while_less_lstm_18_strided_slice_1*
T0*
_output_shapes
: 2
lstm_18/while/Lessu
lstm_18/while/IdentityIdentitylstm_18/while/Less:z:0*
T0
*
_output_shapes
: 2
lstm_18/while/Identity"9
lstm_18_while_identitylstm_18/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :         h:         h: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         h:-)
'
_output_shapes
:         h:

_output_shapes
: :

_output_shapes
:
у
═
while_cond_32590487
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_32590487___redundant_placeholder06
2while_while_cond_32590487___redundant_placeholder16
2while_while_cond_32590487___redundant_placeholder26
2while_while_cond_32590487___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :         ╬:         ╬: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:         ╬:.*
(
_output_shapes
:         ╬:

_output_shapes
: :

_output_shapes
:
╚7
█
$__inference__traced_restore_32590984
file_prefix2
assignvariableop_dense_9_kernel:	╬-
assignvariableop_1_dense_9_bias:A
.assignvariableop_2_lstm_18_lstm_cell_18_kernel:	]аK
8assignvariableop_3_lstm_18_lstm_cell_18_recurrent_kernel:	hа;
,assignvariableop_4_lstm_18_lstm_cell_18_bias:	аA
.assignvariableop_5_lstm_19_lstm_cell_19_kernel:	h╕
L
8assignvariableop_6_lstm_19_lstm_cell_19_recurrent_kernel:
╬╕
;
,assignvariableop_7_lstm_19_lstm_cell_19_bias:	╕
"
assignvariableop_8_total: "
assignvariableop_9_count: %
assignvariableop_10_total_1: %
assignvariableop_11_count_1: 
identity_13ИвAssignVariableOpвAssignVariableOp_1вAssignVariableOp_10вAssignVariableOp_11вAssignVariableOp_2вAssignVariableOp_3вAssignVariableOp_4вAssignVariableOp_5вAssignVariableOp_6вAssignVariableOp_7вAssignVariableOp_8вAssignVariableOp_9Щ
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*е
valueЫBШB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_namesи
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*-
value$B"B B B B B B B B B B B B B 2
RestoreV2/shape_and_slicesь
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*H
_output_shapes6
4:::::::::::::*
dtypes
22
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

IdentityЮ
AssignVariableOpAssignVariableOpassignvariableop_dense_9_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1д
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_9_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2│
AssignVariableOp_2AssignVariableOp.assignvariableop_2_lstm_18_lstm_cell_18_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3╜
AssignVariableOp_3AssignVariableOp8assignvariableop_3_lstm_18_lstm_cell_18_recurrent_kernelIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4▒
AssignVariableOp_4AssignVariableOp,assignvariableop_4_lstm_18_lstm_cell_18_biasIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5│
AssignVariableOp_5AssignVariableOp.assignvariableop_5_lstm_19_lstm_cell_19_kernelIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6╜
AssignVariableOp_6AssignVariableOp8assignvariableop_6_lstm_19_lstm_cell_19_recurrent_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7▒
AssignVariableOp_7AssignVariableOp,assignvariableop_7_lstm_19_lstm_cell_19_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8Э
AssignVariableOp_8AssignVariableOpassignvariableop_8_totalIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9Э
AssignVariableOp_9AssignVariableOpassignvariableop_9_countIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10г
AssignVariableOp_10AssignVariableOpassignvariableop_10_total_1Identity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11г
AssignVariableOp_11AssignVariableOpassignvariableop_11_count_1Identity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_119
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpц
Identity_12Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_12f
Identity_13IdentityIdentity_12:output:0^NoOp_1*
T0*
_output_shapes
: 2
Identity_13╬
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 2
NoOp_1"#
identity_13Identity_13:output:0*-
_input_shapes
: : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
м]
ї
(sequential_9_lstm_19_while_body_32586254F
Bsequential_9_lstm_19_while_sequential_9_lstm_19_while_loop_counterL
Hsequential_9_lstm_19_while_sequential_9_lstm_19_while_maximum_iterations*
&sequential_9_lstm_19_while_placeholder,
(sequential_9_lstm_19_while_placeholder_1,
(sequential_9_lstm_19_while_placeholder_2,
(sequential_9_lstm_19_while_placeholder_3E
Asequential_9_lstm_19_while_sequential_9_lstm_19_strided_slice_1_0Б
}sequential_9_lstm_19_while_tensorarrayv2read_tensorlistgetitem_sequential_9_lstm_19_tensorarrayunstack_tensorlistfromtensor_0[
Hsequential_9_lstm_19_while_lstm_cell_19_matmul_readvariableop_resource_0:	h╕
^
Jsequential_9_lstm_19_while_lstm_cell_19_matmul_1_readvariableop_resource_0:
╬╕
X
Isequential_9_lstm_19_while_lstm_cell_19_biasadd_readvariableop_resource_0:	╕
'
#sequential_9_lstm_19_while_identity)
%sequential_9_lstm_19_while_identity_1)
%sequential_9_lstm_19_while_identity_2)
%sequential_9_lstm_19_while_identity_3)
%sequential_9_lstm_19_while_identity_4)
%sequential_9_lstm_19_while_identity_5C
?sequential_9_lstm_19_while_sequential_9_lstm_19_strided_slice_1
{sequential_9_lstm_19_while_tensorarrayv2read_tensorlistgetitem_sequential_9_lstm_19_tensorarrayunstack_tensorlistfromtensorY
Fsequential_9_lstm_19_while_lstm_cell_19_matmul_readvariableop_resource:	h╕
\
Hsequential_9_lstm_19_while_lstm_cell_19_matmul_1_readvariableop_resource:
╬╕
V
Gsequential_9_lstm_19_while_lstm_cell_19_biasadd_readvariableop_resource:	╕
Ив>sequential_9/lstm_19/while/lstm_cell_19/BiasAdd/ReadVariableOpв=sequential_9/lstm_19/while/lstm_cell_19/MatMul/ReadVariableOpв?sequential_9/lstm_19/while/lstm_cell_19/MatMul_1/ReadVariableOpэ
Lsequential_9/lstm_19/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    h   2N
Lsequential_9/lstm_19/while/TensorArrayV2Read/TensorListGetItem/element_shape╤
>sequential_9/lstm_19/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem}sequential_9_lstm_19_while_tensorarrayv2read_tensorlistgetitem_sequential_9_lstm_19_tensorarrayunstack_tensorlistfromtensor_0&sequential_9_lstm_19_while_placeholderUsequential_9/lstm_19/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         h*
element_dtype02@
>sequential_9/lstm_19/while/TensorArrayV2Read/TensorListGetItemИ
=sequential_9/lstm_19/while/lstm_cell_19/MatMul/ReadVariableOpReadVariableOpHsequential_9_lstm_19_while_lstm_cell_19_matmul_readvariableop_resource_0*
_output_shapes
:	h╕
*
dtype02?
=sequential_9/lstm_19/while/lstm_cell_19/MatMul/ReadVariableOpл
.sequential_9/lstm_19/while/lstm_cell_19/MatMulMatMulEsequential_9/lstm_19/while/TensorArrayV2Read/TensorListGetItem:item:0Esequential_9/lstm_19/while/lstm_cell_19/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╕
20
.sequential_9/lstm_19/while/lstm_cell_19/MatMulП
?sequential_9/lstm_19/while/lstm_cell_19/MatMul_1/ReadVariableOpReadVariableOpJsequential_9_lstm_19_while_lstm_cell_19_matmul_1_readvariableop_resource_0* 
_output_shapes
:
╬╕
*
dtype02A
?sequential_9/lstm_19/while/lstm_cell_19/MatMul_1/ReadVariableOpФ
0sequential_9/lstm_19/while/lstm_cell_19/MatMul_1MatMul(sequential_9_lstm_19_while_placeholder_2Gsequential_9/lstm_19/while/lstm_cell_19/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╕
22
0sequential_9/lstm_19/while/lstm_cell_19/MatMul_1М
+sequential_9/lstm_19/while/lstm_cell_19/addAddV28sequential_9/lstm_19/while/lstm_cell_19/MatMul:product:0:sequential_9/lstm_19/while/lstm_cell_19/MatMul_1:product:0*
T0*(
_output_shapes
:         ╕
2-
+sequential_9/lstm_19/while/lstm_cell_19/addЗ
>sequential_9/lstm_19/while/lstm_cell_19/BiasAdd/ReadVariableOpReadVariableOpIsequential_9_lstm_19_while_lstm_cell_19_biasadd_readvariableop_resource_0*
_output_shapes	
:╕
*
dtype02@
>sequential_9/lstm_19/while/lstm_cell_19/BiasAdd/ReadVariableOpЩ
/sequential_9/lstm_19/while/lstm_cell_19/BiasAddBiasAdd/sequential_9/lstm_19/while/lstm_cell_19/add:z:0Fsequential_9/lstm_19/while/lstm_cell_19/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╕
21
/sequential_9/lstm_19/while/lstm_cell_19/BiasAdd┤
7sequential_9/lstm_19/while/lstm_cell_19/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :29
7sequential_9/lstm_19/while/lstm_cell_19/split/split_dimу
-sequential_9/lstm_19/while/lstm_cell_19/splitSplit@sequential_9/lstm_19/while/lstm_cell_19/split/split_dim:output:08sequential_9/lstm_19/while/lstm_cell_19/BiasAdd:output:0*
T0*d
_output_shapesR
P:         ╬:         ╬:         ╬:         ╬*
	num_split2/
-sequential_9/lstm_19/while/lstm_cell_19/split╪
/sequential_9/lstm_19/while/lstm_cell_19/SigmoidSigmoid6sequential_9/lstm_19/while/lstm_cell_19/split:output:0*
T0*(
_output_shapes
:         ╬21
/sequential_9/lstm_19/while/lstm_cell_19/Sigmoid▄
1sequential_9/lstm_19/while/lstm_cell_19/Sigmoid_1Sigmoid6sequential_9/lstm_19/while/lstm_cell_19/split:output:1*
T0*(
_output_shapes
:         ╬23
1sequential_9/lstm_19/while/lstm_cell_19/Sigmoid_1ї
+sequential_9/lstm_19/while/lstm_cell_19/mulMul5sequential_9/lstm_19/while/lstm_cell_19/Sigmoid_1:y:0(sequential_9_lstm_19_while_placeholder_3*
T0*(
_output_shapes
:         ╬2-
+sequential_9/lstm_19/while/lstm_cell_19/mul╧
,sequential_9/lstm_19/while/lstm_cell_19/ReluRelu6sequential_9/lstm_19/while/lstm_cell_19/split:output:2*
T0*(
_output_shapes
:         ╬2.
,sequential_9/lstm_19/while/lstm_cell_19/ReluЙ
-sequential_9/lstm_19/while/lstm_cell_19/mul_1Mul3sequential_9/lstm_19/while/lstm_cell_19/Sigmoid:y:0:sequential_9/lstm_19/while/lstm_cell_19/Relu:activations:0*
T0*(
_output_shapes
:         ╬2/
-sequential_9/lstm_19/while/lstm_cell_19/mul_1■
-sequential_9/lstm_19/while/lstm_cell_19/add_1AddV2/sequential_9/lstm_19/while/lstm_cell_19/mul:z:01sequential_9/lstm_19/while/lstm_cell_19/mul_1:z:0*
T0*(
_output_shapes
:         ╬2/
-sequential_9/lstm_19/while/lstm_cell_19/add_1▄
1sequential_9/lstm_19/while/lstm_cell_19/Sigmoid_2Sigmoid6sequential_9/lstm_19/while/lstm_cell_19/split:output:3*
T0*(
_output_shapes
:         ╬23
1sequential_9/lstm_19/while/lstm_cell_19/Sigmoid_2╬
.sequential_9/lstm_19/while/lstm_cell_19/Relu_1Relu1sequential_9/lstm_19/while/lstm_cell_19/add_1:z:0*
T0*(
_output_shapes
:         ╬20
.sequential_9/lstm_19/while/lstm_cell_19/Relu_1Н
-sequential_9/lstm_19/while/lstm_cell_19/mul_2Mul5sequential_9/lstm_19/while/lstm_cell_19/Sigmoid_2:y:0<sequential_9/lstm_19/while/lstm_cell_19/Relu_1:activations:0*
T0*(
_output_shapes
:         ╬2/
-sequential_9/lstm_19/while/lstm_cell_19/mul_2╔
?sequential_9/lstm_19/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem(sequential_9_lstm_19_while_placeholder_1&sequential_9_lstm_19_while_placeholder1sequential_9/lstm_19/while/lstm_cell_19/mul_2:z:0*
_output_shapes
: *
element_dtype02A
?sequential_9/lstm_19/while/TensorArrayV2Write/TensorListSetItemЖ
 sequential_9/lstm_19/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2"
 sequential_9/lstm_19/while/add/y╜
sequential_9/lstm_19/while/addAddV2&sequential_9_lstm_19_while_placeholder)sequential_9/lstm_19/while/add/y:output:0*
T0*
_output_shapes
: 2 
sequential_9/lstm_19/while/addК
"sequential_9/lstm_19/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2$
"sequential_9/lstm_19/while/add_1/y▀
 sequential_9/lstm_19/while/add_1AddV2Bsequential_9_lstm_19_while_sequential_9_lstm_19_while_loop_counter+sequential_9/lstm_19/while/add_1/y:output:0*
T0*
_output_shapes
: 2"
 sequential_9/lstm_19/while/add_1┐
#sequential_9/lstm_19/while/IdentityIdentity$sequential_9/lstm_19/while/add_1:z:0 ^sequential_9/lstm_19/while/NoOp*
T0*
_output_shapes
: 2%
#sequential_9/lstm_19/while/Identityч
%sequential_9/lstm_19/while/Identity_1IdentityHsequential_9_lstm_19_while_sequential_9_lstm_19_while_maximum_iterations ^sequential_9/lstm_19/while/NoOp*
T0*
_output_shapes
: 2'
%sequential_9/lstm_19/while/Identity_1┴
%sequential_9/lstm_19/while/Identity_2Identity"sequential_9/lstm_19/while/add:z:0 ^sequential_9/lstm_19/while/NoOp*
T0*
_output_shapes
: 2'
%sequential_9/lstm_19/while/Identity_2ю
%sequential_9/lstm_19/while/Identity_3IdentityOsequential_9/lstm_19/while/TensorArrayV2Write/TensorListSetItem:output_handle:0 ^sequential_9/lstm_19/while/NoOp*
T0*
_output_shapes
: 2'
%sequential_9/lstm_19/while/Identity_3т
%sequential_9/lstm_19/while/Identity_4Identity1sequential_9/lstm_19/while/lstm_cell_19/mul_2:z:0 ^sequential_9/lstm_19/while/NoOp*
T0*(
_output_shapes
:         ╬2'
%sequential_9/lstm_19/while/Identity_4т
%sequential_9/lstm_19/while/Identity_5Identity1sequential_9/lstm_19/while/lstm_cell_19/add_1:z:0 ^sequential_9/lstm_19/while/NoOp*
T0*(
_output_shapes
:         ╬2'
%sequential_9/lstm_19/while/Identity_5╟
sequential_9/lstm_19/while/NoOpNoOp?^sequential_9/lstm_19/while/lstm_cell_19/BiasAdd/ReadVariableOp>^sequential_9/lstm_19/while/lstm_cell_19/MatMul/ReadVariableOp@^sequential_9/lstm_19/while/lstm_cell_19/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2!
sequential_9/lstm_19/while/NoOp"S
#sequential_9_lstm_19_while_identity,sequential_9/lstm_19/while/Identity:output:0"W
%sequential_9_lstm_19_while_identity_1.sequential_9/lstm_19/while/Identity_1:output:0"W
%sequential_9_lstm_19_while_identity_2.sequential_9/lstm_19/while/Identity_2:output:0"W
%sequential_9_lstm_19_while_identity_3.sequential_9/lstm_19/while/Identity_3:output:0"W
%sequential_9_lstm_19_while_identity_4.sequential_9/lstm_19/while/Identity_4:output:0"W
%sequential_9_lstm_19_while_identity_5.sequential_9/lstm_19/while/Identity_5:output:0"Ф
Gsequential_9_lstm_19_while_lstm_cell_19_biasadd_readvariableop_resourceIsequential_9_lstm_19_while_lstm_cell_19_biasadd_readvariableop_resource_0"Ц
Hsequential_9_lstm_19_while_lstm_cell_19_matmul_1_readvariableop_resourceJsequential_9_lstm_19_while_lstm_cell_19_matmul_1_readvariableop_resource_0"Т
Fsequential_9_lstm_19_while_lstm_cell_19_matmul_readvariableop_resourceHsequential_9_lstm_19_while_lstm_cell_19_matmul_readvariableop_resource_0"Д
?sequential_9_lstm_19_while_sequential_9_lstm_19_strided_slice_1Asequential_9_lstm_19_while_sequential_9_lstm_19_strided_slice_1_0"№
{sequential_9_lstm_19_while_tensorarrayv2read_tensorlistgetitem_sequential_9_lstm_19_tensorarrayunstack_tensorlistfromtensor}sequential_9_lstm_19_while_tensorarrayv2read_tensorlistgetitem_sequential_9_lstm_19_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :         ╬:         ╬: : : : : 2А
>sequential_9/lstm_19/while/lstm_cell_19/BiasAdd/ReadVariableOp>sequential_9/lstm_19/while/lstm_cell_19/BiasAdd/ReadVariableOp2~
=sequential_9/lstm_19/while/lstm_cell_19/MatMul/ReadVariableOp=sequential_9/lstm_19/while/lstm_cell_19/MatMul/ReadVariableOp2В
?sequential_9/lstm_19/while/lstm_cell_19/MatMul_1/ReadVariableOp?sequential_9/lstm_19/while/lstm_cell_19/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:         ╬:.*
(
_output_shapes
:         ╬:

_output_shapes
: :

_output_shapes
: 
Й
f
H__inference_dropout_19_layer_call_and_return_conditional_losses_32587961

inputs

identity_1_
IdentityIdentityinputs*
T0*,
_output_shapes
:         ╬2

Identityn

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:         ╬2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ╬:T P
,
_output_shapes
:         ╬
 
_user_specified_nameinputs
╫
g
H__inference_dropout_19_layer_call_and_return_conditional_losses_32588050

inputs
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  а?2
dropout/Constx
dropout/MulMulinputsdropout/Const:output:0*
T0*,
_output_shapes
:         ╬2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape╣
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:         ╬*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>2
dropout/GreaterEqual/y├
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:         ╬2
dropout/GreaterEqualД
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:         ╬2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:         ╬2
dropout/Mul_1j
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:         ╬2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ╬:T P
,
_output_shapes
:         ╬
 
_user_specified_nameinputs
╘!
¤
E__inference_dense_9_layer_call_and_return_conditional_losses_32587994

inputs4
!tensordot_readvariableop_resource:	╬-
biasadd_readvariableop_resource:
identityИвBiasAdd/ReadVariableOpвTensordot/ReadVariableOpЧ
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes
:	╬*
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesq
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axis╤
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis╫
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/ConstА
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1И
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis░
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concatМ
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stackС
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*,
_output_shapes
:         ╬2
Tensordot/transposeЯ
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:                  2
Tensordot/ReshapeЮ
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
Tensordot/MatMulp
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis╜
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1Р
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:         2
	TensordotМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpЗ
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:         2	
BiasAdde
SoftmaxSoftmaxBiasAdd:output:0*
T0*+
_output_shapes
:         2	
Softmaxp
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*+
_output_shapes
:         2

IdentityВ
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         ╬: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:T P
,
_output_shapes
:         ╬
 
_user_specified_nameinputs
·%
ё
while_body_32586665
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_00
while_lstm_cell_18_32586689_0:	]а0
while_lstm_cell_18_32586691_0:	hа,
while_lstm_cell_18_32586693_0:	а
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor.
while_lstm_cell_18_32586689:	]а.
while_lstm_cell_18_32586691:	hа*
while_lstm_cell_18_32586693:	аИв*while/lstm_cell_18/StatefulPartitionedCall├
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ]   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape╙
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         ]*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemщ
*while/lstm_cell_18/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_18_32586689_0while_lstm_cell_18_32586691_0while_lstm_cell_18_32586693_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:         h:         h:         h*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_lstm_cell_18_layer_call_and_return_conditional_losses_325865872,
*while/lstm_cell_18/StatefulPartitionedCallў
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder3while/lstm_cell_18/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1k
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity~
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_1m
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_2Ъ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3д
while/Identity_4Identity3while/lstm_cell_18/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:         h2
while/Identity_4д
while/Identity_5Identity3while/lstm_cell_18/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:         h2
while/Identity_5З

while/NoOpNoOp+^while/lstm_cell_18/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"<
while_lstm_cell_18_32586689while_lstm_cell_18_32586689_0"<
while_lstm_cell_18_32586691while_lstm_cell_18_32586691_0"<
while_lstm_cell_18_32586693while_lstm_cell_18_32586693_0"0
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         h:         h: : : : : 2X
*while/lstm_cell_18/StatefulPartitionedCall*while/lstm_cell_18/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         h:-)
'
_output_shapes
:         h:

_output_shapes
: :

_output_shapes
: 
Е&
є
while_body_32587085
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_00
while_lstm_cell_19_32587109_0:	h╕
1
while_lstm_cell_19_32587111_0:
╬╕
,
while_lstm_cell_19_32587113_0:	╕

while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor.
while_lstm_cell_19_32587109:	h╕
/
while_lstm_cell_19_32587111:
╬╕
*
while_lstm_cell_19_32587113:	╕
Ив*while/lstm_cell_19/StatefulPartitionedCall├
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    h   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape╙
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         h*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemь
*while/lstm_cell_19/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_19_32587109_0while_lstm_cell_19_32587111_0while_lstm_cell_19_32587113_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:         ╬:         ╬:         ╬*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_lstm_cell_19_layer_call_and_return_conditional_losses_325870712,
*while/lstm_cell_19/StatefulPartitionedCallў
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder3while/lstm_cell_19/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1k
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity~
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_1m
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_2Ъ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3е
while/Identity_4Identity3while/lstm_cell_19/StatefulPartitionedCall:output:1^while/NoOp*
T0*(
_output_shapes
:         ╬2
while/Identity_4е
while/Identity_5Identity3while/lstm_cell_19/StatefulPartitionedCall:output:2^while/NoOp*
T0*(
_output_shapes
:         ╬2
while/Identity_5З

while/NoOpNoOp+^while/lstm_cell_19/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"<
while_lstm_cell_19_32587109while_lstm_cell_19_32587109_0"<
while_lstm_cell_19_32587111while_lstm_cell_19_32587111_0"<
while_lstm_cell_19_32587113while_lstm_cell_19_32587113_0"0
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :         ╬:         ╬: : : : : 2X
*while/lstm_cell_19/StatefulPartitionedCall*while/lstm_cell_19/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:         ╬:.*
(
_output_shapes
:         ╬:

_output_shapes
: :

_output_shapes
: 
╨

э
lstm_18_while_cond_32588649,
(lstm_18_while_lstm_18_while_loop_counter2
.lstm_18_while_lstm_18_while_maximum_iterations
lstm_18_while_placeholder
lstm_18_while_placeholder_1
lstm_18_while_placeholder_2
lstm_18_while_placeholder_3.
*lstm_18_while_less_lstm_18_strided_slice_1F
Blstm_18_while_lstm_18_while_cond_32588649___redundant_placeholder0F
Blstm_18_while_lstm_18_while_cond_32588649___redundant_placeholder1F
Blstm_18_while_lstm_18_while_cond_32588649___redundant_placeholder2F
Blstm_18_while_lstm_18_while_cond_32588649___redundant_placeholder3
lstm_18_while_identity
Ш
lstm_18/while/LessLesslstm_18_while_placeholder*lstm_18_while_less_lstm_18_strided_slice_1*
T0*
_output_shapes
: 2
lstm_18/while/Lessu
lstm_18/while/IdentityIdentitylstm_18/while/Less:z:0*
T0
*
_output_shapes
: 2
lstm_18/while/Identity"9
lstm_18_while_identitylstm_18/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :         h:         h: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         h:-)
'
_output_shapes
:         h:

_output_shapes
: :

_output_shapes
:
яЛ
Е
J__inference_sequential_9_layer_call_and_return_conditional_losses_32589251

inputsF
3lstm_18_lstm_cell_18_matmul_readvariableop_resource:	]аH
5lstm_18_lstm_cell_18_matmul_1_readvariableop_resource:	hаC
4lstm_18_lstm_cell_18_biasadd_readvariableop_resource:	аF
3lstm_19_lstm_cell_19_matmul_readvariableop_resource:	h╕
I
5lstm_19_lstm_cell_19_matmul_1_readvariableop_resource:
╬╕
C
4lstm_19_lstm_cell_19_biasadd_readvariableop_resource:	╕
<
)dense_9_tensordot_readvariableop_resource:	╬5
'dense_9_biasadd_readvariableop_resource:
identityИвdense_9/BiasAdd/ReadVariableOpв dense_9/Tensordot/ReadVariableOpв+lstm_18/lstm_cell_18/BiasAdd/ReadVariableOpв*lstm_18/lstm_cell_18/MatMul/ReadVariableOpв,lstm_18/lstm_cell_18/MatMul_1/ReadVariableOpвlstm_18/whileв+lstm_19/lstm_cell_19/BiasAdd/ReadVariableOpв*lstm_19/lstm_cell_19/MatMul/ReadVariableOpв,lstm_19/lstm_cell_19/MatMul_1/ReadVariableOpвlstm_19/whileT
lstm_18/ShapeShapeinputs*
T0*
_output_shapes
:2
lstm_18/ShapeД
lstm_18/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_18/strided_slice/stackИ
lstm_18/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_18/strided_slice/stack_1И
lstm_18/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_18/strided_slice/stack_2Т
lstm_18/strided_sliceStridedSlicelstm_18/Shape:output:0$lstm_18/strided_slice/stack:output:0&lstm_18/strided_slice/stack_1:output:0&lstm_18/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_18/strided_slicel
lstm_18/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :h2
lstm_18/zeros/mul/yМ
lstm_18/zeros/mulMullstm_18/strided_slice:output:0lstm_18/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_18/zeros/mulo
lstm_18/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
lstm_18/zeros/Less/yЗ
lstm_18/zeros/LessLesslstm_18/zeros/mul:z:0lstm_18/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_18/zeros/Lessr
lstm_18/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :h2
lstm_18/zeros/packed/1г
lstm_18/zeros/packedPacklstm_18/strided_slice:output:0lstm_18/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_18/zeros/packedo
lstm_18/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_18/zeros/ConstХ
lstm_18/zerosFilllstm_18/zeros/packed:output:0lstm_18/zeros/Const:output:0*
T0*'
_output_shapes
:         h2
lstm_18/zerosp
lstm_18/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :h2
lstm_18/zeros_1/mul/yТ
lstm_18/zeros_1/mulMullstm_18/strided_slice:output:0lstm_18/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_18/zeros_1/muls
lstm_18/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
lstm_18/zeros_1/Less/yП
lstm_18/zeros_1/LessLesslstm_18/zeros_1/mul:z:0lstm_18/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_18/zeros_1/Lessv
lstm_18/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :h2
lstm_18/zeros_1/packed/1й
lstm_18/zeros_1/packedPacklstm_18/strided_slice:output:0!lstm_18/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_18/zeros_1/packeds
lstm_18/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_18/zeros_1/ConstЭ
lstm_18/zeros_1Filllstm_18/zeros_1/packed:output:0lstm_18/zeros_1/Const:output:0*
T0*'
_output_shapes
:         h2
lstm_18/zeros_1Е
lstm_18/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_18/transpose/permТ
lstm_18/transpose	Transposeinputslstm_18/transpose/perm:output:0*
T0*+
_output_shapes
:         ]2
lstm_18/transposeg
lstm_18/Shape_1Shapelstm_18/transpose:y:0*
T0*
_output_shapes
:2
lstm_18/Shape_1И
lstm_18/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_18/strided_slice_1/stackМ
lstm_18/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_18/strided_slice_1/stack_1М
lstm_18/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_18/strided_slice_1/stack_2Ю
lstm_18/strided_slice_1StridedSlicelstm_18/Shape_1:output:0&lstm_18/strided_slice_1/stack:output:0(lstm_18/strided_slice_1/stack_1:output:0(lstm_18/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_18/strided_slice_1Х
#lstm_18/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         2%
#lstm_18/TensorArrayV2/element_shape╥
lstm_18/TensorArrayV2TensorListReserve,lstm_18/TensorArrayV2/element_shape:output:0 lstm_18/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_18/TensorArrayV2╧
=lstm_18/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ]   2?
=lstm_18/TensorArrayUnstack/TensorListFromTensor/element_shapeШ
/lstm_18/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_18/transpose:y:0Flstm_18/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type021
/lstm_18/TensorArrayUnstack/TensorListFromTensorИ
lstm_18/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_18/strided_slice_2/stackМ
lstm_18/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_18/strided_slice_2/stack_1М
lstm_18/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_18/strided_slice_2/stack_2м
lstm_18/strided_slice_2StridedSlicelstm_18/transpose:y:0&lstm_18/strided_slice_2/stack:output:0(lstm_18/strided_slice_2/stack_1:output:0(lstm_18/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         ]*
shrink_axis_mask2
lstm_18/strided_slice_2═
*lstm_18/lstm_cell_18/MatMul/ReadVariableOpReadVariableOp3lstm_18_lstm_cell_18_matmul_readvariableop_resource*
_output_shapes
:	]а*
dtype02,
*lstm_18/lstm_cell_18/MatMul/ReadVariableOp═
lstm_18/lstm_cell_18/MatMulMatMul lstm_18/strided_slice_2:output:02lstm_18/lstm_cell_18/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         а2
lstm_18/lstm_cell_18/MatMul╙
,lstm_18/lstm_cell_18/MatMul_1/ReadVariableOpReadVariableOp5lstm_18_lstm_cell_18_matmul_1_readvariableop_resource*
_output_shapes
:	hа*
dtype02.
,lstm_18/lstm_cell_18/MatMul_1/ReadVariableOp╔
lstm_18/lstm_cell_18/MatMul_1MatMullstm_18/zeros:output:04lstm_18/lstm_cell_18/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         а2
lstm_18/lstm_cell_18/MatMul_1└
lstm_18/lstm_cell_18/addAddV2%lstm_18/lstm_cell_18/MatMul:product:0'lstm_18/lstm_cell_18/MatMul_1:product:0*
T0*(
_output_shapes
:         а2
lstm_18/lstm_cell_18/add╠
+lstm_18/lstm_cell_18/BiasAdd/ReadVariableOpReadVariableOp4lstm_18_lstm_cell_18_biasadd_readvariableop_resource*
_output_shapes	
:а*
dtype02-
+lstm_18/lstm_cell_18/BiasAdd/ReadVariableOp═
lstm_18/lstm_cell_18/BiasAddBiasAddlstm_18/lstm_cell_18/add:z:03lstm_18/lstm_cell_18/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         а2
lstm_18/lstm_cell_18/BiasAddО
$lstm_18/lstm_cell_18/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2&
$lstm_18/lstm_cell_18/split/split_dimУ
lstm_18/lstm_cell_18/splitSplit-lstm_18/lstm_cell_18/split/split_dim:output:0%lstm_18/lstm_cell_18/BiasAdd:output:0*
T0*`
_output_shapesN
L:         h:         h:         h:         h*
	num_split2
lstm_18/lstm_cell_18/splitЮ
lstm_18/lstm_cell_18/SigmoidSigmoid#lstm_18/lstm_cell_18/split:output:0*
T0*'
_output_shapes
:         h2
lstm_18/lstm_cell_18/Sigmoidв
lstm_18/lstm_cell_18/Sigmoid_1Sigmoid#lstm_18/lstm_cell_18/split:output:1*
T0*'
_output_shapes
:         h2 
lstm_18/lstm_cell_18/Sigmoid_1л
lstm_18/lstm_cell_18/mulMul"lstm_18/lstm_cell_18/Sigmoid_1:y:0lstm_18/zeros_1:output:0*
T0*'
_output_shapes
:         h2
lstm_18/lstm_cell_18/mulХ
lstm_18/lstm_cell_18/ReluRelu#lstm_18/lstm_cell_18/split:output:2*
T0*'
_output_shapes
:         h2
lstm_18/lstm_cell_18/Relu╝
lstm_18/lstm_cell_18/mul_1Mul lstm_18/lstm_cell_18/Sigmoid:y:0'lstm_18/lstm_cell_18/Relu:activations:0*
T0*'
_output_shapes
:         h2
lstm_18/lstm_cell_18/mul_1▒
lstm_18/lstm_cell_18/add_1AddV2lstm_18/lstm_cell_18/mul:z:0lstm_18/lstm_cell_18/mul_1:z:0*
T0*'
_output_shapes
:         h2
lstm_18/lstm_cell_18/add_1в
lstm_18/lstm_cell_18/Sigmoid_2Sigmoid#lstm_18/lstm_cell_18/split:output:3*
T0*'
_output_shapes
:         h2 
lstm_18/lstm_cell_18/Sigmoid_2Ф
lstm_18/lstm_cell_18/Relu_1Relulstm_18/lstm_cell_18/add_1:z:0*
T0*'
_output_shapes
:         h2
lstm_18/lstm_cell_18/Relu_1└
lstm_18/lstm_cell_18/mul_2Mul"lstm_18/lstm_cell_18/Sigmoid_2:y:0)lstm_18/lstm_cell_18/Relu_1:activations:0*
T0*'
_output_shapes
:         h2
lstm_18/lstm_cell_18/mul_2Я
%lstm_18/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    h   2'
%lstm_18/TensorArrayV2_1/element_shape╪
lstm_18/TensorArrayV2_1TensorListReserve.lstm_18/TensorArrayV2_1/element_shape:output:0 lstm_18/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_18/TensorArrayV2_1^
lstm_18/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_18/timeП
 lstm_18/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         2"
 lstm_18/while/maximum_iterationsz
lstm_18/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_18/while/loop_counterЗ
lstm_18/whileWhile#lstm_18/while/loop_counter:output:0)lstm_18/while/maximum_iterations:output:0lstm_18/time:output:0 lstm_18/TensorArrayV2_1:handle:0lstm_18/zeros:output:0lstm_18/zeros_1:output:0 lstm_18/strided_slice_1:output:0?lstm_18/TensorArrayUnstack/TensorListFromTensor:output_handle:03lstm_18_lstm_cell_18_matmul_readvariableop_resource5lstm_18_lstm_cell_18_matmul_1_readvariableop_resource4lstm_18_lstm_cell_18_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         h:         h: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *'
bodyR
lstm_18_while_body_32588977*'
condR
lstm_18_while_cond_32588976*K
output_shapes:
8: : : : :         h:         h: : : : : *
parallel_iterations 2
lstm_18/while┼
8lstm_18/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    h   2:
8lstm_18/TensorArrayV2Stack/TensorListStack/element_shapeИ
*lstm_18/TensorArrayV2Stack/TensorListStackTensorListStacklstm_18/while:output:3Alstm_18/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:         h*
element_dtype02,
*lstm_18/TensorArrayV2Stack/TensorListStackС
lstm_18/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         2
lstm_18/strided_slice_3/stackМ
lstm_18/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2!
lstm_18/strided_slice_3/stack_1М
lstm_18/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_18/strided_slice_3/stack_2╩
lstm_18/strided_slice_3StridedSlice3lstm_18/TensorArrayV2Stack/TensorListStack:tensor:0&lstm_18/strided_slice_3/stack:output:0(lstm_18/strided_slice_3/stack_1:output:0(lstm_18/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         h*
shrink_axis_mask2
lstm_18/strided_slice_3Й
lstm_18/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_18/transpose_1/perm┼
lstm_18/transpose_1	Transpose3lstm_18/TensorArrayV2Stack/TensorListStack:tensor:0!lstm_18/transpose_1/perm:output:0*
T0*+
_output_shapes
:         h2
lstm_18/transpose_1v
lstm_18/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_18/runtimey
dropout_18/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *лкк?2
dropout_18/dropout/Constй
dropout_18/dropout/MulMullstm_18/transpose_1:y:0!dropout_18/dropout/Const:output:0*
T0*+
_output_shapes
:         h2
dropout_18/dropout/Mul{
dropout_18/dropout/ShapeShapelstm_18/transpose_1:y:0*
T0*
_output_shapes
:2
dropout_18/dropout/Shape┘
/dropout_18/dropout/random_uniform/RandomUniformRandomUniform!dropout_18/dropout/Shape:output:0*
T0*+
_output_shapes
:         h*
dtype021
/dropout_18/dropout/random_uniform/RandomUniformЛ
!dropout_18/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А>2#
!dropout_18/dropout/GreaterEqual/yю
dropout_18/dropout/GreaterEqualGreaterEqual8dropout_18/dropout/random_uniform/RandomUniform:output:0*dropout_18/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:         h2!
dropout_18/dropout/GreaterEqualд
dropout_18/dropout/CastCast#dropout_18/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:         h2
dropout_18/dropout/Castк
dropout_18/dropout/Mul_1Muldropout_18/dropout/Mul:z:0dropout_18/dropout/Cast:y:0*
T0*+
_output_shapes
:         h2
dropout_18/dropout/Mul_1j
lstm_19/ShapeShapedropout_18/dropout/Mul_1:z:0*
T0*
_output_shapes
:2
lstm_19/ShapeД
lstm_19/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_19/strided_slice/stackИ
lstm_19/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_19/strided_slice/stack_1И
lstm_19/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_19/strided_slice/stack_2Т
lstm_19/strided_sliceStridedSlicelstm_19/Shape:output:0$lstm_19/strided_slice/stack:output:0&lstm_19/strided_slice/stack_1:output:0&lstm_19/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_19/strided_slicem
lstm_19/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :╬2
lstm_19/zeros/mul/yМ
lstm_19/zeros/mulMullstm_19/strided_slice:output:0lstm_19/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_19/zeros/mulo
lstm_19/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
lstm_19/zeros/Less/yЗ
lstm_19/zeros/LessLesslstm_19/zeros/mul:z:0lstm_19/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_19/zeros/Lesss
lstm_19/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :╬2
lstm_19/zeros/packed/1г
lstm_19/zeros/packedPacklstm_19/strided_slice:output:0lstm_19/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_19/zeros/packedo
lstm_19/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_19/zeros/ConstЦ
lstm_19/zerosFilllstm_19/zeros/packed:output:0lstm_19/zeros/Const:output:0*
T0*(
_output_shapes
:         ╬2
lstm_19/zerosq
lstm_19/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :╬2
lstm_19/zeros_1/mul/yТ
lstm_19/zeros_1/mulMullstm_19/strided_slice:output:0lstm_19/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_19/zeros_1/muls
lstm_19/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
lstm_19/zeros_1/Less/yП
lstm_19/zeros_1/LessLesslstm_19/zeros_1/mul:z:0lstm_19/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_19/zeros_1/Lessw
lstm_19/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :╬2
lstm_19/zeros_1/packed/1й
lstm_19/zeros_1/packedPacklstm_19/strided_slice:output:0!lstm_19/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_19/zeros_1/packeds
lstm_19/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_19/zeros_1/ConstЮ
lstm_19/zeros_1Filllstm_19/zeros_1/packed:output:0lstm_19/zeros_1/Const:output:0*
T0*(
_output_shapes
:         ╬2
lstm_19/zeros_1Е
lstm_19/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_19/transpose/permи
lstm_19/transpose	Transposedropout_18/dropout/Mul_1:z:0lstm_19/transpose/perm:output:0*
T0*+
_output_shapes
:         h2
lstm_19/transposeg
lstm_19/Shape_1Shapelstm_19/transpose:y:0*
T0*
_output_shapes
:2
lstm_19/Shape_1И
lstm_19/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_19/strided_slice_1/stackМ
lstm_19/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_19/strided_slice_1/stack_1М
lstm_19/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_19/strided_slice_1/stack_2Ю
lstm_19/strided_slice_1StridedSlicelstm_19/Shape_1:output:0&lstm_19/strided_slice_1/stack:output:0(lstm_19/strided_slice_1/stack_1:output:0(lstm_19/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_19/strided_slice_1Х
#lstm_19/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         2%
#lstm_19/TensorArrayV2/element_shape╥
lstm_19/TensorArrayV2TensorListReserve,lstm_19/TensorArrayV2/element_shape:output:0 lstm_19/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_19/TensorArrayV2╧
=lstm_19/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    h   2?
=lstm_19/TensorArrayUnstack/TensorListFromTensor/element_shapeШ
/lstm_19/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_19/transpose:y:0Flstm_19/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type021
/lstm_19/TensorArrayUnstack/TensorListFromTensorИ
lstm_19/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_19/strided_slice_2/stackМ
lstm_19/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_19/strided_slice_2/stack_1М
lstm_19/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_19/strided_slice_2/stack_2м
lstm_19/strided_slice_2StridedSlicelstm_19/transpose:y:0&lstm_19/strided_slice_2/stack:output:0(lstm_19/strided_slice_2/stack_1:output:0(lstm_19/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         h*
shrink_axis_mask2
lstm_19/strided_slice_2═
*lstm_19/lstm_cell_19/MatMul/ReadVariableOpReadVariableOp3lstm_19_lstm_cell_19_matmul_readvariableop_resource*
_output_shapes
:	h╕
*
dtype02,
*lstm_19/lstm_cell_19/MatMul/ReadVariableOp═
lstm_19/lstm_cell_19/MatMulMatMul lstm_19/strided_slice_2:output:02lstm_19/lstm_cell_19/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╕
2
lstm_19/lstm_cell_19/MatMul╘
,lstm_19/lstm_cell_19/MatMul_1/ReadVariableOpReadVariableOp5lstm_19_lstm_cell_19_matmul_1_readvariableop_resource* 
_output_shapes
:
╬╕
*
dtype02.
,lstm_19/lstm_cell_19/MatMul_1/ReadVariableOp╔
lstm_19/lstm_cell_19/MatMul_1MatMullstm_19/zeros:output:04lstm_19/lstm_cell_19/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╕
2
lstm_19/lstm_cell_19/MatMul_1└
lstm_19/lstm_cell_19/addAddV2%lstm_19/lstm_cell_19/MatMul:product:0'lstm_19/lstm_cell_19/MatMul_1:product:0*
T0*(
_output_shapes
:         ╕
2
lstm_19/lstm_cell_19/add╠
+lstm_19/lstm_cell_19/BiasAdd/ReadVariableOpReadVariableOp4lstm_19_lstm_cell_19_biasadd_readvariableop_resource*
_output_shapes	
:╕
*
dtype02-
+lstm_19/lstm_cell_19/BiasAdd/ReadVariableOp═
lstm_19/lstm_cell_19/BiasAddBiasAddlstm_19/lstm_cell_19/add:z:03lstm_19/lstm_cell_19/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╕
2
lstm_19/lstm_cell_19/BiasAddО
$lstm_19/lstm_cell_19/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2&
$lstm_19/lstm_cell_19/split/split_dimЧ
lstm_19/lstm_cell_19/splitSplit-lstm_19/lstm_cell_19/split/split_dim:output:0%lstm_19/lstm_cell_19/BiasAdd:output:0*
T0*d
_output_shapesR
P:         ╬:         ╬:         ╬:         ╬*
	num_split2
lstm_19/lstm_cell_19/splitЯ
lstm_19/lstm_cell_19/SigmoidSigmoid#lstm_19/lstm_cell_19/split:output:0*
T0*(
_output_shapes
:         ╬2
lstm_19/lstm_cell_19/Sigmoidг
lstm_19/lstm_cell_19/Sigmoid_1Sigmoid#lstm_19/lstm_cell_19/split:output:1*
T0*(
_output_shapes
:         ╬2 
lstm_19/lstm_cell_19/Sigmoid_1м
lstm_19/lstm_cell_19/mulMul"lstm_19/lstm_cell_19/Sigmoid_1:y:0lstm_19/zeros_1:output:0*
T0*(
_output_shapes
:         ╬2
lstm_19/lstm_cell_19/mulЦ
lstm_19/lstm_cell_19/ReluRelu#lstm_19/lstm_cell_19/split:output:2*
T0*(
_output_shapes
:         ╬2
lstm_19/lstm_cell_19/Relu╜
lstm_19/lstm_cell_19/mul_1Mul lstm_19/lstm_cell_19/Sigmoid:y:0'lstm_19/lstm_cell_19/Relu:activations:0*
T0*(
_output_shapes
:         ╬2
lstm_19/lstm_cell_19/mul_1▓
lstm_19/lstm_cell_19/add_1AddV2lstm_19/lstm_cell_19/mul:z:0lstm_19/lstm_cell_19/mul_1:z:0*
T0*(
_output_shapes
:         ╬2
lstm_19/lstm_cell_19/add_1г
lstm_19/lstm_cell_19/Sigmoid_2Sigmoid#lstm_19/lstm_cell_19/split:output:3*
T0*(
_output_shapes
:         ╬2 
lstm_19/lstm_cell_19/Sigmoid_2Х
lstm_19/lstm_cell_19/Relu_1Relulstm_19/lstm_cell_19/add_1:z:0*
T0*(
_output_shapes
:         ╬2
lstm_19/lstm_cell_19/Relu_1┴
lstm_19/lstm_cell_19/mul_2Mul"lstm_19/lstm_cell_19/Sigmoid_2:y:0)lstm_19/lstm_cell_19/Relu_1:activations:0*
T0*(
_output_shapes
:         ╬2
lstm_19/lstm_cell_19/mul_2Я
%lstm_19/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    N  2'
%lstm_19/TensorArrayV2_1/element_shape╪
lstm_19/TensorArrayV2_1TensorListReserve.lstm_19/TensorArrayV2_1/element_shape:output:0 lstm_19/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_19/TensorArrayV2_1^
lstm_19/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_19/timeП
 lstm_19/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         2"
 lstm_19/while/maximum_iterationsz
lstm_19/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_19/while/loop_counterЛ
lstm_19/whileWhile#lstm_19/while/loop_counter:output:0)lstm_19/while/maximum_iterations:output:0lstm_19/time:output:0 lstm_19/TensorArrayV2_1:handle:0lstm_19/zeros:output:0lstm_19/zeros_1:output:0 lstm_19/strided_slice_1:output:0?lstm_19/TensorArrayUnstack/TensorListFromTensor:output_handle:03lstm_19_lstm_cell_19_matmul_readvariableop_resource5lstm_19_lstm_cell_19_matmul_1_readvariableop_resource4lstm_19_lstm_cell_19_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :         ╬:         ╬: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *'
bodyR
lstm_19_while_body_32589132*'
condR
lstm_19_while_cond_32589131*M
output_shapes<
:: : : : :         ╬:         ╬: : : : : *
parallel_iterations 2
lstm_19/while┼
8lstm_19/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    N  2:
8lstm_19/TensorArrayV2Stack/TensorListStack/element_shapeЙ
*lstm_19/TensorArrayV2Stack/TensorListStackTensorListStacklstm_19/while:output:3Alstm_19/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:         ╬*
element_dtype02,
*lstm_19/TensorArrayV2Stack/TensorListStackС
lstm_19/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         2
lstm_19/strided_slice_3/stackМ
lstm_19/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2!
lstm_19/strided_slice_3/stack_1М
lstm_19/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_19/strided_slice_3/stack_2╦
lstm_19/strided_slice_3StridedSlice3lstm_19/TensorArrayV2Stack/TensorListStack:tensor:0&lstm_19/strided_slice_3/stack:output:0(lstm_19/strided_slice_3/stack_1:output:0(lstm_19/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         ╬*
shrink_axis_mask2
lstm_19/strided_slice_3Й
lstm_19/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_19/transpose_1/perm╞
lstm_19/transpose_1	Transpose3lstm_19/TensorArrayV2Stack/TensorListStack:tensor:0!lstm_19/transpose_1/perm:output:0*
T0*,
_output_shapes
:         ╬2
lstm_19/transpose_1v
lstm_19/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_19/runtimey
dropout_19/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  а?2
dropout_19/dropout/Constк
dropout_19/dropout/MulMullstm_19/transpose_1:y:0!dropout_19/dropout/Const:output:0*
T0*,
_output_shapes
:         ╬2
dropout_19/dropout/Mul{
dropout_19/dropout/ShapeShapelstm_19/transpose_1:y:0*
T0*
_output_shapes
:2
dropout_19/dropout/Shape┌
/dropout_19/dropout/random_uniform/RandomUniformRandomUniform!dropout_19/dropout/Shape:output:0*
T0*,
_output_shapes
:         ╬*
dtype021
/dropout_19/dropout/random_uniform/RandomUniformЛ
!dropout_19/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>2#
!dropout_19/dropout/GreaterEqual/yя
dropout_19/dropout/GreaterEqualGreaterEqual8dropout_19/dropout/random_uniform/RandomUniform:output:0*dropout_19/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:         ╬2!
dropout_19/dropout/GreaterEqualе
dropout_19/dropout/CastCast#dropout_19/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:         ╬2
dropout_19/dropout/Castл
dropout_19/dropout/Mul_1Muldropout_19/dropout/Mul:z:0dropout_19/dropout/Cast:y:0*
T0*,
_output_shapes
:         ╬2
dropout_19/dropout/Mul_1п
 dense_9/Tensordot/ReadVariableOpReadVariableOp)dense_9_tensordot_readvariableop_resource*
_output_shapes
:	╬*
dtype02"
 dense_9/Tensordot/ReadVariableOpz
dense_9/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_9/Tensordot/axesБ
dense_9/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_9/Tensordot/free~
dense_9/Tensordot/ShapeShapedropout_19/dropout/Mul_1:z:0*
T0*
_output_shapes
:2
dense_9/Tensordot/ShapeД
dense_9/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_9/Tensordot/GatherV2/axis∙
dense_9/Tensordot/GatherV2GatherV2 dense_9/Tensordot/Shape:output:0dense_9/Tensordot/free:output:0(dense_9/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_9/Tensordot/GatherV2И
!dense_9/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_9/Tensordot/GatherV2_1/axis 
dense_9/Tensordot/GatherV2_1GatherV2 dense_9/Tensordot/Shape:output:0dense_9/Tensordot/axes:output:0*dense_9/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_9/Tensordot/GatherV2_1|
dense_9/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_9/Tensordot/Constа
dense_9/Tensordot/ProdProd#dense_9/Tensordot/GatherV2:output:0 dense_9/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_9/Tensordot/ProdА
dense_9/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_9/Tensordot/Const_1и
dense_9/Tensordot/Prod_1Prod%dense_9/Tensordot/GatherV2_1:output:0"dense_9/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_9/Tensordot/Prod_1А
dense_9/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense_9/Tensordot/concat/axis╪
dense_9/Tensordot/concatConcatV2dense_9/Tensordot/free:output:0dense_9/Tensordot/axes:output:0&dense_9/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_9/Tensordot/concatм
dense_9/Tensordot/stackPackdense_9/Tensordot/Prod:output:0!dense_9/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_9/Tensordot/stack┐
dense_9/Tensordot/transpose	Transposedropout_19/dropout/Mul_1:z:0!dense_9/Tensordot/concat:output:0*
T0*,
_output_shapes
:         ╬2
dense_9/Tensordot/transpose┐
dense_9/Tensordot/ReshapeReshapedense_9/Tensordot/transpose:y:0 dense_9/Tensordot/stack:output:0*
T0*0
_output_shapes
:                  2
dense_9/Tensordot/Reshape╛
dense_9/Tensordot/MatMulMatMul"dense_9/Tensordot/Reshape:output:0(dense_9/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_9/Tensordot/MatMulА
dense_9/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
dense_9/Tensordot/Const_2Д
dense_9/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_9/Tensordot/concat_1/axisх
dense_9/Tensordot/concat_1ConcatV2#dense_9/Tensordot/GatherV2:output:0"dense_9/Tensordot/Const_2:output:0(dense_9/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_9/Tensordot/concat_1░
dense_9/TensordotReshape"dense_9/Tensordot/MatMul:product:0#dense_9/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:         2
dense_9/Tensordotд
dense_9/BiasAdd/ReadVariableOpReadVariableOp'dense_9_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_9/BiasAdd/ReadVariableOpз
dense_9/BiasAddBiasAdddense_9/Tensordot:output:0&dense_9/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:         2
dense_9/BiasAdd}
dense_9/SoftmaxSoftmaxdense_9/BiasAdd:output:0*
T0*+
_output_shapes
:         2
dense_9/Softmaxx
IdentityIdentitydense_9/Softmax:softmax:0^NoOp*
T0*+
_output_shapes
:         2

Identity╞
NoOpNoOp^dense_9/BiasAdd/ReadVariableOp!^dense_9/Tensordot/ReadVariableOp,^lstm_18/lstm_cell_18/BiasAdd/ReadVariableOp+^lstm_18/lstm_cell_18/MatMul/ReadVariableOp-^lstm_18/lstm_cell_18/MatMul_1/ReadVariableOp^lstm_18/while,^lstm_19/lstm_cell_19/BiasAdd/ReadVariableOp+^lstm_19/lstm_cell_19/MatMul/ReadVariableOp-^lstm_19/lstm_cell_19/MatMul_1/ReadVariableOp^lstm_19/while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':         ]: : : : : : : : 2@
dense_9/BiasAdd/ReadVariableOpdense_9/BiasAdd/ReadVariableOp2D
 dense_9/Tensordot/ReadVariableOp dense_9/Tensordot/ReadVariableOp2Z
+lstm_18/lstm_cell_18/BiasAdd/ReadVariableOp+lstm_18/lstm_cell_18/BiasAdd/ReadVariableOp2X
*lstm_18/lstm_cell_18/MatMul/ReadVariableOp*lstm_18/lstm_cell_18/MatMul/ReadVariableOp2\
,lstm_18/lstm_cell_18/MatMul_1/ReadVariableOp,lstm_18/lstm_cell_18/MatMul_1/ReadVariableOp2
lstm_18/whilelstm_18/while2Z
+lstm_19/lstm_cell_19/BiasAdd/ReadVariableOp+lstm_19/lstm_cell_19/BiasAdd/ReadVariableOp2X
*lstm_19/lstm_cell_19/MatMul/ReadVariableOp*lstm_19/lstm_cell_19/MatMul/ReadVariableOp2\
,lstm_19/lstm_cell_19/MatMul_1/ReadVariableOp,lstm_19/lstm_cell_19/MatMul_1/ReadVariableOp2
lstm_19/whilelstm_19/while:S O
+
_output_shapes
:         ]
 
_user_specified_nameinputs
к

╤
/__inference_sequential_9_layer_call_fn_32588020
lstm_18_input
unknown:	]а
	unknown_0:	hа
	unknown_1:	а
	unknown_2:	h╕

	unknown_3:
╬╕

	unknown_4:	╕

	unknown_5:	╬
	unknown_6:
identityИвStatefulPartitionedCall╙
StatefulPartitionedCallStatefulPartitionedCalllstm_18_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         **
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_sequential_9_layer_call_and_return_conditional_losses_325880012
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:         2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':         ]: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
+
_output_shapes
:         ]
'
_user_specified_namelstm_18_input
╗
f
-__inference_dropout_19_layer_call_fn_32590643

inputs
identityИвStatefulPartitionedCallу
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ╬* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_dropout_19_layer_call_and_return_conditional_losses_325880502
StatefulPartitionedCallА
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:         ╬2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ╬22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:         ╬
 
_user_specified_nameinputs
╒°
Е
J__inference_sequential_9_layer_call_and_return_conditional_losses_32588910

inputsF
3lstm_18_lstm_cell_18_matmul_readvariableop_resource:	]аH
5lstm_18_lstm_cell_18_matmul_1_readvariableop_resource:	hаC
4lstm_18_lstm_cell_18_biasadd_readvariableop_resource:	аF
3lstm_19_lstm_cell_19_matmul_readvariableop_resource:	h╕
I
5lstm_19_lstm_cell_19_matmul_1_readvariableop_resource:
╬╕
C
4lstm_19_lstm_cell_19_biasadd_readvariableop_resource:	╕
<
)dense_9_tensordot_readvariableop_resource:	╬5
'dense_9_biasadd_readvariableop_resource:
identityИвdense_9/BiasAdd/ReadVariableOpв dense_9/Tensordot/ReadVariableOpв+lstm_18/lstm_cell_18/BiasAdd/ReadVariableOpв*lstm_18/lstm_cell_18/MatMul/ReadVariableOpв,lstm_18/lstm_cell_18/MatMul_1/ReadVariableOpвlstm_18/whileв+lstm_19/lstm_cell_19/BiasAdd/ReadVariableOpв*lstm_19/lstm_cell_19/MatMul/ReadVariableOpв,lstm_19/lstm_cell_19/MatMul_1/ReadVariableOpвlstm_19/whileT
lstm_18/ShapeShapeinputs*
T0*
_output_shapes
:2
lstm_18/ShapeД
lstm_18/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_18/strided_slice/stackИ
lstm_18/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_18/strided_slice/stack_1И
lstm_18/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_18/strided_slice/stack_2Т
lstm_18/strided_sliceStridedSlicelstm_18/Shape:output:0$lstm_18/strided_slice/stack:output:0&lstm_18/strided_slice/stack_1:output:0&lstm_18/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_18/strided_slicel
lstm_18/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :h2
lstm_18/zeros/mul/yМ
lstm_18/zeros/mulMullstm_18/strided_slice:output:0lstm_18/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_18/zeros/mulo
lstm_18/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
lstm_18/zeros/Less/yЗ
lstm_18/zeros/LessLesslstm_18/zeros/mul:z:0lstm_18/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_18/zeros/Lessr
lstm_18/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :h2
lstm_18/zeros/packed/1г
lstm_18/zeros/packedPacklstm_18/strided_slice:output:0lstm_18/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_18/zeros/packedo
lstm_18/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_18/zeros/ConstХ
lstm_18/zerosFilllstm_18/zeros/packed:output:0lstm_18/zeros/Const:output:0*
T0*'
_output_shapes
:         h2
lstm_18/zerosp
lstm_18/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :h2
lstm_18/zeros_1/mul/yТ
lstm_18/zeros_1/mulMullstm_18/strided_slice:output:0lstm_18/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_18/zeros_1/muls
lstm_18/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
lstm_18/zeros_1/Less/yП
lstm_18/zeros_1/LessLesslstm_18/zeros_1/mul:z:0lstm_18/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_18/zeros_1/Lessv
lstm_18/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :h2
lstm_18/zeros_1/packed/1й
lstm_18/zeros_1/packedPacklstm_18/strided_slice:output:0!lstm_18/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_18/zeros_1/packeds
lstm_18/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_18/zeros_1/ConstЭ
lstm_18/zeros_1Filllstm_18/zeros_1/packed:output:0lstm_18/zeros_1/Const:output:0*
T0*'
_output_shapes
:         h2
lstm_18/zeros_1Е
lstm_18/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_18/transpose/permТ
lstm_18/transpose	Transposeinputslstm_18/transpose/perm:output:0*
T0*+
_output_shapes
:         ]2
lstm_18/transposeg
lstm_18/Shape_1Shapelstm_18/transpose:y:0*
T0*
_output_shapes
:2
lstm_18/Shape_1И
lstm_18/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_18/strided_slice_1/stackМ
lstm_18/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_18/strided_slice_1/stack_1М
lstm_18/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_18/strided_slice_1/stack_2Ю
lstm_18/strided_slice_1StridedSlicelstm_18/Shape_1:output:0&lstm_18/strided_slice_1/stack:output:0(lstm_18/strided_slice_1/stack_1:output:0(lstm_18/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_18/strided_slice_1Х
#lstm_18/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         2%
#lstm_18/TensorArrayV2/element_shape╥
lstm_18/TensorArrayV2TensorListReserve,lstm_18/TensorArrayV2/element_shape:output:0 lstm_18/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_18/TensorArrayV2╧
=lstm_18/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ]   2?
=lstm_18/TensorArrayUnstack/TensorListFromTensor/element_shapeШ
/lstm_18/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_18/transpose:y:0Flstm_18/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type021
/lstm_18/TensorArrayUnstack/TensorListFromTensorИ
lstm_18/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_18/strided_slice_2/stackМ
lstm_18/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_18/strided_slice_2/stack_1М
lstm_18/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_18/strided_slice_2/stack_2м
lstm_18/strided_slice_2StridedSlicelstm_18/transpose:y:0&lstm_18/strided_slice_2/stack:output:0(lstm_18/strided_slice_2/stack_1:output:0(lstm_18/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         ]*
shrink_axis_mask2
lstm_18/strided_slice_2═
*lstm_18/lstm_cell_18/MatMul/ReadVariableOpReadVariableOp3lstm_18_lstm_cell_18_matmul_readvariableop_resource*
_output_shapes
:	]а*
dtype02,
*lstm_18/lstm_cell_18/MatMul/ReadVariableOp═
lstm_18/lstm_cell_18/MatMulMatMul lstm_18/strided_slice_2:output:02lstm_18/lstm_cell_18/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         а2
lstm_18/lstm_cell_18/MatMul╙
,lstm_18/lstm_cell_18/MatMul_1/ReadVariableOpReadVariableOp5lstm_18_lstm_cell_18_matmul_1_readvariableop_resource*
_output_shapes
:	hа*
dtype02.
,lstm_18/lstm_cell_18/MatMul_1/ReadVariableOp╔
lstm_18/lstm_cell_18/MatMul_1MatMullstm_18/zeros:output:04lstm_18/lstm_cell_18/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         а2
lstm_18/lstm_cell_18/MatMul_1└
lstm_18/lstm_cell_18/addAddV2%lstm_18/lstm_cell_18/MatMul:product:0'lstm_18/lstm_cell_18/MatMul_1:product:0*
T0*(
_output_shapes
:         а2
lstm_18/lstm_cell_18/add╠
+lstm_18/lstm_cell_18/BiasAdd/ReadVariableOpReadVariableOp4lstm_18_lstm_cell_18_biasadd_readvariableop_resource*
_output_shapes	
:а*
dtype02-
+lstm_18/lstm_cell_18/BiasAdd/ReadVariableOp═
lstm_18/lstm_cell_18/BiasAddBiasAddlstm_18/lstm_cell_18/add:z:03lstm_18/lstm_cell_18/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         а2
lstm_18/lstm_cell_18/BiasAddО
$lstm_18/lstm_cell_18/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2&
$lstm_18/lstm_cell_18/split/split_dimУ
lstm_18/lstm_cell_18/splitSplit-lstm_18/lstm_cell_18/split/split_dim:output:0%lstm_18/lstm_cell_18/BiasAdd:output:0*
T0*`
_output_shapesN
L:         h:         h:         h:         h*
	num_split2
lstm_18/lstm_cell_18/splitЮ
lstm_18/lstm_cell_18/SigmoidSigmoid#lstm_18/lstm_cell_18/split:output:0*
T0*'
_output_shapes
:         h2
lstm_18/lstm_cell_18/Sigmoidв
lstm_18/lstm_cell_18/Sigmoid_1Sigmoid#lstm_18/lstm_cell_18/split:output:1*
T0*'
_output_shapes
:         h2 
lstm_18/lstm_cell_18/Sigmoid_1л
lstm_18/lstm_cell_18/mulMul"lstm_18/lstm_cell_18/Sigmoid_1:y:0lstm_18/zeros_1:output:0*
T0*'
_output_shapes
:         h2
lstm_18/lstm_cell_18/mulХ
lstm_18/lstm_cell_18/ReluRelu#lstm_18/lstm_cell_18/split:output:2*
T0*'
_output_shapes
:         h2
lstm_18/lstm_cell_18/Relu╝
lstm_18/lstm_cell_18/mul_1Mul lstm_18/lstm_cell_18/Sigmoid:y:0'lstm_18/lstm_cell_18/Relu:activations:0*
T0*'
_output_shapes
:         h2
lstm_18/lstm_cell_18/mul_1▒
lstm_18/lstm_cell_18/add_1AddV2lstm_18/lstm_cell_18/mul:z:0lstm_18/lstm_cell_18/mul_1:z:0*
T0*'
_output_shapes
:         h2
lstm_18/lstm_cell_18/add_1в
lstm_18/lstm_cell_18/Sigmoid_2Sigmoid#lstm_18/lstm_cell_18/split:output:3*
T0*'
_output_shapes
:         h2 
lstm_18/lstm_cell_18/Sigmoid_2Ф
lstm_18/lstm_cell_18/Relu_1Relulstm_18/lstm_cell_18/add_1:z:0*
T0*'
_output_shapes
:         h2
lstm_18/lstm_cell_18/Relu_1└
lstm_18/lstm_cell_18/mul_2Mul"lstm_18/lstm_cell_18/Sigmoid_2:y:0)lstm_18/lstm_cell_18/Relu_1:activations:0*
T0*'
_output_shapes
:         h2
lstm_18/lstm_cell_18/mul_2Я
%lstm_18/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    h   2'
%lstm_18/TensorArrayV2_1/element_shape╪
lstm_18/TensorArrayV2_1TensorListReserve.lstm_18/TensorArrayV2_1/element_shape:output:0 lstm_18/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_18/TensorArrayV2_1^
lstm_18/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_18/timeП
 lstm_18/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         2"
 lstm_18/while/maximum_iterationsz
lstm_18/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_18/while/loop_counterЗ
lstm_18/whileWhile#lstm_18/while/loop_counter:output:0)lstm_18/while/maximum_iterations:output:0lstm_18/time:output:0 lstm_18/TensorArrayV2_1:handle:0lstm_18/zeros:output:0lstm_18/zeros_1:output:0 lstm_18/strided_slice_1:output:0?lstm_18/TensorArrayUnstack/TensorListFromTensor:output_handle:03lstm_18_lstm_cell_18_matmul_readvariableop_resource5lstm_18_lstm_cell_18_matmul_1_readvariableop_resource4lstm_18_lstm_cell_18_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         h:         h: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *'
bodyR
lstm_18_while_body_32588650*'
condR
lstm_18_while_cond_32588649*K
output_shapes:
8: : : : :         h:         h: : : : : *
parallel_iterations 2
lstm_18/while┼
8lstm_18/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    h   2:
8lstm_18/TensorArrayV2Stack/TensorListStack/element_shapeИ
*lstm_18/TensorArrayV2Stack/TensorListStackTensorListStacklstm_18/while:output:3Alstm_18/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:         h*
element_dtype02,
*lstm_18/TensorArrayV2Stack/TensorListStackС
lstm_18/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         2
lstm_18/strided_slice_3/stackМ
lstm_18/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2!
lstm_18/strided_slice_3/stack_1М
lstm_18/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_18/strided_slice_3/stack_2╩
lstm_18/strided_slice_3StridedSlice3lstm_18/TensorArrayV2Stack/TensorListStack:tensor:0&lstm_18/strided_slice_3/stack:output:0(lstm_18/strided_slice_3/stack_1:output:0(lstm_18/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         h*
shrink_axis_mask2
lstm_18/strided_slice_3Й
lstm_18/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_18/transpose_1/perm┼
lstm_18/transpose_1	Transpose3lstm_18/TensorArrayV2Stack/TensorListStack:tensor:0!lstm_18/transpose_1/perm:output:0*
T0*+
_output_shapes
:         h2
lstm_18/transpose_1v
lstm_18/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_18/runtimeЕ
dropout_18/IdentityIdentitylstm_18/transpose_1:y:0*
T0*+
_output_shapes
:         h2
dropout_18/Identityj
lstm_19/ShapeShapedropout_18/Identity:output:0*
T0*
_output_shapes
:2
lstm_19/ShapeД
lstm_19/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_19/strided_slice/stackИ
lstm_19/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_19/strided_slice/stack_1И
lstm_19/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_19/strided_slice/stack_2Т
lstm_19/strided_sliceStridedSlicelstm_19/Shape:output:0$lstm_19/strided_slice/stack:output:0&lstm_19/strided_slice/stack_1:output:0&lstm_19/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_19/strided_slicem
lstm_19/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :╬2
lstm_19/zeros/mul/yМ
lstm_19/zeros/mulMullstm_19/strided_slice:output:0lstm_19/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_19/zeros/mulo
lstm_19/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
lstm_19/zeros/Less/yЗ
lstm_19/zeros/LessLesslstm_19/zeros/mul:z:0lstm_19/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_19/zeros/Lesss
lstm_19/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :╬2
lstm_19/zeros/packed/1г
lstm_19/zeros/packedPacklstm_19/strided_slice:output:0lstm_19/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_19/zeros/packedo
lstm_19/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_19/zeros/ConstЦ
lstm_19/zerosFilllstm_19/zeros/packed:output:0lstm_19/zeros/Const:output:0*
T0*(
_output_shapes
:         ╬2
lstm_19/zerosq
lstm_19/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :╬2
lstm_19/zeros_1/mul/yТ
lstm_19/zeros_1/mulMullstm_19/strided_slice:output:0lstm_19/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_19/zeros_1/muls
lstm_19/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
lstm_19/zeros_1/Less/yП
lstm_19/zeros_1/LessLesslstm_19/zeros_1/mul:z:0lstm_19/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_19/zeros_1/Lessw
lstm_19/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :╬2
lstm_19/zeros_1/packed/1й
lstm_19/zeros_1/packedPacklstm_19/strided_slice:output:0!lstm_19/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_19/zeros_1/packeds
lstm_19/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_19/zeros_1/ConstЮ
lstm_19/zeros_1Filllstm_19/zeros_1/packed:output:0lstm_19/zeros_1/Const:output:0*
T0*(
_output_shapes
:         ╬2
lstm_19/zeros_1Е
lstm_19/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_19/transpose/permи
lstm_19/transpose	Transposedropout_18/Identity:output:0lstm_19/transpose/perm:output:0*
T0*+
_output_shapes
:         h2
lstm_19/transposeg
lstm_19/Shape_1Shapelstm_19/transpose:y:0*
T0*
_output_shapes
:2
lstm_19/Shape_1И
lstm_19/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_19/strided_slice_1/stackМ
lstm_19/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_19/strided_slice_1/stack_1М
lstm_19/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_19/strided_slice_1/stack_2Ю
lstm_19/strided_slice_1StridedSlicelstm_19/Shape_1:output:0&lstm_19/strided_slice_1/stack:output:0(lstm_19/strided_slice_1/stack_1:output:0(lstm_19/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_19/strided_slice_1Х
#lstm_19/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         2%
#lstm_19/TensorArrayV2/element_shape╥
lstm_19/TensorArrayV2TensorListReserve,lstm_19/TensorArrayV2/element_shape:output:0 lstm_19/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_19/TensorArrayV2╧
=lstm_19/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    h   2?
=lstm_19/TensorArrayUnstack/TensorListFromTensor/element_shapeШ
/lstm_19/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_19/transpose:y:0Flstm_19/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type021
/lstm_19/TensorArrayUnstack/TensorListFromTensorИ
lstm_19/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_19/strided_slice_2/stackМ
lstm_19/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_19/strided_slice_2/stack_1М
lstm_19/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_19/strided_slice_2/stack_2м
lstm_19/strided_slice_2StridedSlicelstm_19/transpose:y:0&lstm_19/strided_slice_2/stack:output:0(lstm_19/strided_slice_2/stack_1:output:0(lstm_19/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         h*
shrink_axis_mask2
lstm_19/strided_slice_2═
*lstm_19/lstm_cell_19/MatMul/ReadVariableOpReadVariableOp3lstm_19_lstm_cell_19_matmul_readvariableop_resource*
_output_shapes
:	h╕
*
dtype02,
*lstm_19/lstm_cell_19/MatMul/ReadVariableOp═
lstm_19/lstm_cell_19/MatMulMatMul lstm_19/strided_slice_2:output:02lstm_19/lstm_cell_19/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╕
2
lstm_19/lstm_cell_19/MatMul╘
,lstm_19/lstm_cell_19/MatMul_1/ReadVariableOpReadVariableOp5lstm_19_lstm_cell_19_matmul_1_readvariableop_resource* 
_output_shapes
:
╬╕
*
dtype02.
,lstm_19/lstm_cell_19/MatMul_1/ReadVariableOp╔
lstm_19/lstm_cell_19/MatMul_1MatMullstm_19/zeros:output:04lstm_19/lstm_cell_19/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╕
2
lstm_19/lstm_cell_19/MatMul_1└
lstm_19/lstm_cell_19/addAddV2%lstm_19/lstm_cell_19/MatMul:product:0'lstm_19/lstm_cell_19/MatMul_1:product:0*
T0*(
_output_shapes
:         ╕
2
lstm_19/lstm_cell_19/add╠
+lstm_19/lstm_cell_19/BiasAdd/ReadVariableOpReadVariableOp4lstm_19_lstm_cell_19_biasadd_readvariableop_resource*
_output_shapes	
:╕
*
dtype02-
+lstm_19/lstm_cell_19/BiasAdd/ReadVariableOp═
lstm_19/lstm_cell_19/BiasAddBiasAddlstm_19/lstm_cell_19/add:z:03lstm_19/lstm_cell_19/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╕
2
lstm_19/lstm_cell_19/BiasAddО
$lstm_19/lstm_cell_19/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2&
$lstm_19/lstm_cell_19/split/split_dimЧ
lstm_19/lstm_cell_19/splitSplit-lstm_19/lstm_cell_19/split/split_dim:output:0%lstm_19/lstm_cell_19/BiasAdd:output:0*
T0*d
_output_shapesR
P:         ╬:         ╬:         ╬:         ╬*
	num_split2
lstm_19/lstm_cell_19/splitЯ
lstm_19/lstm_cell_19/SigmoidSigmoid#lstm_19/lstm_cell_19/split:output:0*
T0*(
_output_shapes
:         ╬2
lstm_19/lstm_cell_19/Sigmoidг
lstm_19/lstm_cell_19/Sigmoid_1Sigmoid#lstm_19/lstm_cell_19/split:output:1*
T0*(
_output_shapes
:         ╬2 
lstm_19/lstm_cell_19/Sigmoid_1м
lstm_19/lstm_cell_19/mulMul"lstm_19/lstm_cell_19/Sigmoid_1:y:0lstm_19/zeros_1:output:0*
T0*(
_output_shapes
:         ╬2
lstm_19/lstm_cell_19/mulЦ
lstm_19/lstm_cell_19/ReluRelu#lstm_19/lstm_cell_19/split:output:2*
T0*(
_output_shapes
:         ╬2
lstm_19/lstm_cell_19/Relu╜
lstm_19/lstm_cell_19/mul_1Mul lstm_19/lstm_cell_19/Sigmoid:y:0'lstm_19/lstm_cell_19/Relu:activations:0*
T0*(
_output_shapes
:         ╬2
lstm_19/lstm_cell_19/mul_1▓
lstm_19/lstm_cell_19/add_1AddV2lstm_19/lstm_cell_19/mul:z:0lstm_19/lstm_cell_19/mul_1:z:0*
T0*(
_output_shapes
:         ╬2
lstm_19/lstm_cell_19/add_1г
lstm_19/lstm_cell_19/Sigmoid_2Sigmoid#lstm_19/lstm_cell_19/split:output:3*
T0*(
_output_shapes
:         ╬2 
lstm_19/lstm_cell_19/Sigmoid_2Х
lstm_19/lstm_cell_19/Relu_1Relulstm_19/lstm_cell_19/add_1:z:0*
T0*(
_output_shapes
:         ╬2
lstm_19/lstm_cell_19/Relu_1┴
lstm_19/lstm_cell_19/mul_2Mul"lstm_19/lstm_cell_19/Sigmoid_2:y:0)lstm_19/lstm_cell_19/Relu_1:activations:0*
T0*(
_output_shapes
:         ╬2
lstm_19/lstm_cell_19/mul_2Я
%lstm_19/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    N  2'
%lstm_19/TensorArrayV2_1/element_shape╪
lstm_19/TensorArrayV2_1TensorListReserve.lstm_19/TensorArrayV2_1/element_shape:output:0 lstm_19/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_19/TensorArrayV2_1^
lstm_19/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_19/timeП
 lstm_19/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         2"
 lstm_19/while/maximum_iterationsz
lstm_19/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_19/while/loop_counterЛ
lstm_19/whileWhile#lstm_19/while/loop_counter:output:0)lstm_19/while/maximum_iterations:output:0lstm_19/time:output:0 lstm_19/TensorArrayV2_1:handle:0lstm_19/zeros:output:0lstm_19/zeros_1:output:0 lstm_19/strided_slice_1:output:0?lstm_19/TensorArrayUnstack/TensorListFromTensor:output_handle:03lstm_19_lstm_cell_19_matmul_readvariableop_resource5lstm_19_lstm_cell_19_matmul_1_readvariableop_resource4lstm_19_lstm_cell_19_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :         ╬:         ╬: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *'
bodyR
lstm_19_while_body_32588798*'
condR
lstm_19_while_cond_32588797*M
output_shapes<
:: : : : :         ╬:         ╬: : : : : *
parallel_iterations 2
lstm_19/while┼
8lstm_19/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    N  2:
8lstm_19/TensorArrayV2Stack/TensorListStack/element_shapeЙ
*lstm_19/TensorArrayV2Stack/TensorListStackTensorListStacklstm_19/while:output:3Alstm_19/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:         ╬*
element_dtype02,
*lstm_19/TensorArrayV2Stack/TensorListStackС
lstm_19/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         2
lstm_19/strided_slice_3/stackМ
lstm_19/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2!
lstm_19/strided_slice_3/stack_1М
lstm_19/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_19/strided_slice_3/stack_2╦
lstm_19/strided_slice_3StridedSlice3lstm_19/TensorArrayV2Stack/TensorListStack:tensor:0&lstm_19/strided_slice_3/stack:output:0(lstm_19/strided_slice_3/stack_1:output:0(lstm_19/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         ╬*
shrink_axis_mask2
lstm_19/strided_slice_3Й
lstm_19/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_19/transpose_1/perm╞
lstm_19/transpose_1	Transpose3lstm_19/TensorArrayV2Stack/TensorListStack:tensor:0!lstm_19/transpose_1/perm:output:0*
T0*,
_output_shapes
:         ╬2
lstm_19/transpose_1v
lstm_19/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_19/runtimeЖ
dropout_19/IdentityIdentitylstm_19/transpose_1:y:0*
T0*,
_output_shapes
:         ╬2
dropout_19/Identityп
 dense_9/Tensordot/ReadVariableOpReadVariableOp)dense_9_tensordot_readvariableop_resource*
_output_shapes
:	╬*
dtype02"
 dense_9/Tensordot/ReadVariableOpz
dense_9/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_9/Tensordot/axesБ
dense_9/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_9/Tensordot/free~
dense_9/Tensordot/ShapeShapedropout_19/Identity:output:0*
T0*
_output_shapes
:2
dense_9/Tensordot/ShapeД
dense_9/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_9/Tensordot/GatherV2/axis∙
dense_9/Tensordot/GatherV2GatherV2 dense_9/Tensordot/Shape:output:0dense_9/Tensordot/free:output:0(dense_9/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_9/Tensordot/GatherV2И
!dense_9/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_9/Tensordot/GatherV2_1/axis 
dense_9/Tensordot/GatherV2_1GatherV2 dense_9/Tensordot/Shape:output:0dense_9/Tensordot/axes:output:0*dense_9/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_9/Tensordot/GatherV2_1|
dense_9/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_9/Tensordot/Constа
dense_9/Tensordot/ProdProd#dense_9/Tensordot/GatherV2:output:0 dense_9/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_9/Tensordot/ProdА
dense_9/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_9/Tensordot/Const_1и
dense_9/Tensordot/Prod_1Prod%dense_9/Tensordot/GatherV2_1:output:0"dense_9/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_9/Tensordot/Prod_1А
dense_9/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense_9/Tensordot/concat/axis╪
dense_9/Tensordot/concatConcatV2dense_9/Tensordot/free:output:0dense_9/Tensordot/axes:output:0&dense_9/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_9/Tensordot/concatм
dense_9/Tensordot/stackPackdense_9/Tensordot/Prod:output:0!dense_9/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_9/Tensordot/stack┐
dense_9/Tensordot/transpose	Transposedropout_19/Identity:output:0!dense_9/Tensordot/concat:output:0*
T0*,
_output_shapes
:         ╬2
dense_9/Tensordot/transpose┐
dense_9/Tensordot/ReshapeReshapedense_9/Tensordot/transpose:y:0 dense_9/Tensordot/stack:output:0*
T0*0
_output_shapes
:                  2
dense_9/Tensordot/Reshape╛
dense_9/Tensordot/MatMulMatMul"dense_9/Tensordot/Reshape:output:0(dense_9/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_9/Tensordot/MatMulА
dense_9/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
dense_9/Tensordot/Const_2Д
dense_9/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_9/Tensordot/concat_1/axisх
dense_9/Tensordot/concat_1ConcatV2#dense_9/Tensordot/GatherV2:output:0"dense_9/Tensordot/Const_2:output:0(dense_9/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_9/Tensordot/concat_1░
dense_9/TensordotReshape"dense_9/Tensordot/MatMul:product:0#dense_9/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:         2
dense_9/Tensordotд
dense_9/BiasAdd/ReadVariableOpReadVariableOp'dense_9_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_9/BiasAdd/ReadVariableOpз
dense_9/BiasAddBiasAdddense_9/Tensordot:output:0&dense_9/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:         2
dense_9/BiasAdd}
dense_9/SoftmaxSoftmaxdense_9/BiasAdd:output:0*
T0*+
_output_shapes
:         2
dense_9/Softmaxx
IdentityIdentitydense_9/Softmax:softmax:0^NoOp*
T0*+
_output_shapes
:         2

Identity╞
NoOpNoOp^dense_9/BiasAdd/ReadVariableOp!^dense_9/Tensordot/ReadVariableOp,^lstm_18/lstm_cell_18/BiasAdd/ReadVariableOp+^lstm_18/lstm_cell_18/MatMul/ReadVariableOp-^lstm_18/lstm_cell_18/MatMul_1/ReadVariableOp^lstm_18/while,^lstm_19/lstm_cell_19/BiasAdd/ReadVariableOp+^lstm_19/lstm_cell_19/MatMul/ReadVariableOp-^lstm_19/lstm_cell_19/MatMul_1/ReadVariableOp^lstm_19/while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':         ]: : : : : : : : 2@
dense_9/BiasAdd/ReadVariableOpdense_9/BiasAdd/ReadVariableOp2D
 dense_9/Tensordot/ReadVariableOp dense_9/Tensordot/ReadVariableOp2Z
+lstm_18/lstm_cell_18/BiasAdd/ReadVariableOp+lstm_18/lstm_cell_18/BiasAdd/ReadVariableOp2X
*lstm_18/lstm_cell_18/MatMul/ReadVariableOp*lstm_18/lstm_cell_18/MatMul/ReadVariableOp2\
,lstm_18/lstm_cell_18/MatMul_1/ReadVariableOp,lstm_18/lstm_cell_18/MatMul_1/ReadVariableOp2
lstm_18/whilelstm_18/while2Z
+lstm_19/lstm_cell_19/BiasAdd/ReadVariableOp+lstm_19/lstm_cell_19/BiasAdd/ReadVariableOp2X
*lstm_19/lstm_cell_19/MatMul/ReadVariableOp*lstm_19/lstm_cell_19/MatMul/ReadVariableOp2\
,lstm_19/lstm_cell_19/MatMul_1/ReadVariableOp,lstm_19/lstm_cell_19/MatMul_1/ReadVariableOp2
lstm_19/whilelstm_19/while:S O
+
_output_shapes
:         ]
 
_user_specified_nameinputs
Ъ?
╥
while_body_32588329
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_18_matmul_readvariableop_resource_0:	]аH
5while_lstm_cell_18_matmul_1_readvariableop_resource_0:	hаC
4while_lstm_cell_18_biasadd_readvariableop_resource_0:	а
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_18_matmul_readvariableop_resource:	]аF
3while_lstm_cell_18_matmul_1_readvariableop_resource:	hаA
2while_lstm_cell_18_biasadd_readvariableop_resource:	аИв)while/lstm_cell_18/BiasAdd/ReadVariableOpв(while/lstm_cell_18/MatMul/ReadVariableOpв*while/lstm_cell_18/MatMul_1/ReadVariableOp├
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ]   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape╙
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         ]*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem╔
(while/lstm_cell_18/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_18_matmul_readvariableop_resource_0*
_output_shapes
:	]а*
dtype02*
(while/lstm_cell_18/MatMul/ReadVariableOp╫
while/lstm_cell_18/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_18/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         а2
while/lstm_cell_18/MatMul╧
*while/lstm_cell_18/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_18_matmul_1_readvariableop_resource_0*
_output_shapes
:	hа*
dtype02,
*while/lstm_cell_18/MatMul_1/ReadVariableOp└
while/lstm_cell_18/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_18/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         а2
while/lstm_cell_18/MatMul_1╕
while/lstm_cell_18/addAddV2#while/lstm_cell_18/MatMul:product:0%while/lstm_cell_18/MatMul_1:product:0*
T0*(
_output_shapes
:         а2
while/lstm_cell_18/add╚
)while/lstm_cell_18/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_18_biasadd_readvariableop_resource_0*
_output_shapes	
:а*
dtype02+
)while/lstm_cell_18/BiasAdd/ReadVariableOp┼
while/lstm_cell_18/BiasAddBiasAddwhile/lstm_cell_18/add:z:01while/lstm_cell_18/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         а2
while/lstm_cell_18/BiasAddК
"while/lstm_cell_18/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_18/split/split_dimЛ
while/lstm_cell_18/splitSplit+while/lstm_cell_18/split/split_dim:output:0#while/lstm_cell_18/BiasAdd:output:0*
T0*`
_output_shapesN
L:         h:         h:         h:         h*
	num_split2
while/lstm_cell_18/splitШ
while/lstm_cell_18/SigmoidSigmoid!while/lstm_cell_18/split:output:0*
T0*'
_output_shapes
:         h2
while/lstm_cell_18/SigmoidЬ
while/lstm_cell_18/Sigmoid_1Sigmoid!while/lstm_cell_18/split:output:1*
T0*'
_output_shapes
:         h2
while/lstm_cell_18/Sigmoid_1а
while/lstm_cell_18/mulMul while/lstm_cell_18/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:         h2
while/lstm_cell_18/mulП
while/lstm_cell_18/ReluRelu!while/lstm_cell_18/split:output:2*
T0*'
_output_shapes
:         h2
while/lstm_cell_18/Relu┤
while/lstm_cell_18/mul_1Mulwhile/lstm_cell_18/Sigmoid:y:0%while/lstm_cell_18/Relu:activations:0*
T0*'
_output_shapes
:         h2
while/lstm_cell_18/mul_1й
while/lstm_cell_18/add_1AddV2while/lstm_cell_18/mul:z:0while/lstm_cell_18/mul_1:z:0*
T0*'
_output_shapes
:         h2
while/lstm_cell_18/add_1Ь
while/lstm_cell_18/Sigmoid_2Sigmoid!while/lstm_cell_18/split:output:3*
T0*'
_output_shapes
:         h2
while/lstm_cell_18/Sigmoid_2О
while/lstm_cell_18/Relu_1Reluwhile/lstm_cell_18/add_1:z:0*
T0*'
_output_shapes
:         h2
while/lstm_cell_18/Relu_1╕
while/lstm_cell_18/mul_2Mul while/lstm_cell_18/Sigmoid_2:y:0'while/lstm_cell_18/Relu_1:activations:0*
T0*'
_output_shapes
:         h2
while/lstm_cell_18/mul_2р
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_18/mul_2:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1k
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity~
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_1m
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_2Ъ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3Н
while/Identity_4Identitywhile/lstm_cell_18/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:         h2
while/Identity_4Н
while/Identity_5Identitywhile/lstm_cell_18/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:         h2
while/Identity_5▐

while/NoOpNoOp*^while/lstm_cell_18/BiasAdd/ReadVariableOp)^while/lstm_cell_18/MatMul/ReadVariableOp+^while/lstm_cell_18/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_18_biasadd_readvariableop_resource4while_lstm_cell_18_biasadd_readvariableop_resource_0"l
3while_lstm_cell_18_matmul_1_readvariableop_resource5while_lstm_cell_18_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_18_matmul_readvariableop_resource3while_lstm_cell_18_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         h:         h: : : : : 2V
)while/lstm_cell_18/BiasAdd/ReadVariableOp)while/lstm_cell_18/BiasAdd/ReadVariableOp2T
(while/lstm_cell_18/MatMul/ReadVariableOp(while/lstm_cell_18/MatMul/ReadVariableOp2X
*while/lstm_cell_18/MatMul_1/ReadVariableOp*while/lstm_cell_18/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         h:-)
'
_output_shapes
:         h:

_output_shapes
: :

_output_shapes
: 
у
═
while_cond_32588132
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_32588132___redundant_placeholder06
2while_while_cond_32588132___redundant_placeholder16
2while_while_cond_32588132___redundant_placeholder26
2while_while_cond_32588132___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :         ╬:         ╬: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:         ╬:.*
(
_output_shapes
:         ╬:

_output_shapes
: :

_output_shapes
:
░?
╘
while_body_32590488
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_19_matmul_readvariableop_resource_0:	h╕
I
5while_lstm_cell_19_matmul_1_readvariableop_resource_0:
╬╕
C
4while_lstm_cell_19_biasadd_readvariableop_resource_0:	╕

while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_19_matmul_readvariableop_resource:	h╕
G
3while_lstm_cell_19_matmul_1_readvariableop_resource:
╬╕
A
2while_lstm_cell_19_biasadd_readvariableop_resource:	╕
Ив)while/lstm_cell_19/BiasAdd/ReadVariableOpв(while/lstm_cell_19/MatMul/ReadVariableOpв*while/lstm_cell_19/MatMul_1/ReadVariableOp├
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    h   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape╙
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         h*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem╔
(while/lstm_cell_19/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_19_matmul_readvariableop_resource_0*
_output_shapes
:	h╕
*
dtype02*
(while/lstm_cell_19/MatMul/ReadVariableOp╫
while/lstm_cell_19/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_19/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╕
2
while/lstm_cell_19/MatMul╨
*while/lstm_cell_19/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_19_matmul_1_readvariableop_resource_0* 
_output_shapes
:
╬╕
*
dtype02,
*while/lstm_cell_19/MatMul_1/ReadVariableOp└
while/lstm_cell_19/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_19/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╕
2
while/lstm_cell_19/MatMul_1╕
while/lstm_cell_19/addAddV2#while/lstm_cell_19/MatMul:product:0%while/lstm_cell_19/MatMul_1:product:0*
T0*(
_output_shapes
:         ╕
2
while/lstm_cell_19/add╚
)while/lstm_cell_19/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_19_biasadd_readvariableop_resource_0*
_output_shapes	
:╕
*
dtype02+
)while/lstm_cell_19/BiasAdd/ReadVariableOp┼
while/lstm_cell_19/BiasAddBiasAddwhile/lstm_cell_19/add:z:01while/lstm_cell_19/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╕
2
while/lstm_cell_19/BiasAddК
"while/lstm_cell_19/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_19/split/split_dimП
while/lstm_cell_19/splitSplit+while/lstm_cell_19/split/split_dim:output:0#while/lstm_cell_19/BiasAdd:output:0*
T0*d
_output_shapesR
P:         ╬:         ╬:         ╬:         ╬*
	num_split2
while/lstm_cell_19/splitЩ
while/lstm_cell_19/SigmoidSigmoid!while/lstm_cell_19/split:output:0*
T0*(
_output_shapes
:         ╬2
while/lstm_cell_19/SigmoidЭ
while/lstm_cell_19/Sigmoid_1Sigmoid!while/lstm_cell_19/split:output:1*
T0*(
_output_shapes
:         ╬2
while/lstm_cell_19/Sigmoid_1б
while/lstm_cell_19/mulMul while/lstm_cell_19/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:         ╬2
while/lstm_cell_19/mulР
while/lstm_cell_19/ReluRelu!while/lstm_cell_19/split:output:2*
T0*(
_output_shapes
:         ╬2
while/lstm_cell_19/Relu╡
while/lstm_cell_19/mul_1Mulwhile/lstm_cell_19/Sigmoid:y:0%while/lstm_cell_19/Relu:activations:0*
T0*(
_output_shapes
:         ╬2
while/lstm_cell_19/mul_1к
while/lstm_cell_19/add_1AddV2while/lstm_cell_19/mul:z:0while/lstm_cell_19/mul_1:z:0*
T0*(
_output_shapes
:         ╬2
while/lstm_cell_19/add_1Э
while/lstm_cell_19/Sigmoid_2Sigmoid!while/lstm_cell_19/split:output:3*
T0*(
_output_shapes
:         ╬2
while/lstm_cell_19/Sigmoid_2П
while/lstm_cell_19/Relu_1Reluwhile/lstm_cell_19/add_1:z:0*
T0*(
_output_shapes
:         ╬2
while/lstm_cell_19/Relu_1╣
while/lstm_cell_19/mul_2Mul while/lstm_cell_19/Sigmoid_2:y:0'while/lstm_cell_19/Relu_1:activations:0*
T0*(
_output_shapes
:         ╬2
while/lstm_cell_19/mul_2р
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_19/mul_2:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1k
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity~
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_1m
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_2Ъ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3О
while/Identity_4Identitywhile/lstm_cell_19/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:         ╬2
while/Identity_4О
while/Identity_5Identitywhile/lstm_cell_19/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:         ╬2
while/Identity_5▐

while/NoOpNoOp*^while/lstm_cell_19/BiasAdd/ReadVariableOp)^while/lstm_cell_19/MatMul/ReadVariableOp+^while/lstm_cell_19/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_19_biasadd_readvariableop_resource4while_lstm_cell_19_biasadd_readvariableop_resource_0"l
3while_lstm_cell_19_matmul_1_readvariableop_resource5while_lstm_cell_19_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_19_matmul_readvariableop_resource3while_lstm_cell_19_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :         ╬:         ╬: : : : : 2V
)while/lstm_cell_19/BiasAdd/ReadVariableOp)while/lstm_cell_19/BiasAdd/ReadVariableOp2T
(while/lstm_cell_19/MatMul/ReadVariableOp(while/lstm_cell_19/MatMul/ReadVariableOp2X
*while/lstm_cell_19/MatMul_1/ReadVariableOp*while/lstm_cell_19/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:         ╬:.*
(
_output_shapes
:         ╬:

_output_shapes
: :

_output_shapes
: 
╘
I
-__inference_dropout_18_layer_call_fn_32589963

inputs
identity╩
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         h* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_dropout_18_layer_call_and_return_conditional_losses_325877962
PartitionedCallp
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:         h2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         h:S O
+
_output_shapes
:         h
 
_user_specified_nameinputs
▓
╖
*__inference_lstm_18_layer_call_fn_32589930

inputs
unknown:	]а
	unknown_0:	hа
	unknown_1:	а
identityИвStatefulPartitionedCallЖ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         h*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_lstm_18_layer_call_and_return_conditional_losses_325877832
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:         h2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         ]: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:         ]
 
_user_specified_nameinputs
ч│
╧	
#__inference__wrapped_model_32586366
lstm_18_inputS
@sequential_9_lstm_18_lstm_cell_18_matmul_readvariableop_resource:	]аU
Bsequential_9_lstm_18_lstm_cell_18_matmul_1_readvariableop_resource:	hаP
Asequential_9_lstm_18_lstm_cell_18_biasadd_readvariableop_resource:	аS
@sequential_9_lstm_19_lstm_cell_19_matmul_readvariableop_resource:	h╕
V
Bsequential_9_lstm_19_lstm_cell_19_matmul_1_readvariableop_resource:
╬╕
P
Asequential_9_lstm_19_lstm_cell_19_biasadd_readvariableop_resource:	╕
I
6sequential_9_dense_9_tensordot_readvariableop_resource:	╬B
4sequential_9_dense_9_biasadd_readvariableop_resource:
identityИв+sequential_9/dense_9/BiasAdd/ReadVariableOpв-sequential_9/dense_9/Tensordot/ReadVariableOpв8sequential_9/lstm_18/lstm_cell_18/BiasAdd/ReadVariableOpв7sequential_9/lstm_18/lstm_cell_18/MatMul/ReadVariableOpв9sequential_9/lstm_18/lstm_cell_18/MatMul_1/ReadVariableOpвsequential_9/lstm_18/whileв8sequential_9/lstm_19/lstm_cell_19/BiasAdd/ReadVariableOpв7sequential_9/lstm_19/lstm_cell_19/MatMul/ReadVariableOpв9sequential_9/lstm_19/lstm_cell_19/MatMul_1/ReadVariableOpвsequential_9/lstm_19/whileu
sequential_9/lstm_18/ShapeShapelstm_18_input*
T0*
_output_shapes
:2
sequential_9/lstm_18/ShapeЮ
(sequential_9/lstm_18/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(sequential_9/lstm_18/strided_slice/stackв
*sequential_9/lstm_18/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*sequential_9/lstm_18/strided_slice/stack_1в
*sequential_9/lstm_18/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*sequential_9/lstm_18/strided_slice/stack_2р
"sequential_9/lstm_18/strided_sliceStridedSlice#sequential_9/lstm_18/Shape:output:01sequential_9/lstm_18/strided_slice/stack:output:03sequential_9/lstm_18/strided_slice/stack_1:output:03sequential_9/lstm_18/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"sequential_9/lstm_18/strided_sliceЖ
 sequential_9/lstm_18/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :h2"
 sequential_9/lstm_18/zeros/mul/y└
sequential_9/lstm_18/zeros/mulMul+sequential_9/lstm_18/strided_slice:output:0)sequential_9/lstm_18/zeros/mul/y:output:0*
T0*
_output_shapes
: 2 
sequential_9/lstm_18/zeros/mulЙ
!sequential_9/lstm_18/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2#
!sequential_9/lstm_18/zeros/Less/y╗
sequential_9/lstm_18/zeros/LessLess"sequential_9/lstm_18/zeros/mul:z:0*sequential_9/lstm_18/zeros/Less/y:output:0*
T0*
_output_shapes
: 2!
sequential_9/lstm_18/zeros/LessМ
#sequential_9/lstm_18/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :h2%
#sequential_9/lstm_18/zeros/packed/1╫
!sequential_9/lstm_18/zeros/packedPack+sequential_9/lstm_18/strided_slice:output:0,sequential_9/lstm_18/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2#
!sequential_9/lstm_18/zeros/packedЙ
 sequential_9/lstm_18/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 sequential_9/lstm_18/zeros/Const╔
sequential_9/lstm_18/zerosFill*sequential_9/lstm_18/zeros/packed:output:0)sequential_9/lstm_18/zeros/Const:output:0*
T0*'
_output_shapes
:         h2
sequential_9/lstm_18/zerosК
"sequential_9/lstm_18/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :h2$
"sequential_9/lstm_18/zeros_1/mul/y╞
 sequential_9/lstm_18/zeros_1/mulMul+sequential_9/lstm_18/strided_slice:output:0+sequential_9/lstm_18/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2"
 sequential_9/lstm_18/zeros_1/mulН
#sequential_9/lstm_18/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2%
#sequential_9/lstm_18/zeros_1/Less/y├
!sequential_9/lstm_18/zeros_1/LessLess$sequential_9/lstm_18/zeros_1/mul:z:0,sequential_9/lstm_18/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2#
!sequential_9/lstm_18/zeros_1/LessР
%sequential_9/lstm_18/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :h2'
%sequential_9/lstm_18/zeros_1/packed/1▌
#sequential_9/lstm_18/zeros_1/packedPack+sequential_9/lstm_18/strided_slice:output:0.sequential_9/lstm_18/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2%
#sequential_9/lstm_18/zeros_1/packedН
"sequential_9/lstm_18/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"sequential_9/lstm_18/zeros_1/Const╤
sequential_9/lstm_18/zeros_1Fill,sequential_9/lstm_18/zeros_1/packed:output:0+sequential_9/lstm_18/zeros_1/Const:output:0*
T0*'
_output_shapes
:         h2
sequential_9/lstm_18/zeros_1Я
#sequential_9/lstm_18/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2%
#sequential_9/lstm_18/transpose/perm└
sequential_9/lstm_18/transpose	Transposelstm_18_input,sequential_9/lstm_18/transpose/perm:output:0*
T0*+
_output_shapes
:         ]2 
sequential_9/lstm_18/transposeО
sequential_9/lstm_18/Shape_1Shape"sequential_9/lstm_18/transpose:y:0*
T0*
_output_shapes
:2
sequential_9/lstm_18/Shape_1в
*sequential_9/lstm_18/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2,
*sequential_9/lstm_18/strided_slice_1/stackж
,sequential_9/lstm_18/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_9/lstm_18/strided_slice_1/stack_1ж
,sequential_9/lstm_18/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_9/lstm_18/strided_slice_1/stack_2ь
$sequential_9/lstm_18/strided_slice_1StridedSlice%sequential_9/lstm_18/Shape_1:output:03sequential_9/lstm_18/strided_slice_1/stack:output:05sequential_9/lstm_18/strided_slice_1/stack_1:output:05sequential_9/lstm_18/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2&
$sequential_9/lstm_18/strided_slice_1п
0sequential_9/lstm_18/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         22
0sequential_9/lstm_18/TensorArrayV2/element_shapeЖ
"sequential_9/lstm_18/TensorArrayV2TensorListReserve9sequential_9/lstm_18/TensorArrayV2/element_shape:output:0-sequential_9/lstm_18/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02$
"sequential_9/lstm_18/TensorArrayV2щ
Jsequential_9/lstm_18/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ]   2L
Jsequential_9/lstm_18/TensorArrayUnstack/TensorListFromTensor/element_shape╠
<sequential_9/lstm_18/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor"sequential_9/lstm_18/transpose:y:0Ssequential_9/lstm_18/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02>
<sequential_9/lstm_18/TensorArrayUnstack/TensorListFromTensorв
*sequential_9/lstm_18/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2,
*sequential_9/lstm_18/strided_slice_2/stackж
,sequential_9/lstm_18/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_9/lstm_18/strided_slice_2/stack_1ж
,sequential_9/lstm_18/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_9/lstm_18/strided_slice_2/stack_2·
$sequential_9/lstm_18/strided_slice_2StridedSlice"sequential_9/lstm_18/transpose:y:03sequential_9/lstm_18/strided_slice_2/stack:output:05sequential_9/lstm_18/strided_slice_2/stack_1:output:05sequential_9/lstm_18/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         ]*
shrink_axis_mask2&
$sequential_9/lstm_18/strided_slice_2Ї
7sequential_9/lstm_18/lstm_cell_18/MatMul/ReadVariableOpReadVariableOp@sequential_9_lstm_18_lstm_cell_18_matmul_readvariableop_resource*
_output_shapes
:	]а*
dtype029
7sequential_9/lstm_18/lstm_cell_18/MatMul/ReadVariableOpБ
(sequential_9/lstm_18/lstm_cell_18/MatMulMatMul-sequential_9/lstm_18/strided_slice_2:output:0?sequential_9/lstm_18/lstm_cell_18/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         а2*
(sequential_9/lstm_18/lstm_cell_18/MatMul·
9sequential_9/lstm_18/lstm_cell_18/MatMul_1/ReadVariableOpReadVariableOpBsequential_9_lstm_18_lstm_cell_18_matmul_1_readvariableop_resource*
_output_shapes
:	hа*
dtype02;
9sequential_9/lstm_18/lstm_cell_18/MatMul_1/ReadVariableOp¤
*sequential_9/lstm_18/lstm_cell_18/MatMul_1MatMul#sequential_9/lstm_18/zeros:output:0Asequential_9/lstm_18/lstm_cell_18/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         а2,
*sequential_9/lstm_18/lstm_cell_18/MatMul_1Ї
%sequential_9/lstm_18/lstm_cell_18/addAddV22sequential_9/lstm_18/lstm_cell_18/MatMul:product:04sequential_9/lstm_18/lstm_cell_18/MatMul_1:product:0*
T0*(
_output_shapes
:         а2'
%sequential_9/lstm_18/lstm_cell_18/addє
8sequential_9/lstm_18/lstm_cell_18/BiasAdd/ReadVariableOpReadVariableOpAsequential_9_lstm_18_lstm_cell_18_biasadd_readvariableop_resource*
_output_shapes	
:а*
dtype02:
8sequential_9/lstm_18/lstm_cell_18/BiasAdd/ReadVariableOpБ
)sequential_9/lstm_18/lstm_cell_18/BiasAddBiasAdd)sequential_9/lstm_18/lstm_cell_18/add:z:0@sequential_9/lstm_18/lstm_cell_18/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         а2+
)sequential_9/lstm_18/lstm_cell_18/BiasAddи
1sequential_9/lstm_18/lstm_cell_18/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :23
1sequential_9/lstm_18/lstm_cell_18/split/split_dim╟
'sequential_9/lstm_18/lstm_cell_18/splitSplit:sequential_9/lstm_18/lstm_cell_18/split/split_dim:output:02sequential_9/lstm_18/lstm_cell_18/BiasAdd:output:0*
T0*`
_output_shapesN
L:         h:         h:         h:         h*
	num_split2)
'sequential_9/lstm_18/lstm_cell_18/split┼
)sequential_9/lstm_18/lstm_cell_18/SigmoidSigmoid0sequential_9/lstm_18/lstm_cell_18/split:output:0*
T0*'
_output_shapes
:         h2+
)sequential_9/lstm_18/lstm_cell_18/Sigmoid╔
+sequential_9/lstm_18/lstm_cell_18/Sigmoid_1Sigmoid0sequential_9/lstm_18/lstm_cell_18/split:output:1*
T0*'
_output_shapes
:         h2-
+sequential_9/lstm_18/lstm_cell_18/Sigmoid_1▀
%sequential_9/lstm_18/lstm_cell_18/mulMul/sequential_9/lstm_18/lstm_cell_18/Sigmoid_1:y:0%sequential_9/lstm_18/zeros_1:output:0*
T0*'
_output_shapes
:         h2'
%sequential_9/lstm_18/lstm_cell_18/mul╝
&sequential_9/lstm_18/lstm_cell_18/ReluRelu0sequential_9/lstm_18/lstm_cell_18/split:output:2*
T0*'
_output_shapes
:         h2(
&sequential_9/lstm_18/lstm_cell_18/ReluЁ
'sequential_9/lstm_18/lstm_cell_18/mul_1Mul-sequential_9/lstm_18/lstm_cell_18/Sigmoid:y:04sequential_9/lstm_18/lstm_cell_18/Relu:activations:0*
T0*'
_output_shapes
:         h2)
'sequential_9/lstm_18/lstm_cell_18/mul_1х
'sequential_9/lstm_18/lstm_cell_18/add_1AddV2)sequential_9/lstm_18/lstm_cell_18/mul:z:0+sequential_9/lstm_18/lstm_cell_18/mul_1:z:0*
T0*'
_output_shapes
:         h2)
'sequential_9/lstm_18/lstm_cell_18/add_1╔
+sequential_9/lstm_18/lstm_cell_18/Sigmoid_2Sigmoid0sequential_9/lstm_18/lstm_cell_18/split:output:3*
T0*'
_output_shapes
:         h2-
+sequential_9/lstm_18/lstm_cell_18/Sigmoid_2╗
(sequential_9/lstm_18/lstm_cell_18/Relu_1Relu+sequential_9/lstm_18/lstm_cell_18/add_1:z:0*
T0*'
_output_shapes
:         h2*
(sequential_9/lstm_18/lstm_cell_18/Relu_1Ї
'sequential_9/lstm_18/lstm_cell_18/mul_2Mul/sequential_9/lstm_18/lstm_cell_18/Sigmoid_2:y:06sequential_9/lstm_18/lstm_cell_18/Relu_1:activations:0*
T0*'
_output_shapes
:         h2)
'sequential_9/lstm_18/lstm_cell_18/mul_2╣
2sequential_9/lstm_18/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    h   24
2sequential_9/lstm_18/TensorArrayV2_1/element_shapeМ
$sequential_9/lstm_18/TensorArrayV2_1TensorListReserve;sequential_9/lstm_18/TensorArrayV2_1/element_shape:output:0-sequential_9/lstm_18/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02&
$sequential_9/lstm_18/TensorArrayV2_1x
sequential_9/lstm_18/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
sequential_9/lstm_18/timeй
-sequential_9/lstm_18/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         2/
-sequential_9/lstm_18/while/maximum_iterationsФ
'sequential_9/lstm_18/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2)
'sequential_9/lstm_18/while/loop_counter╩
sequential_9/lstm_18/whileWhile0sequential_9/lstm_18/while/loop_counter:output:06sequential_9/lstm_18/while/maximum_iterations:output:0"sequential_9/lstm_18/time:output:0-sequential_9/lstm_18/TensorArrayV2_1:handle:0#sequential_9/lstm_18/zeros:output:0%sequential_9/lstm_18/zeros_1:output:0-sequential_9/lstm_18/strided_slice_1:output:0Lsequential_9/lstm_18/TensorArrayUnstack/TensorListFromTensor:output_handle:0@sequential_9_lstm_18_lstm_cell_18_matmul_readvariableop_resourceBsequential_9_lstm_18_lstm_cell_18_matmul_1_readvariableop_resourceAsequential_9_lstm_18_lstm_cell_18_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         h:         h: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *4
body,R*
(sequential_9_lstm_18_while_body_32586106*4
cond,R*
(sequential_9_lstm_18_while_cond_32586105*K
output_shapes:
8: : : : :         h:         h: : : : : *
parallel_iterations 2
sequential_9/lstm_18/while▀
Esequential_9/lstm_18/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    h   2G
Esequential_9/lstm_18/TensorArrayV2Stack/TensorListStack/element_shape╝
7sequential_9/lstm_18/TensorArrayV2Stack/TensorListStackTensorListStack#sequential_9/lstm_18/while:output:3Nsequential_9/lstm_18/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:         h*
element_dtype029
7sequential_9/lstm_18/TensorArrayV2Stack/TensorListStackл
*sequential_9/lstm_18/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         2,
*sequential_9/lstm_18/strided_slice_3/stackж
,sequential_9/lstm_18/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2.
,sequential_9/lstm_18/strided_slice_3/stack_1ж
,sequential_9/lstm_18/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_9/lstm_18/strided_slice_3/stack_2Ш
$sequential_9/lstm_18/strided_slice_3StridedSlice@sequential_9/lstm_18/TensorArrayV2Stack/TensorListStack:tensor:03sequential_9/lstm_18/strided_slice_3/stack:output:05sequential_9/lstm_18/strided_slice_3/stack_1:output:05sequential_9/lstm_18/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         h*
shrink_axis_mask2&
$sequential_9/lstm_18/strided_slice_3г
%sequential_9/lstm_18/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2'
%sequential_9/lstm_18/transpose_1/perm∙
 sequential_9/lstm_18/transpose_1	Transpose@sequential_9/lstm_18/TensorArrayV2Stack/TensorListStack:tensor:0.sequential_9/lstm_18/transpose_1/perm:output:0*
T0*+
_output_shapes
:         h2"
 sequential_9/lstm_18/transpose_1Р
sequential_9/lstm_18/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
sequential_9/lstm_18/runtimeм
 sequential_9/dropout_18/IdentityIdentity$sequential_9/lstm_18/transpose_1:y:0*
T0*+
_output_shapes
:         h2"
 sequential_9/dropout_18/IdentityС
sequential_9/lstm_19/ShapeShape)sequential_9/dropout_18/Identity:output:0*
T0*
_output_shapes
:2
sequential_9/lstm_19/ShapeЮ
(sequential_9/lstm_19/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(sequential_9/lstm_19/strided_slice/stackв
*sequential_9/lstm_19/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*sequential_9/lstm_19/strided_slice/stack_1в
*sequential_9/lstm_19/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*sequential_9/lstm_19/strided_slice/stack_2р
"sequential_9/lstm_19/strided_sliceStridedSlice#sequential_9/lstm_19/Shape:output:01sequential_9/lstm_19/strided_slice/stack:output:03sequential_9/lstm_19/strided_slice/stack_1:output:03sequential_9/lstm_19/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"sequential_9/lstm_19/strided_sliceЗ
 sequential_9/lstm_19/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :╬2"
 sequential_9/lstm_19/zeros/mul/y└
sequential_9/lstm_19/zeros/mulMul+sequential_9/lstm_19/strided_slice:output:0)sequential_9/lstm_19/zeros/mul/y:output:0*
T0*
_output_shapes
: 2 
sequential_9/lstm_19/zeros/mulЙ
!sequential_9/lstm_19/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2#
!sequential_9/lstm_19/zeros/Less/y╗
sequential_9/lstm_19/zeros/LessLess"sequential_9/lstm_19/zeros/mul:z:0*sequential_9/lstm_19/zeros/Less/y:output:0*
T0*
_output_shapes
: 2!
sequential_9/lstm_19/zeros/LessН
#sequential_9/lstm_19/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :╬2%
#sequential_9/lstm_19/zeros/packed/1╫
!sequential_9/lstm_19/zeros/packedPack+sequential_9/lstm_19/strided_slice:output:0,sequential_9/lstm_19/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2#
!sequential_9/lstm_19/zeros/packedЙ
 sequential_9/lstm_19/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 sequential_9/lstm_19/zeros/Const╩
sequential_9/lstm_19/zerosFill*sequential_9/lstm_19/zeros/packed:output:0)sequential_9/lstm_19/zeros/Const:output:0*
T0*(
_output_shapes
:         ╬2
sequential_9/lstm_19/zerosЛ
"sequential_9/lstm_19/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :╬2$
"sequential_9/lstm_19/zeros_1/mul/y╞
 sequential_9/lstm_19/zeros_1/mulMul+sequential_9/lstm_19/strided_slice:output:0+sequential_9/lstm_19/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2"
 sequential_9/lstm_19/zeros_1/mulН
#sequential_9/lstm_19/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2%
#sequential_9/lstm_19/zeros_1/Less/y├
!sequential_9/lstm_19/zeros_1/LessLess$sequential_9/lstm_19/zeros_1/mul:z:0,sequential_9/lstm_19/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2#
!sequential_9/lstm_19/zeros_1/LessС
%sequential_9/lstm_19/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :╬2'
%sequential_9/lstm_19/zeros_1/packed/1▌
#sequential_9/lstm_19/zeros_1/packedPack+sequential_9/lstm_19/strided_slice:output:0.sequential_9/lstm_19/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2%
#sequential_9/lstm_19/zeros_1/packedН
"sequential_9/lstm_19/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"sequential_9/lstm_19/zeros_1/Const╥
sequential_9/lstm_19/zeros_1Fill,sequential_9/lstm_19/zeros_1/packed:output:0+sequential_9/lstm_19/zeros_1/Const:output:0*
T0*(
_output_shapes
:         ╬2
sequential_9/lstm_19/zeros_1Я
#sequential_9/lstm_19/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2%
#sequential_9/lstm_19/transpose/perm▄
sequential_9/lstm_19/transpose	Transpose)sequential_9/dropout_18/Identity:output:0,sequential_9/lstm_19/transpose/perm:output:0*
T0*+
_output_shapes
:         h2 
sequential_9/lstm_19/transposeО
sequential_9/lstm_19/Shape_1Shape"sequential_9/lstm_19/transpose:y:0*
T0*
_output_shapes
:2
sequential_9/lstm_19/Shape_1в
*sequential_9/lstm_19/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2,
*sequential_9/lstm_19/strided_slice_1/stackж
,sequential_9/lstm_19/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_9/lstm_19/strided_slice_1/stack_1ж
,sequential_9/lstm_19/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_9/lstm_19/strided_slice_1/stack_2ь
$sequential_9/lstm_19/strided_slice_1StridedSlice%sequential_9/lstm_19/Shape_1:output:03sequential_9/lstm_19/strided_slice_1/stack:output:05sequential_9/lstm_19/strided_slice_1/stack_1:output:05sequential_9/lstm_19/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2&
$sequential_9/lstm_19/strided_slice_1п
0sequential_9/lstm_19/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         22
0sequential_9/lstm_19/TensorArrayV2/element_shapeЖ
"sequential_9/lstm_19/TensorArrayV2TensorListReserve9sequential_9/lstm_19/TensorArrayV2/element_shape:output:0-sequential_9/lstm_19/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02$
"sequential_9/lstm_19/TensorArrayV2щ
Jsequential_9/lstm_19/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    h   2L
Jsequential_9/lstm_19/TensorArrayUnstack/TensorListFromTensor/element_shape╠
<sequential_9/lstm_19/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor"sequential_9/lstm_19/transpose:y:0Ssequential_9/lstm_19/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02>
<sequential_9/lstm_19/TensorArrayUnstack/TensorListFromTensorв
*sequential_9/lstm_19/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2,
*sequential_9/lstm_19/strided_slice_2/stackж
,sequential_9/lstm_19/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_9/lstm_19/strided_slice_2/stack_1ж
,sequential_9/lstm_19/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_9/lstm_19/strided_slice_2/stack_2·
$sequential_9/lstm_19/strided_slice_2StridedSlice"sequential_9/lstm_19/transpose:y:03sequential_9/lstm_19/strided_slice_2/stack:output:05sequential_9/lstm_19/strided_slice_2/stack_1:output:05sequential_9/lstm_19/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         h*
shrink_axis_mask2&
$sequential_9/lstm_19/strided_slice_2Ї
7sequential_9/lstm_19/lstm_cell_19/MatMul/ReadVariableOpReadVariableOp@sequential_9_lstm_19_lstm_cell_19_matmul_readvariableop_resource*
_output_shapes
:	h╕
*
dtype029
7sequential_9/lstm_19/lstm_cell_19/MatMul/ReadVariableOpБ
(sequential_9/lstm_19/lstm_cell_19/MatMulMatMul-sequential_9/lstm_19/strided_slice_2:output:0?sequential_9/lstm_19/lstm_cell_19/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╕
2*
(sequential_9/lstm_19/lstm_cell_19/MatMul√
9sequential_9/lstm_19/lstm_cell_19/MatMul_1/ReadVariableOpReadVariableOpBsequential_9_lstm_19_lstm_cell_19_matmul_1_readvariableop_resource* 
_output_shapes
:
╬╕
*
dtype02;
9sequential_9/lstm_19/lstm_cell_19/MatMul_1/ReadVariableOp¤
*sequential_9/lstm_19/lstm_cell_19/MatMul_1MatMul#sequential_9/lstm_19/zeros:output:0Asequential_9/lstm_19/lstm_cell_19/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╕
2,
*sequential_9/lstm_19/lstm_cell_19/MatMul_1Ї
%sequential_9/lstm_19/lstm_cell_19/addAddV22sequential_9/lstm_19/lstm_cell_19/MatMul:product:04sequential_9/lstm_19/lstm_cell_19/MatMul_1:product:0*
T0*(
_output_shapes
:         ╕
2'
%sequential_9/lstm_19/lstm_cell_19/addє
8sequential_9/lstm_19/lstm_cell_19/BiasAdd/ReadVariableOpReadVariableOpAsequential_9_lstm_19_lstm_cell_19_biasadd_readvariableop_resource*
_output_shapes	
:╕
*
dtype02:
8sequential_9/lstm_19/lstm_cell_19/BiasAdd/ReadVariableOpБ
)sequential_9/lstm_19/lstm_cell_19/BiasAddBiasAdd)sequential_9/lstm_19/lstm_cell_19/add:z:0@sequential_9/lstm_19/lstm_cell_19/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╕
2+
)sequential_9/lstm_19/lstm_cell_19/BiasAddи
1sequential_9/lstm_19/lstm_cell_19/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :23
1sequential_9/lstm_19/lstm_cell_19/split/split_dim╦
'sequential_9/lstm_19/lstm_cell_19/splitSplit:sequential_9/lstm_19/lstm_cell_19/split/split_dim:output:02sequential_9/lstm_19/lstm_cell_19/BiasAdd:output:0*
T0*d
_output_shapesR
P:         ╬:         ╬:         ╬:         ╬*
	num_split2)
'sequential_9/lstm_19/lstm_cell_19/split╞
)sequential_9/lstm_19/lstm_cell_19/SigmoidSigmoid0sequential_9/lstm_19/lstm_cell_19/split:output:0*
T0*(
_output_shapes
:         ╬2+
)sequential_9/lstm_19/lstm_cell_19/Sigmoid╩
+sequential_9/lstm_19/lstm_cell_19/Sigmoid_1Sigmoid0sequential_9/lstm_19/lstm_cell_19/split:output:1*
T0*(
_output_shapes
:         ╬2-
+sequential_9/lstm_19/lstm_cell_19/Sigmoid_1р
%sequential_9/lstm_19/lstm_cell_19/mulMul/sequential_9/lstm_19/lstm_cell_19/Sigmoid_1:y:0%sequential_9/lstm_19/zeros_1:output:0*
T0*(
_output_shapes
:         ╬2'
%sequential_9/lstm_19/lstm_cell_19/mul╜
&sequential_9/lstm_19/lstm_cell_19/ReluRelu0sequential_9/lstm_19/lstm_cell_19/split:output:2*
T0*(
_output_shapes
:         ╬2(
&sequential_9/lstm_19/lstm_cell_19/Reluё
'sequential_9/lstm_19/lstm_cell_19/mul_1Mul-sequential_9/lstm_19/lstm_cell_19/Sigmoid:y:04sequential_9/lstm_19/lstm_cell_19/Relu:activations:0*
T0*(
_output_shapes
:         ╬2)
'sequential_9/lstm_19/lstm_cell_19/mul_1ц
'sequential_9/lstm_19/lstm_cell_19/add_1AddV2)sequential_9/lstm_19/lstm_cell_19/mul:z:0+sequential_9/lstm_19/lstm_cell_19/mul_1:z:0*
T0*(
_output_shapes
:         ╬2)
'sequential_9/lstm_19/lstm_cell_19/add_1╩
+sequential_9/lstm_19/lstm_cell_19/Sigmoid_2Sigmoid0sequential_9/lstm_19/lstm_cell_19/split:output:3*
T0*(
_output_shapes
:         ╬2-
+sequential_9/lstm_19/lstm_cell_19/Sigmoid_2╝
(sequential_9/lstm_19/lstm_cell_19/Relu_1Relu+sequential_9/lstm_19/lstm_cell_19/add_1:z:0*
T0*(
_output_shapes
:         ╬2*
(sequential_9/lstm_19/lstm_cell_19/Relu_1ї
'sequential_9/lstm_19/lstm_cell_19/mul_2Mul/sequential_9/lstm_19/lstm_cell_19/Sigmoid_2:y:06sequential_9/lstm_19/lstm_cell_19/Relu_1:activations:0*
T0*(
_output_shapes
:         ╬2)
'sequential_9/lstm_19/lstm_cell_19/mul_2╣
2sequential_9/lstm_19/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    N  24
2sequential_9/lstm_19/TensorArrayV2_1/element_shapeМ
$sequential_9/lstm_19/TensorArrayV2_1TensorListReserve;sequential_9/lstm_19/TensorArrayV2_1/element_shape:output:0-sequential_9/lstm_19/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02&
$sequential_9/lstm_19/TensorArrayV2_1x
sequential_9/lstm_19/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
sequential_9/lstm_19/timeй
-sequential_9/lstm_19/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         2/
-sequential_9/lstm_19/while/maximum_iterationsФ
'sequential_9/lstm_19/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2)
'sequential_9/lstm_19/while/loop_counter╬
sequential_9/lstm_19/whileWhile0sequential_9/lstm_19/while/loop_counter:output:06sequential_9/lstm_19/while/maximum_iterations:output:0"sequential_9/lstm_19/time:output:0-sequential_9/lstm_19/TensorArrayV2_1:handle:0#sequential_9/lstm_19/zeros:output:0%sequential_9/lstm_19/zeros_1:output:0-sequential_9/lstm_19/strided_slice_1:output:0Lsequential_9/lstm_19/TensorArrayUnstack/TensorListFromTensor:output_handle:0@sequential_9_lstm_19_lstm_cell_19_matmul_readvariableop_resourceBsequential_9_lstm_19_lstm_cell_19_matmul_1_readvariableop_resourceAsequential_9_lstm_19_lstm_cell_19_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :         ╬:         ╬: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *4
body,R*
(sequential_9_lstm_19_while_body_32586254*4
cond,R*
(sequential_9_lstm_19_while_cond_32586253*M
output_shapes<
:: : : : :         ╬:         ╬: : : : : *
parallel_iterations 2
sequential_9/lstm_19/while▀
Esequential_9/lstm_19/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    N  2G
Esequential_9/lstm_19/TensorArrayV2Stack/TensorListStack/element_shape╜
7sequential_9/lstm_19/TensorArrayV2Stack/TensorListStackTensorListStack#sequential_9/lstm_19/while:output:3Nsequential_9/lstm_19/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:         ╬*
element_dtype029
7sequential_9/lstm_19/TensorArrayV2Stack/TensorListStackл
*sequential_9/lstm_19/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         2,
*sequential_9/lstm_19/strided_slice_3/stackж
,sequential_9/lstm_19/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2.
,sequential_9/lstm_19/strided_slice_3/stack_1ж
,sequential_9/lstm_19/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_9/lstm_19/strided_slice_3/stack_2Щ
$sequential_9/lstm_19/strided_slice_3StridedSlice@sequential_9/lstm_19/TensorArrayV2Stack/TensorListStack:tensor:03sequential_9/lstm_19/strided_slice_3/stack:output:05sequential_9/lstm_19/strided_slice_3/stack_1:output:05sequential_9/lstm_19/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         ╬*
shrink_axis_mask2&
$sequential_9/lstm_19/strided_slice_3г
%sequential_9/lstm_19/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2'
%sequential_9/lstm_19/transpose_1/perm·
 sequential_9/lstm_19/transpose_1	Transpose@sequential_9/lstm_19/TensorArrayV2Stack/TensorListStack:tensor:0.sequential_9/lstm_19/transpose_1/perm:output:0*
T0*,
_output_shapes
:         ╬2"
 sequential_9/lstm_19/transpose_1Р
sequential_9/lstm_19/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
sequential_9/lstm_19/runtimeн
 sequential_9/dropout_19/IdentityIdentity$sequential_9/lstm_19/transpose_1:y:0*
T0*,
_output_shapes
:         ╬2"
 sequential_9/dropout_19/Identity╓
-sequential_9/dense_9/Tensordot/ReadVariableOpReadVariableOp6sequential_9_dense_9_tensordot_readvariableop_resource*
_output_shapes
:	╬*
dtype02/
-sequential_9/dense_9/Tensordot/ReadVariableOpФ
#sequential_9/dense_9/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2%
#sequential_9/dense_9/Tensordot/axesЫ
#sequential_9/dense_9/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2%
#sequential_9/dense_9/Tensordot/freeе
$sequential_9/dense_9/Tensordot/ShapeShape)sequential_9/dropout_19/Identity:output:0*
T0*
_output_shapes
:2&
$sequential_9/dense_9/Tensordot/ShapeЮ
,sequential_9/dense_9/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,sequential_9/dense_9/Tensordot/GatherV2/axis║
'sequential_9/dense_9/Tensordot/GatherV2GatherV2-sequential_9/dense_9/Tensordot/Shape:output:0,sequential_9/dense_9/Tensordot/free:output:05sequential_9/dense_9/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2)
'sequential_9/dense_9/Tensordot/GatherV2в
.sequential_9/dense_9/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.sequential_9/dense_9/Tensordot/GatherV2_1/axis└
)sequential_9/dense_9/Tensordot/GatherV2_1GatherV2-sequential_9/dense_9/Tensordot/Shape:output:0,sequential_9/dense_9/Tensordot/axes:output:07sequential_9/dense_9/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2+
)sequential_9/dense_9/Tensordot/GatherV2_1Ц
$sequential_9/dense_9/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2&
$sequential_9/dense_9/Tensordot/Const╘
#sequential_9/dense_9/Tensordot/ProdProd0sequential_9/dense_9/Tensordot/GatherV2:output:0-sequential_9/dense_9/Tensordot/Const:output:0*
T0*
_output_shapes
: 2%
#sequential_9/dense_9/Tensordot/ProdЪ
&sequential_9/dense_9/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2(
&sequential_9/dense_9/Tensordot/Const_1▄
%sequential_9/dense_9/Tensordot/Prod_1Prod2sequential_9/dense_9/Tensordot/GatherV2_1:output:0/sequential_9/dense_9/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2'
%sequential_9/dense_9/Tensordot/Prod_1Ъ
*sequential_9/dense_9/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2,
*sequential_9/dense_9/Tensordot/concat/axisЩ
%sequential_9/dense_9/Tensordot/concatConcatV2,sequential_9/dense_9/Tensordot/free:output:0,sequential_9/dense_9/Tensordot/axes:output:03sequential_9/dense_9/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2'
%sequential_9/dense_9/Tensordot/concatр
$sequential_9/dense_9/Tensordot/stackPack,sequential_9/dense_9/Tensordot/Prod:output:0.sequential_9/dense_9/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2&
$sequential_9/dense_9/Tensordot/stackє
(sequential_9/dense_9/Tensordot/transpose	Transpose)sequential_9/dropout_19/Identity:output:0.sequential_9/dense_9/Tensordot/concat:output:0*
T0*,
_output_shapes
:         ╬2*
(sequential_9/dense_9/Tensordot/transposeє
&sequential_9/dense_9/Tensordot/ReshapeReshape,sequential_9/dense_9/Tensordot/transpose:y:0-sequential_9/dense_9/Tensordot/stack:output:0*
T0*0
_output_shapes
:                  2(
&sequential_9/dense_9/Tensordot/ReshapeЄ
%sequential_9/dense_9/Tensordot/MatMulMatMul/sequential_9/dense_9/Tensordot/Reshape:output:05sequential_9/dense_9/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2'
%sequential_9/dense_9/Tensordot/MatMulЪ
&sequential_9/dense_9/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&sequential_9/dense_9/Tensordot/Const_2Ю
,sequential_9/dense_9/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,sequential_9/dense_9/Tensordot/concat_1/axisж
'sequential_9/dense_9/Tensordot/concat_1ConcatV20sequential_9/dense_9/Tensordot/GatherV2:output:0/sequential_9/dense_9/Tensordot/Const_2:output:05sequential_9/dense_9/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2)
'sequential_9/dense_9/Tensordot/concat_1ф
sequential_9/dense_9/TensordotReshape/sequential_9/dense_9/Tensordot/MatMul:product:00sequential_9/dense_9/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:         2 
sequential_9/dense_9/Tensordot╦
+sequential_9/dense_9/BiasAdd/ReadVariableOpReadVariableOp4sequential_9_dense_9_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02-
+sequential_9/dense_9/BiasAdd/ReadVariableOp█
sequential_9/dense_9/BiasAddBiasAdd'sequential_9/dense_9/Tensordot:output:03sequential_9/dense_9/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:         2
sequential_9/dense_9/BiasAddд
sequential_9/dense_9/SoftmaxSoftmax%sequential_9/dense_9/BiasAdd:output:0*
T0*+
_output_shapes
:         2
sequential_9/dense_9/SoftmaxЕ
IdentityIdentity&sequential_9/dense_9/Softmax:softmax:0^NoOp*
T0*+
_output_shapes
:         2

Identity╚
NoOpNoOp,^sequential_9/dense_9/BiasAdd/ReadVariableOp.^sequential_9/dense_9/Tensordot/ReadVariableOp9^sequential_9/lstm_18/lstm_cell_18/BiasAdd/ReadVariableOp8^sequential_9/lstm_18/lstm_cell_18/MatMul/ReadVariableOp:^sequential_9/lstm_18/lstm_cell_18/MatMul_1/ReadVariableOp^sequential_9/lstm_18/while9^sequential_9/lstm_19/lstm_cell_19/BiasAdd/ReadVariableOp8^sequential_9/lstm_19/lstm_cell_19/MatMul/ReadVariableOp:^sequential_9/lstm_19/lstm_cell_19/MatMul_1/ReadVariableOp^sequential_9/lstm_19/while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':         ]: : : : : : : : 2Z
+sequential_9/dense_9/BiasAdd/ReadVariableOp+sequential_9/dense_9/BiasAdd/ReadVariableOp2^
-sequential_9/dense_9/Tensordot/ReadVariableOp-sequential_9/dense_9/Tensordot/ReadVariableOp2t
8sequential_9/lstm_18/lstm_cell_18/BiasAdd/ReadVariableOp8sequential_9/lstm_18/lstm_cell_18/BiasAdd/ReadVariableOp2r
7sequential_9/lstm_18/lstm_cell_18/MatMul/ReadVariableOp7sequential_9/lstm_18/lstm_cell_18/MatMul/ReadVariableOp2v
9sequential_9/lstm_18/lstm_cell_18/MatMul_1/ReadVariableOp9sequential_9/lstm_18/lstm_cell_18/MatMul_1/ReadVariableOp28
sequential_9/lstm_18/whilesequential_9/lstm_18/while2t
8sequential_9/lstm_19/lstm_cell_19/BiasAdd/ReadVariableOp8sequential_9/lstm_19/lstm_cell_19/BiasAdd/ReadVariableOp2r
7sequential_9/lstm_19/lstm_cell_19/MatMul/ReadVariableOp7sequential_9/lstm_19/lstm_cell_19/MatMul/ReadVariableOp2v
9sequential_9/lstm_19/lstm_cell_19/MatMul_1/ReadVariableOp9sequential_9/lstm_19/lstm_cell_19/MatMul_1/ReadVariableOp28
sequential_9/lstm_19/whilesequential_9/lstm_19/while:Z V
+
_output_shapes
:         ]
'
_user_specified_namelstm_18_input
Е
f
H__inference_dropout_18_layer_call_and_return_conditional_losses_32587796

inputs

identity_1^
IdentityIdentityinputs*
T0*+
_output_shapes
:         h2

Identitym

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:         h2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         h:S O
+
_output_shapes
:         h
 
_user_specified_nameinputs
╦F
О
E__inference_lstm_19_layer_call_and_return_conditional_losses_32587154

inputs(
lstm_cell_19_32587072:	h╕
)
lstm_cell_19_32587074:
╬╕
$
lstm_cell_19_32587076:	╕

identityИв$lstm_cell_19/StatefulPartitionedCallвwhileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice]
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :╬2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessc
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :╬2
zeros/packed/1Г
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:         ╬2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :╬2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :╬2
zeros_1/packed/1Й
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const~
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:         ╬2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permГ
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :                  h2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ю
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1Е
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         2
TensorArrayV2/element_shape▓
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2┐
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    h   27
5TensorArrayUnstack/TensorListFromTensor/element_shape°
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2№
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         h*
shrink_axis_mask2
strided_slice_2и
$lstm_cell_19/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_19_32587072lstm_cell_19_32587074lstm_cell_19_32587076*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:         ╬:         ╬:         ╬*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_lstm_cell_19_layer_call_and_return_conditional_losses_325870712&
$lstm_cell_19/StatefulPartitionedCallП
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    N  2
TensorArrayV2_1/element_shape╕
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter╬
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_19_32587072lstm_cell_19_32587074lstm_cell_19_32587076*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :         ╬:         ╬: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_32587085*
condR
while_cond_32587084*M
output_shapes<
:: : : : :         ╬:         ╬: : : : : *
parallel_iterations 2
while╡
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    N  22
0TensorArrayV2Stack/TensorListStack/element_shapeЄ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:                  ╬*
element_dtype02$
"TensorArrayV2Stack/TensorListStackБ
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2Ы
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         ╬*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permп
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:                  ╬2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimex
IdentityIdentitytranspose_1:y:0^NoOp*
T0*5
_output_shapes#
!:                  ╬2

Identity}
NoOpNoOp%^lstm_cell_19/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  h: : : 2L
$lstm_cell_19/StatefulPartitionedCall$lstm_cell_19/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :                  h
 
_user_specified_nameinputs
├\
а
E__inference_lstm_19_layer_call_and_return_conditional_losses_32590270
inputs_0>
+lstm_cell_19_matmul_readvariableop_resource:	h╕
A
-lstm_cell_19_matmul_1_readvariableop_resource:
╬╕
;
,lstm_cell_19_biasadd_readvariableop_resource:	╕

identityИв#lstm_cell_19/BiasAdd/ReadVariableOpв"lstm_cell_19/MatMul/ReadVariableOpв$lstm_cell_19/MatMul_1/ReadVariableOpвwhileF
ShapeShapeinputs_0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice]
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :╬2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessc
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :╬2
zeros/packed/1Г
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:         ╬2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :╬2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :╬2
zeros_1/packed/1Й
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const~
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:         ╬2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permЕ
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :                  h2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ю
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1Е
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         2
TensorArrayV2/element_shape▓
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2┐
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    h   27
5TensorArrayUnstack/TensorListFromTensor/element_shape°
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2№
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         h*
shrink_axis_mask2
strided_slice_2╡
"lstm_cell_19/MatMul/ReadVariableOpReadVariableOp+lstm_cell_19_matmul_readvariableop_resource*
_output_shapes
:	h╕
*
dtype02$
"lstm_cell_19/MatMul/ReadVariableOpн
lstm_cell_19/MatMulMatMulstrided_slice_2:output:0*lstm_cell_19/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╕
2
lstm_cell_19/MatMul╝
$lstm_cell_19/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_19_matmul_1_readvariableop_resource* 
_output_shapes
:
╬╕
*
dtype02&
$lstm_cell_19/MatMul_1/ReadVariableOpй
lstm_cell_19/MatMul_1MatMulzeros:output:0,lstm_cell_19/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╕
2
lstm_cell_19/MatMul_1а
lstm_cell_19/addAddV2lstm_cell_19/MatMul:product:0lstm_cell_19/MatMul_1:product:0*
T0*(
_output_shapes
:         ╕
2
lstm_cell_19/add┤
#lstm_cell_19/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_19_biasadd_readvariableop_resource*
_output_shapes	
:╕
*
dtype02%
#lstm_cell_19/BiasAdd/ReadVariableOpн
lstm_cell_19/BiasAddBiasAddlstm_cell_19/add:z:0+lstm_cell_19/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╕
2
lstm_cell_19/BiasAdd~
lstm_cell_19/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_19/split/split_dimў
lstm_cell_19/splitSplit%lstm_cell_19/split/split_dim:output:0lstm_cell_19/BiasAdd:output:0*
T0*d
_output_shapesR
P:         ╬:         ╬:         ╬:         ╬*
	num_split2
lstm_cell_19/splitЗ
lstm_cell_19/SigmoidSigmoidlstm_cell_19/split:output:0*
T0*(
_output_shapes
:         ╬2
lstm_cell_19/SigmoidЛ
lstm_cell_19/Sigmoid_1Sigmoidlstm_cell_19/split:output:1*
T0*(
_output_shapes
:         ╬2
lstm_cell_19/Sigmoid_1М
lstm_cell_19/mulMullstm_cell_19/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:         ╬2
lstm_cell_19/mul~
lstm_cell_19/ReluRelulstm_cell_19/split:output:2*
T0*(
_output_shapes
:         ╬2
lstm_cell_19/ReluЭ
lstm_cell_19/mul_1Mullstm_cell_19/Sigmoid:y:0lstm_cell_19/Relu:activations:0*
T0*(
_output_shapes
:         ╬2
lstm_cell_19/mul_1Т
lstm_cell_19/add_1AddV2lstm_cell_19/mul:z:0lstm_cell_19/mul_1:z:0*
T0*(
_output_shapes
:         ╬2
lstm_cell_19/add_1Л
lstm_cell_19/Sigmoid_2Sigmoidlstm_cell_19/split:output:3*
T0*(
_output_shapes
:         ╬2
lstm_cell_19/Sigmoid_2}
lstm_cell_19/Relu_1Relulstm_cell_19/add_1:z:0*
T0*(
_output_shapes
:         ╬2
lstm_cell_19/Relu_1б
lstm_cell_19/mul_2Mullstm_cell_19/Sigmoid_2:y:0!lstm_cell_19/Relu_1:activations:0*
T0*(
_output_shapes
:         ╬2
lstm_cell_19/mul_2П
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    N  2
TensorArrayV2_1/element_shape╕
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterУ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_19_matmul_readvariableop_resource-lstm_cell_19_matmul_1_readvariableop_resource,lstm_cell_19_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :         ╬:         ╬: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_32590186*
condR
while_cond_32590185*M
output_shapes<
:: : : : :         ╬:         ╬: : : : : *
parallel_iterations 2
while╡
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    N  22
0TensorArrayV2Stack/TensorListStack/element_shapeЄ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:                  ╬*
element_dtype02$
"TensorArrayV2Stack/TensorListStackБ
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2Ы
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         ╬*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permп
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:                  ╬2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimex
IdentityIdentitytranspose_1:y:0^NoOp*
T0*5
_output_shapes#
!:                  ╬2

Identity╚
NoOpNoOp$^lstm_cell_19/BiasAdd/ReadVariableOp#^lstm_cell_19/MatMul/ReadVariableOp%^lstm_cell_19/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  h: : : 2J
#lstm_cell_19/BiasAdd/ReadVariableOp#lstm_cell_19/BiasAdd/ReadVariableOp2H
"lstm_cell_19/MatMul/ReadVariableOp"lstm_cell_19/MatMul/ReadVariableOp2L
$lstm_cell_19/MatMul_1/ReadVariableOp$lstm_cell_19/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :                  h
"
_user_specified_name
inputs/0
╘

э
lstm_19_while_cond_32588797,
(lstm_19_while_lstm_19_while_loop_counter2
.lstm_19_while_lstm_19_while_maximum_iterations
lstm_19_while_placeholder
lstm_19_while_placeholder_1
lstm_19_while_placeholder_2
lstm_19_while_placeholder_3.
*lstm_19_while_less_lstm_19_strided_slice_1F
Blstm_19_while_lstm_19_while_cond_32588797___redundant_placeholder0F
Blstm_19_while_lstm_19_while_cond_32588797___redundant_placeholder1F
Blstm_19_while_lstm_19_while_cond_32588797___redundant_placeholder2F
Blstm_19_while_lstm_19_while_cond_32588797___redundant_placeholder3
lstm_19_while_identity
Ш
lstm_19/while/LessLesslstm_19_while_placeholder*lstm_19_while_less_lstm_19_strided_slice_1*
T0*
_output_shapes
: 2
lstm_19/while/Lessu
lstm_19/while/IdentityIdentitylstm_19/while/Less:z:0*
T0
*
_output_shapes
: 2
lstm_19/while/Identity"9
lstm_19_while_identitylstm_19/while/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :         ╬:         ╬: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:         ╬:.*
(
_output_shapes
:         ╬:

_output_shapes
: :

_output_shapes
:
Ъ?
╥
while_body_32589360
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_18_matmul_readvariableop_resource_0:	]аH
5while_lstm_cell_18_matmul_1_readvariableop_resource_0:	hаC
4while_lstm_cell_18_biasadd_readvariableop_resource_0:	а
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_18_matmul_readvariableop_resource:	]аF
3while_lstm_cell_18_matmul_1_readvariableop_resource:	hаA
2while_lstm_cell_18_biasadd_readvariableop_resource:	аИв)while/lstm_cell_18/BiasAdd/ReadVariableOpв(while/lstm_cell_18/MatMul/ReadVariableOpв*while/lstm_cell_18/MatMul_1/ReadVariableOp├
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ]   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape╙
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         ]*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem╔
(while/lstm_cell_18/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_18_matmul_readvariableop_resource_0*
_output_shapes
:	]а*
dtype02*
(while/lstm_cell_18/MatMul/ReadVariableOp╫
while/lstm_cell_18/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_18/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         а2
while/lstm_cell_18/MatMul╧
*while/lstm_cell_18/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_18_matmul_1_readvariableop_resource_0*
_output_shapes
:	hа*
dtype02,
*while/lstm_cell_18/MatMul_1/ReadVariableOp└
while/lstm_cell_18/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_18/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         а2
while/lstm_cell_18/MatMul_1╕
while/lstm_cell_18/addAddV2#while/lstm_cell_18/MatMul:product:0%while/lstm_cell_18/MatMul_1:product:0*
T0*(
_output_shapes
:         а2
while/lstm_cell_18/add╚
)while/lstm_cell_18/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_18_biasadd_readvariableop_resource_0*
_output_shapes	
:а*
dtype02+
)while/lstm_cell_18/BiasAdd/ReadVariableOp┼
while/lstm_cell_18/BiasAddBiasAddwhile/lstm_cell_18/add:z:01while/lstm_cell_18/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         а2
while/lstm_cell_18/BiasAddК
"while/lstm_cell_18/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_18/split/split_dimЛ
while/lstm_cell_18/splitSplit+while/lstm_cell_18/split/split_dim:output:0#while/lstm_cell_18/BiasAdd:output:0*
T0*`
_output_shapesN
L:         h:         h:         h:         h*
	num_split2
while/lstm_cell_18/splitШ
while/lstm_cell_18/SigmoidSigmoid!while/lstm_cell_18/split:output:0*
T0*'
_output_shapes
:         h2
while/lstm_cell_18/SigmoidЬ
while/lstm_cell_18/Sigmoid_1Sigmoid!while/lstm_cell_18/split:output:1*
T0*'
_output_shapes
:         h2
while/lstm_cell_18/Sigmoid_1а
while/lstm_cell_18/mulMul while/lstm_cell_18/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:         h2
while/lstm_cell_18/mulП
while/lstm_cell_18/ReluRelu!while/lstm_cell_18/split:output:2*
T0*'
_output_shapes
:         h2
while/lstm_cell_18/Relu┤
while/lstm_cell_18/mul_1Mulwhile/lstm_cell_18/Sigmoid:y:0%while/lstm_cell_18/Relu:activations:0*
T0*'
_output_shapes
:         h2
while/lstm_cell_18/mul_1й
while/lstm_cell_18/add_1AddV2while/lstm_cell_18/mul:z:0while/lstm_cell_18/mul_1:z:0*
T0*'
_output_shapes
:         h2
while/lstm_cell_18/add_1Ь
while/lstm_cell_18/Sigmoid_2Sigmoid!while/lstm_cell_18/split:output:3*
T0*'
_output_shapes
:         h2
while/lstm_cell_18/Sigmoid_2О
while/lstm_cell_18/Relu_1Reluwhile/lstm_cell_18/add_1:z:0*
T0*'
_output_shapes
:         h2
while/lstm_cell_18/Relu_1╕
while/lstm_cell_18/mul_2Mul while/lstm_cell_18/Sigmoid_2:y:0'while/lstm_cell_18/Relu_1:activations:0*
T0*'
_output_shapes
:         h2
while/lstm_cell_18/mul_2р
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_18/mul_2:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1k
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity~
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_1m
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_2Ъ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3Н
while/Identity_4Identitywhile/lstm_cell_18/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:         h2
while/Identity_4Н
while/Identity_5Identitywhile/lstm_cell_18/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:         h2
while/Identity_5▐

while/NoOpNoOp*^while/lstm_cell_18/BiasAdd/ReadVariableOp)^while/lstm_cell_18/MatMul/ReadVariableOp+^while/lstm_cell_18/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_18_biasadd_readvariableop_resource4while_lstm_cell_18_biasadd_readvariableop_resource_0"l
3while_lstm_cell_18_matmul_1_readvariableop_resource5while_lstm_cell_18_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_18_matmul_readvariableop_resource3while_lstm_cell_18_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         h:         h: : : : : 2V
)while/lstm_cell_18/BiasAdd/ReadVariableOp)while/lstm_cell_18/BiasAdd/ReadVariableOp2T
(while/lstm_cell_18/MatMul/ReadVariableOp(while/lstm_cell_18/MatMul/ReadVariableOp2X
*while/lstm_cell_18/MatMul_1/ReadVariableOp*while/lstm_cell_18/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         h:-)
'
_output_shapes
:         h:

_output_shapes
: :

_output_shapes
: 
У
Й
J__inference_lstm_cell_19_layer_call_and_return_conditional_losses_32590845

inputs
states_0
states_11
matmul_readvariableop_resource:	h╕
4
 matmul_1_readvariableop_resource:
╬╕
.
biasadd_readvariableop_resource:	╕

identity

identity_1

identity_2ИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpвMatMul_1/ReadVariableOpО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	h╕
*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╕
2
MatMulХ
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
╬╕
*
dtype02
MatMul_1/ReadVariableOp|
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╕
2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:         ╕
2
addН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:╕
*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╕
2	
BiasAddd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim├
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*d
_output_shapesR
P:         ╬:         ╬:         ╬:         ╬*
	num_split2
split`
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:         ╬2	
Sigmoidd
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:         ╬2
	Sigmoid_1]
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:         ╬2
mulW
ReluRelusplit:output:2*
T0*(
_output_shapes
:         ╬2
Relui
mul_1MulSigmoid:y:0Relu:activations:0*
T0*(
_output_shapes
:         ╬2
mul_1^
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:         ╬2
add_1d
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:         ╬2
	Sigmoid_2V
Relu_1Relu	add_1:z:0*
T0*(
_output_shapes
:         ╬2
Relu_1m
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*(
_output_shapes
:         ╬2
mul_2e
IdentityIdentity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:         ╬2

Identityi

Identity_1Identity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:         ╬2

Identity_1i

Identity_2Identity	add_1:z:0^NoOp*
T0*(
_output_shapes
:         ╬2

Identity_2Щ
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:         h:         ╬:         ╬: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:         h
 
_user_specified_nameinputs:RN
(
_output_shapes
:         ╬
"
_user_specified_name
states/0:RN
(
_output_shapes
:         ╬
"
_user_specified_name
states/1
░?
╘
while_body_32587864
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_19_matmul_readvariableop_resource_0:	h╕
I
5while_lstm_cell_19_matmul_1_readvariableop_resource_0:
╬╕
C
4while_lstm_cell_19_biasadd_readvariableop_resource_0:	╕

while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_19_matmul_readvariableop_resource:	h╕
G
3while_lstm_cell_19_matmul_1_readvariableop_resource:
╬╕
A
2while_lstm_cell_19_biasadd_readvariableop_resource:	╕
Ив)while/lstm_cell_19/BiasAdd/ReadVariableOpв(while/lstm_cell_19/MatMul/ReadVariableOpв*while/lstm_cell_19/MatMul_1/ReadVariableOp├
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    h   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape╙
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         h*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem╔
(while/lstm_cell_19/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_19_matmul_readvariableop_resource_0*
_output_shapes
:	h╕
*
dtype02*
(while/lstm_cell_19/MatMul/ReadVariableOp╫
while/lstm_cell_19/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_19/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╕
2
while/lstm_cell_19/MatMul╨
*while/lstm_cell_19/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_19_matmul_1_readvariableop_resource_0* 
_output_shapes
:
╬╕
*
dtype02,
*while/lstm_cell_19/MatMul_1/ReadVariableOp└
while/lstm_cell_19/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_19/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╕
2
while/lstm_cell_19/MatMul_1╕
while/lstm_cell_19/addAddV2#while/lstm_cell_19/MatMul:product:0%while/lstm_cell_19/MatMul_1:product:0*
T0*(
_output_shapes
:         ╕
2
while/lstm_cell_19/add╚
)while/lstm_cell_19/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_19_biasadd_readvariableop_resource_0*
_output_shapes	
:╕
*
dtype02+
)while/lstm_cell_19/BiasAdd/ReadVariableOp┼
while/lstm_cell_19/BiasAddBiasAddwhile/lstm_cell_19/add:z:01while/lstm_cell_19/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╕
2
while/lstm_cell_19/BiasAddК
"while/lstm_cell_19/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_19/split/split_dimП
while/lstm_cell_19/splitSplit+while/lstm_cell_19/split/split_dim:output:0#while/lstm_cell_19/BiasAdd:output:0*
T0*d
_output_shapesR
P:         ╬:         ╬:         ╬:         ╬*
	num_split2
while/lstm_cell_19/splitЩ
while/lstm_cell_19/SigmoidSigmoid!while/lstm_cell_19/split:output:0*
T0*(
_output_shapes
:         ╬2
while/lstm_cell_19/SigmoidЭ
while/lstm_cell_19/Sigmoid_1Sigmoid!while/lstm_cell_19/split:output:1*
T0*(
_output_shapes
:         ╬2
while/lstm_cell_19/Sigmoid_1б
while/lstm_cell_19/mulMul while/lstm_cell_19/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:         ╬2
while/lstm_cell_19/mulР
while/lstm_cell_19/ReluRelu!while/lstm_cell_19/split:output:2*
T0*(
_output_shapes
:         ╬2
while/lstm_cell_19/Relu╡
while/lstm_cell_19/mul_1Mulwhile/lstm_cell_19/Sigmoid:y:0%while/lstm_cell_19/Relu:activations:0*
T0*(
_output_shapes
:         ╬2
while/lstm_cell_19/mul_1к
while/lstm_cell_19/add_1AddV2while/lstm_cell_19/mul:z:0while/lstm_cell_19/mul_1:z:0*
T0*(
_output_shapes
:         ╬2
while/lstm_cell_19/add_1Э
while/lstm_cell_19/Sigmoid_2Sigmoid!while/lstm_cell_19/split:output:3*
T0*(
_output_shapes
:         ╬2
while/lstm_cell_19/Sigmoid_2П
while/lstm_cell_19/Relu_1Reluwhile/lstm_cell_19/add_1:z:0*
T0*(
_output_shapes
:         ╬2
while/lstm_cell_19/Relu_1╣
while/lstm_cell_19/mul_2Mul while/lstm_cell_19/Sigmoid_2:y:0'while/lstm_cell_19/Relu_1:activations:0*
T0*(
_output_shapes
:         ╬2
while/lstm_cell_19/mul_2р
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_19/mul_2:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1k
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity~
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_1m
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_2Ъ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3О
while/Identity_4Identitywhile/lstm_cell_19/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:         ╬2
while/Identity_4О
while/Identity_5Identitywhile/lstm_cell_19/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:         ╬2
while/Identity_5▐

while/NoOpNoOp*^while/lstm_cell_19/BiasAdd/ReadVariableOp)^while/lstm_cell_19/MatMul/ReadVariableOp+^while/lstm_cell_19/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_19_biasadd_readvariableop_resource4while_lstm_cell_19_biasadd_readvariableop_resource_0"l
3while_lstm_cell_19_matmul_1_readvariableop_resource5while_lstm_cell_19_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_19_matmul_readvariableop_resource3while_lstm_cell_19_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :         ╬:         ╬: : : : : 2V
)while/lstm_cell_19/BiasAdd/ReadVariableOp)while/lstm_cell_19/BiasAdd/ReadVariableOp2T
(while/lstm_cell_19/MatMul/ReadVariableOp(while/lstm_cell_19/MatMul/ReadVariableOp2X
*while/lstm_cell_19/MatMul_1/ReadVariableOp*while/lstm_cell_19/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:         ╬:.*
(
_output_shapes
:         ╬:

_output_shapes
: :

_output_shapes
: 
Ч
ы
J__inference_sequential_9_layer_call_and_return_conditional_losses_32588001

inputs#
lstm_18_32587784:	]а#
lstm_18_32587786:	hа
lstm_18_32587788:	а#
lstm_19_32587949:	h╕
$
lstm_19_32587951:
╬╕

lstm_19_32587953:	╕
#
dense_9_32587995:	╬
dense_9_32587997:
identityИвdense_9/StatefulPartitionedCallвlstm_18/StatefulPartitionedCallвlstm_19/StatefulPartitionedCallн
lstm_18/StatefulPartitionedCallStatefulPartitionedCallinputslstm_18_32587784lstm_18_32587786lstm_18_32587788*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         h*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_lstm_18_layer_call_and_return_conditional_losses_325877832!
lstm_18/StatefulPartitionedCallВ
dropout_18/PartitionedCallPartitionedCall(lstm_18/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         h* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_dropout_18_layer_call_and_return_conditional_losses_325877962
dropout_18/PartitionedCall╦
lstm_19/StatefulPartitionedCallStatefulPartitionedCall#dropout_18/PartitionedCall:output:0lstm_19_32587949lstm_19_32587951lstm_19_32587953*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ╬*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_lstm_19_layer_call_and_return_conditional_losses_325879482!
lstm_19/StatefulPartitionedCallГ
dropout_19/PartitionedCallPartitionedCall(lstm_19/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ╬* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_dropout_19_layer_call_and_return_conditional_losses_325879612
dropout_19/PartitionedCall╢
dense_9/StatefulPartitionedCallStatefulPartitionedCall#dropout_19/PartitionedCall:output:0dense_9_32587995dense_9_32587997*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_dense_9_layer_call_and_return_conditional_losses_325879942!
dense_9/StatefulPartitionedCallЗ
IdentityIdentity(dense_9/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:         2

Identity┤
NoOpNoOp ^dense_9/StatefulPartitionedCall ^lstm_18/StatefulPartitionedCall ^lstm_19/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':         ]: : : : : : : : 2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall2B
lstm_18/StatefulPartitionedCalllstm_18/StatefulPartitionedCall2B
lstm_19/StatefulPartitionedCalllstm_19/StatefulPartitionedCall:S O
+
_output_shapes
:         ]
 
_user_specified_nameinputs
╣F
Н
E__inference_lstm_18_layer_call_and_return_conditional_losses_32586734

inputs(
lstm_cell_18_32586652:	]а(
lstm_cell_18_32586654:	hа$
lstm_cell_18_32586656:	а
identityИв$lstm_cell_18/StatefulPartitionedCallвwhileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :h2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :h2
zeros/packed/1Г
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:         h2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :h2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :h2
zeros_1/packed/1Й
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:         h2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permГ
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :                  ]2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ю
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1Е
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         2
TensorArrayV2/element_shape▓
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2┐
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ]   27
5TensorArrayUnstack/TensorListFromTensor/element_shape°
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2№
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         ]*
shrink_axis_mask2
strided_slice_2е
$lstm_cell_18/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_18_32586652lstm_cell_18_32586654lstm_cell_18_32586656*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:         h:         h:         h*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_lstm_cell_18_layer_call_and_return_conditional_losses_325865872&
$lstm_cell_18/StatefulPartitionedCallП
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    h   2
TensorArrayV2_1/element_shape╕
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter╩
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_18_32586652lstm_cell_18_32586654lstm_cell_18_32586656*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         h:         h: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_32586665*
condR
while_cond_32586664*K
output_shapes:
8: : : : :         h:         h: : : : : *
parallel_iterations 2
while╡
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    h   22
0TensorArrayV2Stack/TensorListStack/element_shapeё
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :                  h*
element_dtype02$
"TensorArrayV2Stack/TensorListStackБ
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2Ъ
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         h*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permо
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :                  h2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimew
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :                  h2

Identity}
NoOpNoOp%^lstm_cell_18/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  ]: : : 2L
$lstm_cell_18/StatefulPartitionedCall$lstm_cell_18/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :                  ]
 
_user_specified_nameinputs
ч[
Э
E__inference_lstm_18_layer_call_and_return_conditional_losses_32589746

inputs>
+lstm_cell_18_matmul_readvariableop_resource:	]а@
-lstm_cell_18_matmul_1_readvariableop_resource:	hа;
,lstm_cell_18_biasadd_readvariableop_resource:	а
identityИв#lstm_cell_18/BiasAdd/ReadVariableOpв"lstm_cell_18/MatMul/ReadVariableOpв$lstm_cell_18/MatMul_1/ReadVariableOpвwhileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :h2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :h2
zeros/packed/1Г
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:         h2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :h2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :h2
zeros_1/packed/1Й
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:         h2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:         ]2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ю
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1Е
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         2
TensorArrayV2/element_shape▓
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2┐
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ]   27
5TensorArrayUnstack/TensorListFromTensor/element_shape°
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2№
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         ]*
shrink_axis_mask2
strided_slice_2╡
"lstm_cell_18/MatMul/ReadVariableOpReadVariableOp+lstm_cell_18_matmul_readvariableop_resource*
_output_shapes
:	]а*
dtype02$
"lstm_cell_18/MatMul/ReadVariableOpн
lstm_cell_18/MatMulMatMulstrided_slice_2:output:0*lstm_cell_18/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         а2
lstm_cell_18/MatMul╗
$lstm_cell_18/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_18_matmul_1_readvariableop_resource*
_output_shapes
:	hа*
dtype02&
$lstm_cell_18/MatMul_1/ReadVariableOpй
lstm_cell_18/MatMul_1MatMulzeros:output:0,lstm_cell_18/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         а2
lstm_cell_18/MatMul_1а
lstm_cell_18/addAddV2lstm_cell_18/MatMul:product:0lstm_cell_18/MatMul_1:product:0*
T0*(
_output_shapes
:         а2
lstm_cell_18/add┤
#lstm_cell_18/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_18_biasadd_readvariableop_resource*
_output_shapes	
:а*
dtype02%
#lstm_cell_18/BiasAdd/ReadVariableOpн
lstm_cell_18/BiasAddBiasAddlstm_cell_18/add:z:0+lstm_cell_18/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         а2
lstm_cell_18/BiasAdd~
lstm_cell_18/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_18/split/split_dimє
lstm_cell_18/splitSplit%lstm_cell_18/split/split_dim:output:0lstm_cell_18/BiasAdd:output:0*
T0*`
_output_shapesN
L:         h:         h:         h:         h*
	num_split2
lstm_cell_18/splitЖ
lstm_cell_18/SigmoidSigmoidlstm_cell_18/split:output:0*
T0*'
_output_shapes
:         h2
lstm_cell_18/SigmoidК
lstm_cell_18/Sigmoid_1Sigmoidlstm_cell_18/split:output:1*
T0*'
_output_shapes
:         h2
lstm_cell_18/Sigmoid_1Л
lstm_cell_18/mulMullstm_cell_18/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:         h2
lstm_cell_18/mul}
lstm_cell_18/ReluRelulstm_cell_18/split:output:2*
T0*'
_output_shapes
:         h2
lstm_cell_18/ReluЬ
lstm_cell_18/mul_1Mullstm_cell_18/Sigmoid:y:0lstm_cell_18/Relu:activations:0*
T0*'
_output_shapes
:         h2
lstm_cell_18/mul_1С
lstm_cell_18/add_1AddV2lstm_cell_18/mul:z:0lstm_cell_18/mul_1:z:0*
T0*'
_output_shapes
:         h2
lstm_cell_18/add_1К
lstm_cell_18/Sigmoid_2Sigmoidlstm_cell_18/split:output:3*
T0*'
_output_shapes
:         h2
lstm_cell_18/Sigmoid_2|
lstm_cell_18/Relu_1Relulstm_cell_18/add_1:z:0*
T0*'
_output_shapes
:         h2
lstm_cell_18/Relu_1а
lstm_cell_18/mul_2Mullstm_cell_18/Sigmoid_2:y:0!lstm_cell_18/Relu_1:activations:0*
T0*'
_output_shapes
:         h2
lstm_cell_18/mul_2П
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    h   2
TensorArrayV2_1/element_shape╕
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterП
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_18_matmul_readvariableop_resource-lstm_cell_18_matmul_1_readvariableop_resource,lstm_cell_18_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         h:         h: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_32589662*
condR
while_cond_32589661*K
output_shapes:
8: : : : :         h:         h: : : : : *
parallel_iterations 2
while╡
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    h   22
0TensorArrayV2Stack/TensorListStack/element_shapeш
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:         h*
element_dtype02$
"TensorArrayV2Stack/TensorListStackБ
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2Ъ
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         h*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permе
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:         h2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimen
IdentityIdentitytranspose_1:y:0^NoOp*
T0*+
_output_shapes
:         h2

Identity╚
NoOpNoOp$^lstm_cell_18/BiasAdd/ReadVariableOp#^lstm_cell_18/MatMul/ReadVariableOp%^lstm_cell_18/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         ]: : : 2J
#lstm_cell_18/BiasAdd/ReadVariableOp#lstm_cell_18/BiasAdd/ReadVariableOp2H
"lstm_cell_18/MatMul/ReadVariableOp"lstm_cell_18/MatMul/ReadVariableOp2L
$lstm_cell_18/MatMul_1/ReadVariableOp$lstm_cell_18/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:         ]
 
_user_specified_nameinputs
▀
═
while_cond_32586664
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_32586664___redundant_placeholder06
2while_while_cond_32586664___redundant_placeholder16
2while_while_cond_32586664___redundant_placeholder26
2while_while_cond_32586664___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :         h:         h: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         h:-)
'
_output_shapes
:         h:

_output_shapes
: :

_output_shapes
:
Л
З
J__inference_lstm_cell_19_layer_call_and_return_conditional_losses_32587217

inputs

states
states_11
matmul_readvariableop_resource:	h╕
4
 matmul_1_readvariableop_resource:
╬╕
.
biasadd_readvariableop_resource:	╕

identity

identity_1

identity_2ИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpвMatMul_1/ReadVariableOpО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	h╕
*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╕
2
MatMulХ
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
╬╕
*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╕
2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:         ╕
2
addН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:╕
*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╕
2	
BiasAddd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim├
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*d
_output_shapesR
P:         ╬:         ╬:         ╬:         ╬*
	num_split2
split`
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:         ╬2	
Sigmoidd
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:         ╬2
	Sigmoid_1]
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:         ╬2
mulW
ReluRelusplit:output:2*
T0*(
_output_shapes
:         ╬2
Relui
mul_1MulSigmoid:y:0Relu:activations:0*
T0*(
_output_shapes
:         ╬2
mul_1^
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:         ╬2
add_1d
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:         ╬2
	Sigmoid_2V
Relu_1Relu	add_1:z:0*
T0*(
_output_shapes
:         ╬2
Relu_1m
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*(
_output_shapes
:         ╬2
mul_2e
IdentityIdentity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:         ╬2

Identityi

Identity_1Identity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:         ╬2

Identity_1i

Identity_2Identity	add_1:z:0^NoOp*
T0*(
_output_shapes
:         ╬2

Identity_2Щ
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:         h:         ╬:         ╬: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:         h
 
_user_specified_nameinputs:PL
(
_output_shapes
:         ╬
 
_user_specified_namestates:PL
(
_output_shapes
:         ╬
 
_user_specified_namestates
▌
╣
*__inference_lstm_18_layer_call_fn_32589908
inputs_0
unknown:	]а
	unknown_0:	hа
	unknown_1:	а
identityИвStatefulPartitionedCallС
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :                  h*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_lstm_18_layer_call_and_return_conditional_losses_325865242
StatefulPartitionedCallИ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :                  h2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  ]: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :                  ]
"
_user_specified_name
inputs/0
╪
I
-__inference_dropout_19_layer_call_fn_32590638

inputs
identity╦
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ╬* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_dropout_19_layer_call_and_return_conditional_losses_325879612
PartitionedCallq
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:         ╬2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ╬:T P
,
_output_shapes
:         ╬
 
_user_specified_nameinputs
Д\
Ю
E__inference_lstm_19_layer_call_and_return_conditional_losses_32590572

inputs>
+lstm_cell_19_matmul_readvariableop_resource:	h╕
A
-lstm_cell_19_matmul_1_readvariableop_resource:
╬╕
;
,lstm_cell_19_biasadd_readvariableop_resource:	╕

identityИв#lstm_cell_19/BiasAdd/ReadVariableOpв"lstm_cell_19/MatMul/ReadVariableOpв$lstm_cell_19/MatMul_1/ReadVariableOpвwhileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice]
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :╬2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessc
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :╬2
zeros/packed/1Г
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:         ╬2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :╬2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :╬2
zeros_1/packed/1Й
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const~
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:         ╬2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:         h2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ю
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1Е
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         2
TensorArrayV2/element_shape▓
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2┐
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    h   27
5TensorArrayUnstack/TensorListFromTensor/element_shape°
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2№
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         h*
shrink_axis_mask2
strided_slice_2╡
"lstm_cell_19/MatMul/ReadVariableOpReadVariableOp+lstm_cell_19_matmul_readvariableop_resource*
_output_shapes
:	h╕
*
dtype02$
"lstm_cell_19/MatMul/ReadVariableOpн
lstm_cell_19/MatMulMatMulstrided_slice_2:output:0*lstm_cell_19/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╕
2
lstm_cell_19/MatMul╝
$lstm_cell_19/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_19_matmul_1_readvariableop_resource* 
_output_shapes
:
╬╕
*
dtype02&
$lstm_cell_19/MatMul_1/ReadVariableOpй
lstm_cell_19/MatMul_1MatMulzeros:output:0,lstm_cell_19/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╕
2
lstm_cell_19/MatMul_1а
lstm_cell_19/addAddV2lstm_cell_19/MatMul:product:0lstm_cell_19/MatMul_1:product:0*
T0*(
_output_shapes
:         ╕
2
lstm_cell_19/add┤
#lstm_cell_19/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_19_biasadd_readvariableop_resource*
_output_shapes	
:╕
*
dtype02%
#lstm_cell_19/BiasAdd/ReadVariableOpн
lstm_cell_19/BiasAddBiasAddlstm_cell_19/add:z:0+lstm_cell_19/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╕
2
lstm_cell_19/BiasAdd~
lstm_cell_19/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_19/split/split_dimў
lstm_cell_19/splitSplit%lstm_cell_19/split/split_dim:output:0lstm_cell_19/BiasAdd:output:0*
T0*d
_output_shapesR
P:         ╬:         ╬:         ╬:         ╬*
	num_split2
lstm_cell_19/splitЗ
lstm_cell_19/SigmoidSigmoidlstm_cell_19/split:output:0*
T0*(
_output_shapes
:         ╬2
lstm_cell_19/SigmoidЛ
lstm_cell_19/Sigmoid_1Sigmoidlstm_cell_19/split:output:1*
T0*(
_output_shapes
:         ╬2
lstm_cell_19/Sigmoid_1М
lstm_cell_19/mulMullstm_cell_19/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:         ╬2
lstm_cell_19/mul~
lstm_cell_19/ReluRelulstm_cell_19/split:output:2*
T0*(
_output_shapes
:         ╬2
lstm_cell_19/ReluЭ
lstm_cell_19/mul_1Mullstm_cell_19/Sigmoid:y:0lstm_cell_19/Relu:activations:0*
T0*(
_output_shapes
:         ╬2
lstm_cell_19/mul_1Т
lstm_cell_19/add_1AddV2lstm_cell_19/mul:z:0lstm_cell_19/mul_1:z:0*
T0*(
_output_shapes
:         ╬2
lstm_cell_19/add_1Л
lstm_cell_19/Sigmoid_2Sigmoidlstm_cell_19/split:output:3*
T0*(
_output_shapes
:         ╬2
lstm_cell_19/Sigmoid_2}
lstm_cell_19/Relu_1Relulstm_cell_19/add_1:z:0*
T0*(
_output_shapes
:         ╬2
lstm_cell_19/Relu_1б
lstm_cell_19/mul_2Mullstm_cell_19/Sigmoid_2:y:0!lstm_cell_19/Relu_1:activations:0*
T0*(
_output_shapes
:         ╬2
lstm_cell_19/mul_2П
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    N  2
TensorArrayV2_1/element_shape╕
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterУ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_19_matmul_readvariableop_resource-lstm_cell_19_matmul_1_readvariableop_resource,lstm_cell_19_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :         ╬:         ╬: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_32590488*
condR
while_cond_32590487*M
output_shapes<
:: : : : :         ╬:         ╬: : : : : *
parallel_iterations 2
while╡
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    N  22
0TensorArrayV2Stack/TensorListStack/element_shapeщ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:         ╬*
element_dtype02$
"TensorArrayV2Stack/TensorListStackБ
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2Ы
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         ╬*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permж
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:         ╬2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimeo
IdentityIdentitytranspose_1:y:0^NoOp*
T0*,
_output_shapes
:         ╬2

Identity╚
NoOpNoOp$^lstm_cell_19/BiasAdd/ReadVariableOp#^lstm_cell_19/MatMul/ReadVariableOp%^lstm_cell_19/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         h: : : 2J
#lstm_cell_19/BiasAdd/ReadVariableOp#lstm_cell_19/BiasAdd/ReadVariableOp2H
"lstm_cell_19/MatMul/ReadVariableOp"lstm_cell_19/MatMul/ReadVariableOp2L
$lstm_cell_19/MatMul_1/ReadVariableOp$lstm_cell_19/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:         h
 
_user_specified_nameinputs
╟
∙
/__inference_lstm_cell_19_layer_call_fn_32590862

inputs
states_0
states_1
unknown:	h╕

	unknown_0:
╬╕

	unknown_1:	╕

identity

identity_1

identity_2ИвStatefulPartitionedCall╚
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:         ╬:         ╬:         ╬*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_lstm_cell_19_layer_call_and_return_conditional_losses_325870712
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         ╬2

IdentityА

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:         ╬2

Identity_1А

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:         ╬2

Identity_2h
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:         h:         ╬:         ╬: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         h
 
_user_specified_nameinputs:RN
(
_output_shapes
:         ╬
"
_user_specified_name
states/0:RN
(
_output_shapes
:         ╬
"
_user_specified_name
states/1
Ц]
є
(sequential_9_lstm_18_while_body_32586106F
Bsequential_9_lstm_18_while_sequential_9_lstm_18_while_loop_counterL
Hsequential_9_lstm_18_while_sequential_9_lstm_18_while_maximum_iterations*
&sequential_9_lstm_18_while_placeholder,
(sequential_9_lstm_18_while_placeholder_1,
(sequential_9_lstm_18_while_placeholder_2,
(sequential_9_lstm_18_while_placeholder_3E
Asequential_9_lstm_18_while_sequential_9_lstm_18_strided_slice_1_0Б
}sequential_9_lstm_18_while_tensorarrayv2read_tensorlistgetitem_sequential_9_lstm_18_tensorarrayunstack_tensorlistfromtensor_0[
Hsequential_9_lstm_18_while_lstm_cell_18_matmul_readvariableop_resource_0:	]а]
Jsequential_9_lstm_18_while_lstm_cell_18_matmul_1_readvariableop_resource_0:	hаX
Isequential_9_lstm_18_while_lstm_cell_18_biasadd_readvariableop_resource_0:	а'
#sequential_9_lstm_18_while_identity)
%sequential_9_lstm_18_while_identity_1)
%sequential_9_lstm_18_while_identity_2)
%sequential_9_lstm_18_while_identity_3)
%sequential_9_lstm_18_while_identity_4)
%sequential_9_lstm_18_while_identity_5C
?sequential_9_lstm_18_while_sequential_9_lstm_18_strided_slice_1
{sequential_9_lstm_18_while_tensorarrayv2read_tensorlistgetitem_sequential_9_lstm_18_tensorarrayunstack_tensorlistfromtensorY
Fsequential_9_lstm_18_while_lstm_cell_18_matmul_readvariableop_resource:	]а[
Hsequential_9_lstm_18_while_lstm_cell_18_matmul_1_readvariableop_resource:	hаV
Gsequential_9_lstm_18_while_lstm_cell_18_biasadd_readvariableop_resource:	аИв>sequential_9/lstm_18/while/lstm_cell_18/BiasAdd/ReadVariableOpв=sequential_9/lstm_18/while/lstm_cell_18/MatMul/ReadVariableOpв?sequential_9/lstm_18/while/lstm_cell_18/MatMul_1/ReadVariableOpэ
Lsequential_9/lstm_18/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ]   2N
Lsequential_9/lstm_18/while/TensorArrayV2Read/TensorListGetItem/element_shape╤
>sequential_9/lstm_18/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem}sequential_9_lstm_18_while_tensorarrayv2read_tensorlistgetitem_sequential_9_lstm_18_tensorarrayunstack_tensorlistfromtensor_0&sequential_9_lstm_18_while_placeholderUsequential_9/lstm_18/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         ]*
element_dtype02@
>sequential_9/lstm_18/while/TensorArrayV2Read/TensorListGetItemИ
=sequential_9/lstm_18/while/lstm_cell_18/MatMul/ReadVariableOpReadVariableOpHsequential_9_lstm_18_while_lstm_cell_18_matmul_readvariableop_resource_0*
_output_shapes
:	]а*
dtype02?
=sequential_9/lstm_18/while/lstm_cell_18/MatMul/ReadVariableOpл
.sequential_9/lstm_18/while/lstm_cell_18/MatMulMatMulEsequential_9/lstm_18/while/TensorArrayV2Read/TensorListGetItem:item:0Esequential_9/lstm_18/while/lstm_cell_18/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         а20
.sequential_9/lstm_18/while/lstm_cell_18/MatMulО
?sequential_9/lstm_18/while/lstm_cell_18/MatMul_1/ReadVariableOpReadVariableOpJsequential_9_lstm_18_while_lstm_cell_18_matmul_1_readvariableop_resource_0*
_output_shapes
:	hа*
dtype02A
?sequential_9/lstm_18/while/lstm_cell_18/MatMul_1/ReadVariableOpФ
0sequential_9/lstm_18/while/lstm_cell_18/MatMul_1MatMul(sequential_9_lstm_18_while_placeholder_2Gsequential_9/lstm_18/while/lstm_cell_18/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         а22
0sequential_9/lstm_18/while/lstm_cell_18/MatMul_1М
+sequential_9/lstm_18/while/lstm_cell_18/addAddV28sequential_9/lstm_18/while/lstm_cell_18/MatMul:product:0:sequential_9/lstm_18/while/lstm_cell_18/MatMul_1:product:0*
T0*(
_output_shapes
:         а2-
+sequential_9/lstm_18/while/lstm_cell_18/addЗ
>sequential_9/lstm_18/while/lstm_cell_18/BiasAdd/ReadVariableOpReadVariableOpIsequential_9_lstm_18_while_lstm_cell_18_biasadd_readvariableop_resource_0*
_output_shapes	
:а*
dtype02@
>sequential_9/lstm_18/while/lstm_cell_18/BiasAdd/ReadVariableOpЩ
/sequential_9/lstm_18/while/lstm_cell_18/BiasAddBiasAdd/sequential_9/lstm_18/while/lstm_cell_18/add:z:0Fsequential_9/lstm_18/while/lstm_cell_18/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         а21
/sequential_9/lstm_18/while/lstm_cell_18/BiasAdd┤
7sequential_9/lstm_18/while/lstm_cell_18/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :29
7sequential_9/lstm_18/while/lstm_cell_18/split/split_dim▀
-sequential_9/lstm_18/while/lstm_cell_18/splitSplit@sequential_9/lstm_18/while/lstm_cell_18/split/split_dim:output:08sequential_9/lstm_18/while/lstm_cell_18/BiasAdd:output:0*
T0*`
_output_shapesN
L:         h:         h:         h:         h*
	num_split2/
-sequential_9/lstm_18/while/lstm_cell_18/split╫
/sequential_9/lstm_18/while/lstm_cell_18/SigmoidSigmoid6sequential_9/lstm_18/while/lstm_cell_18/split:output:0*
T0*'
_output_shapes
:         h21
/sequential_9/lstm_18/while/lstm_cell_18/Sigmoid█
1sequential_9/lstm_18/while/lstm_cell_18/Sigmoid_1Sigmoid6sequential_9/lstm_18/while/lstm_cell_18/split:output:1*
T0*'
_output_shapes
:         h23
1sequential_9/lstm_18/while/lstm_cell_18/Sigmoid_1Ї
+sequential_9/lstm_18/while/lstm_cell_18/mulMul5sequential_9/lstm_18/while/lstm_cell_18/Sigmoid_1:y:0(sequential_9_lstm_18_while_placeholder_3*
T0*'
_output_shapes
:         h2-
+sequential_9/lstm_18/while/lstm_cell_18/mul╬
,sequential_9/lstm_18/while/lstm_cell_18/ReluRelu6sequential_9/lstm_18/while/lstm_cell_18/split:output:2*
T0*'
_output_shapes
:         h2.
,sequential_9/lstm_18/while/lstm_cell_18/ReluИ
-sequential_9/lstm_18/while/lstm_cell_18/mul_1Mul3sequential_9/lstm_18/while/lstm_cell_18/Sigmoid:y:0:sequential_9/lstm_18/while/lstm_cell_18/Relu:activations:0*
T0*'
_output_shapes
:         h2/
-sequential_9/lstm_18/while/lstm_cell_18/mul_1¤
-sequential_9/lstm_18/while/lstm_cell_18/add_1AddV2/sequential_9/lstm_18/while/lstm_cell_18/mul:z:01sequential_9/lstm_18/while/lstm_cell_18/mul_1:z:0*
T0*'
_output_shapes
:         h2/
-sequential_9/lstm_18/while/lstm_cell_18/add_1█
1sequential_9/lstm_18/while/lstm_cell_18/Sigmoid_2Sigmoid6sequential_9/lstm_18/while/lstm_cell_18/split:output:3*
T0*'
_output_shapes
:         h23
1sequential_9/lstm_18/while/lstm_cell_18/Sigmoid_2═
.sequential_9/lstm_18/while/lstm_cell_18/Relu_1Relu1sequential_9/lstm_18/while/lstm_cell_18/add_1:z:0*
T0*'
_output_shapes
:         h20
.sequential_9/lstm_18/while/lstm_cell_18/Relu_1М
-sequential_9/lstm_18/while/lstm_cell_18/mul_2Mul5sequential_9/lstm_18/while/lstm_cell_18/Sigmoid_2:y:0<sequential_9/lstm_18/while/lstm_cell_18/Relu_1:activations:0*
T0*'
_output_shapes
:         h2/
-sequential_9/lstm_18/while/lstm_cell_18/mul_2╔
?sequential_9/lstm_18/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem(sequential_9_lstm_18_while_placeholder_1&sequential_9_lstm_18_while_placeholder1sequential_9/lstm_18/while/lstm_cell_18/mul_2:z:0*
_output_shapes
: *
element_dtype02A
?sequential_9/lstm_18/while/TensorArrayV2Write/TensorListSetItemЖ
 sequential_9/lstm_18/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2"
 sequential_9/lstm_18/while/add/y╜
sequential_9/lstm_18/while/addAddV2&sequential_9_lstm_18_while_placeholder)sequential_9/lstm_18/while/add/y:output:0*
T0*
_output_shapes
: 2 
sequential_9/lstm_18/while/addК
"sequential_9/lstm_18/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2$
"sequential_9/lstm_18/while/add_1/y▀
 sequential_9/lstm_18/while/add_1AddV2Bsequential_9_lstm_18_while_sequential_9_lstm_18_while_loop_counter+sequential_9/lstm_18/while/add_1/y:output:0*
T0*
_output_shapes
: 2"
 sequential_9/lstm_18/while/add_1┐
#sequential_9/lstm_18/while/IdentityIdentity$sequential_9/lstm_18/while/add_1:z:0 ^sequential_9/lstm_18/while/NoOp*
T0*
_output_shapes
: 2%
#sequential_9/lstm_18/while/Identityч
%sequential_9/lstm_18/while/Identity_1IdentityHsequential_9_lstm_18_while_sequential_9_lstm_18_while_maximum_iterations ^sequential_9/lstm_18/while/NoOp*
T0*
_output_shapes
: 2'
%sequential_9/lstm_18/while/Identity_1┴
%sequential_9/lstm_18/while/Identity_2Identity"sequential_9/lstm_18/while/add:z:0 ^sequential_9/lstm_18/while/NoOp*
T0*
_output_shapes
: 2'
%sequential_9/lstm_18/while/Identity_2ю
%sequential_9/lstm_18/while/Identity_3IdentityOsequential_9/lstm_18/while/TensorArrayV2Write/TensorListSetItem:output_handle:0 ^sequential_9/lstm_18/while/NoOp*
T0*
_output_shapes
: 2'
%sequential_9/lstm_18/while/Identity_3с
%sequential_9/lstm_18/while/Identity_4Identity1sequential_9/lstm_18/while/lstm_cell_18/mul_2:z:0 ^sequential_9/lstm_18/while/NoOp*
T0*'
_output_shapes
:         h2'
%sequential_9/lstm_18/while/Identity_4с
%sequential_9/lstm_18/while/Identity_5Identity1sequential_9/lstm_18/while/lstm_cell_18/add_1:z:0 ^sequential_9/lstm_18/while/NoOp*
T0*'
_output_shapes
:         h2'
%sequential_9/lstm_18/while/Identity_5╟
sequential_9/lstm_18/while/NoOpNoOp?^sequential_9/lstm_18/while/lstm_cell_18/BiasAdd/ReadVariableOp>^sequential_9/lstm_18/while/lstm_cell_18/MatMul/ReadVariableOp@^sequential_9/lstm_18/while/lstm_cell_18/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2!
sequential_9/lstm_18/while/NoOp"S
#sequential_9_lstm_18_while_identity,sequential_9/lstm_18/while/Identity:output:0"W
%sequential_9_lstm_18_while_identity_1.sequential_9/lstm_18/while/Identity_1:output:0"W
%sequential_9_lstm_18_while_identity_2.sequential_9/lstm_18/while/Identity_2:output:0"W
%sequential_9_lstm_18_while_identity_3.sequential_9/lstm_18/while/Identity_3:output:0"W
%sequential_9_lstm_18_while_identity_4.sequential_9/lstm_18/while/Identity_4:output:0"W
%sequential_9_lstm_18_while_identity_5.sequential_9/lstm_18/while/Identity_5:output:0"Ф
Gsequential_9_lstm_18_while_lstm_cell_18_biasadd_readvariableop_resourceIsequential_9_lstm_18_while_lstm_cell_18_biasadd_readvariableop_resource_0"Ц
Hsequential_9_lstm_18_while_lstm_cell_18_matmul_1_readvariableop_resourceJsequential_9_lstm_18_while_lstm_cell_18_matmul_1_readvariableop_resource_0"Т
Fsequential_9_lstm_18_while_lstm_cell_18_matmul_readvariableop_resourceHsequential_9_lstm_18_while_lstm_cell_18_matmul_readvariableop_resource_0"Д
?sequential_9_lstm_18_while_sequential_9_lstm_18_strided_slice_1Asequential_9_lstm_18_while_sequential_9_lstm_18_strided_slice_1_0"№
{sequential_9_lstm_18_while_tensorarrayv2read_tensorlistgetitem_sequential_9_lstm_18_tensorarrayunstack_tensorlistfromtensor}sequential_9_lstm_18_while_tensorarrayv2read_tensorlistgetitem_sequential_9_lstm_18_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         h:         h: : : : : 2А
>sequential_9/lstm_18/while/lstm_cell_18/BiasAdd/ReadVariableOp>sequential_9/lstm_18/while/lstm_cell_18/BiasAdd/ReadVariableOp2~
=sequential_9/lstm_18/while/lstm_cell_18/MatMul/ReadVariableOp=sequential_9/lstm_18/while/lstm_cell_18/MatMul/ReadVariableOp2В
?sequential_9/lstm_18/while/lstm_cell_18/MatMul_1/ReadVariableOp?sequential_9/lstm_18/while/lstm_cell_18/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         h:-)
'
_output_shapes
:         h:

_output_shapes
: :

_output_shapes
: 
╣
╝
J__inference_sequential_9_layer_call_and_return_conditional_losses_32588560
lstm_18_input#
lstm_18_32588538:	]а#
lstm_18_32588540:	hа
lstm_18_32588542:	а#
lstm_19_32588546:	h╕
$
lstm_19_32588548:
╬╕

lstm_19_32588550:	╕
#
dense_9_32588554:	╬
dense_9_32588556:
identityИвdense_9/StatefulPartitionedCallв"dropout_18/StatefulPartitionedCallв"dropout_19/StatefulPartitionedCallвlstm_18/StatefulPartitionedCallвlstm_19/StatefulPartitionedCall┤
lstm_18/StatefulPartitionedCallStatefulPartitionedCalllstm_18_inputlstm_18_32588538lstm_18_32588540lstm_18_32588542*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         h*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_lstm_18_layer_call_and_return_conditional_losses_325884132!
lstm_18/StatefulPartitionedCallЪ
"dropout_18/StatefulPartitionedCallStatefulPartitionedCall(lstm_18/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         h* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_dropout_18_layer_call_and_return_conditional_losses_325882462$
"dropout_18/StatefulPartitionedCall╙
lstm_19/StatefulPartitionedCallStatefulPartitionedCall+dropout_18/StatefulPartitionedCall:output:0lstm_19_32588546lstm_19_32588548lstm_19_32588550*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ╬*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_lstm_19_layer_call_and_return_conditional_losses_325882172!
lstm_19/StatefulPartitionedCall└
"dropout_19/StatefulPartitionedCallStatefulPartitionedCall(lstm_19/StatefulPartitionedCall:output:0#^dropout_18/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ╬* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_dropout_19_layer_call_and_return_conditional_losses_325880502$
"dropout_19/StatefulPartitionedCall╛
dense_9/StatefulPartitionedCallStatefulPartitionedCall+dropout_19/StatefulPartitionedCall:output:0dense_9_32588554dense_9_32588556*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_dense_9_layer_call_and_return_conditional_losses_325879942!
dense_9/StatefulPartitionedCallЗ
IdentityIdentity(dense_9/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:         2

Identity■
NoOpNoOp ^dense_9/StatefulPartitionedCall#^dropout_18/StatefulPartitionedCall#^dropout_19/StatefulPartitionedCall ^lstm_18/StatefulPartitionedCall ^lstm_19/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':         ]: : : : : : : : 2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall2H
"dropout_18/StatefulPartitionedCall"dropout_18/StatefulPartitionedCall2H
"dropout_19/StatefulPartitionedCall"dropout_19/StatefulPartitionedCall2B
lstm_18/StatefulPartitionedCalllstm_18/StatefulPartitionedCall2B
lstm_19/StatefulPartitionedCalllstm_19/StatefulPartitionedCall:Z V
+
_output_shapes
:         ]
'
_user_specified_namelstm_18_input
░?
╘
while_body_32588133
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_19_matmul_readvariableop_resource_0:	h╕
I
5while_lstm_cell_19_matmul_1_readvariableop_resource_0:
╬╕
C
4while_lstm_cell_19_biasadd_readvariableop_resource_0:	╕

while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_19_matmul_readvariableop_resource:	h╕
G
3while_lstm_cell_19_matmul_1_readvariableop_resource:
╬╕
A
2while_lstm_cell_19_biasadd_readvariableop_resource:	╕
Ив)while/lstm_cell_19/BiasAdd/ReadVariableOpв(while/lstm_cell_19/MatMul/ReadVariableOpв*while/lstm_cell_19/MatMul_1/ReadVariableOp├
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    h   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape╙
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         h*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem╔
(while/lstm_cell_19/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_19_matmul_readvariableop_resource_0*
_output_shapes
:	h╕
*
dtype02*
(while/lstm_cell_19/MatMul/ReadVariableOp╫
while/lstm_cell_19/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_19/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╕
2
while/lstm_cell_19/MatMul╨
*while/lstm_cell_19/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_19_matmul_1_readvariableop_resource_0* 
_output_shapes
:
╬╕
*
dtype02,
*while/lstm_cell_19/MatMul_1/ReadVariableOp└
while/lstm_cell_19/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_19/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╕
2
while/lstm_cell_19/MatMul_1╕
while/lstm_cell_19/addAddV2#while/lstm_cell_19/MatMul:product:0%while/lstm_cell_19/MatMul_1:product:0*
T0*(
_output_shapes
:         ╕
2
while/lstm_cell_19/add╚
)while/lstm_cell_19/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_19_biasadd_readvariableop_resource_0*
_output_shapes	
:╕
*
dtype02+
)while/lstm_cell_19/BiasAdd/ReadVariableOp┼
while/lstm_cell_19/BiasAddBiasAddwhile/lstm_cell_19/add:z:01while/lstm_cell_19/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╕
2
while/lstm_cell_19/BiasAddК
"while/lstm_cell_19/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_19/split/split_dimП
while/lstm_cell_19/splitSplit+while/lstm_cell_19/split/split_dim:output:0#while/lstm_cell_19/BiasAdd:output:0*
T0*d
_output_shapesR
P:         ╬:         ╬:         ╬:         ╬*
	num_split2
while/lstm_cell_19/splitЩ
while/lstm_cell_19/SigmoidSigmoid!while/lstm_cell_19/split:output:0*
T0*(
_output_shapes
:         ╬2
while/lstm_cell_19/SigmoidЭ
while/lstm_cell_19/Sigmoid_1Sigmoid!while/lstm_cell_19/split:output:1*
T0*(
_output_shapes
:         ╬2
while/lstm_cell_19/Sigmoid_1б
while/lstm_cell_19/mulMul while/lstm_cell_19/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:         ╬2
while/lstm_cell_19/mulР
while/lstm_cell_19/ReluRelu!while/lstm_cell_19/split:output:2*
T0*(
_output_shapes
:         ╬2
while/lstm_cell_19/Relu╡
while/lstm_cell_19/mul_1Mulwhile/lstm_cell_19/Sigmoid:y:0%while/lstm_cell_19/Relu:activations:0*
T0*(
_output_shapes
:         ╬2
while/lstm_cell_19/mul_1к
while/lstm_cell_19/add_1AddV2while/lstm_cell_19/mul:z:0while/lstm_cell_19/mul_1:z:0*
T0*(
_output_shapes
:         ╬2
while/lstm_cell_19/add_1Э
while/lstm_cell_19/Sigmoid_2Sigmoid!while/lstm_cell_19/split:output:3*
T0*(
_output_shapes
:         ╬2
while/lstm_cell_19/Sigmoid_2П
while/lstm_cell_19/Relu_1Reluwhile/lstm_cell_19/add_1:z:0*
T0*(
_output_shapes
:         ╬2
while/lstm_cell_19/Relu_1╣
while/lstm_cell_19/mul_2Mul while/lstm_cell_19/Sigmoid_2:y:0'while/lstm_cell_19/Relu_1:activations:0*
T0*(
_output_shapes
:         ╬2
while/lstm_cell_19/mul_2р
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_19/mul_2:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1k
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity~
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_1m
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_2Ъ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3О
while/Identity_4Identitywhile/lstm_cell_19/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:         ╬2
while/Identity_4О
while/Identity_5Identitywhile/lstm_cell_19/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:         ╬2
while/Identity_5▐

while/NoOpNoOp*^while/lstm_cell_19/BiasAdd/ReadVariableOp)^while/lstm_cell_19/MatMul/ReadVariableOp+^while/lstm_cell_19/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_19_biasadd_readvariableop_resource4while_lstm_cell_19_biasadd_readvariableop_resource_0"l
3while_lstm_cell_19_matmul_1_readvariableop_resource5while_lstm_cell_19_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_19_matmul_readvariableop_resource3while_lstm_cell_19_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :         ╬:         ╬: : : : : 2V
)while/lstm_cell_19/BiasAdd/ReadVariableOp)while/lstm_cell_19/BiasAdd/ReadVariableOp2T
(while/lstm_cell_19/MatMul/ReadVariableOp(while/lstm_cell_19/MatMul/ReadVariableOp2X
*while/lstm_cell_19/MatMul_1/ReadVariableOp*while/lstm_cell_19/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:         ╬:.*
(
_output_shapes
:         ╬:

_output_shapes
: :

_output_shapes
: 
╦F
О
E__inference_lstm_19_layer_call_and_return_conditional_losses_32587364

inputs(
lstm_cell_19_32587282:	h╕
)
lstm_cell_19_32587284:
╬╕
$
lstm_cell_19_32587286:	╕

identityИв$lstm_cell_19/StatefulPartitionedCallвwhileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice]
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :╬2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessc
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :╬2
zeros/packed/1Г
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:         ╬2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :╬2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :╬2
zeros_1/packed/1Й
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const~
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:         ╬2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permГ
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :                  h2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ю
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1Е
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         2
TensorArrayV2/element_shape▓
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2┐
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    h   27
5TensorArrayUnstack/TensorListFromTensor/element_shape°
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2№
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         h*
shrink_axis_mask2
strided_slice_2и
$lstm_cell_19/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_19_32587282lstm_cell_19_32587284lstm_cell_19_32587286*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:         ╬:         ╬:         ╬*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_lstm_cell_19_layer_call_and_return_conditional_losses_325872172&
$lstm_cell_19/StatefulPartitionedCallП
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    N  2
TensorArrayV2_1/element_shape╕
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter╬
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_19_32587282lstm_cell_19_32587284lstm_cell_19_32587286*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :         ╬:         ╬: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_32587295*
condR
while_cond_32587294*M
output_shapes<
:: : : : :         ╬:         ╬: : : : : *
parallel_iterations 2
while╡
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    N  22
0TensorArrayV2Stack/TensorListStack/element_shapeЄ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:                  ╬*
element_dtype02$
"TensorArrayV2Stack/TensorListStackБ
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2Ы
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         ╬*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permп
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:                  ╬2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimex
IdentityIdentitytranspose_1:y:0^NoOp*
T0*5
_output_shapes#
!:                  ╬2

Identity}
NoOpNoOp%^lstm_cell_19/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  h: : : 2L
$lstm_cell_19/StatefulPartitionedCall$lstm_cell_19/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :                  h
 
_user_specified_nameinputs
Е&
є
while_body_32587295
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_00
while_lstm_cell_19_32587319_0:	h╕
1
while_lstm_cell_19_32587321_0:
╬╕
,
while_lstm_cell_19_32587323_0:	╕

while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor.
while_lstm_cell_19_32587319:	h╕
/
while_lstm_cell_19_32587321:
╬╕
*
while_lstm_cell_19_32587323:	╕
Ив*while/lstm_cell_19/StatefulPartitionedCall├
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    h   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape╙
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         h*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemь
*while/lstm_cell_19/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_19_32587319_0while_lstm_cell_19_32587321_0while_lstm_cell_19_32587323_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:         ╬:         ╬:         ╬*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_lstm_cell_19_layer_call_and_return_conditional_losses_325872172,
*while/lstm_cell_19/StatefulPartitionedCallў
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder3while/lstm_cell_19/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1k
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity~
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_1m
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_2Ъ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3е
while/Identity_4Identity3while/lstm_cell_19/StatefulPartitionedCall:output:1^while/NoOp*
T0*(
_output_shapes
:         ╬2
while/Identity_4е
while/Identity_5Identity3while/lstm_cell_19/StatefulPartitionedCall:output:2^while/NoOp*
T0*(
_output_shapes
:         ╬2
while/Identity_5З

while/NoOpNoOp+^while/lstm_cell_19/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"<
while_lstm_cell_19_32587319while_lstm_cell_19_32587319_0"<
while_lstm_cell_19_32587321while_lstm_cell_19_32587321_0"<
while_lstm_cell_19_32587323while_lstm_cell_19_32587323_0"0
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :         ╬:         ╬: : : : : 2X
*while/lstm_cell_19/StatefulPartitionedCall*while/lstm_cell_19/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:         ╬:.*
(
_output_shapes
:         ╬:

_output_shapes
: :

_output_shapes
: 
╫
g
H__inference_dropout_19_layer_call_and_return_conditional_losses_32590633

inputs
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  а?2
dropout/Constx
dropout/MulMulinputsdropout/Const:output:0*
T0*,
_output_shapes
:         ╬2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape╣
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:         ╬*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>2
dropout/GreaterEqual/y├
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:         ╬2
dropout/GreaterEqualД
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:         ╬2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:         ╬2
dropout/Mul_1j
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:         ╬2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ╬:T P
,
_output_shapes
:         ╬
 
_user_specified_nameinputs
Ъ?
╥
while_body_32589511
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_18_matmul_readvariableop_resource_0:	]аH
5while_lstm_cell_18_matmul_1_readvariableop_resource_0:	hаC
4while_lstm_cell_18_biasadd_readvariableop_resource_0:	а
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_18_matmul_readvariableop_resource:	]аF
3while_lstm_cell_18_matmul_1_readvariableop_resource:	hаA
2while_lstm_cell_18_biasadd_readvariableop_resource:	аИв)while/lstm_cell_18/BiasAdd/ReadVariableOpв(while/lstm_cell_18/MatMul/ReadVariableOpв*while/lstm_cell_18/MatMul_1/ReadVariableOp├
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ]   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape╙
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         ]*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem╔
(while/lstm_cell_18/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_18_matmul_readvariableop_resource_0*
_output_shapes
:	]а*
dtype02*
(while/lstm_cell_18/MatMul/ReadVariableOp╫
while/lstm_cell_18/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_18/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         а2
while/lstm_cell_18/MatMul╧
*while/lstm_cell_18/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_18_matmul_1_readvariableop_resource_0*
_output_shapes
:	hа*
dtype02,
*while/lstm_cell_18/MatMul_1/ReadVariableOp└
while/lstm_cell_18/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_18/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         а2
while/lstm_cell_18/MatMul_1╕
while/lstm_cell_18/addAddV2#while/lstm_cell_18/MatMul:product:0%while/lstm_cell_18/MatMul_1:product:0*
T0*(
_output_shapes
:         а2
while/lstm_cell_18/add╚
)while/lstm_cell_18/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_18_biasadd_readvariableop_resource_0*
_output_shapes	
:а*
dtype02+
)while/lstm_cell_18/BiasAdd/ReadVariableOp┼
while/lstm_cell_18/BiasAddBiasAddwhile/lstm_cell_18/add:z:01while/lstm_cell_18/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         а2
while/lstm_cell_18/BiasAddК
"while/lstm_cell_18/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_18/split/split_dimЛ
while/lstm_cell_18/splitSplit+while/lstm_cell_18/split/split_dim:output:0#while/lstm_cell_18/BiasAdd:output:0*
T0*`
_output_shapesN
L:         h:         h:         h:         h*
	num_split2
while/lstm_cell_18/splitШ
while/lstm_cell_18/SigmoidSigmoid!while/lstm_cell_18/split:output:0*
T0*'
_output_shapes
:         h2
while/lstm_cell_18/SigmoidЬ
while/lstm_cell_18/Sigmoid_1Sigmoid!while/lstm_cell_18/split:output:1*
T0*'
_output_shapes
:         h2
while/lstm_cell_18/Sigmoid_1а
while/lstm_cell_18/mulMul while/lstm_cell_18/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:         h2
while/lstm_cell_18/mulП
while/lstm_cell_18/ReluRelu!while/lstm_cell_18/split:output:2*
T0*'
_output_shapes
:         h2
while/lstm_cell_18/Relu┤
while/lstm_cell_18/mul_1Mulwhile/lstm_cell_18/Sigmoid:y:0%while/lstm_cell_18/Relu:activations:0*
T0*'
_output_shapes
:         h2
while/lstm_cell_18/mul_1й
while/lstm_cell_18/add_1AddV2while/lstm_cell_18/mul:z:0while/lstm_cell_18/mul_1:z:0*
T0*'
_output_shapes
:         h2
while/lstm_cell_18/add_1Ь
while/lstm_cell_18/Sigmoid_2Sigmoid!while/lstm_cell_18/split:output:3*
T0*'
_output_shapes
:         h2
while/lstm_cell_18/Sigmoid_2О
while/lstm_cell_18/Relu_1Reluwhile/lstm_cell_18/add_1:z:0*
T0*'
_output_shapes
:         h2
while/lstm_cell_18/Relu_1╕
while/lstm_cell_18/mul_2Mul while/lstm_cell_18/Sigmoid_2:y:0'while/lstm_cell_18/Relu_1:activations:0*
T0*'
_output_shapes
:         h2
while/lstm_cell_18/mul_2р
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_18/mul_2:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1k
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity~
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_1m
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_2Ъ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3Н
while/Identity_4Identitywhile/lstm_cell_18/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:         h2
while/Identity_4Н
while/Identity_5Identitywhile/lstm_cell_18/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:         h2
while/Identity_5▐

while/NoOpNoOp*^while/lstm_cell_18/BiasAdd/ReadVariableOp)^while/lstm_cell_18/MatMul/ReadVariableOp+^while/lstm_cell_18/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_18_biasadd_readvariableop_resource4while_lstm_cell_18_biasadd_readvariableop_resource_0"l
3while_lstm_cell_18_matmul_1_readvariableop_resource5while_lstm_cell_18_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_18_matmul_readvariableop_resource3while_lstm_cell_18_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         h:         h: : : : : 2V
)while/lstm_cell_18/BiasAdd/ReadVariableOp)while/lstm_cell_18/BiasAdd/ReadVariableOp2T
(while/lstm_cell_18/MatMul/ReadVariableOp(while/lstm_cell_18/MatMul/ReadVariableOp2X
*while/lstm_cell_18/MatMul_1/ReadVariableOp*while/lstm_cell_18/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         h:-)
'
_output_shapes
:         h:

_output_shapes
: :

_output_shapes
: 
Д\
Ю
E__inference_lstm_19_layer_call_and_return_conditional_losses_32587948

inputs>
+lstm_cell_19_matmul_readvariableop_resource:	h╕
A
-lstm_cell_19_matmul_1_readvariableop_resource:
╬╕
;
,lstm_cell_19_biasadd_readvariableop_resource:	╕

identityИв#lstm_cell_19/BiasAdd/ReadVariableOpв"lstm_cell_19/MatMul/ReadVariableOpв$lstm_cell_19/MatMul_1/ReadVariableOpвwhileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice]
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :╬2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessc
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :╬2
zeros/packed/1Г
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:         ╬2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :╬2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :╬2
zeros_1/packed/1Й
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const~
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:         ╬2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:         h2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ю
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1Е
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         2
TensorArrayV2/element_shape▓
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2┐
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    h   27
5TensorArrayUnstack/TensorListFromTensor/element_shape°
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2№
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         h*
shrink_axis_mask2
strided_slice_2╡
"lstm_cell_19/MatMul/ReadVariableOpReadVariableOp+lstm_cell_19_matmul_readvariableop_resource*
_output_shapes
:	h╕
*
dtype02$
"lstm_cell_19/MatMul/ReadVariableOpн
lstm_cell_19/MatMulMatMulstrided_slice_2:output:0*lstm_cell_19/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╕
2
lstm_cell_19/MatMul╝
$lstm_cell_19/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_19_matmul_1_readvariableop_resource* 
_output_shapes
:
╬╕
*
dtype02&
$lstm_cell_19/MatMul_1/ReadVariableOpй
lstm_cell_19/MatMul_1MatMulzeros:output:0,lstm_cell_19/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╕
2
lstm_cell_19/MatMul_1а
lstm_cell_19/addAddV2lstm_cell_19/MatMul:product:0lstm_cell_19/MatMul_1:product:0*
T0*(
_output_shapes
:         ╕
2
lstm_cell_19/add┤
#lstm_cell_19/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_19_biasadd_readvariableop_resource*
_output_shapes	
:╕
*
dtype02%
#lstm_cell_19/BiasAdd/ReadVariableOpн
lstm_cell_19/BiasAddBiasAddlstm_cell_19/add:z:0+lstm_cell_19/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╕
2
lstm_cell_19/BiasAdd~
lstm_cell_19/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_19/split/split_dimў
lstm_cell_19/splitSplit%lstm_cell_19/split/split_dim:output:0lstm_cell_19/BiasAdd:output:0*
T0*d
_output_shapesR
P:         ╬:         ╬:         ╬:         ╬*
	num_split2
lstm_cell_19/splitЗ
lstm_cell_19/SigmoidSigmoidlstm_cell_19/split:output:0*
T0*(
_output_shapes
:         ╬2
lstm_cell_19/SigmoidЛ
lstm_cell_19/Sigmoid_1Sigmoidlstm_cell_19/split:output:1*
T0*(
_output_shapes
:         ╬2
lstm_cell_19/Sigmoid_1М
lstm_cell_19/mulMullstm_cell_19/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:         ╬2
lstm_cell_19/mul~
lstm_cell_19/ReluRelulstm_cell_19/split:output:2*
T0*(
_output_shapes
:         ╬2
lstm_cell_19/ReluЭ
lstm_cell_19/mul_1Mullstm_cell_19/Sigmoid:y:0lstm_cell_19/Relu:activations:0*
T0*(
_output_shapes
:         ╬2
lstm_cell_19/mul_1Т
lstm_cell_19/add_1AddV2lstm_cell_19/mul:z:0lstm_cell_19/mul_1:z:0*
T0*(
_output_shapes
:         ╬2
lstm_cell_19/add_1Л
lstm_cell_19/Sigmoid_2Sigmoidlstm_cell_19/split:output:3*
T0*(
_output_shapes
:         ╬2
lstm_cell_19/Sigmoid_2}
lstm_cell_19/Relu_1Relulstm_cell_19/add_1:z:0*
T0*(
_output_shapes
:         ╬2
lstm_cell_19/Relu_1б
lstm_cell_19/mul_2Mullstm_cell_19/Sigmoid_2:y:0!lstm_cell_19/Relu_1:activations:0*
T0*(
_output_shapes
:         ╬2
lstm_cell_19/mul_2П
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    N  2
TensorArrayV2_1/element_shape╕
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterУ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_19_matmul_readvariableop_resource-lstm_cell_19_matmul_1_readvariableop_resource,lstm_cell_19_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :         ╬:         ╬: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_32587864*
condR
while_cond_32587863*M
output_shapes<
:: : : : :         ╬:         ╬: : : : : *
parallel_iterations 2
while╡
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    N  22
0TensorArrayV2Stack/TensorListStack/element_shapeщ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:         ╬*
element_dtype02$
"TensorArrayV2Stack/TensorListStackБ
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2Ы
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         ╬*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permж
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:         ╬2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimeo
IdentityIdentitytranspose_1:y:0^NoOp*
T0*,
_output_shapes
:         ╬2

Identity╚
NoOpNoOp$^lstm_cell_19/BiasAdd/ReadVariableOp#^lstm_cell_19/MatMul/ReadVariableOp%^lstm_cell_19/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         h: : : 2J
#lstm_cell_19/BiasAdd/ReadVariableOp#lstm_cell_19/BiasAdd/ReadVariableOp2H
"lstm_cell_19/MatMul/ReadVariableOp"lstm_cell_19/MatMul/ReadVariableOp2L
$lstm_cell_19/MatMul_1/ReadVariableOp$lstm_cell_19/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:         h
 
_user_specified_nameinputs
у
═
while_cond_32590034
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_32590034___redundant_placeholder06
2while_while_cond_32590034___redundant_placeholder16
2while_while_cond_32590034___redundant_placeholder26
2while_while_cond_32590034___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :         ╬:         ╬: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:         ╬:.*
(
_output_shapes
:         ╬:

_output_shapes
: :

_output_shapes
:
р
║
*__inference_lstm_19_layer_call_fn_32590583
inputs_0
unknown:	h╕

	unknown_0:
╬╕

	unknown_1:	╕

identityИвStatefulPartitionedCallТ
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:                  ╬*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_lstm_19_layer_call_and_return_conditional_losses_325871542
StatefulPartitionedCallЙ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:                  ╬2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  h: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :                  h
"
_user_specified_name
inputs/0
░?
╘
while_body_32590337
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_19_matmul_readvariableop_resource_0:	h╕
I
5while_lstm_cell_19_matmul_1_readvariableop_resource_0:
╬╕
C
4while_lstm_cell_19_biasadd_readvariableop_resource_0:	╕

while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_19_matmul_readvariableop_resource:	h╕
G
3while_lstm_cell_19_matmul_1_readvariableop_resource:
╬╕
A
2while_lstm_cell_19_biasadd_readvariableop_resource:	╕
Ив)while/lstm_cell_19/BiasAdd/ReadVariableOpв(while/lstm_cell_19/MatMul/ReadVariableOpв*while/lstm_cell_19/MatMul_1/ReadVariableOp├
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    h   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape╙
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         h*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem╔
(while/lstm_cell_19/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_19_matmul_readvariableop_resource_0*
_output_shapes
:	h╕
*
dtype02*
(while/lstm_cell_19/MatMul/ReadVariableOp╫
while/lstm_cell_19/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_19/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╕
2
while/lstm_cell_19/MatMul╨
*while/lstm_cell_19/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_19_matmul_1_readvariableop_resource_0* 
_output_shapes
:
╬╕
*
dtype02,
*while/lstm_cell_19/MatMul_1/ReadVariableOp└
while/lstm_cell_19/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_19/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╕
2
while/lstm_cell_19/MatMul_1╕
while/lstm_cell_19/addAddV2#while/lstm_cell_19/MatMul:product:0%while/lstm_cell_19/MatMul_1:product:0*
T0*(
_output_shapes
:         ╕
2
while/lstm_cell_19/add╚
)while/lstm_cell_19/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_19_biasadd_readvariableop_resource_0*
_output_shapes	
:╕
*
dtype02+
)while/lstm_cell_19/BiasAdd/ReadVariableOp┼
while/lstm_cell_19/BiasAddBiasAddwhile/lstm_cell_19/add:z:01while/lstm_cell_19/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╕
2
while/lstm_cell_19/BiasAddК
"while/lstm_cell_19/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_19/split/split_dimП
while/lstm_cell_19/splitSplit+while/lstm_cell_19/split/split_dim:output:0#while/lstm_cell_19/BiasAdd:output:0*
T0*d
_output_shapesR
P:         ╬:         ╬:         ╬:         ╬*
	num_split2
while/lstm_cell_19/splitЩ
while/lstm_cell_19/SigmoidSigmoid!while/lstm_cell_19/split:output:0*
T0*(
_output_shapes
:         ╬2
while/lstm_cell_19/SigmoidЭ
while/lstm_cell_19/Sigmoid_1Sigmoid!while/lstm_cell_19/split:output:1*
T0*(
_output_shapes
:         ╬2
while/lstm_cell_19/Sigmoid_1б
while/lstm_cell_19/mulMul while/lstm_cell_19/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:         ╬2
while/lstm_cell_19/mulР
while/lstm_cell_19/ReluRelu!while/lstm_cell_19/split:output:2*
T0*(
_output_shapes
:         ╬2
while/lstm_cell_19/Relu╡
while/lstm_cell_19/mul_1Mulwhile/lstm_cell_19/Sigmoid:y:0%while/lstm_cell_19/Relu:activations:0*
T0*(
_output_shapes
:         ╬2
while/lstm_cell_19/mul_1к
while/lstm_cell_19/add_1AddV2while/lstm_cell_19/mul:z:0while/lstm_cell_19/mul_1:z:0*
T0*(
_output_shapes
:         ╬2
while/lstm_cell_19/add_1Э
while/lstm_cell_19/Sigmoid_2Sigmoid!while/lstm_cell_19/split:output:3*
T0*(
_output_shapes
:         ╬2
while/lstm_cell_19/Sigmoid_2П
while/lstm_cell_19/Relu_1Reluwhile/lstm_cell_19/add_1:z:0*
T0*(
_output_shapes
:         ╬2
while/lstm_cell_19/Relu_1╣
while/lstm_cell_19/mul_2Mul while/lstm_cell_19/Sigmoid_2:y:0'while/lstm_cell_19/Relu_1:activations:0*
T0*(
_output_shapes
:         ╬2
while/lstm_cell_19/mul_2р
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_19/mul_2:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1k
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity~
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_1m
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_2Ъ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3О
while/Identity_4Identitywhile/lstm_cell_19/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:         ╬2
while/Identity_4О
while/Identity_5Identitywhile/lstm_cell_19/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:         ╬2
while/Identity_5▐

while/NoOpNoOp*^while/lstm_cell_19/BiasAdd/ReadVariableOp)^while/lstm_cell_19/MatMul/ReadVariableOp+^while/lstm_cell_19/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_19_biasadd_readvariableop_resource4while_lstm_cell_19_biasadd_readvariableop_resource_0"l
3while_lstm_cell_19_matmul_1_readvariableop_resource5while_lstm_cell_19_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_19_matmul_readvariableop_resource3while_lstm_cell_19_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :         ╬:         ╬: : : : : 2V
)while/lstm_cell_19/BiasAdd/ReadVariableOp)while/lstm_cell_19/BiasAdd/ReadVariableOp2T
(while/lstm_cell_19/MatMul/ReadVariableOp(while/lstm_cell_19/MatMul/ReadVariableOp2X
*while/lstm_cell_19/MatMul_1/ReadVariableOp*while/lstm_cell_19/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:         ╬:.*
(
_output_shapes
:         ╬:

_output_shapes
: :

_output_shapes
: 
ж\
Я
E__inference_lstm_18_layer_call_and_return_conditional_losses_32589444
inputs_0>
+lstm_cell_18_matmul_readvariableop_resource:	]а@
-lstm_cell_18_matmul_1_readvariableop_resource:	hа;
,lstm_cell_18_biasadd_readvariableop_resource:	а
identityИв#lstm_cell_18/BiasAdd/ReadVariableOpв"lstm_cell_18/MatMul/ReadVariableOpв$lstm_cell_18/MatMul_1/ReadVariableOpвwhileF
ShapeShapeinputs_0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :h2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :h2
zeros/packed/1Г
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:         h2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :h2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :h2
zeros_1/packed/1Й
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:         h2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permЕ
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :                  ]2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ю
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1Е
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         2
TensorArrayV2/element_shape▓
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2┐
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ]   27
5TensorArrayUnstack/TensorListFromTensor/element_shape°
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2№
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         ]*
shrink_axis_mask2
strided_slice_2╡
"lstm_cell_18/MatMul/ReadVariableOpReadVariableOp+lstm_cell_18_matmul_readvariableop_resource*
_output_shapes
:	]а*
dtype02$
"lstm_cell_18/MatMul/ReadVariableOpн
lstm_cell_18/MatMulMatMulstrided_slice_2:output:0*lstm_cell_18/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         а2
lstm_cell_18/MatMul╗
$lstm_cell_18/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_18_matmul_1_readvariableop_resource*
_output_shapes
:	hа*
dtype02&
$lstm_cell_18/MatMul_1/ReadVariableOpй
lstm_cell_18/MatMul_1MatMulzeros:output:0,lstm_cell_18/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         а2
lstm_cell_18/MatMul_1а
lstm_cell_18/addAddV2lstm_cell_18/MatMul:product:0lstm_cell_18/MatMul_1:product:0*
T0*(
_output_shapes
:         а2
lstm_cell_18/add┤
#lstm_cell_18/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_18_biasadd_readvariableop_resource*
_output_shapes	
:а*
dtype02%
#lstm_cell_18/BiasAdd/ReadVariableOpн
lstm_cell_18/BiasAddBiasAddlstm_cell_18/add:z:0+lstm_cell_18/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         а2
lstm_cell_18/BiasAdd~
lstm_cell_18/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_18/split/split_dimє
lstm_cell_18/splitSplit%lstm_cell_18/split/split_dim:output:0lstm_cell_18/BiasAdd:output:0*
T0*`
_output_shapesN
L:         h:         h:         h:         h*
	num_split2
lstm_cell_18/splitЖ
lstm_cell_18/SigmoidSigmoidlstm_cell_18/split:output:0*
T0*'
_output_shapes
:         h2
lstm_cell_18/SigmoidК
lstm_cell_18/Sigmoid_1Sigmoidlstm_cell_18/split:output:1*
T0*'
_output_shapes
:         h2
lstm_cell_18/Sigmoid_1Л
lstm_cell_18/mulMullstm_cell_18/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:         h2
lstm_cell_18/mul}
lstm_cell_18/ReluRelulstm_cell_18/split:output:2*
T0*'
_output_shapes
:         h2
lstm_cell_18/ReluЬ
lstm_cell_18/mul_1Mullstm_cell_18/Sigmoid:y:0lstm_cell_18/Relu:activations:0*
T0*'
_output_shapes
:         h2
lstm_cell_18/mul_1С
lstm_cell_18/add_1AddV2lstm_cell_18/mul:z:0lstm_cell_18/mul_1:z:0*
T0*'
_output_shapes
:         h2
lstm_cell_18/add_1К
lstm_cell_18/Sigmoid_2Sigmoidlstm_cell_18/split:output:3*
T0*'
_output_shapes
:         h2
lstm_cell_18/Sigmoid_2|
lstm_cell_18/Relu_1Relulstm_cell_18/add_1:z:0*
T0*'
_output_shapes
:         h2
lstm_cell_18/Relu_1а
lstm_cell_18/mul_2Mullstm_cell_18/Sigmoid_2:y:0!lstm_cell_18/Relu_1:activations:0*
T0*'
_output_shapes
:         h2
lstm_cell_18/mul_2П
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    h   2
TensorArrayV2_1/element_shape╕
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterП
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_18_matmul_readvariableop_resource-lstm_cell_18_matmul_1_readvariableop_resource,lstm_cell_18_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         h:         h: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_32589360*
condR
while_cond_32589359*K
output_shapes:
8: : : : :         h:         h: : : : : *
parallel_iterations 2
while╡
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    h   22
0TensorArrayV2Stack/TensorListStack/element_shapeё
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :                  h*
element_dtype02$
"TensorArrayV2Stack/TensorListStackБ
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2Ъ
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         h*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permо
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :                  h2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimew
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :                  h2

Identity╚
NoOpNoOp$^lstm_cell_18/BiasAdd/ReadVariableOp#^lstm_cell_18/MatMul/ReadVariableOp%^lstm_cell_18/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  ]: : : 2J
#lstm_cell_18/BiasAdd/ReadVariableOp#lstm_cell_18/BiasAdd/ReadVariableOp2H
"lstm_cell_18/MatMul/ReadVariableOp"lstm_cell_18/MatMul/ReadVariableOp2L
$lstm_cell_18/MatMul_1/ReadVariableOp$lstm_cell_18/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :                  ]
"
_user_specified_name
inputs/0
у
═
while_cond_32587084
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_32587084___redundant_placeholder06
2while_while_cond_32587084___redundant_placeholder16
2while_while_cond_32587084___redundant_placeholder26
2while_while_cond_32587084___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :         ╬:         ╬: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:         ╬:.*
(
_output_shapes
:         ╬:

_output_shapes
: :

_output_shapes
:
м
Є
J__inference_sequential_9_layer_call_and_return_conditional_losses_32588535
lstm_18_input#
lstm_18_32588513:	]а#
lstm_18_32588515:	hа
lstm_18_32588517:	а#
lstm_19_32588521:	h╕
$
lstm_19_32588523:
╬╕

lstm_19_32588525:	╕
#
dense_9_32588529:	╬
dense_9_32588531:
identityИвdense_9/StatefulPartitionedCallвlstm_18/StatefulPartitionedCallвlstm_19/StatefulPartitionedCall┤
lstm_18/StatefulPartitionedCallStatefulPartitionedCalllstm_18_inputlstm_18_32588513lstm_18_32588515lstm_18_32588517*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         h*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_lstm_18_layer_call_and_return_conditional_losses_325877832!
lstm_18/StatefulPartitionedCallВ
dropout_18/PartitionedCallPartitionedCall(lstm_18/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         h* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_dropout_18_layer_call_and_return_conditional_losses_325877962
dropout_18/PartitionedCall╦
lstm_19/StatefulPartitionedCallStatefulPartitionedCall#dropout_18/PartitionedCall:output:0lstm_19_32588521lstm_19_32588523lstm_19_32588525*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ╬*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_lstm_19_layer_call_and_return_conditional_losses_325879482!
lstm_19/StatefulPartitionedCallГ
dropout_19/PartitionedCallPartitionedCall(lstm_19/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ╬* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_dropout_19_layer_call_and_return_conditional_losses_325879612
dropout_19/PartitionedCall╢
dense_9/StatefulPartitionedCallStatefulPartitionedCall#dropout_19/PartitionedCall:output:0dense_9_32588529dense_9_32588531*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_dense_9_layer_call_and_return_conditional_losses_325879942!
dense_9/StatefulPartitionedCallЗ
IdentityIdentity(dense_9/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:         2

Identity┤
NoOpNoOp ^dense_9/StatefulPartitionedCall ^lstm_18/StatefulPartitionedCall ^lstm_19/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':         ]: : : : : : : : 2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall2B
lstm_18/StatefulPartitionedCalllstm_18/StatefulPartitionedCall2B
lstm_19/StatefulPartitionedCalllstm_19/StatefulPartitionedCall:Z V
+
_output_shapes
:         ]
'
_user_specified_namelstm_18_input
ї
Ж
J__inference_lstm_cell_18_layer_call_and_return_conditional_losses_32586441

inputs

states
states_11
matmul_readvariableop_resource:	]а3
 matmul_1_readvariableop_resource:	hа.
biasadd_readvariableop_resource:	а
identity

identity_1

identity_2ИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpвMatMul_1/ReadVariableOpО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	]а*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         а2
MatMulФ
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	hа*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         а2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:         а2
addН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:а*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         а2	
BiasAddd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim┐
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:         h:         h:         h:         h*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:         h2	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:         h2
	Sigmoid_1\
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:         h2
mulV
ReluRelusplit:output:2*
T0*'
_output_shapes
:         h2
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:         h2
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:         h2
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:         h2
	Sigmoid_2U
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:         h2
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:         h2
mul_2d
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:         h2

Identityh

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:         h2

Identity_1h

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:         h2

Identity_2Щ
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:         ]:         h:         h: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:         ]
 
_user_specified_nameinputs:OK
'
_output_shapes
:         h
 
_user_specified_namestates:OK
'
_output_shapes
:         h
 
_user_specified_namestates
ч[
Э
E__inference_lstm_18_layer_call_and_return_conditional_losses_32588413

inputs>
+lstm_cell_18_matmul_readvariableop_resource:	]а@
-lstm_cell_18_matmul_1_readvariableop_resource:	hа;
,lstm_cell_18_biasadd_readvariableop_resource:	а
identityИв#lstm_cell_18/BiasAdd/ReadVariableOpв"lstm_cell_18/MatMul/ReadVariableOpв$lstm_cell_18/MatMul_1/ReadVariableOpвwhileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :h2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :h2
zeros/packed/1Г
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:         h2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :h2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :h2
zeros_1/packed/1Й
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:         h2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:         ]2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ю
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1Е
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         2
TensorArrayV2/element_shape▓
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2┐
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ]   27
5TensorArrayUnstack/TensorListFromTensor/element_shape°
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2№
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         ]*
shrink_axis_mask2
strided_slice_2╡
"lstm_cell_18/MatMul/ReadVariableOpReadVariableOp+lstm_cell_18_matmul_readvariableop_resource*
_output_shapes
:	]а*
dtype02$
"lstm_cell_18/MatMul/ReadVariableOpн
lstm_cell_18/MatMulMatMulstrided_slice_2:output:0*lstm_cell_18/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         а2
lstm_cell_18/MatMul╗
$lstm_cell_18/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_18_matmul_1_readvariableop_resource*
_output_shapes
:	hа*
dtype02&
$lstm_cell_18/MatMul_1/ReadVariableOpй
lstm_cell_18/MatMul_1MatMulzeros:output:0,lstm_cell_18/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         а2
lstm_cell_18/MatMul_1а
lstm_cell_18/addAddV2lstm_cell_18/MatMul:product:0lstm_cell_18/MatMul_1:product:0*
T0*(
_output_shapes
:         а2
lstm_cell_18/add┤
#lstm_cell_18/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_18_biasadd_readvariableop_resource*
_output_shapes	
:а*
dtype02%
#lstm_cell_18/BiasAdd/ReadVariableOpн
lstm_cell_18/BiasAddBiasAddlstm_cell_18/add:z:0+lstm_cell_18/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         а2
lstm_cell_18/BiasAdd~
lstm_cell_18/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_18/split/split_dimє
lstm_cell_18/splitSplit%lstm_cell_18/split/split_dim:output:0lstm_cell_18/BiasAdd:output:0*
T0*`
_output_shapesN
L:         h:         h:         h:         h*
	num_split2
lstm_cell_18/splitЖ
lstm_cell_18/SigmoidSigmoidlstm_cell_18/split:output:0*
T0*'
_output_shapes
:         h2
lstm_cell_18/SigmoidК
lstm_cell_18/Sigmoid_1Sigmoidlstm_cell_18/split:output:1*
T0*'
_output_shapes
:         h2
lstm_cell_18/Sigmoid_1Л
lstm_cell_18/mulMullstm_cell_18/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:         h2
lstm_cell_18/mul}
lstm_cell_18/ReluRelulstm_cell_18/split:output:2*
T0*'
_output_shapes
:         h2
lstm_cell_18/ReluЬ
lstm_cell_18/mul_1Mullstm_cell_18/Sigmoid:y:0lstm_cell_18/Relu:activations:0*
T0*'
_output_shapes
:         h2
lstm_cell_18/mul_1С
lstm_cell_18/add_1AddV2lstm_cell_18/mul:z:0lstm_cell_18/mul_1:z:0*
T0*'
_output_shapes
:         h2
lstm_cell_18/add_1К
lstm_cell_18/Sigmoid_2Sigmoidlstm_cell_18/split:output:3*
T0*'
_output_shapes
:         h2
lstm_cell_18/Sigmoid_2|
lstm_cell_18/Relu_1Relulstm_cell_18/add_1:z:0*
T0*'
_output_shapes
:         h2
lstm_cell_18/Relu_1а
lstm_cell_18/mul_2Mullstm_cell_18/Sigmoid_2:y:0!lstm_cell_18/Relu_1:activations:0*
T0*'
_output_shapes
:         h2
lstm_cell_18/mul_2П
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    h   2
TensorArrayV2_1/element_shape╕
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterП
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_18_matmul_readvariableop_resource-lstm_cell_18_matmul_1_readvariableop_resource,lstm_cell_18_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         h:         h: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_32588329*
condR
while_cond_32588328*K
output_shapes:
8: : : : :         h:         h: : : : : *
parallel_iterations 2
while╡
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    h   22
0TensorArrayV2Stack/TensorListStack/element_shapeш
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:         h*
element_dtype02$
"TensorArrayV2Stack/TensorListStackБ
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2Ъ
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         h*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permе
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:         h2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimen
IdentityIdentitytranspose_1:y:0^NoOp*
T0*+
_output_shapes
:         h2

Identity╚
NoOpNoOp$^lstm_cell_18/BiasAdd/ReadVariableOp#^lstm_cell_18/MatMul/ReadVariableOp%^lstm_cell_18/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         ]: : : 2J
#lstm_cell_18/BiasAdd/ReadVariableOp#lstm_cell_18/BiasAdd/ReadVariableOp2H
"lstm_cell_18/MatMul/ReadVariableOp"lstm_cell_18/MatMul/ReadVariableOp2L
$lstm_cell_18/MatMul_1/ReadVariableOp$lstm_cell_18/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:         ]
 
_user_specified_nameinputs
░?
╘
while_body_32590035
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_19_matmul_readvariableop_resource_0:	h╕
I
5while_lstm_cell_19_matmul_1_readvariableop_resource_0:
╬╕
C
4while_lstm_cell_19_biasadd_readvariableop_resource_0:	╕

while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_19_matmul_readvariableop_resource:	h╕
G
3while_lstm_cell_19_matmul_1_readvariableop_resource:
╬╕
A
2while_lstm_cell_19_biasadd_readvariableop_resource:	╕
Ив)while/lstm_cell_19/BiasAdd/ReadVariableOpв(while/lstm_cell_19/MatMul/ReadVariableOpв*while/lstm_cell_19/MatMul_1/ReadVariableOp├
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    h   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape╙
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         h*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem╔
(while/lstm_cell_19/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_19_matmul_readvariableop_resource_0*
_output_shapes
:	h╕
*
dtype02*
(while/lstm_cell_19/MatMul/ReadVariableOp╫
while/lstm_cell_19/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_19/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╕
2
while/lstm_cell_19/MatMul╨
*while/lstm_cell_19/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_19_matmul_1_readvariableop_resource_0* 
_output_shapes
:
╬╕
*
dtype02,
*while/lstm_cell_19/MatMul_1/ReadVariableOp└
while/lstm_cell_19/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_19/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╕
2
while/lstm_cell_19/MatMul_1╕
while/lstm_cell_19/addAddV2#while/lstm_cell_19/MatMul:product:0%while/lstm_cell_19/MatMul_1:product:0*
T0*(
_output_shapes
:         ╕
2
while/lstm_cell_19/add╚
)while/lstm_cell_19/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_19_biasadd_readvariableop_resource_0*
_output_shapes	
:╕
*
dtype02+
)while/lstm_cell_19/BiasAdd/ReadVariableOp┼
while/lstm_cell_19/BiasAddBiasAddwhile/lstm_cell_19/add:z:01while/lstm_cell_19/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╕
2
while/lstm_cell_19/BiasAddК
"while/lstm_cell_19/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_19/split/split_dimП
while/lstm_cell_19/splitSplit+while/lstm_cell_19/split/split_dim:output:0#while/lstm_cell_19/BiasAdd:output:0*
T0*d
_output_shapesR
P:         ╬:         ╬:         ╬:         ╬*
	num_split2
while/lstm_cell_19/splitЩ
while/lstm_cell_19/SigmoidSigmoid!while/lstm_cell_19/split:output:0*
T0*(
_output_shapes
:         ╬2
while/lstm_cell_19/SigmoidЭ
while/lstm_cell_19/Sigmoid_1Sigmoid!while/lstm_cell_19/split:output:1*
T0*(
_output_shapes
:         ╬2
while/lstm_cell_19/Sigmoid_1б
while/lstm_cell_19/mulMul while/lstm_cell_19/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:         ╬2
while/lstm_cell_19/mulР
while/lstm_cell_19/ReluRelu!while/lstm_cell_19/split:output:2*
T0*(
_output_shapes
:         ╬2
while/lstm_cell_19/Relu╡
while/lstm_cell_19/mul_1Mulwhile/lstm_cell_19/Sigmoid:y:0%while/lstm_cell_19/Relu:activations:0*
T0*(
_output_shapes
:         ╬2
while/lstm_cell_19/mul_1к
while/lstm_cell_19/add_1AddV2while/lstm_cell_19/mul:z:0while/lstm_cell_19/mul_1:z:0*
T0*(
_output_shapes
:         ╬2
while/lstm_cell_19/add_1Э
while/lstm_cell_19/Sigmoid_2Sigmoid!while/lstm_cell_19/split:output:3*
T0*(
_output_shapes
:         ╬2
while/lstm_cell_19/Sigmoid_2П
while/lstm_cell_19/Relu_1Reluwhile/lstm_cell_19/add_1:z:0*
T0*(
_output_shapes
:         ╬2
while/lstm_cell_19/Relu_1╣
while/lstm_cell_19/mul_2Mul while/lstm_cell_19/Sigmoid_2:y:0'while/lstm_cell_19/Relu_1:activations:0*
T0*(
_output_shapes
:         ╬2
while/lstm_cell_19/mul_2р
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_19/mul_2:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1k
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity~
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_1m
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_2Ъ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3О
while/Identity_4Identitywhile/lstm_cell_19/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:         ╬2
while/Identity_4О
while/Identity_5Identitywhile/lstm_cell_19/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:         ╬2
while/Identity_5▐

while/NoOpNoOp*^while/lstm_cell_19/BiasAdd/ReadVariableOp)^while/lstm_cell_19/MatMul/ReadVariableOp+^while/lstm_cell_19/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_19_biasadd_readvariableop_resource4while_lstm_cell_19_biasadd_readvariableop_resource_0"l
3while_lstm_cell_19_matmul_1_readvariableop_resource5while_lstm_cell_19_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_19_matmul_readvariableop_resource3while_lstm_cell_19_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :         ╬:         ╬: : : : : 2V
)while/lstm_cell_19/BiasAdd/ReadVariableOp)while/lstm_cell_19/BiasAdd/ReadVariableOp2T
(while/lstm_cell_19/MatMul/ReadVariableOp(while/lstm_cell_19/MatMul/ReadVariableOp2X
*while/lstm_cell_19/MatMul_1/ReadVariableOp*while/lstm_cell_19/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:         ╬:.*
(
_output_shapes
:         ╬:

_output_shapes
: :

_output_shapes
: 
Ё%
▐
!__inference__traced_save_32590938
file_prefix-
)savev2_dense_9_kernel_read_readvariableop+
'savev2_dense_9_bias_read_readvariableop:
6savev2_lstm_18_lstm_cell_18_kernel_read_readvariableopD
@savev2_lstm_18_lstm_cell_18_recurrent_kernel_read_readvariableop8
4savev2_lstm_18_lstm_cell_18_bias_read_readvariableop:
6savev2_lstm_19_lstm_cell_19_kernel_read_readvariableopD
@savev2_lstm_19_lstm_cell_19_recurrent_kernel_read_readvariableop8
4savev2_lstm_19_lstm_cell_19_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop
savev2_const

identity_1ИвMergeV2CheckpointsП
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1Л
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shardж
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilenameУ
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*е
valueЫBШB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_namesв
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*-
value$B"B B B B B B B B B B B B B 2
SaveV2/shape_and_slicesЖ
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0)savev2_dense_9_kernel_read_readvariableop'savev2_dense_9_bias_read_readvariableop6savev2_lstm_18_lstm_cell_18_kernel_read_readvariableop@savev2_lstm_18_lstm_cell_18_recurrent_kernel_read_readvariableop4savev2_lstm_18_lstm_cell_18_bias_read_readvariableop6savev2_lstm_19_lstm_cell_19_kernel_read_readvariableop@savev2_lstm_19_lstm_cell_19_recurrent_kernel_read_readvariableop4savev2_lstm_19_lstm_cell_19_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
22
SaveV2║
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixesб
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: 2

Identity_1c
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"!

identity_1Identity_1:output:0*k
_input_shapesZ
X: :	╬::	]а:	hа:а:	h╕
:
╬╕
:╕
: : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	╬: 

_output_shapes
::%!

_output_shapes
:	]а:%!

_output_shapes
:	hа:!

_output_shapes	
:а:%!

_output_shapes
:	h╕
:&"
 
_output_shapes
:
╬╕
:!

_output_shapes	
:╕
:	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
╧J
╥

lstm_18_while_body_32588977,
(lstm_18_while_lstm_18_while_loop_counter2
.lstm_18_while_lstm_18_while_maximum_iterations
lstm_18_while_placeholder
lstm_18_while_placeholder_1
lstm_18_while_placeholder_2
lstm_18_while_placeholder_3+
'lstm_18_while_lstm_18_strided_slice_1_0g
clstm_18_while_tensorarrayv2read_tensorlistgetitem_lstm_18_tensorarrayunstack_tensorlistfromtensor_0N
;lstm_18_while_lstm_cell_18_matmul_readvariableop_resource_0:	]аP
=lstm_18_while_lstm_cell_18_matmul_1_readvariableop_resource_0:	hаK
<lstm_18_while_lstm_cell_18_biasadd_readvariableop_resource_0:	а
lstm_18_while_identity
lstm_18_while_identity_1
lstm_18_while_identity_2
lstm_18_while_identity_3
lstm_18_while_identity_4
lstm_18_while_identity_5)
%lstm_18_while_lstm_18_strided_slice_1e
alstm_18_while_tensorarrayv2read_tensorlistgetitem_lstm_18_tensorarrayunstack_tensorlistfromtensorL
9lstm_18_while_lstm_cell_18_matmul_readvariableop_resource:	]аN
;lstm_18_while_lstm_cell_18_matmul_1_readvariableop_resource:	hаI
:lstm_18_while_lstm_cell_18_biasadd_readvariableop_resource:	аИв1lstm_18/while/lstm_cell_18/BiasAdd/ReadVariableOpв0lstm_18/while/lstm_cell_18/MatMul/ReadVariableOpв2lstm_18/while/lstm_cell_18/MatMul_1/ReadVariableOp╙
?lstm_18/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ]   2A
?lstm_18/while/TensorArrayV2Read/TensorListGetItem/element_shapeГ
1lstm_18/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemclstm_18_while_tensorarrayv2read_tensorlistgetitem_lstm_18_tensorarrayunstack_tensorlistfromtensor_0lstm_18_while_placeholderHlstm_18/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         ]*
element_dtype023
1lstm_18/while/TensorArrayV2Read/TensorListGetItemс
0lstm_18/while/lstm_cell_18/MatMul/ReadVariableOpReadVariableOp;lstm_18_while_lstm_cell_18_matmul_readvariableop_resource_0*
_output_shapes
:	]а*
dtype022
0lstm_18/while/lstm_cell_18/MatMul/ReadVariableOpў
!lstm_18/while/lstm_cell_18/MatMulMatMul8lstm_18/while/TensorArrayV2Read/TensorListGetItem:item:08lstm_18/while/lstm_cell_18/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         а2#
!lstm_18/while/lstm_cell_18/MatMulч
2lstm_18/while/lstm_cell_18/MatMul_1/ReadVariableOpReadVariableOp=lstm_18_while_lstm_cell_18_matmul_1_readvariableop_resource_0*
_output_shapes
:	hа*
dtype024
2lstm_18/while/lstm_cell_18/MatMul_1/ReadVariableOpр
#lstm_18/while/lstm_cell_18/MatMul_1MatMullstm_18_while_placeholder_2:lstm_18/while/lstm_cell_18/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         а2%
#lstm_18/while/lstm_cell_18/MatMul_1╪
lstm_18/while/lstm_cell_18/addAddV2+lstm_18/while/lstm_cell_18/MatMul:product:0-lstm_18/while/lstm_cell_18/MatMul_1:product:0*
T0*(
_output_shapes
:         а2 
lstm_18/while/lstm_cell_18/addр
1lstm_18/while/lstm_cell_18/BiasAdd/ReadVariableOpReadVariableOp<lstm_18_while_lstm_cell_18_biasadd_readvariableop_resource_0*
_output_shapes	
:а*
dtype023
1lstm_18/while/lstm_cell_18/BiasAdd/ReadVariableOpх
"lstm_18/while/lstm_cell_18/BiasAddBiasAdd"lstm_18/while/lstm_cell_18/add:z:09lstm_18/while/lstm_cell_18/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         а2$
"lstm_18/while/lstm_cell_18/BiasAddЪ
*lstm_18/while/lstm_cell_18/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2,
*lstm_18/while/lstm_cell_18/split/split_dimл
 lstm_18/while/lstm_cell_18/splitSplit3lstm_18/while/lstm_cell_18/split/split_dim:output:0+lstm_18/while/lstm_cell_18/BiasAdd:output:0*
T0*`
_output_shapesN
L:         h:         h:         h:         h*
	num_split2"
 lstm_18/while/lstm_cell_18/split░
"lstm_18/while/lstm_cell_18/SigmoidSigmoid)lstm_18/while/lstm_cell_18/split:output:0*
T0*'
_output_shapes
:         h2$
"lstm_18/while/lstm_cell_18/Sigmoid┤
$lstm_18/while/lstm_cell_18/Sigmoid_1Sigmoid)lstm_18/while/lstm_cell_18/split:output:1*
T0*'
_output_shapes
:         h2&
$lstm_18/while/lstm_cell_18/Sigmoid_1└
lstm_18/while/lstm_cell_18/mulMul(lstm_18/while/lstm_cell_18/Sigmoid_1:y:0lstm_18_while_placeholder_3*
T0*'
_output_shapes
:         h2 
lstm_18/while/lstm_cell_18/mulз
lstm_18/while/lstm_cell_18/ReluRelu)lstm_18/while/lstm_cell_18/split:output:2*
T0*'
_output_shapes
:         h2!
lstm_18/while/lstm_cell_18/Relu╘
 lstm_18/while/lstm_cell_18/mul_1Mul&lstm_18/while/lstm_cell_18/Sigmoid:y:0-lstm_18/while/lstm_cell_18/Relu:activations:0*
T0*'
_output_shapes
:         h2"
 lstm_18/while/lstm_cell_18/mul_1╔
 lstm_18/while/lstm_cell_18/add_1AddV2"lstm_18/while/lstm_cell_18/mul:z:0$lstm_18/while/lstm_cell_18/mul_1:z:0*
T0*'
_output_shapes
:         h2"
 lstm_18/while/lstm_cell_18/add_1┤
$lstm_18/while/lstm_cell_18/Sigmoid_2Sigmoid)lstm_18/while/lstm_cell_18/split:output:3*
T0*'
_output_shapes
:         h2&
$lstm_18/while/lstm_cell_18/Sigmoid_2ж
!lstm_18/while/lstm_cell_18/Relu_1Relu$lstm_18/while/lstm_cell_18/add_1:z:0*
T0*'
_output_shapes
:         h2#
!lstm_18/while/lstm_cell_18/Relu_1╪
 lstm_18/while/lstm_cell_18/mul_2Mul(lstm_18/while/lstm_cell_18/Sigmoid_2:y:0/lstm_18/while/lstm_cell_18/Relu_1:activations:0*
T0*'
_output_shapes
:         h2"
 lstm_18/while/lstm_cell_18/mul_2И
2lstm_18/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_18_while_placeholder_1lstm_18_while_placeholder$lstm_18/while/lstm_cell_18/mul_2:z:0*
_output_shapes
: *
element_dtype024
2lstm_18/while/TensorArrayV2Write/TensorListSetIteml
lstm_18/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_18/while/add/yЙ
lstm_18/while/addAddV2lstm_18_while_placeholderlstm_18/while/add/y:output:0*
T0*
_output_shapes
: 2
lstm_18/while/addp
lstm_18/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_18/while/add_1/yЮ
lstm_18/while/add_1AddV2(lstm_18_while_lstm_18_while_loop_counterlstm_18/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lstm_18/while/add_1Л
lstm_18/while/IdentityIdentitylstm_18/while/add_1:z:0^lstm_18/while/NoOp*
T0*
_output_shapes
: 2
lstm_18/while/Identityж
lstm_18/while/Identity_1Identity.lstm_18_while_lstm_18_while_maximum_iterations^lstm_18/while/NoOp*
T0*
_output_shapes
: 2
lstm_18/while/Identity_1Н
lstm_18/while/Identity_2Identitylstm_18/while/add:z:0^lstm_18/while/NoOp*
T0*
_output_shapes
: 2
lstm_18/while/Identity_2║
lstm_18/while/Identity_3IdentityBlstm_18/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_18/while/NoOp*
T0*
_output_shapes
: 2
lstm_18/while/Identity_3н
lstm_18/while/Identity_4Identity$lstm_18/while/lstm_cell_18/mul_2:z:0^lstm_18/while/NoOp*
T0*'
_output_shapes
:         h2
lstm_18/while/Identity_4н
lstm_18/while/Identity_5Identity$lstm_18/while/lstm_cell_18/add_1:z:0^lstm_18/while/NoOp*
T0*'
_output_shapes
:         h2
lstm_18/while/Identity_5Ж
lstm_18/while/NoOpNoOp2^lstm_18/while/lstm_cell_18/BiasAdd/ReadVariableOp1^lstm_18/while/lstm_cell_18/MatMul/ReadVariableOp3^lstm_18/while/lstm_cell_18/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
lstm_18/while/NoOp"9
lstm_18_while_identitylstm_18/while/Identity:output:0"=
lstm_18_while_identity_1!lstm_18/while/Identity_1:output:0"=
lstm_18_while_identity_2!lstm_18/while/Identity_2:output:0"=
lstm_18_while_identity_3!lstm_18/while/Identity_3:output:0"=
lstm_18_while_identity_4!lstm_18/while/Identity_4:output:0"=
lstm_18_while_identity_5!lstm_18/while/Identity_5:output:0"P
%lstm_18_while_lstm_18_strided_slice_1'lstm_18_while_lstm_18_strided_slice_1_0"z
:lstm_18_while_lstm_cell_18_biasadd_readvariableop_resource<lstm_18_while_lstm_cell_18_biasadd_readvariableop_resource_0"|
;lstm_18_while_lstm_cell_18_matmul_1_readvariableop_resource=lstm_18_while_lstm_cell_18_matmul_1_readvariableop_resource_0"x
9lstm_18_while_lstm_cell_18_matmul_readvariableop_resource;lstm_18_while_lstm_cell_18_matmul_readvariableop_resource_0"╚
alstm_18_while_tensorarrayv2read_tensorlistgetitem_lstm_18_tensorarrayunstack_tensorlistfromtensorclstm_18_while_tensorarrayv2read_tensorlistgetitem_lstm_18_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         h:         h: : : : : 2f
1lstm_18/while/lstm_cell_18/BiasAdd/ReadVariableOp1lstm_18/while/lstm_cell_18/BiasAdd/ReadVariableOp2d
0lstm_18/while/lstm_cell_18/MatMul/ReadVariableOp0lstm_18/while/lstm_cell_18/MatMul/ReadVariableOp2h
2lstm_18/while/lstm_cell_18/MatMul_1/ReadVariableOp2lstm_18/while/lstm_cell_18/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         h:-)
'
_output_shapes
:         h:

_output_shapes
: :

_output_shapes
: 
├\
а
E__inference_lstm_19_layer_call_and_return_conditional_losses_32590119
inputs_0>
+lstm_cell_19_matmul_readvariableop_resource:	h╕
A
-lstm_cell_19_matmul_1_readvariableop_resource:
╬╕
;
,lstm_cell_19_biasadd_readvariableop_resource:	╕

identityИв#lstm_cell_19/BiasAdd/ReadVariableOpв"lstm_cell_19/MatMul/ReadVariableOpв$lstm_cell_19/MatMul_1/ReadVariableOpвwhileF
ShapeShapeinputs_0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice]
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :╬2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessc
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :╬2
zeros/packed/1Г
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:         ╬2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :╬2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :╬2
zeros_1/packed/1Й
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const~
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:         ╬2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permЕ
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :                  h2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ю
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1Е
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         2
TensorArrayV2/element_shape▓
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2┐
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    h   27
5TensorArrayUnstack/TensorListFromTensor/element_shape°
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2№
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         h*
shrink_axis_mask2
strided_slice_2╡
"lstm_cell_19/MatMul/ReadVariableOpReadVariableOp+lstm_cell_19_matmul_readvariableop_resource*
_output_shapes
:	h╕
*
dtype02$
"lstm_cell_19/MatMul/ReadVariableOpн
lstm_cell_19/MatMulMatMulstrided_slice_2:output:0*lstm_cell_19/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╕
2
lstm_cell_19/MatMul╝
$lstm_cell_19/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_19_matmul_1_readvariableop_resource* 
_output_shapes
:
╬╕
*
dtype02&
$lstm_cell_19/MatMul_1/ReadVariableOpй
lstm_cell_19/MatMul_1MatMulzeros:output:0,lstm_cell_19/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╕
2
lstm_cell_19/MatMul_1а
lstm_cell_19/addAddV2lstm_cell_19/MatMul:product:0lstm_cell_19/MatMul_1:product:0*
T0*(
_output_shapes
:         ╕
2
lstm_cell_19/add┤
#lstm_cell_19/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_19_biasadd_readvariableop_resource*
_output_shapes	
:╕
*
dtype02%
#lstm_cell_19/BiasAdd/ReadVariableOpн
lstm_cell_19/BiasAddBiasAddlstm_cell_19/add:z:0+lstm_cell_19/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╕
2
lstm_cell_19/BiasAdd~
lstm_cell_19/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_19/split/split_dimў
lstm_cell_19/splitSplit%lstm_cell_19/split/split_dim:output:0lstm_cell_19/BiasAdd:output:0*
T0*d
_output_shapesR
P:         ╬:         ╬:         ╬:         ╬*
	num_split2
lstm_cell_19/splitЗ
lstm_cell_19/SigmoidSigmoidlstm_cell_19/split:output:0*
T0*(
_output_shapes
:         ╬2
lstm_cell_19/SigmoidЛ
lstm_cell_19/Sigmoid_1Sigmoidlstm_cell_19/split:output:1*
T0*(
_output_shapes
:         ╬2
lstm_cell_19/Sigmoid_1М
lstm_cell_19/mulMullstm_cell_19/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:         ╬2
lstm_cell_19/mul~
lstm_cell_19/ReluRelulstm_cell_19/split:output:2*
T0*(
_output_shapes
:         ╬2
lstm_cell_19/ReluЭ
lstm_cell_19/mul_1Mullstm_cell_19/Sigmoid:y:0lstm_cell_19/Relu:activations:0*
T0*(
_output_shapes
:         ╬2
lstm_cell_19/mul_1Т
lstm_cell_19/add_1AddV2lstm_cell_19/mul:z:0lstm_cell_19/mul_1:z:0*
T0*(
_output_shapes
:         ╬2
lstm_cell_19/add_1Л
lstm_cell_19/Sigmoid_2Sigmoidlstm_cell_19/split:output:3*
T0*(
_output_shapes
:         ╬2
lstm_cell_19/Sigmoid_2}
lstm_cell_19/Relu_1Relulstm_cell_19/add_1:z:0*
T0*(
_output_shapes
:         ╬2
lstm_cell_19/Relu_1б
lstm_cell_19/mul_2Mullstm_cell_19/Sigmoid_2:y:0!lstm_cell_19/Relu_1:activations:0*
T0*(
_output_shapes
:         ╬2
lstm_cell_19/mul_2П
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    N  2
TensorArrayV2_1/element_shape╕
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterУ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_19_matmul_readvariableop_resource-lstm_cell_19_matmul_1_readvariableop_resource,lstm_cell_19_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :         ╬:         ╬: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_32590035*
condR
while_cond_32590034*M
output_shapes<
:: : : : :         ╬:         ╬: : : : : *
parallel_iterations 2
while╡
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    N  22
0TensorArrayV2Stack/TensorListStack/element_shapeЄ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:                  ╬*
element_dtype02$
"TensorArrayV2Stack/TensorListStackБ
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2Ы
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         ╬*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permп
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:                  ╬2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimex
IdentityIdentitytranspose_1:y:0^NoOp*
T0*5
_output_shapes#
!:                  ╬2

Identity╚
NoOpNoOp$^lstm_cell_19/BiasAdd/ReadVariableOp#^lstm_cell_19/MatMul/ReadVariableOp%^lstm_cell_19/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  h: : : 2J
#lstm_cell_19/BiasAdd/ReadVariableOp#lstm_cell_19/BiasAdd/ReadVariableOp2H
"lstm_cell_19/MatMul/ReadVariableOp"lstm_cell_19/MatMul/ReadVariableOp2L
$lstm_cell_19/MatMul_1/ReadVariableOp$lstm_cell_19/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :                  h
"
_user_specified_name
inputs/0
║
°
/__inference_lstm_cell_18_layer_call_fn_32590764

inputs
states_0
states_1
unknown:	]а
	unknown_0:	hа
	unknown_1:	а
identity

identity_1

identity_2ИвStatefulPartitionedCall┼
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:         h:         h:         h*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_lstm_cell_18_layer_call_and_return_conditional_losses_325864412
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         h2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:         h2

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:         h2

Identity_2h
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:         ]:         h:         h: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         ]
 
_user_specified_nameinputs:QM
'
_output_shapes
:         h
"
_user_specified_name
states/0:QM
'
_output_shapes
:         h
"
_user_specified_name
states/1
╢
╕
*__inference_lstm_19_layer_call_fn_32590616

inputs
unknown:	h╕

	unknown_0:
╬╕

	unknown_1:	╕

identityИвStatefulPartitionedCallЗ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ╬*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_lstm_19_layer_call_and_return_conditional_losses_325882172
StatefulPartitionedCallА
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:         ╬2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         h: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:         h
 
_user_specified_nameinputs
╘!
¤
E__inference_dense_9_layer_call_and_return_conditional_losses_32590674

inputs4
!tensordot_readvariableop_resource:	╬-
biasadd_readvariableop_resource:
identityИвBiasAdd/ReadVariableOpвTensordot/ReadVariableOpЧ
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes
:	╬*
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesq
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axis╤
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis╫
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/ConstА
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1И
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis░
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concatМ
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stackС
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*,
_output_shapes
:         ╬2
Tensordot/transposeЯ
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:                  2
Tensordot/ReshapeЮ
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
Tensordot/MatMulp
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis╜
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1Р
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:         2
	TensordotМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpЗ
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:         2	
BiasAdde
SoftmaxSoftmaxBiasAdd:output:0*
T0*+
_output_shapes
:         2	
Softmaxp
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*+
_output_shapes
:         2

IdentityВ
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         ╬: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:T P
,
_output_shapes
:         ╬
 
_user_specified_nameinputs
·	
╚
&__inference_signature_wrapper_32588583
lstm_18_input
unknown:	]а
	unknown_0:	hа
	unknown_1:	а
	unknown_2:	h╕

	unknown_3:
╬╕

	unknown_4:	╕

	unknown_5:	╬
	unknown_6:
identityИвStatefulPartitionedCallм
StatefulPartitionedCallStatefulPartitionedCalllstm_18_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         **
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8В *,
f'R%
#__inference__wrapped_model_325863662
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:         2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':         ]: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
+
_output_shapes
:         ]
'
_user_specified_namelstm_18_input
╧J
╥

lstm_18_while_body_32588650,
(lstm_18_while_lstm_18_while_loop_counter2
.lstm_18_while_lstm_18_while_maximum_iterations
lstm_18_while_placeholder
lstm_18_while_placeholder_1
lstm_18_while_placeholder_2
lstm_18_while_placeholder_3+
'lstm_18_while_lstm_18_strided_slice_1_0g
clstm_18_while_tensorarrayv2read_tensorlistgetitem_lstm_18_tensorarrayunstack_tensorlistfromtensor_0N
;lstm_18_while_lstm_cell_18_matmul_readvariableop_resource_0:	]аP
=lstm_18_while_lstm_cell_18_matmul_1_readvariableop_resource_0:	hаK
<lstm_18_while_lstm_cell_18_biasadd_readvariableop_resource_0:	а
lstm_18_while_identity
lstm_18_while_identity_1
lstm_18_while_identity_2
lstm_18_while_identity_3
lstm_18_while_identity_4
lstm_18_while_identity_5)
%lstm_18_while_lstm_18_strided_slice_1e
alstm_18_while_tensorarrayv2read_tensorlistgetitem_lstm_18_tensorarrayunstack_tensorlistfromtensorL
9lstm_18_while_lstm_cell_18_matmul_readvariableop_resource:	]аN
;lstm_18_while_lstm_cell_18_matmul_1_readvariableop_resource:	hаI
:lstm_18_while_lstm_cell_18_biasadd_readvariableop_resource:	аИв1lstm_18/while/lstm_cell_18/BiasAdd/ReadVariableOpв0lstm_18/while/lstm_cell_18/MatMul/ReadVariableOpв2lstm_18/while/lstm_cell_18/MatMul_1/ReadVariableOp╙
?lstm_18/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ]   2A
?lstm_18/while/TensorArrayV2Read/TensorListGetItem/element_shapeГ
1lstm_18/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemclstm_18_while_tensorarrayv2read_tensorlistgetitem_lstm_18_tensorarrayunstack_tensorlistfromtensor_0lstm_18_while_placeholderHlstm_18/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         ]*
element_dtype023
1lstm_18/while/TensorArrayV2Read/TensorListGetItemс
0lstm_18/while/lstm_cell_18/MatMul/ReadVariableOpReadVariableOp;lstm_18_while_lstm_cell_18_matmul_readvariableop_resource_0*
_output_shapes
:	]а*
dtype022
0lstm_18/while/lstm_cell_18/MatMul/ReadVariableOpў
!lstm_18/while/lstm_cell_18/MatMulMatMul8lstm_18/while/TensorArrayV2Read/TensorListGetItem:item:08lstm_18/while/lstm_cell_18/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         а2#
!lstm_18/while/lstm_cell_18/MatMulч
2lstm_18/while/lstm_cell_18/MatMul_1/ReadVariableOpReadVariableOp=lstm_18_while_lstm_cell_18_matmul_1_readvariableop_resource_0*
_output_shapes
:	hа*
dtype024
2lstm_18/while/lstm_cell_18/MatMul_1/ReadVariableOpр
#lstm_18/while/lstm_cell_18/MatMul_1MatMullstm_18_while_placeholder_2:lstm_18/while/lstm_cell_18/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         а2%
#lstm_18/while/lstm_cell_18/MatMul_1╪
lstm_18/while/lstm_cell_18/addAddV2+lstm_18/while/lstm_cell_18/MatMul:product:0-lstm_18/while/lstm_cell_18/MatMul_1:product:0*
T0*(
_output_shapes
:         а2 
lstm_18/while/lstm_cell_18/addр
1lstm_18/while/lstm_cell_18/BiasAdd/ReadVariableOpReadVariableOp<lstm_18_while_lstm_cell_18_biasadd_readvariableop_resource_0*
_output_shapes	
:а*
dtype023
1lstm_18/while/lstm_cell_18/BiasAdd/ReadVariableOpх
"lstm_18/while/lstm_cell_18/BiasAddBiasAdd"lstm_18/while/lstm_cell_18/add:z:09lstm_18/while/lstm_cell_18/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         а2$
"lstm_18/while/lstm_cell_18/BiasAddЪ
*lstm_18/while/lstm_cell_18/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2,
*lstm_18/while/lstm_cell_18/split/split_dimл
 lstm_18/while/lstm_cell_18/splitSplit3lstm_18/while/lstm_cell_18/split/split_dim:output:0+lstm_18/while/lstm_cell_18/BiasAdd:output:0*
T0*`
_output_shapesN
L:         h:         h:         h:         h*
	num_split2"
 lstm_18/while/lstm_cell_18/split░
"lstm_18/while/lstm_cell_18/SigmoidSigmoid)lstm_18/while/lstm_cell_18/split:output:0*
T0*'
_output_shapes
:         h2$
"lstm_18/while/lstm_cell_18/Sigmoid┤
$lstm_18/while/lstm_cell_18/Sigmoid_1Sigmoid)lstm_18/while/lstm_cell_18/split:output:1*
T0*'
_output_shapes
:         h2&
$lstm_18/while/lstm_cell_18/Sigmoid_1└
lstm_18/while/lstm_cell_18/mulMul(lstm_18/while/lstm_cell_18/Sigmoid_1:y:0lstm_18_while_placeholder_3*
T0*'
_output_shapes
:         h2 
lstm_18/while/lstm_cell_18/mulз
lstm_18/while/lstm_cell_18/ReluRelu)lstm_18/while/lstm_cell_18/split:output:2*
T0*'
_output_shapes
:         h2!
lstm_18/while/lstm_cell_18/Relu╘
 lstm_18/while/lstm_cell_18/mul_1Mul&lstm_18/while/lstm_cell_18/Sigmoid:y:0-lstm_18/while/lstm_cell_18/Relu:activations:0*
T0*'
_output_shapes
:         h2"
 lstm_18/while/lstm_cell_18/mul_1╔
 lstm_18/while/lstm_cell_18/add_1AddV2"lstm_18/while/lstm_cell_18/mul:z:0$lstm_18/while/lstm_cell_18/mul_1:z:0*
T0*'
_output_shapes
:         h2"
 lstm_18/while/lstm_cell_18/add_1┤
$lstm_18/while/lstm_cell_18/Sigmoid_2Sigmoid)lstm_18/while/lstm_cell_18/split:output:3*
T0*'
_output_shapes
:         h2&
$lstm_18/while/lstm_cell_18/Sigmoid_2ж
!lstm_18/while/lstm_cell_18/Relu_1Relu$lstm_18/while/lstm_cell_18/add_1:z:0*
T0*'
_output_shapes
:         h2#
!lstm_18/while/lstm_cell_18/Relu_1╪
 lstm_18/while/lstm_cell_18/mul_2Mul(lstm_18/while/lstm_cell_18/Sigmoid_2:y:0/lstm_18/while/lstm_cell_18/Relu_1:activations:0*
T0*'
_output_shapes
:         h2"
 lstm_18/while/lstm_cell_18/mul_2И
2lstm_18/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_18_while_placeholder_1lstm_18_while_placeholder$lstm_18/while/lstm_cell_18/mul_2:z:0*
_output_shapes
: *
element_dtype024
2lstm_18/while/TensorArrayV2Write/TensorListSetIteml
lstm_18/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_18/while/add/yЙ
lstm_18/while/addAddV2lstm_18_while_placeholderlstm_18/while/add/y:output:0*
T0*
_output_shapes
: 2
lstm_18/while/addp
lstm_18/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_18/while/add_1/yЮ
lstm_18/while/add_1AddV2(lstm_18_while_lstm_18_while_loop_counterlstm_18/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lstm_18/while/add_1Л
lstm_18/while/IdentityIdentitylstm_18/while/add_1:z:0^lstm_18/while/NoOp*
T0*
_output_shapes
: 2
lstm_18/while/Identityж
lstm_18/while/Identity_1Identity.lstm_18_while_lstm_18_while_maximum_iterations^lstm_18/while/NoOp*
T0*
_output_shapes
: 2
lstm_18/while/Identity_1Н
lstm_18/while/Identity_2Identitylstm_18/while/add:z:0^lstm_18/while/NoOp*
T0*
_output_shapes
: 2
lstm_18/while/Identity_2║
lstm_18/while/Identity_3IdentityBlstm_18/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_18/while/NoOp*
T0*
_output_shapes
: 2
lstm_18/while/Identity_3н
lstm_18/while/Identity_4Identity$lstm_18/while/lstm_cell_18/mul_2:z:0^lstm_18/while/NoOp*
T0*'
_output_shapes
:         h2
lstm_18/while/Identity_4н
lstm_18/while/Identity_5Identity$lstm_18/while/lstm_cell_18/add_1:z:0^lstm_18/while/NoOp*
T0*'
_output_shapes
:         h2
lstm_18/while/Identity_5Ж
lstm_18/while/NoOpNoOp2^lstm_18/while/lstm_cell_18/BiasAdd/ReadVariableOp1^lstm_18/while/lstm_cell_18/MatMul/ReadVariableOp3^lstm_18/while/lstm_cell_18/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
lstm_18/while/NoOp"9
lstm_18_while_identitylstm_18/while/Identity:output:0"=
lstm_18_while_identity_1!lstm_18/while/Identity_1:output:0"=
lstm_18_while_identity_2!lstm_18/while/Identity_2:output:0"=
lstm_18_while_identity_3!lstm_18/while/Identity_3:output:0"=
lstm_18_while_identity_4!lstm_18/while/Identity_4:output:0"=
lstm_18_while_identity_5!lstm_18/while/Identity_5:output:0"P
%lstm_18_while_lstm_18_strided_slice_1'lstm_18_while_lstm_18_strided_slice_1_0"z
:lstm_18_while_lstm_cell_18_biasadd_readvariableop_resource<lstm_18_while_lstm_cell_18_biasadd_readvariableop_resource_0"|
;lstm_18_while_lstm_cell_18_matmul_1_readvariableop_resource=lstm_18_while_lstm_cell_18_matmul_1_readvariableop_resource_0"x
9lstm_18_while_lstm_cell_18_matmul_readvariableop_resource;lstm_18_while_lstm_cell_18_matmul_readvariableop_resource_0"╚
alstm_18_while_tensorarrayv2read_tensorlistgetitem_lstm_18_tensorarrayunstack_tensorlistfromtensorclstm_18_while_tensorarrayv2read_tensorlistgetitem_lstm_18_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         h:         h: : : : : 2f
1lstm_18/while/lstm_cell_18/BiasAdd/ReadVariableOp1lstm_18/while/lstm_cell_18/BiasAdd/ReadVariableOp2d
0lstm_18/while/lstm_cell_18/MatMul/ReadVariableOp0lstm_18/while/lstm_cell_18/MatMul/ReadVariableOp2h
2lstm_18/while/lstm_cell_18/MatMul_1/ReadVariableOp2lstm_18/while/lstm_cell_18/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         h:-)
'
_output_shapes
:         h:

_output_shapes
: :

_output_shapes
: 
▀
═
while_cond_32586454
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_32586454___redundant_placeholder06
2while_while_cond_32586454___redundant_placeholder16
2while_while_cond_32586454___redundant_placeholder26
2while_while_cond_32586454___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :         h:         h: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         h:-)
'
_output_shapes
:         h:

_output_shapes
: :

_output_shapes
:
╧
g
H__inference_dropout_18_layer_call_and_return_conditional_losses_32589958

inputs
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *лкк?2
dropout/Constw
dropout/MulMulinputsdropout/Const:output:0*
T0*+
_output_shapes
:         h2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape╕
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:         h*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А>2
dropout/GreaterEqual/y┬
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:         h2
dropout/GreaterEqualГ
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:         h2
dropout/Cast~
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*+
_output_shapes
:         h2
dropout/Mul_1i
IdentityIdentitydropout/Mul_1:z:0*
T0*+
_output_shapes
:         h2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         h:S O
+
_output_shapes
:         h
 
_user_specified_nameinputs
░?
╘
while_body_32590186
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_19_matmul_readvariableop_resource_0:	h╕
I
5while_lstm_cell_19_matmul_1_readvariableop_resource_0:
╬╕
C
4while_lstm_cell_19_biasadd_readvariableop_resource_0:	╕

while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_19_matmul_readvariableop_resource:	h╕
G
3while_lstm_cell_19_matmul_1_readvariableop_resource:
╬╕
A
2while_lstm_cell_19_biasadd_readvariableop_resource:	╕
Ив)while/lstm_cell_19/BiasAdd/ReadVariableOpв(while/lstm_cell_19/MatMul/ReadVariableOpв*while/lstm_cell_19/MatMul_1/ReadVariableOp├
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    h   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape╙
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         h*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem╔
(while/lstm_cell_19/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_19_matmul_readvariableop_resource_0*
_output_shapes
:	h╕
*
dtype02*
(while/lstm_cell_19/MatMul/ReadVariableOp╫
while/lstm_cell_19/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_19/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╕
2
while/lstm_cell_19/MatMul╨
*while/lstm_cell_19/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_19_matmul_1_readvariableop_resource_0* 
_output_shapes
:
╬╕
*
dtype02,
*while/lstm_cell_19/MatMul_1/ReadVariableOp└
while/lstm_cell_19/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_19/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╕
2
while/lstm_cell_19/MatMul_1╕
while/lstm_cell_19/addAddV2#while/lstm_cell_19/MatMul:product:0%while/lstm_cell_19/MatMul_1:product:0*
T0*(
_output_shapes
:         ╕
2
while/lstm_cell_19/add╚
)while/lstm_cell_19/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_19_biasadd_readvariableop_resource_0*
_output_shapes	
:╕
*
dtype02+
)while/lstm_cell_19/BiasAdd/ReadVariableOp┼
while/lstm_cell_19/BiasAddBiasAddwhile/lstm_cell_19/add:z:01while/lstm_cell_19/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╕
2
while/lstm_cell_19/BiasAddК
"while/lstm_cell_19/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_19/split/split_dimП
while/lstm_cell_19/splitSplit+while/lstm_cell_19/split/split_dim:output:0#while/lstm_cell_19/BiasAdd:output:0*
T0*d
_output_shapesR
P:         ╬:         ╬:         ╬:         ╬*
	num_split2
while/lstm_cell_19/splitЩ
while/lstm_cell_19/SigmoidSigmoid!while/lstm_cell_19/split:output:0*
T0*(
_output_shapes
:         ╬2
while/lstm_cell_19/SigmoidЭ
while/lstm_cell_19/Sigmoid_1Sigmoid!while/lstm_cell_19/split:output:1*
T0*(
_output_shapes
:         ╬2
while/lstm_cell_19/Sigmoid_1б
while/lstm_cell_19/mulMul while/lstm_cell_19/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:         ╬2
while/lstm_cell_19/mulР
while/lstm_cell_19/ReluRelu!while/lstm_cell_19/split:output:2*
T0*(
_output_shapes
:         ╬2
while/lstm_cell_19/Relu╡
while/lstm_cell_19/mul_1Mulwhile/lstm_cell_19/Sigmoid:y:0%while/lstm_cell_19/Relu:activations:0*
T0*(
_output_shapes
:         ╬2
while/lstm_cell_19/mul_1к
while/lstm_cell_19/add_1AddV2while/lstm_cell_19/mul:z:0while/lstm_cell_19/mul_1:z:0*
T0*(
_output_shapes
:         ╬2
while/lstm_cell_19/add_1Э
while/lstm_cell_19/Sigmoid_2Sigmoid!while/lstm_cell_19/split:output:3*
T0*(
_output_shapes
:         ╬2
while/lstm_cell_19/Sigmoid_2П
while/lstm_cell_19/Relu_1Reluwhile/lstm_cell_19/add_1:z:0*
T0*(
_output_shapes
:         ╬2
while/lstm_cell_19/Relu_1╣
while/lstm_cell_19/mul_2Mul while/lstm_cell_19/Sigmoid_2:y:0'while/lstm_cell_19/Relu_1:activations:0*
T0*(
_output_shapes
:         ╬2
while/lstm_cell_19/mul_2р
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_19/mul_2:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1k
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity~
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_1m
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_2Ъ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3О
while/Identity_4Identitywhile/lstm_cell_19/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:         ╬2
while/Identity_4О
while/Identity_5Identitywhile/lstm_cell_19/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:         ╬2
while/Identity_5▐

while/NoOpNoOp*^while/lstm_cell_19/BiasAdd/ReadVariableOp)^while/lstm_cell_19/MatMul/ReadVariableOp+^while/lstm_cell_19/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_19_biasadd_readvariableop_resource4while_lstm_cell_19_biasadd_readvariableop_resource_0"l
3while_lstm_cell_19_matmul_1_readvariableop_resource5while_lstm_cell_19_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_19_matmul_readvariableop_resource3while_lstm_cell_19_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :         ╬:         ╬: : : : : 2V
)while/lstm_cell_19/BiasAdd/ReadVariableOp)while/lstm_cell_19/BiasAdd/ReadVariableOp2T
(while/lstm_cell_19/MatMul/ReadVariableOp(while/lstm_cell_19/MatMul/ReadVariableOp2X
*while/lstm_cell_19/MatMul_1/ReadVariableOp*while/lstm_cell_19/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:         ╬:.*
(
_output_shapes
:         ╬:

_output_shapes
: :

_output_shapes
: 
Л
З
J__inference_lstm_cell_19_layer_call_and_return_conditional_losses_32587071

inputs

states
states_11
matmul_readvariableop_resource:	h╕
4
 matmul_1_readvariableop_resource:
╬╕
.
biasadd_readvariableop_resource:	╕

identity

identity_1

identity_2ИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpвMatMul_1/ReadVariableOpО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	h╕
*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╕
2
MatMulХ
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
╬╕
*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╕
2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:         ╕
2
addН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:╕
*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╕
2	
BiasAddd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim├
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*d
_output_shapesR
P:         ╬:         ╬:         ╬:         ╬*
	num_split2
split`
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:         ╬2	
Sigmoidd
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:         ╬2
	Sigmoid_1]
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:         ╬2
mulW
ReluRelusplit:output:2*
T0*(
_output_shapes
:         ╬2
Relui
mul_1MulSigmoid:y:0Relu:activations:0*
T0*(
_output_shapes
:         ╬2
mul_1^
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:         ╬2
add_1d
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:         ╬2
	Sigmoid_2V
Relu_1Relu	add_1:z:0*
T0*(
_output_shapes
:         ╬2
Relu_1m
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*(
_output_shapes
:         ╬2
mul_2e
IdentityIdentity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:         ╬2

Identityi

Identity_1Identity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:         ╬2

Identity_1i

Identity_2Identity	add_1:z:0^NoOp*
T0*(
_output_shapes
:         ╬2

Identity_2Щ
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:         h:         ╬:         ╬: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:         h
 
_user_specified_nameinputs:PL
(
_output_shapes
:         ╬
 
_user_specified_namestates:PL
(
_output_shapes
:         ╬
 
_user_specified_namestates
▀
═
while_cond_32589510
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_32589510___redundant_placeholder06
2while_while_cond_32589510___redundant_placeholder16
2while_while_cond_32589510___redundant_placeholder26
2while_while_cond_32589510___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :         h:         h: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         h:-)
'
_output_shapes
:         h:

_output_shapes
: :

_output_shapes
:
▓
╖
*__inference_lstm_18_layer_call_fn_32589941

inputs
unknown:	]а
	unknown_0:	hа
	unknown_1:	а
identityИвStatefulPartitionedCallЖ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         h*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_lstm_18_layer_call_and_return_conditional_losses_325884132
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:         h2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         ]: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:         ]
 
_user_specified_nameinputs
хJ
╘

lstm_19_while_body_32588798,
(lstm_19_while_lstm_19_while_loop_counter2
.lstm_19_while_lstm_19_while_maximum_iterations
lstm_19_while_placeholder
lstm_19_while_placeholder_1
lstm_19_while_placeholder_2
lstm_19_while_placeholder_3+
'lstm_19_while_lstm_19_strided_slice_1_0g
clstm_19_while_tensorarrayv2read_tensorlistgetitem_lstm_19_tensorarrayunstack_tensorlistfromtensor_0N
;lstm_19_while_lstm_cell_19_matmul_readvariableop_resource_0:	h╕
Q
=lstm_19_while_lstm_cell_19_matmul_1_readvariableop_resource_0:
╬╕
K
<lstm_19_while_lstm_cell_19_biasadd_readvariableop_resource_0:	╕

lstm_19_while_identity
lstm_19_while_identity_1
lstm_19_while_identity_2
lstm_19_while_identity_3
lstm_19_while_identity_4
lstm_19_while_identity_5)
%lstm_19_while_lstm_19_strided_slice_1e
alstm_19_while_tensorarrayv2read_tensorlistgetitem_lstm_19_tensorarrayunstack_tensorlistfromtensorL
9lstm_19_while_lstm_cell_19_matmul_readvariableop_resource:	h╕
O
;lstm_19_while_lstm_cell_19_matmul_1_readvariableop_resource:
╬╕
I
:lstm_19_while_lstm_cell_19_biasadd_readvariableop_resource:	╕
Ив1lstm_19/while/lstm_cell_19/BiasAdd/ReadVariableOpв0lstm_19/while/lstm_cell_19/MatMul/ReadVariableOpв2lstm_19/while/lstm_cell_19/MatMul_1/ReadVariableOp╙
?lstm_19/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    h   2A
?lstm_19/while/TensorArrayV2Read/TensorListGetItem/element_shapeГ
1lstm_19/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemclstm_19_while_tensorarrayv2read_tensorlistgetitem_lstm_19_tensorarrayunstack_tensorlistfromtensor_0lstm_19_while_placeholderHlstm_19/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         h*
element_dtype023
1lstm_19/while/TensorArrayV2Read/TensorListGetItemс
0lstm_19/while/lstm_cell_19/MatMul/ReadVariableOpReadVariableOp;lstm_19_while_lstm_cell_19_matmul_readvariableop_resource_0*
_output_shapes
:	h╕
*
dtype022
0lstm_19/while/lstm_cell_19/MatMul/ReadVariableOpў
!lstm_19/while/lstm_cell_19/MatMulMatMul8lstm_19/while/TensorArrayV2Read/TensorListGetItem:item:08lstm_19/while/lstm_cell_19/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╕
2#
!lstm_19/while/lstm_cell_19/MatMulш
2lstm_19/while/lstm_cell_19/MatMul_1/ReadVariableOpReadVariableOp=lstm_19_while_lstm_cell_19_matmul_1_readvariableop_resource_0* 
_output_shapes
:
╬╕
*
dtype024
2lstm_19/while/lstm_cell_19/MatMul_1/ReadVariableOpр
#lstm_19/while/lstm_cell_19/MatMul_1MatMullstm_19_while_placeholder_2:lstm_19/while/lstm_cell_19/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╕
2%
#lstm_19/while/lstm_cell_19/MatMul_1╪
lstm_19/while/lstm_cell_19/addAddV2+lstm_19/while/lstm_cell_19/MatMul:product:0-lstm_19/while/lstm_cell_19/MatMul_1:product:0*
T0*(
_output_shapes
:         ╕
2 
lstm_19/while/lstm_cell_19/addр
1lstm_19/while/lstm_cell_19/BiasAdd/ReadVariableOpReadVariableOp<lstm_19_while_lstm_cell_19_biasadd_readvariableop_resource_0*
_output_shapes	
:╕
*
dtype023
1lstm_19/while/lstm_cell_19/BiasAdd/ReadVariableOpх
"lstm_19/while/lstm_cell_19/BiasAddBiasAdd"lstm_19/while/lstm_cell_19/add:z:09lstm_19/while/lstm_cell_19/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╕
2$
"lstm_19/while/lstm_cell_19/BiasAddЪ
*lstm_19/while/lstm_cell_19/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2,
*lstm_19/while/lstm_cell_19/split/split_dimп
 lstm_19/while/lstm_cell_19/splitSplit3lstm_19/while/lstm_cell_19/split/split_dim:output:0+lstm_19/while/lstm_cell_19/BiasAdd:output:0*
T0*d
_output_shapesR
P:         ╬:         ╬:         ╬:         ╬*
	num_split2"
 lstm_19/while/lstm_cell_19/split▒
"lstm_19/while/lstm_cell_19/SigmoidSigmoid)lstm_19/while/lstm_cell_19/split:output:0*
T0*(
_output_shapes
:         ╬2$
"lstm_19/while/lstm_cell_19/Sigmoid╡
$lstm_19/while/lstm_cell_19/Sigmoid_1Sigmoid)lstm_19/while/lstm_cell_19/split:output:1*
T0*(
_output_shapes
:         ╬2&
$lstm_19/while/lstm_cell_19/Sigmoid_1┴
lstm_19/while/lstm_cell_19/mulMul(lstm_19/while/lstm_cell_19/Sigmoid_1:y:0lstm_19_while_placeholder_3*
T0*(
_output_shapes
:         ╬2 
lstm_19/while/lstm_cell_19/mulи
lstm_19/while/lstm_cell_19/ReluRelu)lstm_19/while/lstm_cell_19/split:output:2*
T0*(
_output_shapes
:         ╬2!
lstm_19/while/lstm_cell_19/Relu╒
 lstm_19/while/lstm_cell_19/mul_1Mul&lstm_19/while/lstm_cell_19/Sigmoid:y:0-lstm_19/while/lstm_cell_19/Relu:activations:0*
T0*(
_output_shapes
:         ╬2"
 lstm_19/while/lstm_cell_19/mul_1╩
 lstm_19/while/lstm_cell_19/add_1AddV2"lstm_19/while/lstm_cell_19/mul:z:0$lstm_19/while/lstm_cell_19/mul_1:z:0*
T0*(
_output_shapes
:         ╬2"
 lstm_19/while/lstm_cell_19/add_1╡
$lstm_19/while/lstm_cell_19/Sigmoid_2Sigmoid)lstm_19/while/lstm_cell_19/split:output:3*
T0*(
_output_shapes
:         ╬2&
$lstm_19/while/lstm_cell_19/Sigmoid_2з
!lstm_19/while/lstm_cell_19/Relu_1Relu$lstm_19/while/lstm_cell_19/add_1:z:0*
T0*(
_output_shapes
:         ╬2#
!lstm_19/while/lstm_cell_19/Relu_1┘
 lstm_19/while/lstm_cell_19/mul_2Mul(lstm_19/while/lstm_cell_19/Sigmoid_2:y:0/lstm_19/while/lstm_cell_19/Relu_1:activations:0*
T0*(
_output_shapes
:         ╬2"
 lstm_19/while/lstm_cell_19/mul_2И
2lstm_19/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_19_while_placeholder_1lstm_19_while_placeholder$lstm_19/while/lstm_cell_19/mul_2:z:0*
_output_shapes
: *
element_dtype024
2lstm_19/while/TensorArrayV2Write/TensorListSetIteml
lstm_19/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_19/while/add/yЙ
lstm_19/while/addAddV2lstm_19_while_placeholderlstm_19/while/add/y:output:0*
T0*
_output_shapes
: 2
lstm_19/while/addp
lstm_19/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_19/while/add_1/yЮ
lstm_19/while/add_1AddV2(lstm_19_while_lstm_19_while_loop_counterlstm_19/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lstm_19/while/add_1Л
lstm_19/while/IdentityIdentitylstm_19/while/add_1:z:0^lstm_19/while/NoOp*
T0*
_output_shapes
: 2
lstm_19/while/Identityж
lstm_19/while/Identity_1Identity.lstm_19_while_lstm_19_while_maximum_iterations^lstm_19/while/NoOp*
T0*
_output_shapes
: 2
lstm_19/while/Identity_1Н
lstm_19/while/Identity_2Identitylstm_19/while/add:z:0^lstm_19/while/NoOp*
T0*
_output_shapes
: 2
lstm_19/while/Identity_2║
lstm_19/while/Identity_3IdentityBlstm_19/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_19/while/NoOp*
T0*
_output_shapes
: 2
lstm_19/while/Identity_3о
lstm_19/while/Identity_4Identity$lstm_19/while/lstm_cell_19/mul_2:z:0^lstm_19/while/NoOp*
T0*(
_output_shapes
:         ╬2
lstm_19/while/Identity_4о
lstm_19/while/Identity_5Identity$lstm_19/while/lstm_cell_19/add_1:z:0^lstm_19/while/NoOp*
T0*(
_output_shapes
:         ╬2
lstm_19/while/Identity_5Ж
lstm_19/while/NoOpNoOp2^lstm_19/while/lstm_cell_19/BiasAdd/ReadVariableOp1^lstm_19/while/lstm_cell_19/MatMul/ReadVariableOp3^lstm_19/while/lstm_cell_19/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
lstm_19/while/NoOp"9
lstm_19_while_identitylstm_19/while/Identity:output:0"=
lstm_19_while_identity_1!lstm_19/while/Identity_1:output:0"=
lstm_19_while_identity_2!lstm_19/while/Identity_2:output:0"=
lstm_19_while_identity_3!lstm_19/while/Identity_3:output:0"=
lstm_19_while_identity_4!lstm_19/while/Identity_4:output:0"=
lstm_19_while_identity_5!lstm_19/while/Identity_5:output:0"P
%lstm_19_while_lstm_19_strided_slice_1'lstm_19_while_lstm_19_strided_slice_1_0"z
:lstm_19_while_lstm_cell_19_biasadd_readvariableop_resource<lstm_19_while_lstm_cell_19_biasadd_readvariableop_resource_0"|
;lstm_19_while_lstm_cell_19_matmul_1_readvariableop_resource=lstm_19_while_lstm_cell_19_matmul_1_readvariableop_resource_0"x
9lstm_19_while_lstm_cell_19_matmul_readvariableop_resource;lstm_19_while_lstm_cell_19_matmul_readvariableop_resource_0"╚
alstm_19_while_tensorarrayv2read_tensorlistgetitem_lstm_19_tensorarrayunstack_tensorlistfromtensorclstm_19_while_tensorarrayv2read_tensorlistgetitem_lstm_19_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :         ╬:         ╬: : : : : 2f
1lstm_19/while/lstm_cell_19/BiasAdd/ReadVariableOp1lstm_19/while/lstm_cell_19/BiasAdd/ReadVariableOp2d
0lstm_19/while/lstm_cell_19/MatMul/ReadVariableOp0lstm_19/while/lstm_cell_19/MatMul/ReadVariableOp2h
2lstm_19/while/lstm_cell_19/MatMul_1/ReadVariableOp2lstm_19/while/lstm_cell_19/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:         ╬:.*
(
_output_shapes
:         ╬:

_output_shapes
: :

_output_shapes
: 
ї
Ж
J__inference_lstm_cell_18_layer_call_and_return_conditional_losses_32586587

inputs

states
states_11
matmul_readvariableop_resource:	]а3
 matmul_1_readvariableop_resource:	hа.
biasadd_readvariableop_resource:	а
identity

identity_1

identity_2ИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpвMatMul_1/ReadVariableOpО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	]а*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         а2
MatMulФ
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	hа*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         а2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:         а2
addН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:а*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         а2	
BiasAddd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim┐
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:         h:         h:         h:         h*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:         h2	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:         h2
	Sigmoid_1\
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:         h2
mulV
ReluRelusplit:output:2*
T0*'
_output_shapes
:         h2
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:         h2
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:         h2
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:         h2
	Sigmoid_2U
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:         h2
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:         h2
mul_2d
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:         h2

Identityh

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:         h2

Identity_1h

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:         h2

Identity_2Щ
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:         ]:         h:         h: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:         ]
 
_user_specified_nameinputs:OK
'
_output_shapes
:         h
 
_user_specified_namestates:OK
'
_output_shapes
:         h
 
_user_specified_namestates
╢
╕
*__inference_lstm_19_layer_call_fn_32590605

inputs
unknown:	h╕

	unknown_0:
╬╕

	unknown_1:	╕

identityИвStatefulPartitionedCallЗ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ╬*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_lstm_19_layer_call_and_return_conditional_losses_325879482
StatefulPartitionedCallА
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:         ╬2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         h: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:         h
 
_user_specified_nameinputs
ж\
Я
E__inference_lstm_18_layer_call_and_return_conditional_losses_32589595
inputs_0>
+lstm_cell_18_matmul_readvariableop_resource:	]а@
-lstm_cell_18_matmul_1_readvariableop_resource:	hа;
,lstm_cell_18_biasadd_readvariableop_resource:	а
identityИв#lstm_cell_18/BiasAdd/ReadVariableOpв"lstm_cell_18/MatMul/ReadVariableOpв$lstm_cell_18/MatMul_1/ReadVariableOpвwhileF
ShapeShapeinputs_0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :h2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :h2
zeros/packed/1Г
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:         h2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :h2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :h2
zeros_1/packed/1Й
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:         h2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permЕ
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :                  ]2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ю
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1Е
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         2
TensorArrayV2/element_shape▓
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2┐
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ]   27
5TensorArrayUnstack/TensorListFromTensor/element_shape°
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2№
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         ]*
shrink_axis_mask2
strided_slice_2╡
"lstm_cell_18/MatMul/ReadVariableOpReadVariableOp+lstm_cell_18_matmul_readvariableop_resource*
_output_shapes
:	]а*
dtype02$
"lstm_cell_18/MatMul/ReadVariableOpн
lstm_cell_18/MatMulMatMulstrided_slice_2:output:0*lstm_cell_18/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         а2
lstm_cell_18/MatMul╗
$lstm_cell_18/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_18_matmul_1_readvariableop_resource*
_output_shapes
:	hа*
dtype02&
$lstm_cell_18/MatMul_1/ReadVariableOpй
lstm_cell_18/MatMul_1MatMulzeros:output:0,lstm_cell_18/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         а2
lstm_cell_18/MatMul_1а
lstm_cell_18/addAddV2lstm_cell_18/MatMul:product:0lstm_cell_18/MatMul_1:product:0*
T0*(
_output_shapes
:         а2
lstm_cell_18/add┤
#lstm_cell_18/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_18_biasadd_readvariableop_resource*
_output_shapes	
:а*
dtype02%
#lstm_cell_18/BiasAdd/ReadVariableOpн
lstm_cell_18/BiasAddBiasAddlstm_cell_18/add:z:0+lstm_cell_18/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         а2
lstm_cell_18/BiasAdd~
lstm_cell_18/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_18/split/split_dimє
lstm_cell_18/splitSplit%lstm_cell_18/split/split_dim:output:0lstm_cell_18/BiasAdd:output:0*
T0*`
_output_shapesN
L:         h:         h:         h:         h*
	num_split2
lstm_cell_18/splitЖ
lstm_cell_18/SigmoidSigmoidlstm_cell_18/split:output:0*
T0*'
_output_shapes
:         h2
lstm_cell_18/SigmoidК
lstm_cell_18/Sigmoid_1Sigmoidlstm_cell_18/split:output:1*
T0*'
_output_shapes
:         h2
lstm_cell_18/Sigmoid_1Л
lstm_cell_18/mulMullstm_cell_18/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:         h2
lstm_cell_18/mul}
lstm_cell_18/ReluRelulstm_cell_18/split:output:2*
T0*'
_output_shapes
:         h2
lstm_cell_18/ReluЬ
lstm_cell_18/mul_1Mullstm_cell_18/Sigmoid:y:0lstm_cell_18/Relu:activations:0*
T0*'
_output_shapes
:         h2
lstm_cell_18/mul_1С
lstm_cell_18/add_1AddV2lstm_cell_18/mul:z:0lstm_cell_18/mul_1:z:0*
T0*'
_output_shapes
:         h2
lstm_cell_18/add_1К
lstm_cell_18/Sigmoid_2Sigmoidlstm_cell_18/split:output:3*
T0*'
_output_shapes
:         h2
lstm_cell_18/Sigmoid_2|
lstm_cell_18/Relu_1Relulstm_cell_18/add_1:z:0*
T0*'
_output_shapes
:         h2
lstm_cell_18/Relu_1а
lstm_cell_18/mul_2Mullstm_cell_18/Sigmoid_2:y:0!lstm_cell_18/Relu_1:activations:0*
T0*'
_output_shapes
:         h2
lstm_cell_18/mul_2П
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    h   2
TensorArrayV2_1/element_shape╕
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterП
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_18_matmul_readvariableop_resource-lstm_cell_18_matmul_1_readvariableop_resource,lstm_cell_18_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         h:         h: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_32589511*
condR
while_cond_32589510*K
output_shapes:
8: : : : :         h:         h: : : : : *
parallel_iterations 2
while╡
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    h   22
0TensorArrayV2Stack/TensorListStack/element_shapeё
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :                  h*
element_dtype02$
"TensorArrayV2Stack/TensorListStackБ
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2Ъ
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         h*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permо
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :                  h2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimew
IdentityIdentitytranspose_1:y:0^NoOp*
T0*4
_output_shapes"
 :                  h2

Identity╚
NoOpNoOp$^lstm_cell_18/BiasAdd/ReadVariableOp#^lstm_cell_18/MatMul/ReadVariableOp%^lstm_cell_18/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  ]: : : 2J
#lstm_cell_18/BiasAdd/ReadVariableOp#lstm_cell_18/BiasAdd/ReadVariableOp2H
"lstm_cell_18/MatMul/ReadVariableOp"lstm_cell_18/MatMul/ReadVariableOp2L
$lstm_cell_18/MatMul_1/ReadVariableOp$lstm_cell_18/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :                  ]
"
_user_specified_name
inputs/0
У
Й
J__inference_lstm_cell_19_layer_call_and_return_conditional_losses_32590813

inputs
states_0
states_11
matmul_readvariableop_resource:	h╕
4
 matmul_1_readvariableop_resource:
╬╕
.
biasadd_readvariableop_resource:	╕

identity

identity_1

identity_2ИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpвMatMul_1/ReadVariableOpО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	h╕
*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╕
2
MatMulХ
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
╬╕
*
dtype02
MatMul_1/ReadVariableOp|
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╕
2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:         ╕
2
addН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:╕
*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╕
2	
BiasAddd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim├
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*d
_output_shapesR
P:         ╬:         ╬:         ╬:         ╬*
	num_split2
split`
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:         ╬2	
Sigmoidd
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:         ╬2
	Sigmoid_1]
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:         ╬2
mulW
ReluRelusplit:output:2*
T0*(
_output_shapes
:         ╬2
Relui
mul_1MulSigmoid:y:0Relu:activations:0*
T0*(
_output_shapes
:         ╬2
mul_1^
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:         ╬2
add_1d
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:         ╬2
	Sigmoid_2V
Relu_1Relu	add_1:z:0*
T0*(
_output_shapes
:         ╬2
Relu_1m
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*(
_output_shapes
:         ╬2
mul_2e
IdentityIdentity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:         ╬2

Identityi

Identity_1Identity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:         ╬2

Identity_1i

Identity_2Identity	add_1:z:0^NoOp*
T0*(
_output_shapes
:         ╬2

Identity_2Щ
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:         h:         ╬:         ╬: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:         h
 
_user_specified_nameinputs:RN
(
_output_shapes
:         ╬
"
_user_specified_name
states/0:RN
(
_output_shapes
:         ╬
"
_user_specified_name
states/1
╫
ё
(sequential_9_lstm_18_while_cond_32586105F
Bsequential_9_lstm_18_while_sequential_9_lstm_18_while_loop_counterL
Hsequential_9_lstm_18_while_sequential_9_lstm_18_while_maximum_iterations*
&sequential_9_lstm_18_while_placeholder,
(sequential_9_lstm_18_while_placeholder_1,
(sequential_9_lstm_18_while_placeholder_2,
(sequential_9_lstm_18_while_placeholder_3H
Dsequential_9_lstm_18_while_less_sequential_9_lstm_18_strided_slice_1`
\sequential_9_lstm_18_while_sequential_9_lstm_18_while_cond_32586105___redundant_placeholder0`
\sequential_9_lstm_18_while_sequential_9_lstm_18_while_cond_32586105___redundant_placeholder1`
\sequential_9_lstm_18_while_sequential_9_lstm_18_while_cond_32586105___redundant_placeholder2`
\sequential_9_lstm_18_while_sequential_9_lstm_18_while_cond_32586105___redundant_placeholder3'
#sequential_9_lstm_18_while_identity
┘
sequential_9/lstm_18/while/LessLess&sequential_9_lstm_18_while_placeholderDsequential_9_lstm_18_while_less_sequential_9_lstm_18_strided_slice_1*
T0*
_output_shapes
: 2!
sequential_9/lstm_18/while/LessЬ
#sequential_9/lstm_18/while/IdentityIdentity#sequential_9/lstm_18/while/Less:z:0*
T0
*
_output_shapes
: 2%
#sequential_9/lstm_18/while/Identity"S
#sequential_9_lstm_18_while_identity,sequential_9/lstm_18/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :         h:         h: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         h:-)
'
_output_shapes
:         h:

_output_shapes
: :

_output_shapes
:
▀
═
while_cond_32589661
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_32589661___redundant_placeholder06
2while_while_cond_32589661___redundant_placeholder16
2while_while_cond_32589661___redundant_placeholder26
2while_while_cond_32589661___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :         h:         h: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         h:-)
'
_output_shapes
:         h:

_output_shapes
: :

_output_shapes
:
у
═
while_cond_32587294
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_32587294___redundant_placeholder06
2while_while_cond_32587294___redundant_placeholder16
2while_while_cond_32587294___redundant_placeholder26
2while_while_cond_32587294___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :         ╬:         ╬: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:         ╬:.*
(
_output_shapes
:         ╬:

_output_shapes
: :

_output_shapes
:
у
═
while_cond_32590185
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_32590185___redundant_placeholder06
2while_while_cond_32590185___redundant_placeholder16
2while_while_cond_32590185___redundant_placeholder26
2while_while_cond_32590185___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :         ╬:         ╬: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:         ╬:.*
(
_output_shapes
:         ╬:

_output_shapes
: :

_output_shapes
:
ч[
Э
E__inference_lstm_18_layer_call_and_return_conditional_losses_32587783

inputs>
+lstm_cell_18_matmul_readvariableop_resource:	]а@
-lstm_cell_18_matmul_1_readvariableop_resource:	hа;
,lstm_cell_18_biasadd_readvariableop_resource:	а
identityИв#lstm_cell_18/BiasAdd/ReadVariableOpв"lstm_cell_18/MatMul/ReadVariableOpв$lstm_cell_18/MatMul_1/ReadVariableOpвwhileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :h2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :h2
zeros/packed/1Г
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:         h2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :h2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :h2
zeros_1/packed/1Й
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:         h2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:         ]2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ю
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1Е
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         2
TensorArrayV2/element_shape▓
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2┐
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ]   27
5TensorArrayUnstack/TensorListFromTensor/element_shape°
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2№
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         ]*
shrink_axis_mask2
strided_slice_2╡
"lstm_cell_18/MatMul/ReadVariableOpReadVariableOp+lstm_cell_18_matmul_readvariableop_resource*
_output_shapes
:	]а*
dtype02$
"lstm_cell_18/MatMul/ReadVariableOpн
lstm_cell_18/MatMulMatMulstrided_slice_2:output:0*lstm_cell_18/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         а2
lstm_cell_18/MatMul╗
$lstm_cell_18/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_18_matmul_1_readvariableop_resource*
_output_shapes
:	hа*
dtype02&
$lstm_cell_18/MatMul_1/ReadVariableOpй
lstm_cell_18/MatMul_1MatMulzeros:output:0,lstm_cell_18/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         а2
lstm_cell_18/MatMul_1а
lstm_cell_18/addAddV2lstm_cell_18/MatMul:product:0lstm_cell_18/MatMul_1:product:0*
T0*(
_output_shapes
:         а2
lstm_cell_18/add┤
#lstm_cell_18/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_18_biasadd_readvariableop_resource*
_output_shapes	
:а*
dtype02%
#lstm_cell_18/BiasAdd/ReadVariableOpн
lstm_cell_18/BiasAddBiasAddlstm_cell_18/add:z:0+lstm_cell_18/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         а2
lstm_cell_18/BiasAdd~
lstm_cell_18/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_18/split/split_dimє
lstm_cell_18/splitSplit%lstm_cell_18/split/split_dim:output:0lstm_cell_18/BiasAdd:output:0*
T0*`
_output_shapesN
L:         h:         h:         h:         h*
	num_split2
lstm_cell_18/splitЖ
lstm_cell_18/SigmoidSigmoidlstm_cell_18/split:output:0*
T0*'
_output_shapes
:         h2
lstm_cell_18/SigmoidК
lstm_cell_18/Sigmoid_1Sigmoidlstm_cell_18/split:output:1*
T0*'
_output_shapes
:         h2
lstm_cell_18/Sigmoid_1Л
lstm_cell_18/mulMullstm_cell_18/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:         h2
lstm_cell_18/mul}
lstm_cell_18/ReluRelulstm_cell_18/split:output:2*
T0*'
_output_shapes
:         h2
lstm_cell_18/ReluЬ
lstm_cell_18/mul_1Mullstm_cell_18/Sigmoid:y:0lstm_cell_18/Relu:activations:0*
T0*'
_output_shapes
:         h2
lstm_cell_18/mul_1С
lstm_cell_18/add_1AddV2lstm_cell_18/mul:z:0lstm_cell_18/mul_1:z:0*
T0*'
_output_shapes
:         h2
lstm_cell_18/add_1К
lstm_cell_18/Sigmoid_2Sigmoidlstm_cell_18/split:output:3*
T0*'
_output_shapes
:         h2
lstm_cell_18/Sigmoid_2|
lstm_cell_18/Relu_1Relulstm_cell_18/add_1:z:0*
T0*'
_output_shapes
:         h2
lstm_cell_18/Relu_1а
lstm_cell_18/mul_2Mullstm_cell_18/Sigmoid_2:y:0!lstm_cell_18/Relu_1:activations:0*
T0*'
_output_shapes
:         h2
lstm_cell_18/mul_2П
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    h   2
TensorArrayV2_1/element_shape╕
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterП
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_18_matmul_readvariableop_resource-lstm_cell_18_matmul_1_readvariableop_resource,lstm_cell_18_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         h:         h: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_32587699*
condR
while_cond_32587698*K
output_shapes:
8: : : : :         h:         h: : : : : *
parallel_iterations 2
while╡
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    h   22
0TensorArrayV2Stack/TensorListStack/element_shapeш
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:         h*
element_dtype02$
"TensorArrayV2Stack/TensorListStackБ
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2Ъ
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         h*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permе
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:         h2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimen
IdentityIdentitytranspose_1:y:0^NoOp*
T0*+
_output_shapes
:         h2

Identity╚
NoOpNoOp$^lstm_cell_18/BiasAdd/ReadVariableOp#^lstm_cell_18/MatMul/ReadVariableOp%^lstm_cell_18/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         ]: : : 2J
#lstm_cell_18/BiasAdd/ReadVariableOp#lstm_cell_18/BiasAdd/ReadVariableOp2H
"lstm_cell_18/MatMul/ReadVariableOp"lstm_cell_18/MatMul/ReadVariableOp2L
$lstm_cell_18/MatMul_1/ReadVariableOp$lstm_cell_18/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:         ]
 
_user_specified_nameinputs
хJ
╘

lstm_19_while_body_32589132,
(lstm_19_while_lstm_19_while_loop_counter2
.lstm_19_while_lstm_19_while_maximum_iterations
lstm_19_while_placeholder
lstm_19_while_placeholder_1
lstm_19_while_placeholder_2
lstm_19_while_placeholder_3+
'lstm_19_while_lstm_19_strided_slice_1_0g
clstm_19_while_tensorarrayv2read_tensorlistgetitem_lstm_19_tensorarrayunstack_tensorlistfromtensor_0N
;lstm_19_while_lstm_cell_19_matmul_readvariableop_resource_0:	h╕
Q
=lstm_19_while_lstm_cell_19_matmul_1_readvariableop_resource_0:
╬╕
K
<lstm_19_while_lstm_cell_19_biasadd_readvariableop_resource_0:	╕

lstm_19_while_identity
lstm_19_while_identity_1
lstm_19_while_identity_2
lstm_19_while_identity_3
lstm_19_while_identity_4
lstm_19_while_identity_5)
%lstm_19_while_lstm_19_strided_slice_1e
alstm_19_while_tensorarrayv2read_tensorlistgetitem_lstm_19_tensorarrayunstack_tensorlistfromtensorL
9lstm_19_while_lstm_cell_19_matmul_readvariableop_resource:	h╕
O
;lstm_19_while_lstm_cell_19_matmul_1_readvariableop_resource:
╬╕
I
:lstm_19_while_lstm_cell_19_biasadd_readvariableop_resource:	╕
Ив1lstm_19/while/lstm_cell_19/BiasAdd/ReadVariableOpв0lstm_19/while/lstm_cell_19/MatMul/ReadVariableOpв2lstm_19/while/lstm_cell_19/MatMul_1/ReadVariableOp╙
?lstm_19/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    h   2A
?lstm_19/while/TensorArrayV2Read/TensorListGetItem/element_shapeГ
1lstm_19/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemclstm_19_while_tensorarrayv2read_tensorlistgetitem_lstm_19_tensorarrayunstack_tensorlistfromtensor_0lstm_19_while_placeholderHlstm_19/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         h*
element_dtype023
1lstm_19/while/TensorArrayV2Read/TensorListGetItemс
0lstm_19/while/lstm_cell_19/MatMul/ReadVariableOpReadVariableOp;lstm_19_while_lstm_cell_19_matmul_readvariableop_resource_0*
_output_shapes
:	h╕
*
dtype022
0lstm_19/while/lstm_cell_19/MatMul/ReadVariableOpў
!lstm_19/while/lstm_cell_19/MatMulMatMul8lstm_19/while/TensorArrayV2Read/TensorListGetItem:item:08lstm_19/while/lstm_cell_19/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╕
2#
!lstm_19/while/lstm_cell_19/MatMulш
2lstm_19/while/lstm_cell_19/MatMul_1/ReadVariableOpReadVariableOp=lstm_19_while_lstm_cell_19_matmul_1_readvariableop_resource_0* 
_output_shapes
:
╬╕
*
dtype024
2lstm_19/while/lstm_cell_19/MatMul_1/ReadVariableOpр
#lstm_19/while/lstm_cell_19/MatMul_1MatMullstm_19_while_placeholder_2:lstm_19/while/lstm_cell_19/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╕
2%
#lstm_19/while/lstm_cell_19/MatMul_1╪
lstm_19/while/lstm_cell_19/addAddV2+lstm_19/while/lstm_cell_19/MatMul:product:0-lstm_19/while/lstm_cell_19/MatMul_1:product:0*
T0*(
_output_shapes
:         ╕
2 
lstm_19/while/lstm_cell_19/addр
1lstm_19/while/lstm_cell_19/BiasAdd/ReadVariableOpReadVariableOp<lstm_19_while_lstm_cell_19_biasadd_readvariableop_resource_0*
_output_shapes	
:╕
*
dtype023
1lstm_19/while/lstm_cell_19/BiasAdd/ReadVariableOpх
"lstm_19/while/lstm_cell_19/BiasAddBiasAdd"lstm_19/while/lstm_cell_19/add:z:09lstm_19/while/lstm_cell_19/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╕
2$
"lstm_19/while/lstm_cell_19/BiasAddЪ
*lstm_19/while/lstm_cell_19/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2,
*lstm_19/while/lstm_cell_19/split/split_dimп
 lstm_19/while/lstm_cell_19/splitSplit3lstm_19/while/lstm_cell_19/split/split_dim:output:0+lstm_19/while/lstm_cell_19/BiasAdd:output:0*
T0*d
_output_shapesR
P:         ╬:         ╬:         ╬:         ╬*
	num_split2"
 lstm_19/while/lstm_cell_19/split▒
"lstm_19/while/lstm_cell_19/SigmoidSigmoid)lstm_19/while/lstm_cell_19/split:output:0*
T0*(
_output_shapes
:         ╬2$
"lstm_19/while/lstm_cell_19/Sigmoid╡
$lstm_19/while/lstm_cell_19/Sigmoid_1Sigmoid)lstm_19/while/lstm_cell_19/split:output:1*
T0*(
_output_shapes
:         ╬2&
$lstm_19/while/lstm_cell_19/Sigmoid_1┴
lstm_19/while/lstm_cell_19/mulMul(lstm_19/while/lstm_cell_19/Sigmoid_1:y:0lstm_19_while_placeholder_3*
T0*(
_output_shapes
:         ╬2 
lstm_19/while/lstm_cell_19/mulи
lstm_19/while/lstm_cell_19/ReluRelu)lstm_19/while/lstm_cell_19/split:output:2*
T0*(
_output_shapes
:         ╬2!
lstm_19/while/lstm_cell_19/Relu╒
 lstm_19/while/lstm_cell_19/mul_1Mul&lstm_19/while/lstm_cell_19/Sigmoid:y:0-lstm_19/while/lstm_cell_19/Relu:activations:0*
T0*(
_output_shapes
:         ╬2"
 lstm_19/while/lstm_cell_19/mul_1╩
 lstm_19/while/lstm_cell_19/add_1AddV2"lstm_19/while/lstm_cell_19/mul:z:0$lstm_19/while/lstm_cell_19/mul_1:z:0*
T0*(
_output_shapes
:         ╬2"
 lstm_19/while/lstm_cell_19/add_1╡
$lstm_19/while/lstm_cell_19/Sigmoid_2Sigmoid)lstm_19/while/lstm_cell_19/split:output:3*
T0*(
_output_shapes
:         ╬2&
$lstm_19/while/lstm_cell_19/Sigmoid_2з
!lstm_19/while/lstm_cell_19/Relu_1Relu$lstm_19/while/lstm_cell_19/add_1:z:0*
T0*(
_output_shapes
:         ╬2#
!lstm_19/while/lstm_cell_19/Relu_1┘
 lstm_19/while/lstm_cell_19/mul_2Mul(lstm_19/while/lstm_cell_19/Sigmoid_2:y:0/lstm_19/while/lstm_cell_19/Relu_1:activations:0*
T0*(
_output_shapes
:         ╬2"
 lstm_19/while/lstm_cell_19/mul_2И
2lstm_19/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_19_while_placeholder_1lstm_19_while_placeholder$lstm_19/while/lstm_cell_19/mul_2:z:0*
_output_shapes
: *
element_dtype024
2lstm_19/while/TensorArrayV2Write/TensorListSetIteml
lstm_19/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_19/while/add/yЙ
lstm_19/while/addAddV2lstm_19_while_placeholderlstm_19/while/add/y:output:0*
T0*
_output_shapes
: 2
lstm_19/while/addp
lstm_19/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_19/while/add_1/yЮ
lstm_19/while/add_1AddV2(lstm_19_while_lstm_19_while_loop_counterlstm_19/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lstm_19/while/add_1Л
lstm_19/while/IdentityIdentitylstm_19/while/add_1:z:0^lstm_19/while/NoOp*
T0*
_output_shapes
: 2
lstm_19/while/Identityж
lstm_19/while/Identity_1Identity.lstm_19_while_lstm_19_while_maximum_iterations^lstm_19/while/NoOp*
T0*
_output_shapes
: 2
lstm_19/while/Identity_1Н
lstm_19/while/Identity_2Identitylstm_19/while/add:z:0^lstm_19/while/NoOp*
T0*
_output_shapes
: 2
lstm_19/while/Identity_2║
lstm_19/while/Identity_3IdentityBlstm_19/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_19/while/NoOp*
T0*
_output_shapes
: 2
lstm_19/while/Identity_3о
lstm_19/while/Identity_4Identity$lstm_19/while/lstm_cell_19/mul_2:z:0^lstm_19/while/NoOp*
T0*(
_output_shapes
:         ╬2
lstm_19/while/Identity_4о
lstm_19/while/Identity_5Identity$lstm_19/while/lstm_cell_19/add_1:z:0^lstm_19/while/NoOp*
T0*(
_output_shapes
:         ╬2
lstm_19/while/Identity_5Ж
lstm_19/while/NoOpNoOp2^lstm_19/while/lstm_cell_19/BiasAdd/ReadVariableOp1^lstm_19/while/lstm_cell_19/MatMul/ReadVariableOp3^lstm_19/while/lstm_cell_19/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
lstm_19/while/NoOp"9
lstm_19_while_identitylstm_19/while/Identity:output:0"=
lstm_19_while_identity_1!lstm_19/while/Identity_1:output:0"=
lstm_19_while_identity_2!lstm_19/while/Identity_2:output:0"=
lstm_19_while_identity_3!lstm_19/while/Identity_3:output:0"=
lstm_19_while_identity_4!lstm_19/while/Identity_4:output:0"=
lstm_19_while_identity_5!lstm_19/while/Identity_5:output:0"P
%lstm_19_while_lstm_19_strided_slice_1'lstm_19_while_lstm_19_strided_slice_1_0"z
:lstm_19_while_lstm_cell_19_biasadd_readvariableop_resource<lstm_19_while_lstm_cell_19_biasadd_readvariableop_resource_0"|
;lstm_19_while_lstm_cell_19_matmul_1_readvariableop_resource=lstm_19_while_lstm_cell_19_matmul_1_readvariableop_resource_0"x
9lstm_19_while_lstm_cell_19_matmul_readvariableop_resource;lstm_19_while_lstm_cell_19_matmul_readvariableop_resource_0"╚
alstm_19_while_tensorarrayv2read_tensorlistgetitem_lstm_19_tensorarrayunstack_tensorlistfromtensorclstm_19_while_tensorarrayv2read_tensorlistgetitem_lstm_19_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :         ╬:         ╬: : : : : 2f
1lstm_19/while/lstm_cell_19/BiasAdd/ReadVariableOp1lstm_19/while/lstm_cell_19/BiasAdd/ReadVariableOp2d
0lstm_19/while/lstm_cell_19/MatMul/ReadVariableOp0lstm_19/while/lstm_cell_19/MatMul/ReadVariableOp2h
2lstm_19/while/lstm_cell_19/MatMul_1/ReadVariableOp2lstm_19/while/lstm_cell_19/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:         ╬:.*
(
_output_shapes
:         ╬:

_output_shapes
: :

_output_shapes
: 
Ъ?
╥
while_body_32589662
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_18_matmul_readvariableop_resource_0:	]аH
5while_lstm_cell_18_matmul_1_readvariableop_resource_0:	hаC
4while_lstm_cell_18_biasadd_readvariableop_resource_0:	а
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_18_matmul_readvariableop_resource:	]аF
3while_lstm_cell_18_matmul_1_readvariableop_resource:	hаA
2while_lstm_cell_18_biasadd_readvariableop_resource:	аИв)while/lstm_cell_18/BiasAdd/ReadVariableOpв(while/lstm_cell_18/MatMul/ReadVariableOpв*while/lstm_cell_18/MatMul_1/ReadVariableOp├
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ]   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape╙
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         ]*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem╔
(while/lstm_cell_18/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_18_matmul_readvariableop_resource_0*
_output_shapes
:	]а*
dtype02*
(while/lstm_cell_18/MatMul/ReadVariableOp╫
while/lstm_cell_18/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_18/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         а2
while/lstm_cell_18/MatMul╧
*while/lstm_cell_18/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_18_matmul_1_readvariableop_resource_0*
_output_shapes
:	hа*
dtype02,
*while/lstm_cell_18/MatMul_1/ReadVariableOp└
while/lstm_cell_18/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_18/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         а2
while/lstm_cell_18/MatMul_1╕
while/lstm_cell_18/addAddV2#while/lstm_cell_18/MatMul:product:0%while/lstm_cell_18/MatMul_1:product:0*
T0*(
_output_shapes
:         а2
while/lstm_cell_18/add╚
)while/lstm_cell_18/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_18_biasadd_readvariableop_resource_0*
_output_shapes	
:а*
dtype02+
)while/lstm_cell_18/BiasAdd/ReadVariableOp┼
while/lstm_cell_18/BiasAddBiasAddwhile/lstm_cell_18/add:z:01while/lstm_cell_18/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         а2
while/lstm_cell_18/BiasAddК
"while/lstm_cell_18/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_18/split/split_dimЛ
while/lstm_cell_18/splitSplit+while/lstm_cell_18/split/split_dim:output:0#while/lstm_cell_18/BiasAdd:output:0*
T0*`
_output_shapesN
L:         h:         h:         h:         h*
	num_split2
while/lstm_cell_18/splitШ
while/lstm_cell_18/SigmoidSigmoid!while/lstm_cell_18/split:output:0*
T0*'
_output_shapes
:         h2
while/lstm_cell_18/SigmoidЬ
while/lstm_cell_18/Sigmoid_1Sigmoid!while/lstm_cell_18/split:output:1*
T0*'
_output_shapes
:         h2
while/lstm_cell_18/Sigmoid_1а
while/lstm_cell_18/mulMul while/lstm_cell_18/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:         h2
while/lstm_cell_18/mulП
while/lstm_cell_18/ReluRelu!while/lstm_cell_18/split:output:2*
T0*'
_output_shapes
:         h2
while/lstm_cell_18/Relu┤
while/lstm_cell_18/mul_1Mulwhile/lstm_cell_18/Sigmoid:y:0%while/lstm_cell_18/Relu:activations:0*
T0*'
_output_shapes
:         h2
while/lstm_cell_18/mul_1й
while/lstm_cell_18/add_1AddV2while/lstm_cell_18/mul:z:0while/lstm_cell_18/mul_1:z:0*
T0*'
_output_shapes
:         h2
while/lstm_cell_18/add_1Ь
while/lstm_cell_18/Sigmoid_2Sigmoid!while/lstm_cell_18/split:output:3*
T0*'
_output_shapes
:         h2
while/lstm_cell_18/Sigmoid_2О
while/lstm_cell_18/Relu_1Reluwhile/lstm_cell_18/add_1:z:0*
T0*'
_output_shapes
:         h2
while/lstm_cell_18/Relu_1╕
while/lstm_cell_18/mul_2Mul while/lstm_cell_18/Sigmoid_2:y:0'while/lstm_cell_18/Relu_1:activations:0*
T0*'
_output_shapes
:         h2
while/lstm_cell_18/mul_2р
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_18/mul_2:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1k
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity~
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_1m
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_2Ъ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3Н
while/Identity_4Identitywhile/lstm_cell_18/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:         h2
while/Identity_4Н
while/Identity_5Identitywhile/lstm_cell_18/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:         h2
while/Identity_5▐

while/NoOpNoOp*^while/lstm_cell_18/BiasAdd/ReadVariableOp)^while/lstm_cell_18/MatMul/ReadVariableOp+^while/lstm_cell_18/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_18_biasadd_readvariableop_resource4while_lstm_cell_18_biasadd_readvariableop_resource_0"l
3while_lstm_cell_18_matmul_1_readvariableop_resource5while_lstm_cell_18_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_18_matmul_readvariableop_resource3while_lstm_cell_18_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         h:         h: : : : : 2V
)while/lstm_cell_18/BiasAdd/ReadVariableOp)while/lstm_cell_18/BiasAdd/ReadVariableOp2T
(while/lstm_cell_18/MatMul/ReadVariableOp(while/lstm_cell_18/MatMul/ReadVariableOp2X
*while/lstm_cell_18/MatMul_1/ReadVariableOp*while/lstm_cell_18/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         h:-)
'
_output_shapes
:         h:

_output_shapes
: :

_output_shapes
: 
▀
═
while_cond_32589812
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_32589812___redundant_placeholder06
2while_while_cond_32589812___redundant_placeholder16
2while_while_cond_32589812___redundant_placeholder26
2while_while_cond_32589812___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :         h:         h: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         h:-)
'
_output_shapes
:         h:

_output_shapes
: :

_output_shapes
:
Е
f
H__inference_dropout_18_layer_call_and_return_conditional_losses_32589946

inputs

identity_1^
IdentityIdentityinputs*
T0*+
_output_shapes
:         h2

Identitym

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:         h2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         h:S O
+
_output_shapes
:         h
 
_user_specified_nameinputs
ч[
Э
E__inference_lstm_18_layer_call_and_return_conditional_losses_32589897

inputs>
+lstm_cell_18_matmul_readvariableop_resource:	]а@
-lstm_cell_18_matmul_1_readvariableop_resource:	hа;
,lstm_cell_18_biasadd_readvariableop_resource:	а
identityИв#lstm_cell_18/BiasAdd/ReadVariableOpв"lstm_cell_18/MatMul/ReadVariableOpв$lstm_cell_18/MatMul_1/ReadVariableOpвwhileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :h2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :h2
zeros/packed/1Г
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:         h2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :h2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :h2
zeros_1/packed/1Й
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:         h2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:         ]2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ю
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1Е
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         2
TensorArrayV2/element_shape▓
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2┐
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ]   27
5TensorArrayUnstack/TensorListFromTensor/element_shape°
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2№
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         ]*
shrink_axis_mask2
strided_slice_2╡
"lstm_cell_18/MatMul/ReadVariableOpReadVariableOp+lstm_cell_18_matmul_readvariableop_resource*
_output_shapes
:	]а*
dtype02$
"lstm_cell_18/MatMul/ReadVariableOpн
lstm_cell_18/MatMulMatMulstrided_slice_2:output:0*lstm_cell_18/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         а2
lstm_cell_18/MatMul╗
$lstm_cell_18/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_18_matmul_1_readvariableop_resource*
_output_shapes
:	hа*
dtype02&
$lstm_cell_18/MatMul_1/ReadVariableOpй
lstm_cell_18/MatMul_1MatMulzeros:output:0,lstm_cell_18/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         а2
lstm_cell_18/MatMul_1а
lstm_cell_18/addAddV2lstm_cell_18/MatMul:product:0lstm_cell_18/MatMul_1:product:0*
T0*(
_output_shapes
:         а2
lstm_cell_18/add┤
#lstm_cell_18/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_18_biasadd_readvariableop_resource*
_output_shapes	
:а*
dtype02%
#lstm_cell_18/BiasAdd/ReadVariableOpн
lstm_cell_18/BiasAddBiasAddlstm_cell_18/add:z:0+lstm_cell_18/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         а2
lstm_cell_18/BiasAdd~
lstm_cell_18/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_18/split/split_dimє
lstm_cell_18/splitSplit%lstm_cell_18/split/split_dim:output:0lstm_cell_18/BiasAdd:output:0*
T0*`
_output_shapesN
L:         h:         h:         h:         h*
	num_split2
lstm_cell_18/splitЖ
lstm_cell_18/SigmoidSigmoidlstm_cell_18/split:output:0*
T0*'
_output_shapes
:         h2
lstm_cell_18/SigmoidК
lstm_cell_18/Sigmoid_1Sigmoidlstm_cell_18/split:output:1*
T0*'
_output_shapes
:         h2
lstm_cell_18/Sigmoid_1Л
lstm_cell_18/mulMullstm_cell_18/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:         h2
lstm_cell_18/mul}
lstm_cell_18/ReluRelulstm_cell_18/split:output:2*
T0*'
_output_shapes
:         h2
lstm_cell_18/ReluЬ
lstm_cell_18/mul_1Mullstm_cell_18/Sigmoid:y:0lstm_cell_18/Relu:activations:0*
T0*'
_output_shapes
:         h2
lstm_cell_18/mul_1С
lstm_cell_18/add_1AddV2lstm_cell_18/mul:z:0lstm_cell_18/mul_1:z:0*
T0*'
_output_shapes
:         h2
lstm_cell_18/add_1К
lstm_cell_18/Sigmoid_2Sigmoidlstm_cell_18/split:output:3*
T0*'
_output_shapes
:         h2
lstm_cell_18/Sigmoid_2|
lstm_cell_18/Relu_1Relulstm_cell_18/add_1:z:0*
T0*'
_output_shapes
:         h2
lstm_cell_18/Relu_1а
lstm_cell_18/mul_2Mullstm_cell_18/Sigmoid_2:y:0!lstm_cell_18/Relu_1:activations:0*
T0*'
_output_shapes
:         h2
lstm_cell_18/mul_2П
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    h   2
TensorArrayV2_1/element_shape╕
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterП
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_18_matmul_readvariableop_resource-lstm_cell_18_matmul_1_readvariableop_resource,lstm_cell_18_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         h:         h: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_32589813*
condR
while_cond_32589812*K
output_shapes:
8: : : : :         h:         h: : : : : *
parallel_iterations 2
while╡
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    h   22
0TensorArrayV2Stack/TensorListStack/element_shapeш
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:         h*
element_dtype02$
"TensorArrayV2Stack/TensorListStackБ
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2Ъ
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         h*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/permе
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:         h2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimen
IdentityIdentitytranspose_1:y:0^NoOp*
T0*+
_output_shapes
:         h2

Identity╚
NoOpNoOp$^lstm_cell_18/BiasAdd/ReadVariableOp#^lstm_cell_18/MatMul/ReadVariableOp%^lstm_cell_18/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         ]: : : 2J
#lstm_cell_18/BiasAdd/ReadVariableOp#lstm_cell_18/BiasAdd/ReadVariableOp2H
"lstm_cell_18/MatMul/ReadVariableOp"lstm_cell_18/MatMul/ReadVariableOp2L
$lstm_cell_18/MatMul_1/ReadVariableOp$lstm_cell_18/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:         ]
 
_user_specified_nameinputs
р
║
*__inference_lstm_19_layer_call_fn_32590594
inputs_0
unknown:	h╕

	unknown_0:
╬╕

	unknown_1:	╕

identityИвStatefulPartitionedCallТ
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:                  ╬*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_lstm_19_layer_call_and_return_conditional_losses_325873642
StatefulPartitionedCallЙ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:                  ╬2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  h: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :                  h
"
_user_specified_name
inputs/0
·%
ё
while_body_32586455
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_00
while_lstm_cell_18_32586479_0:	]а0
while_lstm_cell_18_32586481_0:	hа,
while_lstm_cell_18_32586483_0:	а
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor.
while_lstm_cell_18_32586479:	]а.
while_lstm_cell_18_32586481:	hа*
while_lstm_cell_18_32586483:	аИв*while/lstm_cell_18/StatefulPartitionedCall├
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ]   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape╙
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         ]*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemщ
*while/lstm_cell_18/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_18_32586479_0while_lstm_cell_18_32586481_0while_lstm_cell_18_32586483_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:         h:         h:         h*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_lstm_cell_18_layer_call_and_return_conditional_losses_325864412,
*while/lstm_cell_18/StatefulPartitionedCallў
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder3while/lstm_cell_18/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1k
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity~
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_1m
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_2Ъ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3д
while/Identity_4Identity3while/lstm_cell_18/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:         h2
while/Identity_4д
while/Identity_5Identity3while/lstm_cell_18/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:         h2
while/Identity_5З

while/NoOpNoOp+^while/lstm_cell_18/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"<
while_lstm_cell_18_32586479while_lstm_cell_18_32586479_0"<
while_lstm_cell_18_32586481while_lstm_cell_18_32586481_0"<
while_lstm_cell_18_32586483while_lstm_cell_18_32586483_0"0
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         h:         h: : : : : 2X
*while/lstm_cell_18/StatefulPartitionedCall*while/lstm_cell_18/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         h:-)
'
_output_shapes
:         h:

_output_shapes
: :

_output_shapes
: 
╟
∙
/__inference_lstm_cell_19_layer_call_fn_32590879

inputs
states_0
states_1
unknown:	h╕

	unknown_0:
╬╕

	unknown_1:	╕

identity

identity_1

identity_2ИвStatefulPartitionedCall╚
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:         ╬:         ╬:         ╬*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_lstm_cell_19_layer_call_and_return_conditional_losses_325872172
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         ╬2

IdentityА

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:         ╬2

Identity_1А

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:         ╬2

Identity_2h
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:         h:         ╬:         ╬: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         h
 
_user_specified_nameinputs:RN
(
_output_shapes
:         ╬
"
_user_specified_name
states/0:RN
(
_output_shapes
:         ╬
"
_user_specified_name
states/1
Х

╩
/__inference_sequential_9_layer_call_fn_32589272

inputs
unknown:	]а
	unknown_0:	hа
	unknown_1:	а
	unknown_2:	h╕

	unknown_3:
╬╕

	unknown_4:	╕

	unknown_5:	╬
	unknown_6:
identityИвStatefulPartitionedCall╠
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         **
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_sequential_9_layer_call_and_return_conditional_losses_325880012
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:         2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':         ]: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:         ]
 
_user_specified_nameinputs
╢
f
-__inference_dropout_18_layer_call_fn_32589968

inputs
identityИвStatefulPartitionedCallт
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         h* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_dropout_18_layer_call_and_return_conditional_losses_325882462
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:         h2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         h22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:         h
 
_user_specified_nameinputs
║
°
/__inference_lstm_cell_18_layer_call_fn_32590781

inputs
states_0
states_1
unknown:	]а
	unknown_0:	hа
	unknown_1:	а
identity

identity_1

identity_2ИвStatefulPartitionedCall┼
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:         h:         h:         h*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_lstm_cell_18_layer_call_and_return_conditional_losses_325865872
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         h2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:         h2

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:         h2

Identity_2h
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:         ]:         h:         h: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         ]
 
_user_specified_nameinputs:QM
'
_output_shapes
:         h
"
_user_specified_name
states/0:QM
'
_output_shapes
:         h
"
_user_specified_name
states/1
╧
g
H__inference_dropout_18_layer_call_and_return_conditional_losses_32588246

inputs
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *лкк?2
dropout/Constw
dropout/MulMulinputsdropout/Const:output:0*
T0*+
_output_shapes
:         h2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape╕
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:         h*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А>2
dropout/GreaterEqual/y┬
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:         h2
dropout/GreaterEqualГ
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:         h2
dropout/Cast~
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*+
_output_shapes
:         h2
dropout/Mul_1i
IdentityIdentitydropout/Mul_1:z:0*
T0*+
_output_shapes
:         h2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         h:S O
+
_output_shapes
:         h
 
_user_specified_nameinputs
▌
╣
*__inference_lstm_18_layer_call_fn_32589919
inputs_0
unknown:	]а
	unknown_0:	hа
	unknown_1:	а
identityИвStatefulPartitionedCallС
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :                  h*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_lstm_18_layer_call_and_return_conditional_losses_325867342
StatefulPartitionedCallИ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :                  h2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  ]: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :                  ]
"
_user_specified_name
inputs/0
▀
═
while_cond_32588328
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_32588328___redundant_placeholder06
2while_while_cond_32588328___redundant_placeholder16
2while_while_cond_32588328___redundant_placeholder26
2while_while_cond_32588328___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :         h:         h: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         h:-)
'
_output_shapes
:         h:

_output_shapes
: :

_output_shapes
:
╘

э
lstm_19_while_cond_32589131,
(lstm_19_while_lstm_19_while_loop_counter2
.lstm_19_while_lstm_19_while_maximum_iterations
lstm_19_while_placeholder
lstm_19_while_placeholder_1
lstm_19_while_placeholder_2
lstm_19_while_placeholder_3.
*lstm_19_while_less_lstm_19_strided_slice_1F
Blstm_19_while_lstm_19_while_cond_32589131___redundant_placeholder0F
Blstm_19_while_lstm_19_while_cond_32589131___redundant_placeholder1F
Blstm_19_while_lstm_19_while_cond_32589131___redundant_placeholder2F
Blstm_19_while_lstm_19_while_cond_32589131___redundant_placeholder3
lstm_19_while_identity
Ш
lstm_19/while/LessLesslstm_19_while_placeholder*lstm_19_while_less_lstm_19_strided_slice_1*
T0*
_output_shapes
: 2
lstm_19/while/Lessu
lstm_19/while/IdentityIdentitylstm_19/while/Less:z:0*
T0
*
_output_shapes
: 2
lstm_19/while/Identity"9
lstm_19_while_identitylstm_19/while/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :         ╬:         ╬: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:         ╬:.*
(
_output_shapes
:         ╬:

_output_shapes
: :

_output_shapes
:
Х

╩
/__inference_sequential_9_layer_call_fn_32589293

inputs
unknown:	]а
	unknown_0:	hа
	unknown_1:	а
	unknown_2:	h╕

	unknown_3:
╬╕

	unknown_4:	╕

	unknown_5:	╬
	unknown_6:
identityИвStatefulPartitionedCall╠
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         **
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_sequential_9_layer_call_and_return_conditional_losses_325884702
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:         2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':         ]: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:         ]
 
_user_specified_nameinputs
¤
И
J__inference_lstm_cell_18_layer_call_and_return_conditional_losses_32590715

inputs
states_0
states_11
matmul_readvariableop_resource:	]а3
 matmul_1_readvariableop_resource:	hа.
biasadd_readvariableop_resource:	а
identity

identity_1

identity_2ИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpвMatMul_1/ReadVariableOpО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	]а*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         а2
MatMulФ
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	hа*
dtype02
MatMul_1/ReadVariableOp|
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         а2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:         а2
addН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:а*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         а2	
BiasAddd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim┐
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:         h:         h:         h:         h*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:         h2	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:         h2
	Sigmoid_1\
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:         h2
mulV
ReluRelusplit:output:2*
T0*'
_output_shapes
:         h2
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:         h2
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:         h2
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:         h2
	Sigmoid_2U
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:         h2
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:         h2
mul_2d
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:         h2

Identityh

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:         h2

Identity_1h

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:         h2

Identity_2Щ
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:         ]:         h:         h: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:         ]
 
_user_specified_nameinputs:QM
'
_output_shapes
:         h
"
_user_specified_name
states/0:QM
'
_output_shapes
:         h
"
_user_specified_name
states/1
█
ё
(sequential_9_lstm_19_while_cond_32586253F
Bsequential_9_lstm_19_while_sequential_9_lstm_19_while_loop_counterL
Hsequential_9_lstm_19_while_sequential_9_lstm_19_while_maximum_iterations*
&sequential_9_lstm_19_while_placeholder,
(sequential_9_lstm_19_while_placeholder_1,
(sequential_9_lstm_19_while_placeholder_2,
(sequential_9_lstm_19_while_placeholder_3H
Dsequential_9_lstm_19_while_less_sequential_9_lstm_19_strided_slice_1`
\sequential_9_lstm_19_while_sequential_9_lstm_19_while_cond_32586253___redundant_placeholder0`
\sequential_9_lstm_19_while_sequential_9_lstm_19_while_cond_32586253___redundant_placeholder1`
\sequential_9_lstm_19_while_sequential_9_lstm_19_while_cond_32586253___redundant_placeholder2`
\sequential_9_lstm_19_while_sequential_9_lstm_19_while_cond_32586253___redundant_placeholder3'
#sequential_9_lstm_19_while_identity
┘
sequential_9/lstm_19/while/LessLess&sequential_9_lstm_19_while_placeholderDsequential_9_lstm_19_while_less_sequential_9_lstm_19_strided_slice_1*
T0*
_output_shapes
: 2!
sequential_9/lstm_19/while/LessЬ
#sequential_9/lstm_19/while/IdentityIdentity#sequential_9/lstm_19/while/Less:z:0*
T0
*
_output_shapes
: 2%
#sequential_9/lstm_19/while/Identity"S
#sequential_9_lstm_19_while_identity,sequential_9/lstm_19/while/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :         ╬:         ╬: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:         ╬:.*
(
_output_shapes
:         ╬:

_output_shapes
: :

_output_shapes
:
Ж
Ш
*__inference_dense_9_layer_call_fn_32590683

inputs
unknown:	╬
	unknown_0:
identityИвStatefulPartitionedCall∙
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_dense_9_layer_call_and_return_conditional_losses_325879942
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:         2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         ╬: : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:         ╬
 
_user_specified_nameinputs"иL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*╛
serving_defaultк
K
lstm_18_input:
serving_default_lstm_18_input:0         ]?
dense_94
StatefulPartitionedCall:0         tensorflow/serving/predict:╛▓
ї
layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
	optimizer
trainable_variables
	variables
	regularization_losses

	keras_api

signatures
k_default_save_signature
*l&call_and_return_all_conditional_losses
m__call__"
_tf_keras_sequential
├
cell

state_spec
trainable_variables
	variables
regularization_losses
	keras_api
*n&call_and_return_all_conditional_losses
o__call__"
_tf_keras_rnn_layer
е
trainable_variables
	variables
regularization_losses
	keras_api
*p&call_and_return_all_conditional_losses
q__call__"
_tf_keras_layer
├
cell

state_spec
trainable_variables
	variables
regularization_losses
	keras_api
*r&call_and_return_all_conditional_losses
s__call__"
_tf_keras_rnn_layer
е
trainable_variables
	variables
regularization_losses
	keras_api
*t&call_and_return_all_conditional_losses
u__call__"
_tf_keras_layer
╗

 kernel
!bias
"trainable_variables
#	variables
$regularization_losses
%	keras_api
*v&call_and_return_all_conditional_losses
w__call__"
_tf_keras_layer
"
	optimizer
X
&0
'1
(2
)3
*4
+5
 6
!7"
trackable_list_wrapper
X
&0
'1
(2
)3
*4
+5
 6
!7"
trackable_list_wrapper
 "
trackable_list_wrapper
╩
trainable_variables
,layer_metrics
-layer_regularization_losses
	variables

.layers
/non_trainable_variables
	regularization_losses
0metrics
m__call__
k_default_save_signature
*l&call_and_return_all_conditional_losses
&l"call_and_return_conditional_losses"
_generic_user_object
,
xserving_default"
signature_map
с
1
state_size

&kernel
'recurrent_kernel
(bias
2trainable_variables
3	variables
4regularization_losses
5	keras_api
*y&call_and_return_all_conditional_losses
z__call__"
_tf_keras_layer
 "
trackable_list_wrapper
5
&0
'1
(2"
trackable_list_wrapper
5
&0
'1
(2"
trackable_list_wrapper
 "
trackable_list_wrapper
╣
trainable_variables
6layer_metrics
7layer_regularization_losses
	variables

8layers
9non_trainable_variables
regularization_losses

:states
;metrics
o__call__
*n&call_and_return_all_conditional_losses
&n"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
н
trainable_variables
<layer_metrics
=layer_regularization_losses
	variables

>layers
?non_trainable_variables
regularization_losses
@metrics
q__call__
*p&call_and_return_all_conditional_losses
&p"call_and_return_conditional_losses"
_generic_user_object
с
A
state_size

)kernel
*recurrent_kernel
+bias
Btrainable_variables
C	variables
Dregularization_losses
E	keras_api
*{&call_and_return_all_conditional_losses
|__call__"
_tf_keras_layer
 "
trackable_list_wrapper
5
)0
*1
+2"
trackable_list_wrapper
5
)0
*1
+2"
trackable_list_wrapper
 "
trackable_list_wrapper
╣
trainable_variables
Flayer_metrics
Glayer_regularization_losses
	variables

Hlayers
Inon_trainable_variables
regularization_losses

Jstates
Kmetrics
s__call__
*r&call_and_return_all_conditional_losses
&r"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
н
trainable_variables
Llayer_metrics
Mlayer_regularization_losses
	variables

Nlayers
Onon_trainable_variables
regularization_losses
Pmetrics
u__call__
*t&call_and_return_all_conditional_losses
&t"call_and_return_conditional_losses"
_generic_user_object
!:	╬2dense_9/kernel
:2dense_9/bias
.
 0
!1"
trackable_list_wrapper
.
 0
!1"
trackable_list_wrapper
 "
trackable_list_wrapper
н
"trainable_variables
Qlayer_metrics
Rlayer_regularization_losses
#	variables

Slayers
Tnon_trainable_variables
$regularization_losses
Umetrics
w__call__
*v&call_and_return_all_conditional_losses
&v"call_and_return_conditional_losses"
_generic_user_object
.:,	]а2lstm_18/lstm_cell_18/kernel
8:6	hа2%lstm_18/lstm_cell_18/recurrent_kernel
(:&а2lstm_18/lstm_cell_18/bias
.:,	h╕
2lstm_19/lstm_cell_19/kernel
9:7
╬╕
2%lstm_19/lstm_cell_19/recurrent_kernel
(:&╕
2lstm_19/lstm_cell_19/bias
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
C
0
1
2
3
4"
trackable_list_wrapper
 "
trackable_list_wrapper
.
V0
W1"
trackable_list_wrapper
 "
trackable_list_wrapper
5
&0
'1
(2"
trackable_list_wrapper
5
&0
'1
(2"
trackable_list_wrapper
 "
trackable_list_wrapper
н
2trainable_variables
Xlayer_metrics
Ylayer_regularization_losses
3	variables

Zlayers
[non_trainable_variables
4regularization_losses
\metrics
z__call__
*y&call_and_return_all_conditional_losses
&y"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
5
)0
*1
+2"
trackable_list_wrapper
5
)0
*1
+2"
trackable_list_wrapper
 "
trackable_list_wrapper
н
Btrainable_variables
]layer_metrics
^layer_regularization_losses
C	variables

_layers
`non_trainable_variables
Dregularization_losses
ametrics
|__call__
*{&call_and_return_all_conditional_losses
&{"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
N
	btotal
	ccount
d	variables
e	keras_api"
_tf_keras_metric
^
	ftotal
	gcount
h
_fn_kwargs
i	variables
j	keras_api"
_tf_keras_metric
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
:  (2total
:  (2count
.
b0
c1"
trackable_list_wrapper
-
d	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
f0
g1"
trackable_list_wrapper
-
i	variables"
_generic_user_object
╘B╤
#__inference__wrapped_model_32586366lstm_18_input"Ш
С▓Н
FullArgSpec
argsЪ 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
Ў2є
J__inference_sequential_9_layer_call_and_return_conditional_losses_32588910
J__inference_sequential_9_layer_call_and_return_conditional_losses_32589251
J__inference_sequential_9_layer_call_and_return_conditional_losses_32588535
J__inference_sequential_9_layer_call_and_return_conditional_losses_32588560└
╖▓│
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
К2З
/__inference_sequential_9_layer_call_fn_32588020
/__inference_sequential_9_layer_call_fn_32589272
/__inference_sequential_9_layer_call_fn_32589293
/__inference_sequential_9_layer_call_fn_32588510└
╖▓│
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
ў2Ї
E__inference_lstm_18_layer_call_and_return_conditional_losses_32589444
E__inference_lstm_18_layer_call_and_return_conditional_losses_32589595
E__inference_lstm_18_layer_call_and_return_conditional_losses_32589746
E__inference_lstm_18_layer_call_and_return_conditional_losses_32589897╒
╠▓╚
FullArgSpecB
args:Ъ7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsЪ

 
p 

 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
Л2И
*__inference_lstm_18_layer_call_fn_32589908
*__inference_lstm_18_layer_call_fn_32589919
*__inference_lstm_18_layer_call_fn_32589930
*__inference_lstm_18_layer_call_fn_32589941╒
╠▓╚
FullArgSpecB
args:Ъ7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsЪ

 
p 

 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
╬2╦
H__inference_dropout_18_layer_call_and_return_conditional_losses_32589946
H__inference_dropout_18_layer_call_and_return_conditional_losses_32589958┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
Ш2Х
-__inference_dropout_18_layer_call_fn_32589963
-__inference_dropout_18_layer_call_fn_32589968┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
ў2Ї
E__inference_lstm_19_layer_call_and_return_conditional_losses_32590119
E__inference_lstm_19_layer_call_and_return_conditional_losses_32590270
E__inference_lstm_19_layer_call_and_return_conditional_losses_32590421
E__inference_lstm_19_layer_call_and_return_conditional_losses_32590572╒
╠▓╚
FullArgSpecB
args:Ъ7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsЪ

 
p 

 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
Л2И
*__inference_lstm_19_layer_call_fn_32590583
*__inference_lstm_19_layer_call_fn_32590594
*__inference_lstm_19_layer_call_fn_32590605
*__inference_lstm_19_layer_call_fn_32590616╒
╠▓╚
FullArgSpecB
args:Ъ7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsЪ

 
p 

 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
╬2╦
H__inference_dropout_19_layer_call_and_return_conditional_losses_32590621
H__inference_dropout_19_layer_call_and_return_conditional_losses_32590633┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
Ш2Х
-__inference_dropout_19_layer_call_fn_32590638
-__inference_dropout_19_layer_call_fn_32590643┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
я2ь
E__inference_dense_9_layer_call_and_return_conditional_losses_32590674в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╘2╤
*__inference_dense_9_layer_call_fn_32590683в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╙B╨
&__inference_signature_wrapper_32588583lstm_18_input"Ф
Н▓Й
FullArgSpec
argsЪ 
varargs
 
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
▄2┘
J__inference_lstm_cell_18_layer_call_and_return_conditional_losses_32590715
J__inference_lstm_cell_18_layer_call_and_return_conditional_losses_32590747╛
╡▓▒
FullArgSpec3
args+Ъ(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
ж2г
/__inference_lstm_cell_18_layer_call_fn_32590764
/__inference_lstm_cell_18_layer_call_fn_32590781╛
╡▓▒
FullArgSpec3
args+Ъ(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
▄2┘
J__inference_lstm_cell_19_layer_call_and_return_conditional_losses_32590813
J__inference_lstm_cell_19_layer_call_and_return_conditional_losses_32590845╛
╡▓▒
FullArgSpec3
args+Ъ(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
ж2г
/__inference_lstm_cell_19_layer_call_fn_32590862
/__inference_lstm_cell_19_layer_call_fn_32590879╛
╡▓▒
FullArgSpec3
args+Ъ(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 д
#__inference__wrapped_model_32586366}&'()*+ !:в7
0в-
+К(
lstm_18_input         ]
к "5к2
0
dense_9%К"
dense_9         о
E__inference_dense_9_layer_call_and_return_conditional_losses_32590674e !4в1
*в'
%К"
inputs         ╬
к ")в&
К
0         
Ъ Ж
*__inference_dense_9_layer_call_fn_32590683X !4в1
*в'
%К"
inputs         ╬
к "К         ░
H__inference_dropout_18_layer_call_and_return_conditional_losses_32589946d7в4
-в*
$К!
inputs         h
p 
к ")в&
К
0         h
Ъ ░
H__inference_dropout_18_layer_call_and_return_conditional_losses_32589958d7в4
-в*
$К!
inputs         h
p
к ")в&
К
0         h
Ъ И
-__inference_dropout_18_layer_call_fn_32589963W7в4
-в*
$К!
inputs         h
p 
к "К         hИ
-__inference_dropout_18_layer_call_fn_32589968W7в4
-в*
$К!
inputs         h
p
к "К         h▓
H__inference_dropout_19_layer_call_and_return_conditional_losses_32590621f8в5
.в+
%К"
inputs         ╬
p 
к "*в'
 К
0         ╬
Ъ ▓
H__inference_dropout_19_layer_call_and_return_conditional_losses_32590633f8в5
.в+
%К"
inputs         ╬
p
к "*в'
 К
0         ╬
Ъ К
-__inference_dropout_19_layer_call_fn_32590638Y8в5
.в+
%К"
inputs         ╬
p 
к "К         ╬К
-__inference_dropout_19_layer_call_fn_32590643Y8в5
.в+
%К"
inputs         ╬
p
к "К         ╬╘
E__inference_lstm_18_layer_call_and_return_conditional_losses_32589444К&'(OвL
EвB
4Ъ1
/К,
inputs/0                  ]

 
p 

 
к "2в/
(К%
0                  h
Ъ ╘
E__inference_lstm_18_layer_call_and_return_conditional_losses_32589595К&'(OвL
EвB
4Ъ1
/К,
inputs/0                  ]

 
p

 
к "2в/
(К%
0                  h
Ъ ║
E__inference_lstm_18_layer_call_and_return_conditional_losses_32589746q&'(?в<
5в2
$К!
inputs         ]

 
p 

 
к ")в&
К
0         h
Ъ ║
E__inference_lstm_18_layer_call_and_return_conditional_losses_32589897q&'(?в<
5в2
$К!
inputs         ]

 
p

 
к ")в&
К
0         h
Ъ л
*__inference_lstm_18_layer_call_fn_32589908}&'(OвL
EвB
4Ъ1
/К,
inputs/0                  ]

 
p 

 
к "%К"                  hл
*__inference_lstm_18_layer_call_fn_32589919}&'(OвL
EвB
4Ъ1
/К,
inputs/0                  ]

 
p

 
к "%К"                  hТ
*__inference_lstm_18_layer_call_fn_32589930d&'(?в<
5в2
$К!
inputs         ]

 
p 

 
к "К         hТ
*__inference_lstm_18_layer_call_fn_32589941d&'(?в<
5в2
$К!
inputs         ]

 
p

 
к "К         h╒
E__inference_lstm_19_layer_call_and_return_conditional_losses_32590119Л)*+OвL
EвB
4Ъ1
/К,
inputs/0                  h

 
p 

 
к "3в0
)К&
0                  ╬
Ъ ╒
E__inference_lstm_19_layer_call_and_return_conditional_losses_32590270Л)*+OвL
EвB
4Ъ1
/К,
inputs/0                  h

 
p

 
к "3в0
)К&
0                  ╬
Ъ ╗
E__inference_lstm_19_layer_call_and_return_conditional_losses_32590421r)*+?в<
5в2
$К!
inputs         h

 
p 

 
к "*в'
 К
0         ╬
Ъ ╗
E__inference_lstm_19_layer_call_and_return_conditional_losses_32590572r)*+?в<
5в2
$К!
inputs         h

 
p

 
к "*в'
 К
0         ╬
Ъ м
*__inference_lstm_19_layer_call_fn_32590583~)*+OвL
EвB
4Ъ1
/К,
inputs/0                  h

 
p 

 
к "&К#                  ╬м
*__inference_lstm_19_layer_call_fn_32590594~)*+OвL
EвB
4Ъ1
/К,
inputs/0                  h

 
p

 
к "&К#                  ╬У
*__inference_lstm_19_layer_call_fn_32590605e)*+?в<
5в2
$К!
inputs         h

 
p 

 
к "К         ╬У
*__inference_lstm_19_layer_call_fn_32590616e)*+?в<
5в2
$К!
inputs         h

 
p

 
к "К         ╬╠
J__inference_lstm_cell_18_layer_call_and_return_conditional_losses_32590715¤&'(Ав}
vвs
 К
inputs         ]
KвH
"К
states/0         h
"К
states/1         h
p 
к "sвp
iвf
К
0/0         h
EЪB
К
0/1/0         h
К
0/1/1         h
Ъ ╠
J__inference_lstm_cell_18_layer_call_and_return_conditional_losses_32590747¤&'(Ав}
vвs
 К
inputs         ]
KвH
"К
states/0         h
"К
states/1         h
p
к "sвp
iвf
К
0/0         h
EЪB
К
0/1/0         h
К
0/1/1         h
Ъ б
/__inference_lstm_cell_18_layer_call_fn_32590764э&'(Ав}
vвs
 К
inputs         ]
KвH
"К
states/0         h
"К
states/1         h
p 
к "cв`
К
0         h
AЪ>
К
1/0         h
К
1/1         hб
/__inference_lstm_cell_18_layer_call_fn_32590781э&'(Ав}
vвs
 К
inputs         ]
KвH
"К
states/0         h
"К
states/1         h
p
к "cв`
К
0         h
AЪ>
К
1/0         h
К
1/1         h╤
J__inference_lstm_cell_19_layer_call_and_return_conditional_losses_32590813В)*+Вв
xвu
 К
inputs         h
MвJ
#К 
states/0         ╬
#К 
states/1         ╬
p 
к "vвs
lвi
К
0/0         ╬
GЪD
 К
0/1/0         ╬
 К
0/1/1         ╬
Ъ ╤
J__inference_lstm_cell_19_layer_call_and_return_conditional_losses_32590845В)*+Вв
xвu
 К
inputs         h
MвJ
#К 
states/0         ╬
#К 
states/1         ╬
p
к "vвs
lвi
К
0/0         ╬
GЪD
 К
0/1/0         ╬
 К
0/1/1         ╬
Ъ ж
/__inference_lstm_cell_19_layer_call_fn_32590862Є)*+Вв
xвu
 К
inputs         h
MвJ
#К 
states/0         ╬
#К 
states/1         ╬
p 
к "fвc
К
0         ╬
CЪ@
К
1/0         ╬
К
1/1         ╬ж
/__inference_lstm_cell_19_layer_call_fn_32590879Є)*+Вв
xвu
 К
inputs         h
MвJ
#К 
states/0         ╬
#К 
states/1         ╬
p
к "fвc
К
0         ╬
CЪ@
К
1/0         ╬
К
1/1         ╬╟
J__inference_sequential_9_layer_call_and_return_conditional_losses_32588535y&'()*+ !Bв?
8в5
+К(
lstm_18_input         ]
p 

 
к ")в&
К
0         
Ъ ╟
J__inference_sequential_9_layer_call_and_return_conditional_losses_32588560y&'()*+ !Bв?
8в5
+К(
lstm_18_input         ]
p

 
к ")в&
К
0         
Ъ └
J__inference_sequential_9_layer_call_and_return_conditional_losses_32588910r&'()*+ !;в8
1в.
$К!
inputs         ]
p 

 
к ")в&
К
0         
Ъ └
J__inference_sequential_9_layer_call_and_return_conditional_losses_32589251r&'()*+ !;в8
1в.
$К!
inputs         ]
p

 
к ")в&
К
0         
Ъ Я
/__inference_sequential_9_layer_call_fn_32588020l&'()*+ !Bв?
8в5
+К(
lstm_18_input         ]
p 

 
к "К         Я
/__inference_sequential_9_layer_call_fn_32588510l&'()*+ !Bв?
8в5
+К(
lstm_18_input         ]
p

 
к "К         Ш
/__inference_sequential_9_layer_call_fn_32589272e&'()*+ !;в8
1в.
$К!
inputs         ]
p 

 
к "К         Ш
/__inference_sequential_9_layer_call_fn_32589293e&'()*+ !;в8
1в.
$К!
inputs         ]
p

 
к "К         ╣
&__inference_signature_wrapper_32588583О&'()*+ !KвH
в 
Aк>
<
lstm_18_input+К(
lstm_18_input         ]"5к2
0
dense_9%К"
dense_9         