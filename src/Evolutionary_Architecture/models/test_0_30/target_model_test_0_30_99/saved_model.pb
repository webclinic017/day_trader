╠Р&
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
И"serve*2.6.02v2.6.0-rc2-32-g919f693420e8н╜$
y
dense_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	Ц*
shared_namedense_7/kernel
r
"dense_7/kernel/Read/ReadVariableOpReadVariableOpdense_7/kernel*
_output_shapes
:	Ц*
dtype0
p
dense_7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_7/bias
i
 dense_7/bias/Read/ReadVariableOpReadVariableOpdense_7/bias*
_output_shapes
:*
dtype0
У
lstm_14/lstm_cell_14/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	]╕*,
shared_namelstm_14/lstm_cell_14/kernel
М
/lstm_14/lstm_cell_14/kernel/Read/ReadVariableOpReadVariableOplstm_14/lstm_cell_14/kernel*
_output_shapes
:	]╕*
dtype0
и
%lstm_14/lstm_cell_14/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ю╕*6
shared_name'%lstm_14/lstm_cell_14/recurrent_kernel
б
9lstm_14/lstm_cell_14/recurrent_kernel/Read/ReadVariableOpReadVariableOp%lstm_14/lstm_cell_14/recurrent_kernel* 
_output_shapes
:
ю╕*
dtype0
Л
lstm_14/lstm_cell_14/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:╕**
shared_namelstm_14/lstm_cell_14/bias
Д
-lstm_14/lstm_cell_14/bias/Read/ReadVariableOpReadVariableOplstm_14/lstm_cell_14/bias*
_output_shapes	
:╕*
dtype0
Ф
lstm_15/lstm_cell_15/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ю╪*,
shared_namelstm_15/lstm_cell_15/kernel
Н
/lstm_15/lstm_cell_15/kernel/Read/ReadVariableOpReadVariableOplstm_15/lstm_cell_15/kernel* 
_output_shapes
:
ю╪*
dtype0
и
%lstm_15/lstm_cell_15/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
Ц╪*6
shared_name'%lstm_15/lstm_cell_15/recurrent_kernel
б
9lstm_15/lstm_cell_15/recurrent_kernel/Read/ReadVariableOpReadVariableOp%lstm_15/lstm_cell_15/recurrent_kernel* 
_output_shapes
:
Ц╪*
dtype0
Л
lstm_15/lstm_cell_15/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:╪**
shared_namelstm_15/lstm_cell_15/bias
Д
-lstm_15/lstm_cell_15/bias/Read/ReadVariableOpReadVariableOplstm_15/lstm_cell_15/bias*
_output_shapes	
:╪*
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
regularization_losses
		variables

	keras_api

signatures
l
cell

state_spec
trainable_variables
regularization_losses
	variables
	keras_api
R
trainable_variables
regularization_losses
	variables
	keras_api
l
cell

state_spec
trainable_variables
regularization_losses
	variables
	keras_api
R
trainable_variables
regularization_losses
	variables
	keras_api
h

 kernel
!bias
"trainable_variables
#regularization_losses
$	variables
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
н

,layers
trainable_variables
regularization_losses
		variables
-layer_metrics
.non_trainable_variables
/metrics
0layer_regularization_losses
 
О
1
state_size

&kernel
'recurrent_kernel
(bias
2trainable_variables
3regularization_losses
4	variables
5	keras_api
 

&0
'1
(2
 

&0
'1
(2
╣

6layers
trainable_variables
regularization_losses
	variables

7states
8layer_metrics
9non_trainable_variables
:metrics
;layer_regularization_losses
 
 
 
н

<layers
trainable_variables
regularization_losses
	variables
=layer_metrics
>non_trainable_variables
?metrics
@layer_regularization_losses
О
A
state_size

)kernel
*recurrent_kernel
+bias
Btrainable_variables
Cregularization_losses
D	variables
E	keras_api
 

)0
*1
+2
 

)0
*1
+2
╣

Flayers
trainable_variables
regularization_losses
	variables

Gstates
Hlayer_metrics
Inon_trainable_variables
Jmetrics
Klayer_regularization_losses
 
 
 
н

Llayers
trainable_variables
regularization_losses
	variables
Mlayer_metrics
Nnon_trainable_variables
Ometrics
Player_regularization_losses
ZX
VARIABLE_VALUEdense_7/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_7/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

 0
!1
 

 0
!1
н

Qlayers
"trainable_variables
#regularization_losses
$	variables
Rlayer_metrics
Snon_trainable_variables
Tmetrics
Ulayer_regularization_losses
a_
VARIABLE_VALUElstm_14/lstm_cell_14/kernel0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUE
ki
VARIABLE_VALUE%lstm_14/lstm_cell_14/recurrent_kernel0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUElstm_14/lstm_cell_14/bias0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUElstm_15/lstm_cell_15/kernel0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUE
ki
VARIABLE_VALUE%lstm_15/lstm_cell_15/recurrent_kernel0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUElstm_15/lstm_cell_15/bias0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUE
#
0
1
2
3
4
 
 

V0
W1
 
 

&0
'1
(2
 

&0
'1
(2
н

Xlayers
2trainable_variables
3regularization_losses
4	variables
Ylayer_metrics
Znon_trainable_variables
[metrics
\layer_regularization_losses
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
 
 

)0
*1
+2
 

)0
*1
+2
н

]layers
Btrainable_variables
Cregularization_losses
D	variables
^layer_metrics
_non_trainable_variables
`metrics
alayer_regularization_losses
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
serving_default_lstm_14_inputPlaceholder*+
_output_shapes
:         ]*
dtype0* 
shape:         ]
к
StatefulPartitionedCallStatefulPartitionedCallserving_default_lstm_14_inputlstm_14/lstm_cell_14/kernel%lstm_14/lstm_cell_14/recurrent_kernellstm_14/lstm_cell_14/biaslstm_15/lstm_cell_15/kernel%lstm_15/lstm_cell_15/recurrent_kernellstm_15/lstm_cell_15/biasdense_7/kerneldense_7/bias*
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
&__inference_signature_wrapper_26072009
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
Х
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename"dense_7/kernel/Read/ReadVariableOp dense_7/bias/Read/ReadVariableOp/lstm_14/lstm_cell_14/kernel/Read/ReadVariableOp9lstm_14/lstm_cell_14/recurrent_kernel/Read/ReadVariableOp-lstm_14/lstm_cell_14/bias/Read/ReadVariableOp/lstm_15/lstm_cell_15/kernel/Read/ReadVariableOp9lstm_15/lstm_cell_15/recurrent_kernel/Read/ReadVariableOp-lstm_15/lstm_cell_15/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOpConst*
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
!__inference__traced_save_26074364
а
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_7/kerneldense_7/biaslstm_14/lstm_cell_14/kernel%lstm_14/lstm_cell_14/recurrent_kernellstm_14/lstm_cell_14/biaslstm_15/lstm_cell_15/kernel%lstm_15/lstm_cell_15/recurrent_kernellstm_15/lstm_cell_15/biastotalcounttotal_1count_1*
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
$__inference__traced_restore_26074410╘Є#
у
═
while_cond_26070510
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_26070510___redundant_placeholder06
2while_while_cond_26070510___redundant_placeholder16
2while_while_cond_26070510___redundant_placeholder26
2while_while_cond_26070510___redundant_placeholder3
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
B: : : : :         Ц:         Ц: ::::: 
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
:         Ц:.*
(
_output_shapes
:         Ц:

_output_shapes
: :

_output_shapes
:
К\
Я
E__inference_lstm_15_layer_call_and_return_conditional_losses_26071643

inputs?
+lstm_cell_15_matmul_readvariableop_resource:
ю╪A
-lstm_cell_15_matmul_1_readvariableop_resource:
Ц╪;
,lstm_cell_15_biasadd_readvariableop_resource:	╪
identityИв#lstm_cell_15/BiasAdd/ReadVariableOpв"lstm_cell_15/MatMul/ReadVariableOpв$lstm_cell_15/MatMul_1/ReadVariableOpвwhileD
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
B :Ц2
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
B :Ц2
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
:         Ц2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :Ц2
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
B :Ц2
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
:         Ц2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm{
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:         ю2
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
valueB"    ю   27
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
strided_slice_2/stack_2¤
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         ю*
shrink_axis_mask2
strided_slice_2╢
"lstm_cell_15/MatMul/ReadVariableOpReadVariableOp+lstm_cell_15_matmul_readvariableop_resource* 
_output_shapes
:
ю╪*
dtype02$
"lstm_cell_15/MatMul/ReadVariableOpн
lstm_cell_15/MatMulMatMulstrided_slice_2:output:0*lstm_cell_15/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╪2
lstm_cell_15/MatMul╝
$lstm_cell_15/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_15_matmul_1_readvariableop_resource* 
_output_shapes
:
Ц╪*
dtype02&
$lstm_cell_15/MatMul_1/ReadVariableOpй
lstm_cell_15/MatMul_1MatMulzeros:output:0,lstm_cell_15/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╪2
lstm_cell_15/MatMul_1а
lstm_cell_15/addAddV2lstm_cell_15/MatMul:product:0lstm_cell_15/MatMul_1:product:0*
T0*(
_output_shapes
:         ╪2
lstm_cell_15/add┤
#lstm_cell_15/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_15_biasadd_readvariableop_resource*
_output_shapes	
:╪*
dtype02%
#lstm_cell_15/BiasAdd/ReadVariableOpн
lstm_cell_15/BiasAddBiasAddlstm_cell_15/add:z:0+lstm_cell_15/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╪2
lstm_cell_15/BiasAdd~
lstm_cell_15/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_15/split/split_dimў
lstm_cell_15/splitSplit%lstm_cell_15/split/split_dim:output:0lstm_cell_15/BiasAdd:output:0*
T0*d
_output_shapesR
P:         Ц:         Ц:         Ц:         Ц*
	num_split2
lstm_cell_15/splitЗ
lstm_cell_15/SigmoidSigmoidlstm_cell_15/split:output:0*
T0*(
_output_shapes
:         Ц2
lstm_cell_15/SigmoidЛ
lstm_cell_15/Sigmoid_1Sigmoidlstm_cell_15/split:output:1*
T0*(
_output_shapes
:         Ц2
lstm_cell_15/Sigmoid_1М
lstm_cell_15/mulMullstm_cell_15/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:         Ц2
lstm_cell_15/mul~
lstm_cell_15/ReluRelulstm_cell_15/split:output:2*
T0*(
_output_shapes
:         Ц2
lstm_cell_15/ReluЭ
lstm_cell_15/mul_1Mullstm_cell_15/Sigmoid:y:0lstm_cell_15/Relu:activations:0*
T0*(
_output_shapes
:         Ц2
lstm_cell_15/mul_1Т
lstm_cell_15/add_1AddV2lstm_cell_15/mul:z:0lstm_cell_15/mul_1:z:0*
T0*(
_output_shapes
:         Ц2
lstm_cell_15/add_1Л
lstm_cell_15/Sigmoid_2Sigmoidlstm_cell_15/split:output:3*
T0*(
_output_shapes
:         Ц2
lstm_cell_15/Sigmoid_2}
lstm_cell_15/Relu_1Relulstm_cell_15/add_1:z:0*
T0*(
_output_shapes
:         Ц2
lstm_cell_15/Relu_1б
lstm_cell_15/mul_2Mullstm_cell_15/Sigmoid_2:y:0!lstm_cell_15/Relu_1:activations:0*
T0*(
_output_shapes
:         Ц2
lstm_cell_15/mul_2П
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    Ц   2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_15_matmul_readvariableop_resource-lstm_cell_15_matmul_1_readvariableop_resource,lstm_cell_15_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :         Ц:         Ц: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_26071559*
condR
while_cond_26071558*M
output_shapes<
:: : : : :         Ц:         Ц: : : : : *
parallel_iterations 2
while╡
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    Ц   22
0TensorArrayV2Stack/TensorListStack/element_shapeщ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:         Ц*
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
:         Ц*
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
:         Ц2
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
:         Ц2

Identity╚
NoOpNoOp$^lstm_cell_15/BiasAdd/ReadVariableOp#^lstm_cell_15/MatMul/ReadVariableOp%^lstm_cell_15/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:         ю: : : 2J
#lstm_cell_15/BiasAdd/ReadVariableOp#lstm_cell_15/BiasAdd/ReadVariableOp2H
"lstm_cell_15/MatMul/ReadVariableOp"lstm_cell_15/MatMul/ReadVariableOp2L
$lstm_cell_15/MatMul_1/ReadVariableOp$lstm_cell_15/MatMul_1/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:         ю
 
_user_specified_nameinputs
Д\
Ю
E__inference_lstm_14_layer_call_and_return_conditional_losses_26073172

inputs>
+lstm_cell_14_matmul_readvariableop_resource:	]╕A
-lstm_cell_14_matmul_1_readvariableop_resource:
ю╕;
,lstm_cell_14_biasadd_readvariableop_resource:	╕
identityИв#lstm_cell_14/BiasAdd/ReadVariableOpв"lstm_cell_14/MatMul/ReadVariableOpв$lstm_cell_14/MatMul_1/ReadVariableOpвwhileD
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
B :ю2
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
B :ю2
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
:         ю2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :ю2
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
B :ю2
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
:         ю2	
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
"lstm_cell_14/MatMul/ReadVariableOpReadVariableOp+lstm_cell_14_matmul_readvariableop_resource*
_output_shapes
:	]╕*
dtype02$
"lstm_cell_14/MatMul/ReadVariableOpн
lstm_cell_14/MatMulMatMulstrided_slice_2:output:0*lstm_cell_14/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╕2
lstm_cell_14/MatMul╝
$lstm_cell_14/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_14_matmul_1_readvariableop_resource* 
_output_shapes
:
ю╕*
dtype02&
$lstm_cell_14/MatMul_1/ReadVariableOpй
lstm_cell_14/MatMul_1MatMulzeros:output:0,lstm_cell_14/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╕2
lstm_cell_14/MatMul_1а
lstm_cell_14/addAddV2lstm_cell_14/MatMul:product:0lstm_cell_14/MatMul_1:product:0*
T0*(
_output_shapes
:         ╕2
lstm_cell_14/add┤
#lstm_cell_14/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_14_biasadd_readvariableop_resource*
_output_shapes	
:╕*
dtype02%
#lstm_cell_14/BiasAdd/ReadVariableOpн
lstm_cell_14/BiasAddBiasAddlstm_cell_14/add:z:0+lstm_cell_14/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╕2
lstm_cell_14/BiasAdd~
lstm_cell_14/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_14/split/split_dimў
lstm_cell_14/splitSplit%lstm_cell_14/split/split_dim:output:0lstm_cell_14/BiasAdd:output:0*
T0*d
_output_shapesR
P:         ю:         ю:         ю:         ю*
	num_split2
lstm_cell_14/splitЗ
lstm_cell_14/SigmoidSigmoidlstm_cell_14/split:output:0*
T0*(
_output_shapes
:         ю2
lstm_cell_14/SigmoidЛ
lstm_cell_14/Sigmoid_1Sigmoidlstm_cell_14/split:output:1*
T0*(
_output_shapes
:         ю2
lstm_cell_14/Sigmoid_1М
lstm_cell_14/mulMullstm_cell_14/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:         ю2
lstm_cell_14/mul~
lstm_cell_14/ReluRelulstm_cell_14/split:output:2*
T0*(
_output_shapes
:         ю2
lstm_cell_14/ReluЭ
lstm_cell_14/mul_1Mullstm_cell_14/Sigmoid:y:0lstm_cell_14/Relu:activations:0*
T0*(
_output_shapes
:         ю2
lstm_cell_14/mul_1Т
lstm_cell_14/add_1AddV2lstm_cell_14/mul:z:0lstm_cell_14/mul_1:z:0*
T0*(
_output_shapes
:         ю2
lstm_cell_14/add_1Л
lstm_cell_14/Sigmoid_2Sigmoidlstm_cell_14/split:output:3*
T0*(
_output_shapes
:         ю2
lstm_cell_14/Sigmoid_2}
lstm_cell_14/Relu_1Relulstm_cell_14/add_1:z:0*
T0*(
_output_shapes
:         ю2
lstm_cell_14/Relu_1б
lstm_cell_14/mul_2Mullstm_cell_14/Sigmoid_2:y:0!lstm_cell_14/Relu_1:activations:0*
T0*(
_output_shapes
:         ю2
lstm_cell_14/mul_2П
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ю   2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_14_matmul_readvariableop_resource-lstm_cell_14_matmul_1_readvariableop_resource,lstm_cell_14_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :         ю:         ю: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_26073088*
condR
while_cond_26073087*M
output_shapes<
:: : : : :         ю:         ю: : : : : *
parallel_iterations 2
while╡
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ю   22
0TensorArrayV2Stack/TensorListStack/element_shapeщ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:         ю*
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
:         ю*
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
:         ю2
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
:         ю2

Identity╚
NoOpNoOp$^lstm_cell_14/BiasAdd/ReadVariableOp#^lstm_cell_14/MatMul/ReadVariableOp%^lstm_cell_14/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         ]: : : 2J
#lstm_cell_14/BiasAdd/ReadVariableOp#lstm_cell_14/BiasAdd/ReadVariableOp2H
"lstm_cell_14/MatMul/ReadVariableOp"lstm_cell_14/MatMul/ReadVariableOp2L
$lstm_cell_14/MatMul_1/ReadVariableOp$lstm_cell_14/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:         ]
 
_user_specified_nameinputs
у
═
while_cond_26073238
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_26073238___redundant_placeholder06
2while_while_cond_26073238___redundant_placeholder16
2while_while_cond_26073238___redundant_placeholder26
2while_while_cond_26073238___redundant_placeholder3
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
B: : : : :         ю:         ю: ::::: 
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
:         ю:.*
(
_output_shapes
:         ю:

_output_shapes
: :

_output_shapes
:
╘

э
lstm_15_while_cond_26072223,
(lstm_15_while_lstm_15_while_loop_counter2
.lstm_15_while_lstm_15_while_maximum_iterations
lstm_15_while_placeholder
lstm_15_while_placeholder_1
lstm_15_while_placeholder_2
lstm_15_while_placeholder_3.
*lstm_15_while_less_lstm_15_strided_slice_1F
Blstm_15_while_lstm_15_while_cond_26072223___redundant_placeholder0F
Blstm_15_while_lstm_15_while_cond_26072223___redundant_placeholder1F
Blstm_15_while_lstm_15_while_cond_26072223___redundant_placeholder2F
Blstm_15_while_lstm_15_while_cond_26072223___redundant_placeholder3
lstm_15_while_identity
Ш
lstm_15/while/LessLesslstm_15_while_placeholder*lstm_15_while_less_lstm_15_strided_slice_1*
T0*
_output_shapes
: 2
lstm_15/while/Lessu
lstm_15/while/IdentityIdentitylstm_15/while/Less:z:0*
T0
*
_output_shapes
: 2
lstm_15/while/Identity"9
lstm_15_while_identitylstm_15/while/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :         Ц:         Ц: ::::: 
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
:         Ц:.*
(
_output_shapes
:         Ц:

_output_shapes
: :

_output_shapes
:
░?
╘
while_body_26071125
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_14_matmul_readvariableop_resource_0:	]╕I
5while_lstm_cell_14_matmul_1_readvariableop_resource_0:
ю╕C
4while_lstm_cell_14_biasadd_readvariableop_resource_0:	╕
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_14_matmul_readvariableop_resource:	]╕G
3while_lstm_cell_14_matmul_1_readvariableop_resource:
ю╕A
2while_lstm_cell_14_biasadd_readvariableop_resource:	╕Ив)while/lstm_cell_14/BiasAdd/ReadVariableOpв(while/lstm_cell_14/MatMul/ReadVariableOpв*while/lstm_cell_14/MatMul_1/ReadVariableOp├
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
(while/lstm_cell_14/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_14_matmul_readvariableop_resource_0*
_output_shapes
:	]╕*
dtype02*
(while/lstm_cell_14/MatMul/ReadVariableOp╫
while/lstm_cell_14/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_14/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╕2
while/lstm_cell_14/MatMul╨
*while/lstm_cell_14/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_14_matmul_1_readvariableop_resource_0* 
_output_shapes
:
ю╕*
dtype02,
*while/lstm_cell_14/MatMul_1/ReadVariableOp└
while/lstm_cell_14/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_14/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╕2
while/lstm_cell_14/MatMul_1╕
while/lstm_cell_14/addAddV2#while/lstm_cell_14/MatMul:product:0%while/lstm_cell_14/MatMul_1:product:0*
T0*(
_output_shapes
:         ╕2
while/lstm_cell_14/add╚
)while/lstm_cell_14/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_14_biasadd_readvariableop_resource_0*
_output_shapes	
:╕*
dtype02+
)while/lstm_cell_14/BiasAdd/ReadVariableOp┼
while/lstm_cell_14/BiasAddBiasAddwhile/lstm_cell_14/add:z:01while/lstm_cell_14/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╕2
while/lstm_cell_14/BiasAddК
"while/lstm_cell_14/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_14/split/split_dimП
while/lstm_cell_14/splitSplit+while/lstm_cell_14/split/split_dim:output:0#while/lstm_cell_14/BiasAdd:output:0*
T0*d
_output_shapesR
P:         ю:         ю:         ю:         ю*
	num_split2
while/lstm_cell_14/splitЩ
while/lstm_cell_14/SigmoidSigmoid!while/lstm_cell_14/split:output:0*
T0*(
_output_shapes
:         ю2
while/lstm_cell_14/SigmoidЭ
while/lstm_cell_14/Sigmoid_1Sigmoid!while/lstm_cell_14/split:output:1*
T0*(
_output_shapes
:         ю2
while/lstm_cell_14/Sigmoid_1б
while/lstm_cell_14/mulMul while/lstm_cell_14/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:         ю2
while/lstm_cell_14/mulР
while/lstm_cell_14/ReluRelu!while/lstm_cell_14/split:output:2*
T0*(
_output_shapes
:         ю2
while/lstm_cell_14/Relu╡
while/lstm_cell_14/mul_1Mulwhile/lstm_cell_14/Sigmoid:y:0%while/lstm_cell_14/Relu:activations:0*
T0*(
_output_shapes
:         ю2
while/lstm_cell_14/mul_1к
while/lstm_cell_14/add_1AddV2while/lstm_cell_14/mul:z:0while/lstm_cell_14/mul_1:z:0*
T0*(
_output_shapes
:         ю2
while/lstm_cell_14/add_1Э
while/lstm_cell_14/Sigmoid_2Sigmoid!while/lstm_cell_14/split:output:3*
T0*(
_output_shapes
:         ю2
while/lstm_cell_14/Sigmoid_2П
while/lstm_cell_14/Relu_1Reluwhile/lstm_cell_14/add_1:z:0*
T0*(
_output_shapes
:         ю2
while/lstm_cell_14/Relu_1╣
while/lstm_cell_14/mul_2Mul while/lstm_cell_14/Sigmoid_2:y:0'while/lstm_cell_14/Relu_1:activations:0*
T0*(
_output_shapes
:         ю2
while/lstm_cell_14/mul_2р
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_14/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_14/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:         ю2
while/Identity_4О
while/Identity_5Identitywhile/lstm_cell_14/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:         ю2
while/Identity_5▐

while/NoOpNoOp*^while/lstm_cell_14/BiasAdd/ReadVariableOp)^while/lstm_cell_14/MatMul/ReadVariableOp+^while/lstm_cell_14/MatMul_1/ReadVariableOp*"
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
2while_lstm_cell_14_biasadd_readvariableop_resource4while_lstm_cell_14_biasadd_readvariableop_resource_0"l
3while_lstm_cell_14_matmul_1_readvariableop_resource5while_lstm_cell_14_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_14_matmul_readvariableop_resource3while_lstm_cell_14_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :         ю:         ю: : : : : 2V
)while/lstm_cell_14/BiasAdd/ReadVariableOp)while/lstm_cell_14/BiasAdd/ReadVariableOp2T
(while/lstm_cell_14/MatMul/ReadVariableOp(while/lstm_cell_14/MatMul/ReadVariableOp2X
*while/lstm_cell_14/MatMul_1/ReadVariableOp*while/lstm_cell_14/MatMul_1/ReadVariableOp: 
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
:         ю:.*
(
_output_shapes
:         ю:

_output_shapes
: :

_output_shapes
: 
┤?
╓
while_body_26073914
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
3while_lstm_cell_15_matmul_readvariableop_resource_0:
ю╪I
5while_lstm_cell_15_matmul_1_readvariableop_resource_0:
Ц╪C
4while_lstm_cell_15_biasadd_readvariableop_resource_0:	╪
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
1while_lstm_cell_15_matmul_readvariableop_resource:
ю╪G
3while_lstm_cell_15_matmul_1_readvariableop_resource:
Ц╪A
2while_lstm_cell_15_biasadd_readvariableop_resource:	╪Ив)while/lstm_cell_15/BiasAdd/ReadVariableOpв(while/lstm_cell_15/MatMul/ReadVariableOpв*while/lstm_cell_15/MatMul_1/ReadVariableOp├
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ю   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape╘
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:         ю*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem╩
(while/lstm_cell_15/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_15_matmul_readvariableop_resource_0* 
_output_shapes
:
ю╪*
dtype02*
(while/lstm_cell_15/MatMul/ReadVariableOp╫
while/lstm_cell_15/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_15/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╪2
while/lstm_cell_15/MatMul╨
*while/lstm_cell_15/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_15_matmul_1_readvariableop_resource_0* 
_output_shapes
:
Ц╪*
dtype02,
*while/lstm_cell_15/MatMul_1/ReadVariableOp└
while/lstm_cell_15/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_15/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╪2
while/lstm_cell_15/MatMul_1╕
while/lstm_cell_15/addAddV2#while/lstm_cell_15/MatMul:product:0%while/lstm_cell_15/MatMul_1:product:0*
T0*(
_output_shapes
:         ╪2
while/lstm_cell_15/add╚
)while/lstm_cell_15/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_15_biasadd_readvariableop_resource_0*
_output_shapes	
:╪*
dtype02+
)while/lstm_cell_15/BiasAdd/ReadVariableOp┼
while/lstm_cell_15/BiasAddBiasAddwhile/lstm_cell_15/add:z:01while/lstm_cell_15/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╪2
while/lstm_cell_15/BiasAddК
"while/lstm_cell_15/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_15/split/split_dimП
while/lstm_cell_15/splitSplit+while/lstm_cell_15/split/split_dim:output:0#while/lstm_cell_15/BiasAdd:output:0*
T0*d
_output_shapesR
P:         Ц:         Ц:         Ц:         Ц*
	num_split2
while/lstm_cell_15/splitЩ
while/lstm_cell_15/SigmoidSigmoid!while/lstm_cell_15/split:output:0*
T0*(
_output_shapes
:         Ц2
while/lstm_cell_15/SigmoidЭ
while/lstm_cell_15/Sigmoid_1Sigmoid!while/lstm_cell_15/split:output:1*
T0*(
_output_shapes
:         Ц2
while/lstm_cell_15/Sigmoid_1б
while/lstm_cell_15/mulMul while/lstm_cell_15/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:         Ц2
while/lstm_cell_15/mulР
while/lstm_cell_15/ReluRelu!while/lstm_cell_15/split:output:2*
T0*(
_output_shapes
:         Ц2
while/lstm_cell_15/Relu╡
while/lstm_cell_15/mul_1Mulwhile/lstm_cell_15/Sigmoid:y:0%while/lstm_cell_15/Relu:activations:0*
T0*(
_output_shapes
:         Ц2
while/lstm_cell_15/mul_1к
while/lstm_cell_15/add_1AddV2while/lstm_cell_15/mul:z:0while/lstm_cell_15/mul_1:z:0*
T0*(
_output_shapes
:         Ц2
while/lstm_cell_15/add_1Э
while/lstm_cell_15/Sigmoid_2Sigmoid!while/lstm_cell_15/split:output:3*
T0*(
_output_shapes
:         Ц2
while/lstm_cell_15/Sigmoid_2П
while/lstm_cell_15/Relu_1Reluwhile/lstm_cell_15/add_1:z:0*
T0*(
_output_shapes
:         Ц2
while/lstm_cell_15/Relu_1╣
while/lstm_cell_15/mul_2Mul while/lstm_cell_15/Sigmoid_2:y:0'while/lstm_cell_15/Relu_1:activations:0*
T0*(
_output_shapes
:         Ц2
while/lstm_cell_15/mul_2р
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_15/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_15/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:         Ц2
while/Identity_4О
while/Identity_5Identitywhile/lstm_cell_15/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:         Ц2
while/Identity_5▐

while/NoOpNoOp*^while/lstm_cell_15/BiasAdd/ReadVariableOp)^while/lstm_cell_15/MatMul/ReadVariableOp+^while/lstm_cell_15/MatMul_1/ReadVariableOp*"
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
2while_lstm_cell_15_biasadd_readvariableop_resource4while_lstm_cell_15_biasadd_readvariableop_resource_0"l
3while_lstm_cell_15_matmul_1_readvariableop_resource5while_lstm_cell_15_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_15_matmul_readvariableop_resource3while_lstm_cell_15_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :         Ц:         Ц: : : : : 2V
)while/lstm_cell_15/BiasAdd/ReadVariableOp)while/lstm_cell_15/BiasAdd/ReadVariableOp2T
(while/lstm_cell_15/MatMul/ReadVariableOp(while/lstm_cell_15/MatMul/ReadVariableOp2X
*while/lstm_cell_15/MatMul_1/ReadVariableOp*while/lstm_cell_15/MatMul_1/ReadVariableOp: 
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
:         Ц:.*
(
_output_shapes
:         Ц:

_output_shapes
: :

_output_shapes
: 
╩
·
/__inference_lstm_cell_15_layer_call_fn_26074288

inputs
states_0
states_1
unknown:
ю╪
	unknown_0:
Ц╪
	unknown_1:	╪
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
<:         Ц:         Ц:         Ц*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_lstm_cell_15_layer_call_and_return_conditional_losses_260704972
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         Ц2

IdentityА

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:         Ц2

Identity_1А

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:         Ц2

Identity_2h
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:         ю:         Ц:         Ц: : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         ю
 
_user_specified_nameinputs:RN
(
_output_shapes
:         Ц
"
_user_specified_name
states/0:RN
(
_output_shapes
:         Ц
"
_user_specified_name
states/1
╘

э
lstm_15_while_cond_26072557,
(lstm_15_while_lstm_15_while_loop_counter2
.lstm_15_while_lstm_15_while_maximum_iterations
lstm_15_while_placeholder
lstm_15_while_placeholder_1
lstm_15_while_placeholder_2
lstm_15_while_placeholder_3.
*lstm_15_while_less_lstm_15_strided_slice_1F
Blstm_15_while_lstm_15_while_cond_26072557___redundant_placeholder0F
Blstm_15_while_lstm_15_while_cond_26072557___redundant_placeholder1F
Blstm_15_while_lstm_15_while_cond_26072557___redundant_placeholder2F
Blstm_15_while_lstm_15_while_cond_26072557___redundant_placeholder3
lstm_15_while_identity
Ш
lstm_15/while/LessLesslstm_15_while_placeholder*lstm_15_while_less_lstm_15_strided_slice_1*
T0*
_output_shapes
: 2
lstm_15/while/Lessu
lstm_15/while/IdentityIdentitylstm_15/while/Less:z:0*
T0
*
_output_shapes
: 2
lstm_15/while/Identity"9
lstm_15_while_identitylstm_15/while/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :         Ц:         Ц: ::::: 
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
:         Ц:.*
(
_output_shapes
:         Ц:

_output_shapes
: :

_output_shapes
:
р
║
*__inference_lstm_14_layer_call_fn_26073345
inputs_0
unknown:	]╕
	unknown_0:
ю╕
	unknown_1:	╕
identityИвStatefulPartitionedCallТ
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:                  ю*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_lstm_14_layer_call_and_return_conditional_losses_260701602
StatefulPartitionedCallЙ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:                  ю2

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
Ч

╠
/__inference_sequential_7_layer_call_fn_26072698

inputs
unknown:	]╕
	unknown_0:
ю╕
	unknown_1:	╕
	unknown_2:
ю╪
	unknown_3:
Ц╪
	unknown_4:	╪
	unknown_5:	Ц
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
J__inference_sequential_7_layer_call_and_return_conditional_losses_260714272
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
у
═
while_cond_26071124
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_26071124___redundant_placeholder06
2while_while_cond_26071124___redundant_placeholder16
2while_while_cond_26071124___redundant_placeholder26
2while_while_cond_26071124___redundant_placeholder3
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
B: : : : :         ю:         ю: ::::: 
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
:         ю:.*
(
_output_shapes
:         ю:

_output_shapes
: :

_output_shapes
:
у
═
while_cond_26072936
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_26072936___redundant_placeholder06
2while_while_cond_26072936___redundant_placeholder16
2while_while_cond_26072936___redundant_placeholder26
2while_while_cond_26072936___redundant_placeholder3
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
B: : : : :         ю:         ю: ::::: 
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
:         ю:.*
(
_output_shapes
:         ю:

_output_shapes
: :

_output_shapes
:
Д\
Ю
E__inference_lstm_14_layer_call_and_return_conditional_losses_26073323

inputs>
+lstm_cell_14_matmul_readvariableop_resource:	]╕A
-lstm_cell_14_matmul_1_readvariableop_resource:
ю╕;
,lstm_cell_14_biasadd_readvariableop_resource:	╕
identityИв#lstm_cell_14/BiasAdd/ReadVariableOpв"lstm_cell_14/MatMul/ReadVariableOpв$lstm_cell_14/MatMul_1/ReadVariableOpвwhileD
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
B :ю2
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
B :ю2
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
:         ю2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :ю2
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
B :ю2
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
:         ю2	
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
"lstm_cell_14/MatMul/ReadVariableOpReadVariableOp+lstm_cell_14_matmul_readvariableop_resource*
_output_shapes
:	]╕*
dtype02$
"lstm_cell_14/MatMul/ReadVariableOpн
lstm_cell_14/MatMulMatMulstrided_slice_2:output:0*lstm_cell_14/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╕2
lstm_cell_14/MatMul╝
$lstm_cell_14/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_14_matmul_1_readvariableop_resource* 
_output_shapes
:
ю╕*
dtype02&
$lstm_cell_14/MatMul_1/ReadVariableOpй
lstm_cell_14/MatMul_1MatMulzeros:output:0,lstm_cell_14/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╕2
lstm_cell_14/MatMul_1а
lstm_cell_14/addAddV2lstm_cell_14/MatMul:product:0lstm_cell_14/MatMul_1:product:0*
T0*(
_output_shapes
:         ╕2
lstm_cell_14/add┤
#lstm_cell_14/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_14_biasadd_readvariableop_resource*
_output_shapes	
:╕*
dtype02%
#lstm_cell_14/BiasAdd/ReadVariableOpн
lstm_cell_14/BiasAddBiasAddlstm_cell_14/add:z:0+lstm_cell_14/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╕2
lstm_cell_14/BiasAdd~
lstm_cell_14/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_14/split/split_dimў
lstm_cell_14/splitSplit%lstm_cell_14/split/split_dim:output:0lstm_cell_14/BiasAdd:output:0*
T0*d
_output_shapesR
P:         ю:         ю:         ю:         ю*
	num_split2
lstm_cell_14/splitЗ
lstm_cell_14/SigmoidSigmoidlstm_cell_14/split:output:0*
T0*(
_output_shapes
:         ю2
lstm_cell_14/SigmoidЛ
lstm_cell_14/Sigmoid_1Sigmoidlstm_cell_14/split:output:1*
T0*(
_output_shapes
:         ю2
lstm_cell_14/Sigmoid_1М
lstm_cell_14/mulMullstm_cell_14/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:         ю2
lstm_cell_14/mul~
lstm_cell_14/ReluRelulstm_cell_14/split:output:2*
T0*(
_output_shapes
:         ю2
lstm_cell_14/ReluЭ
lstm_cell_14/mul_1Mullstm_cell_14/Sigmoid:y:0lstm_cell_14/Relu:activations:0*
T0*(
_output_shapes
:         ю2
lstm_cell_14/mul_1Т
lstm_cell_14/add_1AddV2lstm_cell_14/mul:z:0lstm_cell_14/mul_1:z:0*
T0*(
_output_shapes
:         ю2
lstm_cell_14/add_1Л
lstm_cell_14/Sigmoid_2Sigmoidlstm_cell_14/split:output:3*
T0*(
_output_shapes
:         ю2
lstm_cell_14/Sigmoid_2}
lstm_cell_14/Relu_1Relulstm_cell_14/add_1:z:0*
T0*(
_output_shapes
:         ю2
lstm_cell_14/Relu_1б
lstm_cell_14/mul_2Mullstm_cell_14/Sigmoid_2:y:0!lstm_cell_14/Relu_1:activations:0*
T0*(
_output_shapes
:         ю2
lstm_cell_14/mul_2П
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ю   2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_14_matmul_readvariableop_resource-lstm_cell_14_matmul_1_readvariableop_resource,lstm_cell_14_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :         ю:         ю: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_26073239*
condR
while_cond_26073238*M
output_shapes<
:: : : : :         ю:         ю: : : : : *
parallel_iterations 2
while╡
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ю   22
0TensorArrayV2Stack/TensorListStack/element_shapeщ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:         ю*
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
:         ю*
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
:         ю2
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
:         ю2

Identity╚
NoOpNoOp$^lstm_cell_14/BiasAdd/ReadVariableOp#^lstm_cell_14/MatMul/ReadVariableOp%^lstm_cell_14/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         ]: : : 2J
#lstm_cell_14/BiasAdd/ReadVariableOp#lstm_cell_14/BiasAdd/ReadVariableOp2H
"lstm_cell_14/MatMul/ReadVariableOp"lstm_cell_14/MatMul/ReadVariableOp2L
$lstm_cell_14/MatMul_1/ReadVariableOp$lstm_cell_14/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:         ]
 
_user_specified_nameinputs
╦F
О
E__inference_lstm_14_layer_call_and_return_conditional_losses_26070160

inputs(
lstm_cell_14_26070078:	]╕)
lstm_cell_14_26070080:
ю╕$
lstm_cell_14_26070082:	╕
identityИв$lstm_cell_14/StatefulPartitionedCallвwhileD
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
B :ю2
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
B :ю2
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
:         ю2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :ю2
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
B :ю2
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
:         ю2	
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
strided_slice_2и
$lstm_cell_14/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_14_26070078lstm_cell_14_26070080lstm_cell_14_26070082*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:         ю:         ю:         ю*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_lstm_cell_14_layer_call_and_return_conditional_losses_260700132&
$lstm_cell_14/StatefulPartitionedCallП
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ю   2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_14_26070078lstm_cell_14_26070080lstm_cell_14_26070082*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :         ю:         ю: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_26070091*
condR
while_cond_26070090*M
output_shapes<
:: : : : :         ю:         ю: : : : : *
parallel_iterations 2
while╡
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ю   22
0TensorArrayV2Stack/TensorListStack/element_shapeЄ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:                  ю*
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
:         ю*
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
!:                  ю2
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
!:                  ю2

Identity}
NoOpNoOp%^lstm_cell_14/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  ]: : : 2L
$lstm_cell_14/StatefulPartitionedCall$lstm_cell_14/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :                  ]
 
_user_specified_nameinputs
Й
f
H__inference_dropout_14_layer_call_and_return_conditional_losses_26071222

inputs

identity_1_
IdentityIdentityinputs*
T0*,
_output_shapes
:         ю2

Identityn

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:         ю2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ю:T P
,
_output_shapes
:         ю
 
_user_specified_nameinputs
Й
f
H__inference_dropout_15_layer_call_and_return_conditional_losses_26074047

inputs

identity_1_
IdentityIdentityinputs*
T0*,
_output_shapes
:         Ц2

Identityn

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:         Ц2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         Ц:T P
,
_output_shapes
:         Ц
 
_user_specified_nameinputs
╢
╕
*__inference_lstm_14_layer_call_fn_26073367

inputs
unknown:	]╕
	unknown_0:
ю╕
	unknown_1:	╕
identityИвStatefulPartitionedCallЗ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ю*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_lstm_14_layer_call_and_return_conditional_losses_260718392
StatefulPartitionedCallА
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:         ю2

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
р
║
*__inference_lstm_14_layer_call_fn_26073334
inputs_0
unknown:	]╕
	unknown_0:
ю╕
	unknown_1:	╕
identityИвStatefulPartitionedCallТ
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:                  ю*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_lstm_14_layer_call_and_return_conditional_losses_260699502
StatefulPartitionedCallЙ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:                  ю2

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
Е&
є
while_body_26070091
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_00
while_lstm_cell_14_26070115_0:	]╕1
while_lstm_cell_14_26070117_0:
ю╕,
while_lstm_cell_14_26070119_0:	╕
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor.
while_lstm_cell_14_26070115:	]╕/
while_lstm_cell_14_26070117:
ю╕*
while_lstm_cell_14_26070119:	╕Ив*while/lstm_cell_14/StatefulPartitionedCall├
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
)while/TensorArrayV2Read/TensorListGetItemь
*while/lstm_cell_14/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_14_26070115_0while_lstm_cell_14_26070117_0while_lstm_cell_14_26070119_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:         ю:         ю:         ю*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_lstm_cell_14_layer_call_and_return_conditional_losses_260700132,
*while/lstm_cell_14/StatefulPartitionedCallў
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder3while/lstm_cell_14/StatefulPartitionedCall:output:0*
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
while/Identity_4Identity3while/lstm_cell_14/StatefulPartitionedCall:output:1^while/NoOp*
T0*(
_output_shapes
:         ю2
while/Identity_4е
while/Identity_5Identity3while/lstm_cell_14/StatefulPartitionedCall:output:2^while/NoOp*
T0*(
_output_shapes
:         ю2
while/Identity_5З

while/NoOpNoOp+^while/lstm_cell_14/StatefulPartitionedCall*"
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
while_lstm_cell_14_26070115while_lstm_cell_14_26070115_0"<
while_lstm_cell_14_26070117while_lstm_cell_14_26070117_0"<
while_lstm_cell_14_26070119while_lstm_cell_14_26070119_0"0
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :         ю:         ю: : : : : 2X
*while/lstm_cell_14/StatefulPartitionedCall*while/lstm_cell_14/StatefulPartitionedCall: 
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
:         ю:.*
(
_output_shapes
:         ю:

_output_shapes
: :

_output_shapes
: 
И&
ї
while_body_26070511
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_01
while_lstm_cell_15_26070535_0:
ю╪1
while_lstm_cell_15_26070537_0:
Ц╪,
while_lstm_cell_15_26070539_0:	╪
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor/
while_lstm_cell_15_26070535:
ю╪/
while_lstm_cell_15_26070537:
Ц╪*
while_lstm_cell_15_26070539:	╪Ив*while/lstm_cell_15/StatefulPartitionedCall├
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ю   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape╘
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:         ю*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemь
*while/lstm_cell_15/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_15_26070535_0while_lstm_cell_15_26070537_0while_lstm_cell_15_26070539_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:         Ц:         Ц:         Ц*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_lstm_cell_15_layer_call_and_return_conditional_losses_260704972,
*while/lstm_cell_15/StatefulPartitionedCallў
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder3while/lstm_cell_15/StatefulPartitionedCall:output:0*
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
while/Identity_4Identity3while/lstm_cell_15/StatefulPartitionedCall:output:1^while/NoOp*
T0*(
_output_shapes
:         Ц2
while/Identity_4е
while/Identity_5Identity3while/lstm_cell_15/StatefulPartitionedCall:output:2^while/NoOp*
T0*(
_output_shapes
:         Ц2
while/Identity_5З

while/NoOpNoOp+^while/lstm_cell_15/StatefulPartitionedCall*"
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
while_lstm_cell_15_26070535while_lstm_cell_15_26070535_0"<
while_lstm_cell_15_26070537while_lstm_cell_15_26070537_0"<
while_lstm_cell_15_26070539while_lstm_cell_15_26070539_0"0
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :         Ц:         Ц: : : : : 2X
*while/lstm_cell_15/StatefulPartitionedCall*while/lstm_cell_15/StatefulPartitionedCall: 
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
:         Ц:.*
(
_output_shapes
:         Ц:

_output_shapes
: :

_output_shapes
: 
╘!
¤
E__inference_dense_7_layer_call_and_return_conditional_losses_26071420

inputs4
!tensordot_readvariableop_resource:	Ц-
biasadd_readvariableop_resource:
identityИвBiasAdd/ReadVariableOpвTensordot/ReadVariableOpЧ
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes
:	Ц*
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
:         Ц2
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
:         Ц: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:T P
,
_output_shapes
:         Ц
 
_user_specified_nameinputs
┤?
╓
while_body_26071559
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
3while_lstm_cell_15_matmul_readvariableop_resource_0:
ю╪I
5while_lstm_cell_15_matmul_1_readvariableop_resource_0:
Ц╪C
4while_lstm_cell_15_biasadd_readvariableop_resource_0:	╪
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
1while_lstm_cell_15_matmul_readvariableop_resource:
ю╪G
3while_lstm_cell_15_matmul_1_readvariableop_resource:
Ц╪A
2while_lstm_cell_15_biasadd_readvariableop_resource:	╪Ив)while/lstm_cell_15/BiasAdd/ReadVariableOpв(while/lstm_cell_15/MatMul/ReadVariableOpв*while/lstm_cell_15/MatMul_1/ReadVariableOp├
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ю   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape╘
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:         ю*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem╩
(while/lstm_cell_15/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_15_matmul_readvariableop_resource_0* 
_output_shapes
:
ю╪*
dtype02*
(while/lstm_cell_15/MatMul/ReadVariableOp╫
while/lstm_cell_15/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_15/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╪2
while/lstm_cell_15/MatMul╨
*while/lstm_cell_15/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_15_matmul_1_readvariableop_resource_0* 
_output_shapes
:
Ц╪*
dtype02,
*while/lstm_cell_15/MatMul_1/ReadVariableOp└
while/lstm_cell_15/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_15/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╪2
while/lstm_cell_15/MatMul_1╕
while/lstm_cell_15/addAddV2#while/lstm_cell_15/MatMul:product:0%while/lstm_cell_15/MatMul_1:product:0*
T0*(
_output_shapes
:         ╪2
while/lstm_cell_15/add╚
)while/lstm_cell_15/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_15_biasadd_readvariableop_resource_0*
_output_shapes	
:╪*
dtype02+
)while/lstm_cell_15/BiasAdd/ReadVariableOp┼
while/lstm_cell_15/BiasAddBiasAddwhile/lstm_cell_15/add:z:01while/lstm_cell_15/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╪2
while/lstm_cell_15/BiasAddК
"while/lstm_cell_15/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_15/split/split_dimП
while/lstm_cell_15/splitSplit+while/lstm_cell_15/split/split_dim:output:0#while/lstm_cell_15/BiasAdd:output:0*
T0*d
_output_shapesR
P:         Ц:         Ц:         Ц:         Ц*
	num_split2
while/lstm_cell_15/splitЩ
while/lstm_cell_15/SigmoidSigmoid!while/lstm_cell_15/split:output:0*
T0*(
_output_shapes
:         Ц2
while/lstm_cell_15/SigmoidЭ
while/lstm_cell_15/Sigmoid_1Sigmoid!while/lstm_cell_15/split:output:1*
T0*(
_output_shapes
:         Ц2
while/lstm_cell_15/Sigmoid_1б
while/lstm_cell_15/mulMul while/lstm_cell_15/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:         Ц2
while/lstm_cell_15/mulР
while/lstm_cell_15/ReluRelu!while/lstm_cell_15/split:output:2*
T0*(
_output_shapes
:         Ц2
while/lstm_cell_15/Relu╡
while/lstm_cell_15/mul_1Mulwhile/lstm_cell_15/Sigmoid:y:0%while/lstm_cell_15/Relu:activations:0*
T0*(
_output_shapes
:         Ц2
while/lstm_cell_15/mul_1к
while/lstm_cell_15/add_1AddV2while/lstm_cell_15/mul:z:0while/lstm_cell_15/mul_1:z:0*
T0*(
_output_shapes
:         Ц2
while/lstm_cell_15/add_1Э
while/lstm_cell_15/Sigmoid_2Sigmoid!while/lstm_cell_15/split:output:3*
T0*(
_output_shapes
:         Ц2
while/lstm_cell_15/Sigmoid_2П
while/lstm_cell_15/Relu_1Reluwhile/lstm_cell_15/add_1:z:0*
T0*(
_output_shapes
:         Ц2
while/lstm_cell_15/Relu_1╣
while/lstm_cell_15/mul_2Mul while/lstm_cell_15/Sigmoid_2:y:0'while/lstm_cell_15/Relu_1:activations:0*
T0*(
_output_shapes
:         Ц2
while/lstm_cell_15/mul_2р
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_15/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_15/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:         Ц2
while/Identity_4О
while/Identity_5Identitywhile/lstm_cell_15/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:         Ц2
while/Identity_5▐

while/NoOpNoOp*^while/lstm_cell_15/BiasAdd/ReadVariableOp)^while/lstm_cell_15/MatMul/ReadVariableOp+^while/lstm_cell_15/MatMul_1/ReadVariableOp*"
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
2while_lstm_cell_15_biasadd_readvariableop_resource4while_lstm_cell_15_biasadd_readvariableop_resource_0"l
3while_lstm_cell_15_matmul_1_readvariableop_resource5while_lstm_cell_15_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_15_matmul_readvariableop_resource3while_lstm_cell_15_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :         Ц:         Ц: : : : : 2V
)while/lstm_cell_15/BiasAdd/ReadVariableOp)while/lstm_cell_15/BiasAdd/ReadVariableOp2T
(while/lstm_cell_15/MatMul/ReadVariableOp(while/lstm_cell_15/MatMul/ReadVariableOp2X
*while/lstm_cell_15/MatMul_1/ReadVariableOp*while/lstm_cell_15/MatMul_1/ReadVariableOp: 
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
:         Ц:.*
(
_output_shapes
:         Ц:

_output_shapes
: :

_output_shapes
: 
╣
╣
*__inference_lstm_15_layer_call_fn_26074042

inputs
unknown:
ю╪
	unknown_0:
Ц╪
	unknown_1:	╪
identityИвStatefulPartitionedCallЗ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         Ц*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_lstm_15_layer_call_and_return_conditional_losses_260716432
StatefulPartitionedCallА
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:         Ц2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:         ю: : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:         ю
 
_user_specified_nameinputs
╗
f
-__inference_dropout_15_layer_call_fn_26074069

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
:         Ц* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_dropout_15_layer_call_and_return_conditional_losses_260714762
StatefulPartitionedCallА
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:         Ц2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         Ц22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:         Ц
 
_user_specified_nameinputs
░]
ў
(sequential_7_lstm_15_while_body_26069680F
Bsequential_7_lstm_15_while_sequential_7_lstm_15_while_loop_counterL
Hsequential_7_lstm_15_while_sequential_7_lstm_15_while_maximum_iterations*
&sequential_7_lstm_15_while_placeholder,
(sequential_7_lstm_15_while_placeholder_1,
(sequential_7_lstm_15_while_placeholder_2,
(sequential_7_lstm_15_while_placeholder_3E
Asequential_7_lstm_15_while_sequential_7_lstm_15_strided_slice_1_0Б
}sequential_7_lstm_15_while_tensorarrayv2read_tensorlistgetitem_sequential_7_lstm_15_tensorarrayunstack_tensorlistfromtensor_0\
Hsequential_7_lstm_15_while_lstm_cell_15_matmul_readvariableop_resource_0:
ю╪^
Jsequential_7_lstm_15_while_lstm_cell_15_matmul_1_readvariableop_resource_0:
Ц╪X
Isequential_7_lstm_15_while_lstm_cell_15_biasadd_readvariableop_resource_0:	╪'
#sequential_7_lstm_15_while_identity)
%sequential_7_lstm_15_while_identity_1)
%sequential_7_lstm_15_while_identity_2)
%sequential_7_lstm_15_while_identity_3)
%sequential_7_lstm_15_while_identity_4)
%sequential_7_lstm_15_while_identity_5C
?sequential_7_lstm_15_while_sequential_7_lstm_15_strided_slice_1
{sequential_7_lstm_15_while_tensorarrayv2read_tensorlistgetitem_sequential_7_lstm_15_tensorarrayunstack_tensorlistfromtensorZ
Fsequential_7_lstm_15_while_lstm_cell_15_matmul_readvariableop_resource:
ю╪\
Hsequential_7_lstm_15_while_lstm_cell_15_matmul_1_readvariableop_resource:
Ц╪V
Gsequential_7_lstm_15_while_lstm_cell_15_biasadd_readvariableop_resource:	╪Ив>sequential_7/lstm_15/while/lstm_cell_15/BiasAdd/ReadVariableOpв=sequential_7/lstm_15/while/lstm_cell_15/MatMul/ReadVariableOpв?sequential_7/lstm_15/while/lstm_cell_15/MatMul_1/ReadVariableOpэ
Lsequential_7/lstm_15/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ю   2N
Lsequential_7/lstm_15/while/TensorArrayV2Read/TensorListGetItem/element_shape╥
>sequential_7/lstm_15/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem}sequential_7_lstm_15_while_tensorarrayv2read_tensorlistgetitem_sequential_7_lstm_15_tensorarrayunstack_tensorlistfromtensor_0&sequential_7_lstm_15_while_placeholderUsequential_7/lstm_15/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:         ю*
element_dtype02@
>sequential_7/lstm_15/while/TensorArrayV2Read/TensorListGetItemЙ
=sequential_7/lstm_15/while/lstm_cell_15/MatMul/ReadVariableOpReadVariableOpHsequential_7_lstm_15_while_lstm_cell_15_matmul_readvariableop_resource_0* 
_output_shapes
:
ю╪*
dtype02?
=sequential_7/lstm_15/while/lstm_cell_15/MatMul/ReadVariableOpл
.sequential_7/lstm_15/while/lstm_cell_15/MatMulMatMulEsequential_7/lstm_15/while/TensorArrayV2Read/TensorListGetItem:item:0Esequential_7/lstm_15/while/lstm_cell_15/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╪20
.sequential_7/lstm_15/while/lstm_cell_15/MatMulП
?sequential_7/lstm_15/while/lstm_cell_15/MatMul_1/ReadVariableOpReadVariableOpJsequential_7_lstm_15_while_lstm_cell_15_matmul_1_readvariableop_resource_0* 
_output_shapes
:
Ц╪*
dtype02A
?sequential_7/lstm_15/while/lstm_cell_15/MatMul_1/ReadVariableOpФ
0sequential_7/lstm_15/while/lstm_cell_15/MatMul_1MatMul(sequential_7_lstm_15_while_placeholder_2Gsequential_7/lstm_15/while/lstm_cell_15/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╪22
0sequential_7/lstm_15/while/lstm_cell_15/MatMul_1М
+sequential_7/lstm_15/while/lstm_cell_15/addAddV28sequential_7/lstm_15/while/lstm_cell_15/MatMul:product:0:sequential_7/lstm_15/while/lstm_cell_15/MatMul_1:product:0*
T0*(
_output_shapes
:         ╪2-
+sequential_7/lstm_15/while/lstm_cell_15/addЗ
>sequential_7/lstm_15/while/lstm_cell_15/BiasAdd/ReadVariableOpReadVariableOpIsequential_7_lstm_15_while_lstm_cell_15_biasadd_readvariableop_resource_0*
_output_shapes	
:╪*
dtype02@
>sequential_7/lstm_15/while/lstm_cell_15/BiasAdd/ReadVariableOpЩ
/sequential_7/lstm_15/while/lstm_cell_15/BiasAddBiasAdd/sequential_7/lstm_15/while/lstm_cell_15/add:z:0Fsequential_7/lstm_15/while/lstm_cell_15/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╪21
/sequential_7/lstm_15/while/lstm_cell_15/BiasAdd┤
7sequential_7/lstm_15/while/lstm_cell_15/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :29
7sequential_7/lstm_15/while/lstm_cell_15/split/split_dimу
-sequential_7/lstm_15/while/lstm_cell_15/splitSplit@sequential_7/lstm_15/while/lstm_cell_15/split/split_dim:output:08sequential_7/lstm_15/while/lstm_cell_15/BiasAdd:output:0*
T0*d
_output_shapesR
P:         Ц:         Ц:         Ц:         Ц*
	num_split2/
-sequential_7/lstm_15/while/lstm_cell_15/split╪
/sequential_7/lstm_15/while/lstm_cell_15/SigmoidSigmoid6sequential_7/lstm_15/while/lstm_cell_15/split:output:0*
T0*(
_output_shapes
:         Ц21
/sequential_7/lstm_15/while/lstm_cell_15/Sigmoid▄
1sequential_7/lstm_15/while/lstm_cell_15/Sigmoid_1Sigmoid6sequential_7/lstm_15/while/lstm_cell_15/split:output:1*
T0*(
_output_shapes
:         Ц23
1sequential_7/lstm_15/while/lstm_cell_15/Sigmoid_1ї
+sequential_7/lstm_15/while/lstm_cell_15/mulMul5sequential_7/lstm_15/while/lstm_cell_15/Sigmoid_1:y:0(sequential_7_lstm_15_while_placeholder_3*
T0*(
_output_shapes
:         Ц2-
+sequential_7/lstm_15/while/lstm_cell_15/mul╧
,sequential_7/lstm_15/while/lstm_cell_15/ReluRelu6sequential_7/lstm_15/while/lstm_cell_15/split:output:2*
T0*(
_output_shapes
:         Ц2.
,sequential_7/lstm_15/while/lstm_cell_15/ReluЙ
-sequential_7/lstm_15/while/lstm_cell_15/mul_1Mul3sequential_7/lstm_15/while/lstm_cell_15/Sigmoid:y:0:sequential_7/lstm_15/while/lstm_cell_15/Relu:activations:0*
T0*(
_output_shapes
:         Ц2/
-sequential_7/lstm_15/while/lstm_cell_15/mul_1■
-sequential_7/lstm_15/while/lstm_cell_15/add_1AddV2/sequential_7/lstm_15/while/lstm_cell_15/mul:z:01sequential_7/lstm_15/while/lstm_cell_15/mul_1:z:0*
T0*(
_output_shapes
:         Ц2/
-sequential_7/lstm_15/while/lstm_cell_15/add_1▄
1sequential_7/lstm_15/while/lstm_cell_15/Sigmoid_2Sigmoid6sequential_7/lstm_15/while/lstm_cell_15/split:output:3*
T0*(
_output_shapes
:         Ц23
1sequential_7/lstm_15/while/lstm_cell_15/Sigmoid_2╬
.sequential_7/lstm_15/while/lstm_cell_15/Relu_1Relu1sequential_7/lstm_15/while/lstm_cell_15/add_1:z:0*
T0*(
_output_shapes
:         Ц20
.sequential_7/lstm_15/while/lstm_cell_15/Relu_1Н
-sequential_7/lstm_15/while/lstm_cell_15/mul_2Mul5sequential_7/lstm_15/while/lstm_cell_15/Sigmoid_2:y:0<sequential_7/lstm_15/while/lstm_cell_15/Relu_1:activations:0*
T0*(
_output_shapes
:         Ц2/
-sequential_7/lstm_15/while/lstm_cell_15/mul_2╔
?sequential_7/lstm_15/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem(sequential_7_lstm_15_while_placeholder_1&sequential_7_lstm_15_while_placeholder1sequential_7/lstm_15/while/lstm_cell_15/mul_2:z:0*
_output_shapes
: *
element_dtype02A
?sequential_7/lstm_15/while/TensorArrayV2Write/TensorListSetItemЖ
 sequential_7/lstm_15/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2"
 sequential_7/lstm_15/while/add/y╜
sequential_7/lstm_15/while/addAddV2&sequential_7_lstm_15_while_placeholder)sequential_7/lstm_15/while/add/y:output:0*
T0*
_output_shapes
: 2 
sequential_7/lstm_15/while/addК
"sequential_7/lstm_15/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2$
"sequential_7/lstm_15/while/add_1/y▀
 sequential_7/lstm_15/while/add_1AddV2Bsequential_7_lstm_15_while_sequential_7_lstm_15_while_loop_counter+sequential_7/lstm_15/while/add_1/y:output:0*
T0*
_output_shapes
: 2"
 sequential_7/lstm_15/while/add_1┐
#sequential_7/lstm_15/while/IdentityIdentity$sequential_7/lstm_15/while/add_1:z:0 ^sequential_7/lstm_15/while/NoOp*
T0*
_output_shapes
: 2%
#sequential_7/lstm_15/while/Identityч
%sequential_7/lstm_15/while/Identity_1IdentityHsequential_7_lstm_15_while_sequential_7_lstm_15_while_maximum_iterations ^sequential_7/lstm_15/while/NoOp*
T0*
_output_shapes
: 2'
%sequential_7/lstm_15/while/Identity_1┴
%sequential_7/lstm_15/while/Identity_2Identity"sequential_7/lstm_15/while/add:z:0 ^sequential_7/lstm_15/while/NoOp*
T0*
_output_shapes
: 2'
%sequential_7/lstm_15/while/Identity_2ю
%sequential_7/lstm_15/while/Identity_3IdentityOsequential_7/lstm_15/while/TensorArrayV2Write/TensorListSetItem:output_handle:0 ^sequential_7/lstm_15/while/NoOp*
T0*
_output_shapes
: 2'
%sequential_7/lstm_15/while/Identity_3т
%sequential_7/lstm_15/while/Identity_4Identity1sequential_7/lstm_15/while/lstm_cell_15/mul_2:z:0 ^sequential_7/lstm_15/while/NoOp*
T0*(
_output_shapes
:         Ц2'
%sequential_7/lstm_15/while/Identity_4т
%sequential_7/lstm_15/while/Identity_5Identity1sequential_7/lstm_15/while/lstm_cell_15/add_1:z:0 ^sequential_7/lstm_15/while/NoOp*
T0*(
_output_shapes
:         Ц2'
%sequential_7/lstm_15/while/Identity_5╟
sequential_7/lstm_15/while/NoOpNoOp?^sequential_7/lstm_15/while/lstm_cell_15/BiasAdd/ReadVariableOp>^sequential_7/lstm_15/while/lstm_cell_15/MatMul/ReadVariableOp@^sequential_7/lstm_15/while/lstm_cell_15/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2!
sequential_7/lstm_15/while/NoOp"S
#sequential_7_lstm_15_while_identity,sequential_7/lstm_15/while/Identity:output:0"W
%sequential_7_lstm_15_while_identity_1.sequential_7/lstm_15/while/Identity_1:output:0"W
%sequential_7_lstm_15_while_identity_2.sequential_7/lstm_15/while/Identity_2:output:0"W
%sequential_7_lstm_15_while_identity_3.sequential_7/lstm_15/while/Identity_3:output:0"W
%sequential_7_lstm_15_while_identity_4.sequential_7/lstm_15/while/Identity_4:output:0"W
%sequential_7_lstm_15_while_identity_5.sequential_7/lstm_15/while/Identity_5:output:0"Ф
Gsequential_7_lstm_15_while_lstm_cell_15_biasadd_readvariableop_resourceIsequential_7_lstm_15_while_lstm_cell_15_biasadd_readvariableop_resource_0"Ц
Hsequential_7_lstm_15_while_lstm_cell_15_matmul_1_readvariableop_resourceJsequential_7_lstm_15_while_lstm_cell_15_matmul_1_readvariableop_resource_0"Т
Fsequential_7_lstm_15_while_lstm_cell_15_matmul_readvariableop_resourceHsequential_7_lstm_15_while_lstm_cell_15_matmul_readvariableop_resource_0"Д
?sequential_7_lstm_15_while_sequential_7_lstm_15_strided_slice_1Asequential_7_lstm_15_while_sequential_7_lstm_15_strided_slice_1_0"№
{sequential_7_lstm_15_while_tensorarrayv2read_tensorlistgetitem_sequential_7_lstm_15_tensorarrayunstack_tensorlistfromtensor}sequential_7_lstm_15_while_tensorarrayv2read_tensorlistgetitem_sequential_7_lstm_15_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :         Ц:         Ц: : : : : 2А
>sequential_7/lstm_15/while/lstm_cell_15/BiasAdd/ReadVariableOp>sequential_7/lstm_15/while/lstm_cell_15/BiasAdd/ReadVariableOp2~
=sequential_7/lstm_15/while/lstm_cell_15/MatMul/ReadVariableOp=sequential_7/lstm_15/while/lstm_cell_15/MatMul/ReadVariableOp2В
?sequential_7/lstm_15/while/lstm_cell_15/MatMul_1/ReadVariableOp?sequential_7/lstm_15/while/lstm_cell_15/MatMul_1/ReadVariableOp: 
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
:         Ц:.*
(
_output_shapes
:         Ц:

_output_shapes
: :

_output_shapes
: 
щJ
╓

lstm_15_while_body_26072224,
(lstm_15_while_lstm_15_while_loop_counter2
.lstm_15_while_lstm_15_while_maximum_iterations
lstm_15_while_placeholder
lstm_15_while_placeholder_1
lstm_15_while_placeholder_2
lstm_15_while_placeholder_3+
'lstm_15_while_lstm_15_strided_slice_1_0g
clstm_15_while_tensorarrayv2read_tensorlistgetitem_lstm_15_tensorarrayunstack_tensorlistfromtensor_0O
;lstm_15_while_lstm_cell_15_matmul_readvariableop_resource_0:
ю╪Q
=lstm_15_while_lstm_cell_15_matmul_1_readvariableop_resource_0:
Ц╪K
<lstm_15_while_lstm_cell_15_biasadd_readvariableop_resource_0:	╪
lstm_15_while_identity
lstm_15_while_identity_1
lstm_15_while_identity_2
lstm_15_while_identity_3
lstm_15_while_identity_4
lstm_15_while_identity_5)
%lstm_15_while_lstm_15_strided_slice_1e
alstm_15_while_tensorarrayv2read_tensorlistgetitem_lstm_15_tensorarrayunstack_tensorlistfromtensorM
9lstm_15_while_lstm_cell_15_matmul_readvariableop_resource:
ю╪O
;lstm_15_while_lstm_cell_15_matmul_1_readvariableop_resource:
Ц╪I
:lstm_15_while_lstm_cell_15_biasadd_readvariableop_resource:	╪Ив1lstm_15/while/lstm_cell_15/BiasAdd/ReadVariableOpв0lstm_15/while/lstm_cell_15/MatMul/ReadVariableOpв2lstm_15/while/lstm_cell_15/MatMul_1/ReadVariableOp╙
?lstm_15/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ю   2A
?lstm_15/while/TensorArrayV2Read/TensorListGetItem/element_shapeД
1lstm_15/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemclstm_15_while_tensorarrayv2read_tensorlistgetitem_lstm_15_tensorarrayunstack_tensorlistfromtensor_0lstm_15_while_placeholderHlstm_15/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:         ю*
element_dtype023
1lstm_15/while/TensorArrayV2Read/TensorListGetItemт
0lstm_15/while/lstm_cell_15/MatMul/ReadVariableOpReadVariableOp;lstm_15_while_lstm_cell_15_matmul_readvariableop_resource_0* 
_output_shapes
:
ю╪*
dtype022
0lstm_15/while/lstm_cell_15/MatMul/ReadVariableOpў
!lstm_15/while/lstm_cell_15/MatMulMatMul8lstm_15/while/TensorArrayV2Read/TensorListGetItem:item:08lstm_15/while/lstm_cell_15/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╪2#
!lstm_15/while/lstm_cell_15/MatMulш
2lstm_15/while/lstm_cell_15/MatMul_1/ReadVariableOpReadVariableOp=lstm_15_while_lstm_cell_15_matmul_1_readvariableop_resource_0* 
_output_shapes
:
Ц╪*
dtype024
2lstm_15/while/lstm_cell_15/MatMul_1/ReadVariableOpр
#lstm_15/while/lstm_cell_15/MatMul_1MatMullstm_15_while_placeholder_2:lstm_15/while/lstm_cell_15/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╪2%
#lstm_15/while/lstm_cell_15/MatMul_1╪
lstm_15/while/lstm_cell_15/addAddV2+lstm_15/while/lstm_cell_15/MatMul:product:0-lstm_15/while/lstm_cell_15/MatMul_1:product:0*
T0*(
_output_shapes
:         ╪2 
lstm_15/while/lstm_cell_15/addр
1lstm_15/while/lstm_cell_15/BiasAdd/ReadVariableOpReadVariableOp<lstm_15_while_lstm_cell_15_biasadd_readvariableop_resource_0*
_output_shapes	
:╪*
dtype023
1lstm_15/while/lstm_cell_15/BiasAdd/ReadVariableOpх
"lstm_15/while/lstm_cell_15/BiasAddBiasAdd"lstm_15/while/lstm_cell_15/add:z:09lstm_15/while/lstm_cell_15/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╪2$
"lstm_15/while/lstm_cell_15/BiasAddЪ
*lstm_15/while/lstm_cell_15/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2,
*lstm_15/while/lstm_cell_15/split/split_dimп
 lstm_15/while/lstm_cell_15/splitSplit3lstm_15/while/lstm_cell_15/split/split_dim:output:0+lstm_15/while/lstm_cell_15/BiasAdd:output:0*
T0*d
_output_shapesR
P:         Ц:         Ц:         Ц:         Ц*
	num_split2"
 lstm_15/while/lstm_cell_15/split▒
"lstm_15/while/lstm_cell_15/SigmoidSigmoid)lstm_15/while/lstm_cell_15/split:output:0*
T0*(
_output_shapes
:         Ц2$
"lstm_15/while/lstm_cell_15/Sigmoid╡
$lstm_15/while/lstm_cell_15/Sigmoid_1Sigmoid)lstm_15/while/lstm_cell_15/split:output:1*
T0*(
_output_shapes
:         Ц2&
$lstm_15/while/lstm_cell_15/Sigmoid_1┴
lstm_15/while/lstm_cell_15/mulMul(lstm_15/while/lstm_cell_15/Sigmoid_1:y:0lstm_15_while_placeholder_3*
T0*(
_output_shapes
:         Ц2 
lstm_15/while/lstm_cell_15/mulи
lstm_15/while/lstm_cell_15/ReluRelu)lstm_15/while/lstm_cell_15/split:output:2*
T0*(
_output_shapes
:         Ц2!
lstm_15/while/lstm_cell_15/Relu╒
 lstm_15/while/lstm_cell_15/mul_1Mul&lstm_15/while/lstm_cell_15/Sigmoid:y:0-lstm_15/while/lstm_cell_15/Relu:activations:0*
T0*(
_output_shapes
:         Ц2"
 lstm_15/while/lstm_cell_15/mul_1╩
 lstm_15/while/lstm_cell_15/add_1AddV2"lstm_15/while/lstm_cell_15/mul:z:0$lstm_15/while/lstm_cell_15/mul_1:z:0*
T0*(
_output_shapes
:         Ц2"
 lstm_15/while/lstm_cell_15/add_1╡
$lstm_15/while/lstm_cell_15/Sigmoid_2Sigmoid)lstm_15/while/lstm_cell_15/split:output:3*
T0*(
_output_shapes
:         Ц2&
$lstm_15/while/lstm_cell_15/Sigmoid_2з
!lstm_15/while/lstm_cell_15/Relu_1Relu$lstm_15/while/lstm_cell_15/add_1:z:0*
T0*(
_output_shapes
:         Ц2#
!lstm_15/while/lstm_cell_15/Relu_1┘
 lstm_15/while/lstm_cell_15/mul_2Mul(lstm_15/while/lstm_cell_15/Sigmoid_2:y:0/lstm_15/while/lstm_cell_15/Relu_1:activations:0*
T0*(
_output_shapes
:         Ц2"
 lstm_15/while/lstm_cell_15/mul_2И
2lstm_15/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_15_while_placeholder_1lstm_15_while_placeholder$lstm_15/while/lstm_cell_15/mul_2:z:0*
_output_shapes
: *
element_dtype024
2lstm_15/while/TensorArrayV2Write/TensorListSetIteml
lstm_15/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_15/while/add/yЙ
lstm_15/while/addAddV2lstm_15_while_placeholderlstm_15/while/add/y:output:0*
T0*
_output_shapes
: 2
lstm_15/while/addp
lstm_15/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_15/while/add_1/yЮ
lstm_15/while/add_1AddV2(lstm_15_while_lstm_15_while_loop_counterlstm_15/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lstm_15/while/add_1Л
lstm_15/while/IdentityIdentitylstm_15/while/add_1:z:0^lstm_15/while/NoOp*
T0*
_output_shapes
: 2
lstm_15/while/Identityж
lstm_15/while/Identity_1Identity.lstm_15_while_lstm_15_while_maximum_iterations^lstm_15/while/NoOp*
T0*
_output_shapes
: 2
lstm_15/while/Identity_1Н
lstm_15/while/Identity_2Identitylstm_15/while/add:z:0^lstm_15/while/NoOp*
T0*
_output_shapes
: 2
lstm_15/while/Identity_2║
lstm_15/while/Identity_3IdentityBlstm_15/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_15/while/NoOp*
T0*
_output_shapes
: 2
lstm_15/while/Identity_3о
lstm_15/while/Identity_4Identity$lstm_15/while/lstm_cell_15/mul_2:z:0^lstm_15/while/NoOp*
T0*(
_output_shapes
:         Ц2
lstm_15/while/Identity_4о
lstm_15/while/Identity_5Identity$lstm_15/while/lstm_cell_15/add_1:z:0^lstm_15/while/NoOp*
T0*(
_output_shapes
:         Ц2
lstm_15/while/Identity_5Ж
lstm_15/while/NoOpNoOp2^lstm_15/while/lstm_cell_15/BiasAdd/ReadVariableOp1^lstm_15/while/lstm_cell_15/MatMul/ReadVariableOp3^lstm_15/while/lstm_cell_15/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
lstm_15/while/NoOp"9
lstm_15_while_identitylstm_15/while/Identity:output:0"=
lstm_15_while_identity_1!lstm_15/while/Identity_1:output:0"=
lstm_15_while_identity_2!lstm_15/while/Identity_2:output:0"=
lstm_15_while_identity_3!lstm_15/while/Identity_3:output:0"=
lstm_15_while_identity_4!lstm_15/while/Identity_4:output:0"=
lstm_15_while_identity_5!lstm_15/while/Identity_5:output:0"P
%lstm_15_while_lstm_15_strided_slice_1'lstm_15_while_lstm_15_strided_slice_1_0"z
:lstm_15_while_lstm_cell_15_biasadd_readvariableop_resource<lstm_15_while_lstm_cell_15_biasadd_readvariableop_resource_0"|
;lstm_15_while_lstm_cell_15_matmul_1_readvariableop_resource=lstm_15_while_lstm_cell_15_matmul_1_readvariableop_resource_0"x
9lstm_15_while_lstm_cell_15_matmul_readvariableop_resource;lstm_15_while_lstm_cell_15_matmul_readvariableop_resource_0"╚
alstm_15_while_tensorarrayv2read_tensorlistgetitem_lstm_15_tensorarrayunstack_tensorlistfromtensorclstm_15_while_tensorarrayv2read_tensorlistgetitem_lstm_15_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :         Ц:         Ц: : : : : 2f
1lstm_15/while/lstm_cell_15/BiasAdd/ReadVariableOp1lstm_15/while/lstm_cell_15/BiasAdd/ReadVariableOp2d
0lstm_15/while/lstm_cell_15/MatMul/ReadVariableOp0lstm_15/while/lstm_cell_15/MatMul/ReadVariableOp2h
2lstm_15/while/lstm_cell_15/MatMul_1/ReadVariableOp2lstm_15/while/lstm_cell_15/MatMul_1/ReadVariableOp: 
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
:         Ц:.*
(
_output_shapes
:         Ц:

_output_shapes
: :

_output_shapes
: 
хJ
╘

lstm_14_while_body_26072403,
(lstm_14_while_lstm_14_while_loop_counter2
.lstm_14_while_lstm_14_while_maximum_iterations
lstm_14_while_placeholder
lstm_14_while_placeholder_1
lstm_14_while_placeholder_2
lstm_14_while_placeholder_3+
'lstm_14_while_lstm_14_strided_slice_1_0g
clstm_14_while_tensorarrayv2read_tensorlistgetitem_lstm_14_tensorarrayunstack_tensorlistfromtensor_0N
;lstm_14_while_lstm_cell_14_matmul_readvariableop_resource_0:	]╕Q
=lstm_14_while_lstm_cell_14_matmul_1_readvariableop_resource_0:
ю╕K
<lstm_14_while_lstm_cell_14_biasadd_readvariableop_resource_0:	╕
lstm_14_while_identity
lstm_14_while_identity_1
lstm_14_while_identity_2
lstm_14_while_identity_3
lstm_14_while_identity_4
lstm_14_while_identity_5)
%lstm_14_while_lstm_14_strided_slice_1e
alstm_14_while_tensorarrayv2read_tensorlistgetitem_lstm_14_tensorarrayunstack_tensorlistfromtensorL
9lstm_14_while_lstm_cell_14_matmul_readvariableop_resource:	]╕O
;lstm_14_while_lstm_cell_14_matmul_1_readvariableop_resource:
ю╕I
:lstm_14_while_lstm_cell_14_biasadd_readvariableop_resource:	╕Ив1lstm_14/while/lstm_cell_14/BiasAdd/ReadVariableOpв0lstm_14/while/lstm_cell_14/MatMul/ReadVariableOpв2lstm_14/while/lstm_cell_14/MatMul_1/ReadVariableOp╙
?lstm_14/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ]   2A
?lstm_14/while/TensorArrayV2Read/TensorListGetItem/element_shapeГ
1lstm_14/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemclstm_14_while_tensorarrayv2read_tensorlistgetitem_lstm_14_tensorarrayunstack_tensorlistfromtensor_0lstm_14_while_placeholderHlstm_14/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         ]*
element_dtype023
1lstm_14/while/TensorArrayV2Read/TensorListGetItemс
0lstm_14/while/lstm_cell_14/MatMul/ReadVariableOpReadVariableOp;lstm_14_while_lstm_cell_14_matmul_readvariableop_resource_0*
_output_shapes
:	]╕*
dtype022
0lstm_14/while/lstm_cell_14/MatMul/ReadVariableOpў
!lstm_14/while/lstm_cell_14/MatMulMatMul8lstm_14/while/TensorArrayV2Read/TensorListGetItem:item:08lstm_14/while/lstm_cell_14/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╕2#
!lstm_14/while/lstm_cell_14/MatMulш
2lstm_14/while/lstm_cell_14/MatMul_1/ReadVariableOpReadVariableOp=lstm_14_while_lstm_cell_14_matmul_1_readvariableop_resource_0* 
_output_shapes
:
ю╕*
dtype024
2lstm_14/while/lstm_cell_14/MatMul_1/ReadVariableOpр
#lstm_14/while/lstm_cell_14/MatMul_1MatMullstm_14_while_placeholder_2:lstm_14/while/lstm_cell_14/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╕2%
#lstm_14/while/lstm_cell_14/MatMul_1╪
lstm_14/while/lstm_cell_14/addAddV2+lstm_14/while/lstm_cell_14/MatMul:product:0-lstm_14/while/lstm_cell_14/MatMul_1:product:0*
T0*(
_output_shapes
:         ╕2 
lstm_14/while/lstm_cell_14/addр
1lstm_14/while/lstm_cell_14/BiasAdd/ReadVariableOpReadVariableOp<lstm_14_while_lstm_cell_14_biasadd_readvariableop_resource_0*
_output_shapes	
:╕*
dtype023
1lstm_14/while/lstm_cell_14/BiasAdd/ReadVariableOpх
"lstm_14/while/lstm_cell_14/BiasAddBiasAdd"lstm_14/while/lstm_cell_14/add:z:09lstm_14/while/lstm_cell_14/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╕2$
"lstm_14/while/lstm_cell_14/BiasAddЪ
*lstm_14/while/lstm_cell_14/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2,
*lstm_14/while/lstm_cell_14/split/split_dimп
 lstm_14/while/lstm_cell_14/splitSplit3lstm_14/while/lstm_cell_14/split/split_dim:output:0+lstm_14/while/lstm_cell_14/BiasAdd:output:0*
T0*d
_output_shapesR
P:         ю:         ю:         ю:         ю*
	num_split2"
 lstm_14/while/lstm_cell_14/split▒
"lstm_14/while/lstm_cell_14/SigmoidSigmoid)lstm_14/while/lstm_cell_14/split:output:0*
T0*(
_output_shapes
:         ю2$
"lstm_14/while/lstm_cell_14/Sigmoid╡
$lstm_14/while/lstm_cell_14/Sigmoid_1Sigmoid)lstm_14/while/lstm_cell_14/split:output:1*
T0*(
_output_shapes
:         ю2&
$lstm_14/while/lstm_cell_14/Sigmoid_1┴
lstm_14/while/lstm_cell_14/mulMul(lstm_14/while/lstm_cell_14/Sigmoid_1:y:0lstm_14_while_placeholder_3*
T0*(
_output_shapes
:         ю2 
lstm_14/while/lstm_cell_14/mulи
lstm_14/while/lstm_cell_14/ReluRelu)lstm_14/while/lstm_cell_14/split:output:2*
T0*(
_output_shapes
:         ю2!
lstm_14/while/lstm_cell_14/Relu╒
 lstm_14/while/lstm_cell_14/mul_1Mul&lstm_14/while/lstm_cell_14/Sigmoid:y:0-lstm_14/while/lstm_cell_14/Relu:activations:0*
T0*(
_output_shapes
:         ю2"
 lstm_14/while/lstm_cell_14/mul_1╩
 lstm_14/while/lstm_cell_14/add_1AddV2"lstm_14/while/lstm_cell_14/mul:z:0$lstm_14/while/lstm_cell_14/mul_1:z:0*
T0*(
_output_shapes
:         ю2"
 lstm_14/while/lstm_cell_14/add_1╡
$lstm_14/while/lstm_cell_14/Sigmoid_2Sigmoid)lstm_14/while/lstm_cell_14/split:output:3*
T0*(
_output_shapes
:         ю2&
$lstm_14/while/lstm_cell_14/Sigmoid_2з
!lstm_14/while/lstm_cell_14/Relu_1Relu$lstm_14/while/lstm_cell_14/add_1:z:0*
T0*(
_output_shapes
:         ю2#
!lstm_14/while/lstm_cell_14/Relu_1┘
 lstm_14/while/lstm_cell_14/mul_2Mul(lstm_14/while/lstm_cell_14/Sigmoid_2:y:0/lstm_14/while/lstm_cell_14/Relu_1:activations:0*
T0*(
_output_shapes
:         ю2"
 lstm_14/while/lstm_cell_14/mul_2И
2lstm_14/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_14_while_placeholder_1lstm_14_while_placeholder$lstm_14/while/lstm_cell_14/mul_2:z:0*
_output_shapes
: *
element_dtype024
2lstm_14/while/TensorArrayV2Write/TensorListSetIteml
lstm_14/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_14/while/add/yЙ
lstm_14/while/addAddV2lstm_14_while_placeholderlstm_14/while/add/y:output:0*
T0*
_output_shapes
: 2
lstm_14/while/addp
lstm_14/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_14/while/add_1/yЮ
lstm_14/while/add_1AddV2(lstm_14_while_lstm_14_while_loop_counterlstm_14/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lstm_14/while/add_1Л
lstm_14/while/IdentityIdentitylstm_14/while/add_1:z:0^lstm_14/while/NoOp*
T0*
_output_shapes
: 2
lstm_14/while/Identityж
lstm_14/while/Identity_1Identity.lstm_14_while_lstm_14_while_maximum_iterations^lstm_14/while/NoOp*
T0*
_output_shapes
: 2
lstm_14/while/Identity_1Н
lstm_14/while/Identity_2Identitylstm_14/while/add:z:0^lstm_14/while/NoOp*
T0*
_output_shapes
: 2
lstm_14/while/Identity_2║
lstm_14/while/Identity_3IdentityBlstm_14/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_14/while/NoOp*
T0*
_output_shapes
: 2
lstm_14/while/Identity_3о
lstm_14/while/Identity_4Identity$lstm_14/while/lstm_cell_14/mul_2:z:0^lstm_14/while/NoOp*
T0*(
_output_shapes
:         ю2
lstm_14/while/Identity_4о
lstm_14/while/Identity_5Identity$lstm_14/while/lstm_cell_14/add_1:z:0^lstm_14/while/NoOp*
T0*(
_output_shapes
:         ю2
lstm_14/while/Identity_5Ж
lstm_14/while/NoOpNoOp2^lstm_14/while/lstm_cell_14/BiasAdd/ReadVariableOp1^lstm_14/while/lstm_cell_14/MatMul/ReadVariableOp3^lstm_14/while/lstm_cell_14/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
lstm_14/while/NoOp"9
lstm_14_while_identitylstm_14/while/Identity:output:0"=
lstm_14_while_identity_1!lstm_14/while/Identity_1:output:0"=
lstm_14_while_identity_2!lstm_14/while/Identity_2:output:0"=
lstm_14_while_identity_3!lstm_14/while/Identity_3:output:0"=
lstm_14_while_identity_4!lstm_14/while/Identity_4:output:0"=
lstm_14_while_identity_5!lstm_14/while/Identity_5:output:0"P
%lstm_14_while_lstm_14_strided_slice_1'lstm_14_while_lstm_14_strided_slice_1_0"z
:lstm_14_while_lstm_cell_14_biasadd_readvariableop_resource<lstm_14_while_lstm_cell_14_biasadd_readvariableop_resource_0"|
;lstm_14_while_lstm_cell_14_matmul_1_readvariableop_resource=lstm_14_while_lstm_cell_14_matmul_1_readvariableop_resource_0"x
9lstm_14_while_lstm_cell_14_matmul_readvariableop_resource;lstm_14_while_lstm_cell_14_matmul_readvariableop_resource_0"╚
alstm_14_while_tensorarrayv2read_tensorlistgetitem_lstm_14_tensorarrayunstack_tensorlistfromtensorclstm_14_while_tensorarrayv2read_tensorlistgetitem_lstm_14_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :         ю:         ю: : : : : 2f
1lstm_14/while/lstm_cell_14/BiasAdd/ReadVariableOp1lstm_14/while/lstm_cell_14/BiasAdd/ReadVariableOp2d
0lstm_14/while/lstm_cell_14/MatMul/ReadVariableOp0lstm_14/while/lstm_cell_14/MatMul/ReadVariableOp2h
2lstm_14/while/lstm_cell_14/MatMul_1/ReadVariableOp2lstm_14/while/lstm_cell_14/MatMul_1/ReadVariableOp: 
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
:         ю:.*
(
_output_shapes
:         ю:

_output_shapes
: :

_output_shapes
: 
┤?
╓
while_body_26073763
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
3while_lstm_cell_15_matmul_readvariableop_resource_0:
ю╪I
5while_lstm_cell_15_matmul_1_readvariableop_resource_0:
Ц╪C
4while_lstm_cell_15_biasadd_readvariableop_resource_0:	╪
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
1while_lstm_cell_15_matmul_readvariableop_resource:
ю╪G
3while_lstm_cell_15_matmul_1_readvariableop_resource:
Ц╪A
2while_lstm_cell_15_biasadd_readvariableop_resource:	╪Ив)while/lstm_cell_15/BiasAdd/ReadVariableOpв(while/lstm_cell_15/MatMul/ReadVariableOpв*while/lstm_cell_15/MatMul_1/ReadVariableOp├
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ю   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape╘
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:         ю*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem╩
(while/lstm_cell_15/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_15_matmul_readvariableop_resource_0* 
_output_shapes
:
ю╪*
dtype02*
(while/lstm_cell_15/MatMul/ReadVariableOp╫
while/lstm_cell_15/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_15/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╪2
while/lstm_cell_15/MatMul╨
*while/lstm_cell_15/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_15_matmul_1_readvariableop_resource_0* 
_output_shapes
:
Ц╪*
dtype02,
*while/lstm_cell_15/MatMul_1/ReadVariableOp└
while/lstm_cell_15/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_15/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╪2
while/lstm_cell_15/MatMul_1╕
while/lstm_cell_15/addAddV2#while/lstm_cell_15/MatMul:product:0%while/lstm_cell_15/MatMul_1:product:0*
T0*(
_output_shapes
:         ╪2
while/lstm_cell_15/add╚
)while/lstm_cell_15/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_15_biasadd_readvariableop_resource_0*
_output_shapes	
:╪*
dtype02+
)while/lstm_cell_15/BiasAdd/ReadVariableOp┼
while/lstm_cell_15/BiasAddBiasAddwhile/lstm_cell_15/add:z:01while/lstm_cell_15/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╪2
while/lstm_cell_15/BiasAddК
"while/lstm_cell_15/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_15/split/split_dimП
while/lstm_cell_15/splitSplit+while/lstm_cell_15/split/split_dim:output:0#while/lstm_cell_15/BiasAdd:output:0*
T0*d
_output_shapesR
P:         Ц:         Ц:         Ц:         Ц*
	num_split2
while/lstm_cell_15/splitЩ
while/lstm_cell_15/SigmoidSigmoid!while/lstm_cell_15/split:output:0*
T0*(
_output_shapes
:         Ц2
while/lstm_cell_15/SigmoidЭ
while/lstm_cell_15/Sigmoid_1Sigmoid!while/lstm_cell_15/split:output:1*
T0*(
_output_shapes
:         Ц2
while/lstm_cell_15/Sigmoid_1б
while/lstm_cell_15/mulMul while/lstm_cell_15/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:         Ц2
while/lstm_cell_15/mulР
while/lstm_cell_15/ReluRelu!while/lstm_cell_15/split:output:2*
T0*(
_output_shapes
:         Ц2
while/lstm_cell_15/Relu╡
while/lstm_cell_15/mul_1Mulwhile/lstm_cell_15/Sigmoid:y:0%while/lstm_cell_15/Relu:activations:0*
T0*(
_output_shapes
:         Ц2
while/lstm_cell_15/mul_1к
while/lstm_cell_15/add_1AddV2while/lstm_cell_15/mul:z:0while/lstm_cell_15/mul_1:z:0*
T0*(
_output_shapes
:         Ц2
while/lstm_cell_15/add_1Э
while/lstm_cell_15/Sigmoid_2Sigmoid!while/lstm_cell_15/split:output:3*
T0*(
_output_shapes
:         Ц2
while/lstm_cell_15/Sigmoid_2П
while/lstm_cell_15/Relu_1Reluwhile/lstm_cell_15/add_1:z:0*
T0*(
_output_shapes
:         Ц2
while/lstm_cell_15/Relu_1╣
while/lstm_cell_15/mul_2Mul while/lstm_cell_15/Sigmoid_2:y:0'while/lstm_cell_15/Relu_1:activations:0*
T0*(
_output_shapes
:         Ц2
while/lstm_cell_15/mul_2р
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_15/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_15/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:         Ц2
while/Identity_4О
while/Identity_5Identitywhile/lstm_cell_15/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:         Ц2
while/Identity_5▐

while/NoOpNoOp*^while/lstm_cell_15/BiasAdd/ReadVariableOp)^while/lstm_cell_15/MatMul/ReadVariableOp+^while/lstm_cell_15/MatMul_1/ReadVariableOp*"
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
2while_lstm_cell_15_biasadd_readvariableop_resource4while_lstm_cell_15_biasadd_readvariableop_resource_0"l
3while_lstm_cell_15_matmul_1_readvariableop_resource5while_lstm_cell_15_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_15_matmul_readvariableop_resource3while_lstm_cell_15_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :         Ц:         Ц: : : : : 2V
)while/lstm_cell_15/BiasAdd/ReadVariableOp)while/lstm_cell_15/BiasAdd/ReadVariableOp2T
(while/lstm_cell_15/MatMul/ReadVariableOp(while/lstm_cell_15/MatMul/ReadVariableOp2X
*while/lstm_cell_15/MatMul_1/ReadVariableOp*while/lstm_cell_15/MatMul_1/ReadVariableOp: 
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
:         Ц:.*
(
_output_shapes
:         Ц:

_output_shapes
: :

_output_shapes
: 
№	
╩
&__inference_signature_wrapper_26072009
lstm_14_input
unknown:	]╕
	unknown_0:
ю╕
	unknown_1:	╕
	unknown_2:
ю╪
	unknown_3:
Ц╪
	unknown_4:	╪
	unknown_5:	Ц
	unknown_6:
identityИвStatefulPartitionedCallм
StatefulPartitionedCallStatefulPartitionedCalllstm_14_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
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
#__inference__wrapped_model_260697922
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
_user_specified_namelstm_14_input
Ч
К
J__inference_lstm_cell_15_layer_call_and_return_conditional_losses_26074271

inputs
states_0
states_12
matmul_readvariableop_resource:
ю╪4
 matmul_1_readvariableop_resource:
Ц╪.
biasadd_readvariableop_resource:	╪
identity

identity_1

identity_2ИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpвMatMul_1/ReadVariableOpП
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
ю╪*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╪2
MatMulХ
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
Ц╪*
dtype02
MatMul_1/ReadVariableOp|
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╪2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:         ╪2
addН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:╪*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╪2	
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
P:         Ц:         Ц:         Ц:         Ц*
	num_split2
split`
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:         Ц2	
Sigmoidd
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:         Ц2
	Sigmoid_1]
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:         Ц2
mulW
ReluRelusplit:output:2*
T0*(
_output_shapes
:         Ц2
Relui
mul_1MulSigmoid:y:0Relu:activations:0*
T0*(
_output_shapes
:         Ц2
mul_1^
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:         Ц2
add_1d
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:         Ц2
	Sigmoid_2V
Relu_1Relu	add_1:z:0*
T0*(
_output_shapes
:         Ц2
Relu_1m
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*(
_output_shapes
:         Ц2
mul_2e
IdentityIdentity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:         Ц2

Identityi

Identity_1Identity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:         Ц2

Identity_1i

Identity_2Identity	add_1:z:0^NoOp*
T0*(
_output_shapes
:         Ц2

Identity_2Щ
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:         ю:         Ц:         Ц: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:P L
(
_output_shapes
:         ю
 
_user_specified_nameinputs:RN
(
_output_shapes
:         Ц
"
_user_specified_name
states/0:RN
(
_output_shapes
:         Ц
"
_user_specified_name
states/1
░?
╘
while_body_26073239
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_14_matmul_readvariableop_resource_0:	]╕I
5while_lstm_cell_14_matmul_1_readvariableop_resource_0:
ю╕C
4while_lstm_cell_14_biasadd_readvariableop_resource_0:	╕
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_14_matmul_readvariableop_resource:	]╕G
3while_lstm_cell_14_matmul_1_readvariableop_resource:
ю╕A
2while_lstm_cell_14_biasadd_readvariableop_resource:	╕Ив)while/lstm_cell_14/BiasAdd/ReadVariableOpв(while/lstm_cell_14/MatMul/ReadVariableOpв*while/lstm_cell_14/MatMul_1/ReadVariableOp├
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
(while/lstm_cell_14/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_14_matmul_readvariableop_resource_0*
_output_shapes
:	]╕*
dtype02*
(while/lstm_cell_14/MatMul/ReadVariableOp╫
while/lstm_cell_14/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_14/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╕2
while/lstm_cell_14/MatMul╨
*while/lstm_cell_14/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_14_matmul_1_readvariableop_resource_0* 
_output_shapes
:
ю╕*
dtype02,
*while/lstm_cell_14/MatMul_1/ReadVariableOp└
while/lstm_cell_14/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_14/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╕2
while/lstm_cell_14/MatMul_1╕
while/lstm_cell_14/addAddV2#while/lstm_cell_14/MatMul:product:0%while/lstm_cell_14/MatMul_1:product:0*
T0*(
_output_shapes
:         ╕2
while/lstm_cell_14/add╚
)while/lstm_cell_14/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_14_biasadd_readvariableop_resource_0*
_output_shapes	
:╕*
dtype02+
)while/lstm_cell_14/BiasAdd/ReadVariableOp┼
while/lstm_cell_14/BiasAddBiasAddwhile/lstm_cell_14/add:z:01while/lstm_cell_14/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╕2
while/lstm_cell_14/BiasAddК
"while/lstm_cell_14/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_14/split/split_dimП
while/lstm_cell_14/splitSplit+while/lstm_cell_14/split/split_dim:output:0#while/lstm_cell_14/BiasAdd:output:0*
T0*d
_output_shapesR
P:         ю:         ю:         ю:         ю*
	num_split2
while/lstm_cell_14/splitЩ
while/lstm_cell_14/SigmoidSigmoid!while/lstm_cell_14/split:output:0*
T0*(
_output_shapes
:         ю2
while/lstm_cell_14/SigmoidЭ
while/lstm_cell_14/Sigmoid_1Sigmoid!while/lstm_cell_14/split:output:1*
T0*(
_output_shapes
:         ю2
while/lstm_cell_14/Sigmoid_1б
while/lstm_cell_14/mulMul while/lstm_cell_14/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:         ю2
while/lstm_cell_14/mulР
while/lstm_cell_14/ReluRelu!while/lstm_cell_14/split:output:2*
T0*(
_output_shapes
:         ю2
while/lstm_cell_14/Relu╡
while/lstm_cell_14/mul_1Mulwhile/lstm_cell_14/Sigmoid:y:0%while/lstm_cell_14/Relu:activations:0*
T0*(
_output_shapes
:         ю2
while/lstm_cell_14/mul_1к
while/lstm_cell_14/add_1AddV2while/lstm_cell_14/mul:z:0while/lstm_cell_14/mul_1:z:0*
T0*(
_output_shapes
:         ю2
while/lstm_cell_14/add_1Э
while/lstm_cell_14/Sigmoid_2Sigmoid!while/lstm_cell_14/split:output:3*
T0*(
_output_shapes
:         ю2
while/lstm_cell_14/Sigmoid_2П
while/lstm_cell_14/Relu_1Reluwhile/lstm_cell_14/add_1:z:0*
T0*(
_output_shapes
:         ю2
while/lstm_cell_14/Relu_1╣
while/lstm_cell_14/mul_2Mul while/lstm_cell_14/Sigmoid_2:y:0'while/lstm_cell_14/Relu_1:activations:0*
T0*(
_output_shapes
:         ю2
while/lstm_cell_14/mul_2р
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_14/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_14/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:         ю2
while/Identity_4О
while/Identity_5Identitywhile/lstm_cell_14/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:         ю2
while/Identity_5▐

while/NoOpNoOp*^while/lstm_cell_14/BiasAdd/ReadVariableOp)^while/lstm_cell_14/MatMul/ReadVariableOp+^while/lstm_cell_14/MatMul_1/ReadVariableOp*"
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
2while_lstm_cell_14_biasadd_readvariableop_resource4while_lstm_cell_14_biasadd_readvariableop_resource_0"l
3while_lstm_cell_14_matmul_1_readvariableop_resource5while_lstm_cell_14_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_14_matmul_readvariableop_resource3while_lstm_cell_14_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :         ю:         ю: : : : : 2V
)while/lstm_cell_14/BiasAdd/ReadVariableOp)while/lstm_cell_14/BiasAdd/ReadVariableOp2T
(while/lstm_cell_14/MatMul/ReadVariableOp(while/lstm_cell_14/MatMul/ReadVariableOp2X
*while/lstm_cell_14/MatMul_1/ReadVariableOp*while/lstm_cell_14/MatMul_1/ReadVariableOp: 
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
:         ю:.*
(
_output_shapes
:         ю:

_output_shapes
: :

_output_shapes
: 
╫
g
H__inference_dropout_14_layer_call_and_return_conditional_losses_26073384

inputs
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *лкк?2
dropout/Constx
dropout/MulMulinputsdropout/Const:output:0*
T0*,
_output_shapes
:         ю2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape╣
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:         ю*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А>2
dropout/GreaterEqual/y├
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:         ю2
dropout/GreaterEqualД
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:         ю2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:         ю2
dropout/Mul_1j
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:         ю2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ю:T P
,
_output_shapes
:         ю
 
_user_specified_nameinputs
К\
Я
E__inference_lstm_15_layer_call_and_return_conditional_losses_26073998

inputs?
+lstm_cell_15_matmul_readvariableop_resource:
ю╪A
-lstm_cell_15_matmul_1_readvariableop_resource:
Ц╪;
,lstm_cell_15_biasadd_readvariableop_resource:	╪
identityИв#lstm_cell_15/BiasAdd/ReadVariableOpв"lstm_cell_15/MatMul/ReadVariableOpв$lstm_cell_15/MatMul_1/ReadVariableOpвwhileD
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
B :Ц2
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
B :Ц2
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
:         Ц2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :Ц2
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
B :Ц2
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
:         Ц2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm{
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:         ю2
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
valueB"    ю   27
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
strided_slice_2/stack_2¤
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         ю*
shrink_axis_mask2
strided_slice_2╢
"lstm_cell_15/MatMul/ReadVariableOpReadVariableOp+lstm_cell_15_matmul_readvariableop_resource* 
_output_shapes
:
ю╪*
dtype02$
"lstm_cell_15/MatMul/ReadVariableOpн
lstm_cell_15/MatMulMatMulstrided_slice_2:output:0*lstm_cell_15/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╪2
lstm_cell_15/MatMul╝
$lstm_cell_15/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_15_matmul_1_readvariableop_resource* 
_output_shapes
:
Ц╪*
dtype02&
$lstm_cell_15/MatMul_1/ReadVariableOpй
lstm_cell_15/MatMul_1MatMulzeros:output:0,lstm_cell_15/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╪2
lstm_cell_15/MatMul_1а
lstm_cell_15/addAddV2lstm_cell_15/MatMul:product:0lstm_cell_15/MatMul_1:product:0*
T0*(
_output_shapes
:         ╪2
lstm_cell_15/add┤
#lstm_cell_15/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_15_biasadd_readvariableop_resource*
_output_shapes	
:╪*
dtype02%
#lstm_cell_15/BiasAdd/ReadVariableOpн
lstm_cell_15/BiasAddBiasAddlstm_cell_15/add:z:0+lstm_cell_15/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╪2
lstm_cell_15/BiasAdd~
lstm_cell_15/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_15/split/split_dimў
lstm_cell_15/splitSplit%lstm_cell_15/split/split_dim:output:0lstm_cell_15/BiasAdd:output:0*
T0*d
_output_shapesR
P:         Ц:         Ц:         Ц:         Ц*
	num_split2
lstm_cell_15/splitЗ
lstm_cell_15/SigmoidSigmoidlstm_cell_15/split:output:0*
T0*(
_output_shapes
:         Ц2
lstm_cell_15/SigmoidЛ
lstm_cell_15/Sigmoid_1Sigmoidlstm_cell_15/split:output:1*
T0*(
_output_shapes
:         Ц2
lstm_cell_15/Sigmoid_1М
lstm_cell_15/mulMullstm_cell_15/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:         Ц2
lstm_cell_15/mul~
lstm_cell_15/ReluRelulstm_cell_15/split:output:2*
T0*(
_output_shapes
:         Ц2
lstm_cell_15/ReluЭ
lstm_cell_15/mul_1Mullstm_cell_15/Sigmoid:y:0lstm_cell_15/Relu:activations:0*
T0*(
_output_shapes
:         Ц2
lstm_cell_15/mul_1Т
lstm_cell_15/add_1AddV2lstm_cell_15/mul:z:0lstm_cell_15/mul_1:z:0*
T0*(
_output_shapes
:         Ц2
lstm_cell_15/add_1Л
lstm_cell_15/Sigmoid_2Sigmoidlstm_cell_15/split:output:3*
T0*(
_output_shapes
:         Ц2
lstm_cell_15/Sigmoid_2}
lstm_cell_15/Relu_1Relulstm_cell_15/add_1:z:0*
T0*(
_output_shapes
:         Ц2
lstm_cell_15/Relu_1б
lstm_cell_15/mul_2Mullstm_cell_15/Sigmoid_2:y:0!lstm_cell_15/Relu_1:activations:0*
T0*(
_output_shapes
:         Ц2
lstm_cell_15/mul_2П
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    Ц   2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_15_matmul_readvariableop_resource-lstm_cell_15_matmul_1_readvariableop_resource,lstm_cell_15_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :         Ц:         Ц: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_26073914*
condR
while_cond_26073913*M
output_shapes<
:: : : : :         Ц:         Ц: : : : : *
parallel_iterations 2
while╡
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    Ц   22
0TensorArrayV2Stack/TensorListStack/element_shapeщ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:         Ц*
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
:         Ц*
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
:         Ц2
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
:         Ц2

Identity╚
NoOpNoOp$^lstm_cell_15/BiasAdd/ReadVariableOp#^lstm_cell_15/MatMul/ReadVariableOp%^lstm_cell_15/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:         ю: : : 2J
#lstm_cell_15/BiasAdd/ReadVariableOp#lstm_cell_15/BiasAdd/ReadVariableOp2H
"lstm_cell_15/MatMul/ReadVariableOp"lstm_cell_15/MatMul/ReadVariableOp2L
$lstm_cell_15/MatMul_1/ReadVariableOp$lstm_cell_15/MatMul_1/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:         ю
 
_user_specified_nameinputs
Л
З
J__inference_lstm_cell_14_layer_call_and_return_conditional_losses_26069867

inputs

states
states_11
matmul_readvariableop_resource:	]╕4
 matmul_1_readvariableop_resource:
ю╕.
biasadd_readvariableop_resource:	╕
identity

identity_1

identity_2ИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpвMatMul_1/ReadVariableOpО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	]╕*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╕2
MatMulХ
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
ю╕*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╕2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:         ╕2
addН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:╕*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╕2	
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
P:         ю:         ю:         ю:         ю*
	num_split2
split`
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:         ю2	
Sigmoidd
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:         ю2
	Sigmoid_1]
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:         ю2
mulW
ReluRelusplit:output:2*
T0*(
_output_shapes
:         ю2
Relui
mul_1MulSigmoid:y:0Relu:activations:0*
T0*(
_output_shapes
:         ю2
mul_1^
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:         ю2
add_1d
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:         ю2
	Sigmoid_2V
Relu_1Relu	add_1:z:0*
T0*(
_output_shapes
:         ю2
Relu_1m
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*(
_output_shapes
:         ю2
mul_2e
IdentityIdentity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:         ю2

Identityi

Identity_1Identity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:         ю2

Identity_1i

Identity_2Identity	add_1:z:0^NoOp*
T0*(
_output_shapes
:         ю2

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
A:         ]:         ю:         ю: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:         ]
 
_user_specified_nameinputs:PL
(
_output_shapes
:         ю
 
_user_specified_namestates:PL
(
_output_shapes
:         ю
 
_user_specified_namestates
у
═
while_cond_26070090
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_26070090___redundant_placeholder06
2while_while_cond_26070090___redundant_placeholder16
2while_while_cond_26070090___redundant_placeholder26
2while_while_cond_26070090___redundant_placeholder3
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
B: : : : :         ю:         ю: ::::: 
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
:         ю:.*
(
_output_shapes
:         ю:

_output_shapes
: :

_output_shapes
:
м]
ї
(sequential_7_lstm_14_while_body_26069532F
Bsequential_7_lstm_14_while_sequential_7_lstm_14_while_loop_counterL
Hsequential_7_lstm_14_while_sequential_7_lstm_14_while_maximum_iterations*
&sequential_7_lstm_14_while_placeholder,
(sequential_7_lstm_14_while_placeholder_1,
(sequential_7_lstm_14_while_placeholder_2,
(sequential_7_lstm_14_while_placeholder_3E
Asequential_7_lstm_14_while_sequential_7_lstm_14_strided_slice_1_0Б
}sequential_7_lstm_14_while_tensorarrayv2read_tensorlistgetitem_sequential_7_lstm_14_tensorarrayunstack_tensorlistfromtensor_0[
Hsequential_7_lstm_14_while_lstm_cell_14_matmul_readvariableop_resource_0:	]╕^
Jsequential_7_lstm_14_while_lstm_cell_14_matmul_1_readvariableop_resource_0:
ю╕X
Isequential_7_lstm_14_while_lstm_cell_14_biasadd_readvariableop_resource_0:	╕'
#sequential_7_lstm_14_while_identity)
%sequential_7_lstm_14_while_identity_1)
%sequential_7_lstm_14_while_identity_2)
%sequential_7_lstm_14_while_identity_3)
%sequential_7_lstm_14_while_identity_4)
%sequential_7_lstm_14_while_identity_5C
?sequential_7_lstm_14_while_sequential_7_lstm_14_strided_slice_1
{sequential_7_lstm_14_while_tensorarrayv2read_tensorlistgetitem_sequential_7_lstm_14_tensorarrayunstack_tensorlistfromtensorY
Fsequential_7_lstm_14_while_lstm_cell_14_matmul_readvariableop_resource:	]╕\
Hsequential_7_lstm_14_while_lstm_cell_14_matmul_1_readvariableop_resource:
ю╕V
Gsequential_7_lstm_14_while_lstm_cell_14_biasadd_readvariableop_resource:	╕Ив>sequential_7/lstm_14/while/lstm_cell_14/BiasAdd/ReadVariableOpв=sequential_7/lstm_14/while/lstm_cell_14/MatMul/ReadVariableOpв?sequential_7/lstm_14/while/lstm_cell_14/MatMul_1/ReadVariableOpэ
Lsequential_7/lstm_14/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ]   2N
Lsequential_7/lstm_14/while/TensorArrayV2Read/TensorListGetItem/element_shape╤
>sequential_7/lstm_14/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem}sequential_7_lstm_14_while_tensorarrayv2read_tensorlistgetitem_sequential_7_lstm_14_tensorarrayunstack_tensorlistfromtensor_0&sequential_7_lstm_14_while_placeholderUsequential_7/lstm_14/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         ]*
element_dtype02@
>sequential_7/lstm_14/while/TensorArrayV2Read/TensorListGetItemИ
=sequential_7/lstm_14/while/lstm_cell_14/MatMul/ReadVariableOpReadVariableOpHsequential_7_lstm_14_while_lstm_cell_14_matmul_readvariableop_resource_0*
_output_shapes
:	]╕*
dtype02?
=sequential_7/lstm_14/while/lstm_cell_14/MatMul/ReadVariableOpл
.sequential_7/lstm_14/while/lstm_cell_14/MatMulMatMulEsequential_7/lstm_14/while/TensorArrayV2Read/TensorListGetItem:item:0Esequential_7/lstm_14/while/lstm_cell_14/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╕20
.sequential_7/lstm_14/while/lstm_cell_14/MatMulП
?sequential_7/lstm_14/while/lstm_cell_14/MatMul_1/ReadVariableOpReadVariableOpJsequential_7_lstm_14_while_lstm_cell_14_matmul_1_readvariableop_resource_0* 
_output_shapes
:
ю╕*
dtype02A
?sequential_7/lstm_14/while/lstm_cell_14/MatMul_1/ReadVariableOpФ
0sequential_7/lstm_14/while/lstm_cell_14/MatMul_1MatMul(sequential_7_lstm_14_while_placeholder_2Gsequential_7/lstm_14/while/lstm_cell_14/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╕22
0sequential_7/lstm_14/while/lstm_cell_14/MatMul_1М
+sequential_7/lstm_14/while/lstm_cell_14/addAddV28sequential_7/lstm_14/while/lstm_cell_14/MatMul:product:0:sequential_7/lstm_14/while/lstm_cell_14/MatMul_1:product:0*
T0*(
_output_shapes
:         ╕2-
+sequential_7/lstm_14/while/lstm_cell_14/addЗ
>sequential_7/lstm_14/while/lstm_cell_14/BiasAdd/ReadVariableOpReadVariableOpIsequential_7_lstm_14_while_lstm_cell_14_biasadd_readvariableop_resource_0*
_output_shapes	
:╕*
dtype02@
>sequential_7/lstm_14/while/lstm_cell_14/BiasAdd/ReadVariableOpЩ
/sequential_7/lstm_14/while/lstm_cell_14/BiasAddBiasAdd/sequential_7/lstm_14/while/lstm_cell_14/add:z:0Fsequential_7/lstm_14/while/lstm_cell_14/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╕21
/sequential_7/lstm_14/while/lstm_cell_14/BiasAdd┤
7sequential_7/lstm_14/while/lstm_cell_14/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :29
7sequential_7/lstm_14/while/lstm_cell_14/split/split_dimу
-sequential_7/lstm_14/while/lstm_cell_14/splitSplit@sequential_7/lstm_14/while/lstm_cell_14/split/split_dim:output:08sequential_7/lstm_14/while/lstm_cell_14/BiasAdd:output:0*
T0*d
_output_shapesR
P:         ю:         ю:         ю:         ю*
	num_split2/
-sequential_7/lstm_14/while/lstm_cell_14/split╪
/sequential_7/lstm_14/while/lstm_cell_14/SigmoidSigmoid6sequential_7/lstm_14/while/lstm_cell_14/split:output:0*
T0*(
_output_shapes
:         ю21
/sequential_7/lstm_14/while/lstm_cell_14/Sigmoid▄
1sequential_7/lstm_14/while/lstm_cell_14/Sigmoid_1Sigmoid6sequential_7/lstm_14/while/lstm_cell_14/split:output:1*
T0*(
_output_shapes
:         ю23
1sequential_7/lstm_14/while/lstm_cell_14/Sigmoid_1ї
+sequential_7/lstm_14/while/lstm_cell_14/mulMul5sequential_7/lstm_14/while/lstm_cell_14/Sigmoid_1:y:0(sequential_7_lstm_14_while_placeholder_3*
T0*(
_output_shapes
:         ю2-
+sequential_7/lstm_14/while/lstm_cell_14/mul╧
,sequential_7/lstm_14/while/lstm_cell_14/ReluRelu6sequential_7/lstm_14/while/lstm_cell_14/split:output:2*
T0*(
_output_shapes
:         ю2.
,sequential_7/lstm_14/while/lstm_cell_14/ReluЙ
-sequential_7/lstm_14/while/lstm_cell_14/mul_1Mul3sequential_7/lstm_14/while/lstm_cell_14/Sigmoid:y:0:sequential_7/lstm_14/while/lstm_cell_14/Relu:activations:0*
T0*(
_output_shapes
:         ю2/
-sequential_7/lstm_14/while/lstm_cell_14/mul_1■
-sequential_7/lstm_14/while/lstm_cell_14/add_1AddV2/sequential_7/lstm_14/while/lstm_cell_14/mul:z:01sequential_7/lstm_14/while/lstm_cell_14/mul_1:z:0*
T0*(
_output_shapes
:         ю2/
-sequential_7/lstm_14/while/lstm_cell_14/add_1▄
1sequential_7/lstm_14/while/lstm_cell_14/Sigmoid_2Sigmoid6sequential_7/lstm_14/while/lstm_cell_14/split:output:3*
T0*(
_output_shapes
:         ю23
1sequential_7/lstm_14/while/lstm_cell_14/Sigmoid_2╬
.sequential_7/lstm_14/while/lstm_cell_14/Relu_1Relu1sequential_7/lstm_14/while/lstm_cell_14/add_1:z:0*
T0*(
_output_shapes
:         ю20
.sequential_7/lstm_14/while/lstm_cell_14/Relu_1Н
-sequential_7/lstm_14/while/lstm_cell_14/mul_2Mul5sequential_7/lstm_14/while/lstm_cell_14/Sigmoid_2:y:0<sequential_7/lstm_14/while/lstm_cell_14/Relu_1:activations:0*
T0*(
_output_shapes
:         ю2/
-sequential_7/lstm_14/while/lstm_cell_14/mul_2╔
?sequential_7/lstm_14/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem(sequential_7_lstm_14_while_placeholder_1&sequential_7_lstm_14_while_placeholder1sequential_7/lstm_14/while/lstm_cell_14/mul_2:z:0*
_output_shapes
: *
element_dtype02A
?sequential_7/lstm_14/while/TensorArrayV2Write/TensorListSetItemЖ
 sequential_7/lstm_14/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2"
 sequential_7/lstm_14/while/add/y╜
sequential_7/lstm_14/while/addAddV2&sequential_7_lstm_14_while_placeholder)sequential_7/lstm_14/while/add/y:output:0*
T0*
_output_shapes
: 2 
sequential_7/lstm_14/while/addК
"sequential_7/lstm_14/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2$
"sequential_7/lstm_14/while/add_1/y▀
 sequential_7/lstm_14/while/add_1AddV2Bsequential_7_lstm_14_while_sequential_7_lstm_14_while_loop_counter+sequential_7/lstm_14/while/add_1/y:output:0*
T0*
_output_shapes
: 2"
 sequential_7/lstm_14/while/add_1┐
#sequential_7/lstm_14/while/IdentityIdentity$sequential_7/lstm_14/while/add_1:z:0 ^sequential_7/lstm_14/while/NoOp*
T0*
_output_shapes
: 2%
#sequential_7/lstm_14/while/Identityч
%sequential_7/lstm_14/while/Identity_1IdentityHsequential_7_lstm_14_while_sequential_7_lstm_14_while_maximum_iterations ^sequential_7/lstm_14/while/NoOp*
T0*
_output_shapes
: 2'
%sequential_7/lstm_14/while/Identity_1┴
%sequential_7/lstm_14/while/Identity_2Identity"sequential_7/lstm_14/while/add:z:0 ^sequential_7/lstm_14/while/NoOp*
T0*
_output_shapes
: 2'
%sequential_7/lstm_14/while/Identity_2ю
%sequential_7/lstm_14/while/Identity_3IdentityOsequential_7/lstm_14/while/TensorArrayV2Write/TensorListSetItem:output_handle:0 ^sequential_7/lstm_14/while/NoOp*
T0*
_output_shapes
: 2'
%sequential_7/lstm_14/while/Identity_3т
%sequential_7/lstm_14/while/Identity_4Identity1sequential_7/lstm_14/while/lstm_cell_14/mul_2:z:0 ^sequential_7/lstm_14/while/NoOp*
T0*(
_output_shapes
:         ю2'
%sequential_7/lstm_14/while/Identity_4т
%sequential_7/lstm_14/while/Identity_5Identity1sequential_7/lstm_14/while/lstm_cell_14/add_1:z:0 ^sequential_7/lstm_14/while/NoOp*
T0*(
_output_shapes
:         ю2'
%sequential_7/lstm_14/while/Identity_5╟
sequential_7/lstm_14/while/NoOpNoOp?^sequential_7/lstm_14/while/lstm_cell_14/BiasAdd/ReadVariableOp>^sequential_7/lstm_14/while/lstm_cell_14/MatMul/ReadVariableOp@^sequential_7/lstm_14/while/lstm_cell_14/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2!
sequential_7/lstm_14/while/NoOp"S
#sequential_7_lstm_14_while_identity,sequential_7/lstm_14/while/Identity:output:0"W
%sequential_7_lstm_14_while_identity_1.sequential_7/lstm_14/while/Identity_1:output:0"W
%sequential_7_lstm_14_while_identity_2.sequential_7/lstm_14/while/Identity_2:output:0"W
%sequential_7_lstm_14_while_identity_3.sequential_7/lstm_14/while/Identity_3:output:0"W
%sequential_7_lstm_14_while_identity_4.sequential_7/lstm_14/while/Identity_4:output:0"W
%sequential_7_lstm_14_while_identity_5.sequential_7/lstm_14/while/Identity_5:output:0"Ф
Gsequential_7_lstm_14_while_lstm_cell_14_biasadd_readvariableop_resourceIsequential_7_lstm_14_while_lstm_cell_14_biasadd_readvariableop_resource_0"Ц
Hsequential_7_lstm_14_while_lstm_cell_14_matmul_1_readvariableop_resourceJsequential_7_lstm_14_while_lstm_cell_14_matmul_1_readvariableop_resource_0"Т
Fsequential_7_lstm_14_while_lstm_cell_14_matmul_readvariableop_resourceHsequential_7_lstm_14_while_lstm_cell_14_matmul_readvariableop_resource_0"Д
?sequential_7_lstm_14_while_sequential_7_lstm_14_strided_slice_1Asequential_7_lstm_14_while_sequential_7_lstm_14_strided_slice_1_0"№
{sequential_7_lstm_14_while_tensorarrayv2read_tensorlistgetitem_sequential_7_lstm_14_tensorarrayunstack_tensorlistfromtensor}sequential_7_lstm_14_while_tensorarrayv2read_tensorlistgetitem_sequential_7_lstm_14_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :         ю:         ю: : : : : 2А
>sequential_7/lstm_14/while/lstm_cell_14/BiasAdd/ReadVariableOp>sequential_7/lstm_14/while/lstm_cell_14/BiasAdd/ReadVariableOp2~
=sequential_7/lstm_14/while/lstm_cell_14/MatMul/ReadVariableOp=sequential_7/lstm_14/while/lstm_cell_14/MatMul/ReadVariableOp2В
?sequential_7/lstm_14/while/lstm_cell_14/MatMul_1/ReadVariableOp?sequential_7/lstm_14/while/lstm_cell_14/MatMul_1/ReadVariableOp: 
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
:         ю:.*
(
_output_shapes
:         ю:

_output_shapes
: :

_output_shapes
: 
К\
Я
E__inference_lstm_15_layer_call_and_return_conditional_losses_26073847

inputs?
+lstm_cell_15_matmul_readvariableop_resource:
ю╪A
-lstm_cell_15_matmul_1_readvariableop_resource:
Ц╪;
,lstm_cell_15_biasadd_readvariableop_resource:	╪
identityИв#lstm_cell_15/BiasAdd/ReadVariableOpв"lstm_cell_15/MatMul/ReadVariableOpв$lstm_cell_15/MatMul_1/ReadVariableOpвwhileD
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
B :Ц2
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
B :Ц2
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
:         Ц2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :Ц2
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
B :Ц2
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
:         Ц2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm{
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:         ю2
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
valueB"    ю   27
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
strided_slice_2/stack_2¤
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         ю*
shrink_axis_mask2
strided_slice_2╢
"lstm_cell_15/MatMul/ReadVariableOpReadVariableOp+lstm_cell_15_matmul_readvariableop_resource* 
_output_shapes
:
ю╪*
dtype02$
"lstm_cell_15/MatMul/ReadVariableOpн
lstm_cell_15/MatMulMatMulstrided_slice_2:output:0*lstm_cell_15/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╪2
lstm_cell_15/MatMul╝
$lstm_cell_15/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_15_matmul_1_readvariableop_resource* 
_output_shapes
:
Ц╪*
dtype02&
$lstm_cell_15/MatMul_1/ReadVariableOpй
lstm_cell_15/MatMul_1MatMulzeros:output:0,lstm_cell_15/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╪2
lstm_cell_15/MatMul_1а
lstm_cell_15/addAddV2lstm_cell_15/MatMul:product:0lstm_cell_15/MatMul_1:product:0*
T0*(
_output_shapes
:         ╪2
lstm_cell_15/add┤
#lstm_cell_15/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_15_biasadd_readvariableop_resource*
_output_shapes	
:╪*
dtype02%
#lstm_cell_15/BiasAdd/ReadVariableOpн
lstm_cell_15/BiasAddBiasAddlstm_cell_15/add:z:0+lstm_cell_15/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╪2
lstm_cell_15/BiasAdd~
lstm_cell_15/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_15/split/split_dimў
lstm_cell_15/splitSplit%lstm_cell_15/split/split_dim:output:0lstm_cell_15/BiasAdd:output:0*
T0*d
_output_shapesR
P:         Ц:         Ц:         Ц:         Ц*
	num_split2
lstm_cell_15/splitЗ
lstm_cell_15/SigmoidSigmoidlstm_cell_15/split:output:0*
T0*(
_output_shapes
:         Ц2
lstm_cell_15/SigmoidЛ
lstm_cell_15/Sigmoid_1Sigmoidlstm_cell_15/split:output:1*
T0*(
_output_shapes
:         Ц2
lstm_cell_15/Sigmoid_1М
lstm_cell_15/mulMullstm_cell_15/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:         Ц2
lstm_cell_15/mul~
lstm_cell_15/ReluRelulstm_cell_15/split:output:2*
T0*(
_output_shapes
:         Ц2
lstm_cell_15/ReluЭ
lstm_cell_15/mul_1Mullstm_cell_15/Sigmoid:y:0lstm_cell_15/Relu:activations:0*
T0*(
_output_shapes
:         Ц2
lstm_cell_15/mul_1Т
lstm_cell_15/add_1AddV2lstm_cell_15/mul:z:0lstm_cell_15/mul_1:z:0*
T0*(
_output_shapes
:         Ц2
lstm_cell_15/add_1Л
lstm_cell_15/Sigmoid_2Sigmoidlstm_cell_15/split:output:3*
T0*(
_output_shapes
:         Ц2
lstm_cell_15/Sigmoid_2}
lstm_cell_15/Relu_1Relulstm_cell_15/add_1:z:0*
T0*(
_output_shapes
:         Ц2
lstm_cell_15/Relu_1б
lstm_cell_15/mul_2Mullstm_cell_15/Sigmoid_2:y:0!lstm_cell_15/Relu_1:activations:0*
T0*(
_output_shapes
:         Ц2
lstm_cell_15/mul_2П
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    Ц   2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_15_matmul_readvariableop_resource-lstm_cell_15_matmul_1_readvariableop_resource,lstm_cell_15_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :         Ц:         Ц: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_26073763*
condR
while_cond_26073762*M
output_shapes<
:: : : : :         Ц:         Ц: : : : : *
parallel_iterations 2
while╡
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    Ц   22
0TensorArrayV2Stack/TensorListStack/element_shapeщ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:         Ц*
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
:         Ц*
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
:         Ц2
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
:         Ц2

Identity╚
NoOpNoOp$^lstm_cell_15/BiasAdd/ReadVariableOp#^lstm_cell_15/MatMul/ReadVariableOp%^lstm_cell_15/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:         ю: : : 2J
#lstm_cell_15/BiasAdd/ReadVariableOp#lstm_cell_15/BiasAdd/ReadVariableOp2H
"lstm_cell_15/MatMul/ReadVariableOp"lstm_cell_15/MatMul/ReadVariableOp2L
$lstm_cell_15/MatMul_1/ReadVariableOp$lstm_cell_15/MatMul_1/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:         ю
 
_user_specified_nameinputs
П
И
J__inference_lstm_cell_15_layer_call_and_return_conditional_losses_26070643

inputs

states
states_12
matmul_readvariableop_resource:
ю╪4
 matmul_1_readvariableop_resource:
Ц╪.
biasadd_readvariableop_resource:	╪
identity

identity_1

identity_2ИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpвMatMul_1/ReadVariableOpП
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
ю╪*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╪2
MatMulХ
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
Ц╪*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╪2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:         ╪2
addН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:╪*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╪2	
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
P:         Ц:         Ц:         Ц:         Ц*
	num_split2
split`
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:         Ц2	
Sigmoidd
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:         Ц2
	Sigmoid_1]
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:         Ц2
mulW
ReluRelusplit:output:2*
T0*(
_output_shapes
:         Ц2
Relui
mul_1MulSigmoid:y:0Relu:activations:0*
T0*(
_output_shapes
:         Ц2
mul_1^
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:         Ц2
add_1d
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:         Ц2
	Sigmoid_2V
Relu_1Relu	add_1:z:0*
T0*(
_output_shapes
:         Ц2
Relu_1m
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*(
_output_shapes
:         Ц2
mul_2e
IdentityIdentity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:         Ц2

Identityi

Identity_1Identity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:         Ц2

Identity_1i

Identity_2Identity	add_1:z:0^NoOp*
T0*(
_output_shapes
:         Ц2

Identity_2Щ
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:         ю:         Ц:         Ц: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:P L
(
_output_shapes
:         ю
 
_user_specified_nameinputs:PL
(
_output_shapes
:         Ц
 
_user_specified_namestates:PL
(
_output_shapes
:         Ц
 
_user_specified_namestates
Ы
э
J__inference_sequential_7_layer_call_and_return_conditional_losses_26071427

inputs#
lstm_14_26071210:	]╕$
lstm_14_26071212:
ю╕
lstm_14_26071214:	╕$
lstm_15_26071375:
ю╪$
lstm_15_26071377:
Ц╪
lstm_15_26071379:	╪#
dense_7_26071421:	Ц
dense_7_26071423:
identityИвdense_7/StatefulPartitionedCallвlstm_14/StatefulPartitionedCallвlstm_15/StatefulPartitionedCallо
lstm_14/StatefulPartitionedCallStatefulPartitionedCallinputslstm_14_26071210lstm_14_26071212lstm_14_26071214*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ю*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_lstm_14_layer_call_and_return_conditional_losses_260712092!
lstm_14/StatefulPartitionedCallГ
dropout_14/PartitionedCallPartitionedCall(lstm_14/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ю* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_dropout_14_layer_call_and_return_conditional_losses_260712222
dropout_14/PartitionedCall╦
lstm_15/StatefulPartitionedCallStatefulPartitionedCall#dropout_14/PartitionedCall:output:0lstm_15_26071375lstm_15_26071377lstm_15_26071379*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         Ц*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_lstm_15_layer_call_and_return_conditional_losses_260713742!
lstm_15/StatefulPartitionedCallГ
dropout_15/PartitionedCallPartitionedCall(lstm_15/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         Ц* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_dropout_15_layer_call_and_return_conditional_losses_260713872
dropout_15/PartitionedCall╢
dense_7/StatefulPartitionedCallStatefulPartitionedCall#dropout_15/PartitionedCall:output:0dense_7_26071421dense_7_26071423*
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
E__inference_dense_7_layer_call_and_return_conditional_losses_260714202!
dense_7/StatefulPartitionedCallЗ
IdentityIdentity(dense_7/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:         2

Identity┤
NoOpNoOp ^dense_7/StatefulPartitionedCall ^lstm_14/StatefulPartitionedCall ^lstm_15/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':         ]: : : : : : : : 2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall2B
lstm_14/StatefulPartitionedCalllstm_14/StatefulPartitionedCall2B
lstm_15/StatefulPartitionedCalllstm_15/StatefulPartitionedCall:S O
+
_output_shapes
:         ]
 
_user_specified_nameinputs
у
═
while_cond_26073087
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_26073087___redundant_placeholder06
2while_while_cond_26073087___redundant_placeholder16
2while_while_cond_26073087___redundant_placeholder26
2while_while_cond_26073087___redundant_placeholder3
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
B: : : : :         ю:         ю: ::::: 
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
:         ю:.*
(
_output_shapes
:         ю:

_output_shapes
: :

_output_shapes
:
И┤
╤	
#__inference__wrapped_model_26069792
lstm_14_inputS
@sequential_7_lstm_14_lstm_cell_14_matmul_readvariableop_resource:	]╕V
Bsequential_7_lstm_14_lstm_cell_14_matmul_1_readvariableop_resource:
ю╕P
Asequential_7_lstm_14_lstm_cell_14_biasadd_readvariableop_resource:	╕T
@sequential_7_lstm_15_lstm_cell_15_matmul_readvariableop_resource:
ю╪V
Bsequential_7_lstm_15_lstm_cell_15_matmul_1_readvariableop_resource:
Ц╪P
Asequential_7_lstm_15_lstm_cell_15_biasadd_readvariableop_resource:	╪I
6sequential_7_dense_7_tensordot_readvariableop_resource:	ЦB
4sequential_7_dense_7_biasadd_readvariableop_resource:
identityИв+sequential_7/dense_7/BiasAdd/ReadVariableOpв-sequential_7/dense_7/Tensordot/ReadVariableOpв8sequential_7/lstm_14/lstm_cell_14/BiasAdd/ReadVariableOpв7sequential_7/lstm_14/lstm_cell_14/MatMul/ReadVariableOpв9sequential_7/lstm_14/lstm_cell_14/MatMul_1/ReadVariableOpвsequential_7/lstm_14/whileв8sequential_7/lstm_15/lstm_cell_15/BiasAdd/ReadVariableOpв7sequential_7/lstm_15/lstm_cell_15/MatMul/ReadVariableOpв9sequential_7/lstm_15/lstm_cell_15/MatMul_1/ReadVariableOpвsequential_7/lstm_15/whileu
sequential_7/lstm_14/ShapeShapelstm_14_input*
T0*
_output_shapes
:2
sequential_7/lstm_14/ShapeЮ
(sequential_7/lstm_14/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(sequential_7/lstm_14/strided_slice/stackв
*sequential_7/lstm_14/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*sequential_7/lstm_14/strided_slice/stack_1в
*sequential_7/lstm_14/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*sequential_7/lstm_14/strided_slice/stack_2р
"sequential_7/lstm_14/strided_sliceStridedSlice#sequential_7/lstm_14/Shape:output:01sequential_7/lstm_14/strided_slice/stack:output:03sequential_7/lstm_14/strided_slice/stack_1:output:03sequential_7/lstm_14/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"sequential_7/lstm_14/strided_sliceЗ
 sequential_7/lstm_14/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :ю2"
 sequential_7/lstm_14/zeros/mul/y└
sequential_7/lstm_14/zeros/mulMul+sequential_7/lstm_14/strided_slice:output:0)sequential_7/lstm_14/zeros/mul/y:output:0*
T0*
_output_shapes
: 2 
sequential_7/lstm_14/zeros/mulЙ
!sequential_7/lstm_14/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2#
!sequential_7/lstm_14/zeros/Less/y╗
sequential_7/lstm_14/zeros/LessLess"sequential_7/lstm_14/zeros/mul:z:0*sequential_7/lstm_14/zeros/Less/y:output:0*
T0*
_output_shapes
: 2!
sequential_7/lstm_14/zeros/LessН
#sequential_7/lstm_14/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :ю2%
#sequential_7/lstm_14/zeros/packed/1╫
!sequential_7/lstm_14/zeros/packedPack+sequential_7/lstm_14/strided_slice:output:0,sequential_7/lstm_14/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2#
!sequential_7/lstm_14/zeros/packedЙ
 sequential_7/lstm_14/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 sequential_7/lstm_14/zeros/Const╩
sequential_7/lstm_14/zerosFill*sequential_7/lstm_14/zeros/packed:output:0)sequential_7/lstm_14/zeros/Const:output:0*
T0*(
_output_shapes
:         ю2
sequential_7/lstm_14/zerosЛ
"sequential_7/lstm_14/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :ю2$
"sequential_7/lstm_14/zeros_1/mul/y╞
 sequential_7/lstm_14/zeros_1/mulMul+sequential_7/lstm_14/strided_slice:output:0+sequential_7/lstm_14/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2"
 sequential_7/lstm_14/zeros_1/mulН
#sequential_7/lstm_14/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2%
#sequential_7/lstm_14/zeros_1/Less/y├
!sequential_7/lstm_14/zeros_1/LessLess$sequential_7/lstm_14/zeros_1/mul:z:0,sequential_7/lstm_14/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2#
!sequential_7/lstm_14/zeros_1/LessС
%sequential_7/lstm_14/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :ю2'
%sequential_7/lstm_14/zeros_1/packed/1▌
#sequential_7/lstm_14/zeros_1/packedPack+sequential_7/lstm_14/strided_slice:output:0.sequential_7/lstm_14/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2%
#sequential_7/lstm_14/zeros_1/packedН
"sequential_7/lstm_14/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"sequential_7/lstm_14/zeros_1/Const╥
sequential_7/lstm_14/zeros_1Fill,sequential_7/lstm_14/zeros_1/packed:output:0+sequential_7/lstm_14/zeros_1/Const:output:0*
T0*(
_output_shapes
:         ю2
sequential_7/lstm_14/zeros_1Я
#sequential_7/lstm_14/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2%
#sequential_7/lstm_14/transpose/perm└
sequential_7/lstm_14/transpose	Transposelstm_14_input,sequential_7/lstm_14/transpose/perm:output:0*
T0*+
_output_shapes
:         ]2 
sequential_7/lstm_14/transposeО
sequential_7/lstm_14/Shape_1Shape"sequential_7/lstm_14/transpose:y:0*
T0*
_output_shapes
:2
sequential_7/lstm_14/Shape_1в
*sequential_7/lstm_14/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2,
*sequential_7/lstm_14/strided_slice_1/stackж
,sequential_7/lstm_14/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_7/lstm_14/strided_slice_1/stack_1ж
,sequential_7/lstm_14/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_7/lstm_14/strided_slice_1/stack_2ь
$sequential_7/lstm_14/strided_slice_1StridedSlice%sequential_7/lstm_14/Shape_1:output:03sequential_7/lstm_14/strided_slice_1/stack:output:05sequential_7/lstm_14/strided_slice_1/stack_1:output:05sequential_7/lstm_14/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2&
$sequential_7/lstm_14/strided_slice_1п
0sequential_7/lstm_14/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         22
0sequential_7/lstm_14/TensorArrayV2/element_shapeЖ
"sequential_7/lstm_14/TensorArrayV2TensorListReserve9sequential_7/lstm_14/TensorArrayV2/element_shape:output:0-sequential_7/lstm_14/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02$
"sequential_7/lstm_14/TensorArrayV2щ
Jsequential_7/lstm_14/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ]   2L
Jsequential_7/lstm_14/TensorArrayUnstack/TensorListFromTensor/element_shape╠
<sequential_7/lstm_14/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor"sequential_7/lstm_14/transpose:y:0Ssequential_7/lstm_14/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02>
<sequential_7/lstm_14/TensorArrayUnstack/TensorListFromTensorв
*sequential_7/lstm_14/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2,
*sequential_7/lstm_14/strided_slice_2/stackж
,sequential_7/lstm_14/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_7/lstm_14/strided_slice_2/stack_1ж
,sequential_7/lstm_14/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_7/lstm_14/strided_slice_2/stack_2·
$sequential_7/lstm_14/strided_slice_2StridedSlice"sequential_7/lstm_14/transpose:y:03sequential_7/lstm_14/strided_slice_2/stack:output:05sequential_7/lstm_14/strided_slice_2/stack_1:output:05sequential_7/lstm_14/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         ]*
shrink_axis_mask2&
$sequential_7/lstm_14/strided_slice_2Ї
7sequential_7/lstm_14/lstm_cell_14/MatMul/ReadVariableOpReadVariableOp@sequential_7_lstm_14_lstm_cell_14_matmul_readvariableop_resource*
_output_shapes
:	]╕*
dtype029
7sequential_7/lstm_14/lstm_cell_14/MatMul/ReadVariableOpБ
(sequential_7/lstm_14/lstm_cell_14/MatMulMatMul-sequential_7/lstm_14/strided_slice_2:output:0?sequential_7/lstm_14/lstm_cell_14/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╕2*
(sequential_7/lstm_14/lstm_cell_14/MatMul√
9sequential_7/lstm_14/lstm_cell_14/MatMul_1/ReadVariableOpReadVariableOpBsequential_7_lstm_14_lstm_cell_14_matmul_1_readvariableop_resource* 
_output_shapes
:
ю╕*
dtype02;
9sequential_7/lstm_14/lstm_cell_14/MatMul_1/ReadVariableOp¤
*sequential_7/lstm_14/lstm_cell_14/MatMul_1MatMul#sequential_7/lstm_14/zeros:output:0Asequential_7/lstm_14/lstm_cell_14/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╕2,
*sequential_7/lstm_14/lstm_cell_14/MatMul_1Ї
%sequential_7/lstm_14/lstm_cell_14/addAddV22sequential_7/lstm_14/lstm_cell_14/MatMul:product:04sequential_7/lstm_14/lstm_cell_14/MatMul_1:product:0*
T0*(
_output_shapes
:         ╕2'
%sequential_7/lstm_14/lstm_cell_14/addє
8sequential_7/lstm_14/lstm_cell_14/BiasAdd/ReadVariableOpReadVariableOpAsequential_7_lstm_14_lstm_cell_14_biasadd_readvariableop_resource*
_output_shapes	
:╕*
dtype02:
8sequential_7/lstm_14/lstm_cell_14/BiasAdd/ReadVariableOpБ
)sequential_7/lstm_14/lstm_cell_14/BiasAddBiasAdd)sequential_7/lstm_14/lstm_cell_14/add:z:0@sequential_7/lstm_14/lstm_cell_14/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╕2+
)sequential_7/lstm_14/lstm_cell_14/BiasAddи
1sequential_7/lstm_14/lstm_cell_14/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :23
1sequential_7/lstm_14/lstm_cell_14/split/split_dim╦
'sequential_7/lstm_14/lstm_cell_14/splitSplit:sequential_7/lstm_14/lstm_cell_14/split/split_dim:output:02sequential_7/lstm_14/lstm_cell_14/BiasAdd:output:0*
T0*d
_output_shapesR
P:         ю:         ю:         ю:         ю*
	num_split2)
'sequential_7/lstm_14/lstm_cell_14/split╞
)sequential_7/lstm_14/lstm_cell_14/SigmoidSigmoid0sequential_7/lstm_14/lstm_cell_14/split:output:0*
T0*(
_output_shapes
:         ю2+
)sequential_7/lstm_14/lstm_cell_14/Sigmoid╩
+sequential_7/lstm_14/lstm_cell_14/Sigmoid_1Sigmoid0sequential_7/lstm_14/lstm_cell_14/split:output:1*
T0*(
_output_shapes
:         ю2-
+sequential_7/lstm_14/lstm_cell_14/Sigmoid_1р
%sequential_7/lstm_14/lstm_cell_14/mulMul/sequential_7/lstm_14/lstm_cell_14/Sigmoid_1:y:0%sequential_7/lstm_14/zeros_1:output:0*
T0*(
_output_shapes
:         ю2'
%sequential_7/lstm_14/lstm_cell_14/mul╜
&sequential_7/lstm_14/lstm_cell_14/ReluRelu0sequential_7/lstm_14/lstm_cell_14/split:output:2*
T0*(
_output_shapes
:         ю2(
&sequential_7/lstm_14/lstm_cell_14/Reluё
'sequential_7/lstm_14/lstm_cell_14/mul_1Mul-sequential_7/lstm_14/lstm_cell_14/Sigmoid:y:04sequential_7/lstm_14/lstm_cell_14/Relu:activations:0*
T0*(
_output_shapes
:         ю2)
'sequential_7/lstm_14/lstm_cell_14/mul_1ц
'sequential_7/lstm_14/lstm_cell_14/add_1AddV2)sequential_7/lstm_14/lstm_cell_14/mul:z:0+sequential_7/lstm_14/lstm_cell_14/mul_1:z:0*
T0*(
_output_shapes
:         ю2)
'sequential_7/lstm_14/lstm_cell_14/add_1╩
+sequential_7/lstm_14/lstm_cell_14/Sigmoid_2Sigmoid0sequential_7/lstm_14/lstm_cell_14/split:output:3*
T0*(
_output_shapes
:         ю2-
+sequential_7/lstm_14/lstm_cell_14/Sigmoid_2╝
(sequential_7/lstm_14/lstm_cell_14/Relu_1Relu+sequential_7/lstm_14/lstm_cell_14/add_1:z:0*
T0*(
_output_shapes
:         ю2*
(sequential_7/lstm_14/lstm_cell_14/Relu_1ї
'sequential_7/lstm_14/lstm_cell_14/mul_2Mul/sequential_7/lstm_14/lstm_cell_14/Sigmoid_2:y:06sequential_7/lstm_14/lstm_cell_14/Relu_1:activations:0*
T0*(
_output_shapes
:         ю2)
'sequential_7/lstm_14/lstm_cell_14/mul_2╣
2sequential_7/lstm_14/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ю   24
2sequential_7/lstm_14/TensorArrayV2_1/element_shapeМ
$sequential_7/lstm_14/TensorArrayV2_1TensorListReserve;sequential_7/lstm_14/TensorArrayV2_1/element_shape:output:0-sequential_7/lstm_14/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02&
$sequential_7/lstm_14/TensorArrayV2_1x
sequential_7/lstm_14/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
sequential_7/lstm_14/timeй
-sequential_7/lstm_14/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         2/
-sequential_7/lstm_14/while/maximum_iterationsФ
'sequential_7/lstm_14/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2)
'sequential_7/lstm_14/while/loop_counter╬
sequential_7/lstm_14/whileWhile0sequential_7/lstm_14/while/loop_counter:output:06sequential_7/lstm_14/while/maximum_iterations:output:0"sequential_7/lstm_14/time:output:0-sequential_7/lstm_14/TensorArrayV2_1:handle:0#sequential_7/lstm_14/zeros:output:0%sequential_7/lstm_14/zeros_1:output:0-sequential_7/lstm_14/strided_slice_1:output:0Lsequential_7/lstm_14/TensorArrayUnstack/TensorListFromTensor:output_handle:0@sequential_7_lstm_14_lstm_cell_14_matmul_readvariableop_resourceBsequential_7_lstm_14_lstm_cell_14_matmul_1_readvariableop_resourceAsequential_7_lstm_14_lstm_cell_14_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :         ю:         ю: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *4
body,R*
(sequential_7_lstm_14_while_body_26069532*4
cond,R*
(sequential_7_lstm_14_while_cond_26069531*M
output_shapes<
:: : : : :         ю:         ю: : : : : *
parallel_iterations 2
sequential_7/lstm_14/while▀
Esequential_7/lstm_14/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ю   2G
Esequential_7/lstm_14/TensorArrayV2Stack/TensorListStack/element_shape╜
7sequential_7/lstm_14/TensorArrayV2Stack/TensorListStackTensorListStack#sequential_7/lstm_14/while:output:3Nsequential_7/lstm_14/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:         ю*
element_dtype029
7sequential_7/lstm_14/TensorArrayV2Stack/TensorListStackл
*sequential_7/lstm_14/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         2,
*sequential_7/lstm_14/strided_slice_3/stackж
,sequential_7/lstm_14/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2.
,sequential_7/lstm_14/strided_slice_3/stack_1ж
,sequential_7/lstm_14/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_7/lstm_14/strided_slice_3/stack_2Щ
$sequential_7/lstm_14/strided_slice_3StridedSlice@sequential_7/lstm_14/TensorArrayV2Stack/TensorListStack:tensor:03sequential_7/lstm_14/strided_slice_3/stack:output:05sequential_7/lstm_14/strided_slice_3/stack_1:output:05sequential_7/lstm_14/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         ю*
shrink_axis_mask2&
$sequential_7/lstm_14/strided_slice_3г
%sequential_7/lstm_14/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2'
%sequential_7/lstm_14/transpose_1/perm·
 sequential_7/lstm_14/transpose_1	Transpose@sequential_7/lstm_14/TensorArrayV2Stack/TensorListStack:tensor:0.sequential_7/lstm_14/transpose_1/perm:output:0*
T0*,
_output_shapes
:         ю2"
 sequential_7/lstm_14/transpose_1Р
sequential_7/lstm_14/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
sequential_7/lstm_14/runtimeн
 sequential_7/dropout_14/IdentityIdentity$sequential_7/lstm_14/transpose_1:y:0*
T0*,
_output_shapes
:         ю2"
 sequential_7/dropout_14/IdentityС
sequential_7/lstm_15/ShapeShape)sequential_7/dropout_14/Identity:output:0*
T0*
_output_shapes
:2
sequential_7/lstm_15/ShapeЮ
(sequential_7/lstm_15/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(sequential_7/lstm_15/strided_slice/stackв
*sequential_7/lstm_15/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*sequential_7/lstm_15/strided_slice/stack_1в
*sequential_7/lstm_15/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*sequential_7/lstm_15/strided_slice/stack_2р
"sequential_7/lstm_15/strided_sliceStridedSlice#sequential_7/lstm_15/Shape:output:01sequential_7/lstm_15/strided_slice/stack:output:03sequential_7/lstm_15/strided_slice/stack_1:output:03sequential_7/lstm_15/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"sequential_7/lstm_15/strided_sliceЗ
 sequential_7/lstm_15/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :Ц2"
 sequential_7/lstm_15/zeros/mul/y└
sequential_7/lstm_15/zeros/mulMul+sequential_7/lstm_15/strided_slice:output:0)sequential_7/lstm_15/zeros/mul/y:output:0*
T0*
_output_shapes
: 2 
sequential_7/lstm_15/zeros/mulЙ
!sequential_7/lstm_15/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2#
!sequential_7/lstm_15/zeros/Less/y╗
sequential_7/lstm_15/zeros/LessLess"sequential_7/lstm_15/zeros/mul:z:0*sequential_7/lstm_15/zeros/Less/y:output:0*
T0*
_output_shapes
: 2!
sequential_7/lstm_15/zeros/LessН
#sequential_7/lstm_15/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :Ц2%
#sequential_7/lstm_15/zeros/packed/1╫
!sequential_7/lstm_15/zeros/packedPack+sequential_7/lstm_15/strided_slice:output:0,sequential_7/lstm_15/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2#
!sequential_7/lstm_15/zeros/packedЙ
 sequential_7/lstm_15/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 sequential_7/lstm_15/zeros/Const╩
sequential_7/lstm_15/zerosFill*sequential_7/lstm_15/zeros/packed:output:0)sequential_7/lstm_15/zeros/Const:output:0*
T0*(
_output_shapes
:         Ц2
sequential_7/lstm_15/zerosЛ
"sequential_7/lstm_15/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :Ц2$
"sequential_7/lstm_15/zeros_1/mul/y╞
 sequential_7/lstm_15/zeros_1/mulMul+sequential_7/lstm_15/strided_slice:output:0+sequential_7/lstm_15/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2"
 sequential_7/lstm_15/zeros_1/mulН
#sequential_7/lstm_15/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2%
#sequential_7/lstm_15/zeros_1/Less/y├
!sequential_7/lstm_15/zeros_1/LessLess$sequential_7/lstm_15/zeros_1/mul:z:0,sequential_7/lstm_15/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2#
!sequential_7/lstm_15/zeros_1/LessС
%sequential_7/lstm_15/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :Ц2'
%sequential_7/lstm_15/zeros_1/packed/1▌
#sequential_7/lstm_15/zeros_1/packedPack+sequential_7/lstm_15/strided_slice:output:0.sequential_7/lstm_15/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2%
#sequential_7/lstm_15/zeros_1/packedН
"sequential_7/lstm_15/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"sequential_7/lstm_15/zeros_1/Const╥
sequential_7/lstm_15/zeros_1Fill,sequential_7/lstm_15/zeros_1/packed:output:0+sequential_7/lstm_15/zeros_1/Const:output:0*
T0*(
_output_shapes
:         Ц2
sequential_7/lstm_15/zeros_1Я
#sequential_7/lstm_15/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2%
#sequential_7/lstm_15/transpose/perm▌
sequential_7/lstm_15/transpose	Transpose)sequential_7/dropout_14/Identity:output:0,sequential_7/lstm_15/transpose/perm:output:0*
T0*,
_output_shapes
:         ю2 
sequential_7/lstm_15/transposeО
sequential_7/lstm_15/Shape_1Shape"sequential_7/lstm_15/transpose:y:0*
T0*
_output_shapes
:2
sequential_7/lstm_15/Shape_1в
*sequential_7/lstm_15/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2,
*sequential_7/lstm_15/strided_slice_1/stackж
,sequential_7/lstm_15/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_7/lstm_15/strided_slice_1/stack_1ж
,sequential_7/lstm_15/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_7/lstm_15/strided_slice_1/stack_2ь
$sequential_7/lstm_15/strided_slice_1StridedSlice%sequential_7/lstm_15/Shape_1:output:03sequential_7/lstm_15/strided_slice_1/stack:output:05sequential_7/lstm_15/strided_slice_1/stack_1:output:05sequential_7/lstm_15/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2&
$sequential_7/lstm_15/strided_slice_1п
0sequential_7/lstm_15/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         22
0sequential_7/lstm_15/TensorArrayV2/element_shapeЖ
"sequential_7/lstm_15/TensorArrayV2TensorListReserve9sequential_7/lstm_15/TensorArrayV2/element_shape:output:0-sequential_7/lstm_15/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02$
"sequential_7/lstm_15/TensorArrayV2щ
Jsequential_7/lstm_15/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ю   2L
Jsequential_7/lstm_15/TensorArrayUnstack/TensorListFromTensor/element_shape╠
<sequential_7/lstm_15/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor"sequential_7/lstm_15/transpose:y:0Ssequential_7/lstm_15/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02>
<sequential_7/lstm_15/TensorArrayUnstack/TensorListFromTensorв
*sequential_7/lstm_15/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2,
*sequential_7/lstm_15/strided_slice_2/stackж
,sequential_7/lstm_15/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_7/lstm_15/strided_slice_2/stack_1ж
,sequential_7/lstm_15/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_7/lstm_15/strided_slice_2/stack_2√
$sequential_7/lstm_15/strided_slice_2StridedSlice"sequential_7/lstm_15/transpose:y:03sequential_7/lstm_15/strided_slice_2/stack:output:05sequential_7/lstm_15/strided_slice_2/stack_1:output:05sequential_7/lstm_15/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         ю*
shrink_axis_mask2&
$sequential_7/lstm_15/strided_slice_2ї
7sequential_7/lstm_15/lstm_cell_15/MatMul/ReadVariableOpReadVariableOp@sequential_7_lstm_15_lstm_cell_15_matmul_readvariableop_resource* 
_output_shapes
:
ю╪*
dtype029
7sequential_7/lstm_15/lstm_cell_15/MatMul/ReadVariableOpБ
(sequential_7/lstm_15/lstm_cell_15/MatMulMatMul-sequential_7/lstm_15/strided_slice_2:output:0?sequential_7/lstm_15/lstm_cell_15/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╪2*
(sequential_7/lstm_15/lstm_cell_15/MatMul√
9sequential_7/lstm_15/lstm_cell_15/MatMul_1/ReadVariableOpReadVariableOpBsequential_7_lstm_15_lstm_cell_15_matmul_1_readvariableop_resource* 
_output_shapes
:
Ц╪*
dtype02;
9sequential_7/lstm_15/lstm_cell_15/MatMul_1/ReadVariableOp¤
*sequential_7/lstm_15/lstm_cell_15/MatMul_1MatMul#sequential_7/lstm_15/zeros:output:0Asequential_7/lstm_15/lstm_cell_15/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╪2,
*sequential_7/lstm_15/lstm_cell_15/MatMul_1Ї
%sequential_7/lstm_15/lstm_cell_15/addAddV22sequential_7/lstm_15/lstm_cell_15/MatMul:product:04sequential_7/lstm_15/lstm_cell_15/MatMul_1:product:0*
T0*(
_output_shapes
:         ╪2'
%sequential_7/lstm_15/lstm_cell_15/addє
8sequential_7/lstm_15/lstm_cell_15/BiasAdd/ReadVariableOpReadVariableOpAsequential_7_lstm_15_lstm_cell_15_biasadd_readvariableop_resource*
_output_shapes	
:╪*
dtype02:
8sequential_7/lstm_15/lstm_cell_15/BiasAdd/ReadVariableOpБ
)sequential_7/lstm_15/lstm_cell_15/BiasAddBiasAdd)sequential_7/lstm_15/lstm_cell_15/add:z:0@sequential_7/lstm_15/lstm_cell_15/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╪2+
)sequential_7/lstm_15/lstm_cell_15/BiasAddи
1sequential_7/lstm_15/lstm_cell_15/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :23
1sequential_7/lstm_15/lstm_cell_15/split/split_dim╦
'sequential_7/lstm_15/lstm_cell_15/splitSplit:sequential_7/lstm_15/lstm_cell_15/split/split_dim:output:02sequential_7/lstm_15/lstm_cell_15/BiasAdd:output:0*
T0*d
_output_shapesR
P:         Ц:         Ц:         Ц:         Ц*
	num_split2)
'sequential_7/lstm_15/lstm_cell_15/split╞
)sequential_7/lstm_15/lstm_cell_15/SigmoidSigmoid0sequential_7/lstm_15/lstm_cell_15/split:output:0*
T0*(
_output_shapes
:         Ц2+
)sequential_7/lstm_15/lstm_cell_15/Sigmoid╩
+sequential_7/lstm_15/lstm_cell_15/Sigmoid_1Sigmoid0sequential_7/lstm_15/lstm_cell_15/split:output:1*
T0*(
_output_shapes
:         Ц2-
+sequential_7/lstm_15/lstm_cell_15/Sigmoid_1р
%sequential_7/lstm_15/lstm_cell_15/mulMul/sequential_7/lstm_15/lstm_cell_15/Sigmoid_1:y:0%sequential_7/lstm_15/zeros_1:output:0*
T0*(
_output_shapes
:         Ц2'
%sequential_7/lstm_15/lstm_cell_15/mul╜
&sequential_7/lstm_15/lstm_cell_15/ReluRelu0sequential_7/lstm_15/lstm_cell_15/split:output:2*
T0*(
_output_shapes
:         Ц2(
&sequential_7/lstm_15/lstm_cell_15/Reluё
'sequential_7/lstm_15/lstm_cell_15/mul_1Mul-sequential_7/lstm_15/lstm_cell_15/Sigmoid:y:04sequential_7/lstm_15/lstm_cell_15/Relu:activations:0*
T0*(
_output_shapes
:         Ц2)
'sequential_7/lstm_15/lstm_cell_15/mul_1ц
'sequential_7/lstm_15/lstm_cell_15/add_1AddV2)sequential_7/lstm_15/lstm_cell_15/mul:z:0+sequential_7/lstm_15/lstm_cell_15/mul_1:z:0*
T0*(
_output_shapes
:         Ц2)
'sequential_7/lstm_15/lstm_cell_15/add_1╩
+sequential_7/lstm_15/lstm_cell_15/Sigmoid_2Sigmoid0sequential_7/lstm_15/lstm_cell_15/split:output:3*
T0*(
_output_shapes
:         Ц2-
+sequential_7/lstm_15/lstm_cell_15/Sigmoid_2╝
(sequential_7/lstm_15/lstm_cell_15/Relu_1Relu+sequential_7/lstm_15/lstm_cell_15/add_1:z:0*
T0*(
_output_shapes
:         Ц2*
(sequential_7/lstm_15/lstm_cell_15/Relu_1ї
'sequential_7/lstm_15/lstm_cell_15/mul_2Mul/sequential_7/lstm_15/lstm_cell_15/Sigmoid_2:y:06sequential_7/lstm_15/lstm_cell_15/Relu_1:activations:0*
T0*(
_output_shapes
:         Ц2)
'sequential_7/lstm_15/lstm_cell_15/mul_2╣
2sequential_7/lstm_15/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    Ц   24
2sequential_7/lstm_15/TensorArrayV2_1/element_shapeМ
$sequential_7/lstm_15/TensorArrayV2_1TensorListReserve;sequential_7/lstm_15/TensorArrayV2_1/element_shape:output:0-sequential_7/lstm_15/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02&
$sequential_7/lstm_15/TensorArrayV2_1x
sequential_7/lstm_15/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
sequential_7/lstm_15/timeй
-sequential_7/lstm_15/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         2/
-sequential_7/lstm_15/while/maximum_iterationsФ
'sequential_7/lstm_15/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2)
'sequential_7/lstm_15/while/loop_counter╬
sequential_7/lstm_15/whileWhile0sequential_7/lstm_15/while/loop_counter:output:06sequential_7/lstm_15/while/maximum_iterations:output:0"sequential_7/lstm_15/time:output:0-sequential_7/lstm_15/TensorArrayV2_1:handle:0#sequential_7/lstm_15/zeros:output:0%sequential_7/lstm_15/zeros_1:output:0-sequential_7/lstm_15/strided_slice_1:output:0Lsequential_7/lstm_15/TensorArrayUnstack/TensorListFromTensor:output_handle:0@sequential_7_lstm_15_lstm_cell_15_matmul_readvariableop_resourceBsequential_7_lstm_15_lstm_cell_15_matmul_1_readvariableop_resourceAsequential_7_lstm_15_lstm_cell_15_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :         Ц:         Ц: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *4
body,R*
(sequential_7_lstm_15_while_body_26069680*4
cond,R*
(sequential_7_lstm_15_while_cond_26069679*M
output_shapes<
:: : : : :         Ц:         Ц: : : : : *
parallel_iterations 2
sequential_7/lstm_15/while▀
Esequential_7/lstm_15/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    Ц   2G
Esequential_7/lstm_15/TensorArrayV2Stack/TensorListStack/element_shape╜
7sequential_7/lstm_15/TensorArrayV2Stack/TensorListStackTensorListStack#sequential_7/lstm_15/while:output:3Nsequential_7/lstm_15/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:         Ц*
element_dtype029
7sequential_7/lstm_15/TensorArrayV2Stack/TensorListStackл
*sequential_7/lstm_15/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         2,
*sequential_7/lstm_15/strided_slice_3/stackж
,sequential_7/lstm_15/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2.
,sequential_7/lstm_15/strided_slice_3/stack_1ж
,sequential_7/lstm_15/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_7/lstm_15/strided_slice_3/stack_2Щ
$sequential_7/lstm_15/strided_slice_3StridedSlice@sequential_7/lstm_15/TensorArrayV2Stack/TensorListStack:tensor:03sequential_7/lstm_15/strided_slice_3/stack:output:05sequential_7/lstm_15/strided_slice_3/stack_1:output:05sequential_7/lstm_15/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         Ц*
shrink_axis_mask2&
$sequential_7/lstm_15/strided_slice_3г
%sequential_7/lstm_15/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2'
%sequential_7/lstm_15/transpose_1/perm·
 sequential_7/lstm_15/transpose_1	Transpose@sequential_7/lstm_15/TensorArrayV2Stack/TensorListStack:tensor:0.sequential_7/lstm_15/transpose_1/perm:output:0*
T0*,
_output_shapes
:         Ц2"
 sequential_7/lstm_15/transpose_1Р
sequential_7/lstm_15/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
sequential_7/lstm_15/runtimeн
 sequential_7/dropout_15/IdentityIdentity$sequential_7/lstm_15/transpose_1:y:0*
T0*,
_output_shapes
:         Ц2"
 sequential_7/dropout_15/Identity╓
-sequential_7/dense_7/Tensordot/ReadVariableOpReadVariableOp6sequential_7_dense_7_tensordot_readvariableop_resource*
_output_shapes
:	Ц*
dtype02/
-sequential_7/dense_7/Tensordot/ReadVariableOpФ
#sequential_7/dense_7/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2%
#sequential_7/dense_7/Tensordot/axesЫ
#sequential_7/dense_7/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2%
#sequential_7/dense_7/Tensordot/freeе
$sequential_7/dense_7/Tensordot/ShapeShape)sequential_7/dropout_15/Identity:output:0*
T0*
_output_shapes
:2&
$sequential_7/dense_7/Tensordot/ShapeЮ
,sequential_7/dense_7/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,sequential_7/dense_7/Tensordot/GatherV2/axis║
'sequential_7/dense_7/Tensordot/GatherV2GatherV2-sequential_7/dense_7/Tensordot/Shape:output:0,sequential_7/dense_7/Tensordot/free:output:05sequential_7/dense_7/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2)
'sequential_7/dense_7/Tensordot/GatherV2в
.sequential_7/dense_7/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.sequential_7/dense_7/Tensordot/GatherV2_1/axis└
)sequential_7/dense_7/Tensordot/GatherV2_1GatherV2-sequential_7/dense_7/Tensordot/Shape:output:0,sequential_7/dense_7/Tensordot/axes:output:07sequential_7/dense_7/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2+
)sequential_7/dense_7/Tensordot/GatherV2_1Ц
$sequential_7/dense_7/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2&
$sequential_7/dense_7/Tensordot/Const╘
#sequential_7/dense_7/Tensordot/ProdProd0sequential_7/dense_7/Tensordot/GatherV2:output:0-sequential_7/dense_7/Tensordot/Const:output:0*
T0*
_output_shapes
: 2%
#sequential_7/dense_7/Tensordot/ProdЪ
&sequential_7/dense_7/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2(
&sequential_7/dense_7/Tensordot/Const_1▄
%sequential_7/dense_7/Tensordot/Prod_1Prod2sequential_7/dense_7/Tensordot/GatherV2_1:output:0/sequential_7/dense_7/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2'
%sequential_7/dense_7/Tensordot/Prod_1Ъ
*sequential_7/dense_7/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2,
*sequential_7/dense_7/Tensordot/concat/axisЩ
%sequential_7/dense_7/Tensordot/concatConcatV2,sequential_7/dense_7/Tensordot/free:output:0,sequential_7/dense_7/Tensordot/axes:output:03sequential_7/dense_7/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2'
%sequential_7/dense_7/Tensordot/concatр
$sequential_7/dense_7/Tensordot/stackPack,sequential_7/dense_7/Tensordot/Prod:output:0.sequential_7/dense_7/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2&
$sequential_7/dense_7/Tensordot/stackє
(sequential_7/dense_7/Tensordot/transpose	Transpose)sequential_7/dropout_15/Identity:output:0.sequential_7/dense_7/Tensordot/concat:output:0*
T0*,
_output_shapes
:         Ц2*
(sequential_7/dense_7/Tensordot/transposeє
&sequential_7/dense_7/Tensordot/ReshapeReshape,sequential_7/dense_7/Tensordot/transpose:y:0-sequential_7/dense_7/Tensordot/stack:output:0*
T0*0
_output_shapes
:                  2(
&sequential_7/dense_7/Tensordot/ReshapeЄ
%sequential_7/dense_7/Tensordot/MatMulMatMul/sequential_7/dense_7/Tensordot/Reshape:output:05sequential_7/dense_7/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2'
%sequential_7/dense_7/Tensordot/MatMulЪ
&sequential_7/dense_7/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&sequential_7/dense_7/Tensordot/Const_2Ю
,sequential_7/dense_7/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,sequential_7/dense_7/Tensordot/concat_1/axisж
'sequential_7/dense_7/Tensordot/concat_1ConcatV20sequential_7/dense_7/Tensordot/GatherV2:output:0/sequential_7/dense_7/Tensordot/Const_2:output:05sequential_7/dense_7/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2)
'sequential_7/dense_7/Tensordot/concat_1ф
sequential_7/dense_7/TensordotReshape/sequential_7/dense_7/Tensordot/MatMul:product:00sequential_7/dense_7/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:         2 
sequential_7/dense_7/Tensordot╦
+sequential_7/dense_7/BiasAdd/ReadVariableOpReadVariableOp4sequential_7_dense_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02-
+sequential_7/dense_7/BiasAdd/ReadVariableOp█
sequential_7/dense_7/BiasAddBiasAdd'sequential_7/dense_7/Tensordot:output:03sequential_7/dense_7/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:         2
sequential_7/dense_7/BiasAddд
sequential_7/dense_7/SoftmaxSoftmax%sequential_7/dense_7/BiasAdd:output:0*
T0*+
_output_shapes
:         2
sequential_7/dense_7/SoftmaxЕ
IdentityIdentity&sequential_7/dense_7/Softmax:softmax:0^NoOp*
T0*+
_output_shapes
:         2

Identity╚
NoOpNoOp,^sequential_7/dense_7/BiasAdd/ReadVariableOp.^sequential_7/dense_7/Tensordot/ReadVariableOp9^sequential_7/lstm_14/lstm_cell_14/BiasAdd/ReadVariableOp8^sequential_7/lstm_14/lstm_cell_14/MatMul/ReadVariableOp:^sequential_7/lstm_14/lstm_cell_14/MatMul_1/ReadVariableOp^sequential_7/lstm_14/while9^sequential_7/lstm_15/lstm_cell_15/BiasAdd/ReadVariableOp8^sequential_7/lstm_15/lstm_cell_15/MatMul/ReadVariableOp:^sequential_7/lstm_15/lstm_cell_15/MatMul_1/ReadVariableOp^sequential_7/lstm_15/while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':         ]: : : : : : : : 2Z
+sequential_7/dense_7/BiasAdd/ReadVariableOp+sequential_7/dense_7/BiasAdd/ReadVariableOp2^
-sequential_7/dense_7/Tensordot/ReadVariableOp-sequential_7/dense_7/Tensordot/ReadVariableOp2t
8sequential_7/lstm_14/lstm_cell_14/BiasAdd/ReadVariableOp8sequential_7/lstm_14/lstm_cell_14/BiasAdd/ReadVariableOp2r
7sequential_7/lstm_14/lstm_cell_14/MatMul/ReadVariableOp7sequential_7/lstm_14/lstm_cell_14/MatMul/ReadVariableOp2v
9sequential_7/lstm_14/lstm_cell_14/MatMul_1/ReadVariableOp9sequential_7/lstm_14/lstm_cell_14/MatMul_1/ReadVariableOp28
sequential_7/lstm_14/whilesequential_7/lstm_14/while2t
8sequential_7/lstm_15/lstm_cell_15/BiasAdd/ReadVariableOp8sequential_7/lstm_15/lstm_cell_15/BiasAdd/ReadVariableOp2r
7sequential_7/lstm_15/lstm_cell_15/MatMul/ReadVariableOp7sequential_7/lstm_15/lstm_cell_15/MatMul/ReadVariableOp2v
9sequential_7/lstm_15/lstm_cell_15/MatMul_1/ReadVariableOp9sequential_7/lstm_15/lstm_cell_15/MatMul_1/ReadVariableOp28
sequential_7/lstm_15/whilesequential_7/lstm_15/while:Z V
+
_output_shapes
:         ]
'
_user_specified_namelstm_14_input
░
Ї
J__inference_sequential_7_layer_call_and_return_conditional_losses_26071961
lstm_14_input#
lstm_14_26071939:	]╕$
lstm_14_26071941:
ю╕
lstm_14_26071943:	╕$
lstm_15_26071947:
ю╪$
lstm_15_26071949:
Ц╪
lstm_15_26071951:	╪#
dense_7_26071955:	Ц
dense_7_26071957:
identityИвdense_7/StatefulPartitionedCallвlstm_14/StatefulPartitionedCallвlstm_15/StatefulPartitionedCall╡
lstm_14/StatefulPartitionedCallStatefulPartitionedCalllstm_14_inputlstm_14_26071939lstm_14_26071941lstm_14_26071943*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ю*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_lstm_14_layer_call_and_return_conditional_losses_260712092!
lstm_14/StatefulPartitionedCallГ
dropout_14/PartitionedCallPartitionedCall(lstm_14/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ю* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_dropout_14_layer_call_and_return_conditional_losses_260712222
dropout_14/PartitionedCall╦
lstm_15/StatefulPartitionedCallStatefulPartitionedCall#dropout_14/PartitionedCall:output:0lstm_15_26071947lstm_15_26071949lstm_15_26071951*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         Ц*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_lstm_15_layer_call_and_return_conditional_losses_260713742!
lstm_15/StatefulPartitionedCallГ
dropout_15/PartitionedCallPartitionedCall(lstm_15/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         Ц* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_dropout_15_layer_call_and_return_conditional_losses_260713872
dropout_15/PartitionedCall╢
dense_7/StatefulPartitionedCallStatefulPartitionedCall#dropout_15/PartitionedCall:output:0dense_7_26071955dense_7_26071957*
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
E__inference_dense_7_layer_call_and_return_conditional_losses_260714202!
dense_7/StatefulPartitionedCallЗ
IdentityIdentity(dense_7/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:         2

Identity┤
NoOpNoOp ^dense_7/StatefulPartitionedCall ^lstm_14/StatefulPartitionedCall ^lstm_15/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':         ]: : : : : : : : 2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall2B
lstm_14/StatefulPartitionedCalllstm_14/StatefulPartitionedCall2B
lstm_15/StatefulPartitionedCalllstm_15/StatefulPartitionedCall:Z V
+
_output_shapes
:         ]
'
_user_specified_namelstm_14_input
├\
а
E__inference_lstm_14_layer_call_and_return_conditional_losses_26073021
inputs_0>
+lstm_cell_14_matmul_readvariableop_resource:	]╕A
-lstm_cell_14_matmul_1_readvariableop_resource:
ю╕;
,lstm_cell_14_biasadd_readvariableop_resource:	╕
identityИв#lstm_cell_14/BiasAdd/ReadVariableOpв"lstm_cell_14/MatMul/ReadVariableOpв$lstm_cell_14/MatMul_1/ReadVariableOpвwhileF
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
B :ю2
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
B :ю2
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
:         ю2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :ю2
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
B :ю2
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
:         ю2	
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
"lstm_cell_14/MatMul/ReadVariableOpReadVariableOp+lstm_cell_14_matmul_readvariableop_resource*
_output_shapes
:	]╕*
dtype02$
"lstm_cell_14/MatMul/ReadVariableOpн
lstm_cell_14/MatMulMatMulstrided_slice_2:output:0*lstm_cell_14/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╕2
lstm_cell_14/MatMul╝
$lstm_cell_14/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_14_matmul_1_readvariableop_resource* 
_output_shapes
:
ю╕*
dtype02&
$lstm_cell_14/MatMul_1/ReadVariableOpй
lstm_cell_14/MatMul_1MatMulzeros:output:0,lstm_cell_14/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╕2
lstm_cell_14/MatMul_1а
lstm_cell_14/addAddV2lstm_cell_14/MatMul:product:0lstm_cell_14/MatMul_1:product:0*
T0*(
_output_shapes
:         ╕2
lstm_cell_14/add┤
#lstm_cell_14/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_14_biasadd_readvariableop_resource*
_output_shapes	
:╕*
dtype02%
#lstm_cell_14/BiasAdd/ReadVariableOpн
lstm_cell_14/BiasAddBiasAddlstm_cell_14/add:z:0+lstm_cell_14/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╕2
lstm_cell_14/BiasAdd~
lstm_cell_14/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_14/split/split_dimў
lstm_cell_14/splitSplit%lstm_cell_14/split/split_dim:output:0lstm_cell_14/BiasAdd:output:0*
T0*d
_output_shapesR
P:         ю:         ю:         ю:         ю*
	num_split2
lstm_cell_14/splitЗ
lstm_cell_14/SigmoidSigmoidlstm_cell_14/split:output:0*
T0*(
_output_shapes
:         ю2
lstm_cell_14/SigmoidЛ
lstm_cell_14/Sigmoid_1Sigmoidlstm_cell_14/split:output:1*
T0*(
_output_shapes
:         ю2
lstm_cell_14/Sigmoid_1М
lstm_cell_14/mulMullstm_cell_14/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:         ю2
lstm_cell_14/mul~
lstm_cell_14/ReluRelulstm_cell_14/split:output:2*
T0*(
_output_shapes
:         ю2
lstm_cell_14/ReluЭ
lstm_cell_14/mul_1Mullstm_cell_14/Sigmoid:y:0lstm_cell_14/Relu:activations:0*
T0*(
_output_shapes
:         ю2
lstm_cell_14/mul_1Т
lstm_cell_14/add_1AddV2lstm_cell_14/mul:z:0lstm_cell_14/mul_1:z:0*
T0*(
_output_shapes
:         ю2
lstm_cell_14/add_1Л
lstm_cell_14/Sigmoid_2Sigmoidlstm_cell_14/split:output:3*
T0*(
_output_shapes
:         ю2
lstm_cell_14/Sigmoid_2}
lstm_cell_14/Relu_1Relulstm_cell_14/add_1:z:0*
T0*(
_output_shapes
:         ю2
lstm_cell_14/Relu_1б
lstm_cell_14/mul_2Mullstm_cell_14/Sigmoid_2:y:0!lstm_cell_14/Relu_1:activations:0*
T0*(
_output_shapes
:         ю2
lstm_cell_14/mul_2П
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ю   2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_14_matmul_readvariableop_resource-lstm_cell_14_matmul_1_readvariableop_resource,lstm_cell_14_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :         ю:         ю: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_26072937*
condR
while_cond_26072936*M
output_shapes<
:: : : : :         ю:         ю: : : : : *
parallel_iterations 2
while╡
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ю   22
0TensorArrayV2Stack/TensorListStack/element_shapeЄ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:                  ю*
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
:         ю*
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
!:                  ю2
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
!:                  ю2

Identity╚
NoOpNoOp$^lstm_cell_14/BiasAdd/ReadVariableOp#^lstm_cell_14/MatMul/ReadVariableOp%^lstm_cell_14/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  ]: : : 2J
#lstm_cell_14/BiasAdd/ReadVariableOp#lstm_cell_14/BiasAdd/ReadVariableOp2H
"lstm_cell_14/MatMul/ReadVariableOp"lstm_cell_14/MatMul/ReadVariableOp2L
$lstm_cell_14/MatMul_1/ReadVariableOp$lstm_cell_14/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :                  ]
"
_user_specified_name
inputs/0
┤?
╓
while_body_26073612
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
3while_lstm_cell_15_matmul_readvariableop_resource_0:
ю╪I
5while_lstm_cell_15_matmul_1_readvariableop_resource_0:
Ц╪C
4while_lstm_cell_15_biasadd_readvariableop_resource_0:	╪
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
1while_lstm_cell_15_matmul_readvariableop_resource:
ю╪G
3while_lstm_cell_15_matmul_1_readvariableop_resource:
Ц╪A
2while_lstm_cell_15_biasadd_readvariableop_resource:	╪Ив)while/lstm_cell_15/BiasAdd/ReadVariableOpв(while/lstm_cell_15/MatMul/ReadVariableOpв*while/lstm_cell_15/MatMul_1/ReadVariableOp├
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ю   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape╘
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:         ю*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem╩
(while/lstm_cell_15/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_15_matmul_readvariableop_resource_0* 
_output_shapes
:
ю╪*
dtype02*
(while/lstm_cell_15/MatMul/ReadVariableOp╫
while/lstm_cell_15/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_15/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╪2
while/lstm_cell_15/MatMul╨
*while/lstm_cell_15/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_15_matmul_1_readvariableop_resource_0* 
_output_shapes
:
Ц╪*
dtype02,
*while/lstm_cell_15/MatMul_1/ReadVariableOp└
while/lstm_cell_15/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_15/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╪2
while/lstm_cell_15/MatMul_1╕
while/lstm_cell_15/addAddV2#while/lstm_cell_15/MatMul:product:0%while/lstm_cell_15/MatMul_1:product:0*
T0*(
_output_shapes
:         ╪2
while/lstm_cell_15/add╚
)while/lstm_cell_15/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_15_biasadd_readvariableop_resource_0*
_output_shapes	
:╪*
dtype02+
)while/lstm_cell_15/BiasAdd/ReadVariableOp┼
while/lstm_cell_15/BiasAddBiasAddwhile/lstm_cell_15/add:z:01while/lstm_cell_15/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╪2
while/lstm_cell_15/BiasAddК
"while/lstm_cell_15/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_15/split/split_dimП
while/lstm_cell_15/splitSplit+while/lstm_cell_15/split/split_dim:output:0#while/lstm_cell_15/BiasAdd:output:0*
T0*d
_output_shapesR
P:         Ц:         Ц:         Ц:         Ц*
	num_split2
while/lstm_cell_15/splitЩ
while/lstm_cell_15/SigmoidSigmoid!while/lstm_cell_15/split:output:0*
T0*(
_output_shapes
:         Ц2
while/lstm_cell_15/SigmoidЭ
while/lstm_cell_15/Sigmoid_1Sigmoid!while/lstm_cell_15/split:output:1*
T0*(
_output_shapes
:         Ц2
while/lstm_cell_15/Sigmoid_1б
while/lstm_cell_15/mulMul while/lstm_cell_15/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:         Ц2
while/lstm_cell_15/mulР
while/lstm_cell_15/ReluRelu!while/lstm_cell_15/split:output:2*
T0*(
_output_shapes
:         Ц2
while/lstm_cell_15/Relu╡
while/lstm_cell_15/mul_1Mulwhile/lstm_cell_15/Sigmoid:y:0%while/lstm_cell_15/Relu:activations:0*
T0*(
_output_shapes
:         Ц2
while/lstm_cell_15/mul_1к
while/lstm_cell_15/add_1AddV2while/lstm_cell_15/mul:z:0while/lstm_cell_15/mul_1:z:0*
T0*(
_output_shapes
:         Ц2
while/lstm_cell_15/add_1Э
while/lstm_cell_15/Sigmoid_2Sigmoid!while/lstm_cell_15/split:output:3*
T0*(
_output_shapes
:         Ц2
while/lstm_cell_15/Sigmoid_2П
while/lstm_cell_15/Relu_1Reluwhile/lstm_cell_15/add_1:z:0*
T0*(
_output_shapes
:         Ц2
while/lstm_cell_15/Relu_1╣
while/lstm_cell_15/mul_2Mul while/lstm_cell_15/Sigmoid_2:y:0'while/lstm_cell_15/Relu_1:activations:0*
T0*(
_output_shapes
:         Ц2
while/lstm_cell_15/mul_2р
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_15/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_15/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:         Ц2
while/Identity_4О
while/Identity_5Identitywhile/lstm_cell_15/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:         Ц2
while/Identity_5▐

while/NoOpNoOp*^while/lstm_cell_15/BiasAdd/ReadVariableOp)^while/lstm_cell_15/MatMul/ReadVariableOp+^while/lstm_cell_15/MatMul_1/ReadVariableOp*"
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
2while_lstm_cell_15_biasadd_readvariableop_resource4while_lstm_cell_15_biasadd_readvariableop_resource_0"l
3while_lstm_cell_15_matmul_1_readvariableop_resource5while_lstm_cell_15_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_15_matmul_readvariableop_resource3while_lstm_cell_15_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :         Ц:         Ц: : : : : 2V
)while/lstm_cell_15/BiasAdd/ReadVariableOp)while/lstm_cell_15/BiasAdd/ReadVariableOp2T
(while/lstm_cell_15/MatMul/ReadVariableOp(while/lstm_cell_15/MatMul/ReadVariableOp2X
*while/lstm_cell_15/MatMul_1/ReadVariableOp*while/lstm_cell_15/MatMul_1/ReadVariableOp: 
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
:         Ц:.*
(
_output_shapes
:         Ц:

_output_shapes
: :

_output_shapes
: 
╪
I
-__inference_dropout_15_layer_call_fn_26074064

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
:         Ц* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_dropout_15_layer_call_and_return_conditional_losses_260713872
PartitionedCallq
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:         Ц2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         Ц:T P
,
_output_shapes
:         Ц
 
_user_specified_nameinputs
у
╗
*__inference_lstm_15_layer_call_fn_26074020
inputs_0
unknown:
ю╪
	unknown_0:
Ц╪
	unknown_1:	╪
identityИвStatefulPartitionedCallТ
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:                  Ц*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_lstm_15_layer_call_and_return_conditional_losses_260707902
StatefulPartitionedCallЙ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:                  Ц2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':                  ю: : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
5
_output_shapes#
!:                  ю
"
_user_specified_name
inputs/0
╔\
б
E__inference_lstm_15_layer_call_and_return_conditional_losses_26073545
inputs_0?
+lstm_cell_15_matmul_readvariableop_resource:
ю╪A
-lstm_cell_15_matmul_1_readvariableop_resource:
Ц╪;
,lstm_cell_15_biasadd_readvariableop_resource:	╪
identityИв#lstm_cell_15/BiasAdd/ReadVariableOpв"lstm_cell_15/MatMul/ReadVariableOpв$lstm_cell_15/MatMul_1/ReadVariableOpвwhileF
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
B :Ц2
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
B :Ц2
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
:         Ц2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :Ц2
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
B :Ц2
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
:         Ц2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permЖ
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*5
_output_shapes#
!:                  ю2
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
valueB"    ю   27
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
strided_slice_2/stack_2¤
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         ю*
shrink_axis_mask2
strided_slice_2╢
"lstm_cell_15/MatMul/ReadVariableOpReadVariableOp+lstm_cell_15_matmul_readvariableop_resource* 
_output_shapes
:
ю╪*
dtype02$
"lstm_cell_15/MatMul/ReadVariableOpн
lstm_cell_15/MatMulMatMulstrided_slice_2:output:0*lstm_cell_15/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╪2
lstm_cell_15/MatMul╝
$lstm_cell_15/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_15_matmul_1_readvariableop_resource* 
_output_shapes
:
Ц╪*
dtype02&
$lstm_cell_15/MatMul_1/ReadVariableOpй
lstm_cell_15/MatMul_1MatMulzeros:output:0,lstm_cell_15/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╪2
lstm_cell_15/MatMul_1а
lstm_cell_15/addAddV2lstm_cell_15/MatMul:product:0lstm_cell_15/MatMul_1:product:0*
T0*(
_output_shapes
:         ╪2
lstm_cell_15/add┤
#lstm_cell_15/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_15_biasadd_readvariableop_resource*
_output_shapes	
:╪*
dtype02%
#lstm_cell_15/BiasAdd/ReadVariableOpн
lstm_cell_15/BiasAddBiasAddlstm_cell_15/add:z:0+lstm_cell_15/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╪2
lstm_cell_15/BiasAdd~
lstm_cell_15/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_15/split/split_dimў
lstm_cell_15/splitSplit%lstm_cell_15/split/split_dim:output:0lstm_cell_15/BiasAdd:output:0*
T0*d
_output_shapesR
P:         Ц:         Ц:         Ц:         Ц*
	num_split2
lstm_cell_15/splitЗ
lstm_cell_15/SigmoidSigmoidlstm_cell_15/split:output:0*
T0*(
_output_shapes
:         Ц2
lstm_cell_15/SigmoidЛ
lstm_cell_15/Sigmoid_1Sigmoidlstm_cell_15/split:output:1*
T0*(
_output_shapes
:         Ц2
lstm_cell_15/Sigmoid_1М
lstm_cell_15/mulMullstm_cell_15/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:         Ц2
lstm_cell_15/mul~
lstm_cell_15/ReluRelulstm_cell_15/split:output:2*
T0*(
_output_shapes
:         Ц2
lstm_cell_15/ReluЭ
lstm_cell_15/mul_1Mullstm_cell_15/Sigmoid:y:0lstm_cell_15/Relu:activations:0*
T0*(
_output_shapes
:         Ц2
lstm_cell_15/mul_1Т
lstm_cell_15/add_1AddV2lstm_cell_15/mul:z:0lstm_cell_15/mul_1:z:0*
T0*(
_output_shapes
:         Ц2
lstm_cell_15/add_1Л
lstm_cell_15/Sigmoid_2Sigmoidlstm_cell_15/split:output:3*
T0*(
_output_shapes
:         Ц2
lstm_cell_15/Sigmoid_2}
lstm_cell_15/Relu_1Relulstm_cell_15/add_1:z:0*
T0*(
_output_shapes
:         Ц2
lstm_cell_15/Relu_1б
lstm_cell_15/mul_2Mullstm_cell_15/Sigmoid_2:y:0!lstm_cell_15/Relu_1:activations:0*
T0*(
_output_shapes
:         Ц2
lstm_cell_15/mul_2П
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    Ц   2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_15_matmul_readvariableop_resource-lstm_cell_15_matmul_1_readvariableop_resource,lstm_cell_15_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :         Ц:         Ц: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_26073461*
condR
while_cond_26073460*M
output_shapes<
:: : : : :         Ц:         Ц: : : : : *
parallel_iterations 2
while╡
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    Ц   22
0TensorArrayV2Stack/TensorListStack/element_shapeЄ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:                  Ц*
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
:         Ц*
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
!:                  Ц2
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
!:                  Ц2

Identity╚
NoOpNoOp$^lstm_cell_15/BiasAdd/ReadVariableOp#^lstm_cell_15/MatMul/ReadVariableOp%^lstm_cell_15/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':                  ю: : : 2J
#lstm_cell_15/BiasAdd/ReadVariableOp#lstm_cell_15/BiasAdd/ReadVariableOp2H
"lstm_cell_15/MatMul/ReadVariableOp"lstm_cell_15/MatMul/ReadVariableOp2L
$lstm_cell_15/MatMul_1/ReadVariableOp$lstm_cell_15/MatMul_1/ReadVariableOp2
whilewhile:_ [
5
_output_shapes#
!:                  ю
"
_user_specified_name
inputs/0
Д\
Ю
E__inference_lstm_14_layer_call_and_return_conditional_losses_26071209

inputs>
+lstm_cell_14_matmul_readvariableop_resource:	]╕A
-lstm_cell_14_matmul_1_readvariableop_resource:
ю╕;
,lstm_cell_14_biasadd_readvariableop_resource:	╕
identityИв#lstm_cell_14/BiasAdd/ReadVariableOpв"lstm_cell_14/MatMul/ReadVariableOpв$lstm_cell_14/MatMul_1/ReadVariableOpвwhileD
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
B :ю2
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
B :ю2
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
:         ю2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :ю2
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
B :ю2
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
:         ю2	
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
"lstm_cell_14/MatMul/ReadVariableOpReadVariableOp+lstm_cell_14_matmul_readvariableop_resource*
_output_shapes
:	]╕*
dtype02$
"lstm_cell_14/MatMul/ReadVariableOpн
lstm_cell_14/MatMulMatMulstrided_slice_2:output:0*lstm_cell_14/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╕2
lstm_cell_14/MatMul╝
$lstm_cell_14/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_14_matmul_1_readvariableop_resource* 
_output_shapes
:
ю╕*
dtype02&
$lstm_cell_14/MatMul_1/ReadVariableOpй
lstm_cell_14/MatMul_1MatMulzeros:output:0,lstm_cell_14/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╕2
lstm_cell_14/MatMul_1а
lstm_cell_14/addAddV2lstm_cell_14/MatMul:product:0lstm_cell_14/MatMul_1:product:0*
T0*(
_output_shapes
:         ╕2
lstm_cell_14/add┤
#lstm_cell_14/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_14_biasadd_readvariableop_resource*
_output_shapes	
:╕*
dtype02%
#lstm_cell_14/BiasAdd/ReadVariableOpн
lstm_cell_14/BiasAddBiasAddlstm_cell_14/add:z:0+lstm_cell_14/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╕2
lstm_cell_14/BiasAdd~
lstm_cell_14/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_14/split/split_dimў
lstm_cell_14/splitSplit%lstm_cell_14/split/split_dim:output:0lstm_cell_14/BiasAdd:output:0*
T0*d
_output_shapesR
P:         ю:         ю:         ю:         ю*
	num_split2
lstm_cell_14/splitЗ
lstm_cell_14/SigmoidSigmoidlstm_cell_14/split:output:0*
T0*(
_output_shapes
:         ю2
lstm_cell_14/SigmoidЛ
lstm_cell_14/Sigmoid_1Sigmoidlstm_cell_14/split:output:1*
T0*(
_output_shapes
:         ю2
lstm_cell_14/Sigmoid_1М
lstm_cell_14/mulMullstm_cell_14/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:         ю2
lstm_cell_14/mul~
lstm_cell_14/ReluRelulstm_cell_14/split:output:2*
T0*(
_output_shapes
:         ю2
lstm_cell_14/ReluЭ
lstm_cell_14/mul_1Mullstm_cell_14/Sigmoid:y:0lstm_cell_14/Relu:activations:0*
T0*(
_output_shapes
:         ю2
lstm_cell_14/mul_1Т
lstm_cell_14/add_1AddV2lstm_cell_14/mul:z:0lstm_cell_14/mul_1:z:0*
T0*(
_output_shapes
:         ю2
lstm_cell_14/add_1Л
lstm_cell_14/Sigmoid_2Sigmoidlstm_cell_14/split:output:3*
T0*(
_output_shapes
:         ю2
lstm_cell_14/Sigmoid_2}
lstm_cell_14/Relu_1Relulstm_cell_14/add_1:z:0*
T0*(
_output_shapes
:         ю2
lstm_cell_14/Relu_1б
lstm_cell_14/mul_2Mullstm_cell_14/Sigmoid_2:y:0!lstm_cell_14/Relu_1:activations:0*
T0*(
_output_shapes
:         ю2
lstm_cell_14/mul_2П
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ю   2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_14_matmul_readvariableop_resource-lstm_cell_14_matmul_1_readvariableop_resource,lstm_cell_14_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :         ю:         ю: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_26071125*
condR
while_cond_26071124*M
output_shapes<
:: : : : :         ю:         ю: : : : : *
parallel_iterations 2
while╡
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ю   22
0TensorArrayV2Stack/TensorListStack/element_shapeщ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:         ю*
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
:         ю*
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
:         ю2
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
:         ю2

Identity╚
NoOpNoOp$^lstm_cell_14/BiasAdd/ReadVariableOp#^lstm_cell_14/MatMul/ReadVariableOp%^lstm_cell_14/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         ]: : : 2J
#lstm_cell_14/BiasAdd/ReadVariableOp#lstm_cell_14/BiasAdd/ReadVariableOp2H
"lstm_cell_14/MatMul/ReadVariableOp"lstm_cell_14/MatMul/ReadVariableOp2L
$lstm_cell_14/MatMul_1/ReadVariableOp$lstm_cell_14/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:         ]
 
_user_specified_nameinputs
╘

э
lstm_14_while_cond_26072402,
(lstm_14_while_lstm_14_while_loop_counter2
.lstm_14_while_lstm_14_while_maximum_iterations
lstm_14_while_placeholder
lstm_14_while_placeholder_1
lstm_14_while_placeholder_2
lstm_14_while_placeholder_3.
*lstm_14_while_less_lstm_14_strided_slice_1F
Blstm_14_while_lstm_14_while_cond_26072402___redundant_placeholder0F
Blstm_14_while_lstm_14_while_cond_26072402___redundant_placeholder1F
Blstm_14_while_lstm_14_while_cond_26072402___redundant_placeholder2F
Blstm_14_while_lstm_14_while_cond_26072402___redundant_placeholder3
lstm_14_while_identity
Ш
lstm_14/while/LessLesslstm_14_while_placeholder*lstm_14_while_less_lstm_14_strided_slice_1*
T0*
_output_shapes
: 2
lstm_14/while/Lessu
lstm_14/while/IdentityIdentitylstm_14/while/Less:z:0*
T0
*
_output_shapes
: 2
lstm_14/while/Identity"9
lstm_14_while_identitylstm_14/while/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :         ю:         ю: ::::: 
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
:         ю:.*
(
_output_shapes
:         ю:

_output_shapes
: :

_output_shapes
:
у
═
while_cond_26071289
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_26071289___redundant_placeholder06
2while_while_cond_26071289___redundant_placeholder16
2while_while_cond_26071289___redundant_placeholder26
2while_while_cond_26071289___redundant_placeholder3
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
B: : : : :         Ц:         Ц: ::::: 
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
:         Ц:.*
(
_output_shapes
:         Ц:

_output_shapes
: :

_output_shapes
:
╢
╕
*__inference_lstm_14_layer_call_fn_26073356

inputs
unknown:	]╕
	unknown_0:
ю╕
	unknown_1:	╕
identityИвStatefulPartitionedCallЗ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ю*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_lstm_14_layer_call_and_return_conditional_losses_260712092
StatefulPartitionedCallА
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:         ю2

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
╫
g
H__inference_dropout_14_layer_call_and_return_conditional_losses_26071672

inputs
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *лкк?2
dropout/Constx
dropout/MulMulinputsdropout/Const:output:0*
T0*,
_output_shapes
:         ю2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape╣
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:         ю*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А>2
dropout/GreaterEqual/y├
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:         ю2
dropout/GreaterEqualД
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:         ю2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:         ю2
dropout/Mul_1j
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:         ю2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ю:T P
,
_output_shapes
:         ю
 
_user_specified_nameinputs
╫
g
H__inference_dropout_15_layer_call_and_return_conditional_losses_26074059

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
:         Ц2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape╣
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:         Ц*
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
:         Ц2
dropout/GreaterEqualД
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:         Ц2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:         Ц2
dropout/Mul_1j
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:         Ц2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         Ц:T P
,
_output_shapes
:         Ц
 
_user_specified_nameinputs
Ч

╠
/__inference_sequential_7_layer_call_fn_26072719

inputs
unknown:	]╕
	unknown_0:
ю╕
	unknown_1:	╕
	unknown_2:
ю╪
	unknown_3:
Ц╪
	unknown_4:	╪
	unknown_5:	Ц
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
J__inference_sequential_7_layer_call_and_return_conditional_losses_260718962
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
у
═
while_cond_26073611
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_26073611___redundant_placeholder06
2while_while_cond_26073611___redundant_placeholder16
2while_while_cond_26073611___redundant_placeholder26
2while_while_cond_26073611___redundant_placeholder3
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
B: : : : :         Ц:         Ц: ::::: 
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
:         Ц:.*
(
_output_shapes
:         Ц:

_output_shapes
: :

_output_shapes
:
╣
╣
*__inference_lstm_15_layer_call_fn_26074031

inputs
unknown:
ю╪
	unknown_0:
Ц╪
	unknown_1:	╪
identityИвStatefulPartitionedCallЗ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         Ц*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_lstm_15_layer_call_and_return_conditional_losses_260713742
StatefulPartitionedCallА
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:         Ц2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:         ю: : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:         ю
 
_user_specified_nameinputs
Ў°
З
J__inference_sequential_7_layer_call_and_return_conditional_losses_26072336

inputsF
3lstm_14_lstm_cell_14_matmul_readvariableop_resource:	]╕I
5lstm_14_lstm_cell_14_matmul_1_readvariableop_resource:
ю╕C
4lstm_14_lstm_cell_14_biasadd_readvariableop_resource:	╕G
3lstm_15_lstm_cell_15_matmul_readvariableop_resource:
ю╪I
5lstm_15_lstm_cell_15_matmul_1_readvariableop_resource:
Ц╪C
4lstm_15_lstm_cell_15_biasadd_readvariableop_resource:	╪<
)dense_7_tensordot_readvariableop_resource:	Ц5
'dense_7_biasadd_readvariableop_resource:
identityИвdense_7/BiasAdd/ReadVariableOpв dense_7/Tensordot/ReadVariableOpв+lstm_14/lstm_cell_14/BiasAdd/ReadVariableOpв*lstm_14/lstm_cell_14/MatMul/ReadVariableOpв,lstm_14/lstm_cell_14/MatMul_1/ReadVariableOpвlstm_14/whileв+lstm_15/lstm_cell_15/BiasAdd/ReadVariableOpв*lstm_15/lstm_cell_15/MatMul/ReadVariableOpв,lstm_15/lstm_cell_15/MatMul_1/ReadVariableOpвlstm_15/whileT
lstm_14/ShapeShapeinputs*
T0*
_output_shapes
:2
lstm_14/ShapeД
lstm_14/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_14/strided_slice/stackИ
lstm_14/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_14/strided_slice/stack_1И
lstm_14/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_14/strided_slice/stack_2Т
lstm_14/strided_sliceStridedSlicelstm_14/Shape:output:0$lstm_14/strided_slice/stack:output:0&lstm_14/strided_slice/stack_1:output:0&lstm_14/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_14/strided_slicem
lstm_14/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :ю2
lstm_14/zeros/mul/yМ
lstm_14/zeros/mulMullstm_14/strided_slice:output:0lstm_14/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_14/zeros/mulo
lstm_14/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
lstm_14/zeros/Less/yЗ
lstm_14/zeros/LessLesslstm_14/zeros/mul:z:0lstm_14/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_14/zeros/Lesss
lstm_14/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :ю2
lstm_14/zeros/packed/1г
lstm_14/zeros/packedPacklstm_14/strided_slice:output:0lstm_14/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_14/zeros/packedo
lstm_14/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_14/zeros/ConstЦ
lstm_14/zerosFilllstm_14/zeros/packed:output:0lstm_14/zeros/Const:output:0*
T0*(
_output_shapes
:         ю2
lstm_14/zerosq
lstm_14/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :ю2
lstm_14/zeros_1/mul/yТ
lstm_14/zeros_1/mulMullstm_14/strided_slice:output:0lstm_14/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_14/zeros_1/muls
lstm_14/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
lstm_14/zeros_1/Less/yП
lstm_14/zeros_1/LessLesslstm_14/zeros_1/mul:z:0lstm_14/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_14/zeros_1/Lessw
lstm_14/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :ю2
lstm_14/zeros_1/packed/1й
lstm_14/zeros_1/packedPacklstm_14/strided_slice:output:0!lstm_14/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_14/zeros_1/packeds
lstm_14/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_14/zeros_1/ConstЮ
lstm_14/zeros_1Filllstm_14/zeros_1/packed:output:0lstm_14/zeros_1/Const:output:0*
T0*(
_output_shapes
:         ю2
lstm_14/zeros_1Е
lstm_14/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_14/transpose/permТ
lstm_14/transpose	Transposeinputslstm_14/transpose/perm:output:0*
T0*+
_output_shapes
:         ]2
lstm_14/transposeg
lstm_14/Shape_1Shapelstm_14/transpose:y:0*
T0*
_output_shapes
:2
lstm_14/Shape_1И
lstm_14/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_14/strided_slice_1/stackМ
lstm_14/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_14/strided_slice_1/stack_1М
lstm_14/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_14/strided_slice_1/stack_2Ю
lstm_14/strided_slice_1StridedSlicelstm_14/Shape_1:output:0&lstm_14/strided_slice_1/stack:output:0(lstm_14/strided_slice_1/stack_1:output:0(lstm_14/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_14/strided_slice_1Х
#lstm_14/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         2%
#lstm_14/TensorArrayV2/element_shape╥
lstm_14/TensorArrayV2TensorListReserve,lstm_14/TensorArrayV2/element_shape:output:0 lstm_14/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_14/TensorArrayV2╧
=lstm_14/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ]   2?
=lstm_14/TensorArrayUnstack/TensorListFromTensor/element_shapeШ
/lstm_14/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_14/transpose:y:0Flstm_14/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type021
/lstm_14/TensorArrayUnstack/TensorListFromTensorИ
lstm_14/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_14/strided_slice_2/stackМ
lstm_14/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_14/strided_slice_2/stack_1М
lstm_14/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_14/strided_slice_2/stack_2м
lstm_14/strided_slice_2StridedSlicelstm_14/transpose:y:0&lstm_14/strided_slice_2/stack:output:0(lstm_14/strided_slice_2/stack_1:output:0(lstm_14/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         ]*
shrink_axis_mask2
lstm_14/strided_slice_2═
*lstm_14/lstm_cell_14/MatMul/ReadVariableOpReadVariableOp3lstm_14_lstm_cell_14_matmul_readvariableop_resource*
_output_shapes
:	]╕*
dtype02,
*lstm_14/lstm_cell_14/MatMul/ReadVariableOp═
lstm_14/lstm_cell_14/MatMulMatMul lstm_14/strided_slice_2:output:02lstm_14/lstm_cell_14/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╕2
lstm_14/lstm_cell_14/MatMul╘
,lstm_14/lstm_cell_14/MatMul_1/ReadVariableOpReadVariableOp5lstm_14_lstm_cell_14_matmul_1_readvariableop_resource* 
_output_shapes
:
ю╕*
dtype02.
,lstm_14/lstm_cell_14/MatMul_1/ReadVariableOp╔
lstm_14/lstm_cell_14/MatMul_1MatMullstm_14/zeros:output:04lstm_14/lstm_cell_14/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╕2
lstm_14/lstm_cell_14/MatMul_1└
lstm_14/lstm_cell_14/addAddV2%lstm_14/lstm_cell_14/MatMul:product:0'lstm_14/lstm_cell_14/MatMul_1:product:0*
T0*(
_output_shapes
:         ╕2
lstm_14/lstm_cell_14/add╠
+lstm_14/lstm_cell_14/BiasAdd/ReadVariableOpReadVariableOp4lstm_14_lstm_cell_14_biasadd_readvariableop_resource*
_output_shapes	
:╕*
dtype02-
+lstm_14/lstm_cell_14/BiasAdd/ReadVariableOp═
lstm_14/lstm_cell_14/BiasAddBiasAddlstm_14/lstm_cell_14/add:z:03lstm_14/lstm_cell_14/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╕2
lstm_14/lstm_cell_14/BiasAddО
$lstm_14/lstm_cell_14/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2&
$lstm_14/lstm_cell_14/split/split_dimЧ
lstm_14/lstm_cell_14/splitSplit-lstm_14/lstm_cell_14/split/split_dim:output:0%lstm_14/lstm_cell_14/BiasAdd:output:0*
T0*d
_output_shapesR
P:         ю:         ю:         ю:         ю*
	num_split2
lstm_14/lstm_cell_14/splitЯ
lstm_14/lstm_cell_14/SigmoidSigmoid#lstm_14/lstm_cell_14/split:output:0*
T0*(
_output_shapes
:         ю2
lstm_14/lstm_cell_14/Sigmoidг
lstm_14/lstm_cell_14/Sigmoid_1Sigmoid#lstm_14/lstm_cell_14/split:output:1*
T0*(
_output_shapes
:         ю2 
lstm_14/lstm_cell_14/Sigmoid_1м
lstm_14/lstm_cell_14/mulMul"lstm_14/lstm_cell_14/Sigmoid_1:y:0lstm_14/zeros_1:output:0*
T0*(
_output_shapes
:         ю2
lstm_14/lstm_cell_14/mulЦ
lstm_14/lstm_cell_14/ReluRelu#lstm_14/lstm_cell_14/split:output:2*
T0*(
_output_shapes
:         ю2
lstm_14/lstm_cell_14/Relu╜
lstm_14/lstm_cell_14/mul_1Mul lstm_14/lstm_cell_14/Sigmoid:y:0'lstm_14/lstm_cell_14/Relu:activations:0*
T0*(
_output_shapes
:         ю2
lstm_14/lstm_cell_14/mul_1▓
lstm_14/lstm_cell_14/add_1AddV2lstm_14/lstm_cell_14/mul:z:0lstm_14/lstm_cell_14/mul_1:z:0*
T0*(
_output_shapes
:         ю2
lstm_14/lstm_cell_14/add_1г
lstm_14/lstm_cell_14/Sigmoid_2Sigmoid#lstm_14/lstm_cell_14/split:output:3*
T0*(
_output_shapes
:         ю2 
lstm_14/lstm_cell_14/Sigmoid_2Х
lstm_14/lstm_cell_14/Relu_1Relulstm_14/lstm_cell_14/add_1:z:0*
T0*(
_output_shapes
:         ю2
lstm_14/lstm_cell_14/Relu_1┴
lstm_14/lstm_cell_14/mul_2Mul"lstm_14/lstm_cell_14/Sigmoid_2:y:0)lstm_14/lstm_cell_14/Relu_1:activations:0*
T0*(
_output_shapes
:         ю2
lstm_14/lstm_cell_14/mul_2Я
%lstm_14/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ю   2'
%lstm_14/TensorArrayV2_1/element_shape╪
lstm_14/TensorArrayV2_1TensorListReserve.lstm_14/TensorArrayV2_1/element_shape:output:0 lstm_14/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_14/TensorArrayV2_1^
lstm_14/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_14/timeП
 lstm_14/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         2"
 lstm_14/while/maximum_iterationsz
lstm_14/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_14/while/loop_counterЛ
lstm_14/whileWhile#lstm_14/while/loop_counter:output:0)lstm_14/while/maximum_iterations:output:0lstm_14/time:output:0 lstm_14/TensorArrayV2_1:handle:0lstm_14/zeros:output:0lstm_14/zeros_1:output:0 lstm_14/strided_slice_1:output:0?lstm_14/TensorArrayUnstack/TensorListFromTensor:output_handle:03lstm_14_lstm_cell_14_matmul_readvariableop_resource5lstm_14_lstm_cell_14_matmul_1_readvariableop_resource4lstm_14_lstm_cell_14_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :         ю:         ю: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *'
bodyR
lstm_14_while_body_26072076*'
condR
lstm_14_while_cond_26072075*M
output_shapes<
:: : : : :         ю:         ю: : : : : *
parallel_iterations 2
lstm_14/while┼
8lstm_14/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ю   2:
8lstm_14/TensorArrayV2Stack/TensorListStack/element_shapeЙ
*lstm_14/TensorArrayV2Stack/TensorListStackTensorListStacklstm_14/while:output:3Alstm_14/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:         ю*
element_dtype02,
*lstm_14/TensorArrayV2Stack/TensorListStackС
lstm_14/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         2
lstm_14/strided_slice_3/stackМ
lstm_14/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2!
lstm_14/strided_slice_3/stack_1М
lstm_14/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_14/strided_slice_3/stack_2╦
lstm_14/strided_slice_3StridedSlice3lstm_14/TensorArrayV2Stack/TensorListStack:tensor:0&lstm_14/strided_slice_3/stack:output:0(lstm_14/strided_slice_3/stack_1:output:0(lstm_14/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         ю*
shrink_axis_mask2
lstm_14/strided_slice_3Й
lstm_14/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_14/transpose_1/perm╞
lstm_14/transpose_1	Transpose3lstm_14/TensorArrayV2Stack/TensorListStack:tensor:0!lstm_14/transpose_1/perm:output:0*
T0*,
_output_shapes
:         ю2
lstm_14/transpose_1v
lstm_14/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_14/runtimeЖ
dropout_14/IdentityIdentitylstm_14/transpose_1:y:0*
T0*,
_output_shapes
:         ю2
dropout_14/Identityj
lstm_15/ShapeShapedropout_14/Identity:output:0*
T0*
_output_shapes
:2
lstm_15/ShapeД
lstm_15/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_15/strided_slice/stackИ
lstm_15/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_15/strided_slice/stack_1И
lstm_15/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_15/strided_slice/stack_2Т
lstm_15/strided_sliceStridedSlicelstm_15/Shape:output:0$lstm_15/strided_slice/stack:output:0&lstm_15/strided_slice/stack_1:output:0&lstm_15/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_15/strided_slicem
lstm_15/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :Ц2
lstm_15/zeros/mul/yМ
lstm_15/zeros/mulMullstm_15/strided_slice:output:0lstm_15/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_15/zeros/mulo
lstm_15/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
lstm_15/zeros/Less/yЗ
lstm_15/zeros/LessLesslstm_15/zeros/mul:z:0lstm_15/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_15/zeros/Lesss
lstm_15/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :Ц2
lstm_15/zeros/packed/1г
lstm_15/zeros/packedPacklstm_15/strided_slice:output:0lstm_15/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_15/zeros/packedo
lstm_15/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_15/zeros/ConstЦ
lstm_15/zerosFilllstm_15/zeros/packed:output:0lstm_15/zeros/Const:output:0*
T0*(
_output_shapes
:         Ц2
lstm_15/zerosq
lstm_15/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :Ц2
lstm_15/zeros_1/mul/yТ
lstm_15/zeros_1/mulMullstm_15/strided_slice:output:0lstm_15/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_15/zeros_1/muls
lstm_15/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
lstm_15/zeros_1/Less/yП
lstm_15/zeros_1/LessLesslstm_15/zeros_1/mul:z:0lstm_15/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_15/zeros_1/Lessw
lstm_15/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :Ц2
lstm_15/zeros_1/packed/1й
lstm_15/zeros_1/packedPacklstm_15/strided_slice:output:0!lstm_15/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_15/zeros_1/packeds
lstm_15/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_15/zeros_1/ConstЮ
lstm_15/zeros_1Filllstm_15/zeros_1/packed:output:0lstm_15/zeros_1/Const:output:0*
T0*(
_output_shapes
:         Ц2
lstm_15/zeros_1Е
lstm_15/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_15/transpose/permй
lstm_15/transpose	Transposedropout_14/Identity:output:0lstm_15/transpose/perm:output:0*
T0*,
_output_shapes
:         ю2
lstm_15/transposeg
lstm_15/Shape_1Shapelstm_15/transpose:y:0*
T0*
_output_shapes
:2
lstm_15/Shape_1И
lstm_15/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_15/strided_slice_1/stackМ
lstm_15/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_15/strided_slice_1/stack_1М
lstm_15/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_15/strided_slice_1/stack_2Ю
lstm_15/strided_slice_1StridedSlicelstm_15/Shape_1:output:0&lstm_15/strided_slice_1/stack:output:0(lstm_15/strided_slice_1/stack_1:output:0(lstm_15/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_15/strided_slice_1Х
#lstm_15/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         2%
#lstm_15/TensorArrayV2/element_shape╥
lstm_15/TensorArrayV2TensorListReserve,lstm_15/TensorArrayV2/element_shape:output:0 lstm_15/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_15/TensorArrayV2╧
=lstm_15/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ю   2?
=lstm_15/TensorArrayUnstack/TensorListFromTensor/element_shapeШ
/lstm_15/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_15/transpose:y:0Flstm_15/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type021
/lstm_15/TensorArrayUnstack/TensorListFromTensorИ
lstm_15/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_15/strided_slice_2/stackМ
lstm_15/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_15/strided_slice_2/stack_1М
lstm_15/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_15/strided_slice_2/stack_2н
lstm_15/strided_slice_2StridedSlicelstm_15/transpose:y:0&lstm_15/strided_slice_2/stack:output:0(lstm_15/strided_slice_2/stack_1:output:0(lstm_15/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         ю*
shrink_axis_mask2
lstm_15/strided_slice_2╬
*lstm_15/lstm_cell_15/MatMul/ReadVariableOpReadVariableOp3lstm_15_lstm_cell_15_matmul_readvariableop_resource* 
_output_shapes
:
ю╪*
dtype02,
*lstm_15/lstm_cell_15/MatMul/ReadVariableOp═
lstm_15/lstm_cell_15/MatMulMatMul lstm_15/strided_slice_2:output:02lstm_15/lstm_cell_15/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╪2
lstm_15/lstm_cell_15/MatMul╘
,lstm_15/lstm_cell_15/MatMul_1/ReadVariableOpReadVariableOp5lstm_15_lstm_cell_15_matmul_1_readvariableop_resource* 
_output_shapes
:
Ц╪*
dtype02.
,lstm_15/lstm_cell_15/MatMul_1/ReadVariableOp╔
lstm_15/lstm_cell_15/MatMul_1MatMullstm_15/zeros:output:04lstm_15/lstm_cell_15/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╪2
lstm_15/lstm_cell_15/MatMul_1└
lstm_15/lstm_cell_15/addAddV2%lstm_15/lstm_cell_15/MatMul:product:0'lstm_15/lstm_cell_15/MatMul_1:product:0*
T0*(
_output_shapes
:         ╪2
lstm_15/lstm_cell_15/add╠
+lstm_15/lstm_cell_15/BiasAdd/ReadVariableOpReadVariableOp4lstm_15_lstm_cell_15_biasadd_readvariableop_resource*
_output_shapes	
:╪*
dtype02-
+lstm_15/lstm_cell_15/BiasAdd/ReadVariableOp═
lstm_15/lstm_cell_15/BiasAddBiasAddlstm_15/lstm_cell_15/add:z:03lstm_15/lstm_cell_15/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╪2
lstm_15/lstm_cell_15/BiasAddО
$lstm_15/lstm_cell_15/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2&
$lstm_15/lstm_cell_15/split/split_dimЧ
lstm_15/lstm_cell_15/splitSplit-lstm_15/lstm_cell_15/split/split_dim:output:0%lstm_15/lstm_cell_15/BiasAdd:output:0*
T0*d
_output_shapesR
P:         Ц:         Ц:         Ц:         Ц*
	num_split2
lstm_15/lstm_cell_15/splitЯ
lstm_15/lstm_cell_15/SigmoidSigmoid#lstm_15/lstm_cell_15/split:output:0*
T0*(
_output_shapes
:         Ц2
lstm_15/lstm_cell_15/Sigmoidг
lstm_15/lstm_cell_15/Sigmoid_1Sigmoid#lstm_15/lstm_cell_15/split:output:1*
T0*(
_output_shapes
:         Ц2 
lstm_15/lstm_cell_15/Sigmoid_1м
lstm_15/lstm_cell_15/mulMul"lstm_15/lstm_cell_15/Sigmoid_1:y:0lstm_15/zeros_1:output:0*
T0*(
_output_shapes
:         Ц2
lstm_15/lstm_cell_15/mulЦ
lstm_15/lstm_cell_15/ReluRelu#lstm_15/lstm_cell_15/split:output:2*
T0*(
_output_shapes
:         Ц2
lstm_15/lstm_cell_15/Relu╜
lstm_15/lstm_cell_15/mul_1Mul lstm_15/lstm_cell_15/Sigmoid:y:0'lstm_15/lstm_cell_15/Relu:activations:0*
T0*(
_output_shapes
:         Ц2
lstm_15/lstm_cell_15/mul_1▓
lstm_15/lstm_cell_15/add_1AddV2lstm_15/lstm_cell_15/mul:z:0lstm_15/lstm_cell_15/mul_1:z:0*
T0*(
_output_shapes
:         Ц2
lstm_15/lstm_cell_15/add_1г
lstm_15/lstm_cell_15/Sigmoid_2Sigmoid#lstm_15/lstm_cell_15/split:output:3*
T0*(
_output_shapes
:         Ц2 
lstm_15/lstm_cell_15/Sigmoid_2Х
lstm_15/lstm_cell_15/Relu_1Relulstm_15/lstm_cell_15/add_1:z:0*
T0*(
_output_shapes
:         Ц2
lstm_15/lstm_cell_15/Relu_1┴
lstm_15/lstm_cell_15/mul_2Mul"lstm_15/lstm_cell_15/Sigmoid_2:y:0)lstm_15/lstm_cell_15/Relu_1:activations:0*
T0*(
_output_shapes
:         Ц2
lstm_15/lstm_cell_15/mul_2Я
%lstm_15/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    Ц   2'
%lstm_15/TensorArrayV2_1/element_shape╪
lstm_15/TensorArrayV2_1TensorListReserve.lstm_15/TensorArrayV2_1/element_shape:output:0 lstm_15/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_15/TensorArrayV2_1^
lstm_15/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_15/timeП
 lstm_15/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         2"
 lstm_15/while/maximum_iterationsz
lstm_15/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_15/while/loop_counterЛ
lstm_15/whileWhile#lstm_15/while/loop_counter:output:0)lstm_15/while/maximum_iterations:output:0lstm_15/time:output:0 lstm_15/TensorArrayV2_1:handle:0lstm_15/zeros:output:0lstm_15/zeros_1:output:0 lstm_15/strided_slice_1:output:0?lstm_15/TensorArrayUnstack/TensorListFromTensor:output_handle:03lstm_15_lstm_cell_15_matmul_readvariableop_resource5lstm_15_lstm_cell_15_matmul_1_readvariableop_resource4lstm_15_lstm_cell_15_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :         Ц:         Ц: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *'
bodyR
lstm_15_while_body_26072224*'
condR
lstm_15_while_cond_26072223*M
output_shapes<
:: : : : :         Ц:         Ц: : : : : *
parallel_iterations 2
lstm_15/while┼
8lstm_15/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    Ц   2:
8lstm_15/TensorArrayV2Stack/TensorListStack/element_shapeЙ
*lstm_15/TensorArrayV2Stack/TensorListStackTensorListStacklstm_15/while:output:3Alstm_15/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:         Ц*
element_dtype02,
*lstm_15/TensorArrayV2Stack/TensorListStackС
lstm_15/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         2
lstm_15/strided_slice_3/stackМ
lstm_15/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2!
lstm_15/strided_slice_3/stack_1М
lstm_15/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_15/strided_slice_3/stack_2╦
lstm_15/strided_slice_3StridedSlice3lstm_15/TensorArrayV2Stack/TensorListStack:tensor:0&lstm_15/strided_slice_3/stack:output:0(lstm_15/strided_slice_3/stack_1:output:0(lstm_15/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         Ц*
shrink_axis_mask2
lstm_15/strided_slice_3Й
lstm_15/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_15/transpose_1/perm╞
lstm_15/transpose_1	Transpose3lstm_15/TensorArrayV2Stack/TensorListStack:tensor:0!lstm_15/transpose_1/perm:output:0*
T0*,
_output_shapes
:         Ц2
lstm_15/transpose_1v
lstm_15/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_15/runtimeЖ
dropout_15/IdentityIdentitylstm_15/transpose_1:y:0*
T0*,
_output_shapes
:         Ц2
dropout_15/Identityп
 dense_7/Tensordot/ReadVariableOpReadVariableOp)dense_7_tensordot_readvariableop_resource*
_output_shapes
:	Ц*
dtype02"
 dense_7/Tensordot/ReadVariableOpz
dense_7/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_7/Tensordot/axesБ
dense_7/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_7/Tensordot/free~
dense_7/Tensordot/ShapeShapedropout_15/Identity:output:0*
T0*
_output_shapes
:2
dense_7/Tensordot/ShapeД
dense_7/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_7/Tensordot/GatherV2/axis∙
dense_7/Tensordot/GatherV2GatherV2 dense_7/Tensordot/Shape:output:0dense_7/Tensordot/free:output:0(dense_7/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_7/Tensordot/GatherV2И
!dense_7/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_7/Tensordot/GatherV2_1/axis 
dense_7/Tensordot/GatherV2_1GatherV2 dense_7/Tensordot/Shape:output:0dense_7/Tensordot/axes:output:0*dense_7/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_7/Tensordot/GatherV2_1|
dense_7/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_7/Tensordot/Constа
dense_7/Tensordot/ProdProd#dense_7/Tensordot/GatherV2:output:0 dense_7/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_7/Tensordot/ProdА
dense_7/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_7/Tensordot/Const_1и
dense_7/Tensordot/Prod_1Prod%dense_7/Tensordot/GatherV2_1:output:0"dense_7/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_7/Tensordot/Prod_1А
dense_7/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense_7/Tensordot/concat/axis╪
dense_7/Tensordot/concatConcatV2dense_7/Tensordot/free:output:0dense_7/Tensordot/axes:output:0&dense_7/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_7/Tensordot/concatм
dense_7/Tensordot/stackPackdense_7/Tensordot/Prod:output:0!dense_7/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_7/Tensordot/stack┐
dense_7/Tensordot/transpose	Transposedropout_15/Identity:output:0!dense_7/Tensordot/concat:output:0*
T0*,
_output_shapes
:         Ц2
dense_7/Tensordot/transpose┐
dense_7/Tensordot/ReshapeReshapedense_7/Tensordot/transpose:y:0 dense_7/Tensordot/stack:output:0*
T0*0
_output_shapes
:                  2
dense_7/Tensordot/Reshape╛
dense_7/Tensordot/MatMulMatMul"dense_7/Tensordot/Reshape:output:0(dense_7/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_7/Tensordot/MatMulА
dense_7/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
dense_7/Tensordot/Const_2Д
dense_7/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_7/Tensordot/concat_1/axisх
dense_7/Tensordot/concat_1ConcatV2#dense_7/Tensordot/GatherV2:output:0"dense_7/Tensordot/Const_2:output:0(dense_7/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_7/Tensordot/concat_1░
dense_7/TensordotReshape"dense_7/Tensordot/MatMul:product:0#dense_7/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:         2
dense_7/Tensordotд
dense_7/BiasAdd/ReadVariableOpReadVariableOp'dense_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_7/BiasAdd/ReadVariableOpз
dense_7/BiasAddBiasAdddense_7/Tensordot:output:0&dense_7/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:         2
dense_7/BiasAdd}
dense_7/SoftmaxSoftmaxdense_7/BiasAdd:output:0*
T0*+
_output_shapes
:         2
dense_7/Softmaxx
IdentityIdentitydense_7/Softmax:softmax:0^NoOp*
T0*+
_output_shapes
:         2

Identity╞
NoOpNoOp^dense_7/BiasAdd/ReadVariableOp!^dense_7/Tensordot/ReadVariableOp,^lstm_14/lstm_cell_14/BiasAdd/ReadVariableOp+^lstm_14/lstm_cell_14/MatMul/ReadVariableOp-^lstm_14/lstm_cell_14/MatMul_1/ReadVariableOp^lstm_14/while,^lstm_15/lstm_cell_15/BiasAdd/ReadVariableOp+^lstm_15/lstm_cell_15/MatMul/ReadVariableOp-^lstm_15/lstm_cell_15/MatMul_1/ReadVariableOp^lstm_15/while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':         ]: : : : : : : : 2@
dense_7/BiasAdd/ReadVariableOpdense_7/BiasAdd/ReadVariableOp2D
 dense_7/Tensordot/ReadVariableOp dense_7/Tensordot/ReadVariableOp2Z
+lstm_14/lstm_cell_14/BiasAdd/ReadVariableOp+lstm_14/lstm_cell_14/BiasAdd/ReadVariableOp2X
*lstm_14/lstm_cell_14/MatMul/ReadVariableOp*lstm_14/lstm_cell_14/MatMul/ReadVariableOp2\
,lstm_14/lstm_cell_14/MatMul_1/ReadVariableOp,lstm_14/lstm_cell_14/MatMul_1/ReadVariableOp2
lstm_14/whilelstm_14/while2Z
+lstm_15/lstm_cell_15/BiasAdd/ReadVariableOp+lstm_15/lstm_cell_15/BiasAdd/ReadVariableOp2X
*lstm_15/lstm_cell_15/MatMul/ReadVariableOp*lstm_15/lstm_cell_15/MatMul/ReadVariableOp2\
,lstm_15/lstm_cell_15/MatMul_1/ReadVariableOp,lstm_15/lstm_cell_15/MatMul_1/ReadVariableOp2
lstm_15/whilelstm_15/while:S O
+
_output_shapes
:         ]
 
_user_specified_nameinputs
╩7
▌
$__inference__traced_restore_26074410
file_prefix2
assignvariableop_dense_7_kernel:	Ц-
assignvariableop_1_dense_7_bias:A
.assignvariableop_2_lstm_14_lstm_cell_14_kernel:	]╕L
8assignvariableop_3_lstm_14_lstm_cell_14_recurrent_kernel:
ю╕;
,assignvariableop_4_lstm_14_lstm_cell_14_bias:	╕B
.assignvariableop_5_lstm_15_lstm_cell_15_kernel:
ю╪L
8assignvariableop_6_lstm_15_lstm_cell_15_recurrent_kernel:
Ц╪;
,assignvariableop_7_lstm_15_lstm_cell_15_bias:	╪"
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
AssignVariableOpAssignVariableOpassignvariableop_dense_7_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1д
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_7_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2│
AssignVariableOp_2AssignVariableOp.assignvariableop_2_lstm_14_lstm_cell_14_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3╜
AssignVariableOp_3AssignVariableOp8assignvariableop_3_lstm_14_lstm_cell_14_recurrent_kernelIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4▒
AssignVariableOp_4AssignVariableOp,assignvariableop_4_lstm_14_lstm_cell_14_biasIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5│
AssignVariableOp_5AssignVariableOp.assignvariableop_5_lstm_15_lstm_cell_15_kernelIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6╜
AssignVariableOp_6AssignVariableOp8assignvariableop_6_lstm_15_lstm_cell_15_recurrent_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7▒
AssignVariableOp_7AssignVariableOp,assignvariableop_7_lstm_15_lstm_cell_15_biasIdentity_7:output:0"/device:CPU:0*
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
у
═
while_cond_26073913
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_26073913___redundant_placeholder06
2while_while_cond_26073913___redundant_placeholder16
2while_while_cond_26073913___redundant_placeholder26
2while_while_cond_26073913___redundant_placeholder3
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
B: : : : :         Ц:         Ц: ::::: 
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
:         Ц:.*
(
_output_shapes
:         Ц:

_output_shapes
: :

_output_shapes
:
Ї%
▐
!__inference__traced_save_26074364
file_prefix-
)savev2_dense_7_kernel_read_readvariableop+
'savev2_dense_7_bias_read_readvariableop:
6savev2_lstm_14_lstm_cell_14_kernel_read_readvariableopD
@savev2_lstm_14_lstm_cell_14_recurrent_kernel_read_readvariableop8
4savev2_lstm_14_lstm_cell_14_bias_read_readvariableop:
6savev2_lstm_15_lstm_cell_15_kernel_read_readvariableopD
@savev2_lstm_15_lstm_cell_15_recurrent_kernel_read_readvariableop8
4savev2_lstm_15_lstm_cell_15_bias_read_readvariableop$
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
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0)savev2_dense_7_kernel_read_readvariableop'savev2_dense_7_bias_read_readvariableop6savev2_lstm_14_lstm_cell_14_kernel_read_readvariableop@savev2_lstm_14_lstm_cell_14_recurrent_kernel_read_readvariableop4savev2_lstm_14_lstm_cell_14_bias_read_readvariableop6savev2_lstm_15_lstm_cell_15_kernel_read_readvariableop@savev2_lstm_15_lstm_cell_15_recurrent_kernel_read_readvariableop4savev2_lstm_15_lstm_cell_15_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableopsavev2_const"/device:CPU:0*
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

identity_1Identity_1:output:0*m
_input_shapes\
Z: :	Ц::	]╕:
ю╕:╕:
ю╪:
Ц╪:╪: : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	Ц: 

_output_shapes
::%!

_output_shapes
:	]╕:&"
 
_output_shapes
:
ю╕:!

_output_shapes	
:╕:&"
 
_output_shapes
:
ю╪:&"
 
_output_shapes
:
Ц╪:!

_output_shapes	
:╪:	
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
█
ё
(sequential_7_lstm_14_while_cond_26069531F
Bsequential_7_lstm_14_while_sequential_7_lstm_14_while_loop_counterL
Hsequential_7_lstm_14_while_sequential_7_lstm_14_while_maximum_iterations*
&sequential_7_lstm_14_while_placeholder,
(sequential_7_lstm_14_while_placeholder_1,
(sequential_7_lstm_14_while_placeholder_2,
(sequential_7_lstm_14_while_placeholder_3H
Dsequential_7_lstm_14_while_less_sequential_7_lstm_14_strided_slice_1`
\sequential_7_lstm_14_while_sequential_7_lstm_14_while_cond_26069531___redundant_placeholder0`
\sequential_7_lstm_14_while_sequential_7_lstm_14_while_cond_26069531___redundant_placeholder1`
\sequential_7_lstm_14_while_sequential_7_lstm_14_while_cond_26069531___redundant_placeholder2`
\sequential_7_lstm_14_while_sequential_7_lstm_14_while_cond_26069531___redundant_placeholder3'
#sequential_7_lstm_14_while_identity
┘
sequential_7/lstm_14/while/LessLess&sequential_7_lstm_14_while_placeholderDsequential_7_lstm_14_while_less_sequential_7_lstm_14_strided_slice_1*
T0*
_output_shapes
: 2!
sequential_7/lstm_14/while/LessЬ
#sequential_7/lstm_14/while/IdentityIdentity#sequential_7/lstm_14/while/Less:z:0*
T0
*
_output_shapes
: 2%
#sequential_7/lstm_14/while/Identity"S
#sequential_7_lstm_14_while_identity,sequential_7/lstm_14/while/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :         ю:         ю: ::::: 
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
:         ю:.*
(
_output_shapes
:         ю:

_output_shapes
: :

_output_shapes
:
у
═
while_cond_26073460
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_26073460___redundant_placeholder06
2while_while_cond_26073460___redundant_placeholder16
2while_while_cond_26073460___redundant_placeholder26
2while_while_cond_26073460___redundant_placeholder3
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
B: : : : :         Ц:         Ц: ::::: 
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
:         Ц:.*
(
_output_shapes
:         Ц:

_output_shapes
: :

_output_shapes
:
хJ
╘

lstm_14_while_body_26072076,
(lstm_14_while_lstm_14_while_loop_counter2
.lstm_14_while_lstm_14_while_maximum_iterations
lstm_14_while_placeholder
lstm_14_while_placeholder_1
lstm_14_while_placeholder_2
lstm_14_while_placeholder_3+
'lstm_14_while_lstm_14_strided_slice_1_0g
clstm_14_while_tensorarrayv2read_tensorlistgetitem_lstm_14_tensorarrayunstack_tensorlistfromtensor_0N
;lstm_14_while_lstm_cell_14_matmul_readvariableop_resource_0:	]╕Q
=lstm_14_while_lstm_cell_14_matmul_1_readvariableop_resource_0:
ю╕K
<lstm_14_while_lstm_cell_14_biasadd_readvariableop_resource_0:	╕
lstm_14_while_identity
lstm_14_while_identity_1
lstm_14_while_identity_2
lstm_14_while_identity_3
lstm_14_while_identity_4
lstm_14_while_identity_5)
%lstm_14_while_lstm_14_strided_slice_1e
alstm_14_while_tensorarrayv2read_tensorlistgetitem_lstm_14_tensorarrayunstack_tensorlistfromtensorL
9lstm_14_while_lstm_cell_14_matmul_readvariableop_resource:	]╕O
;lstm_14_while_lstm_cell_14_matmul_1_readvariableop_resource:
ю╕I
:lstm_14_while_lstm_cell_14_biasadd_readvariableop_resource:	╕Ив1lstm_14/while/lstm_cell_14/BiasAdd/ReadVariableOpв0lstm_14/while/lstm_cell_14/MatMul/ReadVariableOpв2lstm_14/while/lstm_cell_14/MatMul_1/ReadVariableOp╙
?lstm_14/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ]   2A
?lstm_14/while/TensorArrayV2Read/TensorListGetItem/element_shapeГ
1lstm_14/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemclstm_14_while_tensorarrayv2read_tensorlistgetitem_lstm_14_tensorarrayunstack_tensorlistfromtensor_0lstm_14_while_placeholderHlstm_14/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         ]*
element_dtype023
1lstm_14/while/TensorArrayV2Read/TensorListGetItemс
0lstm_14/while/lstm_cell_14/MatMul/ReadVariableOpReadVariableOp;lstm_14_while_lstm_cell_14_matmul_readvariableop_resource_0*
_output_shapes
:	]╕*
dtype022
0lstm_14/while/lstm_cell_14/MatMul/ReadVariableOpў
!lstm_14/while/lstm_cell_14/MatMulMatMul8lstm_14/while/TensorArrayV2Read/TensorListGetItem:item:08lstm_14/while/lstm_cell_14/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╕2#
!lstm_14/while/lstm_cell_14/MatMulш
2lstm_14/while/lstm_cell_14/MatMul_1/ReadVariableOpReadVariableOp=lstm_14_while_lstm_cell_14_matmul_1_readvariableop_resource_0* 
_output_shapes
:
ю╕*
dtype024
2lstm_14/while/lstm_cell_14/MatMul_1/ReadVariableOpр
#lstm_14/while/lstm_cell_14/MatMul_1MatMullstm_14_while_placeholder_2:lstm_14/while/lstm_cell_14/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╕2%
#lstm_14/while/lstm_cell_14/MatMul_1╪
lstm_14/while/lstm_cell_14/addAddV2+lstm_14/while/lstm_cell_14/MatMul:product:0-lstm_14/while/lstm_cell_14/MatMul_1:product:0*
T0*(
_output_shapes
:         ╕2 
lstm_14/while/lstm_cell_14/addр
1lstm_14/while/lstm_cell_14/BiasAdd/ReadVariableOpReadVariableOp<lstm_14_while_lstm_cell_14_biasadd_readvariableop_resource_0*
_output_shapes	
:╕*
dtype023
1lstm_14/while/lstm_cell_14/BiasAdd/ReadVariableOpх
"lstm_14/while/lstm_cell_14/BiasAddBiasAdd"lstm_14/while/lstm_cell_14/add:z:09lstm_14/while/lstm_cell_14/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╕2$
"lstm_14/while/lstm_cell_14/BiasAddЪ
*lstm_14/while/lstm_cell_14/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2,
*lstm_14/while/lstm_cell_14/split/split_dimп
 lstm_14/while/lstm_cell_14/splitSplit3lstm_14/while/lstm_cell_14/split/split_dim:output:0+lstm_14/while/lstm_cell_14/BiasAdd:output:0*
T0*d
_output_shapesR
P:         ю:         ю:         ю:         ю*
	num_split2"
 lstm_14/while/lstm_cell_14/split▒
"lstm_14/while/lstm_cell_14/SigmoidSigmoid)lstm_14/while/lstm_cell_14/split:output:0*
T0*(
_output_shapes
:         ю2$
"lstm_14/while/lstm_cell_14/Sigmoid╡
$lstm_14/while/lstm_cell_14/Sigmoid_1Sigmoid)lstm_14/while/lstm_cell_14/split:output:1*
T0*(
_output_shapes
:         ю2&
$lstm_14/while/lstm_cell_14/Sigmoid_1┴
lstm_14/while/lstm_cell_14/mulMul(lstm_14/while/lstm_cell_14/Sigmoid_1:y:0lstm_14_while_placeholder_3*
T0*(
_output_shapes
:         ю2 
lstm_14/while/lstm_cell_14/mulи
lstm_14/while/lstm_cell_14/ReluRelu)lstm_14/while/lstm_cell_14/split:output:2*
T0*(
_output_shapes
:         ю2!
lstm_14/while/lstm_cell_14/Relu╒
 lstm_14/while/lstm_cell_14/mul_1Mul&lstm_14/while/lstm_cell_14/Sigmoid:y:0-lstm_14/while/lstm_cell_14/Relu:activations:0*
T0*(
_output_shapes
:         ю2"
 lstm_14/while/lstm_cell_14/mul_1╩
 lstm_14/while/lstm_cell_14/add_1AddV2"lstm_14/while/lstm_cell_14/mul:z:0$lstm_14/while/lstm_cell_14/mul_1:z:0*
T0*(
_output_shapes
:         ю2"
 lstm_14/while/lstm_cell_14/add_1╡
$lstm_14/while/lstm_cell_14/Sigmoid_2Sigmoid)lstm_14/while/lstm_cell_14/split:output:3*
T0*(
_output_shapes
:         ю2&
$lstm_14/while/lstm_cell_14/Sigmoid_2з
!lstm_14/while/lstm_cell_14/Relu_1Relu$lstm_14/while/lstm_cell_14/add_1:z:0*
T0*(
_output_shapes
:         ю2#
!lstm_14/while/lstm_cell_14/Relu_1┘
 lstm_14/while/lstm_cell_14/mul_2Mul(lstm_14/while/lstm_cell_14/Sigmoid_2:y:0/lstm_14/while/lstm_cell_14/Relu_1:activations:0*
T0*(
_output_shapes
:         ю2"
 lstm_14/while/lstm_cell_14/mul_2И
2lstm_14/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_14_while_placeholder_1lstm_14_while_placeholder$lstm_14/while/lstm_cell_14/mul_2:z:0*
_output_shapes
: *
element_dtype024
2lstm_14/while/TensorArrayV2Write/TensorListSetIteml
lstm_14/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_14/while/add/yЙ
lstm_14/while/addAddV2lstm_14_while_placeholderlstm_14/while/add/y:output:0*
T0*
_output_shapes
: 2
lstm_14/while/addp
lstm_14/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_14/while/add_1/yЮ
lstm_14/while/add_1AddV2(lstm_14_while_lstm_14_while_loop_counterlstm_14/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lstm_14/while/add_1Л
lstm_14/while/IdentityIdentitylstm_14/while/add_1:z:0^lstm_14/while/NoOp*
T0*
_output_shapes
: 2
lstm_14/while/Identityж
lstm_14/while/Identity_1Identity.lstm_14_while_lstm_14_while_maximum_iterations^lstm_14/while/NoOp*
T0*
_output_shapes
: 2
lstm_14/while/Identity_1Н
lstm_14/while/Identity_2Identitylstm_14/while/add:z:0^lstm_14/while/NoOp*
T0*
_output_shapes
: 2
lstm_14/while/Identity_2║
lstm_14/while/Identity_3IdentityBlstm_14/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_14/while/NoOp*
T0*
_output_shapes
: 2
lstm_14/while/Identity_3о
lstm_14/while/Identity_4Identity$lstm_14/while/lstm_cell_14/mul_2:z:0^lstm_14/while/NoOp*
T0*(
_output_shapes
:         ю2
lstm_14/while/Identity_4о
lstm_14/while/Identity_5Identity$lstm_14/while/lstm_cell_14/add_1:z:0^lstm_14/while/NoOp*
T0*(
_output_shapes
:         ю2
lstm_14/while/Identity_5Ж
lstm_14/while/NoOpNoOp2^lstm_14/while/lstm_cell_14/BiasAdd/ReadVariableOp1^lstm_14/while/lstm_cell_14/MatMul/ReadVariableOp3^lstm_14/while/lstm_cell_14/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
lstm_14/while/NoOp"9
lstm_14_while_identitylstm_14/while/Identity:output:0"=
lstm_14_while_identity_1!lstm_14/while/Identity_1:output:0"=
lstm_14_while_identity_2!lstm_14/while/Identity_2:output:0"=
lstm_14_while_identity_3!lstm_14/while/Identity_3:output:0"=
lstm_14_while_identity_4!lstm_14/while/Identity_4:output:0"=
lstm_14_while_identity_5!lstm_14/while/Identity_5:output:0"P
%lstm_14_while_lstm_14_strided_slice_1'lstm_14_while_lstm_14_strided_slice_1_0"z
:lstm_14_while_lstm_cell_14_biasadd_readvariableop_resource<lstm_14_while_lstm_cell_14_biasadd_readvariableop_resource_0"|
;lstm_14_while_lstm_cell_14_matmul_1_readvariableop_resource=lstm_14_while_lstm_cell_14_matmul_1_readvariableop_resource_0"x
9lstm_14_while_lstm_cell_14_matmul_readvariableop_resource;lstm_14_while_lstm_cell_14_matmul_readvariableop_resource_0"╚
alstm_14_while_tensorarrayv2read_tensorlistgetitem_lstm_14_tensorarrayunstack_tensorlistfromtensorclstm_14_while_tensorarrayv2read_tensorlistgetitem_lstm_14_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :         ю:         ю: : : : : 2f
1lstm_14/while/lstm_cell_14/BiasAdd/ReadVariableOp1lstm_14/while/lstm_cell_14/BiasAdd/ReadVariableOp2d
0lstm_14/while/lstm_cell_14/MatMul/ReadVariableOp0lstm_14/while/lstm_cell_14/MatMul/ReadVariableOp2h
2lstm_14/while/lstm_cell_14/MatMul_1/ReadVariableOp2lstm_14/while/lstm_cell_14/MatMul_1/ReadVariableOp: 
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
:         ю:.*
(
_output_shapes
:         ю:

_output_shapes
: :

_output_shapes
: 
ФМ
З
J__inference_sequential_7_layer_call_and_return_conditional_losses_26072677

inputsF
3lstm_14_lstm_cell_14_matmul_readvariableop_resource:	]╕I
5lstm_14_lstm_cell_14_matmul_1_readvariableop_resource:
ю╕C
4lstm_14_lstm_cell_14_biasadd_readvariableop_resource:	╕G
3lstm_15_lstm_cell_15_matmul_readvariableop_resource:
ю╪I
5lstm_15_lstm_cell_15_matmul_1_readvariableop_resource:
Ц╪C
4lstm_15_lstm_cell_15_biasadd_readvariableop_resource:	╪<
)dense_7_tensordot_readvariableop_resource:	Ц5
'dense_7_biasadd_readvariableop_resource:
identityИвdense_7/BiasAdd/ReadVariableOpв dense_7/Tensordot/ReadVariableOpв+lstm_14/lstm_cell_14/BiasAdd/ReadVariableOpв*lstm_14/lstm_cell_14/MatMul/ReadVariableOpв,lstm_14/lstm_cell_14/MatMul_1/ReadVariableOpвlstm_14/whileв+lstm_15/lstm_cell_15/BiasAdd/ReadVariableOpв*lstm_15/lstm_cell_15/MatMul/ReadVariableOpв,lstm_15/lstm_cell_15/MatMul_1/ReadVariableOpвlstm_15/whileT
lstm_14/ShapeShapeinputs*
T0*
_output_shapes
:2
lstm_14/ShapeД
lstm_14/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_14/strided_slice/stackИ
lstm_14/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_14/strided_slice/stack_1И
lstm_14/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_14/strided_slice/stack_2Т
lstm_14/strided_sliceStridedSlicelstm_14/Shape:output:0$lstm_14/strided_slice/stack:output:0&lstm_14/strided_slice/stack_1:output:0&lstm_14/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_14/strided_slicem
lstm_14/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :ю2
lstm_14/zeros/mul/yМ
lstm_14/zeros/mulMullstm_14/strided_slice:output:0lstm_14/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_14/zeros/mulo
lstm_14/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
lstm_14/zeros/Less/yЗ
lstm_14/zeros/LessLesslstm_14/zeros/mul:z:0lstm_14/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_14/zeros/Lesss
lstm_14/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :ю2
lstm_14/zeros/packed/1г
lstm_14/zeros/packedPacklstm_14/strided_slice:output:0lstm_14/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_14/zeros/packedo
lstm_14/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_14/zeros/ConstЦ
lstm_14/zerosFilllstm_14/zeros/packed:output:0lstm_14/zeros/Const:output:0*
T0*(
_output_shapes
:         ю2
lstm_14/zerosq
lstm_14/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :ю2
lstm_14/zeros_1/mul/yТ
lstm_14/zeros_1/mulMullstm_14/strided_slice:output:0lstm_14/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_14/zeros_1/muls
lstm_14/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
lstm_14/zeros_1/Less/yП
lstm_14/zeros_1/LessLesslstm_14/zeros_1/mul:z:0lstm_14/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_14/zeros_1/Lessw
lstm_14/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :ю2
lstm_14/zeros_1/packed/1й
lstm_14/zeros_1/packedPacklstm_14/strided_slice:output:0!lstm_14/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_14/zeros_1/packeds
lstm_14/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_14/zeros_1/ConstЮ
lstm_14/zeros_1Filllstm_14/zeros_1/packed:output:0lstm_14/zeros_1/Const:output:0*
T0*(
_output_shapes
:         ю2
lstm_14/zeros_1Е
lstm_14/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_14/transpose/permТ
lstm_14/transpose	Transposeinputslstm_14/transpose/perm:output:0*
T0*+
_output_shapes
:         ]2
lstm_14/transposeg
lstm_14/Shape_1Shapelstm_14/transpose:y:0*
T0*
_output_shapes
:2
lstm_14/Shape_1И
lstm_14/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_14/strided_slice_1/stackМ
lstm_14/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_14/strided_slice_1/stack_1М
lstm_14/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_14/strided_slice_1/stack_2Ю
lstm_14/strided_slice_1StridedSlicelstm_14/Shape_1:output:0&lstm_14/strided_slice_1/stack:output:0(lstm_14/strided_slice_1/stack_1:output:0(lstm_14/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_14/strided_slice_1Х
#lstm_14/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         2%
#lstm_14/TensorArrayV2/element_shape╥
lstm_14/TensorArrayV2TensorListReserve,lstm_14/TensorArrayV2/element_shape:output:0 lstm_14/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_14/TensorArrayV2╧
=lstm_14/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ]   2?
=lstm_14/TensorArrayUnstack/TensorListFromTensor/element_shapeШ
/lstm_14/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_14/transpose:y:0Flstm_14/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type021
/lstm_14/TensorArrayUnstack/TensorListFromTensorИ
lstm_14/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_14/strided_slice_2/stackМ
lstm_14/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_14/strided_slice_2/stack_1М
lstm_14/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_14/strided_slice_2/stack_2м
lstm_14/strided_slice_2StridedSlicelstm_14/transpose:y:0&lstm_14/strided_slice_2/stack:output:0(lstm_14/strided_slice_2/stack_1:output:0(lstm_14/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         ]*
shrink_axis_mask2
lstm_14/strided_slice_2═
*lstm_14/lstm_cell_14/MatMul/ReadVariableOpReadVariableOp3lstm_14_lstm_cell_14_matmul_readvariableop_resource*
_output_shapes
:	]╕*
dtype02,
*lstm_14/lstm_cell_14/MatMul/ReadVariableOp═
lstm_14/lstm_cell_14/MatMulMatMul lstm_14/strided_slice_2:output:02lstm_14/lstm_cell_14/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╕2
lstm_14/lstm_cell_14/MatMul╘
,lstm_14/lstm_cell_14/MatMul_1/ReadVariableOpReadVariableOp5lstm_14_lstm_cell_14_matmul_1_readvariableop_resource* 
_output_shapes
:
ю╕*
dtype02.
,lstm_14/lstm_cell_14/MatMul_1/ReadVariableOp╔
lstm_14/lstm_cell_14/MatMul_1MatMullstm_14/zeros:output:04lstm_14/lstm_cell_14/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╕2
lstm_14/lstm_cell_14/MatMul_1└
lstm_14/lstm_cell_14/addAddV2%lstm_14/lstm_cell_14/MatMul:product:0'lstm_14/lstm_cell_14/MatMul_1:product:0*
T0*(
_output_shapes
:         ╕2
lstm_14/lstm_cell_14/add╠
+lstm_14/lstm_cell_14/BiasAdd/ReadVariableOpReadVariableOp4lstm_14_lstm_cell_14_biasadd_readvariableop_resource*
_output_shapes	
:╕*
dtype02-
+lstm_14/lstm_cell_14/BiasAdd/ReadVariableOp═
lstm_14/lstm_cell_14/BiasAddBiasAddlstm_14/lstm_cell_14/add:z:03lstm_14/lstm_cell_14/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╕2
lstm_14/lstm_cell_14/BiasAddО
$lstm_14/lstm_cell_14/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2&
$lstm_14/lstm_cell_14/split/split_dimЧ
lstm_14/lstm_cell_14/splitSplit-lstm_14/lstm_cell_14/split/split_dim:output:0%lstm_14/lstm_cell_14/BiasAdd:output:0*
T0*d
_output_shapesR
P:         ю:         ю:         ю:         ю*
	num_split2
lstm_14/lstm_cell_14/splitЯ
lstm_14/lstm_cell_14/SigmoidSigmoid#lstm_14/lstm_cell_14/split:output:0*
T0*(
_output_shapes
:         ю2
lstm_14/lstm_cell_14/Sigmoidг
lstm_14/lstm_cell_14/Sigmoid_1Sigmoid#lstm_14/lstm_cell_14/split:output:1*
T0*(
_output_shapes
:         ю2 
lstm_14/lstm_cell_14/Sigmoid_1м
lstm_14/lstm_cell_14/mulMul"lstm_14/lstm_cell_14/Sigmoid_1:y:0lstm_14/zeros_1:output:0*
T0*(
_output_shapes
:         ю2
lstm_14/lstm_cell_14/mulЦ
lstm_14/lstm_cell_14/ReluRelu#lstm_14/lstm_cell_14/split:output:2*
T0*(
_output_shapes
:         ю2
lstm_14/lstm_cell_14/Relu╜
lstm_14/lstm_cell_14/mul_1Mul lstm_14/lstm_cell_14/Sigmoid:y:0'lstm_14/lstm_cell_14/Relu:activations:0*
T0*(
_output_shapes
:         ю2
lstm_14/lstm_cell_14/mul_1▓
lstm_14/lstm_cell_14/add_1AddV2lstm_14/lstm_cell_14/mul:z:0lstm_14/lstm_cell_14/mul_1:z:0*
T0*(
_output_shapes
:         ю2
lstm_14/lstm_cell_14/add_1г
lstm_14/lstm_cell_14/Sigmoid_2Sigmoid#lstm_14/lstm_cell_14/split:output:3*
T0*(
_output_shapes
:         ю2 
lstm_14/lstm_cell_14/Sigmoid_2Х
lstm_14/lstm_cell_14/Relu_1Relulstm_14/lstm_cell_14/add_1:z:0*
T0*(
_output_shapes
:         ю2
lstm_14/lstm_cell_14/Relu_1┴
lstm_14/lstm_cell_14/mul_2Mul"lstm_14/lstm_cell_14/Sigmoid_2:y:0)lstm_14/lstm_cell_14/Relu_1:activations:0*
T0*(
_output_shapes
:         ю2
lstm_14/lstm_cell_14/mul_2Я
%lstm_14/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ю   2'
%lstm_14/TensorArrayV2_1/element_shape╪
lstm_14/TensorArrayV2_1TensorListReserve.lstm_14/TensorArrayV2_1/element_shape:output:0 lstm_14/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_14/TensorArrayV2_1^
lstm_14/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_14/timeП
 lstm_14/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         2"
 lstm_14/while/maximum_iterationsz
lstm_14/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_14/while/loop_counterЛ
lstm_14/whileWhile#lstm_14/while/loop_counter:output:0)lstm_14/while/maximum_iterations:output:0lstm_14/time:output:0 lstm_14/TensorArrayV2_1:handle:0lstm_14/zeros:output:0lstm_14/zeros_1:output:0 lstm_14/strided_slice_1:output:0?lstm_14/TensorArrayUnstack/TensorListFromTensor:output_handle:03lstm_14_lstm_cell_14_matmul_readvariableop_resource5lstm_14_lstm_cell_14_matmul_1_readvariableop_resource4lstm_14_lstm_cell_14_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :         ю:         ю: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *'
bodyR
lstm_14_while_body_26072403*'
condR
lstm_14_while_cond_26072402*M
output_shapes<
:: : : : :         ю:         ю: : : : : *
parallel_iterations 2
lstm_14/while┼
8lstm_14/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ю   2:
8lstm_14/TensorArrayV2Stack/TensorListStack/element_shapeЙ
*lstm_14/TensorArrayV2Stack/TensorListStackTensorListStacklstm_14/while:output:3Alstm_14/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:         ю*
element_dtype02,
*lstm_14/TensorArrayV2Stack/TensorListStackС
lstm_14/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         2
lstm_14/strided_slice_3/stackМ
lstm_14/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2!
lstm_14/strided_slice_3/stack_1М
lstm_14/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_14/strided_slice_3/stack_2╦
lstm_14/strided_slice_3StridedSlice3lstm_14/TensorArrayV2Stack/TensorListStack:tensor:0&lstm_14/strided_slice_3/stack:output:0(lstm_14/strided_slice_3/stack_1:output:0(lstm_14/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         ю*
shrink_axis_mask2
lstm_14/strided_slice_3Й
lstm_14/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_14/transpose_1/perm╞
lstm_14/transpose_1	Transpose3lstm_14/TensorArrayV2Stack/TensorListStack:tensor:0!lstm_14/transpose_1/perm:output:0*
T0*,
_output_shapes
:         ю2
lstm_14/transpose_1v
lstm_14/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_14/runtimey
dropout_14/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *лкк?2
dropout_14/dropout/Constк
dropout_14/dropout/MulMullstm_14/transpose_1:y:0!dropout_14/dropout/Const:output:0*
T0*,
_output_shapes
:         ю2
dropout_14/dropout/Mul{
dropout_14/dropout/ShapeShapelstm_14/transpose_1:y:0*
T0*
_output_shapes
:2
dropout_14/dropout/Shape┌
/dropout_14/dropout/random_uniform/RandomUniformRandomUniform!dropout_14/dropout/Shape:output:0*
T0*,
_output_shapes
:         ю*
dtype021
/dropout_14/dropout/random_uniform/RandomUniformЛ
!dropout_14/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А>2#
!dropout_14/dropout/GreaterEqual/yя
dropout_14/dropout/GreaterEqualGreaterEqual8dropout_14/dropout/random_uniform/RandomUniform:output:0*dropout_14/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:         ю2!
dropout_14/dropout/GreaterEqualе
dropout_14/dropout/CastCast#dropout_14/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:         ю2
dropout_14/dropout/Castл
dropout_14/dropout/Mul_1Muldropout_14/dropout/Mul:z:0dropout_14/dropout/Cast:y:0*
T0*,
_output_shapes
:         ю2
dropout_14/dropout/Mul_1j
lstm_15/ShapeShapedropout_14/dropout/Mul_1:z:0*
T0*
_output_shapes
:2
lstm_15/ShapeД
lstm_15/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_15/strided_slice/stackИ
lstm_15/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_15/strided_slice/stack_1И
lstm_15/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_15/strided_slice/stack_2Т
lstm_15/strided_sliceStridedSlicelstm_15/Shape:output:0$lstm_15/strided_slice/stack:output:0&lstm_15/strided_slice/stack_1:output:0&lstm_15/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_15/strided_slicem
lstm_15/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :Ц2
lstm_15/zeros/mul/yМ
lstm_15/zeros/mulMullstm_15/strided_slice:output:0lstm_15/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_15/zeros/mulo
lstm_15/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
lstm_15/zeros/Less/yЗ
lstm_15/zeros/LessLesslstm_15/zeros/mul:z:0lstm_15/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_15/zeros/Lesss
lstm_15/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :Ц2
lstm_15/zeros/packed/1г
lstm_15/zeros/packedPacklstm_15/strided_slice:output:0lstm_15/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_15/zeros/packedo
lstm_15/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_15/zeros/ConstЦ
lstm_15/zerosFilllstm_15/zeros/packed:output:0lstm_15/zeros/Const:output:0*
T0*(
_output_shapes
:         Ц2
lstm_15/zerosq
lstm_15/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :Ц2
lstm_15/zeros_1/mul/yТ
lstm_15/zeros_1/mulMullstm_15/strided_slice:output:0lstm_15/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_15/zeros_1/muls
lstm_15/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
lstm_15/zeros_1/Less/yП
lstm_15/zeros_1/LessLesslstm_15/zeros_1/mul:z:0lstm_15/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_15/zeros_1/Lessw
lstm_15/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :Ц2
lstm_15/zeros_1/packed/1й
lstm_15/zeros_1/packedPacklstm_15/strided_slice:output:0!lstm_15/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_15/zeros_1/packeds
lstm_15/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_15/zeros_1/ConstЮ
lstm_15/zeros_1Filllstm_15/zeros_1/packed:output:0lstm_15/zeros_1/Const:output:0*
T0*(
_output_shapes
:         Ц2
lstm_15/zeros_1Е
lstm_15/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_15/transpose/permй
lstm_15/transpose	Transposedropout_14/dropout/Mul_1:z:0lstm_15/transpose/perm:output:0*
T0*,
_output_shapes
:         ю2
lstm_15/transposeg
lstm_15/Shape_1Shapelstm_15/transpose:y:0*
T0*
_output_shapes
:2
lstm_15/Shape_1И
lstm_15/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_15/strided_slice_1/stackМ
lstm_15/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_15/strided_slice_1/stack_1М
lstm_15/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_15/strided_slice_1/stack_2Ю
lstm_15/strided_slice_1StridedSlicelstm_15/Shape_1:output:0&lstm_15/strided_slice_1/stack:output:0(lstm_15/strided_slice_1/stack_1:output:0(lstm_15/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_15/strided_slice_1Х
#lstm_15/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         2%
#lstm_15/TensorArrayV2/element_shape╥
lstm_15/TensorArrayV2TensorListReserve,lstm_15/TensorArrayV2/element_shape:output:0 lstm_15/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_15/TensorArrayV2╧
=lstm_15/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ю   2?
=lstm_15/TensorArrayUnstack/TensorListFromTensor/element_shapeШ
/lstm_15/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_15/transpose:y:0Flstm_15/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type021
/lstm_15/TensorArrayUnstack/TensorListFromTensorИ
lstm_15/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_15/strided_slice_2/stackМ
lstm_15/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_15/strided_slice_2/stack_1М
lstm_15/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_15/strided_slice_2/stack_2н
lstm_15/strided_slice_2StridedSlicelstm_15/transpose:y:0&lstm_15/strided_slice_2/stack:output:0(lstm_15/strided_slice_2/stack_1:output:0(lstm_15/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         ю*
shrink_axis_mask2
lstm_15/strided_slice_2╬
*lstm_15/lstm_cell_15/MatMul/ReadVariableOpReadVariableOp3lstm_15_lstm_cell_15_matmul_readvariableop_resource* 
_output_shapes
:
ю╪*
dtype02,
*lstm_15/lstm_cell_15/MatMul/ReadVariableOp═
lstm_15/lstm_cell_15/MatMulMatMul lstm_15/strided_slice_2:output:02lstm_15/lstm_cell_15/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╪2
lstm_15/lstm_cell_15/MatMul╘
,lstm_15/lstm_cell_15/MatMul_1/ReadVariableOpReadVariableOp5lstm_15_lstm_cell_15_matmul_1_readvariableop_resource* 
_output_shapes
:
Ц╪*
dtype02.
,lstm_15/lstm_cell_15/MatMul_1/ReadVariableOp╔
lstm_15/lstm_cell_15/MatMul_1MatMullstm_15/zeros:output:04lstm_15/lstm_cell_15/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╪2
lstm_15/lstm_cell_15/MatMul_1└
lstm_15/lstm_cell_15/addAddV2%lstm_15/lstm_cell_15/MatMul:product:0'lstm_15/lstm_cell_15/MatMul_1:product:0*
T0*(
_output_shapes
:         ╪2
lstm_15/lstm_cell_15/add╠
+lstm_15/lstm_cell_15/BiasAdd/ReadVariableOpReadVariableOp4lstm_15_lstm_cell_15_biasadd_readvariableop_resource*
_output_shapes	
:╪*
dtype02-
+lstm_15/lstm_cell_15/BiasAdd/ReadVariableOp═
lstm_15/lstm_cell_15/BiasAddBiasAddlstm_15/lstm_cell_15/add:z:03lstm_15/lstm_cell_15/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╪2
lstm_15/lstm_cell_15/BiasAddО
$lstm_15/lstm_cell_15/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2&
$lstm_15/lstm_cell_15/split/split_dimЧ
lstm_15/lstm_cell_15/splitSplit-lstm_15/lstm_cell_15/split/split_dim:output:0%lstm_15/lstm_cell_15/BiasAdd:output:0*
T0*d
_output_shapesR
P:         Ц:         Ц:         Ц:         Ц*
	num_split2
lstm_15/lstm_cell_15/splitЯ
lstm_15/lstm_cell_15/SigmoidSigmoid#lstm_15/lstm_cell_15/split:output:0*
T0*(
_output_shapes
:         Ц2
lstm_15/lstm_cell_15/Sigmoidг
lstm_15/lstm_cell_15/Sigmoid_1Sigmoid#lstm_15/lstm_cell_15/split:output:1*
T0*(
_output_shapes
:         Ц2 
lstm_15/lstm_cell_15/Sigmoid_1м
lstm_15/lstm_cell_15/mulMul"lstm_15/lstm_cell_15/Sigmoid_1:y:0lstm_15/zeros_1:output:0*
T0*(
_output_shapes
:         Ц2
lstm_15/lstm_cell_15/mulЦ
lstm_15/lstm_cell_15/ReluRelu#lstm_15/lstm_cell_15/split:output:2*
T0*(
_output_shapes
:         Ц2
lstm_15/lstm_cell_15/Relu╜
lstm_15/lstm_cell_15/mul_1Mul lstm_15/lstm_cell_15/Sigmoid:y:0'lstm_15/lstm_cell_15/Relu:activations:0*
T0*(
_output_shapes
:         Ц2
lstm_15/lstm_cell_15/mul_1▓
lstm_15/lstm_cell_15/add_1AddV2lstm_15/lstm_cell_15/mul:z:0lstm_15/lstm_cell_15/mul_1:z:0*
T0*(
_output_shapes
:         Ц2
lstm_15/lstm_cell_15/add_1г
lstm_15/lstm_cell_15/Sigmoid_2Sigmoid#lstm_15/lstm_cell_15/split:output:3*
T0*(
_output_shapes
:         Ц2 
lstm_15/lstm_cell_15/Sigmoid_2Х
lstm_15/lstm_cell_15/Relu_1Relulstm_15/lstm_cell_15/add_1:z:0*
T0*(
_output_shapes
:         Ц2
lstm_15/lstm_cell_15/Relu_1┴
lstm_15/lstm_cell_15/mul_2Mul"lstm_15/lstm_cell_15/Sigmoid_2:y:0)lstm_15/lstm_cell_15/Relu_1:activations:0*
T0*(
_output_shapes
:         Ц2
lstm_15/lstm_cell_15/mul_2Я
%lstm_15/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    Ц   2'
%lstm_15/TensorArrayV2_1/element_shape╪
lstm_15/TensorArrayV2_1TensorListReserve.lstm_15/TensorArrayV2_1/element_shape:output:0 lstm_15/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_15/TensorArrayV2_1^
lstm_15/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_15/timeП
 lstm_15/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         2"
 lstm_15/while/maximum_iterationsz
lstm_15/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_15/while/loop_counterЛ
lstm_15/whileWhile#lstm_15/while/loop_counter:output:0)lstm_15/while/maximum_iterations:output:0lstm_15/time:output:0 lstm_15/TensorArrayV2_1:handle:0lstm_15/zeros:output:0lstm_15/zeros_1:output:0 lstm_15/strided_slice_1:output:0?lstm_15/TensorArrayUnstack/TensorListFromTensor:output_handle:03lstm_15_lstm_cell_15_matmul_readvariableop_resource5lstm_15_lstm_cell_15_matmul_1_readvariableop_resource4lstm_15_lstm_cell_15_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :         Ц:         Ц: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *'
bodyR
lstm_15_while_body_26072558*'
condR
lstm_15_while_cond_26072557*M
output_shapes<
:: : : : :         Ц:         Ц: : : : : *
parallel_iterations 2
lstm_15/while┼
8lstm_15/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    Ц   2:
8lstm_15/TensorArrayV2Stack/TensorListStack/element_shapeЙ
*lstm_15/TensorArrayV2Stack/TensorListStackTensorListStacklstm_15/while:output:3Alstm_15/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:         Ц*
element_dtype02,
*lstm_15/TensorArrayV2Stack/TensorListStackС
lstm_15/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         2
lstm_15/strided_slice_3/stackМ
lstm_15/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2!
lstm_15/strided_slice_3/stack_1М
lstm_15/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_15/strided_slice_3/stack_2╦
lstm_15/strided_slice_3StridedSlice3lstm_15/TensorArrayV2Stack/TensorListStack:tensor:0&lstm_15/strided_slice_3/stack:output:0(lstm_15/strided_slice_3/stack_1:output:0(lstm_15/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         Ц*
shrink_axis_mask2
lstm_15/strided_slice_3Й
lstm_15/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_15/transpose_1/perm╞
lstm_15/transpose_1	Transpose3lstm_15/TensorArrayV2Stack/TensorListStack:tensor:0!lstm_15/transpose_1/perm:output:0*
T0*,
_output_shapes
:         Ц2
lstm_15/transpose_1v
lstm_15/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_15/runtimey
dropout_15/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  а?2
dropout_15/dropout/Constк
dropout_15/dropout/MulMullstm_15/transpose_1:y:0!dropout_15/dropout/Const:output:0*
T0*,
_output_shapes
:         Ц2
dropout_15/dropout/Mul{
dropout_15/dropout/ShapeShapelstm_15/transpose_1:y:0*
T0*
_output_shapes
:2
dropout_15/dropout/Shape┌
/dropout_15/dropout/random_uniform/RandomUniformRandomUniform!dropout_15/dropout/Shape:output:0*
T0*,
_output_shapes
:         Ц*
dtype021
/dropout_15/dropout/random_uniform/RandomUniformЛ
!dropout_15/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>2#
!dropout_15/dropout/GreaterEqual/yя
dropout_15/dropout/GreaterEqualGreaterEqual8dropout_15/dropout/random_uniform/RandomUniform:output:0*dropout_15/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:         Ц2!
dropout_15/dropout/GreaterEqualе
dropout_15/dropout/CastCast#dropout_15/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:         Ц2
dropout_15/dropout/Castл
dropout_15/dropout/Mul_1Muldropout_15/dropout/Mul:z:0dropout_15/dropout/Cast:y:0*
T0*,
_output_shapes
:         Ц2
dropout_15/dropout/Mul_1п
 dense_7/Tensordot/ReadVariableOpReadVariableOp)dense_7_tensordot_readvariableop_resource*
_output_shapes
:	Ц*
dtype02"
 dense_7/Tensordot/ReadVariableOpz
dense_7/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_7/Tensordot/axesБ
dense_7/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_7/Tensordot/free~
dense_7/Tensordot/ShapeShapedropout_15/dropout/Mul_1:z:0*
T0*
_output_shapes
:2
dense_7/Tensordot/ShapeД
dense_7/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_7/Tensordot/GatherV2/axis∙
dense_7/Tensordot/GatherV2GatherV2 dense_7/Tensordot/Shape:output:0dense_7/Tensordot/free:output:0(dense_7/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_7/Tensordot/GatherV2И
!dense_7/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_7/Tensordot/GatherV2_1/axis 
dense_7/Tensordot/GatherV2_1GatherV2 dense_7/Tensordot/Shape:output:0dense_7/Tensordot/axes:output:0*dense_7/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_7/Tensordot/GatherV2_1|
dense_7/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_7/Tensordot/Constа
dense_7/Tensordot/ProdProd#dense_7/Tensordot/GatherV2:output:0 dense_7/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_7/Tensordot/ProdА
dense_7/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_7/Tensordot/Const_1и
dense_7/Tensordot/Prod_1Prod%dense_7/Tensordot/GatherV2_1:output:0"dense_7/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_7/Tensordot/Prod_1А
dense_7/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense_7/Tensordot/concat/axis╪
dense_7/Tensordot/concatConcatV2dense_7/Tensordot/free:output:0dense_7/Tensordot/axes:output:0&dense_7/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_7/Tensordot/concatм
dense_7/Tensordot/stackPackdense_7/Tensordot/Prod:output:0!dense_7/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_7/Tensordot/stack┐
dense_7/Tensordot/transpose	Transposedropout_15/dropout/Mul_1:z:0!dense_7/Tensordot/concat:output:0*
T0*,
_output_shapes
:         Ц2
dense_7/Tensordot/transpose┐
dense_7/Tensordot/ReshapeReshapedense_7/Tensordot/transpose:y:0 dense_7/Tensordot/stack:output:0*
T0*0
_output_shapes
:                  2
dense_7/Tensordot/Reshape╛
dense_7/Tensordot/MatMulMatMul"dense_7/Tensordot/Reshape:output:0(dense_7/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_7/Tensordot/MatMulА
dense_7/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
dense_7/Tensordot/Const_2Д
dense_7/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_7/Tensordot/concat_1/axisх
dense_7/Tensordot/concat_1ConcatV2#dense_7/Tensordot/GatherV2:output:0"dense_7/Tensordot/Const_2:output:0(dense_7/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_7/Tensordot/concat_1░
dense_7/TensordotReshape"dense_7/Tensordot/MatMul:product:0#dense_7/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:         2
dense_7/Tensordotд
dense_7/BiasAdd/ReadVariableOpReadVariableOp'dense_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_7/BiasAdd/ReadVariableOpз
dense_7/BiasAddBiasAdddense_7/Tensordot:output:0&dense_7/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:         2
dense_7/BiasAdd}
dense_7/SoftmaxSoftmaxdense_7/BiasAdd:output:0*
T0*+
_output_shapes
:         2
dense_7/Softmaxx
IdentityIdentitydense_7/Softmax:softmax:0^NoOp*
T0*+
_output_shapes
:         2

Identity╞
NoOpNoOp^dense_7/BiasAdd/ReadVariableOp!^dense_7/Tensordot/ReadVariableOp,^lstm_14/lstm_cell_14/BiasAdd/ReadVariableOp+^lstm_14/lstm_cell_14/MatMul/ReadVariableOp-^lstm_14/lstm_cell_14/MatMul_1/ReadVariableOp^lstm_14/while,^lstm_15/lstm_cell_15/BiasAdd/ReadVariableOp+^lstm_15/lstm_cell_15/MatMul/ReadVariableOp-^lstm_15/lstm_cell_15/MatMul_1/ReadVariableOp^lstm_15/while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':         ]: : : : : : : : 2@
dense_7/BiasAdd/ReadVariableOpdense_7/BiasAdd/ReadVariableOp2D
 dense_7/Tensordot/ReadVariableOp dense_7/Tensordot/ReadVariableOp2Z
+lstm_14/lstm_cell_14/BiasAdd/ReadVariableOp+lstm_14/lstm_cell_14/BiasAdd/ReadVariableOp2X
*lstm_14/lstm_cell_14/MatMul/ReadVariableOp*lstm_14/lstm_cell_14/MatMul/ReadVariableOp2\
,lstm_14/lstm_cell_14/MatMul_1/ReadVariableOp,lstm_14/lstm_cell_14/MatMul_1/ReadVariableOp2
lstm_14/whilelstm_14/while2Z
+lstm_15/lstm_cell_15/BiasAdd/ReadVariableOp+lstm_15/lstm_cell_15/BiasAdd/ReadVariableOp2X
*lstm_15/lstm_cell_15/MatMul/ReadVariableOp*lstm_15/lstm_cell_15/MatMul/ReadVariableOp2\
,lstm_15/lstm_cell_15/MatMul_1/ReadVariableOp,lstm_15/lstm_cell_15/MatMul_1/ReadVariableOp2
lstm_15/whilelstm_15/while:S O
+
_output_shapes
:         ]
 
_user_specified_nameinputs
у
╗
*__inference_lstm_15_layer_call_fn_26074009
inputs_0
unknown:
ю╪
	unknown_0:
Ц╪
	unknown_1:	╪
identityИвStatefulPartitionedCallТ
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:                  Ц*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_lstm_15_layer_call_and_return_conditional_losses_260705802
StatefulPartitionedCallЙ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:                  Ц2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':                  ю: : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
5
_output_shapes#
!:                  ю
"
_user_specified_name
inputs/0
щJ
╓

lstm_15_while_body_26072558,
(lstm_15_while_lstm_15_while_loop_counter2
.lstm_15_while_lstm_15_while_maximum_iterations
lstm_15_while_placeholder
lstm_15_while_placeholder_1
lstm_15_while_placeholder_2
lstm_15_while_placeholder_3+
'lstm_15_while_lstm_15_strided_slice_1_0g
clstm_15_while_tensorarrayv2read_tensorlistgetitem_lstm_15_tensorarrayunstack_tensorlistfromtensor_0O
;lstm_15_while_lstm_cell_15_matmul_readvariableop_resource_0:
ю╪Q
=lstm_15_while_lstm_cell_15_matmul_1_readvariableop_resource_0:
Ц╪K
<lstm_15_while_lstm_cell_15_biasadd_readvariableop_resource_0:	╪
lstm_15_while_identity
lstm_15_while_identity_1
lstm_15_while_identity_2
lstm_15_while_identity_3
lstm_15_while_identity_4
lstm_15_while_identity_5)
%lstm_15_while_lstm_15_strided_slice_1e
alstm_15_while_tensorarrayv2read_tensorlistgetitem_lstm_15_tensorarrayunstack_tensorlistfromtensorM
9lstm_15_while_lstm_cell_15_matmul_readvariableop_resource:
ю╪O
;lstm_15_while_lstm_cell_15_matmul_1_readvariableop_resource:
Ц╪I
:lstm_15_while_lstm_cell_15_biasadd_readvariableop_resource:	╪Ив1lstm_15/while/lstm_cell_15/BiasAdd/ReadVariableOpв0lstm_15/while/lstm_cell_15/MatMul/ReadVariableOpв2lstm_15/while/lstm_cell_15/MatMul_1/ReadVariableOp╙
?lstm_15/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ю   2A
?lstm_15/while/TensorArrayV2Read/TensorListGetItem/element_shapeД
1lstm_15/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemclstm_15_while_tensorarrayv2read_tensorlistgetitem_lstm_15_tensorarrayunstack_tensorlistfromtensor_0lstm_15_while_placeholderHlstm_15/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:         ю*
element_dtype023
1lstm_15/while/TensorArrayV2Read/TensorListGetItemт
0lstm_15/while/lstm_cell_15/MatMul/ReadVariableOpReadVariableOp;lstm_15_while_lstm_cell_15_matmul_readvariableop_resource_0* 
_output_shapes
:
ю╪*
dtype022
0lstm_15/while/lstm_cell_15/MatMul/ReadVariableOpў
!lstm_15/while/lstm_cell_15/MatMulMatMul8lstm_15/while/TensorArrayV2Read/TensorListGetItem:item:08lstm_15/while/lstm_cell_15/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╪2#
!lstm_15/while/lstm_cell_15/MatMulш
2lstm_15/while/lstm_cell_15/MatMul_1/ReadVariableOpReadVariableOp=lstm_15_while_lstm_cell_15_matmul_1_readvariableop_resource_0* 
_output_shapes
:
Ц╪*
dtype024
2lstm_15/while/lstm_cell_15/MatMul_1/ReadVariableOpр
#lstm_15/while/lstm_cell_15/MatMul_1MatMullstm_15_while_placeholder_2:lstm_15/while/lstm_cell_15/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╪2%
#lstm_15/while/lstm_cell_15/MatMul_1╪
lstm_15/while/lstm_cell_15/addAddV2+lstm_15/while/lstm_cell_15/MatMul:product:0-lstm_15/while/lstm_cell_15/MatMul_1:product:0*
T0*(
_output_shapes
:         ╪2 
lstm_15/while/lstm_cell_15/addр
1lstm_15/while/lstm_cell_15/BiasAdd/ReadVariableOpReadVariableOp<lstm_15_while_lstm_cell_15_biasadd_readvariableop_resource_0*
_output_shapes	
:╪*
dtype023
1lstm_15/while/lstm_cell_15/BiasAdd/ReadVariableOpх
"lstm_15/while/lstm_cell_15/BiasAddBiasAdd"lstm_15/while/lstm_cell_15/add:z:09lstm_15/while/lstm_cell_15/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╪2$
"lstm_15/while/lstm_cell_15/BiasAddЪ
*lstm_15/while/lstm_cell_15/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2,
*lstm_15/while/lstm_cell_15/split/split_dimп
 lstm_15/while/lstm_cell_15/splitSplit3lstm_15/while/lstm_cell_15/split/split_dim:output:0+lstm_15/while/lstm_cell_15/BiasAdd:output:0*
T0*d
_output_shapesR
P:         Ц:         Ц:         Ц:         Ц*
	num_split2"
 lstm_15/while/lstm_cell_15/split▒
"lstm_15/while/lstm_cell_15/SigmoidSigmoid)lstm_15/while/lstm_cell_15/split:output:0*
T0*(
_output_shapes
:         Ц2$
"lstm_15/while/lstm_cell_15/Sigmoid╡
$lstm_15/while/lstm_cell_15/Sigmoid_1Sigmoid)lstm_15/while/lstm_cell_15/split:output:1*
T0*(
_output_shapes
:         Ц2&
$lstm_15/while/lstm_cell_15/Sigmoid_1┴
lstm_15/while/lstm_cell_15/mulMul(lstm_15/while/lstm_cell_15/Sigmoid_1:y:0lstm_15_while_placeholder_3*
T0*(
_output_shapes
:         Ц2 
lstm_15/while/lstm_cell_15/mulи
lstm_15/while/lstm_cell_15/ReluRelu)lstm_15/while/lstm_cell_15/split:output:2*
T0*(
_output_shapes
:         Ц2!
lstm_15/while/lstm_cell_15/Relu╒
 lstm_15/while/lstm_cell_15/mul_1Mul&lstm_15/while/lstm_cell_15/Sigmoid:y:0-lstm_15/while/lstm_cell_15/Relu:activations:0*
T0*(
_output_shapes
:         Ц2"
 lstm_15/while/lstm_cell_15/mul_1╩
 lstm_15/while/lstm_cell_15/add_1AddV2"lstm_15/while/lstm_cell_15/mul:z:0$lstm_15/while/lstm_cell_15/mul_1:z:0*
T0*(
_output_shapes
:         Ц2"
 lstm_15/while/lstm_cell_15/add_1╡
$lstm_15/while/lstm_cell_15/Sigmoid_2Sigmoid)lstm_15/while/lstm_cell_15/split:output:3*
T0*(
_output_shapes
:         Ц2&
$lstm_15/while/lstm_cell_15/Sigmoid_2з
!lstm_15/while/lstm_cell_15/Relu_1Relu$lstm_15/while/lstm_cell_15/add_1:z:0*
T0*(
_output_shapes
:         Ц2#
!lstm_15/while/lstm_cell_15/Relu_1┘
 lstm_15/while/lstm_cell_15/mul_2Mul(lstm_15/while/lstm_cell_15/Sigmoid_2:y:0/lstm_15/while/lstm_cell_15/Relu_1:activations:0*
T0*(
_output_shapes
:         Ц2"
 lstm_15/while/lstm_cell_15/mul_2И
2lstm_15/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_15_while_placeholder_1lstm_15_while_placeholder$lstm_15/while/lstm_cell_15/mul_2:z:0*
_output_shapes
: *
element_dtype024
2lstm_15/while/TensorArrayV2Write/TensorListSetIteml
lstm_15/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_15/while/add/yЙ
lstm_15/while/addAddV2lstm_15_while_placeholderlstm_15/while/add/y:output:0*
T0*
_output_shapes
: 2
lstm_15/while/addp
lstm_15/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_15/while/add_1/yЮ
lstm_15/while/add_1AddV2(lstm_15_while_lstm_15_while_loop_counterlstm_15/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lstm_15/while/add_1Л
lstm_15/while/IdentityIdentitylstm_15/while/add_1:z:0^lstm_15/while/NoOp*
T0*
_output_shapes
: 2
lstm_15/while/Identityж
lstm_15/while/Identity_1Identity.lstm_15_while_lstm_15_while_maximum_iterations^lstm_15/while/NoOp*
T0*
_output_shapes
: 2
lstm_15/while/Identity_1Н
lstm_15/while/Identity_2Identitylstm_15/while/add:z:0^lstm_15/while/NoOp*
T0*
_output_shapes
: 2
lstm_15/while/Identity_2║
lstm_15/while/Identity_3IdentityBlstm_15/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_15/while/NoOp*
T0*
_output_shapes
: 2
lstm_15/while/Identity_3о
lstm_15/while/Identity_4Identity$lstm_15/while/lstm_cell_15/mul_2:z:0^lstm_15/while/NoOp*
T0*(
_output_shapes
:         Ц2
lstm_15/while/Identity_4о
lstm_15/while/Identity_5Identity$lstm_15/while/lstm_cell_15/add_1:z:0^lstm_15/while/NoOp*
T0*(
_output_shapes
:         Ц2
lstm_15/while/Identity_5Ж
lstm_15/while/NoOpNoOp2^lstm_15/while/lstm_cell_15/BiasAdd/ReadVariableOp1^lstm_15/while/lstm_cell_15/MatMul/ReadVariableOp3^lstm_15/while/lstm_cell_15/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
lstm_15/while/NoOp"9
lstm_15_while_identitylstm_15/while/Identity:output:0"=
lstm_15_while_identity_1!lstm_15/while/Identity_1:output:0"=
lstm_15_while_identity_2!lstm_15/while/Identity_2:output:0"=
lstm_15_while_identity_3!lstm_15/while/Identity_3:output:0"=
lstm_15_while_identity_4!lstm_15/while/Identity_4:output:0"=
lstm_15_while_identity_5!lstm_15/while/Identity_5:output:0"P
%lstm_15_while_lstm_15_strided_slice_1'lstm_15_while_lstm_15_strided_slice_1_0"z
:lstm_15_while_lstm_cell_15_biasadd_readvariableop_resource<lstm_15_while_lstm_cell_15_biasadd_readvariableop_resource_0"|
;lstm_15_while_lstm_cell_15_matmul_1_readvariableop_resource=lstm_15_while_lstm_cell_15_matmul_1_readvariableop_resource_0"x
9lstm_15_while_lstm_cell_15_matmul_readvariableop_resource;lstm_15_while_lstm_cell_15_matmul_readvariableop_resource_0"╚
alstm_15_while_tensorarrayv2read_tensorlistgetitem_lstm_15_tensorarrayunstack_tensorlistfromtensorclstm_15_while_tensorarrayv2read_tensorlistgetitem_lstm_15_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :         Ц:         Ц: : : : : 2f
1lstm_15/while/lstm_cell_15/BiasAdd/ReadVariableOp1lstm_15/while/lstm_cell_15/BiasAdd/ReadVariableOp2d
0lstm_15/while/lstm_cell_15/MatMul/ReadVariableOp0lstm_15/while/lstm_cell_15/MatMul/ReadVariableOp2h
2lstm_15/while/lstm_cell_15/MatMul_1/ReadVariableOp2lstm_15/while/lstm_cell_15/MatMul_1/ReadVariableOp: 
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
:         Ц:.*
(
_output_shapes
:         Ц:

_output_shapes
: :

_output_shapes
: 
░?
╘
while_body_26071755
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_14_matmul_readvariableop_resource_0:	]╕I
5while_lstm_cell_14_matmul_1_readvariableop_resource_0:
ю╕C
4while_lstm_cell_14_biasadd_readvariableop_resource_0:	╕
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_14_matmul_readvariableop_resource:	]╕G
3while_lstm_cell_14_matmul_1_readvariableop_resource:
ю╕A
2while_lstm_cell_14_biasadd_readvariableop_resource:	╕Ив)while/lstm_cell_14/BiasAdd/ReadVariableOpв(while/lstm_cell_14/MatMul/ReadVariableOpв*while/lstm_cell_14/MatMul_1/ReadVariableOp├
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
(while/lstm_cell_14/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_14_matmul_readvariableop_resource_0*
_output_shapes
:	]╕*
dtype02*
(while/lstm_cell_14/MatMul/ReadVariableOp╫
while/lstm_cell_14/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_14/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╕2
while/lstm_cell_14/MatMul╨
*while/lstm_cell_14/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_14_matmul_1_readvariableop_resource_0* 
_output_shapes
:
ю╕*
dtype02,
*while/lstm_cell_14/MatMul_1/ReadVariableOp└
while/lstm_cell_14/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_14/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╕2
while/lstm_cell_14/MatMul_1╕
while/lstm_cell_14/addAddV2#while/lstm_cell_14/MatMul:product:0%while/lstm_cell_14/MatMul_1:product:0*
T0*(
_output_shapes
:         ╕2
while/lstm_cell_14/add╚
)while/lstm_cell_14/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_14_biasadd_readvariableop_resource_0*
_output_shapes	
:╕*
dtype02+
)while/lstm_cell_14/BiasAdd/ReadVariableOp┼
while/lstm_cell_14/BiasAddBiasAddwhile/lstm_cell_14/add:z:01while/lstm_cell_14/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╕2
while/lstm_cell_14/BiasAddК
"while/lstm_cell_14/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_14/split/split_dimП
while/lstm_cell_14/splitSplit+while/lstm_cell_14/split/split_dim:output:0#while/lstm_cell_14/BiasAdd:output:0*
T0*d
_output_shapesR
P:         ю:         ю:         ю:         ю*
	num_split2
while/lstm_cell_14/splitЩ
while/lstm_cell_14/SigmoidSigmoid!while/lstm_cell_14/split:output:0*
T0*(
_output_shapes
:         ю2
while/lstm_cell_14/SigmoidЭ
while/lstm_cell_14/Sigmoid_1Sigmoid!while/lstm_cell_14/split:output:1*
T0*(
_output_shapes
:         ю2
while/lstm_cell_14/Sigmoid_1б
while/lstm_cell_14/mulMul while/lstm_cell_14/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:         ю2
while/lstm_cell_14/mulР
while/lstm_cell_14/ReluRelu!while/lstm_cell_14/split:output:2*
T0*(
_output_shapes
:         ю2
while/lstm_cell_14/Relu╡
while/lstm_cell_14/mul_1Mulwhile/lstm_cell_14/Sigmoid:y:0%while/lstm_cell_14/Relu:activations:0*
T0*(
_output_shapes
:         ю2
while/lstm_cell_14/mul_1к
while/lstm_cell_14/add_1AddV2while/lstm_cell_14/mul:z:0while/lstm_cell_14/mul_1:z:0*
T0*(
_output_shapes
:         ю2
while/lstm_cell_14/add_1Э
while/lstm_cell_14/Sigmoid_2Sigmoid!while/lstm_cell_14/split:output:3*
T0*(
_output_shapes
:         ю2
while/lstm_cell_14/Sigmoid_2П
while/lstm_cell_14/Relu_1Reluwhile/lstm_cell_14/add_1:z:0*
T0*(
_output_shapes
:         ю2
while/lstm_cell_14/Relu_1╣
while/lstm_cell_14/mul_2Mul while/lstm_cell_14/Sigmoid_2:y:0'while/lstm_cell_14/Relu_1:activations:0*
T0*(
_output_shapes
:         ю2
while/lstm_cell_14/mul_2р
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_14/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_14/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:         ю2
while/Identity_4О
while/Identity_5Identitywhile/lstm_cell_14/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:         ю2
while/Identity_5▐

while/NoOpNoOp*^while/lstm_cell_14/BiasAdd/ReadVariableOp)^while/lstm_cell_14/MatMul/ReadVariableOp+^while/lstm_cell_14/MatMul_1/ReadVariableOp*"
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
2while_lstm_cell_14_biasadd_readvariableop_resource4while_lstm_cell_14_biasadd_readvariableop_resource_0"l
3while_lstm_cell_14_matmul_1_readvariableop_resource5while_lstm_cell_14_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_14_matmul_readvariableop_resource3while_lstm_cell_14_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :         ю:         ю: : : : : 2V
)while/lstm_cell_14/BiasAdd/ReadVariableOp)while/lstm_cell_14/BiasAdd/ReadVariableOp2T
(while/lstm_cell_14/MatMul/ReadVariableOp(while/lstm_cell_14/MatMul/ReadVariableOp2X
*while/lstm_cell_14/MatMul_1/ReadVariableOp*while/lstm_cell_14/MatMul_1/ReadVariableOp: 
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
:         ю:.*
(
_output_shapes
:         ю:

_output_shapes
: :

_output_shapes
: 
м

╙
/__inference_sequential_7_layer_call_fn_26071446
lstm_14_input
unknown:	]╕
	unknown_0:
ю╕
	unknown_1:	╕
	unknown_2:
ю╪
	unknown_3:
Ц╪
	unknown_4:	╪
	unknown_5:	Ц
	unknown_6:
identityИвStatefulPartitionedCall╙
StatefulPartitionedCallStatefulPartitionedCalllstm_14_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
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
J__inference_sequential_7_layer_call_and_return_conditional_losses_260714272
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
_user_specified_namelstm_14_input
╔\
б
E__inference_lstm_15_layer_call_and_return_conditional_losses_26073696
inputs_0?
+lstm_cell_15_matmul_readvariableop_resource:
ю╪A
-lstm_cell_15_matmul_1_readvariableop_resource:
Ц╪;
,lstm_cell_15_biasadd_readvariableop_resource:	╪
identityИв#lstm_cell_15/BiasAdd/ReadVariableOpв"lstm_cell_15/MatMul/ReadVariableOpв$lstm_cell_15/MatMul_1/ReadVariableOpвwhileF
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
B :Ц2
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
B :Ц2
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
:         Ц2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :Ц2
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
B :Ц2
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
:         Ц2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permЖ
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*5
_output_shapes#
!:                  ю2
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
valueB"    ю   27
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
strided_slice_2/stack_2¤
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         ю*
shrink_axis_mask2
strided_slice_2╢
"lstm_cell_15/MatMul/ReadVariableOpReadVariableOp+lstm_cell_15_matmul_readvariableop_resource* 
_output_shapes
:
ю╪*
dtype02$
"lstm_cell_15/MatMul/ReadVariableOpн
lstm_cell_15/MatMulMatMulstrided_slice_2:output:0*lstm_cell_15/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╪2
lstm_cell_15/MatMul╝
$lstm_cell_15/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_15_matmul_1_readvariableop_resource* 
_output_shapes
:
Ц╪*
dtype02&
$lstm_cell_15/MatMul_1/ReadVariableOpй
lstm_cell_15/MatMul_1MatMulzeros:output:0,lstm_cell_15/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╪2
lstm_cell_15/MatMul_1а
lstm_cell_15/addAddV2lstm_cell_15/MatMul:product:0lstm_cell_15/MatMul_1:product:0*
T0*(
_output_shapes
:         ╪2
lstm_cell_15/add┤
#lstm_cell_15/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_15_biasadd_readvariableop_resource*
_output_shapes	
:╪*
dtype02%
#lstm_cell_15/BiasAdd/ReadVariableOpн
lstm_cell_15/BiasAddBiasAddlstm_cell_15/add:z:0+lstm_cell_15/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╪2
lstm_cell_15/BiasAdd~
lstm_cell_15/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_15/split/split_dimў
lstm_cell_15/splitSplit%lstm_cell_15/split/split_dim:output:0lstm_cell_15/BiasAdd:output:0*
T0*d
_output_shapesR
P:         Ц:         Ц:         Ц:         Ц*
	num_split2
lstm_cell_15/splitЗ
lstm_cell_15/SigmoidSigmoidlstm_cell_15/split:output:0*
T0*(
_output_shapes
:         Ц2
lstm_cell_15/SigmoidЛ
lstm_cell_15/Sigmoid_1Sigmoidlstm_cell_15/split:output:1*
T0*(
_output_shapes
:         Ц2
lstm_cell_15/Sigmoid_1М
lstm_cell_15/mulMullstm_cell_15/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:         Ц2
lstm_cell_15/mul~
lstm_cell_15/ReluRelulstm_cell_15/split:output:2*
T0*(
_output_shapes
:         Ц2
lstm_cell_15/ReluЭ
lstm_cell_15/mul_1Mullstm_cell_15/Sigmoid:y:0lstm_cell_15/Relu:activations:0*
T0*(
_output_shapes
:         Ц2
lstm_cell_15/mul_1Т
lstm_cell_15/add_1AddV2lstm_cell_15/mul:z:0lstm_cell_15/mul_1:z:0*
T0*(
_output_shapes
:         Ц2
lstm_cell_15/add_1Л
lstm_cell_15/Sigmoid_2Sigmoidlstm_cell_15/split:output:3*
T0*(
_output_shapes
:         Ц2
lstm_cell_15/Sigmoid_2}
lstm_cell_15/Relu_1Relulstm_cell_15/add_1:z:0*
T0*(
_output_shapes
:         Ц2
lstm_cell_15/Relu_1б
lstm_cell_15/mul_2Mullstm_cell_15/Sigmoid_2:y:0!lstm_cell_15/Relu_1:activations:0*
T0*(
_output_shapes
:         Ц2
lstm_cell_15/mul_2П
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    Ц   2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_15_matmul_readvariableop_resource-lstm_cell_15_matmul_1_readvariableop_resource,lstm_cell_15_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :         Ц:         Ц: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_26073612*
condR
while_cond_26073611*M
output_shapes<
:: : : : :         Ц:         Ц: : : : : *
parallel_iterations 2
while╡
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    Ц   22
0TensorArrayV2Stack/TensorListStack/element_shapeЄ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:                  Ц*
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
:         Ц*
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
!:                  Ц2
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
!:                  Ц2

Identity╚
NoOpNoOp$^lstm_cell_15/BiasAdd/ReadVariableOp#^lstm_cell_15/MatMul/ReadVariableOp%^lstm_cell_15/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':                  ю: : : 2J
#lstm_cell_15/BiasAdd/ReadVariableOp#lstm_cell_15/BiasAdd/ReadVariableOp2H
"lstm_cell_15/MatMul/ReadVariableOp"lstm_cell_15/MatMul/ReadVariableOp2L
$lstm_cell_15/MatMul_1/ReadVariableOp$lstm_cell_15/MatMul_1/ReadVariableOp2
whilewhile:_ [
5
_output_shapes#
!:                  ю
"
_user_specified_name
inputs/0
┤?
╓
while_body_26073461
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
3while_lstm_cell_15_matmul_readvariableop_resource_0:
ю╪I
5while_lstm_cell_15_matmul_1_readvariableop_resource_0:
Ц╪C
4while_lstm_cell_15_biasadd_readvariableop_resource_0:	╪
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
1while_lstm_cell_15_matmul_readvariableop_resource:
ю╪G
3while_lstm_cell_15_matmul_1_readvariableop_resource:
Ц╪A
2while_lstm_cell_15_biasadd_readvariableop_resource:	╪Ив)while/lstm_cell_15/BiasAdd/ReadVariableOpв(while/lstm_cell_15/MatMul/ReadVariableOpв*while/lstm_cell_15/MatMul_1/ReadVariableOp├
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ю   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape╘
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:         ю*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem╩
(while/lstm_cell_15/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_15_matmul_readvariableop_resource_0* 
_output_shapes
:
ю╪*
dtype02*
(while/lstm_cell_15/MatMul/ReadVariableOp╫
while/lstm_cell_15/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_15/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╪2
while/lstm_cell_15/MatMul╨
*while/lstm_cell_15/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_15_matmul_1_readvariableop_resource_0* 
_output_shapes
:
Ц╪*
dtype02,
*while/lstm_cell_15/MatMul_1/ReadVariableOp└
while/lstm_cell_15/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_15/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╪2
while/lstm_cell_15/MatMul_1╕
while/lstm_cell_15/addAddV2#while/lstm_cell_15/MatMul:product:0%while/lstm_cell_15/MatMul_1:product:0*
T0*(
_output_shapes
:         ╪2
while/lstm_cell_15/add╚
)while/lstm_cell_15/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_15_biasadd_readvariableop_resource_0*
_output_shapes	
:╪*
dtype02+
)while/lstm_cell_15/BiasAdd/ReadVariableOp┼
while/lstm_cell_15/BiasAddBiasAddwhile/lstm_cell_15/add:z:01while/lstm_cell_15/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╪2
while/lstm_cell_15/BiasAddК
"while/lstm_cell_15/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_15/split/split_dimП
while/lstm_cell_15/splitSplit+while/lstm_cell_15/split/split_dim:output:0#while/lstm_cell_15/BiasAdd:output:0*
T0*d
_output_shapesR
P:         Ц:         Ц:         Ц:         Ц*
	num_split2
while/lstm_cell_15/splitЩ
while/lstm_cell_15/SigmoidSigmoid!while/lstm_cell_15/split:output:0*
T0*(
_output_shapes
:         Ц2
while/lstm_cell_15/SigmoidЭ
while/lstm_cell_15/Sigmoid_1Sigmoid!while/lstm_cell_15/split:output:1*
T0*(
_output_shapes
:         Ц2
while/lstm_cell_15/Sigmoid_1б
while/lstm_cell_15/mulMul while/lstm_cell_15/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:         Ц2
while/lstm_cell_15/mulР
while/lstm_cell_15/ReluRelu!while/lstm_cell_15/split:output:2*
T0*(
_output_shapes
:         Ц2
while/lstm_cell_15/Relu╡
while/lstm_cell_15/mul_1Mulwhile/lstm_cell_15/Sigmoid:y:0%while/lstm_cell_15/Relu:activations:0*
T0*(
_output_shapes
:         Ц2
while/lstm_cell_15/mul_1к
while/lstm_cell_15/add_1AddV2while/lstm_cell_15/mul:z:0while/lstm_cell_15/mul_1:z:0*
T0*(
_output_shapes
:         Ц2
while/lstm_cell_15/add_1Э
while/lstm_cell_15/Sigmoid_2Sigmoid!while/lstm_cell_15/split:output:3*
T0*(
_output_shapes
:         Ц2
while/lstm_cell_15/Sigmoid_2П
while/lstm_cell_15/Relu_1Reluwhile/lstm_cell_15/add_1:z:0*
T0*(
_output_shapes
:         Ц2
while/lstm_cell_15/Relu_1╣
while/lstm_cell_15/mul_2Mul while/lstm_cell_15/Sigmoid_2:y:0'while/lstm_cell_15/Relu_1:activations:0*
T0*(
_output_shapes
:         Ц2
while/lstm_cell_15/mul_2р
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_15/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_15/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:         Ц2
while/Identity_4О
while/Identity_5Identitywhile/lstm_cell_15/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:         Ц2
while/Identity_5▐

while/NoOpNoOp*^while/lstm_cell_15/BiasAdd/ReadVariableOp)^while/lstm_cell_15/MatMul/ReadVariableOp+^while/lstm_cell_15/MatMul_1/ReadVariableOp*"
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
2while_lstm_cell_15_biasadd_readvariableop_resource4while_lstm_cell_15_biasadd_readvariableop_resource_0"l
3while_lstm_cell_15_matmul_1_readvariableop_resource5while_lstm_cell_15_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_15_matmul_readvariableop_resource3while_lstm_cell_15_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :         Ц:         Ц: : : : : 2V
)while/lstm_cell_15/BiasAdd/ReadVariableOp)while/lstm_cell_15/BiasAdd/ReadVariableOp2T
(while/lstm_cell_15/MatMul/ReadVariableOp(while/lstm_cell_15/MatMul/ReadVariableOp2X
*while/lstm_cell_15/MatMul_1/ReadVariableOp*while/lstm_cell_15/MatMul_1/ReadVariableOp: 
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
:         Ц:.*
(
_output_shapes
:         Ц:

_output_shapes
: :

_output_shapes
: 
Ж
Ш
*__inference_dense_7_layer_call_fn_26074109

inputs
unknown:	Ц
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
E__inference_dense_7_layer_call_and_return_conditional_losses_260714202
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
:         Ц: : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:         Ц
 
_user_specified_nameinputs
├\
а
E__inference_lstm_14_layer_call_and_return_conditional_losses_26072870
inputs_0>
+lstm_cell_14_matmul_readvariableop_resource:	]╕A
-lstm_cell_14_matmul_1_readvariableop_resource:
ю╕;
,lstm_cell_14_biasadd_readvariableop_resource:	╕
identityИв#lstm_cell_14/BiasAdd/ReadVariableOpв"lstm_cell_14/MatMul/ReadVariableOpв$lstm_cell_14/MatMul_1/ReadVariableOpвwhileF
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
B :ю2
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
B :ю2
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
:         ю2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :ю2
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
B :ю2
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
:         ю2	
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
"lstm_cell_14/MatMul/ReadVariableOpReadVariableOp+lstm_cell_14_matmul_readvariableop_resource*
_output_shapes
:	]╕*
dtype02$
"lstm_cell_14/MatMul/ReadVariableOpн
lstm_cell_14/MatMulMatMulstrided_slice_2:output:0*lstm_cell_14/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╕2
lstm_cell_14/MatMul╝
$lstm_cell_14/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_14_matmul_1_readvariableop_resource* 
_output_shapes
:
ю╕*
dtype02&
$lstm_cell_14/MatMul_1/ReadVariableOpй
lstm_cell_14/MatMul_1MatMulzeros:output:0,lstm_cell_14/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╕2
lstm_cell_14/MatMul_1а
lstm_cell_14/addAddV2lstm_cell_14/MatMul:product:0lstm_cell_14/MatMul_1:product:0*
T0*(
_output_shapes
:         ╕2
lstm_cell_14/add┤
#lstm_cell_14/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_14_biasadd_readvariableop_resource*
_output_shapes	
:╕*
dtype02%
#lstm_cell_14/BiasAdd/ReadVariableOpн
lstm_cell_14/BiasAddBiasAddlstm_cell_14/add:z:0+lstm_cell_14/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╕2
lstm_cell_14/BiasAdd~
lstm_cell_14/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_14/split/split_dimў
lstm_cell_14/splitSplit%lstm_cell_14/split/split_dim:output:0lstm_cell_14/BiasAdd:output:0*
T0*d
_output_shapesR
P:         ю:         ю:         ю:         ю*
	num_split2
lstm_cell_14/splitЗ
lstm_cell_14/SigmoidSigmoidlstm_cell_14/split:output:0*
T0*(
_output_shapes
:         ю2
lstm_cell_14/SigmoidЛ
lstm_cell_14/Sigmoid_1Sigmoidlstm_cell_14/split:output:1*
T0*(
_output_shapes
:         ю2
lstm_cell_14/Sigmoid_1М
lstm_cell_14/mulMullstm_cell_14/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:         ю2
lstm_cell_14/mul~
lstm_cell_14/ReluRelulstm_cell_14/split:output:2*
T0*(
_output_shapes
:         ю2
lstm_cell_14/ReluЭ
lstm_cell_14/mul_1Mullstm_cell_14/Sigmoid:y:0lstm_cell_14/Relu:activations:0*
T0*(
_output_shapes
:         ю2
lstm_cell_14/mul_1Т
lstm_cell_14/add_1AddV2lstm_cell_14/mul:z:0lstm_cell_14/mul_1:z:0*
T0*(
_output_shapes
:         ю2
lstm_cell_14/add_1Л
lstm_cell_14/Sigmoid_2Sigmoidlstm_cell_14/split:output:3*
T0*(
_output_shapes
:         ю2
lstm_cell_14/Sigmoid_2}
lstm_cell_14/Relu_1Relulstm_cell_14/add_1:z:0*
T0*(
_output_shapes
:         ю2
lstm_cell_14/Relu_1б
lstm_cell_14/mul_2Mullstm_cell_14/Sigmoid_2:y:0!lstm_cell_14/Relu_1:activations:0*
T0*(
_output_shapes
:         ю2
lstm_cell_14/mul_2П
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ю   2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_14_matmul_readvariableop_resource-lstm_cell_14_matmul_1_readvariableop_resource,lstm_cell_14_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :         ю:         ю: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_26072786*
condR
while_cond_26072785*M
output_shapes<
:: : : : :         ю:         ю: : : : : *
parallel_iterations 2
while╡
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ю   22
0TensorArrayV2Stack/TensorListStack/element_shapeЄ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:                  ю*
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
:         ю*
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
!:                  ю2
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
!:                  ю2

Identity╚
NoOpNoOp$^lstm_cell_14/BiasAdd/ReadVariableOp#^lstm_cell_14/MatMul/ReadVariableOp%^lstm_cell_14/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  ]: : : 2J
#lstm_cell_14/BiasAdd/ReadVariableOp#lstm_cell_14/BiasAdd/ReadVariableOp2H
"lstm_cell_14/MatMul/ReadVariableOp"lstm_cell_14/MatMul/ReadVariableOp2L
$lstm_cell_14/MatMul_1/ReadVariableOp$lstm_cell_14/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :                  ]
"
_user_specified_name
inputs/0
╗
f
-__inference_dropout_14_layer_call_fn_26073394

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
:         ю* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_dropout_14_layer_call_and_return_conditional_losses_260716722
StatefulPartitionedCallА
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:         ю2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ю22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:         ю
 
_user_specified_nameinputs
И&
ї
while_body_26070721
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_01
while_lstm_cell_15_26070745_0:
ю╪1
while_lstm_cell_15_26070747_0:
Ц╪,
while_lstm_cell_15_26070749_0:	╪
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor/
while_lstm_cell_15_26070745:
ю╪/
while_lstm_cell_15_26070747:
Ц╪*
while_lstm_cell_15_26070749:	╪Ив*while/lstm_cell_15/StatefulPartitionedCall├
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ю   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape╘
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:         ю*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemь
*while/lstm_cell_15/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_15_26070745_0while_lstm_cell_15_26070747_0while_lstm_cell_15_26070749_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:         Ц:         Ц:         Ц*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_lstm_cell_15_layer_call_and_return_conditional_losses_260706432,
*while/lstm_cell_15/StatefulPartitionedCallў
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder3while/lstm_cell_15/StatefulPartitionedCall:output:0*
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
while/Identity_4Identity3while/lstm_cell_15/StatefulPartitionedCall:output:1^while/NoOp*
T0*(
_output_shapes
:         Ц2
while/Identity_4е
while/Identity_5Identity3while/lstm_cell_15/StatefulPartitionedCall:output:2^while/NoOp*
T0*(
_output_shapes
:         Ц2
while/Identity_5З

while/NoOpNoOp+^while/lstm_cell_15/StatefulPartitionedCall*"
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
while_lstm_cell_15_26070745while_lstm_cell_15_26070745_0"<
while_lstm_cell_15_26070747while_lstm_cell_15_26070747_0"<
while_lstm_cell_15_26070749while_lstm_cell_15_26070749_0"0
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :         Ц:         Ц: : : : : 2X
*while/lstm_cell_15/StatefulPartitionedCall*while/lstm_cell_15/StatefulPartitionedCall: 
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
:         Ц:.*
(
_output_shapes
:         Ц:

_output_shapes
: :

_output_shapes
: 
Д\
Ю
E__inference_lstm_14_layer_call_and_return_conditional_losses_26071839

inputs>
+lstm_cell_14_matmul_readvariableop_resource:	]╕A
-lstm_cell_14_matmul_1_readvariableop_resource:
ю╕;
,lstm_cell_14_biasadd_readvariableop_resource:	╕
identityИв#lstm_cell_14/BiasAdd/ReadVariableOpв"lstm_cell_14/MatMul/ReadVariableOpв$lstm_cell_14/MatMul_1/ReadVariableOpвwhileD
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
B :ю2
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
B :ю2
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
:         ю2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :ю2
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
B :ю2
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
:         ю2	
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
"lstm_cell_14/MatMul/ReadVariableOpReadVariableOp+lstm_cell_14_matmul_readvariableop_resource*
_output_shapes
:	]╕*
dtype02$
"lstm_cell_14/MatMul/ReadVariableOpн
lstm_cell_14/MatMulMatMulstrided_slice_2:output:0*lstm_cell_14/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╕2
lstm_cell_14/MatMul╝
$lstm_cell_14/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_14_matmul_1_readvariableop_resource* 
_output_shapes
:
ю╕*
dtype02&
$lstm_cell_14/MatMul_1/ReadVariableOpй
lstm_cell_14/MatMul_1MatMulzeros:output:0,lstm_cell_14/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╕2
lstm_cell_14/MatMul_1а
lstm_cell_14/addAddV2lstm_cell_14/MatMul:product:0lstm_cell_14/MatMul_1:product:0*
T0*(
_output_shapes
:         ╕2
lstm_cell_14/add┤
#lstm_cell_14/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_14_biasadd_readvariableop_resource*
_output_shapes	
:╕*
dtype02%
#lstm_cell_14/BiasAdd/ReadVariableOpн
lstm_cell_14/BiasAddBiasAddlstm_cell_14/add:z:0+lstm_cell_14/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╕2
lstm_cell_14/BiasAdd~
lstm_cell_14/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_14/split/split_dimў
lstm_cell_14/splitSplit%lstm_cell_14/split/split_dim:output:0lstm_cell_14/BiasAdd:output:0*
T0*d
_output_shapesR
P:         ю:         ю:         ю:         ю*
	num_split2
lstm_cell_14/splitЗ
lstm_cell_14/SigmoidSigmoidlstm_cell_14/split:output:0*
T0*(
_output_shapes
:         ю2
lstm_cell_14/SigmoidЛ
lstm_cell_14/Sigmoid_1Sigmoidlstm_cell_14/split:output:1*
T0*(
_output_shapes
:         ю2
lstm_cell_14/Sigmoid_1М
lstm_cell_14/mulMullstm_cell_14/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:         ю2
lstm_cell_14/mul~
lstm_cell_14/ReluRelulstm_cell_14/split:output:2*
T0*(
_output_shapes
:         ю2
lstm_cell_14/ReluЭ
lstm_cell_14/mul_1Mullstm_cell_14/Sigmoid:y:0lstm_cell_14/Relu:activations:0*
T0*(
_output_shapes
:         ю2
lstm_cell_14/mul_1Т
lstm_cell_14/add_1AddV2lstm_cell_14/mul:z:0lstm_cell_14/mul_1:z:0*
T0*(
_output_shapes
:         ю2
lstm_cell_14/add_1Л
lstm_cell_14/Sigmoid_2Sigmoidlstm_cell_14/split:output:3*
T0*(
_output_shapes
:         ю2
lstm_cell_14/Sigmoid_2}
lstm_cell_14/Relu_1Relulstm_cell_14/add_1:z:0*
T0*(
_output_shapes
:         ю2
lstm_cell_14/Relu_1б
lstm_cell_14/mul_2Mullstm_cell_14/Sigmoid_2:y:0!lstm_cell_14/Relu_1:activations:0*
T0*(
_output_shapes
:         ю2
lstm_cell_14/mul_2П
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ю   2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_14_matmul_readvariableop_resource-lstm_cell_14_matmul_1_readvariableop_resource,lstm_cell_14_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :         ю:         ю: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_26071755*
condR
while_cond_26071754*M
output_shapes<
:: : : : :         ю:         ю: : : : : *
parallel_iterations 2
while╡
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ю   22
0TensorArrayV2Stack/TensorListStack/element_shapeщ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:         ю*
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
:         ю*
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
:         ю2
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
:         ю2

Identity╚
NoOpNoOp$^lstm_cell_14/BiasAdd/ReadVariableOp#^lstm_cell_14/MatMul/ReadVariableOp%^lstm_cell_14/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         ]: : : 2J
#lstm_cell_14/BiasAdd/ReadVariableOp#lstm_cell_14/BiasAdd/ReadVariableOp2H
"lstm_cell_14/MatMul/ReadVariableOp"lstm_cell_14/MatMul/ReadVariableOp2L
$lstm_cell_14/MatMul_1/ReadVariableOp$lstm_cell_14/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:         ]
 
_user_specified_nameinputs
╦F
О
E__inference_lstm_14_layer_call_and_return_conditional_losses_26069950

inputs(
lstm_cell_14_26069868:	]╕)
lstm_cell_14_26069870:
ю╕$
lstm_cell_14_26069872:	╕
identityИв$lstm_cell_14/StatefulPartitionedCallвwhileD
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
B :ю2
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
B :ю2
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
:         ю2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :ю2
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
B :ю2
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
:         ю2	
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
strided_slice_2и
$lstm_cell_14/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_14_26069868lstm_cell_14_26069870lstm_cell_14_26069872*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:         ю:         ю:         ю*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_lstm_cell_14_layer_call_and_return_conditional_losses_260698672&
$lstm_cell_14/StatefulPartitionedCallП
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ю   2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_14_26069868lstm_cell_14_26069870lstm_cell_14_26069872*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :         ю:         ю: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_26069881*
condR
while_cond_26069880*M
output_shapes<
:: : : : :         ю:         ю: : : : : *
parallel_iterations 2
while╡
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ю   22
0TensorArrayV2Stack/TensorListStack/element_shapeЄ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:                  ю*
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
:         ю*
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
!:                  ю2
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
!:                  ю2

Identity}
NoOpNoOp%^lstm_cell_14/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  ]: : : 2L
$lstm_cell_14/StatefulPartitionedCall$lstm_cell_14/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :                  ]
 
_user_specified_nameinputs
и
╖
J__inference_sequential_7_layer_call_and_return_conditional_losses_26071896

inputs#
lstm_14_26071874:	]╕$
lstm_14_26071876:
ю╕
lstm_14_26071878:	╕$
lstm_15_26071882:
ю╪$
lstm_15_26071884:
Ц╪
lstm_15_26071886:	╪#
dense_7_26071890:	Ц
dense_7_26071892:
identityИвdense_7/StatefulPartitionedCallв"dropout_14/StatefulPartitionedCallв"dropout_15/StatefulPartitionedCallвlstm_14/StatefulPartitionedCallвlstm_15/StatefulPartitionedCallо
lstm_14/StatefulPartitionedCallStatefulPartitionedCallinputslstm_14_26071874lstm_14_26071876lstm_14_26071878*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ю*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_lstm_14_layer_call_and_return_conditional_losses_260718392!
lstm_14/StatefulPartitionedCallЫ
"dropout_14/StatefulPartitionedCallStatefulPartitionedCall(lstm_14/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ю* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_dropout_14_layer_call_and_return_conditional_losses_260716722$
"dropout_14/StatefulPartitionedCall╙
lstm_15/StatefulPartitionedCallStatefulPartitionedCall+dropout_14/StatefulPartitionedCall:output:0lstm_15_26071882lstm_15_26071884lstm_15_26071886*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         Ц*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_lstm_15_layer_call_and_return_conditional_losses_260716432!
lstm_15/StatefulPartitionedCall└
"dropout_15/StatefulPartitionedCallStatefulPartitionedCall(lstm_15/StatefulPartitionedCall:output:0#^dropout_14/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         Ц* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_dropout_15_layer_call_and_return_conditional_losses_260714762$
"dropout_15/StatefulPartitionedCall╛
dense_7/StatefulPartitionedCallStatefulPartitionedCall+dropout_15/StatefulPartitionedCall:output:0dense_7_26071890dense_7_26071892*
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
E__inference_dense_7_layer_call_and_return_conditional_losses_260714202!
dense_7/StatefulPartitionedCallЗ
IdentityIdentity(dense_7/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:         2

Identity■
NoOpNoOp ^dense_7/StatefulPartitionedCall#^dropout_14/StatefulPartitionedCall#^dropout_15/StatefulPartitionedCall ^lstm_14/StatefulPartitionedCall ^lstm_15/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':         ]: : : : : : : : 2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall2H
"dropout_14/StatefulPartitionedCall"dropout_14/StatefulPartitionedCall2H
"dropout_15/StatefulPartitionedCall"dropout_15/StatefulPartitionedCall2B
lstm_14/StatefulPartitionedCalllstm_14/StatefulPartitionedCall2B
lstm_15/StatefulPartitionedCalllstm_15/StatefulPartitionedCall:S O
+
_output_shapes
:         ]
 
_user_specified_nameinputs
У
Й
J__inference_lstm_cell_14_layer_call_and_return_conditional_losses_26074141

inputs
states_0
states_11
matmul_readvariableop_resource:	]╕4
 matmul_1_readvariableop_resource:
ю╕.
biasadd_readvariableop_resource:	╕
identity

identity_1

identity_2ИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpвMatMul_1/ReadVariableOpО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	]╕*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╕2
MatMulХ
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
ю╕*
dtype02
MatMul_1/ReadVariableOp|
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╕2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:         ╕2
addН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:╕*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╕2	
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
P:         ю:         ю:         ю:         ю*
	num_split2
split`
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:         ю2	
Sigmoidd
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:         ю2
	Sigmoid_1]
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:         ю2
mulW
ReluRelusplit:output:2*
T0*(
_output_shapes
:         ю2
Relui
mul_1MulSigmoid:y:0Relu:activations:0*
T0*(
_output_shapes
:         ю2
mul_1^
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:         ю2
add_1d
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:         ю2
	Sigmoid_2V
Relu_1Relu	add_1:z:0*
T0*(
_output_shapes
:         ю2
Relu_1m
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*(
_output_shapes
:         ю2
mul_2e
IdentityIdentity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:         ю2

Identityi

Identity_1Identity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:         ю2

Identity_1i

Identity_2Identity	add_1:z:0^NoOp*
T0*(
_output_shapes
:         ю2

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
A:         ]:         ю:         ю: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:         ]
 
_user_specified_nameinputs:RN
(
_output_shapes
:         ю
"
_user_specified_name
states/0:RN
(
_output_shapes
:         ю
"
_user_specified_name
states/1
╪
I
-__inference_dropout_14_layer_call_fn_26073389

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
:         ю* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_dropout_14_layer_call_and_return_conditional_losses_260712222
PartitionedCallq
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:         ю2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ю:T P
,
_output_shapes
:         ю
 
_user_specified_nameinputs
Е&
є
while_body_26069881
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_00
while_lstm_cell_14_26069905_0:	]╕1
while_lstm_cell_14_26069907_0:
ю╕,
while_lstm_cell_14_26069909_0:	╕
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor.
while_lstm_cell_14_26069905:	]╕/
while_lstm_cell_14_26069907:
ю╕*
while_lstm_cell_14_26069909:	╕Ив*while/lstm_cell_14/StatefulPartitionedCall├
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
)while/TensorArrayV2Read/TensorListGetItemь
*while/lstm_cell_14/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_14_26069905_0while_lstm_cell_14_26069907_0while_lstm_cell_14_26069909_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:         ю:         ю:         ю*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_lstm_cell_14_layer_call_and_return_conditional_losses_260698672,
*while/lstm_cell_14/StatefulPartitionedCallў
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder3while/lstm_cell_14/StatefulPartitionedCall:output:0*
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
while/Identity_4Identity3while/lstm_cell_14/StatefulPartitionedCall:output:1^while/NoOp*
T0*(
_output_shapes
:         ю2
while/Identity_4е
while/Identity_5Identity3while/lstm_cell_14/StatefulPartitionedCall:output:2^while/NoOp*
T0*(
_output_shapes
:         ю2
while/Identity_5З

while/NoOpNoOp+^while/lstm_cell_14/StatefulPartitionedCall*"
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
while_lstm_cell_14_26069905while_lstm_cell_14_26069905_0"<
while_lstm_cell_14_26069907while_lstm_cell_14_26069907_0"<
while_lstm_cell_14_26069909while_lstm_cell_14_26069909_0"0
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :         ю:         ю: : : : : 2X
*while/lstm_cell_14/StatefulPartitionedCall*while/lstm_cell_14/StatefulPartitionedCall: 
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
:         ю:.*
(
_output_shapes
:         ю:

_output_shapes
: :

_output_shapes
: 
╘

э
lstm_14_while_cond_26072075,
(lstm_14_while_lstm_14_while_loop_counter2
.lstm_14_while_lstm_14_while_maximum_iterations
lstm_14_while_placeholder
lstm_14_while_placeholder_1
lstm_14_while_placeholder_2
lstm_14_while_placeholder_3.
*lstm_14_while_less_lstm_14_strided_slice_1F
Blstm_14_while_lstm_14_while_cond_26072075___redundant_placeholder0F
Blstm_14_while_lstm_14_while_cond_26072075___redundant_placeholder1F
Blstm_14_while_lstm_14_while_cond_26072075___redundant_placeholder2F
Blstm_14_while_lstm_14_while_cond_26072075___redundant_placeholder3
lstm_14_while_identity
Ш
lstm_14/while/LessLesslstm_14_while_placeholder*lstm_14_while_less_lstm_14_strided_slice_1*
T0*
_output_shapes
: 2
lstm_14/while/Lessu
lstm_14/while/IdentityIdentitylstm_14/while/Less:z:0*
T0
*
_output_shapes
: 2
lstm_14/while/Identity"9
lstm_14_while_identitylstm_14/while/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :         ю:         ю: ::::: 
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
:         ю:.*
(
_output_shapes
:         ю:

_output_shapes
: :

_output_shapes
:
м

╙
/__inference_sequential_7_layer_call_fn_26071936
lstm_14_input
unknown:	]╕
	unknown_0:
ю╕
	unknown_1:	╕
	unknown_2:
ю╪
	unknown_3:
Ц╪
	unknown_4:	╪
	unknown_5:	Ц
	unknown_6:
identityИвStatefulPartitionedCall╙
StatefulPartitionedCallStatefulPartitionedCalllstm_14_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
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
J__inference_sequential_7_layer_call_and_return_conditional_losses_260718962
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
_user_specified_namelstm_14_input
у
═
while_cond_26070720
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_26070720___redundant_placeholder06
2while_while_cond_26070720___redundant_placeholder16
2while_while_cond_26070720___redundant_placeholder26
2while_while_cond_26070720___redundant_placeholder3
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
B: : : : :         Ц:         Ц: ::::: 
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
:         Ц:.*
(
_output_shapes
:         Ц:

_output_shapes
: :

_output_shapes
:
░?
╘
while_body_26073088
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_14_matmul_readvariableop_resource_0:	]╕I
5while_lstm_cell_14_matmul_1_readvariableop_resource_0:
ю╕C
4while_lstm_cell_14_biasadd_readvariableop_resource_0:	╕
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_14_matmul_readvariableop_resource:	]╕G
3while_lstm_cell_14_matmul_1_readvariableop_resource:
ю╕A
2while_lstm_cell_14_biasadd_readvariableop_resource:	╕Ив)while/lstm_cell_14/BiasAdd/ReadVariableOpв(while/lstm_cell_14/MatMul/ReadVariableOpв*while/lstm_cell_14/MatMul_1/ReadVariableOp├
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
(while/lstm_cell_14/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_14_matmul_readvariableop_resource_0*
_output_shapes
:	]╕*
dtype02*
(while/lstm_cell_14/MatMul/ReadVariableOp╫
while/lstm_cell_14/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_14/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╕2
while/lstm_cell_14/MatMul╨
*while/lstm_cell_14/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_14_matmul_1_readvariableop_resource_0* 
_output_shapes
:
ю╕*
dtype02,
*while/lstm_cell_14/MatMul_1/ReadVariableOp└
while/lstm_cell_14/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_14/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╕2
while/lstm_cell_14/MatMul_1╕
while/lstm_cell_14/addAddV2#while/lstm_cell_14/MatMul:product:0%while/lstm_cell_14/MatMul_1:product:0*
T0*(
_output_shapes
:         ╕2
while/lstm_cell_14/add╚
)while/lstm_cell_14/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_14_biasadd_readvariableop_resource_0*
_output_shapes	
:╕*
dtype02+
)while/lstm_cell_14/BiasAdd/ReadVariableOp┼
while/lstm_cell_14/BiasAddBiasAddwhile/lstm_cell_14/add:z:01while/lstm_cell_14/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╕2
while/lstm_cell_14/BiasAddК
"while/lstm_cell_14/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_14/split/split_dimП
while/lstm_cell_14/splitSplit+while/lstm_cell_14/split/split_dim:output:0#while/lstm_cell_14/BiasAdd:output:0*
T0*d
_output_shapesR
P:         ю:         ю:         ю:         ю*
	num_split2
while/lstm_cell_14/splitЩ
while/lstm_cell_14/SigmoidSigmoid!while/lstm_cell_14/split:output:0*
T0*(
_output_shapes
:         ю2
while/lstm_cell_14/SigmoidЭ
while/lstm_cell_14/Sigmoid_1Sigmoid!while/lstm_cell_14/split:output:1*
T0*(
_output_shapes
:         ю2
while/lstm_cell_14/Sigmoid_1б
while/lstm_cell_14/mulMul while/lstm_cell_14/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:         ю2
while/lstm_cell_14/mulР
while/lstm_cell_14/ReluRelu!while/lstm_cell_14/split:output:2*
T0*(
_output_shapes
:         ю2
while/lstm_cell_14/Relu╡
while/lstm_cell_14/mul_1Mulwhile/lstm_cell_14/Sigmoid:y:0%while/lstm_cell_14/Relu:activations:0*
T0*(
_output_shapes
:         ю2
while/lstm_cell_14/mul_1к
while/lstm_cell_14/add_1AddV2while/lstm_cell_14/mul:z:0while/lstm_cell_14/mul_1:z:0*
T0*(
_output_shapes
:         ю2
while/lstm_cell_14/add_1Э
while/lstm_cell_14/Sigmoid_2Sigmoid!while/lstm_cell_14/split:output:3*
T0*(
_output_shapes
:         ю2
while/lstm_cell_14/Sigmoid_2П
while/lstm_cell_14/Relu_1Reluwhile/lstm_cell_14/add_1:z:0*
T0*(
_output_shapes
:         ю2
while/lstm_cell_14/Relu_1╣
while/lstm_cell_14/mul_2Mul while/lstm_cell_14/Sigmoid_2:y:0'while/lstm_cell_14/Relu_1:activations:0*
T0*(
_output_shapes
:         ю2
while/lstm_cell_14/mul_2р
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_14/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_14/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:         ю2
while/Identity_4О
while/Identity_5Identitywhile/lstm_cell_14/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:         ю2
while/Identity_5▐

while/NoOpNoOp*^while/lstm_cell_14/BiasAdd/ReadVariableOp)^while/lstm_cell_14/MatMul/ReadVariableOp+^while/lstm_cell_14/MatMul_1/ReadVariableOp*"
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
2while_lstm_cell_14_biasadd_readvariableop_resource4while_lstm_cell_14_biasadd_readvariableop_resource_0"l
3while_lstm_cell_14_matmul_1_readvariableop_resource5while_lstm_cell_14_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_14_matmul_readvariableop_resource3while_lstm_cell_14_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :         ю:         ю: : : : : 2V
)while/lstm_cell_14/BiasAdd/ReadVariableOp)while/lstm_cell_14/BiasAdd/ReadVariableOp2T
(while/lstm_cell_14/MatMul/ReadVariableOp(while/lstm_cell_14/MatMul/ReadVariableOp2X
*while/lstm_cell_14/MatMul_1/ReadVariableOp*while/lstm_cell_14/MatMul_1/ReadVariableOp: 
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
:         ю:.*
(
_output_shapes
:         ю:

_output_shapes
: :

_output_shapes
: 
У
Й
J__inference_lstm_cell_14_layer_call_and_return_conditional_losses_26074173

inputs
states_0
states_11
matmul_readvariableop_resource:	]╕4
 matmul_1_readvariableop_resource:
ю╕.
biasadd_readvariableop_resource:	╕
identity

identity_1

identity_2ИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpвMatMul_1/ReadVariableOpО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	]╕*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╕2
MatMulХ
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
ю╕*
dtype02
MatMul_1/ReadVariableOp|
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╕2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:         ╕2
addН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:╕*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╕2	
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
P:         ю:         ю:         ю:         ю*
	num_split2
split`
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:         ю2	
Sigmoidd
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:         ю2
	Sigmoid_1]
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:         ю2
mulW
ReluRelusplit:output:2*
T0*(
_output_shapes
:         ю2
Relui
mul_1MulSigmoid:y:0Relu:activations:0*
T0*(
_output_shapes
:         ю2
mul_1^
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:         ю2
add_1d
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:         ю2
	Sigmoid_2V
Relu_1Relu	add_1:z:0*
T0*(
_output_shapes
:         ю2
Relu_1m
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*(
_output_shapes
:         ю2
mul_2e
IdentityIdentity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:         ю2

Identityi

Identity_1Identity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:         ю2

Identity_1i

Identity_2Identity	add_1:z:0^NoOp*
T0*(
_output_shapes
:         ю2

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
A:         ]:         ю:         ю: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:         ]
 
_user_specified_nameinputs:RN
(
_output_shapes
:         ю
"
_user_specified_name
states/0:RN
(
_output_shapes
:         ю
"
_user_specified_name
states/1
╨F
П
E__inference_lstm_15_layer_call_and_return_conditional_losses_26070790

inputs)
lstm_cell_15_26070708:
ю╪)
lstm_cell_15_26070710:
Ц╪$
lstm_cell_15_26070712:	╪
identityИв$lstm_cell_15/StatefulPartitionedCallвwhileD
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
B :Ц2
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
B :Ц2
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
:         Ц2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :Ц2
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
B :Ц2
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
:         Ц2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permД
	transpose	Transposeinputstranspose/perm:output:0*
T0*5
_output_shapes#
!:                  ю2
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
valueB"    ю   27
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
strided_slice_2/stack_2¤
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         ю*
shrink_axis_mask2
strided_slice_2и
$lstm_cell_15/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_15_26070708lstm_cell_15_26070710lstm_cell_15_26070712*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:         Ц:         Ц:         Ц*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_lstm_cell_15_layer_call_and_return_conditional_losses_260706432&
$lstm_cell_15/StatefulPartitionedCallП
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    Ц   2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_15_26070708lstm_cell_15_26070710lstm_cell_15_26070712*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :         Ц:         Ц: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_26070721*
condR
while_cond_26070720*M
output_shapes<
:: : : : :         Ц:         Ц: : : : : *
parallel_iterations 2
while╡
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    Ц   22
0TensorArrayV2Stack/TensorListStack/element_shapeЄ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:                  Ц*
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
:         Ц*
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
!:                  Ц2
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
!:                  Ц2

Identity}
NoOpNoOp%^lstm_cell_15/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':                  ю: : : 2L
$lstm_cell_15/StatefulPartitionedCall$lstm_cell_15/StatefulPartitionedCall2
whilewhile:] Y
5
_output_shapes#
!:                  ю
 
_user_specified_nameinputs
Й
f
H__inference_dropout_14_layer_call_and_return_conditional_losses_26073372

inputs

identity_1_
IdentityIdentityinputs*
T0*,
_output_shapes
:         ю2

Identityn

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:         ю2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ю:T P
,
_output_shapes
:         ю
 
_user_specified_nameinputs
╟
∙
/__inference_lstm_cell_14_layer_call_fn_26074190

inputs
states_0
states_1
unknown:	]╕
	unknown_0:
ю╕
	unknown_1:	╕
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
<:         ю:         ю:         ю*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_lstm_cell_14_layer_call_and_return_conditional_losses_260698672
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         ю2

IdentityА

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:         ю2

Identity_1А

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:         ю2

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
A:         ]:         ю:         ю: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         ]
 
_user_specified_nameinputs:RN
(
_output_shapes
:         ю
"
_user_specified_name
states/0:RN
(
_output_shapes
:         ю
"
_user_specified_name
states/1
█
ё
(sequential_7_lstm_15_while_cond_26069679F
Bsequential_7_lstm_15_while_sequential_7_lstm_15_while_loop_counterL
Hsequential_7_lstm_15_while_sequential_7_lstm_15_while_maximum_iterations*
&sequential_7_lstm_15_while_placeholder,
(sequential_7_lstm_15_while_placeholder_1,
(sequential_7_lstm_15_while_placeholder_2,
(sequential_7_lstm_15_while_placeholder_3H
Dsequential_7_lstm_15_while_less_sequential_7_lstm_15_strided_slice_1`
\sequential_7_lstm_15_while_sequential_7_lstm_15_while_cond_26069679___redundant_placeholder0`
\sequential_7_lstm_15_while_sequential_7_lstm_15_while_cond_26069679___redundant_placeholder1`
\sequential_7_lstm_15_while_sequential_7_lstm_15_while_cond_26069679___redundant_placeholder2`
\sequential_7_lstm_15_while_sequential_7_lstm_15_while_cond_26069679___redundant_placeholder3'
#sequential_7_lstm_15_while_identity
┘
sequential_7/lstm_15/while/LessLess&sequential_7_lstm_15_while_placeholderDsequential_7_lstm_15_while_less_sequential_7_lstm_15_strided_slice_1*
T0*
_output_shapes
: 2!
sequential_7/lstm_15/while/LessЬ
#sequential_7/lstm_15/while/IdentityIdentity#sequential_7/lstm_15/while/Less:z:0*
T0
*
_output_shapes
: 2%
#sequential_7/lstm_15/while/Identity"S
#sequential_7_lstm_15_while_identity,sequential_7/lstm_15/while/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :         Ц:         Ц: ::::: 
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
:         Ц:.*
(
_output_shapes
:         Ц:

_output_shapes
: :

_output_shapes
:
К\
Я
E__inference_lstm_15_layer_call_and_return_conditional_losses_26071374

inputs?
+lstm_cell_15_matmul_readvariableop_resource:
ю╪A
-lstm_cell_15_matmul_1_readvariableop_resource:
Ц╪;
,lstm_cell_15_biasadd_readvariableop_resource:	╪
identityИв#lstm_cell_15/BiasAdd/ReadVariableOpв"lstm_cell_15/MatMul/ReadVariableOpв$lstm_cell_15/MatMul_1/ReadVariableOpвwhileD
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
B :Ц2
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
B :Ц2
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
:         Ц2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :Ц2
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
B :Ц2
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
:         Ц2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm{
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:         ю2
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
valueB"    ю   27
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
strided_slice_2/stack_2¤
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         ю*
shrink_axis_mask2
strided_slice_2╢
"lstm_cell_15/MatMul/ReadVariableOpReadVariableOp+lstm_cell_15_matmul_readvariableop_resource* 
_output_shapes
:
ю╪*
dtype02$
"lstm_cell_15/MatMul/ReadVariableOpн
lstm_cell_15/MatMulMatMulstrided_slice_2:output:0*lstm_cell_15/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╪2
lstm_cell_15/MatMul╝
$lstm_cell_15/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_15_matmul_1_readvariableop_resource* 
_output_shapes
:
Ц╪*
dtype02&
$lstm_cell_15/MatMul_1/ReadVariableOpй
lstm_cell_15/MatMul_1MatMulzeros:output:0,lstm_cell_15/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╪2
lstm_cell_15/MatMul_1а
lstm_cell_15/addAddV2lstm_cell_15/MatMul:product:0lstm_cell_15/MatMul_1:product:0*
T0*(
_output_shapes
:         ╪2
lstm_cell_15/add┤
#lstm_cell_15/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_15_biasadd_readvariableop_resource*
_output_shapes	
:╪*
dtype02%
#lstm_cell_15/BiasAdd/ReadVariableOpн
lstm_cell_15/BiasAddBiasAddlstm_cell_15/add:z:0+lstm_cell_15/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╪2
lstm_cell_15/BiasAdd~
lstm_cell_15/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_15/split/split_dimў
lstm_cell_15/splitSplit%lstm_cell_15/split/split_dim:output:0lstm_cell_15/BiasAdd:output:0*
T0*d
_output_shapesR
P:         Ц:         Ц:         Ц:         Ц*
	num_split2
lstm_cell_15/splitЗ
lstm_cell_15/SigmoidSigmoidlstm_cell_15/split:output:0*
T0*(
_output_shapes
:         Ц2
lstm_cell_15/SigmoidЛ
lstm_cell_15/Sigmoid_1Sigmoidlstm_cell_15/split:output:1*
T0*(
_output_shapes
:         Ц2
lstm_cell_15/Sigmoid_1М
lstm_cell_15/mulMullstm_cell_15/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:         Ц2
lstm_cell_15/mul~
lstm_cell_15/ReluRelulstm_cell_15/split:output:2*
T0*(
_output_shapes
:         Ц2
lstm_cell_15/ReluЭ
lstm_cell_15/mul_1Mullstm_cell_15/Sigmoid:y:0lstm_cell_15/Relu:activations:0*
T0*(
_output_shapes
:         Ц2
lstm_cell_15/mul_1Т
lstm_cell_15/add_1AddV2lstm_cell_15/mul:z:0lstm_cell_15/mul_1:z:0*
T0*(
_output_shapes
:         Ц2
lstm_cell_15/add_1Л
lstm_cell_15/Sigmoid_2Sigmoidlstm_cell_15/split:output:3*
T0*(
_output_shapes
:         Ц2
lstm_cell_15/Sigmoid_2}
lstm_cell_15/Relu_1Relulstm_cell_15/add_1:z:0*
T0*(
_output_shapes
:         Ц2
lstm_cell_15/Relu_1б
lstm_cell_15/mul_2Mullstm_cell_15/Sigmoid_2:y:0!lstm_cell_15/Relu_1:activations:0*
T0*(
_output_shapes
:         Ц2
lstm_cell_15/mul_2П
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    Ц   2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_15_matmul_readvariableop_resource-lstm_cell_15_matmul_1_readvariableop_resource,lstm_cell_15_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :         Ц:         Ц: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_26071290*
condR
while_cond_26071289*M
output_shapes<
:: : : : :         Ц:         Ц: : : : : *
parallel_iterations 2
while╡
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    Ц   22
0TensorArrayV2Stack/TensorListStack/element_shapeщ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:         Ц*
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
:         Ц*
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
:         Ц2
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
:         Ц2

Identity╚
NoOpNoOp$^lstm_cell_15/BiasAdd/ReadVariableOp#^lstm_cell_15/MatMul/ReadVariableOp%^lstm_cell_15/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:         ю: : : 2J
#lstm_cell_15/BiasAdd/ReadVariableOp#lstm_cell_15/BiasAdd/ReadVariableOp2H
"lstm_cell_15/MatMul/ReadVariableOp"lstm_cell_15/MatMul/ReadVariableOp2L
$lstm_cell_15/MatMul_1/ReadVariableOp$lstm_cell_15/MatMul_1/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:         ю
 
_user_specified_nameinputs
╟
∙
/__inference_lstm_cell_14_layer_call_fn_26074207

inputs
states_0
states_1
unknown:	]╕
	unknown_0:
ю╕
	unknown_1:	╕
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
<:         ю:         ю:         ю*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_lstm_cell_14_layer_call_and_return_conditional_losses_260700132
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         ю2

IdentityА

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:         ю2

Identity_1А

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:         ю2

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
A:         ]:         ю:         ю: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         ]
 
_user_specified_nameinputs:RN
(
_output_shapes
:         ю
"
_user_specified_name
states/0:RN
(
_output_shapes
:         ю
"
_user_specified_name
states/1
Ч
К
J__inference_lstm_cell_15_layer_call_and_return_conditional_losses_26074239

inputs
states_0
states_12
matmul_readvariableop_resource:
ю╪4
 matmul_1_readvariableop_resource:
Ц╪.
biasadd_readvariableop_resource:	╪
identity

identity_1

identity_2ИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpвMatMul_1/ReadVariableOpП
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
ю╪*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╪2
MatMulХ
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
Ц╪*
dtype02
MatMul_1/ReadVariableOp|
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╪2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:         ╪2
addН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:╪*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╪2	
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
P:         Ц:         Ц:         Ц:         Ц*
	num_split2
split`
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:         Ц2	
Sigmoidd
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:         Ц2
	Sigmoid_1]
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:         Ц2
mulW
ReluRelusplit:output:2*
T0*(
_output_shapes
:         Ц2
Relui
mul_1MulSigmoid:y:0Relu:activations:0*
T0*(
_output_shapes
:         Ц2
mul_1^
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:         Ц2
add_1d
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:         Ц2
	Sigmoid_2V
Relu_1Relu	add_1:z:0*
T0*(
_output_shapes
:         Ц2
Relu_1m
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*(
_output_shapes
:         Ц2
mul_2e
IdentityIdentity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:         Ц2

Identityi

Identity_1Identity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:         Ц2

Identity_1i

Identity_2Identity	add_1:z:0^NoOp*
T0*(
_output_shapes
:         Ц2

Identity_2Щ
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:         ю:         Ц:         Ц: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:P L
(
_output_shapes
:         ю
 
_user_specified_nameinputs:RN
(
_output_shapes
:         Ц
"
_user_specified_name
states/0:RN
(
_output_shapes
:         Ц
"
_user_specified_name
states/1
у
═
while_cond_26073762
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_26073762___redundant_placeholder06
2while_while_cond_26073762___redundant_placeholder16
2while_while_cond_26073762___redundant_placeholder26
2while_while_cond_26073762___redundant_placeholder3
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
B: : : : :         Ц:         Ц: ::::: 
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
:         Ц:.*
(
_output_shapes
:         Ц:

_output_shapes
: :

_output_shapes
:
┤?
╓
while_body_26071290
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
3while_lstm_cell_15_matmul_readvariableop_resource_0:
ю╪I
5while_lstm_cell_15_matmul_1_readvariableop_resource_0:
Ц╪C
4while_lstm_cell_15_biasadd_readvariableop_resource_0:	╪
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
1while_lstm_cell_15_matmul_readvariableop_resource:
ю╪G
3while_lstm_cell_15_matmul_1_readvariableop_resource:
Ц╪A
2while_lstm_cell_15_biasadd_readvariableop_resource:	╪Ив)while/lstm_cell_15/BiasAdd/ReadVariableOpв(while/lstm_cell_15/MatMul/ReadVariableOpв*while/lstm_cell_15/MatMul_1/ReadVariableOp├
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ю   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape╘
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:         ю*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem╩
(while/lstm_cell_15/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_15_matmul_readvariableop_resource_0* 
_output_shapes
:
ю╪*
dtype02*
(while/lstm_cell_15/MatMul/ReadVariableOp╫
while/lstm_cell_15/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_15/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╪2
while/lstm_cell_15/MatMul╨
*while/lstm_cell_15/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_15_matmul_1_readvariableop_resource_0* 
_output_shapes
:
Ц╪*
dtype02,
*while/lstm_cell_15/MatMul_1/ReadVariableOp└
while/lstm_cell_15/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_15/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╪2
while/lstm_cell_15/MatMul_1╕
while/lstm_cell_15/addAddV2#while/lstm_cell_15/MatMul:product:0%while/lstm_cell_15/MatMul_1:product:0*
T0*(
_output_shapes
:         ╪2
while/lstm_cell_15/add╚
)while/lstm_cell_15/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_15_biasadd_readvariableop_resource_0*
_output_shapes	
:╪*
dtype02+
)while/lstm_cell_15/BiasAdd/ReadVariableOp┼
while/lstm_cell_15/BiasAddBiasAddwhile/lstm_cell_15/add:z:01while/lstm_cell_15/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╪2
while/lstm_cell_15/BiasAddК
"while/lstm_cell_15/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_15/split/split_dimП
while/lstm_cell_15/splitSplit+while/lstm_cell_15/split/split_dim:output:0#while/lstm_cell_15/BiasAdd:output:0*
T0*d
_output_shapesR
P:         Ц:         Ц:         Ц:         Ц*
	num_split2
while/lstm_cell_15/splitЩ
while/lstm_cell_15/SigmoidSigmoid!while/lstm_cell_15/split:output:0*
T0*(
_output_shapes
:         Ц2
while/lstm_cell_15/SigmoidЭ
while/lstm_cell_15/Sigmoid_1Sigmoid!while/lstm_cell_15/split:output:1*
T0*(
_output_shapes
:         Ц2
while/lstm_cell_15/Sigmoid_1б
while/lstm_cell_15/mulMul while/lstm_cell_15/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:         Ц2
while/lstm_cell_15/mulР
while/lstm_cell_15/ReluRelu!while/lstm_cell_15/split:output:2*
T0*(
_output_shapes
:         Ц2
while/lstm_cell_15/Relu╡
while/lstm_cell_15/mul_1Mulwhile/lstm_cell_15/Sigmoid:y:0%while/lstm_cell_15/Relu:activations:0*
T0*(
_output_shapes
:         Ц2
while/lstm_cell_15/mul_1к
while/lstm_cell_15/add_1AddV2while/lstm_cell_15/mul:z:0while/lstm_cell_15/mul_1:z:0*
T0*(
_output_shapes
:         Ц2
while/lstm_cell_15/add_1Э
while/lstm_cell_15/Sigmoid_2Sigmoid!while/lstm_cell_15/split:output:3*
T0*(
_output_shapes
:         Ц2
while/lstm_cell_15/Sigmoid_2П
while/lstm_cell_15/Relu_1Reluwhile/lstm_cell_15/add_1:z:0*
T0*(
_output_shapes
:         Ц2
while/lstm_cell_15/Relu_1╣
while/lstm_cell_15/mul_2Mul while/lstm_cell_15/Sigmoid_2:y:0'while/lstm_cell_15/Relu_1:activations:0*
T0*(
_output_shapes
:         Ц2
while/lstm_cell_15/mul_2р
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_15/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_15/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:         Ц2
while/Identity_4О
while/Identity_5Identitywhile/lstm_cell_15/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:         Ц2
while/Identity_5▐

while/NoOpNoOp*^while/lstm_cell_15/BiasAdd/ReadVariableOp)^while/lstm_cell_15/MatMul/ReadVariableOp+^while/lstm_cell_15/MatMul_1/ReadVariableOp*"
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
2while_lstm_cell_15_biasadd_readvariableop_resource4while_lstm_cell_15_biasadd_readvariableop_resource_0"l
3while_lstm_cell_15_matmul_1_readvariableop_resource5while_lstm_cell_15_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_15_matmul_readvariableop_resource3while_lstm_cell_15_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :         Ц:         Ц: : : : : 2V
)while/lstm_cell_15/BiasAdd/ReadVariableOp)while/lstm_cell_15/BiasAdd/ReadVariableOp2T
(while/lstm_cell_15/MatMul/ReadVariableOp(while/lstm_cell_15/MatMul/ReadVariableOp2X
*while/lstm_cell_15/MatMul_1/ReadVariableOp*while/lstm_cell_15/MatMul_1/ReadVariableOp: 
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
:         Ц:.*
(
_output_shapes
:         Ц:

_output_shapes
: :

_output_shapes
: 
╨F
П
E__inference_lstm_15_layer_call_and_return_conditional_losses_26070580

inputs)
lstm_cell_15_26070498:
ю╪)
lstm_cell_15_26070500:
Ц╪$
lstm_cell_15_26070502:	╪
identityИв$lstm_cell_15/StatefulPartitionedCallвwhileD
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
B :Ц2
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
B :Ц2
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
:         Ц2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :Ц2
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
B :Ц2
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
:         Ц2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permД
	transpose	Transposeinputstranspose/perm:output:0*
T0*5
_output_shapes#
!:                  ю2
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
valueB"    ю   27
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
strided_slice_2/stack_2¤
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         ю*
shrink_axis_mask2
strided_slice_2и
$lstm_cell_15/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_15_26070498lstm_cell_15_26070500lstm_cell_15_26070502*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:         Ц:         Ц:         Ц*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_lstm_cell_15_layer_call_and_return_conditional_losses_260704972&
$lstm_cell_15/StatefulPartitionedCallП
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    Ц   2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_15_26070498lstm_cell_15_26070500lstm_cell_15_26070502*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :         Ц:         Ц: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_26070511*
condR
while_cond_26070510*M
output_shapes<
:: : : : :         Ц:         Ц: : : : : *
parallel_iterations 2
while╡
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    Ц   22
0TensorArrayV2Stack/TensorListStack/element_shapeЄ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:                  Ц*
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
:         Ц*
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
!:                  Ц2
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
!:                  Ц2

Identity}
NoOpNoOp%^lstm_cell_15/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':                  ю: : : 2L
$lstm_cell_15/StatefulPartitionedCall$lstm_cell_15/StatefulPartitionedCall2
whilewhile:] Y
5
_output_shapes#
!:                  ю
 
_user_specified_nameinputs
░?
╘
while_body_26072937
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_14_matmul_readvariableop_resource_0:	]╕I
5while_lstm_cell_14_matmul_1_readvariableop_resource_0:
ю╕C
4while_lstm_cell_14_biasadd_readvariableop_resource_0:	╕
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_14_matmul_readvariableop_resource:	]╕G
3while_lstm_cell_14_matmul_1_readvariableop_resource:
ю╕A
2while_lstm_cell_14_biasadd_readvariableop_resource:	╕Ив)while/lstm_cell_14/BiasAdd/ReadVariableOpв(while/lstm_cell_14/MatMul/ReadVariableOpв*while/lstm_cell_14/MatMul_1/ReadVariableOp├
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
(while/lstm_cell_14/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_14_matmul_readvariableop_resource_0*
_output_shapes
:	]╕*
dtype02*
(while/lstm_cell_14/MatMul/ReadVariableOp╫
while/lstm_cell_14/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_14/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╕2
while/lstm_cell_14/MatMul╨
*while/lstm_cell_14/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_14_matmul_1_readvariableop_resource_0* 
_output_shapes
:
ю╕*
dtype02,
*while/lstm_cell_14/MatMul_1/ReadVariableOp└
while/lstm_cell_14/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_14/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╕2
while/lstm_cell_14/MatMul_1╕
while/lstm_cell_14/addAddV2#while/lstm_cell_14/MatMul:product:0%while/lstm_cell_14/MatMul_1:product:0*
T0*(
_output_shapes
:         ╕2
while/lstm_cell_14/add╚
)while/lstm_cell_14/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_14_biasadd_readvariableop_resource_0*
_output_shapes	
:╕*
dtype02+
)while/lstm_cell_14/BiasAdd/ReadVariableOp┼
while/lstm_cell_14/BiasAddBiasAddwhile/lstm_cell_14/add:z:01while/lstm_cell_14/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╕2
while/lstm_cell_14/BiasAddК
"while/lstm_cell_14/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_14/split/split_dimП
while/lstm_cell_14/splitSplit+while/lstm_cell_14/split/split_dim:output:0#while/lstm_cell_14/BiasAdd:output:0*
T0*d
_output_shapesR
P:         ю:         ю:         ю:         ю*
	num_split2
while/lstm_cell_14/splitЩ
while/lstm_cell_14/SigmoidSigmoid!while/lstm_cell_14/split:output:0*
T0*(
_output_shapes
:         ю2
while/lstm_cell_14/SigmoidЭ
while/lstm_cell_14/Sigmoid_1Sigmoid!while/lstm_cell_14/split:output:1*
T0*(
_output_shapes
:         ю2
while/lstm_cell_14/Sigmoid_1б
while/lstm_cell_14/mulMul while/lstm_cell_14/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:         ю2
while/lstm_cell_14/mulР
while/lstm_cell_14/ReluRelu!while/lstm_cell_14/split:output:2*
T0*(
_output_shapes
:         ю2
while/lstm_cell_14/Relu╡
while/lstm_cell_14/mul_1Mulwhile/lstm_cell_14/Sigmoid:y:0%while/lstm_cell_14/Relu:activations:0*
T0*(
_output_shapes
:         ю2
while/lstm_cell_14/mul_1к
while/lstm_cell_14/add_1AddV2while/lstm_cell_14/mul:z:0while/lstm_cell_14/mul_1:z:0*
T0*(
_output_shapes
:         ю2
while/lstm_cell_14/add_1Э
while/lstm_cell_14/Sigmoid_2Sigmoid!while/lstm_cell_14/split:output:3*
T0*(
_output_shapes
:         ю2
while/lstm_cell_14/Sigmoid_2П
while/lstm_cell_14/Relu_1Reluwhile/lstm_cell_14/add_1:z:0*
T0*(
_output_shapes
:         ю2
while/lstm_cell_14/Relu_1╣
while/lstm_cell_14/mul_2Mul while/lstm_cell_14/Sigmoid_2:y:0'while/lstm_cell_14/Relu_1:activations:0*
T0*(
_output_shapes
:         ю2
while/lstm_cell_14/mul_2р
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_14/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_14/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:         ю2
while/Identity_4О
while/Identity_5Identitywhile/lstm_cell_14/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:         ю2
while/Identity_5▐

while/NoOpNoOp*^while/lstm_cell_14/BiasAdd/ReadVariableOp)^while/lstm_cell_14/MatMul/ReadVariableOp+^while/lstm_cell_14/MatMul_1/ReadVariableOp*"
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
2while_lstm_cell_14_biasadd_readvariableop_resource4while_lstm_cell_14_biasadd_readvariableop_resource_0"l
3while_lstm_cell_14_matmul_1_readvariableop_resource5while_lstm_cell_14_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_14_matmul_readvariableop_resource3while_lstm_cell_14_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :         ю:         ю: : : : : 2V
)while/lstm_cell_14/BiasAdd/ReadVariableOp)while/lstm_cell_14/BiasAdd/ReadVariableOp2T
(while/lstm_cell_14/MatMul/ReadVariableOp(while/lstm_cell_14/MatMul/ReadVariableOp2X
*while/lstm_cell_14/MatMul_1/ReadVariableOp*while/lstm_cell_14/MatMul_1/ReadVariableOp: 
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
:         ю:.*
(
_output_shapes
:         ю:

_output_shapes
: :

_output_shapes
: 
╩
·
/__inference_lstm_cell_15_layer_call_fn_26074305

inputs
states_0
states_1
unknown:
ю╪
	unknown_0:
Ц╪
	unknown_1:	╪
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
<:         Ц:         Ц:         Ц*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_lstm_cell_15_layer_call_and_return_conditional_losses_260706432
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         Ц2

IdentityА

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:         Ц2

Identity_1А

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:         Ц2

Identity_2h
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:         ю:         Ц:         Ц: : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         ю
 
_user_specified_nameinputs:RN
(
_output_shapes
:         Ц
"
_user_specified_name
states/0:RN
(
_output_shapes
:         Ц
"
_user_specified_name
states/1
у
═
while_cond_26071754
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_26071754___redundant_placeholder06
2while_while_cond_26071754___redundant_placeholder16
2while_while_cond_26071754___redundant_placeholder26
2while_while_cond_26071754___redundant_placeholder3
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
B: : : : :         ю:         ю: ::::: 
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
:         ю:.*
(
_output_shapes
:         ю:

_output_shapes
: :

_output_shapes
:
у
═
while_cond_26071558
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_26071558___redundant_placeholder06
2while_while_cond_26071558___redundant_placeholder16
2while_while_cond_26071558___redundant_placeholder26
2while_while_cond_26071558___redundant_placeholder3
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
B: : : : :         Ц:         Ц: ::::: 
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
:         Ц:.*
(
_output_shapes
:         Ц:

_output_shapes
: :

_output_shapes
:
у
═
while_cond_26069880
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_26069880___redundant_placeholder06
2while_while_cond_26069880___redundant_placeholder16
2while_while_cond_26069880___redundant_placeholder26
2while_while_cond_26069880___redundant_placeholder3
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
B: : : : :         ю:         ю: ::::: 
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
:         ю:.*
(
_output_shapes
:         ю:

_output_shapes
: :

_output_shapes
:
Л
З
J__inference_lstm_cell_14_layer_call_and_return_conditional_losses_26070013

inputs

states
states_11
matmul_readvariableop_resource:	]╕4
 matmul_1_readvariableop_resource:
ю╕.
biasadd_readvariableop_resource:	╕
identity

identity_1

identity_2ИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpвMatMul_1/ReadVariableOpО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	]╕*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╕2
MatMulХ
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
ю╕*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╕2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:         ╕2
addН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:╕*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╕2	
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
P:         ю:         ю:         ю:         ю*
	num_split2
split`
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:         ю2	
Sigmoidd
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:         ю2
	Sigmoid_1]
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:         ю2
mulW
ReluRelusplit:output:2*
T0*(
_output_shapes
:         ю2
Relui
mul_1MulSigmoid:y:0Relu:activations:0*
T0*(
_output_shapes
:         ю2
mul_1^
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:         ю2
add_1d
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:         ю2
	Sigmoid_2V
Relu_1Relu	add_1:z:0*
T0*(
_output_shapes
:         ю2
Relu_1m
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*(
_output_shapes
:         ю2
mul_2e
IdentityIdentity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:         ю2

Identityi

Identity_1Identity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:         ю2

Identity_1i

Identity_2Identity	add_1:z:0^NoOp*
T0*(
_output_shapes
:         ю2

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
A:         ]:         ю:         ю: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:         ]
 
_user_specified_nameinputs:PL
(
_output_shapes
:         ю
 
_user_specified_namestates:PL
(
_output_shapes
:         ю
 
_user_specified_namestates
░?
╘
while_body_26072786
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_14_matmul_readvariableop_resource_0:	]╕I
5while_lstm_cell_14_matmul_1_readvariableop_resource_0:
ю╕C
4while_lstm_cell_14_biasadd_readvariableop_resource_0:	╕
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_14_matmul_readvariableop_resource:	]╕G
3while_lstm_cell_14_matmul_1_readvariableop_resource:
ю╕A
2while_lstm_cell_14_biasadd_readvariableop_resource:	╕Ив)while/lstm_cell_14/BiasAdd/ReadVariableOpв(while/lstm_cell_14/MatMul/ReadVariableOpв*while/lstm_cell_14/MatMul_1/ReadVariableOp├
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
(while/lstm_cell_14/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_14_matmul_readvariableop_resource_0*
_output_shapes
:	]╕*
dtype02*
(while/lstm_cell_14/MatMul/ReadVariableOp╫
while/lstm_cell_14/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_14/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╕2
while/lstm_cell_14/MatMul╨
*while/lstm_cell_14/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_14_matmul_1_readvariableop_resource_0* 
_output_shapes
:
ю╕*
dtype02,
*while/lstm_cell_14/MatMul_1/ReadVariableOp└
while/lstm_cell_14/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_14/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╕2
while/lstm_cell_14/MatMul_1╕
while/lstm_cell_14/addAddV2#while/lstm_cell_14/MatMul:product:0%while/lstm_cell_14/MatMul_1:product:0*
T0*(
_output_shapes
:         ╕2
while/lstm_cell_14/add╚
)while/lstm_cell_14/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_14_biasadd_readvariableop_resource_0*
_output_shapes	
:╕*
dtype02+
)while/lstm_cell_14/BiasAdd/ReadVariableOp┼
while/lstm_cell_14/BiasAddBiasAddwhile/lstm_cell_14/add:z:01while/lstm_cell_14/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╕2
while/lstm_cell_14/BiasAddК
"while/lstm_cell_14/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_14/split/split_dimП
while/lstm_cell_14/splitSplit+while/lstm_cell_14/split/split_dim:output:0#while/lstm_cell_14/BiasAdd:output:0*
T0*d
_output_shapesR
P:         ю:         ю:         ю:         ю*
	num_split2
while/lstm_cell_14/splitЩ
while/lstm_cell_14/SigmoidSigmoid!while/lstm_cell_14/split:output:0*
T0*(
_output_shapes
:         ю2
while/lstm_cell_14/SigmoidЭ
while/lstm_cell_14/Sigmoid_1Sigmoid!while/lstm_cell_14/split:output:1*
T0*(
_output_shapes
:         ю2
while/lstm_cell_14/Sigmoid_1б
while/lstm_cell_14/mulMul while/lstm_cell_14/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:         ю2
while/lstm_cell_14/mulР
while/lstm_cell_14/ReluRelu!while/lstm_cell_14/split:output:2*
T0*(
_output_shapes
:         ю2
while/lstm_cell_14/Relu╡
while/lstm_cell_14/mul_1Mulwhile/lstm_cell_14/Sigmoid:y:0%while/lstm_cell_14/Relu:activations:0*
T0*(
_output_shapes
:         ю2
while/lstm_cell_14/mul_1к
while/lstm_cell_14/add_1AddV2while/lstm_cell_14/mul:z:0while/lstm_cell_14/mul_1:z:0*
T0*(
_output_shapes
:         ю2
while/lstm_cell_14/add_1Э
while/lstm_cell_14/Sigmoid_2Sigmoid!while/lstm_cell_14/split:output:3*
T0*(
_output_shapes
:         ю2
while/lstm_cell_14/Sigmoid_2П
while/lstm_cell_14/Relu_1Reluwhile/lstm_cell_14/add_1:z:0*
T0*(
_output_shapes
:         ю2
while/lstm_cell_14/Relu_1╣
while/lstm_cell_14/mul_2Mul while/lstm_cell_14/Sigmoid_2:y:0'while/lstm_cell_14/Relu_1:activations:0*
T0*(
_output_shapes
:         ю2
while/lstm_cell_14/mul_2р
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_14/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_14/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:         ю2
while/Identity_4О
while/Identity_5Identitywhile/lstm_cell_14/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:         ю2
while/Identity_5▐

while/NoOpNoOp*^while/lstm_cell_14/BiasAdd/ReadVariableOp)^while/lstm_cell_14/MatMul/ReadVariableOp+^while/lstm_cell_14/MatMul_1/ReadVariableOp*"
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
2while_lstm_cell_14_biasadd_readvariableop_resource4while_lstm_cell_14_biasadd_readvariableop_resource_0"l
3while_lstm_cell_14_matmul_1_readvariableop_resource5while_lstm_cell_14_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_14_matmul_readvariableop_resource3while_lstm_cell_14_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :         ю:         ю: : : : : 2V
)while/lstm_cell_14/BiasAdd/ReadVariableOp)while/lstm_cell_14/BiasAdd/ReadVariableOp2T
(while/lstm_cell_14/MatMul/ReadVariableOp(while/lstm_cell_14/MatMul/ReadVariableOp2X
*while/lstm_cell_14/MatMul_1/ReadVariableOp*while/lstm_cell_14/MatMul_1/ReadVariableOp: 
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
:         ю:.*
(
_output_shapes
:         ю:

_output_shapes
: :

_output_shapes
: 
╫
g
H__inference_dropout_15_layer_call_and_return_conditional_losses_26071476

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
:         Ц2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape╣
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:         Ц*
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
:         Ц2
dropout/GreaterEqualД
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:         Ц2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:         Ц2
dropout/Mul_1j
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:         Ц2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         Ц:T P
,
_output_shapes
:         Ц
 
_user_specified_nameinputs
у
═
while_cond_26072785
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_26072785___redundant_placeholder06
2while_while_cond_26072785___redundant_placeholder16
2while_while_cond_26072785___redundant_placeholder26
2while_while_cond_26072785___redundant_placeholder3
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
B: : : : :         ю:         ю: ::::: 
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
:         ю:.*
(
_output_shapes
:         ю:

_output_shapes
: :

_output_shapes
:
╜
╛
J__inference_sequential_7_layer_call_and_return_conditional_losses_26071986
lstm_14_input#
lstm_14_26071964:	]╕$
lstm_14_26071966:
ю╕
lstm_14_26071968:	╕$
lstm_15_26071972:
ю╪$
lstm_15_26071974:
Ц╪
lstm_15_26071976:	╪#
dense_7_26071980:	Ц
dense_7_26071982:
identityИвdense_7/StatefulPartitionedCallв"dropout_14/StatefulPartitionedCallв"dropout_15/StatefulPartitionedCallвlstm_14/StatefulPartitionedCallвlstm_15/StatefulPartitionedCall╡
lstm_14/StatefulPartitionedCallStatefulPartitionedCalllstm_14_inputlstm_14_26071964lstm_14_26071966lstm_14_26071968*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ю*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_lstm_14_layer_call_and_return_conditional_losses_260718392!
lstm_14/StatefulPartitionedCallЫ
"dropout_14/StatefulPartitionedCallStatefulPartitionedCall(lstm_14/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ю* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_dropout_14_layer_call_and_return_conditional_losses_260716722$
"dropout_14/StatefulPartitionedCall╙
lstm_15/StatefulPartitionedCallStatefulPartitionedCall+dropout_14/StatefulPartitionedCall:output:0lstm_15_26071972lstm_15_26071974lstm_15_26071976*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         Ц*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_lstm_15_layer_call_and_return_conditional_losses_260716432!
lstm_15/StatefulPartitionedCall└
"dropout_15/StatefulPartitionedCallStatefulPartitionedCall(lstm_15/StatefulPartitionedCall:output:0#^dropout_14/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         Ц* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_dropout_15_layer_call_and_return_conditional_losses_260714762$
"dropout_15/StatefulPartitionedCall╛
dense_7/StatefulPartitionedCallStatefulPartitionedCall+dropout_15/StatefulPartitionedCall:output:0dense_7_26071980dense_7_26071982*
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
E__inference_dense_7_layer_call_and_return_conditional_losses_260714202!
dense_7/StatefulPartitionedCallЗ
IdentityIdentity(dense_7/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:         2

Identity■
NoOpNoOp ^dense_7/StatefulPartitionedCall#^dropout_14/StatefulPartitionedCall#^dropout_15/StatefulPartitionedCall ^lstm_14/StatefulPartitionedCall ^lstm_15/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':         ]: : : : : : : : 2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall2H
"dropout_14/StatefulPartitionedCall"dropout_14/StatefulPartitionedCall2H
"dropout_15/StatefulPartitionedCall"dropout_15/StatefulPartitionedCall2B
lstm_14/StatefulPartitionedCalllstm_14/StatefulPartitionedCall2B
lstm_15/StatefulPartitionedCalllstm_15/StatefulPartitionedCall:Z V
+
_output_shapes
:         ]
'
_user_specified_namelstm_14_input
П
И
J__inference_lstm_cell_15_layer_call_and_return_conditional_losses_26070497

inputs

states
states_12
matmul_readvariableop_resource:
ю╪4
 matmul_1_readvariableop_resource:
Ц╪.
biasadd_readvariableop_resource:	╪
identity

identity_1

identity_2ИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpвMatMul_1/ReadVariableOpП
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
ю╪*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╪2
MatMulХ
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
Ц╪*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╪2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:         ╪2
addН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:╪*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╪2	
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
P:         Ц:         Ц:         Ц:         Ц*
	num_split2
split`
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:         Ц2	
Sigmoidd
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:         Ц2
	Sigmoid_1]
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:         Ц2
mulW
ReluRelusplit:output:2*
T0*(
_output_shapes
:         Ц2
Relui
mul_1MulSigmoid:y:0Relu:activations:0*
T0*(
_output_shapes
:         Ц2
mul_1^
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:         Ц2
add_1d
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:         Ц2
	Sigmoid_2V
Relu_1Relu	add_1:z:0*
T0*(
_output_shapes
:         Ц2
Relu_1m
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*(
_output_shapes
:         Ц2
mul_2e
IdentityIdentity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:         Ц2

Identityi

Identity_1Identity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:         Ц2

Identity_1i

Identity_2Identity	add_1:z:0^NoOp*
T0*(
_output_shapes
:         Ц2

Identity_2Щ
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:         ю:         Ц:         Ц: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:P L
(
_output_shapes
:         ю
 
_user_specified_nameinputs:PL
(
_output_shapes
:         Ц
 
_user_specified_namestates:PL
(
_output_shapes
:         Ц
 
_user_specified_namestates
Й
f
H__inference_dropout_15_layer_call_and_return_conditional_losses_26071387

inputs

identity_1_
IdentityIdentityinputs*
T0*,
_output_shapes
:         Ц2

Identityn

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:         Ц2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         Ц:T P
,
_output_shapes
:         Ц
 
_user_specified_nameinputs
╘!
¤
E__inference_dense_7_layer_call_and_return_conditional_losses_26074100

inputs4
!tensordot_readvariableop_resource:	Ц-
biasadd_readvariableop_resource:
identityИвBiasAdd/ReadVariableOpвTensordot/ReadVariableOpЧ
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes
:	Ц*
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
:         Ц2
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
:         Ц: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:T P
,
_output_shapes
:         Ц
 
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
lstm_14_input:
serving_default_lstm_14_input:0         ]?
dense_74
StatefulPartitionedCall:0         tensorflow/serving/predict:Ї▓
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
regularization_losses
		variables
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
regularization_losses
	variables
	keras_api
*n&call_and_return_all_conditional_losses
o__call__"
_tf_keras_rnn_layer
е
trainable_variables
regularization_losses
	variables
	keras_api
*p&call_and_return_all_conditional_losses
q__call__"
_tf_keras_layer
├
cell

state_spec
trainable_variables
regularization_losses
	variables
	keras_api
*r&call_and_return_all_conditional_losses
s__call__"
_tf_keras_rnn_layer
е
trainable_variables
regularization_losses
	variables
	keras_api
*t&call_and_return_all_conditional_losses
u__call__"
_tf_keras_layer
╗

 kernel
!bias
"trainable_variables
#regularization_losses
$	variables
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
 "
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
╩

,layers
trainable_variables
regularization_losses
		variables
-layer_metrics
.non_trainable_variables
/metrics
0layer_regularization_losses
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
3regularization_losses
4	variables
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
 "
trackable_list_wrapper
5
&0
'1
(2"
trackable_list_wrapper
╣

6layers
trainable_variables
regularization_losses
	variables

7states
8layer_metrics
9non_trainable_variables
:metrics
;layer_regularization_losses
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

<layers
trainable_variables
regularization_losses
	variables
=layer_metrics
>non_trainable_variables
?metrics
@layer_regularization_losses
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
Cregularization_losses
D	variables
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
 "
trackable_list_wrapper
5
)0
*1
+2"
trackable_list_wrapper
╣

Flayers
trainable_variables
regularization_losses
	variables

Gstates
Hlayer_metrics
Inon_trainable_variables
Jmetrics
Klayer_regularization_losses
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

Llayers
trainable_variables
regularization_losses
	variables
Mlayer_metrics
Nnon_trainable_variables
Ometrics
Player_regularization_losses
u__call__
*t&call_and_return_all_conditional_losses
&t"call_and_return_conditional_losses"
_generic_user_object
!:	Ц2dense_7/kernel
:2dense_7/bias
.
 0
!1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
 0
!1"
trackable_list_wrapper
н

Qlayers
"trainable_variables
#regularization_losses
$	variables
Rlayer_metrics
Snon_trainable_variables
Tmetrics
Ulayer_regularization_losses
w__call__
*v&call_and_return_all_conditional_losses
&v"call_and_return_conditional_losses"
_generic_user_object
.:,	]╕2lstm_14/lstm_cell_14/kernel
9:7
ю╕2%lstm_14/lstm_cell_14/recurrent_kernel
(:&╕2lstm_14/lstm_cell_14/bias
/:-
ю╪2lstm_15/lstm_cell_15/kernel
9:7
Ц╪2%lstm_15/lstm_cell_15/recurrent_kernel
(:&╪2lstm_15/lstm_cell_15/bias
C
0
1
2
3
4"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
V0
W1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
5
&0
'1
(2"
trackable_list_wrapper
 "
trackable_list_wrapper
5
&0
'1
(2"
trackable_list_wrapper
н

Xlayers
2trainable_variables
3regularization_losses
4	variables
Ylayer_metrics
Znon_trainable_variables
[metrics
\layer_regularization_losses
z__call__
*y&call_and_return_all_conditional_losses
&y"call_and_return_conditional_losses"
_generic_user_object
'
0"
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
5
)0
*1
+2"
trackable_list_wrapper
 "
trackable_list_wrapper
5
)0
*1
+2"
trackable_list_wrapper
н

]layers
Btrainable_variables
Cregularization_losses
D	variables
^layer_metrics
_non_trainable_variables
`metrics
alayer_regularization_losses
|__call__
*{&call_and_return_all_conditional_losses
&{"call_and_return_conditional_losses"
_generic_user_object
'
0"
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
 "
trackable_dict_wrapper
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
#__inference__wrapped_model_26069792lstm_14_input"Ш
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
J__inference_sequential_7_layer_call_and_return_conditional_losses_26072336
J__inference_sequential_7_layer_call_and_return_conditional_losses_26072677
J__inference_sequential_7_layer_call_and_return_conditional_losses_26071961
J__inference_sequential_7_layer_call_and_return_conditional_losses_26071986└
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
/__inference_sequential_7_layer_call_fn_26071446
/__inference_sequential_7_layer_call_fn_26072698
/__inference_sequential_7_layer_call_fn_26072719
/__inference_sequential_7_layer_call_fn_26071936└
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
E__inference_lstm_14_layer_call_and_return_conditional_losses_26072870
E__inference_lstm_14_layer_call_and_return_conditional_losses_26073021
E__inference_lstm_14_layer_call_and_return_conditional_losses_26073172
E__inference_lstm_14_layer_call_and_return_conditional_losses_26073323╒
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
*__inference_lstm_14_layer_call_fn_26073334
*__inference_lstm_14_layer_call_fn_26073345
*__inference_lstm_14_layer_call_fn_26073356
*__inference_lstm_14_layer_call_fn_26073367╒
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
H__inference_dropout_14_layer_call_and_return_conditional_losses_26073372
H__inference_dropout_14_layer_call_and_return_conditional_losses_26073384┤
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
-__inference_dropout_14_layer_call_fn_26073389
-__inference_dropout_14_layer_call_fn_26073394┤
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
E__inference_lstm_15_layer_call_and_return_conditional_losses_26073545
E__inference_lstm_15_layer_call_and_return_conditional_losses_26073696
E__inference_lstm_15_layer_call_and_return_conditional_losses_26073847
E__inference_lstm_15_layer_call_and_return_conditional_losses_26073998╒
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
*__inference_lstm_15_layer_call_fn_26074009
*__inference_lstm_15_layer_call_fn_26074020
*__inference_lstm_15_layer_call_fn_26074031
*__inference_lstm_15_layer_call_fn_26074042╒
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
H__inference_dropout_15_layer_call_and_return_conditional_losses_26074047
H__inference_dropout_15_layer_call_and_return_conditional_losses_26074059┤
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
-__inference_dropout_15_layer_call_fn_26074064
-__inference_dropout_15_layer_call_fn_26074069┤
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
E__inference_dense_7_layer_call_and_return_conditional_losses_26074100в
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
*__inference_dense_7_layer_call_fn_26074109в
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
&__inference_signature_wrapper_26072009lstm_14_input"Ф
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
J__inference_lstm_cell_14_layer_call_and_return_conditional_losses_26074141
J__inference_lstm_cell_14_layer_call_and_return_conditional_losses_26074173╛
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
/__inference_lstm_cell_14_layer_call_fn_26074190
/__inference_lstm_cell_14_layer_call_fn_26074207╛
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
J__inference_lstm_cell_15_layer_call_and_return_conditional_losses_26074239
J__inference_lstm_cell_15_layer_call_and_return_conditional_losses_26074271╛
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
/__inference_lstm_cell_15_layer_call_fn_26074288
/__inference_lstm_cell_15_layer_call_fn_26074305╛
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
#__inference__wrapped_model_26069792}&'()*+ !:в7
0в-
+К(
lstm_14_input         ]
к "5к2
0
dense_7%К"
dense_7         о
E__inference_dense_7_layer_call_and_return_conditional_losses_26074100e !4в1
*в'
%К"
inputs         Ц
к ")в&
К
0         
Ъ Ж
*__inference_dense_7_layer_call_fn_26074109X !4в1
*в'
%К"
inputs         Ц
к "К         ▓
H__inference_dropout_14_layer_call_and_return_conditional_losses_26073372f8в5
.в+
%К"
inputs         ю
p 
к "*в'
 К
0         ю
Ъ ▓
H__inference_dropout_14_layer_call_and_return_conditional_losses_26073384f8в5
.в+
%К"
inputs         ю
p
к "*в'
 К
0         ю
Ъ К
-__inference_dropout_14_layer_call_fn_26073389Y8в5
.в+
%К"
inputs         ю
p 
к "К         юК
-__inference_dropout_14_layer_call_fn_26073394Y8в5
.в+
%К"
inputs         ю
p
к "К         ю▓
H__inference_dropout_15_layer_call_and_return_conditional_losses_26074047f8в5
.в+
%К"
inputs         Ц
p 
к "*в'
 К
0         Ц
Ъ ▓
H__inference_dropout_15_layer_call_and_return_conditional_losses_26074059f8в5
.в+
%К"
inputs         Ц
p
к "*в'
 К
0         Ц
Ъ К
-__inference_dropout_15_layer_call_fn_26074064Y8в5
.в+
%К"
inputs         Ц
p 
к "К         ЦК
-__inference_dropout_15_layer_call_fn_26074069Y8в5
.в+
%К"
inputs         Ц
p
к "К         Ц╒
E__inference_lstm_14_layer_call_and_return_conditional_losses_26072870Л&'(OвL
EвB
4Ъ1
/К,
inputs/0                  ]

 
p 

 
к "3в0
)К&
0                  ю
Ъ ╒
E__inference_lstm_14_layer_call_and_return_conditional_losses_26073021Л&'(OвL
EвB
4Ъ1
/К,
inputs/0                  ]

 
p

 
к "3в0
)К&
0                  ю
Ъ ╗
E__inference_lstm_14_layer_call_and_return_conditional_losses_26073172r&'(?в<
5в2
$К!
inputs         ]

 
p 

 
к "*в'
 К
0         ю
Ъ ╗
E__inference_lstm_14_layer_call_and_return_conditional_losses_26073323r&'(?в<
5в2
$К!
inputs         ]

 
p

 
к "*в'
 К
0         ю
Ъ м
*__inference_lstm_14_layer_call_fn_26073334~&'(OвL
EвB
4Ъ1
/К,
inputs/0                  ]

 
p 

 
к "&К#                  юм
*__inference_lstm_14_layer_call_fn_26073345~&'(OвL
EвB
4Ъ1
/К,
inputs/0                  ]

 
p

 
к "&К#                  юУ
*__inference_lstm_14_layer_call_fn_26073356e&'(?в<
5в2
$К!
inputs         ]

 
p 

 
к "К         юУ
*__inference_lstm_14_layer_call_fn_26073367e&'(?в<
5в2
$К!
inputs         ]

 
p

 
к "К         ю╓
E__inference_lstm_15_layer_call_and_return_conditional_losses_26073545М)*+PвM
FвC
5Ъ2
0К-
inputs/0                  ю

 
p 

 
к "3в0
)К&
0                  Ц
Ъ ╓
E__inference_lstm_15_layer_call_and_return_conditional_losses_26073696М)*+PвM
FвC
5Ъ2
0К-
inputs/0                  ю

 
p

 
к "3в0
)К&
0                  Ц
Ъ ╝
E__inference_lstm_15_layer_call_and_return_conditional_losses_26073847s)*+@в=
6в3
%К"
inputs         ю

 
p 

 
к "*в'
 К
0         Ц
Ъ ╝
E__inference_lstm_15_layer_call_and_return_conditional_losses_26073998s)*+@в=
6в3
%К"
inputs         ю

 
p

 
к "*в'
 К
0         Ц
Ъ н
*__inference_lstm_15_layer_call_fn_26074009)*+PвM
FвC
5Ъ2
0К-
inputs/0                  ю

 
p 

 
к "&К#                  Цн
*__inference_lstm_15_layer_call_fn_26074020)*+PвM
FвC
5Ъ2
0К-
inputs/0                  ю

 
p

 
к "&К#                  ЦФ
*__inference_lstm_15_layer_call_fn_26074031f)*+@в=
6в3
%К"
inputs         ю

 
p 

 
к "К         ЦФ
*__inference_lstm_15_layer_call_fn_26074042f)*+@в=
6в3
%К"
inputs         ю

 
p

 
к "К         Ц╤
J__inference_lstm_cell_14_layer_call_and_return_conditional_losses_26074141В&'(Вв
xвu
 К
inputs         ]
MвJ
#К 
states/0         ю
#К 
states/1         ю
p 
к "vвs
lвi
К
0/0         ю
GЪD
 К
0/1/0         ю
 К
0/1/1         ю
Ъ ╤
J__inference_lstm_cell_14_layer_call_and_return_conditional_losses_26074173В&'(Вв
xвu
 К
inputs         ]
MвJ
#К 
states/0         ю
#К 
states/1         ю
p
к "vвs
lвi
К
0/0         ю
GЪD
 К
0/1/0         ю
 К
0/1/1         ю
Ъ ж
/__inference_lstm_cell_14_layer_call_fn_26074190Є&'(Вв
xвu
 К
inputs         ]
MвJ
#К 
states/0         ю
#К 
states/1         ю
p 
к "fвc
К
0         ю
CЪ@
К
1/0         ю
К
1/1         юж
/__inference_lstm_cell_14_layer_call_fn_26074207Є&'(Вв
xвu
 К
inputs         ]
MвJ
#К 
states/0         ю
#К 
states/1         ю
p
к "fвc
К
0         ю
CЪ@
К
1/0         ю
К
1/1         ю╙
J__inference_lstm_cell_15_layer_call_and_return_conditional_losses_26074239Д)*+ДвА
yвv
!К
inputs         ю
MвJ
#К 
states/0         Ц
#К 
states/1         Ц
p 
к "vвs
lвi
К
0/0         Ц
GЪD
 К
0/1/0         Ц
 К
0/1/1         Ц
Ъ ╙
J__inference_lstm_cell_15_layer_call_and_return_conditional_losses_26074271Д)*+ДвА
yвv
!К
inputs         ю
MвJ
#К 
states/0         Ц
#К 
states/1         Ц
p
к "vвs
lвi
К
0/0         Ц
GЪD
 К
0/1/0         Ц
 К
0/1/1         Ц
Ъ и
/__inference_lstm_cell_15_layer_call_fn_26074288Ї)*+ДвА
yвv
!К
inputs         ю
MвJ
#К 
states/0         Ц
#К 
states/1         Ц
p 
к "fвc
К
0         Ц
CЪ@
К
1/0         Ц
К
1/1         Ци
/__inference_lstm_cell_15_layer_call_fn_26074305Ї)*+ДвА
yвv
!К
inputs         ю
MвJ
#К 
states/0         Ц
#К 
states/1         Ц
p
к "fвc
К
0         Ц
CЪ@
К
1/0         Ц
К
1/1         Ц╟
J__inference_sequential_7_layer_call_and_return_conditional_losses_26071961y&'()*+ !Bв?
8в5
+К(
lstm_14_input         ]
p 

 
к ")в&
К
0         
Ъ ╟
J__inference_sequential_7_layer_call_and_return_conditional_losses_26071986y&'()*+ !Bв?
8в5
+К(
lstm_14_input         ]
p

 
к ")в&
К
0         
Ъ └
J__inference_sequential_7_layer_call_and_return_conditional_losses_26072336r&'()*+ !;в8
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
J__inference_sequential_7_layer_call_and_return_conditional_losses_26072677r&'()*+ !;в8
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
/__inference_sequential_7_layer_call_fn_26071446l&'()*+ !Bв?
8в5
+К(
lstm_14_input         ]
p 

 
к "К         Я
/__inference_sequential_7_layer_call_fn_26071936l&'()*+ !Bв?
8в5
+К(
lstm_14_input         ]
p

 
к "К         Ш
/__inference_sequential_7_layer_call_fn_26072698e&'()*+ !;в8
1в.
$К!
inputs         ]
p 

 
к "К         Ш
/__inference_sequential_7_layer_call_fn_26072719e&'()*+ !;в8
1в.
$К!
inputs         ]
p

 
к "К         ╣
&__inference_signature_wrapper_26072009О&'()*+ !KвH
в 
Aк>
<
lstm_14_input+К(
lstm_14_input         ]"5к2
0
dense_7%К"
dense_7         