Ç&
°
D
AddV2
x"T
y"T
z"T"
Ttype:
2	
B
AssignVariableOp
resource
value"dtype"
dtypetype
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
­
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
delete_old_dirsbool(
?
Mul
x"T
y"T
z"T"
Ttype:
2	
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

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
dtypetype
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
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
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
¾
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
executor_typestring 
@
StaticRegexFullMatch	
input

output
"
patternstring
ö
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
«
TensorListFromTensor
tensor"element_dtype
element_shape"
shape_type*
output_handleéelement_dtype"
element_dtypetype"

shape_typetype:
2	

TensorListReserve
element_shape"
shape_type
num_elements#
handleéelement_dtype"
element_dtypetype"

shape_typetype:
2	

TensorListStack
input_handle
element_shape
tensor"element_dtype"
element_dtypetype" 
num_elementsintÿÿÿÿÿÿÿÿÿ
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 

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
"serve*2.6.02v2.6.0-rc2-32-g919f693420e8È$
{
dense_11/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	* 
shared_namedense_11/kernel
t
#dense_11/kernel/Read/ReadVariableOpReadVariableOpdense_11/kernel*
_output_shapes
:	*
dtype0
r
dense_11/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_11/bias
k
!dense_11/bias/Read/ReadVariableOpReadVariableOpdense_11/bias*
_output_shapes
:*
dtype0

lstm_22/lstm_cell_22/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	]°*,
shared_namelstm_22/lstm_cell_22/kernel

/lstm_22/lstm_cell_22/kernel/Read/ReadVariableOpReadVariableOplstm_22/lstm_cell_22/kernel*
_output_shapes
:	]°*
dtype0
¨
%lstm_22/lstm_cell_22/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
¬°*6
shared_name'%lstm_22/lstm_cell_22/recurrent_kernel
¡
9lstm_22/lstm_cell_22/recurrent_kernel/Read/ReadVariableOpReadVariableOp%lstm_22/lstm_cell_22/recurrent_kernel* 
_output_shapes
:
¬°*
dtype0

lstm_22/lstm_cell_22/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:°**
shared_namelstm_22/lstm_cell_22/bias

-lstm_22/lstm_cell_22/bias/Read/ReadVariableOpReadVariableOplstm_22/lstm_cell_22/bias*
_output_shapes	
:°*
dtype0

lstm_23/lstm_cell_23/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
¬*,
shared_namelstm_23/lstm_cell_23/kernel

/lstm_23/lstm_cell_23/kernel/Read/ReadVariableOpReadVariableOplstm_23/lstm_cell_23/kernel* 
_output_shapes
:
¬*
dtype0
¨
%lstm_23/lstm_cell_23/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*6
shared_name'%lstm_23/lstm_cell_23/recurrent_kernel
¡
9lstm_23/lstm_cell_23/recurrent_kernel/Read/ReadVariableOpReadVariableOp%lstm_23/lstm_cell_23/recurrent_kernel* 
_output_shapes
:
*
dtype0

lstm_23/lstm_cell_23/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_namelstm_23/lstm_cell_23/bias

-lstm_23/lstm_cell_23/bias/Read/ReadVariableOpReadVariableOplstm_23/lstm_cell_23/bias*
_output_shapes	
:*
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
§"
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*â!
valueØ!BÕ! BÎ!

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
­
,non_trainable_variables
-metrics

.layers
/layer_metrics
trainable_variables
regularization_losses
		variables
0layer_regularization_losses
 

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
¹
6non_trainable_variables
7metrics

8states

9layers
:layer_metrics
trainable_variables
regularization_losses
	variables
;layer_regularization_losses
 
 
 
­
<non_trainable_variables
=metrics

>layers
?layer_metrics
trainable_variables
regularization_losses
	variables
@layer_regularization_losses

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
¹
Fnon_trainable_variables
Gmetrics

Hstates

Ilayers
Jlayer_metrics
trainable_variables
regularization_losses
	variables
Klayer_regularization_losses
 
 
 
­
Lnon_trainable_variables
Mmetrics

Nlayers
Olayer_metrics
trainable_variables
regularization_losses
	variables
Player_regularization_losses
[Y
VARIABLE_VALUEdense_11/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_11/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

 0
!1
 

 0
!1
­
Qnon_trainable_variables
Rmetrics

Slayers
Tlayer_metrics
"trainable_variables
#regularization_losses
$	variables
Ulayer_regularization_losses
a_
VARIABLE_VALUElstm_22/lstm_cell_22/kernel0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUE
ki
VARIABLE_VALUE%lstm_22/lstm_cell_22/recurrent_kernel0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUElstm_22/lstm_cell_22/bias0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUElstm_23/lstm_cell_23/kernel0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUE
ki
VARIABLE_VALUE%lstm_23/lstm_cell_23/recurrent_kernel0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUElstm_23/lstm_cell_23/bias0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUE
 

V0
W1
#
0
1
2
3
4
 
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
­
Xnon_trainable_variables
Ymetrics

Zlayers
[layer_metrics
2trainable_variables
3regularization_losses
4	variables
\layer_regularization_losses
 
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

)0
*1
+2
 

)0
*1
+2
­
]non_trainable_variables
^metrics

_layers
`layer_metrics
Btrainable_variables
Cregularization_losses
D	variables
alayer_regularization_losses
 
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

serving_default_lstm_22_inputPlaceholder*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]*
dtype0* 
shape:ÿÿÿÿÿÿÿÿÿ]
¬
StatefulPartitionedCallStatefulPartitionedCallserving_default_lstm_22_inputlstm_22/lstm_cell_22/kernel%lstm_22/lstm_cell_22/recurrent_kernellstm_22/lstm_cell_22/biaslstm_23/lstm_cell_23/kernel%lstm_23/lstm_cell_23/recurrent_kernellstm_23/lstm_cell_23/biasdense_11/kerneldense_11/bias*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 */
f*R(
&__inference_signature_wrapper_39104765
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#dense_11/kernel/Read/ReadVariableOp!dense_11/bias/Read/ReadVariableOp/lstm_22/lstm_cell_22/kernel/Read/ReadVariableOp9lstm_22/lstm_cell_22/recurrent_kernel/Read/ReadVariableOp-lstm_22/lstm_cell_22/bias/Read/ReadVariableOp/lstm_23/lstm_cell_23/kernel/Read/ReadVariableOp9lstm_23/lstm_cell_23/recurrent_kernel/Read/ReadVariableOp-lstm_23/lstm_cell_23/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOpConst*
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
GPU 2J 8 **
f%R#
!__inference__traced_save_39107120
¢
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_11/kerneldense_11/biaslstm_22/lstm_cell_22/kernel%lstm_22/lstm_cell_22/recurrent_kernellstm_22/lstm_cell_22/biaslstm_23/lstm_cell_23/kernel%lstm_23/lstm_cell_23/recurrent_kernellstm_23/lstm_cell_23/biastotalcounttotal_1count_1*
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
GPU 2J 8 *-
f(R&
$__inference__traced_restore_39107166¤ý#

f
H__inference_dropout_23_layer_call_and_return_conditional_losses_39104143

inputs

identity_1_
IdentityIdentityinputs*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityn

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
\

E__inference_lstm_22_layer_call_and_return_conditional_losses_39103965

inputs>
+lstm_cell_22_matmul_readvariableop_resource:	]°A
-lstm_cell_22_matmul_1_readvariableop_resource:
¬°;
,lstm_cell_22_biasadd_readvariableop_resource:	°
identity¢#lstm_cell_22/BiasAdd/ReadVariableOp¢"lstm_cell_22/MatMul/ReadVariableOp¢$lstm_cell_22/MatMul_1/ReadVariableOp¢whileD
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
strided_slice/stack_2â
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
B :¬2
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
B :è2
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
B :¬2
zeros/packed/1
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
:ÿÿÿÿÿÿÿÿÿ¬2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :¬2
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
B :è2
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
B :¬2
zeros_1/packed/1
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
:ÿÿÿÿÿÿÿÿÿ¬2	
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
:ÿÿÿÿÿÿÿÿÿ]2
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
strided_slice_1/stack_2î
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
TensorArrayV2/element_shape²
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2¿
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ]   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeø
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
strided_slice_2/stack_2ü
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]*
shrink_axis_mask2
strided_slice_2µ
"lstm_cell_22/MatMul/ReadVariableOpReadVariableOp+lstm_cell_22_matmul_readvariableop_resource*
_output_shapes
:	]°*
dtype02$
"lstm_cell_22/MatMul/ReadVariableOp­
lstm_cell_22/MatMulMatMulstrided_slice_2:output:0*lstm_cell_22/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°2
lstm_cell_22/MatMul¼
$lstm_cell_22/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_22_matmul_1_readvariableop_resource* 
_output_shapes
:
¬°*
dtype02&
$lstm_cell_22/MatMul_1/ReadVariableOp©
lstm_cell_22/MatMul_1MatMulzeros:output:0,lstm_cell_22/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°2
lstm_cell_22/MatMul_1 
lstm_cell_22/addAddV2lstm_cell_22/MatMul:product:0lstm_cell_22/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°2
lstm_cell_22/add´
#lstm_cell_22/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_22_biasadd_readvariableop_resource*
_output_shapes	
:°*
dtype02%
#lstm_cell_22/BiasAdd/ReadVariableOp­
lstm_cell_22/BiasAddBiasAddlstm_cell_22/add:z:0+lstm_cell_22/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°2
lstm_cell_22/BiasAdd~
lstm_cell_22/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_22/split/split_dim÷
lstm_cell_22/splitSplit%lstm_cell_22/split/split_dim:output:0lstm_cell_22/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ¬*
	num_split2
lstm_cell_22/split
lstm_cell_22/SigmoidSigmoidlstm_cell_22/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
lstm_cell_22/Sigmoid
lstm_cell_22/Sigmoid_1Sigmoidlstm_cell_22/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
lstm_cell_22/Sigmoid_1
lstm_cell_22/mulMullstm_cell_22/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
lstm_cell_22/mul~
lstm_cell_22/ReluRelulstm_cell_22/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
lstm_cell_22/Relu
lstm_cell_22/mul_1Mullstm_cell_22/Sigmoid:y:0lstm_cell_22/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
lstm_cell_22/mul_1
lstm_cell_22/add_1AddV2lstm_cell_22/mul:z:0lstm_cell_22/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
lstm_cell_22/add_1
lstm_cell_22/Sigmoid_2Sigmoidlstm_cell_22/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
lstm_cell_22/Sigmoid_2}
lstm_cell_22/Relu_1Relulstm_cell_22/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
lstm_cell_22/Relu_1¡
lstm_cell_22/mul_2Mullstm_cell_22/Sigmoid_2:y:0!lstm_cell_22/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
lstm_cell_22/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ¬   2
TensorArrayV2_1/element_shape¸
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
ÿÿÿÿÿÿÿÿÿ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_22_matmul_readvariableop_resource-lstm_cell_22_matmul_1_readvariableop_resource,lstm_cell_22_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ¬: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_39103881*
condR
while_cond_39103880*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ¬: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ¬   22
0TensorArrayV2Stack/TensorListStack/element_shapeé
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
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
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm¦
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
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
:ÿÿÿÿÿÿÿÿÿ¬2

IdentityÈ
NoOpNoOp$^lstm_cell_22/BiasAdd/ReadVariableOp#^lstm_cell_22/MatMul/ReadVariableOp%^lstm_cell_22/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ]: : : 2J
#lstm_cell_22/BiasAdd/ReadVariableOp#lstm_cell_22/BiasAdd/ReadVariableOp2H
"lstm_cell_22/MatMul/ReadVariableOp"lstm_cell_22/MatMul/ReadVariableOp2L
$lstm_cell_22/MatMul_1/ReadVariableOp$lstm_cell_22/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
 
_user_specified_nameinputs
ã
Í
while_cond_39103880
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_39103880___redundant_placeholder06
2while_while_cond_39103880___redundant_placeholder16
2while_while_cond_39103880___redundant_placeholder26
2while_while_cond_39103880___redundant_placeholder3
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
B: : : : :ÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ¬: ::::: 
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
:ÿÿÿÿÿÿÿÿÿ¬:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬:

_output_shapes
: :

_output_shapes
:
°?
Ô
while_body_39103881
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_22_matmul_readvariableop_resource_0:	]°I
5while_lstm_cell_22_matmul_1_readvariableop_resource_0:
¬°C
4while_lstm_cell_22_biasadd_readvariableop_resource_0:	°
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_22_matmul_readvariableop_resource:	]°G
3while_lstm_cell_22_matmul_1_readvariableop_resource:
¬°A
2while_lstm_cell_22_biasadd_readvariableop_resource:	°¢)while/lstm_cell_22/BiasAdd/ReadVariableOp¢(while/lstm_cell_22/MatMul/ReadVariableOp¢*while/lstm_cell_22/MatMul_1/ReadVariableOpÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ]   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÓ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemÉ
(while/lstm_cell_22/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_22_matmul_readvariableop_resource_0*
_output_shapes
:	]°*
dtype02*
(while/lstm_cell_22/MatMul/ReadVariableOp×
while/lstm_cell_22/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_22/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°2
while/lstm_cell_22/MatMulÐ
*while/lstm_cell_22/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_22_matmul_1_readvariableop_resource_0* 
_output_shapes
:
¬°*
dtype02,
*while/lstm_cell_22/MatMul_1/ReadVariableOpÀ
while/lstm_cell_22/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_22/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°2
while/lstm_cell_22/MatMul_1¸
while/lstm_cell_22/addAddV2#while/lstm_cell_22/MatMul:product:0%while/lstm_cell_22/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°2
while/lstm_cell_22/addÈ
)while/lstm_cell_22/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_22_biasadd_readvariableop_resource_0*
_output_shapes	
:°*
dtype02+
)while/lstm_cell_22/BiasAdd/ReadVariableOpÅ
while/lstm_cell_22/BiasAddBiasAddwhile/lstm_cell_22/add:z:01while/lstm_cell_22/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°2
while/lstm_cell_22/BiasAdd
"while/lstm_cell_22/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_22/split/split_dim
while/lstm_cell_22/splitSplit+while/lstm_cell_22/split/split_dim:output:0#while/lstm_cell_22/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ¬*
	num_split2
while/lstm_cell_22/split
while/lstm_cell_22/SigmoidSigmoid!while/lstm_cell_22/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
while/lstm_cell_22/Sigmoid
while/lstm_cell_22/Sigmoid_1Sigmoid!while/lstm_cell_22/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
while/lstm_cell_22/Sigmoid_1¡
while/lstm_cell_22/mulMul while/lstm_cell_22/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
while/lstm_cell_22/mul
while/lstm_cell_22/ReluRelu!while/lstm_cell_22/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
while/lstm_cell_22/Reluµ
while/lstm_cell_22/mul_1Mulwhile/lstm_cell_22/Sigmoid:y:0%while/lstm_cell_22/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
while/lstm_cell_22/mul_1ª
while/lstm_cell_22/add_1AddV2while/lstm_cell_22/mul:z:0while/lstm_cell_22/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
while/lstm_cell_22/add_1
while/lstm_cell_22/Sigmoid_2Sigmoid!while/lstm_cell_22/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
while/lstm_cell_22/Sigmoid_2
while/lstm_cell_22/Relu_1Reluwhile/lstm_cell_22/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
while/lstm_cell_22/Relu_1¹
while/lstm_cell_22/mul_2Mul while/lstm_cell_22/Sigmoid_2:y:0'while/lstm_cell_22/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
while/lstm_cell_22/mul_2à
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_22/mul_2:z:0*
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
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_22/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_22/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
while/Identity_5Þ

while/NoOpNoOp*^while/lstm_cell_22/BiasAdd/ReadVariableOp)^while/lstm_cell_22/MatMul/ReadVariableOp+^while/lstm_cell_22/MatMul_1/ReadVariableOp*"
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
2while_lstm_cell_22_biasadd_readvariableop_resource4while_lstm_cell_22_biasadd_readvariableop_resource_0"l
3while_lstm_cell_22_matmul_1_readvariableop_resource5while_lstm_cell_22_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_22_matmul_readvariableop_resource3while_lstm_cell_22_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ¬: : : : : 2V
)while/lstm_cell_22/BiasAdd/ReadVariableOp)while/lstm_cell_22/BiasAdd/ReadVariableOp2T
(while/lstm_cell_22/MatMul/ReadVariableOp(while/lstm_cell_22/MatMul/ReadVariableOp2X
*while/lstm_cell_22/MatMul_1/ReadVariableOp*while/lstm_cell_22/MatMul_1/ReadVariableOp: 
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
:ÿÿÿÿÿÿÿÿÿ¬:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬:

_output_shapes
: :

_output_shapes
: 
Ø
I
-__inference_dropout_22_layer_call_fn_39106128

inputs
identityË
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_dropout_22_layer_call_and_return_conditional_losses_391039782
PartitionedCallq
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ¬:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
 
_user_specified_nameinputs
è^

)sequential_11_lstm_23_while_body_39102436H
Dsequential_11_lstm_23_while_sequential_11_lstm_23_while_loop_counterN
Jsequential_11_lstm_23_while_sequential_11_lstm_23_while_maximum_iterations+
'sequential_11_lstm_23_while_placeholder-
)sequential_11_lstm_23_while_placeholder_1-
)sequential_11_lstm_23_while_placeholder_2-
)sequential_11_lstm_23_while_placeholder_3G
Csequential_11_lstm_23_while_sequential_11_lstm_23_strided_slice_1_0
sequential_11_lstm_23_while_tensorarrayv2read_tensorlistgetitem_sequential_11_lstm_23_tensorarrayunstack_tensorlistfromtensor_0]
Isequential_11_lstm_23_while_lstm_cell_23_matmul_readvariableop_resource_0:
¬_
Ksequential_11_lstm_23_while_lstm_cell_23_matmul_1_readvariableop_resource_0:
Y
Jsequential_11_lstm_23_while_lstm_cell_23_biasadd_readvariableop_resource_0:	(
$sequential_11_lstm_23_while_identity*
&sequential_11_lstm_23_while_identity_1*
&sequential_11_lstm_23_while_identity_2*
&sequential_11_lstm_23_while_identity_3*
&sequential_11_lstm_23_while_identity_4*
&sequential_11_lstm_23_while_identity_5E
Asequential_11_lstm_23_while_sequential_11_lstm_23_strided_slice_1
}sequential_11_lstm_23_while_tensorarrayv2read_tensorlistgetitem_sequential_11_lstm_23_tensorarrayunstack_tensorlistfromtensor[
Gsequential_11_lstm_23_while_lstm_cell_23_matmul_readvariableop_resource:
¬]
Isequential_11_lstm_23_while_lstm_cell_23_matmul_1_readvariableop_resource:
W
Hsequential_11_lstm_23_while_lstm_cell_23_biasadd_readvariableop_resource:	¢?sequential_11/lstm_23/while/lstm_cell_23/BiasAdd/ReadVariableOp¢>sequential_11/lstm_23/while/lstm_cell_23/MatMul/ReadVariableOp¢@sequential_11/lstm_23/while/lstm_cell_23/MatMul_1/ReadVariableOpï
Msequential_11/lstm_23/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ¬   2O
Msequential_11/lstm_23/while/TensorArrayV2Read/TensorListGetItem/element_shapeØ
?sequential_11/lstm_23/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemsequential_11_lstm_23_while_tensorarrayv2read_tensorlistgetitem_sequential_11_lstm_23_tensorarrayunstack_tensorlistfromtensor_0'sequential_11_lstm_23_while_placeholderVsequential_11/lstm_23/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬*
element_dtype02A
?sequential_11/lstm_23/while/TensorArrayV2Read/TensorListGetItem
>sequential_11/lstm_23/while/lstm_cell_23/MatMul/ReadVariableOpReadVariableOpIsequential_11_lstm_23_while_lstm_cell_23_matmul_readvariableop_resource_0* 
_output_shapes
:
¬*
dtype02@
>sequential_11/lstm_23/while/lstm_cell_23/MatMul/ReadVariableOp¯
/sequential_11/lstm_23/while/lstm_cell_23/MatMulMatMulFsequential_11/lstm_23/while/TensorArrayV2Read/TensorListGetItem:item:0Fsequential_11/lstm_23/while/lstm_cell_23/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ21
/sequential_11/lstm_23/while/lstm_cell_23/MatMul
@sequential_11/lstm_23/while/lstm_cell_23/MatMul_1/ReadVariableOpReadVariableOpKsequential_11_lstm_23_while_lstm_cell_23_matmul_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype02B
@sequential_11/lstm_23/while/lstm_cell_23/MatMul_1/ReadVariableOp
1sequential_11/lstm_23/while/lstm_cell_23/MatMul_1MatMul)sequential_11_lstm_23_while_placeholder_2Hsequential_11/lstm_23/while/lstm_cell_23/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ23
1sequential_11/lstm_23/while/lstm_cell_23/MatMul_1
,sequential_11/lstm_23/while/lstm_cell_23/addAddV29sequential_11/lstm_23/while/lstm_cell_23/MatMul:product:0;sequential_11/lstm_23/while/lstm_cell_23/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2.
,sequential_11/lstm_23/while/lstm_cell_23/add
?sequential_11/lstm_23/while/lstm_cell_23/BiasAdd/ReadVariableOpReadVariableOpJsequential_11_lstm_23_while_lstm_cell_23_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02A
?sequential_11/lstm_23/while/lstm_cell_23/BiasAdd/ReadVariableOp
0sequential_11/lstm_23/while/lstm_cell_23/BiasAddBiasAdd0sequential_11/lstm_23/while/lstm_cell_23/add:z:0Gsequential_11/lstm_23/while/lstm_cell_23/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
0sequential_11/lstm_23/while/lstm_cell_23/BiasAdd¶
8sequential_11/lstm_23/while/lstm_cell_23/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2:
8sequential_11/lstm_23/while/lstm_cell_23/split/split_dimç
.sequential_11/lstm_23/while/lstm_cell_23/splitSplitAsequential_11/lstm_23/while/lstm_cell_23/split/split_dim:output:09sequential_11/lstm_23/while/lstm_cell_23/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split20
.sequential_11/lstm_23/while/lstm_cell_23/splitÛ
0sequential_11/lstm_23/while/lstm_cell_23/SigmoidSigmoid7sequential_11/lstm_23/while/lstm_cell_23/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
0sequential_11/lstm_23/while/lstm_cell_23/Sigmoidß
2sequential_11/lstm_23/while/lstm_cell_23/Sigmoid_1Sigmoid7sequential_11/lstm_23/while/lstm_cell_23/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ24
2sequential_11/lstm_23/while/lstm_cell_23/Sigmoid_1ù
,sequential_11/lstm_23/while/lstm_cell_23/mulMul6sequential_11/lstm_23/while/lstm_cell_23/Sigmoid_1:y:0)sequential_11_lstm_23_while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2.
,sequential_11/lstm_23/while/lstm_cell_23/mulÒ
-sequential_11/lstm_23/while/lstm_cell_23/ReluRelu7sequential_11/lstm_23/while/lstm_cell_23/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2/
-sequential_11/lstm_23/while/lstm_cell_23/Relu
.sequential_11/lstm_23/while/lstm_cell_23/mul_1Mul4sequential_11/lstm_23/while/lstm_cell_23/Sigmoid:y:0;sequential_11/lstm_23/while/lstm_cell_23/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ20
.sequential_11/lstm_23/while/lstm_cell_23/mul_1
.sequential_11/lstm_23/while/lstm_cell_23/add_1AddV20sequential_11/lstm_23/while/lstm_cell_23/mul:z:02sequential_11/lstm_23/while/lstm_cell_23/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ20
.sequential_11/lstm_23/while/lstm_cell_23/add_1ß
2sequential_11/lstm_23/while/lstm_cell_23/Sigmoid_2Sigmoid7sequential_11/lstm_23/while/lstm_cell_23/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ24
2sequential_11/lstm_23/while/lstm_cell_23/Sigmoid_2Ñ
/sequential_11/lstm_23/while/lstm_cell_23/Relu_1Relu2sequential_11/lstm_23/while/lstm_cell_23/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ21
/sequential_11/lstm_23/while/lstm_cell_23/Relu_1
.sequential_11/lstm_23/while/lstm_cell_23/mul_2Mul6sequential_11/lstm_23/while/lstm_cell_23/Sigmoid_2:y:0=sequential_11/lstm_23/while/lstm_cell_23/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ20
.sequential_11/lstm_23/while/lstm_cell_23/mul_2Î
@sequential_11/lstm_23/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem)sequential_11_lstm_23_while_placeholder_1'sequential_11_lstm_23_while_placeholder2sequential_11/lstm_23/while/lstm_cell_23/mul_2:z:0*
_output_shapes
: *
element_dtype02B
@sequential_11/lstm_23/while/TensorArrayV2Write/TensorListSetItem
!sequential_11/lstm_23/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2#
!sequential_11/lstm_23/while/add/yÁ
sequential_11/lstm_23/while/addAddV2'sequential_11_lstm_23_while_placeholder*sequential_11/lstm_23/while/add/y:output:0*
T0*
_output_shapes
: 2!
sequential_11/lstm_23/while/add
#sequential_11/lstm_23/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2%
#sequential_11/lstm_23/while/add_1/yä
!sequential_11/lstm_23/while/add_1AddV2Dsequential_11_lstm_23_while_sequential_11_lstm_23_while_loop_counter,sequential_11/lstm_23/while/add_1/y:output:0*
T0*
_output_shapes
: 2#
!sequential_11/lstm_23/while/add_1Ã
$sequential_11/lstm_23/while/IdentityIdentity%sequential_11/lstm_23/while/add_1:z:0!^sequential_11/lstm_23/while/NoOp*
T0*
_output_shapes
: 2&
$sequential_11/lstm_23/while/Identityì
&sequential_11/lstm_23/while/Identity_1IdentityJsequential_11_lstm_23_while_sequential_11_lstm_23_while_maximum_iterations!^sequential_11/lstm_23/while/NoOp*
T0*
_output_shapes
: 2(
&sequential_11/lstm_23/while/Identity_1Å
&sequential_11/lstm_23/while/Identity_2Identity#sequential_11/lstm_23/while/add:z:0!^sequential_11/lstm_23/while/NoOp*
T0*
_output_shapes
: 2(
&sequential_11/lstm_23/while/Identity_2ò
&sequential_11/lstm_23/while/Identity_3IdentityPsequential_11/lstm_23/while/TensorArrayV2Write/TensorListSetItem:output_handle:0!^sequential_11/lstm_23/while/NoOp*
T0*
_output_shapes
: 2(
&sequential_11/lstm_23/while/Identity_3æ
&sequential_11/lstm_23/while/Identity_4Identity2sequential_11/lstm_23/while/lstm_cell_23/mul_2:z:0!^sequential_11/lstm_23/while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&sequential_11/lstm_23/while/Identity_4æ
&sequential_11/lstm_23/while/Identity_5Identity2sequential_11/lstm_23/while/lstm_cell_23/add_1:z:0!^sequential_11/lstm_23/while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&sequential_11/lstm_23/while/Identity_5Ì
 sequential_11/lstm_23/while/NoOpNoOp@^sequential_11/lstm_23/while/lstm_cell_23/BiasAdd/ReadVariableOp?^sequential_11/lstm_23/while/lstm_cell_23/MatMul/ReadVariableOpA^sequential_11/lstm_23/while/lstm_cell_23/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2"
 sequential_11/lstm_23/while/NoOp"U
$sequential_11_lstm_23_while_identity-sequential_11/lstm_23/while/Identity:output:0"Y
&sequential_11_lstm_23_while_identity_1/sequential_11/lstm_23/while/Identity_1:output:0"Y
&sequential_11_lstm_23_while_identity_2/sequential_11/lstm_23/while/Identity_2:output:0"Y
&sequential_11_lstm_23_while_identity_3/sequential_11/lstm_23/while/Identity_3:output:0"Y
&sequential_11_lstm_23_while_identity_4/sequential_11/lstm_23/while/Identity_4:output:0"Y
&sequential_11_lstm_23_while_identity_5/sequential_11/lstm_23/while/Identity_5:output:0"
Hsequential_11_lstm_23_while_lstm_cell_23_biasadd_readvariableop_resourceJsequential_11_lstm_23_while_lstm_cell_23_biasadd_readvariableop_resource_0"
Isequential_11_lstm_23_while_lstm_cell_23_matmul_1_readvariableop_resourceKsequential_11_lstm_23_while_lstm_cell_23_matmul_1_readvariableop_resource_0"
Gsequential_11_lstm_23_while_lstm_cell_23_matmul_readvariableop_resourceIsequential_11_lstm_23_while_lstm_cell_23_matmul_readvariableop_resource_0"
Asequential_11_lstm_23_while_sequential_11_lstm_23_strided_slice_1Csequential_11_lstm_23_while_sequential_11_lstm_23_strided_slice_1_0"
}sequential_11_lstm_23_while_tensorarrayv2read_tensorlistgetitem_sequential_11_lstm_23_tensorarrayunstack_tensorlistfromtensorsequential_11_lstm_23_while_tensorarrayv2read_tensorlistgetitem_sequential_11_lstm_23_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2
?sequential_11/lstm_23/while/lstm_cell_23/BiasAdd/ReadVariableOp?sequential_11/lstm_23/while/lstm_cell_23/BiasAdd/ReadVariableOp2
>sequential_11/lstm_23/while/lstm_cell_23/MatMul/ReadVariableOp>sequential_11/lstm_23/while/lstm_cell_23/MatMul/ReadVariableOp2
@sequential_11/lstm_23/while/lstm_cell_23/MatMul_1/ReadVariableOp@sequential_11/lstm_23/while/lstm_cell_23/MatMul_1/ReadVariableOp: 
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
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
åJ
Ô

lstm_22_while_body_39104874,
(lstm_22_while_lstm_22_while_loop_counter2
.lstm_22_while_lstm_22_while_maximum_iterations
lstm_22_while_placeholder
lstm_22_while_placeholder_1
lstm_22_while_placeholder_2
lstm_22_while_placeholder_3+
'lstm_22_while_lstm_22_strided_slice_1_0g
clstm_22_while_tensorarrayv2read_tensorlistgetitem_lstm_22_tensorarrayunstack_tensorlistfromtensor_0N
;lstm_22_while_lstm_cell_22_matmul_readvariableop_resource_0:	]°Q
=lstm_22_while_lstm_cell_22_matmul_1_readvariableop_resource_0:
¬°K
<lstm_22_while_lstm_cell_22_biasadd_readvariableop_resource_0:	°
lstm_22_while_identity
lstm_22_while_identity_1
lstm_22_while_identity_2
lstm_22_while_identity_3
lstm_22_while_identity_4
lstm_22_while_identity_5)
%lstm_22_while_lstm_22_strided_slice_1e
alstm_22_while_tensorarrayv2read_tensorlistgetitem_lstm_22_tensorarrayunstack_tensorlistfromtensorL
9lstm_22_while_lstm_cell_22_matmul_readvariableop_resource:	]°O
;lstm_22_while_lstm_cell_22_matmul_1_readvariableop_resource:
¬°I
:lstm_22_while_lstm_cell_22_biasadd_readvariableop_resource:	°¢1lstm_22/while/lstm_cell_22/BiasAdd/ReadVariableOp¢0lstm_22/while/lstm_cell_22/MatMul/ReadVariableOp¢2lstm_22/while/lstm_cell_22/MatMul_1/ReadVariableOpÓ
?lstm_22/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ]   2A
?lstm_22/while/TensorArrayV2Read/TensorListGetItem/element_shape
1lstm_22/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemclstm_22_while_tensorarrayv2read_tensorlistgetitem_lstm_22_tensorarrayunstack_tensorlistfromtensor_0lstm_22_while_placeholderHlstm_22/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]*
element_dtype023
1lstm_22/while/TensorArrayV2Read/TensorListGetItemá
0lstm_22/while/lstm_cell_22/MatMul/ReadVariableOpReadVariableOp;lstm_22_while_lstm_cell_22_matmul_readvariableop_resource_0*
_output_shapes
:	]°*
dtype022
0lstm_22/while/lstm_cell_22/MatMul/ReadVariableOp÷
!lstm_22/while/lstm_cell_22/MatMulMatMul8lstm_22/while/TensorArrayV2Read/TensorListGetItem:item:08lstm_22/while/lstm_cell_22/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°2#
!lstm_22/while/lstm_cell_22/MatMulè
2lstm_22/while/lstm_cell_22/MatMul_1/ReadVariableOpReadVariableOp=lstm_22_while_lstm_cell_22_matmul_1_readvariableop_resource_0* 
_output_shapes
:
¬°*
dtype024
2lstm_22/while/lstm_cell_22/MatMul_1/ReadVariableOpà
#lstm_22/while/lstm_cell_22/MatMul_1MatMullstm_22_while_placeholder_2:lstm_22/while/lstm_cell_22/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°2%
#lstm_22/while/lstm_cell_22/MatMul_1Ø
lstm_22/while/lstm_cell_22/addAddV2+lstm_22/while/lstm_cell_22/MatMul:product:0-lstm_22/while/lstm_cell_22/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°2 
lstm_22/while/lstm_cell_22/addà
1lstm_22/while/lstm_cell_22/BiasAdd/ReadVariableOpReadVariableOp<lstm_22_while_lstm_cell_22_biasadd_readvariableop_resource_0*
_output_shapes	
:°*
dtype023
1lstm_22/while/lstm_cell_22/BiasAdd/ReadVariableOpå
"lstm_22/while/lstm_cell_22/BiasAddBiasAdd"lstm_22/while/lstm_cell_22/add:z:09lstm_22/while/lstm_cell_22/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°2$
"lstm_22/while/lstm_cell_22/BiasAdd
*lstm_22/while/lstm_cell_22/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2,
*lstm_22/while/lstm_cell_22/split/split_dim¯
 lstm_22/while/lstm_cell_22/splitSplit3lstm_22/while/lstm_cell_22/split/split_dim:output:0+lstm_22/while/lstm_cell_22/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ¬*
	num_split2"
 lstm_22/while/lstm_cell_22/split±
"lstm_22/while/lstm_cell_22/SigmoidSigmoid)lstm_22/while/lstm_cell_22/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2$
"lstm_22/while/lstm_cell_22/Sigmoidµ
$lstm_22/while/lstm_cell_22/Sigmoid_1Sigmoid)lstm_22/while/lstm_cell_22/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2&
$lstm_22/while/lstm_cell_22/Sigmoid_1Á
lstm_22/while/lstm_cell_22/mulMul(lstm_22/while/lstm_cell_22/Sigmoid_1:y:0lstm_22_while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2 
lstm_22/while/lstm_cell_22/mul¨
lstm_22/while/lstm_cell_22/ReluRelu)lstm_22/while/lstm_cell_22/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2!
lstm_22/while/lstm_cell_22/ReluÕ
 lstm_22/while/lstm_cell_22/mul_1Mul&lstm_22/while/lstm_cell_22/Sigmoid:y:0-lstm_22/while/lstm_cell_22/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2"
 lstm_22/while/lstm_cell_22/mul_1Ê
 lstm_22/while/lstm_cell_22/add_1AddV2"lstm_22/while/lstm_cell_22/mul:z:0$lstm_22/while/lstm_cell_22/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2"
 lstm_22/while/lstm_cell_22/add_1µ
$lstm_22/while/lstm_cell_22/Sigmoid_2Sigmoid)lstm_22/while/lstm_cell_22/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2&
$lstm_22/while/lstm_cell_22/Sigmoid_2§
!lstm_22/while/lstm_cell_22/Relu_1Relu$lstm_22/while/lstm_cell_22/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2#
!lstm_22/while/lstm_cell_22/Relu_1Ù
 lstm_22/while/lstm_cell_22/mul_2Mul(lstm_22/while/lstm_cell_22/Sigmoid_2:y:0/lstm_22/while/lstm_cell_22/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2"
 lstm_22/while/lstm_cell_22/mul_2
2lstm_22/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_22_while_placeholder_1lstm_22_while_placeholder$lstm_22/while/lstm_cell_22/mul_2:z:0*
_output_shapes
: *
element_dtype024
2lstm_22/while/TensorArrayV2Write/TensorListSetIteml
lstm_22/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_22/while/add/y
lstm_22/while/addAddV2lstm_22_while_placeholderlstm_22/while/add/y:output:0*
T0*
_output_shapes
: 2
lstm_22/while/addp
lstm_22/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_22/while/add_1/y
lstm_22/while/add_1AddV2(lstm_22_while_lstm_22_while_loop_counterlstm_22/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lstm_22/while/add_1
lstm_22/while/IdentityIdentitylstm_22/while/add_1:z:0^lstm_22/while/NoOp*
T0*
_output_shapes
: 2
lstm_22/while/Identity¦
lstm_22/while/Identity_1Identity.lstm_22_while_lstm_22_while_maximum_iterations^lstm_22/while/NoOp*
T0*
_output_shapes
: 2
lstm_22/while/Identity_1
lstm_22/while/Identity_2Identitylstm_22/while/add:z:0^lstm_22/while/NoOp*
T0*
_output_shapes
: 2
lstm_22/while/Identity_2º
lstm_22/while/Identity_3IdentityBlstm_22/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_22/while/NoOp*
T0*
_output_shapes
: 2
lstm_22/while/Identity_3®
lstm_22/while/Identity_4Identity$lstm_22/while/lstm_cell_22/mul_2:z:0^lstm_22/while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
lstm_22/while/Identity_4®
lstm_22/while/Identity_5Identity$lstm_22/while/lstm_cell_22/add_1:z:0^lstm_22/while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
lstm_22/while/Identity_5
lstm_22/while/NoOpNoOp2^lstm_22/while/lstm_cell_22/BiasAdd/ReadVariableOp1^lstm_22/while/lstm_cell_22/MatMul/ReadVariableOp3^lstm_22/while/lstm_cell_22/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
lstm_22/while/NoOp"9
lstm_22_while_identitylstm_22/while/Identity:output:0"=
lstm_22_while_identity_1!lstm_22/while/Identity_1:output:0"=
lstm_22_while_identity_2!lstm_22/while/Identity_2:output:0"=
lstm_22_while_identity_3!lstm_22/while/Identity_3:output:0"=
lstm_22_while_identity_4!lstm_22/while/Identity_4:output:0"=
lstm_22_while_identity_5!lstm_22/while/Identity_5:output:0"P
%lstm_22_while_lstm_22_strided_slice_1'lstm_22_while_lstm_22_strided_slice_1_0"z
:lstm_22_while_lstm_cell_22_biasadd_readvariableop_resource<lstm_22_while_lstm_cell_22_biasadd_readvariableop_resource_0"|
;lstm_22_while_lstm_cell_22_matmul_1_readvariableop_resource=lstm_22_while_lstm_cell_22_matmul_1_readvariableop_resource_0"x
9lstm_22_while_lstm_cell_22_matmul_readvariableop_resource;lstm_22_while_lstm_cell_22_matmul_readvariableop_resource_0"È
alstm_22_while_tensorarrayv2read_tensorlistgetitem_lstm_22_tensorarrayunstack_tensorlistfromtensorclstm_22_while_tensorarrayv2read_tensorlistgetitem_lstm_22_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ¬: : : : : 2f
1lstm_22/while/lstm_cell_22/BiasAdd/ReadVariableOp1lstm_22/while/lstm_cell_22/BiasAdd/ReadVariableOp2d
0lstm_22/while/lstm_cell_22/MatMul/ReadVariableOp0lstm_22/while/lstm_cell_22/MatMul/ReadVariableOp2h
2lstm_22/while/lstm_cell_22/MatMul_1/ReadVariableOp2lstm_22/while/lstm_cell_22/MatMul_1/ReadVariableOp: 
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
:ÿÿÿÿÿÿÿÿÿ¬:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬:

_output_shapes
: :

_output_shapes
: 
ÐF

E__inference_lstm_23_layer_call_and_return_conditional_losses_39103336

inputs)
lstm_cell_23_39103254:
¬)
lstm_cell_23_39103256:
$
lstm_cell_23_39103258:	
identity¢$lstm_cell_23/StatefulPartitionedCall¢whileD
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
strided_slice/stack_2â
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
B :2
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
B :è2
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
B :2
zeros/packed/1
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
:ÿÿÿÿÿÿÿÿÿ2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :2
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
B :è2
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
B :2
zeros_1/packed/1
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
:ÿÿÿÿÿÿÿÿÿ2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputstranspose/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬2
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
strided_slice_1/stack_2î
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
TensorArrayV2/element_shape²
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2¿
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ¬   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeø
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
strided_slice_2/stack_2ý
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬*
shrink_axis_mask2
strided_slice_2¨
$lstm_cell_23/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_23_39103254lstm_cell_23_39103256lstm_cell_23_39103258*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_lstm_cell_23_layer_call_and_return_conditional_losses_391032532&
$lstm_cell_23/StatefulPartitionedCall
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
TensorArrayV2_1/element_shape¸
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
ÿÿÿÿÿÿÿÿÿ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterÎ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_23_39103254lstm_cell_23_39103256lstm_cell_23_39103258*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_39103267*
condR
while_cond_39103266*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   22
0TensorArrayV2Stack/TensorListStack/element_shapeò
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
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
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm¯
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
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
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity}
NoOpNoOp%^lstm_cell_23/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬: : : 2L
$lstm_cell_23/StatefulPartitionedCall$lstm_cell_23/StatefulPartitionedCall2
whilewhile:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬
 
_user_specified_nameinputs
Ø
I
-__inference_dropout_23_layer_call_fn_39106803

inputs
identityË
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_dropout_23_layer_call_and_return_conditional_losses_391041432
PartitionedCallq
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
×
g
H__inference_dropout_23_layer_call_and_return_conditional_losses_39106825

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/Constx
dropout/MulMulinputsdropout/Const:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape¹
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>2
dropout/GreaterEqual/yÃ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/Mul_1j
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ä^

)sequential_11_lstm_22_while_body_39102288H
Dsequential_11_lstm_22_while_sequential_11_lstm_22_while_loop_counterN
Jsequential_11_lstm_22_while_sequential_11_lstm_22_while_maximum_iterations+
'sequential_11_lstm_22_while_placeholder-
)sequential_11_lstm_22_while_placeholder_1-
)sequential_11_lstm_22_while_placeholder_2-
)sequential_11_lstm_22_while_placeholder_3G
Csequential_11_lstm_22_while_sequential_11_lstm_22_strided_slice_1_0
sequential_11_lstm_22_while_tensorarrayv2read_tensorlistgetitem_sequential_11_lstm_22_tensorarrayunstack_tensorlistfromtensor_0\
Isequential_11_lstm_22_while_lstm_cell_22_matmul_readvariableop_resource_0:	]°_
Ksequential_11_lstm_22_while_lstm_cell_22_matmul_1_readvariableop_resource_0:
¬°Y
Jsequential_11_lstm_22_while_lstm_cell_22_biasadd_readvariableop_resource_0:	°(
$sequential_11_lstm_22_while_identity*
&sequential_11_lstm_22_while_identity_1*
&sequential_11_lstm_22_while_identity_2*
&sequential_11_lstm_22_while_identity_3*
&sequential_11_lstm_22_while_identity_4*
&sequential_11_lstm_22_while_identity_5E
Asequential_11_lstm_22_while_sequential_11_lstm_22_strided_slice_1
}sequential_11_lstm_22_while_tensorarrayv2read_tensorlistgetitem_sequential_11_lstm_22_tensorarrayunstack_tensorlistfromtensorZ
Gsequential_11_lstm_22_while_lstm_cell_22_matmul_readvariableop_resource:	]°]
Isequential_11_lstm_22_while_lstm_cell_22_matmul_1_readvariableop_resource:
¬°W
Hsequential_11_lstm_22_while_lstm_cell_22_biasadd_readvariableop_resource:	°¢?sequential_11/lstm_22/while/lstm_cell_22/BiasAdd/ReadVariableOp¢>sequential_11/lstm_22/while/lstm_cell_22/MatMul/ReadVariableOp¢@sequential_11/lstm_22/while/lstm_cell_22/MatMul_1/ReadVariableOpï
Msequential_11/lstm_22/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ]   2O
Msequential_11/lstm_22/while/TensorArrayV2Read/TensorListGetItem/element_shape×
?sequential_11/lstm_22/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemsequential_11_lstm_22_while_tensorarrayv2read_tensorlistgetitem_sequential_11_lstm_22_tensorarrayunstack_tensorlistfromtensor_0'sequential_11_lstm_22_while_placeholderVsequential_11/lstm_22/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]*
element_dtype02A
?sequential_11/lstm_22/while/TensorArrayV2Read/TensorListGetItem
>sequential_11/lstm_22/while/lstm_cell_22/MatMul/ReadVariableOpReadVariableOpIsequential_11_lstm_22_while_lstm_cell_22_matmul_readvariableop_resource_0*
_output_shapes
:	]°*
dtype02@
>sequential_11/lstm_22/while/lstm_cell_22/MatMul/ReadVariableOp¯
/sequential_11/lstm_22/while/lstm_cell_22/MatMulMatMulFsequential_11/lstm_22/while/TensorArrayV2Read/TensorListGetItem:item:0Fsequential_11/lstm_22/while/lstm_cell_22/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°21
/sequential_11/lstm_22/while/lstm_cell_22/MatMul
@sequential_11/lstm_22/while/lstm_cell_22/MatMul_1/ReadVariableOpReadVariableOpKsequential_11_lstm_22_while_lstm_cell_22_matmul_1_readvariableop_resource_0* 
_output_shapes
:
¬°*
dtype02B
@sequential_11/lstm_22/while/lstm_cell_22/MatMul_1/ReadVariableOp
1sequential_11/lstm_22/while/lstm_cell_22/MatMul_1MatMul)sequential_11_lstm_22_while_placeholder_2Hsequential_11/lstm_22/while/lstm_cell_22/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°23
1sequential_11/lstm_22/while/lstm_cell_22/MatMul_1
,sequential_11/lstm_22/while/lstm_cell_22/addAddV29sequential_11/lstm_22/while/lstm_cell_22/MatMul:product:0;sequential_11/lstm_22/while/lstm_cell_22/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°2.
,sequential_11/lstm_22/while/lstm_cell_22/add
?sequential_11/lstm_22/while/lstm_cell_22/BiasAdd/ReadVariableOpReadVariableOpJsequential_11_lstm_22_while_lstm_cell_22_biasadd_readvariableop_resource_0*
_output_shapes	
:°*
dtype02A
?sequential_11/lstm_22/while/lstm_cell_22/BiasAdd/ReadVariableOp
0sequential_11/lstm_22/while/lstm_cell_22/BiasAddBiasAdd0sequential_11/lstm_22/while/lstm_cell_22/add:z:0Gsequential_11/lstm_22/while/lstm_cell_22/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°22
0sequential_11/lstm_22/while/lstm_cell_22/BiasAdd¶
8sequential_11/lstm_22/while/lstm_cell_22/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2:
8sequential_11/lstm_22/while/lstm_cell_22/split/split_dimç
.sequential_11/lstm_22/while/lstm_cell_22/splitSplitAsequential_11/lstm_22/while/lstm_cell_22/split/split_dim:output:09sequential_11/lstm_22/while/lstm_cell_22/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ¬*
	num_split20
.sequential_11/lstm_22/while/lstm_cell_22/splitÛ
0sequential_11/lstm_22/while/lstm_cell_22/SigmoidSigmoid7sequential_11/lstm_22/while/lstm_cell_22/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬22
0sequential_11/lstm_22/while/lstm_cell_22/Sigmoidß
2sequential_11/lstm_22/while/lstm_cell_22/Sigmoid_1Sigmoid7sequential_11/lstm_22/while/lstm_cell_22/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬24
2sequential_11/lstm_22/while/lstm_cell_22/Sigmoid_1ù
,sequential_11/lstm_22/while/lstm_cell_22/mulMul6sequential_11/lstm_22/while/lstm_cell_22/Sigmoid_1:y:0)sequential_11_lstm_22_while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2.
,sequential_11/lstm_22/while/lstm_cell_22/mulÒ
-sequential_11/lstm_22/while/lstm_cell_22/ReluRelu7sequential_11/lstm_22/while/lstm_cell_22/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2/
-sequential_11/lstm_22/while/lstm_cell_22/Relu
.sequential_11/lstm_22/while/lstm_cell_22/mul_1Mul4sequential_11/lstm_22/while/lstm_cell_22/Sigmoid:y:0;sequential_11/lstm_22/while/lstm_cell_22/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬20
.sequential_11/lstm_22/while/lstm_cell_22/mul_1
.sequential_11/lstm_22/while/lstm_cell_22/add_1AddV20sequential_11/lstm_22/while/lstm_cell_22/mul:z:02sequential_11/lstm_22/while/lstm_cell_22/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬20
.sequential_11/lstm_22/while/lstm_cell_22/add_1ß
2sequential_11/lstm_22/while/lstm_cell_22/Sigmoid_2Sigmoid7sequential_11/lstm_22/while/lstm_cell_22/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬24
2sequential_11/lstm_22/while/lstm_cell_22/Sigmoid_2Ñ
/sequential_11/lstm_22/while/lstm_cell_22/Relu_1Relu2sequential_11/lstm_22/while/lstm_cell_22/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬21
/sequential_11/lstm_22/while/lstm_cell_22/Relu_1
.sequential_11/lstm_22/while/lstm_cell_22/mul_2Mul6sequential_11/lstm_22/while/lstm_cell_22/Sigmoid_2:y:0=sequential_11/lstm_22/while/lstm_cell_22/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬20
.sequential_11/lstm_22/while/lstm_cell_22/mul_2Î
@sequential_11/lstm_22/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem)sequential_11_lstm_22_while_placeholder_1'sequential_11_lstm_22_while_placeholder2sequential_11/lstm_22/while/lstm_cell_22/mul_2:z:0*
_output_shapes
: *
element_dtype02B
@sequential_11/lstm_22/while/TensorArrayV2Write/TensorListSetItem
!sequential_11/lstm_22/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2#
!sequential_11/lstm_22/while/add/yÁ
sequential_11/lstm_22/while/addAddV2'sequential_11_lstm_22_while_placeholder*sequential_11/lstm_22/while/add/y:output:0*
T0*
_output_shapes
: 2!
sequential_11/lstm_22/while/add
#sequential_11/lstm_22/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2%
#sequential_11/lstm_22/while/add_1/yä
!sequential_11/lstm_22/while/add_1AddV2Dsequential_11_lstm_22_while_sequential_11_lstm_22_while_loop_counter,sequential_11/lstm_22/while/add_1/y:output:0*
T0*
_output_shapes
: 2#
!sequential_11/lstm_22/while/add_1Ã
$sequential_11/lstm_22/while/IdentityIdentity%sequential_11/lstm_22/while/add_1:z:0!^sequential_11/lstm_22/while/NoOp*
T0*
_output_shapes
: 2&
$sequential_11/lstm_22/while/Identityì
&sequential_11/lstm_22/while/Identity_1IdentityJsequential_11_lstm_22_while_sequential_11_lstm_22_while_maximum_iterations!^sequential_11/lstm_22/while/NoOp*
T0*
_output_shapes
: 2(
&sequential_11/lstm_22/while/Identity_1Å
&sequential_11/lstm_22/while/Identity_2Identity#sequential_11/lstm_22/while/add:z:0!^sequential_11/lstm_22/while/NoOp*
T0*
_output_shapes
: 2(
&sequential_11/lstm_22/while/Identity_2ò
&sequential_11/lstm_22/while/Identity_3IdentityPsequential_11/lstm_22/while/TensorArrayV2Write/TensorListSetItem:output_handle:0!^sequential_11/lstm_22/while/NoOp*
T0*
_output_shapes
: 2(
&sequential_11/lstm_22/while/Identity_3æ
&sequential_11/lstm_22/while/Identity_4Identity2sequential_11/lstm_22/while/lstm_cell_22/mul_2:z:0!^sequential_11/lstm_22/while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2(
&sequential_11/lstm_22/while/Identity_4æ
&sequential_11/lstm_22/while/Identity_5Identity2sequential_11/lstm_22/while/lstm_cell_22/add_1:z:0!^sequential_11/lstm_22/while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2(
&sequential_11/lstm_22/while/Identity_5Ì
 sequential_11/lstm_22/while/NoOpNoOp@^sequential_11/lstm_22/while/lstm_cell_22/BiasAdd/ReadVariableOp?^sequential_11/lstm_22/while/lstm_cell_22/MatMul/ReadVariableOpA^sequential_11/lstm_22/while/lstm_cell_22/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2"
 sequential_11/lstm_22/while/NoOp"U
$sequential_11_lstm_22_while_identity-sequential_11/lstm_22/while/Identity:output:0"Y
&sequential_11_lstm_22_while_identity_1/sequential_11/lstm_22/while/Identity_1:output:0"Y
&sequential_11_lstm_22_while_identity_2/sequential_11/lstm_22/while/Identity_2:output:0"Y
&sequential_11_lstm_22_while_identity_3/sequential_11/lstm_22/while/Identity_3:output:0"Y
&sequential_11_lstm_22_while_identity_4/sequential_11/lstm_22/while/Identity_4:output:0"Y
&sequential_11_lstm_22_while_identity_5/sequential_11/lstm_22/while/Identity_5:output:0"
Hsequential_11_lstm_22_while_lstm_cell_22_biasadd_readvariableop_resourceJsequential_11_lstm_22_while_lstm_cell_22_biasadd_readvariableop_resource_0"
Isequential_11_lstm_22_while_lstm_cell_22_matmul_1_readvariableop_resourceKsequential_11_lstm_22_while_lstm_cell_22_matmul_1_readvariableop_resource_0"
Gsequential_11_lstm_22_while_lstm_cell_22_matmul_readvariableop_resourceIsequential_11_lstm_22_while_lstm_cell_22_matmul_readvariableop_resource_0"
Asequential_11_lstm_22_while_sequential_11_lstm_22_strided_slice_1Csequential_11_lstm_22_while_sequential_11_lstm_22_strided_slice_1_0"
}sequential_11_lstm_22_while_tensorarrayv2read_tensorlistgetitem_sequential_11_lstm_22_tensorarrayunstack_tensorlistfromtensorsequential_11_lstm_22_while_tensorarrayv2read_tensorlistgetitem_sequential_11_lstm_22_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ¬: : : : : 2
?sequential_11/lstm_22/while/lstm_cell_22/BiasAdd/ReadVariableOp?sequential_11/lstm_22/while/lstm_cell_22/BiasAdd/ReadVariableOp2
>sequential_11/lstm_22/while/lstm_cell_22/MatMul/ReadVariableOp>sequential_11/lstm_22/while/lstm_cell_22/MatMul/ReadVariableOp2
@sequential_11/lstm_22/while/lstm_cell_22/MatMul_1/ReadVariableOp@sequential_11/lstm_22/while/lstm_cell_22/MatMul_1/ReadVariableOp: 
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
:ÿÿÿÿÿÿÿÿÿ¬:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬:

_output_shapes
: :

_output_shapes
: 
åJ
Ô

lstm_22_while_body_39105201,
(lstm_22_while_lstm_22_while_loop_counter2
.lstm_22_while_lstm_22_while_maximum_iterations
lstm_22_while_placeholder
lstm_22_while_placeholder_1
lstm_22_while_placeholder_2
lstm_22_while_placeholder_3+
'lstm_22_while_lstm_22_strided_slice_1_0g
clstm_22_while_tensorarrayv2read_tensorlistgetitem_lstm_22_tensorarrayunstack_tensorlistfromtensor_0N
;lstm_22_while_lstm_cell_22_matmul_readvariableop_resource_0:	]°Q
=lstm_22_while_lstm_cell_22_matmul_1_readvariableop_resource_0:
¬°K
<lstm_22_while_lstm_cell_22_biasadd_readvariableop_resource_0:	°
lstm_22_while_identity
lstm_22_while_identity_1
lstm_22_while_identity_2
lstm_22_while_identity_3
lstm_22_while_identity_4
lstm_22_while_identity_5)
%lstm_22_while_lstm_22_strided_slice_1e
alstm_22_while_tensorarrayv2read_tensorlistgetitem_lstm_22_tensorarrayunstack_tensorlistfromtensorL
9lstm_22_while_lstm_cell_22_matmul_readvariableop_resource:	]°O
;lstm_22_while_lstm_cell_22_matmul_1_readvariableop_resource:
¬°I
:lstm_22_while_lstm_cell_22_biasadd_readvariableop_resource:	°¢1lstm_22/while/lstm_cell_22/BiasAdd/ReadVariableOp¢0lstm_22/while/lstm_cell_22/MatMul/ReadVariableOp¢2lstm_22/while/lstm_cell_22/MatMul_1/ReadVariableOpÓ
?lstm_22/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ]   2A
?lstm_22/while/TensorArrayV2Read/TensorListGetItem/element_shape
1lstm_22/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemclstm_22_while_tensorarrayv2read_tensorlistgetitem_lstm_22_tensorarrayunstack_tensorlistfromtensor_0lstm_22_while_placeholderHlstm_22/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]*
element_dtype023
1lstm_22/while/TensorArrayV2Read/TensorListGetItemá
0lstm_22/while/lstm_cell_22/MatMul/ReadVariableOpReadVariableOp;lstm_22_while_lstm_cell_22_matmul_readvariableop_resource_0*
_output_shapes
:	]°*
dtype022
0lstm_22/while/lstm_cell_22/MatMul/ReadVariableOp÷
!lstm_22/while/lstm_cell_22/MatMulMatMul8lstm_22/while/TensorArrayV2Read/TensorListGetItem:item:08lstm_22/while/lstm_cell_22/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°2#
!lstm_22/while/lstm_cell_22/MatMulè
2lstm_22/while/lstm_cell_22/MatMul_1/ReadVariableOpReadVariableOp=lstm_22_while_lstm_cell_22_matmul_1_readvariableop_resource_0* 
_output_shapes
:
¬°*
dtype024
2lstm_22/while/lstm_cell_22/MatMul_1/ReadVariableOpà
#lstm_22/while/lstm_cell_22/MatMul_1MatMullstm_22_while_placeholder_2:lstm_22/while/lstm_cell_22/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°2%
#lstm_22/while/lstm_cell_22/MatMul_1Ø
lstm_22/while/lstm_cell_22/addAddV2+lstm_22/while/lstm_cell_22/MatMul:product:0-lstm_22/while/lstm_cell_22/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°2 
lstm_22/while/lstm_cell_22/addà
1lstm_22/while/lstm_cell_22/BiasAdd/ReadVariableOpReadVariableOp<lstm_22_while_lstm_cell_22_biasadd_readvariableop_resource_0*
_output_shapes	
:°*
dtype023
1lstm_22/while/lstm_cell_22/BiasAdd/ReadVariableOpå
"lstm_22/while/lstm_cell_22/BiasAddBiasAdd"lstm_22/while/lstm_cell_22/add:z:09lstm_22/while/lstm_cell_22/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°2$
"lstm_22/while/lstm_cell_22/BiasAdd
*lstm_22/while/lstm_cell_22/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2,
*lstm_22/while/lstm_cell_22/split/split_dim¯
 lstm_22/while/lstm_cell_22/splitSplit3lstm_22/while/lstm_cell_22/split/split_dim:output:0+lstm_22/while/lstm_cell_22/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ¬*
	num_split2"
 lstm_22/while/lstm_cell_22/split±
"lstm_22/while/lstm_cell_22/SigmoidSigmoid)lstm_22/while/lstm_cell_22/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2$
"lstm_22/while/lstm_cell_22/Sigmoidµ
$lstm_22/while/lstm_cell_22/Sigmoid_1Sigmoid)lstm_22/while/lstm_cell_22/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2&
$lstm_22/while/lstm_cell_22/Sigmoid_1Á
lstm_22/while/lstm_cell_22/mulMul(lstm_22/while/lstm_cell_22/Sigmoid_1:y:0lstm_22_while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2 
lstm_22/while/lstm_cell_22/mul¨
lstm_22/while/lstm_cell_22/ReluRelu)lstm_22/while/lstm_cell_22/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2!
lstm_22/while/lstm_cell_22/ReluÕ
 lstm_22/while/lstm_cell_22/mul_1Mul&lstm_22/while/lstm_cell_22/Sigmoid:y:0-lstm_22/while/lstm_cell_22/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2"
 lstm_22/while/lstm_cell_22/mul_1Ê
 lstm_22/while/lstm_cell_22/add_1AddV2"lstm_22/while/lstm_cell_22/mul:z:0$lstm_22/while/lstm_cell_22/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2"
 lstm_22/while/lstm_cell_22/add_1µ
$lstm_22/while/lstm_cell_22/Sigmoid_2Sigmoid)lstm_22/while/lstm_cell_22/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2&
$lstm_22/while/lstm_cell_22/Sigmoid_2§
!lstm_22/while/lstm_cell_22/Relu_1Relu$lstm_22/while/lstm_cell_22/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2#
!lstm_22/while/lstm_cell_22/Relu_1Ù
 lstm_22/while/lstm_cell_22/mul_2Mul(lstm_22/while/lstm_cell_22/Sigmoid_2:y:0/lstm_22/while/lstm_cell_22/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2"
 lstm_22/while/lstm_cell_22/mul_2
2lstm_22/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_22_while_placeholder_1lstm_22_while_placeholder$lstm_22/while/lstm_cell_22/mul_2:z:0*
_output_shapes
: *
element_dtype024
2lstm_22/while/TensorArrayV2Write/TensorListSetIteml
lstm_22/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_22/while/add/y
lstm_22/while/addAddV2lstm_22_while_placeholderlstm_22/while/add/y:output:0*
T0*
_output_shapes
: 2
lstm_22/while/addp
lstm_22/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_22/while/add_1/y
lstm_22/while/add_1AddV2(lstm_22_while_lstm_22_while_loop_counterlstm_22/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lstm_22/while/add_1
lstm_22/while/IdentityIdentitylstm_22/while/add_1:z:0^lstm_22/while/NoOp*
T0*
_output_shapes
: 2
lstm_22/while/Identity¦
lstm_22/while/Identity_1Identity.lstm_22_while_lstm_22_while_maximum_iterations^lstm_22/while/NoOp*
T0*
_output_shapes
: 2
lstm_22/while/Identity_1
lstm_22/while/Identity_2Identitylstm_22/while/add:z:0^lstm_22/while/NoOp*
T0*
_output_shapes
: 2
lstm_22/while/Identity_2º
lstm_22/while/Identity_3IdentityBlstm_22/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_22/while/NoOp*
T0*
_output_shapes
: 2
lstm_22/while/Identity_3®
lstm_22/while/Identity_4Identity$lstm_22/while/lstm_cell_22/mul_2:z:0^lstm_22/while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
lstm_22/while/Identity_4®
lstm_22/while/Identity_5Identity$lstm_22/while/lstm_cell_22/add_1:z:0^lstm_22/while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
lstm_22/while/Identity_5
lstm_22/while/NoOpNoOp2^lstm_22/while/lstm_cell_22/BiasAdd/ReadVariableOp1^lstm_22/while/lstm_cell_22/MatMul/ReadVariableOp3^lstm_22/while/lstm_cell_22/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
lstm_22/while/NoOp"9
lstm_22_while_identitylstm_22/while/Identity:output:0"=
lstm_22_while_identity_1!lstm_22/while/Identity_1:output:0"=
lstm_22_while_identity_2!lstm_22/while/Identity_2:output:0"=
lstm_22_while_identity_3!lstm_22/while/Identity_3:output:0"=
lstm_22_while_identity_4!lstm_22/while/Identity_4:output:0"=
lstm_22_while_identity_5!lstm_22/while/Identity_5:output:0"P
%lstm_22_while_lstm_22_strided_slice_1'lstm_22_while_lstm_22_strided_slice_1_0"z
:lstm_22_while_lstm_cell_22_biasadd_readvariableop_resource<lstm_22_while_lstm_cell_22_biasadd_readvariableop_resource_0"|
;lstm_22_while_lstm_cell_22_matmul_1_readvariableop_resource=lstm_22_while_lstm_cell_22_matmul_1_readvariableop_resource_0"x
9lstm_22_while_lstm_cell_22_matmul_readvariableop_resource;lstm_22_while_lstm_cell_22_matmul_readvariableop_resource_0"È
alstm_22_while_tensorarrayv2read_tensorlistgetitem_lstm_22_tensorarrayunstack_tensorlistfromtensorclstm_22_while_tensorarrayv2read_tensorlistgetitem_lstm_22_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ¬: : : : : 2f
1lstm_22/while/lstm_cell_22/BiasAdd/ReadVariableOp1lstm_22/while/lstm_cell_22/BiasAdd/ReadVariableOp2d
0lstm_22/while/lstm_cell_22/MatMul/ReadVariableOp0lstm_22/while/lstm_cell_22/MatMul/ReadVariableOp2h
2lstm_22/while/lstm_cell_22/MatMul_1/ReadVariableOp2lstm_22/while/lstm_cell_22/MatMul_1/ReadVariableOp: 
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
:ÿÿÿÿÿÿÿÿÿ¬:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬:

_output_shapes
: :

_output_shapes
: 
à
º
*__inference_lstm_22_layer_call_fn_39105486
inputs_0
unknown:	]°
	unknown_0:
¬°
	unknown_1:	°
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_lstm_22_layer_call_and_return_conditional_losses_391027062
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ]: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ]
"
_user_specified_name
inputs/0
\

E__inference_lstm_23_layer_call_and_return_conditional_losses_39106798

inputs?
+lstm_cell_23_matmul_readvariableop_resource:
¬A
-lstm_cell_23_matmul_1_readvariableop_resource:
;
,lstm_cell_23_biasadd_readvariableop_resource:	
identity¢#lstm_cell_23/BiasAdd/ReadVariableOp¢"lstm_cell_23/MatMul/ReadVariableOp¢$lstm_cell_23/MatMul_1/ReadVariableOp¢whileD
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
strided_slice/stack_2â
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
B :2
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
B :è2
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
B :2
zeros/packed/1
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
:ÿÿÿÿÿÿÿÿÿ2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :2
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
B :è2
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
B :2
zeros_1/packed/1
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
:ÿÿÿÿÿÿÿÿÿ2	
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
:ÿÿÿÿÿÿÿÿÿ¬2
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
strided_slice_1/stack_2î
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
TensorArrayV2/element_shape²
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2¿
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ¬   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeø
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
strided_slice_2/stack_2ý
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬*
shrink_axis_mask2
strided_slice_2¶
"lstm_cell_23/MatMul/ReadVariableOpReadVariableOp+lstm_cell_23_matmul_readvariableop_resource* 
_output_shapes
:
¬*
dtype02$
"lstm_cell_23/MatMul/ReadVariableOp­
lstm_cell_23/MatMulMatMulstrided_slice_2:output:0*lstm_cell_23/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_23/MatMul¼
$lstm_cell_23/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_23_matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype02&
$lstm_cell_23/MatMul_1/ReadVariableOp©
lstm_cell_23/MatMul_1MatMulzeros:output:0,lstm_cell_23/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_23/MatMul_1 
lstm_cell_23/addAddV2lstm_cell_23/MatMul:product:0lstm_cell_23/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_23/add´
#lstm_cell_23/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_23_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02%
#lstm_cell_23/BiasAdd/ReadVariableOp­
lstm_cell_23/BiasAddBiasAddlstm_cell_23/add:z:0+lstm_cell_23/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_23/BiasAdd~
lstm_cell_23/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_23/split/split_dim÷
lstm_cell_23/splitSplit%lstm_cell_23/split/split_dim:output:0lstm_cell_23/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2
lstm_cell_23/split
lstm_cell_23/SigmoidSigmoidlstm_cell_23/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_23/Sigmoid
lstm_cell_23/Sigmoid_1Sigmoidlstm_cell_23/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_23/Sigmoid_1
lstm_cell_23/mulMullstm_cell_23/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_23/mul~
lstm_cell_23/ReluRelulstm_cell_23/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_23/Relu
lstm_cell_23/mul_1Mullstm_cell_23/Sigmoid:y:0lstm_cell_23/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_23/mul_1
lstm_cell_23/add_1AddV2lstm_cell_23/mul:z:0lstm_cell_23/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_23/add_1
lstm_cell_23/Sigmoid_2Sigmoidlstm_cell_23/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_23/Sigmoid_2}
lstm_cell_23/Relu_1Relulstm_cell_23/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_23/Relu_1¡
lstm_cell_23/mul_2Mullstm_cell_23/Sigmoid_2:y:0!lstm_cell_23/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_23/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
TensorArrayV2_1/element_shape¸
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
ÿÿÿÿÿÿÿÿÿ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_23_matmul_readvariableop_resource-lstm_cell_23_matmul_1_readvariableop_resource,lstm_cell_23_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_39106714*
condR
while_cond_39106713*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   22
0TensorArrayV2Stack/TensorListStack/element_shapeé
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
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
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm¦
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
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
:ÿÿÿÿÿÿÿÿÿ2

IdentityÈ
NoOpNoOp$^lstm_cell_23/BiasAdd/ReadVariableOp#^lstm_cell_23/MatMul/ReadVariableOp%^lstm_cell_23/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:ÿÿÿÿÿÿÿÿÿ¬: : : 2J
#lstm_cell_23/BiasAdd/ReadVariableOp#lstm_cell_23/BiasAdd/ReadVariableOp2H
"lstm_cell_23/MatMul/ReadVariableOp"lstm_cell_23/MatMul/ReadVariableOp2L
$lstm_cell_23/MatMul_1/ReadVariableOp$lstm_cell_23/MatMul_1/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
 
_user_specified_nameinputs


Í
0__inference_sequential_11_layer_call_fn_39104786

inputs
unknown:	]°
	unknown_0:
¬°
	unknown_1:	°
	unknown_2:
¬
	unknown_3:

	unknown_4:	
	unknown_5:	
	unknown_6:
identity¢StatefulPartitionedCallÍ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_sequential_11_layer_call_and_return_conditional_losses_391041832
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ]: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
 
_user_specified_nameinputs
´?
Ö
while_body_39104046
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
3while_lstm_cell_23_matmul_readvariableop_resource_0:
¬I
5while_lstm_cell_23_matmul_1_readvariableop_resource_0:
C
4while_lstm_cell_23_biasadd_readvariableop_resource_0:	
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
1while_lstm_cell_23_matmul_readvariableop_resource:
¬G
3while_lstm_cell_23_matmul_1_readvariableop_resource:
A
2while_lstm_cell_23_biasadd_readvariableop_resource:	¢)while/lstm_cell_23/BiasAdd/ReadVariableOp¢(while/lstm_cell_23/MatMul/ReadVariableOp¢*while/lstm_cell_23/MatMul_1/ReadVariableOpÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ¬   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÔ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemÊ
(while/lstm_cell_23/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_23_matmul_readvariableop_resource_0* 
_output_shapes
:
¬*
dtype02*
(while/lstm_cell_23/MatMul/ReadVariableOp×
while/lstm_cell_23/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_23/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_23/MatMulÐ
*while/lstm_cell_23/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_23_matmul_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype02,
*while/lstm_cell_23/MatMul_1/ReadVariableOpÀ
while/lstm_cell_23/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_23/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_23/MatMul_1¸
while/lstm_cell_23/addAddV2#while/lstm_cell_23/MatMul:product:0%while/lstm_cell_23/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_23/addÈ
)while/lstm_cell_23/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_23_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02+
)while/lstm_cell_23/BiasAdd/ReadVariableOpÅ
while/lstm_cell_23/BiasAddBiasAddwhile/lstm_cell_23/add:z:01while/lstm_cell_23/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_23/BiasAdd
"while/lstm_cell_23/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_23/split/split_dim
while/lstm_cell_23/splitSplit+while/lstm_cell_23/split/split_dim:output:0#while/lstm_cell_23/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2
while/lstm_cell_23/split
while/lstm_cell_23/SigmoidSigmoid!while/lstm_cell_23/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_23/Sigmoid
while/lstm_cell_23/Sigmoid_1Sigmoid!while/lstm_cell_23/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_23/Sigmoid_1¡
while/lstm_cell_23/mulMul while/lstm_cell_23/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_23/mul
while/lstm_cell_23/ReluRelu!while/lstm_cell_23/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_23/Reluµ
while/lstm_cell_23/mul_1Mulwhile/lstm_cell_23/Sigmoid:y:0%while/lstm_cell_23/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_23/mul_1ª
while/lstm_cell_23/add_1AddV2while/lstm_cell_23/mul:z:0while/lstm_cell_23/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_23/add_1
while/lstm_cell_23/Sigmoid_2Sigmoid!while/lstm_cell_23/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_23/Sigmoid_2
while/lstm_cell_23/Relu_1Reluwhile/lstm_cell_23/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_23/Relu_1¹
while/lstm_cell_23/mul_2Mul while/lstm_cell_23/Sigmoid_2:y:0'while/lstm_cell_23/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_23/mul_2à
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_23/mul_2:z:0*
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
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_23/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_23/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/Identity_5Þ

while/NoOpNoOp*^while/lstm_cell_23/BiasAdd/ReadVariableOp)^while/lstm_cell_23/MatMul/ReadVariableOp+^while/lstm_cell_23/MatMul_1/ReadVariableOp*"
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
2while_lstm_cell_23_biasadd_readvariableop_resource4while_lstm_cell_23_biasadd_readvariableop_resource_0"l
3while_lstm_cell_23_matmul_1_readvariableop_resource5while_lstm_cell_23_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_23_matmul_readvariableop_resource3while_lstm_cell_23_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2V
)while/lstm_cell_23/BiasAdd/ReadVariableOp)while/lstm_cell_23/BiasAdd/ReadVariableOp2T
(while/lstm_cell_23/MatMul/ReadVariableOp(while/lstm_cell_23/MatMul/ReadVariableOp2X
*while/lstm_cell_23/MatMul_1/ReadVariableOp*while/lstm_cell_23/MatMul_1/ReadVariableOp: 
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
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
¹
¹
*__inference_lstm_23_layer_call_fn_39106183

inputs
unknown:
¬
	unknown_0:

	unknown_1:	
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_lstm_23_layer_call_and_return_conditional_losses_391041302
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:ÿÿÿÿÿÿÿÿÿ¬: : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
 
_user_specified_nameinputs
Î7
ß
$__inference__traced_restore_39107166
file_prefix3
 assignvariableop_dense_11_kernel:	.
 assignvariableop_1_dense_11_bias:A
.assignvariableop_2_lstm_22_lstm_cell_22_kernel:	]°L
8assignvariableop_3_lstm_22_lstm_cell_22_recurrent_kernel:
¬°;
,assignvariableop_4_lstm_22_lstm_cell_22_bias:	°B
.assignvariableop_5_lstm_23_lstm_cell_23_kernel:
¬L
8assignvariableop_6_lstm_23_lstm_cell_23_recurrent_kernel:
;
,assignvariableop_7_lstm_23_lstm_cell_23_bias:	"
assignvariableop_8_total: "
assignvariableop_9_count: %
assignvariableop_10_total_1: %
assignvariableop_11_count_1: 
identity_13¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_2¢AssignVariableOp_3¢AssignVariableOp_4¢AssignVariableOp_5¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*¥
valueBB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names¨
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*-
value$B"B B B B B B B B B B B B B 2
RestoreV2/shape_and_slicesì
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

Identity
AssignVariableOpAssignVariableOp assignvariableop_dense_11_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1¥
AssignVariableOp_1AssignVariableOp assignvariableop_1_dense_11_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2³
AssignVariableOp_2AssignVariableOp.assignvariableop_2_lstm_22_lstm_cell_22_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3½
AssignVariableOp_3AssignVariableOp8assignvariableop_3_lstm_22_lstm_cell_22_recurrent_kernelIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4±
AssignVariableOp_4AssignVariableOp,assignvariableop_4_lstm_22_lstm_cell_22_biasIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5³
AssignVariableOp_5AssignVariableOp.assignvariableop_5_lstm_23_lstm_cell_23_kernelIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6½
AssignVariableOp_6AssignVariableOp8assignvariableop_6_lstm_23_lstm_cell_23_recurrent_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7±
AssignVariableOp_7AssignVariableOp,assignvariableop_7_lstm_23_lstm_cell_23_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8
AssignVariableOp_8AssignVariableOpassignvariableop_8_totalIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9
AssignVariableOp_9AssignVariableOpassignvariableop_9_countIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10£
AssignVariableOp_10AssignVariableOpassignvariableop_10_total_1Identity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11£
AssignVariableOp_11AssignVariableOpassignvariableop_11_count_1Identity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_119
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpæ
Identity_12Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_12f
Identity_13IdentityIdentity_12:output:0^NoOp_1*
T0*
_output_shapes
: 2
Identity_13Î
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


+__inference_dense_11_layer_call_fn_39106834

inputs
unknown:	
	unknown_0:
identity¢StatefulPartitionedCallú
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_11_layer_call_and_return_conditional_losses_391041762
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ü	
Ê
&__inference_signature_wrapper_39104765
lstm_22_input
unknown:	]°
	unknown_0:
¬°
	unknown_1:	°
	unknown_2:
¬
	unknown_3:

	unknown_4:	
	unknown_5:	
	unknown_6:
identity¢StatefulPartitionedCall¬
StatefulPartitionedCallStatefulPartitionedCalllstm_22_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *,
f'R%
#__inference__wrapped_model_391025482
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ]: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
'
_user_specified_namelstm_22_input
°?
Ô
while_body_39104511
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_22_matmul_readvariableop_resource_0:	]°I
5while_lstm_cell_22_matmul_1_readvariableop_resource_0:
¬°C
4while_lstm_cell_22_biasadd_readvariableop_resource_0:	°
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_22_matmul_readvariableop_resource:	]°G
3while_lstm_cell_22_matmul_1_readvariableop_resource:
¬°A
2while_lstm_cell_22_biasadd_readvariableop_resource:	°¢)while/lstm_cell_22/BiasAdd/ReadVariableOp¢(while/lstm_cell_22/MatMul/ReadVariableOp¢*while/lstm_cell_22/MatMul_1/ReadVariableOpÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ]   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÓ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemÉ
(while/lstm_cell_22/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_22_matmul_readvariableop_resource_0*
_output_shapes
:	]°*
dtype02*
(while/lstm_cell_22/MatMul/ReadVariableOp×
while/lstm_cell_22/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_22/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°2
while/lstm_cell_22/MatMulÐ
*while/lstm_cell_22/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_22_matmul_1_readvariableop_resource_0* 
_output_shapes
:
¬°*
dtype02,
*while/lstm_cell_22/MatMul_1/ReadVariableOpÀ
while/lstm_cell_22/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_22/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°2
while/lstm_cell_22/MatMul_1¸
while/lstm_cell_22/addAddV2#while/lstm_cell_22/MatMul:product:0%while/lstm_cell_22/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°2
while/lstm_cell_22/addÈ
)while/lstm_cell_22/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_22_biasadd_readvariableop_resource_0*
_output_shapes	
:°*
dtype02+
)while/lstm_cell_22/BiasAdd/ReadVariableOpÅ
while/lstm_cell_22/BiasAddBiasAddwhile/lstm_cell_22/add:z:01while/lstm_cell_22/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°2
while/lstm_cell_22/BiasAdd
"while/lstm_cell_22/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_22/split/split_dim
while/lstm_cell_22/splitSplit+while/lstm_cell_22/split/split_dim:output:0#while/lstm_cell_22/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ¬*
	num_split2
while/lstm_cell_22/split
while/lstm_cell_22/SigmoidSigmoid!while/lstm_cell_22/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
while/lstm_cell_22/Sigmoid
while/lstm_cell_22/Sigmoid_1Sigmoid!while/lstm_cell_22/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
while/lstm_cell_22/Sigmoid_1¡
while/lstm_cell_22/mulMul while/lstm_cell_22/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
while/lstm_cell_22/mul
while/lstm_cell_22/ReluRelu!while/lstm_cell_22/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
while/lstm_cell_22/Reluµ
while/lstm_cell_22/mul_1Mulwhile/lstm_cell_22/Sigmoid:y:0%while/lstm_cell_22/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
while/lstm_cell_22/mul_1ª
while/lstm_cell_22/add_1AddV2while/lstm_cell_22/mul:z:0while/lstm_cell_22/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
while/lstm_cell_22/add_1
while/lstm_cell_22/Sigmoid_2Sigmoid!while/lstm_cell_22/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
while/lstm_cell_22/Sigmoid_2
while/lstm_cell_22/Relu_1Reluwhile/lstm_cell_22/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
while/lstm_cell_22/Relu_1¹
while/lstm_cell_22/mul_2Mul while/lstm_cell_22/Sigmoid_2:y:0'while/lstm_cell_22/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
while/lstm_cell_22/mul_2à
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_22/mul_2:z:0*
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
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_22/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_22/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
while/Identity_5Þ

while/NoOpNoOp*^while/lstm_cell_22/BiasAdd/ReadVariableOp)^while/lstm_cell_22/MatMul/ReadVariableOp+^while/lstm_cell_22/MatMul_1/ReadVariableOp*"
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
2while_lstm_cell_22_biasadd_readvariableop_resource4while_lstm_cell_22_biasadd_readvariableop_resource_0"l
3while_lstm_cell_22_matmul_1_readvariableop_resource5while_lstm_cell_22_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_22_matmul_readvariableop_resource3while_lstm_cell_22_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ¬: : : : : 2V
)while/lstm_cell_22/BiasAdd/ReadVariableOp)while/lstm_cell_22/BiasAdd/ReadVariableOp2T
(while/lstm_cell_22/MatMul/ReadVariableOp(while/lstm_cell_22/MatMul/ReadVariableOp2X
*while/lstm_cell_22/MatMul_1/ReadVariableOp*while/lstm_cell_22/MatMul_1/ReadVariableOp: 
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
:ÿÿÿÿÿÿÿÿÿ¬:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬:

_output_shapes
: :

_output_shapes
: 
ã
Í
while_cond_39102846
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_39102846___redundant_placeholder06
2while_while_cond_39102846___redundant_placeholder16
2while_while_cond_39102846___redundant_placeholder26
2while_while_cond_39102846___redundant_placeholder3
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
B: : : : :ÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ¬: ::::: 
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
:ÿÿÿÿÿÿÿÿÿ¬:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬:

_output_shapes
: :

_output_shapes
:


J__inference_lstm_cell_23_layer_call_and_return_conditional_losses_39103253

inputs

states
states_12
matmul_readvariableop_resource:
¬4
 matmul_1_readvariableop_resource:
.
biasadd_readvariableop_resource:	
identity

identity_1

identity_2¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
¬*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
add
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dimÃ
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2
split`
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
Sigmoidd
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	Sigmoid_1]
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mulW
ReluRelusplit:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relui
mul_1MulSigmoid:y:0Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mul_1^
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
add_1d
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	Sigmoid_2V
Relu_1Relu	add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu_1m
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mul_2e
IdentityIdentity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityi

Identity_1Identity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1i

Identity_2Identity	add_1:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_2
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
B:ÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
 
_user_specified_nameinputs:PL
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_namestates:PL
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_namestates
&
ó
while_body_39102847
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_00
while_lstm_cell_22_39102871_0:	]°1
while_lstm_cell_22_39102873_0:
¬°,
while_lstm_cell_22_39102875_0:	°
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor.
while_lstm_cell_22_39102871:	]°/
while_lstm_cell_22_39102873:
¬°*
while_lstm_cell_22_39102875:	°¢*while/lstm_cell_22/StatefulPartitionedCallÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ]   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÓ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemì
*while/lstm_cell_22/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_22_39102871_0while_lstm_cell_22_39102873_0while_lstm_cell_22_39102875_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ¬*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_lstm_cell_22_layer_call_and_return_conditional_losses_391027692,
*while/lstm_cell_22/StatefulPartitionedCall÷
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder3while/lstm_cell_22/StatefulPartitionedCall:output:0*
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
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3¥
while/Identity_4Identity3while/lstm_cell_22/StatefulPartitionedCall:output:1^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
while/Identity_4¥
while/Identity_5Identity3while/lstm_cell_22/StatefulPartitionedCall:output:2^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
while/Identity_5

while/NoOpNoOp+^while/lstm_cell_22/StatefulPartitionedCall*"
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
while_lstm_cell_22_39102871while_lstm_cell_22_39102871_0"<
while_lstm_cell_22_39102873while_lstm_cell_22_39102873_0"<
while_lstm_cell_22_39102875while_lstm_cell_22_39102875_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ¬: : : : : 2X
*while/lstm_cell_22/StatefulPartitionedCall*while/lstm_cell_22/StatefulPartitionedCall: 
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
:ÿÿÿÿÿÿÿÿÿ¬:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬:

_output_shapes
: :

_output_shapes
: 
°?
Ô
while_body_39105737
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_22_matmul_readvariableop_resource_0:	]°I
5while_lstm_cell_22_matmul_1_readvariableop_resource_0:
¬°C
4while_lstm_cell_22_biasadd_readvariableop_resource_0:	°
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_22_matmul_readvariableop_resource:	]°G
3while_lstm_cell_22_matmul_1_readvariableop_resource:
¬°A
2while_lstm_cell_22_biasadd_readvariableop_resource:	°¢)while/lstm_cell_22/BiasAdd/ReadVariableOp¢(while/lstm_cell_22/MatMul/ReadVariableOp¢*while/lstm_cell_22/MatMul_1/ReadVariableOpÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ]   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÓ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemÉ
(while/lstm_cell_22/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_22_matmul_readvariableop_resource_0*
_output_shapes
:	]°*
dtype02*
(while/lstm_cell_22/MatMul/ReadVariableOp×
while/lstm_cell_22/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_22/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°2
while/lstm_cell_22/MatMulÐ
*while/lstm_cell_22/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_22_matmul_1_readvariableop_resource_0* 
_output_shapes
:
¬°*
dtype02,
*while/lstm_cell_22/MatMul_1/ReadVariableOpÀ
while/lstm_cell_22/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_22/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°2
while/lstm_cell_22/MatMul_1¸
while/lstm_cell_22/addAddV2#while/lstm_cell_22/MatMul:product:0%while/lstm_cell_22/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°2
while/lstm_cell_22/addÈ
)while/lstm_cell_22/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_22_biasadd_readvariableop_resource_0*
_output_shapes	
:°*
dtype02+
)while/lstm_cell_22/BiasAdd/ReadVariableOpÅ
while/lstm_cell_22/BiasAddBiasAddwhile/lstm_cell_22/add:z:01while/lstm_cell_22/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°2
while/lstm_cell_22/BiasAdd
"while/lstm_cell_22/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_22/split/split_dim
while/lstm_cell_22/splitSplit+while/lstm_cell_22/split/split_dim:output:0#while/lstm_cell_22/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ¬*
	num_split2
while/lstm_cell_22/split
while/lstm_cell_22/SigmoidSigmoid!while/lstm_cell_22/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
while/lstm_cell_22/Sigmoid
while/lstm_cell_22/Sigmoid_1Sigmoid!while/lstm_cell_22/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
while/lstm_cell_22/Sigmoid_1¡
while/lstm_cell_22/mulMul while/lstm_cell_22/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
while/lstm_cell_22/mul
while/lstm_cell_22/ReluRelu!while/lstm_cell_22/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
while/lstm_cell_22/Reluµ
while/lstm_cell_22/mul_1Mulwhile/lstm_cell_22/Sigmoid:y:0%while/lstm_cell_22/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
while/lstm_cell_22/mul_1ª
while/lstm_cell_22/add_1AddV2while/lstm_cell_22/mul:z:0while/lstm_cell_22/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
while/lstm_cell_22/add_1
while/lstm_cell_22/Sigmoid_2Sigmoid!while/lstm_cell_22/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
while/lstm_cell_22/Sigmoid_2
while/lstm_cell_22/Relu_1Reluwhile/lstm_cell_22/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
while/lstm_cell_22/Relu_1¹
while/lstm_cell_22/mul_2Mul while/lstm_cell_22/Sigmoid_2:y:0'while/lstm_cell_22/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
while/lstm_cell_22/mul_2à
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_22/mul_2:z:0*
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
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_22/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_22/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
while/Identity_5Þ

while/NoOpNoOp*^while/lstm_cell_22/BiasAdd/ReadVariableOp)^while/lstm_cell_22/MatMul/ReadVariableOp+^while/lstm_cell_22/MatMul_1/ReadVariableOp*"
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
2while_lstm_cell_22_biasadd_readvariableop_resource4while_lstm_cell_22_biasadd_readvariableop_resource_0"l
3while_lstm_cell_22_matmul_1_readvariableop_resource5while_lstm_cell_22_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_22_matmul_readvariableop_resource3while_lstm_cell_22_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ¬: : : : : 2V
)while/lstm_cell_22/BiasAdd/ReadVariableOp)while/lstm_cell_22/BiasAdd/ReadVariableOp2T
(while/lstm_cell_22/MatMul/ReadVariableOp(while/lstm_cell_22/MatMul/ReadVariableOp2X
*while/lstm_cell_22/MatMul_1/ReadVariableOp*while/lstm_cell_22/MatMul_1/ReadVariableOp: 
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
:ÿÿÿÿÿÿÿÿÿ¬:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬:

_output_shapes
: :

_output_shapes
: 
ËF

E__inference_lstm_22_layer_call_and_return_conditional_losses_39102916

inputs(
lstm_cell_22_39102834:	]°)
lstm_cell_22_39102836:
¬°$
lstm_cell_22_39102838:	°
identity¢$lstm_cell_22/StatefulPartitionedCall¢whileD
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
strided_slice/stack_2â
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
B :¬2
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
B :è2
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
B :¬2
zeros/packed/1
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
:ÿÿÿÿÿÿÿÿÿ¬2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :¬2
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
B :è2
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
B :¬2
zeros_1/packed/1
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
:ÿÿÿÿÿÿÿÿÿ¬2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ]2
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
strided_slice_1/stack_2î
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
TensorArrayV2/element_shape²
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2¿
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ]   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeø
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
strided_slice_2/stack_2ü
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]*
shrink_axis_mask2
strided_slice_2¨
$lstm_cell_22/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_22_39102834lstm_cell_22_39102836lstm_cell_22_39102838*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ¬*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_lstm_cell_22_layer_call_and_return_conditional_losses_391027692&
$lstm_cell_22/StatefulPartitionedCall
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ¬   2
TensorArrayV2_1/element_shape¸
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
ÿÿÿÿÿÿÿÿÿ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterÎ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_22_39102834lstm_cell_22_39102836lstm_cell_22_39102838*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ¬: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_39102847*
condR
while_cond_39102846*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ¬: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ¬   22
0TensorArrayV2Stack/TensorListStack/element_shapeò
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
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
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm¯
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬2
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
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬2

Identity}
NoOpNoOp%^lstm_cell_22/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ]: : : 2L
$lstm_cell_22/StatefulPartitionedCall$lstm_cell_22/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ]
 
_user_specified_nameinputs
É\
¡
E__inference_lstm_23_layer_call_and_return_conditional_losses_39106496
inputs_0?
+lstm_cell_23_matmul_readvariableop_resource:
¬A
-lstm_cell_23_matmul_1_readvariableop_resource:
;
,lstm_cell_23_biasadd_readvariableop_resource:	
identity¢#lstm_cell_23/BiasAdd/ReadVariableOp¢"lstm_cell_23/MatMul/ReadVariableOp¢$lstm_cell_23/MatMul_1/ReadVariableOp¢whileF
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
strided_slice/stack_2â
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
B :2
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
B :è2
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
B :2
zeros/packed/1
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
:ÿÿÿÿÿÿÿÿÿ2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :2
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
B :è2
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
B :2
zeros_1/packed/1
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
:ÿÿÿÿÿÿÿÿÿ2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬2
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
strided_slice_1/stack_2î
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
TensorArrayV2/element_shape²
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2¿
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ¬   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeø
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
strided_slice_2/stack_2ý
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬*
shrink_axis_mask2
strided_slice_2¶
"lstm_cell_23/MatMul/ReadVariableOpReadVariableOp+lstm_cell_23_matmul_readvariableop_resource* 
_output_shapes
:
¬*
dtype02$
"lstm_cell_23/MatMul/ReadVariableOp­
lstm_cell_23/MatMulMatMulstrided_slice_2:output:0*lstm_cell_23/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_23/MatMul¼
$lstm_cell_23/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_23_matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype02&
$lstm_cell_23/MatMul_1/ReadVariableOp©
lstm_cell_23/MatMul_1MatMulzeros:output:0,lstm_cell_23/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_23/MatMul_1 
lstm_cell_23/addAddV2lstm_cell_23/MatMul:product:0lstm_cell_23/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_23/add´
#lstm_cell_23/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_23_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02%
#lstm_cell_23/BiasAdd/ReadVariableOp­
lstm_cell_23/BiasAddBiasAddlstm_cell_23/add:z:0+lstm_cell_23/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_23/BiasAdd~
lstm_cell_23/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_23/split/split_dim÷
lstm_cell_23/splitSplit%lstm_cell_23/split/split_dim:output:0lstm_cell_23/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2
lstm_cell_23/split
lstm_cell_23/SigmoidSigmoidlstm_cell_23/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_23/Sigmoid
lstm_cell_23/Sigmoid_1Sigmoidlstm_cell_23/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_23/Sigmoid_1
lstm_cell_23/mulMullstm_cell_23/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_23/mul~
lstm_cell_23/ReluRelulstm_cell_23/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_23/Relu
lstm_cell_23/mul_1Mullstm_cell_23/Sigmoid:y:0lstm_cell_23/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_23/mul_1
lstm_cell_23/add_1AddV2lstm_cell_23/mul:z:0lstm_cell_23/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_23/add_1
lstm_cell_23/Sigmoid_2Sigmoidlstm_cell_23/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_23/Sigmoid_2}
lstm_cell_23/Relu_1Relulstm_cell_23/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_23/Relu_1¡
lstm_cell_23/mul_2Mullstm_cell_23/Sigmoid_2:y:0!lstm_cell_23/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_23/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
TensorArrayV2_1/element_shape¸
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
ÿÿÿÿÿÿÿÿÿ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_23_matmul_readvariableop_resource-lstm_cell_23_matmul_1_readvariableop_resource,lstm_cell_23_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_39106412*
condR
while_cond_39106411*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   22
0TensorArrayV2Stack/TensorListStack/element_shapeò
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
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
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm¯
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
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
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

IdentityÈ
NoOpNoOp$^lstm_cell_23/BiasAdd/ReadVariableOp#^lstm_cell_23/MatMul/ReadVariableOp%^lstm_cell_23/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬: : : 2J
#lstm_cell_23/BiasAdd/ReadVariableOp#lstm_cell_23/BiasAdd/ReadVariableOp2H
"lstm_cell_23/MatMul/ReadVariableOp"lstm_cell_23/MatMul/ReadVariableOp2L
$lstm_cell_23/MatMul_1/ReadVariableOp$lstm_cell_23/MatMul_1/ReadVariableOp2
whilewhile:_ [
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬
"
_user_specified_name
inputs/0
\

E__inference_lstm_23_layer_call_and_return_conditional_losses_39106647

inputs?
+lstm_cell_23_matmul_readvariableop_resource:
¬A
-lstm_cell_23_matmul_1_readvariableop_resource:
;
,lstm_cell_23_biasadd_readvariableop_resource:	
identity¢#lstm_cell_23/BiasAdd/ReadVariableOp¢"lstm_cell_23/MatMul/ReadVariableOp¢$lstm_cell_23/MatMul_1/ReadVariableOp¢whileD
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
strided_slice/stack_2â
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
B :2
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
B :è2
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
B :2
zeros/packed/1
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
:ÿÿÿÿÿÿÿÿÿ2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :2
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
B :è2
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
B :2
zeros_1/packed/1
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
:ÿÿÿÿÿÿÿÿÿ2	
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
:ÿÿÿÿÿÿÿÿÿ¬2
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
strided_slice_1/stack_2î
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
TensorArrayV2/element_shape²
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2¿
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ¬   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeø
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
strided_slice_2/stack_2ý
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬*
shrink_axis_mask2
strided_slice_2¶
"lstm_cell_23/MatMul/ReadVariableOpReadVariableOp+lstm_cell_23_matmul_readvariableop_resource* 
_output_shapes
:
¬*
dtype02$
"lstm_cell_23/MatMul/ReadVariableOp­
lstm_cell_23/MatMulMatMulstrided_slice_2:output:0*lstm_cell_23/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_23/MatMul¼
$lstm_cell_23/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_23_matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype02&
$lstm_cell_23/MatMul_1/ReadVariableOp©
lstm_cell_23/MatMul_1MatMulzeros:output:0,lstm_cell_23/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_23/MatMul_1 
lstm_cell_23/addAddV2lstm_cell_23/MatMul:product:0lstm_cell_23/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_23/add´
#lstm_cell_23/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_23_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02%
#lstm_cell_23/BiasAdd/ReadVariableOp­
lstm_cell_23/BiasAddBiasAddlstm_cell_23/add:z:0+lstm_cell_23/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_23/BiasAdd~
lstm_cell_23/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_23/split/split_dim÷
lstm_cell_23/splitSplit%lstm_cell_23/split/split_dim:output:0lstm_cell_23/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2
lstm_cell_23/split
lstm_cell_23/SigmoidSigmoidlstm_cell_23/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_23/Sigmoid
lstm_cell_23/Sigmoid_1Sigmoidlstm_cell_23/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_23/Sigmoid_1
lstm_cell_23/mulMullstm_cell_23/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_23/mul~
lstm_cell_23/ReluRelulstm_cell_23/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_23/Relu
lstm_cell_23/mul_1Mullstm_cell_23/Sigmoid:y:0lstm_cell_23/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_23/mul_1
lstm_cell_23/add_1AddV2lstm_cell_23/mul:z:0lstm_cell_23/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_23/add_1
lstm_cell_23/Sigmoid_2Sigmoidlstm_cell_23/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_23/Sigmoid_2}
lstm_cell_23/Relu_1Relulstm_cell_23/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_23/Relu_1¡
lstm_cell_23/mul_2Mullstm_cell_23/Sigmoid_2:y:0!lstm_cell_23/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_23/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
TensorArrayV2_1/element_shape¸
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
ÿÿÿÿÿÿÿÿÿ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_23_matmul_readvariableop_resource-lstm_cell_23_matmul_1_readvariableop_resource,lstm_cell_23_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_39106563*
condR
while_cond_39106562*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   22
0TensorArrayV2Stack/TensorListStack/element_shapeé
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
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
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm¦
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
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
:ÿÿÿÿÿÿÿÿÿ2

IdentityÈ
NoOpNoOp$^lstm_cell_23/BiasAdd/ReadVariableOp#^lstm_cell_23/MatMul/ReadVariableOp%^lstm_cell_23/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:ÿÿÿÿÿÿÿÿÿ¬: : : 2J
#lstm_cell_23/BiasAdd/ReadVariableOp#lstm_cell_23/BiasAdd/ReadVariableOp2H
"lstm_cell_23/MatMul/ReadVariableOp"lstm_cell_23/MatMul/ReadVariableOp2L
$lstm_cell_23/MatMul_1/ReadVariableOp$lstm_cell_23/MatMul_1/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
 
_user_specified_nameinputs
ã
»
*__inference_lstm_23_layer_call_fn_39106161
inputs_0
unknown:
¬
	unknown_0:

	unknown_1:	
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_lstm_23_layer_call_and_return_conditional_losses_391033362
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬: : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬
"
_user_specified_name
inputs/0


J__inference_lstm_cell_23_layer_call_and_return_conditional_losses_39107029

inputs
states_0
states_12
matmul_readvariableop_resource:
¬4
 matmul_1_readvariableop_resource:
.
biasadd_readvariableop_resource:	
identity

identity_1

identity_2¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
¬*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul_1/ReadVariableOp|
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
add
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dimÃ
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2
split`
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
Sigmoidd
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	Sigmoid_1]
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mulW
ReluRelusplit:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relui
mul_1MulSigmoid:y:0Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mul_1^
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
add_1d
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	Sigmoid_2V
Relu_1Relu	add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu_1m
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mul_2e
IdentityIdentity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityi

Identity_1Identity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1i

Identity_2Identity	add_1:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_2
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
B:ÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
 
_user_specified_nameinputs:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
states/0:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
states/1
&
õ
while_body_39103477
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_01
while_lstm_cell_23_39103501_0:
¬1
while_lstm_cell_23_39103503_0:
,
while_lstm_cell_23_39103505_0:	
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor/
while_lstm_cell_23_39103501:
¬/
while_lstm_cell_23_39103503:
*
while_lstm_cell_23_39103505:	¢*while/lstm_cell_23/StatefulPartitionedCallÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ¬   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÔ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemì
*while/lstm_cell_23/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_23_39103501_0while_lstm_cell_23_39103503_0while_lstm_cell_23_39103505_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_lstm_cell_23_layer_call_and_return_conditional_losses_391033992,
*while/lstm_cell_23/StatefulPartitionedCall÷
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder3while/lstm_cell_23/StatefulPartitionedCall:output:0*
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
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3¥
while/Identity_4Identity3while/lstm_cell_23/StatefulPartitionedCall:output:1^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/Identity_4¥
while/Identity_5Identity3while/lstm_cell_23/StatefulPartitionedCall:output:2^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/Identity_5

while/NoOpNoOp+^while/lstm_cell_23/StatefulPartitionedCall*"
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
while_lstm_cell_23_39103501while_lstm_cell_23_39103501_0"<
while_lstm_cell_23_39103503while_lstm_cell_23_39103503_0"<
while_lstm_cell_23_39103505while_lstm_cell_23_39103505_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2X
*while/lstm_cell_23/StatefulPartitionedCall*while/lstm_cell_23/StatefulPartitionedCall: 
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
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 


J__inference_lstm_cell_22_layer_call_and_return_conditional_losses_39102623

inputs

states
states_11
matmul_readvariableop_resource:	]°4
 matmul_1_readvariableop_resource:
¬°.
biasadd_readvariableop_resource:	°
identity

identity_1

identity_2¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	]°*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°2
MatMul
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
¬°*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°2
add
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:°*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°2	
BiasAddd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dimÃ
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ¬*
	num_split2
split`
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2	
Sigmoidd
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
	Sigmoid_1]
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
mulW
ReluRelusplit:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
Relui
mul_1MulSigmoid:y:0Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
mul_1^
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
add_1d
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
	Sigmoid_2V
Relu_1Relu	add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
Relu_1m
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
mul_2e
IdentityIdentity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2

Identityi

Identity_1Identity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2

Identity_1i

Identity_2Identity	add_1:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2

Identity_2
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
A:ÿÿÿÿÿÿÿÿÿ]:ÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ¬: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
 
_user_specified_nameinputs:PL
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
 
_user_specified_namestates:PL
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
 
_user_specified_namestates
¶
¸
*__inference_lstm_22_layer_call_fn_39105508

inputs
unknown:	]°
	unknown_0:
¬°
	unknown_1:	°
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_lstm_22_layer_call_and_return_conditional_losses_391039652
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ]: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
 
_user_specified_nameinputs
ã
Í
while_cond_39105736
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_39105736___redundant_placeholder06
2while_while_cond_39105736___redundant_placeholder16
2while_while_cond_39105736___redundant_placeholder26
2while_while_cond_39105736___redundant_placeholder3
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
B: : : : :ÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ¬: ::::: 
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
:ÿÿÿÿÿÿÿÿÿ¬:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬:

_output_shapes
: :

_output_shapes
:
ã
Í
while_cond_39106713
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_39106713___redundant_placeholder06
2while_while_cond_39106713___redundant_placeholder16
2while_while_cond_39106713___redundant_placeholder26
2while_while_cond_39106713___redundant_placeholder3
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
B: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 
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
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
°?
Ô
while_body_39105888
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_22_matmul_readvariableop_resource_0:	]°I
5while_lstm_cell_22_matmul_1_readvariableop_resource_0:
¬°C
4while_lstm_cell_22_biasadd_readvariableop_resource_0:	°
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_22_matmul_readvariableop_resource:	]°G
3while_lstm_cell_22_matmul_1_readvariableop_resource:
¬°A
2while_lstm_cell_22_biasadd_readvariableop_resource:	°¢)while/lstm_cell_22/BiasAdd/ReadVariableOp¢(while/lstm_cell_22/MatMul/ReadVariableOp¢*while/lstm_cell_22/MatMul_1/ReadVariableOpÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ]   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÓ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemÉ
(while/lstm_cell_22/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_22_matmul_readvariableop_resource_0*
_output_shapes
:	]°*
dtype02*
(while/lstm_cell_22/MatMul/ReadVariableOp×
while/lstm_cell_22/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_22/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°2
while/lstm_cell_22/MatMulÐ
*while/lstm_cell_22/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_22_matmul_1_readvariableop_resource_0* 
_output_shapes
:
¬°*
dtype02,
*while/lstm_cell_22/MatMul_1/ReadVariableOpÀ
while/lstm_cell_22/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_22/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°2
while/lstm_cell_22/MatMul_1¸
while/lstm_cell_22/addAddV2#while/lstm_cell_22/MatMul:product:0%while/lstm_cell_22/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°2
while/lstm_cell_22/addÈ
)while/lstm_cell_22/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_22_biasadd_readvariableop_resource_0*
_output_shapes	
:°*
dtype02+
)while/lstm_cell_22/BiasAdd/ReadVariableOpÅ
while/lstm_cell_22/BiasAddBiasAddwhile/lstm_cell_22/add:z:01while/lstm_cell_22/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°2
while/lstm_cell_22/BiasAdd
"while/lstm_cell_22/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_22/split/split_dim
while/lstm_cell_22/splitSplit+while/lstm_cell_22/split/split_dim:output:0#while/lstm_cell_22/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ¬*
	num_split2
while/lstm_cell_22/split
while/lstm_cell_22/SigmoidSigmoid!while/lstm_cell_22/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
while/lstm_cell_22/Sigmoid
while/lstm_cell_22/Sigmoid_1Sigmoid!while/lstm_cell_22/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
while/lstm_cell_22/Sigmoid_1¡
while/lstm_cell_22/mulMul while/lstm_cell_22/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
while/lstm_cell_22/mul
while/lstm_cell_22/ReluRelu!while/lstm_cell_22/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
while/lstm_cell_22/Reluµ
while/lstm_cell_22/mul_1Mulwhile/lstm_cell_22/Sigmoid:y:0%while/lstm_cell_22/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
while/lstm_cell_22/mul_1ª
while/lstm_cell_22/add_1AddV2while/lstm_cell_22/mul:z:0while/lstm_cell_22/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
while/lstm_cell_22/add_1
while/lstm_cell_22/Sigmoid_2Sigmoid!while/lstm_cell_22/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
while/lstm_cell_22/Sigmoid_2
while/lstm_cell_22/Relu_1Reluwhile/lstm_cell_22/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
while/lstm_cell_22/Relu_1¹
while/lstm_cell_22/mul_2Mul while/lstm_cell_22/Sigmoid_2:y:0'while/lstm_cell_22/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
while/lstm_cell_22/mul_2à
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_22/mul_2:z:0*
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
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_22/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_22/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
while/Identity_5Þ

while/NoOpNoOp*^while/lstm_cell_22/BiasAdd/ReadVariableOp)^while/lstm_cell_22/MatMul/ReadVariableOp+^while/lstm_cell_22/MatMul_1/ReadVariableOp*"
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
2while_lstm_cell_22_biasadd_readvariableop_resource4while_lstm_cell_22_biasadd_readvariableop_resource_0"l
3while_lstm_cell_22_matmul_1_readvariableop_resource5while_lstm_cell_22_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_22_matmul_readvariableop_resource3while_lstm_cell_22_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ¬: : : : : 2V
)while/lstm_cell_22/BiasAdd/ReadVariableOp)while/lstm_cell_22/BiasAdd/ReadVariableOp2T
(while/lstm_cell_22/MatMul/ReadVariableOp(while/lstm_cell_22/MatMul/ReadVariableOp2X
*while/lstm_cell_22/MatMul_1/ReadVariableOp*while/lstm_cell_22/MatMul_1/ReadVariableOp: 
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
:ÿÿÿÿÿÿÿÿÿ¬:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬:

_output_shapes
: :

_output_shapes
: 
É\
¡
E__inference_lstm_23_layer_call_and_return_conditional_losses_39106345
inputs_0?
+lstm_cell_23_matmul_readvariableop_resource:
¬A
-lstm_cell_23_matmul_1_readvariableop_resource:
;
,lstm_cell_23_biasadd_readvariableop_resource:	
identity¢#lstm_cell_23/BiasAdd/ReadVariableOp¢"lstm_cell_23/MatMul/ReadVariableOp¢$lstm_cell_23/MatMul_1/ReadVariableOp¢whileF
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
strided_slice/stack_2â
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
B :2
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
B :è2
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
B :2
zeros/packed/1
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
:ÿÿÿÿÿÿÿÿÿ2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :2
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
B :è2
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
B :2
zeros_1/packed/1
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
:ÿÿÿÿÿÿÿÿÿ2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬2
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
strided_slice_1/stack_2î
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
TensorArrayV2/element_shape²
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2¿
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ¬   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeø
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
strided_slice_2/stack_2ý
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬*
shrink_axis_mask2
strided_slice_2¶
"lstm_cell_23/MatMul/ReadVariableOpReadVariableOp+lstm_cell_23_matmul_readvariableop_resource* 
_output_shapes
:
¬*
dtype02$
"lstm_cell_23/MatMul/ReadVariableOp­
lstm_cell_23/MatMulMatMulstrided_slice_2:output:0*lstm_cell_23/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_23/MatMul¼
$lstm_cell_23/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_23_matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype02&
$lstm_cell_23/MatMul_1/ReadVariableOp©
lstm_cell_23/MatMul_1MatMulzeros:output:0,lstm_cell_23/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_23/MatMul_1 
lstm_cell_23/addAddV2lstm_cell_23/MatMul:product:0lstm_cell_23/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_23/add´
#lstm_cell_23/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_23_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02%
#lstm_cell_23/BiasAdd/ReadVariableOp­
lstm_cell_23/BiasAddBiasAddlstm_cell_23/add:z:0+lstm_cell_23/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_23/BiasAdd~
lstm_cell_23/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_23/split/split_dim÷
lstm_cell_23/splitSplit%lstm_cell_23/split/split_dim:output:0lstm_cell_23/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2
lstm_cell_23/split
lstm_cell_23/SigmoidSigmoidlstm_cell_23/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_23/Sigmoid
lstm_cell_23/Sigmoid_1Sigmoidlstm_cell_23/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_23/Sigmoid_1
lstm_cell_23/mulMullstm_cell_23/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_23/mul~
lstm_cell_23/ReluRelulstm_cell_23/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_23/Relu
lstm_cell_23/mul_1Mullstm_cell_23/Sigmoid:y:0lstm_cell_23/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_23/mul_1
lstm_cell_23/add_1AddV2lstm_cell_23/mul:z:0lstm_cell_23/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_23/add_1
lstm_cell_23/Sigmoid_2Sigmoidlstm_cell_23/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_23/Sigmoid_2}
lstm_cell_23/Relu_1Relulstm_cell_23/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_23/Relu_1¡
lstm_cell_23/mul_2Mullstm_cell_23/Sigmoid_2:y:0!lstm_cell_23/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_23/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
TensorArrayV2_1/element_shape¸
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
ÿÿÿÿÿÿÿÿÿ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_23_matmul_readvariableop_resource-lstm_cell_23_matmul_1_readvariableop_resource,lstm_cell_23_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_39106261*
condR
while_cond_39106260*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   22
0TensorArrayV2Stack/TensorListStack/element_shapeò
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
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
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm¯
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
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
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

IdentityÈ
NoOpNoOp$^lstm_cell_23/BiasAdd/ReadVariableOp#^lstm_cell_23/MatMul/ReadVariableOp%^lstm_cell_23/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬: : : 2J
#lstm_cell_23/BiasAdd/ReadVariableOp#lstm_cell_23/BiasAdd/ReadVariableOp2H
"lstm_cell_23/MatMul/ReadVariableOp"lstm_cell_23/MatMul/ReadVariableOp2L
$lstm_cell_23/MatMul_1/ReadVariableOp$lstm_cell_23/MatMul_1/ReadVariableOp2
whilewhile:_ [
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬
"
_user_specified_name
inputs/0
¶
¸
*__inference_lstm_22_layer_call_fn_39105519

inputs
unknown:	]°
	unknown_0:
¬°
	unknown_1:	°
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_lstm_22_layer_call_and_return_conditional_losses_391045952
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ]: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
 
_user_specified_nameinputs
ã
»
*__inference_lstm_23_layer_call_fn_39106172
inputs_0
unknown:
¬
	unknown_0:

	unknown_1:	
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_lstm_23_layer_call_and_return_conditional_losses_391035462
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬: : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬
"
_user_specified_name
inputs/0
´?
Ö
while_body_39106261
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
3while_lstm_cell_23_matmul_readvariableop_resource_0:
¬I
5while_lstm_cell_23_matmul_1_readvariableop_resource_0:
C
4while_lstm_cell_23_biasadd_readvariableop_resource_0:	
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
1while_lstm_cell_23_matmul_readvariableop_resource:
¬G
3while_lstm_cell_23_matmul_1_readvariableop_resource:
A
2while_lstm_cell_23_biasadd_readvariableop_resource:	¢)while/lstm_cell_23/BiasAdd/ReadVariableOp¢(while/lstm_cell_23/MatMul/ReadVariableOp¢*while/lstm_cell_23/MatMul_1/ReadVariableOpÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ¬   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÔ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemÊ
(while/lstm_cell_23/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_23_matmul_readvariableop_resource_0* 
_output_shapes
:
¬*
dtype02*
(while/lstm_cell_23/MatMul/ReadVariableOp×
while/lstm_cell_23/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_23/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_23/MatMulÐ
*while/lstm_cell_23/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_23_matmul_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype02,
*while/lstm_cell_23/MatMul_1/ReadVariableOpÀ
while/lstm_cell_23/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_23/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_23/MatMul_1¸
while/lstm_cell_23/addAddV2#while/lstm_cell_23/MatMul:product:0%while/lstm_cell_23/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_23/addÈ
)while/lstm_cell_23/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_23_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02+
)while/lstm_cell_23/BiasAdd/ReadVariableOpÅ
while/lstm_cell_23/BiasAddBiasAddwhile/lstm_cell_23/add:z:01while/lstm_cell_23/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_23/BiasAdd
"while/lstm_cell_23/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_23/split/split_dim
while/lstm_cell_23/splitSplit+while/lstm_cell_23/split/split_dim:output:0#while/lstm_cell_23/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2
while/lstm_cell_23/split
while/lstm_cell_23/SigmoidSigmoid!while/lstm_cell_23/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_23/Sigmoid
while/lstm_cell_23/Sigmoid_1Sigmoid!while/lstm_cell_23/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_23/Sigmoid_1¡
while/lstm_cell_23/mulMul while/lstm_cell_23/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_23/mul
while/lstm_cell_23/ReluRelu!while/lstm_cell_23/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_23/Reluµ
while/lstm_cell_23/mul_1Mulwhile/lstm_cell_23/Sigmoid:y:0%while/lstm_cell_23/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_23/mul_1ª
while/lstm_cell_23/add_1AddV2while/lstm_cell_23/mul:z:0while/lstm_cell_23/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_23/add_1
while/lstm_cell_23/Sigmoid_2Sigmoid!while/lstm_cell_23/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_23/Sigmoid_2
while/lstm_cell_23/Relu_1Reluwhile/lstm_cell_23/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_23/Relu_1¹
while/lstm_cell_23/mul_2Mul while/lstm_cell_23/Sigmoid_2:y:0'while/lstm_cell_23/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_23/mul_2à
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_23/mul_2:z:0*
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
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_23/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_23/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/Identity_5Þ

while/NoOpNoOp*^while/lstm_cell_23/BiasAdd/ReadVariableOp)^while/lstm_cell_23/MatMul/ReadVariableOp+^while/lstm_cell_23/MatMul_1/ReadVariableOp*"
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
2while_lstm_cell_23_biasadd_readvariableop_resource4while_lstm_cell_23_biasadd_readvariableop_resource_0"l
3while_lstm_cell_23_matmul_1_readvariableop_resource5while_lstm_cell_23_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_23_matmul_readvariableop_resource3while_lstm_cell_23_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2V
)while/lstm_cell_23/BiasAdd/ReadVariableOp)while/lstm_cell_23/BiasAdd/ReadVariableOp2T
(while/lstm_cell_23/MatMul/ReadVariableOp(while/lstm_cell_23/MatMul/ReadVariableOp2X
*while/lstm_cell_23/MatMul_1/ReadVariableOp*while/lstm_cell_23/MatMul_1/ReadVariableOp: 
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
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
Ô

í
lstm_22_while_cond_39104873,
(lstm_22_while_lstm_22_while_loop_counter2
.lstm_22_while_lstm_22_while_maximum_iterations
lstm_22_while_placeholder
lstm_22_while_placeholder_1
lstm_22_while_placeholder_2
lstm_22_while_placeholder_3.
*lstm_22_while_less_lstm_22_strided_slice_1F
Blstm_22_while_lstm_22_while_cond_39104873___redundant_placeholder0F
Blstm_22_while_lstm_22_while_cond_39104873___redundant_placeholder1F
Blstm_22_while_lstm_22_while_cond_39104873___redundant_placeholder2F
Blstm_22_while_lstm_22_while_cond_39104873___redundant_placeholder3
lstm_22_while_identity

lstm_22/while/LessLesslstm_22_while_placeholder*lstm_22_while_less_lstm_22_strided_slice_1*
T0*
_output_shapes
: 2
lstm_22/while/Lessu
lstm_22/while/IdentityIdentitylstm_22/while/Less:z:0*
T0
*
_output_shapes
: 2
lstm_22/while/Identity"9
lstm_22_while_identitylstm_22/while/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :ÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ¬: ::::: 
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
:ÿÿÿÿÿÿÿÿÿ¬:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬:

_output_shapes
: :

_output_shapes
:
ËF

E__inference_lstm_22_layer_call_and_return_conditional_losses_39102706

inputs(
lstm_cell_22_39102624:	]°)
lstm_cell_22_39102626:
¬°$
lstm_cell_22_39102628:	°
identity¢$lstm_cell_22/StatefulPartitionedCall¢whileD
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
strided_slice/stack_2â
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
B :¬2
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
B :è2
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
B :¬2
zeros/packed/1
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
:ÿÿÿÿÿÿÿÿÿ¬2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :¬2
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
B :è2
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
B :¬2
zeros_1/packed/1
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
:ÿÿÿÿÿÿÿÿÿ¬2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ]2
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
strided_slice_1/stack_2î
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
TensorArrayV2/element_shape²
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2¿
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ]   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeø
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
strided_slice_2/stack_2ü
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]*
shrink_axis_mask2
strided_slice_2¨
$lstm_cell_22/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_22_39102624lstm_cell_22_39102626lstm_cell_22_39102628*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ¬*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_lstm_cell_22_layer_call_and_return_conditional_losses_391026232&
$lstm_cell_22/StatefulPartitionedCall
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ¬   2
TensorArrayV2_1/element_shape¸
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
ÿÿÿÿÿÿÿÿÿ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterÎ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_22_39102624lstm_cell_22_39102626lstm_cell_22_39102628*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ¬: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_39102637*
condR
while_cond_39102636*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ¬: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ¬   22
0TensorArrayV2Stack/TensorListStack/element_shapeò
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
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
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm¯
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬2
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
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬2

Identity}
NoOpNoOp%^lstm_cell_22/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ]: : : 2L
$lstm_cell_22/StatefulPartitionedCall$lstm_cell_22/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ]
 
_user_specified_nameinputs
\

E__inference_lstm_22_layer_call_and_return_conditional_losses_39105972

inputs>
+lstm_cell_22_matmul_readvariableop_resource:	]°A
-lstm_cell_22_matmul_1_readvariableop_resource:
¬°;
,lstm_cell_22_biasadd_readvariableop_resource:	°
identity¢#lstm_cell_22/BiasAdd/ReadVariableOp¢"lstm_cell_22/MatMul/ReadVariableOp¢$lstm_cell_22/MatMul_1/ReadVariableOp¢whileD
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
strided_slice/stack_2â
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
B :¬2
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
B :è2
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
B :¬2
zeros/packed/1
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
:ÿÿÿÿÿÿÿÿÿ¬2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :¬2
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
B :è2
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
B :¬2
zeros_1/packed/1
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
:ÿÿÿÿÿÿÿÿÿ¬2	
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
:ÿÿÿÿÿÿÿÿÿ]2
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
strided_slice_1/stack_2î
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
TensorArrayV2/element_shape²
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2¿
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ]   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeø
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
strided_slice_2/stack_2ü
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]*
shrink_axis_mask2
strided_slice_2µ
"lstm_cell_22/MatMul/ReadVariableOpReadVariableOp+lstm_cell_22_matmul_readvariableop_resource*
_output_shapes
:	]°*
dtype02$
"lstm_cell_22/MatMul/ReadVariableOp­
lstm_cell_22/MatMulMatMulstrided_slice_2:output:0*lstm_cell_22/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°2
lstm_cell_22/MatMul¼
$lstm_cell_22/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_22_matmul_1_readvariableop_resource* 
_output_shapes
:
¬°*
dtype02&
$lstm_cell_22/MatMul_1/ReadVariableOp©
lstm_cell_22/MatMul_1MatMulzeros:output:0,lstm_cell_22/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°2
lstm_cell_22/MatMul_1 
lstm_cell_22/addAddV2lstm_cell_22/MatMul:product:0lstm_cell_22/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°2
lstm_cell_22/add´
#lstm_cell_22/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_22_biasadd_readvariableop_resource*
_output_shapes	
:°*
dtype02%
#lstm_cell_22/BiasAdd/ReadVariableOp­
lstm_cell_22/BiasAddBiasAddlstm_cell_22/add:z:0+lstm_cell_22/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°2
lstm_cell_22/BiasAdd~
lstm_cell_22/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_22/split/split_dim÷
lstm_cell_22/splitSplit%lstm_cell_22/split/split_dim:output:0lstm_cell_22/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ¬*
	num_split2
lstm_cell_22/split
lstm_cell_22/SigmoidSigmoidlstm_cell_22/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
lstm_cell_22/Sigmoid
lstm_cell_22/Sigmoid_1Sigmoidlstm_cell_22/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
lstm_cell_22/Sigmoid_1
lstm_cell_22/mulMullstm_cell_22/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
lstm_cell_22/mul~
lstm_cell_22/ReluRelulstm_cell_22/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
lstm_cell_22/Relu
lstm_cell_22/mul_1Mullstm_cell_22/Sigmoid:y:0lstm_cell_22/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
lstm_cell_22/mul_1
lstm_cell_22/add_1AddV2lstm_cell_22/mul:z:0lstm_cell_22/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
lstm_cell_22/add_1
lstm_cell_22/Sigmoid_2Sigmoidlstm_cell_22/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
lstm_cell_22/Sigmoid_2}
lstm_cell_22/Relu_1Relulstm_cell_22/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
lstm_cell_22/Relu_1¡
lstm_cell_22/mul_2Mullstm_cell_22/Sigmoid_2:y:0!lstm_cell_22/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
lstm_cell_22/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ¬   2
TensorArrayV2_1/element_shape¸
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
ÿÿÿÿÿÿÿÿÿ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_22_matmul_readvariableop_resource-lstm_cell_22_matmul_1_readvariableop_resource,lstm_cell_22_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ¬: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_39105888*
condR
while_cond_39105887*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ¬: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ¬   22
0TensorArrayV2Stack/TensorListStack/element_shapeé
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
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
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm¦
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
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
:ÿÿÿÿÿÿÿÿÿ¬2

IdentityÈ
NoOpNoOp$^lstm_cell_22/BiasAdd/ReadVariableOp#^lstm_cell_22/MatMul/ReadVariableOp%^lstm_cell_22/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ]: : : 2J
#lstm_cell_22/BiasAdd/ReadVariableOp#lstm_cell_22/BiasAdd/ReadVariableOp2H
"lstm_cell_22/MatMul/ReadVariableOp"lstm_cell_22/MatMul/ReadVariableOp2L
$lstm_cell_22/MatMul_1/ReadVariableOp$lstm_cell_22/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
 
_user_specified_nameinputs
à
º
*__inference_lstm_22_layer_call_fn_39105497
inputs_0
unknown:	]°
	unknown_0:
¬°
	unknown_1:	°
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_lstm_22_layer_call_and_return_conditional_losses_391029162
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ]: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ]
"
_user_specified_name
inputs/0
Ô

í
lstm_23_while_cond_39105021,
(lstm_23_while_lstm_23_while_loop_counter2
.lstm_23_while_lstm_23_while_maximum_iterations
lstm_23_while_placeholder
lstm_23_while_placeholder_1
lstm_23_while_placeholder_2
lstm_23_while_placeholder_3.
*lstm_23_while_less_lstm_23_strided_slice_1F
Blstm_23_while_lstm_23_while_cond_39105021___redundant_placeholder0F
Blstm_23_while_lstm_23_while_cond_39105021___redundant_placeholder1F
Blstm_23_while_lstm_23_while_cond_39105021___redundant_placeholder2F
Blstm_23_while_lstm_23_while_cond_39105021___redundant_placeholder3
lstm_23_while_identity

lstm_23/while/LessLesslstm_23_while_placeholder*lstm_23_while_less_lstm_23_strided_slice_1*
T0*
_output_shapes
: 2
lstm_23/while/Lessu
lstm_23/while/IdentityIdentitylstm_23/while/Less:z:0*
T0
*
_output_shapes
: 2
lstm_23/while/Identity"9
lstm_23_while_identitylstm_23/while/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 
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
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
éJ
Ö

lstm_23_while_body_39105356,
(lstm_23_while_lstm_23_while_loop_counter2
.lstm_23_while_lstm_23_while_maximum_iterations
lstm_23_while_placeholder
lstm_23_while_placeholder_1
lstm_23_while_placeholder_2
lstm_23_while_placeholder_3+
'lstm_23_while_lstm_23_strided_slice_1_0g
clstm_23_while_tensorarrayv2read_tensorlistgetitem_lstm_23_tensorarrayunstack_tensorlistfromtensor_0O
;lstm_23_while_lstm_cell_23_matmul_readvariableop_resource_0:
¬Q
=lstm_23_while_lstm_cell_23_matmul_1_readvariableop_resource_0:
K
<lstm_23_while_lstm_cell_23_biasadd_readvariableop_resource_0:	
lstm_23_while_identity
lstm_23_while_identity_1
lstm_23_while_identity_2
lstm_23_while_identity_3
lstm_23_while_identity_4
lstm_23_while_identity_5)
%lstm_23_while_lstm_23_strided_slice_1e
alstm_23_while_tensorarrayv2read_tensorlistgetitem_lstm_23_tensorarrayunstack_tensorlistfromtensorM
9lstm_23_while_lstm_cell_23_matmul_readvariableop_resource:
¬O
;lstm_23_while_lstm_cell_23_matmul_1_readvariableop_resource:
I
:lstm_23_while_lstm_cell_23_biasadd_readvariableop_resource:	¢1lstm_23/while/lstm_cell_23/BiasAdd/ReadVariableOp¢0lstm_23/while/lstm_cell_23/MatMul/ReadVariableOp¢2lstm_23/while/lstm_cell_23/MatMul_1/ReadVariableOpÓ
?lstm_23/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ¬   2A
?lstm_23/while/TensorArrayV2Read/TensorListGetItem/element_shape
1lstm_23/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemclstm_23_while_tensorarrayv2read_tensorlistgetitem_lstm_23_tensorarrayunstack_tensorlistfromtensor_0lstm_23_while_placeholderHlstm_23/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬*
element_dtype023
1lstm_23/while/TensorArrayV2Read/TensorListGetItemâ
0lstm_23/while/lstm_cell_23/MatMul/ReadVariableOpReadVariableOp;lstm_23_while_lstm_cell_23_matmul_readvariableop_resource_0* 
_output_shapes
:
¬*
dtype022
0lstm_23/while/lstm_cell_23/MatMul/ReadVariableOp÷
!lstm_23/while/lstm_cell_23/MatMulMatMul8lstm_23/while/TensorArrayV2Read/TensorListGetItem:item:08lstm_23/while/lstm_cell_23/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!lstm_23/while/lstm_cell_23/MatMulè
2lstm_23/while/lstm_cell_23/MatMul_1/ReadVariableOpReadVariableOp=lstm_23_while_lstm_cell_23_matmul_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype024
2lstm_23/while/lstm_cell_23/MatMul_1/ReadVariableOpà
#lstm_23/while/lstm_cell_23/MatMul_1MatMullstm_23_while_placeholder_2:lstm_23/while/lstm_cell_23/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2%
#lstm_23/while/lstm_cell_23/MatMul_1Ø
lstm_23/while/lstm_cell_23/addAddV2+lstm_23/while/lstm_cell_23/MatMul:product:0-lstm_23/while/lstm_cell_23/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
lstm_23/while/lstm_cell_23/addà
1lstm_23/while/lstm_cell_23/BiasAdd/ReadVariableOpReadVariableOp<lstm_23_while_lstm_cell_23_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype023
1lstm_23/while/lstm_cell_23/BiasAdd/ReadVariableOpå
"lstm_23/while/lstm_cell_23/BiasAddBiasAdd"lstm_23/while/lstm_cell_23/add:z:09lstm_23/while/lstm_cell_23/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2$
"lstm_23/while/lstm_cell_23/BiasAdd
*lstm_23/while/lstm_cell_23/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2,
*lstm_23/while/lstm_cell_23/split/split_dim¯
 lstm_23/while/lstm_cell_23/splitSplit3lstm_23/while/lstm_cell_23/split/split_dim:output:0+lstm_23/while/lstm_cell_23/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2"
 lstm_23/while/lstm_cell_23/split±
"lstm_23/while/lstm_cell_23/SigmoidSigmoid)lstm_23/while/lstm_cell_23/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2$
"lstm_23/while/lstm_cell_23/Sigmoidµ
$lstm_23/while/lstm_cell_23/Sigmoid_1Sigmoid)lstm_23/while/lstm_cell_23/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$lstm_23/while/lstm_cell_23/Sigmoid_1Á
lstm_23/while/lstm_cell_23/mulMul(lstm_23/while/lstm_cell_23/Sigmoid_1:y:0lstm_23_while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
lstm_23/while/lstm_cell_23/mul¨
lstm_23/while/lstm_cell_23/ReluRelu)lstm_23/while/lstm_cell_23/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
lstm_23/while/lstm_cell_23/ReluÕ
 lstm_23/while/lstm_cell_23/mul_1Mul&lstm_23/while/lstm_cell_23/Sigmoid:y:0-lstm_23/while/lstm_cell_23/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 lstm_23/while/lstm_cell_23/mul_1Ê
 lstm_23/while/lstm_cell_23/add_1AddV2"lstm_23/while/lstm_cell_23/mul:z:0$lstm_23/while/lstm_cell_23/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 lstm_23/while/lstm_cell_23/add_1µ
$lstm_23/while/lstm_cell_23/Sigmoid_2Sigmoid)lstm_23/while/lstm_cell_23/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$lstm_23/while/lstm_cell_23/Sigmoid_2§
!lstm_23/while/lstm_cell_23/Relu_1Relu$lstm_23/while/lstm_cell_23/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!lstm_23/while/lstm_cell_23/Relu_1Ù
 lstm_23/while/lstm_cell_23/mul_2Mul(lstm_23/while/lstm_cell_23/Sigmoid_2:y:0/lstm_23/while/lstm_cell_23/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 lstm_23/while/lstm_cell_23/mul_2
2lstm_23/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_23_while_placeholder_1lstm_23_while_placeholder$lstm_23/while/lstm_cell_23/mul_2:z:0*
_output_shapes
: *
element_dtype024
2lstm_23/while/TensorArrayV2Write/TensorListSetIteml
lstm_23/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_23/while/add/y
lstm_23/while/addAddV2lstm_23_while_placeholderlstm_23/while/add/y:output:0*
T0*
_output_shapes
: 2
lstm_23/while/addp
lstm_23/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_23/while/add_1/y
lstm_23/while/add_1AddV2(lstm_23_while_lstm_23_while_loop_counterlstm_23/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lstm_23/while/add_1
lstm_23/while/IdentityIdentitylstm_23/while/add_1:z:0^lstm_23/while/NoOp*
T0*
_output_shapes
: 2
lstm_23/while/Identity¦
lstm_23/while/Identity_1Identity.lstm_23_while_lstm_23_while_maximum_iterations^lstm_23/while/NoOp*
T0*
_output_shapes
: 2
lstm_23/while/Identity_1
lstm_23/while/Identity_2Identitylstm_23/while/add:z:0^lstm_23/while/NoOp*
T0*
_output_shapes
: 2
lstm_23/while/Identity_2º
lstm_23/while/Identity_3IdentityBlstm_23/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_23/while/NoOp*
T0*
_output_shapes
: 2
lstm_23/while/Identity_3®
lstm_23/while/Identity_4Identity$lstm_23/while/lstm_cell_23/mul_2:z:0^lstm_23/while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_23/while/Identity_4®
lstm_23/while/Identity_5Identity$lstm_23/while/lstm_cell_23/add_1:z:0^lstm_23/while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_23/while/Identity_5
lstm_23/while/NoOpNoOp2^lstm_23/while/lstm_cell_23/BiasAdd/ReadVariableOp1^lstm_23/while/lstm_cell_23/MatMul/ReadVariableOp3^lstm_23/while/lstm_cell_23/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
lstm_23/while/NoOp"9
lstm_23_while_identitylstm_23/while/Identity:output:0"=
lstm_23_while_identity_1!lstm_23/while/Identity_1:output:0"=
lstm_23_while_identity_2!lstm_23/while/Identity_2:output:0"=
lstm_23_while_identity_3!lstm_23/while/Identity_3:output:0"=
lstm_23_while_identity_4!lstm_23/while/Identity_4:output:0"=
lstm_23_while_identity_5!lstm_23/while/Identity_5:output:0"P
%lstm_23_while_lstm_23_strided_slice_1'lstm_23_while_lstm_23_strided_slice_1_0"z
:lstm_23_while_lstm_cell_23_biasadd_readvariableop_resource<lstm_23_while_lstm_cell_23_biasadd_readvariableop_resource_0"|
;lstm_23_while_lstm_cell_23_matmul_1_readvariableop_resource=lstm_23_while_lstm_cell_23_matmul_1_readvariableop_resource_0"x
9lstm_23_while_lstm_cell_23_matmul_readvariableop_resource;lstm_23_while_lstm_cell_23_matmul_readvariableop_resource_0"È
alstm_23_while_tensorarrayv2read_tensorlistgetitem_lstm_23_tensorarrayunstack_tensorlistfromtensorclstm_23_while_tensorarrayv2read_tensorlistgetitem_lstm_23_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2f
1lstm_23/while/lstm_cell_23/BiasAdd/ReadVariableOp1lstm_23/while/lstm_cell_23/BiasAdd/ReadVariableOp2d
0lstm_23/while/lstm_cell_23/MatMul/ReadVariableOp0lstm_23/while/lstm_cell_23/MatMul/ReadVariableOp2h
2lstm_23/while/lstm_cell_23/MatMul_1/ReadVariableOp2lstm_23/while/lstm_cell_23/MatMul_1/ReadVariableOp: 
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
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
ã
Í
while_cond_39105585
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_39105585___redundant_placeholder06
2while_while_cond_39105585___redundant_placeholder16
2while_while_cond_39105585___redundant_placeholder26
2while_while_cond_39105585___redundant_placeholder3
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
B: : : : :ÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ¬: ::::: 
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
:ÿÿÿÿÿÿÿÿÿ¬:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬:

_output_shapes
: :

_output_shapes
:
ã
Í
while_cond_39106038
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_39106038___redundant_placeholder06
2while_while_cond_39106038___redundant_placeholder16
2while_while_cond_39106038___redundant_placeholder26
2while_while_cond_39106038___redundant_placeholder3
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
B: : : : :ÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ¬: ::::: 
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
:ÿÿÿÿÿÿÿÿÿ¬:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬:

_output_shapes
: :

_output_shapes
:
Ê
ú
/__inference_lstm_cell_23_layer_call_fn_39106980

inputs
states_0
states_1
unknown:
¬
	unknown_0:

	unknown_1:	
identity

identity_1

identity_2¢StatefulPartitionedCallÈ
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_lstm_cell_23_layer_call_and_return_conditional_losses_391032532
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

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
B:ÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
 
_user_specified_nameinputs:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
states/0:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
states/1
ã
Í
while_cond_39102636
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_39102636___redundant_placeholder06
2while_while_cond_39102636___redundant_placeholder16
2while_while_cond_39102636___redundant_placeholder26
2while_while_cond_39102636___redundant_placeholder3
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
B: : : : :ÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ¬: ::::: 
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
:ÿÿÿÿÿÿÿÿÿ¬:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬:

_output_shapes
: :

_output_shapes
:
Ç
ù
/__inference_lstm_cell_22_layer_call_fn_39106899

inputs
states_0
states_1
unknown:	]°
	unknown_0:
¬°
	unknown_1:	°
identity

identity_1

identity_2¢StatefulPartitionedCallÈ
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ¬*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_lstm_cell_22_layer_call_and_return_conditional_losses_391027692
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2

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
A:ÿÿÿÿÿÿÿÿÿ]:ÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ¬: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
 
_user_specified_nameinputs:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
"
_user_specified_name
states/0:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
"
_user_specified_name
states/1
®

Ô
0__inference_sequential_11_layer_call_fn_39104202
lstm_22_input
unknown:	]°
	unknown_0:
¬°
	unknown_1:	°
	unknown_2:
¬
	unknown_3:

	unknown_4:	
	unknown_5:	
	unknown_6:
identity¢StatefulPartitionedCallÔ
StatefulPartitionedCallStatefulPartitionedCalllstm_22_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_sequential_11_layer_call_and_return_conditional_losses_391041832
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ]: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
'
_user_specified_namelstm_22_input
¨
ñ
K__inference_sequential_11_layer_call_and_return_conditional_losses_39104183

inputs#
lstm_22_39103966:	]°$
lstm_22_39103968:
¬°
lstm_22_39103970:	°$
lstm_23_39104131:
¬$
lstm_23_39104133:

lstm_23_39104135:	$
dense_11_39104177:	
dense_11_39104179:
identity¢ dense_11/StatefulPartitionedCall¢lstm_22/StatefulPartitionedCall¢lstm_23/StatefulPartitionedCall®
lstm_22/StatefulPartitionedCallStatefulPartitionedCallinputslstm_22_39103966lstm_22_39103968lstm_22_39103970*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_lstm_22_layer_call_and_return_conditional_losses_391039652!
lstm_22/StatefulPartitionedCall
dropout_22/PartitionedCallPartitionedCall(lstm_22/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_dropout_22_layer_call_and_return_conditional_losses_391039782
dropout_22/PartitionedCallË
lstm_23/StatefulPartitionedCallStatefulPartitionedCall#dropout_22/PartitionedCall:output:0lstm_23_39104131lstm_23_39104133lstm_23_39104135*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_lstm_23_layer_call_and_return_conditional_losses_391041302!
lstm_23/StatefulPartitionedCall
dropout_23/PartitionedCallPartitionedCall(lstm_23/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_dropout_23_layer_call_and_return_conditional_losses_391041432
dropout_23/PartitionedCall»
 dense_11/StatefulPartitionedCallStatefulPartitionedCall#dropout_23/PartitionedCall:output:0dense_11_39104177dense_11_39104179*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_11_layer_call_and_return_conditional_losses_391041762"
 dense_11/StatefulPartitionedCall
IdentityIdentity)dense_11/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityµ
NoOpNoOp!^dense_11/StatefulPartitionedCall ^lstm_22/StatefulPartitionedCall ^lstm_23/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ]: : : : : : : : 2D
 dense_11/StatefulPartitionedCall dense_11/StatefulPartitionedCall2B
lstm_22/StatefulPartitionedCalllstm_22/StatefulPartitionedCall2B
lstm_23/StatefulPartitionedCalllstm_23/StatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
 
_user_specified_nameinputs
Ê
ú
/__inference_lstm_cell_23_layer_call_fn_39106997

inputs
states_0
states_1
unknown:
¬
	unknown_0:

	unknown_1:	
identity

identity_1

identity_2¢StatefulPartitionedCallÈ
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_lstm_cell_23_layer_call_and_return_conditional_losses_391033992
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

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
B:ÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
 
_user_specified_nameinputs:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
states/0:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
states/1
½
ø
K__inference_sequential_11_layer_call_and_return_conditional_losses_39104717
lstm_22_input#
lstm_22_39104695:	]°$
lstm_22_39104697:
¬°
lstm_22_39104699:	°$
lstm_23_39104703:
¬$
lstm_23_39104705:

lstm_23_39104707:	$
dense_11_39104711:	
dense_11_39104713:
identity¢ dense_11/StatefulPartitionedCall¢lstm_22/StatefulPartitionedCall¢lstm_23/StatefulPartitionedCallµ
lstm_22/StatefulPartitionedCallStatefulPartitionedCalllstm_22_inputlstm_22_39104695lstm_22_39104697lstm_22_39104699*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_lstm_22_layer_call_and_return_conditional_losses_391039652!
lstm_22/StatefulPartitionedCall
dropout_22/PartitionedCallPartitionedCall(lstm_22/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_dropout_22_layer_call_and_return_conditional_losses_391039782
dropout_22/PartitionedCallË
lstm_23/StatefulPartitionedCallStatefulPartitionedCall#dropout_22/PartitionedCall:output:0lstm_23_39104703lstm_23_39104705lstm_23_39104707*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_lstm_23_layer_call_and_return_conditional_losses_391041302!
lstm_23/StatefulPartitionedCall
dropout_23/PartitionedCallPartitionedCall(lstm_23/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_dropout_23_layer_call_and_return_conditional_losses_391041432
dropout_23/PartitionedCall»
 dense_11/StatefulPartitionedCallStatefulPartitionedCall#dropout_23/PartitionedCall:output:0dense_11_39104711dense_11_39104713*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_11_layer_call_and_return_conditional_losses_391041762"
 dense_11/StatefulPartitionedCall
IdentityIdentity)dense_11/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityµ
NoOpNoOp!^dense_11/StatefulPartitionedCall ^lstm_22/StatefulPartitionedCall ^lstm_23/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ]: : : : : : : : 2D
 dense_11/StatefulPartitionedCall dense_11/StatefulPartitionedCall2B
lstm_22/StatefulPartitionedCalllstm_22/StatefulPartitionedCall2B
lstm_23/StatefulPartitionedCalllstm_23/StatefulPartitionedCall:Z V
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
'
_user_specified_namelstm_22_input


J__inference_lstm_cell_23_layer_call_and_return_conditional_losses_39103399

inputs

states
states_12
matmul_readvariableop_resource:
¬4
 matmul_1_readvariableop_resource:
.
biasadd_readvariableop_resource:	
identity

identity_1

identity_2¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
¬*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
add
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dimÃ
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2
split`
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
Sigmoidd
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	Sigmoid_1]
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mulW
ReluRelusplit:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relui
mul_1MulSigmoid:y:0Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mul_1^
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
add_1d
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	Sigmoid_2V
Relu_1Relu	add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu_1m
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mul_2e
IdentityIdentity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityi

Identity_1Identity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1i

Identity_2Identity	add_1:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_2
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
B:ÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
 
_user_specified_nameinputs:PL
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_namestates:PL
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_namestates
¹
¹
*__inference_lstm_23_layer_call_fn_39106194

inputs
unknown:
¬
	unknown_0:

	unknown_1:	
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_lstm_23_layer_call_and_return_conditional_losses_391043992
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:ÿÿÿÿÿÿÿÿÿ¬: : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
 
_user_specified_nameinputs
ã
Í
while_cond_39103266
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_39103266___redundant_placeholder06
2while_while_cond_39103266___redundant_placeholder16
2while_while_cond_39103266___redundant_placeholder26
2while_while_cond_39103266___redundant_placeholder3
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
B: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 
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
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:

f
H__inference_dropout_22_layer_call_and_return_conditional_losses_39103978

inputs

identity_1_
IdentityIdentityinputs*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2

Identityn

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ¬:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
 
_user_specified_nameinputs
\

E__inference_lstm_23_layer_call_and_return_conditional_losses_39104130

inputs?
+lstm_cell_23_matmul_readvariableop_resource:
¬A
-lstm_cell_23_matmul_1_readvariableop_resource:
;
,lstm_cell_23_biasadd_readvariableop_resource:	
identity¢#lstm_cell_23/BiasAdd/ReadVariableOp¢"lstm_cell_23/MatMul/ReadVariableOp¢$lstm_cell_23/MatMul_1/ReadVariableOp¢whileD
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
strided_slice/stack_2â
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
B :2
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
B :è2
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
B :2
zeros/packed/1
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
:ÿÿÿÿÿÿÿÿÿ2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :2
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
B :è2
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
B :2
zeros_1/packed/1
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
:ÿÿÿÿÿÿÿÿÿ2	
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
:ÿÿÿÿÿÿÿÿÿ¬2
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
strided_slice_1/stack_2î
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
TensorArrayV2/element_shape²
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2¿
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ¬   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeø
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
strided_slice_2/stack_2ý
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬*
shrink_axis_mask2
strided_slice_2¶
"lstm_cell_23/MatMul/ReadVariableOpReadVariableOp+lstm_cell_23_matmul_readvariableop_resource* 
_output_shapes
:
¬*
dtype02$
"lstm_cell_23/MatMul/ReadVariableOp­
lstm_cell_23/MatMulMatMulstrided_slice_2:output:0*lstm_cell_23/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_23/MatMul¼
$lstm_cell_23/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_23_matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype02&
$lstm_cell_23/MatMul_1/ReadVariableOp©
lstm_cell_23/MatMul_1MatMulzeros:output:0,lstm_cell_23/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_23/MatMul_1 
lstm_cell_23/addAddV2lstm_cell_23/MatMul:product:0lstm_cell_23/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_23/add´
#lstm_cell_23/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_23_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02%
#lstm_cell_23/BiasAdd/ReadVariableOp­
lstm_cell_23/BiasAddBiasAddlstm_cell_23/add:z:0+lstm_cell_23/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_23/BiasAdd~
lstm_cell_23/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_23/split/split_dim÷
lstm_cell_23/splitSplit%lstm_cell_23/split/split_dim:output:0lstm_cell_23/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2
lstm_cell_23/split
lstm_cell_23/SigmoidSigmoidlstm_cell_23/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_23/Sigmoid
lstm_cell_23/Sigmoid_1Sigmoidlstm_cell_23/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_23/Sigmoid_1
lstm_cell_23/mulMullstm_cell_23/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_23/mul~
lstm_cell_23/ReluRelulstm_cell_23/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_23/Relu
lstm_cell_23/mul_1Mullstm_cell_23/Sigmoid:y:0lstm_cell_23/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_23/mul_1
lstm_cell_23/add_1AddV2lstm_cell_23/mul:z:0lstm_cell_23/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_23/add_1
lstm_cell_23/Sigmoid_2Sigmoidlstm_cell_23/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_23/Sigmoid_2}
lstm_cell_23/Relu_1Relulstm_cell_23/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_23/Relu_1¡
lstm_cell_23/mul_2Mullstm_cell_23/Sigmoid_2:y:0!lstm_cell_23/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_23/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
TensorArrayV2_1/element_shape¸
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
ÿÿÿÿÿÿÿÿÿ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_23_matmul_readvariableop_resource-lstm_cell_23_matmul_1_readvariableop_resource,lstm_cell_23_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_39104046*
condR
while_cond_39104045*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   22
0TensorArrayV2Stack/TensorListStack/element_shapeé
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
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
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm¦
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
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
:ÿÿÿÿÿÿÿÿÿ2

IdentityÈ
NoOpNoOp$^lstm_cell_23/BiasAdd/ReadVariableOp#^lstm_cell_23/MatMul/ReadVariableOp%^lstm_cell_23/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:ÿÿÿÿÿÿÿÿÿ¬: : : 2J
#lstm_cell_23/BiasAdd/ReadVariableOp#lstm_cell_23/BiasAdd/ReadVariableOp2H
"lstm_cell_23/MatMul/ReadVariableOp"lstm_cell_23/MatMul/ReadVariableOp2L
$lstm_cell_23/MatMul_1/ReadVariableOp$lstm_cell_23/MatMul_1/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
 
_user_specified_nameinputs
®

Ô
0__inference_sequential_11_layer_call_fn_39104692
lstm_22_input
unknown:	]°
	unknown_0:
¬°
	unknown_1:	°
	unknown_2:
¬
	unknown_3:

	unknown_4:	
	unknown_5:	
	unknown_6:
identity¢StatefulPartitionedCallÔ
StatefulPartitionedCallStatefulPartitionedCalllstm_22_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_sequential_11_layer_call_and_return_conditional_losses_391046522
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ]: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
'
_user_specified_namelstm_22_input
Ô

í
lstm_22_while_cond_39105200,
(lstm_22_while_lstm_22_while_loop_counter2
.lstm_22_while_lstm_22_while_maximum_iterations
lstm_22_while_placeholder
lstm_22_while_placeholder_1
lstm_22_while_placeholder_2
lstm_22_while_placeholder_3.
*lstm_22_while_less_lstm_22_strided_slice_1F
Blstm_22_while_lstm_22_while_cond_39105200___redundant_placeholder0F
Blstm_22_while_lstm_22_while_cond_39105200___redundant_placeholder1F
Blstm_22_while_lstm_22_while_cond_39105200___redundant_placeholder2F
Blstm_22_while_lstm_22_while_cond_39105200___redundant_placeholder3
lstm_22_while_identity

lstm_22/while/LessLesslstm_22_while_placeholder*lstm_22_while_less_lstm_22_strided_slice_1*
T0*
_output_shapes
: 2
lstm_22/while/Lessu
lstm_22/while/IdentityIdentitylstm_22/while/Less:z:0*
T0
*
_output_shapes
: 2
lstm_22/while/Identity"9
lstm_22_while_identitylstm_22/while/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :ÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ¬: ::::: 
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
:ÿÿÿÿÿÿÿÿÿ¬:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬:

_output_shapes
: :

_output_shapes
:


J__inference_lstm_cell_23_layer_call_and_return_conditional_losses_39107061

inputs
states_0
states_12
matmul_readvariableop_resource:
¬4
 matmul_1_readvariableop_resource:
.
biasadd_readvariableop_resource:	
identity

identity_1

identity_2¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
¬*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul_1/ReadVariableOp|
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
add
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dimÃ
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2
split`
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
Sigmoidd
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	Sigmoid_1]
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mulW
ReluRelusplit:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relui
mul_1MulSigmoid:y:0Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mul_1^
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
add_1d
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	Sigmoid_2V
Relu_1Relu	add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu_1m
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mul_2e
IdentityIdentity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityi

Identity_1Identity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1i

Identity_2Identity	add_1:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_2
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
B:ÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
 
_user_specified_nameinputs:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
states/0:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
states/1
\

E__inference_lstm_22_layer_call_and_return_conditional_losses_39106123

inputs>
+lstm_cell_22_matmul_readvariableop_resource:	]°A
-lstm_cell_22_matmul_1_readvariableop_resource:
¬°;
,lstm_cell_22_biasadd_readvariableop_resource:	°
identity¢#lstm_cell_22/BiasAdd/ReadVariableOp¢"lstm_cell_22/MatMul/ReadVariableOp¢$lstm_cell_22/MatMul_1/ReadVariableOp¢whileD
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
strided_slice/stack_2â
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
B :¬2
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
B :è2
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
B :¬2
zeros/packed/1
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
:ÿÿÿÿÿÿÿÿÿ¬2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :¬2
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
B :è2
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
B :¬2
zeros_1/packed/1
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
:ÿÿÿÿÿÿÿÿÿ¬2	
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
:ÿÿÿÿÿÿÿÿÿ]2
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
strided_slice_1/stack_2î
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
TensorArrayV2/element_shape²
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2¿
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ]   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeø
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
strided_slice_2/stack_2ü
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]*
shrink_axis_mask2
strided_slice_2µ
"lstm_cell_22/MatMul/ReadVariableOpReadVariableOp+lstm_cell_22_matmul_readvariableop_resource*
_output_shapes
:	]°*
dtype02$
"lstm_cell_22/MatMul/ReadVariableOp­
lstm_cell_22/MatMulMatMulstrided_slice_2:output:0*lstm_cell_22/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°2
lstm_cell_22/MatMul¼
$lstm_cell_22/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_22_matmul_1_readvariableop_resource* 
_output_shapes
:
¬°*
dtype02&
$lstm_cell_22/MatMul_1/ReadVariableOp©
lstm_cell_22/MatMul_1MatMulzeros:output:0,lstm_cell_22/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°2
lstm_cell_22/MatMul_1 
lstm_cell_22/addAddV2lstm_cell_22/MatMul:product:0lstm_cell_22/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°2
lstm_cell_22/add´
#lstm_cell_22/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_22_biasadd_readvariableop_resource*
_output_shapes	
:°*
dtype02%
#lstm_cell_22/BiasAdd/ReadVariableOp­
lstm_cell_22/BiasAddBiasAddlstm_cell_22/add:z:0+lstm_cell_22/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°2
lstm_cell_22/BiasAdd~
lstm_cell_22/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_22/split/split_dim÷
lstm_cell_22/splitSplit%lstm_cell_22/split/split_dim:output:0lstm_cell_22/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ¬*
	num_split2
lstm_cell_22/split
lstm_cell_22/SigmoidSigmoidlstm_cell_22/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
lstm_cell_22/Sigmoid
lstm_cell_22/Sigmoid_1Sigmoidlstm_cell_22/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
lstm_cell_22/Sigmoid_1
lstm_cell_22/mulMullstm_cell_22/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
lstm_cell_22/mul~
lstm_cell_22/ReluRelulstm_cell_22/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
lstm_cell_22/Relu
lstm_cell_22/mul_1Mullstm_cell_22/Sigmoid:y:0lstm_cell_22/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
lstm_cell_22/mul_1
lstm_cell_22/add_1AddV2lstm_cell_22/mul:z:0lstm_cell_22/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
lstm_cell_22/add_1
lstm_cell_22/Sigmoid_2Sigmoidlstm_cell_22/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
lstm_cell_22/Sigmoid_2}
lstm_cell_22/Relu_1Relulstm_cell_22/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
lstm_cell_22/Relu_1¡
lstm_cell_22/mul_2Mullstm_cell_22/Sigmoid_2:y:0!lstm_cell_22/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
lstm_cell_22/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ¬   2
TensorArrayV2_1/element_shape¸
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
ÿÿÿÿÿÿÿÿÿ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_22_matmul_readvariableop_resource-lstm_cell_22_matmul_1_readvariableop_resource,lstm_cell_22_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ¬: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_39106039*
condR
while_cond_39106038*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ¬: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ¬   22
0TensorArrayV2Stack/TensorListStack/element_shapeé
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
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
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm¦
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
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
:ÿÿÿÿÿÿÿÿÿ¬2

IdentityÈ
NoOpNoOp$^lstm_cell_22/BiasAdd/ReadVariableOp#^lstm_cell_22/MatMul/ReadVariableOp%^lstm_cell_22/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ]: : : 2J
#lstm_cell_22/BiasAdd/ReadVariableOp#lstm_cell_22/BiasAdd/ReadVariableOp2H
"lstm_cell_22/MatMul/ReadVariableOp"lstm_cell_22/MatMul/ReadVariableOp2L
$lstm_cell_22/MatMul_1/ReadVariableOp$lstm_cell_22/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
 
_user_specified_nameinputs
ã
Í
while_cond_39106411
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_39106411___redundant_placeholder06
2while_while_cond_39106411___redundant_placeholder16
2while_while_cond_39106411___redundant_placeholder26
2while_while_cond_39106411___redundant_placeholder3
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
B: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 
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
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
Ç
ù
/__inference_lstm_cell_22_layer_call_fn_39106882

inputs
states_0
states_1
unknown:	]°
	unknown_0:
¬°
	unknown_1:	°
identity

identity_1

identity_2¢StatefulPartitionedCallÈ
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ¬*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_lstm_cell_22_layer_call_and_return_conditional_losses_391026232
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2

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
A:ÿÿÿÿÿÿÿÿÿ]:ÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ¬: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
 
_user_specified_nameinputs:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
"
_user_specified_name
states/0:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
"
_user_specified_name
states/1
»
f
-__inference_dropout_22_layer_call_fn_39106133

inputs
identity¢StatefulPartitionedCallã
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_dropout_22_layer_call_and_return_conditional_losses_391044282
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ¬22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
 
_user_specified_nameinputs
&
ó
while_body_39102637
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_00
while_lstm_cell_22_39102661_0:	]°1
while_lstm_cell_22_39102663_0:
¬°,
while_lstm_cell_22_39102665_0:	°
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor.
while_lstm_cell_22_39102661:	]°/
while_lstm_cell_22_39102663:
¬°*
while_lstm_cell_22_39102665:	°¢*while/lstm_cell_22/StatefulPartitionedCallÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ]   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÓ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemì
*while/lstm_cell_22/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_22_39102661_0while_lstm_cell_22_39102663_0while_lstm_cell_22_39102665_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ¬*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_lstm_cell_22_layer_call_and_return_conditional_losses_391026232,
*while/lstm_cell_22/StatefulPartitionedCall÷
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder3while/lstm_cell_22/StatefulPartitionedCall:output:0*
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
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3¥
while/Identity_4Identity3while/lstm_cell_22/StatefulPartitionedCall:output:1^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
while/Identity_4¥
while/Identity_5Identity3while/lstm_cell_22/StatefulPartitionedCall:output:2^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
while/Identity_5

while/NoOpNoOp+^while/lstm_cell_22/StatefulPartitionedCall*"
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
while_lstm_cell_22_39102661while_lstm_cell_22_39102661_0"<
while_lstm_cell_22_39102663while_lstm_cell_22_39102663_0"<
while_lstm_cell_22_39102665while_lstm_cell_22_39102665_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ¬: : : : : 2X
*while/lstm_cell_22/StatefulPartitionedCall*while/lstm_cell_22/StatefulPartitionedCall: 
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
:ÿÿÿÿÿÿÿÿÿ¬:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬:

_output_shapes
: :

_output_shapes
: 
´?
Ö
while_body_39106563
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
3while_lstm_cell_23_matmul_readvariableop_resource_0:
¬I
5while_lstm_cell_23_matmul_1_readvariableop_resource_0:
C
4while_lstm_cell_23_biasadd_readvariableop_resource_0:	
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
1while_lstm_cell_23_matmul_readvariableop_resource:
¬G
3while_lstm_cell_23_matmul_1_readvariableop_resource:
A
2while_lstm_cell_23_biasadd_readvariableop_resource:	¢)while/lstm_cell_23/BiasAdd/ReadVariableOp¢(while/lstm_cell_23/MatMul/ReadVariableOp¢*while/lstm_cell_23/MatMul_1/ReadVariableOpÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ¬   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÔ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemÊ
(while/lstm_cell_23/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_23_matmul_readvariableop_resource_0* 
_output_shapes
:
¬*
dtype02*
(while/lstm_cell_23/MatMul/ReadVariableOp×
while/lstm_cell_23/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_23/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_23/MatMulÐ
*while/lstm_cell_23/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_23_matmul_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype02,
*while/lstm_cell_23/MatMul_1/ReadVariableOpÀ
while/lstm_cell_23/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_23/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_23/MatMul_1¸
while/lstm_cell_23/addAddV2#while/lstm_cell_23/MatMul:product:0%while/lstm_cell_23/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_23/addÈ
)while/lstm_cell_23/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_23_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02+
)while/lstm_cell_23/BiasAdd/ReadVariableOpÅ
while/lstm_cell_23/BiasAddBiasAddwhile/lstm_cell_23/add:z:01while/lstm_cell_23/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_23/BiasAdd
"while/lstm_cell_23/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_23/split/split_dim
while/lstm_cell_23/splitSplit+while/lstm_cell_23/split/split_dim:output:0#while/lstm_cell_23/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2
while/lstm_cell_23/split
while/lstm_cell_23/SigmoidSigmoid!while/lstm_cell_23/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_23/Sigmoid
while/lstm_cell_23/Sigmoid_1Sigmoid!while/lstm_cell_23/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_23/Sigmoid_1¡
while/lstm_cell_23/mulMul while/lstm_cell_23/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_23/mul
while/lstm_cell_23/ReluRelu!while/lstm_cell_23/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_23/Reluµ
while/lstm_cell_23/mul_1Mulwhile/lstm_cell_23/Sigmoid:y:0%while/lstm_cell_23/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_23/mul_1ª
while/lstm_cell_23/add_1AddV2while/lstm_cell_23/mul:z:0while/lstm_cell_23/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_23/add_1
while/lstm_cell_23/Sigmoid_2Sigmoid!while/lstm_cell_23/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_23/Sigmoid_2
while/lstm_cell_23/Relu_1Reluwhile/lstm_cell_23/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_23/Relu_1¹
while/lstm_cell_23/mul_2Mul while/lstm_cell_23/Sigmoid_2:y:0'while/lstm_cell_23/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_23/mul_2à
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_23/mul_2:z:0*
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
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_23/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_23/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/Identity_5Þ

while/NoOpNoOp*^while/lstm_cell_23/BiasAdd/ReadVariableOp)^while/lstm_cell_23/MatMul/ReadVariableOp+^while/lstm_cell_23/MatMul_1/ReadVariableOp*"
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
2while_lstm_cell_23_biasadd_readvariableop_resource4while_lstm_cell_23_biasadd_readvariableop_resource_0"l
3while_lstm_cell_23_matmul_1_readvariableop_resource5while_lstm_cell_23_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_23_matmul_readvariableop_resource3while_lstm_cell_23_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2V
)while/lstm_cell_23/BiasAdd/ReadVariableOp)while/lstm_cell_23/BiasAdd/ReadVariableOp2T
(while/lstm_cell_23/MatMul/ReadVariableOp(while/lstm_cell_23/MatMul/ReadVariableOp2X
*while/lstm_cell_23/MatMul_1/ReadVariableOp*while/lstm_cell_23/MatMul_1/ReadVariableOp: 
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
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
×
g
H__inference_dropout_22_layer_call_and_return_conditional_losses_39106150

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *«ªª?2
dropout/Constx
dropout/MulMulinputsdropout/Const:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape¹
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  >2
dropout/GreaterEqual/yÃ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
dropout/Mul_1j
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ¬:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
 
_user_specified_nameinputs
Õ!
þ
F__inference_dense_11_layer_call_and_return_conditional_losses_39106865

inputs4
!tensordot_readvariableop_resource:	-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢Tensordot/ReadVariableOp
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes
:	*
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
Tensordot/GatherV2/axisÑ
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
Tensordot/GatherV2_1/axis×
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
Tensordot/Const
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
Tensordot/Const_1
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
Tensordot/concat/axis°
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Tensordot/transpose
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Tensordot/Reshape
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
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
Tensordot/concat_1/axis½
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	Tensordot
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdde
SoftmaxSoftmaxBiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
Softmaxp
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
´?
Ö
while_body_39104315
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
3while_lstm_cell_23_matmul_readvariableop_resource_0:
¬I
5while_lstm_cell_23_matmul_1_readvariableop_resource_0:
C
4while_lstm_cell_23_biasadd_readvariableop_resource_0:	
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
1while_lstm_cell_23_matmul_readvariableop_resource:
¬G
3while_lstm_cell_23_matmul_1_readvariableop_resource:
A
2while_lstm_cell_23_biasadd_readvariableop_resource:	¢)while/lstm_cell_23/BiasAdd/ReadVariableOp¢(while/lstm_cell_23/MatMul/ReadVariableOp¢*while/lstm_cell_23/MatMul_1/ReadVariableOpÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ¬   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÔ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemÊ
(while/lstm_cell_23/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_23_matmul_readvariableop_resource_0* 
_output_shapes
:
¬*
dtype02*
(while/lstm_cell_23/MatMul/ReadVariableOp×
while/lstm_cell_23/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_23/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_23/MatMulÐ
*while/lstm_cell_23/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_23_matmul_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype02,
*while/lstm_cell_23/MatMul_1/ReadVariableOpÀ
while/lstm_cell_23/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_23/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_23/MatMul_1¸
while/lstm_cell_23/addAddV2#while/lstm_cell_23/MatMul:product:0%while/lstm_cell_23/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_23/addÈ
)while/lstm_cell_23/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_23_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02+
)while/lstm_cell_23/BiasAdd/ReadVariableOpÅ
while/lstm_cell_23/BiasAddBiasAddwhile/lstm_cell_23/add:z:01while/lstm_cell_23/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_23/BiasAdd
"while/lstm_cell_23/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_23/split/split_dim
while/lstm_cell_23/splitSplit+while/lstm_cell_23/split/split_dim:output:0#while/lstm_cell_23/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2
while/lstm_cell_23/split
while/lstm_cell_23/SigmoidSigmoid!while/lstm_cell_23/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_23/Sigmoid
while/lstm_cell_23/Sigmoid_1Sigmoid!while/lstm_cell_23/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_23/Sigmoid_1¡
while/lstm_cell_23/mulMul while/lstm_cell_23/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_23/mul
while/lstm_cell_23/ReluRelu!while/lstm_cell_23/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_23/Reluµ
while/lstm_cell_23/mul_1Mulwhile/lstm_cell_23/Sigmoid:y:0%while/lstm_cell_23/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_23/mul_1ª
while/lstm_cell_23/add_1AddV2while/lstm_cell_23/mul:z:0while/lstm_cell_23/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_23/add_1
while/lstm_cell_23/Sigmoid_2Sigmoid!while/lstm_cell_23/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_23/Sigmoid_2
while/lstm_cell_23/Relu_1Reluwhile/lstm_cell_23/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_23/Relu_1¹
while/lstm_cell_23/mul_2Mul while/lstm_cell_23/Sigmoid_2:y:0'while/lstm_cell_23/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_23/mul_2à
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_23/mul_2:z:0*
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
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_23/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_23/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/Identity_5Þ

while/NoOpNoOp*^while/lstm_cell_23/BiasAdd/ReadVariableOp)^while/lstm_cell_23/MatMul/ReadVariableOp+^while/lstm_cell_23/MatMul_1/ReadVariableOp*"
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
2while_lstm_cell_23_biasadd_readvariableop_resource4while_lstm_cell_23_biasadd_readvariableop_resource_0"l
3while_lstm_cell_23_matmul_1_readvariableop_resource5while_lstm_cell_23_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_23_matmul_readvariableop_resource3while_lstm_cell_23_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2V
)while/lstm_cell_23/BiasAdd/ReadVariableOp)while/lstm_cell_23/BiasAdd/ReadVariableOp2T
(while/lstm_cell_23/MatMul/ReadVariableOp(while/lstm_cell_23/MatMul/ReadVariableOp2X
*while/lstm_cell_23/MatMul_1/ReadVariableOp*while/lstm_cell_23/MatMul_1/ReadVariableOp: 
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
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
´?
Ö
while_body_39106412
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
3while_lstm_cell_23_matmul_readvariableop_resource_0:
¬I
5while_lstm_cell_23_matmul_1_readvariableop_resource_0:
C
4while_lstm_cell_23_biasadd_readvariableop_resource_0:	
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
1while_lstm_cell_23_matmul_readvariableop_resource:
¬G
3while_lstm_cell_23_matmul_1_readvariableop_resource:
A
2while_lstm_cell_23_biasadd_readvariableop_resource:	¢)while/lstm_cell_23/BiasAdd/ReadVariableOp¢(while/lstm_cell_23/MatMul/ReadVariableOp¢*while/lstm_cell_23/MatMul_1/ReadVariableOpÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ¬   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÔ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemÊ
(while/lstm_cell_23/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_23_matmul_readvariableop_resource_0* 
_output_shapes
:
¬*
dtype02*
(while/lstm_cell_23/MatMul/ReadVariableOp×
while/lstm_cell_23/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_23/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_23/MatMulÐ
*while/lstm_cell_23/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_23_matmul_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype02,
*while/lstm_cell_23/MatMul_1/ReadVariableOpÀ
while/lstm_cell_23/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_23/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_23/MatMul_1¸
while/lstm_cell_23/addAddV2#while/lstm_cell_23/MatMul:product:0%while/lstm_cell_23/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_23/addÈ
)while/lstm_cell_23/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_23_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02+
)while/lstm_cell_23/BiasAdd/ReadVariableOpÅ
while/lstm_cell_23/BiasAddBiasAddwhile/lstm_cell_23/add:z:01while/lstm_cell_23/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_23/BiasAdd
"while/lstm_cell_23/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_23/split/split_dim
while/lstm_cell_23/splitSplit+while/lstm_cell_23/split/split_dim:output:0#while/lstm_cell_23/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2
while/lstm_cell_23/split
while/lstm_cell_23/SigmoidSigmoid!while/lstm_cell_23/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_23/Sigmoid
while/lstm_cell_23/Sigmoid_1Sigmoid!while/lstm_cell_23/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_23/Sigmoid_1¡
while/lstm_cell_23/mulMul while/lstm_cell_23/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_23/mul
while/lstm_cell_23/ReluRelu!while/lstm_cell_23/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_23/Reluµ
while/lstm_cell_23/mul_1Mulwhile/lstm_cell_23/Sigmoid:y:0%while/lstm_cell_23/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_23/mul_1ª
while/lstm_cell_23/add_1AddV2while/lstm_cell_23/mul:z:0while/lstm_cell_23/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_23/add_1
while/lstm_cell_23/Sigmoid_2Sigmoid!while/lstm_cell_23/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_23/Sigmoid_2
while/lstm_cell_23/Relu_1Reluwhile/lstm_cell_23/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_23/Relu_1¹
while/lstm_cell_23/mul_2Mul while/lstm_cell_23/Sigmoid_2:y:0'while/lstm_cell_23/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_23/mul_2à
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_23/mul_2:z:0*
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
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_23/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_23/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/Identity_5Þ

while/NoOpNoOp*^while/lstm_cell_23/BiasAdd/ReadVariableOp)^while/lstm_cell_23/MatMul/ReadVariableOp+^while/lstm_cell_23/MatMul_1/ReadVariableOp*"
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
2while_lstm_cell_23_biasadd_readvariableop_resource4while_lstm_cell_23_biasadd_readvariableop_resource_0"l
3while_lstm_cell_23_matmul_1_readvariableop_resource5while_lstm_cell_23_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_23_matmul_readvariableop_resource3while_lstm_cell_23_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2V
)while/lstm_cell_23/BiasAdd/ReadVariableOp)while/lstm_cell_23/BiasAdd/ReadVariableOp2T
(while/lstm_cell_23/MatMul/ReadVariableOp(while/lstm_cell_23/MatMul/ReadVariableOp2X
*while/lstm_cell_23/MatMul_1/ReadVariableOp*while/lstm_cell_23/MatMul_1/ReadVariableOp: 
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
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 


J__inference_lstm_cell_22_layer_call_and_return_conditional_losses_39106963

inputs
states_0
states_11
matmul_readvariableop_resource:	]°4
 matmul_1_readvariableop_resource:
¬°.
biasadd_readvariableop_resource:	°
identity

identity_1

identity_2¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	]°*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°2
MatMul
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
¬°*
dtype02
MatMul_1/ReadVariableOp|
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°2
add
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:°*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°2	
BiasAddd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dimÃ
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ¬*
	num_split2
split`
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2	
Sigmoidd
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
	Sigmoid_1]
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
mulW
ReluRelusplit:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
Relui
mul_1MulSigmoid:y:0Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
mul_1^
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
add_1d
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
	Sigmoid_2V
Relu_1Relu	add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
Relu_1m
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
mul_2e
IdentityIdentity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2

Identityi

Identity_1Identity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2

Identity_1i

Identity_2Identity	add_1:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2

Identity_2
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
A:ÿÿÿÿÿÿÿÿÿ]:ÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ¬: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
 
_user_specified_nameinputs:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
"
_user_specified_name
states/0:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
"
_user_specified_name
states/1
ù

)sequential_11_lstm_22_while_cond_39102287H
Dsequential_11_lstm_22_while_sequential_11_lstm_22_while_loop_counterN
Jsequential_11_lstm_22_while_sequential_11_lstm_22_while_maximum_iterations+
'sequential_11_lstm_22_while_placeholder-
)sequential_11_lstm_22_while_placeholder_1-
)sequential_11_lstm_22_while_placeholder_2-
)sequential_11_lstm_22_while_placeholder_3J
Fsequential_11_lstm_22_while_less_sequential_11_lstm_22_strided_slice_1b
^sequential_11_lstm_22_while_sequential_11_lstm_22_while_cond_39102287___redundant_placeholder0b
^sequential_11_lstm_22_while_sequential_11_lstm_22_while_cond_39102287___redundant_placeholder1b
^sequential_11_lstm_22_while_sequential_11_lstm_22_while_cond_39102287___redundant_placeholder2b
^sequential_11_lstm_22_while_sequential_11_lstm_22_while_cond_39102287___redundant_placeholder3(
$sequential_11_lstm_22_while_identity
Þ
 sequential_11/lstm_22/while/LessLess'sequential_11_lstm_22_while_placeholderFsequential_11_lstm_22_while_less_sequential_11_lstm_22_strided_slice_1*
T0*
_output_shapes
: 2"
 sequential_11/lstm_22/while/Less
$sequential_11/lstm_22/while/IdentityIdentity$sequential_11/lstm_22/while/Less:z:0*
T0
*
_output_shapes
: 2&
$sequential_11/lstm_22/while/Identity"U
$sequential_11_lstm_22_while_identity-sequential_11/lstm_22/while/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :ÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ¬: ::::: 
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
:ÿÿÿÿÿÿÿÿÿ¬:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬:

_output_shapes
: :

_output_shapes
:
ø%
à
!__inference__traced_save_39107120
file_prefix.
*savev2_dense_11_kernel_read_readvariableop,
(savev2_dense_11_bias_read_readvariableop:
6savev2_lstm_22_lstm_cell_22_kernel_read_readvariableopD
@savev2_lstm_22_lstm_cell_22_recurrent_kernel_read_readvariableop8
4savev2_lstm_22_lstm_cell_22_bias_read_readvariableop:
6savev2_lstm_23_lstm_cell_23_kernel_read_readvariableopD
@savev2_lstm_23_lstm_cell_23_recurrent_kernel_read_readvariableop8
4savev2_lstm_23_lstm_cell_23_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop
savev2_const

identity_1¢MergeV2Checkpoints
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
Const_1
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
ShardedFilename/shard¦
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*¥
valueBB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names¢
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*-
value$B"B B B B B B B B B B B B B 2
SaveV2/shape_and_slices
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_dense_11_kernel_read_readvariableop(savev2_dense_11_bias_read_readvariableop6savev2_lstm_22_lstm_cell_22_kernel_read_readvariableop@savev2_lstm_22_lstm_cell_22_recurrent_kernel_read_readvariableop4savev2_lstm_22_lstm_cell_22_bias_read_readvariableop6savev2_lstm_23_lstm_cell_23_kernel_read_readvariableop@savev2_lstm_23_lstm_cell_23_recurrent_kernel_read_readvariableop4savev2_lstm_23_lstm_cell_23_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
22
SaveV2º
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes¡
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
Z: :	::	]°:
¬°:°:
¬:
:: : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	: 

_output_shapes
::%!

_output_shapes
:	]°:&"
 
_output_shapes
:
¬°:!

_output_shapes	
:°:&"
 
_output_shapes
:
¬:&"
 
_output_shapes
:
:!

_output_shapes	
::	
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
°?
Ô
while_body_39106039
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_22_matmul_readvariableop_resource_0:	]°I
5while_lstm_cell_22_matmul_1_readvariableop_resource_0:
¬°C
4while_lstm_cell_22_biasadd_readvariableop_resource_0:	°
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_22_matmul_readvariableop_resource:	]°G
3while_lstm_cell_22_matmul_1_readvariableop_resource:
¬°A
2while_lstm_cell_22_biasadd_readvariableop_resource:	°¢)while/lstm_cell_22/BiasAdd/ReadVariableOp¢(while/lstm_cell_22/MatMul/ReadVariableOp¢*while/lstm_cell_22/MatMul_1/ReadVariableOpÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ]   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÓ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemÉ
(while/lstm_cell_22/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_22_matmul_readvariableop_resource_0*
_output_shapes
:	]°*
dtype02*
(while/lstm_cell_22/MatMul/ReadVariableOp×
while/lstm_cell_22/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_22/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°2
while/lstm_cell_22/MatMulÐ
*while/lstm_cell_22/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_22_matmul_1_readvariableop_resource_0* 
_output_shapes
:
¬°*
dtype02,
*while/lstm_cell_22/MatMul_1/ReadVariableOpÀ
while/lstm_cell_22/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_22/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°2
while/lstm_cell_22/MatMul_1¸
while/lstm_cell_22/addAddV2#while/lstm_cell_22/MatMul:product:0%while/lstm_cell_22/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°2
while/lstm_cell_22/addÈ
)while/lstm_cell_22/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_22_biasadd_readvariableop_resource_0*
_output_shapes	
:°*
dtype02+
)while/lstm_cell_22/BiasAdd/ReadVariableOpÅ
while/lstm_cell_22/BiasAddBiasAddwhile/lstm_cell_22/add:z:01while/lstm_cell_22/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°2
while/lstm_cell_22/BiasAdd
"while/lstm_cell_22/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_22/split/split_dim
while/lstm_cell_22/splitSplit+while/lstm_cell_22/split/split_dim:output:0#while/lstm_cell_22/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ¬*
	num_split2
while/lstm_cell_22/split
while/lstm_cell_22/SigmoidSigmoid!while/lstm_cell_22/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
while/lstm_cell_22/Sigmoid
while/lstm_cell_22/Sigmoid_1Sigmoid!while/lstm_cell_22/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
while/lstm_cell_22/Sigmoid_1¡
while/lstm_cell_22/mulMul while/lstm_cell_22/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
while/lstm_cell_22/mul
while/lstm_cell_22/ReluRelu!while/lstm_cell_22/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
while/lstm_cell_22/Reluµ
while/lstm_cell_22/mul_1Mulwhile/lstm_cell_22/Sigmoid:y:0%while/lstm_cell_22/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
while/lstm_cell_22/mul_1ª
while/lstm_cell_22/add_1AddV2while/lstm_cell_22/mul:z:0while/lstm_cell_22/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
while/lstm_cell_22/add_1
while/lstm_cell_22/Sigmoid_2Sigmoid!while/lstm_cell_22/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
while/lstm_cell_22/Sigmoid_2
while/lstm_cell_22/Relu_1Reluwhile/lstm_cell_22/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
while/lstm_cell_22/Relu_1¹
while/lstm_cell_22/mul_2Mul while/lstm_cell_22/Sigmoid_2:y:0'while/lstm_cell_22/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
while/lstm_cell_22/mul_2à
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_22/mul_2:z:0*
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
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_22/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_22/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
while/Identity_5Þ

while/NoOpNoOp*^while/lstm_cell_22/BiasAdd/ReadVariableOp)^while/lstm_cell_22/MatMul/ReadVariableOp+^while/lstm_cell_22/MatMul_1/ReadVariableOp*"
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
2while_lstm_cell_22_biasadd_readvariableop_resource4while_lstm_cell_22_biasadd_readvariableop_resource_0"l
3while_lstm_cell_22_matmul_1_readvariableop_resource5while_lstm_cell_22_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_22_matmul_readvariableop_resource3while_lstm_cell_22_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ¬: : : : : 2V
)while/lstm_cell_22/BiasAdd/ReadVariableOp)while/lstm_cell_22/BiasAdd/ReadVariableOp2T
(while/lstm_cell_22/MatMul/ReadVariableOp(while/lstm_cell_22/MatMul/ReadVariableOp2X
*while/lstm_cell_22/MatMul_1/ReadVariableOp*while/lstm_cell_22/MatMul_1/ReadVariableOp: 
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
:ÿÿÿÿÿÿÿÿÿ¬:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬:

_output_shapes
: :

_output_shapes
: 


J__inference_lstm_cell_22_layer_call_and_return_conditional_losses_39102769

inputs

states
states_11
matmul_readvariableop_resource:	]°4
 matmul_1_readvariableop_resource:
¬°.
biasadd_readvariableop_resource:	°
identity

identity_1

identity_2¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	]°*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°2
MatMul
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
¬°*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°2
add
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:°*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°2	
BiasAddd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dimÃ
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ¬*
	num_split2
split`
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2	
Sigmoidd
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
	Sigmoid_1]
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
mulW
ReluRelusplit:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
Relui
mul_1MulSigmoid:y:0Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
mul_1^
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
add_1d
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
	Sigmoid_2V
Relu_1Relu	add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
Relu_1m
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
mul_2e
IdentityIdentity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2

Identityi

Identity_1Identity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2

Identity_1i

Identity_2Identity	add_1:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2

Identity_2
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
A:ÿÿÿÿÿÿÿÿÿ]:ÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ¬: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
 
_user_specified_nameinputs:PL
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
 
_user_specified_namestates:PL
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
 
_user_specified_namestates


Í
0__inference_sequential_11_layer_call_fn_39104807

inputs
unknown:	]°
	unknown_0:
¬°
	unknown_1:	°
	unknown_2:
¬
	unknown_3:

	unknown_4:	
	unknown_5:	
	unknown_6:
identity¢StatefulPartitionedCallÍ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_sequential_11_layer_call_and_return_conditional_losses_391046522
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ]: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
 
_user_specified_nameinputs
»
f
-__inference_dropout_23_layer_call_fn_39106808

inputs
identity¢StatefulPartitionedCallã
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_dropout_23_layer_call_and_return_conditional_losses_391042322
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
\

E__inference_lstm_22_layer_call_and_return_conditional_losses_39104595

inputs>
+lstm_cell_22_matmul_readvariableop_resource:	]°A
-lstm_cell_22_matmul_1_readvariableop_resource:
¬°;
,lstm_cell_22_biasadd_readvariableop_resource:	°
identity¢#lstm_cell_22/BiasAdd/ReadVariableOp¢"lstm_cell_22/MatMul/ReadVariableOp¢$lstm_cell_22/MatMul_1/ReadVariableOp¢whileD
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
strided_slice/stack_2â
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
B :¬2
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
B :è2
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
B :¬2
zeros/packed/1
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
:ÿÿÿÿÿÿÿÿÿ¬2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :¬2
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
B :è2
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
B :¬2
zeros_1/packed/1
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
:ÿÿÿÿÿÿÿÿÿ¬2	
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
:ÿÿÿÿÿÿÿÿÿ]2
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
strided_slice_1/stack_2î
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
TensorArrayV2/element_shape²
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2¿
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ]   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeø
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
strided_slice_2/stack_2ü
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]*
shrink_axis_mask2
strided_slice_2µ
"lstm_cell_22/MatMul/ReadVariableOpReadVariableOp+lstm_cell_22_matmul_readvariableop_resource*
_output_shapes
:	]°*
dtype02$
"lstm_cell_22/MatMul/ReadVariableOp­
lstm_cell_22/MatMulMatMulstrided_slice_2:output:0*lstm_cell_22/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°2
lstm_cell_22/MatMul¼
$lstm_cell_22/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_22_matmul_1_readvariableop_resource* 
_output_shapes
:
¬°*
dtype02&
$lstm_cell_22/MatMul_1/ReadVariableOp©
lstm_cell_22/MatMul_1MatMulzeros:output:0,lstm_cell_22/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°2
lstm_cell_22/MatMul_1 
lstm_cell_22/addAddV2lstm_cell_22/MatMul:product:0lstm_cell_22/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°2
lstm_cell_22/add´
#lstm_cell_22/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_22_biasadd_readvariableop_resource*
_output_shapes	
:°*
dtype02%
#lstm_cell_22/BiasAdd/ReadVariableOp­
lstm_cell_22/BiasAddBiasAddlstm_cell_22/add:z:0+lstm_cell_22/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°2
lstm_cell_22/BiasAdd~
lstm_cell_22/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_22/split/split_dim÷
lstm_cell_22/splitSplit%lstm_cell_22/split/split_dim:output:0lstm_cell_22/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ¬*
	num_split2
lstm_cell_22/split
lstm_cell_22/SigmoidSigmoidlstm_cell_22/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
lstm_cell_22/Sigmoid
lstm_cell_22/Sigmoid_1Sigmoidlstm_cell_22/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
lstm_cell_22/Sigmoid_1
lstm_cell_22/mulMullstm_cell_22/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
lstm_cell_22/mul~
lstm_cell_22/ReluRelulstm_cell_22/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
lstm_cell_22/Relu
lstm_cell_22/mul_1Mullstm_cell_22/Sigmoid:y:0lstm_cell_22/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
lstm_cell_22/mul_1
lstm_cell_22/add_1AddV2lstm_cell_22/mul:z:0lstm_cell_22/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
lstm_cell_22/add_1
lstm_cell_22/Sigmoid_2Sigmoidlstm_cell_22/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
lstm_cell_22/Sigmoid_2}
lstm_cell_22/Relu_1Relulstm_cell_22/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
lstm_cell_22/Relu_1¡
lstm_cell_22/mul_2Mullstm_cell_22/Sigmoid_2:y:0!lstm_cell_22/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
lstm_cell_22/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ¬   2
TensorArrayV2_1/element_shape¸
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
ÿÿÿÿÿÿÿÿÿ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_22_matmul_readvariableop_resource-lstm_cell_22_matmul_1_readvariableop_resource,lstm_cell_22_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ¬: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_39104511*
condR
while_cond_39104510*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ¬: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ¬   22
0TensorArrayV2Stack/TensorListStack/element_shapeé
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
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
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm¦
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
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
:ÿÿÿÿÿÿÿÿÿ¬2

IdentityÈ
NoOpNoOp$^lstm_cell_22/BiasAdd/ReadVariableOp#^lstm_cell_22/MatMul/ReadVariableOp%^lstm_cell_22/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ]: : : 2J
#lstm_cell_22/BiasAdd/ReadVariableOp#lstm_cell_22/BiasAdd/ReadVariableOp2H
"lstm_cell_22/MatMul/ReadVariableOp"lstm_cell_22/MatMul/ReadVariableOp2L
$lstm_cell_22/MatMul_1/ReadVariableOp$lstm_cell_22/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
 
_user_specified_nameinputs
ã
Í
while_cond_39106562
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_39106562___redundant_placeholder06
2while_while_cond_39106562___redundant_placeholder16
2while_while_cond_39106562___redundant_placeholder26
2while_while_cond_39106562___redundant_placeholder3
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
B: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 
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
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:

f
H__inference_dropout_22_layer_call_and_return_conditional_losses_39106138

inputs

identity_1_
IdentityIdentityinputs*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2

Identityn

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ¬:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
 
_user_specified_nameinputs
Ôù

K__inference_sequential_11_layer_call_and_return_conditional_losses_39105134

inputsF
3lstm_22_lstm_cell_22_matmul_readvariableop_resource:	]°I
5lstm_22_lstm_cell_22_matmul_1_readvariableop_resource:
¬°C
4lstm_22_lstm_cell_22_biasadd_readvariableop_resource:	°G
3lstm_23_lstm_cell_23_matmul_readvariableop_resource:
¬I
5lstm_23_lstm_cell_23_matmul_1_readvariableop_resource:
C
4lstm_23_lstm_cell_23_biasadd_readvariableop_resource:	=
*dense_11_tensordot_readvariableop_resource:	6
(dense_11_biasadd_readvariableop_resource:
identity¢dense_11/BiasAdd/ReadVariableOp¢!dense_11/Tensordot/ReadVariableOp¢+lstm_22/lstm_cell_22/BiasAdd/ReadVariableOp¢*lstm_22/lstm_cell_22/MatMul/ReadVariableOp¢,lstm_22/lstm_cell_22/MatMul_1/ReadVariableOp¢lstm_22/while¢+lstm_23/lstm_cell_23/BiasAdd/ReadVariableOp¢*lstm_23/lstm_cell_23/MatMul/ReadVariableOp¢,lstm_23/lstm_cell_23/MatMul_1/ReadVariableOp¢lstm_23/whileT
lstm_22/ShapeShapeinputs*
T0*
_output_shapes
:2
lstm_22/Shape
lstm_22/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_22/strided_slice/stack
lstm_22/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_22/strided_slice/stack_1
lstm_22/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_22/strided_slice/stack_2
lstm_22/strided_sliceStridedSlicelstm_22/Shape:output:0$lstm_22/strided_slice/stack:output:0&lstm_22/strided_slice/stack_1:output:0&lstm_22/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_22/strided_slicem
lstm_22/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :¬2
lstm_22/zeros/mul/y
lstm_22/zeros/mulMullstm_22/strided_slice:output:0lstm_22/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_22/zeros/mulo
lstm_22/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
lstm_22/zeros/Less/y
lstm_22/zeros/LessLesslstm_22/zeros/mul:z:0lstm_22/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_22/zeros/Lesss
lstm_22/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :¬2
lstm_22/zeros/packed/1£
lstm_22/zeros/packedPacklstm_22/strided_slice:output:0lstm_22/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_22/zeros/packedo
lstm_22/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_22/zeros/Const
lstm_22/zerosFilllstm_22/zeros/packed:output:0lstm_22/zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
lstm_22/zerosq
lstm_22/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :¬2
lstm_22/zeros_1/mul/y
lstm_22/zeros_1/mulMullstm_22/strided_slice:output:0lstm_22/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_22/zeros_1/muls
lstm_22/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
lstm_22/zeros_1/Less/y
lstm_22/zeros_1/LessLesslstm_22/zeros_1/mul:z:0lstm_22/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_22/zeros_1/Lessw
lstm_22/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :¬2
lstm_22/zeros_1/packed/1©
lstm_22/zeros_1/packedPacklstm_22/strided_slice:output:0!lstm_22/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_22/zeros_1/packeds
lstm_22/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_22/zeros_1/Const
lstm_22/zeros_1Filllstm_22/zeros_1/packed:output:0lstm_22/zeros_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
lstm_22/zeros_1
lstm_22/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_22/transpose/perm
lstm_22/transpose	Transposeinputslstm_22/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]2
lstm_22/transposeg
lstm_22/Shape_1Shapelstm_22/transpose:y:0*
T0*
_output_shapes
:2
lstm_22/Shape_1
lstm_22/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_22/strided_slice_1/stack
lstm_22/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_22/strided_slice_1/stack_1
lstm_22/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_22/strided_slice_1/stack_2
lstm_22/strided_slice_1StridedSlicelstm_22/Shape_1:output:0&lstm_22/strided_slice_1/stack:output:0(lstm_22/strided_slice_1/stack_1:output:0(lstm_22/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_22/strided_slice_1
#lstm_22/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2%
#lstm_22/TensorArrayV2/element_shapeÒ
lstm_22/TensorArrayV2TensorListReserve,lstm_22/TensorArrayV2/element_shape:output:0 lstm_22/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_22/TensorArrayV2Ï
=lstm_22/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ]   2?
=lstm_22/TensorArrayUnstack/TensorListFromTensor/element_shape
/lstm_22/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_22/transpose:y:0Flstm_22/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type021
/lstm_22/TensorArrayUnstack/TensorListFromTensor
lstm_22/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_22/strided_slice_2/stack
lstm_22/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_22/strided_slice_2/stack_1
lstm_22/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_22/strided_slice_2/stack_2¬
lstm_22/strided_slice_2StridedSlicelstm_22/transpose:y:0&lstm_22/strided_slice_2/stack:output:0(lstm_22/strided_slice_2/stack_1:output:0(lstm_22/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]*
shrink_axis_mask2
lstm_22/strided_slice_2Í
*lstm_22/lstm_cell_22/MatMul/ReadVariableOpReadVariableOp3lstm_22_lstm_cell_22_matmul_readvariableop_resource*
_output_shapes
:	]°*
dtype02,
*lstm_22/lstm_cell_22/MatMul/ReadVariableOpÍ
lstm_22/lstm_cell_22/MatMulMatMul lstm_22/strided_slice_2:output:02lstm_22/lstm_cell_22/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°2
lstm_22/lstm_cell_22/MatMulÔ
,lstm_22/lstm_cell_22/MatMul_1/ReadVariableOpReadVariableOp5lstm_22_lstm_cell_22_matmul_1_readvariableop_resource* 
_output_shapes
:
¬°*
dtype02.
,lstm_22/lstm_cell_22/MatMul_1/ReadVariableOpÉ
lstm_22/lstm_cell_22/MatMul_1MatMullstm_22/zeros:output:04lstm_22/lstm_cell_22/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°2
lstm_22/lstm_cell_22/MatMul_1À
lstm_22/lstm_cell_22/addAddV2%lstm_22/lstm_cell_22/MatMul:product:0'lstm_22/lstm_cell_22/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°2
lstm_22/lstm_cell_22/addÌ
+lstm_22/lstm_cell_22/BiasAdd/ReadVariableOpReadVariableOp4lstm_22_lstm_cell_22_biasadd_readvariableop_resource*
_output_shapes	
:°*
dtype02-
+lstm_22/lstm_cell_22/BiasAdd/ReadVariableOpÍ
lstm_22/lstm_cell_22/BiasAddBiasAddlstm_22/lstm_cell_22/add:z:03lstm_22/lstm_cell_22/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°2
lstm_22/lstm_cell_22/BiasAdd
$lstm_22/lstm_cell_22/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2&
$lstm_22/lstm_cell_22/split/split_dim
lstm_22/lstm_cell_22/splitSplit-lstm_22/lstm_cell_22/split/split_dim:output:0%lstm_22/lstm_cell_22/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ¬*
	num_split2
lstm_22/lstm_cell_22/split
lstm_22/lstm_cell_22/SigmoidSigmoid#lstm_22/lstm_cell_22/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
lstm_22/lstm_cell_22/Sigmoid£
lstm_22/lstm_cell_22/Sigmoid_1Sigmoid#lstm_22/lstm_cell_22/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2 
lstm_22/lstm_cell_22/Sigmoid_1¬
lstm_22/lstm_cell_22/mulMul"lstm_22/lstm_cell_22/Sigmoid_1:y:0lstm_22/zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
lstm_22/lstm_cell_22/mul
lstm_22/lstm_cell_22/ReluRelu#lstm_22/lstm_cell_22/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
lstm_22/lstm_cell_22/Relu½
lstm_22/lstm_cell_22/mul_1Mul lstm_22/lstm_cell_22/Sigmoid:y:0'lstm_22/lstm_cell_22/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
lstm_22/lstm_cell_22/mul_1²
lstm_22/lstm_cell_22/add_1AddV2lstm_22/lstm_cell_22/mul:z:0lstm_22/lstm_cell_22/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
lstm_22/lstm_cell_22/add_1£
lstm_22/lstm_cell_22/Sigmoid_2Sigmoid#lstm_22/lstm_cell_22/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2 
lstm_22/lstm_cell_22/Sigmoid_2
lstm_22/lstm_cell_22/Relu_1Relulstm_22/lstm_cell_22/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
lstm_22/lstm_cell_22/Relu_1Á
lstm_22/lstm_cell_22/mul_2Mul"lstm_22/lstm_cell_22/Sigmoid_2:y:0)lstm_22/lstm_cell_22/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
lstm_22/lstm_cell_22/mul_2
%lstm_22/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ¬   2'
%lstm_22/TensorArrayV2_1/element_shapeØ
lstm_22/TensorArrayV2_1TensorListReserve.lstm_22/TensorArrayV2_1/element_shape:output:0 lstm_22/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_22/TensorArrayV2_1^
lstm_22/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_22/time
 lstm_22/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2"
 lstm_22/while/maximum_iterationsz
lstm_22/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_22/while/loop_counter
lstm_22/whileWhile#lstm_22/while/loop_counter:output:0)lstm_22/while/maximum_iterations:output:0lstm_22/time:output:0 lstm_22/TensorArrayV2_1:handle:0lstm_22/zeros:output:0lstm_22/zeros_1:output:0 lstm_22/strided_slice_1:output:0?lstm_22/TensorArrayUnstack/TensorListFromTensor:output_handle:03lstm_22_lstm_cell_22_matmul_readvariableop_resource5lstm_22_lstm_cell_22_matmul_1_readvariableop_resource4lstm_22_lstm_cell_22_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ¬: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *'
bodyR
lstm_22_while_body_39104874*'
condR
lstm_22_while_cond_39104873*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ¬: : : : : *
parallel_iterations 2
lstm_22/whileÅ
8lstm_22/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ¬   2:
8lstm_22/TensorArrayV2Stack/TensorListStack/element_shape
*lstm_22/TensorArrayV2Stack/TensorListStackTensorListStacklstm_22/while:output:3Alstm_22/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬*
element_dtype02,
*lstm_22/TensorArrayV2Stack/TensorListStack
lstm_22/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
lstm_22/strided_slice_3/stack
lstm_22/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2!
lstm_22/strided_slice_3/stack_1
lstm_22/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_22/strided_slice_3/stack_2Ë
lstm_22/strided_slice_3StridedSlice3lstm_22/TensorArrayV2Stack/TensorListStack:tensor:0&lstm_22/strided_slice_3/stack:output:0(lstm_22/strided_slice_3/stack_1:output:0(lstm_22/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬*
shrink_axis_mask2
lstm_22/strided_slice_3
lstm_22/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_22/transpose_1/permÆ
lstm_22/transpose_1	Transpose3lstm_22/TensorArrayV2Stack/TensorListStack:tensor:0!lstm_22/transpose_1/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
lstm_22/transpose_1v
lstm_22/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_22/runtime
dropout_22/IdentityIdentitylstm_22/transpose_1:y:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
dropout_22/Identityj
lstm_23/ShapeShapedropout_22/Identity:output:0*
T0*
_output_shapes
:2
lstm_23/Shape
lstm_23/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_23/strided_slice/stack
lstm_23/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_23/strided_slice/stack_1
lstm_23/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_23/strided_slice/stack_2
lstm_23/strided_sliceStridedSlicelstm_23/Shape:output:0$lstm_23/strided_slice/stack:output:0&lstm_23/strided_slice/stack_1:output:0&lstm_23/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_23/strided_slicem
lstm_23/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :2
lstm_23/zeros/mul/y
lstm_23/zeros/mulMullstm_23/strided_slice:output:0lstm_23/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_23/zeros/mulo
lstm_23/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
lstm_23/zeros/Less/y
lstm_23/zeros/LessLesslstm_23/zeros/mul:z:0lstm_23/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_23/zeros/Lesss
lstm_23/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :2
lstm_23/zeros/packed/1£
lstm_23/zeros/packedPacklstm_23/strided_slice:output:0lstm_23/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_23/zeros/packedo
lstm_23/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_23/zeros/Const
lstm_23/zerosFilllstm_23/zeros/packed:output:0lstm_23/zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_23/zerosq
lstm_23/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :2
lstm_23/zeros_1/mul/y
lstm_23/zeros_1/mulMullstm_23/strided_slice:output:0lstm_23/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_23/zeros_1/muls
lstm_23/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
lstm_23/zeros_1/Less/y
lstm_23/zeros_1/LessLesslstm_23/zeros_1/mul:z:0lstm_23/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_23/zeros_1/Lessw
lstm_23/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :2
lstm_23/zeros_1/packed/1©
lstm_23/zeros_1/packedPacklstm_23/strided_slice:output:0!lstm_23/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_23/zeros_1/packeds
lstm_23/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_23/zeros_1/Const
lstm_23/zeros_1Filllstm_23/zeros_1/packed:output:0lstm_23/zeros_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_23/zeros_1
lstm_23/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_23/transpose/perm©
lstm_23/transpose	Transposedropout_22/Identity:output:0lstm_23/transpose/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
lstm_23/transposeg
lstm_23/Shape_1Shapelstm_23/transpose:y:0*
T0*
_output_shapes
:2
lstm_23/Shape_1
lstm_23/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_23/strided_slice_1/stack
lstm_23/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_23/strided_slice_1/stack_1
lstm_23/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_23/strided_slice_1/stack_2
lstm_23/strided_slice_1StridedSlicelstm_23/Shape_1:output:0&lstm_23/strided_slice_1/stack:output:0(lstm_23/strided_slice_1/stack_1:output:0(lstm_23/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_23/strided_slice_1
#lstm_23/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2%
#lstm_23/TensorArrayV2/element_shapeÒ
lstm_23/TensorArrayV2TensorListReserve,lstm_23/TensorArrayV2/element_shape:output:0 lstm_23/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_23/TensorArrayV2Ï
=lstm_23/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ¬   2?
=lstm_23/TensorArrayUnstack/TensorListFromTensor/element_shape
/lstm_23/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_23/transpose:y:0Flstm_23/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type021
/lstm_23/TensorArrayUnstack/TensorListFromTensor
lstm_23/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_23/strided_slice_2/stack
lstm_23/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_23/strided_slice_2/stack_1
lstm_23/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_23/strided_slice_2/stack_2­
lstm_23/strided_slice_2StridedSlicelstm_23/transpose:y:0&lstm_23/strided_slice_2/stack:output:0(lstm_23/strided_slice_2/stack_1:output:0(lstm_23/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬*
shrink_axis_mask2
lstm_23/strided_slice_2Î
*lstm_23/lstm_cell_23/MatMul/ReadVariableOpReadVariableOp3lstm_23_lstm_cell_23_matmul_readvariableop_resource* 
_output_shapes
:
¬*
dtype02,
*lstm_23/lstm_cell_23/MatMul/ReadVariableOpÍ
lstm_23/lstm_cell_23/MatMulMatMul lstm_23/strided_slice_2:output:02lstm_23/lstm_cell_23/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_23/lstm_cell_23/MatMulÔ
,lstm_23/lstm_cell_23/MatMul_1/ReadVariableOpReadVariableOp5lstm_23_lstm_cell_23_matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype02.
,lstm_23/lstm_cell_23/MatMul_1/ReadVariableOpÉ
lstm_23/lstm_cell_23/MatMul_1MatMullstm_23/zeros:output:04lstm_23/lstm_cell_23/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_23/lstm_cell_23/MatMul_1À
lstm_23/lstm_cell_23/addAddV2%lstm_23/lstm_cell_23/MatMul:product:0'lstm_23/lstm_cell_23/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_23/lstm_cell_23/addÌ
+lstm_23/lstm_cell_23/BiasAdd/ReadVariableOpReadVariableOp4lstm_23_lstm_cell_23_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02-
+lstm_23/lstm_cell_23/BiasAdd/ReadVariableOpÍ
lstm_23/lstm_cell_23/BiasAddBiasAddlstm_23/lstm_cell_23/add:z:03lstm_23/lstm_cell_23/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_23/lstm_cell_23/BiasAdd
$lstm_23/lstm_cell_23/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2&
$lstm_23/lstm_cell_23/split/split_dim
lstm_23/lstm_cell_23/splitSplit-lstm_23/lstm_cell_23/split/split_dim:output:0%lstm_23/lstm_cell_23/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2
lstm_23/lstm_cell_23/split
lstm_23/lstm_cell_23/SigmoidSigmoid#lstm_23/lstm_cell_23/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_23/lstm_cell_23/Sigmoid£
lstm_23/lstm_cell_23/Sigmoid_1Sigmoid#lstm_23/lstm_cell_23/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
lstm_23/lstm_cell_23/Sigmoid_1¬
lstm_23/lstm_cell_23/mulMul"lstm_23/lstm_cell_23/Sigmoid_1:y:0lstm_23/zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_23/lstm_cell_23/mul
lstm_23/lstm_cell_23/ReluRelu#lstm_23/lstm_cell_23/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_23/lstm_cell_23/Relu½
lstm_23/lstm_cell_23/mul_1Mul lstm_23/lstm_cell_23/Sigmoid:y:0'lstm_23/lstm_cell_23/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_23/lstm_cell_23/mul_1²
lstm_23/lstm_cell_23/add_1AddV2lstm_23/lstm_cell_23/mul:z:0lstm_23/lstm_cell_23/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_23/lstm_cell_23/add_1£
lstm_23/lstm_cell_23/Sigmoid_2Sigmoid#lstm_23/lstm_cell_23/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
lstm_23/lstm_cell_23/Sigmoid_2
lstm_23/lstm_cell_23/Relu_1Relulstm_23/lstm_cell_23/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_23/lstm_cell_23/Relu_1Á
lstm_23/lstm_cell_23/mul_2Mul"lstm_23/lstm_cell_23/Sigmoid_2:y:0)lstm_23/lstm_cell_23/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_23/lstm_cell_23/mul_2
%lstm_23/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2'
%lstm_23/TensorArrayV2_1/element_shapeØ
lstm_23/TensorArrayV2_1TensorListReserve.lstm_23/TensorArrayV2_1/element_shape:output:0 lstm_23/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_23/TensorArrayV2_1^
lstm_23/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_23/time
 lstm_23/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2"
 lstm_23/while/maximum_iterationsz
lstm_23/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_23/while/loop_counter
lstm_23/whileWhile#lstm_23/while/loop_counter:output:0)lstm_23/while/maximum_iterations:output:0lstm_23/time:output:0 lstm_23/TensorArrayV2_1:handle:0lstm_23/zeros:output:0lstm_23/zeros_1:output:0 lstm_23/strided_slice_1:output:0?lstm_23/TensorArrayUnstack/TensorListFromTensor:output_handle:03lstm_23_lstm_cell_23_matmul_readvariableop_resource5lstm_23_lstm_cell_23_matmul_1_readvariableop_resource4lstm_23_lstm_cell_23_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *'
bodyR
lstm_23_while_body_39105022*'
condR
lstm_23_while_cond_39105021*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 2
lstm_23/whileÅ
8lstm_23/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2:
8lstm_23/TensorArrayV2Stack/TensorListStack/element_shape
*lstm_23/TensorArrayV2Stack/TensorListStackTensorListStacklstm_23/while:output:3Alstm_23/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype02,
*lstm_23/TensorArrayV2Stack/TensorListStack
lstm_23/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
lstm_23/strided_slice_3/stack
lstm_23/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2!
lstm_23/strided_slice_3/stack_1
lstm_23/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_23/strided_slice_3/stack_2Ë
lstm_23/strided_slice_3StridedSlice3lstm_23/TensorArrayV2Stack/TensorListStack:tensor:0&lstm_23/strided_slice_3/stack:output:0(lstm_23/strided_slice_3/stack_1:output:0(lstm_23/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
lstm_23/strided_slice_3
lstm_23/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_23/transpose_1/permÆ
lstm_23/transpose_1	Transpose3lstm_23/TensorArrayV2Stack/TensorListStack:tensor:0!lstm_23/transpose_1/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_23/transpose_1v
lstm_23/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_23/runtime
dropout_23/IdentityIdentitylstm_23/transpose_1:y:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout_23/Identity²
!dense_11/Tensordot/ReadVariableOpReadVariableOp*dense_11_tensordot_readvariableop_resource*
_output_shapes
:	*
dtype02#
!dense_11/Tensordot/ReadVariableOp|
dense_11/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_11/Tensordot/axes
dense_11/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_11/Tensordot/free
dense_11/Tensordot/ShapeShapedropout_23/Identity:output:0*
T0*
_output_shapes
:2
dense_11/Tensordot/Shape
 dense_11/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_11/Tensordot/GatherV2/axisþ
dense_11/Tensordot/GatherV2GatherV2!dense_11/Tensordot/Shape:output:0 dense_11/Tensordot/free:output:0)dense_11/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_11/Tensordot/GatherV2
"dense_11/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"dense_11/Tensordot/GatherV2_1/axis
dense_11/Tensordot/GatherV2_1GatherV2!dense_11/Tensordot/Shape:output:0 dense_11/Tensordot/axes:output:0+dense_11/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_11/Tensordot/GatherV2_1~
dense_11/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_11/Tensordot/Const¤
dense_11/Tensordot/ProdProd$dense_11/Tensordot/GatherV2:output:0!dense_11/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_11/Tensordot/Prod
dense_11/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_11/Tensordot/Const_1¬
dense_11/Tensordot/Prod_1Prod&dense_11/Tensordot/GatherV2_1:output:0#dense_11/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_11/Tensordot/Prod_1
dense_11/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2 
dense_11/Tensordot/concat/axisÝ
dense_11/Tensordot/concatConcatV2 dense_11/Tensordot/free:output:0 dense_11/Tensordot/axes:output:0'dense_11/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_11/Tensordot/concat°
dense_11/Tensordot/stackPack dense_11/Tensordot/Prod:output:0"dense_11/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_11/Tensordot/stackÂ
dense_11/Tensordot/transpose	Transposedropout_23/Identity:output:0"dense_11/Tensordot/concat:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_11/Tensordot/transposeÃ
dense_11/Tensordot/ReshapeReshape dense_11/Tensordot/transpose:y:0!dense_11/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
dense_11/Tensordot/ReshapeÂ
dense_11/Tensordot/MatMulMatMul#dense_11/Tensordot/Reshape:output:0)dense_11/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_11/Tensordot/MatMul
dense_11/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
dense_11/Tensordot/Const_2
 dense_11/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_11/Tensordot/concat_1/axisê
dense_11/Tensordot/concat_1ConcatV2$dense_11/Tensordot/GatherV2:output:0#dense_11/Tensordot/Const_2:output:0)dense_11/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_11/Tensordot/concat_1´
dense_11/TensordotReshape#dense_11/Tensordot/MatMul:product:0$dense_11/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_11/Tensordot§
dense_11/BiasAdd/ReadVariableOpReadVariableOp(dense_11_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_11/BiasAdd/ReadVariableOp«
dense_11/BiasAddBiasAdddense_11/Tensordot:output:0'dense_11/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_11/BiasAdd
dense_11/SoftmaxSoftmaxdense_11/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_11/Softmaxy
IdentityIdentitydense_11/Softmax:softmax:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

IdentityÈ
NoOpNoOp ^dense_11/BiasAdd/ReadVariableOp"^dense_11/Tensordot/ReadVariableOp,^lstm_22/lstm_cell_22/BiasAdd/ReadVariableOp+^lstm_22/lstm_cell_22/MatMul/ReadVariableOp-^lstm_22/lstm_cell_22/MatMul_1/ReadVariableOp^lstm_22/while,^lstm_23/lstm_cell_23/BiasAdd/ReadVariableOp+^lstm_23/lstm_cell_23/MatMul/ReadVariableOp-^lstm_23/lstm_cell_23/MatMul_1/ReadVariableOp^lstm_23/while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ]: : : : : : : : 2B
dense_11/BiasAdd/ReadVariableOpdense_11/BiasAdd/ReadVariableOp2F
!dense_11/Tensordot/ReadVariableOp!dense_11/Tensordot/ReadVariableOp2Z
+lstm_22/lstm_cell_22/BiasAdd/ReadVariableOp+lstm_22/lstm_cell_22/BiasAdd/ReadVariableOp2X
*lstm_22/lstm_cell_22/MatMul/ReadVariableOp*lstm_22/lstm_cell_22/MatMul/ReadVariableOp2\
,lstm_22/lstm_cell_22/MatMul_1/ReadVariableOp,lstm_22/lstm_cell_22/MatMul_1/ReadVariableOp2
lstm_22/whilelstm_22/while2Z
+lstm_23/lstm_cell_23/BiasAdd/ReadVariableOp+lstm_23/lstm_cell_23/BiasAdd/ReadVariableOp2X
*lstm_23/lstm_cell_23/MatMul/ReadVariableOp*lstm_23/lstm_cell_23/MatMul/ReadVariableOp2\
,lstm_23/lstm_cell_23/MatMul_1/ReadVariableOp,lstm_23/lstm_cell_23/MatMul_1/ReadVariableOp2
lstm_23/whilelstm_23/while:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
 
_user_specified_nameinputs
×
g
H__inference_dropout_23_layer_call_and_return_conditional_losses_39104232

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/Constx
dropout/MulMulinputsdropout/Const:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape¹
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>2
dropout/GreaterEqual/yÃ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/Mul_1j
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Õ!
þ
F__inference_dense_11_layer_call_and_return_conditional_losses_39104176

inputs4
!tensordot_readvariableop_resource:	-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢Tensordot/ReadVariableOp
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes
:	*
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
Tensordot/GatherV2/axisÑ
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
Tensordot/GatherV2_1/axis×
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
Tensordot/Const
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
Tensordot/Const_1
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
Tensordot/concat/axis°
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Tensordot/transpose
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Tensordot/Reshape
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
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
Tensordot/concat_1/axis½
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	Tensordot
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdde
SoftmaxSoftmaxBiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
Softmaxp
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ù

)sequential_11_lstm_23_while_cond_39102435H
Dsequential_11_lstm_23_while_sequential_11_lstm_23_while_loop_counterN
Jsequential_11_lstm_23_while_sequential_11_lstm_23_while_maximum_iterations+
'sequential_11_lstm_23_while_placeholder-
)sequential_11_lstm_23_while_placeholder_1-
)sequential_11_lstm_23_while_placeholder_2-
)sequential_11_lstm_23_while_placeholder_3J
Fsequential_11_lstm_23_while_less_sequential_11_lstm_23_strided_slice_1b
^sequential_11_lstm_23_while_sequential_11_lstm_23_while_cond_39102435___redundant_placeholder0b
^sequential_11_lstm_23_while_sequential_11_lstm_23_while_cond_39102435___redundant_placeholder1b
^sequential_11_lstm_23_while_sequential_11_lstm_23_while_cond_39102435___redundant_placeholder2b
^sequential_11_lstm_23_while_sequential_11_lstm_23_while_cond_39102435___redundant_placeholder3(
$sequential_11_lstm_23_while_identity
Þ
 sequential_11/lstm_23/while/LessLess'sequential_11_lstm_23_while_placeholderFsequential_11_lstm_23_while_less_sequential_11_lstm_23_strided_slice_1*
T0*
_output_shapes
: 2"
 sequential_11/lstm_23/while/Less
$sequential_11/lstm_23/while/IdentityIdentity$sequential_11/lstm_23/while/Less:z:0*
T0
*
_output_shapes
: 2&
$sequential_11/lstm_23/while/Identity"U
$sequential_11_lstm_23_while_identity-sequential_11/lstm_23/while/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 
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
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
Ã\
 
E__inference_lstm_22_layer_call_and_return_conditional_losses_39105670
inputs_0>
+lstm_cell_22_matmul_readvariableop_resource:	]°A
-lstm_cell_22_matmul_1_readvariableop_resource:
¬°;
,lstm_cell_22_biasadd_readvariableop_resource:	°
identity¢#lstm_cell_22/BiasAdd/ReadVariableOp¢"lstm_cell_22/MatMul/ReadVariableOp¢$lstm_cell_22/MatMul_1/ReadVariableOp¢whileF
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
strided_slice/stack_2â
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
B :¬2
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
B :è2
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
B :¬2
zeros/packed/1
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
:ÿÿÿÿÿÿÿÿÿ¬2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :¬2
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
B :è2
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
B :¬2
zeros_1/packed/1
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
:ÿÿÿÿÿÿÿÿÿ¬2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ]2
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
strided_slice_1/stack_2î
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
TensorArrayV2/element_shape²
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2¿
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ]   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeø
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
strided_slice_2/stack_2ü
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]*
shrink_axis_mask2
strided_slice_2µ
"lstm_cell_22/MatMul/ReadVariableOpReadVariableOp+lstm_cell_22_matmul_readvariableop_resource*
_output_shapes
:	]°*
dtype02$
"lstm_cell_22/MatMul/ReadVariableOp­
lstm_cell_22/MatMulMatMulstrided_slice_2:output:0*lstm_cell_22/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°2
lstm_cell_22/MatMul¼
$lstm_cell_22/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_22_matmul_1_readvariableop_resource* 
_output_shapes
:
¬°*
dtype02&
$lstm_cell_22/MatMul_1/ReadVariableOp©
lstm_cell_22/MatMul_1MatMulzeros:output:0,lstm_cell_22/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°2
lstm_cell_22/MatMul_1 
lstm_cell_22/addAddV2lstm_cell_22/MatMul:product:0lstm_cell_22/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°2
lstm_cell_22/add´
#lstm_cell_22/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_22_biasadd_readvariableop_resource*
_output_shapes	
:°*
dtype02%
#lstm_cell_22/BiasAdd/ReadVariableOp­
lstm_cell_22/BiasAddBiasAddlstm_cell_22/add:z:0+lstm_cell_22/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°2
lstm_cell_22/BiasAdd~
lstm_cell_22/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_22/split/split_dim÷
lstm_cell_22/splitSplit%lstm_cell_22/split/split_dim:output:0lstm_cell_22/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ¬*
	num_split2
lstm_cell_22/split
lstm_cell_22/SigmoidSigmoidlstm_cell_22/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
lstm_cell_22/Sigmoid
lstm_cell_22/Sigmoid_1Sigmoidlstm_cell_22/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
lstm_cell_22/Sigmoid_1
lstm_cell_22/mulMullstm_cell_22/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
lstm_cell_22/mul~
lstm_cell_22/ReluRelulstm_cell_22/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
lstm_cell_22/Relu
lstm_cell_22/mul_1Mullstm_cell_22/Sigmoid:y:0lstm_cell_22/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
lstm_cell_22/mul_1
lstm_cell_22/add_1AddV2lstm_cell_22/mul:z:0lstm_cell_22/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
lstm_cell_22/add_1
lstm_cell_22/Sigmoid_2Sigmoidlstm_cell_22/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
lstm_cell_22/Sigmoid_2}
lstm_cell_22/Relu_1Relulstm_cell_22/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
lstm_cell_22/Relu_1¡
lstm_cell_22/mul_2Mullstm_cell_22/Sigmoid_2:y:0!lstm_cell_22/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
lstm_cell_22/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ¬   2
TensorArrayV2_1/element_shape¸
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
ÿÿÿÿÿÿÿÿÿ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_22_matmul_readvariableop_resource-lstm_cell_22_matmul_1_readvariableop_resource,lstm_cell_22_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ¬: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_39105586*
condR
while_cond_39105585*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ¬: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ¬   22
0TensorArrayV2Stack/TensorListStack/element_shapeò
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
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
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm¯
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬2
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
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬2

IdentityÈ
NoOpNoOp$^lstm_cell_22/BiasAdd/ReadVariableOp#^lstm_cell_22/MatMul/ReadVariableOp%^lstm_cell_22/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ]: : : 2J
#lstm_cell_22/BiasAdd/ReadVariableOp#lstm_cell_22/BiasAdd/ReadVariableOp2H
"lstm_cell_22/MatMul/ReadVariableOp"lstm_cell_22/MatMul/ReadVariableOp2L
$lstm_cell_22/MatMul_1/ReadVariableOp$lstm_cell_22/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ]
"
_user_specified_name
inputs/0


J__inference_lstm_cell_22_layer_call_and_return_conditional_losses_39106931

inputs
states_0
states_11
matmul_readvariableop_resource:	]°4
 matmul_1_readvariableop_resource:
¬°.
biasadd_readvariableop_resource:	°
identity

identity_1

identity_2¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	]°*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°2
MatMul
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
¬°*
dtype02
MatMul_1/ReadVariableOp|
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°2
add
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:°*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°2	
BiasAddd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dimÃ
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ¬*
	num_split2
split`
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2	
Sigmoidd
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
	Sigmoid_1]
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
mulW
ReluRelusplit:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
Relui
mul_1MulSigmoid:y:0Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
mul_1^
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
add_1d
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
	Sigmoid_2V
Relu_1Relu	add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
Relu_1m
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
mul_2e
IdentityIdentity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2

Identityi

Identity_1Identity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2

Identity_1i

Identity_2Identity	add_1:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2

Identity_2
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
A:ÿÿÿÿÿÿÿÿÿ]:ÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ¬: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
 
_user_specified_nameinputs:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
"
_user_specified_name
states/0:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
"
_user_specified_name
states/1
&
õ
while_body_39103267
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_01
while_lstm_cell_23_39103291_0:
¬1
while_lstm_cell_23_39103293_0:
,
while_lstm_cell_23_39103295_0:	
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor/
while_lstm_cell_23_39103291:
¬/
while_lstm_cell_23_39103293:
*
while_lstm_cell_23_39103295:	¢*while/lstm_cell_23/StatefulPartitionedCallÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ¬   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÔ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemì
*while/lstm_cell_23/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_23_39103291_0while_lstm_cell_23_39103293_0while_lstm_cell_23_39103295_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_lstm_cell_23_layer_call_and_return_conditional_losses_391032532,
*while/lstm_cell_23/StatefulPartitionedCall÷
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder3while/lstm_cell_23/StatefulPartitionedCall:output:0*
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
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3¥
while/Identity_4Identity3while/lstm_cell_23/StatefulPartitionedCall:output:1^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/Identity_4¥
while/Identity_5Identity3while/lstm_cell_23/StatefulPartitionedCall:output:2^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/Identity_5

while/NoOpNoOp+^while/lstm_cell_23/StatefulPartitionedCall*"
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
while_lstm_cell_23_39103291while_lstm_cell_23_39103291_0"<
while_lstm_cell_23_39103293while_lstm_cell_23_39103293_0"<
while_lstm_cell_23_39103295while_lstm_cell_23_39103295_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2X
*while/lstm_cell_23/StatefulPartitionedCall*while/lstm_cell_23/StatefulPartitionedCall: 
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
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
ã
Í
while_cond_39105887
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_39105887___redundant_placeholder06
2while_while_cond_39105887___redundant_placeholder16
2while_while_cond_39105887___redundant_placeholder26
2while_while_cond_39105887___redundant_placeholder3
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
B: : : : :ÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ¬: ::::: 
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
:ÿÿÿÿÿÿÿÿÿ¬:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬:

_output_shapes
: :

_output_shapes
:
Ê
Â
K__inference_sequential_11_layer_call_and_return_conditional_losses_39104742
lstm_22_input#
lstm_22_39104720:	]°$
lstm_22_39104722:
¬°
lstm_22_39104724:	°$
lstm_23_39104728:
¬$
lstm_23_39104730:

lstm_23_39104732:	$
dense_11_39104736:	
dense_11_39104738:
identity¢ dense_11/StatefulPartitionedCall¢"dropout_22/StatefulPartitionedCall¢"dropout_23/StatefulPartitionedCall¢lstm_22/StatefulPartitionedCall¢lstm_23/StatefulPartitionedCallµ
lstm_22/StatefulPartitionedCallStatefulPartitionedCalllstm_22_inputlstm_22_39104720lstm_22_39104722lstm_22_39104724*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_lstm_22_layer_call_and_return_conditional_losses_391045952!
lstm_22/StatefulPartitionedCall
"dropout_22/StatefulPartitionedCallStatefulPartitionedCall(lstm_22/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_dropout_22_layer_call_and_return_conditional_losses_391044282$
"dropout_22/StatefulPartitionedCallÓ
lstm_23/StatefulPartitionedCallStatefulPartitionedCall+dropout_22/StatefulPartitionedCall:output:0lstm_23_39104728lstm_23_39104730lstm_23_39104732*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_lstm_23_layer_call_and_return_conditional_losses_391043992!
lstm_23/StatefulPartitionedCallÀ
"dropout_23/StatefulPartitionedCallStatefulPartitionedCall(lstm_23/StatefulPartitionedCall:output:0#^dropout_22/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_dropout_23_layer_call_and_return_conditional_losses_391042322$
"dropout_23/StatefulPartitionedCallÃ
 dense_11/StatefulPartitionedCallStatefulPartitionedCall+dropout_23/StatefulPartitionedCall:output:0dense_11_39104736dense_11_39104738*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_11_layer_call_and_return_conditional_losses_391041762"
 dense_11/StatefulPartitionedCall
IdentityIdentity)dense_11/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityÿ
NoOpNoOp!^dense_11/StatefulPartitionedCall#^dropout_22/StatefulPartitionedCall#^dropout_23/StatefulPartitionedCall ^lstm_22/StatefulPartitionedCall ^lstm_23/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ]: : : : : : : : 2D
 dense_11/StatefulPartitionedCall dense_11/StatefulPartitionedCall2H
"dropout_22/StatefulPartitionedCall"dropout_22/StatefulPartitionedCall2H
"dropout_23/StatefulPartitionedCall"dropout_23/StatefulPartitionedCall2B
lstm_22/StatefulPartitionedCalllstm_22/StatefulPartitionedCall2B
lstm_23/StatefulPartitionedCalllstm_23/StatefulPartitionedCall:Z V
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
'
_user_specified_namelstm_22_input
\

E__inference_lstm_23_layer_call_and_return_conditional_losses_39104399

inputs?
+lstm_cell_23_matmul_readvariableop_resource:
¬A
-lstm_cell_23_matmul_1_readvariableop_resource:
;
,lstm_cell_23_biasadd_readvariableop_resource:	
identity¢#lstm_cell_23/BiasAdd/ReadVariableOp¢"lstm_cell_23/MatMul/ReadVariableOp¢$lstm_cell_23/MatMul_1/ReadVariableOp¢whileD
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
strided_slice/stack_2â
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
B :2
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
B :è2
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
B :2
zeros/packed/1
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
:ÿÿÿÿÿÿÿÿÿ2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :2
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
B :è2
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
B :2
zeros_1/packed/1
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
:ÿÿÿÿÿÿÿÿÿ2	
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
:ÿÿÿÿÿÿÿÿÿ¬2
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
strided_slice_1/stack_2î
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
TensorArrayV2/element_shape²
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2¿
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ¬   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeø
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
strided_slice_2/stack_2ý
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬*
shrink_axis_mask2
strided_slice_2¶
"lstm_cell_23/MatMul/ReadVariableOpReadVariableOp+lstm_cell_23_matmul_readvariableop_resource* 
_output_shapes
:
¬*
dtype02$
"lstm_cell_23/MatMul/ReadVariableOp­
lstm_cell_23/MatMulMatMulstrided_slice_2:output:0*lstm_cell_23/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_23/MatMul¼
$lstm_cell_23/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_23_matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype02&
$lstm_cell_23/MatMul_1/ReadVariableOp©
lstm_cell_23/MatMul_1MatMulzeros:output:0,lstm_cell_23/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_23/MatMul_1 
lstm_cell_23/addAddV2lstm_cell_23/MatMul:product:0lstm_cell_23/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_23/add´
#lstm_cell_23/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_23_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02%
#lstm_cell_23/BiasAdd/ReadVariableOp­
lstm_cell_23/BiasAddBiasAddlstm_cell_23/add:z:0+lstm_cell_23/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_23/BiasAdd~
lstm_cell_23/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_23/split/split_dim÷
lstm_cell_23/splitSplit%lstm_cell_23/split/split_dim:output:0lstm_cell_23/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2
lstm_cell_23/split
lstm_cell_23/SigmoidSigmoidlstm_cell_23/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_23/Sigmoid
lstm_cell_23/Sigmoid_1Sigmoidlstm_cell_23/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_23/Sigmoid_1
lstm_cell_23/mulMullstm_cell_23/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_23/mul~
lstm_cell_23/ReluRelulstm_cell_23/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_23/Relu
lstm_cell_23/mul_1Mullstm_cell_23/Sigmoid:y:0lstm_cell_23/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_23/mul_1
lstm_cell_23/add_1AddV2lstm_cell_23/mul:z:0lstm_cell_23/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_23/add_1
lstm_cell_23/Sigmoid_2Sigmoidlstm_cell_23/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_23/Sigmoid_2}
lstm_cell_23/Relu_1Relulstm_cell_23/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_23/Relu_1¡
lstm_cell_23/mul_2Mullstm_cell_23/Sigmoid_2:y:0!lstm_cell_23/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_23/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
TensorArrayV2_1/element_shape¸
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
ÿÿÿÿÿÿÿÿÿ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_23_matmul_readvariableop_resource-lstm_cell_23_matmul_1_readvariableop_resource,lstm_cell_23_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_39104315*
condR
while_cond_39104314*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   22
0TensorArrayV2Stack/TensorListStack/element_shapeé
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
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
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm¦
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
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
:ÿÿÿÿÿÿÿÿÿ2

IdentityÈ
NoOpNoOp$^lstm_cell_23/BiasAdd/ReadVariableOp#^lstm_cell_23/MatMul/ReadVariableOp%^lstm_cell_23/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:ÿÿÿÿÿÿÿÿÿ¬: : : 2J
#lstm_cell_23/BiasAdd/ReadVariableOp#lstm_cell_23/BiasAdd/ReadVariableOp2H
"lstm_cell_23/MatMul/ReadVariableOp"lstm_cell_23/MatMul/ReadVariableOp2L
$lstm_cell_23/MatMul_1/ReadVariableOp$lstm_cell_23/MatMul_1/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
 
_user_specified_nameinputs
´?
Ö
while_body_39106714
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
3while_lstm_cell_23_matmul_readvariableop_resource_0:
¬I
5while_lstm_cell_23_matmul_1_readvariableop_resource_0:
C
4while_lstm_cell_23_biasadd_readvariableop_resource_0:	
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
1while_lstm_cell_23_matmul_readvariableop_resource:
¬G
3while_lstm_cell_23_matmul_1_readvariableop_resource:
A
2while_lstm_cell_23_biasadd_readvariableop_resource:	¢)while/lstm_cell_23/BiasAdd/ReadVariableOp¢(while/lstm_cell_23/MatMul/ReadVariableOp¢*while/lstm_cell_23/MatMul_1/ReadVariableOpÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ¬   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÔ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemÊ
(while/lstm_cell_23/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_23_matmul_readvariableop_resource_0* 
_output_shapes
:
¬*
dtype02*
(while/lstm_cell_23/MatMul/ReadVariableOp×
while/lstm_cell_23/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_23/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_23/MatMulÐ
*while/lstm_cell_23/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_23_matmul_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype02,
*while/lstm_cell_23/MatMul_1/ReadVariableOpÀ
while/lstm_cell_23/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_23/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_23/MatMul_1¸
while/lstm_cell_23/addAddV2#while/lstm_cell_23/MatMul:product:0%while/lstm_cell_23/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_23/addÈ
)while/lstm_cell_23/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_23_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02+
)while/lstm_cell_23/BiasAdd/ReadVariableOpÅ
while/lstm_cell_23/BiasAddBiasAddwhile/lstm_cell_23/add:z:01while/lstm_cell_23/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_23/BiasAdd
"while/lstm_cell_23/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_23/split/split_dim
while/lstm_cell_23/splitSplit+while/lstm_cell_23/split/split_dim:output:0#while/lstm_cell_23/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2
while/lstm_cell_23/split
while/lstm_cell_23/SigmoidSigmoid!while/lstm_cell_23/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_23/Sigmoid
while/lstm_cell_23/Sigmoid_1Sigmoid!while/lstm_cell_23/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_23/Sigmoid_1¡
while/lstm_cell_23/mulMul while/lstm_cell_23/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_23/mul
while/lstm_cell_23/ReluRelu!while/lstm_cell_23/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_23/Reluµ
while/lstm_cell_23/mul_1Mulwhile/lstm_cell_23/Sigmoid:y:0%while/lstm_cell_23/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_23/mul_1ª
while/lstm_cell_23/add_1AddV2while/lstm_cell_23/mul:z:0while/lstm_cell_23/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_23/add_1
while/lstm_cell_23/Sigmoid_2Sigmoid!while/lstm_cell_23/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_23/Sigmoid_2
while/lstm_cell_23/Relu_1Reluwhile/lstm_cell_23/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_23/Relu_1¹
while/lstm_cell_23/mul_2Mul while/lstm_cell_23/Sigmoid_2:y:0'while/lstm_cell_23/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_23/mul_2à
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_23/mul_2:z:0*
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
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_23/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_23/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/Identity_5Þ

while/NoOpNoOp*^while/lstm_cell_23/BiasAdd/ReadVariableOp)^while/lstm_cell_23/MatMul/ReadVariableOp+^while/lstm_cell_23/MatMul_1/ReadVariableOp*"
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
2while_lstm_cell_23_biasadd_readvariableop_resource4while_lstm_cell_23_biasadd_readvariableop_resource_0"l
3while_lstm_cell_23_matmul_1_readvariableop_resource5while_lstm_cell_23_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_23_matmul_readvariableop_resource3while_lstm_cell_23_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2V
)while/lstm_cell_23/BiasAdd/ReadVariableOp)while/lstm_cell_23/BiasAdd/ReadVariableOp2T
(while/lstm_cell_23/MatMul/ReadVariableOp(while/lstm_cell_23/MatMul/ReadVariableOp2X
*while/lstm_cell_23/MatMul_1/ReadVariableOp*while/lstm_cell_23/MatMul_1/ReadVariableOp: 
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
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
µ
»
K__inference_sequential_11_layer_call_and_return_conditional_losses_39104652

inputs#
lstm_22_39104630:	]°$
lstm_22_39104632:
¬°
lstm_22_39104634:	°$
lstm_23_39104638:
¬$
lstm_23_39104640:

lstm_23_39104642:	$
dense_11_39104646:	
dense_11_39104648:
identity¢ dense_11/StatefulPartitionedCall¢"dropout_22/StatefulPartitionedCall¢"dropout_23/StatefulPartitionedCall¢lstm_22/StatefulPartitionedCall¢lstm_23/StatefulPartitionedCall®
lstm_22/StatefulPartitionedCallStatefulPartitionedCallinputslstm_22_39104630lstm_22_39104632lstm_22_39104634*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_lstm_22_layer_call_and_return_conditional_losses_391045952!
lstm_22/StatefulPartitionedCall
"dropout_22/StatefulPartitionedCallStatefulPartitionedCall(lstm_22/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_dropout_22_layer_call_and_return_conditional_losses_391044282$
"dropout_22/StatefulPartitionedCallÓ
lstm_23/StatefulPartitionedCallStatefulPartitionedCall+dropout_22/StatefulPartitionedCall:output:0lstm_23_39104638lstm_23_39104640lstm_23_39104642*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_lstm_23_layer_call_and_return_conditional_losses_391043992!
lstm_23/StatefulPartitionedCallÀ
"dropout_23/StatefulPartitionedCallStatefulPartitionedCall(lstm_23/StatefulPartitionedCall:output:0#^dropout_22/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_dropout_23_layer_call_and_return_conditional_losses_391042322$
"dropout_23/StatefulPartitionedCallÃ
 dense_11/StatefulPartitionedCallStatefulPartitionedCall+dropout_23/StatefulPartitionedCall:output:0dense_11_39104646dense_11_39104648*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_11_layer_call_and_return_conditional_losses_391041762"
 dense_11/StatefulPartitionedCall
IdentityIdentity)dense_11/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityÿ
NoOpNoOp!^dense_11/StatefulPartitionedCall#^dropout_22/StatefulPartitionedCall#^dropout_23/StatefulPartitionedCall ^lstm_22/StatefulPartitionedCall ^lstm_23/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ]: : : : : : : : 2D
 dense_11/StatefulPartitionedCall dense_11/StatefulPartitionedCall2H
"dropout_22/StatefulPartitionedCall"dropout_22/StatefulPartitionedCall2H
"dropout_23/StatefulPartitionedCall"dropout_23/StatefulPartitionedCall2B
lstm_22/StatefulPartitionedCalllstm_22/StatefulPartitionedCall2B
lstm_23/StatefulPartitionedCalllstm_23/StatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
 
_user_specified_nameinputs
¨¹
ç	
#__inference__wrapped_model_39102548
lstm_22_inputT
Asequential_11_lstm_22_lstm_cell_22_matmul_readvariableop_resource:	]°W
Csequential_11_lstm_22_lstm_cell_22_matmul_1_readvariableop_resource:
¬°Q
Bsequential_11_lstm_22_lstm_cell_22_biasadd_readvariableop_resource:	°U
Asequential_11_lstm_23_lstm_cell_23_matmul_readvariableop_resource:
¬W
Csequential_11_lstm_23_lstm_cell_23_matmul_1_readvariableop_resource:
Q
Bsequential_11_lstm_23_lstm_cell_23_biasadd_readvariableop_resource:	K
8sequential_11_dense_11_tensordot_readvariableop_resource:	D
6sequential_11_dense_11_biasadd_readvariableop_resource:
identity¢-sequential_11/dense_11/BiasAdd/ReadVariableOp¢/sequential_11/dense_11/Tensordot/ReadVariableOp¢9sequential_11/lstm_22/lstm_cell_22/BiasAdd/ReadVariableOp¢8sequential_11/lstm_22/lstm_cell_22/MatMul/ReadVariableOp¢:sequential_11/lstm_22/lstm_cell_22/MatMul_1/ReadVariableOp¢sequential_11/lstm_22/while¢9sequential_11/lstm_23/lstm_cell_23/BiasAdd/ReadVariableOp¢8sequential_11/lstm_23/lstm_cell_23/MatMul/ReadVariableOp¢:sequential_11/lstm_23/lstm_cell_23/MatMul_1/ReadVariableOp¢sequential_11/lstm_23/whilew
sequential_11/lstm_22/ShapeShapelstm_22_input*
T0*
_output_shapes
:2
sequential_11/lstm_22/Shape 
)sequential_11/lstm_22/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)sequential_11/lstm_22/strided_slice/stack¤
+sequential_11/lstm_22/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential_11/lstm_22/strided_slice/stack_1¤
+sequential_11/lstm_22/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential_11/lstm_22/strided_slice/stack_2æ
#sequential_11/lstm_22/strided_sliceStridedSlice$sequential_11/lstm_22/Shape:output:02sequential_11/lstm_22/strided_slice/stack:output:04sequential_11/lstm_22/strided_slice/stack_1:output:04sequential_11/lstm_22/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#sequential_11/lstm_22/strided_slice
!sequential_11/lstm_22/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :¬2#
!sequential_11/lstm_22/zeros/mul/yÄ
sequential_11/lstm_22/zeros/mulMul,sequential_11/lstm_22/strided_slice:output:0*sequential_11/lstm_22/zeros/mul/y:output:0*
T0*
_output_shapes
: 2!
sequential_11/lstm_22/zeros/mul
"sequential_11/lstm_22/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2$
"sequential_11/lstm_22/zeros/Less/y¿
 sequential_11/lstm_22/zeros/LessLess#sequential_11/lstm_22/zeros/mul:z:0+sequential_11/lstm_22/zeros/Less/y:output:0*
T0*
_output_shapes
: 2"
 sequential_11/lstm_22/zeros/Less
$sequential_11/lstm_22/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :¬2&
$sequential_11/lstm_22/zeros/packed/1Û
"sequential_11/lstm_22/zeros/packedPack,sequential_11/lstm_22/strided_slice:output:0-sequential_11/lstm_22/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2$
"sequential_11/lstm_22/zeros/packed
!sequential_11/lstm_22/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!sequential_11/lstm_22/zeros/ConstÎ
sequential_11/lstm_22/zerosFill+sequential_11/lstm_22/zeros/packed:output:0*sequential_11/lstm_22/zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
sequential_11/lstm_22/zeros
#sequential_11/lstm_22/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :¬2%
#sequential_11/lstm_22/zeros_1/mul/yÊ
!sequential_11/lstm_22/zeros_1/mulMul,sequential_11/lstm_22/strided_slice:output:0,sequential_11/lstm_22/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2#
!sequential_11/lstm_22/zeros_1/mul
$sequential_11/lstm_22/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2&
$sequential_11/lstm_22/zeros_1/Less/yÇ
"sequential_11/lstm_22/zeros_1/LessLess%sequential_11/lstm_22/zeros_1/mul:z:0-sequential_11/lstm_22/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2$
"sequential_11/lstm_22/zeros_1/Less
&sequential_11/lstm_22/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :¬2(
&sequential_11/lstm_22/zeros_1/packed/1á
$sequential_11/lstm_22/zeros_1/packedPack,sequential_11/lstm_22/strided_slice:output:0/sequential_11/lstm_22/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2&
$sequential_11/lstm_22/zeros_1/packed
#sequential_11/lstm_22/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2%
#sequential_11/lstm_22/zeros_1/ConstÖ
sequential_11/lstm_22/zeros_1Fill-sequential_11/lstm_22/zeros_1/packed:output:0,sequential_11/lstm_22/zeros_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
sequential_11/lstm_22/zeros_1¡
$sequential_11/lstm_22/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2&
$sequential_11/lstm_22/transpose/permÃ
sequential_11/lstm_22/transpose	Transposelstm_22_input-sequential_11/lstm_22/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]2!
sequential_11/lstm_22/transpose
sequential_11/lstm_22/Shape_1Shape#sequential_11/lstm_22/transpose:y:0*
T0*
_output_shapes
:2
sequential_11/lstm_22/Shape_1¤
+sequential_11/lstm_22/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2-
+sequential_11/lstm_22/strided_slice_1/stack¨
-sequential_11/lstm_22/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_11/lstm_22/strided_slice_1/stack_1¨
-sequential_11/lstm_22/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_11/lstm_22/strided_slice_1/stack_2ò
%sequential_11/lstm_22/strided_slice_1StridedSlice&sequential_11/lstm_22/Shape_1:output:04sequential_11/lstm_22/strided_slice_1/stack:output:06sequential_11/lstm_22/strided_slice_1/stack_1:output:06sequential_11/lstm_22/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%sequential_11/lstm_22/strided_slice_1±
1sequential_11/lstm_22/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ23
1sequential_11/lstm_22/TensorArrayV2/element_shape
#sequential_11/lstm_22/TensorArrayV2TensorListReserve:sequential_11/lstm_22/TensorArrayV2/element_shape:output:0.sequential_11/lstm_22/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02%
#sequential_11/lstm_22/TensorArrayV2ë
Ksequential_11/lstm_22/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ]   2M
Ksequential_11/lstm_22/TensorArrayUnstack/TensorListFromTensor/element_shapeÐ
=sequential_11/lstm_22/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor#sequential_11/lstm_22/transpose:y:0Tsequential_11/lstm_22/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02?
=sequential_11/lstm_22/TensorArrayUnstack/TensorListFromTensor¤
+sequential_11/lstm_22/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2-
+sequential_11/lstm_22/strided_slice_2/stack¨
-sequential_11/lstm_22/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_11/lstm_22/strided_slice_2/stack_1¨
-sequential_11/lstm_22/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_11/lstm_22/strided_slice_2/stack_2
%sequential_11/lstm_22/strided_slice_2StridedSlice#sequential_11/lstm_22/transpose:y:04sequential_11/lstm_22/strided_slice_2/stack:output:06sequential_11/lstm_22/strided_slice_2/stack_1:output:06sequential_11/lstm_22/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]*
shrink_axis_mask2'
%sequential_11/lstm_22/strided_slice_2÷
8sequential_11/lstm_22/lstm_cell_22/MatMul/ReadVariableOpReadVariableOpAsequential_11_lstm_22_lstm_cell_22_matmul_readvariableop_resource*
_output_shapes
:	]°*
dtype02:
8sequential_11/lstm_22/lstm_cell_22/MatMul/ReadVariableOp
)sequential_11/lstm_22/lstm_cell_22/MatMulMatMul.sequential_11/lstm_22/strided_slice_2:output:0@sequential_11/lstm_22/lstm_cell_22/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°2+
)sequential_11/lstm_22/lstm_cell_22/MatMulþ
:sequential_11/lstm_22/lstm_cell_22/MatMul_1/ReadVariableOpReadVariableOpCsequential_11_lstm_22_lstm_cell_22_matmul_1_readvariableop_resource* 
_output_shapes
:
¬°*
dtype02<
:sequential_11/lstm_22/lstm_cell_22/MatMul_1/ReadVariableOp
+sequential_11/lstm_22/lstm_cell_22/MatMul_1MatMul$sequential_11/lstm_22/zeros:output:0Bsequential_11/lstm_22/lstm_cell_22/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°2-
+sequential_11/lstm_22/lstm_cell_22/MatMul_1ø
&sequential_11/lstm_22/lstm_cell_22/addAddV23sequential_11/lstm_22/lstm_cell_22/MatMul:product:05sequential_11/lstm_22/lstm_cell_22/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°2(
&sequential_11/lstm_22/lstm_cell_22/addö
9sequential_11/lstm_22/lstm_cell_22/BiasAdd/ReadVariableOpReadVariableOpBsequential_11_lstm_22_lstm_cell_22_biasadd_readvariableop_resource*
_output_shapes	
:°*
dtype02;
9sequential_11/lstm_22/lstm_cell_22/BiasAdd/ReadVariableOp
*sequential_11/lstm_22/lstm_cell_22/BiasAddBiasAdd*sequential_11/lstm_22/lstm_cell_22/add:z:0Asequential_11/lstm_22/lstm_cell_22/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°2,
*sequential_11/lstm_22/lstm_cell_22/BiasAddª
2sequential_11/lstm_22/lstm_cell_22/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :24
2sequential_11/lstm_22/lstm_cell_22/split/split_dimÏ
(sequential_11/lstm_22/lstm_cell_22/splitSplit;sequential_11/lstm_22/lstm_cell_22/split/split_dim:output:03sequential_11/lstm_22/lstm_cell_22/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ¬*
	num_split2*
(sequential_11/lstm_22/lstm_cell_22/splitÉ
*sequential_11/lstm_22/lstm_cell_22/SigmoidSigmoid1sequential_11/lstm_22/lstm_cell_22/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2,
*sequential_11/lstm_22/lstm_cell_22/SigmoidÍ
,sequential_11/lstm_22/lstm_cell_22/Sigmoid_1Sigmoid1sequential_11/lstm_22/lstm_cell_22/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2.
,sequential_11/lstm_22/lstm_cell_22/Sigmoid_1ä
&sequential_11/lstm_22/lstm_cell_22/mulMul0sequential_11/lstm_22/lstm_cell_22/Sigmoid_1:y:0&sequential_11/lstm_22/zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2(
&sequential_11/lstm_22/lstm_cell_22/mulÀ
'sequential_11/lstm_22/lstm_cell_22/ReluRelu1sequential_11/lstm_22/lstm_cell_22/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2)
'sequential_11/lstm_22/lstm_cell_22/Reluõ
(sequential_11/lstm_22/lstm_cell_22/mul_1Mul.sequential_11/lstm_22/lstm_cell_22/Sigmoid:y:05sequential_11/lstm_22/lstm_cell_22/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2*
(sequential_11/lstm_22/lstm_cell_22/mul_1ê
(sequential_11/lstm_22/lstm_cell_22/add_1AddV2*sequential_11/lstm_22/lstm_cell_22/mul:z:0,sequential_11/lstm_22/lstm_cell_22/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2*
(sequential_11/lstm_22/lstm_cell_22/add_1Í
,sequential_11/lstm_22/lstm_cell_22/Sigmoid_2Sigmoid1sequential_11/lstm_22/lstm_cell_22/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2.
,sequential_11/lstm_22/lstm_cell_22/Sigmoid_2¿
)sequential_11/lstm_22/lstm_cell_22/Relu_1Relu,sequential_11/lstm_22/lstm_cell_22/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2+
)sequential_11/lstm_22/lstm_cell_22/Relu_1ù
(sequential_11/lstm_22/lstm_cell_22/mul_2Mul0sequential_11/lstm_22/lstm_cell_22/Sigmoid_2:y:07sequential_11/lstm_22/lstm_cell_22/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2*
(sequential_11/lstm_22/lstm_cell_22/mul_2»
3sequential_11/lstm_22/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ¬   25
3sequential_11/lstm_22/TensorArrayV2_1/element_shape
%sequential_11/lstm_22/TensorArrayV2_1TensorListReserve<sequential_11/lstm_22/TensorArrayV2_1/element_shape:output:0.sequential_11/lstm_22/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02'
%sequential_11/lstm_22/TensorArrayV2_1z
sequential_11/lstm_22/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
sequential_11/lstm_22/time«
.sequential_11/lstm_22/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ20
.sequential_11/lstm_22/while/maximum_iterations
(sequential_11/lstm_22/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2*
(sequential_11/lstm_22/while/loop_counterÝ
sequential_11/lstm_22/whileWhile1sequential_11/lstm_22/while/loop_counter:output:07sequential_11/lstm_22/while/maximum_iterations:output:0#sequential_11/lstm_22/time:output:0.sequential_11/lstm_22/TensorArrayV2_1:handle:0$sequential_11/lstm_22/zeros:output:0&sequential_11/lstm_22/zeros_1:output:0.sequential_11/lstm_22/strided_slice_1:output:0Msequential_11/lstm_22/TensorArrayUnstack/TensorListFromTensor:output_handle:0Asequential_11_lstm_22_lstm_cell_22_matmul_readvariableop_resourceCsequential_11_lstm_22_lstm_cell_22_matmul_1_readvariableop_resourceBsequential_11_lstm_22_lstm_cell_22_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ¬: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *5
body-R+
)sequential_11_lstm_22_while_body_39102288*5
cond-R+
)sequential_11_lstm_22_while_cond_39102287*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ¬: : : : : *
parallel_iterations 2
sequential_11/lstm_22/whileá
Fsequential_11/lstm_22/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ¬   2H
Fsequential_11/lstm_22/TensorArrayV2Stack/TensorListStack/element_shapeÁ
8sequential_11/lstm_22/TensorArrayV2Stack/TensorListStackTensorListStack$sequential_11/lstm_22/while:output:3Osequential_11/lstm_22/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬*
element_dtype02:
8sequential_11/lstm_22/TensorArrayV2Stack/TensorListStack­
+sequential_11/lstm_22/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2-
+sequential_11/lstm_22/strided_slice_3/stack¨
-sequential_11/lstm_22/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2/
-sequential_11/lstm_22/strided_slice_3/stack_1¨
-sequential_11/lstm_22/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_11/lstm_22/strided_slice_3/stack_2
%sequential_11/lstm_22/strided_slice_3StridedSliceAsequential_11/lstm_22/TensorArrayV2Stack/TensorListStack:tensor:04sequential_11/lstm_22/strided_slice_3/stack:output:06sequential_11/lstm_22/strided_slice_3/stack_1:output:06sequential_11/lstm_22/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬*
shrink_axis_mask2'
%sequential_11/lstm_22/strided_slice_3¥
&sequential_11/lstm_22/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2(
&sequential_11/lstm_22/transpose_1/permþ
!sequential_11/lstm_22/transpose_1	TransposeAsequential_11/lstm_22/TensorArrayV2Stack/TensorListStack:tensor:0/sequential_11/lstm_22/transpose_1/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2#
!sequential_11/lstm_22/transpose_1
sequential_11/lstm_22/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
sequential_11/lstm_22/runtime°
!sequential_11/dropout_22/IdentityIdentity%sequential_11/lstm_22/transpose_1:y:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2#
!sequential_11/dropout_22/Identity
sequential_11/lstm_23/ShapeShape*sequential_11/dropout_22/Identity:output:0*
T0*
_output_shapes
:2
sequential_11/lstm_23/Shape 
)sequential_11/lstm_23/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)sequential_11/lstm_23/strided_slice/stack¤
+sequential_11/lstm_23/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential_11/lstm_23/strided_slice/stack_1¤
+sequential_11/lstm_23/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential_11/lstm_23/strided_slice/stack_2æ
#sequential_11/lstm_23/strided_sliceStridedSlice$sequential_11/lstm_23/Shape:output:02sequential_11/lstm_23/strided_slice/stack:output:04sequential_11/lstm_23/strided_slice/stack_1:output:04sequential_11/lstm_23/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#sequential_11/lstm_23/strided_slice
!sequential_11/lstm_23/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :2#
!sequential_11/lstm_23/zeros/mul/yÄ
sequential_11/lstm_23/zeros/mulMul,sequential_11/lstm_23/strided_slice:output:0*sequential_11/lstm_23/zeros/mul/y:output:0*
T0*
_output_shapes
: 2!
sequential_11/lstm_23/zeros/mul
"sequential_11/lstm_23/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2$
"sequential_11/lstm_23/zeros/Less/y¿
 sequential_11/lstm_23/zeros/LessLess#sequential_11/lstm_23/zeros/mul:z:0+sequential_11/lstm_23/zeros/Less/y:output:0*
T0*
_output_shapes
: 2"
 sequential_11/lstm_23/zeros/Less
$sequential_11/lstm_23/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :2&
$sequential_11/lstm_23/zeros/packed/1Û
"sequential_11/lstm_23/zeros/packedPack,sequential_11/lstm_23/strided_slice:output:0-sequential_11/lstm_23/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2$
"sequential_11/lstm_23/zeros/packed
!sequential_11/lstm_23/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!sequential_11/lstm_23/zeros/ConstÎ
sequential_11/lstm_23/zerosFill+sequential_11/lstm_23/zeros/packed:output:0*sequential_11/lstm_23/zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential_11/lstm_23/zeros
#sequential_11/lstm_23/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :2%
#sequential_11/lstm_23/zeros_1/mul/yÊ
!sequential_11/lstm_23/zeros_1/mulMul,sequential_11/lstm_23/strided_slice:output:0,sequential_11/lstm_23/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2#
!sequential_11/lstm_23/zeros_1/mul
$sequential_11/lstm_23/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2&
$sequential_11/lstm_23/zeros_1/Less/yÇ
"sequential_11/lstm_23/zeros_1/LessLess%sequential_11/lstm_23/zeros_1/mul:z:0-sequential_11/lstm_23/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2$
"sequential_11/lstm_23/zeros_1/Less
&sequential_11/lstm_23/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :2(
&sequential_11/lstm_23/zeros_1/packed/1á
$sequential_11/lstm_23/zeros_1/packedPack,sequential_11/lstm_23/strided_slice:output:0/sequential_11/lstm_23/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2&
$sequential_11/lstm_23/zeros_1/packed
#sequential_11/lstm_23/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2%
#sequential_11/lstm_23/zeros_1/ConstÖ
sequential_11/lstm_23/zeros_1Fill-sequential_11/lstm_23/zeros_1/packed:output:0,sequential_11/lstm_23/zeros_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential_11/lstm_23/zeros_1¡
$sequential_11/lstm_23/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2&
$sequential_11/lstm_23/transpose/permá
sequential_11/lstm_23/transpose	Transpose*sequential_11/dropout_22/Identity:output:0-sequential_11/lstm_23/transpose/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2!
sequential_11/lstm_23/transpose
sequential_11/lstm_23/Shape_1Shape#sequential_11/lstm_23/transpose:y:0*
T0*
_output_shapes
:2
sequential_11/lstm_23/Shape_1¤
+sequential_11/lstm_23/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2-
+sequential_11/lstm_23/strided_slice_1/stack¨
-sequential_11/lstm_23/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_11/lstm_23/strided_slice_1/stack_1¨
-sequential_11/lstm_23/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_11/lstm_23/strided_slice_1/stack_2ò
%sequential_11/lstm_23/strided_slice_1StridedSlice&sequential_11/lstm_23/Shape_1:output:04sequential_11/lstm_23/strided_slice_1/stack:output:06sequential_11/lstm_23/strided_slice_1/stack_1:output:06sequential_11/lstm_23/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%sequential_11/lstm_23/strided_slice_1±
1sequential_11/lstm_23/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ23
1sequential_11/lstm_23/TensorArrayV2/element_shape
#sequential_11/lstm_23/TensorArrayV2TensorListReserve:sequential_11/lstm_23/TensorArrayV2/element_shape:output:0.sequential_11/lstm_23/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02%
#sequential_11/lstm_23/TensorArrayV2ë
Ksequential_11/lstm_23/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ¬   2M
Ksequential_11/lstm_23/TensorArrayUnstack/TensorListFromTensor/element_shapeÐ
=sequential_11/lstm_23/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor#sequential_11/lstm_23/transpose:y:0Tsequential_11/lstm_23/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02?
=sequential_11/lstm_23/TensorArrayUnstack/TensorListFromTensor¤
+sequential_11/lstm_23/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2-
+sequential_11/lstm_23/strided_slice_2/stack¨
-sequential_11/lstm_23/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_11/lstm_23/strided_slice_2/stack_1¨
-sequential_11/lstm_23/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_11/lstm_23/strided_slice_2/stack_2
%sequential_11/lstm_23/strided_slice_2StridedSlice#sequential_11/lstm_23/transpose:y:04sequential_11/lstm_23/strided_slice_2/stack:output:06sequential_11/lstm_23/strided_slice_2/stack_1:output:06sequential_11/lstm_23/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬*
shrink_axis_mask2'
%sequential_11/lstm_23/strided_slice_2ø
8sequential_11/lstm_23/lstm_cell_23/MatMul/ReadVariableOpReadVariableOpAsequential_11_lstm_23_lstm_cell_23_matmul_readvariableop_resource* 
_output_shapes
:
¬*
dtype02:
8sequential_11/lstm_23/lstm_cell_23/MatMul/ReadVariableOp
)sequential_11/lstm_23/lstm_cell_23/MatMulMatMul.sequential_11/lstm_23/strided_slice_2:output:0@sequential_11/lstm_23/lstm_cell_23/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2+
)sequential_11/lstm_23/lstm_cell_23/MatMulþ
:sequential_11/lstm_23/lstm_cell_23/MatMul_1/ReadVariableOpReadVariableOpCsequential_11_lstm_23_lstm_cell_23_matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype02<
:sequential_11/lstm_23/lstm_cell_23/MatMul_1/ReadVariableOp
+sequential_11/lstm_23/lstm_cell_23/MatMul_1MatMul$sequential_11/lstm_23/zeros:output:0Bsequential_11/lstm_23/lstm_cell_23/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2-
+sequential_11/lstm_23/lstm_cell_23/MatMul_1ø
&sequential_11/lstm_23/lstm_cell_23/addAddV23sequential_11/lstm_23/lstm_cell_23/MatMul:product:05sequential_11/lstm_23/lstm_cell_23/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&sequential_11/lstm_23/lstm_cell_23/addö
9sequential_11/lstm_23/lstm_cell_23/BiasAdd/ReadVariableOpReadVariableOpBsequential_11_lstm_23_lstm_cell_23_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02;
9sequential_11/lstm_23/lstm_cell_23/BiasAdd/ReadVariableOp
*sequential_11/lstm_23/lstm_cell_23/BiasAddBiasAdd*sequential_11/lstm_23/lstm_cell_23/add:z:0Asequential_11/lstm_23/lstm_cell_23/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2,
*sequential_11/lstm_23/lstm_cell_23/BiasAddª
2sequential_11/lstm_23/lstm_cell_23/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :24
2sequential_11/lstm_23/lstm_cell_23/split/split_dimÏ
(sequential_11/lstm_23/lstm_cell_23/splitSplit;sequential_11/lstm_23/lstm_cell_23/split/split_dim:output:03sequential_11/lstm_23/lstm_cell_23/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2*
(sequential_11/lstm_23/lstm_cell_23/splitÉ
*sequential_11/lstm_23/lstm_cell_23/SigmoidSigmoid1sequential_11/lstm_23/lstm_cell_23/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2,
*sequential_11/lstm_23/lstm_cell_23/SigmoidÍ
,sequential_11/lstm_23/lstm_cell_23/Sigmoid_1Sigmoid1sequential_11/lstm_23/lstm_cell_23/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2.
,sequential_11/lstm_23/lstm_cell_23/Sigmoid_1ä
&sequential_11/lstm_23/lstm_cell_23/mulMul0sequential_11/lstm_23/lstm_cell_23/Sigmoid_1:y:0&sequential_11/lstm_23/zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&sequential_11/lstm_23/lstm_cell_23/mulÀ
'sequential_11/lstm_23/lstm_cell_23/ReluRelu1sequential_11/lstm_23/lstm_cell_23/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'sequential_11/lstm_23/lstm_cell_23/Reluõ
(sequential_11/lstm_23/lstm_cell_23/mul_1Mul.sequential_11/lstm_23/lstm_cell_23/Sigmoid:y:05sequential_11/lstm_23/lstm_cell_23/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
(sequential_11/lstm_23/lstm_cell_23/mul_1ê
(sequential_11/lstm_23/lstm_cell_23/add_1AddV2*sequential_11/lstm_23/lstm_cell_23/mul:z:0,sequential_11/lstm_23/lstm_cell_23/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
(sequential_11/lstm_23/lstm_cell_23/add_1Í
,sequential_11/lstm_23/lstm_cell_23/Sigmoid_2Sigmoid1sequential_11/lstm_23/lstm_cell_23/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2.
,sequential_11/lstm_23/lstm_cell_23/Sigmoid_2¿
)sequential_11/lstm_23/lstm_cell_23/Relu_1Relu,sequential_11/lstm_23/lstm_cell_23/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2+
)sequential_11/lstm_23/lstm_cell_23/Relu_1ù
(sequential_11/lstm_23/lstm_cell_23/mul_2Mul0sequential_11/lstm_23/lstm_cell_23/Sigmoid_2:y:07sequential_11/lstm_23/lstm_cell_23/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
(sequential_11/lstm_23/lstm_cell_23/mul_2»
3sequential_11/lstm_23/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   25
3sequential_11/lstm_23/TensorArrayV2_1/element_shape
%sequential_11/lstm_23/TensorArrayV2_1TensorListReserve<sequential_11/lstm_23/TensorArrayV2_1/element_shape:output:0.sequential_11/lstm_23/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02'
%sequential_11/lstm_23/TensorArrayV2_1z
sequential_11/lstm_23/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
sequential_11/lstm_23/time«
.sequential_11/lstm_23/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ20
.sequential_11/lstm_23/while/maximum_iterations
(sequential_11/lstm_23/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2*
(sequential_11/lstm_23/while/loop_counterÝ
sequential_11/lstm_23/whileWhile1sequential_11/lstm_23/while/loop_counter:output:07sequential_11/lstm_23/while/maximum_iterations:output:0#sequential_11/lstm_23/time:output:0.sequential_11/lstm_23/TensorArrayV2_1:handle:0$sequential_11/lstm_23/zeros:output:0&sequential_11/lstm_23/zeros_1:output:0.sequential_11/lstm_23/strided_slice_1:output:0Msequential_11/lstm_23/TensorArrayUnstack/TensorListFromTensor:output_handle:0Asequential_11_lstm_23_lstm_cell_23_matmul_readvariableop_resourceCsequential_11_lstm_23_lstm_cell_23_matmul_1_readvariableop_resourceBsequential_11_lstm_23_lstm_cell_23_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *5
body-R+
)sequential_11_lstm_23_while_body_39102436*5
cond-R+
)sequential_11_lstm_23_while_cond_39102435*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 2
sequential_11/lstm_23/whileá
Fsequential_11/lstm_23/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2H
Fsequential_11/lstm_23/TensorArrayV2Stack/TensorListStack/element_shapeÁ
8sequential_11/lstm_23/TensorArrayV2Stack/TensorListStackTensorListStack$sequential_11/lstm_23/while:output:3Osequential_11/lstm_23/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype02:
8sequential_11/lstm_23/TensorArrayV2Stack/TensorListStack­
+sequential_11/lstm_23/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2-
+sequential_11/lstm_23/strided_slice_3/stack¨
-sequential_11/lstm_23/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2/
-sequential_11/lstm_23/strided_slice_3/stack_1¨
-sequential_11/lstm_23/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_11/lstm_23/strided_slice_3/stack_2
%sequential_11/lstm_23/strided_slice_3StridedSliceAsequential_11/lstm_23/TensorArrayV2Stack/TensorListStack:tensor:04sequential_11/lstm_23/strided_slice_3/stack:output:06sequential_11/lstm_23/strided_slice_3/stack_1:output:06sequential_11/lstm_23/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2'
%sequential_11/lstm_23/strided_slice_3¥
&sequential_11/lstm_23/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2(
&sequential_11/lstm_23/transpose_1/permþ
!sequential_11/lstm_23/transpose_1	TransposeAsequential_11/lstm_23/TensorArrayV2Stack/TensorListStack:tensor:0/sequential_11/lstm_23/transpose_1/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!sequential_11/lstm_23/transpose_1
sequential_11/lstm_23/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
sequential_11/lstm_23/runtime°
!sequential_11/dropout_23/IdentityIdentity%sequential_11/lstm_23/transpose_1:y:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!sequential_11/dropout_23/IdentityÜ
/sequential_11/dense_11/Tensordot/ReadVariableOpReadVariableOp8sequential_11_dense_11_tensordot_readvariableop_resource*
_output_shapes
:	*
dtype021
/sequential_11/dense_11/Tensordot/ReadVariableOp
%sequential_11/dense_11/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2'
%sequential_11/dense_11/Tensordot/axes
%sequential_11/dense_11/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2'
%sequential_11/dense_11/Tensordot/freeª
&sequential_11/dense_11/Tensordot/ShapeShape*sequential_11/dropout_23/Identity:output:0*
T0*
_output_shapes
:2(
&sequential_11/dense_11/Tensordot/Shape¢
.sequential_11/dense_11/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.sequential_11/dense_11/Tensordot/GatherV2/axisÄ
)sequential_11/dense_11/Tensordot/GatherV2GatherV2/sequential_11/dense_11/Tensordot/Shape:output:0.sequential_11/dense_11/Tensordot/free:output:07sequential_11/dense_11/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2+
)sequential_11/dense_11/Tensordot/GatherV2¦
0sequential_11/dense_11/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 22
0sequential_11/dense_11/Tensordot/GatherV2_1/axisÊ
+sequential_11/dense_11/Tensordot/GatherV2_1GatherV2/sequential_11/dense_11/Tensordot/Shape:output:0.sequential_11/dense_11/Tensordot/axes:output:09sequential_11/dense_11/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2-
+sequential_11/dense_11/Tensordot/GatherV2_1
&sequential_11/dense_11/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2(
&sequential_11/dense_11/Tensordot/ConstÜ
%sequential_11/dense_11/Tensordot/ProdProd2sequential_11/dense_11/Tensordot/GatherV2:output:0/sequential_11/dense_11/Tensordot/Const:output:0*
T0*
_output_shapes
: 2'
%sequential_11/dense_11/Tensordot/Prod
(sequential_11/dense_11/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(sequential_11/dense_11/Tensordot/Const_1ä
'sequential_11/dense_11/Tensordot/Prod_1Prod4sequential_11/dense_11/Tensordot/GatherV2_1:output:01sequential_11/dense_11/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2)
'sequential_11/dense_11/Tensordot/Prod_1
,sequential_11/dense_11/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,sequential_11/dense_11/Tensordot/concat/axis£
'sequential_11/dense_11/Tensordot/concatConcatV2.sequential_11/dense_11/Tensordot/free:output:0.sequential_11/dense_11/Tensordot/axes:output:05sequential_11/dense_11/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2)
'sequential_11/dense_11/Tensordot/concatè
&sequential_11/dense_11/Tensordot/stackPack.sequential_11/dense_11/Tensordot/Prod:output:00sequential_11/dense_11/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2(
&sequential_11/dense_11/Tensordot/stackú
*sequential_11/dense_11/Tensordot/transpose	Transpose*sequential_11/dropout_23/Identity:output:00sequential_11/dense_11/Tensordot/concat:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2,
*sequential_11/dense_11/Tensordot/transposeû
(sequential_11/dense_11/Tensordot/ReshapeReshape.sequential_11/dense_11/Tensordot/transpose:y:0/sequential_11/dense_11/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2*
(sequential_11/dense_11/Tensordot/Reshapeú
'sequential_11/dense_11/Tensordot/MatMulMatMul1sequential_11/dense_11/Tensordot/Reshape:output:07sequential_11/dense_11/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'sequential_11/dense_11/Tensordot/MatMul
(sequential_11/dense_11/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(sequential_11/dense_11/Tensordot/Const_2¢
.sequential_11/dense_11/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.sequential_11/dense_11/Tensordot/concat_1/axis°
)sequential_11/dense_11/Tensordot/concat_1ConcatV22sequential_11/dense_11/Tensordot/GatherV2:output:01sequential_11/dense_11/Tensordot/Const_2:output:07sequential_11/dense_11/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2+
)sequential_11/dense_11/Tensordot/concat_1ì
 sequential_11/dense_11/TensordotReshape1sequential_11/dense_11/Tensordot/MatMul:product:02sequential_11/dense_11/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 sequential_11/dense_11/TensordotÑ
-sequential_11/dense_11/BiasAdd/ReadVariableOpReadVariableOp6sequential_11_dense_11_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02/
-sequential_11/dense_11/BiasAdd/ReadVariableOpã
sequential_11/dense_11/BiasAddBiasAdd)sequential_11/dense_11/Tensordot:output:05sequential_11/dense_11/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
sequential_11/dense_11/BiasAddª
sequential_11/dense_11/SoftmaxSoftmax'sequential_11/dense_11/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
sequential_11/dense_11/Softmax
IdentityIdentity(sequential_11/dense_11/Softmax:softmax:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

IdentityÔ
NoOpNoOp.^sequential_11/dense_11/BiasAdd/ReadVariableOp0^sequential_11/dense_11/Tensordot/ReadVariableOp:^sequential_11/lstm_22/lstm_cell_22/BiasAdd/ReadVariableOp9^sequential_11/lstm_22/lstm_cell_22/MatMul/ReadVariableOp;^sequential_11/lstm_22/lstm_cell_22/MatMul_1/ReadVariableOp^sequential_11/lstm_22/while:^sequential_11/lstm_23/lstm_cell_23/BiasAdd/ReadVariableOp9^sequential_11/lstm_23/lstm_cell_23/MatMul/ReadVariableOp;^sequential_11/lstm_23/lstm_cell_23/MatMul_1/ReadVariableOp^sequential_11/lstm_23/while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ]: : : : : : : : 2^
-sequential_11/dense_11/BiasAdd/ReadVariableOp-sequential_11/dense_11/BiasAdd/ReadVariableOp2b
/sequential_11/dense_11/Tensordot/ReadVariableOp/sequential_11/dense_11/Tensordot/ReadVariableOp2v
9sequential_11/lstm_22/lstm_cell_22/BiasAdd/ReadVariableOp9sequential_11/lstm_22/lstm_cell_22/BiasAdd/ReadVariableOp2t
8sequential_11/lstm_22/lstm_cell_22/MatMul/ReadVariableOp8sequential_11/lstm_22/lstm_cell_22/MatMul/ReadVariableOp2x
:sequential_11/lstm_22/lstm_cell_22/MatMul_1/ReadVariableOp:sequential_11/lstm_22/lstm_cell_22/MatMul_1/ReadVariableOp2:
sequential_11/lstm_22/whilesequential_11/lstm_22/while2v
9sequential_11/lstm_23/lstm_cell_23/BiasAdd/ReadVariableOp9sequential_11/lstm_23/lstm_cell_23/BiasAdd/ReadVariableOp2t
8sequential_11/lstm_23/lstm_cell_23/MatMul/ReadVariableOp8sequential_11/lstm_23/lstm_cell_23/MatMul/ReadVariableOp2x
:sequential_11/lstm_23/lstm_cell_23/MatMul_1/ReadVariableOp:sequential_11/lstm_23/lstm_cell_23/MatMul_1/ReadVariableOp2:
sequential_11/lstm_23/whilesequential_11/lstm_23/while:Z V
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
'
_user_specified_namelstm_22_input
ã
Í
while_cond_39104045
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_39104045___redundant_placeholder06
2while_while_cond_39104045___redundant_placeholder16
2while_while_cond_39104045___redundant_placeholder26
2while_while_cond_39104045___redundant_placeholder3
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
B: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 
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
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
Ã\
 
E__inference_lstm_22_layer_call_and_return_conditional_losses_39105821
inputs_0>
+lstm_cell_22_matmul_readvariableop_resource:	]°A
-lstm_cell_22_matmul_1_readvariableop_resource:
¬°;
,lstm_cell_22_biasadd_readvariableop_resource:	°
identity¢#lstm_cell_22/BiasAdd/ReadVariableOp¢"lstm_cell_22/MatMul/ReadVariableOp¢$lstm_cell_22/MatMul_1/ReadVariableOp¢whileF
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
strided_slice/stack_2â
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
B :¬2
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
B :è2
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
B :¬2
zeros/packed/1
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
:ÿÿÿÿÿÿÿÿÿ¬2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :¬2
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
B :è2
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
B :¬2
zeros_1/packed/1
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
:ÿÿÿÿÿÿÿÿÿ¬2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ]2
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
strided_slice_1/stack_2î
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
TensorArrayV2/element_shape²
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2¿
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ]   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeø
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
strided_slice_2/stack_2ü
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]*
shrink_axis_mask2
strided_slice_2µ
"lstm_cell_22/MatMul/ReadVariableOpReadVariableOp+lstm_cell_22_matmul_readvariableop_resource*
_output_shapes
:	]°*
dtype02$
"lstm_cell_22/MatMul/ReadVariableOp­
lstm_cell_22/MatMulMatMulstrided_slice_2:output:0*lstm_cell_22/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°2
lstm_cell_22/MatMul¼
$lstm_cell_22/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_22_matmul_1_readvariableop_resource* 
_output_shapes
:
¬°*
dtype02&
$lstm_cell_22/MatMul_1/ReadVariableOp©
lstm_cell_22/MatMul_1MatMulzeros:output:0,lstm_cell_22/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°2
lstm_cell_22/MatMul_1 
lstm_cell_22/addAddV2lstm_cell_22/MatMul:product:0lstm_cell_22/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°2
lstm_cell_22/add´
#lstm_cell_22/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_22_biasadd_readvariableop_resource*
_output_shapes	
:°*
dtype02%
#lstm_cell_22/BiasAdd/ReadVariableOp­
lstm_cell_22/BiasAddBiasAddlstm_cell_22/add:z:0+lstm_cell_22/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°2
lstm_cell_22/BiasAdd~
lstm_cell_22/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_22/split/split_dim÷
lstm_cell_22/splitSplit%lstm_cell_22/split/split_dim:output:0lstm_cell_22/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ¬*
	num_split2
lstm_cell_22/split
lstm_cell_22/SigmoidSigmoidlstm_cell_22/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
lstm_cell_22/Sigmoid
lstm_cell_22/Sigmoid_1Sigmoidlstm_cell_22/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
lstm_cell_22/Sigmoid_1
lstm_cell_22/mulMullstm_cell_22/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
lstm_cell_22/mul~
lstm_cell_22/ReluRelulstm_cell_22/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
lstm_cell_22/Relu
lstm_cell_22/mul_1Mullstm_cell_22/Sigmoid:y:0lstm_cell_22/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
lstm_cell_22/mul_1
lstm_cell_22/add_1AddV2lstm_cell_22/mul:z:0lstm_cell_22/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
lstm_cell_22/add_1
lstm_cell_22/Sigmoid_2Sigmoidlstm_cell_22/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
lstm_cell_22/Sigmoid_2}
lstm_cell_22/Relu_1Relulstm_cell_22/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
lstm_cell_22/Relu_1¡
lstm_cell_22/mul_2Mullstm_cell_22/Sigmoid_2:y:0!lstm_cell_22/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
lstm_cell_22/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ¬   2
TensorArrayV2_1/element_shape¸
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
ÿÿÿÿÿÿÿÿÿ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_22_matmul_readvariableop_resource-lstm_cell_22_matmul_1_readvariableop_resource,lstm_cell_22_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ¬: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_39105737*
condR
while_cond_39105736*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ¬: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ¬   22
0TensorArrayV2Stack/TensorListStack/element_shapeò
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
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
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm¯
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬2
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
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬2

IdentityÈ
NoOpNoOp$^lstm_cell_22/BiasAdd/ReadVariableOp#^lstm_cell_22/MatMul/ReadVariableOp%^lstm_cell_22/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ]: : : 2J
#lstm_cell_22/BiasAdd/ReadVariableOp#lstm_cell_22/BiasAdd/ReadVariableOp2H
"lstm_cell_22/MatMul/ReadVariableOp"lstm_cell_22/MatMul/ReadVariableOp2L
$lstm_cell_22/MatMul_1/ReadVariableOp$lstm_cell_22/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ]
"
_user_specified_name
inputs/0
Ô

í
lstm_23_while_cond_39105355,
(lstm_23_while_lstm_23_while_loop_counter2
.lstm_23_while_lstm_23_while_maximum_iterations
lstm_23_while_placeholder
lstm_23_while_placeholder_1
lstm_23_while_placeholder_2
lstm_23_while_placeholder_3.
*lstm_23_while_less_lstm_23_strided_slice_1F
Blstm_23_while_lstm_23_while_cond_39105355___redundant_placeholder0F
Blstm_23_while_lstm_23_while_cond_39105355___redundant_placeholder1F
Blstm_23_while_lstm_23_while_cond_39105355___redundant_placeholder2F
Blstm_23_while_lstm_23_while_cond_39105355___redundant_placeholder3
lstm_23_while_identity

lstm_23/while/LessLesslstm_23_while_placeholder*lstm_23_while_less_lstm_23_strided_slice_1*
T0*
_output_shapes
: 2
lstm_23/while/Lessu
lstm_23/while/IdentityIdentitylstm_23/while/Less:z:0*
T0
*
_output_shapes
: 2
lstm_23/while/Identity"9
lstm_23_while_identitylstm_23/while/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 
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
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
ã
Í
while_cond_39104510
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_39104510___redundant_placeholder06
2while_while_cond_39104510___redundant_placeholder16
2while_while_cond_39104510___redundant_placeholder26
2while_while_cond_39104510___redundant_placeholder3
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
B: : : : :ÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ¬: ::::: 
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
:ÿÿÿÿÿÿÿÿÿ¬:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬:

_output_shapes
: :

_output_shapes
:

f
H__inference_dropout_23_layer_call_and_return_conditional_losses_39106813

inputs

identity_1_
IdentityIdentityinputs*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityn

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ã
Í
while_cond_39104314
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_39104314___redundant_placeholder06
2while_while_cond_39104314___redundant_placeholder16
2while_while_cond_39104314___redundant_placeholder26
2while_while_cond_39104314___redundant_placeholder3
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
B: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 
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
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
ò

K__inference_sequential_11_layer_call_and_return_conditional_losses_39105475

inputsF
3lstm_22_lstm_cell_22_matmul_readvariableop_resource:	]°I
5lstm_22_lstm_cell_22_matmul_1_readvariableop_resource:
¬°C
4lstm_22_lstm_cell_22_biasadd_readvariableop_resource:	°G
3lstm_23_lstm_cell_23_matmul_readvariableop_resource:
¬I
5lstm_23_lstm_cell_23_matmul_1_readvariableop_resource:
C
4lstm_23_lstm_cell_23_biasadd_readvariableop_resource:	=
*dense_11_tensordot_readvariableop_resource:	6
(dense_11_biasadd_readvariableop_resource:
identity¢dense_11/BiasAdd/ReadVariableOp¢!dense_11/Tensordot/ReadVariableOp¢+lstm_22/lstm_cell_22/BiasAdd/ReadVariableOp¢*lstm_22/lstm_cell_22/MatMul/ReadVariableOp¢,lstm_22/lstm_cell_22/MatMul_1/ReadVariableOp¢lstm_22/while¢+lstm_23/lstm_cell_23/BiasAdd/ReadVariableOp¢*lstm_23/lstm_cell_23/MatMul/ReadVariableOp¢,lstm_23/lstm_cell_23/MatMul_1/ReadVariableOp¢lstm_23/whileT
lstm_22/ShapeShapeinputs*
T0*
_output_shapes
:2
lstm_22/Shape
lstm_22/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_22/strided_slice/stack
lstm_22/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_22/strided_slice/stack_1
lstm_22/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_22/strided_slice/stack_2
lstm_22/strided_sliceStridedSlicelstm_22/Shape:output:0$lstm_22/strided_slice/stack:output:0&lstm_22/strided_slice/stack_1:output:0&lstm_22/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_22/strided_slicem
lstm_22/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :¬2
lstm_22/zeros/mul/y
lstm_22/zeros/mulMullstm_22/strided_slice:output:0lstm_22/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_22/zeros/mulo
lstm_22/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
lstm_22/zeros/Less/y
lstm_22/zeros/LessLesslstm_22/zeros/mul:z:0lstm_22/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_22/zeros/Lesss
lstm_22/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :¬2
lstm_22/zeros/packed/1£
lstm_22/zeros/packedPacklstm_22/strided_slice:output:0lstm_22/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_22/zeros/packedo
lstm_22/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_22/zeros/Const
lstm_22/zerosFilllstm_22/zeros/packed:output:0lstm_22/zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
lstm_22/zerosq
lstm_22/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :¬2
lstm_22/zeros_1/mul/y
lstm_22/zeros_1/mulMullstm_22/strided_slice:output:0lstm_22/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_22/zeros_1/muls
lstm_22/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
lstm_22/zeros_1/Less/y
lstm_22/zeros_1/LessLesslstm_22/zeros_1/mul:z:0lstm_22/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_22/zeros_1/Lessw
lstm_22/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :¬2
lstm_22/zeros_1/packed/1©
lstm_22/zeros_1/packedPacklstm_22/strided_slice:output:0!lstm_22/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_22/zeros_1/packeds
lstm_22/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_22/zeros_1/Const
lstm_22/zeros_1Filllstm_22/zeros_1/packed:output:0lstm_22/zeros_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
lstm_22/zeros_1
lstm_22/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_22/transpose/perm
lstm_22/transpose	Transposeinputslstm_22/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]2
lstm_22/transposeg
lstm_22/Shape_1Shapelstm_22/transpose:y:0*
T0*
_output_shapes
:2
lstm_22/Shape_1
lstm_22/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_22/strided_slice_1/stack
lstm_22/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_22/strided_slice_1/stack_1
lstm_22/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_22/strided_slice_1/stack_2
lstm_22/strided_slice_1StridedSlicelstm_22/Shape_1:output:0&lstm_22/strided_slice_1/stack:output:0(lstm_22/strided_slice_1/stack_1:output:0(lstm_22/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_22/strided_slice_1
#lstm_22/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2%
#lstm_22/TensorArrayV2/element_shapeÒ
lstm_22/TensorArrayV2TensorListReserve,lstm_22/TensorArrayV2/element_shape:output:0 lstm_22/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_22/TensorArrayV2Ï
=lstm_22/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ]   2?
=lstm_22/TensorArrayUnstack/TensorListFromTensor/element_shape
/lstm_22/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_22/transpose:y:0Flstm_22/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type021
/lstm_22/TensorArrayUnstack/TensorListFromTensor
lstm_22/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_22/strided_slice_2/stack
lstm_22/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_22/strided_slice_2/stack_1
lstm_22/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_22/strided_slice_2/stack_2¬
lstm_22/strided_slice_2StridedSlicelstm_22/transpose:y:0&lstm_22/strided_slice_2/stack:output:0(lstm_22/strided_slice_2/stack_1:output:0(lstm_22/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]*
shrink_axis_mask2
lstm_22/strided_slice_2Í
*lstm_22/lstm_cell_22/MatMul/ReadVariableOpReadVariableOp3lstm_22_lstm_cell_22_matmul_readvariableop_resource*
_output_shapes
:	]°*
dtype02,
*lstm_22/lstm_cell_22/MatMul/ReadVariableOpÍ
lstm_22/lstm_cell_22/MatMulMatMul lstm_22/strided_slice_2:output:02lstm_22/lstm_cell_22/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°2
lstm_22/lstm_cell_22/MatMulÔ
,lstm_22/lstm_cell_22/MatMul_1/ReadVariableOpReadVariableOp5lstm_22_lstm_cell_22_matmul_1_readvariableop_resource* 
_output_shapes
:
¬°*
dtype02.
,lstm_22/lstm_cell_22/MatMul_1/ReadVariableOpÉ
lstm_22/lstm_cell_22/MatMul_1MatMullstm_22/zeros:output:04lstm_22/lstm_cell_22/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°2
lstm_22/lstm_cell_22/MatMul_1À
lstm_22/lstm_cell_22/addAddV2%lstm_22/lstm_cell_22/MatMul:product:0'lstm_22/lstm_cell_22/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°2
lstm_22/lstm_cell_22/addÌ
+lstm_22/lstm_cell_22/BiasAdd/ReadVariableOpReadVariableOp4lstm_22_lstm_cell_22_biasadd_readvariableop_resource*
_output_shapes	
:°*
dtype02-
+lstm_22/lstm_cell_22/BiasAdd/ReadVariableOpÍ
lstm_22/lstm_cell_22/BiasAddBiasAddlstm_22/lstm_cell_22/add:z:03lstm_22/lstm_cell_22/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°2
lstm_22/lstm_cell_22/BiasAdd
$lstm_22/lstm_cell_22/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2&
$lstm_22/lstm_cell_22/split/split_dim
lstm_22/lstm_cell_22/splitSplit-lstm_22/lstm_cell_22/split/split_dim:output:0%lstm_22/lstm_cell_22/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ¬*
	num_split2
lstm_22/lstm_cell_22/split
lstm_22/lstm_cell_22/SigmoidSigmoid#lstm_22/lstm_cell_22/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
lstm_22/lstm_cell_22/Sigmoid£
lstm_22/lstm_cell_22/Sigmoid_1Sigmoid#lstm_22/lstm_cell_22/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2 
lstm_22/lstm_cell_22/Sigmoid_1¬
lstm_22/lstm_cell_22/mulMul"lstm_22/lstm_cell_22/Sigmoid_1:y:0lstm_22/zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
lstm_22/lstm_cell_22/mul
lstm_22/lstm_cell_22/ReluRelu#lstm_22/lstm_cell_22/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
lstm_22/lstm_cell_22/Relu½
lstm_22/lstm_cell_22/mul_1Mul lstm_22/lstm_cell_22/Sigmoid:y:0'lstm_22/lstm_cell_22/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
lstm_22/lstm_cell_22/mul_1²
lstm_22/lstm_cell_22/add_1AddV2lstm_22/lstm_cell_22/mul:z:0lstm_22/lstm_cell_22/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
lstm_22/lstm_cell_22/add_1£
lstm_22/lstm_cell_22/Sigmoid_2Sigmoid#lstm_22/lstm_cell_22/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2 
lstm_22/lstm_cell_22/Sigmoid_2
lstm_22/lstm_cell_22/Relu_1Relulstm_22/lstm_cell_22/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
lstm_22/lstm_cell_22/Relu_1Á
lstm_22/lstm_cell_22/mul_2Mul"lstm_22/lstm_cell_22/Sigmoid_2:y:0)lstm_22/lstm_cell_22/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
lstm_22/lstm_cell_22/mul_2
%lstm_22/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ¬   2'
%lstm_22/TensorArrayV2_1/element_shapeØ
lstm_22/TensorArrayV2_1TensorListReserve.lstm_22/TensorArrayV2_1/element_shape:output:0 lstm_22/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_22/TensorArrayV2_1^
lstm_22/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_22/time
 lstm_22/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2"
 lstm_22/while/maximum_iterationsz
lstm_22/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_22/while/loop_counter
lstm_22/whileWhile#lstm_22/while/loop_counter:output:0)lstm_22/while/maximum_iterations:output:0lstm_22/time:output:0 lstm_22/TensorArrayV2_1:handle:0lstm_22/zeros:output:0lstm_22/zeros_1:output:0 lstm_22/strided_slice_1:output:0?lstm_22/TensorArrayUnstack/TensorListFromTensor:output_handle:03lstm_22_lstm_cell_22_matmul_readvariableop_resource5lstm_22_lstm_cell_22_matmul_1_readvariableop_resource4lstm_22_lstm_cell_22_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ¬: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *'
bodyR
lstm_22_while_body_39105201*'
condR
lstm_22_while_cond_39105200*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ¬: : : : : *
parallel_iterations 2
lstm_22/whileÅ
8lstm_22/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ¬   2:
8lstm_22/TensorArrayV2Stack/TensorListStack/element_shape
*lstm_22/TensorArrayV2Stack/TensorListStackTensorListStacklstm_22/while:output:3Alstm_22/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬*
element_dtype02,
*lstm_22/TensorArrayV2Stack/TensorListStack
lstm_22/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
lstm_22/strided_slice_3/stack
lstm_22/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2!
lstm_22/strided_slice_3/stack_1
lstm_22/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_22/strided_slice_3/stack_2Ë
lstm_22/strided_slice_3StridedSlice3lstm_22/TensorArrayV2Stack/TensorListStack:tensor:0&lstm_22/strided_slice_3/stack:output:0(lstm_22/strided_slice_3/stack_1:output:0(lstm_22/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬*
shrink_axis_mask2
lstm_22/strided_slice_3
lstm_22/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_22/transpose_1/permÆ
lstm_22/transpose_1	Transpose3lstm_22/TensorArrayV2Stack/TensorListStack:tensor:0!lstm_22/transpose_1/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
lstm_22/transpose_1v
lstm_22/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_22/runtimey
dropout_22/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *«ªª?2
dropout_22/dropout/Constª
dropout_22/dropout/MulMullstm_22/transpose_1:y:0!dropout_22/dropout/Const:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
dropout_22/dropout/Mul{
dropout_22/dropout/ShapeShapelstm_22/transpose_1:y:0*
T0*
_output_shapes
:2
dropout_22/dropout/ShapeÚ
/dropout_22/dropout/random_uniform/RandomUniformRandomUniform!dropout_22/dropout/Shape:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬*
dtype021
/dropout_22/dropout/random_uniform/RandomUniform
!dropout_22/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  >2#
!dropout_22/dropout/GreaterEqual/yï
dropout_22/dropout/GreaterEqualGreaterEqual8dropout_22/dropout/random_uniform/RandomUniform:output:0*dropout_22/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2!
dropout_22/dropout/GreaterEqual¥
dropout_22/dropout/CastCast#dropout_22/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
dropout_22/dropout/Cast«
dropout_22/dropout/Mul_1Muldropout_22/dropout/Mul:z:0dropout_22/dropout/Cast:y:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
dropout_22/dropout/Mul_1j
lstm_23/ShapeShapedropout_22/dropout/Mul_1:z:0*
T0*
_output_shapes
:2
lstm_23/Shape
lstm_23/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_23/strided_slice/stack
lstm_23/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_23/strided_slice/stack_1
lstm_23/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_23/strided_slice/stack_2
lstm_23/strided_sliceStridedSlicelstm_23/Shape:output:0$lstm_23/strided_slice/stack:output:0&lstm_23/strided_slice/stack_1:output:0&lstm_23/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_23/strided_slicem
lstm_23/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :2
lstm_23/zeros/mul/y
lstm_23/zeros/mulMullstm_23/strided_slice:output:0lstm_23/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_23/zeros/mulo
lstm_23/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
lstm_23/zeros/Less/y
lstm_23/zeros/LessLesslstm_23/zeros/mul:z:0lstm_23/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_23/zeros/Lesss
lstm_23/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :2
lstm_23/zeros/packed/1£
lstm_23/zeros/packedPacklstm_23/strided_slice:output:0lstm_23/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_23/zeros/packedo
lstm_23/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_23/zeros/Const
lstm_23/zerosFilllstm_23/zeros/packed:output:0lstm_23/zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_23/zerosq
lstm_23/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :2
lstm_23/zeros_1/mul/y
lstm_23/zeros_1/mulMullstm_23/strided_slice:output:0lstm_23/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_23/zeros_1/muls
lstm_23/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
lstm_23/zeros_1/Less/y
lstm_23/zeros_1/LessLesslstm_23/zeros_1/mul:z:0lstm_23/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_23/zeros_1/Lessw
lstm_23/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :2
lstm_23/zeros_1/packed/1©
lstm_23/zeros_1/packedPacklstm_23/strided_slice:output:0!lstm_23/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_23/zeros_1/packeds
lstm_23/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_23/zeros_1/Const
lstm_23/zeros_1Filllstm_23/zeros_1/packed:output:0lstm_23/zeros_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_23/zeros_1
lstm_23/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_23/transpose/perm©
lstm_23/transpose	Transposedropout_22/dropout/Mul_1:z:0lstm_23/transpose/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
lstm_23/transposeg
lstm_23/Shape_1Shapelstm_23/transpose:y:0*
T0*
_output_shapes
:2
lstm_23/Shape_1
lstm_23/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_23/strided_slice_1/stack
lstm_23/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_23/strided_slice_1/stack_1
lstm_23/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_23/strided_slice_1/stack_2
lstm_23/strided_slice_1StridedSlicelstm_23/Shape_1:output:0&lstm_23/strided_slice_1/stack:output:0(lstm_23/strided_slice_1/stack_1:output:0(lstm_23/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_23/strided_slice_1
#lstm_23/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2%
#lstm_23/TensorArrayV2/element_shapeÒ
lstm_23/TensorArrayV2TensorListReserve,lstm_23/TensorArrayV2/element_shape:output:0 lstm_23/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_23/TensorArrayV2Ï
=lstm_23/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ¬   2?
=lstm_23/TensorArrayUnstack/TensorListFromTensor/element_shape
/lstm_23/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_23/transpose:y:0Flstm_23/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type021
/lstm_23/TensorArrayUnstack/TensorListFromTensor
lstm_23/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_23/strided_slice_2/stack
lstm_23/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_23/strided_slice_2/stack_1
lstm_23/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_23/strided_slice_2/stack_2­
lstm_23/strided_slice_2StridedSlicelstm_23/transpose:y:0&lstm_23/strided_slice_2/stack:output:0(lstm_23/strided_slice_2/stack_1:output:0(lstm_23/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬*
shrink_axis_mask2
lstm_23/strided_slice_2Î
*lstm_23/lstm_cell_23/MatMul/ReadVariableOpReadVariableOp3lstm_23_lstm_cell_23_matmul_readvariableop_resource* 
_output_shapes
:
¬*
dtype02,
*lstm_23/lstm_cell_23/MatMul/ReadVariableOpÍ
lstm_23/lstm_cell_23/MatMulMatMul lstm_23/strided_slice_2:output:02lstm_23/lstm_cell_23/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_23/lstm_cell_23/MatMulÔ
,lstm_23/lstm_cell_23/MatMul_1/ReadVariableOpReadVariableOp5lstm_23_lstm_cell_23_matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype02.
,lstm_23/lstm_cell_23/MatMul_1/ReadVariableOpÉ
lstm_23/lstm_cell_23/MatMul_1MatMullstm_23/zeros:output:04lstm_23/lstm_cell_23/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_23/lstm_cell_23/MatMul_1À
lstm_23/lstm_cell_23/addAddV2%lstm_23/lstm_cell_23/MatMul:product:0'lstm_23/lstm_cell_23/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_23/lstm_cell_23/addÌ
+lstm_23/lstm_cell_23/BiasAdd/ReadVariableOpReadVariableOp4lstm_23_lstm_cell_23_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02-
+lstm_23/lstm_cell_23/BiasAdd/ReadVariableOpÍ
lstm_23/lstm_cell_23/BiasAddBiasAddlstm_23/lstm_cell_23/add:z:03lstm_23/lstm_cell_23/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_23/lstm_cell_23/BiasAdd
$lstm_23/lstm_cell_23/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2&
$lstm_23/lstm_cell_23/split/split_dim
lstm_23/lstm_cell_23/splitSplit-lstm_23/lstm_cell_23/split/split_dim:output:0%lstm_23/lstm_cell_23/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2
lstm_23/lstm_cell_23/split
lstm_23/lstm_cell_23/SigmoidSigmoid#lstm_23/lstm_cell_23/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_23/lstm_cell_23/Sigmoid£
lstm_23/lstm_cell_23/Sigmoid_1Sigmoid#lstm_23/lstm_cell_23/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
lstm_23/lstm_cell_23/Sigmoid_1¬
lstm_23/lstm_cell_23/mulMul"lstm_23/lstm_cell_23/Sigmoid_1:y:0lstm_23/zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_23/lstm_cell_23/mul
lstm_23/lstm_cell_23/ReluRelu#lstm_23/lstm_cell_23/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_23/lstm_cell_23/Relu½
lstm_23/lstm_cell_23/mul_1Mul lstm_23/lstm_cell_23/Sigmoid:y:0'lstm_23/lstm_cell_23/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_23/lstm_cell_23/mul_1²
lstm_23/lstm_cell_23/add_1AddV2lstm_23/lstm_cell_23/mul:z:0lstm_23/lstm_cell_23/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_23/lstm_cell_23/add_1£
lstm_23/lstm_cell_23/Sigmoid_2Sigmoid#lstm_23/lstm_cell_23/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
lstm_23/lstm_cell_23/Sigmoid_2
lstm_23/lstm_cell_23/Relu_1Relulstm_23/lstm_cell_23/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_23/lstm_cell_23/Relu_1Á
lstm_23/lstm_cell_23/mul_2Mul"lstm_23/lstm_cell_23/Sigmoid_2:y:0)lstm_23/lstm_cell_23/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_23/lstm_cell_23/mul_2
%lstm_23/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2'
%lstm_23/TensorArrayV2_1/element_shapeØ
lstm_23/TensorArrayV2_1TensorListReserve.lstm_23/TensorArrayV2_1/element_shape:output:0 lstm_23/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_23/TensorArrayV2_1^
lstm_23/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_23/time
 lstm_23/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2"
 lstm_23/while/maximum_iterationsz
lstm_23/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_23/while/loop_counter
lstm_23/whileWhile#lstm_23/while/loop_counter:output:0)lstm_23/while/maximum_iterations:output:0lstm_23/time:output:0 lstm_23/TensorArrayV2_1:handle:0lstm_23/zeros:output:0lstm_23/zeros_1:output:0 lstm_23/strided_slice_1:output:0?lstm_23/TensorArrayUnstack/TensorListFromTensor:output_handle:03lstm_23_lstm_cell_23_matmul_readvariableop_resource5lstm_23_lstm_cell_23_matmul_1_readvariableop_resource4lstm_23_lstm_cell_23_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *'
bodyR
lstm_23_while_body_39105356*'
condR
lstm_23_while_cond_39105355*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 2
lstm_23/whileÅ
8lstm_23/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2:
8lstm_23/TensorArrayV2Stack/TensorListStack/element_shape
*lstm_23/TensorArrayV2Stack/TensorListStackTensorListStacklstm_23/while:output:3Alstm_23/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype02,
*lstm_23/TensorArrayV2Stack/TensorListStack
lstm_23/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
lstm_23/strided_slice_3/stack
lstm_23/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2!
lstm_23/strided_slice_3/stack_1
lstm_23/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_23/strided_slice_3/stack_2Ë
lstm_23/strided_slice_3StridedSlice3lstm_23/TensorArrayV2Stack/TensorListStack:tensor:0&lstm_23/strided_slice_3/stack:output:0(lstm_23/strided_slice_3/stack_1:output:0(lstm_23/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
lstm_23/strided_slice_3
lstm_23/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_23/transpose_1/permÆ
lstm_23/transpose_1	Transpose3lstm_23/TensorArrayV2Stack/TensorListStack:tensor:0!lstm_23/transpose_1/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_23/transpose_1v
lstm_23/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_23/runtimey
dropout_23/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout_23/dropout/Constª
dropout_23/dropout/MulMullstm_23/transpose_1:y:0!dropout_23/dropout/Const:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout_23/dropout/Mul{
dropout_23/dropout/ShapeShapelstm_23/transpose_1:y:0*
T0*
_output_shapes
:2
dropout_23/dropout/ShapeÚ
/dropout_23/dropout/random_uniform/RandomUniformRandomUniform!dropout_23/dropout/Shape:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype021
/dropout_23/dropout/random_uniform/RandomUniform
!dropout_23/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>2#
!dropout_23/dropout/GreaterEqual/yï
dropout_23/dropout/GreaterEqualGreaterEqual8dropout_23/dropout/random_uniform/RandomUniform:output:0*dropout_23/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
dropout_23/dropout/GreaterEqual¥
dropout_23/dropout/CastCast#dropout_23/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout_23/dropout/Cast«
dropout_23/dropout/Mul_1Muldropout_23/dropout/Mul:z:0dropout_23/dropout/Cast:y:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout_23/dropout/Mul_1²
!dense_11/Tensordot/ReadVariableOpReadVariableOp*dense_11_tensordot_readvariableop_resource*
_output_shapes
:	*
dtype02#
!dense_11/Tensordot/ReadVariableOp|
dense_11/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_11/Tensordot/axes
dense_11/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_11/Tensordot/free
dense_11/Tensordot/ShapeShapedropout_23/dropout/Mul_1:z:0*
T0*
_output_shapes
:2
dense_11/Tensordot/Shape
 dense_11/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_11/Tensordot/GatherV2/axisþ
dense_11/Tensordot/GatherV2GatherV2!dense_11/Tensordot/Shape:output:0 dense_11/Tensordot/free:output:0)dense_11/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_11/Tensordot/GatherV2
"dense_11/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"dense_11/Tensordot/GatherV2_1/axis
dense_11/Tensordot/GatherV2_1GatherV2!dense_11/Tensordot/Shape:output:0 dense_11/Tensordot/axes:output:0+dense_11/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_11/Tensordot/GatherV2_1~
dense_11/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_11/Tensordot/Const¤
dense_11/Tensordot/ProdProd$dense_11/Tensordot/GatherV2:output:0!dense_11/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_11/Tensordot/Prod
dense_11/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_11/Tensordot/Const_1¬
dense_11/Tensordot/Prod_1Prod&dense_11/Tensordot/GatherV2_1:output:0#dense_11/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_11/Tensordot/Prod_1
dense_11/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2 
dense_11/Tensordot/concat/axisÝ
dense_11/Tensordot/concatConcatV2 dense_11/Tensordot/free:output:0 dense_11/Tensordot/axes:output:0'dense_11/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_11/Tensordot/concat°
dense_11/Tensordot/stackPack dense_11/Tensordot/Prod:output:0"dense_11/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_11/Tensordot/stackÂ
dense_11/Tensordot/transpose	Transposedropout_23/dropout/Mul_1:z:0"dense_11/Tensordot/concat:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_11/Tensordot/transposeÃ
dense_11/Tensordot/ReshapeReshape dense_11/Tensordot/transpose:y:0!dense_11/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
dense_11/Tensordot/ReshapeÂ
dense_11/Tensordot/MatMulMatMul#dense_11/Tensordot/Reshape:output:0)dense_11/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_11/Tensordot/MatMul
dense_11/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
dense_11/Tensordot/Const_2
 dense_11/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_11/Tensordot/concat_1/axisê
dense_11/Tensordot/concat_1ConcatV2$dense_11/Tensordot/GatherV2:output:0#dense_11/Tensordot/Const_2:output:0)dense_11/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_11/Tensordot/concat_1´
dense_11/TensordotReshape#dense_11/Tensordot/MatMul:product:0$dense_11/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_11/Tensordot§
dense_11/BiasAdd/ReadVariableOpReadVariableOp(dense_11_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_11/BiasAdd/ReadVariableOp«
dense_11/BiasAddBiasAdddense_11/Tensordot:output:0'dense_11/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_11/BiasAdd
dense_11/SoftmaxSoftmaxdense_11/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_11/Softmaxy
IdentityIdentitydense_11/Softmax:softmax:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

IdentityÈ
NoOpNoOp ^dense_11/BiasAdd/ReadVariableOp"^dense_11/Tensordot/ReadVariableOp,^lstm_22/lstm_cell_22/BiasAdd/ReadVariableOp+^lstm_22/lstm_cell_22/MatMul/ReadVariableOp-^lstm_22/lstm_cell_22/MatMul_1/ReadVariableOp^lstm_22/while,^lstm_23/lstm_cell_23/BiasAdd/ReadVariableOp+^lstm_23/lstm_cell_23/MatMul/ReadVariableOp-^lstm_23/lstm_cell_23/MatMul_1/ReadVariableOp^lstm_23/while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ]: : : : : : : : 2B
dense_11/BiasAdd/ReadVariableOpdense_11/BiasAdd/ReadVariableOp2F
!dense_11/Tensordot/ReadVariableOp!dense_11/Tensordot/ReadVariableOp2Z
+lstm_22/lstm_cell_22/BiasAdd/ReadVariableOp+lstm_22/lstm_cell_22/BiasAdd/ReadVariableOp2X
*lstm_22/lstm_cell_22/MatMul/ReadVariableOp*lstm_22/lstm_cell_22/MatMul/ReadVariableOp2\
,lstm_22/lstm_cell_22/MatMul_1/ReadVariableOp,lstm_22/lstm_cell_22/MatMul_1/ReadVariableOp2
lstm_22/whilelstm_22/while2Z
+lstm_23/lstm_cell_23/BiasAdd/ReadVariableOp+lstm_23/lstm_cell_23/BiasAdd/ReadVariableOp2X
*lstm_23/lstm_cell_23/MatMul/ReadVariableOp*lstm_23/lstm_cell_23/MatMul/ReadVariableOp2\
,lstm_23/lstm_cell_23/MatMul_1/ReadVariableOp,lstm_23/lstm_cell_23/MatMul_1/ReadVariableOp2
lstm_23/whilelstm_23/while:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
 
_user_specified_nameinputs
×
g
H__inference_dropout_22_layer_call_and_return_conditional_losses_39104428

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *«ªª?2
dropout/Constx
dropout/MulMulinputsdropout/Const:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape¹
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  >2
dropout/GreaterEqual/yÃ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
dropout/Mul_1j
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ¬:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
 
_user_specified_nameinputs
ÐF

E__inference_lstm_23_layer_call_and_return_conditional_losses_39103546

inputs)
lstm_cell_23_39103464:
¬)
lstm_cell_23_39103466:
$
lstm_cell_23_39103468:	
identity¢$lstm_cell_23/StatefulPartitionedCall¢whileD
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
strided_slice/stack_2â
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
B :2
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
B :è2
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
B :2
zeros/packed/1
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
:ÿÿÿÿÿÿÿÿÿ2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :2
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
B :è2
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
B :2
zeros_1/packed/1
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
:ÿÿÿÿÿÿÿÿÿ2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputstranspose/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬2
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
strided_slice_1/stack_2î
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
TensorArrayV2/element_shape²
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2¿
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ¬   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeø
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
strided_slice_2/stack_2ý
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬*
shrink_axis_mask2
strided_slice_2¨
$lstm_cell_23/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_23_39103464lstm_cell_23_39103466lstm_cell_23_39103468*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_lstm_cell_23_layer_call_and_return_conditional_losses_391033992&
$lstm_cell_23/StatefulPartitionedCall
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
TensorArrayV2_1/element_shape¸
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
ÿÿÿÿÿÿÿÿÿ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterÎ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_23_39103464lstm_cell_23_39103466lstm_cell_23_39103468*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_39103477*
condR
while_cond_39103476*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   22
0TensorArrayV2Stack/TensorListStack/element_shapeò
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
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
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm¯
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
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
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity}
NoOpNoOp%^lstm_cell_23/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬: : : 2L
$lstm_cell_23/StatefulPartitionedCall$lstm_cell_23/StatefulPartitionedCall2
whilewhile:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬
 
_user_specified_nameinputs
ã
Í
while_cond_39106260
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_39106260___redundant_placeholder06
2while_while_cond_39106260___redundant_placeholder16
2while_while_cond_39106260___redundant_placeholder26
2while_while_cond_39106260___redundant_placeholder3
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
B: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 
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
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
éJ
Ö

lstm_23_while_body_39105022,
(lstm_23_while_lstm_23_while_loop_counter2
.lstm_23_while_lstm_23_while_maximum_iterations
lstm_23_while_placeholder
lstm_23_while_placeholder_1
lstm_23_while_placeholder_2
lstm_23_while_placeholder_3+
'lstm_23_while_lstm_23_strided_slice_1_0g
clstm_23_while_tensorarrayv2read_tensorlistgetitem_lstm_23_tensorarrayunstack_tensorlistfromtensor_0O
;lstm_23_while_lstm_cell_23_matmul_readvariableop_resource_0:
¬Q
=lstm_23_while_lstm_cell_23_matmul_1_readvariableop_resource_0:
K
<lstm_23_while_lstm_cell_23_biasadd_readvariableop_resource_0:	
lstm_23_while_identity
lstm_23_while_identity_1
lstm_23_while_identity_2
lstm_23_while_identity_3
lstm_23_while_identity_4
lstm_23_while_identity_5)
%lstm_23_while_lstm_23_strided_slice_1e
alstm_23_while_tensorarrayv2read_tensorlistgetitem_lstm_23_tensorarrayunstack_tensorlistfromtensorM
9lstm_23_while_lstm_cell_23_matmul_readvariableop_resource:
¬O
;lstm_23_while_lstm_cell_23_matmul_1_readvariableop_resource:
I
:lstm_23_while_lstm_cell_23_biasadd_readvariableop_resource:	¢1lstm_23/while/lstm_cell_23/BiasAdd/ReadVariableOp¢0lstm_23/while/lstm_cell_23/MatMul/ReadVariableOp¢2lstm_23/while/lstm_cell_23/MatMul_1/ReadVariableOpÓ
?lstm_23/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ¬   2A
?lstm_23/while/TensorArrayV2Read/TensorListGetItem/element_shape
1lstm_23/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemclstm_23_while_tensorarrayv2read_tensorlistgetitem_lstm_23_tensorarrayunstack_tensorlistfromtensor_0lstm_23_while_placeholderHlstm_23/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬*
element_dtype023
1lstm_23/while/TensorArrayV2Read/TensorListGetItemâ
0lstm_23/while/lstm_cell_23/MatMul/ReadVariableOpReadVariableOp;lstm_23_while_lstm_cell_23_matmul_readvariableop_resource_0* 
_output_shapes
:
¬*
dtype022
0lstm_23/while/lstm_cell_23/MatMul/ReadVariableOp÷
!lstm_23/while/lstm_cell_23/MatMulMatMul8lstm_23/while/TensorArrayV2Read/TensorListGetItem:item:08lstm_23/while/lstm_cell_23/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!lstm_23/while/lstm_cell_23/MatMulè
2lstm_23/while/lstm_cell_23/MatMul_1/ReadVariableOpReadVariableOp=lstm_23_while_lstm_cell_23_matmul_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype024
2lstm_23/while/lstm_cell_23/MatMul_1/ReadVariableOpà
#lstm_23/while/lstm_cell_23/MatMul_1MatMullstm_23_while_placeholder_2:lstm_23/while/lstm_cell_23/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2%
#lstm_23/while/lstm_cell_23/MatMul_1Ø
lstm_23/while/lstm_cell_23/addAddV2+lstm_23/while/lstm_cell_23/MatMul:product:0-lstm_23/while/lstm_cell_23/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
lstm_23/while/lstm_cell_23/addà
1lstm_23/while/lstm_cell_23/BiasAdd/ReadVariableOpReadVariableOp<lstm_23_while_lstm_cell_23_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype023
1lstm_23/while/lstm_cell_23/BiasAdd/ReadVariableOpå
"lstm_23/while/lstm_cell_23/BiasAddBiasAdd"lstm_23/while/lstm_cell_23/add:z:09lstm_23/while/lstm_cell_23/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2$
"lstm_23/while/lstm_cell_23/BiasAdd
*lstm_23/while/lstm_cell_23/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2,
*lstm_23/while/lstm_cell_23/split/split_dim¯
 lstm_23/while/lstm_cell_23/splitSplit3lstm_23/while/lstm_cell_23/split/split_dim:output:0+lstm_23/while/lstm_cell_23/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2"
 lstm_23/while/lstm_cell_23/split±
"lstm_23/while/lstm_cell_23/SigmoidSigmoid)lstm_23/while/lstm_cell_23/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2$
"lstm_23/while/lstm_cell_23/Sigmoidµ
$lstm_23/while/lstm_cell_23/Sigmoid_1Sigmoid)lstm_23/while/lstm_cell_23/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$lstm_23/while/lstm_cell_23/Sigmoid_1Á
lstm_23/while/lstm_cell_23/mulMul(lstm_23/while/lstm_cell_23/Sigmoid_1:y:0lstm_23_while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
lstm_23/while/lstm_cell_23/mul¨
lstm_23/while/lstm_cell_23/ReluRelu)lstm_23/while/lstm_cell_23/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
lstm_23/while/lstm_cell_23/ReluÕ
 lstm_23/while/lstm_cell_23/mul_1Mul&lstm_23/while/lstm_cell_23/Sigmoid:y:0-lstm_23/while/lstm_cell_23/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 lstm_23/while/lstm_cell_23/mul_1Ê
 lstm_23/while/lstm_cell_23/add_1AddV2"lstm_23/while/lstm_cell_23/mul:z:0$lstm_23/while/lstm_cell_23/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 lstm_23/while/lstm_cell_23/add_1µ
$lstm_23/while/lstm_cell_23/Sigmoid_2Sigmoid)lstm_23/while/lstm_cell_23/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$lstm_23/while/lstm_cell_23/Sigmoid_2§
!lstm_23/while/lstm_cell_23/Relu_1Relu$lstm_23/while/lstm_cell_23/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!lstm_23/while/lstm_cell_23/Relu_1Ù
 lstm_23/while/lstm_cell_23/mul_2Mul(lstm_23/while/lstm_cell_23/Sigmoid_2:y:0/lstm_23/while/lstm_cell_23/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 lstm_23/while/lstm_cell_23/mul_2
2lstm_23/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_23_while_placeholder_1lstm_23_while_placeholder$lstm_23/while/lstm_cell_23/mul_2:z:0*
_output_shapes
: *
element_dtype024
2lstm_23/while/TensorArrayV2Write/TensorListSetIteml
lstm_23/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_23/while/add/y
lstm_23/while/addAddV2lstm_23_while_placeholderlstm_23/while/add/y:output:0*
T0*
_output_shapes
: 2
lstm_23/while/addp
lstm_23/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_23/while/add_1/y
lstm_23/while/add_1AddV2(lstm_23_while_lstm_23_while_loop_counterlstm_23/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lstm_23/while/add_1
lstm_23/while/IdentityIdentitylstm_23/while/add_1:z:0^lstm_23/while/NoOp*
T0*
_output_shapes
: 2
lstm_23/while/Identity¦
lstm_23/while/Identity_1Identity.lstm_23_while_lstm_23_while_maximum_iterations^lstm_23/while/NoOp*
T0*
_output_shapes
: 2
lstm_23/while/Identity_1
lstm_23/while/Identity_2Identitylstm_23/while/add:z:0^lstm_23/while/NoOp*
T0*
_output_shapes
: 2
lstm_23/while/Identity_2º
lstm_23/while/Identity_3IdentityBlstm_23/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_23/while/NoOp*
T0*
_output_shapes
: 2
lstm_23/while/Identity_3®
lstm_23/while/Identity_4Identity$lstm_23/while/lstm_cell_23/mul_2:z:0^lstm_23/while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_23/while/Identity_4®
lstm_23/while/Identity_5Identity$lstm_23/while/lstm_cell_23/add_1:z:0^lstm_23/while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_23/while/Identity_5
lstm_23/while/NoOpNoOp2^lstm_23/while/lstm_cell_23/BiasAdd/ReadVariableOp1^lstm_23/while/lstm_cell_23/MatMul/ReadVariableOp3^lstm_23/while/lstm_cell_23/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
lstm_23/while/NoOp"9
lstm_23_while_identitylstm_23/while/Identity:output:0"=
lstm_23_while_identity_1!lstm_23/while/Identity_1:output:0"=
lstm_23_while_identity_2!lstm_23/while/Identity_2:output:0"=
lstm_23_while_identity_3!lstm_23/while/Identity_3:output:0"=
lstm_23_while_identity_4!lstm_23/while/Identity_4:output:0"=
lstm_23_while_identity_5!lstm_23/while/Identity_5:output:0"P
%lstm_23_while_lstm_23_strided_slice_1'lstm_23_while_lstm_23_strided_slice_1_0"z
:lstm_23_while_lstm_cell_23_biasadd_readvariableop_resource<lstm_23_while_lstm_cell_23_biasadd_readvariableop_resource_0"|
;lstm_23_while_lstm_cell_23_matmul_1_readvariableop_resource=lstm_23_while_lstm_cell_23_matmul_1_readvariableop_resource_0"x
9lstm_23_while_lstm_cell_23_matmul_readvariableop_resource;lstm_23_while_lstm_cell_23_matmul_readvariableop_resource_0"È
alstm_23_while_tensorarrayv2read_tensorlistgetitem_lstm_23_tensorarrayunstack_tensorlistfromtensorclstm_23_while_tensorarrayv2read_tensorlistgetitem_lstm_23_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2f
1lstm_23/while/lstm_cell_23/BiasAdd/ReadVariableOp1lstm_23/while/lstm_cell_23/BiasAdd/ReadVariableOp2d
0lstm_23/while/lstm_cell_23/MatMul/ReadVariableOp0lstm_23/while/lstm_cell_23/MatMul/ReadVariableOp2h
2lstm_23/while/lstm_cell_23/MatMul_1/ReadVariableOp2lstm_23/while/lstm_cell_23/MatMul_1/ReadVariableOp: 
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
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
°?
Ô
while_body_39105586
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_22_matmul_readvariableop_resource_0:	]°I
5while_lstm_cell_22_matmul_1_readvariableop_resource_0:
¬°C
4while_lstm_cell_22_biasadd_readvariableop_resource_0:	°
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_22_matmul_readvariableop_resource:	]°G
3while_lstm_cell_22_matmul_1_readvariableop_resource:
¬°A
2while_lstm_cell_22_biasadd_readvariableop_resource:	°¢)while/lstm_cell_22/BiasAdd/ReadVariableOp¢(while/lstm_cell_22/MatMul/ReadVariableOp¢*while/lstm_cell_22/MatMul_1/ReadVariableOpÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ]   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÓ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemÉ
(while/lstm_cell_22/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_22_matmul_readvariableop_resource_0*
_output_shapes
:	]°*
dtype02*
(while/lstm_cell_22/MatMul/ReadVariableOp×
while/lstm_cell_22/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_22/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°2
while/lstm_cell_22/MatMulÐ
*while/lstm_cell_22/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_22_matmul_1_readvariableop_resource_0* 
_output_shapes
:
¬°*
dtype02,
*while/lstm_cell_22/MatMul_1/ReadVariableOpÀ
while/lstm_cell_22/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_22/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°2
while/lstm_cell_22/MatMul_1¸
while/lstm_cell_22/addAddV2#while/lstm_cell_22/MatMul:product:0%while/lstm_cell_22/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°2
while/lstm_cell_22/addÈ
)while/lstm_cell_22/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_22_biasadd_readvariableop_resource_0*
_output_shapes	
:°*
dtype02+
)while/lstm_cell_22/BiasAdd/ReadVariableOpÅ
while/lstm_cell_22/BiasAddBiasAddwhile/lstm_cell_22/add:z:01while/lstm_cell_22/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°2
while/lstm_cell_22/BiasAdd
"while/lstm_cell_22/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_22/split/split_dim
while/lstm_cell_22/splitSplit+while/lstm_cell_22/split/split_dim:output:0#while/lstm_cell_22/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ¬*
	num_split2
while/lstm_cell_22/split
while/lstm_cell_22/SigmoidSigmoid!while/lstm_cell_22/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
while/lstm_cell_22/Sigmoid
while/lstm_cell_22/Sigmoid_1Sigmoid!while/lstm_cell_22/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
while/lstm_cell_22/Sigmoid_1¡
while/lstm_cell_22/mulMul while/lstm_cell_22/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
while/lstm_cell_22/mul
while/lstm_cell_22/ReluRelu!while/lstm_cell_22/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
while/lstm_cell_22/Reluµ
while/lstm_cell_22/mul_1Mulwhile/lstm_cell_22/Sigmoid:y:0%while/lstm_cell_22/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
while/lstm_cell_22/mul_1ª
while/lstm_cell_22/add_1AddV2while/lstm_cell_22/mul:z:0while/lstm_cell_22/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
while/lstm_cell_22/add_1
while/lstm_cell_22/Sigmoid_2Sigmoid!while/lstm_cell_22/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
while/lstm_cell_22/Sigmoid_2
while/lstm_cell_22/Relu_1Reluwhile/lstm_cell_22/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
while/lstm_cell_22/Relu_1¹
while/lstm_cell_22/mul_2Mul while/lstm_cell_22/Sigmoid_2:y:0'while/lstm_cell_22/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
while/lstm_cell_22/mul_2à
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_22/mul_2:z:0*
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
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_22/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_22/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬2
while/Identity_5Þ

while/NoOpNoOp*^while/lstm_cell_22/BiasAdd/ReadVariableOp)^while/lstm_cell_22/MatMul/ReadVariableOp+^while/lstm_cell_22/MatMul_1/ReadVariableOp*"
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
2while_lstm_cell_22_biasadd_readvariableop_resource4while_lstm_cell_22_biasadd_readvariableop_resource_0"l
3while_lstm_cell_22_matmul_1_readvariableop_resource5while_lstm_cell_22_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_22_matmul_readvariableop_resource3while_lstm_cell_22_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ¬:ÿÿÿÿÿÿÿÿÿ¬: : : : : 2V
)while/lstm_cell_22/BiasAdd/ReadVariableOp)while/lstm_cell_22/BiasAdd/ReadVariableOp2T
(while/lstm_cell_22/MatMul/ReadVariableOp(while/lstm_cell_22/MatMul/ReadVariableOp2X
*while/lstm_cell_22/MatMul_1/ReadVariableOp*while/lstm_cell_22/MatMul_1/ReadVariableOp: 
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
:ÿÿÿÿÿÿÿÿÿ¬:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬:

_output_shapes
: :

_output_shapes
: 
ã
Í
while_cond_39103476
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_39103476___redundant_placeholder06
2while_while_cond_39103476___redundant_placeholder16
2while_while_cond_39103476___redundant_placeholder26
2while_while_cond_39103476___redundant_placeholder3
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
B: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 
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
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:"¨L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*¿
serving_default«
K
lstm_22_input:
serving_default_lstm_22_input:0ÿÿÿÿÿÿÿÿÿ]@
dense_114
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:³
õ
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
k__call__
*l&call_and_return_all_conditional_losses
m_default_save_signature"
_tf_keras_sequential
Ã
cell

state_spec
trainable_variables
regularization_losses
	variables
	keras_api
n__call__
*o&call_and_return_all_conditional_losses"
_tf_keras_rnn_layer
¥
trainable_variables
regularization_losses
	variables
	keras_api
p__call__
*q&call_and_return_all_conditional_losses"
_tf_keras_layer
Ã
cell

state_spec
trainable_variables
regularization_losses
	variables
	keras_api
r__call__
*s&call_and_return_all_conditional_losses"
_tf_keras_rnn_layer
¥
trainable_variables
regularization_losses
	variables
	keras_api
t__call__
*u&call_and_return_all_conditional_losses"
_tf_keras_layer
»

 kernel
!bias
"trainable_variables
#regularization_losses
$	variables
%	keras_api
v__call__
*w&call_and_return_all_conditional_losses"
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
Ê
,non_trainable_variables
-metrics

.layers
/layer_metrics
trainable_variables
regularization_losses
		variables
0layer_regularization_losses
k__call__
m_default_save_signature
*l&call_and_return_all_conditional_losses
&l"call_and_return_conditional_losses"
_generic_user_object
,
xserving_default"
signature_map
á
1
state_size

&kernel
'recurrent_kernel
(bias
2trainable_variables
3regularization_losses
4	variables
5	keras_api
y__call__
*z&call_and_return_all_conditional_losses"
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
¹
6non_trainable_variables
7metrics

8states

9layers
:layer_metrics
trainable_variables
regularization_losses
	variables
;layer_regularization_losses
n__call__
*o&call_and_return_all_conditional_losses
&o"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
<non_trainable_variables
=metrics

>layers
?layer_metrics
trainable_variables
regularization_losses
	variables
@layer_regularization_losses
p__call__
*q&call_and_return_all_conditional_losses
&q"call_and_return_conditional_losses"
_generic_user_object
á
A
state_size

)kernel
*recurrent_kernel
+bias
Btrainable_variables
Cregularization_losses
D	variables
E	keras_api
{__call__
*|&call_and_return_all_conditional_losses"
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
¹
Fnon_trainable_variables
Gmetrics

Hstates

Ilayers
Jlayer_metrics
trainable_variables
regularization_losses
	variables
Klayer_regularization_losses
r__call__
*s&call_and_return_all_conditional_losses
&s"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
Lnon_trainable_variables
Mmetrics

Nlayers
Olayer_metrics
trainable_variables
regularization_losses
	variables
Player_regularization_losses
t__call__
*u&call_and_return_all_conditional_losses
&u"call_and_return_conditional_losses"
_generic_user_object
": 	2dense_11/kernel
:2dense_11/bias
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
­
Qnon_trainable_variables
Rmetrics

Slayers
Tlayer_metrics
"trainable_variables
#regularization_losses
$	variables
Ulayer_regularization_losses
v__call__
*w&call_and_return_all_conditional_losses
&w"call_and_return_conditional_losses"
_generic_user_object
.:,	]°2lstm_22/lstm_cell_22/kernel
9:7
¬°2%lstm_22/lstm_cell_22/recurrent_kernel
(:&°2lstm_22/lstm_cell_22/bias
/:-
¬2lstm_23/lstm_cell_23/kernel
9:7
2%lstm_23/lstm_cell_23/recurrent_kernel
(:&2lstm_23/lstm_cell_23/bias
 "
trackable_list_wrapper
.
V0
W1"
trackable_list_wrapper
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
­
Xnon_trainable_variables
Ymetrics

Zlayers
[layer_metrics
2trainable_variables
3regularization_losses
4	variables
\layer_regularization_losses
y__call__
*z&call_and_return_all_conditional_losses
&z"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
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
­
]non_trainable_variables
^metrics

_layers
`layer_metrics
Btrainable_variables
Cregularization_losses
D	variables
alayer_regularization_losses
{__call__
*|&call_and_return_all_conditional_losses
&|"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
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
2
0__inference_sequential_11_layer_call_fn_39104202
0__inference_sequential_11_layer_call_fn_39104786
0__inference_sequential_11_layer_call_fn_39104807
0__inference_sequential_11_layer_call_fn_39104692À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ú2÷
K__inference_sequential_11_layer_call_and_return_conditional_losses_39105134
K__inference_sequential_11_layer_call_and_return_conditional_losses_39105475
K__inference_sequential_11_layer_call_and_return_conditional_losses_39104717
K__inference_sequential_11_layer_call_and_return_conditional_losses_39104742À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ÔBÑ
#__inference__wrapped_model_39102548lstm_22_input"
²
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
2
*__inference_lstm_22_layer_call_fn_39105486
*__inference_lstm_22_layer_call_fn_39105497
*__inference_lstm_22_layer_call_fn_39105508
*__inference_lstm_22_layer_call_fn_39105519Õ
Ì²È
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
÷2ô
E__inference_lstm_22_layer_call_and_return_conditional_losses_39105670
E__inference_lstm_22_layer_call_and_return_conditional_losses_39105821
E__inference_lstm_22_layer_call_and_return_conditional_losses_39105972
E__inference_lstm_22_layer_call_and_return_conditional_losses_39106123Õ
Ì²È
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
2
-__inference_dropout_22_layer_call_fn_39106128
-__inference_dropout_22_layer_call_fn_39106133´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Î2Ë
H__inference_dropout_22_layer_call_and_return_conditional_losses_39106138
H__inference_dropout_22_layer_call_and_return_conditional_losses_39106150´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
2
*__inference_lstm_23_layer_call_fn_39106161
*__inference_lstm_23_layer_call_fn_39106172
*__inference_lstm_23_layer_call_fn_39106183
*__inference_lstm_23_layer_call_fn_39106194Õ
Ì²È
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
÷2ô
E__inference_lstm_23_layer_call_and_return_conditional_losses_39106345
E__inference_lstm_23_layer_call_and_return_conditional_losses_39106496
E__inference_lstm_23_layer_call_and_return_conditional_losses_39106647
E__inference_lstm_23_layer_call_and_return_conditional_losses_39106798Õ
Ì²È
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
2
-__inference_dropout_23_layer_call_fn_39106803
-__inference_dropout_23_layer_call_fn_39106808´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Î2Ë
H__inference_dropout_23_layer_call_and_return_conditional_losses_39106813
H__inference_dropout_23_layer_call_and_return_conditional_losses_39106825´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Õ2Ò
+__inference_dense_11_layer_call_fn_39106834¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ð2í
F__inference_dense_11_layer_call_and_return_conditional_losses_39106865¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ÓBÐ
&__inference_signature_wrapper_39104765lstm_22_input"
²
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¦2£
/__inference_lstm_cell_22_layer_call_fn_39106882
/__inference_lstm_cell_22_layer_call_fn_39106899¾
µ²±
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ü2Ù
J__inference_lstm_cell_22_layer_call_and_return_conditional_losses_39106931
J__inference_lstm_cell_22_layer_call_and_return_conditional_losses_39106963¾
µ²±
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
¦2£
/__inference_lstm_cell_23_layer_call_fn_39106980
/__inference_lstm_cell_23_layer_call_fn_39106997¾
µ²±
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ü2Ù
J__inference_lstm_cell_23_layer_call_and_return_conditional_losses_39107029
J__inference_lstm_cell_23_layer_call_and_return_conditional_losses_39107061¾
µ²±
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 ¦
#__inference__wrapped_model_39102548&'()*+ !:¢7
0¢-
+(
lstm_22_inputÿÿÿÿÿÿÿÿÿ]
ª "7ª4
2
dense_11&#
dense_11ÿÿÿÿÿÿÿÿÿ¯
F__inference_dense_11_layer_call_and_return_conditional_losses_39106865e !4¢1
*¢'
%"
inputsÿÿÿÿÿÿÿÿÿ
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ
 
+__inference_dense_11_layer_call_fn_39106834X !4¢1
*¢'
%"
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ²
H__inference_dropout_22_layer_call_and_return_conditional_losses_39106138f8¢5
.¢+
%"
inputsÿÿÿÿÿÿÿÿÿ¬
p 
ª "*¢'
 
0ÿÿÿÿÿÿÿÿÿ¬
 ²
H__inference_dropout_22_layer_call_and_return_conditional_losses_39106150f8¢5
.¢+
%"
inputsÿÿÿÿÿÿÿÿÿ¬
p
ª "*¢'
 
0ÿÿÿÿÿÿÿÿÿ¬
 
-__inference_dropout_22_layer_call_fn_39106128Y8¢5
.¢+
%"
inputsÿÿÿÿÿÿÿÿÿ¬
p 
ª "ÿÿÿÿÿÿÿÿÿ¬
-__inference_dropout_22_layer_call_fn_39106133Y8¢5
.¢+
%"
inputsÿÿÿÿÿÿÿÿÿ¬
p
ª "ÿÿÿÿÿÿÿÿÿ¬²
H__inference_dropout_23_layer_call_and_return_conditional_losses_39106813f8¢5
.¢+
%"
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "*¢'
 
0ÿÿÿÿÿÿÿÿÿ
 ²
H__inference_dropout_23_layer_call_and_return_conditional_losses_39106825f8¢5
.¢+
%"
inputsÿÿÿÿÿÿÿÿÿ
p
ª "*¢'
 
0ÿÿÿÿÿÿÿÿÿ
 
-__inference_dropout_23_layer_call_fn_39106803Y8¢5
.¢+
%"
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "ÿÿÿÿÿÿÿÿÿ
-__inference_dropout_23_layer_call_fn_39106808Y8¢5
.¢+
%"
inputsÿÿÿÿÿÿÿÿÿ
p
ª "ÿÿÿÿÿÿÿÿÿÕ
E__inference_lstm_22_layer_call_and_return_conditional_losses_39105670&'(O¢L
E¢B
41
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ]

 
p 

 
ª "3¢0
)&
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬
 Õ
E__inference_lstm_22_layer_call_and_return_conditional_losses_39105821&'(O¢L
E¢B
41
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ]

 
p

 
ª "3¢0
)&
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬
 »
E__inference_lstm_22_layer_call_and_return_conditional_losses_39105972r&'(?¢<
5¢2
$!
inputsÿÿÿÿÿÿÿÿÿ]

 
p 

 
ª "*¢'
 
0ÿÿÿÿÿÿÿÿÿ¬
 »
E__inference_lstm_22_layer_call_and_return_conditional_losses_39106123r&'(?¢<
5¢2
$!
inputsÿÿÿÿÿÿÿÿÿ]

 
p

 
ª "*¢'
 
0ÿÿÿÿÿÿÿÿÿ¬
 ¬
*__inference_lstm_22_layer_call_fn_39105486~&'(O¢L
E¢B
41
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ]

 
p 

 
ª "&#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬¬
*__inference_lstm_22_layer_call_fn_39105497~&'(O¢L
E¢B
41
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ]

 
p

 
ª "&#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬
*__inference_lstm_22_layer_call_fn_39105508e&'(?¢<
5¢2
$!
inputsÿÿÿÿÿÿÿÿÿ]

 
p 

 
ª "ÿÿÿÿÿÿÿÿÿ¬
*__inference_lstm_22_layer_call_fn_39105519e&'(?¢<
5¢2
$!
inputsÿÿÿÿÿÿÿÿÿ]

 
p

 
ª "ÿÿÿÿÿÿÿÿÿ¬Ö
E__inference_lstm_23_layer_call_and_return_conditional_losses_39106345)*+P¢M
F¢C
52
0-
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬

 
p 

 
ª "3¢0
)&
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Ö
E__inference_lstm_23_layer_call_and_return_conditional_losses_39106496)*+P¢M
F¢C
52
0-
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬

 
p

 
ª "3¢0
)&
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ¼
E__inference_lstm_23_layer_call_and_return_conditional_losses_39106647s)*+@¢=
6¢3
%"
inputsÿÿÿÿÿÿÿÿÿ¬

 
p 

 
ª "*¢'
 
0ÿÿÿÿÿÿÿÿÿ
 ¼
E__inference_lstm_23_layer_call_and_return_conditional_losses_39106798s)*+@¢=
6¢3
%"
inputsÿÿÿÿÿÿÿÿÿ¬

 
p

 
ª "*¢'
 
0ÿÿÿÿÿÿÿÿÿ
 ­
*__inference_lstm_23_layer_call_fn_39106161)*+P¢M
F¢C
52
0-
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬

 
p 

 
ª "&#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ­
*__inference_lstm_23_layer_call_fn_39106172)*+P¢M
F¢C
52
0-
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬

 
p

 
ª "&#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
*__inference_lstm_23_layer_call_fn_39106183f)*+@¢=
6¢3
%"
inputsÿÿÿÿÿÿÿÿÿ¬

 
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
*__inference_lstm_23_layer_call_fn_39106194f)*+@¢=
6¢3
%"
inputsÿÿÿÿÿÿÿÿÿ¬

 
p

 
ª "ÿÿÿÿÿÿÿÿÿÑ
J__inference_lstm_cell_22_layer_call_and_return_conditional_losses_39106931&'(¢
x¢u
 
inputsÿÿÿÿÿÿÿÿÿ]
M¢J
# 
states/0ÿÿÿÿÿÿÿÿÿ¬
# 
states/1ÿÿÿÿÿÿÿÿÿ¬
p 
ª "v¢s
l¢i

0/0ÿÿÿÿÿÿÿÿÿ¬
GD
 
0/1/0ÿÿÿÿÿÿÿÿÿ¬
 
0/1/1ÿÿÿÿÿÿÿÿÿ¬
 Ñ
J__inference_lstm_cell_22_layer_call_and_return_conditional_losses_39106963&'(¢
x¢u
 
inputsÿÿÿÿÿÿÿÿÿ]
M¢J
# 
states/0ÿÿÿÿÿÿÿÿÿ¬
# 
states/1ÿÿÿÿÿÿÿÿÿ¬
p
ª "v¢s
l¢i

0/0ÿÿÿÿÿÿÿÿÿ¬
GD
 
0/1/0ÿÿÿÿÿÿÿÿÿ¬
 
0/1/1ÿÿÿÿÿÿÿÿÿ¬
 ¦
/__inference_lstm_cell_22_layer_call_fn_39106882ò&'(¢
x¢u
 
inputsÿÿÿÿÿÿÿÿÿ]
M¢J
# 
states/0ÿÿÿÿÿÿÿÿÿ¬
# 
states/1ÿÿÿÿÿÿÿÿÿ¬
p 
ª "f¢c

0ÿÿÿÿÿÿÿÿÿ¬
C@

1/0ÿÿÿÿÿÿÿÿÿ¬

1/1ÿÿÿÿÿÿÿÿÿ¬¦
/__inference_lstm_cell_22_layer_call_fn_39106899ò&'(¢
x¢u
 
inputsÿÿÿÿÿÿÿÿÿ]
M¢J
# 
states/0ÿÿÿÿÿÿÿÿÿ¬
# 
states/1ÿÿÿÿÿÿÿÿÿ¬
p
ª "f¢c

0ÿÿÿÿÿÿÿÿÿ¬
C@

1/0ÿÿÿÿÿÿÿÿÿ¬

1/1ÿÿÿÿÿÿÿÿÿ¬Ó
J__inference_lstm_cell_23_layer_call_and_return_conditional_losses_39107029)*+¢
y¢v
!
inputsÿÿÿÿÿÿÿÿÿ¬
M¢J
# 
states/0ÿÿÿÿÿÿÿÿÿ
# 
states/1ÿÿÿÿÿÿÿÿÿ
p 
ª "v¢s
l¢i

0/0ÿÿÿÿÿÿÿÿÿ
GD
 
0/1/0ÿÿÿÿÿÿÿÿÿ
 
0/1/1ÿÿÿÿÿÿÿÿÿ
 Ó
J__inference_lstm_cell_23_layer_call_and_return_conditional_losses_39107061)*+¢
y¢v
!
inputsÿÿÿÿÿÿÿÿÿ¬
M¢J
# 
states/0ÿÿÿÿÿÿÿÿÿ
# 
states/1ÿÿÿÿÿÿÿÿÿ
p
ª "v¢s
l¢i

0/0ÿÿÿÿÿÿÿÿÿ
GD
 
0/1/0ÿÿÿÿÿÿÿÿÿ
 
0/1/1ÿÿÿÿÿÿÿÿÿ
 ¨
/__inference_lstm_cell_23_layer_call_fn_39106980ô)*+¢
y¢v
!
inputsÿÿÿÿÿÿÿÿÿ¬
M¢J
# 
states/0ÿÿÿÿÿÿÿÿÿ
# 
states/1ÿÿÿÿÿÿÿÿÿ
p 
ª "f¢c

0ÿÿÿÿÿÿÿÿÿ
C@

1/0ÿÿÿÿÿÿÿÿÿ

1/1ÿÿÿÿÿÿÿÿÿ¨
/__inference_lstm_cell_23_layer_call_fn_39106997ô)*+¢
y¢v
!
inputsÿÿÿÿÿÿÿÿÿ¬
M¢J
# 
states/0ÿÿÿÿÿÿÿÿÿ
# 
states/1ÿÿÿÿÿÿÿÿÿ
p
ª "f¢c

0ÿÿÿÿÿÿÿÿÿ
C@

1/0ÿÿÿÿÿÿÿÿÿ

1/1ÿÿÿÿÿÿÿÿÿÈ
K__inference_sequential_11_layer_call_and_return_conditional_losses_39104717y&'()*+ !B¢?
8¢5
+(
lstm_22_inputÿÿÿÿÿÿÿÿÿ]
p 

 
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ
 È
K__inference_sequential_11_layer_call_and_return_conditional_losses_39104742y&'()*+ !B¢?
8¢5
+(
lstm_22_inputÿÿÿÿÿÿÿÿÿ]
p

 
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ
 Á
K__inference_sequential_11_layer_call_and_return_conditional_losses_39105134r&'()*+ !;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿ]
p 

 
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ
 Á
K__inference_sequential_11_layer_call_and_return_conditional_losses_39105475r&'()*+ !;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿ]
p

 
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ
  
0__inference_sequential_11_layer_call_fn_39104202l&'()*+ !B¢?
8¢5
+(
lstm_22_inputÿÿÿÿÿÿÿÿÿ]
p 

 
ª "ÿÿÿÿÿÿÿÿÿ 
0__inference_sequential_11_layer_call_fn_39104692l&'()*+ !B¢?
8¢5
+(
lstm_22_inputÿÿÿÿÿÿÿÿÿ]
p

 
ª "ÿÿÿÿÿÿÿÿÿ
0__inference_sequential_11_layer_call_fn_39104786e&'()*+ !;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿ]
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
0__inference_sequential_11_layer_call_fn_39104807e&'()*+ !;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿ]
p

 
ª "ÿÿÿÿÿÿÿÿÿ»
&__inference_signature_wrapper_39104765&'()*+ !K¢H
¢ 
Aª>
<
lstm_22_input+(
lstm_22_inputÿÿÿÿÿÿÿÿÿ]"7ª4
2
dense_11&#
dense_11ÿÿÿÿÿÿÿÿÿ