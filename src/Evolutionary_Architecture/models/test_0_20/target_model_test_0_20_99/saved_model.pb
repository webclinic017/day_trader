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
dense_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	в*
shared_namedense_5/kernel
r
"dense_5/kernel/Read/ReadVariableOpReadVariableOpdense_5/kernel*
_output_shapes
:	в*
dtype0
p
dense_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_5/bias
i
 dense_5/bias/Read/ReadVariableOpReadVariableOpdense_5/bias*
_output_shapes
:*
dtype0
У
lstm_10/lstm_cell_10/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	]Ь*,
shared_namelstm_10/lstm_cell_10/kernel
М
/lstm_10/lstm_cell_10/kernel/Read/ReadVariableOpReadVariableOplstm_10/lstm_cell_10/kernel*
_output_shapes
:	]Ь*
dtype0
з
%lstm_10/lstm_cell_10/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	GЬ*6
shared_name'%lstm_10/lstm_cell_10/recurrent_kernel
а
9lstm_10/lstm_cell_10/recurrent_kernel/Read/ReadVariableOpReadVariableOp%lstm_10/lstm_cell_10/recurrent_kernel*
_output_shapes
:	GЬ*
dtype0
Л
lstm_10/lstm_cell_10/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:Ь**
shared_namelstm_10/lstm_cell_10/bias
Д
-lstm_10/lstm_cell_10/bias/Read/ReadVariableOpReadVariableOplstm_10/lstm_cell_10/bias*
_output_shapes	
:Ь*
dtype0
У
lstm_11/lstm_cell_11/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	GИ*,
shared_namelstm_11/lstm_cell_11/kernel
М
/lstm_11/lstm_cell_11/kernel/Read/ReadVariableOpReadVariableOplstm_11/lstm_cell_11/kernel*
_output_shapes
:	GИ*
dtype0
и
%lstm_11/lstm_cell_11/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
вИ*6
shared_name'%lstm_11/lstm_cell_11/recurrent_kernel
б
9lstm_11/lstm_cell_11/recurrent_kernel/Read/ReadVariableOpReadVariableOp%lstm_11/lstm_cell_11/recurrent_kernel* 
_output_shapes
:
вИ*
dtype0
Л
lstm_11/lstm_cell_11/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:И**
shared_namelstm_11/lstm_cell_11/bias
Д
-lstm_11/lstm_cell_11/bias/Read/ReadVariableOpReadVariableOplstm_11/lstm_cell_11/bias*
_output_shapes	
:И*
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
,metrics
trainable_variables
regularization_losses
-layer_regularization_losses
.non_trainable_variables

/layers
		variables
0layer_metrics
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
6metrics
trainable_variables

7states
8layer_regularization_losses
9non_trainable_variables
regularization_losses

:layers
	variables
;layer_metrics
 
 
 
н
<metrics
trainable_variables
=layer_regularization_losses
>non_trainable_variables
regularization_losses

?layers
	variables
@layer_metrics
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
Fmetrics
trainable_variables

Gstates
Hlayer_regularization_losses
Inon_trainable_variables
regularization_losses

Jlayers
	variables
Klayer_metrics
 
 
 
н
Lmetrics
trainable_variables
Mlayer_regularization_losses
Nnon_trainable_variables
regularization_losses

Olayers
	variables
Player_metrics
ZX
VARIABLE_VALUEdense_5/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_5/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

 0
!1
 

 0
!1
н
Qmetrics
"trainable_variables
Rlayer_regularization_losses
Snon_trainable_variables
#regularization_losses

Tlayers
$	variables
Ulayer_metrics
a_
VARIABLE_VALUElstm_10/lstm_cell_10/kernel0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUE
ki
VARIABLE_VALUE%lstm_10/lstm_cell_10/recurrent_kernel0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUElstm_10/lstm_cell_10/bias0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUElstm_11/lstm_cell_11/kernel0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUE
ki
VARIABLE_VALUE%lstm_11/lstm_cell_11/recurrent_kernel0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUElstm_11/lstm_cell_11/bias0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUE

V0
W1
 
 
#
0
1
2
3
4
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
Xmetrics
2trainable_variables
Ylayer_regularization_losses
Znon_trainable_variables
3regularization_losses

[layers
4	variables
\layer_metrics
 
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
]metrics
Btrainable_variables
^layer_regularization_losses
_non_trainable_variables
Cregularization_losses

`layers
D	variables
alayer_metrics
 
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
serving_default_lstm_10_inputPlaceholder*+
_output_shapes
:         ]*
dtype0* 
shape:         ]
к
StatefulPartitionedCallStatefulPartitionedCallserving_default_lstm_10_inputlstm_10/lstm_cell_10/kernel%lstm_10/lstm_cell_10/recurrent_kernellstm_10/lstm_cell_10/biaslstm_11/lstm_cell_11/kernel%lstm_11/lstm_cell_11/recurrent_kernellstm_11/lstm_cell_11/biasdense_5/kerneldense_5/bias*
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
&__inference_signature_wrapper_19547371
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
Х
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename"dense_5/kernel/Read/ReadVariableOp dense_5/bias/Read/ReadVariableOp/lstm_10/lstm_cell_10/kernel/Read/ReadVariableOp9lstm_10/lstm_cell_10/recurrent_kernel/Read/ReadVariableOp-lstm_10/lstm_cell_10/bias/Read/ReadVariableOp/lstm_11/lstm_cell_11/kernel/Read/ReadVariableOp9lstm_11/lstm_cell_11/recurrent_kernel/Read/ReadVariableOp-lstm_11/lstm_cell_11/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOpConst*
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
!__inference__traced_save_19549726
а
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_5/kerneldense_5/biaslstm_10/lstm_cell_10/kernel%lstm_10/lstm_cell_10/recurrent_kernellstm_10/lstm_cell_10/biaslstm_11/lstm_cell_11/kernel%lstm_11/lstm_cell_11/recurrent_kernellstm_11/lstm_cell_11/biastotalcounttotal_1count_1*
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
$__inference__traced_restore_19549772╪ы#
╧
g
H__inference_dropout_10_layer_call_and_return_conditional_losses_19547034

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
:         G2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape╕
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:         G*
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
:         G2
dropout/GreaterEqualГ
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:         G2
dropout/Cast~
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*+
_output_shapes
:         G2
dropout/Mul_1i
IdentityIdentitydropout/Mul_1:z:0*
T0*+
_output_shapes
:         G2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         G:S O
+
_output_shapes
:         G
 
_user_specified_nameinputs
у
═
while_cond_19545872
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_19545872___redundant_placeholder06
2while_while_cond_19545872___redundant_placeholder16
2while_while_cond_19545872___redundant_placeholder26
2while_while_cond_19545872___redundant_placeholder3
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
B: : : : :         в:         в: ::::: 
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
:         в:.*
(
_output_shapes
:         в:

_output_shapes
: :

_output_shapes
:
Л
З
J__inference_lstm_cell_11_layer_call_and_return_conditional_losses_19546005

inputs

states
states_11
matmul_readvariableop_resource:	GИ4
 matmul_1_readvariableop_resource:
вИ.
biasadd_readvariableop_resource:	И
identity

identity_1

identity_2ИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpвMatMul_1/ReadVariableOpО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	GИ*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         И2
MatMulХ
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
вИ*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         И2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:         И2
addН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:И*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         И2	
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
P:         в:         в:         в:         в*
	num_split2
split`
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:         в2	
Sigmoidd
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:         в2
	Sigmoid_1]
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:         в2
mulW
ReluRelusplit:output:2*
T0*(
_output_shapes
:         в2
Relui
mul_1MulSigmoid:y:0Relu:activations:0*
T0*(
_output_shapes
:         в2
mul_1^
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:         в2
add_1d
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:         в2
	Sigmoid_2V
Relu_1Relu	add_1:z:0*
T0*(
_output_shapes
:         в2
Relu_1m
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*(
_output_shapes
:         в2
mul_2e
IdentityIdentity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:         в2

Identityi

Identity_1Identity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:         в2

Identity_1i

Identity_2Identity	add_1:z:0^NoOp*
T0*(
_output_shapes
:         в2

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
A:         G:         в:         в: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:         G
 
_user_specified_nameinputs:PL
(
_output_shapes
:         в
 
_user_specified_namestates:PL
(
_output_shapes
:         в
 
_user_specified_namestates
╢
╕
*__inference_lstm_11_layer_call_fn_19549404

inputs
unknown:	GИ
	unknown_0:
вИ
	unknown_1:	И
identityИвStatefulPartitionedCallЗ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         в*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_lstm_11_layer_call_and_return_conditional_losses_195470052
StatefulPartitionedCallА
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:         в2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         G: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:         G
 
_user_specified_nameinputs
╦F
О
E__inference_lstm_11_layer_call_and_return_conditional_losses_19545942

inputs(
lstm_cell_11_19545860:	GИ)
lstm_cell_11_19545862:
вИ$
lstm_cell_11_19545864:	И
identityИв$lstm_cell_11/StatefulPartitionedCallвwhileD
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
B :в2
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
B :в2
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
:         в2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :в2
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
B :в2
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
:         в2	
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
 :                  G2
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
valueB"    G   27
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
:         G*
shrink_axis_mask2
strided_slice_2и
$lstm_cell_11/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_11_19545860lstm_cell_11_19545862lstm_cell_11_19545864*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:         в:         в:         в*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_lstm_cell_11_layer_call_and_return_conditional_losses_195458592&
$lstm_cell_11/StatefulPartitionedCallП
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    в   2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_11_19545860lstm_cell_11_19545862lstm_cell_11_19545864*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :         в:         в: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_19545873*
condR
while_cond_19545872*M
output_shapes<
:: : : : :         в:         в: : : : : *
parallel_iterations 2
while╡
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    в   22
0TensorArrayV2Stack/TensorListStack/element_shapeЄ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:                  в*
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
:         в*
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
!:                  в2
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
!:                  в2

Identity}
NoOpNoOp%^lstm_cell_11/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  G: : : 2L
$lstm_cell_11/StatefulPartitionedCall$lstm_cell_11/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :                  G
 
_user_specified_nameinputs
ч[
Э
E__inference_lstm_10_layer_call_and_return_conditional_losses_19546571

inputs>
+lstm_cell_10_matmul_readvariableop_resource:	]Ь@
-lstm_cell_10_matmul_1_readvariableop_resource:	GЬ;
,lstm_cell_10_biasadd_readvariableop_resource:	Ь
identityИв#lstm_cell_10/BiasAdd/ReadVariableOpв"lstm_cell_10/MatMul/ReadVariableOpв$lstm_cell_10/MatMul_1/ReadVariableOpвwhileD
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
value	B :G2
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
value	B :G2
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
:         G2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :G2
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
value	B :G2
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
:         G2	
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
"lstm_cell_10/MatMul/ReadVariableOpReadVariableOp+lstm_cell_10_matmul_readvariableop_resource*
_output_shapes
:	]Ь*
dtype02$
"lstm_cell_10/MatMul/ReadVariableOpн
lstm_cell_10/MatMulMatMulstrided_slice_2:output:0*lstm_cell_10/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ь2
lstm_cell_10/MatMul╗
$lstm_cell_10/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_10_matmul_1_readvariableop_resource*
_output_shapes
:	GЬ*
dtype02&
$lstm_cell_10/MatMul_1/ReadVariableOpй
lstm_cell_10/MatMul_1MatMulzeros:output:0,lstm_cell_10/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ь2
lstm_cell_10/MatMul_1а
lstm_cell_10/addAddV2lstm_cell_10/MatMul:product:0lstm_cell_10/MatMul_1:product:0*
T0*(
_output_shapes
:         Ь2
lstm_cell_10/add┤
#lstm_cell_10/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_10_biasadd_readvariableop_resource*
_output_shapes	
:Ь*
dtype02%
#lstm_cell_10/BiasAdd/ReadVariableOpн
lstm_cell_10/BiasAddBiasAddlstm_cell_10/add:z:0+lstm_cell_10/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ь2
lstm_cell_10/BiasAdd~
lstm_cell_10/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_10/split/split_dimє
lstm_cell_10/splitSplit%lstm_cell_10/split/split_dim:output:0lstm_cell_10/BiasAdd:output:0*
T0*`
_output_shapesN
L:         G:         G:         G:         G*
	num_split2
lstm_cell_10/splitЖ
lstm_cell_10/SigmoidSigmoidlstm_cell_10/split:output:0*
T0*'
_output_shapes
:         G2
lstm_cell_10/SigmoidК
lstm_cell_10/Sigmoid_1Sigmoidlstm_cell_10/split:output:1*
T0*'
_output_shapes
:         G2
lstm_cell_10/Sigmoid_1Л
lstm_cell_10/mulMullstm_cell_10/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:         G2
lstm_cell_10/mul}
lstm_cell_10/ReluRelulstm_cell_10/split:output:2*
T0*'
_output_shapes
:         G2
lstm_cell_10/ReluЬ
lstm_cell_10/mul_1Mullstm_cell_10/Sigmoid:y:0lstm_cell_10/Relu:activations:0*
T0*'
_output_shapes
:         G2
lstm_cell_10/mul_1С
lstm_cell_10/add_1AddV2lstm_cell_10/mul:z:0lstm_cell_10/mul_1:z:0*
T0*'
_output_shapes
:         G2
lstm_cell_10/add_1К
lstm_cell_10/Sigmoid_2Sigmoidlstm_cell_10/split:output:3*
T0*'
_output_shapes
:         G2
lstm_cell_10/Sigmoid_2|
lstm_cell_10/Relu_1Relulstm_cell_10/add_1:z:0*
T0*'
_output_shapes
:         G2
lstm_cell_10/Relu_1а
lstm_cell_10/mul_2Mullstm_cell_10/Sigmoid_2:y:0!lstm_cell_10/Relu_1:activations:0*
T0*'
_output_shapes
:         G2
lstm_cell_10/mul_2П
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    G   2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_10_matmul_readvariableop_resource-lstm_cell_10_matmul_1_readvariableop_resource,lstm_cell_10_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         G:         G: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_19546487*
condR
while_cond_19546486*K
output_shapes:
8: : : : :         G:         G: : : : : *
parallel_iterations 2
while╡
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    G   22
0TensorArrayV2Stack/TensorListStack/element_shapeш
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:         G*
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
:         G*
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
:         G2
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
:         G2

Identity╚
NoOpNoOp$^lstm_cell_10/BiasAdd/ReadVariableOp#^lstm_cell_10/MatMul/ReadVariableOp%^lstm_cell_10/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         ]: : : 2J
#lstm_cell_10/BiasAdd/ReadVariableOp#lstm_cell_10/BiasAdd/ReadVariableOp2H
"lstm_cell_10/MatMul/ReadVariableOp"lstm_cell_10/MatMul/ReadVariableOp2L
$lstm_cell_10/MatMul_1/ReadVariableOp$lstm_cell_10/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:         ]
 
_user_specified_nameinputs
Ж
Ш
*__inference_dense_5_layer_call_fn_19549471

inputs
unknown:	в
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
E__inference_dense_5_layer_call_and_return_conditional_losses_195467822
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
:         в: : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:         в
 
_user_specified_nameinputs
▀
═
while_cond_19547116
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_19547116___redundant_placeholder06
2while_while_cond_19547116___redundant_placeholder16
2while_while_cond_19547116___redundant_placeholder26
2while_while_cond_19547116___redundant_placeholder3
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
@: : : : :         G:         G: ::::: 
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
:         G:-)
'
_output_shapes
:         G:

_output_shapes
: :

_output_shapes
:
╟
∙
/__inference_lstm_cell_11_layer_call_fn_19549667

inputs
states_0
states_1
unknown:	GИ
	unknown_0:
вИ
	unknown_1:	И
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
<:         в:         в:         в*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_lstm_cell_11_layer_call_and_return_conditional_losses_195460052
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         в2

IdentityА

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:         в2

Identity_1А

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:         в2

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
A:         G:         в:         в: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         G
 
_user_specified_nameinputs:RN
(
_output_shapes
:         в
"
_user_specified_name
states/0:RN
(
_output_shapes
:         в
"
_user_specified_name
states/1
▌
╣
*__inference_lstm_10_layer_call_fn_19548696
inputs_0
unknown:	]Ь
	unknown_0:	GЬ
	unknown_1:	Ь
identityИвStatefulPartitionedCallС
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :                  G*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_lstm_10_layer_call_and_return_conditional_losses_195453122
StatefulPartitionedCallИ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :                  G2

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
у
═
while_cond_19546082
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_19546082___redundant_placeholder06
2while_while_cond_19546082___redundant_placeholder16
2while_while_cond_19546082___redundant_placeholder26
2while_while_cond_19546082___redundant_placeholder3
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
B: : : : :         в:         в: ::::: 
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
:         в:.*
(
_output_shapes
:         в:

_output_shapes
: :

_output_shapes
:
▀
═
while_cond_19548449
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_19548449___redundant_placeholder06
2while_while_cond_19548449___redundant_placeholder16
2while_while_cond_19548449___redundant_placeholder26
2while_while_cond_19548449___redundant_placeholder3
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
@: : : : :         G:         G: ::::: 
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
:         G:-)
'
_output_shapes
:         G:

_output_shapes
: :

_output_shapes
:
▀
═
while_cond_19545242
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_19545242___redundant_placeholder06
2while_while_cond_19545242___redundant_placeholder16
2while_while_cond_19545242___redundant_placeholder26
2while_while_cond_19545242___redundant_placeholder3
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
@: : : : :         G:         G: ::::: 
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
:         G:-)
'
_output_shapes
:         G:

_output_shapes
: :

_output_shapes
:
у
═
while_cond_19549124
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_19549124___redundant_placeholder06
2while_while_cond_19549124___redundant_placeholder16
2while_while_cond_19549124___redundant_placeholder26
2while_while_cond_19549124___redundant_placeholder3
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
B: : : : :         в:         в: ::::: 
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
:         в:.*
(
_output_shapes
:         в:

_output_shapes
: :

_output_shapes
:
д
╡
J__inference_sequential_5_layer_call_and_return_conditional_losses_19547258

inputs#
lstm_10_19547236:	]Ь#
lstm_10_19547238:	GЬ
lstm_10_19547240:	Ь#
lstm_11_19547244:	GИ$
lstm_11_19547246:
вИ
lstm_11_19547248:	И#
dense_5_19547252:	в
dense_5_19547254:
identityИвdense_5/StatefulPartitionedCallв"dropout_10/StatefulPartitionedCallв"dropout_11/StatefulPartitionedCallвlstm_10/StatefulPartitionedCallвlstm_11/StatefulPartitionedCallн
lstm_10/StatefulPartitionedCallStatefulPartitionedCallinputslstm_10_19547236lstm_10_19547238lstm_10_19547240*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         G*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_lstm_10_layer_call_and_return_conditional_losses_195472012!
lstm_10/StatefulPartitionedCallЪ
"dropout_10/StatefulPartitionedCallStatefulPartitionedCall(lstm_10/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         G* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_dropout_10_layer_call_and_return_conditional_losses_195470342$
"dropout_10/StatefulPartitionedCall╙
lstm_11/StatefulPartitionedCallStatefulPartitionedCall+dropout_10/StatefulPartitionedCall:output:0lstm_11_19547244lstm_11_19547246lstm_11_19547248*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         в*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_lstm_11_layer_call_and_return_conditional_losses_195470052!
lstm_11/StatefulPartitionedCall└
"dropout_11/StatefulPartitionedCallStatefulPartitionedCall(lstm_11/StatefulPartitionedCall:output:0#^dropout_10/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         в* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_dropout_11_layer_call_and_return_conditional_losses_195468382$
"dropout_11/StatefulPartitionedCall╛
dense_5/StatefulPartitionedCallStatefulPartitionedCall+dropout_11/StatefulPartitionedCall:output:0dense_5_19547252dense_5_19547254*
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
E__inference_dense_5_layer_call_and_return_conditional_losses_195467822!
dense_5/StatefulPartitionedCallЗ
IdentityIdentity(dense_5/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:         2

Identity■
NoOpNoOp ^dense_5/StatefulPartitionedCall#^dropout_10/StatefulPartitionedCall#^dropout_11/StatefulPartitionedCall ^lstm_10/StatefulPartitionedCall ^lstm_11/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':         ]: : : : : : : : 2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2H
"dropout_10/StatefulPartitionedCall"dropout_10/StatefulPartitionedCall2H
"dropout_11/StatefulPartitionedCall"dropout_11/StatefulPartitionedCall2B
lstm_10/StatefulPartitionedCalllstm_10/StatefulPartitionedCall2B
lstm_11/StatefulPartitionedCalllstm_11/StatefulPartitionedCall:S O
+
_output_shapes
:         ]
 
_user_specified_nameinputs
Ъ?
╥
while_body_19548148
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_10_matmul_readvariableop_resource_0:	]ЬH
5while_lstm_cell_10_matmul_1_readvariableop_resource_0:	GЬC
4while_lstm_cell_10_biasadd_readvariableop_resource_0:	Ь
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_10_matmul_readvariableop_resource:	]ЬF
3while_lstm_cell_10_matmul_1_readvariableop_resource:	GЬA
2while_lstm_cell_10_biasadd_readvariableop_resource:	ЬИв)while/lstm_cell_10/BiasAdd/ReadVariableOpв(while/lstm_cell_10/MatMul/ReadVariableOpв*while/lstm_cell_10/MatMul_1/ReadVariableOp├
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
(while/lstm_cell_10/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_10_matmul_readvariableop_resource_0*
_output_shapes
:	]Ь*
dtype02*
(while/lstm_cell_10/MatMul/ReadVariableOp╫
while/lstm_cell_10/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_10/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ь2
while/lstm_cell_10/MatMul╧
*while/lstm_cell_10/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_10_matmul_1_readvariableop_resource_0*
_output_shapes
:	GЬ*
dtype02,
*while/lstm_cell_10/MatMul_1/ReadVariableOp└
while/lstm_cell_10/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_10/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ь2
while/lstm_cell_10/MatMul_1╕
while/lstm_cell_10/addAddV2#while/lstm_cell_10/MatMul:product:0%while/lstm_cell_10/MatMul_1:product:0*
T0*(
_output_shapes
:         Ь2
while/lstm_cell_10/add╚
)while/lstm_cell_10/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_10_biasadd_readvariableop_resource_0*
_output_shapes	
:Ь*
dtype02+
)while/lstm_cell_10/BiasAdd/ReadVariableOp┼
while/lstm_cell_10/BiasAddBiasAddwhile/lstm_cell_10/add:z:01while/lstm_cell_10/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ь2
while/lstm_cell_10/BiasAddК
"while/lstm_cell_10/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_10/split/split_dimЛ
while/lstm_cell_10/splitSplit+while/lstm_cell_10/split/split_dim:output:0#while/lstm_cell_10/BiasAdd:output:0*
T0*`
_output_shapesN
L:         G:         G:         G:         G*
	num_split2
while/lstm_cell_10/splitШ
while/lstm_cell_10/SigmoidSigmoid!while/lstm_cell_10/split:output:0*
T0*'
_output_shapes
:         G2
while/lstm_cell_10/SigmoidЬ
while/lstm_cell_10/Sigmoid_1Sigmoid!while/lstm_cell_10/split:output:1*
T0*'
_output_shapes
:         G2
while/lstm_cell_10/Sigmoid_1а
while/lstm_cell_10/mulMul while/lstm_cell_10/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:         G2
while/lstm_cell_10/mulП
while/lstm_cell_10/ReluRelu!while/lstm_cell_10/split:output:2*
T0*'
_output_shapes
:         G2
while/lstm_cell_10/Relu┤
while/lstm_cell_10/mul_1Mulwhile/lstm_cell_10/Sigmoid:y:0%while/lstm_cell_10/Relu:activations:0*
T0*'
_output_shapes
:         G2
while/lstm_cell_10/mul_1й
while/lstm_cell_10/add_1AddV2while/lstm_cell_10/mul:z:0while/lstm_cell_10/mul_1:z:0*
T0*'
_output_shapes
:         G2
while/lstm_cell_10/add_1Ь
while/lstm_cell_10/Sigmoid_2Sigmoid!while/lstm_cell_10/split:output:3*
T0*'
_output_shapes
:         G2
while/lstm_cell_10/Sigmoid_2О
while/lstm_cell_10/Relu_1Reluwhile/lstm_cell_10/add_1:z:0*
T0*'
_output_shapes
:         G2
while/lstm_cell_10/Relu_1╕
while/lstm_cell_10/mul_2Mul while/lstm_cell_10/Sigmoid_2:y:0'while/lstm_cell_10/Relu_1:activations:0*
T0*'
_output_shapes
:         G2
while/lstm_cell_10/mul_2р
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_10/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_10/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:         G2
while/Identity_4Н
while/Identity_5Identitywhile/lstm_cell_10/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:         G2
while/Identity_5▐

while/NoOpNoOp*^while/lstm_cell_10/BiasAdd/ReadVariableOp)^while/lstm_cell_10/MatMul/ReadVariableOp+^while/lstm_cell_10/MatMul_1/ReadVariableOp*"
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
2while_lstm_cell_10_biasadd_readvariableop_resource4while_lstm_cell_10_biasadd_readvariableop_resource_0"l
3while_lstm_cell_10_matmul_1_readvariableop_resource5while_lstm_cell_10_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_10_matmul_readvariableop_resource3while_lstm_cell_10_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         G:         G: : : : : 2V
)while/lstm_cell_10/BiasAdd/ReadVariableOp)while/lstm_cell_10/BiasAdd/ReadVariableOp2T
(while/lstm_cell_10/MatMul/ReadVariableOp(while/lstm_cell_10/MatMul/ReadVariableOp2X
*while/lstm_cell_10/MatMul_1/ReadVariableOp*while/lstm_cell_10/MatMul_1/ReadVariableOp: 
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
:         G:-)
'
_output_shapes
:         G:

_output_shapes
: :

_output_shapes
: 
╪
I
-__inference_dropout_11_layer_call_fn_19549426

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
:         в* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_dropout_11_layer_call_and_return_conditional_losses_195467492
PartitionedCallq
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:         в2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         в:T P
,
_output_shapes
:         в
 
_user_specified_nameinputs
├\
а
E__inference_lstm_11_layer_call_and_return_conditional_losses_19549058
inputs_0>
+lstm_cell_11_matmul_readvariableop_resource:	GИA
-lstm_cell_11_matmul_1_readvariableop_resource:
вИ;
,lstm_cell_11_biasadd_readvariableop_resource:	И
identityИв#lstm_cell_11/BiasAdd/ReadVariableOpв"lstm_cell_11/MatMul/ReadVariableOpв$lstm_cell_11/MatMul_1/ReadVariableOpвwhileF
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
B :в2
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
B :в2
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
:         в2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :в2
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
B :в2
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
:         в2	
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
 :                  G2
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
valueB"    G   27
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
:         G*
shrink_axis_mask2
strided_slice_2╡
"lstm_cell_11/MatMul/ReadVariableOpReadVariableOp+lstm_cell_11_matmul_readvariableop_resource*
_output_shapes
:	GИ*
dtype02$
"lstm_cell_11/MatMul/ReadVariableOpн
lstm_cell_11/MatMulMatMulstrided_slice_2:output:0*lstm_cell_11/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         И2
lstm_cell_11/MatMul╝
$lstm_cell_11/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_11_matmul_1_readvariableop_resource* 
_output_shapes
:
вИ*
dtype02&
$lstm_cell_11/MatMul_1/ReadVariableOpй
lstm_cell_11/MatMul_1MatMulzeros:output:0,lstm_cell_11/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         И2
lstm_cell_11/MatMul_1а
lstm_cell_11/addAddV2lstm_cell_11/MatMul:product:0lstm_cell_11/MatMul_1:product:0*
T0*(
_output_shapes
:         И2
lstm_cell_11/add┤
#lstm_cell_11/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_11_biasadd_readvariableop_resource*
_output_shapes	
:И*
dtype02%
#lstm_cell_11/BiasAdd/ReadVariableOpн
lstm_cell_11/BiasAddBiasAddlstm_cell_11/add:z:0+lstm_cell_11/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         И2
lstm_cell_11/BiasAdd~
lstm_cell_11/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_11/split/split_dimў
lstm_cell_11/splitSplit%lstm_cell_11/split/split_dim:output:0lstm_cell_11/BiasAdd:output:0*
T0*d
_output_shapesR
P:         в:         в:         в:         в*
	num_split2
lstm_cell_11/splitЗ
lstm_cell_11/SigmoidSigmoidlstm_cell_11/split:output:0*
T0*(
_output_shapes
:         в2
lstm_cell_11/SigmoidЛ
lstm_cell_11/Sigmoid_1Sigmoidlstm_cell_11/split:output:1*
T0*(
_output_shapes
:         в2
lstm_cell_11/Sigmoid_1М
lstm_cell_11/mulMullstm_cell_11/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:         в2
lstm_cell_11/mul~
lstm_cell_11/ReluRelulstm_cell_11/split:output:2*
T0*(
_output_shapes
:         в2
lstm_cell_11/ReluЭ
lstm_cell_11/mul_1Mullstm_cell_11/Sigmoid:y:0lstm_cell_11/Relu:activations:0*
T0*(
_output_shapes
:         в2
lstm_cell_11/mul_1Т
lstm_cell_11/add_1AddV2lstm_cell_11/mul:z:0lstm_cell_11/mul_1:z:0*
T0*(
_output_shapes
:         в2
lstm_cell_11/add_1Л
lstm_cell_11/Sigmoid_2Sigmoidlstm_cell_11/split:output:3*
T0*(
_output_shapes
:         в2
lstm_cell_11/Sigmoid_2}
lstm_cell_11/Relu_1Relulstm_cell_11/add_1:z:0*
T0*(
_output_shapes
:         в2
lstm_cell_11/Relu_1б
lstm_cell_11/mul_2Mullstm_cell_11/Sigmoid_2:y:0!lstm_cell_11/Relu_1:activations:0*
T0*(
_output_shapes
:         в2
lstm_cell_11/mul_2П
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    в   2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_11_matmul_readvariableop_resource-lstm_cell_11_matmul_1_readvariableop_resource,lstm_cell_11_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :         в:         в: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_19548974*
condR
while_cond_19548973*M
output_shapes<
:: : : : :         в:         в: : : : : *
parallel_iterations 2
while╡
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    в   22
0TensorArrayV2Stack/TensorListStack/element_shapeЄ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:                  в*
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
:         в*
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
!:                  в2
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
!:                  в2

Identity╚
NoOpNoOp$^lstm_cell_11/BiasAdd/ReadVariableOp#^lstm_cell_11/MatMul/ReadVariableOp%^lstm_cell_11/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  G: : : 2J
#lstm_cell_11/BiasAdd/ReadVariableOp#lstm_cell_11/BiasAdd/ReadVariableOp2H
"lstm_cell_11/MatMul/ReadVariableOp"lstm_cell_11/MatMul/ReadVariableOp2L
$lstm_cell_11/MatMul_1/ReadVariableOp$lstm_cell_11/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :                  G
"
_user_specified_name
inputs/0
Ё%
▐
!__inference__traced_save_19549726
file_prefix-
)savev2_dense_5_kernel_read_readvariableop+
'savev2_dense_5_bias_read_readvariableop:
6savev2_lstm_10_lstm_cell_10_kernel_read_readvariableopD
@savev2_lstm_10_lstm_cell_10_recurrent_kernel_read_readvariableop8
4savev2_lstm_10_lstm_cell_10_bias_read_readvariableop:
6savev2_lstm_11_lstm_cell_11_kernel_read_readvariableopD
@savev2_lstm_11_lstm_cell_11_recurrent_kernel_read_readvariableop8
4savev2_lstm_11_lstm_cell_11_bias_read_readvariableop$
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
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0)savev2_dense_5_kernel_read_readvariableop'savev2_dense_5_bias_read_readvariableop6savev2_lstm_10_lstm_cell_10_kernel_read_readvariableop@savev2_lstm_10_lstm_cell_10_recurrent_kernel_read_readvariableop4savev2_lstm_10_lstm_cell_10_bias_read_readvariableop6savev2_lstm_11_lstm_cell_11_kernel_read_readvariableop@savev2_lstm_11_lstm_cell_11_recurrent_kernel_read_readvariableop4savev2_lstm_11_lstm_cell_11_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableopsavev2_const"/device:CPU:0*
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
X: :	в::	]Ь:	GЬ:Ь:	GИ:
вИ:И: : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	в: 

_output_shapes
::%!

_output_shapes
:	]Ь:%!

_output_shapes
:	GЬ:!

_output_shapes	
:Ь:%!

_output_shapes
:	GИ:&"
 
_output_shapes
:
вИ:!

_output_shapes	
:И:	
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
¤
И
J__inference_lstm_cell_10_layer_call_and_return_conditional_losses_19549535

inputs
states_0
states_11
matmul_readvariableop_resource:	]Ь3
 matmul_1_readvariableop_resource:	GЬ.
biasadd_readvariableop_resource:	Ь
identity

identity_1

identity_2ИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpвMatMul_1/ReadVariableOpО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	]Ь*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ь2
MatMulФ
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	GЬ*
dtype02
MatMul_1/ReadVariableOp|
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ь2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:         Ь2
addН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:Ь*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ь2	
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
L:         G:         G:         G:         G*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:         G2	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:         G2
	Sigmoid_1\
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:         G2
mulV
ReluRelusplit:output:2*
T0*'
_output_shapes
:         G2
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:         G2
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:         G2
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:         G2
	Sigmoid_2U
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:         G2
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:         G2
mul_2d
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:         G2

Identityh

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:         G2

Identity_1h

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:         G2

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
?:         ]:         G:         G: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:         ]
 
_user_specified_nameinputs:QM
'
_output_shapes
:         G
"
_user_specified_name
states/0:QM
'
_output_shapes
:         G
"
_user_specified_name
states/1
╨

э
lstm_10_while_cond_19547764,
(lstm_10_while_lstm_10_while_loop_counter2
.lstm_10_while_lstm_10_while_maximum_iterations
lstm_10_while_placeholder
lstm_10_while_placeholder_1
lstm_10_while_placeholder_2
lstm_10_while_placeholder_3.
*lstm_10_while_less_lstm_10_strided_slice_1F
Blstm_10_while_lstm_10_while_cond_19547764___redundant_placeholder0F
Blstm_10_while_lstm_10_while_cond_19547764___redundant_placeholder1F
Blstm_10_while_lstm_10_while_cond_19547764___redundant_placeholder2F
Blstm_10_while_lstm_10_while_cond_19547764___redundant_placeholder3
lstm_10_while_identity
Ш
lstm_10/while/LessLesslstm_10_while_placeholder*lstm_10_while_less_lstm_10_strided_slice_1*
T0*
_output_shapes
: 2
lstm_10/while/Lessu
lstm_10/while/IdentityIdentitylstm_10/while/Less:z:0*
T0
*
_output_shapes
: 2
lstm_10/while/Identity"9
lstm_10_while_identitylstm_10/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :         G:         G: ::::: 
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
:         G:-)
'
_output_shapes
:         G:

_output_shapes
: :

_output_shapes
:
Й
f
H__inference_dropout_11_layer_call_and_return_conditional_losses_19549409

inputs

identity_1_
IdentityIdentityinputs*
T0*,
_output_shapes
:         в2

Identityn

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:         в2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         в:T P
,
_output_shapes
:         в
 
_user_specified_nameinputs
╧J
╥

lstm_10_while_body_19547438,
(lstm_10_while_lstm_10_while_loop_counter2
.lstm_10_while_lstm_10_while_maximum_iterations
lstm_10_while_placeholder
lstm_10_while_placeholder_1
lstm_10_while_placeholder_2
lstm_10_while_placeholder_3+
'lstm_10_while_lstm_10_strided_slice_1_0g
clstm_10_while_tensorarrayv2read_tensorlistgetitem_lstm_10_tensorarrayunstack_tensorlistfromtensor_0N
;lstm_10_while_lstm_cell_10_matmul_readvariableop_resource_0:	]ЬP
=lstm_10_while_lstm_cell_10_matmul_1_readvariableop_resource_0:	GЬK
<lstm_10_while_lstm_cell_10_biasadd_readvariableop_resource_0:	Ь
lstm_10_while_identity
lstm_10_while_identity_1
lstm_10_while_identity_2
lstm_10_while_identity_3
lstm_10_while_identity_4
lstm_10_while_identity_5)
%lstm_10_while_lstm_10_strided_slice_1e
alstm_10_while_tensorarrayv2read_tensorlistgetitem_lstm_10_tensorarrayunstack_tensorlistfromtensorL
9lstm_10_while_lstm_cell_10_matmul_readvariableop_resource:	]ЬN
;lstm_10_while_lstm_cell_10_matmul_1_readvariableop_resource:	GЬI
:lstm_10_while_lstm_cell_10_biasadd_readvariableop_resource:	ЬИв1lstm_10/while/lstm_cell_10/BiasAdd/ReadVariableOpв0lstm_10/while/lstm_cell_10/MatMul/ReadVariableOpв2lstm_10/while/lstm_cell_10/MatMul_1/ReadVariableOp╙
?lstm_10/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ]   2A
?lstm_10/while/TensorArrayV2Read/TensorListGetItem/element_shapeГ
1lstm_10/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemclstm_10_while_tensorarrayv2read_tensorlistgetitem_lstm_10_tensorarrayunstack_tensorlistfromtensor_0lstm_10_while_placeholderHlstm_10/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         ]*
element_dtype023
1lstm_10/while/TensorArrayV2Read/TensorListGetItemс
0lstm_10/while/lstm_cell_10/MatMul/ReadVariableOpReadVariableOp;lstm_10_while_lstm_cell_10_matmul_readvariableop_resource_0*
_output_shapes
:	]Ь*
dtype022
0lstm_10/while/lstm_cell_10/MatMul/ReadVariableOpў
!lstm_10/while/lstm_cell_10/MatMulMatMul8lstm_10/while/TensorArrayV2Read/TensorListGetItem:item:08lstm_10/while/lstm_cell_10/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ь2#
!lstm_10/while/lstm_cell_10/MatMulч
2lstm_10/while/lstm_cell_10/MatMul_1/ReadVariableOpReadVariableOp=lstm_10_while_lstm_cell_10_matmul_1_readvariableop_resource_0*
_output_shapes
:	GЬ*
dtype024
2lstm_10/while/lstm_cell_10/MatMul_1/ReadVariableOpр
#lstm_10/while/lstm_cell_10/MatMul_1MatMullstm_10_while_placeholder_2:lstm_10/while/lstm_cell_10/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ь2%
#lstm_10/while/lstm_cell_10/MatMul_1╪
lstm_10/while/lstm_cell_10/addAddV2+lstm_10/while/lstm_cell_10/MatMul:product:0-lstm_10/while/lstm_cell_10/MatMul_1:product:0*
T0*(
_output_shapes
:         Ь2 
lstm_10/while/lstm_cell_10/addр
1lstm_10/while/lstm_cell_10/BiasAdd/ReadVariableOpReadVariableOp<lstm_10_while_lstm_cell_10_biasadd_readvariableop_resource_0*
_output_shapes	
:Ь*
dtype023
1lstm_10/while/lstm_cell_10/BiasAdd/ReadVariableOpх
"lstm_10/while/lstm_cell_10/BiasAddBiasAdd"lstm_10/while/lstm_cell_10/add:z:09lstm_10/while/lstm_cell_10/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ь2$
"lstm_10/while/lstm_cell_10/BiasAddЪ
*lstm_10/while/lstm_cell_10/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2,
*lstm_10/while/lstm_cell_10/split/split_dimл
 lstm_10/while/lstm_cell_10/splitSplit3lstm_10/while/lstm_cell_10/split/split_dim:output:0+lstm_10/while/lstm_cell_10/BiasAdd:output:0*
T0*`
_output_shapesN
L:         G:         G:         G:         G*
	num_split2"
 lstm_10/while/lstm_cell_10/split░
"lstm_10/while/lstm_cell_10/SigmoidSigmoid)lstm_10/while/lstm_cell_10/split:output:0*
T0*'
_output_shapes
:         G2$
"lstm_10/while/lstm_cell_10/Sigmoid┤
$lstm_10/while/lstm_cell_10/Sigmoid_1Sigmoid)lstm_10/while/lstm_cell_10/split:output:1*
T0*'
_output_shapes
:         G2&
$lstm_10/while/lstm_cell_10/Sigmoid_1└
lstm_10/while/lstm_cell_10/mulMul(lstm_10/while/lstm_cell_10/Sigmoid_1:y:0lstm_10_while_placeholder_3*
T0*'
_output_shapes
:         G2 
lstm_10/while/lstm_cell_10/mulз
lstm_10/while/lstm_cell_10/ReluRelu)lstm_10/while/lstm_cell_10/split:output:2*
T0*'
_output_shapes
:         G2!
lstm_10/while/lstm_cell_10/Relu╘
 lstm_10/while/lstm_cell_10/mul_1Mul&lstm_10/while/lstm_cell_10/Sigmoid:y:0-lstm_10/while/lstm_cell_10/Relu:activations:0*
T0*'
_output_shapes
:         G2"
 lstm_10/while/lstm_cell_10/mul_1╔
 lstm_10/while/lstm_cell_10/add_1AddV2"lstm_10/while/lstm_cell_10/mul:z:0$lstm_10/while/lstm_cell_10/mul_1:z:0*
T0*'
_output_shapes
:         G2"
 lstm_10/while/lstm_cell_10/add_1┤
$lstm_10/while/lstm_cell_10/Sigmoid_2Sigmoid)lstm_10/while/lstm_cell_10/split:output:3*
T0*'
_output_shapes
:         G2&
$lstm_10/while/lstm_cell_10/Sigmoid_2ж
!lstm_10/while/lstm_cell_10/Relu_1Relu$lstm_10/while/lstm_cell_10/add_1:z:0*
T0*'
_output_shapes
:         G2#
!lstm_10/while/lstm_cell_10/Relu_1╪
 lstm_10/while/lstm_cell_10/mul_2Mul(lstm_10/while/lstm_cell_10/Sigmoid_2:y:0/lstm_10/while/lstm_cell_10/Relu_1:activations:0*
T0*'
_output_shapes
:         G2"
 lstm_10/while/lstm_cell_10/mul_2И
2lstm_10/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_10_while_placeholder_1lstm_10_while_placeholder$lstm_10/while/lstm_cell_10/mul_2:z:0*
_output_shapes
: *
element_dtype024
2lstm_10/while/TensorArrayV2Write/TensorListSetIteml
lstm_10/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_10/while/add/yЙ
lstm_10/while/addAddV2lstm_10_while_placeholderlstm_10/while/add/y:output:0*
T0*
_output_shapes
: 2
lstm_10/while/addp
lstm_10/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_10/while/add_1/yЮ
lstm_10/while/add_1AddV2(lstm_10_while_lstm_10_while_loop_counterlstm_10/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lstm_10/while/add_1Л
lstm_10/while/IdentityIdentitylstm_10/while/add_1:z:0^lstm_10/while/NoOp*
T0*
_output_shapes
: 2
lstm_10/while/Identityж
lstm_10/while/Identity_1Identity.lstm_10_while_lstm_10_while_maximum_iterations^lstm_10/while/NoOp*
T0*
_output_shapes
: 2
lstm_10/while/Identity_1Н
lstm_10/while/Identity_2Identitylstm_10/while/add:z:0^lstm_10/while/NoOp*
T0*
_output_shapes
: 2
lstm_10/while/Identity_2║
lstm_10/while/Identity_3IdentityBlstm_10/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_10/while/NoOp*
T0*
_output_shapes
: 2
lstm_10/while/Identity_3н
lstm_10/while/Identity_4Identity$lstm_10/while/lstm_cell_10/mul_2:z:0^lstm_10/while/NoOp*
T0*'
_output_shapes
:         G2
lstm_10/while/Identity_4н
lstm_10/while/Identity_5Identity$lstm_10/while/lstm_cell_10/add_1:z:0^lstm_10/while/NoOp*
T0*'
_output_shapes
:         G2
lstm_10/while/Identity_5Ж
lstm_10/while/NoOpNoOp2^lstm_10/while/lstm_cell_10/BiasAdd/ReadVariableOp1^lstm_10/while/lstm_cell_10/MatMul/ReadVariableOp3^lstm_10/while/lstm_cell_10/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
lstm_10/while/NoOp"9
lstm_10_while_identitylstm_10/while/Identity:output:0"=
lstm_10_while_identity_1!lstm_10/while/Identity_1:output:0"=
lstm_10_while_identity_2!lstm_10/while/Identity_2:output:0"=
lstm_10_while_identity_3!lstm_10/while/Identity_3:output:0"=
lstm_10_while_identity_4!lstm_10/while/Identity_4:output:0"=
lstm_10_while_identity_5!lstm_10/while/Identity_5:output:0"P
%lstm_10_while_lstm_10_strided_slice_1'lstm_10_while_lstm_10_strided_slice_1_0"z
:lstm_10_while_lstm_cell_10_biasadd_readvariableop_resource<lstm_10_while_lstm_cell_10_biasadd_readvariableop_resource_0"|
;lstm_10_while_lstm_cell_10_matmul_1_readvariableop_resource=lstm_10_while_lstm_cell_10_matmul_1_readvariableop_resource_0"x
9lstm_10_while_lstm_cell_10_matmul_readvariableop_resource;lstm_10_while_lstm_cell_10_matmul_readvariableop_resource_0"╚
alstm_10_while_tensorarrayv2read_tensorlistgetitem_lstm_10_tensorarrayunstack_tensorlistfromtensorclstm_10_while_tensorarrayv2read_tensorlistgetitem_lstm_10_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         G:         G: : : : : 2f
1lstm_10/while/lstm_cell_10/BiasAdd/ReadVariableOp1lstm_10/while/lstm_cell_10/BiasAdd/ReadVariableOp2d
0lstm_10/while/lstm_cell_10/MatMul/ReadVariableOp0lstm_10/while/lstm_cell_10/MatMul/ReadVariableOp2h
2lstm_10/while/lstm_cell_10/MatMul_1/ReadVariableOp2lstm_10/while/lstm_cell_10/MatMul_1/ReadVariableOp: 
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
:         G:-)
'
_output_shapes
:         G:

_output_shapes
: :

_output_shapes
: 
╟
∙
/__inference_lstm_cell_11_layer_call_fn_19549650

inputs
states_0
states_1
unknown:	GИ
	unknown_0:
вИ
	unknown_1:	И
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
<:         в:         в:         в*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_lstm_cell_11_layer_call_and_return_conditional_losses_195458592
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         в2

IdentityА

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:         в2

Identity_1А

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:         в2

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
A:         G:         в:         в: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         G
 
_user_specified_nameinputs:RN
(
_output_shapes
:         в
"
_user_specified_name
states/0:RN
(
_output_shapes
:         в
"
_user_specified_name
states/1
▀
═
while_cond_19548147
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_19548147___redundant_placeholder06
2while_while_cond_19548147___redundant_placeholder16
2while_while_cond_19548147___redundant_placeholder26
2while_while_cond_19548147___redundant_placeholder3
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
@: : : : :         G:         G: ::::: 
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
:         G:-)
'
_output_shapes
:         G:

_output_shapes
: :

_output_shapes
:
╘

э
lstm_11_while_cond_19547919,
(lstm_11_while_lstm_11_while_loop_counter2
.lstm_11_while_lstm_11_while_maximum_iterations
lstm_11_while_placeholder
lstm_11_while_placeholder_1
lstm_11_while_placeholder_2
lstm_11_while_placeholder_3.
*lstm_11_while_less_lstm_11_strided_slice_1F
Blstm_11_while_lstm_11_while_cond_19547919___redundant_placeholder0F
Blstm_11_while_lstm_11_while_cond_19547919___redundant_placeholder1F
Blstm_11_while_lstm_11_while_cond_19547919___redundant_placeholder2F
Blstm_11_while_lstm_11_while_cond_19547919___redundant_placeholder3
lstm_11_while_identity
Ш
lstm_11/while/LessLesslstm_11_while_placeholder*lstm_11_while_less_lstm_11_strided_slice_1*
T0*
_output_shapes
: 2
lstm_11/while/Lessu
lstm_11/while/IdentityIdentitylstm_11/while/Less:z:0*
T0
*
_output_shapes
: 2
lstm_11/while/Identity"9
lstm_11_while_identitylstm_11/while/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :         в:         в: ::::: 
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
:         в:.*
(
_output_shapes
:         в:

_output_shapes
: :

_output_shapes
:
Ъ?
╥
while_body_19547117
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_10_matmul_readvariableop_resource_0:	]ЬH
5while_lstm_cell_10_matmul_1_readvariableop_resource_0:	GЬC
4while_lstm_cell_10_biasadd_readvariableop_resource_0:	Ь
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_10_matmul_readvariableop_resource:	]ЬF
3while_lstm_cell_10_matmul_1_readvariableop_resource:	GЬA
2while_lstm_cell_10_biasadd_readvariableop_resource:	ЬИв)while/lstm_cell_10/BiasAdd/ReadVariableOpв(while/lstm_cell_10/MatMul/ReadVariableOpв*while/lstm_cell_10/MatMul_1/ReadVariableOp├
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
(while/lstm_cell_10/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_10_matmul_readvariableop_resource_0*
_output_shapes
:	]Ь*
dtype02*
(while/lstm_cell_10/MatMul/ReadVariableOp╫
while/lstm_cell_10/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_10/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ь2
while/lstm_cell_10/MatMul╧
*while/lstm_cell_10/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_10_matmul_1_readvariableop_resource_0*
_output_shapes
:	GЬ*
dtype02,
*while/lstm_cell_10/MatMul_1/ReadVariableOp└
while/lstm_cell_10/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_10/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ь2
while/lstm_cell_10/MatMul_1╕
while/lstm_cell_10/addAddV2#while/lstm_cell_10/MatMul:product:0%while/lstm_cell_10/MatMul_1:product:0*
T0*(
_output_shapes
:         Ь2
while/lstm_cell_10/add╚
)while/lstm_cell_10/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_10_biasadd_readvariableop_resource_0*
_output_shapes	
:Ь*
dtype02+
)while/lstm_cell_10/BiasAdd/ReadVariableOp┼
while/lstm_cell_10/BiasAddBiasAddwhile/lstm_cell_10/add:z:01while/lstm_cell_10/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ь2
while/lstm_cell_10/BiasAddК
"while/lstm_cell_10/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_10/split/split_dimЛ
while/lstm_cell_10/splitSplit+while/lstm_cell_10/split/split_dim:output:0#while/lstm_cell_10/BiasAdd:output:0*
T0*`
_output_shapesN
L:         G:         G:         G:         G*
	num_split2
while/lstm_cell_10/splitШ
while/lstm_cell_10/SigmoidSigmoid!while/lstm_cell_10/split:output:0*
T0*'
_output_shapes
:         G2
while/lstm_cell_10/SigmoidЬ
while/lstm_cell_10/Sigmoid_1Sigmoid!while/lstm_cell_10/split:output:1*
T0*'
_output_shapes
:         G2
while/lstm_cell_10/Sigmoid_1а
while/lstm_cell_10/mulMul while/lstm_cell_10/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:         G2
while/lstm_cell_10/mulП
while/lstm_cell_10/ReluRelu!while/lstm_cell_10/split:output:2*
T0*'
_output_shapes
:         G2
while/lstm_cell_10/Relu┤
while/lstm_cell_10/mul_1Mulwhile/lstm_cell_10/Sigmoid:y:0%while/lstm_cell_10/Relu:activations:0*
T0*'
_output_shapes
:         G2
while/lstm_cell_10/mul_1й
while/lstm_cell_10/add_1AddV2while/lstm_cell_10/mul:z:0while/lstm_cell_10/mul_1:z:0*
T0*'
_output_shapes
:         G2
while/lstm_cell_10/add_1Ь
while/lstm_cell_10/Sigmoid_2Sigmoid!while/lstm_cell_10/split:output:3*
T0*'
_output_shapes
:         G2
while/lstm_cell_10/Sigmoid_2О
while/lstm_cell_10/Relu_1Reluwhile/lstm_cell_10/add_1:z:0*
T0*'
_output_shapes
:         G2
while/lstm_cell_10/Relu_1╕
while/lstm_cell_10/mul_2Mul while/lstm_cell_10/Sigmoid_2:y:0'while/lstm_cell_10/Relu_1:activations:0*
T0*'
_output_shapes
:         G2
while/lstm_cell_10/mul_2р
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_10/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_10/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:         G2
while/Identity_4Н
while/Identity_5Identitywhile/lstm_cell_10/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:         G2
while/Identity_5▐

while/NoOpNoOp*^while/lstm_cell_10/BiasAdd/ReadVariableOp)^while/lstm_cell_10/MatMul/ReadVariableOp+^while/lstm_cell_10/MatMul_1/ReadVariableOp*"
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
2while_lstm_cell_10_biasadd_readvariableop_resource4while_lstm_cell_10_biasadd_readvariableop_resource_0"l
3while_lstm_cell_10_matmul_1_readvariableop_resource5while_lstm_cell_10_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_10_matmul_readvariableop_resource3while_lstm_cell_10_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         G:         G: : : : : 2V
)while/lstm_cell_10/BiasAdd/ReadVariableOp)while/lstm_cell_10/BiasAdd/ReadVariableOp2T
(while/lstm_cell_10/MatMul/ReadVariableOp(while/lstm_cell_10/MatMul/ReadVariableOp2X
*while/lstm_cell_10/MatMul_1/ReadVariableOp*while/lstm_cell_10/MatMul_1/ReadVariableOp: 
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
:         G:-)
'
_output_shapes
:         G:

_output_shapes
: :

_output_shapes
: 
Е&
є
while_body_19546083
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_00
while_lstm_cell_11_19546107_0:	GИ1
while_lstm_cell_11_19546109_0:
вИ,
while_lstm_cell_11_19546111_0:	И
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor.
while_lstm_cell_11_19546107:	GИ/
while_lstm_cell_11_19546109:
вИ*
while_lstm_cell_11_19546111:	ИИв*while/lstm_cell_11/StatefulPartitionedCall├
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    G   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape╙
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         G*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemь
*while/lstm_cell_11/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_11_19546107_0while_lstm_cell_11_19546109_0while_lstm_cell_11_19546111_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:         в:         в:         в*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_lstm_cell_11_layer_call_and_return_conditional_losses_195460052,
*while/lstm_cell_11/StatefulPartitionedCallў
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder3while/lstm_cell_11/StatefulPartitionedCall:output:0*
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
while/Identity_4Identity3while/lstm_cell_11/StatefulPartitionedCall:output:1^while/NoOp*
T0*(
_output_shapes
:         в2
while/Identity_4е
while/Identity_5Identity3while/lstm_cell_11/StatefulPartitionedCall:output:2^while/NoOp*
T0*(
_output_shapes
:         в2
while/Identity_5З

while/NoOpNoOp+^while/lstm_cell_11/StatefulPartitionedCall*"
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
while_lstm_cell_11_19546107while_lstm_cell_11_19546107_0"<
while_lstm_cell_11_19546109while_lstm_cell_11_19546109_0"<
while_lstm_cell_11_19546111while_lstm_cell_11_19546111_0"0
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :         в:         в: : : : : 2X
*while/lstm_cell_11/StatefulPartitionedCall*while/lstm_cell_11/StatefulPartitionedCall: 
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
:         в:.*
(
_output_shapes
:         в:

_output_shapes
: :

_output_shapes
: 
Д\
Ю
E__inference_lstm_11_layer_call_and_return_conditional_losses_19547005

inputs>
+lstm_cell_11_matmul_readvariableop_resource:	GИA
-lstm_cell_11_matmul_1_readvariableop_resource:
вИ;
,lstm_cell_11_biasadd_readvariableop_resource:	И
identityИв#lstm_cell_11/BiasAdd/ReadVariableOpв"lstm_cell_11/MatMul/ReadVariableOpв$lstm_cell_11/MatMul_1/ReadVariableOpвwhileD
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
B :в2
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
B :в2
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
:         в2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :в2
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
B :в2
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
:         в2	
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
:         G2
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
valueB"    G   27
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
:         G*
shrink_axis_mask2
strided_slice_2╡
"lstm_cell_11/MatMul/ReadVariableOpReadVariableOp+lstm_cell_11_matmul_readvariableop_resource*
_output_shapes
:	GИ*
dtype02$
"lstm_cell_11/MatMul/ReadVariableOpн
lstm_cell_11/MatMulMatMulstrided_slice_2:output:0*lstm_cell_11/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         И2
lstm_cell_11/MatMul╝
$lstm_cell_11/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_11_matmul_1_readvariableop_resource* 
_output_shapes
:
вИ*
dtype02&
$lstm_cell_11/MatMul_1/ReadVariableOpй
lstm_cell_11/MatMul_1MatMulzeros:output:0,lstm_cell_11/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         И2
lstm_cell_11/MatMul_1а
lstm_cell_11/addAddV2lstm_cell_11/MatMul:product:0lstm_cell_11/MatMul_1:product:0*
T0*(
_output_shapes
:         И2
lstm_cell_11/add┤
#lstm_cell_11/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_11_biasadd_readvariableop_resource*
_output_shapes	
:И*
dtype02%
#lstm_cell_11/BiasAdd/ReadVariableOpн
lstm_cell_11/BiasAddBiasAddlstm_cell_11/add:z:0+lstm_cell_11/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         И2
lstm_cell_11/BiasAdd~
lstm_cell_11/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_11/split/split_dimў
lstm_cell_11/splitSplit%lstm_cell_11/split/split_dim:output:0lstm_cell_11/BiasAdd:output:0*
T0*d
_output_shapesR
P:         в:         в:         в:         в*
	num_split2
lstm_cell_11/splitЗ
lstm_cell_11/SigmoidSigmoidlstm_cell_11/split:output:0*
T0*(
_output_shapes
:         в2
lstm_cell_11/SigmoidЛ
lstm_cell_11/Sigmoid_1Sigmoidlstm_cell_11/split:output:1*
T0*(
_output_shapes
:         в2
lstm_cell_11/Sigmoid_1М
lstm_cell_11/mulMullstm_cell_11/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:         в2
lstm_cell_11/mul~
lstm_cell_11/ReluRelulstm_cell_11/split:output:2*
T0*(
_output_shapes
:         в2
lstm_cell_11/ReluЭ
lstm_cell_11/mul_1Mullstm_cell_11/Sigmoid:y:0lstm_cell_11/Relu:activations:0*
T0*(
_output_shapes
:         в2
lstm_cell_11/mul_1Т
lstm_cell_11/add_1AddV2lstm_cell_11/mul:z:0lstm_cell_11/mul_1:z:0*
T0*(
_output_shapes
:         в2
lstm_cell_11/add_1Л
lstm_cell_11/Sigmoid_2Sigmoidlstm_cell_11/split:output:3*
T0*(
_output_shapes
:         в2
lstm_cell_11/Sigmoid_2}
lstm_cell_11/Relu_1Relulstm_cell_11/add_1:z:0*
T0*(
_output_shapes
:         в2
lstm_cell_11/Relu_1б
lstm_cell_11/mul_2Mullstm_cell_11/Sigmoid_2:y:0!lstm_cell_11/Relu_1:activations:0*
T0*(
_output_shapes
:         в2
lstm_cell_11/mul_2П
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    в   2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_11_matmul_readvariableop_resource-lstm_cell_11_matmul_1_readvariableop_resource,lstm_cell_11_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :         в:         в: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_19546921*
condR
while_cond_19546920*M
output_shapes<
:: : : : :         в:         в: : : : : *
parallel_iterations 2
while╡
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    в   22
0TensorArrayV2Stack/TensorListStack/element_shapeщ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:         в*
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
:         в*
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
:         в2
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
:         в2

Identity╚
NoOpNoOp$^lstm_cell_11/BiasAdd/ReadVariableOp#^lstm_cell_11/MatMul/ReadVariableOp%^lstm_cell_11/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         G: : : 2J
#lstm_cell_11/BiasAdd/ReadVariableOp#lstm_cell_11/BiasAdd/ReadVariableOp2H
"lstm_cell_11/MatMul/ReadVariableOp"lstm_cell_11/MatMul/ReadVariableOp2L
$lstm_cell_11/MatMul_1/ReadVariableOp$lstm_cell_11/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:         G
 
_user_specified_nameinputs
Д\
Ю
E__inference_lstm_11_layer_call_and_return_conditional_losses_19549209

inputs>
+lstm_cell_11_matmul_readvariableop_resource:	GИA
-lstm_cell_11_matmul_1_readvariableop_resource:
вИ;
,lstm_cell_11_biasadd_readvariableop_resource:	И
identityИв#lstm_cell_11/BiasAdd/ReadVariableOpв"lstm_cell_11/MatMul/ReadVariableOpв$lstm_cell_11/MatMul_1/ReadVariableOpвwhileD
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
B :в2
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
B :в2
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
:         в2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :в2
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
B :в2
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
:         в2	
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
:         G2
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
valueB"    G   27
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
:         G*
shrink_axis_mask2
strided_slice_2╡
"lstm_cell_11/MatMul/ReadVariableOpReadVariableOp+lstm_cell_11_matmul_readvariableop_resource*
_output_shapes
:	GИ*
dtype02$
"lstm_cell_11/MatMul/ReadVariableOpн
lstm_cell_11/MatMulMatMulstrided_slice_2:output:0*lstm_cell_11/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         И2
lstm_cell_11/MatMul╝
$lstm_cell_11/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_11_matmul_1_readvariableop_resource* 
_output_shapes
:
вИ*
dtype02&
$lstm_cell_11/MatMul_1/ReadVariableOpй
lstm_cell_11/MatMul_1MatMulzeros:output:0,lstm_cell_11/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         И2
lstm_cell_11/MatMul_1а
lstm_cell_11/addAddV2lstm_cell_11/MatMul:product:0lstm_cell_11/MatMul_1:product:0*
T0*(
_output_shapes
:         И2
lstm_cell_11/add┤
#lstm_cell_11/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_11_biasadd_readvariableop_resource*
_output_shapes	
:И*
dtype02%
#lstm_cell_11/BiasAdd/ReadVariableOpн
lstm_cell_11/BiasAddBiasAddlstm_cell_11/add:z:0+lstm_cell_11/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         И2
lstm_cell_11/BiasAdd~
lstm_cell_11/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_11/split/split_dimў
lstm_cell_11/splitSplit%lstm_cell_11/split/split_dim:output:0lstm_cell_11/BiasAdd:output:0*
T0*d
_output_shapesR
P:         в:         в:         в:         в*
	num_split2
lstm_cell_11/splitЗ
lstm_cell_11/SigmoidSigmoidlstm_cell_11/split:output:0*
T0*(
_output_shapes
:         в2
lstm_cell_11/SigmoidЛ
lstm_cell_11/Sigmoid_1Sigmoidlstm_cell_11/split:output:1*
T0*(
_output_shapes
:         в2
lstm_cell_11/Sigmoid_1М
lstm_cell_11/mulMullstm_cell_11/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:         в2
lstm_cell_11/mul~
lstm_cell_11/ReluRelulstm_cell_11/split:output:2*
T0*(
_output_shapes
:         в2
lstm_cell_11/ReluЭ
lstm_cell_11/mul_1Mullstm_cell_11/Sigmoid:y:0lstm_cell_11/Relu:activations:0*
T0*(
_output_shapes
:         в2
lstm_cell_11/mul_1Т
lstm_cell_11/add_1AddV2lstm_cell_11/mul:z:0lstm_cell_11/mul_1:z:0*
T0*(
_output_shapes
:         в2
lstm_cell_11/add_1Л
lstm_cell_11/Sigmoid_2Sigmoidlstm_cell_11/split:output:3*
T0*(
_output_shapes
:         в2
lstm_cell_11/Sigmoid_2}
lstm_cell_11/Relu_1Relulstm_cell_11/add_1:z:0*
T0*(
_output_shapes
:         в2
lstm_cell_11/Relu_1б
lstm_cell_11/mul_2Mullstm_cell_11/Sigmoid_2:y:0!lstm_cell_11/Relu_1:activations:0*
T0*(
_output_shapes
:         в2
lstm_cell_11/mul_2П
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    в   2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_11_matmul_readvariableop_resource-lstm_cell_11_matmul_1_readvariableop_resource,lstm_cell_11_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :         в:         в: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_19549125*
condR
while_cond_19549124*M
output_shapes<
:: : : : :         в:         в: : : : : *
parallel_iterations 2
while╡
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    в   22
0TensorArrayV2Stack/TensorListStack/element_shapeщ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:         в*
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
:         в*
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
:         в2
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
:         в2

Identity╚
NoOpNoOp$^lstm_cell_11/BiasAdd/ReadVariableOp#^lstm_cell_11/MatMul/ReadVariableOp%^lstm_cell_11/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         G: : : 2J
#lstm_cell_11/BiasAdd/ReadVariableOp#lstm_cell_11/BiasAdd/ReadVariableOp2H
"lstm_cell_11/MatMul/ReadVariableOp"lstm_cell_11/MatMul/ReadVariableOp2L
$lstm_cell_11/MatMul_1/ReadVariableOp$lstm_cell_11/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:         G
 
_user_specified_nameinputs
хJ
╘

lstm_11_while_body_19547920,
(lstm_11_while_lstm_11_while_loop_counter2
.lstm_11_while_lstm_11_while_maximum_iterations
lstm_11_while_placeholder
lstm_11_while_placeholder_1
lstm_11_while_placeholder_2
lstm_11_while_placeholder_3+
'lstm_11_while_lstm_11_strided_slice_1_0g
clstm_11_while_tensorarrayv2read_tensorlistgetitem_lstm_11_tensorarrayunstack_tensorlistfromtensor_0N
;lstm_11_while_lstm_cell_11_matmul_readvariableop_resource_0:	GИQ
=lstm_11_while_lstm_cell_11_matmul_1_readvariableop_resource_0:
вИK
<lstm_11_while_lstm_cell_11_biasadd_readvariableop_resource_0:	И
lstm_11_while_identity
lstm_11_while_identity_1
lstm_11_while_identity_2
lstm_11_while_identity_3
lstm_11_while_identity_4
lstm_11_while_identity_5)
%lstm_11_while_lstm_11_strided_slice_1e
alstm_11_while_tensorarrayv2read_tensorlistgetitem_lstm_11_tensorarrayunstack_tensorlistfromtensorL
9lstm_11_while_lstm_cell_11_matmul_readvariableop_resource:	GИO
;lstm_11_while_lstm_cell_11_matmul_1_readvariableop_resource:
вИI
:lstm_11_while_lstm_cell_11_biasadd_readvariableop_resource:	ИИв1lstm_11/while/lstm_cell_11/BiasAdd/ReadVariableOpв0lstm_11/while/lstm_cell_11/MatMul/ReadVariableOpв2lstm_11/while/lstm_cell_11/MatMul_1/ReadVariableOp╙
?lstm_11/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    G   2A
?lstm_11/while/TensorArrayV2Read/TensorListGetItem/element_shapeГ
1lstm_11/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemclstm_11_while_tensorarrayv2read_tensorlistgetitem_lstm_11_tensorarrayunstack_tensorlistfromtensor_0lstm_11_while_placeholderHlstm_11/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         G*
element_dtype023
1lstm_11/while/TensorArrayV2Read/TensorListGetItemс
0lstm_11/while/lstm_cell_11/MatMul/ReadVariableOpReadVariableOp;lstm_11_while_lstm_cell_11_matmul_readvariableop_resource_0*
_output_shapes
:	GИ*
dtype022
0lstm_11/while/lstm_cell_11/MatMul/ReadVariableOpў
!lstm_11/while/lstm_cell_11/MatMulMatMul8lstm_11/while/TensorArrayV2Read/TensorListGetItem:item:08lstm_11/while/lstm_cell_11/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         И2#
!lstm_11/while/lstm_cell_11/MatMulш
2lstm_11/while/lstm_cell_11/MatMul_1/ReadVariableOpReadVariableOp=lstm_11_while_lstm_cell_11_matmul_1_readvariableop_resource_0* 
_output_shapes
:
вИ*
dtype024
2lstm_11/while/lstm_cell_11/MatMul_1/ReadVariableOpр
#lstm_11/while/lstm_cell_11/MatMul_1MatMullstm_11_while_placeholder_2:lstm_11/while/lstm_cell_11/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         И2%
#lstm_11/while/lstm_cell_11/MatMul_1╪
lstm_11/while/lstm_cell_11/addAddV2+lstm_11/while/lstm_cell_11/MatMul:product:0-lstm_11/while/lstm_cell_11/MatMul_1:product:0*
T0*(
_output_shapes
:         И2 
lstm_11/while/lstm_cell_11/addр
1lstm_11/while/lstm_cell_11/BiasAdd/ReadVariableOpReadVariableOp<lstm_11_while_lstm_cell_11_biasadd_readvariableop_resource_0*
_output_shapes	
:И*
dtype023
1lstm_11/while/lstm_cell_11/BiasAdd/ReadVariableOpх
"lstm_11/while/lstm_cell_11/BiasAddBiasAdd"lstm_11/while/lstm_cell_11/add:z:09lstm_11/while/lstm_cell_11/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         И2$
"lstm_11/while/lstm_cell_11/BiasAddЪ
*lstm_11/while/lstm_cell_11/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2,
*lstm_11/while/lstm_cell_11/split/split_dimп
 lstm_11/while/lstm_cell_11/splitSplit3lstm_11/while/lstm_cell_11/split/split_dim:output:0+lstm_11/while/lstm_cell_11/BiasAdd:output:0*
T0*d
_output_shapesR
P:         в:         в:         в:         в*
	num_split2"
 lstm_11/while/lstm_cell_11/split▒
"lstm_11/while/lstm_cell_11/SigmoidSigmoid)lstm_11/while/lstm_cell_11/split:output:0*
T0*(
_output_shapes
:         в2$
"lstm_11/while/lstm_cell_11/Sigmoid╡
$lstm_11/while/lstm_cell_11/Sigmoid_1Sigmoid)lstm_11/while/lstm_cell_11/split:output:1*
T0*(
_output_shapes
:         в2&
$lstm_11/while/lstm_cell_11/Sigmoid_1┴
lstm_11/while/lstm_cell_11/mulMul(lstm_11/while/lstm_cell_11/Sigmoid_1:y:0lstm_11_while_placeholder_3*
T0*(
_output_shapes
:         в2 
lstm_11/while/lstm_cell_11/mulи
lstm_11/while/lstm_cell_11/ReluRelu)lstm_11/while/lstm_cell_11/split:output:2*
T0*(
_output_shapes
:         в2!
lstm_11/while/lstm_cell_11/Relu╒
 lstm_11/while/lstm_cell_11/mul_1Mul&lstm_11/while/lstm_cell_11/Sigmoid:y:0-lstm_11/while/lstm_cell_11/Relu:activations:0*
T0*(
_output_shapes
:         в2"
 lstm_11/while/lstm_cell_11/mul_1╩
 lstm_11/while/lstm_cell_11/add_1AddV2"lstm_11/while/lstm_cell_11/mul:z:0$lstm_11/while/lstm_cell_11/mul_1:z:0*
T0*(
_output_shapes
:         в2"
 lstm_11/while/lstm_cell_11/add_1╡
$lstm_11/while/lstm_cell_11/Sigmoid_2Sigmoid)lstm_11/while/lstm_cell_11/split:output:3*
T0*(
_output_shapes
:         в2&
$lstm_11/while/lstm_cell_11/Sigmoid_2з
!lstm_11/while/lstm_cell_11/Relu_1Relu$lstm_11/while/lstm_cell_11/add_1:z:0*
T0*(
_output_shapes
:         в2#
!lstm_11/while/lstm_cell_11/Relu_1┘
 lstm_11/while/lstm_cell_11/mul_2Mul(lstm_11/while/lstm_cell_11/Sigmoid_2:y:0/lstm_11/while/lstm_cell_11/Relu_1:activations:0*
T0*(
_output_shapes
:         в2"
 lstm_11/while/lstm_cell_11/mul_2И
2lstm_11/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_11_while_placeholder_1lstm_11_while_placeholder$lstm_11/while/lstm_cell_11/mul_2:z:0*
_output_shapes
: *
element_dtype024
2lstm_11/while/TensorArrayV2Write/TensorListSetIteml
lstm_11/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_11/while/add/yЙ
lstm_11/while/addAddV2lstm_11_while_placeholderlstm_11/while/add/y:output:0*
T0*
_output_shapes
: 2
lstm_11/while/addp
lstm_11/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_11/while/add_1/yЮ
lstm_11/while/add_1AddV2(lstm_11_while_lstm_11_while_loop_counterlstm_11/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lstm_11/while/add_1Л
lstm_11/while/IdentityIdentitylstm_11/while/add_1:z:0^lstm_11/while/NoOp*
T0*
_output_shapes
: 2
lstm_11/while/Identityж
lstm_11/while/Identity_1Identity.lstm_11_while_lstm_11_while_maximum_iterations^lstm_11/while/NoOp*
T0*
_output_shapes
: 2
lstm_11/while/Identity_1Н
lstm_11/while/Identity_2Identitylstm_11/while/add:z:0^lstm_11/while/NoOp*
T0*
_output_shapes
: 2
lstm_11/while/Identity_2║
lstm_11/while/Identity_3IdentityBlstm_11/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_11/while/NoOp*
T0*
_output_shapes
: 2
lstm_11/while/Identity_3о
lstm_11/while/Identity_4Identity$lstm_11/while/lstm_cell_11/mul_2:z:0^lstm_11/while/NoOp*
T0*(
_output_shapes
:         в2
lstm_11/while/Identity_4о
lstm_11/while/Identity_5Identity$lstm_11/while/lstm_cell_11/add_1:z:0^lstm_11/while/NoOp*
T0*(
_output_shapes
:         в2
lstm_11/while/Identity_5Ж
lstm_11/while/NoOpNoOp2^lstm_11/while/lstm_cell_11/BiasAdd/ReadVariableOp1^lstm_11/while/lstm_cell_11/MatMul/ReadVariableOp3^lstm_11/while/lstm_cell_11/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
lstm_11/while/NoOp"9
lstm_11_while_identitylstm_11/while/Identity:output:0"=
lstm_11_while_identity_1!lstm_11/while/Identity_1:output:0"=
lstm_11_while_identity_2!lstm_11/while/Identity_2:output:0"=
lstm_11_while_identity_3!lstm_11/while/Identity_3:output:0"=
lstm_11_while_identity_4!lstm_11/while/Identity_4:output:0"=
lstm_11_while_identity_5!lstm_11/while/Identity_5:output:0"P
%lstm_11_while_lstm_11_strided_slice_1'lstm_11_while_lstm_11_strided_slice_1_0"z
:lstm_11_while_lstm_cell_11_biasadd_readvariableop_resource<lstm_11_while_lstm_cell_11_biasadd_readvariableop_resource_0"|
;lstm_11_while_lstm_cell_11_matmul_1_readvariableop_resource=lstm_11_while_lstm_cell_11_matmul_1_readvariableop_resource_0"x
9lstm_11_while_lstm_cell_11_matmul_readvariableop_resource;lstm_11_while_lstm_cell_11_matmul_readvariableop_resource_0"╚
alstm_11_while_tensorarrayv2read_tensorlistgetitem_lstm_11_tensorarrayunstack_tensorlistfromtensorclstm_11_while_tensorarrayv2read_tensorlistgetitem_lstm_11_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :         в:         в: : : : : 2f
1lstm_11/while/lstm_cell_11/BiasAdd/ReadVariableOp1lstm_11/while/lstm_cell_11/BiasAdd/ReadVariableOp2d
0lstm_11/while/lstm_cell_11/MatMul/ReadVariableOp0lstm_11/while/lstm_cell_11/MatMul/ReadVariableOp2h
2lstm_11/while/lstm_cell_11/MatMul_1/ReadVariableOp2lstm_11/while/lstm_cell_11/MatMul_1/ReadVariableOp: 
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
:         в:.*
(
_output_shapes
:         в:

_output_shapes
: :

_output_shapes
: 
╣F
Н
E__inference_lstm_10_layer_call_and_return_conditional_losses_19545312

inputs(
lstm_cell_10_19545230:	]Ь(
lstm_cell_10_19545232:	GЬ$
lstm_cell_10_19545234:	Ь
identityИв$lstm_cell_10/StatefulPartitionedCallвwhileD
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
value	B :G2
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
value	B :G2
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
:         G2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :G2
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
value	B :G2
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
:         G2	
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
$lstm_cell_10/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_10_19545230lstm_cell_10_19545232lstm_cell_10_19545234*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:         G:         G:         G*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_lstm_cell_10_layer_call_and_return_conditional_losses_195452292&
$lstm_cell_10/StatefulPartitionedCallП
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    G   2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_10_19545230lstm_cell_10_19545232lstm_cell_10_19545234*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         G:         G: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_19545243*
condR
while_cond_19545242*K
output_shapes:
8: : : : :         G:         G: : : : : *
parallel_iterations 2
while╡
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    G   22
0TensorArrayV2Stack/TensorListStack/element_shapeё
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :                  G*
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
:         G*
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
 :                  G2
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
 :                  G2

Identity}
NoOpNoOp%^lstm_cell_10/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  ]: : : 2L
$lstm_cell_10/StatefulPartitionedCall$lstm_cell_10/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :                  ]
 
_user_specified_nameinputs
╢
╕
*__inference_lstm_11_layer_call_fn_19549393

inputs
unknown:	GИ
	unknown_0:
вИ
	unknown_1:	И
identityИвStatefulPartitionedCallЗ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         в*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_lstm_11_layer_call_and_return_conditional_losses_195467362
StatefulPartitionedCallА
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:         в2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         G: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:         G
 
_user_specified_nameinputs
Х

╩
/__inference_sequential_5_layer_call_fn_19548081

inputs
unknown:	]Ь
	unknown_0:	GЬ
	unknown_1:	Ь
	unknown_2:	GИ
	unknown_3:
вИ
	unknown_4:	И
	unknown_5:	в
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
J__inference_sequential_5_layer_call_and_return_conditional_losses_195472582
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
while_cond_19549275
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_19549275___redundant_placeholder06
2while_while_cond_19549275___redundant_placeholder16
2while_while_cond_19549275___redundant_placeholder26
2while_while_cond_19549275___redundant_placeholder3
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
B: : : : :         в:         в: ::::: 
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
:         в:.*
(
_output_shapes
:         в:

_output_shapes
: :

_output_shapes
:
╧J
╥

lstm_10_while_body_19547765,
(lstm_10_while_lstm_10_while_loop_counter2
.lstm_10_while_lstm_10_while_maximum_iterations
lstm_10_while_placeholder
lstm_10_while_placeholder_1
lstm_10_while_placeholder_2
lstm_10_while_placeholder_3+
'lstm_10_while_lstm_10_strided_slice_1_0g
clstm_10_while_tensorarrayv2read_tensorlistgetitem_lstm_10_tensorarrayunstack_tensorlistfromtensor_0N
;lstm_10_while_lstm_cell_10_matmul_readvariableop_resource_0:	]ЬP
=lstm_10_while_lstm_cell_10_matmul_1_readvariableop_resource_0:	GЬK
<lstm_10_while_lstm_cell_10_biasadd_readvariableop_resource_0:	Ь
lstm_10_while_identity
lstm_10_while_identity_1
lstm_10_while_identity_2
lstm_10_while_identity_3
lstm_10_while_identity_4
lstm_10_while_identity_5)
%lstm_10_while_lstm_10_strided_slice_1e
alstm_10_while_tensorarrayv2read_tensorlistgetitem_lstm_10_tensorarrayunstack_tensorlistfromtensorL
9lstm_10_while_lstm_cell_10_matmul_readvariableop_resource:	]ЬN
;lstm_10_while_lstm_cell_10_matmul_1_readvariableop_resource:	GЬI
:lstm_10_while_lstm_cell_10_biasadd_readvariableop_resource:	ЬИв1lstm_10/while/lstm_cell_10/BiasAdd/ReadVariableOpв0lstm_10/while/lstm_cell_10/MatMul/ReadVariableOpв2lstm_10/while/lstm_cell_10/MatMul_1/ReadVariableOp╙
?lstm_10/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ]   2A
?lstm_10/while/TensorArrayV2Read/TensorListGetItem/element_shapeГ
1lstm_10/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemclstm_10_while_tensorarrayv2read_tensorlistgetitem_lstm_10_tensorarrayunstack_tensorlistfromtensor_0lstm_10_while_placeholderHlstm_10/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         ]*
element_dtype023
1lstm_10/while/TensorArrayV2Read/TensorListGetItemс
0lstm_10/while/lstm_cell_10/MatMul/ReadVariableOpReadVariableOp;lstm_10_while_lstm_cell_10_matmul_readvariableop_resource_0*
_output_shapes
:	]Ь*
dtype022
0lstm_10/while/lstm_cell_10/MatMul/ReadVariableOpў
!lstm_10/while/lstm_cell_10/MatMulMatMul8lstm_10/while/TensorArrayV2Read/TensorListGetItem:item:08lstm_10/while/lstm_cell_10/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ь2#
!lstm_10/while/lstm_cell_10/MatMulч
2lstm_10/while/lstm_cell_10/MatMul_1/ReadVariableOpReadVariableOp=lstm_10_while_lstm_cell_10_matmul_1_readvariableop_resource_0*
_output_shapes
:	GЬ*
dtype024
2lstm_10/while/lstm_cell_10/MatMul_1/ReadVariableOpр
#lstm_10/while/lstm_cell_10/MatMul_1MatMullstm_10_while_placeholder_2:lstm_10/while/lstm_cell_10/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ь2%
#lstm_10/while/lstm_cell_10/MatMul_1╪
lstm_10/while/lstm_cell_10/addAddV2+lstm_10/while/lstm_cell_10/MatMul:product:0-lstm_10/while/lstm_cell_10/MatMul_1:product:0*
T0*(
_output_shapes
:         Ь2 
lstm_10/while/lstm_cell_10/addр
1lstm_10/while/lstm_cell_10/BiasAdd/ReadVariableOpReadVariableOp<lstm_10_while_lstm_cell_10_biasadd_readvariableop_resource_0*
_output_shapes	
:Ь*
dtype023
1lstm_10/while/lstm_cell_10/BiasAdd/ReadVariableOpх
"lstm_10/while/lstm_cell_10/BiasAddBiasAdd"lstm_10/while/lstm_cell_10/add:z:09lstm_10/while/lstm_cell_10/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ь2$
"lstm_10/while/lstm_cell_10/BiasAddЪ
*lstm_10/while/lstm_cell_10/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2,
*lstm_10/while/lstm_cell_10/split/split_dimл
 lstm_10/while/lstm_cell_10/splitSplit3lstm_10/while/lstm_cell_10/split/split_dim:output:0+lstm_10/while/lstm_cell_10/BiasAdd:output:0*
T0*`
_output_shapesN
L:         G:         G:         G:         G*
	num_split2"
 lstm_10/while/lstm_cell_10/split░
"lstm_10/while/lstm_cell_10/SigmoidSigmoid)lstm_10/while/lstm_cell_10/split:output:0*
T0*'
_output_shapes
:         G2$
"lstm_10/while/lstm_cell_10/Sigmoid┤
$lstm_10/while/lstm_cell_10/Sigmoid_1Sigmoid)lstm_10/while/lstm_cell_10/split:output:1*
T0*'
_output_shapes
:         G2&
$lstm_10/while/lstm_cell_10/Sigmoid_1└
lstm_10/while/lstm_cell_10/mulMul(lstm_10/while/lstm_cell_10/Sigmoid_1:y:0lstm_10_while_placeholder_3*
T0*'
_output_shapes
:         G2 
lstm_10/while/lstm_cell_10/mulз
lstm_10/while/lstm_cell_10/ReluRelu)lstm_10/while/lstm_cell_10/split:output:2*
T0*'
_output_shapes
:         G2!
lstm_10/while/lstm_cell_10/Relu╘
 lstm_10/while/lstm_cell_10/mul_1Mul&lstm_10/while/lstm_cell_10/Sigmoid:y:0-lstm_10/while/lstm_cell_10/Relu:activations:0*
T0*'
_output_shapes
:         G2"
 lstm_10/while/lstm_cell_10/mul_1╔
 lstm_10/while/lstm_cell_10/add_1AddV2"lstm_10/while/lstm_cell_10/mul:z:0$lstm_10/while/lstm_cell_10/mul_1:z:0*
T0*'
_output_shapes
:         G2"
 lstm_10/while/lstm_cell_10/add_1┤
$lstm_10/while/lstm_cell_10/Sigmoid_2Sigmoid)lstm_10/while/lstm_cell_10/split:output:3*
T0*'
_output_shapes
:         G2&
$lstm_10/while/lstm_cell_10/Sigmoid_2ж
!lstm_10/while/lstm_cell_10/Relu_1Relu$lstm_10/while/lstm_cell_10/add_1:z:0*
T0*'
_output_shapes
:         G2#
!lstm_10/while/lstm_cell_10/Relu_1╪
 lstm_10/while/lstm_cell_10/mul_2Mul(lstm_10/while/lstm_cell_10/Sigmoid_2:y:0/lstm_10/while/lstm_cell_10/Relu_1:activations:0*
T0*'
_output_shapes
:         G2"
 lstm_10/while/lstm_cell_10/mul_2И
2lstm_10/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_10_while_placeholder_1lstm_10_while_placeholder$lstm_10/while/lstm_cell_10/mul_2:z:0*
_output_shapes
: *
element_dtype024
2lstm_10/while/TensorArrayV2Write/TensorListSetIteml
lstm_10/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_10/while/add/yЙ
lstm_10/while/addAddV2lstm_10_while_placeholderlstm_10/while/add/y:output:0*
T0*
_output_shapes
: 2
lstm_10/while/addp
lstm_10/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_10/while/add_1/yЮ
lstm_10/while/add_1AddV2(lstm_10_while_lstm_10_while_loop_counterlstm_10/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lstm_10/while/add_1Л
lstm_10/while/IdentityIdentitylstm_10/while/add_1:z:0^lstm_10/while/NoOp*
T0*
_output_shapes
: 2
lstm_10/while/Identityж
lstm_10/while/Identity_1Identity.lstm_10_while_lstm_10_while_maximum_iterations^lstm_10/while/NoOp*
T0*
_output_shapes
: 2
lstm_10/while/Identity_1Н
lstm_10/while/Identity_2Identitylstm_10/while/add:z:0^lstm_10/while/NoOp*
T0*
_output_shapes
: 2
lstm_10/while/Identity_2║
lstm_10/while/Identity_3IdentityBlstm_10/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_10/while/NoOp*
T0*
_output_shapes
: 2
lstm_10/while/Identity_3н
lstm_10/while/Identity_4Identity$lstm_10/while/lstm_cell_10/mul_2:z:0^lstm_10/while/NoOp*
T0*'
_output_shapes
:         G2
lstm_10/while/Identity_4н
lstm_10/while/Identity_5Identity$lstm_10/while/lstm_cell_10/add_1:z:0^lstm_10/while/NoOp*
T0*'
_output_shapes
:         G2
lstm_10/while/Identity_5Ж
lstm_10/while/NoOpNoOp2^lstm_10/while/lstm_cell_10/BiasAdd/ReadVariableOp1^lstm_10/while/lstm_cell_10/MatMul/ReadVariableOp3^lstm_10/while/lstm_cell_10/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
lstm_10/while/NoOp"9
lstm_10_while_identitylstm_10/while/Identity:output:0"=
lstm_10_while_identity_1!lstm_10/while/Identity_1:output:0"=
lstm_10_while_identity_2!lstm_10/while/Identity_2:output:0"=
lstm_10_while_identity_3!lstm_10/while/Identity_3:output:0"=
lstm_10_while_identity_4!lstm_10/while/Identity_4:output:0"=
lstm_10_while_identity_5!lstm_10/while/Identity_5:output:0"P
%lstm_10_while_lstm_10_strided_slice_1'lstm_10_while_lstm_10_strided_slice_1_0"z
:lstm_10_while_lstm_cell_10_biasadd_readvariableop_resource<lstm_10_while_lstm_cell_10_biasadd_readvariableop_resource_0"|
;lstm_10_while_lstm_cell_10_matmul_1_readvariableop_resource=lstm_10_while_lstm_cell_10_matmul_1_readvariableop_resource_0"x
9lstm_10_while_lstm_cell_10_matmul_readvariableop_resource;lstm_10_while_lstm_cell_10_matmul_readvariableop_resource_0"╚
alstm_10_while_tensorarrayv2read_tensorlistgetitem_lstm_10_tensorarrayunstack_tensorlistfromtensorclstm_10_while_tensorarrayv2read_tensorlistgetitem_lstm_10_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         G:         G: : : : : 2f
1lstm_10/while/lstm_cell_10/BiasAdd/ReadVariableOp1lstm_10/while/lstm_cell_10/BiasAdd/ReadVariableOp2d
0lstm_10/while/lstm_cell_10/MatMul/ReadVariableOp0lstm_10/while/lstm_cell_10/MatMul/ReadVariableOp2h
2lstm_10/while/lstm_cell_10/MatMul_1/ReadVariableOp2lstm_10/while/lstm_cell_10/MatMul_1/ReadVariableOp: 
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
:         G:-)
'
_output_shapes
:         G:

_output_shapes
: :

_output_shapes
: 
║
°
/__inference_lstm_cell_10_layer_call_fn_19549552

inputs
states_0
states_1
unknown:	]Ь
	unknown_0:	GЬ
	unknown_1:	Ь
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
9:         G:         G:         G*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_lstm_cell_10_layer_call_and_return_conditional_losses_195452292
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         G2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:         G2

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:         G2

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
?:         ]:         G:         G: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         ]
 
_user_specified_nameinputs:QM
'
_output_shapes
:         G
"
_user_specified_name
states/0:QM
'
_output_shapes
:         G
"
_user_specified_name
states/1
Е
f
H__inference_dropout_10_layer_call_and_return_conditional_losses_19548734

inputs

identity_1^
IdentityIdentityinputs*
T0*+
_output_shapes
:         G2

Identitym

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:         G2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         G:S O
+
_output_shapes
:         G
 
_user_specified_nameinputs
м
Є
J__inference_sequential_5_layer_call_and_return_conditional_losses_19547323
lstm_10_input#
lstm_10_19547301:	]Ь#
lstm_10_19547303:	GЬ
lstm_10_19547305:	Ь#
lstm_11_19547309:	GИ$
lstm_11_19547311:
вИ
lstm_11_19547313:	И#
dense_5_19547317:	в
dense_5_19547319:
identityИвdense_5/StatefulPartitionedCallвlstm_10/StatefulPartitionedCallвlstm_11/StatefulPartitionedCall┤
lstm_10/StatefulPartitionedCallStatefulPartitionedCalllstm_10_inputlstm_10_19547301lstm_10_19547303lstm_10_19547305*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         G*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_lstm_10_layer_call_and_return_conditional_losses_195465712!
lstm_10/StatefulPartitionedCallВ
dropout_10/PartitionedCallPartitionedCall(lstm_10/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         G* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_dropout_10_layer_call_and_return_conditional_losses_195465842
dropout_10/PartitionedCall╦
lstm_11/StatefulPartitionedCallStatefulPartitionedCall#dropout_10/PartitionedCall:output:0lstm_11_19547309lstm_11_19547311lstm_11_19547313*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         в*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_lstm_11_layer_call_and_return_conditional_losses_195467362!
lstm_11/StatefulPartitionedCallГ
dropout_11/PartitionedCallPartitionedCall(lstm_11/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         в* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_dropout_11_layer_call_and_return_conditional_losses_195467492
dropout_11/PartitionedCall╢
dense_5/StatefulPartitionedCallStatefulPartitionedCall#dropout_11/PartitionedCall:output:0dense_5_19547317dense_5_19547319*
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
E__inference_dense_5_layer_call_and_return_conditional_losses_195467822!
dense_5/StatefulPartitionedCallЗ
IdentityIdentity(dense_5/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:         2

Identity┤
NoOpNoOp ^dense_5/StatefulPartitionedCall ^lstm_10/StatefulPartitionedCall ^lstm_11/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':         ]: : : : : : : : 2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2B
lstm_10/StatefulPartitionedCalllstm_10/StatefulPartitionedCall2B
lstm_11/StatefulPartitionedCalllstm_11/StatefulPartitionedCall:Z V
+
_output_shapes
:         ]
'
_user_specified_namelstm_10_input
р
║
*__inference_lstm_11_layer_call_fn_19549382
inputs_0
unknown:	GИ
	unknown_0:
вИ
	unknown_1:	И
identityИвStatefulPartitionedCallТ
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:                  в*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_lstm_11_layer_call_and_return_conditional_losses_195461522
StatefulPartitionedCallЙ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:                  в2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  G: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :                  G
"
_user_specified_name
inputs/0
█
ё
(sequential_5_lstm_11_while_cond_19545041F
Bsequential_5_lstm_11_while_sequential_5_lstm_11_while_loop_counterL
Hsequential_5_lstm_11_while_sequential_5_lstm_11_while_maximum_iterations*
&sequential_5_lstm_11_while_placeholder,
(sequential_5_lstm_11_while_placeholder_1,
(sequential_5_lstm_11_while_placeholder_2,
(sequential_5_lstm_11_while_placeholder_3H
Dsequential_5_lstm_11_while_less_sequential_5_lstm_11_strided_slice_1`
\sequential_5_lstm_11_while_sequential_5_lstm_11_while_cond_19545041___redundant_placeholder0`
\sequential_5_lstm_11_while_sequential_5_lstm_11_while_cond_19545041___redundant_placeholder1`
\sequential_5_lstm_11_while_sequential_5_lstm_11_while_cond_19545041___redundant_placeholder2`
\sequential_5_lstm_11_while_sequential_5_lstm_11_while_cond_19545041___redundant_placeholder3'
#sequential_5_lstm_11_while_identity
┘
sequential_5/lstm_11/while/LessLess&sequential_5_lstm_11_while_placeholderDsequential_5_lstm_11_while_less_sequential_5_lstm_11_strided_slice_1*
T0*
_output_shapes
: 2!
sequential_5/lstm_11/while/LessЬ
#sequential_5/lstm_11/while/IdentityIdentity#sequential_5/lstm_11/while/Less:z:0*
T0
*
_output_shapes
: 2%
#sequential_5/lstm_11/while/Identity"S
#sequential_5_lstm_11_while_identity,sequential_5/lstm_11/while/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :         в:         в: ::::: 
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
:         в:.*
(
_output_shapes
:         в:

_output_shapes
: :

_output_shapes
:
У
Й
J__inference_lstm_cell_11_layer_call_and_return_conditional_losses_19549601

inputs
states_0
states_11
matmul_readvariableop_resource:	GИ4
 matmul_1_readvariableop_resource:
вИ.
biasadd_readvariableop_resource:	И
identity

identity_1

identity_2ИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpвMatMul_1/ReadVariableOpО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	GИ*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         И2
MatMulХ
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
вИ*
dtype02
MatMul_1/ReadVariableOp|
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         И2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:         И2
addН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:И*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         И2	
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
P:         в:         в:         в:         в*
	num_split2
split`
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:         в2	
Sigmoidd
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:         в2
	Sigmoid_1]
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:         в2
mulW
ReluRelusplit:output:2*
T0*(
_output_shapes
:         в2
Relui
mul_1MulSigmoid:y:0Relu:activations:0*
T0*(
_output_shapes
:         в2
mul_1^
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:         в2
add_1d
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:         в2
	Sigmoid_2V
Relu_1Relu	add_1:z:0*
T0*(
_output_shapes
:         в2
Relu_1m
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*(
_output_shapes
:         в2
mul_2e
IdentityIdentity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:         в2

Identityi

Identity_1Identity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:         в2

Identity_1i

Identity_2Identity	add_1:z:0^NoOp*
T0*(
_output_shapes
:         в2

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
A:         G:         в:         в: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:         G
 
_user_specified_nameinputs:RN
(
_output_shapes
:         в
"
_user_specified_name
states/0:RN
(
_output_shapes
:         в
"
_user_specified_name
states/1
╨

э
lstm_10_while_cond_19547437,
(lstm_10_while_lstm_10_while_loop_counter2
.lstm_10_while_lstm_10_while_maximum_iterations
lstm_10_while_placeholder
lstm_10_while_placeholder_1
lstm_10_while_placeholder_2
lstm_10_while_placeholder_3.
*lstm_10_while_less_lstm_10_strided_slice_1F
Blstm_10_while_lstm_10_while_cond_19547437___redundant_placeholder0F
Blstm_10_while_lstm_10_while_cond_19547437___redundant_placeholder1F
Blstm_10_while_lstm_10_while_cond_19547437___redundant_placeholder2F
Blstm_10_while_lstm_10_while_cond_19547437___redundant_placeholder3
lstm_10_while_identity
Ш
lstm_10/while/LessLesslstm_10_while_placeholder*lstm_10_while_less_lstm_10_strided_slice_1*
T0*
_output_shapes
: 2
lstm_10/while/Lessu
lstm_10/while/IdentityIdentitylstm_10/while/Less:z:0*
T0
*
_output_shapes
: 2
lstm_10/while/Identity"9
lstm_10_while_identitylstm_10/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :         G:         G: ::::: 
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
:         G:-)
'
_output_shapes
:         G:

_output_shapes
: :

_output_shapes
:
Е&
є
while_body_19545873
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_00
while_lstm_cell_11_19545897_0:	GИ1
while_lstm_cell_11_19545899_0:
вИ,
while_lstm_cell_11_19545901_0:	И
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor.
while_lstm_cell_11_19545897:	GИ/
while_lstm_cell_11_19545899:
вИ*
while_lstm_cell_11_19545901:	ИИв*while/lstm_cell_11/StatefulPartitionedCall├
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    G   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape╙
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         G*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemь
*while/lstm_cell_11/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_11_19545897_0while_lstm_cell_11_19545899_0while_lstm_cell_11_19545901_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:         в:         в:         в*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_lstm_cell_11_layer_call_and_return_conditional_losses_195458592,
*while/lstm_cell_11/StatefulPartitionedCallў
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder3while/lstm_cell_11/StatefulPartitionedCall:output:0*
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
while/Identity_4Identity3while/lstm_cell_11/StatefulPartitionedCall:output:1^while/NoOp*
T0*(
_output_shapes
:         в2
while/Identity_4е
while/Identity_5Identity3while/lstm_cell_11/StatefulPartitionedCall:output:2^while/NoOp*
T0*(
_output_shapes
:         в2
while/Identity_5З

while/NoOpNoOp+^while/lstm_cell_11/StatefulPartitionedCall*"
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
while_lstm_cell_11_19545897while_lstm_cell_11_19545897_0"<
while_lstm_cell_11_19545899while_lstm_cell_11_19545899_0"<
while_lstm_cell_11_19545901while_lstm_cell_11_19545901_0"0
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :         в:         в: : : : : 2X
*while/lstm_cell_11/StatefulPartitionedCall*while/lstm_cell_11/StatefulPartitionedCall: 
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
:         в:.*
(
_output_shapes
:         в:

_output_shapes
: :

_output_shapes
: 
╚7
█
$__inference__traced_restore_19549772
file_prefix2
assignvariableop_dense_5_kernel:	в-
assignvariableop_1_dense_5_bias:A
.assignvariableop_2_lstm_10_lstm_cell_10_kernel:	]ЬK
8assignvariableop_3_lstm_10_lstm_cell_10_recurrent_kernel:	GЬ;
,assignvariableop_4_lstm_10_lstm_cell_10_bias:	ЬA
.assignvariableop_5_lstm_11_lstm_cell_11_kernel:	GИL
8assignvariableop_6_lstm_11_lstm_cell_11_recurrent_kernel:
вИ;
,assignvariableop_7_lstm_11_lstm_cell_11_bias:	И"
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
AssignVariableOpAssignVariableOpassignvariableop_dense_5_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1д
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_5_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2│
AssignVariableOp_2AssignVariableOp.assignvariableop_2_lstm_10_lstm_cell_10_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3╜
AssignVariableOp_3AssignVariableOp8assignvariableop_3_lstm_10_lstm_cell_10_recurrent_kernelIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4▒
AssignVariableOp_4AssignVariableOp,assignvariableop_4_lstm_10_lstm_cell_10_biasIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5│
AssignVariableOp_5AssignVariableOp.assignvariableop_5_lstm_11_lstm_cell_11_kernelIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6╜
AssignVariableOp_6AssignVariableOp8assignvariableop_6_lstm_11_lstm_cell_11_recurrent_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7▒
AssignVariableOp_7AssignVariableOp,assignvariableop_7_lstm_11_lstm_cell_11_biasIdentity_7:output:0"/device:CPU:0*
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
к

╤
/__inference_sequential_5_layer_call_fn_19547298
lstm_10_input
unknown:	]Ь
	unknown_0:	GЬ
	unknown_1:	Ь
	unknown_2:	GИ
	unknown_3:
вИ
	unknown_4:	И
	unknown_5:	в
	unknown_6:
identityИвStatefulPartitionedCall╙
StatefulPartitionedCallStatefulPartitionedCalllstm_10_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
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
J__inference_sequential_5_layer_call_and_return_conditional_losses_195472582
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
_user_specified_namelstm_10_input
р
║
*__inference_lstm_11_layer_call_fn_19549371
inputs_0
unknown:	GИ
	unknown_0:
вИ
	unknown_1:	И
identityИвStatefulPartitionedCallТ
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:                  в*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_lstm_11_layer_call_and_return_conditional_losses_195459422
StatefulPartitionedCallЙ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:                  в2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  G: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :                  G
"
_user_specified_name
inputs/0
╣F
Н
E__inference_lstm_10_layer_call_and_return_conditional_losses_19545522

inputs(
lstm_cell_10_19545440:	]Ь(
lstm_cell_10_19545442:	GЬ$
lstm_cell_10_19545444:	Ь
identityИв$lstm_cell_10/StatefulPartitionedCallвwhileD
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
value	B :G2
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
value	B :G2
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
:         G2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :G2
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
value	B :G2
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
:         G2	
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
$lstm_cell_10/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_10_19545440lstm_cell_10_19545442lstm_cell_10_19545444*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:         G:         G:         G*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_lstm_cell_10_layer_call_and_return_conditional_losses_195453752&
$lstm_cell_10/StatefulPartitionedCallП
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    G   2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_10_19545440lstm_cell_10_19545442lstm_cell_10_19545444*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         G:         G: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_19545453*
condR
while_cond_19545452*K
output_shapes:
8: : : : :         G:         G: : : : : *
parallel_iterations 2
while╡
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    G   22
0TensorArrayV2Stack/TensorListStack/element_shapeё
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :                  G*
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
:         G*
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
 :                  G2
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
 :                  G2

Identity}
NoOpNoOp%^lstm_cell_10/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  ]: : : 2L
$lstm_cell_10/StatefulPartitionedCall$lstm_cell_10/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :                  ]
 
_user_specified_nameinputs
░?
╘
while_body_19548823
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_11_matmul_readvariableop_resource_0:	GИI
5while_lstm_cell_11_matmul_1_readvariableop_resource_0:
вИC
4while_lstm_cell_11_biasadd_readvariableop_resource_0:	И
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_11_matmul_readvariableop_resource:	GИG
3while_lstm_cell_11_matmul_1_readvariableop_resource:
вИA
2while_lstm_cell_11_biasadd_readvariableop_resource:	ИИв)while/lstm_cell_11/BiasAdd/ReadVariableOpв(while/lstm_cell_11/MatMul/ReadVariableOpв*while/lstm_cell_11/MatMul_1/ReadVariableOp├
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    G   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape╙
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         G*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem╔
(while/lstm_cell_11/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_11_matmul_readvariableop_resource_0*
_output_shapes
:	GИ*
dtype02*
(while/lstm_cell_11/MatMul/ReadVariableOp╫
while/lstm_cell_11/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_11/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         И2
while/lstm_cell_11/MatMul╨
*while/lstm_cell_11/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_11_matmul_1_readvariableop_resource_0* 
_output_shapes
:
вИ*
dtype02,
*while/lstm_cell_11/MatMul_1/ReadVariableOp└
while/lstm_cell_11/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_11/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         И2
while/lstm_cell_11/MatMul_1╕
while/lstm_cell_11/addAddV2#while/lstm_cell_11/MatMul:product:0%while/lstm_cell_11/MatMul_1:product:0*
T0*(
_output_shapes
:         И2
while/lstm_cell_11/add╚
)while/lstm_cell_11/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_11_biasadd_readvariableop_resource_0*
_output_shapes	
:И*
dtype02+
)while/lstm_cell_11/BiasAdd/ReadVariableOp┼
while/lstm_cell_11/BiasAddBiasAddwhile/lstm_cell_11/add:z:01while/lstm_cell_11/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         И2
while/lstm_cell_11/BiasAddК
"while/lstm_cell_11/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_11/split/split_dimП
while/lstm_cell_11/splitSplit+while/lstm_cell_11/split/split_dim:output:0#while/lstm_cell_11/BiasAdd:output:0*
T0*d
_output_shapesR
P:         в:         в:         в:         в*
	num_split2
while/lstm_cell_11/splitЩ
while/lstm_cell_11/SigmoidSigmoid!while/lstm_cell_11/split:output:0*
T0*(
_output_shapes
:         в2
while/lstm_cell_11/SigmoidЭ
while/lstm_cell_11/Sigmoid_1Sigmoid!while/lstm_cell_11/split:output:1*
T0*(
_output_shapes
:         в2
while/lstm_cell_11/Sigmoid_1б
while/lstm_cell_11/mulMul while/lstm_cell_11/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:         в2
while/lstm_cell_11/mulР
while/lstm_cell_11/ReluRelu!while/lstm_cell_11/split:output:2*
T0*(
_output_shapes
:         в2
while/lstm_cell_11/Relu╡
while/lstm_cell_11/mul_1Mulwhile/lstm_cell_11/Sigmoid:y:0%while/lstm_cell_11/Relu:activations:0*
T0*(
_output_shapes
:         в2
while/lstm_cell_11/mul_1к
while/lstm_cell_11/add_1AddV2while/lstm_cell_11/mul:z:0while/lstm_cell_11/mul_1:z:0*
T0*(
_output_shapes
:         в2
while/lstm_cell_11/add_1Э
while/lstm_cell_11/Sigmoid_2Sigmoid!while/lstm_cell_11/split:output:3*
T0*(
_output_shapes
:         в2
while/lstm_cell_11/Sigmoid_2П
while/lstm_cell_11/Relu_1Reluwhile/lstm_cell_11/add_1:z:0*
T0*(
_output_shapes
:         в2
while/lstm_cell_11/Relu_1╣
while/lstm_cell_11/mul_2Mul while/lstm_cell_11/Sigmoid_2:y:0'while/lstm_cell_11/Relu_1:activations:0*
T0*(
_output_shapes
:         в2
while/lstm_cell_11/mul_2р
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_11/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_11/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:         в2
while/Identity_4О
while/Identity_5Identitywhile/lstm_cell_11/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:         в2
while/Identity_5▐

while/NoOpNoOp*^while/lstm_cell_11/BiasAdd/ReadVariableOp)^while/lstm_cell_11/MatMul/ReadVariableOp+^while/lstm_cell_11/MatMul_1/ReadVariableOp*"
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
2while_lstm_cell_11_biasadd_readvariableop_resource4while_lstm_cell_11_biasadd_readvariableop_resource_0"l
3while_lstm_cell_11_matmul_1_readvariableop_resource5while_lstm_cell_11_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_11_matmul_readvariableop_resource3while_lstm_cell_11_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :         в:         в: : : : : 2V
)while/lstm_cell_11/BiasAdd/ReadVariableOp)while/lstm_cell_11/BiasAdd/ReadVariableOp2T
(while/lstm_cell_11/MatMul/ReadVariableOp(while/lstm_cell_11/MatMul/ReadVariableOp2X
*while/lstm_cell_11/MatMul_1/ReadVariableOp*while/lstm_cell_11/MatMul_1/ReadVariableOp: 
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
:         в:.*
(
_output_shapes
:         в:

_output_shapes
: :

_output_shapes
: 
у
═
while_cond_19548822
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_19548822___redundant_placeholder06
2while_while_cond_19548822___redundant_placeholder16
2while_while_cond_19548822___redundant_placeholder26
2while_while_cond_19548822___redundant_placeholder3
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
B: : : : :         в:         в: ::::: 
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
:         в:.*
(
_output_shapes
:         в:

_output_shapes
: :

_output_shapes
:
Х

╩
/__inference_sequential_5_layer_call_fn_19548060

inputs
unknown:	]Ь
	unknown_0:	GЬ
	unknown_1:	Ь
	unknown_2:	GИ
	unknown_3:
вИ
	unknown_4:	И
	unknown_5:	в
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
J__inference_sequential_5_layer_call_and_return_conditional_losses_195467892
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
╫
g
H__inference_dropout_11_layer_call_and_return_conditional_losses_19549421

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
:         в2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape╣
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:         в*
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
:         в2
dropout/GreaterEqualД
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:         в2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:         в2
dropout/Mul_1j
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:         в2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         в:T P
,
_output_shapes
:         в
 
_user_specified_nameinputs
ч[
Э
E__inference_lstm_10_layer_call_and_return_conditional_losses_19548685

inputs>
+lstm_cell_10_matmul_readvariableop_resource:	]Ь@
-lstm_cell_10_matmul_1_readvariableop_resource:	GЬ;
,lstm_cell_10_biasadd_readvariableop_resource:	Ь
identityИв#lstm_cell_10/BiasAdd/ReadVariableOpв"lstm_cell_10/MatMul/ReadVariableOpв$lstm_cell_10/MatMul_1/ReadVariableOpвwhileD
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
value	B :G2
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
value	B :G2
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
:         G2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :G2
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
value	B :G2
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
:         G2	
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
"lstm_cell_10/MatMul/ReadVariableOpReadVariableOp+lstm_cell_10_matmul_readvariableop_resource*
_output_shapes
:	]Ь*
dtype02$
"lstm_cell_10/MatMul/ReadVariableOpн
lstm_cell_10/MatMulMatMulstrided_slice_2:output:0*lstm_cell_10/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ь2
lstm_cell_10/MatMul╗
$lstm_cell_10/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_10_matmul_1_readvariableop_resource*
_output_shapes
:	GЬ*
dtype02&
$lstm_cell_10/MatMul_1/ReadVariableOpй
lstm_cell_10/MatMul_1MatMulzeros:output:0,lstm_cell_10/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ь2
lstm_cell_10/MatMul_1а
lstm_cell_10/addAddV2lstm_cell_10/MatMul:product:0lstm_cell_10/MatMul_1:product:0*
T0*(
_output_shapes
:         Ь2
lstm_cell_10/add┤
#lstm_cell_10/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_10_biasadd_readvariableop_resource*
_output_shapes	
:Ь*
dtype02%
#lstm_cell_10/BiasAdd/ReadVariableOpн
lstm_cell_10/BiasAddBiasAddlstm_cell_10/add:z:0+lstm_cell_10/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ь2
lstm_cell_10/BiasAdd~
lstm_cell_10/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_10/split/split_dimє
lstm_cell_10/splitSplit%lstm_cell_10/split/split_dim:output:0lstm_cell_10/BiasAdd:output:0*
T0*`
_output_shapesN
L:         G:         G:         G:         G*
	num_split2
lstm_cell_10/splitЖ
lstm_cell_10/SigmoidSigmoidlstm_cell_10/split:output:0*
T0*'
_output_shapes
:         G2
lstm_cell_10/SigmoidК
lstm_cell_10/Sigmoid_1Sigmoidlstm_cell_10/split:output:1*
T0*'
_output_shapes
:         G2
lstm_cell_10/Sigmoid_1Л
lstm_cell_10/mulMullstm_cell_10/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:         G2
lstm_cell_10/mul}
lstm_cell_10/ReluRelulstm_cell_10/split:output:2*
T0*'
_output_shapes
:         G2
lstm_cell_10/ReluЬ
lstm_cell_10/mul_1Mullstm_cell_10/Sigmoid:y:0lstm_cell_10/Relu:activations:0*
T0*'
_output_shapes
:         G2
lstm_cell_10/mul_1С
lstm_cell_10/add_1AddV2lstm_cell_10/mul:z:0lstm_cell_10/mul_1:z:0*
T0*'
_output_shapes
:         G2
lstm_cell_10/add_1К
lstm_cell_10/Sigmoid_2Sigmoidlstm_cell_10/split:output:3*
T0*'
_output_shapes
:         G2
lstm_cell_10/Sigmoid_2|
lstm_cell_10/Relu_1Relulstm_cell_10/add_1:z:0*
T0*'
_output_shapes
:         G2
lstm_cell_10/Relu_1а
lstm_cell_10/mul_2Mullstm_cell_10/Sigmoid_2:y:0!lstm_cell_10/Relu_1:activations:0*
T0*'
_output_shapes
:         G2
lstm_cell_10/mul_2П
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    G   2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_10_matmul_readvariableop_resource-lstm_cell_10_matmul_1_readvariableop_resource,lstm_cell_10_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         G:         G: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_19548601*
condR
while_cond_19548600*K
output_shapes:
8: : : : :         G:         G: : : : : *
parallel_iterations 2
while╡
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    G   22
0TensorArrayV2Stack/TensorListStack/element_shapeш
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:         G*
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
:         G*
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
:         G2
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
:         G2

Identity╚
NoOpNoOp$^lstm_cell_10/BiasAdd/ReadVariableOp#^lstm_cell_10/MatMul/ReadVariableOp%^lstm_cell_10/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         ]: : : 2J
#lstm_cell_10/BiasAdd/ReadVariableOp#lstm_cell_10/BiasAdd/ReadVariableOp2H
"lstm_cell_10/MatMul/ReadVariableOp"lstm_cell_10/MatMul/ReadVariableOp2L
$lstm_cell_10/MatMul_1/ReadVariableOp$lstm_cell_10/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:         ]
 
_user_specified_nameinputs
▓
╖
*__inference_lstm_10_layer_call_fn_19548718

inputs
unknown:	]Ь
	unknown_0:	GЬ
	unknown_1:	Ь
identityИвStatefulPartitionedCallЖ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         G*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_lstm_10_layer_call_and_return_conditional_losses_195465712
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:         G2

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

lstm_11_while_body_19547586,
(lstm_11_while_lstm_11_while_loop_counter2
.lstm_11_while_lstm_11_while_maximum_iterations
lstm_11_while_placeholder
lstm_11_while_placeholder_1
lstm_11_while_placeholder_2
lstm_11_while_placeholder_3+
'lstm_11_while_lstm_11_strided_slice_1_0g
clstm_11_while_tensorarrayv2read_tensorlistgetitem_lstm_11_tensorarrayunstack_tensorlistfromtensor_0N
;lstm_11_while_lstm_cell_11_matmul_readvariableop_resource_0:	GИQ
=lstm_11_while_lstm_cell_11_matmul_1_readvariableop_resource_0:
вИK
<lstm_11_while_lstm_cell_11_biasadd_readvariableop_resource_0:	И
lstm_11_while_identity
lstm_11_while_identity_1
lstm_11_while_identity_2
lstm_11_while_identity_3
lstm_11_while_identity_4
lstm_11_while_identity_5)
%lstm_11_while_lstm_11_strided_slice_1e
alstm_11_while_tensorarrayv2read_tensorlistgetitem_lstm_11_tensorarrayunstack_tensorlistfromtensorL
9lstm_11_while_lstm_cell_11_matmul_readvariableop_resource:	GИO
;lstm_11_while_lstm_cell_11_matmul_1_readvariableop_resource:
вИI
:lstm_11_while_lstm_cell_11_biasadd_readvariableop_resource:	ИИв1lstm_11/while/lstm_cell_11/BiasAdd/ReadVariableOpв0lstm_11/while/lstm_cell_11/MatMul/ReadVariableOpв2lstm_11/while/lstm_cell_11/MatMul_1/ReadVariableOp╙
?lstm_11/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    G   2A
?lstm_11/while/TensorArrayV2Read/TensorListGetItem/element_shapeГ
1lstm_11/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemclstm_11_while_tensorarrayv2read_tensorlistgetitem_lstm_11_tensorarrayunstack_tensorlistfromtensor_0lstm_11_while_placeholderHlstm_11/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         G*
element_dtype023
1lstm_11/while/TensorArrayV2Read/TensorListGetItemс
0lstm_11/while/lstm_cell_11/MatMul/ReadVariableOpReadVariableOp;lstm_11_while_lstm_cell_11_matmul_readvariableop_resource_0*
_output_shapes
:	GИ*
dtype022
0lstm_11/while/lstm_cell_11/MatMul/ReadVariableOpў
!lstm_11/while/lstm_cell_11/MatMulMatMul8lstm_11/while/TensorArrayV2Read/TensorListGetItem:item:08lstm_11/while/lstm_cell_11/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         И2#
!lstm_11/while/lstm_cell_11/MatMulш
2lstm_11/while/lstm_cell_11/MatMul_1/ReadVariableOpReadVariableOp=lstm_11_while_lstm_cell_11_matmul_1_readvariableop_resource_0* 
_output_shapes
:
вИ*
dtype024
2lstm_11/while/lstm_cell_11/MatMul_1/ReadVariableOpр
#lstm_11/while/lstm_cell_11/MatMul_1MatMullstm_11_while_placeholder_2:lstm_11/while/lstm_cell_11/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         И2%
#lstm_11/while/lstm_cell_11/MatMul_1╪
lstm_11/while/lstm_cell_11/addAddV2+lstm_11/while/lstm_cell_11/MatMul:product:0-lstm_11/while/lstm_cell_11/MatMul_1:product:0*
T0*(
_output_shapes
:         И2 
lstm_11/while/lstm_cell_11/addр
1lstm_11/while/lstm_cell_11/BiasAdd/ReadVariableOpReadVariableOp<lstm_11_while_lstm_cell_11_biasadd_readvariableop_resource_0*
_output_shapes	
:И*
dtype023
1lstm_11/while/lstm_cell_11/BiasAdd/ReadVariableOpх
"lstm_11/while/lstm_cell_11/BiasAddBiasAdd"lstm_11/while/lstm_cell_11/add:z:09lstm_11/while/lstm_cell_11/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         И2$
"lstm_11/while/lstm_cell_11/BiasAddЪ
*lstm_11/while/lstm_cell_11/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2,
*lstm_11/while/lstm_cell_11/split/split_dimп
 lstm_11/while/lstm_cell_11/splitSplit3lstm_11/while/lstm_cell_11/split/split_dim:output:0+lstm_11/while/lstm_cell_11/BiasAdd:output:0*
T0*d
_output_shapesR
P:         в:         в:         в:         в*
	num_split2"
 lstm_11/while/lstm_cell_11/split▒
"lstm_11/while/lstm_cell_11/SigmoidSigmoid)lstm_11/while/lstm_cell_11/split:output:0*
T0*(
_output_shapes
:         в2$
"lstm_11/while/lstm_cell_11/Sigmoid╡
$lstm_11/while/lstm_cell_11/Sigmoid_1Sigmoid)lstm_11/while/lstm_cell_11/split:output:1*
T0*(
_output_shapes
:         в2&
$lstm_11/while/lstm_cell_11/Sigmoid_1┴
lstm_11/while/lstm_cell_11/mulMul(lstm_11/while/lstm_cell_11/Sigmoid_1:y:0lstm_11_while_placeholder_3*
T0*(
_output_shapes
:         в2 
lstm_11/while/lstm_cell_11/mulи
lstm_11/while/lstm_cell_11/ReluRelu)lstm_11/while/lstm_cell_11/split:output:2*
T0*(
_output_shapes
:         в2!
lstm_11/while/lstm_cell_11/Relu╒
 lstm_11/while/lstm_cell_11/mul_1Mul&lstm_11/while/lstm_cell_11/Sigmoid:y:0-lstm_11/while/lstm_cell_11/Relu:activations:0*
T0*(
_output_shapes
:         в2"
 lstm_11/while/lstm_cell_11/mul_1╩
 lstm_11/while/lstm_cell_11/add_1AddV2"lstm_11/while/lstm_cell_11/mul:z:0$lstm_11/while/lstm_cell_11/mul_1:z:0*
T0*(
_output_shapes
:         в2"
 lstm_11/while/lstm_cell_11/add_1╡
$lstm_11/while/lstm_cell_11/Sigmoid_2Sigmoid)lstm_11/while/lstm_cell_11/split:output:3*
T0*(
_output_shapes
:         в2&
$lstm_11/while/lstm_cell_11/Sigmoid_2з
!lstm_11/while/lstm_cell_11/Relu_1Relu$lstm_11/while/lstm_cell_11/add_1:z:0*
T0*(
_output_shapes
:         в2#
!lstm_11/while/lstm_cell_11/Relu_1┘
 lstm_11/while/lstm_cell_11/mul_2Mul(lstm_11/while/lstm_cell_11/Sigmoid_2:y:0/lstm_11/while/lstm_cell_11/Relu_1:activations:0*
T0*(
_output_shapes
:         в2"
 lstm_11/while/lstm_cell_11/mul_2И
2lstm_11/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_11_while_placeholder_1lstm_11_while_placeholder$lstm_11/while/lstm_cell_11/mul_2:z:0*
_output_shapes
: *
element_dtype024
2lstm_11/while/TensorArrayV2Write/TensorListSetIteml
lstm_11/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_11/while/add/yЙ
lstm_11/while/addAddV2lstm_11_while_placeholderlstm_11/while/add/y:output:0*
T0*
_output_shapes
: 2
lstm_11/while/addp
lstm_11/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_11/while/add_1/yЮ
lstm_11/while/add_1AddV2(lstm_11_while_lstm_11_while_loop_counterlstm_11/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lstm_11/while/add_1Л
lstm_11/while/IdentityIdentitylstm_11/while/add_1:z:0^lstm_11/while/NoOp*
T0*
_output_shapes
: 2
lstm_11/while/Identityж
lstm_11/while/Identity_1Identity.lstm_11_while_lstm_11_while_maximum_iterations^lstm_11/while/NoOp*
T0*
_output_shapes
: 2
lstm_11/while/Identity_1Н
lstm_11/while/Identity_2Identitylstm_11/while/add:z:0^lstm_11/while/NoOp*
T0*
_output_shapes
: 2
lstm_11/while/Identity_2║
lstm_11/while/Identity_3IdentityBlstm_11/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_11/while/NoOp*
T0*
_output_shapes
: 2
lstm_11/while/Identity_3о
lstm_11/while/Identity_4Identity$lstm_11/while/lstm_cell_11/mul_2:z:0^lstm_11/while/NoOp*
T0*(
_output_shapes
:         в2
lstm_11/while/Identity_4о
lstm_11/while/Identity_5Identity$lstm_11/while/lstm_cell_11/add_1:z:0^lstm_11/while/NoOp*
T0*(
_output_shapes
:         в2
lstm_11/while/Identity_5Ж
lstm_11/while/NoOpNoOp2^lstm_11/while/lstm_cell_11/BiasAdd/ReadVariableOp1^lstm_11/while/lstm_cell_11/MatMul/ReadVariableOp3^lstm_11/while/lstm_cell_11/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
lstm_11/while/NoOp"9
lstm_11_while_identitylstm_11/while/Identity:output:0"=
lstm_11_while_identity_1!lstm_11/while/Identity_1:output:0"=
lstm_11_while_identity_2!lstm_11/while/Identity_2:output:0"=
lstm_11_while_identity_3!lstm_11/while/Identity_3:output:0"=
lstm_11_while_identity_4!lstm_11/while/Identity_4:output:0"=
lstm_11_while_identity_5!lstm_11/while/Identity_5:output:0"P
%lstm_11_while_lstm_11_strided_slice_1'lstm_11_while_lstm_11_strided_slice_1_0"z
:lstm_11_while_lstm_cell_11_biasadd_readvariableop_resource<lstm_11_while_lstm_cell_11_biasadd_readvariableop_resource_0"|
;lstm_11_while_lstm_cell_11_matmul_1_readvariableop_resource=lstm_11_while_lstm_cell_11_matmul_1_readvariableop_resource_0"x
9lstm_11_while_lstm_cell_11_matmul_readvariableop_resource;lstm_11_while_lstm_cell_11_matmul_readvariableop_resource_0"╚
alstm_11_while_tensorarrayv2read_tensorlistgetitem_lstm_11_tensorarrayunstack_tensorlistfromtensorclstm_11_while_tensorarrayv2read_tensorlistgetitem_lstm_11_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :         в:         в: : : : : 2f
1lstm_11/while/lstm_cell_11/BiasAdd/ReadVariableOp1lstm_11/while/lstm_cell_11/BiasAdd/ReadVariableOp2d
0lstm_11/while/lstm_cell_11/MatMul/ReadVariableOp0lstm_11/while/lstm_cell_11/MatMul/ReadVariableOp2h
2lstm_11/while/lstm_cell_11/MatMul_1/ReadVariableOp2lstm_11/while/lstm_cell_11/MatMul_1/ReadVariableOp: 
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
:         в:.*
(
_output_shapes
:         в:

_output_shapes
: :

_output_shapes
: 
ч[
Э
E__inference_lstm_10_layer_call_and_return_conditional_losses_19548534

inputs>
+lstm_cell_10_matmul_readvariableop_resource:	]Ь@
-lstm_cell_10_matmul_1_readvariableop_resource:	GЬ;
,lstm_cell_10_biasadd_readvariableop_resource:	Ь
identityИв#lstm_cell_10/BiasAdd/ReadVariableOpв"lstm_cell_10/MatMul/ReadVariableOpв$lstm_cell_10/MatMul_1/ReadVariableOpвwhileD
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
value	B :G2
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
value	B :G2
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
:         G2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :G2
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
value	B :G2
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
:         G2	
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
"lstm_cell_10/MatMul/ReadVariableOpReadVariableOp+lstm_cell_10_matmul_readvariableop_resource*
_output_shapes
:	]Ь*
dtype02$
"lstm_cell_10/MatMul/ReadVariableOpн
lstm_cell_10/MatMulMatMulstrided_slice_2:output:0*lstm_cell_10/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ь2
lstm_cell_10/MatMul╗
$lstm_cell_10/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_10_matmul_1_readvariableop_resource*
_output_shapes
:	GЬ*
dtype02&
$lstm_cell_10/MatMul_1/ReadVariableOpй
lstm_cell_10/MatMul_1MatMulzeros:output:0,lstm_cell_10/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ь2
lstm_cell_10/MatMul_1а
lstm_cell_10/addAddV2lstm_cell_10/MatMul:product:0lstm_cell_10/MatMul_1:product:0*
T0*(
_output_shapes
:         Ь2
lstm_cell_10/add┤
#lstm_cell_10/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_10_biasadd_readvariableop_resource*
_output_shapes	
:Ь*
dtype02%
#lstm_cell_10/BiasAdd/ReadVariableOpн
lstm_cell_10/BiasAddBiasAddlstm_cell_10/add:z:0+lstm_cell_10/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ь2
lstm_cell_10/BiasAdd~
lstm_cell_10/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_10/split/split_dimє
lstm_cell_10/splitSplit%lstm_cell_10/split/split_dim:output:0lstm_cell_10/BiasAdd:output:0*
T0*`
_output_shapesN
L:         G:         G:         G:         G*
	num_split2
lstm_cell_10/splitЖ
lstm_cell_10/SigmoidSigmoidlstm_cell_10/split:output:0*
T0*'
_output_shapes
:         G2
lstm_cell_10/SigmoidК
lstm_cell_10/Sigmoid_1Sigmoidlstm_cell_10/split:output:1*
T0*'
_output_shapes
:         G2
lstm_cell_10/Sigmoid_1Л
lstm_cell_10/mulMullstm_cell_10/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:         G2
lstm_cell_10/mul}
lstm_cell_10/ReluRelulstm_cell_10/split:output:2*
T0*'
_output_shapes
:         G2
lstm_cell_10/ReluЬ
lstm_cell_10/mul_1Mullstm_cell_10/Sigmoid:y:0lstm_cell_10/Relu:activations:0*
T0*'
_output_shapes
:         G2
lstm_cell_10/mul_1С
lstm_cell_10/add_1AddV2lstm_cell_10/mul:z:0lstm_cell_10/mul_1:z:0*
T0*'
_output_shapes
:         G2
lstm_cell_10/add_1К
lstm_cell_10/Sigmoid_2Sigmoidlstm_cell_10/split:output:3*
T0*'
_output_shapes
:         G2
lstm_cell_10/Sigmoid_2|
lstm_cell_10/Relu_1Relulstm_cell_10/add_1:z:0*
T0*'
_output_shapes
:         G2
lstm_cell_10/Relu_1а
lstm_cell_10/mul_2Mullstm_cell_10/Sigmoid_2:y:0!lstm_cell_10/Relu_1:activations:0*
T0*'
_output_shapes
:         G2
lstm_cell_10/mul_2П
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    G   2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_10_matmul_readvariableop_resource-lstm_cell_10_matmul_1_readvariableop_resource,lstm_cell_10_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         G:         G: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_19548450*
condR
while_cond_19548449*K
output_shapes:
8: : : : :         G:         G: : : : : *
parallel_iterations 2
while╡
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    G   22
0TensorArrayV2Stack/TensorListStack/element_shapeш
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:         G*
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
:         G*
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
:         G2
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
:         G2

Identity╚
NoOpNoOp$^lstm_cell_10/BiasAdd/ReadVariableOp#^lstm_cell_10/MatMul/ReadVariableOp%^lstm_cell_10/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         ]: : : 2J
#lstm_cell_10/BiasAdd/ReadVariableOp#lstm_cell_10/BiasAdd/ReadVariableOp2H
"lstm_cell_10/MatMul/ReadVariableOp"lstm_cell_10/MatMul/ReadVariableOp2L
$lstm_cell_10/MatMul_1/ReadVariableOp$lstm_cell_10/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:         ]
 
_user_specified_nameinputs
░?
╘
while_body_19546652
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_11_matmul_readvariableop_resource_0:	GИI
5while_lstm_cell_11_matmul_1_readvariableop_resource_0:
вИC
4while_lstm_cell_11_biasadd_readvariableop_resource_0:	И
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_11_matmul_readvariableop_resource:	GИG
3while_lstm_cell_11_matmul_1_readvariableop_resource:
вИA
2while_lstm_cell_11_biasadd_readvariableop_resource:	ИИв)while/lstm_cell_11/BiasAdd/ReadVariableOpв(while/lstm_cell_11/MatMul/ReadVariableOpв*while/lstm_cell_11/MatMul_1/ReadVariableOp├
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    G   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape╙
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         G*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem╔
(while/lstm_cell_11/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_11_matmul_readvariableop_resource_0*
_output_shapes
:	GИ*
dtype02*
(while/lstm_cell_11/MatMul/ReadVariableOp╫
while/lstm_cell_11/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_11/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         И2
while/lstm_cell_11/MatMul╨
*while/lstm_cell_11/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_11_matmul_1_readvariableop_resource_0* 
_output_shapes
:
вИ*
dtype02,
*while/lstm_cell_11/MatMul_1/ReadVariableOp└
while/lstm_cell_11/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_11/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         И2
while/lstm_cell_11/MatMul_1╕
while/lstm_cell_11/addAddV2#while/lstm_cell_11/MatMul:product:0%while/lstm_cell_11/MatMul_1:product:0*
T0*(
_output_shapes
:         И2
while/lstm_cell_11/add╚
)while/lstm_cell_11/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_11_biasadd_readvariableop_resource_0*
_output_shapes	
:И*
dtype02+
)while/lstm_cell_11/BiasAdd/ReadVariableOp┼
while/lstm_cell_11/BiasAddBiasAddwhile/lstm_cell_11/add:z:01while/lstm_cell_11/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         И2
while/lstm_cell_11/BiasAddК
"while/lstm_cell_11/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_11/split/split_dimП
while/lstm_cell_11/splitSplit+while/lstm_cell_11/split/split_dim:output:0#while/lstm_cell_11/BiasAdd:output:0*
T0*d
_output_shapesR
P:         в:         в:         в:         в*
	num_split2
while/lstm_cell_11/splitЩ
while/lstm_cell_11/SigmoidSigmoid!while/lstm_cell_11/split:output:0*
T0*(
_output_shapes
:         в2
while/lstm_cell_11/SigmoidЭ
while/lstm_cell_11/Sigmoid_1Sigmoid!while/lstm_cell_11/split:output:1*
T0*(
_output_shapes
:         в2
while/lstm_cell_11/Sigmoid_1б
while/lstm_cell_11/mulMul while/lstm_cell_11/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:         в2
while/lstm_cell_11/mulР
while/lstm_cell_11/ReluRelu!while/lstm_cell_11/split:output:2*
T0*(
_output_shapes
:         в2
while/lstm_cell_11/Relu╡
while/lstm_cell_11/mul_1Mulwhile/lstm_cell_11/Sigmoid:y:0%while/lstm_cell_11/Relu:activations:0*
T0*(
_output_shapes
:         в2
while/lstm_cell_11/mul_1к
while/lstm_cell_11/add_1AddV2while/lstm_cell_11/mul:z:0while/lstm_cell_11/mul_1:z:0*
T0*(
_output_shapes
:         в2
while/lstm_cell_11/add_1Э
while/lstm_cell_11/Sigmoid_2Sigmoid!while/lstm_cell_11/split:output:3*
T0*(
_output_shapes
:         в2
while/lstm_cell_11/Sigmoid_2П
while/lstm_cell_11/Relu_1Reluwhile/lstm_cell_11/add_1:z:0*
T0*(
_output_shapes
:         в2
while/lstm_cell_11/Relu_1╣
while/lstm_cell_11/mul_2Mul while/lstm_cell_11/Sigmoid_2:y:0'while/lstm_cell_11/Relu_1:activations:0*
T0*(
_output_shapes
:         в2
while/lstm_cell_11/mul_2р
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_11/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_11/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:         в2
while/Identity_4О
while/Identity_5Identitywhile/lstm_cell_11/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:         в2
while/Identity_5▐

while/NoOpNoOp*^while/lstm_cell_11/BiasAdd/ReadVariableOp)^while/lstm_cell_11/MatMul/ReadVariableOp+^while/lstm_cell_11/MatMul_1/ReadVariableOp*"
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
2while_lstm_cell_11_biasadd_readvariableop_resource4while_lstm_cell_11_biasadd_readvariableop_resource_0"l
3while_lstm_cell_11_matmul_1_readvariableop_resource5while_lstm_cell_11_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_11_matmul_readvariableop_resource3while_lstm_cell_11_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :         в:         в: : : : : 2V
)while/lstm_cell_11/BiasAdd/ReadVariableOp)while/lstm_cell_11/BiasAdd/ReadVariableOp2T
(while/lstm_cell_11/MatMul/ReadVariableOp(while/lstm_cell_11/MatMul/ReadVariableOp2X
*while/lstm_cell_11/MatMul_1/ReadVariableOp*while/lstm_cell_11/MatMul_1/ReadVariableOp: 
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
:         в:.*
(
_output_shapes
:         в:

_output_shapes
: :

_output_shapes
: 
░?
╘
while_body_19546921
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_11_matmul_readvariableop_resource_0:	GИI
5while_lstm_cell_11_matmul_1_readvariableop_resource_0:
вИC
4while_lstm_cell_11_biasadd_readvariableop_resource_0:	И
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_11_matmul_readvariableop_resource:	GИG
3while_lstm_cell_11_matmul_1_readvariableop_resource:
вИA
2while_lstm_cell_11_biasadd_readvariableop_resource:	ИИв)while/lstm_cell_11/BiasAdd/ReadVariableOpв(while/lstm_cell_11/MatMul/ReadVariableOpв*while/lstm_cell_11/MatMul_1/ReadVariableOp├
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    G   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape╙
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         G*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem╔
(while/lstm_cell_11/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_11_matmul_readvariableop_resource_0*
_output_shapes
:	GИ*
dtype02*
(while/lstm_cell_11/MatMul/ReadVariableOp╫
while/lstm_cell_11/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_11/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         И2
while/lstm_cell_11/MatMul╨
*while/lstm_cell_11/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_11_matmul_1_readvariableop_resource_0* 
_output_shapes
:
вИ*
dtype02,
*while/lstm_cell_11/MatMul_1/ReadVariableOp└
while/lstm_cell_11/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_11/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         И2
while/lstm_cell_11/MatMul_1╕
while/lstm_cell_11/addAddV2#while/lstm_cell_11/MatMul:product:0%while/lstm_cell_11/MatMul_1:product:0*
T0*(
_output_shapes
:         И2
while/lstm_cell_11/add╚
)while/lstm_cell_11/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_11_biasadd_readvariableop_resource_0*
_output_shapes	
:И*
dtype02+
)while/lstm_cell_11/BiasAdd/ReadVariableOp┼
while/lstm_cell_11/BiasAddBiasAddwhile/lstm_cell_11/add:z:01while/lstm_cell_11/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         И2
while/lstm_cell_11/BiasAddК
"while/lstm_cell_11/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_11/split/split_dimП
while/lstm_cell_11/splitSplit+while/lstm_cell_11/split/split_dim:output:0#while/lstm_cell_11/BiasAdd:output:0*
T0*d
_output_shapesR
P:         в:         в:         в:         в*
	num_split2
while/lstm_cell_11/splitЩ
while/lstm_cell_11/SigmoidSigmoid!while/lstm_cell_11/split:output:0*
T0*(
_output_shapes
:         в2
while/lstm_cell_11/SigmoidЭ
while/lstm_cell_11/Sigmoid_1Sigmoid!while/lstm_cell_11/split:output:1*
T0*(
_output_shapes
:         в2
while/lstm_cell_11/Sigmoid_1б
while/lstm_cell_11/mulMul while/lstm_cell_11/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:         в2
while/lstm_cell_11/mulР
while/lstm_cell_11/ReluRelu!while/lstm_cell_11/split:output:2*
T0*(
_output_shapes
:         в2
while/lstm_cell_11/Relu╡
while/lstm_cell_11/mul_1Mulwhile/lstm_cell_11/Sigmoid:y:0%while/lstm_cell_11/Relu:activations:0*
T0*(
_output_shapes
:         в2
while/lstm_cell_11/mul_1к
while/lstm_cell_11/add_1AddV2while/lstm_cell_11/mul:z:0while/lstm_cell_11/mul_1:z:0*
T0*(
_output_shapes
:         в2
while/lstm_cell_11/add_1Э
while/lstm_cell_11/Sigmoid_2Sigmoid!while/lstm_cell_11/split:output:3*
T0*(
_output_shapes
:         в2
while/lstm_cell_11/Sigmoid_2П
while/lstm_cell_11/Relu_1Reluwhile/lstm_cell_11/add_1:z:0*
T0*(
_output_shapes
:         в2
while/lstm_cell_11/Relu_1╣
while/lstm_cell_11/mul_2Mul while/lstm_cell_11/Sigmoid_2:y:0'while/lstm_cell_11/Relu_1:activations:0*
T0*(
_output_shapes
:         в2
while/lstm_cell_11/mul_2р
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_11/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_11/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:         в2
while/Identity_4О
while/Identity_5Identitywhile/lstm_cell_11/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:         в2
while/Identity_5▐

while/NoOpNoOp*^while/lstm_cell_11/BiasAdd/ReadVariableOp)^while/lstm_cell_11/MatMul/ReadVariableOp+^while/lstm_cell_11/MatMul_1/ReadVariableOp*"
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
2while_lstm_cell_11_biasadd_readvariableop_resource4while_lstm_cell_11_biasadd_readvariableop_resource_0"l
3while_lstm_cell_11_matmul_1_readvariableop_resource5while_lstm_cell_11_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_11_matmul_readvariableop_resource3while_lstm_cell_11_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :         в:         в: : : : : 2V
)while/lstm_cell_11/BiasAdd/ReadVariableOp)while/lstm_cell_11/BiasAdd/ReadVariableOp2T
(while/lstm_cell_11/MatMul/ReadVariableOp(while/lstm_cell_11/MatMul/ReadVariableOp2X
*while/lstm_cell_11/MatMul_1/ReadVariableOp*while/lstm_cell_11/MatMul_1/ReadVariableOp: 
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
:         в:.*
(
_output_shapes
:         в:

_output_shapes
: :

_output_shapes
: 
├\
а
E__inference_lstm_11_layer_call_and_return_conditional_losses_19548907
inputs_0>
+lstm_cell_11_matmul_readvariableop_resource:	GИA
-lstm_cell_11_matmul_1_readvariableop_resource:
вИ;
,lstm_cell_11_biasadd_readvariableop_resource:	И
identityИв#lstm_cell_11/BiasAdd/ReadVariableOpв"lstm_cell_11/MatMul/ReadVariableOpв$lstm_cell_11/MatMul_1/ReadVariableOpвwhileF
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
B :в2
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
B :в2
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
:         в2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :в2
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
B :в2
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
:         в2	
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
 :                  G2
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
valueB"    G   27
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
:         G*
shrink_axis_mask2
strided_slice_2╡
"lstm_cell_11/MatMul/ReadVariableOpReadVariableOp+lstm_cell_11_matmul_readvariableop_resource*
_output_shapes
:	GИ*
dtype02$
"lstm_cell_11/MatMul/ReadVariableOpн
lstm_cell_11/MatMulMatMulstrided_slice_2:output:0*lstm_cell_11/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         И2
lstm_cell_11/MatMul╝
$lstm_cell_11/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_11_matmul_1_readvariableop_resource* 
_output_shapes
:
вИ*
dtype02&
$lstm_cell_11/MatMul_1/ReadVariableOpй
lstm_cell_11/MatMul_1MatMulzeros:output:0,lstm_cell_11/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         И2
lstm_cell_11/MatMul_1а
lstm_cell_11/addAddV2lstm_cell_11/MatMul:product:0lstm_cell_11/MatMul_1:product:0*
T0*(
_output_shapes
:         И2
lstm_cell_11/add┤
#lstm_cell_11/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_11_biasadd_readvariableop_resource*
_output_shapes	
:И*
dtype02%
#lstm_cell_11/BiasAdd/ReadVariableOpн
lstm_cell_11/BiasAddBiasAddlstm_cell_11/add:z:0+lstm_cell_11/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         И2
lstm_cell_11/BiasAdd~
lstm_cell_11/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_11/split/split_dimў
lstm_cell_11/splitSplit%lstm_cell_11/split/split_dim:output:0lstm_cell_11/BiasAdd:output:0*
T0*d
_output_shapesR
P:         в:         в:         в:         в*
	num_split2
lstm_cell_11/splitЗ
lstm_cell_11/SigmoidSigmoidlstm_cell_11/split:output:0*
T0*(
_output_shapes
:         в2
lstm_cell_11/SigmoidЛ
lstm_cell_11/Sigmoid_1Sigmoidlstm_cell_11/split:output:1*
T0*(
_output_shapes
:         в2
lstm_cell_11/Sigmoid_1М
lstm_cell_11/mulMullstm_cell_11/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:         в2
lstm_cell_11/mul~
lstm_cell_11/ReluRelulstm_cell_11/split:output:2*
T0*(
_output_shapes
:         в2
lstm_cell_11/ReluЭ
lstm_cell_11/mul_1Mullstm_cell_11/Sigmoid:y:0lstm_cell_11/Relu:activations:0*
T0*(
_output_shapes
:         в2
lstm_cell_11/mul_1Т
lstm_cell_11/add_1AddV2lstm_cell_11/mul:z:0lstm_cell_11/mul_1:z:0*
T0*(
_output_shapes
:         в2
lstm_cell_11/add_1Л
lstm_cell_11/Sigmoid_2Sigmoidlstm_cell_11/split:output:3*
T0*(
_output_shapes
:         в2
lstm_cell_11/Sigmoid_2}
lstm_cell_11/Relu_1Relulstm_cell_11/add_1:z:0*
T0*(
_output_shapes
:         в2
lstm_cell_11/Relu_1б
lstm_cell_11/mul_2Mullstm_cell_11/Sigmoid_2:y:0!lstm_cell_11/Relu_1:activations:0*
T0*(
_output_shapes
:         в2
lstm_cell_11/mul_2П
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    в   2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_11_matmul_readvariableop_resource-lstm_cell_11_matmul_1_readvariableop_resource,lstm_cell_11_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :         в:         в: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_19548823*
condR
while_cond_19548822*M
output_shapes<
:: : : : :         в:         в: : : : : *
parallel_iterations 2
while╡
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    в   22
0TensorArrayV2Stack/TensorListStack/element_shapeЄ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:                  в*
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
:         в*
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
!:                  в2
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
!:                  в2

Identity╚
NoOpNoOp$^lstm_cell_11/BiasAdd/ReadVariableOp#^lstm_cell_11/MatMul/ReadVariableOp%^lstm_cell_11/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  G: : : 2J
#lstm_cell_11/BiasAdd/ReadVariableOp#lstm_cell_11/BiasAdd/ReadVariableOp2H
"lstm_cell_11/MatMul/ReadVariableOp"lstm_cell_11/MatMul/ReadVariableOp2L
$lstm_cell_11/MatMul_1/ReadVariableOp$lstm_cell_11/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :                  G
"
_user_specified_name
inputs/0
╘!
¤
E__inference_dense_5_layer_call_and_return_conditional_losses_19546782

inputs4
!tensordot_readvariableop_resource:	в-
biasadd_readvariableop_resource:
identityИвBiasAdd/ReadVariableOpвTensordot/ReadVariableOpЧ
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes
:	в*
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
:         в2
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
:         в: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:T P
,
_output_shapes
:         в
 
_user_specified_nameinputs
У
Й
J__inference_lstm_cell_11_layer_call_and_return_conditional_losses_19549633

inputs
states_0
states_11
matmul_readvariableop_resource:	GИ4
 matmul_1_readvariableop_resource:
вИ.
biasadd_readvariableop_resource:	И
identity

identity_1

identity_2ИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpвMatMul_1/ReadVariableOpО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	GИ*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         И2
MatMulХ
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
вИ*
dtype02
MatMul_1/ReadVariableOp|
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         И2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:         И2
addН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:И*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         И2	
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
P:         в:         в:         в:         в*
	num_split2
split`
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:         в2	
Sigmoidd
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:         в2
	Sigmoid_1]
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:         в2
mulW
ReluRelusplit:output:2*
T0*(
_output_shapes
:         в2
Relui
mul_1MulSigmoid:y:0Relu:activations:0*
T0*(
_output_shapes
:         в2
mul_1^
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:         в2
add_1d
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:         в2
	Sigmoid_2V
Relu_1Relu	add_1:z:0*
T0*(
_output_shapes
:         в2
Relu_1m
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*(
_output_shapes
:         в2
mul_2e
IdentityIdentity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:         в2

Identityi

Identity_1Identity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:         в2

Identity_1i

Identity_2Identity	add_1:z:0^NoOp*
T0*(
_output_shapes
:         в2

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
A:         G:         в:         в: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:         G
 
_user_specified_nameinputs:RN
(
_output_shapes
:         в
"
_user_specified_name
states/0:RN
(
_output_shapes
:         в
"
_user_specified_name
states/1
Д\
Ю
E__inference_lstm_11_layer_call_and_return_conditional_losses_19546736

inputs>
+lstm_cell_11_matmul_readvariableop_resource:	GИA
-lstm_cell_11_matmul_1_readvariableop_resource:
вИ;
,lstm_cell_11_biasadd_readvariableop_resource:	И
identityИв#lstm_cell_11/BiasAdd/ReadVariableOpв"lstm_cell_11/MatMul/ReadVariableOpв$lstm_cell_11/MatMul_1/ReadVariableOpвwhileD
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
B :в2
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
B :в2
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
:         в2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :в2
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
B :в2
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
:         в2	
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
:         G2
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
valueB"    G   27
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
:         G*
shrink_axis_mask2
strided_slice_2╡
"lstm_cell_11/MatMul/ReadVariableOpReadVariableOp+lstm_cell_11_matmul_readvariableop_resource*
_output_shapes
:	GИ*
dtype02$
"lstm_cell_11/MatMul/ReadVariableOpн
lstm_cell_11/MatMulMatMulstrided_slice_2:output:0*lstm_cell_11/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         И2
lstm_cell_11/MatMul╝
$lstm_cell_11/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_11_matmul_1_readvariableop_resource* 
_output_shapes
:
вИ*
dtype02&
$lstm_cell_11/MatMul_1/ReadVariableOpй
lstm_cell_11/MatMul_1MatMulzeros:output:0,lstm_cell_11/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         И2
lstm_cell_11/MatMul_1а
lstm_cell_11/addAddV2lstm_cell_11/MatMul:product:0lstm_cell_11/MatMul_1:product:0*
T0*(
_output_shapes
:         И2
lstm_cell_11/add┤
#lstm_cell_11/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_11_biasadd_readvariableop_resource*
_output_shapes	
:И*
dtype02%
#lstm_cell_11/BiasAdd/ReadVariableOpн
lstm_cell_11/BiasAddBiasAddlstm_cell_11/add:z:0+lstm_cell_11/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         И2
lstm_cell_11/BiasAdd~
lstm_cell_11/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_11/split/split_dimў
lstm_cell_11/splitSplit%lstm_cell_11/split/split_dim:output:0lstm_cell_11/BiasAdd:output:0*
T0*d
_output_shapesR
P:         в:         в:         в:         в*
	num_split2
lstm_cell_11/splitЗ
lstm_cell_11/SigmoidSigmoidlstm_cell_11/split:output:0*
T0*(
_output_shapes
:         в2
lstm_cell_11/SigmoidЛ
lstm_cell_11/Sigmoid_1Sigmoidlstm_cell_11/split:output:1*
T0*(
_output_shapes
:         в2
lstm_cell_11/Sigmoid_1М
lstm_cell_11/mulMullstm_cell_11/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:         в2
lstm_cell_11/mul~
lstm_cell_11/ReluRelulstm_cell_11/split:output:2*
T0*(
_output_shapes
:         в2
lstm_cell_11/ReluЭ
lstm_cell_11/mul_1Mullstm_cell_11/Sigmoid:y:0lstm_cell_11/Relu:activations:0*
T0*(
_output_shapes
:         в2
lstm_cell_11/mul_1Т
lstm_cell_11/add_1AddV2lstm_cell_11/mul:z:0lstm_cell_11/mul_1:z:0*
T0*(
_output_shapes
:         в2
lstm_cell_11/add_1Л
lstm_cell_11/Sigmoid_2Sigmoidlstm_cell_11/split:output:3*
T0*(
_output_shapes
:         в2
lstm_cell_11/Sigmoid_2}
lstm_cell_11/Relu_1Relulstm_cell_11/add_1:z:0*
T0*(
_output_shapes
:         в2
lstm_cell_11/Relu_1б
lstm_cell_11/mul_2Mullstm_cell_11/Sigmoid_2:y:0!lstm_cell_11/Relu_1:activations:0*
T0*(
_output_shapes
:         в2
lstm_cell_11/mul_2П
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    в   2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_11_matmul_readvariableop_resource-lstm_cell_11_matmul_1_readvariableop_resource,lstm_cell_11_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :         в:         в: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_19546652*
condR
while_cond_19546651*M
output_shapes<
:: : : : :         в:         в: : : : : *
parallel_iterations 2
while╡
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    в   22
0TensorArrayV2Stack/TensorListStack/element_shapeщ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:         в*
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
:         в*
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
:         в2
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
:         в2

Identity╚
NoOpNoOp$^lstm_cell_11/BiasAdd/ReadVariableOp#^lstm_cell_11/MatMul/ReadVariableOp%^lstm_cell_11/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         G: : : 2J
#lstm_cell_11/BiasAdd/ReadVariableOp#lstm_cell_11/BiasAdd/ReadVariableOp2H
"lstm_cell_11/MatMul/ReadVariableOp"lstm_cell_11/MatMul/ReadVariableOp2L
$lstm_cell_11/MatMul_1/ReadVariableOp$lstm_cell_11/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:         G
 
_user_specified_nameinputs
╢
f
-__inference_dropout_10_layer_call_fn_19548756

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
:         G* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_dropout_10_layer_call_and_return_conditional_losses_195470342
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:         G2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         G22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:         G
 
_user_specified_nameinputs
яЛ
Е
J__inference_sequential_5_layer_call_and_return_conditional_losses_19548039

inputsF
3lstm_10_lstm_cell_10_matmul_readvariableop_resource:	]ЬH
5lstm_10_lstm_cell_10_matmul_1_readvariableop_resource:	GЬC
4lstm_10_lstm_cell_10_biasadd_readvariableop_resource:	ЬF
3lstm_11_lstm_cell_11_matmul_readvariableop_resource:	GИI
5lstm_11_lstm_cell_11_matmul_1_readvariableop_resource:
вИC
4lstm_11_lstm_cell_11_biasadd_readvariableop_resource:	И<
)dense_5_tensordot_readvariableop_resource:	в5
'dense_5_biasadd_readvariableop_resource:
identityИвdense_5/BiasAdd/ReadVariableOpв dense_5/Tensordot/ReadVariableOpв+lstm_10/lstm_cell_10/BiasAdd/ReadVariableOpв*lstm_10/lstm_cell_10/MatMul/ReadVariableOpв,lstm_10/lstm_cell_10/MatMul_1/ReadVariableOpвlstm_10/whileв+lstm_11/lstm_cell_11/BiasAdd/ReadVariableOpв*lstm_11/lstm_cell_11/MatMul/ReadVariableOpв,lstm_11/lstm_cell_11/MatMul_1/ReadVariableOpвlstm_11/whileT
lstm_10/ShapeShapeinputs*
T0*
_output_shapes
:2
lstm_10/ShapeД
lstm_10/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_10/strided_slice/stackИ
lstm_10/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_10/strided_slice/stack_1И
lstm_10/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_10/strided_slice/stack_2Т
lstm_10/strided_sliceStridedSlicelstm_10/Shape:output:0$lstm_10/strided_slice/stack:output:0&lstm_10/strided_slice/stack_1:output:0&lstm_10/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_10/strided_slicel
lstm_10/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :G2
lstm_10/zeros/mul/yМ
lstm_10/zeros/mulMullstm_10/strided_slice:output:0lstm_10/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_10/zeros/mulo
lstm_10/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
lstm_10/zeros/Less/yЗ
lstm_10/zeros/LessLesslstm_10/zeros/mul:z:0lstm_10/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_10/zeros/Lessr
lstm_10/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :G2
lstm_10/zeros/packed/1г
lstm_10/zeros/packedPacklstm_10/strided_slice:output:0lstm_10/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_10/zeros/packedo
lstm_10/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_10/zeros/ConstХ
lstm_10/zerosFilllstm_10/zeros/packed:output:0lstm_10/zeros/Const:output:0*
T0*'
_output_shapes
:         G2
lstm_10/zerosp
lstm_10/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :G2
lstm_10/zeros_1/mul/yТ
lstm_10/zeros_1/mulMullstm_10/strided_slice:output:0lstm_10/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_10/zeros_1/muls
lstm_10/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
lstm_10/zeros_1/Less/yП
lstm_10/zeros_1/LessLesslstm_10/zeros_1/mul:z:0lstm_10/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_10/zeros_1/Lessv
lstm_10/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :G2
lstm_10/zeros_1/packed/1й
lstm_10/zeros_1/packedPacklstm_10/strided_slice:output:0!lstm_10/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_10/zeros_1/packeds
lstm_10/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_10/zeros_1/ConstЭ
lstm_10/zeros_1Filllstm_10/zeros_1/packed:output:0lstm_10/zeros_1/Const:output:0*
T0*'
_output_shapes
:         G2
lstm_10/zeros_1Е
lstm_10/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_10/transpose/permТ
lstm_10/transpose	Transposeinputslstm_10/transpose/perm:output:0*
T0*+
_output_shapes
:         ]2
lstm_10/transposeg
lstm_10/Shape_1Shapelstm_10/transpose:y:0*
T0*
_output_shapes
:2
lstm_10/Shape_1И
lstm_10/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_10/strided_slice_1/stackМ
lstm_10/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_10/strided_slice_1/stack_1М
lstm_10/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_10/strided_slice_1/stack_2Ю
lstm_10/strided_slice_1StridedSlicelstm_10/Shape_1:output:0&lstm_10/strided_slice_1/stack:output:0(lstm_10/strided_slice_1/stack_1:output:0(lstm_10/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_10/strided_slice_1Х
#lstm_10/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         2%
#lstm_10/TensorArrayV2/element_shape╥
lstm_10/TensorArrayV2TensorListReserve,lstm_10/TensorArrayV2/element_shape:output:0 lstm_10/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_10/TensorArrayV2╧
=lstm_10/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ]   2?
=lstm_10/TensorArrayUnstack/TensorListFromTensor/element_shapeШ
/lstm_10/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_10/transpose:y:0Flstm_10/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type021
/lstm_10/TensorArrayUnstack/TensorListFromTensorИ
lstm_10/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_10/strided_slice_2/stackМ
lstm_10/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_10/strided_slice_2/stack_1М
lstm_10/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_10/strided_slice_2/stack_2м
lstm_10/strided_slice_2StridedSlicelstm_10/transpose:y:0&lstm_10/strided_slice_2/stack:output:0(lstm_10/strided_slice_2/stack_1:output:0(lstm_10/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         ]*
shrink_axis_mask2
lstm_10/strided_slice_2═
*lstm_10/lstm_cell_10/MatMul/ReadVariableOpReadVariableOp3lstm_10_lstm_cell_10_matmul_readvariableop_resource*
_output_shapes
:	]Ь*
dtype02,
*lstm_10/lstm_cell_10/MatMul/ReadVariableOp═
lstm_10/lstm_cell_10/MatMulMatMul lstm_10/strided_slice_2:output:02lstm_10/lstm_cell_10/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ь2
lstm_10/lstm_cell_10/MatMul╙
,lstm_10/lstm_cell_10/MatMul_1/ReadVariableOpReadVariableOp5lstm_10_lstm_cell_10_matmul_1_readvariableop_resource*
_output_shapes
:	GЬ*
dtype02.
,lstm_10/lstm_cell_10/MatMul_1/ReadVariableOp╔
lstm_10/lstm_cell_10/MatMul_1MatMullstm_10/zeros:output:04lstm_10/lstm_cell_10/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ь2
lstm_10/lstm_cell_10/MatMul_1└
lstm_10/lstm_cell_10/addAddV2%lstm_10/lstm_cell_10/MatMul:product:0'lstm_10/lstm_cell_10/MatMul_1:product:0*
T0*(
_output_shapes
:         Ь2
lstm_10/lstm_cell_10/add╠
+lstm_10/lstm_cell_10/BiasAdd/ReadVariableOpReadVariableOp4lstm_10_lstm_cell_10_biasadd_readvariableop_resource*
_output_shapes	
:Ь*
dtype02-
+lstm_10/lstm_cell_10/BiasAdd/ReadVariableOp═
lstm_10/lstm_cell_10/BiasAddBiasAddlstm_10/lstm_cell_10/add:z:03lstm_10/lstm_cell_10/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ь2
lstm_10/lstm_cell_10/BiasAddО
$lstm_10/lstm_cell_10/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2&
$lstm_10/lstm_cell_10/split/split_dimУ
lstm_10/lstm_cell_10/splitSplit-lstm_10/lstm_cell_10/split/split_dim:output:0%lstm_10/lstm_cell_10/BiasAdd:output:0*
T0*`
_output_shapesN
L:         G:         G:         G:         G*
	num_split2
lstm_10/lstm_cell_10/splitЮ
lstm_10/lstm_cell_10/SigmoidSigmoid#lstm_10/lstm_cell_10/split:output:0*
T0*'
_output_shapes
:         G2
lstm_10/lstm_cell_10/Sigmoidв
lstm_10/lstm_cell_10/Sigmoid_1Sigmoid#lstm_10/lstm_cell_10/split:output:1*
T0*'
_output_shapes
:         G2 
lstm_10/lstm_cell_10/Sigmoid_1л
lstm_10/lstm_cell_10/mulMul"lstm_10/lstm_cell_10/Sigmoid_1:y:0lstm_10/zeros_1:output:0*
T0*'
_output_shapes
:         G2
lstm_10/lstm_cell_10/mulХ
lstm_10/lstm_cell_10/ReluRelu#lstm_10/lstm_cell_10/split:output:2*
T0*'
_output_shapes
:         G2
lstm_10/lstm_cell_10/Relu╝
lstm_10/lstm_cell_10/mul_1Mul lstm_10/lstm_cell_10/Sigmoid:y:0'lstm_10/lstm_cell_10/Relu:activations:0*
T0*'
_output_shapes
:         G2
lstm_10/lstm_cell_10/mul_1▒
lstm_10/lstm_cell_10/add_1AddV2lstm_10/lstm_cell_10/mul:z:0lstm_10/lstm_cell_10/mul_1:z:0*
T0*'
_output_shapes
:         G2
lstm_10/lstm_cell_10/add_1в
lstm_10/lstm_cell_10/Sigmoid_2Sigmoid#lstm_10/lstm_cell_10/split:output:3*
T0*'
_output_shapes
:         G2 
lstm_10/lstm_cell_10/Sigmoid_2Ф
lstm_10/lstm_cell_10/Relu_1Relulstm_10/lstm_cell_10/add_1:z:0*
T0*'
_output_shapes
:         G2
lstm_10/lstm_cell_10/Relu_1└
lstm_10/lstm_cell_10/mul_2Mul"lstm_10/lstm_cell_10/Sigmoid_2:y:0)lstm_10/lstm_cell_10/Relu_1:activations:0*
T0*'
_output_shapes
:         G2
lstm_10/lstm_cell_10/mul_2Я
%lstm_10/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    G   2'
%lstm_10/TensorArrayV2_1/element_shape╪
lstm_10/TensorArrayV2_1TensorListReserve.lstm_10/TensorArrayV2_1/element_shape:output:0 lstm_10/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_10/TensorArrayV2_1^
lstm_10/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_10/timeП
 lstm_10/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         2"
 lstm_10/while/maximum_iterationsz
lstm_10/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_10/while/loop_counterЗ
lstm_10/whileWhile#lstm_10/while/loop_counter:output:0)lstm_10/while/maximum_iterations:output:0lstm_10/time:output:0 lstm_10/TensorArrayV2_1:handle:0lstm_10/zeros:output:0lstm_10/zeros_1:output:0 lstm_10/strided_slice_1:output:0?lstm_10/TensorArrayUnstack/TensorListFromTensor:output_handle:03lstm_10_lstm_cell_10_matmul_readvariableop_resource5lstm_10_lstm_cell_10_matmul_1_readvariableop_resource4lstm_10_lstm_cell_10_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         G:         G: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *'
bodyR
lstm_10_while_body_19547765*'
condR
lstm_10_while_cond_19547764*K
output_shapes:
8: : : : :         G:         G: : : : : *
parallel_iterations 2
lstm_10/while┼
8lstm_10/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    G   2:
8lstm_10/TensorArrayV2Stack/TensorListStack/element_shapeИ
*lstm_10/TensorArrayV2Stack/TensorListStackTensorListStacklstm_10/while:output:3Alstm_10/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:         G*
element_dtype02,
*lstm_10/TensorArrayV2Stack/TensorListStackС
lstm_10/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         2
lstm_10/strided_slice_3/stackМ
lstm_10/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2!
lstm_10/strided_slice_3/stack_1М
lstm_10/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_10/strided_slice_3/stack_2╩
lstm_10/strided_slice_3StridedSlice3lstm_10/TensorArrayV2Stack/TensorListStack:tensor:0&lstm_10/strided_slice_3/stack:output:0(lstm_10/strided_slice_3/stack_1:output:0(lstm_10/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         G*
shrink_axis_mask2
lstm_10/strided_slice_3Й
lstm_10/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_10/transpose_1/perm┼
lstm_10/transpose_1	Transpose3lstm_10/TensorArrayV2Stack/TensorListStack:tensor:0!lstm_10/transpose_1/perm:output:0*
T0*+
_output_shapes
:         G2
lstm_10/transpose_1v
lstm_10/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_10/runtimey
dropout_10/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *лкк?2
dropout_10/dropout/Constй
dropout_10/dropout/MulMullstm_10/transpose_1:y:0!dropout_10/dropout/Const:output:0*
T0*+
_output_shapes
:         G2
dropout_10/dropout/Mul{
dropout_10/dropout/ShapeShapelstm_10/transpose_1:y:0*
T0*
_output_shapes
:2
dropout_10/dropout/Shape┘
/dropout_10/dropout/random_uniform/RandomUniformRandomUniform!dropout_10/dropout/Shape:output:0*
T0*+
_output_shapes
:         G*
dtype021
/dropout_10/dropout/random_uniform/RandomUniformЛ
!dropout_10/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А>2#
!dropout_10/dropout/GreaterEqual/yю
dropout_10/dropout/GreaterEqualGreaterEqual8dropout_10/dropout/random_uniform/RandomUniform:output:0*dropout_10/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:         G2!
dropout_10/dropout/GreaterEqualд
dropout_10/dropout/CastCast#dropout_10/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:         G2
dropout_10/dropout/Castк
dropout_10/dropout/Mul_1Muldropout_10/dropout/Mul:z:0dropout_10/dropout/Cast:y:0*
T0*+
_output_shapes
:         G2
dropout_10/dropout/Mul_1j
lstm_11/ShapeShapedropout_10/dropout/Mul_1:z:0*
T0*
_output_shapes
:2
lstm_11/ShapeД
lstm_11/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_11/strided_slice/stackИ
lstm_11/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_11/strided_slice/stack_1И
lstm_11/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_11/strided_slice/stack_2Т
lstm_11/strided_sliceStridedSlicelstm_11/Shape:output:0$lstm_11/strided_slice/stack:output:0&lstm_11/strided_slice/stack_1:output:0&lstm_11/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_11/strided_slicem
lstm_11/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :в2
lstm_11/zeros/mul/yМ
lstm_11/zeros/mulMullstm_11/strided_slice:output:0lstm_11/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_11/zeros/mulo
lstm_11/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
lstm_11/zeros/Less/yЗ
lstm_11/zeros/LessLesslstm_11/zeros/mul:z:0lstm_11/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_11/zeros/Lesss
lstm_11/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :в2
lstm_11/zeros/packed/1г
lstm_11/zeros/packedPacklstm_11/strided_slice:output:0lstm_11/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_11/zeros/packedo
lstm_11/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_11/zeros/ConstЦ
lstm_11/zerosFilllstm_11/zeros/packed:output:0lstm_11/zeros/Const:output:0*
T0*(
_output_shapes
:         в2
lstm_11/zerosq
lstm_11/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :в2
lstm_11/zeros_1/mul/yТ
lstm_11/zeros_1/mulMullstm_11/strided_slice:output:0lstm_11/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_11/zeros_1/muls
lstm_11/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
lstm_11/zeros_1/Less/yП
lstm_11/zeros_1/LessLesslstm_11/zeros_1/mul:z:0lstm_11/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_11/zeros_1/Lessw
lstm_11/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :в2
lstm_11/zeros_1/packed/1й
lstm_11/zeros_1/packedPacklstm_11/strided_slice:output:0!lstm_11/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_11/zeros_1/packeds
lstm_11/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_11/zeros_1/ConstЮ
lstm_11/zeros_1Filllstm_11/zeros_1/packed:output:0lstm_11/zeros_1/Const:output:0*
T0*(
_output_shapes
:         в2
lstm_11/zeros_1Е
lstm_11/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_11/transpose/permи
lstm_11/transpose	Transposedropout_10/dropout/Mul_1:z:0lstm_11/transpose/perm:output:0*
T0*+
_output_shapes
:         G2
lstm_11/transposeg
lstm_11/Shape_1Shapelstm_11/transpose:y:0*
T0*
_output_shapes
:2
lstm_11/Shape_1И
lstm_11/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_11/strided_slice_1/stackМ
lstm_11/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_11/strided_slice_1/stack_1М
lstm_11/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_11/strided_slice_1/stack_2Ю
lstm_11/strided_slice_1StridedSlicelstm_11/Shape_1:output:0&lstm_11/strided_slice_1/stack:output:0(lstm_11/strided_slice_1/stack_1:output:0(lstm_11/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_11/strided_slice_1Х
#lstm_11/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         2%
#lstm_11/TensorArrayV2/element_shape╥
lstm_11/TensorArrayV2TensorListReserve,lstm_11/TensorArrayV2/element_shape:output:0 lstm_11/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_11/TensorArrayV2╧
=lstm_11/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    G   2?
=lstm_11/TensorArrayUnstack/TensorListFromTensor/element_shapeШ
/lstm_11/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_11/transpose:y:0Flstm_11/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type021
/lstm_11/TensorArrayUnstack/TensorListFromTensorИ
lstm_11/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_11/strided_slice_2/stackМ
lstm_11/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_11/strided_slice_2/stack_1М
lstm_11/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_11/strided_slice_2/stack_2м
lstm_11/strided_slice_2StridedSlicelstm_11/transpose:y:0&lstm_11/strided_slice_2/stack:output:0(lstm_11/strided_slice_2/stack_1:output:0(lstm_11/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         G*
shrink_axis_mask2
lstm_11/strided_slice_2═
*lstm_11/lstm_cell_11/MatMul/ReadVariableOpReadVariableOp3lstm_11_lstm_cell_11_matmul_readvariableop_resource*
_output_shapes
:	GИ*
dtype02,
*lstm_11/lstm_cell_11/MatMul/ReadVariableOp═
lstm_11/lstm_cell_11/MatMulMatMul lstm_11/strided_slice_2:output:02lstm_11/lstm_cell_11/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         И2
lstm_11/lstm_cell_11/MatMul╘
,lstm_11/lstm_cell_11/MatMul_1/ReadVariableOpReadVariableOp5lstm_11_lstm_cell_11_matmul_1_readvariableop_resource* 
_output_shapes
:
вИ*
dtype02.
,lstm_11/lstm_cell_11/MatMul_1/ReadVariableOp╔
lstm_11/lstm_cell_11/MatMul_1MatMullstm_11/zeros:output:04lstm_11/lstm_cell_11/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         И2
lstm_11/lstm_cell_11/MatMul_1└
lstm_11/lstm_cell_11/addAddV2%lstm_11/lstm_cell_11/MatMul:product:0'lstm_11/lstm_cell_11/MatMul_1:product:0*
T0*(
_output_shapes
:         И2
lstm_11/lstm_cell_11/add╠
+lstm_11/lstm_cell_11/BiasAdd/ReadVariableOpReadVariableOp4lstm_11_lstm_cell_11_biasadd_readvariableop_resource*
_output_shapes	
:И*
dtype02-
+lstm_11/lstm_cell_11/BiasAdd/ReadVariableOp═
lstm_11/lstm_cell_11/BiasAddBiasAddlstm_11/lstm_cell_11/add:z:03lstm_11/lstm_cell_11/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         И2
lstm_11/lstm_cell_11/BiasAddО
$lstm_11/lstm_cell_11/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2&
$lstm_11/lstm_cell_11/split/split_dimЧ
lstm_11/lstm_cell_11/splitSplit-lstm_11/lstm_cell_11/split/split_dim:output:0%lstm_11/lstm_cell_11/BiasAdd:output:0*
T0*d
_output_shapesR
P:         в:         в:         в:         в*
	num_split2
lstm_11/lstm_cell_11/splitЯ
lstm_11/lstm_cell_11/SigmoidSigmoid#lstm_11/lstm_cell_11/split:output:0*
T0*(
_output_shapes
:         в2
lstm_11/lstm_cell_11/Sigmoidг
lstm_11/lstm_cell_11/Sigmoid_1Sigmoid#lstm_11/lstm_cell_11/split:output:1*
T0*(
_output_shapes
:         в2 
lstm_11/lstm_cell_11/Sigmoid_1м
lstm_11/lstm_cell_11/mulMul"lstm_11/lstm_cell_11/Sigmoid_1:y:0lstm_11/zeros_1:output:0*
T0*(
_output_shapes
:         в2
lstm_11/lstm_cell_11/mulЦ
lstm_11/lstm_cell_11/ReluRelu#lstm_11/lstm_cell_11/split:output:2*
T0*(
_output_shapes
:         в2
lstm_11/lstm_cell_11/Relu╜
lstm_11/lstm_cell_11/mul_1Mul lstm_11/lstm_cell_11/Sigmoid:y:0'lstm_11/lstm_cell_11/Relu:activations:0*
T0*(
_output_shapes
:         в2
lstm_11/lstm_cell_11/mul_1▓
lstm_11/lstm_cell_11/add_1AddV2lstm_11/lstm_cell_11/mul:z:0lstm_11/lstm_cell_11/mul_1:z:0*
T0*(
_output_shapes
:         в2
lstm_11/lstm_cell_11/add_1г
lstm_11/lstm_cell_11/Sigmoid_2Sigmoid#lstm_11/lstm_cell_11/split:output:3*
T0*(
_output_shapes
:         в2 
lstm_11/lstm_cell_11/Sigmoid_2Х
lstm_11/lstm_cell_11/Relu_1Relulstm_11/lstm_cell_11/add_1:z:0*
T0*(
_output_shapes
:         в2
lstm_11/lstm_cell_11/Relu_1┴
lstm_11/lstm_cell_11/mul_2Mul"lstm_11/lstm_cell_11/Sigmoid_2:y:0)lstm_11/lstm_cell_11/Relu_1:activations:0*
T0*(
_output_shapes
:         в2
lstm_11/lstm_cell_11/mul_2Я
%lstm_11/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    в   2'
%lstm_11/TensorArrayV2_1/element_shape╪
lstm_11/TensorArrayV2_1TensorListReserve.lstm_11/TensorArrayV2_1/element_shape:output:0 lstm_11/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_11/TensorArrayV2_1^
lstm_11/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_11/timeП
 lstm_11/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         2"
 lstm_11/while/maximum_iterationsz
lstm_11/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_11/while/loop_counterЛ
lstm_11/whileWhile#lstm_11/while/loop_counter:output:0)lstm_11/while/maximum_iterations:output:0lstm_11/time:output:0 lstm_11/TensorArrayV2_1:handle:0lstm_11/zeros:output:0lstm_11/zeros_1:output:0 lstm_11/strided_slice_1:output:0?lstm_11/TensorArrayUnstack/TensorListFromTensor:output_handle:03lstm_11_lstm_cell_11_matmul_readvariableop_resource5lstm_11_lstm_cell_11_matmul_1_readvariableop_resource4lstm_11_lstm_cell_11_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :         в:         в: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *'
bodyR
lstm_11_while_body_19547920*'
condR
lstm_11_while_cond_19547919*M
output_shapes<
:: : : : :         в:         в: : : : : *
parallel_iterations 2
lstm_11/while┼
8lstm_11/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    в   2:
8lstm_11/TensorArrayV2Stack/TensorListStack/element_shapeЙ
*lstm_11/TensorArrayV2Stack/TensorListStackTensorListStacklstm_11/while:output:3Alstm_11/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:         в*
element_dtype02,
*lstm_11/TensorArrayV2Stack/TensorListStackС
lstm_11/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         2
lstm_11/strided_slice_3/stackМ
lstm_11/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2!
lstm_11/strided_slice_3/stack_1М
lstm_11/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_11/strided_slice_3/stack_2╦
lstm_11/strided_slice_3StridedSlice3lstm_11/TensorArrayV2Stack/TensorListStack:tensor:0&lstm_11/strided_slice_3/stack:output:0(lstm_11/strided_slice_3/stack_1:output:0(lstm_11/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         в*
shrink_axis_mask2
lstm_11/strided_slice_3Й
lstm_11/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_11/transpose_1/perm╞
lstm_11/transpose_1	Transpose3lstm_11/TensorArrayV2Stack/TensorListStack:tensor:0!lstm_11/transpose_1/perm:output:0*
T0*,
_output_shapes
:         в2
lstm_11/transpose_1v
lstm_11/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_11/runtimey
dropout_11/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  а?2
dropout_11/dropout/Constк
dropout_11/dropout/MulMullstm_11/transpose_1:y:0!dropout_11/dropout/Const:output:0*
T0*,
_output_shapes
:         в2
dropout_11/dropout/Mul{
dropout_11/dropout/ShapeShapelstm_11/transpose_1:y:0*
T0*
_output_shapes
:2
dropout_11/dropout/Shape┌
/dropout_11/dropout/random_uniform/RandomUniformRandomUniform!dropout_11/dropout/Shape:output:0*
T0*,
_output_shapes
:         в*
dtype021
/dropout_11/dropout/random_uniform/RandomUniformЛ
!dropout_11/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>2#
!dropout_11/dropout/GreaterEqual/yя
dropout_11/dropout/GreaterEqualGreaterEqual8dropout_11/dropout/random_uniform/RandomUniform:output:0*dropout_11/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:         в2!
dropout_11/dropout/GreaterEqualе
dropout_11/dropout/CastCast#dropout_11/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:         в2
dropout_11/dropout/Castл
dropout_11/dropout/Mul_1Muldropout_11/dropout/Mul:z:0dropout_11/dropout/Cast:y:0*
T0*,
_output_shapes
:         в2
dropout_11/dropout/Mul_1п
 dense_5/Tensordot/ReadVariableOpReadVariableOp)dense_5_tensordot_readvariableop_resource*
_output_shapes
:	в*
dtype02"
 dense_5/Tensordot/ReadVariableOpz
dense_5/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_5/Tensordot/axesБ
dense_5/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_5/Tensordot/free~
dense_5/Tensordot/ShapeShapedropout_11/dropout/Mul_1:z:0*
T0*
_output_shapes
:2
dense_5/Tensordot/ShapeД
dense_5/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_5/Tensordot/GatherV2/axis∙
dense_5/Tensordot/GatherV2GatherV2 dense_5/Tensordot/Shape:output:0dense_5/Tensordot/free:output:0(dense_5/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_5/Tensordot/GatherV2И
!dense_5/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_5/Tensordot/GatherV2_1/axis 
dense_5/Tensordot/GatherV2_1GatherV2 dense_5/Tensordot/Shape:output:0dense_5/Tensordot/axes:output:0*dense_5/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_5/Tensordot/GatherV2_1|
dense_5/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_5/Tensordot/Constа
dense_5/Tensordot/ProdProd#dense_5/Tensordot/GatherV2:output:0 dense_5/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_5/Tensordot/ProdА
dense_5/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_5/Tensordot/Const_1и
dense_5/Tensordot/Prod_1Prod%dense_5/Tensordot/GatherV2_1:output:0"dense_5/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_5/Tensordot/Prod_1А
dense_5/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense_5/Tensordot/concat/axis╪
dense_5/Tensordot/concatConcatV2dense_5/Tensordot/free:output:0dense_5/Tensordot/axes:output:0&dense_5/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_5/Tensordot/concatм
dense_5/Tensordot/stackPackdense_5/Tensordot/Prod:output:0!dense_5/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_5/Tensordot/stack┐
dense_5/Tensordot/transpose	Transposedropout_11/dropout/Mul_1:z:0!dense_5/Tensordot/concat:output:0*
T0*,
_output_shapes
:         в2
dense_5/Tensordot/transpose┐
dense_5/Tensordot/ReshapeReshapedense_5/Tensordot/transpose:y:0 dense_5/Tensordot/stack:output:0*
T0*0
_output_shapes
:                  2
dense_5/Tensordot/Reshape╛
dense_5/Tensordot/MatMulMatMul"dense_5/Tensordot/Reshape:output:0(dense_5/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_5/Tensordot/MatMulА
dense_5/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
dense_5/Tensordot/Const_2Д
dense_5/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_5/Tensordot/concat_1/axisх
dense_5/Tensordot/concat_1ConcatV2#dense_5/Tensordot/GatherV2:output:0"dense_5/Tensordot/Const_2:output:0(dense_5/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_5/Tensordot/concat_1░
dense_5/TensordotReshape"dense_5/Tensordot/MatMul:product:0#dense_5/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:         2
dense_5/Tensordotд
dense_5/BiasAdd/ReadVariableOpReadVariableOp'dense_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_5/BiasAdd/ReadVariableOpз
dense_5/BiasAddBiasAdddense_5/Tensordot:output:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:         2
dense_5/BiasAdd}
dense_5/SoftmaxSoftmaxdense_5/BiasAdd:output:0*
T0*+
_output_shapes
:         2
dense_5/Softmaxx
IdentityIdentitydense_5/Softmax:softmax:0^NoOp*
T0*+
_output_shapes
:         2

Identity╞
NoOpNoOp^dense_5/BiasAdd/ReadVariableOp!^dense_5/Tensordot/ReadVariableOp,^lstm_10/lstm_cell_10/BiasAdd/ReadVariableOp+^lstm_10/lstm_cell_10/MatMul/ReadVariableOp-^lstm_10/lstm_cell_10/MatMul_1/ReadVariableOp^lstm_10/while,^lstm_11/lstm_cell_11/BiasAdd/ReadVariableOp+^lstm_11/lstm_cell_11/MatMul/ReadVariableOp-^lstm_11/lstm_cell_11/MatMul_1/ReadVariableOp^lstm_11/while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':         ]: : : : : : : : 2@
dense_5/BiasAdd/ReadVariableOpdense_5/BiasAdd/ReadVariableOp2D
 dense_5/Tensordot/ReadVariableOp dense_5/Tensordot/ReadVariableOp2Z
+lstm_10/lstm_cell_10/BiasAdd/ReadVariableOp+lstm_10/lstm_cell_10/BiasAdd/ReadVariableOp2X
*lstm_10/lstm_cell_10/MatMul/ReadVariableOp*lstm_10/lstm_cell_10/MatMul/ReadVariableOp2\
,lstm_10/lstm_cell_10/MatMul_1/ReadVariableOp,lstm_10/lstm_cell_10/MatMul_1/ReadVariableOp2
lstm_10/whilelstm_10/while2Z
+lstm_11/lstm_cell_11/BiasAdd/ReadVariableOp+lstm_11/lstm_cell_11/BiasAdd/ReadVariableOp2X
*lstm_11/lstm_cell_11/MatMul/ReadVariableOp*lstm_11/lstm_cell_11/MatMul/ReadVariableOp2\
,lstm_11/lstm_cell_11/MatMul_1/ReadVariableOp,lstm_11/lstm_cell_11/MatMul_1/ReadVariableOp2
lstm_11/whilelstm_11/while:S O
+
_output_shapes
:         ]
 
_user_specified_nameinputs
░?
╘
while_body_19548974
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_11_matmul_readvariableop_resource_0:	GИI
5while_lstm_cell_11_matmul_1_readvariableop_resource_0:
вИC
4while_lstm_cell_11_biasadd_readvariableop_resource_0:	И
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_11_matmul_readvariableop_resource:	GИG
3while_lstm_cell_11_matmul_1_readvariableop_resource:
вИA
2while_lstm_cell_11_biasadd_readvariableop_resource:	ИИв)while/lstm_cell_11/BiasAdd/ReadVariableOpв(while/lstm_cell_11/MatMul/ReadVariableOpв*while/lstm_cell_11/MatMul_1/ReadVariableOp├
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    G   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape╙
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         G*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem╔
(while/lstm_cell_11/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_11_matmul_readvariableop_resource_0*
_output_shapes
:	GИ*
dtype02*
(while/lstm_cell_11/MatMul/ReadVariableOp╫
while/lstm_cell_11/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_11/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         И2
while/lstm_cell_11/MatMul╨
*while/lstm_cell_11/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_11_matmul_1_readvariableop_resource_0* 
_output_shapes
:
вИ*
dtype02,
*while/lstm_cell_11/MatMul_1/ReadVariableOp└
while/lstm_cell_11/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_11/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         И2
while/lstm_cell_11/MatMul_1╕
while/lstm_cell_11/addAddV2#while/lstm_cell_11/MatMul:product:0%while/lstm_cell_11/MatMul_1:product:0*
T0*(
_output_shapes
:         И2
while/lstm_cell_11/add╚
)while/lstm_cell_11/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_11_biasadd_readvariableop_resource_0*
_output_shapes	
:И*
dtype02+
)while/lstm_cell_11/BiasAdd/ReadVariableOp┼
while/lstm_cell_11/BiasAddBiasAddwhile/lstm_cell_11/add:z:01while/lstm_cell_11/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         И2
while/lstm_cell_11/BiasAddК
"while/lstm_cell_11/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_11/split/split_dimП
while/lstm_cell_11/splitSplit+while/lstm_cell_11/split/split_dim:output:0#while/lstm_cell_11/BiasAdd:output:0*
T0*d
_output_shapesR
P:         в:         в:         в:         в*
	num_split2
while/lstm_cell_11/splitЩ
while/lstm_cell_11/SigmoidSigmoid!while/lstm_cell_11/split:output:0*
T0*(
_output_shapes
:         в2
while/lstm_cell_11/SigmoidЭ
while/lstm_cell_11/Sigmoid_1Sigmoid!while/lstm_cell_11/split:output:1*
T0*(
_output_shapes
:         в2
while/lstm_cell_11/Sigmoid_1б
while/lstm_cell_11/mulMul while/lstm_cell_11/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:         в2
while/lstm_cell_11/mulР
while/lstm_cell_11/ReluRelu!while/lstm_cell_11/split:output:2*
T0*(
_output_shapes
:         в2
while/lstm_cell_11/Relu╡
while/lstm_cell_11/mul_1Mulwhile/lstm_cell_11/Sigmoid:y:0%while/lstm_cell_11/Relu:activations:0*
T0*(
_output_shapes
:         в2
while/lstm_cell_11/mul_1к
while/lstm_cell_11/add_1AddV2while/lstm_cell_11/mul:z:0while/lstm_cell_11/mul_1:z:0*
T0*(
_output_shapes
:         в2
while/lstm_cell_11/add_1Э
while/lstm_cell_11/Sigmoid_2Sigmoid!while/lstm_cell_11/split:output:3*
T0*(
_output_shapes
:         в2
while/lstm_cell_11/Sigmoid_2П
while/lstm_cell_11/Relu_1Reluwhile/lstm_cell_11/add_1:z:0*
T0*(
_output_shapes
:         в2
while/lstm_cell_11/Relu_1╣
while/lstm_cell_11/mul_2Mul while/lstm_cell_11/Sigmoid_2:y:0'while/lstm_cell_11/Relu_1:activations:0*
T0*(
_output_shapes
:         в2
while/lstm_cell_11/mul_2р
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_11/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_11/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:         в2
while/Identity_4О
while/Identity_5Identitywhile/lstm_cell_11/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:         в2
while/Identity_5▐

while/NoOpNoOp*^while/lstm_cell_11/BiasAdd/ReadVariableOp)^while/lstm_cell_11/MatMul/ReadVariableOp+^while/lstm_cell_11/MatMul_1/ReadVariableOp*"
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
2while_lstm_cell_11_biasadd_readvariableop_resource4while_lstm_cell_11_biasadd_readvariableop_resource_0"l
3while_lstm_cell_11_matmul_1_readvariableop_resource5while_lstm_cell_11_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_11_matmul_readvariableop_resource3while_lstm_cell_11_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :         в:         в: : : : : 2V
)while/lstm_cell_11/BiasAdd/ReadVariableOp)while/lstm_cell_11/BiasAdd/ReadVariableOp2T
(while/lstm_cell_11/MatMul/ReadVariableOp(while/lstm_cell_11/MatMul/ReadVariableOp2X
*while/lstm_cell_11/MatMul_1/ReadVariableOp*while/lstm_cell_11/MatMul_1/ReadVariableOp: 
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
:         в:.*
(
_output_shapes
:         в:

_output_shapes
: :

_output_shapes
: 
Й
f
H__inference_dropout_11_layer_call_and_return_conditional_losses_19546749

inputs

identity_1_
IdentityIdentityinputs*
T0*,
_output_shapes
:         в2

Identityn

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:         в2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         в:T P
,
_output_shapes
:         в
 
_user_specified_nameinputs
▀
═
while_cond_19548600
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_19548600___redundant_placeholder06
2while_while_cond_19548600___redundant_placeholder16
2while_while_cond_19548600___redundant_placeholder26
2while_while_cond_19548600___redundant_placeholder3
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
@: : : : :         G:         G: ::::: 
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
:         G:-)
'
_output_shapes
:         G:

_output_shapes
: :

_output_shapes
:
·%
ё
while_body_19545453
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_00
while_lstm_cell_10_19545477_0:	]Ь0
while_lstm_cell_10_19545479_0:	GЬ,
while_lstm_cell_10_19545481_0:	Ь
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor.
while_lstm_cell_10_19545477:	]Ь.
while_lstm_cell_10_19545479:	GЬ*
while_lstm_cell_10_19545481:	ЬИв*while/lstm_cell_10/StatefulPartitionedCall├
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
*while/lstm_cell_10/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_10_19545477_0while_lstm_cell_10_19545479_0while_lstm_cell_10_19545481_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:         G:         G:         G*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_lstm_cell_10_layer_call_and_return_conditional_losses_195453752,
*while/lstm_cell_10/StatefulPartitionedCallў
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder3while/lstm_cell_10/StatefulPartitionedCall:output:0*
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
while/Identity_4Identity3while/lstm_cell_10/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:         G2
while/Identity_4д
while/Identity_5Identity3while/lstm_cell_10/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:         G2
while/Identity_5З

while/NoOpNoOp+^while/lstm_cell_10/StatefulPartitionedCall*"
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
while_lstm_cell_10_19545477while_lstm_cell_10_19545477_0"<
while_lstm_cell_10_19545479while_lstm_cell_10_19545479_0"<
while_lstm_cell_10_19545481while_lstm_cell_10_19545481_0"0
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         G:         G: : : : : 2X
*while/lstm_cell_10/StatefulPartitionedCall*while/lstm_cell_10/StatefulPartitionedCall: 
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
:         G:-)
'
_output_shapes
:         G:

_output_shapes
: :

_output_shapes
: 
ч[
Э
E__inference_lstm_10_layer_call_and_return_conditional_losses_19547201

inputs>
+lstm_cell_10_matmul_readvariableop_resource:	]Ь@
-lstm_cell_10_matmul_1_readvariableop_resource:	GЬ;
,lstm_cell_10_biasadd_readvariableop_resource:	Ь
identityИв#lstm_cell_10/BiasAdd/ReadVariableOpв"lstm_cell_10/MatMul/ReadVariableOpв$lstm_cell_10/MatMul_1/ReadVariableOpвwhileD
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
value	B :G2
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
value	B :G2
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
:         G2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :G2
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
value	B :G2
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
:         G2	
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
"lstm_cell_10/MatMul/ReadVariableOpReadVariableOp+lstm_cell_10_matmul_readvariableop_resource*
_output_shapes
:	]Ь*
dtype02$
"lstm_cell_10/MatMul/ReadVariableOpн
lstm_cell_10/MatMulMatMulstrided_slice_2:output:0*lstm_cell_10/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ь2
lstm_cell_10/MatMul╗
$lstm_cell_10/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_10_matmul_1_readvariableop_resource*
_output_shapes
:	GЬ*
dtype02&
$lstm_cell_10/MatMul_1/ReadVariableOpй
lstm_cell_10/MatMul_1MatMulzeros:output:0,lstm_cell_10/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ь2
lstm_cell_10/MatMul_1а
lstm_cell_10/addAddV2lstm_cell_10/MatMul:product:0lstm_cell_10/MatMul_1:product:0*
T0*(
_output_shapes
:         Ь2
lstm_cell_10/add┤
#lstm_cell_10/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_10_biasadd_readvariableop_resource*
_output_shapes	
:Ь*
dtype02%
#lstm_cell_10/BiasAdd/ReadVariableOpн
lstm_cell_10/BiasAddBiasAddlstm_cell_10/add:z:0+lstm_cell_10/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ь2
lstm_cell_10/BiasAdd~
lstm_cell_10/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_10/split/split_dimє
lstm_cell_10/splitSplit%lstm_cell_10/split/split_dim:output:0lstm_cell_10/BiasAdd:output:0*
T0*`
_output_shapesN
L:         G:         G:         G:         G*
	num_split2
lstm_cell_10/splitЖ
lstm_cell_10/SigmoidSigmoidlstm_cell_10/split:output:0*
T0*'
_output_shapes
:         G2
lstm_cell_10/SigmoidК
lstm_cell_10/Sigmoid_1Sigmoidlstm_cell_10/split:output:1*
T0*'
_output_shapes
:         G2
lstm_cell_10/Sigmoid_1Л
lstm_cell_10/mulMullstm_cell_10/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:         G2
lstm_cell_10/mul}
lstm_cell_10/ReluRelulstm_cell_10/split:output:2*
T0*'
_output_shapes
:         G2
lstm_cell_10/ReluЬ
lstm_cell_10/mul_1Mullstm_cell_10/Sigmoid:y:0lstm_cell_10/Relu:activations:0*
T0*'
_output_shapes
:         G2
lstm_cell_10/mul_1С
lstm_cell_10/add_1AddV2lstm_cell_10/mul:z:0lstm_cell_10/mul_1:z:0*
T0*'
_output_shapes
:         G2
lstm_cell_10/add_1К
lstm_cell_10/Sigmoid_2Sigmoidlstm_cell_10/split:output:3*
T0*'
_output_shapes
:         G2
lstm_cell_10/Sigmoid_2|
lstm_cell_10/Relu_1Relulstm_cell_10/add_1:z:0*
T0*'
_output_shapes
:         G2
lstm_cell_10/Relu_1а
lstm_cell_10/mul_2Mullstm_cell_10/Sigmoid_2:y:0!lstm_cell_10/Relu_1:activations:0*
T0*'
_output_shapes
:         G2
lstm_cell_10/mul_2П
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    G   2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_10_matmul_readvariableop_resource-lstm_cell_10_matmul_1_readvariableop_resource,lstm_cell_10_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         G:         G: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_19547117*
condR
while_cond_19547116*K
output_shapes:
8: : : : :         G:         G: : : : : *
parallel_iterations 2
while╡
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    G   22
0TensorArrayV2Stack/TensorListStack/element_shapeш
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:         G*
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
:         G*
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
:         G2
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
:         G2

Identity╚
NoOpNoOp$^lstm_cell_10/BiasAdd/ReadVariableOp#^lstm_cell_10/MatMul/ReadVariableOp%^lstm_cell_10/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         ]: : : 2J
#lstm_cell_10/BiasAdd/ReadVariableOp#lstm_cell_10/BiasAdd/ReadVariableOp2H
"lstm_cell_10/MatMul/ReadVariableOp"lstm_cell_10/MatMul/ReadVariableOp2L
$lstm_cell_10/MatMul_1/ReadVariableOp$lstm_cell_10/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:         ]
 
_user_specified_nameinputs
╗
f
-__inference_dropout_11_layer_call_fn_19549431

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
:         в* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_dropout_11_layer_call_and_return_conditional_losses_195468382
StatefulPartitionedCallА
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:         в2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         в22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:         в
 
_user_specified_nameinputs
м]
ї
(sequential_5_lstm_11_while_body_19545042F
Bsequential_5_lstm_11_while_sequential_5_lstm_11_while_loop_counterL
Hsequential_5_lstm_11_while_sequential_5_lstm_11_while_maximum_iterations*
&sequential_5_lstm_11_while_placeholder,
(sequential_5_lstm_11_while_placeholder_1,
(sequential_5_lstm_11_while_placeholder_2,
(sequential_5_lstm_11_while_placeholder_3E
Asequential_5_lstm_11_while_sequential_5_lstm_11_strided_slice_1_0Б
}sequential_5_lstm_11_while_tensorarrayv2read_tensorlistgetitem_sequential_5_lstm_11_tensorarrayunstack_tensorlistfromtensor_0[
Hsequential_5_lstm_11_while_lstm_cell_11_matmul_readvariableop_resource_0:	GИ^
Jsequential_5_lstm_11_while_lstm_cell_11_matmul_1_readvariableop_resource_0:
вИX
Isequential_5_lstm_11_while_lstm_cell_11_biasadd_readvariableop_resource_0:	И'
#sequential_5_lstm_11_while_identity)
%sequential_5_lstm_11_while_identity_1)
%sequential_5_lstm_11_while_identity_2)
%sequential_5_lstm_11_while_identity_3)
%sequential_5_lstm_11_while_identity_4)
%sequential_5_lstm_11_while_identity_5C
?sequential_5_lstm_11_while_sequential_5_lstm_11_strided_slice_1
{sequential_5_lstm_11_while_tensorarrayv2read_tensorlistgetitem_sequential_5_lstm_11_tensorarrayunstack_tensorlistfromtensorY
Fsequential_5_lstm_11_while_lstm_cell_11_matmul_readvariableop_resource:	GИ\
Hsequential_5_lstm_11_while_lstm_cell_11_matmul_1_readvariableop_resource:
вИV
Gsequential_5_lstm_11_while_lstm_cell_11_biasadd_readvariableop_resource:	ИИв>sequential_5/lstm_11/while/lstm_cell_11/BiasAdd/ReadVariableOpв=sequential_5/lstm_11/while/lstm_cell_11/MatMul/ReadVariableOpв?sequential_5/lstm_11/while/lstm_cell_11/MatMul_1/ReadVariableOpэ
Lsequential_5/lstm_11/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    G   2N
Lsequential_5/lstm_11/while/TensorArrayV2Read/TensorListGetItem/element_shape╤
>sequential_5/lstm_11/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem}sequential_5_lstm_11_while_tensorarrayv2read_tensorlistgetitem_sequential_5_lstm_11_tensorarrayunstack_tensorlistfromtensor_0&sequential_5_lstm_11_while_placeholderUsequential_5/lstm_11/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         G*
element_dtype02@
>sequential_5/lstm_11/while/TensorArrayV2Read/TensorListGetItemИ
=sequential_5/lstm_11/while/lstm_cell_11/MatMul/ReadVariableOpReadVariableOpHsequential_5_lstm_11_while_lstm_cell_11_matmul_readvariableop_resource_0*
_output_shapes
:	GИ*
dtype02?
=sequential_5/lstm_11/while/lstm_cell_11/MatMul/ReadVariableOpл
.sequential_5/lstm_11/while/lstm_cell_11/MatMulMatMulEsequential_5/lstm_11/while/TensorArrayV2Read/TensorListGetItem:item:0Esequential_5/lstm_11/while/lstm_cell_11/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         И20
.sequential_5/lstm_11/while/lstm_cell_11/MatMulП
?sequential_5/lstm_11/while/lstm_cell_11/MatMul_1/ReadVariableOpReadVariableOpJsequential_5_lstm_11_while_lstm_cell_11_matmul_1_readvariableop_resource_0* 
_output_shapes
:
вИ*
dtype02A
?sequential_5/lstm_11/while/lstm_cell_11/MatMul_1/ReadVariableOpФ
0sequential_5/lstm_11/while/lstm_cell_11/MatMul_1MatMul(sequential_5_lstm_11_while_placeholder_2Gsequential_5/lstm_11/while/lstm_cell_11/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         И22
0sequential_5/lstm_11/while/lstm_cell_11/MatMul_1М
+sequential_5/lstm_11/while/lstm_cell_11/addAddV28sequential_5/lstm_11/while/lstm_cell_11/MatMul:product:0:sequential_5/lstm_11/while/lstm_cell_11/MatMul_1:product:0*
T0*(
_output_shapes
:         И2-
+sequential_5/lstm_11/while/lstm_cell_11/addЗ
>sequential_5/lstm_11/while/lstm_cell_11/BiasAdd/ReadVariableOpReadVariableOpIsequential_5_lstm_11_while_lstm_cell_11_biasadd_readvariableop_resource_0*
_output_shapes	
:И*
dtype02@
>sequential_5/lstm_11/while/lstm_cell_11/BiasAdd/ReadVariableOpЩ
/sequential_5/lstm_11/while/lstm_cell_11/BiasAddBiasAdd/sequential_5/lstm_11/while/lstm_cell_11/add:z:0Fsequential_5/lstm_11/while/lstm_cell_11/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         И21
/sequential_5/lstm_11/while/lstm_cell_11/BiasAdd┤
7sequential_5/lstm_11/while/lstm_cell_11/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :29
7sequential_5/lstm_11/while/lstm_cell_11/split/split_dimу
-sequential_5/lstm_11/while/lstm_cell_11/splitSplit@sequential_5/lstm_11/while/lstm_cell_11/split/split_dim:output:08sequential_5/lstm_11/while/lstm_cell_11/BiasAdd:output:0*
T0*d
_output_shapesR
P:         в:         в:         в:         в*
	num_split2/
-sequential_5/lstm_11/while/lstm_cell_11/split╪
/sequential_5/lstm_11/while/lstm_cell_11/SigmoidSigmoid6sequential_5/lstm_11/while/lstm_cell_11/split:output:0*
T0*(
_output_shapes
:         в21
/sequential_5/lstm_11/while/lstm_cell_11/Sigmoid▄
1sequential_5/lstm_11/while/lstm_cell_11/Sigmoid_1Sigmoid6sequential_5/lstm_11/while/lstm_cell_11/split:output:1*
T0*(
_output_shapes
:         в23
1sequential_5/lstm_11/while/lstm_cell_11/Sigmoid_1ї
+sequential_5/lstm_11/while/lstm_cell_11/mulMul5sequential_5/lstm_11/while/lstm_cell_11/Sigmoid_1:y:0(sequential_5_lstm_11_while_placeholder_3*
T0*(
_output_shapes
:         в2-
+sequential_5/lstm_11/while/lstm_cell_11/mul╧
,sequential_5/lstm_11/while/lstm_cell_11/ReluRelu6sequential_5/lstm_11/while/lstm_cell_11/split:output:2*
T0*(
_output_shapes
:         в2.
,sequential_5/lstm_11/while/lstm_cell_11/ReluЙ
-sequential_5/lstm_11/while/lstm_cell_11/mul_1Mul3sequential_5/lstm_11/while/lstm_cell_11/Sigmoid:y:0:sequential_5/lstm_11/while/lstm_cell_11/Relu:activations:0*
T0*(
_output_shapes
:         в2/
-sequential_5/lstm_11/while/lstm_cell_11/mul_1■
-sequential_5/lstm_11/while/lstm_cell_11/add_1AddV2/sequential_5/lstm_11/while/lstm_cell_11/mul:z:01sequential_5/lstm_11/while/lstm_cell_11/mul_1:z:0*
T0*(
_output_shapes
:         в2/
-sequential_5/lstm_11/while/lstm_cell_11/add_1▄
1sequential_5/lstm_11/while/lstm_cell_11/Sigmoid_2Sigmoid6sequential_5/lstm_11/while/lstm_cell_11/split:output:3*
T0*(
_output_shapes
:         в23
1sequential_5/lstm_11/while/lstm_cell_11/Sigmoid_2╬
.sequential_5/lstm_11/while/lstm_cell_11/Relu_1Relu1sequential_5/lstm_11/while/lstm_cell_11/add_1:z:0*
T0*(
_output_shapes
:         в20
.sequential_5/lstm_11/while/lstm_cell_11/Relu_1Н
-sequential_5/lstm_11/while/lstm_cell_11/mul_2Mul5sequential_5/lstm_11/while/lstm_cell_11/Sigmoid_2:y:0<sequential_5/lstm_11/while/lstm_cell_11/Relu_1:activations:0*
T0*(
_output_shapes
:         в2/
-sequential_5/lstm_11/while/lstm_cell_11/mul_2╔
?sequential_5/lstm_11/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem(sequential_5_lstm_11_while_placeholder_1&sequential_5_lstm_11_while_placeholder1sequential_5/lstm_11/while/lstm_cell_11/mul_2:z:0*
_output_shapes
: *
element_dtype02A
?sequential_5/lstm_11/while/TensorArrayV2Write/TensorListSetItemЖ
 sequential_5/lstm_11/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2"
 sequential_5/lstm_11/while/add/y╜
sequential_5/lstm_11/while/addAddV2&sequential_5_lstm_11_while_placeholder)sequential_5/lstm_11/while/add/y:output:0*
T0*
_output_shapes
: 2 
sequential_5/lstm_11/while/addК
"sequential_5/lstm_11/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2$
"sequential_5/lstm_11/while/add_1/y▀
 sequential_5/lstm_11/while/add_1AddV2Bsequential_5_lstm_11_while_sequential_5_lstm_11_while_loop_counter+sequential_5/lstm_11/while/add_1/y:output:0*
T0*
_output_shapes
: 2"
 sequential_5/lstm_11/while/add_1┐
#sequential_5/lstm_11/while/IdentityIdentity$sequential_5/lstm_11/while/add_1:z:0 ^sequential_5/lstm_11/while/NoOp*
T0*
_output_shapes
: 2%
#sequential_5/lstm_11/while/Identityч
%sequential_5/lstm_11/while/Identity_1IdentityHsequential_5_lstm_11_while_sequential_5_lstm_11_while_maximum_iterations ^sequential_5/lstm_11/while/NoOp*
T0*
_output_shapes
: 2'
%sequential_5/lstm_11/while/Identity_1┴
%sequential_5/lstm_11/while/Identity_2Identity"sequential_5/lstm_11/while/add:z:0 ^sequential_5/lstm_11/while/NoOp*
T0*
_output_shapes
: 2'
%sequential_5/lstm_11/while/Identity_2ю
%sequential_5/lstm_11/while/Identity_3IdentityOsequential_5/lstm_11/while/TensorArrayV2Write/TensorListSetItem:output_handle:0 ^sequential_5/lstm_11/while/NoOp*
T0*
_output_shapes
: 2'
%sequential_5/lstm_11/while/Identity_3т
%sequential_5/lstm_11/while/Identity_4Identity1sequential_5/lstm_11/while/lstm_cell_11/mul_2:z:0 ^sequential_5/lstm_11/while/NoOp*
T0*(
_output_shapes
:         в2'
%sequential_5/lstm_11/while/Identity_4т
%sequential_5/lstm_11/while/Identity_5Identity1sequential_5/lstm_11/while/lstm_cell_11/add_1:z:0 ^sequential_5/lstm_11/while/NoOp*
T0*(
_output_shapes
:         в2'
%sequential_5/lstm_11/while/Identity_5╟
sequential_5/lstm_11/while/NoOpNoOp?^sequential_5/lstm_11/while/lstm_cell_11/BiasAdd/ReadVariableOp>^sequential_5/lstm_11/while/lstm_cell_11/MatMul/ReadVariableOp@^sequential_5/lstm_11/while/lstm_cell_11/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2!
sequential_5/lstm_11/while/NoOp"S
#sequential_5_lstm_11_while_identity,sequential_5/lstm_11/while/Identity:output:0"W
%sequential_5_lstm_11_while_identity_1.sequential_5/lstm_11/while/Identity_1:output:0"W
%sequential_5_lstm_11_while_identity_2.sequential_5/lstm_11/while/Identity_2:output:0"W
%sequential_5_lstm_11_while_identity_3.sequential_5/lstm_11/while/Identity_3:output:0"W
%sequential_5_lstm_11_while_identity_4.sequential_5/lstm_11/while/Identity_4:output:0"W
%sequential_5_lstm_11_while_identity_5.sequential_5/lstm_11/while/Identity_5:output:0"Ф
Gsequential_5_lstm_11_while_lstm_cell_11_biasadd_readvariableop_resourceIsequential_5_lstm_11_while_lstm_cell_11_biasadd_readvariableop_resource_0"Ц
Hsequential_5_lstm_11_while_lstm_cell_11_matmul_1_readvariableop_resourceJsequential_5_lstm_11_while_lstm_cell_11_matmul_1_readvariableop_resource_0"Т
Fsequential_5_lstm_11_while_lstm_cell_11_matmul_readvariableop_resourceHsequential_5_lstm_11_while_lstm_cell_11_matmul_readvariableop_resource_0"Д
?sequential_5_lstm_11_while_sequential_5_lstm_11_strided_slice_1Asequential_5_lstm_11_while_sequential_5_lstm_11_strided_slice_1_0"№
{sequential_5_lstm_11_while_tensorarrayv2read_tensorlistgetitem_sequential_5_lstm_11_tensorarrayunstack_tensorlistfromtensor}sequential_5_lstm_11_while_tensorarrayv2read_tensorlistgetitem_sequential_5_lstm_11_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :         в:         в: : : : : 2А
>sequential_5/lstm_11/while/lstm_cell_11/BiasAdd/ReadVariableOp>sequential_5/lstm_11/while/lstm_cell_11/BiasAdd/ReadVariableOp2~
=sequential_5/lstm_11/while/lstm_cell_11/MatMul/ReadVariableOp=sequential_5/lstm_11/while/lstm_cell_11/MatMul/ReadVariableOp2В
?sequential_5/lstm_11/while/lstm_cell_11/MatMul_1/ReadVariableOp?sequential_5/lstm_11/while/lstm_cell_11/MatMul_1/ReadVariableOp: 
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
:         в:.*
(
_output_shapes
:         в:

_output_shapes
: :

_output_shapes
: 
ї
Ж
J__inference_lstm_cell_10_layer_call_and_return_conditional_losses_19545375

inputs

states
states_11
matmul_readvariableop_resource:	]Ь3
 matmul_1_readvariableop_resource:	GЬ.
biasadd_readvariableop_resource:	Ь
identity

identity_1

identity_2ИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpвMatMul_1/ReadVariableOpО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	]Ь*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ь2
MatMulФ
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	GЬ*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ь2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:         Ь2
addН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:Ь*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ь2	
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
L:         G:         G:         G:         G*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:         G2	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:         G2
	Sigmoid_1\
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:         G2
mulV
ReluRelusplit:output:2*
T0*'
_output_shapes
:         G2
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:         G2
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:         G2
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:         G2
	Sigmoid_2U
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:         G2
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:         G2
mul_2d
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:         G2

Identityh

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:         G2

Identity_1h

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:         G2

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
?:         ]:         G:         G: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:         ]
 
_user_specified_nameinputs:OK
'
_output_shapes
:         G
 
_user_specified_namestates:OK
'
_output_shapes
:         G
 
_user_specified_namestates
ї
Ж
J__inference_lstm_cell_10_layer_call_and_return_conditional_losses_19545229

inputs

states
states_11
matmul_readvariableop_resource:	]Ь3
 matmul_1_readvariableop_resource:	GЬ.
biasadd_readvariableop_resource:	Ь
identity

identity_1

identity_2ИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpвMatMul_1/ReadVariableOpО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	]Ь*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ь2
MatMulФ
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	GЬ*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ь2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:         Ь2
addН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:Ь*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ь2	
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
L:         G:         G:         G:         G*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:         G2	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:         G2
	Sigmoid_1\
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:         G2
mulV
ReluRelusplit:output:2*
T0*'
_output_shapes
:         G2
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:         G2
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:         G2
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:         G2
	Sigmoid_2U
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:         G2
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:         G2
mul_2d
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:         G2

Identityh

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:         G2

Identity_1h

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:         G2

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
?:         ]:         G:         G: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:         ]
 
_user_specified_nameinputs:OK
'
_output_shapes
:         G
 
_user_specified_namestates:OK
'
_output_shapes
:         G
 
_user_specified_namestates
у
═
while_cond_19548973
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_19548973___redundant_placeholder06
2while_while_cond_19548973___redundant_placeholder16
2while_while_cond_19548973___redundant_placeholder26
2while_while_cond_19548973___redundant_placeholder3
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
B: : : : :         в:         в: ::::: 
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
:         в:.*
(
_output_shapes
:         в:

_output_shapes
: :

_output_shapes
:
╘!
¤
E__inference_dense_5_layer_call_and_return_conditional_losses_19549462

inputs4
!tensordot_readvariableop_resource:	в-
biasadd_readvariableop_resource:
identityИвBiasAdd/ReadVariableOpвTensordot/ReadVariableOpЧ
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes
:	в*
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
:         в2
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
:         в: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:T P
,
_output_shapes
:         в
 
_user_specified_nameinputs
Л
З
J__inference_lstm_cell_11_layer_call_and_return_conditional_losses_19545859

inputs

states
states_11
matmul_readvariableop_resource:	GИ4
 matmul_1_readvariableop_resource:
вИ.
biasadd_readvariableop_resource:	И
identity

identity_1

identity_2ИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpвMatMul_1/ReadVariableOpО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	GИ*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         И2
MatMulХ
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
вИ*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         И2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:         И2
addН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:И*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         И2	
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
P:         в:         в:         в:         в*
	num_split2
split`
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:         в2	
Sigmoidd
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:         в2
	Sigmoid_1]
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:         в2
mulW
ReluRelusplit:output:2*
T0*(
_output_shapes
:         в2
Relui
mul_1MulSigmoid:y:0Relu:activations:0*
T0*(
_output_shapes
:         в2
mul_1^
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:         в2
add_1d
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:         в2
	Sigmoid_2V
Relu_1Relu	add_1:z:0*
T0*(
_output_shapes
:         в2
Relu_1m
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*(
_output_shapes
:         в2
mul_2e
IdentityIdentity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:         в2

Identityi

Identity_1Identity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:         в2

Identity_1i

Identity_2Identity	add_1:z:0^NoOp*
T0*(
_output_shapes
:         в2

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
A:         G:         в:         в: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:         G
 
_user_specified_nameinputs:PL
(
_output_shapes
:         в
 
_user_specified_namestates:PL
(
_output_shapes
:         в
 
_user_specified_namestates
╫
g
H__inference_dropout_11_layer_call_and_return_conditional_losses_19546838

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
:         в2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape╣
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:         в*
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
:         в2
dropout/GreaterEqualД
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:         в2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:         в2
dropout/Mul_1j
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:         в2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         в:T P
,
_output_shapes
:         в
 
_user_specified_nameinputs
к

╤
/__inference_sequential_5_layer_call_fn_19546808
lstm_10_input
unknown:	]Ь
	unknown_0:	GЬ
	unknown_1:	Ь
	unknown_2:	GИ
	unknown_3:
вИ
	unknown_4:	И
	unknown_5:	в
	unknown_6:
identityИвStatefulPartitionedCall╙
StatefulPartitionedCallStatefulPartitionedCalllstm_10_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
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
J__inference_sequential_5_layer_call_and_return_conditional_losses_195467892
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
_user_specified_namelstm_10_input
Ъ?
╥
while_body_19548601
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_10_matmul_readvariableop_resource_0:	]ЬH
5while_lstm_cell_10_matmul_1_readvariableop_resource_0:	GЬC
4while_lstm_cell_10_biasadd_readvariableop_resource_0:	Ь
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_10_matmul_readvariableop_resource:	]ЬF
3while_lstm_cell_10_matmul_1_readvariableop_resource:	GЬA
2while_lstm_cell_10_biasadd_readvariableop_resource:	ЬИв)while/lstm_cell_10/BiasAdd/ReadVariableOpв(while/lstm_cell_10/MatMul/ReadVariableOpв*while/lstm_cell_10/MatMul_1/ReadVariableOp├
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
(while/lstm_cell_10/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_10_matmul_readvariableop_resource_0*
_output_shapes
:	]Ь*
dtype02*
(while/lstm_cell_10/MatMul/ReadVariableOp╫
while/lstm_cell_10/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_10/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ь2
while/lstm_cell_10/MatMul╧
*while/lstm_cell_10/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_10_matmul_1_readvariableop_resource_0*
_output_shapes
:	GЬ*
dtype02,
*while/lstm_cell_10/MatMul_1/ReadVariableOp└
while/lstm_cell_10/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_10/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ь2
while/lstm_cell_10/MatMul_1╕
while/lstm_cell_10/addAddV2#while/lstm_cell_10/MatMul:product:0%while/lstm_cell_10/MatMul_1:product:0*
T0*(
_output_shapes
:         Ь2
while/lstm_cell_10/add╚
)while/lstm_cell_10/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_10_biasadd_readvariableop_resource_0*
_output_shapes	
:Ь*
dtype02+
)while/lstm_cell_10/BiasAdd/ReadVariableOp┼
while/lstm_cell_10/BiasAddBiasAddwhile/lstm_cell_10/add:z:01while/lstm_cell_10/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ь2
while/lstm_cell_10/BiasAddК
"while/lstm_cell_10/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_10/split/split_dimЛ
while/lstm_cell_10/splitSplit+while/lstm_cell_10/split/split_dim:output:0#while/lstm_cell_10/BiasAdd:output:0*
T0*`
_output_shapesN
L:         G:         G:         G:         G*
	num_split2
while/lstm_cell_10/splitШ
while/lstm_cell_10/SigmoidSigmoid!while/lstm_cell_10/split:output:0*
T0*'
_output_shapes
:         G2
while/lstm_cell_10/SigmoidЬ
while/lstm_cell_10/Sigmoid_1Sigmoid!while/lstm_cell_10/split:output:1*
T0*'
_output_shapes
:         G2
while/lstm_cell_10/Sigmoid_1а
while/lstm_cell_10/mulMul while/lstm_cell_10/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:         G2
while/lstm_cell_10/mulП
while/lstm_cell_10/ReluRelu!while/lstm_cell_10/split:output:2*
T0*'
_output_shapes
:         G2
while/lstm_cell_10/Relu┤
while/lstm_cell_10/mul_1Mulwhile/lstm_cell_10/Sigmoid:y:0%while/lstm_cell_10/Relu:activations:0*
T0*'
_output_shapes
:         G2
while/lstm_cell_10/mul_1й
while/lstm_cell_10/add_1AddV2while/lstm_cell_10/mul:z:0while/lstm_cell_10/mul_1:z:0*
T0*'
_output_shapes
:         G2
while/lstm_cell_10/add_1Ь
while/lstm_cell_10/Sigmoid_2Sigmoid!while/lstm_cell_10/split:output:3*
T0*'
_output_shapes
:         G2
while/lstm_cell_10/Sigmoid_2О
while/lstm_cell_10/Relu_1Reluwhile/lstm_cell_10/add_1:z:0*
T0*'
_output_shapes
:         G2
while/lstm_cell_10/Relu_1╕
while/lstm_cell_10/mul_2Mul while/lstm_cell_10/Sigmoid_2:y:0'while/lstm_cell_10/Relu_1:activations:0*
T0*'
_output_shapes
:         G2
while/lstm_cell_10/mul_2р
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_10/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_10/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:         G2
while/Identity_4Н
while/Identity_5Identitywhile/lstm_cell_10/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:         G2
while/Identity_5▐

while/NoOpNoOp*^while/lstm_cell_10/BiasAdd/ReadVariableOp)^while/lstm_cell_10/MatMul/ReadVariableOp+^while/lstm_cell_10/MatMul_1/ReadVariableOp*"
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
2while_lstm_cell_10_biasadd_readvariableop_resource4while_lstm_cell_10_biasadd_readvariableop_resource_0"l
3while_lstm_cell_10_matmul_1_readvariableop_resource5while_lstm_cell_10_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_10_matmul_readvariableop_resource3while_lstm_cell_10_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         G:         G: : : : : 2V
)while/lstm_cell_10/BiasAdd/ReadVariableOp)while/lstm_cell_10/BiasAdd/ReadVariableOp2T
(while/lstm_cell_10/MatMul/ReadVariableOp(while/lstm_cell_10/MatMul/ReadVariableOp2X
*while/lstm_cell_10/MatMul_1/ReadVariableOp*while/lstm_cell_10/MatMul_1/ReadVariableOp: 
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
:         G:-)
'
_output_shapes
:         G:

_output_shapes
: :

_output_shapes
: 
·%
ё
while_body_19545243
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_00
while_lstm_cell_10_19545267_0:	]Ь0
while_lstm_cell_10_19545269_0:	GЬ,
while_lstm_cell_10_19545271_0:	Ь
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor.
while_lstm_cell_10_19545267:	]Ь.
while_lstm_cell_10_19545269:	GЬ*
while_lstm_cell_10_19545271:	ЬИв*while/lstm_cell_10/StatefulPartitionedCall├
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
*while/lstm_cell_10/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_10_19545267_0while_lstm_cell_10_19545269_0while_lstm_cell_10_19545271_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:         G:         G:         G*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_lstm_cell_10_layer_call_and_return_conditional_losses_195452292,
*while/lstm_cell_10/StatefulPartitionedCallў
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder3while/lstm_cell_10/StatefulPartitionedCall:output:0*
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
while/Identity_4Identity3while/lstm_cell_10/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:         G2
while/Identity_4д
while/Identity_5Identity3while/lstm_cell_10/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:         G2
while/Identity_5З

while/NoOpNoOp+^while/lstm_cell_10/StatefulPartitionedCall*"
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
while_lstm_cell_10_19545267while_lstm_cell_10_19545267_0"<
while_lstm_cell_10_19545269while_lstm_cell_10_19545269_0"<
while_lstm_cell_10_19545271while_lstm_cell_10_19545271_0"0
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         G:         G: : : : : 2X
*while/lstm_cell_10/StatefulPartitionedCall*while/lstm_cell_10/StatefulPartitionedCall: 
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
:         G:-)
'
_output_shapes
:         G:

_output_shapes
: :

_output_shapes
: 
▓
╖
*__inference_lstm_10_layer_call_fn_19548729

inputs
unknown:	]Ь
	unknown_0:	GЬ
	unknown_1:	Ь
identityИвStatefulPartitionedCallЖ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         G*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_lstm_10_layer_call_and_return_conditional_losses_195472012
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:         G2

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
ж\
Я
E__inference_lstm_10_layer_call_and_return_conditional_losses_19548383
inputs_0>
+lstm_cell_10_matmul_readvariableop_resource:	]Ь@
-lstm_cell_10_matmul_1_readvariableop_resource:	GЬ;
,lstm_cell_10_biasadd_readvariableop_resource:	Ь
identityИв#lstm_cell_10/BiasAdd/ReadVariableOpв"lstm_cell_10/MatMul/ReadVariableOpв$lstm_cell_10/MatMul_1/ReadVariableOpвwhileF
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
value	B :G2
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
value	B :G2
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
:         G2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :G2
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
value	B :G2
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
:         G2	
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
"lstm_cell_10/MatMul/ReadVariableOpReadVariableOp+lstm_cell_10_matmul_readvariableop_resource*
_output_shapes
:	]Ь*
dtype02$
"lstm_cell_10/MatMul/ReadVariableOpн
lstm_cell_10/MatMulMatMulstrided_slice_2:output:0*lstm_cell_10/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ь2
lstm_cell_10/MatMul╗
$lstm_cell_10/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_10_matmul_1_readvariableop_resource*
_output_shapes
:	GЬ*
dtype02&
$lstm_cell_10/MatMul_1/ReadVariableOpй
lstm_cell_10/MatMul_1MatMulzeros:output:0,lstm_cell_10/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ь2
lstm_cell_10/MatMul_1а
lstm_cell_10/addAddV2lstm_cell_10/MatMul:product:0lstm_cell_10/MatMul_1:product:0*
T0*(
_output_shapes
:         Ь2
lstm_cell_10/add┤
#lstm_cell_10/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_10_biasadd_readvariableop_resource*
_output_shapes	
:Ь*
dtype02%
#lstm_cell_10/BiasAdd/ReadVariableOpн
lstm_cell_10/BiasAddBiasAddlstm_cell_10/add:z:0+lstm_cell_10/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ь2
lstm_cell_10/BiasAdd~
lstm_cell_10/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_10/split/split_dimє
lstm_cell_10/splitSplit%lstm_cell_10/split/split_dim:output:0lstm_cell_10/BiasAdd:output:0*
T0*`
_output_shapesN
L:         G:         G:         G:         G*
	num_split2
lstm_cell_10/splitЖ
lstm_cell_10/SigmoidSigmoidlstm_cell_10/split:output:0*
T0*'
_output_shapes
:         G2
lstm_cell_10/SigmoidК
lstm_cell_10/Sigmoid_1Sigmoidlstm_cell_10/split:output:1*
T0*'
_output_shapes
:         G2
lstm_cell_10/Sigmoid_1Л
lstm_cell_10/mulMullstm_cell_10/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:         G2
lstm_cell_10/mul}
lstm_cell_10/ReluRelulstm_cell_10/split:output:2*
T0*'
_output_shapes
:         G2
lstm_cell_10/ReluЬ
lstm_cell_10/mul_1Mullstm_cell_10/Sigmoid:y:0lstm_cell_10/Relu:activations:0*
T0*'
_output_shapes
:         G2
lstm_cell_10/mul_1С
lstm_cell_10/add_1AddV2lstm_cell_10/mul:z:0lstm_cell_10/mul_1:z:0*
T0*'
_output_shapes
:         G2
lstm_cell_10/add_1К
lstm_cell_10/Sigmoid_2Sigmoidlstm_cell_10/split:output:3*
T0*'
_output_shapes
:         G2
lstm_cell_10/Sigmoid_2|
lstm_cell_10/Relu_1Relulstm_cell_10/add_1:z:0*
T0*'
_output_shapes
:         G2
lstm_cell_10/Relu_1а
lstm_cell_10/mul_2Mullstm_cell_10/Sigmoid_2:y:0!lstm_cell_10/Relu_1:activations:0*
T0*'
_output_shapes
:         G2
lstm_cell_10/mul_2П
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    G   2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_10_matmul_readvariableop_resource-lstm_cell_10_matmul_1_readvariableop_resource,lstm_cell_10_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         G:         G: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_19548299*
condR
while_cond_19548298*K
output_shapes:
8: : : : :         G:         G: : : : : *
parallel_iterations 2
while╡
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    G   22
0TensorArrayV2Stack/TensorListStack/element_shapeё
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :                  G*
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
:         G*
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
 :                  G2
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
 :                  G2

Identity╚
NoOpNoOp$^lstm_cell_10/BiasAdd/ReadVariableOp#^lstm_cell_10/MatMul/ReadVariableOp%^lstm_cell_10/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  ]: : : 2J
#lstm_cell_10/BiasAdd/ReadVariableOp#lstm_cell_10/BiasAdd/ReadVariableOp2H
"lstm_cell_10/MatMul/ReadVariableOp"lstm_cell_10/MatMul/ReadVariableOp2L
$lstm_cell_10/MatMul_1/ReadVariableOp$lstm_cell_10/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :                  ]
"
_user_specified_name
inputs/0
Е
f
H__inference_dropout_10_layer_call_and_return_conditional_losses_19546584

inputs

identity_1^
IdentityIdentityinputs*
T0*+
_output_shapes
:         G2

Identitym

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:         G2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         G:S O
+
_output_shapes
:         G
 
_user_specified_nameinputs
ж\
Я
E__inference_lstm_10_layer_call_and_return_conditional_losses_19548232
inputs_0>
+lstm_cell_10_matmul_readvariableop_resource:	]Ь@
-lstm_cell_10_matmul_1_readvariableop_resource:	GЬ;
,lstm_cell_10_biasadd_readvariableop_resource:	Ь
identityИв#lstm_cell_10/BiasAdd/ReadVariableOpв"lstm_cell_10/MatMul/ReadVariableOpв$lstm_cell_10/MatMul_1/ReadVariableOpвwhileF
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
value	B :G2
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
value	B :G2
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
:         G2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :G2
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
value	B :G2
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
:         G2	
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
"lstm_cell_10/MatMul/ReadVariableOpReadVariableOp+lstm_cell_10_matmul_readvariableop_resource*
_output_shapes
:	]Ь*
dtype02$
"lstm_cell_10/MatMul/ReadVariableOpн
lstm_cell_10/MatMulMatMulstrided_slice_2:output:0*lstm_cell_10/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ь2
lstm_cell_10/MatMul╗
$lstm_cell_10/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_10_matmul_1_readvariableop_resource*
_output_shapes
:	GЬ*
dtype02&
$lstm_cell_10/MatMul_1/ReadVariableOpй
lstm_cell_10/MatMul_1MatMulzeros:output:0,lstm_cell_10/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ь2
lstm_cell_10/MatMul_1а
lstm_cell_10/addAddV2lstm_cell_10/MatMul:product:0lstm_cell_10/MatMul_1:product:0*
T0*(
_output_shapes
:         Ь2
lstm_cell_10/add┤
#lstm_cell_10/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_10_biasadd_readvariableop_resource*
_output_shapes	
:Ь*
dtype02%
#lstm_cell_10/BiasAdd/ReadVariableOpн
lstm_cell_10/BiasAddBiasAddlstm_cell_10/add:z:0+lstm_cell_10/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ь2
lstm_cell_10/BiasAdd~
lstm_cell_10/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_10/split/split_dimє
lstm_cell_10/splitSplit%lstm_cell_10/split/split_dim:output:0lstm_cell_10/BiasAdd:output:0*
T0*`
_output_shapesN
L:         G:         G:         G:         G*
	num_split2
lstm_cell_10/splitЖ
lstm_cell_10/SigmoidSigmoidlstm_cell_10/split:output:0*
T0*'
_output_shapes
:         G2
lstm_cell_10/SigmoidК
lstm_cell_10/Sigmoid_1Sigmoidlstm_cell_10/split:output:1*
T0*'
_output_shapes
:         G2
lstm_cell_10/Sigmoid_1Л
lstm_cell_10/mulMullstm_cell_10/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:         G2
lstm_cell_10/mul}
lstm_cell_10/ReluRelulstm_cell_10/split:output:2*
T0*'
_output_shapes
:         G2
lstm_cell_10/ReluЬ
lstm_cell_10/mul_1Mullstm_cell_10/Sigmoid:y:0lstm_cell_10/Relu:activations:0*
T0*'
_output_shapes
:         G2
lstm_cell_10/mul_1С
lstm_cell_10/add_1AddV2lstm_cell_10/mul:z:0lstm_cell_10/mul_1:z:0*
T0*'
_output_shapes
:         G2
lstm_cell_10/add_1К
lstm_cell_10/Sigmoid_2Sigmoidlstm_cell_10/split:output:3*
T0*'
_output_shapes
:         G2
lstm_cell_10/Sigmoid_2|
lstm_cell_10/Relu_1Relulstm_cell_10/add_1:z:0*
T0*'
_output_shapes
:         G2
lstm_cell_10/Relu_1а
lstm_cell_10/mul_2Mullstm_cell_10/Sigmoid_2:y:0!lstm_cell_10/Relu_1:activations:0*
T0*'
_output_shapes
:         G2
lstm_cell_10/mul_2П
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    G   2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_10_matmul_readvariableop_resource-lstm_cell_10_matmul_1_readvariableop_resource,lstm_cell_10_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         G:         G: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_19548148*
condR
while_cond_19548147*K
output_shapes:
8: : : : :         G:         G: : : : : *
parallel_iterations 2
while╡
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    G   22
0TensorArrayV2Stack/TensorListStack/element_shapeё
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :                  G*
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
:         G*
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
 :                  G2
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
 :                  G2

Identity╚
NoOpNoOp$^lstm_cell_10/BiasAdd/ReadVariableOp#^lstm_cell_10/MatMul/ReadVariableOp%^lstm_cell_10/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  ]: : : 2J
#lstm_cell_10/BiasAdd/ReadVariableOp#lstm_cell_10/BiasAdd/ReadVariableOp2H
"lstm_cell_10/MatMul/ReadVariableOp"lstm_cell_10/MatMul/ReadVariableOp2L
$lstm_cell_10/MatMul_1/ReadVariableOp$lstm_cell_10/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :                  ]
"
_user_specified_name
inputs/0
·	
╚
&__inference_signature_wrapper_19547371
lstm_10_input
unknown:	]Ь
	unknown_0:	GЬ
	unknown_1:	Ь
	unknown_2:	GИ
	unknown_3:
вИ
	unknown_4:	И
	unknown_5:	в
	unknown_6:
identityИвStatefulPartitionedCallм
StatefulPartitionedCallStatefulPartitionedCalllstm_10_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
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
#__inference__wrapped_model_195451542
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
_user_specified_namelstm_10_input
Ц]
є
(sequential_5_lstm_10_while_body_19544894F
Bsequential_5_lstm_10_while_sequential_5_lstm_10_while_loop_counterL
Hsequential_5_lstm_10_while_sequential_5_lstm_10_while_maximum_iterations*
&sequential_5_lstm_10_while_placeholder,
(sequential_5_lstm_10_while_placeholder_1,
(sequential_5_lstm_10_while_placeholder_2,
(sequential_5_lstm_10_while_placeholder_3E
Asequential_5_lstm_10_while_sequential_5_lstm_10_strided_slice_1_0Б
}sequential_5_lstm_10_while_tensorarrayv2read_tensorlistgetitem_sequential_5_lstm_10_tensorarrayunstack_tensorlistfromtensor_0[
Hsequential_5_lstm_10_while_lstm_cell_10_matmul_readvariableop_resource_0:	]Ь]
Jsequential_5_lstm_10_while_lstm_cell_10_matmul_1_readvariableop_resource_0:	GЬX
Isequential_5_lstm_10_while_lstm_cell_10_biasadd_readvariableop_resource_0:	Ь'
#sequential_5_lstm_10_while_identity)
%sequential_5_lstm_10_while_identity_1)
%sequential_5_lstm_10_while_identity_2)
%sequential_5_lstm_10_while_identity_3)
%sequential_5_lstm_10_while_identity_4)
%sequential_5_lstm_10_while_identity_5C
?sequential_5_lstm_10_while_sequential_5_lstm_10_strided_slice_1
{sequential_5_lstm_10_while_tensorarrayv2read_tensorlistgetitem_sequential_5_lstm_10_tensorarrayunstack_tensorlistfromtensorY
Fsequential_5_lstm_10_while_lstm_cell_10_matmul_readvariableop_resource:	]Ь[
Hsequential_5_lstm_10_while_lstm_cell_10_matmul_1_readvariableop_resource:	GЬV
Gsequential_5_lstm_10_while_lstm_cell_10_biasadd_readvariableop_resource:	ЬИв>sequential_5/lstm_10/while/lstm_cell_10/BiasAdd/ReadVariableOpв=sequential_5/lstm_10/while/lstm_cell_10/MatMul/ReadVariableOpв?sequential_5/lstm_10/while/lstm_cell_10/MatMul_1/ReadVariableOpэ
Lsequential_5/lstm_10/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ]   2N
Lsequential_5/lstm_10/while/TensorArrayV2Read/TensorListGetItem/element_shape╤
>sequential_5/lstm_10/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem}sequential_5_lstm_10_while_tensorarrayv2read_tensorlistgetitem_sequential_5_lstm_10_tensorarrayunstack_tensorlistfromtensor_0&sequential_5_lstm_10_while_placeholderUsequential_5/lstm_10/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         ]*
element_dtype02@
>sequential_5/lstm_10/while/TensorArrayV2Read/TensorListGetItemИ
=sequential_5/lstm_10/while/lstm_cell_10/MatMul/ReadVariableOpReadVariableOpHsequential_5_lstm_10_while_lstm_cell_10_matmul_readvariableop_resource_0*
_output_shapes
:	]Ь*
dtype02?
=sequential_5/lstm_10/while/lstm_cell_10/MatMul/ReadVariableOpл
.sequential_5/lstm_10/while/lstm_cell_10/MatMulMatMulEsequential_5/lstm_10/while/TensorArrayV2Read/TensorListGetItem:item:0Esequential_5/lstm_10/while/lstm_cell_10/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ь20
.sequential_5/lstm_10/while/lstm_cell_10/MatMulО
?sequential_5/lstm_10/while/lstm_cell_10/MatMul_1/ReadVariableOpReadVariableOpJsequential_5_lstm_10_while_lstm_cell_10_matmul_1_readvariableop_resource_0*
_output_shapes
:	GЬ*
dtype02A
?sequential_5/lstm_10/while/lstm_cell_10/MatMul_1/ReadVariableOpФ
0sequential_5/lstm_10/while/lstm_cell_10/MatMul_1MatMul(sequential_5_lstm_10_while_placeholder_2Gsequential_5/lstm_10/while/lstm_cell_10/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ь22
0sequential_5/lstm_10/while/lstm_cell_10/MatMul_1М
+sequential_5/lstm_10/while/lstm_cell_10/addAddV28sequential_5/lstm_10/while/lstm_cell_10/MatMul:product:0:sequential_5/lstm_10/while/lstm_cell_10/MatMul_1:product:0*
T0*(
_output_shapes
:         Ь2-
+sequential_5/lstm_10/while/lstm_cell_10/addЗ
>sequential_5/lstm_10/while/lstm_cell_10/BiasAdd/ReadVariableOpReadVariableOpIsequential_5_lstm_10_while_lstm_cell_10_biasadd_readvariableop_resource_0*
_output_shapes	
:Ь*
dtype02@
>sequential_5/lstm_10/while/lstm_cell_10/BiasAdd/ReadVariableOpЩ
/sequential_5/lstm_10/while/lstm_cell_10/BiasAddBiasAdd/sequential_5/lstm_10/while/lstm_cell_10/add:z:0Fsequential_5/lstm_10/while/lstm_cell_10/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ь21
/sequential_5/lstm_10/while/lstm_cell_10/BiasAdd┤
7sequential_5/lstm_10/while/lstm_cell_10/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :29
7sequential_5/lstm_10/while/lstm_cell_10/split/split_dim▀
-sequential_5/lstm_10/while/lstm_cell_10/splitSplit@sequential_5/lstm_10/while/lstm_cell_10/split/split_dim:output:08sequential_5/lstm_10/while/lstm_cell_10/BiasAdd:output:0*
T0*`
_output_shapesN
L:         G:         G:         G:         G*
	num_split2/
-sequential_5/lstm_10/while/lstm_cell_10/split╫
/sequential_5/lstm_10/while/lstm_cell_10/SigmoidSigmoid6sequential_5/lstm_10/while/lstm_cell_10/split:output:0*
T0*'
_output_shapes
:         G21
/sequential_5/lstm_10/while/lstm_cell_10/Sigmoid█
1sequential_5/lstm_10/while/lstm_cell_10/Sigmoid_1Sigmoid6sequential_5/lstm_10/while/lstm_cell_10/split:output:1*
T0*'
_output_shapes
:         G23
1sequential_5/lstm_10/while/lstm_cell_10/Sigmoid_1Ї
+sequential_5/lstm_10/while/lstm_cell_10/mulMul5sequential_5/lstm_10/while/lstm_cell_10/Sigmoid_1:y:0(sequential_5_lstm_10_while_placeholder_3*
T0*'
_output_shapes
:         G2-
+sequential_5/lstm_10/while/lstm_cell_10/mul╬
,sequential_5/lstm_10/while/lstm_cell_10/ReluRelu6sequential_5/lstm_10/while/lstm_cell_10/split:output:2*
T0*'
_output_shapes
:         G2.
,sequential_5/lstm_10/while/lstm_cell_10/ReluИ
-sequential_5/lstm_10/while/lstm_cell_10/mul_1Mul3sequential_5/lstm_10/while/lstm_cell_10/Sigmoid:y:0:sequential_5/lstm_10/while/lstm_cell_10/Relu:activations:0*
T0*'
_output_shapes
:         G2/
-sequential_5/lstm_10/while/lstm_cell_10/mul_1¤
-sequential_5/lstm_10/while/lstm_cell_10/add_1AddV2/sequential_5/lstm_10/while/lstm_cell_10/mul:z:01sequential_5/lstm_10/while/lstm_cell_10/mul_1:z:0*
T0*'
_output_shapes
:         G2/
-sequential_5/lstm_10/while/lstm_cell_10/add_1█
1sequential_5/lstm_10/while/lstm_cell_10/Sigmoid_2Sigmoid6sequential_5/lstm_10/while/lstm_cell_10/split:output:3*
T0*'
_output_shapes
:         G23
1sequential_5/lstm_10/while/lstm_cell_10/Sigmoid_2═
.sequential_5/lstm_10/while/lstm_cell_10/Relu_1Relu1sequential_5/lstm_10/while/lstm_cell_10/add_1:z:0*
T0*'
_output_shapes
:         G20
.sequential_5/lstm_10/while/lstm_cell_10/Relu_1М
-sequential_5/lstm_10/while/lstm_cell_10/mul_2Mul5sequential_5/lstm_10/while/lstm_cell_10/Sigmoid_2:y:0<sequential_5/lstm_10/while/lstm_cell_10/Relu_1:activations:0*
T0*'
_output_shapes
:         G2/
-sequential_5/lstm_10/while/lstm_cell_10/mul_2╔
?sequential_5/lstm_10/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem(sequential_5_lstm_10_while_placeholder_1&sequential_5_lstm_10_while_placeholder1sequential_5/lstm_10/while/lstm_cell_10/mul_2:z:0*
_output_shapes
: *
element_dtype02A
?sequential_5/lstm_10/while/TensorArrayV2Write/TensorListSetItemЖ
 sequential_5/lstm_10/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2"
 sequential_5/lstm_10/while/add/y╜
sequential_5/lstm_10/while/addAddV2&sequential_5_lstm_10_while_placeholder)sequential_5/lstm_10/while/add/y:output:0*
T0*
_output_shapes
: 2 
sequential_5/lstm_10/while/addК
"sequential_5/lstm_10/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2$
"sequential_5/lstm_10/while/add_1/y▀
 sequential_5/lstm_10/while/add_1AddV2Bsequential_5_lstm_10_while_sequential_5_lstm_10_while_loop_counter+sequential_5/lstm_10/while/add_1/y:output:0*
T0*
_output_shapes
: 2"
 sequential_5/lstm_10/while/add_1┐
#sequential_5/lstm_10/while/IdentityIdentity$sequential_5/lstm_10/while/add_1:z:0 ^sequential_5/lstm_10/while/NoOp*
T0*
_output_shapes
: 2%
#sequential_5/lstm_10/while/Identityч
%sequential_5/lstm_10/while/Identity_1IdentityHsequential_5_lstm_10_while_sequential_5_lstm_10_while_maximum_iterations ^sequential_5/lstm_10/while/NoOp*
T0*
_output_shapes
: 2'
%sequential_5/lstm_10/while/Identity_1┴
%sequential_5/lstm_10/while/Identity_2Identity"sequential_5/lstm_10/while/add:z:0 ^sequential_5/lstm_10/while/NoOp*
T0*
_output_shapes
: 2'
%sequential_5/lstm_10/while/Identity_2ю
%sequential_5/lstm_10/while/Identity_3IdentityOsequential_5/lstm_10/while/TensorArrayV2Write/TensorListSetItem:output_handle:0 ^sequential_5/lstm_10/while/NoOp*
T0*
_output_shapes
: 2'
%sequential_5/lstm_10/while/Identity_3с
%sequential_5/lstm_10/while/Identity_4Identity1sequential_5/lstm_10/while/lstm_cell_10/mul_2:z:0 ^sequential_5/lstm_10/while/NoOp*
T0*'
_output_shapes
:         G2'
%sequential_5/lstm_10/while/Identity_4с
%sequential_5/lstm_10/while/Identity_5Identity1sequential_5/lstm_10/while/lstm_cell_10/add_1:z:0 ^sequential_5/lstm_10/while/NoOp*
T0*'
_output_shapes
:         G2'
%sequential_5/lstm_10/while/Identity_5╟
sequential_5/lstm_10/while/NoOpNoOp?^sequential_5/lstm_10/while/lstm_cell_10/BiasAdd/ReadVariableOp>^sequential_5/lstm_10/while/lstm_cell_10/MatMul/ReadVariableOp@^sequential_5/lstm_10/while/lstm_cell_10/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2!
sequential_5/lstm_10/while/NoOp"S
#sequential_5_lstm_10_while_identity,sequential_5/lstm_10/while/Identity:output:0"W
%sequential_5_lstm_10_while_identity_1.sequential_5/lstm_10/while/Identity_1:output:0"W
%sequential_5_lstm_10_while_identity_2.sequential_5/lstm_10/while/Identity_2:output:0"W
%sequential_5_lstm_10_while_identity_3.sequential_5/lstm_10/while/Identity_3:output:0"W
%sequential_5_lstm_10_while_identity_4.sequential_5/lstm_10/while/Identity_4:output:0"W
%sequential_5_lstm_10_while_identity_5.sequential_5/lstm_10/while/Identity_5:output:0"Ф
Gsequential_5_lstm_10_while_lstm_cell_10_biasadd_readvariableop_resourceIsequential_5_lstm_10_while_lstm_cell_10_biasadd_readvariableop_resource_0"Ц
Hsequential_5_lstm_10_while_lstm_cell_10_matmul_1_readvariableop_resourceJsequential_5_lstm_10_while_lstm_cell_10_matmul_1_readvariableop_resource_0"Т
Fsequential_5_lstm_10_while_lstm_cell_10_matmul_readvariableop_resourceHsequential_5_lstm_10_while_lstm_cell_10_matmul_readvariableop_resource_0"Д
?sequential_5_lstm_10_while_sequential_5_lstm_10_strided_slice_1Asequential_5_lstm_10_while_sequential_5_lstm_10_strided_slice_1_0"№
{sequential_5_lstm_10_while_tensorarrayv2read_tensorlistgetitem_sequential_5_lstm_10_tensorarrayunstack_tensorlistfromtensor}sequential_5_lstm_10_while_tensorarrayv2read_tensorlistgetitem_sequential_5_lstm_10_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         G:         G: : : : : 2А
>sequential_5/lstm_10/while/lstm_cell_10/BiasAdd/ReadVariableOp>sequential_5/lstm_10/while/lstm_cell_10/BiasAdd/ReadVariableOp2~
=sequential_5/lstm_10/while/lstm_cell_10/MatMul/ReadVariableOp=sequential_5/lstm_10/while/lstm_cell_10/MatMul/ReadVariableOp2В
?sequential_5/lstm_10/while/lstm_cell_10/MatMul_1/ReadVariableOp?sequential_5/lstm_10/while/lstm_cell_10/MatMul_1/ReadVariableOp: 
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
:         G:-)
'
_output_shapes
:         G:

_output_shapes
: :

_output_shapes
: 
ч│
╧	
#__inference__wrapped_model_19545154
lstm_10_inputS
@sequential_5_lstm_10_lstm_cell_10_matmul_readvariableop_resource:	]ЬU
Bsequential_5_lstm_10_lstm_cell_10_matmul_1_readvariableop_resource:	GЬP
Asequential_5_lstm_10_lstm_cell_10_biasadd_readvariableop_resource:	ЬS
@sequential_5_lstm_11_lstm_cell_11_matmul_readvariableop_resource:	GИV
Bsequential_5_lstm_11_lstm_cell_11_matmul_1_readvariableop_resource:
вИP
Asequential_5_lstm_11_lstm_cell_11_biasadd_readvariableop_resource:	ИI
6sequential_5_dense_5_tensordot_readvariableop_resource:	вB
4sequential_5_dense_5_biasadd_readvariableop_resource:
identityИв+sequential_5/dense_5/BiasAdd/ReadVariableOpв-sequential_5/dense_5/Tensordot/ReadVariableOpв8sequential_5/lstm_10/lstm_cell_10/BiasAdd/ReadVariableOpв7sequential_5/lstm_10/lstm_cell_10/MatMul/ReadVariableOpв9sequential_5/lstm_10/lstm_cell_10/MatMul_1/ReadVariableOpвsequential_5/lstm_10/whileв8sequential_5/lstm_11/lstm_cell_11/BiasAdd/ReadVariableOpв7sequential_5/lstm_11/lstm_cell_11/MatMul/ReadVariableOpв9sequential_5/lstm_11/lstm_cell_11/MatMul_1/ReadVariableOpвsequential_5/lstm_11/whileu
sequential_5/lstm_10/ShapeShapelstm_10_input*
T0*
_output_shapes
:2
sequential_5/lstm_10/ShapeЮ
(sequential_5/lstm_10/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(sequential_5/lstm_10/strided_slice/stackв
*sequential_5/lstm_10/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*sequential_5/lstm_10/strided_slice/stack_1в
*sequential_5/lstm_10/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*sequential_5/lstm_10/strided_slice/stack_2р
"sequential_5/lstm_10/strided_sliceStridedSlice#sequential_5/lstm_10/Shape:output:01sequential_5/lstm_10/strided_slice/stack:output:03sequential_5/lstm_10/strided_slice/stack_1:output:03sequential_5/lstm_10/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"sequential_5/lstm_10/strided_sliceЖ
 sequential_5/lstm_10/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :G2"
 sequential_5/lstm_10/zeros/mul/y└
sequential_5/lstm_10/zeros/mulMul+sequential_5/lstm_10/strided_slice:output:0)sequential_5/lstm_10/zeros/mul/y:output:0*
T0*
_output_shapes
: 2 
sequential_5/lstm_10/zeros/mulЙ
!sequential_5/lstm_10/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2#
!sequential_5/lstm_10/zeros/Less/y╗
sequential_5/lstm_10/zeros/LessLess"sequential_5/lstm_10/zeros/mul:z:0*sequential_5/lstm_10/zeros/Less/y:output:0*
T0*
_output_shapes
: 2!
sequential_5/lstm_10/zeros/LessМ
#sequential_5/lstm_10/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :G2%
#sequential_5/lstm_10/zeros/packed/1╫
!sequential_5/lstm_10/zeros/packedPack+sequential_5/lstm_10/strided_slice:output:0,sequential_5/lstm_10/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2#
!sequential_5/lstm_10/zeros/packedЙ
 sequential_5/lstm_10/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 sequential_5/lstm_10/zeros/Const╔
sequential_5/lstm_10/zerosFill*sequential_5/lstm_10/zeros/packed:output:0)sequential_5/lstm_10/zeros/Const:output:0*
T0*'
_output_shapes
:         G2
sequential_5/lstm_10/zerosК
"sequential_5/lstm_10/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :G2$
"sequential_5/lstm_10/zeros_1/mul/y╞
 sequential_5/lstm_10/zeros_1/mulMul+sequential_5/lstm_10/strided_slice:output:0+sequential_5/lstm_10/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2"
 sequential_5/lstm_10/zeros_1/mulН
#sequential_5/lstm_10/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2%
#sequential_5/lstm_10/zeros_1/Less/y├
!sequential_5/lstm_10/zeros_1/LessLess$sequential_5/lstm_10/zeros_1/mul:z:0,sequential_5/lstm_10/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2#
!sequential_5/lstm_10/zeros_1/LessР
%sequential_5/lstm_10/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :G2'
%sequential_5/lstm_10/zeros_1/packed/1▌
#sequential_5/lstm_10/zeros_1/packedPack+sequential_5/lstm_10/strided_slice:output:0.sequential_5/lstm_10/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2%
#sequential_5/lstm_10/zeros_1/packedН
"sequential_5/lstm_10/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"sequential_5/lstm_10/zeros_1/Const╤
sequential_5/lstm_10/zeros_1Fill,sequential_5/lstm_10/zeros_1/packed:output:0+sequential_5/lstm_10/zeros_1/Const:output:0*
T0*'
_output_shapes
:         G2
sequential_5/lstm_10/zeros_1Я
#sequential_5/lstm_10/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2%
#sequential_5/lstm_10/transpose/perm└
sequential_5/lstm_10/transpose	Transposelstm_10_input,sequential_5/lstm_10/transpose/perm:output:0*
T0*+
_output_shapes
:         ]2 
sequential_5/lstm_10/transposeО
sequential_5/lstm_10/Shape_1Shape"sequential_5/lstm_10/transpose:y:0*
T0*
_output_shapes
:2
sequential_5/lstm_10/Shape_1в
*sequential_5/lstm_10/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2,
*sequential_5/lstm_10/strided_slice_1/stackж
,sequential_5/lstm_10/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_5/lstm_10/strided_slice_1/stack_1ж
,sequential_5/lstm_10/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_5/lstm_10/strided_slice_1/stack_2ь
$sequential_5/lstm_10/strided_slice_1StridedSlice%sequential_5/lstm_10/Shape_1:output:03sequential_5/lstm_10/strided_slice_1/stack:output:05sequential_5/lstm_10/strided_slice_1/stack_1:output:05sequential_5/lstm_10/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2&
$sequential_5/lstm_10/strided_slice_1п
0sequential_5/lstm_10/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         22
0sequential_5/lstm_10/TensorArrayV2/element_shapeЖ
"sequential_5/lstm_10/TensorArrayV2TensorListReserve9sequential_5/lstm_10/TensorArrayV2/element_shape:output:0-sequential_5/lstm_10/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02$
"sequential_5/lstm_10/TensorArrayV2щ
Jsequential_5/lstm_10/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ]   2L
Jsequential_5/lstm_10/TensorArrayUnstack/TensorListFromTensor/element_shape╠
<sequential_5/lstm_10/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor"sequential_5/lstm_10/transpose:y:0Ssequential_5/lstm_10/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02>
<sequential_5/lstm_10/TensorArrayUnstack/TensorListFromTensorв
*sequential_5/lstm_10/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2,
*sequential_5/lstm_10/strided_slice_2/stackж
,sequential_5/lstm_10/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_5/lstm_10/strided_slice_2/stack_1ж
,sequential_5/lstm_10/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_5/lstm_10/strided_slice_2/stack_2·
$sequential_5/lstm_10/strided_slice_2StridedSlice"sequential_5/lstm_10/transpose:y:03sequential_5/lstm_10/strided_slice_2/stack:output:05sequential_5/lstm_10/strided_slice_2/stack_1:output:05sequential_5/lstm_10/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         ]*
shrink_axis_mask2&
$sequential_5/lstm_10/strided_slice_2Ї
7sequential_5/lstm_10/lstm_cell_10/MatMul/ReadVariableOpReadVariableOp@sequential_5_lstm_10_lstm_cell_10_matmul_readvariableop_resource*
_output_shapes
:	]Ь*
dtype029
7sequential_5/lstm_10/lstm_cell_10/MatMul/ReadVariableOpБ
(sequential_5/lstm_10/lstm_cell_10/MatMulMatMul-sequential_5/lstm_10/strided_slice_2:output:0?sequential_5/lstm_10/lstm_cell_10/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ь2*
(sequential_5/lstm_10/lstm_cell_10/MatMul·
9sequential_5/lstm_10/lstm_cell_10/MatMul_1/ReadVariableOpReadVariableOpBsequential_5_lstm_10_lstm_cell_10_matmul_1_readvariableop_resource*
_output_shapes
:	GЬ*
dtype02;
9sequential_5/lstm_10/lstm_cell_10/MatMul_1/ReadVariableOp¤
*sequential_5/lstm_10/lstm_cell_10/MatMul_1MatMul#sequential_5/lstm_10/zeros:output:0Asequential_5/lstm_10/lstm_cell_10/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ь2,
*sequential_5/lstm_10/lstm_cell_10/MatMul_1Ї
%sequential_5/lstm_10/lstm_cell_10/addAddV22sequential_5/lstm_10/lstm_cell_10/MatMul:product:04sequential_5/lstm_10/lstm_cell_10/MatMul_1:product:0*
T0*(
_output_shapes
:         Ь2'
%sequential_5/lstm_10/lstm_cell_10/addє
8sequential_5/lstm_10/lstm_cell_10/BiasAdd/ReadVariableOpReadVariableOpAsequential_5_lstm_10_lstm_cell_10_biasadd_readvariableop_resource*
_output_shapes	
:Ь*
dtype02:
8sequential_5/lstm_10/lstm_cell_10/BiasAdd/ReadVariableOpБ
)sequential_5/lstm_10/lstm_cell_10/BiasAddBiasAdd)sequential_5/lstm_10/lstm_cell_10/add:z:0@sequential_5/lstm_10/lstm_cell_10/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ь2+
)sequential_5/lstm_10/lstm_cell_10/BiasAddи
1sequential_5/lstm_10/lstm_cell_10/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :23
1sequential_5/lstm_10/lstm_cell_10/split/split_dim╟
'sequential_5/lstm_10/lstm_cell_10/splitSplit:sequential_5/lstm_10/lstm_cell_10/split/split_dim:output:02sequential_5/lstm_10/lstm_cell_10/BiasAdd:output:0*
T0*`
_output_shapesN
L:         G:         G:         G:         G*
	num_split2)
'sequential_5/lstm_10/lstm_cell_10/split┼
)sequential_5/lstm_10/lstm_cell_10/SigmoidSigmoid0sequential_5/lstm_10/lstm_cell_10/split:output:0*
T0*'
_output_shapes
:         G2+
)sequential_5/lstm_10/lstm_cell_10/Sigmoid╔
+sequential_5/lstm_10/lstm_cell_10/Sigmoid_1Sigmoid0sequential_5/lstm_10/lstm_cell_10/split:output:1*
T0*'
_output_shapes
:         G2-
+sequential_5/lstm_10/lstm_cell_10/Sigmoid_1▀
%sequential_5/lstm_10/lstm_cell_10/mulMul/sequential_5/lstm_10/lstm_cell_10/Sigmoid_1:y:0%sequential_5/lstm_10/zeros_1:output:0*
T0*'
_output_shapes
:         G2'
%sequential_5/lstm_10/lstm_cell_10/mul╝
&sequential_5/lstm_10/lstm_cell_10/ReluRelu0sequential_5/lstm_10/lstm_cell_10/split:output:2*
T0*'
_output_shapes
:         G2(
&sequential_5/lstm_10/lstm_cell_10/ReluЁ
'sequential_5/lstm_10/lstm_cell_10/mul_1Mul-sequential_5/lstm_10/lstm_cell_10/Sigmoid:y:04sequential_5/lstm_10/lstm_cell_10/Relu:activations:0*
T0*'
_output_shapes
:         G2)
'sequential_5/lstm_10/lstm_cell_10/mul_1х
'sequential_5/lstm_10/lstm_cell_10/add_1AddV2)sequential_5/lstm_10/lstm_cell_10/mul:z:0+sequential_5/lstm_10/lstm_cell_10/mul_1:z:0*
T0*'
_output_shapes
:         G2)
'sequential_5/lstm_10/lstm_cell_10/add_1╔
+sequential_5/lstm_10/lstm_cell_10/Sigmoid_2Sigmoid0sequential_5/lstm_10/lstm_cell_10/split:output:3*
T0*'
_output_shapes
:         G2-
+sequential_5/lstm_10/lstm_cell_10/Sigmoid_2╗
(sequential_5/lstm_10/lstm_cell_10/Relu_1Relu+sequential_5/lstm_10/lstm_cell_10/add_1:z:0*
T0*'
_output_shapes
:         G2*
(sequential_5/lstm_10/lstm_cell_10/Relu_1Ї
'sequential_5/lstm_10/lstm_cell_10/mul_2Mul/sequential_5/lstm_10/lstm_cell_10/Sigmoid_2:y:06sequential_5/lstm_10/lstm_cell_10/Relu_1:activations:0*
T0*'
_output_shapes
:         G2)
'sequential_5/lstm_10/lstm_cell_10/mul_2╣
2sequential_5/lstm_10/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    G   24
2sequential_5/lstm_10/TensorArrayV2_1/element_shapeМ
$sequential_5/lstm_10/TensorArrayV2_1TensorListReserve;sequential_5/lstm_10/TensorArrayV2_1/element_shape:output:0-sequential_5/lstm_10/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02&
$sequential_5/lstm_10/TensorArrayV2_1x
sequential_5/lstm_10/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
sequential_5/lstm_10/timeй
-sequential_5/lstm_10/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         2/
-sequential_5/lstm_10/while/maximum_iterationsФ
'sequential_5/lstm_10/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2)
'sequential_5/lstm_10/while/loop_counter╩
sequential_5/lstm_10/whileWhile0sequential_5/lstm_10/while/loop_counter:output:06sequential_5/lstm_10/while/maximum_iterations:output:0"sequential_5/lstm_10/time:output:0-sequential_5/lstm_10/TensorArrayV2_1:handle:0#sequential_5/lstm_10/zeros:output:0%sequential_5/lstm_10/zeros_1:output:0-sequential_5/lstm_10/strided_slice_1:output:0Lsequential_5/lstm_10/TensorArrayUnstack/TensorListFromTensor:output_handle:0@sequential_5_lstm_10_lstm_cell_10_matmul_readvariableop_resourceBsequential_5_lstm_10_lstm_cell_10_matmul_1_readvariableop_resourceAsequential_5_lstm_10_lstm_cell_10_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         G:         G: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *4
body,R*
(sequential_5_lstm_10_while_body_19544894*4
cond,R*
(sequential_5_lstm_10_while_cond_19544893*K
output_shapes:
8: : : : :         G:         G: : : : : *
parallel_iterations 2
sequential_5/lstm_10/while▀
Esequential_5/lstm_10/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    G   2G
Esequential_5/lstm_10/TensorArrayV2Stack/TensorListStack/element_shape╝
7sequential_5/lstm_10/TensorArrayV2Stack/TensorListStackTensorListStack#sequential_5/lstm_10/while:output:3Nsequential_5/lstm_10/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:         G*
element_dtype029
7sequential_5/lstm_10/TensorArrayV2Stack/TensorListStackл
*sequential_5/lstm_10/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         2,
*sequential_5/lstm_10/strided_slice_3/stackж
,sequential_5/lstm_10/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2.
,sequential_5/lstm_10/strided_slice_3/stack_1ж
,sequential_5/lstm_10/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_5/lstm_10/strided_slice_3/stack_2Ш
$sequential_5/lstm_10/strided_slice_3StridedSlice@sequential_5/lstm_10/TensorArrayV2Stack/TensorListStack:tensor:03sequential_5/lstm_10/strided_slice_3/stack:output:05sequential_5/lstm_10/strided_slice_3/stack_1:output:05sequential_5/lstm_10/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         G*
shrink_axis_mask2&
$sequential_5/lstm_10/strided_slice_3г
%sequential_5/lstm_10/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2'
%sequential_5/lstm_10/transpose_1/perm∙
 sequential_5/lstm_10/transpose_1	Transpose@sequential_5/lstm_10/TensorArrayV2Stack/TensorListStack:tensor:0.sequential_5/lstm_10/transpose_1/perm:output:0*
T0*+
_output_shapes
:         G2"
 sequential_5/lstm_10/transpose_1Р
sequential_5/lstm_10/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
sequential_5/lstm_10/runtimeм
 sequential_5/dropout_10/IdentityIdentity$sequential_5/lstm_10/transpose_1:y:0*
T0*+
_output_shapes
:         G2"
 sequential_5/dropout_10/IdentityС
sequential_5/lstm_11/ShapeShape)sequential_5/dropout_10/Identity:output:0*
T0*
_output_shapes
:2
sequential_5/lstm_11/ShapeЮ
(sequential_5/lstm_11/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(sequential_5/lstm_11/strided_slice/stackв
*sequential_5/lstm_11/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*sequential_5/lstm_11/strided_slice/stack_1в
*sequential_5/lstm_11/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*sequential_5/lstm_11/strided_slice/stack_2р
"sequential_5/lstm_11/strided_sliceStridedSlice#sequential_5/lstm_11/Shape:output:01sequential_5/lstm_11/strided_slice/stack:output:03sequential_5/lstm_11/strided_slice/stack_1:output:03sequential_5/lstm_11/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"sequential_5/lstm_11/strided_sliceЗ
 sequential_5/lstm_11/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :в2"
 sequential_5/lstm_11/zeros/mul/y└
sequential_5/lstm_11/zeros/mulMul+sequential_5/lstm_11/strided_slice:output:0)sequential_5/lstm_11/zeros/mul/y:output:0*
T0*
_output_shapes
: 2 
sequential_5/lstm_11/zeros/mulЙ
!sequential_5/lstm_11/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2#
!sequential_5/lstm_11/zeros/Less/y╗
sequential_5/lstm_11/zeros/LessLess"sequential_5/lstm_11/zeros/mul:z:0*sequential_5/lstm_11/zeros/Less/y:output:0*
T0*
_output_shapes
: 2!
sequential_5/lstm_11/zeros/LessН
#sequential_5/lstm_11/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :в2%
#sequential_5/lstm_11/zeros/packed/1╫
!sequential_5/lstm_11/zeros/packedPack+sequential_5/lstm_11/strided_slice:output:0,sequential_5/lstm_11/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2#
!sequential_5/lstm_11/zeros/packedЙ
 sequential_5/lstm_11/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 sequential_5/lstm_11/zeros/Const╩
sequential_5/lstm_11/zerosFill*sequential_5/lstm_11/zeros/packed:output:0)sequential_5/lstm_11/zeros/Const:output:0*
T0*(
_output_shapes
:         в2
sequential_5/lstm_11/zerosЛ
"sequential_5/lstm_11/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :в2$
"sequential_5/lstm_11/zeros_1/mul/y╞
 sequential_5/lstm_11/zeros_1/mulMul+sequential_5/lstm_11/strided_slice:output:0+sequential_5/lstm_11/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2"
 sequential_5/lstm_11/zeros_1/mulН
#sequential_5/lstm_11/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2%
#sequential_5/lstm_11/zeros_1/Less/y├
!sequential_5/lstm_11/zeros_1/LessLess$sequential_5/lstm_11/zeros_1/mul:z:0,sequential_5/lstm_11/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2#
!sequential_5/lstm_11/zeros_1/LessС
%sequential_5/lstm_11/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :в2'
%sequential_5/lstm_11/zeros_1/packed/1▌
#sequential_5/lstm_11/zeros_1/packedPack+sequential_5/lstm_11/strided_slice:output:0.sequential_5/lstm_11/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2%
#sequential_5/lstm_11/zeros_1/packedН
"sequential_5/lstm_11/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"sequential_5/lstm_11/zeros_1/Const╥
sequential_5/lstm_11/zeros_1Fill,sequential_5/lstm_11/zeros_1/packed:output:0+sequential_5/lstm_11/zeros_1/Const:output:0*
T0*(
_output_shapes
:         в2
sequential_5/lstm_11/zeros_1Я
#sequential_5/lstm_11/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2%
#sequential_5/lstm_11/transpose/perm▄
sequential_5/lstm_11/transpose	Transpose)sequential_5/dropout_10/Identity:output:0,sequential_5/lstm_11/transpose/perm:output:0*
T0*+
_output_shapes
:         G2 
sequential_5/lstm_11/transposeО
sequential_5/lstm_11/Shape_1Shape"sequential_5/lstm_11/transpose:y:0*
T0*
_output_shapes
:2
sequential_5/lstm_11/Shape_1в
*sequential_5/lstm_11/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2,
*sequential_5/lstm_11/strided_slice_1/stackж
,sequential_5/lstm_11/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_5/lstm_11/strided_slice_1/stack_1ж
,sequential_5/lstm_11/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_5/lstm_11/strided_slice_1/stack_2ь
$sequential_5/lstm_11/strided_slice_1StridedSlice%sequential_5/lstm_11/Shape_1:output:03sequential_5/lstm_11/strided_slice_1/stack:output:05sequential_5/lstm_11/strided_slice_1/stack_1:output:05sequential_5/lstm_11/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2&
$sequential_5/lstm_11/strided_slice_1п
0sequential_5/lstm_11/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         22
0sequential_5/lstm_11/TensorArrayV2/element_shapeЖ
"sequential_5/lstm_11/TensorArrayV2TensorListReserve9sequential_5/lstm_11/TensorArrayV2/element_shape:output:0-sequential_5/lstm_11/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02$
"sequential_5/lstm_11/TensorArrayV2щ
Jsequential_5/lstm_11/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    G   2L
Jsequential_5/lstm_11/TensorArrayUnstack/TensorListFromTensor/element_shape╠
<sequential_5/lstm_11/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor"sequential_5/lstm_11/transpose:y:0Ssequential_5/lstm_11/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02>
<sequential_5/lstm_11/TensorArrayUnstack/TensorListFromTensorв
*sequential_5/lstm_11/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2,
*sequential_5/lstm_11/strided_slice_2/stackж
,sequential_5/lstm_11/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_5/lstm_11/strided_slice_2/stack_1ж
,sequential_5/lstm_11/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_5/lstm_11/strided_slice_2/stack_2·
$sequential_5/lstm_11/strided_slice_2StridedSlice"sequential_5/lstm_11/transpose:y:03sequential_5/lstm_11/strided_slice_2/stack:output:05sequential_5/lstm_11/strided_slice_2/stack_1:output:05sequential_5/lstm_11/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         G*
shrink_axis_mask2&
$sequential_5/lstm_11/strided_slice_2Ї
7sequential_5/lstm_11/lstm_cell_11/MatMul/ReadVariableOpReadVariableOp@sequential_5_lstm_11_lstm_cell_11_matmul_readvariableop_resource*
_output_shapes
:	GИ*
dtype029
7sequential_5/lstm_11/lstm_cell_11/MatMul/ReadVariableOpБ
(sequential_5/lstm_11/lstm_cell_11/MatMulMatMul-sequential_5/lstm_11/strided_slice_2:output:0?sequential_5/lstm_11/lstm_cell_11/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         И2*
(sequential_5/lstm_11/lstm_cell_11/MatMul√
9sequential_5/lstm_11/lstm_cell_11/MatMul_1/ReadVariableOpReadVariableOpBsequential_5_lstm_11_lstm_cell_11_matmul_1_readvariableop_resource* 
_output_shapes
:
вИ*
dtype02;
9sequential_5/lstm_11/lstm_cell_11/MatMul_1/ReadVariableOp¤
*sequential_5/lstm_11/lstm_cell_11/MatMul_1MatMul#sequential_5/lstm_11/zeros:output:0Asequential_5/lstm_11/lstm_cell_11/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         И2,
*sequential_5/lstm_11/lstm_cell_11/MatMul_1Ї
%sequential_5/lstm_11/lstm_cell_11/addAddV22sequential_5/lstm_11/lstm_cell_11/MatMul:product:04sequential_5/lstm_11/lstm_cell_11/MatMul_1:product:0*
T0*(
_output_shapes
:         И2'
%sequential_5/lstm_11/lstm_cell_11/addє
8sequential_5/lstm_11/lstm_cell_11/BiasAdd/ReadVariableOpReadVariableOpAsequential_5_lstm_11_lstm_cell_11_biasadd_readvariableop_resource*
_output_shapes	
:И*
dtype02:
8sequential_5/lstm_11/lstm_cell_11/BiasAdd/ReadVariableOpБ
)sequential_5/lstm_11/lstm_cell_11/BiasAddBiasAdd)sequential_5/lstm_11/lstm_cell_11/add:z:0@sequential_5/lstm_11/lstm_cell_11/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         И2+
)sequential_5/lstm_11/lstm_cell_11/BiasAddи
1sequential_5/lstm_11/lstm_cell_11/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :23
1sequential_5/lstm_11/lstm_cell_11/split/split_dim╦
'sequential_5/lstm_11/lstm_cell_11/splitSplit:sequential_5/lstm_11/lstm_cell_11/split/split_dim:output:02sequential_5/lstm_11/lstm_cell_11/BiasAdd:output:0*
T0*d
_output_shapesR
P:         в:         в:         в:         в*
	num_split2)
'sequential_5/lstm_11/lstm_cell_11/split╞
)sequential_5/lstm_11/lstm_cell_11/SigmoidSigmoid0sequential_5/lstm_11/lstm_cell_11/split:output:0*
T0*(
_output_shapes
:         в2+
)sequential_5/lstm_11/lstm_cell_11/Sigmoid╩
+sequential_5/lstm_11/lstm_cell_11/Sigmoid_1Sigmoid0sequential_5/lstm_11/lstm_cell_11/split:output:1*
T0*(
_output_shapes
:         в2-
+sequential_5/lstm_11/lstm_cell_11/Sigmoid_1р
%sequential_5/lstm_11/lstm_cell_11/mulMul/sequential_5/lstm_11/lstm_cell_11/Sigmoid_1:y:0%sequential_5/lstm_11/zeros_1:output:0*
T0*(
_output_shapes
:         в2'
%sequential_5/lstm_11/lstm_cell_11/mul╜
&sequential_5/lstm_11/lstm_cell_11/ReluRelu0sequential_5/lstm_11/lstm_cell_11/split:output:2*
T0*(
_output_shapes
:         в2(
&sequential_5/lstm_11/lstm_cell_11/Reluё
'sequential_5/lstm_11/lstm_cell_11/mul_1Mul-sequential_5/lstm_11/lstm_cell_11/Sigmoid:y:04sequential_5/lstm_11/lstm_cell_11/Relu:activations:0*
T0*(
_output_shapes
:         в2)
'sequential_5/lstm_11/lstm_cell_11/mul_1ц
'sequential_5/lstm_11/lstm_cell_11/add_1AddV2)sequential_5/lstm_11/lstm_cell_11/mul:z:0+sequential_5/lstm_11/lstm_cell_11/mul_1:z:0*
T0*(
_output_shapes
:         в2)
'sequential_5/lstm_11/lstm_cell_11/add_1╩
+sequential_5/lstm_11/lstm_cell_11/Sigmoid_2Sigmoid0sequential_5/lstm_11/lstm_cell_11/split:output:3*
T0*(
_output_shapes
:         в2-
+sequential_5/lstm_11/lstm_cell_11/Sigmoid_2╝
(sequential_5/lstm_11/lstm_cell_11/Relu_1Relu+sequential_5/lstm_11/lstm_cell_11/add_1:z:0*
T0*(
_output_shapes
:         в2*
(sequential_5/lstm_11/lstm_cell_11/Relu_1ї
'sequential_5/lstm_11/lstm_cell_11/mul_2Mul/sequential_5/lstm_11/lstm_cell_11/Sigmoid_2:y:06sequential_5/lstm_11/lstm_cell_11/Relu_1:activations:0*
T0*(
_output_shapes
:         в2)
'sequential_5/lstm_11/lstm_cell_11/mul_2╣
2sequential_5/lstm_11/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    в   24
2sequential_5/lstm_11/TensorArrayV2_1/element_shapeМ
$sequential_5/lstm_11/TensorArrayV2_1TensorListReserve;sequential_5/lstm_11/TensorArrayV2_1/element_shape:output:0-sequential_5/lstm_11/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02&
$sequential_5/lstm_11/TensorArrayV2_1x
sequential_5/lstm_11/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
sequential_5/lstm_11/timeй
-sequential_5/lstm_11/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         2/
-sequential_5/lstm_11/while/maximum_iterationsФ
'sequential_5/lstm_11/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2)
'sequential_5/lstm_11/while/loop_counter╬
sequential_5/lstm_11/whileWhile0sequential_5/lstm_11/while/loop_counter:output:06sequential_5/lstm_11/while/maximum_iterations:output:0"sequential_5/lstm_11/time:output:0-sequential_5/lstm_11/TensorArrayV2_1:handle:0#sequential_5/lstm_11/zeros:output:0%sequential_5/lstm_11/zeros_1:output:0-sequential_5/lstm_11/strided_slice_1:output:0Lsequential_5/lstm_11/TensorArrayUnstack/TensorListFromTensor:output_handle:0@sequential_5_lstm_11_lstm_cell_11_matmul_readvariableop_resourceBsequential_5_lstm_11_lstm_cell_11_matmul_1_readvariableop_resourceAsequential_5_lstm_11_lstm_cell_11_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :         в:         в: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *4
body,R*
(sequential_5_lstm_11_while_body_19545042*4
cond,R*
(sequential_5_lstm_11_while_cond_19545041*M
output_shapes<
:: : : : :         в:         в: : : : : *
parallel_iterations 2
sequential_5/lstm_11/while▀
Esequential_5/lstm_11/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    в   2G
Esequential_5/lstm_11/TensorArrayV2Stack/TensorListStack/element_shape╜
7sequential_5/lstm_11/TensorArrayV2Stack/TensorListStackTensorListStack#sequential_5/lstm_11/while:output:3Nsequential_5/lstm_11/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:         в*
element_dtype029
7sequential_5/lstm_11/TensorArrayV2Stack/TensorListStackл
*sequential_5/lstm_11/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         2,
*sequential_5/lstm_11/strided_slice_3/stackж
,sequential_5/lstm_11/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2.
,sequential_5/lstm_11/strided_slice_3/stack_1ж
,sequential_5/lstm_11/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_5/lstm_11/strided_slice_3/stack_2Щ
$sequential_5/lstm_11/strided_slice_3StridedSlice@sequential_5/lstm_11/TensorArrayV2Stack/TensorListStack:tensor:03sequential_5/lstm_11/strided_slice_3/stack:output:05sequential_5/lstm_11/strided_slice_3/stack_1:output:05sequential_5/lstm_11/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         в*
shrink_axis_mask2&
$sequential_5/lstm_11/strided_slice_3г
%sequential_5/lstm_11/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2'
%sequential_5/lstm_11/transpose_1/perm·
 sequential_5/lstm_11/transpose_1	Transpose@sequential_5/lstm_11/TensorArrayV2Stack/TensorListStack:tensor:0.sequential_5/lstm_11/transpose_1/perm:output:0*
T0*,
_output_shapes
:         в2"
 sequential_5/lstm_11/transpose_1Р
sequential_5/lstm_11/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
sequential_5/lstm_11/runtimeн
 sequential_5/dropout_11/IdentityIdentity$sequential_5/lstm_11/transpose_1:y:0*
T0*,
_output_shapes
:         в2"
 sequential_5/dropout_11/Identity╓
-sequential_5/dense_5/Tensordot/ReadVariableOpReadVariableOp6sequential_5_dense_5_tensordot_readvariableop_resource*
_output_shapes
:	в*
dtype02/
-sequential_5/dense_5/Tensordot/ReadVariableOpФ
#sequential_5/dense_5/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2%
#sequential_5/dense_5/Tensordot/axesЫ
#sequential_5/dense_5/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2%
#sequential_5/dense_5/Tensordot/freeе
$sequential_5/dense_5/Tensordot/ShapeShape)sequential_5/dropout_11/Identity:output:0*
T0*
_output_shapes
:2&
$sequential_5/dense_5/Tensordot/ShapeЮ
,sequential_5/dense_5/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,sequential_5/dense_5/Tensordot/GatherV2/axis║
'sequential_5/dense_5/Tensordot/GatherV2GatherV2-sequential_5/dense_5/Tensordot/Shape:output:0,sequential_5/dense_5/Tensordot/free:output:05sequential_5/dense_5/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2)
'sequential_5/dense_5/Tensordot/GatherV2в
.sequential_5/dense_5/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.sequential_5/dense_5/Tensordot/GatherV2_1/axis└
)sequential_5/dense_5/Tensordot/GatherV2_1GatherV2-sequential_5/dense_5/Tensordot/Shape:output:0,sequential_5/dense_5/Tensordot/axes:output:07sequential_5/dense_5/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2+
)sequential_5/dense_5/Tensordot/GatherV2_1Ц
$sequential_5/dense_5/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2&
$sequential_5/dense_5/Tensordot/Const╘
#sequential_5/dense_5/Tensordot/ProdProd0sequential_5/dense_5/Tensordot/GatherV2:output:0-sequential_5/dense_5/Tensordot/Const:output:0*
T0*
_output_shapes
: 2%
#sequential_5/dense_5/Tensordot/ProdЪ
&sequential_5/dense_5/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2(
&sequential_5/dense_5/Tensordot/Const_1▄
%sequential_5/dense_5/Tensordot/Prod_1Prod2sequential_5/dense_5/Tensordot/GatherV2_1:output:0/sequential_5/dense_5/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2'
%sequential_5/dense_5/Tensordot/Prod_1Ъ
*sequential_5/dense_5/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2,
*sequential_5/dense_5/Tensordot/concat/axisЩ
%sequential_5/dense_5/Tensordot/concatConcatV2,sequential_5/dense_5/Tensordot/free:output:0,sequential_5/dense_5/Tensordot/axes:output:03sequential_5/dense_5/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2'
%sequential_5/dense_5/Tensordot/concatр
$sequential_5/dense_5/Tensordot/stackPack,sequential_5/dense_5/Tensordot/Prod:output:0.sequential_5/dense_5/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2&
$sequential_5/dense_5/Tensordot/stackє
(sequential_5/dense_5/Tensordot/transpose	Transpose)sequential_5/dropout_11/Identity:output:0.sequential_5/dense_5/Tensordot/concat:output:0*
T0*,
_output_shapes
:         в2*
(sequential_5/dense_5/Tensordot/transposeє
&sequential_5/dense_5/Tensordot/ReshapeReshape,sequential_5/dense_5/Tensordot/transpose:y:0-sequential_5/dense_5/Tensordot/stack:output:0*
T0*0
_output_shapes
:                  2(
&sequential_5/dense_5/Tensordot/ReshapeЄ
%sequential_5/dense_5/Tensordot/MatMulMatMul/sequential_5/dense_5/Tensordot/Reshape:output:05sequential_5/dense_5/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2'
%sequential_5/dense_5/Tensordot/MatMulЪ
&sequential_5/dense_5/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&sequential_5/dense_5/Tensordot/Const_2Ю
,sequential_5/dense_5/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,sequential_5/dense_5/Tensordot/concat_1/axisж
'sequential_5/dense_5/Tensordot/concat_1ConcatV20sequential_5/dense_5/Tensordot/GatherV2:output:0/sequential_5/dense_5/Tensordot/Const_2:output:05sequential_5/dense_5/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2)
'sequential_5/dense_5/Tensordot/concat_1ф
sequential_5/dense_5/TensordotReshape/sequential_5/dense_5/Tensordot/MatMul:product:00sequential_5/dense_5/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:         2 
sequential_5/dense_5/Tensordot╦
+sequential_5/dense_5/BiasAdd/ReadVariableOpReadVariableOp4sequential_5_dense_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02-
+sequential_5/dense_5/BiasAdd/ReadVariableOp█
sequential_5/dense_5/BiasAddBiasAdd'sequential_5/dense_5/Tensordot:output:03sequential_5/dense_5/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:         2
sequential_5/dense_5/BiasAddд
sequential_5/dense_5/SoftmaxSoftmax%sequential_5/dense_5/BiasAdd:output:0*
T0*+
_output_shapes
:         2
sequential_5/dense_5/SoftmaxЕ
IdentityIdentity&sequential_5/dense_5/Softmax:softmax:0^NoOp*
T0*+
_output_shapes
:         2

Identity╚
NoOpNoOp,^sequential_5/dense_5/BiasAdd/ReadVariableOp.^sequential_5/dense_5/Tensordot/ReadVariableOp9^sequential_5/lstm_10/lstm_cell_10/BiasAdd/ReadVariableOp8^sequential_5/lstm_10/lstm_cell_10/MatMul/ReadVariableOp:^sequential_5/lstm_10/lstm_cell_10/MatMul_1/ReadVariableOp^sequential_5/lstm_10/while9^sequential_5/lstm_11/lstm_cell_11/BiasAdd/ReadVariableOp8^sequential_5/lstm_11/lstm_cell_11/MatMul/ReadVariableOp:^sequential_5/lstm_11/lstm_cell_11/MatMul_1/ReadVariableOp^sequential_5/lstm_11/while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':         ]: : : : : : : : 2Z
+sequential_5/dense_5/BiasAdd/ReadVariableOp+sequential_5/dense_5/BiasAdd/ReadVariableOp2^
-sequential_5/dense_5/Tensordot/ReadVariableOp-sequential_5/dense_5/Tensordot/ReadVariableOp2t
8sequential_5/lstm_10/lstm_cell_10/BiasAdd/ReadVariableOp8sequential_5/lstm_10/lstm_cell_10/BiasAdd/ReadVariableOp2r
7sequential_5/lstm_10/lstm_cell_10/MatMul/ReadVariableOp7sequential_5/lstm_10/lstm_cell_10/MatMul/ReadVariableOp2v
9sequential_5/lstm_10/lstm_cell_10/MatMul_1/ReadVariableOp9sequential_5/lstm_10/lstm_cell_10/MatMul_1/ReadVariableOp28
sequential_5/lstm_10/whilesequential_5/lstm_10/while2t
8sequential_5/lstm_11/lstm_cell_11/BiasAdd/ReadVariableOp8sequential_5/lstm_11/lstm_cell_11/BiasAdd/ReadVariableOp2r
7sequential_5/lstm_11/lstm_cell_11/MatMul/ReadVariableOp7sequential_5/lstm_11/lstm_cell_11/MatMul/ReadVariableOp2v
9sequential_5/lstm_11/lstm_cell_11/MatMul_1/ReadVariableOp9sequential_5/lstm_11/lstm_cell_11/MatMul_1/ReadVariableOp28
sequential_5/lstm_11/whilesequential_5/lstm_11/while:Z V
+
_output_shapes
:         ]
'
_user_specified_namelstm_10_input
Ъ?
╥
while_body_19548450
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_10_matmul_readvariableop_resource_0:	]ЬH
5while_lstm_cell_10_matmul_1_readvariableop_resource_0:	GЬC
4while_lstm_cell_10_biasadd_readvariableop_resource_0:	Ь
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_10_matmul_readvariableop_resource:	]ЬF
3while_lstm_cell_10_matmul_1_readvariableop_resource:	GЬA
2while_lstm_cell_10_biasadd_readvariableop_resource:	ЬИв)while/lstm_cell_10/BiasAdd/ReadVariableOpв(while/lstm_cell_10/MatMul/ReadVariableOpв*while/lstm_cell_10/MatMul_1/ReadVariableOp├
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
(while/lstm_cell_10/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_10_matmul_readvariableop_resource_0*
_output_shapes
:	]Ь*
dtype02*
(while/lstm_cell_10/MatMul/ReadVariableOp╫
while/lstm_cell_10/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_10/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ь2
while/lstm_cell_10/MatMul╧
*while/lstm_cell_10/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_10_matmul_1_readvariableop_resource_0*
_output_shapes
:	GЬ*
dtype02,
*while/lstm_cell_10/MatMul_1/ReadVariableOp└
while/lstm_cell_10/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_10/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ь2
while/lstm_cell_10/MatMul_1╕
while/lstm_cell_10/addAddV2#while/lstm_cell_10/MatMul:product:0%while/lstm_cell_10/MatMul_1:product:0*
T0*(
_output_shapes
:         Ь2
while/lstm_cell_10/add╚
)while/lstm_cell_10/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_10_biasadd_readvariableop_resource_0*
_output_shapes	
:Ь*
dtype02+
)while/lstm_cell_10/BiasAdd/ReadVariableOp┼
while/lstm_cell_10/BiasAddBiasAddwhile/lstm_cell_10/add:z:01while/lstm_cell_10/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ь2
while/lstm_cell_10/BiasAddК
"while/lstm_cell_10/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_10/split/split_dimЛ
while/lstm_cell_10/splitSplit+while/lstm_cell_10/split/split_dim:output:0#while/lstm_cell_10/BiasAdd:output:0*
T0*`
_output_shapesN
L:         G:         G:         G:         G*
	num_split2
while/lstm_cell_10/splitШ
while/lstm_cell_10/SigmoidSigmoid!while/lstm_cell_10/split:output:0*
T0*'
_output_shapes
:         G2
while/lstm_cell_10/SigmoidЬ
while/lstm_cell_10/Sigmoid_1Sigmoid!while/lstm_cell_10/split:output:1*
T0*'
_output_shapes
:         G2
while/lstm_cell_10/Sigmoid_1а
while/lstm_cell_10/mulMul while/lstm_cell_10/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:         G2
while/lstm_cell_10/mulП
while/lstm_cell_10/ReluRelu!while/lstm_cell_10/split:output:2*
T0*'
_output_shapes
:         G2
while/lstm_cell_10/Relu┤
while/lstm_cell_10/mul_1Mulwhile/lstm_cell_10/Sigmoid:y:0%while/lstm_cell_10/Relu:activations:0*
T0*'
_output_shapes
:         G2
while/lstm_cell_10/mul_1й
while/lstm_cell_10/add_1AddV2while/lstm_cell_10/mul:z:0while/lstm_cell_10/mul_1:z:0*
T0*'
_output_shapes
:         G2
while/lstm_cell_10/add_1Ь
while/lstm_cell_10/Sigmoid_2Sigmoid!while/lstm_cell_10/split:output:3*
T0*'
_output_shapes
:         G2
while/lstm_cell_10/Sigmoid_2О
while/lstm_cell_10/Relu_1Reluwhile/lstm_cell_10/add_1:z:0*
T0*'
_output_shapes
:         G2
while/lstm_cell_10/Relu_1╕
while/lstm_cell_10/mul_2Mul while/lstm_cell_10/Sigmoid_2:y:0'while/lstm_cell_10/Relu_1:activations:0*
T0*'
_output_shapes
:         G2
while/lstm_cell_10/mul_2р
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_10/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_10/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:         G2
while/Identity_4Н
while/Identity_5Identitywhile/lstm_cell_10/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:         G2
while/Identity_5▐

while/NoOpNoOp*^while/lstm_cell_10/BiasAdd/ReadVariableOp)^while/lstm_cell_10/MatMul/ReadVariableOp+^while/lstm_cell_10/MatMul_1/ReadVariableOp*"
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
2while_lstm_cell_10_biasadd_readvariableop_resource4while_lstm_cell_10_biasadd_readvariableop_resource_0"l
3while_lstm_cell_10_matmul_1_readvariableop_resource5while_lstm_cell_10_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_10_matmul_readvariableop_resource3while_lstm_cell_10_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         G:         G: : : : : 2V
)while/lstm_cell_10/BiasAdd/ReadVariableOp)while/lstm_cell_10/BiasAdd/ReadVariableOp2T
(while/lstm_cell_10/MatMul/ReadVariableOp(while/lstm_cell_10/MatMul/ReadVariableOp2X
*while/lstm_cell_10/MatMul_1/ReadVariableOp*while/lstm_cell_10/MatMul_1/ReadVariableOp: 
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
:         G:-)
'
_output_shapes
:         G:

_output_shapes
: :

_output_shapes
: 
░?
╘
while_body_19549276
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_11_matmul_readvariableop_resource_0:	GИI
5while_lstm_cell_11_matmul_1_readvariableop_resource_0:
вИC
4while_lstm_cell_11_biasadd_readvariableop_resource_0:	И
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_11_matmul_readvariableop_resource:	GИG
3while_lstm_cell_11_matmul_1_readvariableop_resource:
вИA
2while_lstm_cell_11_biasadd_readvariableop_resource:	ИИв)while/lstm_cell_11/BiasAdd/ReadVariableOpв(while/lstm_cell_11/MatMul/ReadVariableOpв*while/lstm_cell_11/MatMul_1/ReadVariableOp├
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    G   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape╙
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         G*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem╔
(while/lstm_cell_11/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_11_matmul_readvariableop_resource_0*
_output_shapes
:	GИ*
dtype02*
(while/lstm_cell_11/MatMul/ReadVariableOp╫
while/lstm_cell_11/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_11/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         И2
while/lstm_cell_11/MatMul╨
*while/lstm_cell_11/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_11_matmul_1_readvariableop_resource_0* 
_output_shapes
:
вИ*
dtype02,
*while/lstm_cell_11/MatMul_1/ReadVariableOp└
while/lstm_cell_11/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_11/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         И2
while/lstm_cell_11/MatMul_1╕
while/lstm_cell_11/addAddV2#while/lstm_cell_11/MatMul:product:0%while/lstm_cell_11/MatMul_1:product:0*
T0*(
_output_shapes
:         И2
while/lstm_cell_11/add╚
)while/lstm_cell_11/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_11_biasadd_readvariableop_resource_0*
_output_shapes	
:И*
dtype02+
)while/lstm_cell_11/BiasAdd/ReadVariableOp┼
while/lstm_cell_11/BiasAddBiasAddwhile/lstm_cell_11/add:z:01while/lstm_cell_11/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         И2
while/lstm_cell_11/BiasAddК
"while/lstm_cell_11/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_11/split/split_dimП
while/lstm_cell_11/splitSplit+while/lstm_cell_11/split/split_dim:output:0#while/lstm_cell_11/BiasAdd:output:0*
T0*d
_output_shapesR
P:         в:         в:         в:         в*
	num_split2
while/lstm_cell_11/splitЩ
while/lstm_cell_11/SigmoidSigmoid!while/lstm_cell_11/split:output:0*
T0*(
_output_shapes
:         в2
while/lstm_cell_11/SigmoidЭ
while/lstm_cell_11/Sigmoid_1Sigmoid!while/lstm_cell_11/split:output:1*
T0*(
_output_shapes
:         в2
while/lstm_cell_11/Sigmoid_1б
while/lstm_cell_11/mulMul while/lstm_cell_11/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:         в2
while/lstm_cell_11/mulР
while/lstm_cell_11/ReluRelu!while/lstm_cell_11/split:output:2*
T0*(
_output_shapes
:         в2
while/lstm_cell_11/Relu╡
while/lstm_cell_11/mul_1Mulwhile/lstm_cell_11/Sigmoid:y:0%while/lstm_cell_11/Relu:activations:0*
T0*(
_output_shapes
:         в2
while/lstm_cell_11/mul_1к
while/lstm_cell_11/add_1AddV2while/lstm_cell_11/mul:z:0while/lstm_cell_11/mul_1:z:0*
T0*(
_output_shapes
:         в2
while/lstm_cell_11/add_1Э
while/lstm_cell_11/Sigmoid_2Sigmoid!while/lstm_cell_11/split:output:3*
T0*(
_output_shapes
:         в2
while/lstm_cell_11/Sigmoid_2П
while/lstm_cell_11/Relu_1Reluwhile/lstm_cell_11/add_1:z:0*
T0*(
_output_shapes
:         в2
while/lstm_cell_11/Relu_1╣
while/lstm_cell_11/mul_2Mul while/lstm_cell_11/Sigmoid_2:y:0'while/lstm_cell_11/Relu_1:activations:0*
T0*(
_output_shapes
:         в2
while/lstm_cell_11/mul_2р
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_11/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_11/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:         в2
while/Identity_4О
while/Identity_5Identitywhile/lstm_cell_11/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:         в2
while/Identity_5▐

while/NoOpNoOp*^while/lstm_cell_11/BiasAdd/ReadVariableOp)^while/lstm_cell_11/MatMul/ReadVariableOp+^while/lstm_cell_11/MatMul_1/ReadVariableOp*"
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
2while_lstm_cell_11_biasadd_readvariableop_resource4while_lstm_cell_11_biasadd_readvariableop_resource_0"l
3while_lstm_cell_11_matmul_1_readvariableop_resource5while_lstm_cell_11_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_11_matmul_readvariableop_resource3while_lstm_cell_11_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :         в:         в: : : : : 2V
)while/lstm_cell_11/BiasAdd/ReadVariableOp)while/lstm_cell_11/BiasAdd/ReadVariableOp2T
(while/lstm_cell_11/MatMul/ReadVariableOp(while/lstm_cell_11/MatMul/ReadVariableOp2X
*while/lstm_cell_11/MatMul_1/ReadVariableOp*while/lstm_cell_11/MatMul_1/ReadVariableOp: 
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
:         в:.*
(
_output_shapes
:         в:

_output_shapes
: :

_output_shapes
: 
╒°
Е
J__inference_sequential_5_layer_call_and_return_conditional_losses_19547698

inputsF
3lstm_10_lstm_cell_10_matmul_readvariableop_resource:	]ЬH
5lstm_10_lstm_cell_10_matmul_1_readvariableop_resource:	GЬC
4lstm_10_lstm_cell_10_biasadd_readvariableop_resource:	ЬF
3lstm_11_lstm_cell_11_matmul_readvariableop_resource:	GИI
5lstm_11_lstm_cell_11_matmul_1_readvariableop_resource:
вИC
4lstm_11_lstm_cell_11_biasadd_readvariableop_resource:	И<
)dense_5_tensordot_readvariableop_resource:	в5
'dense_5_biasadd_readvariableop_resource:
identityИвdense_5/BiasAdd/ReadVariableOpв dense_5/Tensordot/ReadVariableOpв+lstm_10/lstm_cell_10/BiasAdd/ReadVariableOpв*lstm_10/lstm_cell_10/MatMul/ReadVariableOpв,lstm_10/lstm_cell_10/MatMul_1/ReadVariableOpвlstm_10/whileв+lstm_11/lstm_cell_11/BiasAdd/ReadVariableOpв*lstm_11/lstm_cell_11/MatMul/ReadVariableOpв,lstm_11/lstm_cell_11/MatMul_1/ReadVariableOpвlstm_11/whileT
lstm_10/ShapeShapeinputs*
T0*
_output_shapes
:2
lstm_10/ShapeД
lstm_10/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_10/strided_slice/stackИ
lstm_10/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_10/strided_slice/stack_1И
lstm_10/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_10/strided_slice/stack_2Т
lstm_10/strided_sliceStridedSlicelstm_10/Shape:output:0$lstm_10/strided_slice/stack:output:0&lstm_10/strided_slice/stack_1:output:0&lstm_10/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_10/strided_slicel
lstm_10/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :G2
lstm_10/zeros/mul/yМ
lstm_10/zeros/mulMullstm_10/strided_slice:output:0lstm_10/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_10/zeros/mulo
lstm_10/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
lstm_10/zeros/Less/yЗ
lstm_10/zeros/LessLesslstm_10/zeros/mul:z:0lstm_10/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_10/zeros/Lessr
lstm_10/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :G2
lstm_10/zeros/packed/1г
lstm_10/zeros/packedPacklstm_10/strided_slice:output:0lstm_10/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_10/zeros/packedo
lstm_10/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_10/zeros/ConstХ
lstm_10/zerosFilllstm_10/zeros/packed:output:0lstm_10/zeros/Const:output:0*
T0*'
_output_shapes
:         G2
lstm_10/zerosp
lstm_10/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :G2
lstm_10/zeros_1/mul/yТ
lstm_10/zeros_1/mulMullstm_10/strided_slice:output:0lstm_10/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_10/zeros_1/muls
lstm_10/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
lstm_10/zeros_1/Less/yП
lstm_10/zeros_1/LessLesslstm_10/zeros_1/mul:z:0lstm_10/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_10/zeros_1/Lessv
lstm_10/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :G2
lstm_10/zeros_1/packed/1й
lstm_10/zeros_1/packedPacklstm_10/strided_slice:output:0!lstm_10/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_10/zeros_1/packeds
lstm_10/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_10/zeros_1/ConstЭ
lstm_10/zeros_1Filllstm_10/zeros_1/packed:output:0lstm_10/zeros_1/Const:output:0*
T0*'
_output_shapes
:         G2
lstm_10/zeros_1Е
lstm_10/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_10/transpose/permТ
lstm_10/transpose	Transposeinputslstm_10/transpose/perm:output:0*
T0*+
_output_shapes
:         ]2
lstm_10/transposeg
lstm_10/Shape_1Shapelstm_10/transpose:y:0*
T0*
_output_shapes
:2
lstm_10/Shape_1И
lstm_10/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_10/strided_slice_1/stackМ
lstm_10/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_10/strided_slice_1/stack_1М
lstm_10/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_10/strided_slice_1/stack_2Ю
lstm_10/strided_slice_1StridedSlicelstm_10/Shape_1:output:0&lstm_10/strided_slice_1/stack:output:0(lstm_10/strided_slice_1/stack_1:output:0(lstm_10/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_10/strided_slice_1Х
#lstm_10/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         2%
#lstm_10/TensorArrayV2/element_shape╥
lstm_10/TensorArrayV2TensorListReserve,lstm_10/TensorArrayV2/element_shape:output:0 lstm_10/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_10/TensorArrayV2╧
=lstm_10/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ]   2?
=lstm_10/TensorArrayUnstack/TensorListFromTensor/element_shapeШ
/lstm_10/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_10/transpose:y:0Flstm_10/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type021
/lstm_10/TensorArrayUnstack/TensorListFromTensorИ
lstm_10/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_10/strided_slice_2/stackМ
lstm_10/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_10/strided_slice_2/stack_1М
lstm_10/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_10/strided_slice_2/stack_2м
lstm_10/strided_slice_2StridedSlicelstm_10/transpose:y:0&lstm_10/strided_slice_2/stack:output:0(lstm_10/strided_slice_2/stack_1:output:0(lstm_10/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         ]*
shrink_axis_mask2
lstm_10/strided_slice_2═
*lstm_10/lstm_cell_10/MatMul/ReadVariableOpReadVariableOp3lstm_10_lstm_cell_10_matmul_readvariableop_resource*
_output_shapes
:	]Ь*
dtype02,
*lstm_10/lstm_cell_10/MatMul/ReadVariableOp═
lstm_10/lstm_cell_10/MatMulMatMul lstm_10/strided_slice_2:output:02lstm_10/lstm_cell_10/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ь2
lstm_10/lstm_cell_10/MatMul╙
,lstm_10/lstm_cell_10/MatMul_1/ReadVariableOpReadVariableOp5lstm_10_lstm_cell_10_matmul_1_readvariableop_resource*
_output_shapes
:	GЬ*
dtype02.
,lstm_10/lstm_cell_10/MatMul_1/ReadVariableOp╔
lstm_10/lstm_cell_10/MatMul_1MatMullstm_10/zeros:output:04lstm_10/lstm_cell_10/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ь2
lstm_10/lstm_cell_10/MatMul_1└
lstm_10/lstm_cell_10/addAddV2%lstm_10/lstm_cell_10/MatMul:product:0'lstm_10/lstm_cell_10/MatMul_1:product:0*
T0*(
_output_shapes
:         Ь2
lstm_10/lstm_cell_10/add╠
+lstm_10/lstm_cell_10/BiasAdd/ReadVariableOpReadVariableOp4lstm_10_lstm_cell_10_biasadd_readvariableop_resource*
_output_shapes	
:Ь*
dtype02-
+lstm_10/lstm_cell_10/BiasAdd/ReadVariableOp═
lstm_10/lstm_cell_10/BiasAddBiasAddlstm_10/lstm_cell_10/add:z:03lstm_10/lstm_cell_10/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ь2
lstm_10/lstm_cell_10/BiasAddО
$lstm_10/lstm_cell_10/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2&
$lstm_10/lstm_cell_10/split/split_dimУ
lstm_10/lstm_cell_10/splitSplit-lstm_10/lstm_cell_10/split/split_dim:output:0%lstm_10/lstm_cell_10/BiasAdd:output:0*
T0*`
_output_shapesN
L:         G:         G:         G:         G*
	num_split2
lstm_10/lstm_cell_10/splitЮ
lstm_10/lstm_cell_10/SigmoidSigmoid#lstm_10/lstm_cell_10/split:output:0*
T0*'
_output_shapes
:         G2
lstm_10/lstm_cell_10/Sigmoidв
lstm_10/lstm_cell_10/Sigmoid_1Sigmoid#lstm_10/lstm_cell_10/split:output:1*
T0*'
_output_shapes
:         G2 
lstm_10/lstm_cell_10/Sigmoid_1л
lstm_10/lstm_cell_10/mulMul"lstm_10/lstm_cell_10/Sigmoid_1:y:0lstm_10/zeros_1:output:0*
T0*'
_output_shapes
:         G2
lstm_10/lstm_cell_10/mulХ
lstm_10/lstm_cell_10/ReluRelu#lstm_10/lstm_cell_10/split:output:2*
T0*'
_output_shapes
:         G2
lstm_10/lstm_cell_10/Relu╝
lstm_10/lstm_cell_10/mul_1Mul lstm_10/lstm_cell_10/Sigmoid:y:0'lstm_10/lstm_cell_10/Relu:activations:0*
T0*'
_output_shapes
:         G2
lstm_10/lstm_cell_10/mul_1▒
lstm_10/lstm_cell_10/add_1AddV2lstm_10/lstm_cell_10/mul:z:0lstm_10/lstm_cell_10/mul_1:z:0*
T0*'
_output_shapes
:         G2
lstm_10/lstm_cell_10/add_1в
lstm_10/lstm_cell_10/Sigmoid_2Sigmoid#lstm_10/lstm_cell_10/split:output:3*
T0*'
_output_shapes
:         G2 
lstm_10/lstm_cell_10/Sigmoid_2Ф
lstm_10/lstm_cell_10/Relu_1Relulstm_10/lstm_cell_10/add_1:z:0*
T0*'
_output_shapes
:         G2
lstm_10/lstm_cell_10/Relu_1└
lstm_10/lstm_cell_10/mul_2Mul"lstm_10/lstm_cell_10/Sigmoid_2:y:0)lstm_10/lstm_cell_10/Relu_1:activations:0*
T0*'
_output_shapes
:         G2
lstm_10/lstm_cell_10/mul_2Я
%lstm_10/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    G   2'
%lstm_10/TensorArrayV2_1/element_shape╪
lstm_10/TensorArrayV2_1TensorListReserve.lstm_10/TensorArrayV2_1/element_shape:output:0 lstm_10/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_10/TensorArrayV2_1^
lstm_10/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_10/timeП
 lstm_10/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         2"
 lstm_10/while/maximum_iterationsz
lstm_10/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_10/while/loop_counterЗ
lstm_10/whileWhile#lstm_10/while/loop_counter:output:0)lstm_10/while/maximum_iterations:output:0lstm_10/time:output:0 lstm_10/TensorArrayV2_1:handle:0lstm_10/zeros:output:0lstm_10/zeros_1:output:0 lstm_10/strided_slice_1:output:0?lstm_10/TensorArrayUnstack/TensorListFromTensor:output_handle:03lstm_10_lstm_cell_10_matmul_readvariableop_resource5lstm_10_lstm_cell_10_matmul_1_readvariableop_resource4lstm_10_lstm_cell_10_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         G:         G: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *'
bodyR
lstm_10_while_body_19547438*'
condR
lstm_10_while_cond_19547437*K
output_shapes:
8: : : : :         G:         G: : : : : *
parallel_iterations 2
lstm_10/while┼
8lstm_10/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    G   2:
8lstm_10/TensorArrayV2Stack/TensorListStack/element_shapeИ
*lstm_10/TensorArrayV2Stack/TensorListStackTensorListStacklstm_10/while:output:3Alstm_10/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:         G*
element_dtype02,
*lstm_10/TensorArrayV2Stack/TensorListStackС
lstm_10/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         2
lstm_10/strided_slice_3/stackМ
lstm_10/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2!
lstm_10/strided_slice_3/stack_1М
lstm_10/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_10/strided_slice_3/stack_2╩
lstm_10/strided_slice_3StridedSlice3lstm_10/TensorArrayV2Stack/TensorListStack:tensor:0&lstm_10/strided_slice_3/stack:output:0(lstm_10/strided_slice_3/stack_1:output:0(lstm_10/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         G*
shrink_axis_mask2
lstm_10/strided_slice_3Й
lstm_10/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_10/transpose_1/perm┼
lstm_10/transpose_1	Transpose3lstm_10/TensorArrayV2Stack/TensorListStack:tensor:0!lstm_10/transpose_1/perm:output:0*
T0*+
_output_shapes
:         G2
lstm_10/transpose_1v
lstm_10/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_10/runtimeЕ
dropout_10/IdentityIdentitylstm_10/transpose_1:y:0*
T0*+
_output_shapes
:         G2
dropout_10/Identityj
lstm_11/ShapeShapedropout_10/Identity:output:0*
T0*
_output_shapes
:2
lstm_11/ShapeД
lstm_11/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_11/strided_slice/stackИ
lstm_11/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_11/strided_slice/stack_1И
lstm_11/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_11/strided_slice/stack_2Т
lstm_11/strided_sliceStridedSlicelstm_11/Shape:output:0$lstm_11/strided_slice/stack:output:0&lstm_11/strided_slice/stack_1:output:0&lstm_11/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_11/strided_slicem
lstm_11/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :в2
lstm_11/zeros/mul/yМ
lstm_11/zeros/mulMullstm_11/strided_slice:output:0lstm_11/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_11/zeros/mulo
lstm_11/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
lstm_11/zeros/Less/yЗ
lstm_11/zeros/LessLesslstm_11/zeros/mul:z:0lstm_11/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_11/zeros/Lesss
lstm_11/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :в2
lstm_11/zeros/packed/1г
lstm_11/zeros/packedPacklstm_11/strided_slice:output:0lstm_11/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_11/zeros/packedo
lstm_11/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_11/zeros/ConstЦ
lstm_11/zerosFilllstm_11/zeros/packed:output:0lstm_11/zeros/Const:output:0*
T0*(
_output_shapes
:         в2
lstm_11/zerosq
lstm_11/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :в2
lstm_11/zeros_1/mul/yТ
lstm_11/zeros_1/mulMullstm_11/strided_slice:output:0lstm_11/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_11/zeros_1/muls
lstm_11/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
lstm_11/zeros_1/Less/yП
lstm_11/zeros_1/LessLesslstm_11/zeros_1/mul:z:0lstm_11/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_11/zeros_1/Lessw
lstm_11/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :в2
lstm_11/zeros_1/packed/1й
lstm_11/zeros_1/packedPacklstm_11/strided_slice:output:0!lstm_11/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_11/zeros_1/packeds
lstm_11/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_11/zeros_1/ConstЮ
lstm_11/zeros_1Filllstm_11/zeros_1/packed:output:0lstm_11/zeros_1/Const:output:0*
T0*(
_output_shapes
:         в2
lstm_11/zeros_1Е
lstm_11/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_11/transpose/permи
lstm_11/transpose	Transposedropout_10/Identity:output:0lstm_11/transpose/perm:output:0*
T0*+
_output_shapes
:         G2
lstm_11/transposeg
lstm_11/Shape_1Shapelstm_11/transpose:y:0*
T0*
_output_shapes
:2
lstm_11/Shape_1И
lstm_11/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_11/strided_slice_1/stackМ
lstm_11/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_11/strided_slice_1/stack_1М
lstm_11/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_11/strided_slice_1/stack_2Ю
lstm_11/strided_slice_1StridedSlicelstm_11/Shape_1:output:0&lstm_11/strided_slice_1/stack:output:0(lstm_11/strided_slice_1/stack_1:output:0(lstm_11/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_11/strided_slice_1Х
#lstm_11/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         2%
#lstm_11/TensorArrayV2/element_shape╥
lstm_11/TensorArrayV2TensorListReserve,lstm_11/TensorArrayV2/element_shape:output:0 lstm_11/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_11/TensorArrayV2╧
=lstm_11/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    G   2?
=lstm_11/TensorArrayUnstack/TensorListFromTensor/element_shapeШ
/lstm_11/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_11/transpose:y:0Flstm_11/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type021
/lstm_11/TensorArrayUnstack/TensorListFromTensorИ
lstm_11/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_11/strided_slice_2/stackМ
lstm_11/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_11/strided_slice_2/stack_1М
lstm_11/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_11/strided_slice_2/stack_2м
lstm_11/strided_slice_2StridedSlicelstm_11/transpose:y:0&lstm_11/strided_slice_2/stack:output:0(lstm_11/strided_slice_2/stack_1:output:0(lstm_11/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         G*
shrink_axis_mask2
lstm_11/strided_slice_2═
*lstm_11/lstm_cell_11/MatMul/ReadVariableOpReadVariableOp3lstm_11_lstm_cell_11_matmul_readvariableop_resource*
_output_shapes
:	GИ*
dtype02,
*lstm_11/lstm_cell_11/MatMul/ReadVariableOp═
lstm_11/lstm_cell_11/MatMulMatMul lstm_11/strided_slice_2:output:02lstm_11/lstm_cell_11/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         И2
lstm_11/lstm_cell_11/MatMul╘
,lstm_11/lstm_cell_11/MatMul_1/ReadVariableOpReadVariableOp5lstm_11_lstm_cell_11_matmul_1_readvariableop_resource* 
_output_shapes
:
вИ*
dtype02.
,lstm_11/lstm_cell_11/MatMul_1/ReadVariableOp╔
lstm_11/lstm_cell_11/MatMul_1MatMullstm_11/zeros:output:04lstm_11/lstm_cell_11/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         И2
lstm_11/lstm_cell_11/MatMul_1└
lstm_11/lstm_cell_11/addAddV2%lstm_11/lstm_cell_11/MatMul:product:0'lstm_11/lstm_cell_11/MatMul_1:product:0*
T0*(
_output_shapes
:         И2
lstm_11/lstm_cell_11/add╠
+lstm_11/lstm_cell_11/BiasAdd/ReadVariableOpReadVariableOp4lstm_11_lstm_cell_11_biasadd_readvariableop_resource*
_output_shapes	
:И*
dtype02-
+lstm_11/lstm_cell_11/BiasAdd/ReadVariableOp═
lstm_11/lstm_cell_11/BiasAddBiasAddlstm_11/lstm_cell_11/add:z:03lstm_11/lstm_cell_11/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         И2
lstm_11/lstm_cell_11/BiasAddО
$lstm_11/lstm_cell_11/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2&
$lstm_11/lstm_cell_11/split/split_dimЧ
lstm_11/lstm_cell_11/splitSplit-lstm_11/lstm_cell_11/split/split_dim:output:0%lstm_11/lstm_cell_11/BiasAdd:output:0*
T0*d
_output_shapesR
P:         в:         в:         в:         в*
	num_split2
lstm_11/lstm_cell_11/splitЯ
lstm_11/lstm_cell_11/SigmoidSigmoid#lstm_11/lstm_cell_11/split:output:0*
T0*(
_output_shapes
:         в2
lstm_11/lstm_cell_11/Sigmoidг
lstm_11/lstm_cell_11/Sigmoid_1Sigmoid#lstm_11/lstm_cell_11/split:output:1*
T0*(
_output_shapes
:         в2 
lstm_11/lstm_cell_11/Sigmoid_1м
lstm_11/lstm_cell_11/mulMul"lstm_11/lstm_cell_11/Sigmoid_1:y:0lstm_11/zeros_1:output:0*
T0*(
_output_shapes
:         в2
lstm_11/lstm_cell_11/mulЦ
lstm_11/lstm_cell_11/ReluRelu#lstm_11/lstm_cell_11/split:output:2*
T0*(
_output_shapes
:         в2
lstm_11/lstm_cell_11/Relu╜
lstm_11/lstm_cell_11/mul_1Mul lstm_11/lstm_cell_11/Sigmoid:y:0'lstm_11/lstm_cell_11/Relu:activations:0*
T0*(
_output_shapes
:         в2
lstm_11/lstm_cell_11/mul_1▓
lstm_11/lstm_cell_11/add_1AddV2lstm_11/lstm_cell_11/mul:z:0lstm_11/lstm_cell_11/mul_1:z:0*
T0*(
_output_shapes
:         в2
lstm_11/lstm_cell_11/add_1г
lstm_11/lstm_cell_11/Sigmoid_2Sigmoid#lstm_11/lstm_cell_11/split:output:3*
T0*(
_output_shapes
:         в2 
lstm_11/lstm_cell_11/Sigmoid_2Х
lstm_11/lstm_cell_11/Relu_1Relulstm_11/lstm_cell_11/add_1:z:0*
T0*(
_output_shapes
:         в2
lstm_11/lstm_cell_11/Relu_1┴
lstm_11/lstm_cell_11/mul_2Mul"lstm_11/lstm_cell_11/Sigmoid_2:y:0)lstm_11/lstm_cell_11/Relu_1:activations:0*
T0*(
_output_shapes
:         в2
lstm_11/lstm_cell_11/mul_2Я
%lstm_11/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    в   2'
%lstm_11/TensorArrayV2_1/element_shape╪
lstm_11/TensorArrayV2_1TensorListReserve.lstm_11/TensorArrayV2_1/element_shape:output:0 lstm_11/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_11/TensorArrayV2_1^
lstm_11/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_11/timeП
 lstm_11/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         2"
 lstm_11/while/maximum_iterationsz
lstm_11/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_11/while/loop_counterЛ
lstm_11/whileWhile#lstm_11/while/loop_counter:output:0)lstm_11/while/maximum_iterations:output:0lstm_11/time:output:0 lstm_11/TensorArrayV2_1:handle:0lstm_11/zeros:output:0lstm_11/zeros_1:output:0 lstm_11/strided_slice_1:output:0?lstm_11/TensorArrayUnstack/TensorListFromTensor:output_handle:03lstm_11_lstm_cell_11_matmul_readvariableop_resource5lstm_11_lstm_cell_11_matmul_1_readvariableop_resource4lstm_11_lstm_cell_11_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :         в:         в: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *'
bodyR
lstm_11_while_body_19547586*'
condR
lstm_11_while_cond_19547585*M
output_shapes<
:: : : : :         в:         в: : : : : *
parallel_iterations 2
lstm_11/while┼
8lstm_11/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    в   2:
8lstm_11/TensorArrayV2Stack/TensorListStack/element_shapeЙ
*lstm_11/TensorArrayV2Stack/TensorListStackTensorListStacklstm_11/while:output:3Alstm_11/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:         в*
element_dtype02,
*lstm_11/TensorArrayV2Stack/TensorListStackС
lstm_11/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         2
lstm_11/strided_slice_3/stackМ
lstm_11/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2!
lstm_11/strided_slice_3/stack_1М
lstm_11/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_11/strided_slice_3/stack_2╦
lstm_11/strided_slice_3StridedSlice3lstm_11/TensorArrayV2Stack/TensorListStack:tensor:0&lstm_11/strided_slice_3/stack:output:0(lstm_11/strided_slice_3/stack_1:output:0(lstm_11/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         в*
shrink_axis_mask2
lstm_11/strided_slice_3Й
lstm_11/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_11/transpose_1/perm╞
lstm_11/transpose_1	Transpose3lstm_11/TensorArrayV2Stack/TensorListStack:tensor:0!lstm_11/transpose_1/perm:output:0*
T0*,
_output_shapes
:         в2
lstm_11/transpose_1v
lstm_11/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_11/runtimeЖ
dropout_11/IdentityIdentitylstm_11/transpose_1:y:0*
T0*,
_output_shapes
:         в2
dropout_11/Identityп
 dense_5/Tensordot/ReadVariableOpReadVariableOp)dense_5_tensordot_readvariableop_resource*
_output_shapes
:	в*
dtype02"
 dense_5/Tensordot/ReadVariableOpz
dense_5/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_5/Tensordot/axesБ
dense_5/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_5/Tensordot/free~
dense_5/Tensordot/ShapeShapedropout_11/Identity:output:0*
T0*
_output_shapes
:2
dense_5/Tensordot/ShapeД
dense_5/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_5/Tensordot/GatherV2/axis∙
dense_5/Tensordot/GatherV2GatherV2 dense_5/Tensordot/Shape:output:0dense_5/Tensordot/free:output:0(dense_5/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_5/Tensordot/GatherV2И
!dense_5/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_5/Tensordot/GatherV2_1/axis 
dense_5/Tensordot/GatherV2_1GatherV2 dense_5/Tensordot/Shape:output:0dense_5/Tensordot/axes:output:0*dense_5/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_5/Tensordot/GatherV2_1|
dense_5/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_5/Tensordot/Constа
dense_5/Tensordot/ProdProd#dense_5/Tensordot/GatherV2:output:0 dense_5/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_5/Tensordot/ProdА
dense_5/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_5/Tensordot/Const_1и
dense_5/Tensordot/Prod_1Prod%dense_5/Tensordot/GatherV2_1:output:0"dense_5/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_5/Tensordot/Prod_1А
dense_5/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense_5/Tensordot/concat/axis╪
dense_5/Tensordot/concatConcatV2dense_5/Tensordot/free:output:0dense_5/Tensordot/axes:output:0&dense_5/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_5/Tensordot/concatм
dense_5/Tensordot/stackPackdense_5/Tensordot/Prod:output:0!dense_5/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_5/Tensordot/stack┐
dense_5/Tensordot/transpose	Transposedropout_11/Identity:output:0!dense_5/Tensordot/concat:output:0*
T0*,
_output_shapes
:         в2
dense_5/Tensordot/transpose┐
dense_5/Tensordot/ReshapeReshapedense_5/Tensordot/transpose:y:0 dense_5/Tensordot/stack:output:0*
T0*0
_output_shapes
:                  2
dense_5/Tensordot/Reshape╛
dense_5/Tensordot/MatMulMatMul"dense_5/Tensordot/Reshape:output:0(dense_5/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_5/Tensordot/MatMulА
dense_5/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
dense_5/Tensordot/Const_2Д
dense_5/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_5/Tensordot/concat_1/axisх
dense_5/Tensordot/concat_1ConcatV2#dense_5/Tensordot/GatherV2:output:0"dense_5/Tensordot/Const_2:output:0(dense_5/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_5/Tensordot/concat_1░
dense_5/TensordotReshape"dense_5/Tensordot/MatMul:product:0#dense_5/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:         2
dense_5/Tensordotд
dense_5/BiasAdd/ReadVariableOpReadVariableOp'dense_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_5/BiasAdd/ReadVariableOpз
dense_5/BiasAddBiasAdddense_5/Tensordot:output:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:         2
dense_5/BiasAdd}
dense_5/SoftmaxSoftmaxdense_5/BiasAdd:output:0*
T0*+
_output_shapes
:         2
dense_5/Softmaxx
IdentityIdentitydense_5/Softmax:softmax:0^NoOp*
T0*+
_output_shapes
:         2

Identity╞
NoOpNoOp^dense_5/BiasAdd/ReadVariableOp!^dense_5/Tensordot/ReadVariableOp,^lstm_10/lstm_cell_10/BiasAdd/ReadVariableOp+^lstm_10/lstm_cell_10/MatMul/ReadVariableOp-^lstm_10/lstm_cell_10/MatMul_1/ReadVariableOp^lstm_10/while,^lstm_11/lstm_cell_11/BiasAdd/ReadVariableOp+^lstm_11/lstm_cell_11/MatMul/ReadVariableOp-^lstm_11/lstm_cell_11/MatMul_1/ReadVariableOp^lstm_11/while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':         ]: : : : : : : : 2@
dense_5/BiasAdd/ReadVariableOpdense_5/BiasAdd/ReadVariableOp2D
 dense_5/Tensordot/ReadVariableOp dense_5/Tensordot/ReadVariableOp2Z
+lstm_10/lstm_cell_10/BiasAdd/ReadVariableOp+lstm_10/lstm_cell_10/BiasAdd/ReadVariableOp2X
*lstm_10/lstm_cell_10/MatMul/ReadVariableOp*lstm_10/lstm_cell_10/MatMul/ReadVariableOp2\
,lstm_10/lstm_cell_10/MatMul_1/ReadVariableOp,lstm_10/lstm_cell_10/MatMul_1/ReadVariableOp2
lstm_10/whilelstm_10/while2Z
+lstm_11/lstm_cell_11/BiasAdd/ReadVariableOp+lstm_11/lstm_cell_11/BiasAdd/ReadVariableOp2X
*lstm_11/lstm_cell_11/MatMul/ReadVariableOp*lstm_11/lstm_cell_11/MatMul/ReadVariableOp2\
,lstm_11/lstm_cell_11/MatMul_1/ReadVariableOp,lstm_11/lstm_cell_11/MatMul_1/ReadVariableOp2
lstm_11/whilelstm_11/while:S O
+
_output_shapes
:         ]
 
_user_specified_nameinputs
Ъ?
╥
while_body_19546487
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_10_matmul_readvariableop_resource_0:	]ЬH
5while_lstm_cell_10_matmul_1_readvariableop_resource_0:	GЬC
4while_lstm_cell_10_biasadd_readvariableop_resource_0:	Ь
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_10_matmul_readvariableop_resource:	]ЬF
3while_lstm_cell_10_matmul_1_readvariableop_resource:	GЬA
2while_lstm_cell_10_biasadd_readvariableop_resource:	ЬИв)while/lstm_cell_10/BiasAdd/ReadVariableOpв(while/lstm_cell_10/MatMul/ReadVariableOpв*while/lstm_cell_10/MatMul_1/ReadVariableOp├
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
(while/lstm_cell_10/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_10_matmul_readvariableop_resource_0*
_output_shapes
:	]Ь*
dtype02*
(while/lstm_cell_10/MatMul/ReadVariableOp╫
while/lstm_cell_10/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_10/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ь2
while/lstm_cell_10/MatMul╧
*while/lstm_cell_10/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_10_matmul_1_readvariableop_resource_0*
_output_shapes
:	GЬ*
dtype02,
*while/lstm_cell_10/MatMul_1/ReadVariableOp└
while/lstm_cell_10/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_10/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ь2
while/lstm_cell_10/MatMul_1╕
while/lstm_cell_10/addAddV2#while/lstm_cell_10/MatMul:product:0%while/lstm_cell_10/MatMul_1:product:0*
T0*(
_output_shapes
:         Ь2
while/lstm_cell_10/add╚
)while/lstm_cell_10/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_10_biasadd_readvariableop_resource_0*
_output_shapes	
:Ь*
dtype02+
)while/lstm_cell_10/BiasAdd/ReadVariableOp┼
while/lstm_cell_10/BiasAddBiasAddwhile/lstm_cell_10/add:z:01while/lstm_cell_10/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ь2
while/lstm_cell_10/BiasAddК
"while/lstm_cell_10/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_10/split/split_dimЛ
while/lstm_cell_10/splitSplit+while/lstm_cell_10/split/split_dim:output:0#while/lstm_cell_10/BiasAdd:output:0*
T0*`
_output_shapesN
L:         G:         G:         G:         G*
	num_split2
while/lstm_cell_10/splitШ
while/lstm_cell_10/SigmoidSigmoid!while/lstm_cell_10/split:output:0*
T0*'
_output_shapes
:         G2
while/lstm_cell_10/SigmoidЬ
while/lstm_cell_10/Sigmoid_1Sigmoid!while/lstm_cell_10/split:output:1*
T0*'
_output_shapes
:         G2
while/lstm_cell_10/Sigmoid_1а
while/lstm_cell_10/mulMul while/lstm_cell_10/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:         G2
while/lstm_cell_10/mulП
while/lstm_cell_10/ReluRelu!while/lstm_cell_10/split:output:2*
T0*'
_output_shapes
:         G2
while/lstm_cell_10/Relu┤
while/lstm_cell_10/mul_1Mulwhile/lstm_cell_10/Sigmoid:y:0%while/lstm_cell_10/Relu:activations:0*
T0*'
_output_shapes
:         G2
while/lstm_cell_10/mul_1й
while/lstm_cell_10/add_1AddV2while/lstm_cell_10/mul:z:0while/lstm_cell_10/mul_1:z:0*
T0*'
_output_shapes
:         G2
while/lstm_cell_10/add_1Ь
while/lstm_cell_10/Sigmoid_2Sigmoid!while/lstm_cell_10/split:output:3*
T0*'
_output_shapes
:         G2
while/lstm_cell_10/Sigmoid_2О
while/lstm_cell_10/Relu_1Reluwhile/lstm_cell_10/add_1:z:0*
T0*'
_output_shapes
:         G2
while/lstm_cell_10/Relu_1╕
while/lstm_cell_10/mul_2Mul while/lstm_cell_10/Sigmoid_2:y:0'while/lstm_cell_10/Relu_1:activations:0*
T0*'
_output_shapes
:         G2
while/lstm_cell_10/mul_2р
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_10/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_10/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:         G2
while/Identity_4Н
while/Identity_5Identitywhile/lstm_cell_10/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:         G2
while/Identity_5▐

while/NoOpNoOp*^while/lstm_cell_10/BiasAdd/ReadVariableOp)^while/lstm_cell_10/MatMul/ReadVariableOp+^while/lstm_cell_10/MatMul_1/ReadVariableOp*"
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
2while_lstm_cell_10_biasadd_readvariableop_resource4while_lstm_cell_10_biasadd_readvariableop_resource_0"l
3while_lstm_cell_10_matmul_1_readvariableop_resource5while_lstm_cell_10_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_10_matmul_readvariableop_resource3while_lstm_cell_10_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         G:         G: : : : : 2V
)while/lstm_cell_10/BiasAdd/ReadVariableOp)while/lstm_cell_10/BiasAdd/ReadVariableOp2T
(while/lstm_cell_10/MatMul/ReadVariableOp(while/lstm_cell_10/MatMul/ReadVariableOp2X
*while/lstm_cell_10/MatMul_1/ReadVariableOp*while/lstm_cell_10/MatMul_1/ReadVariableOp: 
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
:         G:-)
'
_output_shapes
:         G:

_output_shapes
: :

_output_shapes
: 
Ч
ы
J__inference_sequential_5_layer_call_and_return_conditional_losses_19546789

inputs#
lstm_10_19546572:	]Ь#
lstm_10_19546574:	GЬ
lstm_10_19546576:	Ь#
lstm_11_19546737:	GИ$
lstm_11_19546739:
вИ
lstm_11_19546741:	И#
dense_5_19546783:	в
dense_5_19546785:
identityИвdense_5/StatefulPartitionedCallвlstm_10/StatefulPartitionedCallвlstm_11/StatefulPartitionedCallн
lstm_10/StatefulPartitionedCallStatefulPartitionedCallinputslstm_10_19546572lstm_10_19546574lstm_10_19546576*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         G*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_lstm_10_layer_call_and_return_conditional_losses_195465712!
lstm_10/StatefulPartitionedCallВ
dropout_10/PartitionedCallPartitionedCall(lstm_10/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         G* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_dropout_10_layer_call_and_return_conditional_losses_195465842
dropout_10/PartitionedCall╦
lstm_11/StatefulPartitionedCallStatefulPartitionedCall#dropout_10/PartitionedCall:output:0lstm_11_19546737lstm_11_19546739lstm_11_19546741*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         в*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_lstm_11_layer_call_and_return_conditional_losses_195467362!
lstm_11/StatefulPartitionedCallГ
dropout_11/PartitionedCallPartitionedCall(lstm_11/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         в* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_dropout_11_layer_call_and_return_conditional_losses_195467492
dropout_11/PartitionedCall╢
dense_5/StatefulPartitionedCallStatefulPartitionedCall#dropout_11/PartitionedCall:output:0dense_5_19546783dense_5_19546785*
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
E__inference_dense_5_layer_call_and_return_conditional_losses_195467822!
dense_5/StatefulPartitionedCallЗ
IdentityIdentity(dense_5/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:         2

Identity┤
NoOpNoOp ^dense_5/StatefulPartitionedCall ^lstm_10/StatefulPartitionedCall ^lstm_11/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':         ]: : : : : : : : 2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2B
lstm_10/StatefulPartitionedCalllstm_10/StatefulPartitionedCall2B
lstm_11/StatefulPartitionedCalllstm_11/StatefulPartitionedCall:S O
+
_output_shapes
:         ]
 
_user_specified_nameinputs
▀
═
while_cond_19545452
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_19545452___redundant_placeholder06
2while_while_cond_19545452___redundant_placeholder16
2while_while_cond_19545452___redundant_placeholder26
2while_while_cond_19545452___redundant_placeholder3
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
@: : : : :         G:         G: ::::: 
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
:         G:-)
'
_output_shapes
:         G:

_output_shapes
: :

_output_shapes
:
╧
g
H__inference_dropout_10_layer_call_and_return_conditional_losses_19548746

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
:         G2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape╕
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:         G*
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
:         G2
dropout/GreaterEqualГ
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:         G2
dropout/Cast~
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*+
_output_shapes
:         G2
dropout/Mul_1i
IdentityIdentitydropout/Mul_1:z:0*
T0*+
_output_shapes
:         G2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         G:S O
+
_output_shapes
:         G
 
_user_specified_nameinputs
Ъ?
╥
while_body_19548299
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_10_matmul_readvariableop_resource_0:	]ЬH
5while_lstm_cell_10_matmul_1_readvariableop_resource_0:	GЬC
4while_lstm_cell_10_biasadd_readvariableop_resource_0:	Ь
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_10_matmul_readvariableop_resource:	]ЬF
3while_lstm_cell_10_matmul_1_readvariableop_resource:	GЬA
2while_lstm_cell_10_biasadd_readvariableop_resource:	ЬИв)while/lstm_cell_10/BiasAdd/ReadVariableOpв(while/lstm_cell_10/MatMul/ReadVariableOpв*while/lstm_cell_10/MatMul_1/ReadVariableOp├
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
(while/lstm_cell_10/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_10_matmul_readvariableop_resource_0*
_output_shapes
:	]Ь*
dtype02*
(while/lstm_cell_10/MatMul/ReadVariableOp╫
while/lstm_cell_10/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_10/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ь2
while/lstm_cell_10/MatMul╧
*while/lstm_cell_10/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_10_matmul_1_readvariableop_resource_0*
_output_shapes
:	GЬ*
dtype02,
*while/lstm_cell_10/MatMul_1/ReadVariableOp└
while/lstm_cell_10/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_10/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ь2
while/lstm_cell_10/MatMul_1╕
while/lstm_cell_10/addAddV2#while/lstm_cell_10/MatMul:product:0%while/lstm_cell_10/MatMul_1:product:0*
T0*(
_output_shapes
:         Ь2
while/lstm_cell_10/add╚
)while/lstm_cell_10/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_10_biasadd_readvariableop_resource_0*
_output_shapes	
:Ь*
dtype02+
)while/lstm_cell_10/BiasAdd/ReadVariableOp┼
while/lstm_cell_10/BiasAddBiasAddwhile/lstm_cell_10/add:z:01while/lstm_cell_10/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ь2
while/lstm_cell_10/BiasAddК
"while/lstm_cell_10/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_10/split/split_dimЛ
while/lstm_cell_10/splitSplit+while/lstm_cell_10/split/split_dim:output:0#while/lstm_cell_10/BiasAdd:output:0*
T0*`
_output_shapesN
L:         G:         G:         G:         G*
	num_split2
while/lstm_cell_10/splitШ
while/lstm_cell_10/SigmoidSigmoid!while/lstm_cell_10/split:output:0*
T0*'
_output_shapes
:         G2
while/lstm_cell_10/SigmoidЬ
while/lstm_cell_10/Sigmoid_1Sigmoid!while/lstm_cell_10/split:output:1*
T0*'
_output_shapes
:         G2
while/lstm_cell_10/Sigmoid_1а
while/lstm_cell_10/mulMul while/lstm_cell_10/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:         G2
while/lstm_cell_10/mulП
while/lstm_cell_10/ReluRelu!while/lstm_cell_10/split:output:2*
T0*'
_output_shapes
:         G2
while/lstm_cell_10/Relu┤
while/lstm_cell_10/mul_1Mulwhile/lstm_cell_10/Sigmoid:y:0%while/lstm_cell_10/Relu:activations:0*
T0*'
_output_shapes
:         G2
while/lstm_cell_10/mul_1й
while/lstm_cell_10/add_1AddV2while/lstm_cell_10/mul:z:0while/lstm_cell_10/mul_1:z:0*
T0*'
_output_shapes
:         G2
while/lstm_cell_10/add_1Ь
while/lstm_cell_10/Sigmoid_2Sigmoid!while/lstm_cell_10/split:output:3*
T0*'
_output_shapes
:         G2
while/lstm_cell_10/Sigmoid_2О
while/lstm_cell_10/Relu_1Reluwhile/lstm_cell_10/add_1:z:0*
T0*'
_output_shapes
:         G2
while/lstm_cell_10/Relu_1╕
while/lstm_cell_10/mul_2Mul while/lstm_cell_10/Sigmoid_2:y:0'while/lstm_cell_10/Relu_1:activations:0*
T0*'
_output_shapes
:         G2
while/lstm_cell_10/mul_2р
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_10/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_10/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:         G2
while/Identity_4Н
while/Identity_5Identitywhile/lstm_cell_10/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:         G2
while/Identity_5▐

while/NoOpNoOp*^while/lstm_cell_10/BiasAdd/ReadVariableOp)^while/lstm_cell_10/MatMul/ReadVariableOp+^while/lstm_cell_10/MatMul_1/ReadVariableOp*"
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
2while_lstm_cell_10_biasadd_readvariableop_resource4while_lstm_cell_10_biasadd_readvariableop_resource_0"l
3while_lstm_cell_10_matmul_1_readvariableop_resource5while_lstm_cell_10_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_10_matmul_readvariableop_resource3while_lstm_cell_10_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         G:         G: : : : : 2V
)while/lstm_cell_10/BiasAdd/ReadVariableOp)while/lstm_cell_10/BiasAdd/ReadVariableOp2T
(while/lstm_cell_10/MatMul/ReadVariableOp(while/lstm_cell_10/MatMul/ReadVariableOp2X
*while/lstm_cell_10/MatMul_1/ReadVariableOp*while/lstm_cell_10/MatMul_1/ReadVariableOp: 
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
:         G:-)
'
_output_shapes
:         G:

_output_shapes
: :

_output_shapes
: 
¤
И
J__inference_lstm_cell_10_layer_call_and_return_conditional_losses_19549503

inputs
states_0
states_11
matmul_readvariableop_resource:	]Ь3
 matmul_1_readvariableop_resource:	GЬ.
biasadd_readvariableop_resource:	Ь
identity

identity_1

identity_2ИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpвMatMul_1/ReadVariableOpО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	]Ь*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ь2
MatMulФ
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	GЬ*
dtype02
MatMul_1/ReadVariableOp|
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ь2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:         Ь2
addН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:Ь*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ь2	
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
L:         G:         G:         G:         G*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:         G2	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:         G2
	Sigmoid_1\
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:         G2
mulV
ReluRelusplit:output:2*
T0*'
_output_shapes
:         G2
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:         G2
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:         G2
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:         G2
	Sigmoid_2U
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:         G2
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:         G2
mul_2d
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:         G2

Identityh

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:         G2

Identity_1h

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:         G2

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
?:         ]:         G:         G: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:         ]
 
_user_specified_nameinputs:QM
'
_output_shapes
:         G
"
_user_specified_name
states/0:QM
'
_output_shapes
:         G
"
_user_specified_name
states/1
▀
═
while_cond_19548298
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_19548298___redundant_placeholder06
2while_while_cond_19548298___redundant_placeholder16
2while_while_cond_19548298___redundant_placeholder26
2while_while_cond_19548298___redundant_placeholder3
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
@: : : : :         G:         G: ::::: 
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
:         G:-)
'
_output_shapes
:         G:

_output_shapes
: :

_output_shapes
:
╦F
О
E__inference_lstm_11_layer_call_and_return_conditional_losses_19546152

inputs(
lstm_cell_11_19546070:	GИ)
lstm_cell_11_19546072:
вИ$
lstm_cell_11_19546074:	И
identityИв$lstm_cell_11/StatefulPartitionedCallвwhileD
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
B :в2
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
B :в2
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
:         в2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :в2
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
B :в2
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
:         в2	
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
 :                  G2
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
valueB"    G   27
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
:         G*
shrink_axis_mask2
strided_slice_2и
$lstm_cell_11/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_11_19546070lstm_cell_11_19546072lstm_cell_11_19546074*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:         в:         в:         в*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_lstm_cell_11_layer_call_and_return_conditional_losses_195460052&
$lstm_cell_11/StatefulPartitionedCallП
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    в   2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_11_19546070lstm_cell_11_19546072lstm_cell_11_19546074*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :         в:         в: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_19546083*
condR
while_cond_19546082*M
output_shapes<
:: : : : :         в:         в: : : : : *
parallel_iterations 2
while╡
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    в   22
0TensorArrayV2Stack/TensorListStack/element_shapeЄ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:                  в*
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
:         в*
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
!:                  в2
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
!:                  в2

Identity}
NoOpNoOp%^lstm_cell_11/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  G: : : 2L
$lstm_cell_11/StatefulPartitionedCall$lstm_cell_11/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :                  G
 
_user_specified_nameinputs
╣
╝
J__inference_sequential_5_layer_call_and_return_conditional_losses_19547348
lstm_10_input#
lstm_10_19547326:	]Ь#
lstm_10_19547328:	GЬ
lstm_10_19547330:	Ь#
lstm_11_19547334:	GИ$
lstm_11_19547336:
вИ
lstm_11_19547338:	И#
dense_5_19547342:	в
dense_5_19547344:
identityИвdense_5/StatefulPartitionedCallв"dropout_10/StatefulPartitionedCallв"dropout_11/StatefulPartitionedCallвlstm_10/StatefulPartitionedCallвlstm_11/StatefulPartitionedCall┤
lstm_10/StatefulPartitionedCallStatefulPartitionedCalllstm_10_inputlstm_10_19547326lstm_10_19547328lstm_10_19547330*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         G*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_lstm_10_layer_call_and_return_conditional_losses_195472012!
lstm_10/StatefulPartitionedCallЪ
"dropout_10/StatefulPartitionedCallStatefulPartitionedCall(lstm_10/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         G* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_dropout_10_layer_call_and_return_conditional_losses_195470342$
"dropout_10/StatefulPartitionedCall╙
lstm_11/StatefulPartitionedCallStatefulPartitionedCall+dropout_10/StatefulPartitionedCall:output:0lstm_11_19547334lstm_11_19547336lstm_11_19547338*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         в*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_lstm_11_layer_call_and_return_conditional_losses_195470052!
lstm_11/StatefulPartitionedCall└
"dropout_11/StatefulPartitionedCallStatefulPartitionedCall(lstm_11/StatefulPartitionedCall:output:0#^dropout_10/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         в* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_dropout_11_layer_call_and_return_conditional_losses_195468382$
"dropout_11/StatefulPartitionedCall╛
dense_5/StatefulPartitionedCallStatefulPartitionedCall+dropout_11/StatefulPartitionedCall:output:0dense_5_19547342dense_5_19547344*
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
E__inference_dense_5_layer_call_and_return_conditional_losses_195467822!
dense_5/StatefulPartitionedCallЗ
IdentityIdentity(dense_5/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:         2

Identity■
NoOpNoOp ^dense_5/StatefulPartitionedCall#^dropout_10/StatefulPartitionedCall#^dropout_11/StatefulPartitionedCall ^lstm_10/StatefulPartitionedCall ^lstm_11/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':         ]: : : : : : : : 2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2H
"dropout_10/StatefulPartitionedCall"dropout_10/StatefulPartitionedCall2H
"dropout_11/StatefulPartitionedCall"dropout_11/StatefulPartitionedCall2B
lstm_10/StatefulPartitionedCalllstm_10/StatefulPartitionedCall2B
lstm_11/StatefulPartitionedCalllstm_11/StatefulPartitionedCall:Z V
+
_output_shapes
:         ]
'
_user_specified_namelstm_10_input
╫
ё
(sequential_5_lstm_10_while_cond_19544893F
Bsequential_5_lstm_10_while_sequential_5_lstm_10_while_loop_counterL
Hsequential_5_lstm_10_while_sequential_5_lstm_10_while_maximum_iterations*
&sequential_5_lstm_10_while_placeholder,
(sequential_5_lstm_10_while_placeholder_1,
(sequential_5_lstm_10_while_placeholder_2,
(sequential_5_lstm_10_while_placeholder_3H
Dsequential_5_lstm_10_while_less_sequential_5_lstm_10_strided_slice_1`
\sequential_5_lstm_10_while_sequential_5_lstm_10_while_cond_19544893___redundant_placeholder0`
\sequential_5_lstm_10_while_sequential_5_lstm_10_while_cond_19544893___redundant_placeholder1`
\sequential_5_lstm_10_while_sequential_5_lstm_10_while_cond_19544893___redundant_placeholder2`
\sequential_5_lstm_10_while_sequential_5_lstm_10_while_cond_19544893___redundant_placeholder3'
#sequential_5_lstm_10_while_identity
┘
sequential_5/lstm_10/while/LessLess&sequential_5_lstm_10_while_placeholderDsequential_5_lstm_10_while_less_sequential_5_lstm_10_strided_slice_1*
T0*
_output_shapes
: 2!
sequential_5/lstm_10/while/LessЬ
#sequential_5/lstm_10/while/IdentityIdentity#sequential_5/lstm_10/while/Less:z:0*
T0
*
_output_shapes
: 2%
#sequential_5/lstm_10/while/Identity"S
#sequential_5_lstm_10_while_identity,sequential_5/lstm_10/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :         G:         G: ::::: 
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
:         G:-)
'
_output_shapes
:         G:

_output_shapes
: :

_output_shapes
:
░?
╘
while_body_19549125
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_11_matmul_readvariableop_resource_0:	GИI
5while_lstm_cell_11_matmul_1_readvariableop_resource_0:
вИC
4while_lstm_cell_11_biasadd_readvariableop_resource_0:	И
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_11_matmul_readvariableop_resource:	GИG
3while_lstm_cell_11_matmul_1_readvariableop_resource:
вИA
2while_lstm_cell_11_biasadd_readvariableop_resource:	ИИв)while/lstm_cell_11/BiasAdd/ReadVariableOpв(while/lstm_cell_11/MatMul/ReadVariableOpв*while/lstm_cell_11/MatMul_1/ReadVariableOp├
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    G   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape╙
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         G*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem╔
(while/lstm_cell_11/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_11_matmul_readvariableop_resource_0*
_output_shapes
:	GИ*
dtype02*
(while/lstm_cell_11/MatMul/ReadVariableOp╫
while/lstm_cell_11/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_11/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         И2
while/lstm_cell_11/MatMul╨
*while/lstm_cell_11/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_11_matmul_1_readvariableop_resource_0* 
_output_shapes
:
вИ*
dtype02,
*while/lstm_cell_11/MatMul_1/ReadVariableOp└
while/lstm_cell_11/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_11/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         И2
while/lstm_cell_11/MatMul_1╕
while/lstm_cell_11/addAddV2#while/lstm_cell_11/MatMul:product:0%while/lstm_cell_11/MatMul_1:product:0*
T0*(
_output_shapes
:         И2
while/lstm_cell_11/add╚
)while/lstm_cell_11/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_11_biasadd_readvariableop_resource_0*
_output_shapes	
:И*
dtype02+
)while/lstm_cell_11/BiasAdd/ReadVariableOp┼
while/lstm_cell_11/BiasAddBiasAddwhile/lstm_cell_11/add:z:01while/lstm_cell_11/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         И2
while/lstm_cell_11/BiasAddК
"while/lstm_cell_11/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_11/split/split_dimП
while/lstm_cell_11/splitSplit+while/lstm_cell_11/split/split_dim:output:0#while/lstm_cell_11/BiasAdd:output:0*
T0*d
_output_shapesR
P:         в:         в:         в:         в*
	num_split2
while/lstm_cell_11/splitЩ
while/lstm_cell_11/SigmoidSigmoid!while/lstm_cell_11/split:output:0*
T0*(
_output_shapes
:         в2
while/lstm_cell_11/SigmoidЭ
while/lstm_cell_11/Sigmoid_1Sigmoid!while/lstm_cell_11/split:output:1*
T0*(
_output_shapes
:         в2
while/lstm_cell_11/Sigmoid_1б
while/lstm_cell_11/mulMul while/lstm_cell_11/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:         в2
while/lstm_cell_11/mulР
while/lstm_cell_11/ReluRelu!while/lstm_cell_11/split:output:2*
T0*(
_output_shapes
:         в2
while/lstm_cell_11/Relu╡
while/lstm_cell_11/mul_1Mulwhile/lstm_cell_11/Sigmoid:y:0%while/lstm_cell_11/Relu:activations:0*
T0*(
_output_shapes
:         в2
while/lstm_cell_11/mul_1к
while/lstm_cell_11/add_1AddV2while/lstm_cell_11/mul:z:0while/lstm_cell_11/mul_1:z:0*
T0*(
_output_shapes
:         в2
while/lstm_cell_11/add_1Э
while/lstm_cell_11/Sigmoid_2Sigmoid!while/lstm_cell_11/split:output:3*
T0*(
_output_shapes
:         в2
while/lstm_cell_11/Sigmoid_2П
while/lstm_cell_11/Relu_1Reluwhile/lstm_cell_11/add_1:z:0*
T0*(
_output_shapes
:         в2
while/lstm_cell_11/Relu_1╣
while/lstm_cell_11/mul_2Mul while/lstm_cell_11/Sigmoid_2:y:0'while/lstm_cell_11/Relu_1:activations:0*
T0*(
_output_shapes
:         в2
while/lstm_cell_11/mul_2р
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_11/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_11/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:         в2
while/Identity_4О
while/Identity_5Identitywhile/lstm_cell_11/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:         в2
while/Identity_5▐

while/NoOpNoOp*^while/lstm_cell_11/BiasAdd/ReadVariableOp)^while/lstm_cell_11/MatMul/ReadVariableOp+^while/lstm_cell_11/MatMul_1/ReadVariableOp*"
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
2while_lstm_cell_11_biasadd_readvariableop_resource4while_lstm_cell_11_biasadd_readvariableop_resource_0"l
3while_lstm_cell_11_matmul_1_readvariableop_resource5while_lstm_cell_11_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_11_matmul_readvariableop_resource3while_lstm_cell_11_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :         в:         в: : : : : 2V
)while/lstm_cell_11/BiasAdd/ReadVariableOp)while/lstm_cell_11/BiasAdd/ReadVariableOp2T
(while/lstm_cell_11/MatMul/ReadVariableOp(while/lstm_cell_11/MatMul/ReadVariableOp2X
*while/lstm_cell_11/MatMul_1/ReadVariableOp*while/lstm_cell_11/MatMul_1/ReadVariableOp: 
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
:         в:.*
(
_output_shapes
:         в:

_output_shapes
: :

_output_shapes
: 
Д\
Ю
E__inference_lstm_11_layer_call_and_return_conditional_losses_19549360

inputs>
+lstm_cell_11_matmul_readvariableop_resource:	GИA
-lstm_cell_11_matmul_1_readvariableop_resource:
вИ;
,lstm_cell_11_biasadd_readvariableop_resource:	И
identityИв#lstm_cell_11/BiasAdd/ReadVariableOpв"lstm_cell_11/MatMul/ReadVariableOpв$lstm_cell_11/MatMul_1/ReadVariableOpвwhileD
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
B :в2
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
B :в2
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
:         в2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :в2
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
B :в2
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
:         в2	
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
:         G2
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
valueB"    G   27
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
:         G*
shrink_axis_mask2
strided_slice_2╡
"lstm_cell_11/MatMul/ReadVariableOpReadVariableOp+lstm_cell_11_matmul_readvariableop_resource*
_output_shapes
:	GИ*
dtype02$
"lstm_cell_11/MatMul/ReadVariableOpн
lstm_cell_11/MatMulMatMulstrided_slice_2:output:0*lstm_cell_11/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         И2
lstm_cell_11/MatMul╝
$lstm_cell_11/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_11_matmul_1_readvariableop_resource* 
_output_shapes
:
вИ*
dtype02&
$lstm_cell_11/MatMul_1/ReadVariableOpй
lstm_cell_11/MatMul_1MatMulzeros:output:0,lstm_cell_11/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         И2
lstm_cell_11/MatMul_1а
lstm_cell_11/addAddV2lstm_cell_11/MatMul:product:0lstm_cell_11/MatMul_1:product:0*
T0*(
_output_shapes
:         И2
lstm_cell_11/add┤
#lstm_cell_11/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_11_biasadd_readvariableop_resource*
_output_shapes	
:И*
dtype02%
#lstm_cell_11/BiasAdd/ReadVariableOpн
lstm_cell_11/BiasAddBiasAddlstm_cell_11/add:z:0+lstm_cell_11/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         И2
lstm_cell_11/BiasAdd~
lstm_cell_11/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_11/split/split_dimў
lstm_cell_11/splitSplit%lstm_cell_11/split/split_dim:output:0lstm_cell_11/BiasAdd:output:0*
T0*d
_output_shapesR
P:         в:         в:         в:         в*
	num_split2
lstm_cell_11/splitЗ
lstm_cell_11/SigmoidSigmoidlstm_cell_11/split:output:0*
T0*(
_output_shapes
:         в2
lstm_cell_11/SigmoidЛ
lstm_cell_11/Sigmoid_1Sigmoidlstm_cell_11/split:output:1*
T0*(
_output_shapes
:         в2
lstm_cell_11/Sigmoid_1М
lstm_cell_11/mulMullstm_cell_11/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:         в2
lstm_cell_11/mul~
lstm_cell_11/ReluRelulstm_cell_11/split:output:2*
T0*(
_output_shapes
:         в2
lstm_cell_11/ReluЭ
lstm_cell_11/mul_1Mullstm_cell_11/Sigmoid:y:0lstm_cell_11/Relu:activations:0*
T0*(
_output_shapes
:         в2
lstm_cell_11/mul_1Т
lstm_cell_11/add_1AddV2lstm_cell_11/mul:z:0lstm_cell_11/mul_1:z:0*
T0*(
_output_shapes
:         в2
lstm_cell_11/add_1Л
lstm_cell_11/Sigmoid_2Sigmoidlstm_cell_11/split:output:3*
T0*(
_output_shapes
:         в2
lstm_cell_11/Sigmoid_2}
lstm_cell_11/Relu_1Relulstm_cell_11/add_1:z:0*
T0*(
_output_shapes
:         в2
lstm_cell_11/Relu_1б
lstm_cell_11/mul_2Mullstm_cell_11/Sigmoid_2:y:0!lstm_cell_11/Relu_1:activations:0*
T0*(
_output_shapes
:         в2
lstm_cell_11/mul_2П
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    в   2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_11_matmul_readvariableop_resource-lstm_cell_11_matmul_1_readvariableop_resource,lstm_cell_11_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :         в:         в: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_19549276*
condR
while_cond_19549275*M
output_shapes<
:: : : : :         в:         в: : : : : *
parallel_iterations 2
while╡
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    в   22
0TensorArrayV2Stack/TensorListStack/element_shapeщ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:         в*
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
:         в*
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
:         в2
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
:         в2

Identity╚
NoOpNoOp$^lstm_cell_11/BiasAdd/ReadVariableOp#^lstm_cell_11/MatMul/ReadVariableOp%^lstm_cell_11/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         G: : : 2J
#lstm_cell_11/BiasAdd/ReadVariableOp#lstm_cell_11/BiasAdd/ReadVariableOp2H
"lstm_cell_11/MatMul/ReadVariableOp"lstm_cell_11/MatMul/ReadVariableOp2L
$lstm_cell_11/MatMul_1/ReadVariableOp$lstm_cell_11/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:         G
 
_user_specified_nameinputs
у
═
while_cond_19546920
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_19546920___redundant_placeholder06
2while_while_cond_19546920___redundant_placeholder16
2while_while_cond_19546920___redundant_placeholder26
2while_while_cond_19546920___redundant_placeholder3
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
B: : : : :         в:         в: ::::: 
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
:         в:.*
(
_output_shapes
:         в:

_output_shapes
: :

_output_shapes
:
у
═
while_cond_19546651
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_19546651___redundant_placeholder06
2while_while_cond_19546651___redundant_placeholder16
2while_while_cond_19546651___redundant_placeholder26
2while_while_cond_19546651___redundant_placeholder3
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
B: : : : :         в:         в: ::::: 
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
:         в:.*
(
_output_shapes
:         в:

_output_shapes
: :

_output_shapes
:
╘

э
lstm_11_while_cond_19547585,
(lstm_11_while_lstm_11_while_loop_counter2
.lstm_11_while_lstm_11_while_maximum_iterations
lstm_11_while_placeholder
lstm_11_while_placeholder_1
lstm_11_while_placeholder_2
lstm_11_while_placeholder_3.
*lstm_11_while_less_lstm_11_strided_slice_1F
Blstm_11_while_lstm_11_while_cond_19547585___redundant_placeholder0F
Blstm_11_while_lstm_11_while_cond_19547585___redundant_placeholder1F
Blstm_11_while_lstm_11_while_cond_19547585___redundant_placeholder2F
Blstm_11_while_lstm_11_while_cond_19547585___redundant_placeholder3
lstm_11_while_identity
Ш
lstm_11/while/LessLesslstm_11_while_placeholder*lstm_11_while_less_lstm_11_strided_slice_1*
T0*
_output_shapes
: 2
lstm_11/while/Lessu
lstm_11/while/IdentityIdentitylstm_11/while/Less:z:0*
T0
*
_output_shapes
: 2
lstm_11/while/Identity"9
lstm_11_while_identitylstm_11/while/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :         в:         в: ::::: 
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
:         в:.*
(
_output_shapes
:         в:

_output_shapes
: :

_output_shapes
:
▀
═
while_cond_19546486
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_19546486___redundant_placeholder06
2while_while_cond_19546486___redundant_placeholder16
2while_while_cond_19546486___redundant_placeholder26
2while_while_cond_19546486___redundant_placeholder3
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
@: : : : :         G:         G: ::::: 
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
:         G:-)
'
_output_shapes
:         G:

_output_shapes
: :

_output_shapes
:
║
°
/__inference_lstm_cell_10_layer_call_fn_19549569

inputs
states_0
states_1
unknown:	]Ь
	unknown_0:	GЬ
	unknown_1:	Ь
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
9:         G:         G:         G*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_lstm_cell_10_layer_call_and_return_conditional_losses_195453752
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         G2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:         G2

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:         G2

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
?:         ]:         G:         G: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         ]
 
_user_specified_nameinputs:QM
'
_output_shapes
:         G
"
_user_specified_name
states/0:QM
'
_output_shapes
:         G
"
_user_specified_name
states/1
╘
I
-__inference_dropout_10_layer_call_fn_19548751

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
:         G* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_dropout_10_layer_call_and_return_conditional_losses_195465842
PartitionedCallp
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:         G2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         G:S O
+
_output_shapes
:         G
 
_user_specified_nameinputs
▌
╣
*__inference_lstm_10_layer_call_fn_19548707
inputs_0
unknown:	]Ь
	unknown_0:	GЬ
	unknown_1:	Ь
identityИвStatefulPartitionedCallС
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :                  G*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_lstm_10_layer_call_and_return_conditional_losses_195455222
StatefulPartitionedCallИ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :                  G2

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
inputs/0"иL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*╛
serving_defaultк
K
lstm_10_input:
serving_default_lstm_10_input:0         ]?
dense_54
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
,metrics
trainable_variables
regularization_losses
-layer_regularization_losses
.non_trainable_variables

/layers
		variables
0layer_metrics
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
6metrics
trainable_variables

7states
8layer_regularization_losses
9non_trainable_variables
regularization_losses

:layers
	variables
;layer_metrics
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
<metrics
trainable_variables
=layer_regularization_losses
>non_trainable_variables
regularization_losses

?layers
	variables
@layer_metrics
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
Fmetrics
trainable_variables

Gstates
Hlayer_regularization_losses
Inon_trainable_variables
regularization_losses

Jlayers
	variables
Klayer_metrics
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
Lmetrics
trainable_variables
Mlayer_regularization_losses
Nnon_trainable_variables
regularization_losses

Olayers
	variables
Player_metrics
u__call__
*t&call_and_return_all_conditional_losses
&t"call_and_return_conditional_losses"
_generic_user_object
!:	в2dense_5/kernel
:2dense_5/bias
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
Qmetrics
"trainable_variables
Rlayer_regularization_losses
Snon_trainable_variables
#regularization_losses

Tlayers
$	variables
Ulayer_metrics
w__call__
*v&call_and_return_all_conditional_losses
&v"call_and_return_conditional_losses"
_generic_user_object
.:,	]Ь2lstm_10/lstm_cell_10/kernel
8:6	GЬ2%lstm_10/lstm_cell_10/recurrent_kernel
(:&Ь2lstm_10/lstm_cell_10/bias
.:,	GИ2lstm_11/lstm_cell_11/kernel
9:7
вИ2%lstm_11/lstm_cell_11/recurrent_kernel
(:&И2lstm_11/lstm_cell_11/bias
.
V0
W1"
trackable_list_wrapper
 "
trackable_list_wrapper
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
trackable_dict_wrapper
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
Xmetrics
2trainable_variables
Ylayer_regularization_losses
Znon_trainable_variables
3regularization_losses

[layers
4	variables
\layer_metrics
z__call__
*y&call_and_return_all_conditional_losses
&y"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
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
]metrics
Btrainable_variables
^layer_regularization_losses
_non_trainable_variables
Cregularization_losses

`layers
D	variables
alayer_metrics
|__call__
*{&call_and_return_all_conditional_losses
&{"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
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
#__inference__wrapped_model_19545154lstm_10_input"Ш
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
J__inference_sequential_5_layer_call_and_return_conditional_losses_19547698
J__inference_sequential_5_layer_call_and_return_conditional_losses_19548039
J__inference_sequential_5_layer_call_and_return_conditional_losses_19547323
J__inference_sequential_5_layer_call_and_return_conditional_losses_19547348└
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
/__inference_sequential_5_layer_call_fn_19546808
/__inference_sequential_5_layer_call_fn_19548060
/__inference_sequential_5_layer_call_fn_19548081
/__inference_sequential_5_layer_call_fn_19547298└
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
E__inference_lstm_10_layer_call_and_return_conditional_losses_19548232
E__inference_lstm_10_layer_call_and_return_conditional_losses_19548383
E__inference_lstm_10_layer_call_and_return_conditional_losses_19548534
E__inference_lstm_10_layer_call_and_return_conditional_losses_19548685╒
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
*__inference_lstm_10_layer_call_fn_19548696
*__inference_lstm_10_layer_call_fn_19548707
*__inference_lstm_10_layer_call_fn_19548718
*__inference_lstm_10_layer_call_fn_19548729╒
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
H__inference_dropout_10_layer_call_and_return_conditional_losses_19548734
H__inference_dropout_10_layer_call_and_return_conditional_losses_19548746┤
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
-__inference_dropout_10_layer_call_fn_19548751
-__inference_dropout_10_layer_call_fn_19548756┤
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
E__inference_lstm_11_layer_call_and_return_conditional_losses_19548907
E__inference_lstm_11_layer_call_and_return_conditional_losses_19549058
E__inference_lstm_11_layer_call_and_return_conditional_losses_19549209
E__inference_lstm_11_layer_call_and_return_conditional_losses_19549360╒
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
*__inference_lstm_11_layer_call_fn_19549371
*__inference_lstm_11_layer_call_fn_19549382
*__inference_lstm_11_layer_call_fn_19549393
*__inference_lstm_11_layer_call_fn_19549404╒
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
H__inference_dropout_11_layer_call_and_return_conditional_losses_19549409
H__inference_dropout_11_layer_call_and_return_conditional_losses_19549421┤
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
-__inference_dropout_11_layer_call_fn_19549426
-__inference_dropout_11_layer_call_fn_19549431┤
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
E__inference_dense_5_layer_call_and_return_conditional_losses_19549462в
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
*__inference_dense_5_layer_call_fn_19549471в
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
&__inference_signature_wrapper_19547371lstm_10_input"Ф
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
J__inference_lstm_cell_10_layer_call_and_return_conditional_losses_19549503
J__inference_lstm_cell_10_layer_call_and_return_conditional_losses_19549535╛
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
/__inference_lstm_cell_10_layer_call_fn_19549552
/__inference_lstm_cell_10_layer_call_fn_19549569╛
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
J__inference_lstm_cell_11_layer_call_and_return_conditional_losses_19549601
J__inference_lstm_cell_11_layer_call_and_return_conditional_losses_19549633╛
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
/__inference_lstm_cell_11_layer_call_fn_19549650
/__inference_lstm_cell_11_layer_call_fn_19549667╛
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
#__inference__wrapped_model_19545154}&'()*+ !:в7
0в-
+К(
lstm_10_input         ]
к "5к2
0
dense_5%К"
dense_5         о
E__inference_dense_5_layer_call_and_return_conditional_losses_19549462e !4в1
*в'
%К"
inputs         в
к ")в&
К
0         
Ъ Ж
*__inference_dense_5_layer_call_fn_19549471X !4в1
*в'
%К"
inputs         в
к "К         ░
H__inference_dropout_10_layer_call_and_return_conditional_losses_19548734d7в4
-в*
$К!
inputs         G
p 
к ")в&
К
0         G
Ъ ░
H__inference_dropout_10_layer_call_and_return_conditional_losses_19548746d7в4
-в*
$К!
inputs         G
p
к ")в&
К
0         G
Ъ И
-__inference_dropout_10_layer_call_fn_19548751W7в4
-в*
$К!
inputs         G
p 
к "К         GИ
-__inference_dropout_10_layer_call_fn_19548756W7в4
-в*
$К!
inputs         G
p
к "К         G▓
H__inference_dropout_11_layer_call_and_return_conditional_losses_19549409f8в5
.в+
%К"
inputs         в
p 
к "*в'
 К
0         в
Ъ ▓
H__inference_dropout_11_layer_call_and_return_conditional_losses_19549421f8в5
.в+
%К"
inputs         в
p
к "*в'
 К
0         в
Ъ К
-__inference_dropout_11_layer_call_fn_19549426Y8в5
.в+
%К"
inputs         в
p 
к "К         вК
-__inference_dropout_11_layer_call_fn_19549431Y8в5
.в+
%К"
inputs         в
p
к "К         в╘
E__inference_lstm_10_layer_call_and_return_conditional_losses_19548232К&'(OвL
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
0                  G
Ъ ╘
E__inference_lstm_10_layer_call_and_return_conditional_losses_19548383К&'(OвL
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
0                  G
Ъ ║
E__inference_lstm_10_layer_call_and_return_conditional_losses_19548534q&'(?в<
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
0         G
Ъ ║
E__inference_lstm_10_layer_call_and_return_conditional_losses_19548685q&'(?в<
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
0         G
Ъ л
*__inference_lstm_10_layer_call_fn_19548696}&'(OвL
EвB
4Ъ1
/К,
inputs/0                  ]

 
p 

 
к "%К"                  Gл
*__inference_lstm_10_layer_call_fn_19548707}&'(OвL
EвB
4Ъ1
/К,
inputs/0                  ]

 
p

 
к "%К"                  GТ
*__inference_lstm_10_layer_call_fn_19548718d&'(?в<
5в2
$К!
inputs         ]

 
p 

 
к "К         GТ
*__inference_lstm_10_layer_call_fn_19548729d&'(?в<
5в2
$К!
inputs         ]

 
p

 
к "К         G╒
E__inference_lstm_11_layer_call_and_return_conditional_losses_19548907Л)*+OвL
EвB
4Ъ1
/К,
inputs/0                  G

 
p 

 
к "3в0
)К&
0                  в
Ъ ╒
E__inference_lstm_11_layer_call_and_return_conditional_losses_19549058Л)*+OвL
EвB
4Ъ1
/К,
inputs/0                  G

 
p

 
к "3в0
)К&
0                  в
Ъ ╗
E__inference_lstm_11_layer_call_and_return_conditional_losses_19549209r)*+?в<
5в2
$К!
inputs         G

 
p 

 
к "*в'
 К
0         в
Ъ ╗
E__inference_lstm_11_layer_call_and_return_conditional_losses_19549360r)*+?в<
5в2
$К!
inputs         G

 
p

 
к "*в'
 К
0         в
Ъ м
*__inference_lstm_11_layer_call_fn_19549371~)*+OвL
EвB
4Ъ1
/К,
inputs/0                  G

 
p 

 
к "&К#                  вм
*__inference_lstm_11_layer_call_fn_19549382~)*+OвL
EвB
4Ъ1
/К,
inputs/0                  G

 
p

 
к "&К#                  вУ
*__inference_lstm_11_layer_call_fn_19549393e)*+?в<
5в2
$К!
inputs         G

 
p 

 
к "К         вУ
*__inference_lstm_11_layer_call_fn_19549404e)*+?в<
5в2
$К!
inputs         G

 
p

 
к "К         в╠
J__inference_lstm_cell_10_layer_call_and_return_conditional_losses_19549503¤&'(Ав}
vвs
 К
inputs         ]
KвH
"К
states/0         G
"К
states/1         G
p 
к "sвp
iвf
К
0/0         G
EЪB
К
0/1/0         G
К
0/1/1         G
Ъ ╠
J__inference_lstm_cell_10_layer_call_and_return_conditional_losses_19549535¤&'(Ав}
vвs
 К
inputs         ]
KвH
"К
states/0         G
"К
states/1         G
p
к "sвp
iвf
К
0/0         G
EЪB
К
0/1/0         G
К
0/1/1         G
Ъ б
/__inference_lstm_cell_10_layer_call_fn_19549552э&'(Ав}
vвs
 К
inputs         ]
KвH
"К
states/0         G
"К
states/1         G
p 
к "cв`
К
0         G
AЪ>
К
1/0         G
К
1/1         Gб
/__inference_lstm_cell_10_layer_call_fn_19549569э&'(Ав}
vвs
 К
inputs         ]
KвH
"К
states/0         G
"К
states/1         G
p
к "cв`
К
0         G
AЪ>
К
1/0         G
К
1/1         G╤
J__inference_lstm_cell_11_layer_call_and_return_conditional_losses_19549601В)*+Вв
xвu
 К
inputs         G
MвJ
#К 
states/0         в
#К 
states/1         в
p 
к "vвs
lвi
К
0/0         в
GЪD
 К
0/1/0         в
 К
0/1/1         в
Ъ ╤
J__inference_lstm_cell_11_layer_call_and_return_conditional_losses_19549633В)*+Вв
xвu
 К
inputs         G
MвJ
#К 
states/0         в
#К 
states/1         в
p
к "vвs
lвi
К
0/0         в
GЪD
 К
0/1/0         в
 К
0/1/1         в
Ъ ж
/__inference_lstm_cell_11_layer_call_fn_19549650Є)*+Вв
xвu
 К
inputs         G
MвJ
#К 
states/0         в
#К 
states/1         в
p 
к "fвc
К
0         в
CЪ@
К
1/0         в
К
1/1         вж
/__inference_lstm_cell_11_layer_call_fn_19549667Є)*+Вв
xвu
 К
inputs         G
MвJ
#К 
states/0         в
#К 
states/1         в
p
к "fвc
К
0         в
CЪ@
К
1/0         в
К
1/1         в╟
J__inference_sequential_5_layer_call_and_return_conditional_losses_19547323y&'()*+ !Bв?
8в5
+К(
lstm_10_input         ]
p 

 
к ")в&
К
0         
Ъ ╟
J__inference_sequential_5_layer_call_and_return_conditional_losses_19547348y&'()*+ !Bв?
8в5
+К(
lstm_10_input         ]
p

 
к ")в&
К
0         
Ъ └
J__inference_sequential_5_layer_call_and_return_conditional_losses_19547698r&'()*+ !;в8
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
J__inference_sequential_5_layer_call_and_return_conditional_losses_19548039r&'()*+ !;в8
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
/__inference_sequential_5_layer_call_fn_19546808l&'()*+ !Bв?
8в5
+К(
lstm_10_input         ]
p 

 
к "К         Я
/__inference_sequential_5_layer_call_fn_19547298l&'()*+ !Bв?
8в5
+К(
lstm_10_input         ]
p

 
к "К         Ш
/__inference_sequential_5_layer_call_fn_19548060e&'()*+ !;в8
1в.
$К!
inputs         ]
p 

 
к "К         Ш
/__inference_sequential_5_layer_call_fn_19548081e&'()*+ !;в8
1в.
$К!
inputs         ]
p

 
к "К         ╣
&__inference_signature_wrapper_19547371О&'()*+ !KвH
в 
Aк>
<
lstm_10_input+К(
lstm_10_input         ]"5к2
0
dense_5%К"
dense_5         