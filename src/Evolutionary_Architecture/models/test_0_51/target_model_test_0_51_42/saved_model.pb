╟Ы&
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
И"serve*2.6.02v2.6.0-rc2-32-g919f693420e8Н╚$
{
dense_13/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	Ь* 
shared_namedense_13/kernel
t
#dense_13/kernel/Read/ReadVariableOpReadVariableOpdense_13/kernel*
_output_shapes
:	Ь*
dtype0
r
dense_13/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_13/bias
k
!dense_13/bias/Read/ReadVariableOpReadVariableOpdense_13/bias*
_output_shapes
:*
dtype0
У
lstm_26/lstm_cell_26/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	]И*,
shared_namelstm_26/lstm_cell_26/kernel
М
/lstm_26/lstm_cell_26/kernel/Read/ReadVariableOpReadVariableOplstm_26/lstm_cell_26/kernel*
_output_shapes
:	]И*
dtype0
и
%lstm_26/lstm_cell_26/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ВИ*6
shared_name'%lstm_26/lstm_cell_26/recurrent_kernel
б
9lstm_26/lstm_cell_26/recurrent_kernel/Read/ReadVariableOpReadVariableOp%lstm_26/lstm_cell_26/recurrent_kernel* 
_output_shapes
:
ВИ*
dtype0
Л
lstm_26/lstm_cell_26/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:И**
shared_namelstm_26/lstm_cell_26/bias
Д
-lstm_26/lstm_cell_26/bias/Read/ReadVariableOpReadVariableOplstm_26/lstm_cell_26/bias*
_output_shapes	
:И*
dtype0
Ф
lstm_27/lstm_cell_27/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ВЁ*,
shared_namelstm_27/lstm_cell_27/kernel
Н
/lstm_27/lstm_cell_27/kernel/Read/ReadVariableOpReadVariableOplstm_27/lstm_cell_27/kernel* 
_output_shapes
:
ВЁ*
dtype0
и
%lstm_27/lstm_cell_27/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ЬЁ*6
shared_name'%lstm_27/lstm_cell_27/recurrent_kernel
б
9lstm_27/lstm_cell_27/recurrent_kernel/Read/ReadVariableOpReadVariableOp%lstm_27/lstm_cell_27/recurrent_kernel* 
_output_shapes
:
ЬЁ*
dtype0
Л
lstm_27/lstm_cell_27/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:Ё**
shared_namelstm_27/lstm_cell_27/bias
Д
-lstm_27/lstm_cell_27/bias/Read/ReadVariableOpReadVariableOplstm_27/lstm_cell_27/bias*
_output_shapes	
:Ё*
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
з"
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*т!
value╪!B╒! B╬!
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
[Y
VARIABLE_VALUEdense_13/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_13/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
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
VARIABLE_VALUElstm_26/lstm_cell_26/kernel0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUE
ki
VARIABLE_VALUE%lstm_26/lstm_cell_26/recurrent_kernel0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUElstm_26/lstm_cell_26/bias0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUElstm_27/lstm_cell_27/kernel0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUE
ki
VARIABLE_VALUE%lstm_27/lstm_cell_27/recurrent_kernel0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUElstm_27/lstm_cell_27/bias0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUE
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
serving_default_lstm_26_inputPlaceholder*+
_output_shapes
:         ]*
dtype0* 
shape:         ]
м
StatefulPartitionedCallStatefulPartitionedCallserving_default_lstm_26_inputlstm_26/lstm_cell_26/kernel%lstm_26/lstm_cell_26/recurrent_kernellstm_26/lstm_cell_26/biaslstm_27/lstm_cell_27/kernel%lstm_27/lstm_cell_27/recurrent_kernellstm_27/lstm_cell_27/biasdense_13/kerneldense_13/bias*
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
&__inference_signature_wrapper_40074275
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
Ч
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#dense_13/kernel/Read/ReadVariableOp!dense_13/bias/Read/ReadVariableOp/lstm_26/lstm_cell_26/kernel/Read/ReadVariableOp9lstm_26/lstm_cell_26/recurrent_kernel/Read/ReadVariableOp-lstm_26/lstm_cell_26/bias/Read/ReadVariableOp/lstm_27/lstm_cell_27/kernel/Read/ReadVariableOp9lstm_27/lstm_cell_27/recurrent_kernel/Read/ReadVariableOp-lstm_27/lstm_cell_27/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOpConst*
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
!__inference__traced_save_40076630
в
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_13/kerneldense_13/biaslstm_26/lstm_cell_26/kernel%lstm_26/lstm_cell_26/recurrent_kernellstm_26/lstm_cell_26/biaslstm_27/lstm_cell_27/kernel%lstm_27/lstm_cell_27/recurrent_kernellstm_27/lstm_cell_27/biastotalcounttotal_1count_1*
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
$__inference__traced_restore_40076676д¤#
Л
З
J__inference_lstm_cell_26_layer_call_and_return_conditional_losses_40072279

inputs

states
states_11
matmul_readvariableop_resource:	]И4
 matmul_1_readvariableop_resource:
ВИ.
biasadd_readvariableop_resource:	И
identity

identity_1

identity_2ИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpвMatMul_1/ReadVariableOpО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	]И*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         И2
MatMulХ
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
ВИ*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         И2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:         И2
addН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:И*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         И2	
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
P:         В:         В:         В:         В*
	num_split2
split`
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:         В2	
Sigmoidd
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:         В2
	Sigmoid_1]
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:         В2
mulW
ReluRelusplit:output:2*
T0*(
_output_shapes
:         В2
Relui
mul_1MulSigmoid:y:0Relu:activations:0*
T0*(
_output_shapes
:         В2
mul_1^
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:         В2
add_1d
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:         В2
	Sigmoid_2V
Relu_1Relu	add_1:z:0*
T0*(
_output_shapes
:         В2
Relu_1m
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*(
_output_shapes
:         В2
mul_2e
IdentityIdentity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:         В2

Identityi

Identity_1Identity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:         В2

Identity_1i

Identity_2Identity	add_1:z:0^NoOp*
T0*(
_output_shapes
:         В2

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
A:         ]:         В:         В: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:         ]
 
_user_specified_nameinputs:PL
(
_output_shapes
:         В
 
_user_specified_namestates:PL
(
_output_shapes
:         В
 
_user_specified_namestates
у
═
while_cond_40072776
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_40072776___redundant_placeholder06
2while_while_cond_40072776___redundant_placeholder16
2while_while_cond_40072776___redundant_placeholder26
2while_while_cond_40072776___redundant_placeholder3
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
B: : : : :         Ь:         Ь: ::::: 
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
:         Ь:.*
(
_output_shapes
:         Ь:

_output_shapes
: :

_output_shapes
:
о

╘
0__inference_sequential_13_layer_call_fn_40073712
lstm_26_input
unknown:	]И
	unknown_0:
ВИ
	unknown_1:	И
	unknown_2:
ВЁ
	unknown_3:
ЬЁ
	unknown_4:	Ё
	unknown_5:	Ь
	unknown_6:
identityИвStatefulPartitionedCall╘
StatefulPartitionedCallStatefulPartitionedCalllstm_26_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
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
GPU 2J 8В *T
fORM
K__inference_sequential_13_layer_call_and_return_conditional_losses_400736932
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
_user_specified_namelstm_26_input
┤?
╓
while_body_40073825
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
3while_lstm_cell_27_matmul_readvariableop_resource_0:
ВЁI
5while_lstm_cell_27_matmul_1_readvariableop_resource_0:
ЬЁC
4while_lstm_cell_27_biasadd_readvariableop_resource_0:	Ё
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
1while_lstm_cell_27_matmul_readvariableop_resource:
ВЁG
3while_lstm_cell_27_matmul_1_readvariableop_resource:
ЬЁA
2while_lstm_cell_27_biasadd_readvariableop_resource:	ЁИв)while/lstm_cell_27/BiasAdd/ReadVariableOpв(while/lstm_cell_27/MatMul/ReadVariableOpв*while/lstm_cell_27/MatMul_1/ReadVariableOp├
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    В   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape╘
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:         В*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem╩
(while/lstm_cell_27/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_27_matmul_readvariableop_resource_0* 
_output_shapes
:
ВЁ*
dtype02*
(while/lstm_cell_27/MatMul/ReadVariableOp╫
while/lstm_cell_27/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_27/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ё2
while/lstm_cell_27/MatMul╨
*while/lstm_cell_27/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_27_matmul_1_readvariableop_resource_0* 
_output_shapes
:
ЬЁ*
dtype02,
*while/lstm_cell_27/MatMul_1/ReadVariableOp└
while/lstm_cell_27/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_27/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ё2
while/lstm_cell_27/MatMul_1╕
while/lstm_cell_27/addAddV2#while/lstm_cell_27/MatMul:product:0%while/lstm_cell_27/MatMul_1:product:0*
T0*(
_output_shapes
:         Ё2
while/lstm_cell_27/add╚
)while/lstm_cell_27/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_27_biasadd_readvariableop_resource_0*
_output_shapes	
:Ё*
dtype02+
)while/lstm_cell_27/BiasAdd/ReadVariableOp┼
while/lstm_cell_27/BiasAddBiasAddwhile/lstm_cell_27/add:z:01while/lstm_cell_27/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ё2
while/lstm_cell_27/BiasAddК
"while/lstm_cell_27/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_27/split/split_dimП
while/lstm_cell_27/splitSplit+while/lstm_cell_27/split/split_dim:output:0#while/lstm_cell_27/BiasAdd:output:0*
T0*d
_output_shapesR
P:         Ь:         Ь:         Ь:         Ь*
	num_split2
while/lstm_cell_27/splitЩ
while/lstm_cell_27/SigmoidSigmoid!while/lstm_cell_27/split:output:0*
T0*(
_output_shapes
:         Ь2
while/lstm_cell_27/SigmoidЭ
while/lstm_cell_27/Sigmoid_1Sigmoid!while/lstm_cell_27/split:output:1*
T0*(
_output_shapes
:         Ь2
while/lstm_cell_27/Sigmoid_1б
while/lstm_cell_27/mulMul while/lstm_cell_27/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:         Ь2
while/lstm_cell_27/mulР
while/lstm_cell_27/ReluRelu!while/lstm_cell_27/split:output:2*
T0*(
_output_shapes
:         Ь2
while/lstm_cell_27/Relu╡
while/lstm_cell_27/mul_1Mulwhile/lstm_cell_27/Sigmoid:y:0%while/lstm_cell_27/Relu:activations:0*
T0*(
_output_shapes
:         Ь2
while/lstm_cell_27/mul_1к
while/lstm_cell_27/add_1AddV2while/lstm_cell_27/mul:z:0while/lstm_cell_27/mul_1:z:0*
T0*(
_output_shapes
:         Ь2
while/lstm_cell_27/add_1Э
while/lstm_cell_27/Sigmoid_2Sigmoid!while/lstm_cell_27/split:output:3*
T0*(
_output_shapes
:         Ь2
while/lstm_cell_27/Sigmoid_2П
while/lstm_cell_27/Relu_1Reluwhile/lstm_cell_27/add_1:z:0*
T0*(
_output_shapes
:         Ь2
while/lstm_cell_27/Relu_1╣
while/lstm_cell_27/mul_2Mul while/lstm_cell_27/Sigmoid_2:y:0'while/lstm_cell_27/Relu_1:activations:0*
T0*(
_output_shapes
:         Ь2
while/lstm_cell_27/mul_2р
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_27/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_27/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:         Ь2
while/Identity_4О
while/Identity_5Identitywhile/lstm_cell_27/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:         Ь2
while/Identity_5▐

while/NoOpNoOp*^while/lstm_cell_27/BiasAdd/ReadVariableOp)^while/lstm_cell_27/MatMul/ReadVariableOp+^while/lstm_cell_27/MatMul_1/ReadVariableOp*"
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
2while_lstm_cell_27_biasadd_readvariableop_resource4while_lstm_cell_27_biasadd_readvariableop_resource_0"l
3while_lstm_cell_27_matmul_1_readvariableop_resource5while_lstm_cell_27_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_27_matmul_readvariableop_resource3while_lstm_cell_27_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :         Ь:         Ь: : : : : 2V
)while/lstm_cell_27/BiasAdd/ReadVariableOp)while/lstm_cell_27/BiasAdd/ReadVariableOp2T
(while/lstm_cell_27/MatMul/ReadVariableOp(while/lstm_cell_27/MatMul/ReadVariableOp2X
*while/lstm_cell_27/MatMul_1/ReadVariableOp*while/lstm_cell_27/MatMul_1/ReadVariableOp: 
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
:         Ь:.*
(
_output_shapes
:         Ь:

_output_shapes
: :

_output_shapes
: 
├\
а
E__inference_lstm_26_layer_call_and_return_conditional_losses_40075287
inputs_0>
+lstm_cell_26_matmul_readvariableop_resource:	]ИA
-lstm_cell_26_matmul_1_readvariableop_resource:
ВИ;
,lstm_cell_26_biasadd_readvariableop_resource:	И
identityИв#lstm_cell_26/BiasAdd/ReadVariableOpв"lstm_cell_26/MatMul/ReadVariableOpв$lstm_cell_26/MatMul_1/ReadVariableOpвwhileF
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
B :В2
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
B :В2
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
:         В2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :В2
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
B :В2
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
:         В2	
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
"lstm_cell_26/MatMul/ReadVariableOpReadVariableOp+lstm_cell_26_matmul_readvariableop_resource*
_output_shapes
:	]И*
dtype02$
"lstm_cell_26/MatMul/ReadVariableOpн
lstm_cell_26/MatMulMatMulstrided_slice_2:output:0*lstm_cell_26/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         И2
lstm_cell_26/MatMul╝
$lstm_cell_26/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_26_matmul_1_readvariableop_resource* 
_output_shapes
:
ВИ*
dtype02&
$lstm_cell_26/MatMul_1/ReadVariableOpй
lstm_cell_26/MatMul_1MatMulzeros:output:0,lstm_cell_26/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         И2
lstm_cell_26/MatMul_1а
lstm_cell_26/addAddV2lstm_cell_26/MatMul:product:0lstm_cell_26/MatMul_1:product:0*
T0*(
_output_shapes
:         И2
lstm_cell_26/add┤
#lstm_cell_26/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_26_biasadd_readvariableop_resource*
_output_shapes	
:И*
dtype02%
#lstm_cell_26/BiasAdd/ReadVariableOpн
lstm_cell_26/BiasAddBiasAddlstm_cell_26/add:z:0+lstm_cell_26/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         И2
lstm_cell_26/BiasAdd~
lstm_cell_26/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_26/split/split_dimў
lstm_cell_26/splitSplit%lstm_cell_26/split/split_dim:output:0lstm_cell_26/BiasAdd:output:0*
T0*d
_output_shapesR
P:         В:         В:         В:         В*
	num_split2
lstm_cell_26/splitЗ
lstm_cell_26/SigmoidSigmoidlstm_cell_26/split:output:0*
T0*(
_output_shapes
:         В2
lstm_cell_26/SigmoidЛ
lstm_cell_26/Sigmoid_1Sigmoidlstm_cell_26/split:output:1*
T0*(
_output_shapes
:         В2
lstm_cell_26/Sigmoid_1М
lstm_cell_26/mulMullstm_cell_26/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:         В2
lstm_cell_26/mul~
lstm_cell_26/ReluRelulstm_cell_26/split:output:2*
T0*(
_output_shapes
:         В2
lstm_cell_26/ReluЭ
lstm_cell_26/mul_1Mullstm_cell_26/Sigmoid:y:0lstm_cell_26/Relu:activations:0*
T0*(
_output_shapes
:         В2
lstm_cell_26/mul_1Т
lstm_cell_26/add_1AddV2lstm_cell_26/mul:z:0lstm_cell_26/mul_1:z:0*
T0*(
_output_shapes
:         В2
lstm_cell_26/add_1Л
lstm_cell_26/Sigmoid_2Sigmoidlstm_cell_26/split:output:3*
T0*(
_output_shapes
:         В2
lstm_cell_26/Sigmoid_2}
lstm_cell_26/Relu_1Relulstm_cell_26/add_1:z:0*
T0*(
_output_shapes
:         В2
lstm_cell_26/Relu_1б
lstm_cell_26/mul_2Mullstm_cell_26/Sigmoid_2:y:0!lstm_cell_26/Relu_1:activations:0*
T0*(
_output_shapes
:         В2
lstm_cell_26/mul_2П
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    В   2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_26_matmul_readvariableop_resource-lstm_cell_26_matmul_1_readvariableop_resource,lstm_cell_26_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :         В:         В: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_40075203*
condR
while_cond_40075202*M
output_shapes<
:: : : : :         В:         В: : : : : *
parallel_iterations 2
while╡
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    В   22
0TensorArrayV2Stack/TensorListStack/element_shapeЄ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:                  В*
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
:         В*
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
!:                  В2
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
!:                  В2

Identity╚
NoOpNoOp$^lstm_cell_26/BiasAdd/ReadVariableOp#^lstm_cell_26/MatMul/ReadVariableOp%^lstm_cell_26/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  ]: : : 2J
#lstm_cell_26/BiasAdd/ReadVariableOp#lstm_cell_26/BiasAdd/ReadVariableOp2H
"lstm_cell_26/MatMul/ReadVariableOp"lstm_cell_26/MatMul/ReadVariableOp2L
$lstm_cell_26/MatMul_1/ReadVariableOp$lstm_cell_26/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :                  ]
"
_user_specified_name
inputs/0
у
═
while_cond_40073555
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_40073555___redundant_placeholder06
2while_while_cond_40073555___redundant_placeholder16
2while_while_cond_40073555___redundant_placeholder26
2while_while_cond_40073555___redundant_placeholder3
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
B: : : : :         Ь:         Ь: ::::: 
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
:         Ь:.*
(
_output_shapes
:         Ь:

_output_shapes
: :

_output_shapes
:
Е&
є
while_body_40072357
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_00
while_lstm_cell_26_40072381_0:	]И1
while_lstm_cell_26_40072383_0:
ВИ,
while_lstm_cell_26_40072385_0:	И
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor.
while_lstm_cell_26_40072381:	]И/
while_lstm_cell_26_40072383:
ВИ*
while_lstm_cell_26_40072385:	ИИв*while/lstm_cell_26/StatefulPartitionedCall├
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
*while/lstm_cell_26/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_26_40072381_0while_lstm_cell_26_40072383_0while_lstm_cell_26_40072385_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:         В:         В:         В*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_lstm_cell_26_layer_call_and_return_conditional_losses_400722792,
*while/lstm_cell_26/StatefulPartitionedCallў
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder3while/lstm_cell_26/StatefulPartitionedCall:output:0*
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
while/Identity_4Identity3while/lstm_cell_26/StatefulPartitionedCall:output:1^while/NoOp*
T0*(
_output_shapes
:         В2
while/Identity_4е
while/Identity_5Identity3while/lstm_cell_26/StatefulPartitionedCall:output:2^while/NoOp*
T0*(
_output_shapes
:         В2
while/Identity_5З

while/NoOpNoOp+^while/lstm_cell_26/StatefulPartitionedCall*"
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
while_lstm_cell_26_40072381while_lstm_cell_26_40072381_0"<
while_lstm_cell_26_40072383while_lstm_cell_26_40072383_0"<
while_lstm_cell_26_40072385while_lstm_cell_26_40072385_0"0
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :         В:         В: : : : : 2X
*while/lstm_cell_26/StatefulPartitionedCall*while/lstm_cell_26/StatefulPartitionedCall: 
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
:         В:.*
(
_output_shapes
:         В:

_output_shapes
: :

_output_shapes
: 
К\
Я
E__inference_lstm_27_layer_call_and_return_conditional_losses_40076113

inputs?
+lstm_cell_27_matmul_readvariableop_resource:
ВЁA
-lstm_cell_27_matmul_1_readvariableop_resource:
ЬЁ;
,lstm_cell_27_biasadd_readvariableop_resource:	Ё
identityИв#lstm_cell_27/BiasAdd/ReadVariableOpв"lstm_cell_27/MatMul/ReadVariableOpв$lstm_cell_27/MatMul_1/ReadVariableOpвwhileD
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
B :Ь2
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
B :Ь2
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
:         Ь2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :Ь2
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
B :Ь2
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
:         Ь2	
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
:         В2
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
valueB"    В   27
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
:         В*
shrink_axis_mask2
strided_slice_2╢
"lstm_cell_27/MatMul/ReadVariableOpReadVariableOp+lstm_cell_27_matmul_readvariableop_resource* 
_output_shapes
:
ВЁ*
dtype02$
"lstm_cell_27/MatMul/ReadVariableOpн
lstm_cell_27/MatMulMatMulstrided_slice_2:output:0*lstm_cell_27/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ё2
lstm_cell_27/MatMul╝
$lstm_cell_27/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_27_matmul_1_readvariableop_resource* 
_output_shapes
:
ЬЁ*
dtype02&
$lstm_cell_27/MatMul_1/ReadVariableOpй
lstm_cell_27/MatMul_1MatMulzeros:output:0,lstm_cell_27/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ё2
lstm_cell_27/MatMul_1а
lstm_cell_27/addAddV2lstm_cell_27/MatMul:product:0lstm_cell_27/MatMul_1:product:0*
T0*(
_output_shapes
:         Ё2
lstm_cell_27/add┤
#lstm_cell_27/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_27_biasadd_readvariableop_resource*
_output_shapes	
:Ё*
dtype02%
#lstm_cell_27/BiasAdd/ReadVariableOpн
lstm_cell_27/BiasAddBiasAddlstm_cell_27/add:z:0+lstm_cell_27/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ё2
lstm_cell_27/BiasAdd~
lstm_cell_27/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_27/split/split_dimў
lstm_cell_27/splitSplit%lstm_cell_27/split/split_dim:output:0lstm_cell_27/BiasAdd:output:0*
T0*d
_output_shapesR
P:         Ь:         Ь:         Ь:         Ь*
	num_split2
lstm_cell_27/splitЗ
lstm_cell_27/SigmoidSigmoidlstm_cell_27/split:output:0*
T0*(
_output_shapes
:         Ь2
lstm_cell_27/SigmoidЛ
lstm_cell_27/Sigmoid_1Sigmoidlstm_cell_27/split:output:1*
T0*(
_output_shapes
:         Ь2
lstm_cell_27/Sigmoid_1М
lstm_cell_27/mulMullstm_cell_27/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:         Ь2
lstm_cell_27/mul~
lstm_cell_27/ReluRelulstm_cell_27/split:output:2*
T0*(
_output_shapes
:         Ь2
lstm_cell_27/ReluЭ
lstm_cell_27/mul_1Mullstm_cell_27/Sigmoid:y:0lstm_cell_27/Relu:activations:0*
T0*(
_output_shapes
:         Ь2
lstm_cell_27/mul_1Т
lstm_cell_27/add_1AddV2lstm_cell_27/mul:z:0lstm_cell_27/mul_1:z:0*
T0*(
_output_shapes
:         Ь2
lstm_cell_27/add_1Л
lstm_cell_27/Sigmoid_2Sigmoidlstm_cell_27/split:output:3*
T0*(
_output_shapes
:         Ь2
lstm_cell_27/Sigmoid_2}
lstm_cell_27/Relu_1Relulstm_cell_27/add_1:z:0*
T0*(
_output_shapes
:         Ь2
lstm_cell_27/Relu_1б
lstm_cell_27/mul_2Mullstm_cell_27/Sigmoid_2:y:0!lstm_cell_27/Relu_1:activations:0*
T0*(
_output_shapes
:         Ь2
lstm_cell_27/mul_2П
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_27_matmul_readvariableop_resource-lstm_cell_27_matmul_1_readvariableop_resource,lstm_cell_27_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :         Ь:         Ь: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_40076029*
condR
while_cond_40076028*M
output_shapes<
:: : : : :         Ь:         Ь: : : : : *
parallel_iterations 2
while╡
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      22
0TensorArrayV2Stack/TensorListStack/element_shapeщ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:         Ь*
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
:         Ь*
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
:         Ь2
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
:         Ь2

Identity╚
NoOpNoOp$^lstm_cell_27/BiasAdd/ReadVariableOp#^lstm_cell_27/MatMul/ReadVariableOp%^lstm_cell_27/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:         В: : : 2J
#lstm_cell_27/BiasAdd/ReadVariableOp#lstm_cell_27/BiasAdd/ReadVariableOp2H
"lstm_cell_27/MatMul/ReadVariableOp"lstm_cell_27/MatMul/ReadVariableOp2L
$lstm_cell_27/MatMul_1/ReadVariableOp$lstm_cell_27/MatMul_1/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:         В
 
_user_specified_nameinputs
∙
Е
)sequential_13_lstm_26_while_cond_40071797H
Dsequential_13_lstm_26_while_sequential_13_lstm_26_while_loop_counterN
Jsequential_13_lstm_26_while_sequential_13_lstm_26_while_maximum_iterations+
'sequential_13_lstm_26_while_placeholder-
)sequential_13_lstm_26_while_placeholder_1-
)sequential_13_lstm_26_while_placeholder_2-
)sequential_13_lstm_26_while_placeholder_3J
Fsequential_13_lstm_26_while_less_sequential_13_lstm_26_strided_slice_1b
^sequential_13_lstm_26_while_sequential_13_lstm_26_while_cond_40071797___redundant_placeholder0b
^sequential_13_lstm_26_while_sequential_13_lstm_26_while_cond_40071797___redundant_placeholder1b
^sequential_13_lstm_26_while_sequential_13_lstm_26_while_cond_40071797___redundant_placeholder2b
^sequential_13_lstm_26_while_sequential_13_lstm_26_while_cond_40071797___redundant_placeholder3(
$sequential_13_lstm_26_while_identity
▐
 sequential_13/lstm_26/while/LessLess'sequential_13_lstm_26_while_placeholderFsequential_13_lstm_26_while_less_sequential_13_lstm_26_strided_slice_1*
T0*
_output_shapes
: 2"
 sequential_13/lstm_26/while/LessЯ
$sequential_13/lstm_26/while/IdentityIdentity$sequential_13/lstm_26/while/Less:z:0*
T0
*
_output_shapes
: 2&
$sequential_13/lstm_26/while/Identity"U
$sequential_13_lstm_26_while_identity-sequential_13/lstm_26/while/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :         В:         В: ::::: 
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
:         В:.*
(
_output_shapes
:         В:

_output_shapes
: :

_output_shapes
:
╡
╗
K__inference_sequential_13_layer_call_and_return_conditional_losses_40074162

inputs#
lstm_26_40074140:	]И$
lstm_26_40074142:
ВИ
lstm_26_40074144:	И$
lstm_27_40074148:
ВЁ$
lstm_27_40074150:
ЬЁ
lstm_27_40074152:	Ё$
dense_13_40074156:	Ь
dense_13_40074158:
identityИв dense_13/StatefulPartitionedCallв"dropout_26/StatefulPartitionedCallв"dropout_27/StatefulPartitionedCallвlstm_26/StatefulPartitionedCallвlstm_27/StatefulPartitionedCallо
lstm_26/StatefulPartitionedCallStatefulPartitionedCallinputslstm_26_40074140lstm_26_40074142lstm_26_40074144*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         В*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_lstm_26_layer_call_and_return_conditional_losses_400741052!
lstm_26/StatefulPartitionedCallЫ
"dropout_26/StatefulPartitionedCallStatefulPartitionedCall(lstm_26/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         В* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_dropout_26_layer_call_and_return_conditional_losses_400739382$
"dropout_26/StatefulPartitionedCall╙
lstm_27/StatefulPartitionedCallStatefulPartitionedCall+dropout_26/StatefulPartitionedCall:output:0lstm_27_40074148lstm_27_40074150lstm_27_40074152*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         Ь*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_lstm_27_layer_call_and_return_conditional_losses_400739092!
lstm_27/StatefulPartitionedCall└
"dropout_27/StatefulPartitionedCallStatefulPartitionedCall(lstm_27/StatefulPartitionedCall:output:0#^dropout_26/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         Ь* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_dropout_27_layer_call_and_return_conditional_losses_400737422$
"dropout_27/StatefulPartitionedCall├
 dense_13/StatefulPartitionedCallStatefulPartitionedCall+dropout_27/StatefulPartitionedCall:output:0dense_13_40074156dense_13_40074158*
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
GPU 2J 8В *O
fJRH
F__inference_dense_13_layer_call_and_return_conditional_losses_400736862"
 dense_13/StatefulPartitionedCallИ
IdentityIdentity)dense_13/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:         2

Identity 
NoOpNoOp!^dense_13/StatefulPartitionedCall#^dropout_26/StatefulPartitionedCall#^dropout_27/StatefulPartitionedCall ^lstm_26/StatefulPartitionedCall ^lstm_27/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':         ]: : : : : : : : 2D
 dense_13/StatefulPartitionedCall dense_13/StatefulPartitionedCall2H
"dropout_26/StatefulPartitionedCall"dropout_26/StatefulPartitionedCall2H
"dropout_27/StatefulPartitionedCall"dropout_27/StatefulPartitionedCall2B
lstm_26/StatefulPartitionedCalllstm_26/StatefulPartitionedCall2B
lstm_27/StatefulPartitionedCalllstm_27/StatefulPartitionedCall:S O
+
_output_shapes
:         ]
 
_user_specified_nameinputs
у
═
while_cond_40072986
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_40072986___redundant_placeholder06
2while_while_cond_40072986___redundant_placeholder16
2while_while_cond_40072986___redundant_placeholder26
2while_while_cond_40072986___redundant_placeholder3
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
B: : : : :         Ь:         Ь: ::::: 
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
:         Ь:.*
(
_output_shapes
:         Ь:

_output_shapes
: :

_output_shapes
:
°%
р
!__inference__traced_save_40076630
file_prefix.
*savev2_dense_13_kernel_read_readvariableop,
(savev2_dense_13_bias_read_readvariableop:
6savev2_lstm_26_lstm_cell_26_kernel_read_readvariableopD
@savev2_lstm_26_lstm_cell_26_recurrent_kernel_read_readvariableop8
4savev2_lstm_26_lstm_cell_26_bias_read_readvariableop:
6savev2_lstm_27_lstm_cell_27_kernel_read_readvariableopD
@savev2_lstm_27_lstm_cell_27_recurrent_kernel_read_readvariableop8
4savev2_lstm_27_lstm_cell_27_bias_read_readvariableop$
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
SaveV2/shape_and_slicesИ
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_dense_13_kernel_read_readvariableop(savev2_dense_13_bias_read_readvariableop6savev2_lstm_26_lstm_cell_26_kernel_read_readvariableop@savev2_lstm_26_lstm_cell_26_recurrent_kernel_read_readvariableop4savev2_lstm_26_lstm_cell_26_bias_read_readvariableop6savev2_lstm_27_lstm_cell_27_kernel_read_readvariableop@savev2_lstm_27_lstm_cell_27_recurrent_kernel_read_readvariableop4savev2_lstm_27_lstm_cell_27_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableopsavev2_const"/device:CPU:0*
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
Z: :	Ь::	]И:
ВИ:И:
ВЁ:
ЬЁ:Ё: : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	Ь: 

_output_shapes
::%!

_output_shapes
:	]И:&"
 
_output_shapes
:
ВИ:!

_output_shapes	
:И:&"
 
_output_shapes
:
ВЁ:&"
 
_output_shapes
:
ЬЁ:!

_output_shapes	
:Ё:	
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
╘

э
lstm_26_while_cond_40074668,
(lstm_26_while_lstm_26_while_loop_counter2
.lstm_26_while_lstm_26_while_maximum_iterations
lstm_26_while_placeholder
lstm_26_while_placeholder_1
lstm_26_while_placeholder_2
lstm_26_while_placeholder_3.
*lstm_26_while_less_lstm_26_strided_slice_1F
Blstm_26_while_lstm_26_while_cond_40074668___redundant_placeholder0F
Blstm_26_while_lstm_26_while_cond_40074668___redundant_placeholder1F
Blstm_26_while_lstm_26_while_cond_40074668___redundant_placeholder2F
Blstm_26_while_lstm_26_while_cond_40074668___redundant_placeholder3
lstm_26_while_identity
Ш
lstm_26/while/LessLesslstm_26_while_placeholder*lstm_26_while_less_lstm_26_strided_slice_1*
T0*
_output_shapes
: 2
lstm_26/while/Lessu
lstm_26/while/IdentityIdentitylstm_26/while/Less:z:0*
T0
*
_output_shapes
: 2
lstm_26/while/Identity"9
lstm_26_while_identitylstm_26/while/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :         В:         В: ::::: 
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
:         В:.*
(
_output_shapes
:         В:

_output_shapes
: :

_output_shapes
:
у
═
while_cond_40075353
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_40075353___redundant_placeholder06
2while_while_cond_40075353___redundant_placeholder16
2while_while_cond_40075353___redundant_placeholder26
2while_while_cond_40075353___redundant_placeholder3
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
B: : : : :         В:         В: ::::: 
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
:         В:.*
(
_output_shapes
:         В:

_output_shapes
: :

_output_shapes
:
░?
╘
while_body_40075203
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_26_matmul_readvariableop_resource_0:	]ИI
5while_lstm_cell_26_matmul_1_readvariableop_resource_0:
ВИC
4while_lstm_cell_26_biasadd_readvariableop_resource_0:	И
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_26_matmul_readvariableop_resource:	]ИG
3while_lstm_cell_26_matmul_1_readvariableop_resource:
ВИA
2while_lstm_cell_26_biasadd_readvariableop_resource:	ИИв)while/lstm_cell_26/BiasAdd/ReadVariableOpв(while/lstm_cell_26/MatMul/ReadVariableOpв*while/lstm_cell_26/MatMul_1/ReadVariableOp├
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
(while/lstm_cell_26/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_26_matmul_readvariableop_resource_0*
_output_shapes
:	]И*
dtype02*
(while/lstm_cell_26/MatMul/ReadVariableOp╫
while/lstm_cell_26/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_26/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         И2
while/lstm_cell_26/MatMul╨
*while/lstm_cell_26/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_26_matmul_1_readvariableop_resource_0* 
_output_shapes
:
ВИ*
dtype02,
*while/lstm_cell_26/MatMul_1/ReadVariableOp└
while/lstm_cell_26/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_26/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         И2
while/lstm_cell_26/MatMul_1╕
while/lstm_cell_26/addAddV2#while/lstm_cell_26/MatMul:product:0%while/lstm_cell_26/MatMul_1:product:0*
T0*(
_output_shapes
:         И2
while/lstm_cell_26/add╚
)while/lstm_cell_26/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_26_biasadd_readvariableop_resource_0*
_output_shapes	
:И*
dtype02+
)while/lstm_cell_26/BiasAdd/ReadVariableOp┼
while/lstm_cell_26/BiasAddBiasAddwhile/lstm_cell_26/add:z:01while/lstm_cell_26/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         И2
while/lstm_cell_26/BiasAddК
"while/lstm_cell_26/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_26/split/split_dimП
while/lstm_cell_26/splitSplit+while/lstm_cell_26/split/split_dim:output:0#while/lstm_cell_26/BiasAdd:output:0*
T0*d
_output_shapesR
P:         В:         В:         В:         В*
	num_split2
while/lstm_cell_26/splitЩ
while/lstm_cell_26/SigmoidSigmoid!while/lstm_cell_26/split:output:0*
T0*(
_output_shapes
:         В2
while/lstm_cell_26/SigmoidЭ
while/lstm_cell_26/Sigmoid_1Sigmoid!while/lstm_cell_26/split:output:1*
T0*(
_output_shapes
:         В2
while/lstm_cell_26/Sigmoid_1б
while/lstm_cell_26/mulMul while/lstm_cell_26/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:         В2
while/lstm_cell_26/mulР
while/lstm_cell_26/ReluRelu!while/lstm_cell_26/split:output:2*
T0*(
_output_shapes
:         В2
while/lstm_cell_26/Relu╡
while/lstm_cell_26/mul_1Mulwhile/lstm_cell_26/Sigmoid:y:0%while/lstm_cell_26/Relu:activations:0*
T0*(
_output_shapes
:         В2
while/lstm_cell_26/mul_1к
while/lstm_cell_26/add_1AddV2while/lstm_cell_26/mul:z:0while/lstm_cell_26/mul_1:z:0*
T0*(
_output_shapes
:         В2
while/lstm_cell_26/add_1Э
while/lstm_cell_26/Sigmoid_2Sigmoid!while/lstm_cell_26/split:output:3*
T0*(
_output_shapes
:         В2
while/lstm_cell_26/Sigmoid_2П
while/lstm_cell_26/Relu_1Reluwhile/lstm_cell_26/add_1:z:0*
T0*(
_output_shapes
:         В2
while/lstm_cell_26/Relu_1╣
while/lstm_cell_26/mul_2Mul while/lstm_cell_26/Sigmoid_2:y:0'while/lstm_cell_26/Relu_1:activations:0*
T0*(
_output_shapes
:         В2
while/lstm_cell_26/mul_2р
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_26/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_26/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:         В2
while/Identity_4О
while/Identity_5Identitywhile/lstm_cell_26/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:         В2
while/Identity_5▐

while/NoOpNoOp*^while/lstm_cell_26/BiasAdd/ReadVariableOp)^while/lstm_cell_26/MatMul/ReadVariableOp+^while/lstm_cell_26/MatMul_1/ReadVariableOp*"
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
2while_lstm_cell_26_biasadd_readvariableop_resource4while_lstm_cell_26_biasadd_readvariableop_resource_0"l
3while_lstm_cell_26_matmul_1_readvariableop_resource5while_lstm_cell_26_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_26_matmul_readvariableop_resource3while_lstm_cell_26_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :         В:         В: : : : : 2V
)while/lstm_cell_26/BiasAdd/ReadVariableOp)while/lstm_cell_26/BiasAdd/ReadVariableOp2T
(while/lstm_cell_26/MatMul/ReadVariableOp(while/lstm_cell_26/MatMul/ReadVariableOp2X
*while/lstm_cell_26/MatMul_1/ReadVariableOp*while/lstm_cell_26/MatMul_1/ReadVariableOp: 
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
:         В:.*
(
_output_shapes
:         В:

_output_shapes
: :

_output_shapes
: 
ш^
Ш
)sequential_13_lstm_27_while_body_40071946H
Dsequential_13_lstm_27_while_sequential_13_lstm_27_while_loop_counterN
Jsequential_13_lstm_27_while_sequential_13_lstm_27_while_maximum_iterations+
'sequential_13_lstm_27_while_placeholder-
)sequential_13_lstm_27_while_placeholder_1-
)sequential_13_lstm_27_while_placeholder_2-
)sequential_13_lstm_27_while_placeholder_3G
Csequential_13_lstm_27_while_sequential_13_lstm_27_strided_slice_1_0Г
sequential_13_lstm_27_while_tensorarrayv2read_tensorlistgetitem_sequential_13_lstm_27_tensorarrayunstack_tensorlistfromtensor_0]
Isequential_13_lstm_27_while_lstm_cell_27_matmul_readvariableop_resource_0:
ВЁ_
Ksequential_13_lstm_27_while_lstm_cell_27_matmul_1_readvariableop_resource_0:
ЬЁY
Jsequential_13_lstm_27_while_lstm_cell_27_biasadd_readvariableop_resource_0:	Ё(
$sequential_13_lstm_27_while_identity*
&sequential_13_lstm_27_while_identity_1*
&sequential_13_lstm_27_while_identity_2*
&sequential_13_lstm_27_while_identity_3*
&sequential_13_lstm_27_while_identity_4*
&sequential_13_lstm_27_while_identity_5E
Asequential_13_lstm_27_while_sequential_13_lstm_27_strided_slice_1Б
}sequential_13_lstm_27_while_tensorarrayv2read_tensorlistgetitem_sequential_13_lstm_27_tensorarrayunstack_tensorlistfromtensor[
Gsequential_13_lstm_27_while_lstm_cell_27_matmul_readvariableop_resource:
ВЁ]
Isequential_13_lstm_27_while_lstm_cell_27_matmul_1_readvariableop_resource:
ЬЁW
Hsequential_13_lstm_27_while_lstm_cell_27_biasadd_readvariableop_resource:	ЁИв?sequential_13/lstm_27/while/lstm_cell_27/BiasAdd/ReadVariableOpв>sequential_13/lstm_27/while/lstm_cell_27/MatMul/ReadVariableOpв@sequential_13/lstm_27/while/lstm_cell_27/MatMul_1/ReadVariableOpя
Msequential_13/lstm_27/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    В   2O
Msequential_13/lstm_27/while/TensorArrayV2Read/TensorListGetItem/element_shape╪
?sequential_13/lstm_27/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemsequential_13_lstm_27_while_tensorarrayv2read_tensorlistgetitem_sequential_13_lstm_27_tensorarrayunstack_tensorlistfromtensor_0'sequential_13_lstm_27_while_placeholderVsequential_13/lstm_27/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:         В*
element_dtype02A
?sequential_13/lstm_27/while/TensorArrayV2Read/TensorListGetItemМ
>sequential_13/lstm_27/while/lstm_cell_27/MatMul/ReadVariableOpReadVariableOpIsequential_13_lstm_27_while_lstm_cell_27_matmul_readvariableop_resource_0* 
_output_shapes
:
ВЁ*
dtype02@
>sequential_13/lstm_27/while/lstm_cell_27/MatMul/ReadVariableOpп
/sequential_13/lstm_27/while/lstm_cell_27/MatMulMatMulFsequential_13/lstm_27/while/TensorArrayV2Read/TensorListGetItem:item:0Fsequential_13/lstm_27/while/lstm_cell_27/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ё21
/sequential_13/lstm_27/while/lstm_cell_27/MatMulТ
@sequential_13/lstm_27/while/lstm_cell_27/MatMul_1/ReadVariableOpReadVariableOpKsequential_13_lstm_27_while_lstm_cell_27_matmul_1_readvariableop_resource_0* 
_output_shapes
:
ЬЁ*
dtype02B
@sequential_13/lstm_27/while/lstm_cell_27/MatMul_1/ReadVariableOpШ
1sequential_13/lstm_27/while/lstm_cell_27/MatMul_1MatMul)sequential_13_lstm_27_while_placeholder_2Hsequential_13/lstm_27/while/lstm_cell_27/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ё23
1sequential_13/lstm_27/while/lstm_cell_27/MatMul_1Р
,sequential_13/lstm_27/while/lstm_cell_27/addAddV29sequential_13/lstm_27/while/lstm_cell_27/MatMul:product:0;sequential_13/lstm_27/while/lstm_cell_27/MatMul_1:product:0*
T0*(
_output_shapes
:         Ё2.
,sequential_13/lstm_27/while/lstm_cell_27/addК
?sequential_13/lstm_27/while/lstm_cell_27/BiasAdd/ReadVariableOpReadVariableOpJsequential_13_lstm_27_while_lstm_cell_27_biasadd_readvariableop_resource_0*
_output_shapes	
:Ё*
dtype02A
?sequential_13/lstm_27/while/lstm_cell_27/BiasAdd/ReadVariableOpЭ
0sequential_13/lstm_27/while/lstm_cell_27/BiasAddBiasAdd0sequential_13/lstm_27/while/lstm_cell_27/add:z:0Gsequential_13/lstm_27/while/lstm_cell_27/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ё22
0sequential_13/lstm_27/while/lstm_cell_27/BiasAdd╢
8sequential_13/lstm_27/while/lstm_cell_27/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2:
8sequential_13/lstm_27/while/lstm_cell_27/split/split_dimч
.sequential_13/lstm_27/while/lstm_cell_27/splitSplitAsequential_13/lstm_27/while/lstm_cell_27/split/split_dim:output:09sequential_13/lstm_27/while/lstm_cell_27/BiasAdd:output:0*
T0*d
_output_shapesR
P:         Ь:         Ь:         Ь:         Ь*
	num_split20
.sequential_13/lstm_27/while/lstm_cell_27/split█
0sequential_13/lstm_27/while/lstm_cell_27/SigmoidSigmoid7sequential_13/lstm_27/while/lstm_cell_27/split:output:0*
T0*(
_output_shapes
:         Ь22
0sequential_13/lstm_27/while/lstm_cell_27/Sigmoid▀
2sequential_13/lstm_27/while/lstm_cell_27/Sigmoid_1Sigmoid7sequential_13/lstm_27/while/lstm_cell_27/split:output:1*
T0*(
_output_shapes
:         Ь24
2sequential_13/lstm_27/while/lstm_cell_27/Sigmoid_1∙
,sequential_13/lstm_27/while/lstm_cell_27/mulMul6sequential_13/lstm_27/while/lstm_cell_27/Sigmoid_1:y:0)sequential_13_lstm_27_while_placeholder_3*
T0*(
_output_shapes
:         Ь2.
,sequential_13/lstm_27/while/lstm_cell_27/mul╥
-sequential_13/lstm_27/while/lstm_cell_27/ReluRelu7sequential_13/lstm_27/while/lstm_cell_27/split:output:2*
T0*(
_output_shapes
:         Ь2/
-sequential_13/lstm_27/while/lstm_cell_27/ReluН
.sequential_13/lstm_27/while/lstm_cell_27/mul_1Mul4sequential_13/lstm_27/while/lstm_cell_27/Sigmoid:y:0;sequential_13/lstm_27/while/lstm_cell_27/Relu:activations:0*
T0*(
_output_shapes
:         Ь20
.sequential_13/lstm_27/while/lstm_cell_27/mul_1В
.sequential_13/lstm_27/while/lstm_cell_27/add_1AddV20sequential_13/lstm_27/while/lstm_cell_27/mul:z:02sequential_13/lstm_27/while/lstm_cell_27/mul_1:z:0*
T0*(
_output_shapes
:         Ь20
.sequential_13/lstm_27/while/lstm_cell_27/add_1▀
2sequential_13/lstm_27/while/lstm_cell_27/Sigmoid_2Sigmoid7sequential_13/lstm_27/while/lstm_cell_27/split:output:3*
T0*(
_output_shapes
:         Ь24
2sequential_13/lstm_27/while/lstm_cell_27/Sigmoid_2╤
/sequential_13/lstm_27/while/lstm_cell_27/Relu_1Relu2sequential_13/lstm_27/while/lstm_cell_27/add_1:z:0*
T0*(
_output_shapes
:         Ь21
/sequential_13/lstm_27/while/lstm_cell_27/Relu_1С
.sequential_13/lstm_27/while/lstm_cell_27/mul_2Mul6sequential_13/lstm_27/while/lstm_cell_27/Sigmoid_2:y:0=sequential_13/lstm_27/while/lstm_cell_27/Relu_1:activations:0*
T0*(
_output_shapes
:         Ь20
.sequential_13/lstm_27/while/lstm_cell_27/mul_2╬
@sequential_13/lstm_27/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem)sequential_13_lstm_27_while_placeholder_1'sequential_13_lstm_27_while_placeholder2sequential_13/lstm_27/while/lstm_cell_27/mul_2:z:0*
_output_shapes
: *
element_dtype02B
@sequential_13/lstm_27/while/TensorArrayV2Write/TensorListSetItemИ
!sequential_13/lstm_27/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2#
!sequential_13/lstm_27/while/add/y┴
sequential_13/lstm_27/while/addAddV2'sequential_13_lstm_27_while_placeholder*sequential_13/lstm_27/while/add/y:output:0*
T0*
_output_shapes
: 2!
sequential_13/lstm_27/while/addМ
#sequential_13/lstm_27/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2%
#sequential_13/lstm_27/while/add_1/yф
!sequential_13/lstm_27/while/add_1AddV2Dsequential_13_lstm_27_while_sequential_13_lstm_27_while_loop_counter,sequential_13/lstm_27/while/add_1/y:output:0*
T0*
_output_shapes
: 2#
!sequential_13/lstm_27/while/add_1├
$sequential_13/lstm_27/while/IdentityIdentity%sequential_13/lstm_27/while/add_1:z:0!^sequential_13/lstm_27/while/NoOp*
T0*
_output_shapes
: 2&
$sequential_13/lstm_27/while/Identityь
&sequential_13/lstm_27/while/Identity_1IdentityJsequential_13_lstm_27_while_sequential_13_lstm_27_while_maximum_iterations!^sequential_13/lstm_27/while/NoOp*
T0*
_output_shapes
: 2(
&sequential_13/lstm_27/while/Identity_1┼
&sequential_13/lstm_27/while/Identity_2Identity#sequential_13/lstm_27/while/add:z:0!^sequential_13/lstm_27/while/NoOp*
T0*
_output_shapes
: 2(
&sequential_13/lstm_27/while/Identity_2Є
&sequential_13/lstm_27/while/Identity_3IdentityPsequential_13/lstm_27/while/TensorArrayV2Write/TensorListSetItem:output_handle:0!^sequential_13/lstm_27/while/NoOp*
T0*
_output_shapes
: 2(
&sequential_13/lstm_27/while/Identity_3ц
&sequential_13/lstm_27/while/Identity_4Identity2sequential_13/lstm_27/while/lstm_cell_27/mul_2:z:0!^sequential_13/lstm_27/while/NoOp*
T0*(
_output_shapes
:         Ь2(
&sequential_13/lstm_27/while/Identity_4ц
&sequential_13/lstm_27/while/Identity_5Identity2sequential_13/lstm_27/while/lstm_cell_27/add_1:z:0!^sequential_13/lstm_27/while/NoOp*
T0*(
_output_shapes
:         Ь2(
&sequential_13/lstm_27/while/Identity_5╠
 sequential_13/lstm_27/while/NoOpNoOp@^sequential_13/lstm_27/while/lstm_cell_27/BiasAdd/ReadVariableOp?^sequential_13/lstm_27/while/lstm_cell_27/MatMul/ReadVariableOpA^sequential_13/lstm_27/while/lstm_cell_27/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2"
 sequential_13/lstm_27/while/NoOp"U
$sequential_13_lstm_27_while_identity-sequential_13/lstm_27/while/Identity:output:0"Y
&sequential_13_lstm_27_while_identity_1/sequential_13/lstm_27/while/Identity_1:output:0"Y
&sequential_13_lstm_27_while_identity_2/sequential_13/lstm_27/while/Identity_2:output:0"Y
&sequential_13_lstm_27_while_identity_3/sequential_13/lstm_27/while/Identity_3:output:0"Y
&sequential_13_lstm_27_while_identity_4/sequential_13/lstm_27/while/Identity_4:output:0"Y
&sequential_13_lstm_27_while_identity_5/sequential_13/lstm_27/while/Identity_5:output:0"Ц
Hsequential_13_lstm_27_while_lstm_cell_27_biasadd_readvariableop_resourceJsequential_13_lstm_27_while_lstm_cell_27_biasadd_readvariableop_resource_0"Ш
Isequential_13_lstm_27_while_lstm_cell_27_matmul_1_readvariableop_resourceKsequential_13_lstm_27_while_lstm_cell_27_matmul_1_readvariableop_resource_0"Ф
Gsequential_13_lstm_27_while_lstm_cell_27_matmul_readvariableop_resourceIsequential_13_lstm_27_while_lstm_cell_27_matmul_readvariableop_resource_0"И
Asequential_13_lstm_27_while_sequential_13_lstm_27_strided_slice_1Csequential_13_lstm_27_while_sequential_13_lstm_27_strided_slice_1_0"А
}sequential_13_lstm_27_while_tensorarrayv2read_tensorlistgetitem_sequential_13_lstm_27_tensorarrayunstack_tensorlistfromtensorsequential_13_lstm_27_while_tensorarrayv2read_tensorlistgetitem_sequential_13_lstm_27_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :         Ь:         Ь: : : : : 2В
?sequential_13/lstm_27/while/lstm_cell_27/BiasAdd/ReadVariableOp?sequential_13/lstm_27/while/lstm_cell_27/BiasAdd/ReadVariableOp2А
>sequential_13/lstm_27/while/lstm_cell_27/MatMul/ReadVariableOp>sequential_13/lstm_27/while/lstm_cell_27/MatMul/ReadVariableOp2Д
@sequential_13/lstm_27/while/lstm_cell_27/MatMul_1/ReadVariableOp@sequential_13/lstm_27/while/lstm_cell_27/MatMul_1/ReadVariableOp: 
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
:         Ь:.*
(
_output_shapes
:         Ь:

_output_shapes
: :

_output_shapes
: 
┤?
╓
while_body_40076029
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
3while_lstm_cell_27_matmul_readvariableop_resource_0:
ВЁI
5while_lstm_cell_27_matmul_1_readvariableop_resource_0:
ЬЁC
4while_lstm_cell_27_biasadd_readvariableop_resource_0:	Ё
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
1while_lstm_cell_27_matmul_readvariableop_resource:
ВЁG
3while_lstm_cell_27_matmul_1_readvariableop_resource:
ЬЁA
2while_lstm_cell_27_biasadd_readvariableop_resource:	ЁИв)while/lstm_cell_27/BiasAdd/ReadVariableOpв(while/lstm_cell_27/MatMul/ReadVariableOpв*while/lstm_cell_27/MatMul_1/ReadVariableOp├
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    В   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape╘
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:         В*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem╩
(while/lstm_cell_27/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_27_matmul_readvariableop_resource_0* 
_output_shapes
:
ВЁ*
dtype02*
(while/lstm_cell_27/MatMul/ReadVariableOp╫
while/lstm_cell_27/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_27/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ё2
while/lstm_cell_27/MatMul╨
*while/lstm_cell_27/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_27_matmul_1_readvariableop_resource_0* 
_output_shapes
:
ЬЁ*
dtype02,
*while/lstm_cell_27/MatMul_1/ReadVariableOp└
while/lstm_cell_27/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_27/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ё2
while/lstm_cell_27/MatMul_1╕
while/lstm_cell_27/addAddV2#while/lstm_cell_27/MatMul:product:0%while/lstm_cell_27/MatMul_1:product:0*
T0*(
_output_shapes
:         Ё2
while/lstm_cell_27/add╚
)while/lstm_cell_27/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_27_biasadd_readvariableop_resource_0*
_output_shapes	
:Ё*
dtype02+
)while/lstm_cell_27/BiasAdd/ReadVariableOp┼
while/lstm_cell_27/BiasAddBiasAddwhile/lstm_cell_27/add:z:01while/lstm_cell_27/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ё2
while/lstm_cell_27/BiasAddК
"while/lstm_cell_27/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_27/split/split_dimП
while/lstm_cell_27/splitSplit+while/lstm_cell_27/split/split_dim:output:0#while/lstm_cell_27/BiasAdd:output:0*
T0*d
_output_shapesR
P:         Ь:         Ь:         Ь:         Ь*
	num_split2
while/lstm_cell_27/splitЩ
while/lstm_cell_27/SigmoidSigmoid!while/lstm_cell_27/split:output:0*
T0*(
_output_shapes
:         Ь2
while/lstm_cell_27/SigmoidЭ
while/lstm_cell_27/Sigmoid_1Sigmoid!while/lstm_cell_27/split:output:1*
T0*(
_output_shapes
:         Ь2
while/lstm_cell_27/Sigmoid_1б
while/lstm_cell_27/mulMul while/lstm_cell_27/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:         Ь2
while/lstm_cell_27/mulР
while/lstm_cell_27/ReluRelu!while/lstm_cell_27/split:output:2*
T0*(
_output_shapes
:         Ь2
while/lstm_cell_27/Relu╡
while/lstm_cell_27/mul_1Mulwhile/lstm_cell_27/Sigmoid:y:0%while/lstm_cell_27/Relu:activations:0*
T0*(
_output_shapes
:         Ь2
while/lstm_cell_27/mul_1к
while/lstm_cell_27/add_1AddV2while/lstm_cell_27/mul:z:0while/lstm_cell_27/mul_1:z:0*
T0*(
_output_shapes
:         Ь2
while/lstm_cell_27/add_1Э
while/lstm_cell_27/Sigmoid_2Sigmoid!while/lstm_cell_27/split:output:3*
T0*(
_output_shapes
:         Ь2
while/lstm_cell_27/Sigmoid_2П
while/lstm_cell_27/Relu_1Reluwhile/lstm_cell_27/add_1:z:0*
T0*(
_output_shapes
:         Ь2
while/lstm_cell_27/Relu_1╣
while/lstm_cell_27/mul_2Mul while/lstm_cell_27/Sigmoid_2:y:0'while/lstm_cell_27/Relu_1:activations:0*
T0*(
_output_shapes
:         Ь2
while/lstm_cell_27/mul_2р
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_27/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_27/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:         Ь2
while/Identity_4О
while/Identity_5Identitywhile/lstm_cell_27/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:         Ь2
while/Identity_5▐

while/NoOpNoOp*^while/lstm_cell_27/BiasAdd/ReadVariableOp)^while/lstm_cell_27/MatMul/ReadVariableOp+^while/lstm_cell_27/MatMul_1/ReadVariableOp*"
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
2while_lstm_cell_27_biasadd_readvariableop_resource4while_lstm_cell_27_biasadd_readvariableop_resource_0"l
3while_lstm_cell_27_matmul_1_readvariableop_resource5while_lstm_cell_27_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_27_matmul_readvariableop_resource3while_lstm_cell_27_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :         Ь:         Ь: : : : : 2V
)while/lstm_cell_27/BiasAdd/ReadVariableOp)while/lstm_cell_27/BiasAdd/ReadVariableOp2T
(while/lstm_cell_27/MatMul/ReadVariableOp(while/lstm_cell_27/MatMul/ReadVariableOp2X
*while/lstm_cell_27/MatMul_1/ReadVariableOp*while/lstm_cell_27/MatMul_1/ReadVariableOp: 
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
:         Ь:.*
(
_output_shapes
:         Ь:

_output_shapes
: :

_output_shapes
: 
И
Щ
+__inference_dense_13_layer_call_fn_40076375

inputs
unknown:	Ь
	unknown_0:
identityИвStatefulPartitionedCall·
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
GPU 2J 8В *O
fJRH
F__inference_dense_13_layer_call_and_return_conditional_losses_400736862
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
:         Ь: : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:         Ь
 
_user_specified_nameinputs
о

╘
0__inference_sequential_13_layer_call_fn_40074202
lstm_26_input
unknown:	]И
	unknown_0:
ВИ
	unknown_1:	И
	unknown_2:
ВЁ
	unknown_3:
ЬЁ
	unknown_4:	Ё
	unknown_5:	Ь
	unknown_6:
identityИвStatefulPartitionedCall╘
StatefulPartitionedCallStatefulPartitionedCalllstm_26_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
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
GPU 2J 8В *T
fORM
K__inference_sequential_13_layer_call_and_return_conditional_losses_400741622
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
_user_specified_namelstm_26_input
┤?
╓
while_body_40075878
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
3while_lstm_cell_27_matmul_readvariableop_resource_0:
ВЁI
5while_lstm_cell_27_matmul_1_readvariableop_resource_0:
ЬЁC
4while_lstm_cell_27_biasadd_readvariableop_resource_0:	Ё
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
1while_lstm_cell_27_matmul_readvariableop_resource:
ВЁG
3while_lstm_cell_27_matmul_1_readvariableop_resource:
ЬЁA
2while_lstm_cell_27_biasadd_readvariableop_resource:	ЁИв)while/lstm_cell_27/BiasAdd/ReadVariableOpв(while/lstm_cell_27/MatMul/ReadVariableOpв*while/lstm_cell_27/MatMul_1/ReadVariableOp├
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    В   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape╘
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:         В*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem╩
(while/lstm_cell_27/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_27_matmul_readvariableop_resource_0* 
_output_shapes
:
ВЁ*
dtype02*
(while/lstm_cell_27/MatMul/ReadVariableOp╫
while/lstm_cell_27/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_27/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ё2
while/lstm_cell_27/MatMul╨
*while/lstm_cell_27/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_27_matmul_1_readvariableop_resource_0* 
_output_shapes
:
ЬЁ*
dtype02,
*while/lstm_cell_27/MatMul_1/ReadVariableOp└
while/lstm_cell_27/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_27/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ё2
while/lstm_cell_27/MatMul_1╕
while/lstm_cell_27/addAddV2#while/lstm_cell_27/MatMul:product:0%while/lstm_cell_27/MatMul_1:product:0*
T0*(
_output_shapes
:         Ё2
while/lstm_cell_27/add╚
)while/lstm_cell_27/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_27_biasadd_readvariableop_resource_0*
_output_shapes	
:Ё*
dtype02+
)while/lstm_cell_27/BiasAdd/ReadVariableOp┼
while/lstm_cell_27/BiasAddBiasAddwhile/lstm_cell_27/add:z:01while/lstm_cell_27/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ё2
while/lstm_cell_27/BiasAddК
"while/lstm_cell_27/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_27/split/split_dimП
while/lstm_cell_27/splitSplit+while/lstm_cell_27/split/split_dim:output:0#while/lstm_cell_27/BiasAdd:output:0*
T0*d
_output_shapesR
P:         Ь:         Ь:         Ь:         Ь*
	num_split2
while/lstm_cell_27/splitЩ
while/lstm_cell_27/SigmoidSigmoid!while/lstm_cell_27/split:output:0*
T0*(
_output_shapes
:         Ь2
while/lstm_cell_27/SigmoidЭ
while/lstm_cell_27/Sigmoid_1Sigmoid!while/lstm_cell_27/split:output:1*
T0*(
_output_shapes
:         Ь2
while/lstm_cell_27/Sigmoid_1б
while/lstm_cell_27/mulMul while/lstm_cell_27/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:         Ь2
while/lstm_cell_27/mulР
while/lstm_cell_27/ReluRelu!while/lstm_cell_27/split:output:2*
T0*(
_output_shapes
:         Ь2
while/lstm_cell_27/Relu╡
while/lstm_cell_27/mul_1Mulwhile/lstm_cell_27/Sigmoid:y:0%while/lstm_cell_27/Relu:activations:0*
T0*(
_output_shapes
:         Ь2
while/lstm_cell_27/mul_1к
while/lstm_cell_27/add_1AddV2while/lstm_cell_27/mul:z:0while/lstm_cell_27/mul_1:z:0*
T0*(
_output_shapes
:         Ь2
while/lstm_cell_27/add_1Э
while/lstm_cell_27/Sigmoid_2Sigmoid!while/lstm_cell_27/split:output:3*
T0*(
_output_shapes
:         Ь2
while/lstm_cell_27/Sigmoid_2П
while/lstm_cell_27/Relu_1Reluwhile/lstm_cell_27/add_1:z:0*
T0*(
_output_shapes
:         Ь2
while/lstm_cell_27/Relu_1╣
while/lstm_cell_27/mul_2Mul while/lstm_cell_27/Sigmoid_2:y:0'while/lstm_cell_27/Relu_1:activations:0*
T0*(
_output_shapes
:         Ь2
while/lstm_cell_27/mul_2р
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_27/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_27/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:         Ь2
while/Identity_4О
while/Identity_5Identitywhile/lstm_cell_27/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:         Ь2
while/Identity_5▐

while/NoOpNoOp*^while/lstm_cell_27/BiasAdd/ReadVariableOp)^while/lstm_cell_27/MatMul/ReadVariableOp+^while/lstm_cell_27/MatMul_1/ReadVariableOp*"
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
2while_lstm_cell_27_biasadd_readvariableop_resource4while_lstm_cell_27_biasadd_readvariableop_resource_0"l
3while_lstm_cell_27_matmul_1_readvariableop_resource5while_lstm_cell_27_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_27_matmul_readvariableop_resource3while_lstm_cell_27_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :         Ь:         Ь: : : : : 2V
)while/lstm_cell_27/BiasAdd/ReadVariableOp)while/lstm_cell_27/BiasAdd/ReadVariableOp2T
(while/lstm_cell_27/MatMul/ReadVariableOp(while/lstm_cell_27/MatMul/ReadVariableOp2X
*while/lstm_cell_27/MatMul_1/ReadVariableOp*while/lstm_cell_27/MatMul_1/ReadVariableOp: 
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
:         Ь:.*
(
_output_shapes
:         Ь:

_output_shapes
: :

_output_shapes
: 
Е&
є
while_body_40072147
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_00
while_lstm_cell_26_40072171_0:	]И1
while_lstm_cell_26_40072173_0:
ВИ,
while_lstm_cell_26_40072175_0:	И
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor.
while_lstm_cell_26_40072171:	]И/
while_lstm_cell_26_40072173:
ВИ*
while_lstm_cell_26_40072175:	ИИв*while/lstm_cell_26/StatefulPartitionedCall├
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
*while/lstm_cell_26/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_26_40072171_0while_lstm_cell_26_40072173_0while_lstm_cell_26_40072175_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:         В:         В:         В*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_lstm_cell_26_layer_call_and_return_conditional_losses_400721332,
*while/lstm_cell_26/StatefulPartitionedCallў
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder3while/lstm_cell_26/StatefulPartitionedCall:output:0*
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
while/Identity_4Identity3while/lstm_cell_26/StatefulPartitionedCall:output:1^while/NoOp*
T0*(
_output_shapes
:         В2
while/Identity_4е
while/Identity_5Identity3while/lstm_cell_26/StatefulPartitionedCall:output:2^while/NoOp*
T0*(
_output_shapes
:         В2
while/Identity_5З

while/NoOpNoOp+^while/lstm_cell_26/StatefulPartitionedCall*"
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
while_lstm_cell_26_40072171while_lstm_cell_26_40072171_0"<
while_lstm_cell_26_40072173while_lstm_cell_26_40072173_0"<
while_lstm_cell_26_40072175while_lstm_cell_26_40072175_0"0
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :         В:         В: : : : : 2X
*while/lstm_cell_26/StatefulPartitionedCall*while/lstm_cell_26/StatefulPartitionedCall: 
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
:         В:.*
(
_output_shapes
:         В:

_output_shapes
: :

_output_shapes
: 
╬7
▀
$__inference__traced_restore_40076676
file_prefix3
 assignvariableop_dense_13_kernel:	Ь.
 assignvariableop_1_dense_13_bias:A
.assignvariableop_2_lstm_26_lstm_cell_26_kernel:	]ИL
8assignvariableop_3_lstm_26_lstm_cell_26_recurrent_kernel:
ВИ;
,assignvariableop_4_lstm_26_lstm_cell_26_bias:	ИB
.assignvariableop_5_lstm_27_lstm_cell_27_kernel:
ВЁL
8assignvariableop_6_lstm_27_lstm_cell_27_recurrent_kernel:
ЬЁ;
,assignvariableop_7_lstm_27_lstm_cell_27_bias:	Ё"
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

IdentityЯ
AssignVariableOpAssignVariableOp assignvariableop_dense_13_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1е
AssignVariableOp_1AssignVariableOp assignvariableop_1_dense_13_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2│
AssignVariableOp_2AssignVariableOp.assignvariableop_2_lstm_26_lstm_cell_26_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3╜
AssignVariableOp_3AssignVariableOp8assignvariableop_3_lstm_26_lstm_cell_26_recurrent_kernelIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4▒
AssignVariableOp_4AssignVariableOp,assignvariableop_4_lstm_26_lstm_cell_26_biasIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5│
AssignVariableOp_5AssignVariableOp.assignvariableop_5_lstm_27_lstm_cell_27_kernelIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6╜
AssignVariableOp_6AssignVariableOp8assignvariableop_6_lstm_27_lstm_cell_27_recurrent_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7▒
AssignVariableOp_7AssignVariableOp,assignvariableop_7_lstm_27_lstm_cell_27_biasIdentity_7:output:0"/device:CPU:0*
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
хJ
╘

lstm_26_while_body_40074669,
(lstm_26_while_lstm_26_while_loop_counter2
.lstm_26_while_lstm_26_while_maximum_iterations
lstm_26_while_placeholder
lstm_26_while_placeholder_1
lstm_26_while_placeholder_2
lstm_26_while_placeholder_3+
'lstm_26_while_lstm_26_strided_slice_1_0g
clstm_26_while_tensorarrayv2read_tensorlistgetitem_lstm_26_tensorarrayunstack_tensorlistfromtensor_0N
;lstm_26_while_lstm_cell_26_matmul_readvariableop_resource_0:	]ИQ
=lstm_26_while_lstm_cell_26_matmul_1_readvariableop_resource_0:
ВИK
<lstm_26_while_lstm_cell_26_biasadd_readvariableop_resource_0:	И
lstm_26_while_identity
lstm_26_while_identity_1
lstm_26_while_identity_2
lstm_26_while_identity_3
lstm_26_while_identity_4
lstm_26_while_identity_5)
%lstm_26_while_lstm_26_strided_slice_1e
alstm_26_while_tensorarrayv2read_tensorlistgetitem_lstm_26_tensorarrayunstack_tensorlistfromtensorL
9lstm_26_while_lstm_cell_26_matmul_readvariableop_resource:	]ИO
;lstm_26_while_lstm_cell_26_matmul_1_readvariableop_resource:
ВИI
:lstm_26_while_lstm_cell_26_biasadd_readvariableop_resource:	ИИв1lstm_26/while/lstm_cell_26/BiasAdd/ReadVariableOpв0lstm_26/while/lstm_cell_26/MatMul/ReadVariableOpв2lstm_26/while/lstm_cell_26/MatMul_1/ReadVariableOp╙
?lstm_26/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ]   2A
?lstm_26/while/TensorArrayV2Read/TensorListGetItem/element_shapeГ
1lstm_26/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemclstm_26_while_tensorarrayv2read_tensorlistgetitem_lstm_26_tensorarrayunstack_tensorlistfromtensor_0lstm_26_while_placeholderHlstm_26/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         ]*
element_dtype023
1lstm_26/while/TensorArrayV2Read/TensorListGetItemс
0lstm_26/while/lstm_cell_26/MatMul/ReadVariableOpReadVariableOp;lstm_26_while_lstm_cell_26_matmul_readvariableop_resource_0*
_output_shapes
:	]И*
dtype022
0lstm_26/while/lstm_cell_26/MatMul/ReadVariableOpў
!lstm_26/while/lstm_cell_26/MatMulMatMul8lstm_26/while/TensorArrayV2Read/TensorListGetItem:item:08lstm_26/while/lstm_cell_26/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         И2#
!lstm_26/while/lstm_cell_26/MatMulш
2lstm_26/while/lstm_cell_26/MatMul_1/ReadVariableOpReadVariableOp=lstm_26_while_lstm_cell_26_matmul_1_readvariableop_resource_0* 
_output_shapes
:
ВИ*
dtype024
2lstm_26/while/lstm_cell_26/MatMul_1/ReadVariableOpр
#lstm_26/while/lstm_cell_26/MatMul_1MatMullstm_26_while_placeholder_2:lstm_26/while/lstm_cell_26/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         И2%
#lstm_26/while/lstm_cell_26/MatMul_1╪
lstm_26/while/lstm_cell_26/addAddV2+lstm_26/while/lstm_cell_26/MatMul:product:0-lstm_26/while/lstm_cell_26/MatMul_1:product:0*
T0*(
_output_shapes
:         И2 
lstm_26/while/lstm_cell_26/addр
1lstm_26/while/lstm_cell_26/BiasAdd/ReadVariableOpReadVariableOp<lstm_26_while_lstm_cell_26_biasadd_readvariableop_resource_0*
_output_shapes	
:И*
dtype023
1lstm_26/while/lstm_cell_26/BiasAdd/ReadVariableOpх
"lstm_26/while/lstm_cell_26/BiasAddBiasAdd"lstm_26/while/lstm_cell_26/add:z:09lstm_26/while/lstm_cell_26/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         И2$
"lstm_26/while/lstm_cell_26/BiasAddЪ
*lstm_26/while/lstm_cell_26/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2,
*lstm_26/while/lstm_cell_26/split/split_dimп
 lstm_26/while/lstm_cell_26/splitSplit3lstm_26/while/lstm_cell_26/split/split_dim:output:0+lstm_26/while/lstm_cell_26/BiasAdd:output:0*
T0*d
_output_shapesR
P:         В:         В:         В:         В*
	num_split2"
 lstm_26/while/lstm_cell_26/split▒
"lstm_26/while/lstm_cell_26/SigmoidSigmoid)lstm_26/while/lstm_cell_26/split:output:0*
T0*(
_output_shapes
:         В2$
"lstm_26/while/lstm_cell_26/Sigmoid╡
$lstm_26/while/lstm_cell_26/Sigmoid_1Sigmoid)lstm_26/while/lstm_cell_26/split:output:1*
T0*(
_output_shapes
:         В2&
$lstm_26/while/lstm_cell_26/Sigmoid_1┴
lstm_26/while/lstm_cell_26/mulMul(lstm_26/while/lstm_cell_26/Sigmoid_1:y:0lstm_26_while_placeholder_3*
T0*(
_output_shapes
:         В2 
lstm_26/while/lstm_cell_26/mulи
lstm_26/while/lstm_cell_26/ReluRelu)lstm_26/while/lstm_cell_26/split:output:2*
T0*(
_output_shapes
:         В2!
lstm_26/while/lstm_cell_26/Relu╒
 lstm_26/while/lstm_cell_26/mul_1Mul&lstm_26/while/lstm_cell_26/Sigmoid:y:0-lstm_26/while/lstm_cell_26/Relu:activations:0*
T0*(
_output_shapes
:         В2"
 lstm_26/while/lstm_cell_26/mul_1╩
 lstm_26/while/lstm_cell_26/add_1AddV2"lstm_26/while/lstm_cell_26/mul:z:0$lstm_26/while/lstm_cell_26/mul_1:z:0*
T0*(
_output_shapes
:         В2"
 lstm_26/while/lstm_cell_26/add_1╡
$lstm_26/while/lstm_cell_26/Sigmoid_2Sigmoid)lstm_26/while/lstm_cell_26/split:output:3*
T0*(
_output_shapes
:         В2&
$lstm_26/while/lstm_cell_26/Sigmoid_2з
!lstm_26/while/lstm_cell_26/Relu_1Relu$lstm_26/while/lstm_cell_26/add_1:z:0*
T0*(
_output_shapes
:         В2#
!lstm_26/while/lstm_cell_26/Relu_1┘
 lstm_26/while/lstm_cell_26/mul_2Mul(lstm_26/while/lstm_cell_26/Sigmoid_2:y:0/lstm_26/while/lstm_cell_26/Relu_1:activations:0*
T0*(
_output_shapes
:         В2"
 lstm_26/while/lstm_cell_26/mul_2И
2lstm_26/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_26_while_placeholder_1lstm_26_while_placeholder$lstm_26/while/lstm_cell_26/mul_2:z:0*
_output_shapes
: *
element_dtype024
2lstm_26/while/TensorArrayV2Write/TensorListSetIteml
lstm_26/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_26/while/add/yЙ
lstm_26/while/addAddV2lstm_26_while_placeholderlstm_26/while/add/y:output:0*
T0*
_output_shapes
: 2
lstm_26/while/addp
lstm_26/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_26/while/add_1/yЮ
lstm_26/while/add_1AddV2(lstm_26_while_lstm_26_while_loop_counterlstm_26/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lstm_26/while/add_1Л
lstm_26/while/IdentityIdentitylstm_26/while/add_1:z:0^lstm_26/while/NoOp*
T0*
_output_shapes
: 2
lstm_26/while/Identityж
lstm_26/while/Identity_1Identity.lstm_26_while_lstm_26_while_maximum_iterations^lstm_26/while/NoOp*
T0*
_output_shapes
: 2
lstm_26/while/Identity_1Н
lstm_26/while/Identity_2Identitylstm_26/while/add:z:0^lstm_26/while/NoOp*
T0*
_output_shapes
: 2
lstm_26/while/Identity_2║
lstm_26/while/Identity_3IdentityBlstm_26/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_26/while/NoOp*
T0*
_output_shapes
: 2
lstm_26/while/Identity_3о
lstm_26/while/Identity_4Identity$lstm_26/while/lstm_cell_26/mul_2:z:0^lstm_26/while/NoOp*
T0*(
_output_shapes
:         В2
lstm_26/while/Identity_4о
lstm_26/while/Identity_5Identity$lstm_26/while/lstm_cell_26/add_1:z:0^lstm_26/while/NoOp*
T0*(
_output_shapes
:         В2
lstm_26/while/Identity_5Ж
lstm_26/while/NoOpNoOp2^lstm_26/while/lstm_cell_26/BiasAdd/ReadVariableOp1^lstm_26/while/lstm_cell_26/MatMul/ReadVariableOp3^lstm_26/while/lstm_cell_26/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
lstm_26/while/NoOp"9
lstm_26_while_identitylstm_26/while/Identity:output:0"=
lstm_26_while_identity_1!lstm_26/while/Identity_1:output:0"=
lstm_26_while_identity_2!lstm_26/while/Identity_2:output:0"=
lstm_26_while_identity_3!lstm_26/while/Identity_3:output:0"=
lstm_26_while_identity_4!lstm_26/while/Identity_4:output:0"=
lstm_26_while_identity_5!lstm_26/while/Identity_5:output:0"P
%lstm_26_while_lstm_26_strided_slice_1'lstm_26_while_lstm_26_strided_slice_1_0"z
:lstm_26_while_lstm_cell_26_biasadd_readvariableop_resource<lstm_26_while_lstm_cell_26_biasadd_readvariableop_resource_0"|
;lstm_26_while_lstm_cell_26_matmul_1_readvariableop_resource=lstm_26_while_lstm_cell_26_matmul_1_readvariableop_resource_0"x
9lstm_26_while_lstm_cell_26_matmul_readvariableop_resource;lstm_26_while_lstm_cell_26_matmul_readvariableop_resource_0"╚
alstm_26_while_tensorarrayv2read_tensorlistgetitem_lstm_26_tensorarrayunstack_tensorlistfromtensorclstm_26_while_tensorarrayv2read_tensorlistgetitem_lstm_26_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :         В:         В: : : : : 2f
1lstm_26/while/lstm_cell_26/BiasAdd/ReadVariableOp1lstm_26/while/lstm_cell_26/BiasAdd/ReadVariableOp2d
0lstm_26/while/lstm_cell_26/MatMul/ReadVariableOp0lstm_26/while/lstm_cell_26/MatMul/ReadVariableOp2h
2lstm_26/while/lstm_cell_26/MatMul_1/ReadVariableOp2lstm_26/while/lstm_cell_26/MatMul_1/ReadVariableOp: 
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
:         В:.*
(
_output_shapes
:         В:

_output_shapes
: :

_output_shapes
: 
у
═
while_cond_40075202
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_40075202___redundant_placeholder06
2while_while_cond_40075202___redundant_placeholder16
2while_while_cond_40075202___redundant_placeholder26
2while_while_cond_40075202___redundant_placeholder3
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
B: : : : :         В:         В: ::::: 
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
:         В:.*
(
_output_shapes
:         В:

_output_shapes
: :

_output_shapes
:
К\
Я
E__inference_lstm_27_layer_call_and_return_conditional_losses_40076264

inputs?
+lstm_cell_27_matmul_readvariableop_resource:
ВЁA
-lstm_cell_27_matmul_1_readvariableop_resource:
ЬЁ;
,lstm_cell_27_biasadd_readvariableop_resource:	Ё
identityИв#lstm_cell_27/BiasAdd/ReadVariableOpв"lstm_cell_27/MatMul/ReadVariableOpв$lstm_cell_27/MatMul_1/ReadVariableOpвwhileD
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
B :Ь2
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
B :Ь2
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
:         Ь2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :Ь2
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
B :Ь2
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
:         Ь2	
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
:         В2
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
valueB"    В   27
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
:         В*
shrink_axis_mask2
strided_slice_2╢
"lstm_cell_27/MatMul/ReadVariableOpReadVariableOp+lstm_cell_27_matmul_readvariableop_resource* 
_output_shapes
:
ВЁ*
dtype02$
"lstm_cell_27/MatMul/ReadVariableOpн
lstm_cell_27/MatMulMatMulstrided_slice_2:output:0*lstm_cell_27/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ё2
lstm_cell_27/MatMul╝
$lstm_cell_27/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_27_matmul_1_readvariableop_resource* 
_output_shapes
:
ЬЁ*
dtype02&
$lstm_cell_27/MatMul_1/ReadVariableOpй
lstm_cell_27/MatMul_1MatMulzeros:output:0,lstm_cell_27/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ё2
lstm_cell_27/MatMul_1а
lstm_cell_27/addAddV2lstm_cell_27/MatMul:product:0lstm_cell_27/MatMul_1:product:0*
T0*(
_output_shapes
:         Ё2
lstm_cell_27/add┤
#lstm_cell_27/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_27_biasadd_readvariableop_resource*
_output_shapes	
:Ё*
dtype02%
#lstm_cell_27/BiasAdd/ReadVariableOpн
lstm_cell_27/BiasAddBiasAddlstm_cell_27/add:z:0+lstm_cell_27/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ё2
lstm_cell_27/BiasAdd~
lstm_cell_27/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_27/split/split_dimў
lstm_cell_27/splitSplit%lstm_cell_27/split/split_dim:output:0lstm_cell_27/BiasAdd:output:0*
T0*d
_output_shapesR
P:         Ь:         Ь:         Ь:         Ь*
	num_split2
lstm_cell_27/splitЗ
lstm_cell_27/SigmoidSigmoidlstm_cell_27/split:output:0*
T0*(
_output_shapes
:         Ь2
lstm_cell_27/SigmoidЛ
lstm_cell_27/Sigmoid_1Sigmoidlstm_cell_27/split:output:1*
T0*(
_output_shapes
:         Ь2
lstm_cell_27/Sigmoid_1М
lstm_cell_27/mulMullstm_cell_27/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:         Ь2
lstm_cell_27/mul~
lstm_cell_27/ReluRelulstm_cell_27/split:output:2*
T0*(
_output_shapes
:         Ь2
lstm_cell_27/ReluЭ
lstm_cell_27/mul_1Mullstm_cell_27/Sigmoid:y:0lstm_cell_27/Relu:activations:0*
T0*(
_output_shapes
:         Ь2
lstm_cell_27/mul_1Т
lstm_cell_27/add_1AddV2lstm_cell_27/mul:z:0lstm_cell_27/mul_1:z:0*
T0*(
_output_shapes
:         Ь2
lstm_cell_27/add_1Л
lstm_cell_27/Sigmoid_2Sigmoidlstm_cell_27/split:output:3*
T0*(
_output_shapes
:         Ь2
lstm_cell_27/Sigmoid_2}
lstm_cell_27/Relu_1Relulstm_cell_27/add_1:z:0*
T0*(
_output_shapes
:         Ь2
lstm_cell_27/Relu_1б
lstm_cell_27/mul_2Mullstm_cell_27/Sigmoid_2:y:0!lstm_cell_27/Relu_1:activations:0*
T0*(
_output_shapes
:         Ь2
lstm_cell_27/mul_2П
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_27_matmul_readvariableop_resource-lstm_cell_27_matmul_1_readvariableop_resource,lstm_cell_27_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :         Ь:         Ь: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_40076180*
condR
while_cond_40076179*M
output_shapes<
:: : : : :         Ь:         Ь: : : : : *
parallel_iterations 2
while╡
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      22
0TensorArrayV2Stack/TensorListStack/element_shapeщ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:         Ь*
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
:         Ь*
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
:         Ь2
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
:         Ь2

Identity╚
NoOpNoOp$^lstm_cell_27/BiasAdd/ReadVariableOp#^lstm_cell_27/MatMul/ReadVariableOp%^lstm_cell_27/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:         В: : : 2J
#lstm_cell_27/BiasAdd/ReadVariableOp#lstm_cell_27/BiasAdd/ReadVariableOp2H
"lstm_cell_27/MatMul/ReadVariableOp"lstm_cell_27/MatMul/ReadVariableOp2L
$lstm_cell_27/MatMul_1/ReadVariableOp$lstm_cell_27/MatMul_1/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:         В
 
_user_specified_nameinputs
Д\
Ю
E__inference_lstm_26_layer_call_and_return_conditional_losses_40073475

inputs>
+lstm_cell_26_matmul_readvariableop_resource:	]ИA
-lstm_cell_26_matmul_1_readvariableop_resource:
ВИ;
,lstm_cell_26_biasadd_readvariableop_resource:	И
identityИв#lstm_cell_26/BiasAdd/ReadVariableOpв"lstm_cell_26/MatMul/ReadVariableOpв$lstm_cell_26/MatMul_1/ReadVariableOpвwhileD
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
B :В2
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
B :В2
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
:         В2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :В2
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
B :В2
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
:         В2	
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
"lstm_cell_26/MatMul/ReadVariableOpReadVariableOp+lstm_cell_26_matmul_readvariableop_resource*
_output_shapes
:	]И*
dtype02$
"lstm_cell_26/MatMul/ReadVariableOpн
lstm_cell_26/MatMulMatMulstrided_slice_2:output:0*lstm_cell_26/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         И2
lstm_cell_26/MatMul╝
$lstm_cell_26/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_26_matmul_1_readvariableop_resource* 
_output_shapes
:
ВИ*
dtype02&
$lstm_cell_26/MatMul_1/ReadVariableOpй
lstm_cell_26/MatMul_1MatMulzeros:output:0,lstm_cell_26/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         И2
lstm_cell_26/MatMul_1а
lstm_cell_26/addAddV2lstm_cell_26/MatMul:product:0lstm_cell_26/MatMul_1:product:0*
T0*(
_output_shapes
:         И2
lstm_cell_26/add┤
#lstm_cell_26/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_26_biasadd_readvariableop_resource*
_output_shapes	
:И*
dtype02%
#lstm_cell_26/BiasAdd/ReadVariableOpн
lstm_cell_26/BiasAddBiasAddlstm_cell_26/add:z:0+lstm_cell_26/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         И2
lstm_cell_26/BiasAdd~
lstm_cell_26/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_26/split/split_dimў
lstm_cell_26/splitSplit%lstm_cell_26/split/split_dim:output:0lstm_cell_26/BiasAdd:output:0*
T0*d
_output_shapesR
P:         В:         В:         В:         В*
	num_split2
lstm_cell_26/splitЗ
lstm_cell_26/SigmoidSigmoidlstm_cell_26/split:output:0*
T0*(
_output_shapes
:         В2
lstm_cell_26/SigmoidЛ
lstm_cell_26/Sigmoid_1Sigmoidlstm_cell_26/split:output:1*
T0*(
_output_shapes
:         В2
lstm_cell_26/Sigmoid_1М
lstm_cell_26/mulMullstm_cell_26/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:         В2
lstm_cell_26/mul~
lstm_cell_26/ReluRelulstm_cell_26/split:output:2*
T0*(
_output_shapes
:         В2
lstm_cell_26/ReluЭ
lstm_cell_26/mul_1Mullstm_cell_26/Sigmoid:y:0lstm_cell_26/Relu:activations:0*
T0*(
_output_shapes
:         В2
lstm_cell_26/mul_1Т
lstm_cell_26/add_1AddV2lstm_cell_26/mul:z:0lstm_cell_26/mul_1:z:0*
T0*(
_output_shapes
:         В2
lstm_cell_26/add_1Л
lstm_cell_26/Sigmoid_2Sigmoidlstm_cell_26/split:output:3*
T0*(
_output_shapes
:         В2
lstm_cell_26/Sigmoid_2}
lstm_cell_26/Relu_1Relulstm_cell_26/add_1:z:0*
T0*(
_output_shapes
:         В2
lstm_cell_26/Relu_1б
lstm_cell_26/mul_2Mullstm_cell_26/Sigmoid_2:y:0!lstm_cell_26/Relu_1:activations:0*
T0*(
_output_shapes
:         В2
lstm_cell_26/mul_2П
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    В   2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_26_matmul_readvariableop_resource-lstm_cell_26_matmul_1_readvariableop_resource,lstm_cell_26_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :         В:         В: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_40073391*
condR
while_cond_40073390*M
output_shapes<
:: : : : :         В:         В: : : : : *
parallel_iterations 2
while╡
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    В   22
0TensorArrayV2Stack/TensorListStack/element_shapeщ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:         В*
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
:         В*
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
:         В2
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
:         В2

Identity╚
NoOpNoOp$^lstm_cell_26/BiasAdd/ReadVariableOp#^lstm_cell_26/MatMul/ReadVariableOp%^lstm_cell_26/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         ]: : : 2J
#lstm_cell_26/BiasAdd/ReadVariableOp#lstm_cell_26/BiasAdd/ReadVariableOp2H
"lstm_cell_26/MatMul/ReadVariableOp"lstm_cell_26/MatMul/ReadVariableOp2L
$lstm_cell_26/MatMul_1/ReadVariableOp$lstm_cell_26/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:         ]
 
_user_specified_nameinputs
╔\
б
E__inference_lstm_27_layer_call_and_return_conditional_losses_40075962
inputs_0?
+lstm_cell_27_matmul_readvariableop_resource:
ВЁA
-lstm_cell_27_matmul_1_readvariableop_resource:
ЬЁ;
,lstm_cell_27_biasadd_readvariableop_resource:	Ё
identityИв#lstm_cell_27/BiasAdd/ReadVariableOpв"lstm_cell_27/MatMul/ReadVariableOpв$lstm_cell_27/MatMul_1/ReadVariableOpвwhileF
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
B :Ь2
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
B :Ь2
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
:         Ь2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :Ь2
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
B :Ь2
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
:         Ь2	
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
!:                  В2
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
valueB"    В   27
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
:         В*
shrink_axis_mask2
strided_slice_2╢
"lstm_cell_27/MatMul/ReadVariableOpReadVariableOp+lstm_cell_27_matmul_readvariableop_resource* 
_output_shapes
:
ВЁ*
dtype02$
"lstm_cell_27/MatMul/ReadVariableOpн
lstm_cell_27/MatMulMatMulstrided_slice_2:output:0*lstm_cell_27/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ё2
lstm_cell_27/MatMul╝
$lstm_cell_27/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_27_matmul_1_readvariableop_resource* 
_output_shapes
:
ЬЁ*
dtype02&
$lstm_cell_27/MatMul_1/ReadVariableOpй
lstm_cell_27/MatMul_1MatMulzeros:output:0,lstm_cell_27/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ё2
lstm_cell_27/MatMul_1а
lstm_cell_27/addAddV2lstm_cell_27/MatMul:product:0lstm_cell_27/MatMul_1:product:0*
T0*(
_output_shapes
:         Ё2
lstm_cell_27/add┤
#lstm_cell_27/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_27_biasadd_readvariableop_resource*
_output_shapes	
:Ё*
dtype02%
#lstm_cell_27/BiasAdd/ReadVariableOpн
lstm_cell_27/BiasAddBiasAddlstm_cell_27/add:z:0+lstm_cell_27/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ё2
lstm_cell_27/BiasAdd~
lstm_cell_27/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_27/split/split_dimў
lstm_cell_27/splitSplit%lstm_cell_27/split/split_dim:output:0lstm_cell_27/BiasAdd:output:0*
T0*d
_output_shapesR
P:         Ь:         Ь:         Ь:         Ь*
	num_split2
lstm_cell_27/splitЗ
lstm_cell_27/SigmoidSigmoidlstm_cell_27/split:output:0*
T0*(
_output_shapes
:         Ь2
lstm_cell_27/SigmoidЛ
lstm_cell_27/Sigmoid_1Sigmoidlstm_cell_27/split:output:1*
T0*(
_output_shapes
:         Ь2
lstm_cell_27/Sigmoid_1М
lstm_cell_27/mulMullstm_cell_27/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:         Ь2
lstm_cell_27/mul~
lstm_cell_27/ReluRelulstm_cell_27/split:output:2*
T0*(
_output_shapes
:         Ь2
lstm_cell_27/ReluЭ
lstm_cell_27/mul_1Mullstm_cell_27/Sigmoid:y:0lstm_cell_27/Relu:activations:0*
T0*(
_output_shapes
:         Ь2
lstm_cell_27/mul_1Т
lstm_cell_27/add_1AddV2lstm_cell_27/mul:z:0lstm_cell_27/mul_1:z:0*
T0*(
_output_shapes
:         Ь2
lstm_cell_27/add_1Л
lstm_cell_27/Sigmoid_2Sigmoidlstm_cell_27/split:output:3*
T0*(
_output_shapes
:         Ь2
lstm_cell_27/Sigmoid_2}
lstm_cell_27/Relu_1Relulstm_cell_27/add_1:z:0*
T0*(
_output_shapes
:         Ь2
lstm_cell_27/Relu_1б
lstm_cell_27/mul_2Mullstm_cell_27/Sigmoid_2:y:0!lstm_cell_27/Relu_1:activations:0*
T0*(
_output_shapes
:         Ь2
lstm_cell_27/mul_2П
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_27_matmul_readvariableop_resource-lstm_cell_27_matmul_1_readvariableop_resource,lstm_cell_27_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :         Ь:         Ь: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_40075878*
condR
while_cond_40075877*M
output_shapes<
:: : : : :         Ь:         Ь: : : : : *
parallel_iterations 2
while╡
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      22
0TensorArrayV2Stack/TensorListStack/element_shapeЄ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:                  Ь*
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
:         Ь*
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
!:                  Ь2
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
!:                  Ь2

Identity╚
NoOpNoOp$^lstm_cell_27/BiasAdd/ReadVariableOp#^lstm_cell_27/MatMul/ReadVariableOp%^lstm_cell_27/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':                  В: : : 2J
#lstm_cell_27/BiasAdd/ReadVariableOp#lstm_cell_27/BiasAdd/ReadVariableOp2H
"lstm_cell_27/MatMul/ReadVariableOp"lstm_cell_27/MatMul/ReadVariableOp2L
$lstm_cell_27/MatMul_1/ReadVariableOp$lstm_cell_27/MatMul_1/ReadVariableOp2
whilewhile:_ [
5
_output_shapes#
!:                  В
"
_user_specified_name
inputs/0
╪
I
-__inference_dropout_27_layer_call_fn_40076330

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
:         Ь* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_dropout_27_layer_call_and_return_conditional_losses_400736532
PartitionedCallq
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:         Ь2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         Ь:T P
,
_output_shapes
:         Ь
 
_user_specified_nameinputs
Й
f
H__inference_dropout_27_layer_call_and_return_conditional_losses_40073653

inputs

identity_1_
IdentityIdentityinputs*
T0*,
_output_shapes
:         Ь2

Identityn

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:         Ь2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         Ь:T P
,
_output_shapes
:         Ь
 
_user_specified_nameinputs
Й
f
H__inference_dropout_26_layer_call_and_return_conditional_losses_40073488

inputs

identity_1_
IdentityIdentityinputs*
T0*,
_output_shapes
:         В2

Identityn

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:         В2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         В:T P
,
_output_shapes
:         В
 
_user_specified_nameinputs
щJ
╓

lstm_27_while_body_40074490,
(lstm_27_while_lstm_27_while_loop_counter2
.lstm_27_while_lstm_27_while_maximum_iterations
lstm_27_while_placeholder
lstm_27_while_placeholder_1
lstm_27_while_placeholder_2
lstm_27_while_placeholder_3+
'lstm_27_while_lstm_27_strided_slice_1_0g
clstm_27_while_tensorarrayv2read_tensorlistgetitem_lstm_27_tensorarrayunstack_tensorlistfromtensor_0O
;lstm_27_while_lstm_cell_27_matmul_readvariableop_resource_0:
ВЁQ
=lstm_27_while_lstm_cell_27_matmul_1_readvariableop_resource_0:
ЬЁK
<lstm_27_while_lstm_cell_27_biasadd_readvariableop_resource_0:	Ё
lstm_27_while_identity
lstm_27_while_identity_1
lstm_27_while_identity_2
lstm_27_while_identity_3
lstm_27_while_identity_4
lstm_27_while_identity_5)
%lstm_27_while_lstm_27_strided_slice_1e
alstm_27_while_tensorarrayv2read_tensorlistgetitem_lstm_27_tensorarrayunstack_tensorlistfromtensorM
9lstm_27_while_lstm_cell_27_matmul_readvariableop_resource:
ВЁO
;lstm_27_while_lstm_cell_27_matmul_1_readvariableop_resource:
ЬЁI
:lstm_27_while_lstm_cell_27_biasadd_readvariableop_resource:	ЁИв1lstm_27/while/lstm_cell_27/BiasAdd/ReadVariableOpв0lstm_27/while/lstm_cell_27/MatMul/ReadVariableOpв2lstm_27/while/lstm_cell_27/MatMul_1/ReadVariableOp╙
?lstm_27/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    В   2A
?lstm_27/while/TensorArrayV2Read/TensorListGetItem/element_shapeД
1lstm_27/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemclstm_27_while_tensorarrayv2read_tensorlistgetitem_lstm_27_tensorarrayunstack_tensorlistfromtensor_0lstm_27_while_placeholderHlstm_27/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:         В*
element_dtype023
1lstm_27/while/TensorArrayV2Read/TensorListGetItemт
0lstm_27/while/lstm_cell_27/MatMul/ReadVariableOpReadVariableOp;lstm_27_while_lstm_cell_27_matmul_readvariableop_resource_0* 
_output_shapes
:
ВЁ*
dtype022
0lstm_27/while/lstm_cell_27/MatMul/ReadVariableOpў
!lstm_27/while/lstm_cell_27/MatMulMatMul8lstm_27/while/TensorArrayV2Read/TensorListGetItem:item:08lstm_27/while/lstm_cell_27/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ё2#
!lstm_27/while/lstm_cell_27/MatMulш
2lstm_27/while/lstm_cell_27/MatMul_1/ReadVariableOpReadVariableOp=lstm_27_while_lstm_cell_27_matmul_1_readvariableop_resource_0* 
_output_shapes
:
ЬЁ*
dtype024
2lstm_27/while/lstm_cell_27/MatMul_1/ReadVariableOpр
#lstm_27/while/lstm_cell_27/MatMul_1MatMullstm_27_while_placeholder_2:lstm_27/while/lstm_cell_27/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ё2%
#lstm_27/while/lstm_cell_27/MatMul_1╪
lstm_27/while/lstm_cell_27/addAddV2+lstm_27/while/lstm_cell_27/MatMul:product:0-lstm_27/while/lstm_cell_27/MatMul_1:product:0*
T0*(
_output_shapes
:         Ё2 
lstm_27/while/lstm_cell_27/addр
1lstm_27/while/lstm_cell_27/BiasAdd/ReadVariableOpReadVariableOp<lstm_27_while_lstm_cell_27_biasadd_readvariableop_resource_0*
_output_shapes	
:Ё*
dtype023
1lstm_27/while/lstm_cell_27/BiasAdd/ReadVariableOpх
"lstm_27/while/lstm_cell_27/BiasAddBiasAdd"lstm_27/while/lstm_cell_27/add:z:09lstm_27/while/lstm_cell_27/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ё2$
"lstm_27/while/lstm_cell_27/BiasAddЪ
*lstm_27/while/lstm_cell_27/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2,
*lstm_27/while/lstm_cell_27/split/split_dimп
 lstm_27/while/lstm_cell_27/splitSplit3lstm_27/while/lstm_cell_27/split/split_dim:output:0+lstm_27/while/lstm_cell_27/BiasAdd:output:0*
T0*d
_output_shapesR
P:         Ь:         Ь:         Ь:         Ь*
	num_split2"
 lstm_27/while/lstm_cell_27/split▒
"lstm_27/while/lstm_cell_27/SigmoidSigmoid)lstm_27/while/lstm_cell_27/split:output:0*
T0*(
_output_shapes
:         Ь2$
"lstm_27/while/lstm_cell_27/Sigmoid╡
$lstm_27/while/lstm_cell_27/Sigmoid_1Sigmoid)lstm_27/while/lstm_cell_27/split:output:1*
T0*(
_output_shapes
:         Ь2&
$lstm_27/while/lstm_cell_27/Sigmoid_1┴
lstm_27/while/lstm_cell_27/mulMul(lstm_27/while/lstm_cell_27/Sigmoid_1:y:0lstm_27_while_placeholder_3*
T0*(
_output_shapes
:         Ь2 
lstm_27/while/lstm_cell_27/mulи
lstm_27/while/lstm_cell_27/ReluRelu)lstm_27/while/lstm_cell_27/split:output:2*
T0*(
_output_shapes
:         Ь2!
lstm_27/while/lstm_cell_27/Relu╒
 lstm_27/while/lstm_cell_27/mul_1Mul&lstm_27/while/lstm_cell_27/Sigmoid:y:0-lstm_27/while/lstm_cell_27/Relu:activations:0*
T0*(
_output_shapes
:         Ь2"
 lstm_27/while/lstm_cell_27/mul_1╩
 lstm_27/while/lstm_cell_27/add_1AddV2"lstm_27/while/lstm_cell_27/mul:z:0$lstm_27/while/lstm_cell_27/mul_1:z:0*
T0*(
_output_shapes
:         Ь2"
 lstm_27/while/lstm_cell_27/add_1╡
$lstm_27/while/lstm_cell_27/Sigmoid_2Sigmoid)lstm_27/while/lstm_cell_27/split:output:3*
T0*(
_output_shapes
:         Ь2&
$lstm_27/while/lstm_cell_27/Sigmoid_2з
!lstm_27/while/lstm_cell_27/Relu_1Relu$lstm_27/while/lstm_cell_27/add_1:z:0*
T0*(
_output_shapes
:         Ь2#
!lstm_27/while/lstm_cell_27/Relu_1┘
 lstm_27/while/lstm_cell_27/mul_2Mul(lstm_27/while/lstm_cell_27/Sigmoid_2:y:0/lstm_27/while/lstm_cell_27/Relu_1:activations:0*
T0*(
_output_shapes
:         Ь2"
 lstm_27/while/lstm_cell_27/mul_2И
2lstm_27/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_27_while_placeholder_1lstm_27_while_placeholder$lstm_27/while/lstm_cell_27/mul_2:z:0*
_output_shapes
: *
element_dtype024
2lstm_27/while/TensorArrayV2Write/TensorListSetIteml
lstm_27/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_27/while/add/yЙ
lstm_27/while/addAddV2lstm_27_while_placeholderlstm_27/while/add/y:output:0*
T0*
_output_shapes
: 2
lstm_27/while/addp
lstm_27/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_27/while/add_1/yЮ
lstm_27/while/add_1AddV2(lstm_27_while_lstm_27_while_loop_counterlstm_27/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lstm_27/while/add_1Л
lstm_27/while/IdentityIdentitylstm_27/while/add_1:z:0^lstm_27/while/NoOp*
T0*
_output_shapes
: 2
lstm_27/while/Identityж
lstm_27/while/Identity_1Identity.lstm_27_while_lstm_27_while_maximum_iterations^lstm_27/while/NoOp*
T0*
_output_shapes
: 2
lstm_27/while/Identity_1Н
lstm_27/while/Identity_2Identitylstm_27/while/add:z:0^lstm_27/while/NoOp*
T0*
_output_shapes
: 2
lstm_27/while/Identity_2║
lstm_27/while/Identity_3IdentityBlstm_27/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_27/while/NoOp*
T0*
_output_shapes
: 2
lstm_27/while/Identity_3о
lstm_27/while/Identity_4Identity$lstm_27/while/lstm_cell_27/mul_2:z:0^lstm_27/while/NoOp*
T0*(
_output_shapes
:         Ь2
lstm_27/while/Identity_4о
lstm_27/while/Identity_5Identity$lstm_27/while/lstm_cell_27/add_1:z:0^lstm_27/while/NoOp*
T0*(
_output_shapes
:         Ь2
lstm_27/while/Identity_5Ж
lstm_27/while/NoOpNoOp2^lstm_27/while/lstm_cell_27/BiasAdd/ReadVariableOp1^lstm_27/while/lstm_cell_27/MatMul/ReadVariableOp3^lstm_27/while/lstm_cell_27/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
lstm_27/while/NoOp"9
lstm_27_while_identitylstm_27/while/Identity:output:0"=
lstm_27_while_identity_1!lstm_27/while/Identity_1:output:0"=
lstm_27_while_identity_2!lstm_27/while/Identity_2:output:0"=
lstm_27_while_identity_3!lstm_27/while/Identity_3:output:0"=
lstm_27_while_identity_4!lstm_27/while/Identity_4:output:0"=
lstm_27_while_identity_5!lstm_27/while/Identity_5:output:0"P
%lstm_27_while_lstm_27_strided_slice_1'lstm_27_while_lstm_27_strided_slice_1_0"z
:lstm_27_while_lstm_cell_27_biasadd_readvariableop_resource<lstm_27_while_lstm_cell_27_biasadd_readvariableop_resource_0"|
;lstm_27_while_lstm_cell_27_matmul_1_readvariableop_resource=lstm_27_while_lstm_cell_27_matmul_1_readvariableop_resource_0"x
9lstm_27_while_lstm_cell_27_matmul_readvariableop_resource;lstm_27_while_lstm_cell_27_matmul_readvariableop_resource_0"╚
alstm_27_while_tensorarrayv2read_tensorlistgetitem_lstm_27_tensorarrayunstack_tensorlistfromtensorclstm_27_while_tensorarrayv2read_tensorlistgetitem_lstm_27_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :         Ь:         Ь: : : : : 2f
1lstm_27/while/lstm_cell_27/BiasAdd/ReadVariableOp1lstm_27/while/lstm_cell_27/BiasAdd/ReadVariableOp2d
0lstm_27/while/lstm_cell_27/MatMul/ReadVariableOp0lstm_27/while/lstm_cell_27/MatMul/ReadVariableOp2h
2lstm_27/while/lstm_cell_27/MatMul_1/ReadVariableOp2lstm_27/while/lstm_cell_27/MatMul_1/ReadVariableOp: 
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
:         Ь:.*
(
_output_shapes
:         Ь:

_output_shapes
: :

_output_shapes
: 
ЄМ
М
K__inference_sequential_13_layer_call_and_return_conditional_losses_40074943

inputsF
3lstm_26_lstm_cell_26_matmul_readvariableop_resource:	]ИI
5lstm_26_lstm_cell_26_matmul_1_readvariableop_resource:
ВИC
4lstm_26_lstm_cell_26_biasadd_readvariableop_resource:	ИG
3lstm_27_lstm_cell_27_matmul_readvariableop_resource:
ВЁI
5lstm_27_lstm_cell_27_matmul_1_readvariableop_resource:
ЬЁC
4lstm_27_lstm_cell_27_biasadd_readvariableop_resource:	Ё=
*dense_13_tensordot_readvariableop_resource:	Ь6
(dense_13_biasadd_readvariableop_resource:
identityИвdense_13/BiasAdd/ReadVariableOpв!dense_13/Tensordot/ReadVariableOpв+lstm_26/lstm_cell_26/BiasAdd/ReadVariableOpв*lstm_26/lstm_cell_26/MatMul/ReadVariableOpв,lstm_26/lstm_cell_26/MatMul_1/ReadVariableOpвlstm_26/whileв+lstm_27/lstm_cell_27/BiasAdd/ReadVariableOpв*lstm_27/lstm_cell_27/MatMul/ReadVariableOpв,lstm_27/lstm_cell_27/MatMul_1/ReadVariableOpвlstm_27/whileT
lstm_26/ShapeShapeinputs*
T0*
_output_shapes
:2
lstm_26/ShapeД
lstm_26/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_26/strided_slice/stackИ
lstm_26/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_26/strided_slice/stack_1И
lstm_26/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_26/strided_slice/stack_2Т
lstm_26/strided_sliceStridedSlicelstm_26/Shape:output:0$lstm_26/strided_slice/stack:output:0&lstm_26/strided_slice/stack_1:output:0&lstm_26/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_26/strided_slicem
lstm_26/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :В2
lstm_26/zeros/mul/yМ
lstm_26/zeros/mulMullstm_26/strided_slice:output:0lstm_26/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_26/zeros/mulo
lstm_26/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
lstm_26/zeros/Less/yЗ
lstm_26/zeros/LessLesslstm_26/zeros/mul:z:0lstm_26/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_26/zeros/Lesss
lstm_26/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :В2
lstm_26/zeros/packed/1г
lstm_26/zeros/packedPacklstm_26/strided_slice:output:0lstm_26/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_26/zeros/packedo
lstm_26/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_26/zeros/ConstЦ
lstm_26/zerosFilllstm_26/zeros/packed:output:0lstm_26/zeros/Const:output:0*
T0*(
_output_shapes
:         В2
lstm_26/zerosq
lstm_26/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :В2
lstm_26/zeros_1/mul/yТ
lstm_26/zeros_1/mulMullstm_26/strided_slice:output:0lstm_26/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_26/zeros_1/muls
lstm_26/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
lstm_26/zeros_1/Less/yП
lstm_26/zeros_1/LessLesslstm_26/zeros_1/mul:z:0lstm_26/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_26/zeros_1/Lessw
lstm_26/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :В2
lstm_26/zeros_1/packed/1й
lstm_26/zeros_1/packedPacklstm_26/strided_slice:output:0!lstm_26/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_26/zeros_1/packeds
lstm_26/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_26/zeros_1/ConstЮ
lstm_26/zeros_1Filllstm_26/zeros_1/packed:output:0lstm_26/zeros_1/Const:output:0*
T0*(
_output_shapes
:         В2
lstm_26/zeros_1Е
lstm_26/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_26/transpose/permТ
lstm_26/transpose	Transposeinputslstm_26/transpose/perm:output:0*
T0*+
_output_shapes
:         ]2
lstm_26/transposeg
lstm_26/Shape_1Shapelstm_26/transpose:y:0*
T0*
_output_shapes
:2
lstm_26/Shape_1И
lstm_26/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_26/strided_slice_1/stackМ
lstm_26/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_26/strided_slice_1/stack_1М
lstm_26/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_26/strided_slice_1/stack_2Ю
lstm_26/strided_slice_1StridedSlicelstm_26/Shape_1:output:0&lstm_26/strided_slice_1/stack:output:0(lstm_26/strided_slice_1/stack_1:output:0(lstm_26/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_26/strided_slice_1Х
#lstm_26/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         2%
#lstm_26/TensorArrayV2/element_shape╥
lstm_26/TensorArrayV2TensorListReserve,lstm_26/TensorArrayV2/element_shape:output:0 lstm_26/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_26/TensorArrayV2╧
=lstm_26/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ]   2?
=lstm_26/TensorArrayUnstack/TensorListFromTensor/element_shapeШ
/lstm_26/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_26/transpose:y:0Flstm_26/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type021
/lstm_26/TensorArrayUnstack/TensorListFromTensorИ
lstm_26/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_26/strided_slice_2/stackМ
lstm_26/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_26/strided_slice_2/stack_1М
lstm_26/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_26/strided_slice_2/stack_2м
lstm_26/strided_slice_2StridedSlicelstm_26/transpose:y:0&lstm_26/strided_slice_2/stack:output:0(lstm_26/strided_slice_2/stack_1:output:0(lstm_26/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         ]*
shrink_axis_mask2
lstm_26/strided_slice_2═
*lstm_26/lstm_cell_26/MatMul/ReadVariableOpReadVariableOp3lstm_26_lstm_cell_26_matmul_readvariableop_resource*
_output_shapes
:	]И*
dtype02,
*lstm_26/lstm_cell_26/MatMul/ReadVariableOp═
lstm_26/lstm_cell_26/MatMulMatMul lstm_26/strided_slice_2:output:02lstm_26/lstm_cell_26/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         И2
lstm_26/lstm_cell_26/MatMul╘
,lstm_26/lstm_cell_26/MatMul_1/ReadVariableOpReadVariableOp5lstm_26_lstm_cell_26_matmul_1_readvariableop_resource* 
_output_shapes
:
ВИ*
dtype02.
,lstm_26/lstm_cell_26/MatMul_1/ReadVariableOp╔
lstm_26/lstm_cell_26/MatMul_1MatMullstm_26/zeros:output:04lstm_26/lstm_cell_26/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         И2
lstm_26/lstm_cell_26/MatMul_1└
lstm_26/lstm_cell_26/addAddV2%lstm_26/lstm_cell_26/MatMul:product:0'lstm_26/lstm_cell_26/MatMul_1:product:0*
T0*(
_output_shapes
:         И2
lstm_26/lstm_cell_26/add╠
+lstm_26/lstm_cell_26/BiasAdd/ReadVariableOpReadVariableOp4lstm_26_lstm_cell_26_biasadd_readvariableop_resource*
_output_shapes	
:И*
dtype02-
+lstm_26/lstm_cell_26/BiasAdd/ReadVariableOp═
lstm_26/lstm_cell_26/BiasAddBiasAddlstm_26/lstm_cell_26/add:z:03lstm_26/lstm_cell_26/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         И2
lstm_26/lstm_cell_26/BiasAddО
$lstm_26/lstm_cell_26/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2&
$lstm_26/lstm_cell_26/split/split_dimЧ
lstm_26/lstm_cell_26/splitSplit-lstm_26/lstm_cell_26/split/split_dim:output:0%lstm_26/lstm_cell_26/BiasAdd:output:0*
T0*d
_output_shapesR
P:         В:         В:         В:         В*
	num_split2
lstm_26/lstm_cell_26/splitЯ
lstm_26/lstm_cell_26/SigmoidSigmoid#lstm_26/lstm_cell_26/split:output:0*
T0*(
_output_shapes
:         В2
lstm_26/lstm_cell_26/Sigmoidг
lstm_26/lstm_cell_26/Sigmoid_1Sigmoid#lstm_26/lstm_cell_26/split:output:1*
T0*(
_output_shapes
:         В2 
lstm_26/lstm_cell_26/Sigmoid_1м
lstm_26/lstm_cell_26/mulMul"lstm_26/lstm_cell_26/Sigmoid_1:y:0lstm_26/zeros_1:output:0*
T0*(
_output_shapes
:         В2
lstm_26/lstm_cell_26/mulЦ
lstm_26/lstm_cell_26/ReluRelu#lstm_26/lstm_cell_26/split:output:2*
T0*(
_output_shapes
:         В2
lstm_26/lstm_cell_26/Relu╜
lstm_26/lstm_cell_26/mul_1Mul lstm_26/lstm_cell_26/Sigmoid:y:0'lstm_26/lstm_cell_26/Relu:activations:0*
T0*(
_output_shapes
:         В2
lstm_26/lstm_cell_26/mul_1▓
lstm_26/lstm_cell_26/add_1AddV2lstm_26/lstm_cell_26/mul:z:0lstm_26/lstm_cell_26/mul_1:z:0*
T0*(
_output_shapes
:         В2
lstm_26/lstm_cell_26/add_1г
lstm_26/lstm_cell_26/Sigmoid_2Sigmoid#lstm_26/lstm_cell_26/split:output:3*
T0*(
_output_shapes
:         В2 
lstm_26/lstm_cell_26/Sigmoid_2Х
lstm_26/lstm_cell_26/Relu_1Relulstm_26/lstm_cell_26/add_1:z:0*
T0*(
_output_shapes
:         В2
lstm_26/lstm_cell_26/Relu_1┴
lstm_26/lstm_cell_26/mul_2Mul"lstm_26/lstm_cell_26/Sigmoid_2:y:0)lstm_26/lstm_cell_26/Relu_1:activations:0*
T0*(
_output_shapes
:         В2
lstm_26/lstm_cell_26/mul_2Я
%lstm_26/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    В   2'
%lstm_26/TensorArrayV2_1/element_shape╪
lstm_26/TensorArrayV2_1TensorListReserve.lstm_26/TensorArrayV2_1/element_shape:output:0 lstm_26/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_26/TensorArrayV2_1^
lstm_26/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_26/timeП
 lstm_26/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         2"
 lstm_26/while/maximum_iterationsz
lstm_26/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_26/while/loop_counterЛ
lstm_26/whileWhile#lstm_26/while/loop_counter:output:0)lstm_26/while/maximum_iterations:output:0lstm_26/time:output:0 lstm_26/TensorArrayV2_1:handle:0lstm_26/zeros:output:0lstm_26/zeros_1:output:0 lstm_26/strided_slice_1:output:0?lstm_26/TensorArrayUnstack/TensorListFromTensor:output_handle:03lstm_26_lstm_cell_26_matmul_readvariableop_resource5lstm_26_lstm_cell_26_matmul_1_readvariableop_resource4lstm_26_lstm_cell_26_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :         В:         В: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *'
bodyR
lstm_26_while_body_40074669*'
condR
lstm_26_while_cond_40074668*M
output_shapes<
:: : : : :         В:         В: : : : : *
parallel_iterations 2
lstm_26/while┼
8lstm_26/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    В   2:
8lstm_26/TensorArrayV2Stack/TensorListStack/element_shapeЙ
*lstm_26/TensorArrayV2Stack/TensorListStackTensorListStacklstm_26/while:output:3Alstm_26/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:         В*
element_dtype02,
*lstm_26/TensorArrayV2Stack/TensorListStackС
lstm_26/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         2
lstm_26/strided_slice_3/stackМ
lstm_26/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2!
lstm_26/strided_slice_3/stack_1М
lstm_26/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_26/strided_slice_3/stack_2╦
lstm_26/strided_slice_3StridedSlice3lstm_26/TensorArrayV2Stack/TensorListStack:tensor:0&lstm_26/strided_slice_3/stack:output:0(lstm_26/strided_slice_3/stack_1:output:0(lstm_26/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         В*
shrink_axis_mask2
lstm_26/strided_slice_3Й
lstm_26/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_26/transpose_1/perm╞
lstm_26/transpose_1	Transpose3lstm_26/TensorArrayV2Stack/TensorListStack:tensor:0!lstm_26/transpose_1/perm:output:0*
T0*,
_output_shapes
:         В2
lstm_26/transpose_1v
lstm_26/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_26/runtimey
dropout_26/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *лкк?2
dropout_26/dropout/Constк
dropout_26/dropout/MulMullstm_26/transpose_1:y:0!dropout_26/dropout/Const:output:0*
T0*,
_output_shapes
:         В2
dropout_26/dropout/Mul{
dropout_26/dropout/ShapeShapelstm_26/transpose_1:y:0*
T0*
_output_shapes
:2
dropout_26/dropout/Shape┌
/dropout_26/dropout/random_uniform/RandomUniformRandomUniform!dropout_26/dropout/Shape:output:0*
T0*,
_output_shapes
:         В*
dtype021
/dropout_26/dropout/random_uniform/RandomUniformЛ
!dropout_26/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А>2#
!dropout_26/dropout/GreaterEqual/yя
dropout_26/dropout/GreaterEqualGreaterEqual8dropout_26/dropout/random_uniform/RandomUniform:output:0*dropout_26/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:         В2!
dropout_26/dropout/GreaterEqualе
dropout_26/dropout/CastCast#dropout_26/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:         В2
dropout_26/dropout/Castл
dropout_26/dropout/Mul_1Muldropout_26/dropout/Mul:z:0dropout_26/dropout/Cast:y:0*
T0*,
_output_shapes
:         В2
dropout_26/dropout/Mul_1j
lstm_27/ShapeShapedropout_26/dropout/Mul_1:z:0*
T0*
_output_shapes
:2
lstm_27/ShapeД
lstm_27/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_27/strided_slice/stackИ
lstm_27/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_27/strided_slice/stack_1И
lstm_27/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_27/strided_slice/stack_2Т
lstm_27/strided_sliceStridedSlicelstm_27/Shape:output:0$lstm_27/strided_slice/stack:output:0&lstm_27/strided_slice/stack_1:output:0&lstm_27/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_27/strided_slicem
lstm_27/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :Ь2
lstm_27/zeros/mul/yМ
lstm_27/zeros/mulMullstm_27/strided_slice:output:0lstm_27/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_27/zeros/mulo
lstm_27/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
lstm_27/zeros/Less/yЗ
lstm_27/zeros/LessLesslstm_27/zeros/mul:z:0lstm_27/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_27/zeros/Lesss
lstm_27/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :Ь2
lstm_27/zeros/packed/1г
lstm_27/zeros/packedPacklstm_27/strided_slice:output:0lstm_27/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_27/zeros/packedo
lstm_27/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_27/zeros/ConstЦ
lstm_27/zerosFilllstm_27/zeros/packed:output:0lstm_27/zeros/Const:output:0*
T0*(
_output_shapes
:         Ь2
lstm_27/zerosq
lstm_27/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :Ь2
lstm_27/zeros_1/mul/yТ
lstm_27/zeros_1/mulMullstm_27/strided_slice:output:0lstm_27/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_27/zeros_1/muls
lstm_27/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
lstm_27/zeros_1/Less/yП
lstm_27/zeros_1/LessLesslstm_27/zeros_1/mul:z:0lstm_27/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_27/zeros_1/Lessw
lstm_27/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :Ь2
lstm_27/zeros_1/packed/1й
lstm_27/zeros_1/packedPacklstm_27/strided_slice:output:0!lstm_27/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_27/zeros_1/packeds
lstm_27/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_27/zeros_1/ConstЮ
lstm_27/zeros_1Filllstm_27/zeros_1/packed:output:0lstm_27/zeros_1/Const:output:0*
T0*(
_output_shapes
:         Ь2
lstm_27/zeros_1Е
lstm_27/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_27/transpose/permй
lstm_27/transpose	Transposedropout_26/dropout/Mul_1:z:0lstm_27/transpose/perm:output:0*
T0*,
_output_shapes
:         В2
lstm_27/transposeg
lstm_27/Shape_1Shapelstm_27/transpose:y:0*
T0*
_output_shapes
:2
lstm_27/Shape_1И
lstm_27/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_27/strided_slice_1/stackМ
lstm_27/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_27/strided_slice_1/stack_1М
lstm_27/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_27/strided_slice_1/stack_2Ю
lstm_27/strided_slice_1StridedSlicelstm_27/Shape_1:output:0&lstm_27/strided_slice_1/stack:output:0(lstm_27/strided_slice_1/stack_1:output:0(lstm_27/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_27/strided_slice_1Х
#lstm_27/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         2%
#lstm_27/TensorArrayV2/element_shape╥
lstm_27/TensorArrayV2TensorListReserve,lstm_27/TensorArrayV2/element_shape:output:0 lstm_27/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_27/TensorArrayV2╧
=lstm_27/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    В   2?
=lstm_27/TensorArrayUnstack/TensorListFromTensor/element_shapeШ
/lstm_27/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_27/transpose:y:0Flstm_27/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type021
/lstm_27/TensorArrayUnstack/TensorListFromTensorИ
lstm_27/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_27/strided_slice_2/stackМ
lstm_27/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_27/strided_slice_2/stack_1М
lstm_27/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_27/strided_slice_2/stack_2н
lstm_27/strided_slice_2StridedSlicelstm_27/transpose:y:0&lstm_27/strided_slice_2/stack:output:0(lstm_27/strided_slice_2/stack_1:output:0(lstm_27/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         В*
shrink_axis_mask2
lstm_27/strided_slice_2╬
*lstm_27/lstm_cell_27/MatMul/ReadVariableOpReadVariableOp3lstm_27_lstm_cell_27_matmul_readvariableop_resource* 
_output_shapes
:
ВЁ*
dtype02,
*lstm_27/lstm_cell_27/MatMul/ReadVariableOp═
lstm_27/lstm_cell_27/MatMulMatMul lstm_27/strided_slice_2:output:02lstm_27/lstm_cell_27/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ё2
lstm_27/lstm_cell_27/MatMul╘
,lstm_27/lstm_cell_27/MatMul_1/ReadVariableOpReadVariableOp5lstm_27_lstm_cell_27_matmul_1_readvariableop_resource* 
_output_shapes
:
ЬЁ*
dtype02.
,lstm_27/lstm_cell_27/MatMul_1/ReadVariableOp╔
lstm_27/lstm_cell_27/MatMul_1MatMullstm_27/zeros:output:04lstm_27/lstm_cell_27/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ё2
lstm_27/lstm_cell_27/MatMul_1└
lstm_27/lstm_cell_27/addAddV2%lstm_27/lstm_cell_27/MatMul:product:0'lstm_27/lstm_cell_27/MatMul_1:product:0*
T0*(
_output_shapes
:         Ё2
lstm_27/lstm_cell_27/add╠
+lstm_27/lstm_cell_27/BiasAdd/ReadVariableOpReadVariableOp4lstm_27_lstm_cell_27_biasadd_readvariableop_resource*
_output_shapes	
:Ё*
dtype02-
+lstm_27/lstm_cell_27/BiasAdd/ReadVariableOp═
lstm_27/lstm_cell_27/BiasAddBiasAddlstm_27/lstm_cell_27/add:z:03lstm_27/lstm_cell_27/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ё2
lstm_27/lstm_cell_27/BiasAddО
$lstm_27/lstm_cell_27/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2&
$lstm_27/lstm_cell_27/split/split_dimЧ
lstm_27/lstm_cell_27/splitSplit-lstm_27/lstm_cell_27/split/split_dim:output:0%lstm_27/lstm_cell_27/BiasAdd:output:0*
T0*d
_output_shapesR
P:         Ь:         Ь:         Ь:         Ь*
	num_split2
lstm_27/lstm_cell_27/splitЯ
lstm_27/lstm_cell_27/SigmoidSigmoid#lstm_27/lstm_cell_27/split:output:0*
T0*(
_output_shapes
:         Ь2
lstm_27/lstm_cell_27/Sigmoidг
lstm_27/lstm_cell_27/Sigmoid_1Sigmoid#lstm_27/lstm_cell_27/split:output:1*
T0*(
_output_shapes
:         Ь2 
lstm_27/lstm_cell_27/Sigmoid_1м
lstm_27/lstm_cell_27/mulMul"lstm_27/lstm_cell_27/Sigmoid_1:y:0lstm_27/zeros_1:output:0*
T0*(
_output_shapes
:         Ь2
lstm_27/lstm_cell_27/mulЦ
lstm_27/lstm_cell_27/ReluRelu#lstm_27/lstm_cell_27/split:output:2*
T0*(
_output_shapes
:         Ь2
lstm_27/lstm_cell_27/Relu╜
lstm_27/lstm_cell_27/mul_1Mul lstm_27/lstm_cell_27/Sigmoid:y:0'lstm_27/lstm_cell_27/Relu:activations:0*
T0*(
_output_shapes
:         Ь2
lstm_27/lstm_cell_27/mul_1▓
lstm_27/lstm_cell_27/add_1AddV2lstm_27/lstm_cell_27/mul:z:0lstm_27/lstm_cell_27/mul_1:z:0*
T0*(
_output_shapes
:         Ь2
lstm_27/lstm_cell_27/add_1г
lstm_27/lstm_cell_27/Sigmoid_2Sigmoid#lstm_27/lstm_cell_27/split:output:3*
T0*(
_output_shapes
:         Ь2 
lstm_27/lstm_cell_27/Sigmoid_2Х
lstm_27/lstm_cell_27/Relu_1Relulstm_27/lstm_cell_27/add_1:z:0*
T0*(
_output_shapes
:         Ь2
lstm_27/lstm_cell_27/Relu_1┴
lstm_27/lstm_cell_27/mul_2Mul"lstm_27/lstm_cell_27/Sigmoid_2:y:0)lstm_27/lstm_cell_27/Relu_1:activations:0*
T0*(
_output_shapes
:         Ь2
lstm_27/lstm_cell_27/mul_2Я
%lstm_27/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      2'
%lstm_27/TensorArrayV2_1/element_shape╪
lstm_27/TensorArrayV2_1TensorListReserve.lstm_27/TensorArrayV2_1/element_shape:output:0 lstm_27/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_27/TensorArrayV2_1^
lstm_27/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_27/timeП
 lstm_27/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         2"
 lstm_27/while/maximum_iterationsz
lstm_27/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_27/while/loop_counterЛ
lstm_27/whileWhile#lstm_27/while/loop_counter:output:0)lstm_27/while/maximum_iterations:output:0lstm_27/time:output:0 lstm_27/TensorArrayV2_1:handle:0lstm_27/zeros:output:0lstm_27/zeros_1:output:0 lstm_27/strided_slice_1:output:0?lstm_27/TensorArrayUnstack/TensorListFromTensor:output_handle:03lstm_27_lstm_cell_27_matmul_readvariableop_resource5lstm_27_lstm_cell_27_matmul_1_readvariableop_resource4lstm_27_lstm_cell_27_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :         Ь:         Ь: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *'
bodyR
lstm_27_while_body_40074824*'
condR
lstm_27_while_cond_40074823*M
output_shapes<
:: : : : :         Ь:         Ь: : : : : *
parallel_iterations 2
lstm_27/while┼
8lstm_27/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      2:
8lstm_27/TensorArrayV2Stack/TensorListStack/element_shapeЙ
*lstm_27/TensorArrayV2Stack/TensorListStackTensorListStacklstm_27/while:output:3Alstm_27/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:         Ь*
element_dtype02,
*lstm_27/TensorArrayV2Stack/TensorListStackС
lstm_27/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         2
lstm_27/strided_slice_3/stackМ
lstm_27/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2!
lstm_27/strided_slice_3/stack_1М
lstm_27/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_27/strided_slice_3/stack_2╦
lstm_27/strided_slice_3StridedSlice3lstm_27/TensorArrayV2Stack/TensorListStack:tensor:0&lstm_27/strided_slice_3/stack:output:0(lstm_27/strided_slice_3/stack_1:output:0(lstm_27/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         Ь*
shrink_axis_mask2
lstm_27/strided_slice_3Й
lstm_27/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_27/transpose_1/perm╞
lstm_27/transpose_1	Transpose3lstm_27/TensorArrayV2Stack/TensorListStack:tensor:0!lstm_27/transpose_1/perm:output:0*
T0*,
_output_shapes
:         Ь2
lstm_27/transpose_1v
lstm_27/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_27/runtimey
dropout_27/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  а?2
dropout_27/dropout/Constк
dropout_27/dropout/MulMullstm_27/transpose_1:y:0!dropout_27/dropout/Const:output:0*
T0*,
_output_shapes
:         Ь2
dropout_27/dropout/Mul{
dropout_27/dropout/ShapeShapelstm_27/transpose_1:y:0*
T0*
_output_shapes
:2
dropout_27/dropout/Shape┌
/dropout_27/dropout/random_uniform/RandomUniformRandomUniform!dropout_27/dropout/Shape:output:0*
T0*,
_output_shapes
:         Ь*
dtype021
/dropout_27/dropout/random_uniform/RandomUniformЛ
!dropout_27/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>2#
!dropout_27/dropout/GreaterEqual/yя
dropout_27/dropout/GreaterEqualGreaterEqual8dropout_27/dropout/random_uniform/RandomUniform:output:0*dropout_27/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:         Ь2!
dropout_27/dropout/GreaterEqualе
dropout_27/dropout/CastCast#dropout_27/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:         Ь2
dropout_27/dropout/Castл
dropout_27/dropout/Mul_1Muldropout_27/dropout/Mul:z:0dropout_27/dropout/Cast:y:0*
T0*,
_output_shapes
:         Ь2
dropout_27/dropout/Mul_1▓
!dense_13/Tensordot/ReadVariableOpReadVariableOp*dense_13_tensordot_readvariableop_resource*
_output_shapes
:	Ь*
dtype02#
!dense_13/Tensordot/ReadVariableOp|
dense_13/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_13/Tensordot/axesГ
dense_13/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_13/Tensordot/freeА
dense_13/Tensordot/ShapeShapedropout_27/dropout/Mul_1:z:0*
T0*
_output_shapes
:2
dense_13/Tensordot/ShapeЖ
 dense_13/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_13/Tensordot/GatherV2/axis■
dense_13/Tensordot/GatherV2GatherV2!dense_13/Tensordot/Shape:output:0 dense_13/Tensordot/free:output:0)dense_13/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_13/Tensordot/GatherV2К
"dense_13/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"dense_13/Tensordot/GatherV2_1/axisД
dense_13/Tensordot/GatherV2_1GatherV2!dense_13/Tensordot/Shape:output:0 dense_13/Tensordot/axes:output:0+dense_13/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_13/Tensordot/GatherV2_1~
dense_13/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_13/Tensordot/Constд
dense_13/Tensordot/ProdProd$dense_13/Tensordot/GatherV2:output:0!dense_13/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_13/Tensordot/ProdВ
dense_13/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_13/Tensordot/Const_1м
dense_13/Tensordot/Prod_1Prod&dense_13/Tensordot/GatherV2_1:output:0#dense_13/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_13/Tensordot/Prod_1В
dense_13/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2 
dense_13/Tensordot/concat/axis▌
dense_13/Tensordot/concatConcatV2 dense_13/Tensordot/free:output:0 dense_13/Tensordot/axes:output:0'dense_13/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_13/Tensordot/concat░
dense_13/Tensordot/stackPack dense_13/Tensordot/Prod:output:0"dense_13/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_13/Tensordot/stack┬
dense_13/Tensordot/transpose	Transposedropout_27/dropout/Mul_1:z:0"dense_13/Tensordot/concat:output:0*
T0*,
_output_shapes
:         Ь2
dense_13/Tensordot/transpose├
dense_13/Tensordot/ReshapeReshape dense_13/Tensordot/transpose:y:0!dense_13/Tensordot/stack:output:0*
T0*0
_output_shapes
:                  2
dense_13/Tensordot/Reshape┬
dense_13/Tensordot/MatMulMatMul#dense_13/Tensordot/Reshape:output:0)dense_13/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_13/Tensordot/MatMulВ
dense_13/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
dense_13/Tensordot/Const_2Ж
 dense_13/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_13/Tensordot/concat_1/axisъ
dense_13/Tensordot/concat_1ConcatV2$dense_13/Tensordot/GatherV2:output:0#dense_13/Tensordot/Const_2:output:0)dense_13/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_13/Tensordot/concat_1┤
dense_13/TensordotReshape#dense_13/Tensordot/MatMul:product:0$dense_13/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:         2
dense_13/Tensordotз
dense_13/BiasAdd/ReadVariableOpReadVariableOp(dense_13_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_13/BiasAdd/ReadVariableOpл
dense_13/BiasAddBiasAdddense_13/Tensordot:output:0'dense_13/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:         2
dense_13/BiasAddА
dense_13/SoftmaxSoftmaxdense_13/BiasAdd:output:0*
T0*+
_output_shapes
:         2
dense_13/Softmaxy
IdentityIdentitydense_13/Softmax:softmax:0^NoOp*
T0*+
_output_shapes
:         2

Identity╚
NoOpNoOp ^dense_13/BiasAdd/ReadVariableOp"^dense_13/Tensordot/ReadVariableOp,^lstm_26/lstm_cell_26/BiasAdd/ReadVariableOp+^lstm_26/lstm_cell_26/MatMul/ReadVariableOp-^lstm_26/lstm_cell_26/MatMul_1/ReadVariableOp^lstm_26/while,^lstm_27/lstm_cell_27/BiasAdd/ReadVariableOp+^lstm_27/lstm_cell_27/MatMul/ReadVariableOp-^lstm_27/lstm_cell_27/MatMul_1/ReadVariableOp^lstm_27/while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':         ]: : : : : : : : 2B
dense_13/BiasAdd/ReadVariableOpdense_13/BiasAdd/ReadVariableOp2F
!dense_13/Tensordot/ReadVariableOp!dense_13/Tensordot/ReadVariableOp2Z
+lstm_26/lstm_cell_26/BiasAdd/ReadVariableOp+lstm_26/lstm_cell_26/BiasAdd/ReadVariableOp2X
*lstm_26/lstm_cell_26/MatMul/ReadVariableOp*lstm_26/lstm_cell_26/MatMul/ReadVariableOp2\
,lstm_26/lstm_cell_26/MatMul_1/ReadVariableOp,lstm_26/lstm_cell_26/MatMul_1/ReadVariableOp2
lstm_26/whilelstm_26/while2Z
+lstm_27/lstm_cell_27/BiasAdd/ReadVariableOp+lstm_27/lstm_cell_27/BiasAdd/ReadVariableOp2X
*lstm_27/lstm_cell_27/MatMul/ReadVariableOp*lstm_27/lstm_cell_27/MatMul/ReadVariableOp2\
,lstm_27/lstm_cell_27/MatMul_1/ReadVariableOp,lstm_27/lstm_cell_27/MatMul_1/ReadVariableOp2
lstm_27/whilelstm_27/while:S O
+
_output_shapes
:         ]
 
_user_specified_nameinputs
у
╗
*__inference_lstm_27_layer_call_fn_40076286
inputs_0
unknown:
ВЁ
	unknown_0:
ЬЁ
	unknown_1:	Ё
identityИвStatefulPartitionedCallТ
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:                  Ь*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_lstm_27_layer_call_and_return_conditional_losses_400730562
StatefulPartitionedCallЙ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:                  Ь2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':                  В: : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
5
_output_shapes#
!:                  В
"
_user_specified_name
inputs/0
╘

э
lstm_27_while_cond_40074823,
(lstm_27_while_lstm_27_while_loop_counter2
.lstm_27_while_lstm_27_while_maximum_iterations
lstm_27_while_placeholder
lstm_27_while_placeholder_1
lstm_27_while_placeholder_2
lstm_27_while_placeholder_3.
*lstm_27_while_less_lstm_27_strided_slice_1F
Blstm_27_while_lstm_27_while_cond_40074823___redundant_placeholder0F
Blstm_27_while_lstm_27_while_cond_40074823___redundant_placeholder1F
Blstm_27_while_lstm_27_while_cond_40074823___redundant_placeholder2F
Blstm_27_while_lstm_27_while_cond_40074823___redundant_placeholder3
lstm_27_while_identity
Ш
lstm_27/while/LessLesslstm_27_while_placeholder*lstm_27_while_less_lstm_27_strided_slice_1*
T0*
_output_shapes
: 2
lstm_27/while/Lessu
lstm_27/while/IdentityIdentitylstm_27/while/Less:z:0*
T0
*
_output_shapes
: 2
lstm_27/while/Identity"9
lstm_27_while_identitylstm_27/while/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :         Ь:         Ь: ::::: 
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
:         Ь:.*
(
_output_shapes
:         Ь:

_output_shapes
: :

_output_shapes
:
Д\
Ю
E__inference_lstm_26_layer_call_and_return_conditional_losses_40075589

inputs>
+lstm_cell_26_matmul_readvariableop_resource:	]ИA
-lstm_cell_26_matmul_1_readvariableop_resource:
ВИ;
,lstm_cell_26_biasadd_readvariableop_resource:	И
identityИв#lstm_cell_26/BiasAdd/ReadVariableOpв"lstm_cell_26/MatMul/ReadVariableOpв$lstm_cell_26/MatMul_1/ReadVariableOpвwhileD
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
B :В2
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
B :В2
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
:         В2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :В2
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
B :В2
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
:         В2	
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
"lstm_cell_26/MatMul/ReadVariableOpReadVariableOp+lstm_cell_26_matmul_readvariableop_resource*
_output_shapes
:	]И*
dtype02$
"lstm_cell_26/MatMul/ReadVariableOpн
lstm_cell_26/MatMulMatMulstrided_slice_2:output:0*lstm_cell_26/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         И2
lstm_cell_26/MatMul╝
$lstm_cell_26/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_26_matmul_1_readvariableop_resource* 
_output_shapes
:
ВИ*
dtype02&
$lstm_cell_26/MatMul_1/ReadVariableOpй
lstm_cell_26/MatMul_1MatMulzeros:output:0,lstm_cell_26/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         И2
lstm_cell_26/MatMul_1а
lstm_cell_26/addAddV2lstm_cell_26/MatMul:product:0lstm_cell_26/MatMul_1:product:0*
T0*(
_output_shapes
:         И2
lstm_cell_26/add┤
#lstm_cell_26/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_26_biasadd_readvariableop_resource*
_output_shapes	
:И*
dtype02%
#lstm_cell_26/BiasAdd/ReadVariableOpн
lstm_cell_26/BiasAddBiasAddlstm_cell_26/add:z:0+lstm_cell_26/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         И2
lstm_cell_26/BiasAdd~
lstm_cell_26/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_26/split/split_dimў
lstm_cell_26/splitSplit%lstm_cell_26/split/split_dim:output:0lstm_cell_26/BiasAdd:output:0*
T0*d
_output_shapesR
P:         В:         В:         В:         В*
	num_split2
lstm_cell_26/splitЗ
lstm_cell_26/SigmoidSigmoidlstm_cell_26/split:output:0*
T0*(
_output_shapes
:         В2
lstm_cell_26/SigmoidЛ
lstm_cell_26/Sigmoid_1Sigmoidlstm_cell_26/split:output:1*
T0*(
_output_shapes
:         В2
lstm_cell_26/Sigmoid_1М
lstm_cell_26/mulMullstm_cell_26/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:         В2
lstm_cell_26/mul~
lstm_cell_26/ReluRelulstm_cell_26/split:output:2*
T0*(
_output_shapes
:         В2
lstm_cell_26/ReluЭ
lstm_cell_26/mul_1Mullstm_cell_26/Sigmoid:y:0lstm_cell_26/Relu:activations:0*
T0*(
_output_shapes
:         В2
lstm_cell_26/mul_1Т
lstm_cell_26/add_1AddV2lstm_cell_26/mul:z:0lstm_cell_26/mul_1:z:0*
T0*(
_output_shapes
:         В2
lstm_cell_26/add_1Л
lstm_cell_26/Sigmoid_2Sigmoidlstm_cell_26/split:output:3*
T0*(
_output_shapes
:         В2
lstm_cell_26/Sigmoid_2}
lstm_cell_26/Relu_1Relulstm_cell_26/add_1:z:0*
T0*(
_output_shapes
:         В2
lstm_cell_26/Relu_1б
lstm_cell_26/mul_2Mullstm_cell_26/Sigmoid_2:y:0!lstm_cell_26/Relu_1:activations:0*
T0*(
_output_shapes
:         В2
lstm_cell_26/mul_2П
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    В   2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_26_matmul_readvariableop_resource-lstm_cell_26_matmul_1_readvariableop_resource,lstm_cell_26_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :         В:         В: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_40075505*
condR
while_cond_40075504*M
output_shapes<
:: : : : :         В:         В: : : : : *
parallel_iterations 2
while╡
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    В   22
0TensorArrayV2Stack/TensorListStack/element_shapeщ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:         В*
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
:         В*
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
:         В2
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
:         В2

Identity╚
NoOpNoOp$^lstm_cell_26/BiasAdd/ReadVariableOp#^lstm_cell_26/MatMul/ReadVariableOp%^lstm_cell_26/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         ]: : : 2J
#lstm_cell_26/BiasAdd/ReadVariableOp#lstm_cell_26/BiasAdd/ReadVariableOp2H
"lstm_cell_26/MatMul/ReadVariableOp"lstm_cell_26/MatMul/ReadVariableOp2L
$lstm_cell_26/MatMul_1/ReadVariableOp$lstm_cell_26/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:         ]
 
_user_specified_nameinputs
ф^
Ц
)sequential_13_lstm_26_while_body_40071798H
Dsequential_13_lstm_26_while_sequential_13_lstm_26_while_loop_counterN
Jsequential_13_lstm_26_while_sequential_13_lstm_26_while_maximum_iterations+
'sequential_13_lstm_26_while_placeholder-
)sequential_13_lstm_26_while_placeholder_1-
)sequential_13_lstm_26_while_placeholder_2-
)sequential_13_lstm_26_while_placeholder_3G
Csequential_13_lstm_26_while_sequential_13_lstm_26_strided_slice_1_0Г
sequential_13_lstm_26_while_tensorarrayv2read_tensorlistgetitem_sequential_13_lstm_26_tensorarrayunstack_tensorlistfromtensor_0\
Isequential_13_lstm_26_while_lstm_cell_26_matmul_readvariableop_resource_0:	]И_
Ksequential_13_lstm_26_while_lstm_cell_26_matmul_1_readvariableop_resource_0:
ВИY
Jsequential_13_lstm_26_while_lstm_cell_26_biasadd_readvariableop_resource_0:	И(
$sequential_13_lstm_26_while_identity*
&sequential_13_lstm_26_while_identity_1*
&sequential_13_lstm_26_while_identity_2*
&sequential_13_lstm_26_while_identity_3*
&sequential_13_lstm_26_while_identity_4*
&sequential_13_lstm_26_while_identity_5E
Asequential_13_lstm_26_while_sequential_13_lstm_26_strided_slice_1Б
}sequential_13_lstm_26_while_tensorarrayv2read_tensorlistgetitem_sequential_13_lstm_26_tensorarrayunstack_tensorlistfromtensorZ
Gsequential_13_lstm_26_while_lstm_cell_26_matmul_readvariableop_resource:	]И]
Isequential_13_lstm_26_while_lstm_cell_26_matmul_1_readvariableop_resource:
ВИW
Hsequential_13_lstm_26_while_lstm_cell_26_biasadd_readvariableop_resource:	ИИв?sequential_13/lstm_26/while/lstm_cell_26/BiasAdd/ReadVariableOpв>sequential_13/lstm_26/while/lstm_cell_26/MatMul/ReadVariableOpв@sequential_13/lstm_26/while/lstm_cell_26/MatMul_1/ReadVariableOpя
Msequential_13/lstm_26/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ]   2O
Msequential_13/lstm_26/while/TensorArrayV2Read/TensorListGetItem/element_shape╫
?sequential_13/lstm_26/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemsequential_13_lstm_26_while_tensorarrayv2read_tensorlistgetitem_sequential_13_lstm_26_tensorarrayunstack_tensorlistfromtensor_0'sequential_13_lstm_26_while_placeholderVsequential_13/lstm_26/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         ]*
element_dtype02A
?sequential_13/lstm_26/while/TensorArrayV2Read/TensorListGetItemЛ
>sequential_13/lstm_26/while/lstm_cell_26/MatMul/ReadVariableOpReadVariableOpIsequential_13_lstm_26_while_lstm_cell_26_matmul_readvariableop_resource_0*
_output_shapes
:	]И*
dtype02@
>sequential_13/lstm_26/while/lstm_cell_26/MatMul/ReadVariableOpп
/sequential_13/lstm_26/while/lstm_cell_26/MatMulMatMulFsequential_13/lstm_26/while/TensorArrayV2Read/TensorListGetItem:item:0Fsequential_13/lstm_26/while/lstm_cell_26/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         И21
/sequential_13/lstm_26/while/lstm_cell_26/MatMulТ
@sequential_13/lstm_26/while/lstm_cell_26/MatMul_1/ReadVariableOpReadVariableOpKsequential_13_lstm_26_while_lstm_cell_26_matmul_1_readvariableop_resource_0* 
_output_shapes
:
ВИ*
dtype02B
@sequential_13/lstm_26/while/lstm_cell_26/MatMul_1/ReadVariableOpШ
1sequential_13/lstm_26/while/lstm_cell_26/MatMul_1MatMul)sequential_13_lstm_26_while_placeholder_2Hsequential_13/lstm_26/while/lstm_cell_26/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         И23
1sequential_13/lstm_26/while/lstm_cell_26/MatMul_1Р
,sequential_13/lstm_26/while/lstm_cell_26/addAddV29sequential_13/lstm_26/while/lstm_cell_26/MatMul:product:0;sequential_13/lstm_26/while/lstm_cell_26/MatMul_1:product:0*
T0*(
_output_shapes
:         И2.
,sequential_13/lstm_26/while/lstm_cell_26/addК
?sequential_13/lstm_26/while/lstm_cell_26/BiasAdd/ReadVariableOpReadVariableOpJsequential_13_lstm_26_while_lstm_cell_26_biasadd_readvariableop_resource_0*
_output_shapes	
:И*
dtype02A
?sequential_13/lstm_26/while/lstm_cell_26/BiasAdd/ReadVariableOpЭ
0sequential_13/lstm_26/while/lstm_cell_26/BiasAddBiasAdd0sequential_13/lstm_26/while/lstm_cell_26/add:z:0Gsequential_13/lstm_26/while/lstm_cell_26/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         И22
0sequential_13/lstm_26/while/lstm_cell_26/BiasAdd╢
8sequential_13/lstm_26/while/lstm_cell_26/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2:
8sequential_13/lstm_26/while/lstm_cell_26/split/split_dimч
.sequential_13/lstm_26/while/lstm_cell_26/splitSplitAsequential_13/lstm_26/while/lstm_cell_26/split/split_dim:output:09sequential_13/lstm_26/while/lstm_cell_26/BiasAdd:output:0*
T0*d
_output_shapesR
P:         В:         В:         В:         В*
	num_split20
.sequential_13/lstm_26/while/lstm_cell_26/split█
0sequential_13/lstm_26/while/lstm_cell_26/SigmoidSigmoid7sequential_13/lstm_26/while/lstm_cell_26/split:output:0*
T0*(
_output_shapes
:         В22
0sequential_13/lstm_26/while/lstm_cell_26/Sigmoid▀
2sequential_13/lstm_26/while/lstm_cell_26/Sigmoid_1Sigmoid7sequential_13/lstm_26/while/lstm_cell_26/split:output:1*
T0*(
_output_shapes
:         В24
2sequential_13/lstm_26/while/lstm_cell_26/Sigmoid_1∙
,sequential_13/lstm_26/while/lstm_cell_26/mulMul6sequential_13/lstm_26/while/lstm_cell_26/Sigmoid_1:y:0)sequential_13_lstm_26_while_placeholder_3*
T0*(
_output_shapes
:         В2.
,sequential_13/lstm_26/while/lstm_cell_26/mul╥
-sequential_13/lstm_26/while/lstm_cell_26/ReluRelu7sequential_13/lstm_26/while/lstm_cell_26/split:output:2*
T0*(
_output_shapes
:         В2/
-sequential_13/lstm_26/while/lstm_cell_26/ReluН
.sequential_13/lstm_26/while/lstm_cell_26/mul_1Mul4sequential_13/lstm_26/while/lstm_cell_26/Sigmoid:y:0;sequential_13/lstm_26/while/lstm_cell_26/Relu:activations:0*
T0*(
_output_shapes
:         В20
.sequential_13/lstm_26/while/lstm_cell_26/mul_1В
.sequential_13/lstm_26/while/lstm_cell_26/add_1AddV20sequential_13/lstm_26/while/lstm_cell_26/mul:z:02sequential_13/lstm_26/while/lstm_cell_26/mul_1:z:0*
T0*(
_output_shapes
:         В20
.sequential_13/lstm_26/while/lstm_cell_26/add_1▀
2sequential_13/lstm_26/while/lstm_cell_26/Sigmoid_2Sigmoid7sequential_13/lstm_26/while/lstm_cell_26/split:output:3*
T0*(
_output_shapes
:         В24
2sequential_13/lstm_26/while/lstm_cell_26/Sigmoid_2╤
/sequential_13/lstm_26/while/lstm_cell_26/Relu_1Relu2sequential_13/lstm_26/while/lstm_cell_26/add_1:z:0*
T0*(
_output_shapes
:         В21
/sequential_13/lstm_26/while/lstm_cell_26/Relu_1С
.sequential_13/lstm_26/while/lstm_cell_26/mul_2Mul6sequential_13/lstm_26/while/lstm_cell_26/Sigmoid_2:y:0=sequential_13/lstm_26/while/lstm_cell_26/Relu_1:activations:0*
T0*(
_output_shapes
:         В20
.sequential_13/lstm_26/while/lstm_cell_26/mul_2╬
@sequential_13/lstm_26/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem)sequential_13_lstm_26_while_placeholder_1'sequential_13_lstm_26_while_placeholder2sequential_13/lstm_26/while/lstm_cell_26/mul_2:z:0*
_output_shapes
: *
element_dtype02B
@sequential_13/lstm_26/while/TensorArrayV2Write/TensorListSetItemИ
!sequential_13/lstm_26/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2#
!sequential_13/lstm_26/while/add/y┴
sequential_13/lstm_26/while/addAddV2'sequential_13_lstm_26_while_placeholder*sequential_13/lstm_26/while/add/y:output:0*
T0*
_output_shapes
: 2!
sequential_13/lstm_26/while/addМ
#sequential_13/lstm_26/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2%
#sequential_13/lstm_26/while/add_1/yф
!sequential_13/lstm_26/while/add_1AddV2Dsequential_13_lstm_26_while_sequential_13_lstm_26_while_loop_counter,sequential_13/lstm_26/while/add_1/y:output:0*
T0*
_output_shapes
: 2#
!sequential_13/lstm_26/while/add_1├
$sequential_13/lstm_26/while/IdentityIdentity%sequential_13/lstm_26/while/add_1:z:0!^sequential_13/lstm_26/while/NoOp*
T0*
_output_shapes
: 2&
$sequential_13/lstm_26/while/Identityь
&sequential_13/lstm_26/while/Identity_1IdentityJsequential_13_lstm_26_while_sequential_13_lstm_26_while_maximum_iterations!^sequential_13/lstm_26/while/NoOp*
T0*
_output_shapes
: 2(
&sequential_13/lstm_26/while/Identity_1┼
&sequential_13/lstm_26/while/Identity_2Identity#sequential_13/lstm_26/while/add:z:0!^sequential_13/lstm_26/while/NoOp*
T0*
_output_shapes
: 2(
&sequential_13/lstm_26/while/Identity_2Є
&sequential_13/lstm_26/while/Identity_3IdentityPsequential_13/lstm_26/while/TensorArrayV2Write/TensorListSetItem:output_handle:0!^sequential_13/lstm_26/while/NoOp*
T0*
_output_shapes
: 2(
&sequential_13/lstm_26/while/Identity_3ц
&sequential_13/lstm_26/while/Identity_4Identity2sequential_13/lstm_26/while/lstm_cell_26/mul_2:z:0!^sequential_13/lstm_26/while/NoOp*
T0*(
_output_shapes
:         В2(
&sequential_13/lstm_26/while/Identity_4ц
&sequential_13/lstm_26/while/Identity_5Identity2sequential_13/lstm_26/while/lstm_cell_26/add_1:z:0!^sequential_13/lstm_26/while/NoOp*
T0*(
_output_shapes
:         В2(
&sequential_13/lstm_26/while/Identity_5╠
 sequential_13/lstm_26/while/NoOpNoOp@^sequential_13/lstm_26/while/lstm_cell_26/BiasAdd/ReadVariableOp?^sequential_13/lstm_26/while/lstm_cell_26/MatMul/ReadVariableOpA^sequential_13/lstm_26/while/lstm_cell_26/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2"
 sequential_13/lstm_26/while/NoOp"U
$sequential_13_lstm_26_while_identity-sequential_13/lstm_26/while/Identity:output:0"Y
&sequential_13_lstm_26_while_identity_1/sequential_13/lstm_26/while/Identity_1:output:0"Y
&sequential_13_lstm_26_while_identity_2/sequential_13/lstm_26/while/Identity_2:output:0"Y
&sequential_13_lstm_26_while_identity_3/sequential_13/lstm_26/while/Identity_3:output:0"Y
&sequential_13_lstm_26_while_identity_4/sequential_13/lstm_26/while/Identity_4:output:0"Y
&sequential_13_lstm_26_while_identity_5/sequential_13/lstm_26/while/Identity_5:output:0"Ц
Hsequential_13_lstm_26_while_lstm_cell_26_biasadd_readvariableop_resourceJsequential_13_lstm_26_while_lstm_cell_26_biasadd_readvariableop_resource_0"Ш
Isequential_13_lstm_26_while_lstm_cell_26_matmul_1_readvariableop_resourceKsequential_13_lstm_26_while_lstm_cell_26_matmul_1_readvariableop_resource_0"Ф
Gsequential_13_lstm_26_while_lstm_cell_26_matmul_readvariableop_resourceIsequential_13_lstm_26_while_lstm_cell_26_matmul_readvariableop_resource_0"И
Asequential_13_lstm_26_while_sequential_13_lstm_26_strided_slice_1Csequential_13_lstm_26_while_sequential_13_lstm_26_strided_slice_1_0"А
}sequential_13_lstm_26_while_tensorarrayv2read_tensorlistgetitem_sequential_13_lstm_26_tensorarrayunstack_tensorlistfromtensorsequential_13_lstm_26_while_tensorarrayv2read_tensorlistgetitem_sequential_13_lstm_26_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :         В:         В: : : : : 2В
?sequential_13/lstm_26/while/lstm_cell_26/BiasAdd/ReadVariableOp?sequential_13/lstm_26/while/lstm_cell_26/BiasAdd/ReadVariableOp2А
>sequential_13/lstm_26/while/lstm_cell_26/MatMul/ReadVariableOp>sequential_13/lstm_26/while/lstm_cell_26/MatMul/ReadVariableOp2Д
@sequential_13/lstm_26/while/lstm_cell_26/MatMul_1/ReadVariableOp@sequential_13/lstm_26/while/lstm_cell_26/MatMul_1/ReadVariableOp: 
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
:         В:.*
(
_output_shapes
:         В:

_output_shapes
: :

_output_shapes
: 
у
═
while_cond_40074020
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_40074020___redundant_placeholder06
2while_while_cond_40074020___redundant_placeholder16
2while_while_cond_40074020___redundant_placeholder26
2while_while_cond_40074020___redundant_placeholder3
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
B: : : : :         В:         В: ::::: 
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
:         В:.*
(
_output_shapes
:         В:

_output_shapes
: :

_output_shapes
:
┤?
╓
while_body_40073556
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
3while_lstm_cell_27_matmul_readvariableop_resource_0:
ВЁI
5while_lstm_cell_27_matmul_1_readvariableop_resource_0:
ЬЁC
4while_lstm_cell_27_biasadd_readvariableop_resource_0:	Ё
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
1while_lstm_cell_27_matmul_readvariableop_resource:
ВЁG
3while_lstm_cell_27_matmul_1_readvariableop_resource:
ЬЁA
2while_lstm_cell_27_biasadd_readvariableop_resource:	ЁИв)while/lstm_cell_27/BiasAdd/ReadVariableOpв(while/lstm_cell_27/MatMul/ReadVariableOpв*while/lstm_cell_27/MatMul_1/ReadVariableOp├
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    В   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape╘
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:         В*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem╩
(while/lstm_cell_27/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_27_matmul_readvariableop_resource_0* 
_output_shapes
:
ВЁ*
dtype02*
(while/lstm_cell_27/MatMul/ReadVariableOp╫
while/lstm_cell_27/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_27/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ё2
while/lstm_cell_27/MatMul╨
*while/lstm_cell_27/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_27_matmul_1_readvariableop_resource_0* 
_output_shapes
:
ЬЁ*
dtype02,
*while/lstm_cell_27/MatMul_1/ReadVariableOp└
while/lstm_cell_27/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_27/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ё2
while/lstm_cell_27/MatMul_1╕
while/lstm_cell_27/addAddV2#while/lstm_cell_27/MatMul:product:0%while/lstm_cell_27/MatMul_1:product:0*
T0*(
_output_shapes
:         Ё2
while/lstm_cell_27/add╚
)while/lstm_cell_27/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_27_biasadd_readvariableop_resource_0*
_output_shapes	
:Ё*
dtype02+
)while/lstm_cell_27/BiasAdd/ReadVariableOp┼
while/lstm_cell_27/BiasAddBiasAddwhile/lstm_cell_27/add:z:01while/lstm_cell_27/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ё2
while/lstm_cell_27/BiasAddК
"while/lstm_cell_27/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_27/split/split_dimП
while/lstm_cell_27/splitSplit+while/lstm_cell_27/split/split_dim:output:0#while/lstm_cell_27/BiasAdd:output:0*
T0*d
_output_shapesR
P:         Ь:         Ь:         Ь:         Ь*
	num_split2
while/lstm_cell_27/splitЩ
while/lstm_cell_27/SigmoidSigmoid!while/lstm_cell_27/split:output:0*
T0*(
_output_shapes
:         Ь2
while/lstm_cell_27/SigmoidЭ
while/lstm_cell_27/Sigmoid_1Sigmoid!while/lstm_cell_27/split:output:1*
T0*(
_output_shapes
:         Ь2
while/lstm_cell_27/Sigmoid_1б
while/lstm_cell_27/mulMul while/lstm_cell_27/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:         Ь2
while/lstm_cell_27/mulР
while/lstm_cell_27/ReluRelu!while/lstm_cell_27/split:output:2*
T0*(
_output_shapes
:         Ь2
while/lstm_cell_27/Relu╡
while/lstm_cell_27/mul_1Mulwhile/lstm_cell_27/Sigmoid:y:0%while/lstm_cell_27/Relu:activations:0*
T0*(
_output_shapes
:         Ь2
while/lstm_cell_27/mul_1к
while/lstm_cell_27/add_1AddV2while/lstm_cell_27/mul:z:0while/lstm_cell_27/mul_1:z:0*
T0*(
_output_shapes
:         Ь2
while/lstm_cell_27/add_1Э
while/lstm_cell_27/Sigmoid_2Sigmoid!while/lstm_cell_27/split:output:3*
T0*(
_output_shapes
:         Ь2
while/lstm_cell_27/Sigmoid_2П
while/lstm_cell_27/Relu_1Reluwhile/lstm_cell_27/add_1:z:0*
T0*(
_output_shapes
:         Ь2
while/lstm_cell_27/Relu_1╣
while/lstm_cell_27/mul_2Mul while/lstm_cell_27/Sigmoid_2:y:0'while/lstm_cell_27/Relu_1:activations:0*
T0*(
_output_shapes
:         Ь2
while/lstm_cell_27/mul_2р
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_27/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_27/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:         Ь2
while/Identity_4О
while/Identity_5Identitywhile/lstm_cell_27/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:         Ь2
while/Identity_5▐

while/NoOpNoOp*^while/lstm_cell_27/BiasAdd/ReadVariableOp)^while/lstm_cell_27/MatMul/ReadVariableOp+^while/lstm_cell_27/MatMul_1/ReadVariableOp*"
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
2while_lstm_cell_27_biasadd_readvariableop_resource4while_lstm_cell_27_biasadd_readvariableop_resource_0"l
3while_lstm_cell_27_matmul_1_readvariableop_resource5while_lstm_cell_27_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_27_matmul_readvariableop_resource3while_lstm_cell_27_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :         Ь:         Ь: : : : : 2V
)while/lstm_cell_27/BiasAdd/ReadVariableOp)while/lstm_cell_27/BiasAdd/ReadVariableOp2T
(while/lstm_cell_27/MatMul/ReadVariableOp(while/lstm_cell_27/MatMul/ReadVariableOp2X
*while/lstm_cell_27/MatMul_1/ReadVariableOp*while/lstm_cell_27/MatMul_1/ReadVariableOp: 
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
:         Ь:.*
(
_output_shapes
:         Ь:

_output_shapes
: :

_output_shapes
: 
Й
f
H__inference_dropout_27_layer_call_and_return_conditional_losses_40076313

inputs

identity_1_
IdentityIdentityinputs*
T0*,
_output_shapes
:         Ь2

Identityn

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:         Ь2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         Ь:T P
,
_output_shapes
:         Ь
 
_user_specified_nameinputs
р
║
*__inference_lstm_26_layer_call_fn_40075600
inputs_0
unknown:	]И
	unknown_0:
ВИ
	unknown_1:	И
identityИвStatefulPartitionedCallТ
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:                  В*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_lstm_26_layer_call_and_return_conditional_losses_400722162
StatefulPartitionedCallЙ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:                  В2

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
├\
а
E__inference_lstm_26_layer_call_and_return_conditional_losses_40075136
inputs_0>
+lstm_cell_26_matmul_readvariableop_resource:	]ИA
-lstm_cell_26_matmul_1_readvariableop_resource:
ВИ;
,lstm_cell_26_biasadd_readvariableop_resource:	И
identityИв#lstm_cell_26/BiasAdd/ReadVariableOpв"lstm_cell_26/MatMul/ReadVariableOpв$lstm_cell_26/MatMul_1/ReadVariableOpвwhileF
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
B :В2
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
B :В2
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
:         В2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :В2
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
B :В2
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
:         В2	
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
"lstm_cell_26/MatMul/ReadVariableOpReadVariableOp+lstm_cell_26_matmul_readvariableop_resource*
_output_shapes
:	]И*
dtype02$
"lstm_cell_26/MatMul/ReadVariableOpн
lstm_cell_26/MatMulMatMulstrided_slice_2:output:0*lstm_cell_26/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         И2
lstm_cell_26/MatMul╝
$lstm_cell_26/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_26_matmul_1_readvariableop_resource* 
_output_shapes
:
ВИ*
dtype02&
$lstm_cell_26/MatMul_1/ReadVariableOpй
lstm_cell_26/MatMul_1MatMulzeros:output:0,lstm_cell_26/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         И2
lstm_cell_26/MatMul_1а
lstm_cell_26/addAddV2lstm_cell_26/MatMul:product:0lstm_cell_26/MatMul_1:product:0*
T0*(
_output_shapes
:         И2
lstm_cell_26/add┤
#lstm_cell_26/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_26_biasadd_readvariableop_resource*
_output_shapes	
:И*
dtype02%
#lstm_cell_26/BiasAdd/ReadVariableOpн
lstm_cell_26/BiasAddBiasAddlstm_cell_26/add:z:0+lstm_cell_26/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         И2
lstm_cell_26/BiasAdd~
lstm_cell_26/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_26/split/split_dimў
lstm_cell_26/splitSplit%lstm_cell_26/split/split_dim:output:0lstm_cell_26/BiasAdd:output:0*
T0*d
_output_shapesR
P:         В:         В:         В:         В*
	num_split2
lstm_cell_26/splitЗ
lstm_cell_26/SigmoidSigmoidlstm_cell_26/split:output:0*
T0*(
_output_shapes
:         В2
lstm_cell_26/SigmoidЛ
lstm_cell_26/Sigmoid_1Sigmoidlstm_cell_26/split:output:1*
T0*(
_output_shapes
:         В2
lstm_cell_26/Sigmoid_1М
lstm_cell_26/mulMullstm_cell_26/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:         В2
lstm_cell_26/mul~
lstm_cell_26/ReluRelulstm_cell_26/split:output:2*
T0*(
_output_shapes
:         В2
lstm_cell_26/ReluЭ
lstm_cell_26/mul_1Mullstm_cell_26/Sigmoid:y:0lstm_cell_26/Relu:activations:0*
T0*(
_output_shapes
:         В2
lstm_cell_26/mul_1Т
lstm_cell_26/add_1AddV2lstm_cell_26/mul:z:0lstm_cell_26/mul_1:z:0*
T0*(
_output_shapes
:         В2
lstm_cell_26/add_1Л
lstm_cell_26/Sigmoid_2Sigmoidlstm_cell_26/split:output:3*
T0*(
_output_shapes
:         В2
lstm_cell_26/Sigmoid_2}
lstm_cell_26/Relu_1Relulstm_cell_26/add_1:z:0*
T0*(
_output_shapes
:         В2
lstm_cell_26/Relu_1б
lstm_cell_26/mul_2Mullstm_cell_26/Sigmoid_2:y:0!lstm_cell_26/Relu_1:activations:0*
T0*(
_output_shapes
:         В2
lstm_cell_26/mul_2П
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    В   2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_26_matmul_readvariableop_resource-lstm_cell_26_matmul_1_readvariableop_resource,lstm_cell_26_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :         В:         В: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_40075052*
condR
while_cond_40075051*M
output_shapes<
:: : : : :         В:         В: : : : : *
parallel_iterations 2
while╡
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    В   22
0TensorArrayV2Stack/TensorListStack/element_shapeЄ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:                  В*
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
:         В*
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
!:                  В2
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
!:                  В2

Identity╚
NoOpNoOp$^lstm_cell_26/BiasAdd/ReadVariableOp#^lstm_cell_26/MatMul/ReadVariableOp%^lstm_cell_26/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  ]: : : 2J
#lstm_cell_26/BiasAdd/ReadVariableOp#lstm_cell_26/BiasAdd/ReadVariableOp2H
"lstm_cell_26/MatMul/ReadVariableOp"lstm_cell_26/MatMul/ReadVariableOp2L
$lstm_cell_26/MatMul_1/ReadVariableOp$lstm_cell_26/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :                  ]
"
_user_specified_name
inputs/0
╪
I
-__inference_dropout_26_layer_call_fn_40075655

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
:         В* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_dropout_26_layer_call_and_return_conditional_losses_400734882
PartitionedCallq
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:         В2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         В:T P
,
_output_shapes
:         В
 
_user_specified_nameinputs
№	
╩
&__inference_signature_wrapper_40074275
lstm_26_input
unknown:	]И
	unknown_0:
ВИ
	unknown_1:	И
	unknown_2:
ВЁ
	unknown_3:
ЬЁ
	unknown_4:	Ё
	unknown_5:	Ь
	unknown_6:
identityИвStatefulPartitionedCallм
StatefulPartitionedCallStatefulPartitionedCalllstm_26_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
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
#__inference__wrapped_model_400720582
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
_user_specified_namelstm_26_input
у
═
while_cond_40075504
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_40075504___redundant_placeholder06
2while_while_cond_40075504___redundant_placeholder16
2while_while_cond_40075504___redundant_placeholder26
2while_while_cond_40075504___redundant_placeholder3
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
B: : : : :         В:         В: ::::: 
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
:         В:.*
(
_output_shapes
:         В:

_output_shapes
: :

_output_shapes
:
╫
g
H__inference_dropout_26_layer_call_and_return_conditional_losses_40073938

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
:         В2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape╣
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:         В*
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
:         В2
dropout/GreaterEqualД
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:         В2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:         В2
dropout/Mul_1j
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:         В2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         В:T P
,
_output_shapes
:         В
 
_user_specified_nameinputs
╜
°
K__inference_sequential_13_layer_call_and_return_conditional_losses_40074227
lstm_26_input#
lstm_26_40074205:	]И$
lstm_26_40074207:
ВИ
lstm_26_40074209:	И$
lstm_27_40074213:
ВЁ$
lstm_27_40074215:
ЬЁ
lstm_27_40074217:	Ё$
dense_13_40074221:	Ь
dense_13_40074223:
identityИв dense_13/StatefulPartitionedCallвlstm_26/StatefulPartitionedCallвlstm_27/StatefulPartitionedCall╡
lstm_26/StatefulPartitionedCallStatefulPartitionedCalllstm_26_inputlstm_26_40074205lstm_26_40074207lstm_26_40074209*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         В*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_lstm_26_layer_call_and_return_conditional_losses_400734752!
lstm_26/StatefulPartitionedCallГ
dropout_26/PartitionedCallPartitionedCall(lstm_26/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         В* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_dropout_26_layer_call_and_return_conditional_losses_400734882
dropout_26/PartitionedCall╦
lstm_27/StatefulPartitionedCallStatefulPartitionedCall#dropout_26/PartitionedCall:output:0lstm_27_40074213lstm_27_40074215lstm_27_40074217*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         Ь*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_lstm_27_layer_call_and_return_conditional_losses_400736402!
lstm_27/StatefulPartitionedCallГ
dropout_27/PartitionedCallPartitionedCall(lstm_27/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         Ь* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_dropout_27_layer_call_and_return_conditional_losses_400736532
dropout_27/PartitionedCall╗
 dense_13/StatefulPartitionedCallStatefulPartitionedCall#dropout_27/PartitionedCall:output:0dense_13_40074221dense_13_40074223*
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
GPU 2J 8В *O
fJRH
F__inference_dense_13_layer_call_and_return_conditional_losses_400736862"
 dense_13/StatefulPartitionedCallИ
IdentityIdentity)dense_13/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:         2

Identity╡
NoOpNoOp!^dense_13/StatefulPartitionedCall ^lstm_26/StatefulPartitionedCall ^lstm_27/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':         ]: : : : : : : : 2D
 dense_13/StatefulPartitionedCall dense_13/StatefulPartitionedCall2B
lstm_26/StatefulPartitionedCalllstm_26/StatefulPartitionedCall2B
lstm_27/StatefulPartitionedCalllstm_27/StatefulPartitionedCall:Z V
+
_output_shapes
:         ]
'
_user_specified_namelstm_26_input
╟
∙
/__inference_lstm_cell_26_layer_call_fn_40076456

inputs
states_0
states_1
unknown:	]И
	unknown_0:
ВИ
	unknown_1:	И
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
<:         В:         В:         В*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_lstm_cell_26_layer_call_and_return_conditional_losses_400721332
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         В2

IdentityА

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:         В2

Identity_1А

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:         В2

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
A:         ]:         В:         В: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         ]
 
_user_specified_nameinputs:RN
(
_output_shapes
:         В
"
_user_specified_name
states/0:RN
(
_output_shapes
:         В
"
_user_specified_name
states/1
у
╗
*__inference_lstm_27_layer_call_fn_40076275
inputs_0
unknown:
ВЁ
	unknown_0:
ЬЁ
	unknown_1:	Ё
identityИвStatefulPartitionedCallТ
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:                  Ь*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_lstm_27_layer_call_and_return_conditional_losses_400728462
StatefulPartitionedCallЙ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:                  Ь2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':                  В: : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
5
_output_shapes#
!:                  В
"
_user_specified_name
inputs/0
Д\
Ю
E__inference_lstm_26_layer_call_and_return_conditional_losses_40075438

inputs>
+lstm_cell_26_matmul_readvariableop_resource:	]ИA
-lstm_cell_26_matmul_1_readvariableop_resource:
ВИ;
,lstm_cell_26_biasadd_readvariableop_resource:	И
identityИв#lstm_cell_26/BiasAdd/ReadVariableOpв"lstm_cell_26/MatMul/ReadVariableOpв$lstm_cell_26/MatMul_1/ReadVariableOpвwhileD
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
B :В2
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
B :В2
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
:         В2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :В2
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
B :В2
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
:         В2	
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
"lstm_cell_26/MatMul/ReadVariableOpReadVariableOp+lstm_cell_26_matmul_readvariableop_resource*
_output_shapes
:	]И*
dtype02$
"lstm_cell_26/MatMul/ReadVariableOpн
lstm_cell_26/MatMulMatMulstrided_slice_2:output:0*lstm_cell_26/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         И2
lstm_cell_26/MatMul╝
$lstm_cell_26/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_26_matmul_1_readvariableop_resource* 
_output_shapes
:
ВИ*
dtype02&
$lstm_cell_26/MatMul_1/ReadVariableOpй
lstm_cell_26/MatMul_1MatMulzeros:output:0,lstm_cell_26/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         И2
lstm_cell_26/MatMul_1а
lstm_cell_26/addAddV2lstm_cell_26/MatMul:product:0lstm_cell_26/MatMul_1:product:0*
T0*(
_output_shapes
:         И2
lstm_cell_26/add┤
#lstm_cell_26/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_26_biasadd_readvariableop_resource*
_output_shapes	
:И*
dtype02%
#lstm_cell_26/BiasAdd/ReadVariableOpн
lstm_cell_26/BiasAddBiasAddlstm_cell_26/add:z:0+lstm_cell_26/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         И2
lstm_cell_26/BiasAdd~
lstm_cell_26/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_26/split/split_dimў
lstm_cell_26/splitSplit%lstm_cell_26/split/split_dim:output:0lstm_cell_26/BiasAdd:output:0*
T0*d
_output_shapesR
P:         В:         В:         В:         В*
	num_split2
lstm_cell_26/splitЗ
lstm_cell_26/SigmoidSigmoidlstm_cell_26/split:output:0*
T0*(
_output_shapes
:         В2
lstm_cell_26/SigmoidЛ
lstm_cell_26/Sigmoid_1Sigmoidlstm_cell_26/split:output:1*
T0*(
_output_shapes
:         В2
lstm_cell_26/Sigmoid_1М
lstm_cell_26/mulMullstm_cell_26/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:         В2
lstm_cell_26/mul~
lstm_cell_26/ReluRelulstm_cell_26/split:output:2*
T0*(
_output_shapes
:         В2
lstm_cell_26/ReluЭ
lstm_cell_26/mul_1Mullstm_cell_26/Sigmoid:y:0lstm_cell_26/Relu:activations:0*
T0*(
_output_shapes
:         В2
lstm_cell_26/mul_1Т
lstm_cell_26/add_1AddV2lstm_cell_26/mul:z:0lstm_cell_26/mul_1:z:0*
T0*(
_output_shapes
:         В2
lstm_cell_26/add_1Л
lstm_cell_26/Sigmoid_2Sigmoidlstm_cell_26/split:output:3*
T0*(
_output_shapes
:         В2
lstm_cell_26/Sigmoid_2}
lstm_cell_26/Relu_1Relulstm_cell_26/add_1:z:0*
T0*(
_output_shapes
:         В2
lstm_cell_26/Relu_1б
lstm_cell_26/mul_2Mullstm_cell_26/Sigmoid_2:y:0!lstm_cell_26/Relu_1:activations:0*
T0*(
_output_shapes
:         В2
lstm_cell_26/mul_2П
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    В   2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_26_matmul_readvariableop_resource-lstm_cell_26_matmul_1_readvariableop_resource,lstm_cell_26_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :         В:         В: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_40075354*
condR
while_cond_40075353*M
output_shapes<
:: : : : :         В:         В: : : : : *
parallel_iterations 2
while╡
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    В   22
0TensorArrayV2Stack/TensorListStack/element_shapeщ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:         В*
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
:         В*
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
:         В2
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
:         В2

Identity╚
NoOpNoOp$^lstm_cell_26/BiasAdd/ReadVariableOp#^lstm_cell_26/MatMul/ReadVariableOp%^lstm_cell_26/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         ]: : : 2J
#lstm_cell_26/BiasAdd/ReadVariableOp#lstm_cell_26/BiasAdd/ReadVariableOp2H
"lstm_cell_26/MatMul/ReadVariableOp"lstm_cell_26/MatMul/ReadVariableOp2L
$lstm_cell_26/MatMul_1/ReadVariableOp$lstm_cell_26/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:         ]
 
_user_specified_nameinputs
К\
Я
E__inference_lstm_27_layer_call_and_return_conditional_losses_40073640

inputs?
+lstm_cell_27_matmul_readvariableop_resource:
ВЁA
-lstm_cell_27_matmul_1_readvariableop_resource:
ЬЁ;
,lstm_cell_27_biasadd_readvariableop_resource:	Ё
identityИв#lstm_cell_27/BiasAdd/ReadVariableOpв"lstm_cell_27/MatMul/ReadVariableOpв$lstm_cell_27/MatMul_1/ReadVariableOpвwhileD
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
B :Ь2
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
B :Ь2
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
:         Ь2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :Ь2
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
B :Ь2
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
:         Ь2	
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
:         В2
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
valueB"    В   27
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
:         В*
shrink_axis_mask2
strided_slice_2╢
"lstm_cell_27/MatMul/ReadVariableOpReadVariableOp+lstm_cell_27_matmul_readvariableop_resource* 
_output_shapes
:
ВЁ*
dtype02$
"lstm_cell_27/MatMul/ReadVariableOpн
lstm_cell_27/MatMulMatMulstrided_slice_2:output:0*lstm_cell_27/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ё2
lstm_cell_27/MatMul╝
$lstm_cell_27/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_27_matmul_1_readvariableop_resource* 
_output_shapes
:
ЬЁ*
dtype02&
$lstm_cell_27/MatMul_1/ReadVariableOpй
lstm_cell_27/MatMul_1MatMulzeros:output:0,lstm_cell_27/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ё2
lstm_cell_27/MatMul_1а
lstm_cell_27/addAddV2lstm_cell_27/MatMul:product:0lstm_cell_27/MatMul_1:product:0*
T0*(
_output_shapes
:         Ё2
lstm_cell_27/add┤
#lstm_cell_27/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_27_biasadd_readvariableop_resource*
_output_shapes	
:Ё*
dtype02%
#lstm_cell_27/BiasAdd/ReadVariableOpн
lstm_cell_27/BiasAddBiasAddlstm_cell_27/add:z:0+lstm_cell_27/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ё2
lstm_cell_27/BiasAdd~
lstm_cell_27/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_27/split/split_dimў
lstm_cell_27/splitSplit%lstm_cell_27/split/split_dim:output:0lstm_cell_27/BiasAdd:output:0*
T0*d
_output_shapesR
P:         Ь:         Ь:         Ь:         Ь*
	num_split2
lstm_cell_27/splitЗ
lstm_cell_27/SigmoidSigmoidlstm_cell_27/split:output:0*
T0*(
_output_shapes
:         Ь2
lstm_cell_27/SigmoidЛ
lstm_cell_27/Sigmoid_1Sigmoidlstm_cell_27/split:output:1*
T0*(
_output_shapes
:         Ь2
lstm_cell_27/Sigmoid_1М
lstm_cell_27/mulMullstm_cell_27/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:         Ь2
lstm_cell_27/mul~
lstm_cell_27/ReluRelulstm_cell_27/split:output:2*
T0*(
_output_shapes
:         Ь2
lstm_cell_27/ReluЭ
lstm_cell_27/mul_1Mullstm_cell_27/Sigmoid:y:0lstm_cell_27/Relu:activations:0*
T0*(
_output_shapes
:         Ь2
lstm_cell_27/mul_1Т
lstm_cell_27/add_1AddV2lstm_cell_27/mul:z:0lstm_cell_27/mul_1:z:0*
T0*(
_output_shapes
:         Ь2
lstm_cell_27/add_1Л
lstm_cell_27/Sigmoid_2Sigmoidlstm_cell_27/split:output:3*
T0*(
_output_shapes
:         Ь2
lstm_cell_27/Sigmoid_2}
lstm_cell_27/Relu_1Relulstm_cell_27/add_1:z:0*
T0*(
_output_shapes
:         Ь2
lstm_cell_27/Relu_1б
lstm_cell_27/mul_2Mullstm_cell_27/Sigmoid_2:y:0!lstm_cell_27/Relu_1:activations:0*
T0*(
_output_shapes
:         Ь2
lstm_cell_27/mul_2П
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_27_matmul_readvariableop_resource-lstm_cell_27_matmul_1_readvariableop_resource,lstm_cell_27_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :         Ь:         Ь: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_40073556*
condR
while_cond_40073555*M
output_shapes<
:: : : : :         Ь:         Ь: : : : : *
parallel_iterations 2
while╡
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      22
0TensorArrayV2Stack/TensorListStack/element_shapeщ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:         Ь*
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
:         Ь*
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
:         Ь2
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
:         Ь2

Identity╚
NoOpNoOp$^lstm_cell_27/BiasAdd/ReadVariableOp#^lstm_cell_27/MatMul/ReadVariableOp%^lstm_cell_27/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:         В: : : 2J
#lstm_cell_27/BiasAdd/ReadVariableOp#lstm_cell_27/BiasAdd/ReadVariableOp2H
"lstm_cell_27/MatMul/ReadVariableOp"lstm_cell_27/MatMul/ReadVariableOp2L
$lstm_cell_27/MatMul_1/ReadVariableOp$lstm_cell_27/MatMul_1/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:         В
 
_user_specified_nameinputs
Й
f
H__inference_dropout_26_layer_call_and_return_conditional_losses_40075638

inputs

identity_1_
IdentityIdentityinputs*
T0*,
_output_shapes
:         В2

Identityn

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:         В2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         В:T P
,
_output_shapes
:         В
 
_user_specified_nameinputs
╘

э
lstm_26_while_cond_40074341,
(lstm_26_while_lstm_26_while_loop_counter2
.lstm_26_while_lstm_26_while_maximum_iterations
lstm_26_while_placeholder
lstm_26_while_placeholder_1
lstm_26_while_placeholder_2
lstm_26_while_placeholder_3.
*lstm_26_while_less_lstm_26_strided_slice_1F
Blstm_26_while_lstm_26_while_cond_40074341___redundant_placeholder0F
Blstm_26_while_lstm_26_while_cond_40074341___redundant_placeholder1F
Blstm_26_while_lstm_26_while_cond_40074341___redundant_placeholder2F
Blstm_26_while_lstm_26_while_cond_40074341___redundant_placeholder3
lstm_26_while_identity
Ш
lstm_26/while/LessLesslstm_26_while_placeholder*lstm_26_while_less_lstm_26_strided_slice_1*
T0*
_output_shapes
: 2
lstm_26/while/Lessu
lstm_26/while/IdentityIdentitylstm_26/while/Less:z:0*
T0
*
_output_shapes
: 2
lstm_26/while/Identity"9
lstm_26_while_identitylstm_26/while/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :         В:         В: ::::: 
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
:         В:.*
(
_output_shapes
:         В:

_output_shapes
: :

_output_shapes
:
░?
╘
while_body_40073391
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_26_matmul_readvariableop_resource_0:	]ИI
5while_lstm_cell_26_matmul_1_readvariableop_resource_0:
ВИC
4while_lstm_cell_26_biasadd_readvariableop_resource_0:	И
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_26_matmul_readvariableop_resource:	]ИG
3while_lstm_cell_26_matmul_1_readvariableop_resource:
ВИA
2while_lstm_cell_26_biasadd_readvariableop_resource:	ИИв)while/lstm_cell_26/BiasAdd/ReadVariableOpв(while/lstm_cell_26/MatMul/ReadVariableOpв*while/lstm_cell_26/MatMul_1/ReadVariableOp├
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
(while/lstm_cell_26/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_26_matmul_readvariableop_resource_0*
_output_shapes
:	]И*
dtype02*
(while/lstm_cell_26/MatMul/ReadVariableOp╫
while/lstm_cell_26/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_26/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         И2
while/lstm_cell_26/MatMul╨
*while/lstm_cell_26/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_26_matmul_1_readvariableop_resource_0* 
_output_shapes
:
ВИ*
dtype02,
*while/lstm_cell_26/MatMul_1/ReadVariableOp└
while/lstm_cell_26/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_26/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         И2
while/lstm_cell_26/MatMul_1╕
while/lstm_cell_26/addAddV2#while/lstm_cell_26/MatMul:product:0%while/lstm_cell_26/MatMul_1:product:0*
T0*(
_output_shapes
:         И2
while/lstm_cell_26/add╚
)while/lstm_cell_26/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_26_biasadd_readvariableop_resource_0*
_output_shapes	
:И*
dtype02+
)while/lstm_cell_26/BiasAdd/ReadVariableOp┼
while/lstm_cell_26/BiasAddBiasAddwhile/lstm_cell_26/add:z:01while/lstm_cell_26/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         И2
while/lstm_cell_26/BiasAddК
"while/lstm_cell_26/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_26/split/split_dimП
while/lstm_cell_26/splitSplit+while/lstm_cell_26/split/split_dim:output:0#while/lstm_cell_26/BiasAdd:output:0*
T0*d
_output_shapesR
P:         В:         В:         В:         В*
	num_split2
while/lstm_cell_26/splitЩ
while/lstm_cell_26/SigmoidSigmoid!while/lstm_cell_26/split:output:0*
T0*(
_output_shapes
:         В2
while/lstm_cell_26/SigmoidЭ
while/lstm_cell_26/Sigmoid_1Sigmoid!while/lstm_cell_26/split:output:1*
T0*(
_output_shapes
:         В2
while/lstm_cell_26/Sigmoid_1б
while/lstm_cell_26/mulMul while/lstm_cell_26/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:         В2
while/lstm_cell_26/mulР
while/lstm_cell_26/ReluRelu!while/lstm_cell_26/split:output:2*
T0*(
_output_shapes
:         В2
while/lstm_cell_26/Relu╡
while/lstm_cell_26/mul_1Mulwhile/lstm_cell_26/Sigmoid:y:0%while/lstm_cell_26/Relu:activations:0*
T0*(
_output_shapes
:         В2
while/lstm_cell_26/mul_1к
while/lstm_cell_26/add_1AddV2while/lstm_cell_26/mul:z:0while/lstm_cell_26/mul_1:z:0*
T0*(
_output_shapes
:         В2
while/lstm_cell_26/add_1Э
while/lstm_cell_26/Sigmoid_2Sigmoid!while/lstm_cell_26/split:output:3*
T0*(
_output_shapes
:         В2
while/lstm_cell_26/Sigmoid_2П
while/lstm_cell_26/Relu_1Reluwhile/lstm_cell_26/add_1:z:0*
T0*(
_output_shapes
:         В2
while/lstm_cell_26/Relu_1╣
while/lstm_cell_26/mul_2Mul while/lstm_cell_26/Sigmoid_2:y:0'while/lstm_cell_26/Relu_1:activations:0*
T0*(
_output_shapes
:         В2
while/lstm_cell_26/mul_2р
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_26/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_26/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:         В2
while/Identity_4О
while/Identity_5Identitywhile/lstm_cell_26/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:         В2
while/Identity_5▐

while/NoOpNoOp*^while/lstm_cell_26/BiasAdd/ReadVariableOp)^while/lstm_cell_26/MatMul/ReadVariableOp+^while/lstm_cell_26/MatMul_1/ReadVariableOp*"
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
2while_lstm_cell_26_biasadd_readvariableop_resource4while_lstm_cell_26_biasadd_readvariableop_resource_0"l
3while_lstm_cell_26_matmul_1_readvariableop_resource5while_lstm_cell_26_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_26_matmul_readvariableop_resource3while_lstm_cell_26_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :         В:         В: : : : : 2V
)while/lstm_cell_26/BiasAdd/ReadVariableOp)while/lstm_cell_26/BiasAdd/ReadVariableOp2T
(while/lstm_cell_26/MatMul/ReadVariableOp(while/lstm_cell_26/MatMul/ReadVariableOp2X
*while/lstm_cell_26/MatMul_1/ReadVariableOp*while/lstm_cell_26/MatMul_1/ReadVariableOp: 
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
:         В:.*
(
_output_shapes
:         В:

_output_shapes
: :

_output_shapes
: 
┤?
╓
while_body_40075727
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
3while_lstm_cell_27_matmul_readvariableop_resource_0:
ВЁI
5while_lstm_cell_27_matmul_1_readvariableop_resource_0:
ЬЁC
4while_lstm_cell_27_biasadd_readvariableop_resource_0:	Ё
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
1while_lstm_cell_27_matmul_readvariableop_resource:
ВЁG
3while_lstm_cell_27_matmul_1_readvariableop_resource:
ЬЁA
2while_lstm_cell_27_biasadd_readvariableop_resource:	ЁИв)while/lstm_cell_27/BiasAdd/ReadVariableOpв(while/lstm_cell_27/MatMul/ReadVariableOpв*while/lstm_cell_27/MatMul_1/ReadVariableOp├
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    В   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape╘
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:         В*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem╩
(while/lstm_cell_27/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_27_matmul_readvariableop_resource_0* 
_output_shapes
:
ВЁ*
dtype02*
(while/lstm_cell_27/MatMul/ReadVariableOp╫
while/lstm_cell_27/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_27/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ё2
while/lstm_cell_27/MatMul╨
*while/lstm_cell_27/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_27_matmul_1_readvariableop_resource_0* 
_output_shapes
:
ЬЁ*
dtype02,
*while/lstm_cell_27/MatMul_1/ReadVariableOp└
while/lstm_cell_27/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_27/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ё2
while/lstm_cell_27/MatMul_1╕
while/lstm_cell_27/addAddV2#while/lstm_cell_27/MatMul:product:0%while/lstm_cell_27/MatMul_1:product:0*
T0*(
_output_shapes
:         Ё2
while/lstm_cell_27/add╚
)while/lstm_cell_27/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_27_biasadd_readvariableop_resource_0*
_output_shapes	
:Ё*
dtype02+
)while/lstm_cell_27/BiasAdd/ReadVariableOp┼
while/lstm_cell_27/BiasAddBiasAddwhile/lstm_cell_27/add:z:01while/lstm_cell_27/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ё2
while/lstm_cell_27/BiasAddК
"while/lstm_cell_27/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_27/split/split_dimП
while/lstm_cell_27/splitSplit+while/lstm_cell_27/split/split_dim:output:0#while/lstm_cell_27/BiasAdd:output:0*
T0*d
_output_shapesR
P:         Ь:         Ь:         Ь:         Ь*
	num_split2
while/lstm_cell_27/splitЩ
while/lstm_cell_27/SigmoidSigmoid!while/lstm_cell_27/split:output:0*
T0*(
_output_shapes
:         Ь2
while/lstm_cell_27/SigmoidЭ
while/lstm_cell_27/Sigmoid_1Sigmoid!while/lstm_cell_27/split:output:1*
T0*(
_output_shapes
:         Ь2
while/lstm_cell_27/Sigmoid_1б
while/lstm_cell_27/mulMul while/lstm_cell_27/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:         Ь2
while/lstm_cell_27/mulР
while/lstm_cell_27/ReluRelu!while/lstm_cell_27/split:output:2*
T0*(
_output_shapes
:         Ь2
while/lstm_cell_27/Relu╡
while/lstm_cell_27/mul_1Mulwhile/lstm_cell_27/Sigmoid:y:0%while/lstm_cell_27/Relu:activations:0*
T0*(
_output_shapes
:         Ь2
while/lstm_cell_27/mul_1к
while/lstm_cell_27/add_1AddV2while/lstm_cell_27/mul:z:0while/lstm_cell_27/mul_1:z:0*
T0*(
_output_shapes
:         Ь2
while/lstm_cell_27/add_1Э
while/lstm_cell_27/Sigmoid_2Sigmoid!while/lstm_cell_27/split:output:3*
T0*(
_output_shapes
:         Ь2
while/lstm_cell_27/Sigmoid_2П
while/lstm_cell_27/Relu_1Reluwhile/lstm_cell_27/add_1:z:0*
T0*(
_output_shapes
:         Ь2
while/lstm_cell_27/Relu_1╣
while/lstm_cell_27/mul_2Mul while/lstm_cell_27/Sigmoid_2:y:0'while/lstm_cell_27/Relu_1:activations:0*
T0*(
_output_shapes
:         Ь2
while/lstm_cell_27/mul_2р
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_27/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_27/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:         Ь2
while/Identity_4О
while/Identity_5Identitywhile/lstm_cell_27/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:         Ь2
while/Identity_5▐

while/NoOpNoOp*^while/lstm_cell_27/BiasAdd/ReadVariableOp)^while/lstm_cell_27/MatMul/ReadVariableOp+^while/lstm_cell_27/MatMul_1/ReadVariableOp*"
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
2while_lstm_cell_27_biasadd_readvariableop_resource4while_lstm_cell_27_biasadd_readvariableop_resource_0"l
3while_lstm_cell_27_matmul_1_readvariableop_resource5while_lstm_cell_27_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_27_matmul_readvariableop_resource3while_lstm_cell_27_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :         Ь:         Ь: : : : : 2V
)while/lstm_cell_27/BiasAdd/ReadVariableOp)while/lstm_cell_27/BiasAdd/ReadVariableOp2T
(while/lstm_cell_27/MatMul/ReadVariableOp(while/lstm_cell_27/MatMul/ReadVariableOp2X
*while/lstm_cell_27/MatMul_1/ReadVariableOp*while/lstm_cell_27/MatMul_1/ReadVariableOp: 
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
:         Ь:.*
(
_output_shapes
:         Ь:

_output_shapes
: :

_output_shapes
: 
╦F
О
E__inference_lstm_26_layer_call_and_return_conditional_losses_40072426

inputs(
lstm_cell_26_40072344:	]И)
lstm_cell_26_40072346:
ВИ$
lstm_cell_26_40072348:	И
identityИв$lstm_cell_26/StatefulPartitionedCallвwhileD
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
B :В2
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
B :В2
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
:         В2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :В2
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
B :В2
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
:         В2	
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
$lstm_cell_26/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_26_40072344lstm_cell_26_40072346lstm_cell_26_40072348*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:         В:         В:         В*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_lstm_cell_26_layer_call_and_return_conditional_losses_400722792&
$lstm_cell_26/StatefulPartitionedCallП
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    В   2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_26_40072344lstm_cell_26_40072346lstm_cell_26_40072348*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :         В:         В: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_40072357*
condR
while_cond_40072356*M
output_shapes<
:: : : : :         В:         В: : : : : *
parallel_iterations 2
while╡
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    В   22
0TensorArrayV2Stack/TensorListStack/element_shapeЄ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:                  В*
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
:         В*
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
!:                  В2
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
!:                  В2

Identity}
NoOpNoOp%^lstm_cell_26/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  ]: : : 2L
$lstm_cell_26/StatefulPartitionedCall$lstm_cell_26/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :                  ]
 
_user_specified_nameinputs
Щ

═
0__inference_sequential_13_layer_call_fn_40074964

inputs
unknown:	]И
	unknown_0:
ВИ
	unknown_1:	И
	unknown_2:
ВЁ
	unknown_3:
ЬЁ
	unknown_4:	Ё
	unknown_5:	Ь
	unknown_6:
identityИвStatefulPartitionedCall═
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
GPU 2J 8В *T
fORM
K__inference_sequential_13_layer_call_and_return_conditional_losses_400736932
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
У
Й
J__inference_lstm_cell_26_layer_call_and_return_conditional_losses_40076407

inputs
states_0
states_11
matmul_readvariableop_resource:	]И4
 matmul_1_readvariableop_resource:
ВИ.
biasadd_readvariableop_resource:	И
identity

identity_1

identity_2ИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpвMatMul_1/ReadVariableOpО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	]И*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         И2
MatMulХ
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
ВИ*
dtype02
MatMul_1/ReadVariableOp|
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         И2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:         И2
addН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:И*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         И2	
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
P:         В:         В:         В:         В*
	num_split2
split`
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:         В2	
Sigmoidd
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:         В2
	Sigmoid_1]
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:         В2
mulW
ReluRelusplit:output:2*
T0*(
_output_shapes
:         В2
Relui
mul_1MulSigmoid:y:0Relu:activations:0*
T0*(
_output_shapes
:         В2
mul_1^
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:         В2
add_1d
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:         В2
	Sigmoid_2V
Relu_1Relu	add_1:z:0*
T0*(
_output_shapes
:         В2
Relu_1m
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*(
_output_shapes
:         В2
mul_2e
IdentityIdentity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:         В2

Identityi

Identity_1Identity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:         В2

Identity_1i

Identity_2Identity	add_1:z:0^NoOp*
T0*(
_output_shapes
:         В2

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
A:         ]:         В:         В: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:         ]
 
_user_specified_nameinputs:RN
(
_output_shapes
:         В
"
_user_specified_name
states/0:RN
(
_output_shapes
:         В
"
_user_specified_name
states/1
╔\
б
E__inference_lstm_27_layer_call_and_return_conditional_losses_40075811
inputs_0?
+lstm_cell_27_matmul_readvariableop_resource:
ВЁA
-lstm_cell_27_matmul_1_readvariableop_resource:
ЬЁ;
,lstm_cell_27_biasadd_readvariableop_resource:	Ё
identityИв#lstm_cell_27/BiasAdd/ReadVariableOpв"lstm_cell_27/MatMul/ReadVariableOpв$lstm_cell_27/MatMul_1/ReadVariableOpвwhileF
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
B :Ь2
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
B :Ь2
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
:         Ь2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :Ь2
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
B :Ь2
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
:         Ь2	
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
!:                  В2
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
valueB"    В   27
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
:         В*
shrink_axis_mask2
strided_slice_2╢
"lstm_cell_27/MatMul/ReadVariableOpReadVariableOp+lstm_cell_27_matmul_readvariableop_resource* 
_output_shapes
:
ВЁ*
dtype02$
"lstm_cell_27/MatMul/ReadVariableOpн
lstm_cell_27/MatMulMatMulstrided_slice_2:output:0*lstm_cell_27/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ё2
lstm_cell_27/MatMul╝
$lstm_cell_27/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_27_matmul_1_readvariableop_resource* 
_output_shapes
:
ЬЁ*
dtype02&
$lstm_cell_27/MatMul_1/ReadVariableOpй
lstm_cell_27/MatMul_1MatMulzeros:output:0,lstm_cell_27/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ё2
lstm_cell_27/MatMul_1а
lstm_cell_27/addAddV2lstm_cell_27/MatMul:product:0lstm_cell_27/MatMul_1:product:0*
T0*(
_output_shapes
:         Ё2
lstm_cell_27/add┤
#lstm_cell_27/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_27_biasadd_readvariableop_resource*
_output_shapes	
:Ё*
dtype02%
#lstm_cell_27/BiasAdd/ReadVariableOpн
lstm_cell_27/BiasAddBiasAddlstm_cell_27/add:z:0+lstm_cell_27/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ё2
lstm_cell_27/BiasAdd~
lstm_cell_27/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_27/split/split_dimў
lstm_cell_27/splitSplit%lstm_cell_27/split/split_dim:output:0lstm_cell_27/BiasAdd:output:0*
T0*d
_output_shapesR
P:         Ь:         Ь:         Ь:         Ь*
	num_split2
lstm_cell_27/splitЗ
lstm_cell_27/SigmoidSigmoidlstm_cell_27/split:output:0*
T0*(
_output_shapes
:         Ь2
lstm_cell_27/SigmoidЛ
lstm_cell_27/Sigmoid_1Sigmoidlstm_cell_27/split:output:1*
T0*(
_output_shapes
:         Ь2
lstm_cell_27/Sigmoid_1М
lstm_cell_27/mulMullstm_cell_27/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:         Ь2
lstm_cell_27/mul~
lstm_cell_27/ReluRelulstm_cell_27/split:output:2*
T0*(
_output_shapes
:         Ь2
lstm_cell_27/ReluЭ
lstm_cell_27/mul_1Mullstm_cell_27/Sigmoid:y:0lstm_cell_27/Relu:activations:0*
T0*(
_output_shapes
:         Ь2
lstm_cell_27/mul_1Т
lstm_cell_27/add_1AddV2lstm_cell_27/mul:z:0lstm_cell_27/mul_1:z:0*
T0*(
_output_shapes
:         Ь2
lstm_cell_27/add_1Л
lstm_cell_27/Sigmoid_2Sigmoidlstm_cell_27/split:output:3*
T0*(
_output_shapes
:         Ь2
lstm_cell_27/Sigmoid_2}
lstm_cell_27/Relu_1Relulstm_cell_27/add_1:z:0*
T0*(
_output_shapes
:         Ь2
lstm_cell_27/Relu_1б
lstm_cell_27/mul_2Mullstm_cell_27/Sigmoid_2:y:0!lstm_cell_27/Relu_1:activations:0*
T0*(
_output_shapes
:         Ь2
lstm_cell_27/mul_2П
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_27_matmul_readvariableop_resource-lstm_cell_27_matmul_1_readvariableop_resource,lstm_cell_27_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :         Ь:         Ь: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_40075727*
condR
while_cond_40075726*M
output_shapes<
:: : : : :         Ь:         Ь: : : : : *
parallel_iterations 2
while╡
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      22
0TensorArrayV2Stack/TensorListStack/element_shapeЄ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:                  Ь*
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
:         Ь*
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
!:                  Ь2
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
!:                  Ь2

Identity╚
NoOpNoOp$^lstm_cell_27/BiasAdd/ReadVariableOp#^lstm_cell_27/MatMul/ReadVariableOp%^lstm_cell_27/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':                  В: : : 2J
#lstm_cell_27/BiasAdd/ReadVariableOp#lstm_cell_27/BiasAdd/ReadVariableOp2H
"lstm_cell_27/MatMul/ReadVariableOp"lstm_cell_27/MatMul/ReadVariableOp2L
$lstm_cell_27/MatMul_1/ReadVariableOp$lstm_cell_27/MatMul_1/ReadVariableOp2
whilewhile:_ [
5
_output_shapes#
!:                  В
"
_user_specified_name
inputs/0
у
═
while_cond_40076179
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_40076179___redundant_placeholder06
2while_while_cond_40076179___redundant_placeholder16
2while_while_cond_40076179___redundant_placeholder26
2while_while_cond_40076179___redundant_placeholder3
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
B: : : : :         Ь:         Ь: ::::: 
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
:         Ь:.*
(
_output_shapes
:         Ь:

_output_shapes
: :

_output_shapes
:
╫
g
H__inference_dropout_27_layer_call_and_return_conditional_losses_40073742

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
:         Ь2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape╣
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:         Ь*
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
:         Ь2
dropout/GreaterEqualД
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:         Ь2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:         Ь2
dropout/Mul_1j
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:         Ь2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         Ь:T P
,
_output_shapes
:         Ь
 
_user_specified_nameinputs
и╣
ч	
#__inference__wrapped_model_40072058
lstm_26_inputT
Asequential_13_lstm_26_lstm_cell_26_matmul_readvariableop_resource:	]ИW
Csequential_13_lstm_26_lstm_cell_26_matmul_1_readvariableop_resource:
ВИQ
Bsequential_13_lstm_26_lstm_cell_26_biasadd_readvariableop_resource:	ИU
Asequential_13_lstm_27_lstm_cell_27_matmul_readvariableop_resource:
ВЁW
Csequential_13_lstm_27_lstm_cell_27_matmul_1_readvariableop_resource:
ЬЁQ
Bsequential_13_lstm_27_lstm_cell_27_biasadd_readvariableop_resource:	ЁK
8sequential_13_dense_13_tensordot_readvariableop_resource:	ЬD
6sequential_13_dense_13_biasadd_readvariableop_resource:
identityИв-sequential_13/dense_13/BiasAdd/ReadVariableOpв/sequential_13/dense_13/Tensordot/ReadVariableOpв9sequential_13/lstm_26/lstm_cell_26/BiasAdd/ReadVariableOpв8sequential_13/lstm_26/lstm_cell_26/MatMul/ReadVariableOpв:sequential_13/lstm_26/lstm_cell_26/MatMul_1/ReadVariableOpвsequential_13/lstm_26/whileв9sequential_13/lstm_27/lstm_cell_27/BiasAdd/ReadVariableOpв8sequential_13/lstm_27/lstm_cell_27/MatMul/ReadVariableOpв:sequential_13/lstm_27/lstm_cell_27/MatMul_1/ReadVariableOpвsequential_13/lstm_27/whilew
sequential_13/lstm_26/ShapeShapelstm_26_input*
T0*
_output_shapes
:2
sequential_13/lstm_26/Shapeа
)sequential_13/lstm_26/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)sequential_13/lstm_26/strided_slice/stackд
+sequential_13/lstm_26/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential_13/lstm_26/strided_slice/stack_1д
+sequential_13/lstm_26/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential_13/lstm_26/strided_slice/stack_2ц
#sequential_13/lstm_26/strided_sliceStridedSlice$sequential_13/lstm_26/Shape:output:02sequential_13/lstm_26/strided_slice/stack:output:04sequential_13/lstm_26/strided_slice/stack_1:output:04sequential_13/lstm_26/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#sequential_13/lstm_26/strided_sliceЙ
!sequential_13/lstm_26/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :В2#
!sequential_13/lstm_26/zeros/mul/y─
sequential_13/lstm_26/zeros/mulMul,sequential_13/lstm_26/strided_slice:output:0*sequential_13/lstm_26/zeros/mul/y:output:0*
T0*
_output_shapes
: 2!
sequential_13/lstm_26/zeros/mulЛ
"sequential_13/lstm_26/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2$
"sequential_13/lstm_26/zeros/Less/y┐
 sequential_13/lstm_26/zeros/LessLess#sequential_13/lstm_26/zeros/mul:z:0+sequential_13/lstm_26/zeros/Less/y:output:0*
T0*
_output_shapes
: 2"
 sequential_13/lstm_26/zeros/LessП
$sequential_13/lstm_26/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :В2&
$sequential_13/lstm_26/zeros/packed/1█
"sequential_13/lstm_26/zeros/packedPack,sequential_13/lstm_26/strided_slice:output:0-sequential_13/lstm_26/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2$
"sequential_13/lstm_26/zeros/packedЛ
!sequential_13/lstm_26/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!sequential_13/lstm_26/zeros/Const╬
sequential_13/lstm_26/zerosFill+sequential_13/lstm_26/zeros/packed:output:0*sequential_13/lstm_26/zeros/Const:output:0*
T0*(
_output_shapes
:         В2
sequential_13/lstm_26/zerosН
#sequential_13/lstm_26/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :В2%
#sequential_13/lstm_26/zeros_1/mul/y╩
!sequential_13/lstm_26/zeros_1/mulMul,sequential_13/lstm_26/strided_slice:output:0,sequential_13/lstm_26/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2#
!sequential_13/lstm_26/zeros_1/mulП
$sequential_13/lstm_26/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2&
$sequential_13/lstm_26/zeros_1/Less/y╟
"sequential_13/lstm_26/zeros_1/LessLess%sequential_13/lstm_26/zeros_1/mul:z:0-sequential_13/lstm_26/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2$
"sequential_13/lstm_26/zeros_1/LessУ
&sequential_13/lstm_26/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :В2(
&sequential_13/lstm_26/zeros_1/packed/1с
$sequential_13/lstm_26/zeros_1/packedPack,sequential_13/lstm_26/strided_slice:output:0/sequential_13/lstm_26/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2&
$sequential_13/lstm_26/zeros_1/packedП
#sequential_13/lstm_26/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2%
#sequential_13/lstm_26/zeros_1/Const╓
sequential_13/lstm_26/zeros_1Fill-sequential_13/lstm_26/zeros_1/packed:output:0,sequential_13/lstm_26/zeros_1/Const:output:0*
T0*(
_output_shapes
:         В2
sequential_13/lstm_26/zeros_1б
$sequential_13/lstm_26/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2&
$sequential_13/lstm_26/transpose/perm├
sequential_13/lstm_26/transpose	Transposelstm_26_input-sequential_13/lstm_26/transpose/perm:output:0*
T0*+
_output_shapes
:         ]2!
sequential_13/lstm_26/transposeС
sequential_13/lstm_26/Shape_1Shape#sequential_13/lstm_26/transpose:y:0*
T0*
_output_shapes
:2
sequential_13/lstm_26/Shape_1д
+sequential_13/lstm_26/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2-
+sequential_13/lstm_26/strided_slice_1/stackи
-sequential_13/lstm_26/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_13/lstm_26/strided_slice_1/stack_1и
-sequential_13/lstm_26/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_13/lstm_26/strided_slice_1/stack_2Є
%sequential_13/lstm_26/strided_slice_1StridedSlice&sequential_13/lstm_26/Shape_1:output:04sequential_13/lstm_26/strided_slice_1/stack:output:06sequential_13/lstm_26/strided_slice_1/stack_1:output:06sequential_13/lstm_26/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%sequential_13/lstm_26/strided_slice_1▒
1sequential_13/lstm_26/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         23
1sequential_13/lstm_26/TensorArrayV2/element_shapeК
#sequential_13/lstm_26/TensorArrayV2TensorListReserve:sequential_13/lstm_26/TensorArrayV2/element_shape:output:0.sequential_13/lstm_26/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02%
#sequential_13/lstm_26/TensorArrayV2ы
Ksequential_13/lstm_26/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ]   2M
Ksequential_13/lstm_26/TensorArrayUnstack/TensorListFromTensor/element_shape╨
=sequential_13/lstm_26/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor#sequential_13/lstm_26/transpose:y:0Tsequential_13/lstm_26/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02?
=sequential_13/lstm_26/TensorArrayUnstack/TensorListFromTensorд
+sequential_13/lstm_26/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2-
+sequential_13/lstm_26/strided_slice_2/stackи
-sequential_13/lstm_26/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_13/lstm_26/strided_slice_2/stack_1и
-sequential_13/lstm_26/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_13/lstm_26/strided_slice_2/stack_2А
%sequential_13/lstm_26/strided_slice_2StridedSlice#sequential_13/lstm_26/transpose:y:04sequential_13/lstm_26/strided_slice_2/stack:output:06sequential_13/lstm_26/strided_slice_2/stack_1:output:06sequential_13/lstm_26/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         ]*
shrink_axis_mask2'
%sequential_13/lstm_26/strided_slice_2ў
8sequential_13/lstm_26/lstm_cell_26/MatMul/ReadVariableOpReadVariableOpAsequential_13_lstm_26_lstm_cell_26_matmul_readvariableop_resource*
_output_shapes
:	]И*
dtype02:
8sequential_13/lstm_26/lstm_cell_26/MatMul/ReadVariableOpЕ
)sequential_13/lstm_26/lstm_cell_26/MatMulMatMul.sequential_13/lstm_26/strided_slice_2:output:0@sequential_13/lstm_26/lstm_cell_26/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         И2+
)sequential_13/lstm_26/lstm_cell_26/MatMul■
:sequential_13/lstm_26/lstm_cell_26/MatMul_1/ReadVariableOpReadVariableOpCsequential_13_lstm_26_lstm_cell_26_matmul_1_readvariableop_resource* 
_output_shapes
:
ВИ*
dtype02<
:sequential_13/lstm_26/lstm_cell_26/MatMul_1/ReadVariableOpБ
+sequential_13/lstm_26/lstm_cell_26/MatMul_1MatMul$sequential_13/lstm_26/zeros:output:0Bsequential_13/lstm_26/lstm_cell_26/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         И2-
+sequential_13/lstm_26/lstm_cell_26/MatMul_1°
&sequential_13/lstm_26/lstm_cell_26/addAddV23sequential_13/lstm_26/lstm_cell_26/MatMul:product:05sequential_13/lstm_26/lstm_cell_26/MatMul_1:product:0*
T0*(
_output_shapes
:         И2(
&sequential_13/lstm_26/lstm_cell_26/addЎ
9sequential_13/lstm_26/lstm_cell_26/BiasAdd/ReadVariableOpReadVariableOpBsequential_13_lstm_26_lstm_cell_26_biasadd_readvariableop_resource*
_output_shapes	
:И*
dtype02;
9sequential_13/lstm_26/lstm_cell_26/BiasAdd/ReadVariableOpЕ
*sequential_13/lstm_26/lstm_cell_26/BiasAddBiasAdd*sequential_13/lstm_26/lstm_cell_26/add:z:0Asequential_13/lstm_26/lstm_cell_26/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         И2,
*sequential_13/lstm_26/lstm_cell_26/BiasAddк
2sequential_13/lstm_26/lstm_cell_26/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :24
2sequential_13/lstm_26/lstm_cell_26/split/split_dim╧
(sequential_13/lstm_26/lstm_cell_26/splitSplit;sequential_13/lstm_26/lstm_cell_26/split/split_dim:output:03sequential_13/lstm_26/lstm_cell_26/BiasAdd:output:0*
T0*d
_output_shapesR
P:         В:         В:         В:         В*
	num_split2*
(sequential_13/lstm_26/lstm_cell_26/split╔
*sequential_13/lstm_26/lstm_cell_26/SigmoidSigmoid1sequential_13/lstm_26/lstm_cell_26/split:output:0*
T0*(
_output_shapes
:         В2,
*sequential_13/lstm_26/lstm_cell_26/Sigmoid═
,sequential_13/lstm_26/lstm_cell_26/Sigmoid_1Sigmoid1sequential_13/lstm_26/lstm_cell_26/split:output:1*
T0*(
_output_shapes
:         В2.
,sequential_13/lstm_26/lstm_cell_26/Sigmoid_1ф
&sequential_13/lstm_26/lstm_cell_26/mulMul0sequential_13/lstm_26/lstm_cell_26/Sigmoid_1:y:0&sequential_13/lstm_26/zeros_1:output:0*
T0*(
_output_shapes
:         В2(
&sequential_13/lstm_26/lstm_cell_26/mul└
'sequential_13/lstm_26/lstm_cell_26/ReluRelu1sequential_13/lstm_26/lstm_cell_26/split:output:2*
T0*(
_output_shapes
:         В2)
'sequential_13/lstm_26/lstm_cell_26/Reluї
(sequential_13/lstm_26/lstm_cell_26/mul_1Mul.sequential_13/lstm_26/lstm_cell_26/Sigmoid:y:05sequential_13/lstm_26/lstm_cell_26/Relu:activations:0*
T0*(
_output_shapes
:         В2*
(sequential_13/lstm_26/lstm_cell_26/mul_1ъ
(sequential_13/lstm_26/lstm_cell_26/add_1AddV2*sequential_13/lstm_26/lstm_cell_26/mul:z:0,sequential_13/lstm_26/lstm_cell_26/mul_1:z:0*
T0*(
_output_shapes
:         В2*
(sequential_13/lstm_26/lstm_cell_26/add_1═
,sequential_13/lstm_26/lstm_cell_26/Sigmoid_2Sigmoid1sequential_13/lstm_26/lstm_cell_26/split:output:3*
T0*(
_output_shapes
:         В2.
,sequential_13/lstm_26/lstm_cell_26/Sigmoid_2┐
)sequential_13/lstm_26/lstm_cell_26/Relu_1Relu,sequential_13/lstm_26/lstm_cell_26/add_1:z:0*
T0*(
_output_shapes
:         В2+
)sequential_13/lstm_26/lstm_cell_26/Relu_1∙
(sequential_13/lstm_26/lstm_cell_26/mul_2Mul0sequential_13/lstm_26/lstm_cell_26/Sigmoid_2:y:07sequential_13/lstm_26/lstm_cell_26/Relu_1:activations:0*
T0*(
_output_shapes
:         В2*
(sequential_13/lstm_26/lstm_cell_26/mul_2╗
3sequential_13/lstm_26/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    В   25
3sequential_13/lstm_26/TensorArrayV2_1/element_shapeР
%sequential_13/lstm_26/TensorArrayV2_1TensorListReserve<sequential_13/lstm_26/TensorArrayV2_1/element_shape:output:0.sequential_13/lstm_26/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02'
%sequential_13/lstm_26/TensorArrayV2_1z
sequential_13/lstm_26/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
sequential_13/lstm_26/timeл
.sequential_13/lstm_26/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         20
.sequential_13/lstm_26/while/maximum_iterationsЦ
(sequential_13/lstm_26/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2*
(sequential_13/lstm_26/while/loop_counter▌
sequential_13/lstm_26/whileWhile1sequential_13/lstm_26/while/loop_counter:output:07sequential_13/lstm_26/while/maximum_iterations:output:0#sequential_13/lstm_26/time:output:0.sequential_13/lstm_26/TensorArrayV2_1:handle:0$sequential_13/lstm_26/zeros:output:0&sequential_13/lstm_26/zeros_1:output:0.sequential_13/lstm_26/strided_slice_1:output:0Msequential_13/lstm_26/TensorArrayUnstack/TensorListFromTensor:output_handle:0Asequential_13_lstm_26_lstm_cell_26_matmul_readvariableop_resourceCsequential_13_lstm_26_lstm_cell_26_matmul_1_readvariableop_resourceBsequential_13_lstm_26_lstm_cell_26_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :         В:         В: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *5
body-R+
)sequential_13_lstm_26_while_body_40071798*5
cond-R+
)sequential_13_lstm_26_while_cond_40071797*M
output_shapes<
:: : : : :         В:         В: : : : : *
parallel_iterations 2
sequential_13/lstm_26/whileс
Fsequential_13/lstm_26/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    В   2H
Fsequential_13/lstm_26/TensorArrayV2Stack/TensorListStack/element_shape┴
8sequential_13/lstm_26/TensorArrayV2Stack/TensorListStackTensorListStack$sequential_13/lstm_26/while:output:3Osequential_13/lstm_26/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:         В*
element_dtype02:
8sequential_13/lstm_26/TensorArrayV2Stack/TensorListStackн
+sequential_13/lstm_26/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         2-
+sequential_13/lstm_26/strided_slice_3/stackи
-sequential_13/lstm_26/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2/
-sequential_13/lstm_26/strided_slice_3/stack_1и
-sequential_13/lstm_26/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_13/lstm_26/strided_slice_3/stack_2Я
%sequential_13/lstm_26/strided_slice_3StridedSliceAsequential_13/lstm_26/TensorArrayV2Stack/TensorListStack:tensor:04sequential_13/lstm_26/strided_slice_3/stack:output:06sequential_13/lstm_26/strided_slice_3/stack_1:output:06sequential_13/lstm_26/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         В*
shrink_axis_mask2'
%sequential_13/lstm_26/strided_slice_3е
&sequential_13/lstm_26/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2(
&sequential_13/lstm_26/transpose_1/perm■
!sequential_13/lstm_26/transpose_1	TransposeAsequential_13/lstm_26/TensorArrayV2Stack/TensorListStack:tensor:0/sequential_13/lstm_26/transpose_1/perm:output:0*
T0*,
_output_shapes
:         В2#
!sequential_13/lstm_26/transpose_1Т
sequential_13/lstm_26/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
sequential_13/lstm_26/runtime░
!sequential_13/dropout_26/IdentityIdentity%sequential_13/lstm_26/transpose_1:y:0*
T0*,
_output_shapes
:         В2#
!sequential_13/dropout_26/IdentityФ
sequential_13/lstm_27/ShapeShape*sequential_13/dropout_26/Identity:output:0*
T0*
_output_shapes
:2
sequential_13/lstm_27/Shapeа
)sequential_13/lstm_27/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)sequential_13/lstm_27/strided_slice/stackд
+sequential_13/lstm_27/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential_13/lstm_27/strided_slice/stack_1д
+sequential_13/lstm_27/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential_13/lstm_27/strided_slice/stack_2ц
#sequential_13/lstm_27/strided_sliceStridedSlice$sequential_13/lstm_27/Shape:output:02sequential_13/lstm_27/strided_slice/stack:output:04sequential_13/lstm_27/strided_slice/stack_1:output:04sequential_13/lstm_27/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#sequential_13/lstm_27/strided_sliceЙ
!sequential_13/lstm_27/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :Ь2#
!sequential_13/lstm_27/zeros/mul/y─
sequential_13/lstm_27/zeros/mulMul,sequential_13/lstm_27/strided_slice:output:0*sequential_13/lstm_27/zeros/mul/y:output:0*
T0*
_output_shapes
: 2!
sequential_13/lstm_27/zeros/mulЛ
"sequential_13/lstm_27/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2$
"sequential_13/lstm_27/zeros/Less/y┐
 sequential_13/lstm_27/zeros/LessLess#sequential_13/lstm_27/zeros/mul:z:0+sequential_13/lstm_27/zeros/Less/y:output:0*
T0*
_output_shapes
: 2"
 sequential_13/lstm_27/zeros/LessП
$sequential_13/lstm_27/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :Ь2&
$sequential_13/lstm_27/zeros/packed/1█
"sequential_13/lstm_27/zeros/packedPack,sequential_13/lstm_27/strided_slice:output:0-sequential_13/lstm_27/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2$
"sequential_13/lstm_27/zeros/packedЛ
!sequential_13/lstm_27/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!sequential_13/lstm_27/zeros/Const╬
sequential_13/lstm_27/zerosFill+sequential_13/lstm_27/zeros/packed:output:0*sequential_13/lstm_27/zeros/Const:output:0*
T0*(
_output_shapes
:         Ь2
sequential_13/lstm_27/zerosН
#sequential_13/lstm_27/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :Ь2%
#sequential_13/lstm_27/zeros_1/mul/y╩
!sequential_13/lstm_27/zeros_1/mulMul,sequential_13/lstm_27/strided_slice:output:0,sequential_13/lstm_27/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2#
!sequential_13/lstm_27/zeros_1/mulП
$sequential_13/lstm_27/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2&
$sequential_13/lstm_27/zeros_1/Less/y╟
"sequential_13/lstm_27/zeros_1/LessLess%sequential_13/lstm_27/zeros_1/mul:z:0-sequential_13/lstm_27/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2$
"sequential_13/lstm_27/zeros_1/LessУ
&sequential_13/lstm_27/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :Ь2(
&sequential_13/lstm_27/zeros_1/packed/1с
$sequential_13/lstm_27/zeros_1/packedPack,sequential_13/lstm_27/strided_slice:output:0/sequential_13/lstm_27/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2&
$sequential_13/lstm_27/zeros_1/packedП
#sequential_13/lstm_27/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2%
#sequential_13/lstm_27/zeros_1/Const╓
sequential_13/lstm_27/zeros_1Fill-sequential_13/lstm_27/zeros_1/packed:output:0,sequential_13/lstm_27/zeros_1/Const:output:0*
T0*(
_output_shapes
:         Ь2
sequential_13/lstm_27/zeros_1б
$sequential_13/lstm_27/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2&
$sequential_13/lstm_27/transpose/permс
sequential_13/lstm_27/transpose	Transpose*sequential_13/dropout_26/Identity:output:0-sequential_13/lstm_27/transpose/perm:output:0*
T0*,
_output_shapes
:         В2!
sequential_13/lstm_27/transposeС
sequential_13/lstm_27/Shape_1Shape#sequential_13/lstm_27/transpose:y:0*
T0*
_output_shapes
:2
sequential_13/lstm_27/Shape_1д
+sequential_13/lstm_27/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2-
+sequential_13/lstm_27/strided_slice_1/stackи
-sequential_13/lstm_27/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_13/lstm_27/strided_slice_1/stack_1и
-sequential_13/lstm_27/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_13/lstm_27/strided_slice_1/stack_2Є
%sequential_13/lstm_27/strided_slice_1StridedSlice&sequential_13/lstm_27/Shape_1:output:04sequential_13/lstm_27/strided_slice_1/stack:output:06sequential_13/lstm_27/strided_slice_1/stack_1:output:06sequential_13/lstm_27/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%sequential_13/lstm_27/strided_slice_1▒
1sequential_13/lstm_27/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         23
1sequential_13/lstm_27/TensorArrayV2/element_shapeК
#sequential_13/lstm_27/TensorArrayV2TensorListReserve:sequential_13/lstm_27/TensorArrayV2/element_shape:output:0.sequential_13/lstm_27/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02%
#sequential_13/lstm_27/TensorArrayV2ы
Ksequential_13/lstm_27/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    В   2M
Ksequential_13/lstm_27/TensorArrayUnstack/TensorListFromTensor/element_shape╨
=sequential_13/lstm_27/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor#sequential_13/lstm_27/transpose:y:0Tsequential_13/lstm_27/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02?
=sequential_13/lstm_27/TensorArrayUnstack/TensorListFromTensorд
+sequential_13/lstm_27/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2-
+sequential_13/lstm_27/strided_slice_2/stackи
-sequential_13/lstm_27/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_13/lstm_27/strided_slice_2/stack_1и
-sequential_13/lstm_27/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_13/lstm_27/strided_slice_2/stack_2Б
%sequential_13/lstm_27/strided_slice_2StridedSlice#sequential_13/lstm_27/transpose:y:04sequential_13/lstm_27/strided_slice_2/stack:output:06sequential_13/lstm_27/strided_slice_2/stack_1:output:06sequential_13/lstm_27/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         В*
shrink_axis_mask2'
%sequential_13/lstm_27/strided_slice_2°
8sequential_13/lstm_27/lstm_cell_27/MatMul/ReadVariableOpReadVariableOpAsequential_13_lstm_27_lstm_cell_27_matmul_readvariableop_resource* 
_output_shapes
:
ВЁ*
dtype02:
8sequential_13/lstm_27/lstm_cell_27/MatMul/ReadVariableOpЕ
)sequential_13/lstm_27/lstm_cell_27/MatMulMatMul.sequential_13/lstm_27/strided_slice_2:output:0@sequential_13/lstm_27/lstm_cell_27/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ё2+
)sequential_13/lstm_27/lstm_cell_27/MatMul■
:sequential_13/lstm_27/lstm_cell_27/MatMul_1/ReadVariableOpReadVariableOpCsequential_13_lstm_27_lstm_cell_27_matmul_1_readvariableop_resource* 
_output_shapes
:
ЬЁ*
dtype02<
:sequential_13/lstm_27/lstm_cell_27/MatMul_1/ReadVariableOpБ
+sequential_13/lstm_27/lstm_cell_27/MatMul_1MatMul$sequential_13/lstm_27/zeros:output:0Bsequential_13/lstm_27/lstm_cell_27/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ё2-
+sequential_13/lstm_27/lstm_cell_27/MatMul_1°
&sequential_13/lstm_27/lstm_cell_27/addAddV23sequential_13/lstm_27/lstm_cell_27/MatMul:product:05sequential_13/lstm_27/lstm_cell_27/MatMul_1:product:0*
T0*(
_output_shapes
:         Ё2(
&sequential_13/lstm_27/lstm_cell_27/addЎ
9sequential_13/lstm_27/lstm_cell_27/BiasAdd/ReadVariableOpReadVariableOpBsequential_13_lstm_27_lstm_cell_27_biasadd_readvariableop_resource*
_output_shapes	
:Ё*
dtype02;
9sequential_13/lstm_27/lstm_cell_27/BiasAdd/ReadVariableOpЕ
*sequential_13/lstm_27/lstm_cell_27/BiasAddBiasAdd*sequential_13/lstm_27/lstm_cell_27/add:z:0Asequential_13/lstm_27/lstm_cell_27/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ё2,
*sequential_13/lstm_27/lstm_cell_27/BiasAddк
2sequential_13/lstm_27/lstm_cell_27/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :24
2sequential_13/lstm_27/lstm_cell_27/split/split_dim╧
(sequential_13/lstm_27/lstm_cell_27/splitSplit;sequential_13/lstm_27/lstm_cell_27/split/split_dim:output:03sequential_13/lstm_27/lstm_cell_27/BiasAdd:output:0*
T0*d
_output_shapesR
P:         Ь:         Ь:         Ь:         Ь*
	num_split2*
(sequential_13/lstm_27/lstm_cell_27/split╔
*sequential_13/lstm_27/lstm_cell_27/SigmoidSigmoid1sequential_13/lstm_27/lstm_cell_27/split:output:0*
T0*(
_output_shapes
:         Ь2,
*sequential_13/lstm_27/lstm_cell_27/Sigmoid═
,sequential_13/lstm_27/lstm_cell_27/Sigmoid_1Sigmoid1sequential_13/lstm_27/lstm_cell_27/split:output:1*
T0*(
_output_shapes
:         Ь2.
,sequential_13/lstm_27/lstm_cell_27/Sigmoid_1ф
&sequential_13/lstm_27/lstm_cell_27/mulMul0sequential_13/lstm_27/lstm_cell_27/Sigmoid_1:y:0&sequential_13/lstm_27/zeros_1:output:0*
T0*(
_output_shapes
:         Ь2(
&sequential_13/lstm_27/lstm_cell_27/mul└
'sequential_13/lstm_27/lstm_cell_27/ReluRelu1sequential_13/lstm_27/lstm_cell_27/split:output:2*
T0*(
_output_shapes
:         Ь2)
'sequential_13/lstm_27/lstm_cell_27/Reluї
(sequential_13/lstm_27/lstm_cell_27/mul_1Mul.sequential_13/lstm_27/lstm_cell_27/Sigmoid:y:05sequential_13/lstm_27/lstm_cell_27/Relu:activations:0*
T0*(
_output_shapes
:         Ь2*
(sequential_13/lstm_27/lstm_cell_27/mul_1ъ
(sequential_13/lstm_27/lstm_cell_27/add_1AddV2*sequential_13/lstm_27/lstm_cell_27/mul:z:0,sequential_13/lstm_27/lstm_cell_27/mul_1:z:0*
T0*(
_output_shapes
:         Ь2*
(sequential_13/lstm_27/lstm_cell_27/add_1═
,sequential_13/lstm_27/lstm_cell_27/Sigmoid_2Sigmoid1sequential_13/lstm_27/lstm_cell_27/split:output:3*
T0*(
_output_shapes
:         Ь2.
,sequential_13/lstm_27/lstm_cell_27/Sigmoid_2┐
)sequential_13/lstm_27/lstm_cell_27/Relu_1Relu,sequential_13/lstm_27/lstm_cell_27/add_1:z:0*
T0*(
_output_shapes
:         Ь2+
)sequential_13/lstm_27/lstm_cell_27/Relu_1∙
(sequential_13/lstm_27/lstm_cell_27/mul_2Mul0sequential_13/lstm_27/lstm_cell_27/Sigmoid_2:y:07sequential_13/lstm_27/lstm_cell_27/Relu_1:activations:0*
T0*(
_output_shapes
:         Ь2*
(sequential_13/lstm_27/lstm_cell_27/mul_2╗
3sequential_13/lstm_27/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      25
3sequential_13/lstm_27/TensorArrayV2_1/element_shapeР
%sequential_13/lstm_27/TensorArrayV2_1TensorListReserve<sequential_13/lstm_27/TensorArrayV2_1/element_shape:output:0.sequential_13/lstm_27/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02'
%sequential_13/lstm_27/TensorArrayV2_1z
sequential_13/lstm_27/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
sequential_13/lstm_27/timeл
.sequential_13/lstm_27/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         20
.sequential_13/lstm_27/while/maximum_iterationsЦ
(sequential_13/lstm_27/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2*
(sequential_13/lstm_27/while/loop_counter▌
sequential_13/lstm_27/whileWhile1sequential_13/lstm_27/while/loop_counter:output:07sequential_13/lstm_27/while/maximum_iterations:output:0#sequential_13/lstm_27/time:output:0.sequential_13/lstm_27/TensorArrayV2_1:handle:0$sequential_13/lstm_27/zeros:output:0&sequential_13/lstm_27/zeros_1:output:0.sequential_13/lstm_27/strided_slice_1:output:0Msequential_13/lstm_27/TensorArrayUnstack/TensorListFromTensor:output_handle:0Asequential_13_lstm_27_lstm_cell_27_matmul_readvariableop_resourceCsequential_13_lstm_27_lstm_cell_27_matmul_1_readvariableop_resourceBsequential_13_lstm_27_lstm_cell_27_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :         Ь:         Ь: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *5
body-R+
)sequential_13_lstm_27_while_body_40071946*5
cond-R+
)sequential_13_lstm_27_while_cond_40071945*M
output_shapes<
:: : : : :         Ь:         Ь: : : : : *
parallel_iterations 2
sequential_13/lstm_27/whileс
Fsequential_13/lstm_27/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      2H
Fsequential_13/lstm_27/TensorArrayV2Stack/TensorListStack/element_shape┴
8sequential_13/lstm_27/TensorArrayV2Stack/TensorListStackTensorListStack$sequential_13/lstm_27/while:output:3Osequential_13/lstm_27/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:         Ь*
element_dtype02:
8sequential_13/lstm_27/TensorArrayV2Stack/TensorListStackн
+sequential_13/lstm_27/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         2-
+sequential_13/lstm_27/strided_slice_3/stackи
-sequential_13/lstm_27/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2/
-sequential_13/lstm_27/strided_slice_3/stack_1и
-sequential_13/lstm_27/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_13/lstm_27/strided_slice_3/stack_2Я
%sequential_13/lstm_27/strided_slice_3StridedSliceAsequential_13/lstm_27/TensorArrayV2Stack/TensorListStack:tensor:04sequential_13/lstm_27/strided_slice_3/stack:output:06sequential_13/lstm_27/strided_slice_3/stack_1:output:06sequential_13/lstm_27/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         Ь*
shrink_axis_mask2'
%sequential_13/lstm_27/strided_slice_3е
&sequential_13/lstm_27/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2(
&sequential_13/lstm_27/transpose_1/perm■
!sequential_13/lstm_27/transpose_1	TransposeAsequential_13/lstm_27/TensorArrayV2Stack/TensorListStack:tensor:0/sequential_13/lstm_27/transpose_1/perm:output:0*
T0*,
_output_shapes
:         Ь2#
!sequential_13/lstm_27/transpose_1Т
sequential_13/lstm_27/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
sequential_13/lstm_27/runtime░
!sequential_13/dropout_27/IdentityIdentity%sequential_13/lstm_27/transpose_1:y:0*
T0*,
_output_shapes
:         Ь2#
!sequential_13/dropout_27/Identity▄
/sequential_13/dense_13/Tensordot/ReadVariableOpReadVariableOp8sequential_13_dense_13_tensordot_readvariableop_resource*
_output_shapes
:	Ь*
dtype021
/sequential_13/dense_13/Tensordot/ReadVariableOpШ
%sequential_13/dense_13/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2'
%sequential_13/dense_13/Tensordot/axesЯ
%sequential_13/dense_13/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2'
%sequential_13/dense_13/Tensordot/freeк
&sequential_13/dense_13/Tensordot/ShapeShape*sequential_13/dropout_27/Identity:output:0*
T0*
_output_shapes
:2(
&sequential_13/dense_13/Tensordot/Shapeв
.sequential_13/dense_13/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.sequential_13/dense_13/Tensordot/GatherV2/axis─
)sequential_13/dense_13/Tensordot/GatherV2GatherV2/sequential_13/dense_13/Tensordot/Shape:output:0.sequential_13/dense_13/Tensordot/free:output:07sequential_13/dense_13/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2+
)sequential_13/dense_13/Tensordot/GatherV2ж
0sequential_13/dense_13/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 22
0sequential_13/dense_13/Tensordot/GatherV2_1/axis╩
+sequential_13/dense_13/Tensordot/GatherV2_1GatherV2/sequential_13/dense_13/Tensordot/Shape:output:0.sequential_13/dense_13/Tensordot/axes:output:09sequential_13/dense_13/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2-
+sequential_13/dense_13/Tensordot/GatherV2_1Ъ
&sequential_13/dense_13/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2(
&sequential_13/dense_13/Tensordot/Const▄
%sequential_13/dense_13/Tensordot/ProdProd2sequential_13/dense_13/Tensordot/GatherV2:output:0/sequential_13/dense_13/Tensordot/Const:output:0*
T0*
_output_shapes
: 2'
%sequential_13/dense_13/Tensordot/ProdЮ
(sequential_13/dense_13/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(sequential_13/dense_13/Tensordot/Const_1ф
'sequential_13/dense_13/Tensordot/Prod_1Prod4sequential_13/dense_13/Tensordot/GatherV2_1:output:01sequential_13/dense_13/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2)
'sequential_13/dense_13/Tensordot/Prod_1Ю
,sequential_13/dense_13/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,sequential_13/dense_13/Tensordot/concat/axisг
'sequential_13/dense_13/Tensordot/concatConcatV2.sequential_13/dense_13/Tensordot/free:output:0.sequential_13/dense_13/Tensordot/axes:output:05sequential_13/dense_13/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2)
'sequential_13/dense_13/Tensordot/concatш
&sequential_13/dense_13/Tensordot/stackPack.sequential_13/dense_13/Tensordot/Prod:output:00sequential_13/dense_13/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2(
&sequential_13/dense_13/Tensordot/stack·
*sequential_13/dense_13/Tensordot/transpose	Transpose*sequential_13/dropout_27/Identity:output:00sequential_13/dense_13/Tensordot/concat:output:0*
T0*,
_output_shapes
:         Ь2,
*sequential_13/dense_13/Tensordot/transpose√
(sequential_13/dense_13/Tensordot/ReshapeReshape.sequential_13/dense_13/Tensordot/transpose:y:0/sequential_13/dense_13/Tensordot/stack:output:0*
T0*0
_output_shapes
:                  2*
(sequential_13/dense_13/Tensordot/Reshape·
'sequential_13/dense_13/Tensordot/MatMulMatMul1sequential_13/dense_13/Tensordot/Reshape:output:07sequential_13/dense_13/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2)
'sequential_13/dense_13/Tensordot/MatMulЮ
(sequential_13/dense_13/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(sequential_13/dense_13/Tensordot/Const_2в
.sequential_13/dense_13/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.sequential_13/dense_13/Tensordot/concat_1/axis░
)sequential_13/dense_13/Tensordot/concat_1ConcatV22sequential_13/dense_13/Tensordot/GatherV2:output:01sequential_13/dense_13/Tensordot/Const_2:output:07sequential_13/dense_13/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2+
)sequential_13/dense_13/Tensordot/concat_1ь
 sequential_13/dense_13/TensordotReshape1sequential_13/dense_13/Tensordot/MatMul:product:02sequential_13/dense_13/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:         2"
 sequential_13/dense_13/Tensordot╤
-sequential_13/dense_13/BiasAdd/ReadVariableOpReadVariableOp6sequential_13_dense_13_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02/
-sequential_13/dense_13/BiasAdd/ReadVariableOpу
sequential_13/dense_13/BiasAddBiasAdd)sequential_13/dense_13/Tensordot:output:05sequential_13/dense_13/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:         2 
sequential_13/dense_13/BiasAddк
sequential_13/dense_13/SoftmaxSoftmax'sequential_13/dense_13/BiasAdd:output:0*
T0*+
_output_shapes
:         2 
sequential_13/dense_13/SoftmaxЗ
IdentityIdentity(sequential_13/dense_13/Softmax:softmax:0^NoOp*
T0*+
_output_shapes
:         2

Identity╘
NoOpNoOp.^sequential_13/dense_13/BiasAdd/ReadVariableOp0^sequential_13/dense_13/Tensordot/ReadVariableOp:^sequential_13/lstm_26/lstm_cell_26/BiasAdd/ReadVariableOp9^sequential_13/lstm_26/lstm_cell_26/MatMul/ReadVariableOp;^sequential_13/lstm_26/lstm_cell_26/MatMul_1/ReadVariableOp^sequential_13/lstm_26/while:^sequential_13/lstm_27/lstm_cell_27/BiasAdd/ReadVariableOp9^sequential_13/lstm_27/lstm_cell_27/MatMul/ReadVariableOp;^sequential_13/lstm_27/lstm_cell_27/MatMul_1/ReadVariableOp^sequential_13/lstm_27/while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':         ]: : : : : : : : 2^
-sequential_13/dense_13/BiasAdd/ReadVariableOp-sequential_13/dense_13/BiasAdd/ReadVariableOp2b
/sequential_13/dense_13/Tensordot/ReadVariableOp/sequential_13/dense_13/Tensordot/ReadVariableOp2v
9sequential_13/lstm_26/lstm_cell_26/BiasAdd/ReadVariableOp9sequential_13/lstm_26/lstm_cell_26/BiasAdd/ReadVariableOp2t
8sequential_13/lstm_26/lstm_cell_26/MatMul/ReadVariableOp8sequential_13/lstm_26/lstm_cell_26/MatMul/ReadVariableOp2x
:sequential_13/lstm_26/lstm_cell_26/MatMul_1/ReadVariableOp:sequential_13/lstm_26/lstm_cell_26/MatMul_1/ReadVariableOp2:
sequential_13/lstm_26/whilesequential_13/lstm_26/while2v
9sequential_13/lstm_27/lstm_cell_27/BiasAdd/ReadVariableOp9sequential_13/lstm_27/lstm_cell_27/BiasAdd/ReadVariableOp2t
8sequential_13/lstm_27/lstm_cell_27/MatMul/ReadVariableOp8sequential_13/lstm_27/lstm_cell_27/MatMul/ReadVariableOp2x
:sequential_13/lstm_27/lstm_cell_27/MatMul_1/ReadVariableOp:sequential_13/lstm_27/lstm_cell_27/MatMul_1/ReadVariableOp2:
sequential_13/lstm_27/whilesequential_13/lstm_27/while:Z V
+
_output_shapes
:         ]
'
_user_specified_namelstm_26_input
╩
·
/__inference_lstm_cell_27_layer_call_fn_40076571

inputs
states_0
states_1
unknown:
ВЁ
	unknown_0:
ЬЁ
	unknown_1:	Ё
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
<:         Ь:         Ь:         Ь*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_lstm_cell_27_layer_call_and_return_conditional_losses_400729092
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         Ь2

IdentityА

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:         Ь2

Identity_1А

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:         Ь2

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
B:         В:         Ь:         Ь: : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         В
 
_user_specified_nameinputs:RN
(
_output_shapes
:         Ь
"
_user_specified_name
states/0:RN
(
_output_shapes
:         Ь
"
_user_specified_name
states/1
у
═
while_cond_40076028
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_40076028___redundant_placeholder06
2while_while_cond_40076028___redundant_placeholder16
2while_while_cond_40076028___redundant_placeholder26
2while_while_cond_40076028___redundant_placeholder3
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
B: : : : :         Ь:         Ь: ::::: 
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
:         Ь:.*
(
_output_shapes
:         Ь:

_output_shapes
: :

_output_shapes
:
╣
╣
*__inference_lstm_27_layer_call_fn_40076297

inputs
unknown:
ВЁ
	unknown_0:
ЬЁ
	unknown_1:	Ё
identityИвStatefulPartitionedCallЗ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         Ь*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_lstm_27_layer_call_and_return_conditional_losses_400736402
StatefulPartitionedCallА
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:         Ь2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:         В: : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:         В
 
_user_specified_nameinputs
░?
╘
while_body_40075052
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_26_matmul_readvariableop_resource_0:	]ИI
5while_lstm_cell_26_matmul_1_readvariableop_resource_0:
ВИC
4while_lstm_cell_26_biasadd_readvariableop_resource_0:	И
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_26_matmul_readvariableop_resource:	]ИG
3while_lstm_cell_26_matmul_1_readvariableop_resource:
ВИA
2while_lstm_cell_26_biasadd_readvariableop_resource:	ИИв)while/lstm_cell_26/BiasAdd/ReadVariableOpв(while/lstm_cell_26/MatMul/ReadVariableOpв*while/lstm_cell_26/MatMul_1/ReadVariableOp├
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
(while/lstm_cell_26/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_26_matmul_readvariableop_resource_0*
_output_shapes
:	]И*
dtype02*
(while/lstm_cell_26/MatMul/ReadVariableOp╫
while/lstm_cell_26/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_26/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         И2
while/lstm_cell_26/MatMul╨
*while/lstm_cell_26/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_26_matmul_1_readvariableop_resource_0* 
_output_shapes
:
ВИ*
dtype02,
*while/lstm_cell_26/MatMul_1/ReadVariableOp└
while/lstm_cell_26/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_26/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         И2
while/lstm_cell_26/MatMul_1╕
while/lstm_cell_26/addAddV2#while/lstm_cell_26/MatMul:product:0%while/lstm_cell_26/MatMul_1:product:0*
T0*(
_output_shapes
:         И2
while/lstm_cell_26/add╚
)while/lstm_cell_26/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_26_biasadd_readvariableop_resource_0*
_output_shapes	
:И*
dtype02+
)while/lstm_cell_26/BiasAdd/ReadVariableOp┼
while/lstm_cell_26/BiasAddBiasAddwhile/lstm_cell_26/add:z:01while/lstm_cell_26/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         И2
while/lstm_cell_26/BiasAddК
"while/lstm_cell_26/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_26/split/split_dimП
while/lstm_cell_26/splitSplit+while/lstm_cell_26/split/split_dim:output:0#while/lstm_cell_26/BiasAdd:output:0*
T0*d
_output_shapesR
P:         В:         В:         В:         В*
	num_split2
while/lstm_cell_26/splitЩ
while/lstm_cell_26/SigmoidSigmoid!while/lstm_cell_26/split:output:0*
T0*(
_output_shapes
:         В2
while/lstm_cell_26/SigmoidЭ
while/lstm_cell_26/Sigmoid_1Sigmoid!while/lstm_cell_26/split:output:1*
T0*(
_output_shapes
:         В2
while/lstm_cell_26/Sigmoid_1б
while/lstm_cell_26/mulMul while/lstm_cell_26/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:         В2
while/lstm_cell_26/mulР
while/lstm_cell_26/ReluRelu!while/lstm_cell_26/split:output:2*
T0*(
_output_shapes
:         В2
while/lstm_cell_26/Relu╡
while/lstm_cell_26/mul_1Mulwhile/lstm_cell_26/Sigmoid:y:0%while/lstm_cell_26/Relu:activations:0*
T0*(
_output_shapes
:         В2
while/lstm_cell_26/mul_1к
while/lstm_cell_26/add_1AddV2while/lstm_cell_26/mul:z:0while/lstm_cell_26/mul_1:z:0*
T0*(
_output_shapes
:         В2
while/lstm_cell_26/add_1Э
while/lstm_cell_26/Sigmoid_2Sigmoid!while/lstm_cell_26/split:output:3*
T0*(
_output_shapes
:         В2
while/lstm_cell_26/Sigmoid_2П
while/lstm_cell_26/Relu_1Reluwhile/lstm_cell_26/add_1:z:0*
T0*(
_output_shapes
:         В2
while/lstm_cell_26/Relu_1╣
while/lstm_cell_26/mul_2Mul while/lstm_cell_26/Sigmoid_2:y:0'while/lstm_cell_26/Relu_1:activations:0*
T0*(
_output_shapes
:         В2
while/lstm_cell_26/mul_2р
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_26/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_26/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:         В2
while/Identity_4О
while/Identity_5Identitywhile/lstm_cell_26/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:         В2
while/Identity_5▐

while/NoOpNoOp*^while/lstm_cell_26/BiasAdd/ReadVariableOp)^while/lstm_cell_26/MatMul/ReadVariableOp+^while/lstm_cell_26/MatMul_1/ReadVariableOp*"
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
2while_lstm_cell_26_biasadd_readvariableop_resource4while_lstm_cell_26_biasadd_readvariableop_resource_0"l
3while_lstm_cell_26_matmul_1_readvariableop_resource5while_lstm_cell_26_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_26_matmul_readvariableop_resource3while_lstm_cell_26_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :         В:         В: : : : : 2V
)while/lstm_cell_26/BiasAdd/ReadVariableOp)while/lstm_cell_26/BiasAdd/ReadVariableOp2T
(while/lstm_cell_26/MatMul/ReadVariableOp(while/lstm_cell_26/MatMul/ReadVariableOp2X
*while/lstm_cell_26/MatMul_1/ReadVariableOp*while/lstm_cell_26/MatMul_1/ReadVariableOp: 
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
:         В:.*
(
_output_shapes
:         В:

_output_shapes
: :

_output_shapes
: 
╢
╕
*__inference_lstm_26_layer_call_fn_40075633

inputs
unknown:	]И
	unknown_0:
ВИ
	unknown_1:	И
identityИвStatefulPartitionedCallЗ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         В*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_lstm_26_layer_call_and_return_conditional_losses_400741052
StatefulPartitionedCallА
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:         В2

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
░?
╘
while_body_40075505
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_26_matmul_readvariableop_resource_0:	]ИI
5while_lstm_cell_26_matmul_1_readvariableop_resource_0:
ВИC
4while_lstm_cell_26_biasadd_readvariableop_resource_0:	И
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_26_matmul_readvariableop_resource:	]ИG
3while_lstm_cell_26_matmul_1_readvariableop_resource:
ВИA
2while_lstm_cell_26_biasadd_readvariableop_resource:	ИИв)while/lstm_cell_26/BiasAdd/ReadVariableOpв(while/lstm_cell_26/MatMul/ReadVariableOpв*while/lstm_cell_26/MatMul_1/ReadVariableOp├
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
(while/lstm_cell_26/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_26_matmul_readvariableop_resource_0*
_output_shapes
:	]И*
dtype02*
(while/lstm_cell_26/MatMul/ReadVariableOp╫
while/lstm_cell_26/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_26/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         И2
while/lstm_cell_26/MatMul╨
*while/lstm_cell_26/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_26_matmul_1_readvariableop_resource_0* 
_output_shapes
:
ВИ*
dtype02,
*while/lstm_cell_26/MatMul_1/ReadVariableOp└
while/lstm_cell_26/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_26/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         И2
while/lstm_cell_26/MatMul_1╕
while/lstm_cell_26/addAddV2#while/lstm_cell_26/MatMul:product:0%while/lstm_cell_26/MatMul_1:product:0*
T0*(
_output_shapes
:         И2
while/lstm_cell_26/add╚
)while/lstm_cell_26/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_26_biasadd_readvariableop_resource_0*
_output_shapes	
:И*
dtype02+
)while/lstm_cell_26/BiasAdd/ReadVariableOp┼
while/lstm_cell_26/BiasAddBiasAddwhile/lstm_cell_26/add:z:01while/lstm_cell_26/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         И2
while/lstm_cell_26/BiasAddК
"while/lstm_cell_26/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_26/split/split_dimП
while/lstm_cell_26/splitSplit+while/lstm_cell_26/split/split_dim:output:0#while/lstm_cell_26/BiasAdd:output:0*
T0*d
_output_shapesR
P:         В:         В:         В:         В*
	num_split2
while/lstm_cell_26/splitЩ
while/lstm_cell_26/SigmoidSigmoid!while/lstm_cell_26/split:output:0*
T0*(
_output_shapes
:         В2
while/lstm_cell_26/SigmoidЭ
while/lstm_cell_26/Sigmoid_1Sigmoid!while/lstm_cell_26/split:output:1*
T0*(
_output_shapes
:         В2
while/lstm_cell_26/Sigmoid_1б
while/lstm_cell_26/mulMul while/lstm_cell_26/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:         В2
while/lstm_cell_26/mulР
while/lstm_cell_26/ReluRelu!while/lstm_cell_26/split:output:2*
T0*(
_output_shapes
:         В2
while/lstm_cell_26/Relu╡
while/lstm_cell_26/mul_1Mulwhile/lstm_cell_26/Sigmoid:y:0%while/lstm_cell_26/Relu:activations:0*
T0*(
_output_shapes
:         В2
while/lstm_cell_26/mul_1к
while/lstm_cell_26/add_1AddV2while/lstm_cell_26/mul:z:0while/lstm_cell_26/mul_1:z:0*
T0*(
_output_shapes
:         В2
while/lstm_cell_26/add_1Э
while/lstm_cell_26/Sigmoid_2Sigmoid!while/lstm_cell_26/split:output:3*
T0*(
_output_shapes
:         В2
while/lstm_cell_26/Sigmoid_2П
while/lstm_cell_26/Relu_1Reluwhile/lstm_cell_26/add_1:z:0*
T0*(
_output_shapes
:         В2
while/lstm_cell_26/Relu_1╣
while/lstm_cell_26/mul_2Mul while/lstm_cell_26/Sigmoid_2:y:0'while/lstm_cell_26/Relu_1:activations:0*
T0*(
_output_shapes
:         В2
while/lstm_cell_26/mul_2р
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_26/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_26/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:         В2
while/Identity_4О
while/Identity_5Identitywhile/lstm_cell_26/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:         В2
while/Identity_5▐

while/NoOpNoOp*^while/lstm_cell_26/BiasAdd/ReadVariableOp)^while/lstm_cell_26/MatMul/ReadVariableOp+^while/lstm_cell_26/MatMul_1/ReadVariableOp*"
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
2while_lstm_cell_26_biasadd_readvariableop_resource4while_lstm_cell_26_biasadd_readvariableop_resource_0"l
3while_lstm_cell_26_matmul_1_readvariableop_resource5while_lstm_cell_26_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_26_matmul_readvariableop_resource3while_lstm_cell_26_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :         В:         В: : : : : 2V
)while/lstm_cell_26/BiasAdd/ReadVariableOp)while/lstm_cell_26/BiasAdd/ReadVariableOp2T
(while/lstm_cell_26/MatMul/ReadVariableOp(while/lstm_cell_26/MatMul/ReadVariableOp2X
*while/lstm_cell_26/MatMul_1/ReadVariableOp*while/lstm_cell_26/MatMul_1/ReadVariableOp: 
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
:         В:.*
(
_output_shapes
:         В:

_output_shapes
: :

_output_shapes
: 
у
═
while_cond_40075051
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_40075051___redundant_placeholder06
2while_while_cond_40075051___redundant_placeholder16
2while_while_cond_40075051___redundant_placeholder26
2while_while_cond_40075051___redundant_placeholder3
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
B: : : : :         В:         В: ::::: 
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
:         В:.*
(
_output_shapes
:         В:

_output_shapes
: :

_output_shapes
:
╘

э
lstm_27_while_cond_40074489,
(lstm_27_while_lstm_27_while_loop_counter2
.lstm_27_while_lstm_27_while_maximum_iterations
lstm_27_while_placeholder
lstm_27_while_placeholder_1
lstm_27_while_placeholder_2
lstm_27_while_placeholder_3.
*lstm_27_while_less_lstm_27_strided_slice_1F
Blstm_27_while_lstm_27_while_cond_40074489___redundant_placeholder0F
Blstm_27_while_lstm_27_while_cond_40074489___redundant_placeholder1F
Blstm_27_while_lstm_27_while_cond_40074489___redundant_placeholder2F
Blstm_27_while_lstm_27_while_cond_40074489___redundant_placeholder3
lstm_27_while_identity
Ш
lstm_27/while/LessLesslstm_27_while_placeholder*lstm_27_while_less_lstm_27_strided_slice_1*
T0*
_output_shapes
: 2
lstm_27/while/Lessu
lstm_27/while/IdentityIdentitylstm_27/while/Less:z:0*
T0
*
_output_shapes
: 2
lstm_27/while/Identity"9
lstm_27_while_identitylstm_27/while/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :         Ь:         Ь: ::::: 
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
:         Ь:.*
(
_output_shapes
:         Ь:

_output_shapes
: :

_output_shapes
:
░?
╘
while_body_40074021
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_26_matmul_readvariableop_resource_0:	]ИI
5while_lstm_cell_26_matmul_1_readvariableop_resource_0:
ВИC
4while_lstm_cell_26_biasadd_readvariableop_resource_0:	И
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_26_matmul_readvariableop_resource:	]ИG
3while_lstm_cell_26_matmul_1_readvariableop_resource:
ВИA
2while_lstm_cell_26_biasadd_readvariableop_resource:	ИИв)while/lstm_cell_26/BiasAdd/ReadVariableOpв(while/lstm_cell_26/MatMul/ReadVariableOpв*while/lstm_cell_26/MatMul_1/ReadVariableOp├
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
(while/lstm_cell_26/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_26_matmul_readvariableop_resource_0*
_output_shapes
:	]И*
dtype02*
(while/lstm_cell_26/MatMul/ReadVariableOp╫
while/lstm_cell_26/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_26/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         И2
while/lstm_cell_26/MatMul╨
*while/lstm_cell_26/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_26_matmul_1_readvariableop_resource_0* 
_output_shapes
:
ВИ*
dtype02,
*while/lstm_cell_26/MatMul_1/ReadVariableOp└
while/lstm_cell_26/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_26/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         И2
while/lstm_cell_26/MatMul_1╕
while/lstm_cell_26/addAddV2#while/lstm_cell_26/MatMul:product:0%while/lstm_cell_26/MatMul_1:product:0*
T0*(
_output_shapes
:         И2
while/lstm_cell_26/add╚
)while/lstm_cell_26/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_26_biasadd_readvariableop_resource_0*
_output_shapes	
:И*
dtype02+
)while/lstm_cell_26/BiasAdd/ReadVariableOp┼
while/lstm_cell_26/BiasAddBiasAddwhile/lstm_cell_26/add:z:01while/lstm_cell_26/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         И2
while/lstm_cell_26/BiasAddК
"while/lstm_cell_26/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_26/split/split_dimП
while/lstm_cell_26/splitSplit+while/lstm_cell_26/split/split_dim:output:0#while/lstm_cell_26/BiasAdd:output:0*
T0*d
_output_shapesR
P:         В:         В:         В:         В*
	num_split2
while/lstm_cell_26/splitЩ
while/lstm_cell_26/SigmoidSigmoid!while/lstm_cell_26/split:output:0*
T0*(
_output_shapes
:         В2
while/lstm_cell_26/SigmoidЭ
while/lstm_cell_26/Sigmoid_1Sigmoid!while/lstm_cell_26/split:output:1*
T0*(
_output_shapes
:         В2
while/lstm_cell_26/Sigmoid_1б
while/lstm_cell_26/mulMul while/lstm_cell_26/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:         В2
while/lstm_cell_26/mulР
while/lstm_cell_26/ReluRelu!while/lstm_cell_26/split:output:2*
T0*(
_output_shapes
:         В2
while/lstm_cell_26/Relu╡
while/lstm_cell_26/mul_1Mulwhile/lstm_cell_26/Sigmoid:y:0%while/lstm_cell_26/Relu:activations:0*
T0*(
_output_shapes
:         В2
while/lstm_cell_26/mul_1к
while/lstm_cell_26/add_1AddV2while/lstm_cell_26/mul:z:0while/lstm_cell_26/mul_1:z:0*
T0*(
_output_shapes
:         В2
while/lstm_cell_26/add_1Э
while/lstm_cell_26/Sigmoid_2Sigmoid!while/lstm_cell_26/split:output:3*
T0*(
_output_shapes
:         В2
while/lstm_cell_26/Sigmoid_2П
while/lstm_cell_26/Relu_1Reluwhile/lstm_cell_26/add_1:z:0*
T0*(
_output_shapes
:         В2
while/lstm_cell_26/Relu_1╣
while/lstm_cell_26/mul_2Mul while/lstm_cell_26/Sigmoid_2:y:0'while/lstm_cell_26/Relu_1:activations:0*
T0*(
_output_shapes
:         В2
while/lstm_cell_26/mul_2р
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_26/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_26/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:         В2
while/Identity_4О
while/Identity_5Identitywhile/lstm_cell_26/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:         В2
while/Identity_5▐

while/NoOpNoOp*^while/lstm_cell_26/BiasAdd/ReadVariableOp)^while/lstm_cell_26/MatMul/ReadVariableOp+^while/lstm_cell_26/MatMul_1/ReadVariableOp*"
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
2while_lstm_cell_26_biasadd_readvariableop_resource4while_lstm_cell_26_biasadd_readvariableop_resource_0"l
3while_lstm_cell_26_matmul_1_readvariableop_resource5while_lstm_cell_26_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_26_matmul_readvariableop_resource3while_lstm_cell_26_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :         В:         В: : : : : 2V
)while/lstm_cell_26/BiasAdd/ReadVariableOp)while/lstm_cell_26/BiasAdd/ReadVariableOp2T
(while/lstm_cell_26/MatMul/ReadVariableOp(while/lstm_cell_26/MatMul/ReadVariableOp2X
*while/lstm_cell_26/MatMul_1/ReadVariableOp*while/lstm_cell_26/MatMul_1/ReadVariableOp: 
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
:         В:.*
(
_output_shapes
:         В:

_output_shapes
: :

_output_shapes
: 
╗
f
-__inference_dropout_27_layer_call_fn_40076335

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
:         Ь* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_dropout_27_layer_call_and_return_conditional_losses_400737422
StatefulPartitionedCallА
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:         Ь2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         Ь22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:         Ь
 
_user_specified_nameinputs
Ч
К
J__inference_lstm_cell_27_layer_call_and_return_conditional_losses_40076505

inputs
states_0
states_12
matmul_readvariableop_resource:
ВЁ4
 matmul_1_readvariableop_resource:
ЬЁ.
biasadd_readvariableop_resource:	Ё
identity

identity_1

identity_2ИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpвMatMul_1/ReadVariableOpП
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
ВЁ*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ё2
MatMulХ
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
ЬЁ*
dtype02
MatMul_1/ReadVariableOp|
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ё2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:         Ё2
addН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:Ё*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ё2	
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
P:         Ь:         Ь:         Ь:         Ь*
	num_split2
split`
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:         Ь2	
Sigmoidd
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:         Ь2
	Sigmoid_1]
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:         Ь2
mulW
ReluRelusplit:output:2*
T0*(
_output_shapes
:         Ь2
Relui
mul_1MulSigmoid:y:0Relu:activations:0*
T0*(
_output_shapes
:         Ь2
mul_1^
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:         Ь2
add_1d
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:         Ь2
	Sigmoid_2V
Relu_1Relu	add_1:z:0*
T0*(
_output_shapes
:         Ь2
Relu_1m
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*(
_output_shapes
:         Ь2
mul_2e
IdentityIdentity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:         Ь2

Identityi

Identity_1Identity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:         Ь2

Identity_1i

Identity_2Identity	add_1:z:0^NoOp*
T0*(
_output_shapes
:         Ь2

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
B:         В:         Ь:         Ь: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:P L
(
_output_shapes
:         В
 
_user_specified_nameinputs:RN
(
_output_shapes
:         Ь
"
_user_specified_name
states/0:RN
(
_output_shapes
:         Ь
"
_user_specified_name
states/1
р
║
*__inference_lstm_26_layer_call_fn_40075611
inputs_0
unknown:	]И
	unknown_0:
ВИ
	unknown_1:	И
identityИвStatefulPartitionedCallТ
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:                  В*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_lstm_26_layer_call_and_return_conditional_losses_400724262
StatefulPartitionedCallЙ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:                  В2

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
while_cond_40072146
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_40072146___redundant_placeholder06
2while_while_cond_40072146___redundant_placeholder16
2while_while_cond_40072146___redundant_placeholder26
2while_while_cond_40072146___redundant_placeholder3
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
B: : : : :         В:         В: ::::: 
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
:         В:.*
(
_output_shapes
:         В:

_output_shapes
: :

_output_shapes
:
П
И
J__inference_lstm_cell_27_layer_call_and_return_conditional_losses_40072763

inputs

states
states_12
matmul_readvariableop_resource:
ВЁ4
 matmul_1_readvariableop_resource:
ЬЁ.
biasadd_readvariableop_resource:	Ё
identity

identity_1

identity_2ИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpвMatMul_1/ReadVariableOpП
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
ВЁ*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ё2
MatMulХ
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
ЬЁ*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ё2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:         Ё2
addН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:Ё*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ё2	
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
P:         Ь:         Ь:         Ь:         Ь*
	num_split2
split`
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:         Ь2	
Sigmoidd
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:         Ь2
	Sigmoid_1]
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:         Ь2
mulW
ReluRelusplit:output:2*
T0*(
_output_shapes
:         Ь2
Relui
mul_1MulSigmoid:y:0Relu:activations:0*
T0*(
_output_shapes
:         Ь2
mul_1^
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:         Ь2
add_1d
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:         Ь2
	Sigmoid_2V
Relu_1Relu	add_1:z:0*
T0*(
_output_shapes
:         Ь2
Relu_1m
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*(
_output_shapes
:         Ь2
mul_2e
IdentityIdentity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:         Ь2

Identityi

Identity_1Identity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:         Ь2

Identity_1i

Identity_2Identity	add_1:z:0^NoOp*
T0*(
_output_shapes
:         Ь2

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
B:         В:         Ь:         Ь: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:P L
(
_output_shapes
:         В
 
_user_specified_nameinputs:PL
(
_output_shapes
:         Ь
 
_user_specified_namestates:PL
(
_output_shapes
:         Ь
 
_user_specified_namestates
╩
·
/__inference_lstm_cell_27_layer_call_fn_40076554

inputs
states_0
states_1
unknown:
ВЁ
	unknown_0:
ЬЁ
	unknown_1:	Ё
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
<:         Ь:         Ь:         Ь*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_lstm_cell_27_layer_call_and_return_conditional_losses_400727632
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         Ь2

IdentityА

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:         Ь2

Identity_1А

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:         Ь2

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
B:         В:         Ь:         Ь: : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         В
 
_user_specified_nameinputs:RN
(
_output_shapes
:         Ь
"
_user_specified_name
states/0:RN
(
_output_shapes
:         Ь
"
_user_specified_name
states/1
░?
╘
while_body_40075354
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_26_matmul_readvariableop_resource_0:	]ИI
5while_lstm_cell_26_matmul_1_readvariableop_resource_0:
ВИC
4while_lstm_cell_26_biasadd_readvariableop_resource_0:	И
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_26_matmul_readvariableop_resource:	]ИG
3while_lstm_cell_26_matmul_1_readvariableop_resource:
ВИA
2while_lstm_cell_26_biasadd_readvariableop_resource:	ИИв)while/lstm_cell_26/BiasAdd/ReadVariableOpв(while/lstm_cell_26/MatMul/ReadVariableOpв*while/lstm_cell_26/MatMul_1/ReadVariableOp├
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
(while/lstm_cell_26/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_26_matmul_readvariableop_resource_0*
_output_shapes
:	]И*
dtype02*
(while/lstm_cell_26/MatMul/ReadVariableOp╫
while/lstm_cell_26/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_26/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         И2
while/lstm_cell_26/MatMul╨
*while/lstm_cell_26/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_26_matmul_1_readvariableop_resource_0* 
_output_shapes
:
ВИ*
dtype02,
*while/lstm_cell_26/MatMul_1/ReadVariableOp└
while/lstm_cell_26/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_26/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         И2
while/lstm_cell_26/MatMul_1╕
while/lstm_cell_26/addAddV2#while/lstm_cell_26/MatMul:product:0%while/lstm_cell_26/MatMul_1:product:0*
T0*(
_output_shapes
:         И2
while/lstm_cell_26/add╚
)while/lstm_cell_26/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_26_biasadd_readvariableop_resource_0*
_output_shapes	
:И*
dtype02+
)while/lstm_cell_26/BiasAdd/ReadVariableOp┼
while/lstm_cell_26/BiasAddBiasAddwhile/lstm_cell_26/add:z:01while/lstm_cell_26/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         И2
while/lstm_cell_26/BiasAddК
"while/lstm_cell_26/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_26/split/split_dimП
while/lstm_cell_26/splitSplit+while/lstm_cell_26/split/split_dim:output:0#while/lstm_cell_26/BiasAdd:output:0*
T0*d
_output_shapesR
P:         В:         В:         В:         В*
	num_split2
while/lstm_cell_26/splitЩ
while/lstm_cell_26/SigmoidSigmoid!while/lstm_cell_26/split:output:0*
T0*(
_output_shapes
:         В2
while/lstm_cell_26/SigmoidЭ
while/lstm_cell_26/Sigmoid_1Sigmoid!while/lstm_cell_26/split:output:1*
T0*(
_output_shapes
:         В2
while/lstm_cell_26/Sigmoid_1б
while/lstm_cell_26/mulMul while/lstm_cell_26/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:         В2
while/lstm_cell_26/mulР
while/lstm_cell_26/ReluRelu!while/lstm_cell_26/split:output:2*
T0*(
_output_shapes
:         В2
while/lstm_cell_26/Relu╡
while/lstm_cell_26/mul_1Mulwhile/lstm_cell_26/Sigmoid:y:0%while/lstm_cell_26/Relu:activations:0*
T0*(
_output_shapes
:         В2
while/lstm_cell_26/mul_1к
while/lstm_cell_26/add_1AddV2while/lstm_cell_26/mul:z:0while/lstm_cell_26/mul_1:z:0*
T0*(
_output_shapes
:         В2
while/lstm_cell_26/add_1Э
while/lstm_cell_26/Sigmoid_2Sigmoid!while/lstm_cell_26/split:output:3*
T0*(
_output_shapes
:         В2
while/lstm_cell_26/Sigmoid_2П
while/lstm_cell_26/Relu_1Reluwhile/lstm_cell_26/add_1:z:0*
T0*(
_output_shapes
:         В2
while/lstm_cell_26/Relu_1╣
while/lstm_cell_26/mul_2Mul while/lstm_cell_26/Sigmoid_2:y:0'while/lstm_cell_26/Relu_1:activations:0*
T0*(
_output_shapes
:         В2
while/lstm_cell_26/mul_2р
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_26/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_26/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:         В2
while/Identity_4О
while/Identity_5Identitywhile/lstm_cell_26/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:         В2
while/Identity_5▐

while/NoOpNoOp*^while/lstm_cell_26/BiasAdd/ReadVariableOp)^while/lstm_cell_26/MatMul/ReadVariableOp+^while/lstm_cell_26/MatMul_1/ReadVariableOp*"
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
2while_lstm_cell_26_biasadd_readvariableop_resource4while_lstm_cell_26_biasadd_readvariableop_resource_0"l
3while_lstm_cell_26_matmul_1_readvariableop_resource5while_lstm_cell_26_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_26_matmul_readvariableop_resource3while_lstm_cell_26_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :         В:         В: : : : : 2V
)while/lstm_cell_26/BiasAdd/ReadVariableOp)while/lstm_cell_26/BiasAdd/ReadVariableOp2T
(while/lstm_cell_26/MatMul/ReadVariableOp(while/lstm_cell_26/MatMul/ReadVariableOp2X
*while/lstm_cell_26/MatMul_1/ReadVariableOp*while/lstm_cell_26/MatMul_1/ReadVariableOp: 
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
:         В:.*
(
_output_shapes
:         В:

_output_shapes
: :

_output_shapes
: 
И&
ї
while_body_40072777
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_01
while_lstm_cell_27_40072801_0:
ВЁ1
while_lstm_cell_27_40072803_0:
ЬЁ,
while_lstm_cell_27_40072805_0:	Ё
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor/
while_lstm_cell_27_40072801:
ВЁ/
while_lstm_cell_27_40072803:
ЬЁ*
while_lstm_cell_27_40072805:	ЁИв*while/lstm_cell_27/StatefulPartitionedCall├
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    В   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape╘
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:         В*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemь
*while/lstm_cell_27/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_27_40072801_0while_lstm_cell_27_40072803_0while_lstm_cell_27_40072805_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:         Ь:         Ь:         Ь*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_lstm_cell_27_layer_call_and_return_conditional_losses_400727632,
*while/lstm_cell_27/StatefulPartitionedCallў
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder3while/lstm_cell_27/StatefulPartitionedCall:output:0*
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
while/Identity_4Identity3while/lstm_cell_27/StatefulPartitionedCall:output:1^while/NoOp*
T0*(
_output_shapes
:         Ь2
while/Identity_4е
while/Identity_5Identity3while/lstm_cell_27/StatefulPartitionedCall:output:2^while/NoOp*
T0*(
_output_shapes
:         Ь2
while/Identity_5З

while/NoOpNoOp+^while/lstm_cell_27/StatefulPartitionedCall*"
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
while_lstm_cell_27_40072801while_lstm_cell_27_40072801_0"<
while_lstm_cell_27_40072803while_lstm_cell_27_40072803_0"<
while_lstm_cell_27_40072805while_lstm_cell_27_40072805_0"0
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :         Ь:         Ь: : : : : 2X
*while/lstm_cell_27/StatefulPartitionedCall*while/lstm_cell_27/StatefulPartitionedCall: 
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
:         Ь:.*
(
_output_shapes
:         Ь:

_output_shapes
: :

_output_shapes
: 
у
═
while_cond_40075726
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_40075726___redundant_placeholder06
2while_while_cond_40075726___redundant_placeholder16
2while_while_cond_40075726___redundant_placeholder26
2while_while_cond_40075726___redundant_placeholder3
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
B: : : : :         Ь:         Ь: ::::: 
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
:         Ь:.*
(
_output_shapes
:         Ь:

_output_shapes
: :

_output_shapes
:
П
И
J__inference_lstm_cell_27_layer_call_and_return_conditional_losses_40072909

inputs

states
states_12
matmul_readvariableop_resource:
ВЁ4
 matmul_1_readvariableop_resource:
ЬЁ.
biasadd_readvariableop_resource:	Ё
identity

identity_1

identity_2ИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpвMatMul_1/ReadVariableOpП
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
ВЁ*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ё2
MatMulХ
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
ЬЁ*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ё2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:         Ё2
addН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:Ё*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ё2	
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
P:         Ь:         Ь:         Ь:         Ь*
	num_split2
split`
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:         Ь2	
Sigmoidd
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:         Ь2
	Sigmoid_1]
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:         Ь2
mulW
ReluRelusplit:output:2*
T0*(
_output_shapes
:         Ь2
Relui
mul_1MulSigmoid:y:0Relu:activations:0*
T0*(
_output_shapes
:         Ь2
mul_1^
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:         Ь2
add_1d
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:         Ь2
	Sigmoid_2V
Relu_1Relu	add_1:z:0*
T0*(
_output_shapes
:         Ь2
Relu_1m
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*(
_output_shapes
:         Ь2
mul_2e
IdentityIdentity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:         Ь2

Identityi

Identity_1Identity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:         Ь2

Identity_1i

Identity_2Identity	add_1:z:0^NoOp*
T0*(
_output_shapes
:         Ь2

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
B:         В:         Ь:         Ь: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:P L
(
_output_shapes
:         В
 
_user_specified_nameinputs:PL
(
_output_shapes
:         Ь
 
_user_specified_namestates:PL
(
_output_shapes
:         Ь
 
_user_specified_namestates
Л
З
J__inference_lstm_cell_26_layer_call_and_return_conditional_losses_40072133

inputs

states
states_11
matmul_readvariableop_resource:	]И4
 matmul_1_readvariableop_resource:
ВИ.
biasadd_readvariableop_resource:	И
identity

identity_1

identity_2ИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpвMatMul_1/ReadVariableOpО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	]И*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         И2
MatMulХ
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
ВИ*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         И2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:         И2
addН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:И*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         И2	
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
P:         В:         В:         В:         В*
	num_split2
split`
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:         В2	
Sigmoidd
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:         В2
	Sigmoid_1]
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:         В2
mulW
ReluRelusplit:output:2*
T0*(
_output_shapes
:         В2
Relui
mul_1MulSigmoid:y:0Relu:activations:0*
T0*(
_output_shapes
:         В2
mul_1^
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:         В2
add_1d
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:         В2
	Sigmoid_2V
Relu_1Relu	add_1:z:0*
T0*(
_output_shapes
:         В2
Relu_1m
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*(
_output_shapes
:         В2
mul_2e
IdentityIdentity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:         В2

Identityi

Identity_1Identity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:         В2

Identity_1i

Identity_2Identity	add_1:z:0^NoOp*
T0*(
_output_shapes
:         В2

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
A:         ]:         В:         В: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:         ]
 
_user_specified_nameinputs:PL
(
_output_shapes
:         В
 
_user_specified_namestates:PL
(
_output_shapes
:         В
 
_user_specified_namestates
Д\
Ю
E__inference_lstm_26_layer_call_and_return_conditional_losses_40074105

inputs>
+lstm_cell_26_matmul_readvariableop_resource:	]ИA
-lstm_cell_26_matmul_1_readvariableop_resource:
ВИ;
,lstm_cell_26_biasadd_readvariableop_resource:	И
identityИв#lstm_cell_26/BiasAdd/ReadVariableOpв"lstm_cell_26/MatMul/ReadVariableOpв$lstm_cell_26/MatMul_1/ReadVariableOpвwhileD
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
B :В2
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
B :В2
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
:         В2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :В2
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
B :В2
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
:         В2	
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
"lstm_cell_26/MatMul/ReadVariableOpReadVariableOp+lstm_cell_26_matmul_readvariableop_resource*
_output_shapes
:	]И*
dtype02$
"lstm_cell_26/MatMul/ReadVariableOpн
lstm_cell_26/MatMulMatMulstrided_slice_2:output:0*lstm_cell_26/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         И2
lstm_cell_26/MatMul╝
$lstm_cell_26/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_26_matmul_1_readvariableop_resource* 
_output_shapes
:
ВИ*
dtype02&
$lstm_cell_26/MatMul_1/ReadVariableOpй
lstm_cell_26/MatMul_1MatMulzeros:output:0,lstm_cell_26/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         И2
lstm_cell_26/MatMul_1а
lstm_cell_26/addAddV2lstm_cell_26/MatMul:product:0lstm_cell_26/MatMul_1:product:0*
T0*(
_output_shapes
:         И2
lstm_cell_26/add┤
#lstm_cell_26/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_26_biasadd_readvariableop_resource*
_output_shapes	
:И*
dtype02%
#lstm_cell_26/BiasAdd/ReadVariableOpн
lstm_cell_26/BiasAddBiasAddlstm_cell_26/add:z:0+lstm_cell_26/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         И2
lstm_cell_26/BiasAdd~
lstm_cell_26/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_26/split/split_dimў
lstm_cell_26/splitSplit%lstm_cell_26/split/split_dim:output:0lstm_cell_26/BiasAdd:output:0*
T0*d
_output_shapesR
P:         В:         В:         В:         В*
	num_split2
lstm_cell_26/splitЗ
lstm_cell_26/SigmoidSigmoidlstm_cell_26/split:output:0*
T0*(
_output_shapes
:         В2
lstm_cell_26/SigmoidЛ
lstm_cell_26/Sigmoid_1Sigmoidlstm_cell_26/split:output:1*
T0*(
_output_shapes
:         В2
lstm_cell_26/Sigmoid_1М
lstm_cell_26/mulMullstm_cell_26/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:         В2
lstm_cell_26/mul~
lstm_cell_26/ReluRelulstm_cell_26/split:output:2*
T0*(
_output_shapes
:         В2
lstm_cell_26/ReluЭ
lstm_cell_26/mul_1Mullstm_cell_26/Sigmoid:y:0lstm_cell_26/Relu:activations:0*
T0*(
_output_shapes
:         В2
lstm_cell_26/mul_1Т
lstm_cell_26/add_1AddV2lstm_cell_26/mul:z:0lstm_cell_26/mul_1:z:0*
T0*(
_output_shapes
:         В2
lstm_cell_26/add_1Л
lstm_cell_26/Sigmoid_2Sigmoidlstm_cell_26/split:output:3*
T0*(
_output_shapes
:         В2
lstm_cell_26/Sigmoid_2}
lstm_cell_26/Relu_1Relulstm_cell_26/add_1:z:0*
T0*(
_output_shapes
:         В2
lstm_cell_26/Relu_1б
lstm_cell_26/mul_2Mullstm_cell_26/Sigmoid_2:y:0!lstm_cell_26/Relu_1:activations:0*
T0*(
_output_shapes
:         В2
lstm_cell_26/mul_2П
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    В   2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_26_matmul_readvariableop_resource-lstm_cell_26_matmul_1_readvariableop_resource,lstm_cell_26_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :         В:         В: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_40074021*
condR
while_cond_40074020*M
output_shapes<
:: : : : :         В:         В: : : : : *
parallel_iterations 2
while╡
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    В   22
0TensorArrayV2Stack/TensorListStack/element_shapeщ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:         В*
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
:         В*
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
:         В2
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
:         В2

Identity╚
NoOpNoOp$^lstm_cell_26/BiasAdd/ReadVariableOp#^lstm_cell_26/MatMul/ReadVariableOp%^lstm_cell_26/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         ]: : : 2J
#lstm_cell_26/BiasAdd/ReadVariableOp#lstm_cell_26/BiasAdd/ReadVariableOp2H
"lstm_cell_26/MatMul/ReadVariableOp"lstm_cell_26/MatMul/ReadVariableOp2L
$lstm_cell_26/MatMul_1/ReadVariableOp$lstm_cell_26/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:         ]
 
_user_specified_nameinputs
Ч
К
J__inference_lstm_cell_27_layer_call_and_return_conditional_losses_40076537

inputs
states_0
states_12
matmul_readvariableop_resource:
ВЁ4
 matmul_1_readvariableop_resource:
ЬЁ.
biasadd_readvariableop_resource:	Ё
identity

identity_1

identity_2ИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpвMatMul_1/ReadVariableOpП
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
ВЁ*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ё2
MatMulХ
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
ЬЁ*
dtype02
MatMul_1/ReadVariableOp|
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ё2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:         Ё2
addН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:Ё*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ё2	
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
P:         Ь:         Ь:         Ь:         Ь*
	num_split2
split`
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:         Ь2	
Sigmoidd
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:         Ь2
	Sigmoid_1]
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:         Ь2
mulW
ReluRelusplit:output:2*
T0*(
_output_shapes
:         Ь2
Relui
mul_1MulSigmoid:y:0Relu:activations:0*
T0*(
_output_shapes
:         Ь2
mul_1^
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:         Ь2
add_1d
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:         Ь2
	Sigmoid_2V
Relu_1Relu	add_1:z:0*
T0*(
_output_shapes
:         Ь2
Relu_1m
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*(
_output_shapes
:         Ь2
mul_2e
IdentityIdentity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:         Ь2

Identityi

Identity_1Identity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:         Ь2

Identity_1i

Identity_2Identity	add_1:z:0^NoOp*
T0*(
_output_shapes
:         Ь2

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
B:         В:         Ь:         Ь: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:P L
(
_output_shapes
:         В
 
_user_specified_nameinputs:RN
(
_output_shapes
:         Ь
"
_user_specified_name
states/0:RN
(
_output_shapes
:         Ь
"
_user_specified_name
states/1
у
═
while_cond_40073390
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_40073390___redundant_placeholder06
2while_while_cond_40073390___redundant_placeholder16
2while_while_cond_40073390___redundant_placeholder26
2while_while_cond_40073390___redundant_placeholder3
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
B: : : : :         В:         В: ::::: 
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
:         В:.*
(
_output_shapes
:         В:

_output_shapes
: :

_output_shapes
:
╨F
П
E__inference_lstm_27_layer_call_and_return_conditional_losses_40072846

inputs)
lstm_cell_27_40072764:
ВЁ)
lstm_cell_27_40072766:
ЬЁ$
lstm_cell_27_40072768:	Ё
identityИв$lstm_cell_27/StatefulPartitionedCallвwhileD
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
B :Ь2
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
B :Ь2
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
:         Ь2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :Ь2
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
B :Ь2
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
:         Ь2	
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
!:                  В2
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
valueB"    В   27
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
:         В*
shrink_axis_mask2
strided_slice_2и
$lstm_cell_27/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_27_40072764lstm_cell_27_40072766lstm_cell_27_40072768*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:         Ь:         Ь:         Ь*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_lstm_cell_27_layer_call_and_return_conditional_losses_400727632&
$lstm_cell_27/StatefulPartitionedCallП
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_27_40072764lstm_cell_27_40072766lstm_cell_27_40072768*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :         Ь:         Ь: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_40072777*
condR
while_cond_40072776*M
output_shapes<
:: : : : :         Ь:         Ь: : : : : *
parallel_iterations 2
while╡
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      22
0TensorArrayV2Stack/TensorListStack/element_shapeЄ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:                  Ь*
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
:         Ь*
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
!:                  Ь2
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
!:                  Ь2

Identity}
NoOpNoOp%^lstm_cell_27/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':                  В: : : 2L
$lstm_cell_27/StatefulPartitionedCall$lstm_cell_27/StatefulPartitionedCall2
whilewhile:] Y
5
_output_shapes#
!:                  В
 
_user_specified_nameinputs
и
ё
K__inference_sequential_13_layer_call_and_return_conditional_losses_40073693

inputs#
lstm_26_40073476:	]И$
lstm_26_40073478:
ВИ
lstm_26_40073480:	И$
lstm_27_40073641:
ВЁ$
lstm_27_40073643:
ЬЁ
lstm_27_40073645:	Ё$
dense_13_40073687:	Ь
dense_13_40073689:
identityИв dense_13/StatefulPartitionedCallвlstm_26/StatefulPartitionedCallвlstm_27/StatefulPartitionedCallо
lstm_26/StatefulPartitionedCallStatefulPartitionedCallinputslstm_26_40073476lstm_26_40073478lstm_26_40073480*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         В*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_lstm_26_layer_call_and_return_conditional_losses_400734752!
lstm_26/StatefulPartitionedCallГ
dropout_26/PartitionedCallPartitionedCall(lstm_26/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         В* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_dropout_26_layer_call_and_return_conditional_losses_400734882
dropout_26/PartitionedCall╦
lstm_27/StatefulPartitionedCallStatefulPartitionedCall#dropout_26/PartitionedCall:output:0lstm_27_40073641lstm_27_40073643lstm_27_40073645*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         Ь*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_lstm_27_layer_call_and_return_conditional_losses_400736402!
lstm_27/StatefulPartitionedCallГ
dropout_27/PartitionedCallPartitionedCall(lstm_27/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         Ь* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_dropout_27_layer_call_and_return_conditional_losses_400736532
dropout_27/PartitionedCall╗
 dense_13/StatefulPartitionedCallStatefulPartitionedCall#dropout_27/PartitionedCall:output:0dense_13_40073687dense_13_40073689*
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
GPU 2J 8В *O
fJRH
F__inference_dense_13_layer_call_and_return_conditional_losses_400736862"
 dense_13/StatefulPartitionedCallИ
IdentityIdentity)dense_13/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:         2

Identity╡
NoOpNoOp!^dense_13/StatefulPartitionedCall ^lstm_26/StatefulPartitionedCall ^lstm_27/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':         ]: : : : : : : : 2D
 dense_13/StatefulPartitionedCall dense_13/StatefulPartitionedCall2B
lstm_26/StatefulPartitionedCalllstm_26/StatefulPartitionedCall2B
lstm_27/StatefulPartitionedCalllstm_27/StatefulPartitionedCall:S O
+
_output_shapes
:         ]
 
_user_specified_nameinputs
Щ

═
0__inference_sequential_13_layer_call_fn_40074985

inputs
unknown:	]И
	unknown_0:
ВИ
	unknown_1:	И
	unknown_2:
ВЁ
	unknown_3:
ЬЁ
	unknown_4:	Ё
	unknown_5:	Ь
	unknown_6:
identityИвStatefulPartitionedCall═
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
GPU 2J 8В *T
fORM
K__inference_sequential_13_layer_call_and_return_conditional_losses_400741622
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
╢
╕
*__inference_lstm_26_layer_call_fn_40075622

inputs
unknown:	]И
	unknown_0:
ВИ
	unknown_1:	И
identityИвStatefulPartitionedCallЗ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         В*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_lstm_26_layer_call_and_return_conditional_losses_400734752
StatefulPartitionedCallА
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:         В2

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
И&
ї
while_body_40072987
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_01
while_lstm_cell_27_40073011_0:
ВЁ1
while_lstm_cell_27_40073013_0:
ЬЁ,
while_lstm_cell_27_40073015_0:	Ё
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor/
while_lstm_cell_27_40073011:
ВЁ/
while_lstm_cell_27_40073013:
ЬЁ*
while_lstm_cell_27_40073015:	ЁИв*while/lstm_cell_27/StatefulPartitionedCall├
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    В   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape╘
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:         В*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemь
*while/lstm_cell_27/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_27_40073011_0while_lstm_cell_27_40073013_0while_lstm_cell_27_40073015_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:         Ь:         Ь:         Ь*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_lstm_cell_27_layer_call_and_return_conditional_losses_400729092,
*while/lstm_cell_27/StatefulPartitionedCallў
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder3while/lstm_cell_27/StatefulPartitionedCall:output:0*
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
while/Identity_4Identity3while/lstm_cell_27/StatefulPartitionedCall:output:1^while/NoOp*
T0*(
_output_shapes
:         Ь2
while/Identity_4е
while/Identity_5Identity3while/lstm_cell_27/StatefulPartitionedCall:output:2^while/NoOp*
T0*(
_output_shapes
:         Ь2
while/Identity_5З

while/NoOpNoOp+^while/lstm_cell_27/StatefulPartitionedCall*"
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
while_lstm_cell_27_40073011while_lstm_cell_27_40073011_0"<
while_lstm_cell_27_40073013while_lstm_cell_27_40073013_0"<
while_lstm_cell_27_40073015while_lstm_cell_27_40073015_0"0
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :         Ь:         Ь: : : : : 2X
*while/lstm_cell_27/StatefulPartitionedCall*while/lstm_cell_27/StatefulPartitionedCall: 
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
:         Ь:.*
(
_output_shapes
:         Ь:

_output_shapes
: :

_output_shapes
: 
╟
∙
/__inference_lstm_cell_26_layer_call_fn_40076473

inputs
states_0
states_1
unknown:	]И
	unknown_0:
ВИ
	unknown_1:	И
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
<:         В:         В:         В*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_lstm_cell_26_layer_call_and_return_conditional_losses_400722792
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         В2

IdentityА

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:         В2

Identity_1А

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:         В2

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
A:         ]:         В:         В: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         ]
 
_user_specified_nameinputs:RN
(
_output_shapes
:         В
"
_user_specified_name
states/0:RN
(
_output_shapes
:         В
"
_user_specified_name
states/1
К\
Я
E__inference_lstm_27_layer_call_and_return_conditional_losses_40073909

inputs?
+lstm_cell_27_matmul_readvariableop_resource:
ВЁA
-lstm_cell_27_matmul_1_readvariableop_resource:
ЬЁ;
,lstm_cell_27_biasadd_readvariableop_resource:	Ё
identityИв#lstm_cell_27/BiasAdd/ReadVariableOpв"lstm_cell_27/MatMul/ReadVariableOpв$lstm_cell_27/MatMul_1/ReadVariableOpвwhileD
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
B :Ь2
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
B :Ь2
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
:         Ь2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :Ь2
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
B :Ь2
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
:         Ь2	
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
:         В2
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
valueB"    В   27
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
:         В*
shrink_axis_mask2
strided_slice_2╢
"lstm_cell_27/MatMul/ReadVariableOpReadVariableOp+lstm_cell_27_matmul_readvariableop_resource* 
_output_shapes
:
ВЁ*
dtype02$
"lstm_cell_27/MatMul/ReadVariableOpн
lstm_cell_27/MatMulMatMulstrided_slice_2:output:0*lstm_cell_27/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ё2
lstm_cell_27/MatMul╝
$lstm_cell_27/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_27_matmul_1_readvariableop_resource* 
_output_shapes
:
ЬЁ*
dtype02&
$lstm_cell_27/MatMul_1/ReadVariableOpй
lstm_cell_27/MatMul_1MatMulzeros:output:0,lstm_cell_27/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ё2
lstm_cell_27/MatMul_1а
lstm_cell_27/addAddV2lstm_cell_27/MatMul:product:0lstm_cell_27/MatMul_1:product:0*
T0*(
_output_shapes
:         Ё2
lstm_cell_27/add┤
#lstm_cell_27/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_27_biasadd_readvariableop_resource*
_output_shapes	
:Ё*
dtype02%
#lstm_cell_27/BiasAdd/ReadVariableOpн
lstm_cell_27/BiasAddBiasAddlstm_cell_27/add:z:0+lstm_cell_27/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ё2
lstm_cell_27/BiasAdd~
lstm_cell_27/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_27/split/split_dimў
lstm_cell_27/splitSplit%lstm_cell_27/split/split_dim:output:0lstm_cell_27/BiasAdd:output:0*
T0*d
_output_shapesR
P:         Ь:         Ь:         Ь:         Ь*
	num_split2
lstm_cell_27/splitЗ
lstm_cell_27/SigmoidSigmoidlstm_cell_27/split:output:0*
T0*(
_output_shapes
:         Ь2
lstm_cell_27/SigmoidЛ
lstm_cell_27/Sigmoid_1Sigmoidlstm_cell_27/split:output:1*
T0*(
_output_shapes
:         Ь2
lstm_cell_27/Sigmoid_1М
lstm_cell_27/mulMullstm_cell_27/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:         Ь2
lstm_cell_27/mul~
lstm_cell_27/ReluRelulstm_cell_27/split:output:2*
T0*(
_output_shapes
:         Ь2
lstm_cell_27/ReluЭ
lstm_cell_27/mul_1Mullstm_cell_27/Sigmoid:y:0lstm_cell_27/Relu:activations:0*
T0*(
_output_shapes
:         Ь2
lstm_cell_27/mul_1Т
lstm_cell_27/add_1AddV2lstm_cell_27/mul:z:0lstm_cell_27/mul_1:z:0*
T0*(
_output_shapes
:         Ь2
lstm_cell_27/add_1Л
lstm_cell_27/Sigmoid_2Sigmoidlstm_cell_27/split:output:3*
T0*(
_output_shapes
:         Ь2
lstm_cell_27/Sigmoid_2}
lstm_cell_27/Relu_1Relulstm_cell_27/add_1:z:0*
T0*(
_output_shapes
:         Ь2
lstm_cell_27/Relu_1б
lstm_cell_27/mul_2Mullstm_cell_27/Sigmoid_2:y:0!lstm_cell_27/Relu_1:activations:0*
T0*(
_output_shapes
:         Ь2
lstm_cell_27/mul_2П
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_27_matmul_readvariableop_resource-lstm_cell_27_matmul_1_readvariableop_resource,lstm_cell_27_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :         Ь:         Ь: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_40073825*
condR
while_cond_40073824*M
output_shapes<
:: : : : :         Ь:         Ь: : : : : *
parallel_iterations 2
while╡
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      22
0TensorArrayV2Stack/TensorListStack/element_shapeщ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:         Ь*
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
:         Ь*
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
:         Ь2
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
:         Ь2

Identity╚
NoOpNoOp$^lstm_cell_27/BiasAdd/ReadVariableOp#^lstm_cell_27/MatMul/ReadVariableOp%^lstm_cell_27/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:         В: : : 2J
#lstm_cell_27/BiasAdd/ReadVariableOp#lstm_cell_27/BiasAdd/ReadVariableOp2H
"lstm_cell_27/MatMul/ReadVariableOp"lstm_cell_27/MatMul/ReadVariableOp2L
$lstm_cell_27/MatMul_1/ReadVariableOp$lstm_cell_27/MatMul_1/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:         В
 
_user_specified_nameinputs
╘∙
М
K__inference_sequential_13_layer_call_and_return_conditional_losses_40074602

inputsF
3lstm_26_lstm_cell_26_matmul_readvariableop_resource:	]ИI
5lstm_26_lstm_cell_26_matmul_1_readvariableop_resource:
ВИC
4lstm_26_lstm_cell_26_biasadd_readvariableop_resource:	ИG
3lstm_27_lstm_cell_27_matmul_readvariableop_resource:
ВЁI
5lstm_27_lstm_cell_27_matmul_1_readvariableop_resource:
ЬЁC
4lstm_27_lstm_cell_27_biasadd_readvariableop_resource:	Ё=
*dense_13_tensordot_readvariableop_resource:	Ь6
(dense_13_biasadd_readvariableop_resource:
identityИвdense_13/BiasAdd/ReadVariableOpв!dense_13/Tensordot/ReadVariableOpв+lstm_26/lstm_cell_26/BiasAdd/ReadVariableOpв*lstm_26/lstm_cell_26/MatMul/ReadVariableOpв,lstm_26/lstm_cell_26/MatMul_1/ReadVariableOpвlstm_26/whileв+lstm_27/lstm_cell_27/BiasAdd/ReadVariableOpв*lstm_27/lstm_cell_27/MatMul/ReadVariableOpв,lstm_27/lstm_cell_27/MatMul_1/ReadVariableOpвlstm_27/whileT
lstm_26/ShapeShapeinputs*
T0*
_output_shapes
:2
lstm_26/ShapeД
lstm_26/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_26/strided_slice/stackИ
lstm_26/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_26/strided_slice/stack_1И
lstm_26/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_26/strided_slice/stack_2Т
lstm_26/strided_sliceStridedSlicelstm_26/Shape:output:0$lstm_26/strided_slice/stack:output:0&lstm_26/strided_slice/stack_1:output:0&lstm_26/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_26/strided_slicem
lstm_26/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :В2
lstm_26/zeros/mul/yМ
lstm_26/zeros/mulMullstm_26/strided_slice:output:0lstm_26/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_26/zeros/mulo
lstm_26/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
lstm_26/zeros/Less/yЗ
lstm_26/zeros/LessLesslstm_26/zeros/mul:z:0lstm_26/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_26/zeros/Lesss
lstm_26/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :В2
lstm_26/zeros/packed/1г
lstm_26/zeros/packedPacklstm_26/strided_slice:output:0lstm_26/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_26/zeros/packedo
lstm_26/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_26/zeros/ConstЦ
lstm_26/zerosFilllstm_26/zeros/packed:output:0lstm_26/zeros/Const:output:0*
T0*(
_output_shapes
:         В2
lstm_26/zerosq
lstm_26/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :В2
lstm_26/zeros_1/mul/yТ
lstm_26/zeros_1/mulMullstm_26/strided_slice:output:0lstm_26/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_26/zeros_1/muls
lstm_26/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
lstm_26/zeros_1/Less/yП
lstm_26/zeros_1/LessLesslstm_26/zeros_1/mul:z:0lstm_26/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_26/zeros_1/Lessw
lstm_26/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :В2
lstm_26/zeros_1/packed/1й
lstm_26/zeros_1/packedPacklstm_26/strided_slice:output:0!lstm_26/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_26/zeros_1/packeds
lstm_26/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_26/zeros_1/ConstЮ
lstm_26/zeros_1Filllstm_26/zeros_1/packed:output:0lstm_26/zeros_1/Const:output:0*
T0*(
_output_shapes
:         В2
lstm_26/zeros_1Е
lstm_26/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_26/transpose/permТ
lstm_26/transpose	Transposeinputslstm_26/transpose/perm:output:0*
T0*+
_output_shapes
:         ]2
lstm_26/transposeg
lstm_26/Shape_1Shapelstm_26/transpose:y:0*
T0*
_output_shapes
:2
lstm_26/Shape_1И
lstm_26/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_26/strided_slice_1/stackМ
lstm_26/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_26/strided_slice_1/stack_1М
lstm_26/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_26/strided_slice_1/stack_2Ю
lstm_26/strided_slice_1StridedSlicelstm_26/Shape_1:output:0&lstm_26/strided_slice_1/stack:output:0(lstm_26/strided_slice_1/stack_1:output:0(lstm_26/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_26/strided_slice_1Х
#lstm_26/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         2%
#lstm_26/TensorArrayV2/element_shape╥
lstm_26/TensorArrayV2TensorListReserve,lstm_26/TensorArrayV2/element_shape:output:0 lstm_26/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_26/TensorArrayV2╧
=lstm_26/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ]   2?
=lstm_26/TensorArrayUnstack/TensorListFromTensor/element_shapeШ
/lstm_26/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_26/transpose:y:0Flstm_26/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type021
/lstm_26/TensorArrayUnstack/TensorListFromTensorИ
lstm_26/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_26/strided_slice_2/stackМ
lstm_26/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_26/strided_slice_2/stack_1М
lstm_26/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_26/strided_slice_2/stack_2м
lstm_26/strided_slice_2StridedSlicelstm_26/transpose:y:0&lstm_26/strided_slice_2/stack:output:0(lstm_26/strided_slice_2/stack_1:output:0(lstm_26/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         ]*
shrink_axis_mask2
lstm_26/strided_slice_2═
*lstm_26/lstm_cell_26/MatMul/ReadVariableOpReadVariableOp3lstm_26_lstm_cell_26_matmul_readvariableop_resource*
_output_shapes
:	]И*
dtype02,
*lstm_26/lstm_cell_26/MatMul/ReadVariableOp═
lstm_26/lstm_cell_26/MatMulMatMul lstm_26/strided_slice_2:output:02lstm_26/lstm_cell_26/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         И2
lstm_26/lstm_cell_26/MatMul╘
,lstm_26/lstm_cell_26/MatMul_1/ReadVariableOpReadVariableOp5lstm_26_lstm_cell_26_matmul_1_readvariableop_resource* 
_output_shapes
:
ВИ*
dtype02.
,lstm_26/lstm_cell_26/MatMul_1/ReadVariableOp╔
lstm_26/lstm_cell_26/MatMul_1MatMullstm_26/zeros:output:04lstm_26/lstm_cell_26/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         И2
lstm_26/lstm_cell_26/MatMul_1└
lstm_26/lstm_cell_26/addAddV2%lstm_26/lstm_cell_26/MatMul:product:0'lstm_26/lstm_cell_26/MatMul_1:product:0*
T0*(
_output_shapes
:         И2
lstm_26/lstm_cell_26/add╠
+lstm_26/lstm_cell_26/BiasAdd/ReadVariableOpReadVariableOp4lstm_26_lstm_cell_26_biasadd_readvariableop_resource*
_output_shapes	
:И*
dtype02-
+lstm_26/lstm_cell_26/BiasAdd/ReadVariableOp═
lstm_26/lstm_cell_26/BiasAddBiasAddlstm_26/lstm_cell_26/add:z:03lstm_26/lstm_cell_26/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         И2
lstm_26/lstm_cell_26/BiasAddО
$lstm_26/lstm_cell_26/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2&
$lstm_26/lstm_cell_26/split/split_dimЧ
lstm_26/lstm_cell_26/splitSplit-lstm_26/lstm_cell_26/split/split_dim:output:0%lstm_26/lstm_cell_26/BiasAdd:output:0*
T0*d
_output_shapesR
P:         В:         В:         В:         В*
	num_split2
lstm_26/lstm_cell_26/splitЯ
lstm_26/lstm_cell_26/SigmoidSigmoid#lstm_26/lstm_cell_26/split:output:0*
T0*(
_output_shapes
:         В2
lstm_26/lstm_cell_26/Sigmoidг
lstm_26/lstm_cell_26/Sigmoid_1Sigmoid#lstm_26/lstm_cell_26/split:output:1*
T0*(
_output_shapes
:         В2 
lstm_26/lstm_cell_26/Sigmoid_1м
lstm_26/lstm_cell_26/mulMul"lstm_26/lstm_cell_26/Sigmoid_1:y:0lstm_26/zeros_1:output:0*
T0*(
_output_shapes
:         В2
lstm_26/lstm_cell_26/mulЦ
lstm_26/lstm_cell_26/ReluRelu#lstm_26/lstm_cell_26/split:output:2*
T0*(
_output_shapes
:         В2
lstm_26/lstm_cell_26/Relu╜
lstm_26/lstm_cell_26/mul_1Mul lstm_26/lstm_cell_26/Sigmoid:y:0'lstm_26/lstm_cell_26/Relu:activations:0*
T0*(
_output_shapes
:         В2
lstm_26/lstm_cell_26/mul_1▓
lstm_26/lstm_cell_26/add_1AddV2lstm_26/lstm_cell_26/mul:z:0lstm_26/lstm_cell_26/mul_1:z:0*
T0*(
_output_shapes
:         В2
lstm_26/lstm_cell_26/add_1г
lstm_26/lstm_cell_26/Sigmoid_2Sigmoid#lstm_26/lstm_cell_26/split:output:3*
T0*(
_output_shapes
:         В2 
lstm_26/lstm_cell_26/Sigmoid_2Х
lstm_26/lstm_cell_26/Relu_1Relulstm_26/lstm_cell_26/add_1:z:0*
T0*(
_output_shapes
:         В2
lstm_26/lstm_cell_26/Relu_1┴
lstm_26/lstm_cell_26/mul_2Mul"lstm_26/lstm_cell_26/Sigmoid_2:y:0)lstm_26/lstm_cell_26/Relu_1:activations:0*
T0*(
_output_shapes
:         В2
lstm_26/lstm_cell_26/mul_2Я
%lstm_26/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    В   2'
%lstm_26/TensorArrayV2_1/element_shape╪
lstm_26/TensorArrayV2_1TensorListReserve.lstm_26/TensorArrayV2_1/element_shape:output:0 lstm_26/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_26/TensorArrayV2_1^
lstm_26/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_26/timeП
 lstm_26/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         2"
 lstm_26/while/maximum_iterationsz
lstm_26/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_26/while/loop_counterЛ
lstm_26/whileWhile#lstm_26/while/loop_counter:output:0)lstm_26/while/maximum_iterations:output:0lstm_26/time:output:0 lstm_26/TensorArrayV2_1:handle:0lstm_26/zeros:output:0lstm_26/zeros_1:output:0 lstm_26/strided_slice_1:output:0?lstm_26/TensorArrayUnstack/TensorListFromTensor:output_handle:03lstm_26_lstm_cell_26_matmul_readvariableop_resource5lstm_26_lstm_cell_26_matmul_1_readvariableop_resource4lstm_26_lstm_cell_26_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :         В:         В: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *'
bodyR
lstm_26_while_body_40074342*'
condR
lstm_26_while_cond_40074341*M
output_shapes<
:: : : : :         В:         В: : : : : *
parallel_iterations 2
lstm_26/while┼
8lstm_26/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    В   2:
8lstm_26/TensorArrayV2Stack/TensorListStack/element_shapeЙ
*lstm_26/TensorArrayV2Stack/TensorListStackTensorListStacklstm_26/while:output:3Alstm_26/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:         В*
element_dtype02,
*lstm_26/TensorArrayV2Stack/TensorListStackС
lstm_26/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         2
lstm_26/strided_slice_3/stackМ
lstm_26/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2!
lstm_26/strided_slice_3/stack_1М
lstm_26/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_26/strided_slice_3/stack_2╦
lstm_26/strided_slice_3StridedSlice3lstm_26/TensorArrayV2Stack/TensorListStack:tensor:0&lstm_26/strided_slice_3/stack:output:0(lstm_26/strided_slice_3/stack_1:output:0(lstm_26/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         В*
shrink_axis_mask2
lstm_26/strided_slice_3Й
lstm_26/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_26/transpose_1/perm╞
lstm_26/transpose_1	Transpose3lstm_26/TensorArrayV2Stack/TensorListStack:tensor:0!lstm_26/transpose_1/perm:output:0*
T0*,
_output_shapes
:         В2
lstm_26/transpose_1v
lstm_26/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_26/runtimeЖ
dropout_26/IdentityIdentitylstm_26/transpose_1:y:0*
T0*,
_output_shapes
:         В2
dropout_26/Identityj
lstm_27/ShapeShapedropout_26/Identity:output:0*
T0*
_output_shapes
:2
lstm_27/ShapeД
lstm_27/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_27/strided_slice/stackИ
lstm_27/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_27/strided_slice/stack_1И
lstm_27/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_27/strided_slice/stack_2Т
lstm_27/strided_sliceStridedSlicelstm_27/Shape:output:0$lstm_27/strided_slice/stack:output:0&lstm_27/strided_slice/stack_1:output:0&lstm_27/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_27/strided_slicem
lstm_27/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :Ь2
lstm_27/zeros/mul/yМ
lstm_27/zeros/mulMullstm_27/strided_slice:output:0lstm_27/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_27/zeros/mulo
lstm_27/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
lstm_27/zeros/Less/yЗ
lstm_27/zeros/LessLesslstm_27/zeros/mul:z:0lstm_27/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_27/zeros/Lesss
lstm_27/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :Ь2
lstm_27/zeros/packed/1г
lstm_27/zeros/packedPacklstm_27/strided_slice:output:0lstm_27/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_27/zeros/packedo
lstm_27/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_27/zeros/ConstЦ
lstm_27/zerosFilllstm_27/zeros/packed:output:0lstm_27/zeros/Const:output:0*
T0*(
_output_shapes
:         Ь2
lstm_27/zerosq
lstm_27/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :Ь2
lstm_27/zeros_1/mul/yТ
lstm_27/zeros_1/mulMullstm_27/strided_slice:output:0lstm_27/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_27/zeros_1/muls
lstm_27/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2
lstm_27/zeros_1/Less/yП
lstm_27/zeros_1/LessLesslstm_27/zeros_1/mul:z:0lstm_27/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_27/zeros_1/Lessw
lstm_27/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :Ь2
lstm_27/zeros_1/packed/1й
lstm_27/zeros_1/packedPacklstm_27/strided_slice:output:0!lstm_27/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_27/zeros_1/packeds
lstm_27/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_27/zeros_1/ConstЮ
lstm_27/zeros_1Filllstm_27/zeros_1/packed:output:0lstm_27/zeros_1/Const:output:0*
T0*(
_output_shapes
:         Ь2
lstm_27/zeros_1Е
lstm_27/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_27/transpose/permй
lstm_27/transpose	Transposedropout_26/Identity:output:0lstm_27/transpose/perm:output:0*
T0*,
_output_shapes
:         В2
lstm_27/transposeg
lstm_27/Shape_1Shapelstm_27/transpose:y:0*
T0*
_output_shapes
:2
lstm_27/Shape_1И
lstm_27/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_27/strided_slice_1/stackМ
lstm_27/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_27/strided_slice_1/stack_1М
lstm_27/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_27/strided_slice_1/stack_2Ю
lstm_27/strided_slice_1StridedSlicelstm_27/Shape_1:output:0&lstm_27/strided_slice_1/stack:output:0(lstm_27/strided_slice_1/stack_1:output:0(lstm_27/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_27/strided_slice_1Х
#lstm_27/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         2%
#lstm_27/TensorArrayV2/element_shape╥
lstm_27/TensorArrayV2TensorListReserve,lstm_27/TensorArrayV2/element_shape:output:0 lstm_27/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_27/TensorArrayV2╧
=lstm_27/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    В   2?
=lstm_27/TensorArrayUnstack/TensorListFromTensor/element_shapeШ
/lstm_27/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_27/transpose:y:0Flstm_27/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type021
/lstm_27/TensorArrayUnstack/TensorListFromTensorИ
lstm_27/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_27/strided_slice_2/stackМ
lstm_27/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_27/strided_slice_2/stack_1М
lstm_27/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_27/strided_slice_2/stack_2н
lstm_27/strided_slice_2StridedSlicelstm_27/transpose:y:0&lstm_27/strided_slice_2/stack:output:0(lstm_27/strided_slice_2/stack_1:output:0(lstm_27/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         В*
shrink_axis_mask2
lstm_27/strided_slice_2╬
*lstm_27/lstm_cell_27/MatMul/ReadVariableOpReadVariableOp3lstm_27_lstm_cell_27_matmul_readvariableop_resource* 
_output_shapes
:
ВЁ*
dtype02,
*lstm_27/lstm_cell_27/MatMul/ReadVariableOp═
lstm_27/lstm_cell_27/MatMulMatMul lstm_27/strided_slice_2:output:02lstm_27/lstm_cell_27/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ё2
lstm_27/lstm_cell_27/MatMul╘
,lstm_27/lstm_cell_27/MatMul_1/ReadVariableOpReadVariableOp5lstm_27_lstm_cell_27_matmul_1_readvariableop_resource* 
_output_shapes
:
ЬЁ*
dtype02.
,lstm_27/lstm_cell_27/MatMul_1/ReadVariableOp╔
lstm_27/lstm_cell_27/MatMul_1MatMullstm_27/zeros:output:04lstm_27/lstm_cell_27/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ё2
lstm_27/lstm_cell_27/MatMul_1└
lstm_27/lstm_cell_27/addAddV2%lstm_27/lstm_cell_27/MatMul:product:0'lstm_27/lstm_cell_27/MatMul_1:product:0*
T0*(
_output_shapes
:         Ё2
lstm_27/lstm_cell_27/add╠
+lstm_27/lstm_cell_27/BiasAdd/ReadVariableOpReadVariableOp4lstm_27_lstm_cell_27_biasadd_readvariableop_resource*
_output_shapes	
:Ё*
dtype02-
+lstm_27/lstm_cell_27/BiasAdd/ReadVariableOp═
lstm_27/lstm_cell_27/BiasAddBiasAddlstm_27/lstm_cell_27/add:z:03lstm_27/lstm_cell_27/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ё2
lstm_27/lstm_cell_27/BiasAddО
$lstm_27/lstm_cell_27/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2&
$lstm_27/lstm_cell_27/split/split_dimЧ
lstm_27/lstm_cell_27/splitSplit-lstm_27/lstm_cell_27/split/split_dim:output:0%lstm_27/lstm_cell_27/BiasAdd:output:0*
T0*d
_output_shapesR
P:         Ь:         Ь:         Ь:         Ь*
	num_split2
lstm_27/lstm_cell_27/splitЯ
lstm_27/lstm_cell_27/SigmoidSigmoid#lstm_27/lstm_cell_27/split:output:0*
T0*(
_output_shapes
:         Ь2
lstm_27/lstm_cell_27/Sigmoidг
lstm_27/lstm_cell_27/Sigmoid_1Sigmoid#lstm_27/lstm_cell_27/split:output:1*
T0*(
_output_shapes
:         Ь2 
lstm_27/lstm_cell_27/Sigmoid_1м
lstm_27/lstm_cell_27/mulMul"lstm_27/lstm_cell_27/Sigmoid_1:y:0lstm_27/zeros_1:output:0*
T0*(
_output_shapes
:         Ь2
lstm_27/lstm_cell_27/mulЦ
lstm_27/lstm_cell_27/ReluRelu#lstm_27/lstm_cell_27/split:output:2*
T0*(
_output_shapes
:         Ь2
lstm_27/lstm_cell_27/Relu╜
lstm_27/lstm_cell_27/mul_1Mul lstm_27/lstm_cell_27/Sigmoid:y:0'lstm_27/lstm_cell_27/Relu:activations:0*
T0*(
_output_shapes
:         Ь2
lstm_27/lstm_cell_27/mul_1▓
lstm_27/lstm_cell_27/add_1AddV2lstm_27/lstm_cell_27/mul:z:0lstm_27/lstm_cell_27/mul_1:z:0*
T0*(
_output_shapes
:         Ь2
lstm_27/lstm_cell_27/add_1г
lstm_27/lstm_cell_27/Sigmoid_2Sigmoid#lstm_27/lstm_cell_27/split:output:3*
T0*(
_output_shapes
:         Ь2 
lstm_27/lstm_cell_27/Sigmoid_2Х
lstm_27/lstm_cell_27/Relu_1Relulstm_27/lstm_cell_27/add_1:z:0*
T0*(
_output_shapes
:         Ь2
lstm_27/lstm_cell_27/Relu_1┴
lstm_27/lstm_cell_27/mul_2Mul"lstm_27/lstm_cell_27/Sigmoid_2:y:0)lstm_27/lstm_cell_27/Relu_1:activations:0*
T0*(
_output_shapes
:         Ь2
lstm_27/lstm_cell_27/mul_2Я
%lstm_27/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      2'
%lstm_27/TensorArrayV2_1/element_shape╪
lstm_27/TensorArrayV2_1TensorListReserve.lstm_27/TensorArrayV2_1/element_shape:output:0 lstm_27/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_27/TensorArrayV2_1^
lstm_27/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_27/timeП
 lstm_27/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         2"
 lstm_27/while/maximum_iterationsz
lstm_27/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_27/while/loop_counterЛ
lstm_27/whileWhile#lstm_27/while/loop_counter:output:0)lstm_27/while/maximum_iterations:output:0lstm_27/time:output:0 lstm_27/TensorArrayV2_1:handle:0lstm_27/zeros:output:0lstm_27/zeros_1:output:0 lstm_27/strided_slice_1:output:0?lstm_27/TensorArrayUnstack/TensorListFromTensor:output_handle:03lstm_27_lstm_cell_27_matmul_readvariableop_resource5lstm_27_lstm_cell_27_matmul_1_readvariableop_resource4lstm_27_lstm_cell_27_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :         Ь:         Ь: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *'
bodyR
lstm_27_while_body_40074490*'
condR
lstm_27_while_cond_40074489*M
output_shapes<
:: : : : :         Ь:         Ь: : : : : *
parallel_iterations 2
lstm_27/while┼
8lstm_27/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      2:
8lstm_27/TensorArrayV2Stack/TensorListStack/element_shapeЙ
*lstm_27/TensorArrayV2Stack/TensorListStackTensorListStacklstm_27/while:output:3Alstm_27/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:         Ь*
element_dtype02,
*lstm_27/TensorArrayV2Stack/TensorListStackС
lstm_27/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         2
lstm_27/strided_slice_3/stackМ
lstm_27/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2!
lstm_27/strided_slice_3/stack_1М
lstm_27/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_27/strided_slice_3/stack_2╦
lstm_27/strided_slice_3StridedSlice3lstm_27/TensorArrayV2Stack/TensorListStack:tensor:0&lstm_27/strided_slice_3/stack:output:0(lstm_27/strided_slice_3/stack_1:output:0(lstm_27/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         Ь*
shrink_axis_mask2
lstm_27/strided_slice_3Й
lstm_27/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_27/transpose_1/perm╞
lstm_27/transpose_1	Transpose3lstm_27/TensorArrayV2Stack/TensorListStack:tensor:0!lstm_27/transpose_1/perm:output:0*
T0*,
_output_shapes
:         Ь2
lstm_27/transpose_1v
lstm_27/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_27/runtimeЖ
dropout_27/IdentityIdentitylstm_27/transpose_1:y:0*
T0*,
_output_shapes
:         Ь2
dropout_27/Identity▓
!dense_13/Tensordot/ReadVariableOpReadVariableOp*dense_13_tensordot_readvariableop_resource*
_output_shapes
:	Ь*
dtype02#
!dense_13/Tensordot/ReadVariableOp|
dense_13/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_13/Tensordot/axesГ
dense_13/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_13/Tensordot/freeА
dense_13/Tensordot/ShapeShapedropout_27/Identity:output:0*
T0*
_output_shapes
:2
dense_13/Tensordot/ShapeЖ
 dense_13/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_13/Tensordot/GatherV2/axis■
dense_13/Tensordot/GatherV2GatherV2!dense_13/Tensordot/Shape:output:0 dense_13/Tensordot/free:output:0)dense_13/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_13/Tensordot/GatherV2К
"dense_13/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"dense_13/Tensordot/GatherV2_1/axisД
dense_13/Tensordot/GatherV2_1GatherV2!dense_13/Tensordot/Shape:output:0 dense_13/Tensordot/axes:output:0+dense_13/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_13/Tensordot/GatherV2_1~
dense_13/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_13/Tensordot/Constд
dense_13/Tensordot/ProdProd$dense_13/Tensordot/GatherV2:output:0!dense_13/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_13/Tensordot/ProdВ
dense_13/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_13/Tensordot/Const_1м
dense_13/Tensordot/Prod_1Prod&dense_13/Tensordot/GatherV2_1:output:0#dense_13/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_13/Tensordot/Prod_1В
dense_13/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2 
dense_13/Tensordot/concat/axis▌
dense_13/Tensordot/concatConcatV2 dense_13/Tensordot/free:output:0 dense_13/Tensordot/axes:output:0'dense_13/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_13/Tensordot/concat░
dense_13/Tensordot/stackPack dense_13/Tensordot/Prod:output:0"dense_13/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_13/Tensordot/stack┬
dense_13/Tensordot/transpose	Transposedropout_27/Identity:output:0"dense_13/Tensordot/concat:output:0*
T0*,
_output_shapes
:         Ь2
dense_13/Tensordot/transpose├
dense_13/Tensordot/ReshapeReshape dense_13/Tensordot/transpose:y:0!dense_13/Tensordot/stack:output:0*
T0*0
_output_shapes
:                  2
dense_13/Tensordot/Reshape┬
dense_13/Tensordot/MatMulMatMul#dense_13/Tensordot/Reshape:output:0)dense_13/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_13/Tensordot/MatMulВ
dense_13/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
dense_13/Tensordot/Const_2Ж
 dense_13/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_13/Tensordot/concat_1/axisъ
dense_13/Tensordot/concat_1ConcatV2$dense_13/Tensordot/GatherV2:output:0#dense_13/Tensordot/Const_2:output:0)dense_13/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_13/Tensordot/concat_1┤
dense_13/TensordotReshape#dense_13/Tensordot/MatMul:product:0$dense_13/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:         2
dense_13/Tensordotз
dense_13/BiasAdd/ReadVariableOpReadVariableOp(dense_13_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_13/BiasAdd/ReadVariableOpл
dense_13/BiasAddBiasAdddense_13/Tensordot:output:0'dense_13/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:         2
dense_13/BiasAddА
dense_13/SoftmaxSoftmaxdense_13/BiasAdd:output:0*
T0*+
_output_shapes
:         2
dense_13/Softmaxy
IdentityIdentitydense_13/Softmax:softmax:0^NoOp*
T0*+
_output_shapes
:         2

Identity╚
NoOpNoOp ^dense_13/BiasAdd/ReadVariableOp"^dense_13/Tensordot/ReadVariableOp,^lstm_26/lstm_cell_26/BiasAdd/ReadVariableOp+^lstm_26/lstm_cell_26/MatMul/ReadVariableOp-^lstm_26/lstm_cell_26/MatMul_1/ReadVariableOp^lstm_26/while,^lstm_27/lstm_cell_27/BiasAdd/ReadVariableOp+^lstm_27/lstm_cell_27/MatMul/ReadVariableOp-^lstm_27/lstm_cell_27/MatMul_1/ReadVariableOp^lstm_27/while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':         ]: : : : : : : : 2B
dense_13/BiasAdd/ReadVariableOpdense_13/BiasAdd/ReadVariableOp2F
!dense_13/Tensordot/ReadVariableOp!dense_13/Tensordot/ReadVariableOp2Z
+lstm_26/lstm_cell_26/BiasAdd/ReadVariableOp+lstm_26/lstm_cell_26/BiasAdd/ReadVariableOp2X
*lstm_26/lstm_cell_26/MatMul/ReadVariableOp*lstm_26/lstm_cell_26/MatMul/ReadVariableOp2\
,lstm_26/lstm_cell_26/MatMul_1/ReadVariableOp,lstm_26/lstm_cell_26/MatMul_1/ReadVariableOp2
lstm_26/whilelstm_26/while2Z
+lstm_27/lstm_cell_27/BiasAdd/ReadVariableOp+lstm_27/lstm_cell_27/BiasAdd/ReadVariableOp2X
*lstm_27/lstm_cell_27/MatMul/ReadVariableOp*lstm_27/lstm_cell_27/MatMul/ReadVariableOp2\
,lstm_27/lstm_cell_27/MatMul_1/ReadVariableOp,lstm_27/lstm_cell_27/MatMul_1/ReadVariableOp2
lstm_27/whilelstm_27/while:S O
+
_output_shapes
:         ]
 
_user_specified_nameinputs
╦F
О
E__inference_lstm_26_layer_call_and_return_conditional_losses_40072216

inputs(
lstm_cell_26_40072134:	]И)
lstm_cell_26_40072136:
ВИ$
lstm_cell_26_40072138:	И
identityИв$lstm_cell_26/StatefulPartitionedCallвwhileD
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
B :В2
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
B :В2
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
:         В2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :В2
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
B :В2
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
:         В2	
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
$lstm_cell_26/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_26_40072134lstm_cell_26_40072136lstm_cell_26_40072138*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:         В:         В:         В*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_lstm_cell_26_layer_call_and_return_conditional_losses_400721332&
$lstm_cell_26/StatefulPartitionedCallП
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    В   2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_26_40072134lstm_cell_26_40072136lstm_cell_26_40072138*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :         В:         В: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_40072147*
condR
while_cond_40072146*M
output_shapes<
:: : : : :         В:         В: : : : : *
parallel_iterations 2
while╡
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    В   22
0TensorArrayV2Stack/TensorListStack/element_shapeЄ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:                  В*
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
:         В*
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
!:                  В2
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
!:                  В2

Identity}
NoOpNoOp%^lstm_cell_26/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  ]: : : 2L
$lstm_cell_26/StatefulPartitionedCall$lstm_cell_26/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :                  ]
 
_user_specified_nameinputs
╫
g
H__inference_dropout_26_layer_call_and_return_conditional_losses_40075650

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
:         В2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape╣
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:         В*
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
:         В2
dropout/GreaterEqualД
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:         В2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:         В2
dropout/Mul_1j
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:         В2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         В:T P
,
_output_shapes
:         В
 
_user_specified_nameinputs
у
═
while_cond_40072356
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_40072356___redundant_placeholder06
2while_while_cond_40072356___redundant_placeholder16
2while_while_cond_40072356___redundant_placeholder26
2while_while_cond_40072356___redundant_placeholder3
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
B: : : : :         В:         В: ::::: 
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
:         В:.*
(
_output_shapes
:         В:

_output_shapes
: :

_output_shapes
:
╨F
П
E__inference_lstm_27_layer_call_and_return_conditional_losses_40073056

inputs)
lstm_cell_27_40072974:
ВЁ)
lstm_cell_27_40072976:
ЬЁ$
lstm_cell_27_40072978:	Ё
identityИв$lstm_cell_27/StatefulPartitionedCallвwhileD
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
B :Ь2
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
B :Ь2
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
:         Ь2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :Ь2
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
B :Ь2
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
:         Ь2	
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
!:                  В2
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
valueB"    В   27
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
:         В*
shrink_axis_mask2
strided_slice_2и
$lstm_cell_27/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_27_40072974lstm_cell_27_40072976lstm_cell_27_40072978*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:         Ь:         Ь:         Ь*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_lstm_cell_27_layer_call_and_return_conditional_losses_400729092&
$lstm_cell_27/StatefulPartitionedCallП
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_27_40072974lstm_cell_27_40072976lstm_cell_27_40072978*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :         Ь:         Ь: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_40072987*
condR
while_cond_40072986*M
output_shapes<
:: : : : :         Ь:         Ь: : : : : *
parallel_iterations 2
while╡
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      22
0TensorArrayV2Stack/TensorListStack/element_shapeЄ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:                  Ь*
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
:         Ь*
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
!:                  Ь2
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
!:                  Ь2

Identity}
NoOpNoOp%^lstm_cell_27/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':                  В: : : 2L
$lstm_cell_27/StatefulPartitionedCall$lstm_cell_27/StatefulPartitionedCall2
whilewhile:] Y
5
_output_shapes#
!:                  В
 
_user_specified_nameinputs
╒!
■
F__inference_dense_13_layer_call_and_return_conditional_losses_40073686

inputs4
!tensordot_readvariableop_resource:	Ь-
biasadd_readvariableop_resource:
identityИвBiasAdd/ReadVariableOpвTensordot/ReadVariableOpЧ
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes
:	Ь*
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
:         Ь2
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
:         Ь: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:T P
,
_output_shapes
:         Ь
 
_user_specified_nameinputs
у
═
while_cond_40075877
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_40075877___redundant_placeholder06
2while_while_cond_40075877___redundant_placeholder16
2while_while_cond_40075877___redundant_placeholder26
2while_while_cond_40075877___redundant_placeholder3
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
B: : : : :         Ь:         Ь: ::::: 
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
:         Ь:.*
(
_output_shapes
:         Ь:

_output_shapes
: :

_output_shapes
:
╗
f
-__inference_dropout_26_layer_call_fn_40075660

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
:         В* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_dropout_26_layer_call_and_return_conditional_losses_400739382
StatefulPartitionedCallА
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:         В2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         В22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:         В
 
_user_specified_nameinputs
╣
╣
*__inference_lstm_27_layer_call_fn_40076308

inputs
unknown:
ВЁ
	unknown_0:
ЬЁ
	unknown_1:	Ё
identityИвStatefulPartitionedCallЗ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         Ь*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_lstm_27_layer_call_and_return_conditional_losses_400739092
StatefulPartitionedCallА
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:         Ь2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:         В: : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:         В
 
_user_specified_nameinputs
у
═
while_cond_40073824
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_40073824___redundant_placeholder06
2while_while_cond_40073824___redundant_placeholder16
2while_while_cond_40073824___redundant_placeholder26
2while_while_cond_40073824___redundant_placeholder3
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
B: : : : :         Ь:         Ь: ::::: 
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
:         Ь:.*
(
_output_shapes
:         Ь:

_output_shapes
: :

_output_shapes
:
У
Й
J__inference_lstm_cell_26_layer_call_and_return_conditional_losses_40076439

inputs
states_0
states_11
matmul_readvariableop_resource:	]И4
 matmul_1_readvariableop_resource:
ВИ.
biasadd_readvariableop_resource:	И
identity

identity_1

identity_2ИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpвMatMul_1/ReadVariableOpО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	]И*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         И2
MatMulХ
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
ВИ*
dtype02
MatMul_1/ReadVariableOp|
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         И2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:         И2
addН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:И*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         И2	
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
P:         В:         В:         В:         В*
	num_split2
split`
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:         В2	
Sigmoidd
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:         В2
	Sigmoid_1]
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:         В2
mulW
ReluRelusplit:output:2*
T0*(
_output_shapes
:         В2
Relui
mul_1MulSigmoid:y:0Relu:activations:0*
T0*(
_output_shapes
:         В2
mul_1^
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:         В2
add_1d
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:         В2
	Sigmoid_2V
Relu_1Relu	add_1:z:0*
T0*(
_output_shapes
:         В2
Relu_1m
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*(
_output_shapes
:         В2
mul_2e
IdentityIdentity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:         В2

Identityi

Identity_1Identity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:         В2

Identity_1i

Identity_2Identity	add_1:z:0^NoOp*
T0*(
_output_shapes
:         В2

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
A:         ]:         В:         В: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:         ]
 
_user_specified_nameinputs:RN
(
_output_shapes
:         В
"
_user_specified_name
states/0:RN
(
_output_shapes
:         В
"
_user_specified_name
states/1
∙
Е
)sequential_13_lstm_27_while_cond_40071945H
Dsequential_13_lstm_27_while_sequential_13_lstm_27_while_loop_counterN
Jsequential_13_lstm_27_while_sequential_13_lstm_27_while_maximum_iterations+
'sequential_13_lstm_27_while_placeholder-
)sequential_13_lstm_27_while_placeholder_1-
)sequential_13_lstm_27_while_placeholder_2-
)sequential_13_lstm_27_while_placeholder_3J
Fsequential_13_lstm_27_while_less_sequential_13_lstm_27_strided_slice_1b
^sequential_13_lstm_27_while_sequential_13_lstm_27_while_cond_40071945___redundant_placeholder0b
^sequential_13_lstm_27_while_sequential_13_lstm_27_while_cond_40071945___redundant_placeholder1b
^sequential_13_lstm_27_while_sequential_13_lstm_27_while_cond_40071945___redundant_placeholder2b
^sequential_13_lstm_27_while_sequential_13_lstm_27_while_cond_40071945___redundant_placeholder3(
$sequential_13_lstm_27_while_identity
▐
 sequential_13/lstm_27/while/LessLess'sequential_13_lstm_27_while_placeholderFsequential_13_lstm_27_while_less_sequential_13_lstm_27_strided_slice_1*
T0*
_output_shapes
: 2"
 sequential_13/lstm_27/while/LessЯ
$sequential_13/lstm_27/while/IdentityIdentity$sequential_13/lstm_27/while/Less:z:0*
T0
*
_output_shapes
: 2&
$sequential_13/lstm_27/while/Identity"U
$sequential_13_lstm_27_while_identity-sequential_13/lstm_27/while/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :         Ь:         Ь: ::::: 
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
:         Ь:.*
(
_output_shapes
:         Ь:

_output_shapes
: :

_output_shapes
:
╩
┬
K__inference_sequential_13_layer_call_and_return_conditional_losses_40074252
lstm_26_input#
lstm_26_40074230:	]И$
lstm_26_40074232:
ВИ
lstm_26_40074234:	И$
lstm_27_40074238:
ВЁ$
lstm_27_40074240:
ЬЁ
lstm_27_40074242:	Ё$
dense_13_40074246:	Ь
dense_13_40074248:
identityИв dense_13/StatefulPartitionedCallв"dropout_26/StatefulPartitionedCallв"dropout_27/StatefulPartitionedCallвlstm_26/StatefulPartitionedCallвlstm_27/StatefulPartitionedCall╡
lstm_26/StatefulPartitionedCallStatefulPartitionedCalllstm_26_inputlstm_26_40074230lstm_26_40074232lstm_26_40074234*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         В*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_lstm_26_layer_call_and_return_conditional_losses_400741052!
lstm_26/StatefulPartitionedCallЫ
"dropout_26/StatefulPartitionedCallStatefulPartitionedCall(lstm_26/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         В* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_dropout_26_layer_call_and_return_conditional_losses_400739382$
"dropout_26/StatefulPartitionedCall╙
lstm_27/StatefulPartitionedCallStatefulPartitionedCall+dropout_26/StatefulPartitionedCall:output:0lstm_27_40074238lstm_27_40074240lstm_27_40074242*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         Ь*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_lstm_27_layer_call_and_return_conditional_losses_400739092!
lstm_27/StatefulPartitionedCall└
"dropout_27/StatefulPartitionedCallStatefulPartitionedCall(lstm_27/StatefulPartitionedCall:output:0#^dropout_26/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         Ь* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_dropout_27_layer_call_and_return_conditional_losses_400737422$
"dropout_27/StatefulPartitionedCall├
 dense_13/StatefulPartitionedCallStatefulPartitionedCall+dropout_27/StatefulPartitionedCall:output:0dense_13_40074246dense_13_40074248*
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
GPU 2J 8В *O
fJRH
F__inference_dense_13_layer_call_and_return_conditional_losses_400736862"
 dense_13/StatefulPartitionedCallИ
IdentityIdentity)dense_13/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:         2

Identity 
NoOpNoOp!^dense_13/StatefulPartitionedCall#^dropout_26/StatefulPartitionedCall#^dropout_27/StatefulPartitionedCall ^lstm_26/StatefulPartitionedCall ^lstm_27/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':         ]: : : : : : : : 2D
 dense_13/StatefulPartitionedCall dense_13/StatefulPartitionedCall2H
"dropout_26/StatefulPartitionedCall"dropout_26/StatefulPartitionedCall2H
"dropout_27/StatefulPartitionedCall"dropout_27/StatefulPartitionedCall2B
lstm_26/StatefulPartitionedCalllstm_26/StatefulPartitionedCall2B
lstm_27/StatefulPartitionedCalllstm_27/StatefulPartitionedCall:Z V
+
_output_shapes
:         ]
'
_user_specified_namelstm_26_input
┤?
╓
while_body_40076180
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
3while_lstm_cell_27_matmul_readvariableop_resource_0:
ВЁI
5while_lstm_cell_27_matmul_1_readvariableop_resource_0:
ЬЁC
4while_lstm_cell_27_biasadd_readvariableop_resource_0:	Ё
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
1while_lstm_cell_27_matmul_readvariableop_resource:
ВЁG
3while_lstm_cell_27_matmul_1_readvariableop_resource:
ЬЁA
2while_lstm_cell_27_biasadd_readvariableop_resource:	ЁИв)while/lstm_cell_27/BiasAdd/ReadVariableOpв(while/lstm_cell_27/MatMul/ReadVariableOpв*while/lstm_cell_27/MatMul_1/ReadVariableOp├
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    В   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape╘
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:         В*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem╩
(while/lstm_cell_27/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_27_matmul_readvariableop_resource_0* 
_output_shapes
:
ВЁ*
dtype02*
(while/lstm_cell_27/MatMul/ReadVariableOp╫
while/lstm_cell_27/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_27/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ё2
while/lstm_cell_27/MatMul╨
*while/lstm_cell_27/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_27_matmul_1_readvariableop_resource_0* 
_output_shapes
:
ЬЁ*
dtype02,
*while/lstm_cell_27/MatMul_1/ReadVariableOp└
while/lstm_cell_27/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_27/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ё2
while/lstm_cell_27/MatMul_1╕
while/lstm_cell_27/addAddV2#while/lstm_cell_27/MatMul:product:0%while/lstm_cell_27/MatMul_1:product:0*
T0*(
_output_shapes
:         Ё2
while/lstm_cell_27/add╚
)while/lstm_cell_27/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_27_biasadd_readvariableop_resource_0*
_output_shapes	
:Ё*
dtype02+
)while/lstm_cell_27/BiasAdd/ReadVariableOp┼
while/lstm_cell_27/BiasAddBiasAddwhile/lstm_cell_27/add:z:01while/lstm_cell_27/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ё2
while/lstm_cell_27/BiasAddК
"while/lstm_cell_27/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_27/split/split_dimП
while/lstm_cell_27/splitSplit+while/lstm_cell_27/split/split_dim:output:0#while/lstm_cell_27/BiasAdd:output:0*
T0*d
_output_shapesR
P:         Ь:         Ь:         Ь:         Ь*
	num_split2
while/lstm_cell_27/splitЩ
while/lstm_cell_27/SigmoidSigmoid!while/lstm_cell_27/split:output:0*
T0*(
_output_shapes
:         Ь2
while/lstm_cell_27/SigmoidЭ
while/lstm_cell_27/Sigmoid_1Sigmoid!while/lstm_cell_27/split:output:1*
T0*(
_output_shapes
:         Ь2
while/lstm_cell_27/Sigmoid_1б
while/lstm_cell_27/mulMul while/lstm_cell_27/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:         Ь2
while/lstm_cell_27/mulР
while/lstm_cell_27/ReluRelu!while/lstm_cell_27/split:output:2*
T0*(
_output_shapes
:         Ь2
while/lstm_cell_27/Relu╡
while/lstm_cell_27/mul_1Mulwhile/lstm_cell_27/Sigmoid:y:0%while/lstm_cell_27/Relu:activations:0*
T0*(
_output_shapes
:         Ь2
while/lstm_cell_27/mul_1к
while/lstm_cell_27/add_1AddV2while/lstm_cell_27/mul:z:0while/lstm_cell_27/mul_1:z:0*
T0*(
_output_shapes
:         Ь2
while/lstm_cell_27/add_1Э
while/lstm_cell_27/Sigmoid_2Sigmoid!while/lstm_cell_27/split:output:3*
T0*(
_output_shapes
:         Ь2
while/lstm_cell_27/Sigmoid_2П
while/lstm_cell_27/Relu_1Reluwhile/lstm_cell_27/add_1:z:0*
T0*(
_output_shapes
:         Ь2
while/lstm_cell_27/Relu_1╣
while/lstm_cell_27/mul_2Mul while/lstm_cell_27/Sigmoid_2:y:0'while/lstm_cell_27/Relu_1:activations:0*
T0*(
_output_shapes
:         Ь2
while/lstm_cell_27/mul_2р
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_27/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_27/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:         Ь2
while/Identity_4О
while/Identity_5Identitywhile/lstm_cell_27/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:         Ь2
while/Identity_5▐

while/NoOpNoOp*^while/lstm_cell_27/BiasAdd/ReadVariableOp)^while/lstm_cell_27/MatMul/ReadVariableOp+^while/lstm_cell_27/MatMul_1/ReadVariableOp*"
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
2while_lstm_cell_27_biasadd_readvariableop_resource4while_lstm_cell_27_biasadd_readvariableop_resource_0"l
3while_lstm_cell_27_matmul_1_readvariableop_resource5while_lstm_cell_27_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_27_matmul_readvariableop_resource3while_lstm_cell_27_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :         Ь:         Ь: : : : : 2V
)while/lstm_cell_27/BiasAdd/ReadVariableOp)while/lstm_cell_27/BiasAdd/ReadVariableOp2T
(while/lstm_cell_27/MatMul/ReadVariableOp(while/lstm_cell_27/MatMul/ReadVariableOp2X
*while/lstm_cell_27/MatMul_1/ReadVariableOp*while/lstm_cell_27/MatMul_1/ReadVariableOp: 
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
:         Ь:.*
(
_output_shapes
:         Ь:

_output_shapes
: :

_output_shapes
: 
╫
g
H__inference_dropout_27_layer_call_and_return_conditional_losses_40076325

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
:         Ь2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape╣
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:         Ь*
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
:         Ь2
dropout/GreaterEqualД
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:         Ь2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:         Ь2
dropout/Mul_1j
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:         Ь2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         Ь:T P
,
_output_shapes
:         Ь
 
_user_specified_nameinputs
хJ
╘

lstm_26_while_body_40074342,
(lstm_26_while_lstm_26_while_loop_counter2
.lstm_26_while_lstm_26_while_maximum_iterations
lstm_26_while_placeholder
lstm_26_while_placeholder_1
lstm_26_while_placeholder_2
lstm_26_while_placeholder_3+
'lstm_26_while_lstm_26_strided_slice_1_0g
clstm_26_while_tensorarrayv2read_tensorlistgetitem_lstm_26_tensorarrayunstack_tensorlistfromtensor_0N
;lstm_26_while_lstm_cell_26_matmul_readvariableop_resource_0:	]ИQ
=lstm_26_while_lstm_cell_26_matmul_1_readvariableop_resource_0:
ВИK
<lstm_26_while_lstm_cell_26_biasadd_readvariableop_resource_0:	И
lstm_26_while_identity
lstm_26_while_identity_1
lstm_26_while_identity_2
lstm_26_while_identity_3
lstm_26_while_identity_4
lstm_26_while_identity_5)
%lstm_26_while_lstm_26_strided_slice_1e
alstm_26_while_tensorarrayv2read_tensorlistgetitem_lstm_26_tensorarrayunstack_tensorlistfromtensorL
9lstm_26_while_lstm_cell_26_matmul_readvariableop_resource:	]ИO
;lstm_26_while_lstm_cell_26_matmul_1_readvariableop_resource:
ВИI
:lstm_26_while_lstm_cell_26_biasadd_readvariableop_resource:	ИИв1lstm_26/while/lstm_cell_26/BiasAdd/ReadVariableOpв0lstm_26/while/lstm_cell_26/MatMul/ReadVariableOpв2lstm_26/while/lstm_cell_26/MatMul_1/ReadVariableOp╙
?lstm_26/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ]   2A
?lstm_26/while/TensorArrayV2Read/TensorListGetItem/element_shapeГ
1lstm_26/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemclstm_26_while_tensorarrayv2read_tensorlistgetitem_lstm_26_tensorarrayunstack_tensorlistfromtensor_0lstm_26_while_placeholderHlstm_26/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         ]*
element_dtype023
1lstm_26/while/TensorArrayV2Read/TensorListGetItemс
0lstm_26/while/lstm_cell_26/MatMul/ReadVariableOpReadVariableOp;lstm_26_while_lstm_cell_26_matmul_readvariableop_resource_0*
_output_shapes
:	]И*
dtype022
0lstm_26/while/lstm_cell_26/MatMul/ReadVariableOpў
!lstm_26/while/lstm_cell_26/MatMulMatMul8lstm_26/while/TensorArrayV2Read/TensorListGetItem:item:08lstm_26/while/lstm_cell_26/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         И2#
!lstm_26/while/lstm_cell_26/MatMulш
2lstm_26/while/lstm_cell_26/MatMul_1/ReadVariableOpReadVariableOp=lstm_26_while_lstm_cell_26_matmul_1_readvariableop_resource_0* 
_output_shapes
:
ВИ*
dtype024
2lstm_26/while/lstm_cell_26/MatMul_1/ReadVariableOpр
#lstm_26/while/lstm_cell_26/MatMul_1MatMullstm_26_while_placeholder_2:lstm_26/while/lstm_cell_26/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         И2%
#lstm_26/while/lstm_cell_26/MatMul_1╪
lstm_26/while/lstm_cell_26/addAddV2+lstm_26/while/lstm_cell_26/MatMul:product:0-lstm_26/while/lstm_cell_26/MatMul_1:product:0*
T0*(
_output_shapes
:         И2 
lstm_26/while/lstm_cell_26/addр
1lstm_26/while/lstm_cell_26/BiasAdd/ReadVariableOpReadVariableOp<lstm_26_while_lstm_cell_26_biasadd_readvariableop_resource_0*
_output_shapes	
:И*
dtype023
1lstm_26/while/lstm_cell_26/BiasAdd/ReadVariableOpх
"lstm_26/while/lstm_cell_26/BiasAddBiasAdd"lstm_26/while/lstm_cell_26/add:z:09lstm_26/while/lstm_cell_26/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         И2$
"lstm_26/while/lstm_cell_26/BiasAddЪ
*lstm_26/while/lstm_cell_26/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2,
*lstm_26/while/lstm_cell_26/split/split_dimп
 lstm_26/while/lstm_cell_26/splitSplit3lstm_26/while/lstm_cell_26/split/split_dim:output:0+lstm_26/while/lstm_cell_26/BiasAdd:output:0*
T0*d
_output_shapesR
P:         В:         В:         В:         В*
	num_split2"
 lstm_26/while/lstm_cell_26/split▒
"lstm_26/while/lstm_cell_26/SigmoidSigmoid)lstm_26/while/lstm_cell_26/split:output:0*
T0*(
_output_shapes
:         В2$
"lstm_26/while/lstm_cell_26/Sigmoid╡
$lstm_26/while/lstm_cell_26/Sigmoid_1Sigmoid)lstm_26/while/lstm_cell_26/split:output:1*
T0*(
_output_shapes
:         В2&
$lstm_26/while/lstm_cell_26/Sigmoid_1┴
lstm_26/while/lstm_cell_26/mulMul(lstm_26/while/lstm_cell_26/Sigmoid_1:y:0lstm_26_while_placeholder_3*
T0*(
_output_shapes
:         В2 
lstm_26/while/lstm_cell_26/mulи
lstm_26/while/lstm_cell_26/ReluRelu)lstm_26/while/lstm_cell_26/split:output:2*
T0*(
_output_shapes
:         В2!
lstm_26/while/lstm_cell_26/Relu╒
 lstm_26/while/lstm_cell_26/mul_1Mul&lstm_26/while/lstm_cell_26/Sigmoid:y:0-lstm_26/while/lstm_cell_26/Relu:activations:0*
T0*(
_output_shapes
:         В2"
 lstm_26/while/lstm_cell_26/mul_1╩
 lstm_26/while/lstm_cell_26/add_1AddV2"lstm_26/while/lstm_cell_26/mul:z:0$lstm_26/while/lstm_cell_26/mul_1:z:0*
T0*(
_output_shapes
:         В2"
 lstm_26/while/lstm_cell_26/add_1╡
$lstm_26/while/lstm_cell_26/Sigmoid_2Sigmoid)lstm_26/while/lstm_cell_26/split:output:3*
T0*(
_output_shapes
:         В2&
$lstm_26/while/lstm_cell_26/Sigmoid_2з
!lstm_26/while/lstm_cell_26/Relu_1Relu$lstm_26/while/lstm_cell_26/add_1:z:0*
T0*(
_output_shapes
:         В2#
!lstm_26/while/lstm_cell_26/Relu_1┘
 lstm_26/while/lstm_cell_26/mul_2Mul(lstm_26/while/lstm_cell_26/Sigmoid_2:y:0/lstm_26/while/lstm_cell_26/Relu_1:activations:0*
T0*(
_output_shapes
:         В2"
 lstm_26/while/lstm_cell_26/mul_2И
2lstm_26/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_26_while_placeholder_1lstm_26_while_placeholder$lstm_26/while/lstm_cell_26/mul_2:z:0*
_output_shapes
: *
element_dtype024
2lstm_26/while/TensorArrayV2Write/TensorListSetIteml
lstm_26/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_26/while/add/yЙ
lstm_26/while/addAddV2lstm_26_while_placeholderlstm_26/while/add/y:output:0*
T0*
_output_shapes
: 2
lstm_26/while/addp
lstm_26/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_26/while/add_1/yЮ
lstm_26/while/add_1AddV2(lstm_26_while_lstm_26_while_loop_counterlstm_26/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lstm_26/while/add_1Л
lstm_26/while/IdentityIdentitylstm_26/while/add_1:z:0^lstm_26/while/NoOp*
T0*
_output_shapes
: 2
lstm_26/while/Identityж
lstm_26/while/Identity_1Identity.lstm_26_while_lstm_26_while_maximum_iterations^lstm_26/while/NoOp*
T0*
_output_shapes
: 2
lstm_26/while/Identity_1Н
lstm_26/while/Identity_2Identitylstm_26/while/add:z:0^lstm_26/while/NoOp*
T0*
_output_shapes
: 2
lstm_26/while/Identity_2║
lstm_26/while/Identity_3IdentityBlstm_26/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_26/while/NoOp*
T0*
_output_shapes
: 2
lstm_26/while/Identity_3о
lstm_26/while/Identity_4Identity$lstm_26/while/lstm_cell_26/mul_2:z:0^lstm_26/while/NoOp*
T0*(
_output_shapes
:         В2
lstm_26/while/Identity_4о
lstm_26/while/Identity_5Identity$lstm_26/while/lstm_cell_26/add_1:z:0^lstm_26/while/NoOp*
T0*(
_output_shapes
:         В2
lstm_26/while/Identity_5Ж
lstm_26/while/NoOpNoOp2^lstm_26/while/lstm_cell_26/BiasAdd/ReadVariableOp1^lstm_26/while/lstm_cell_26/MatMul/ReadVariableOp3^lstm_26/while/lstm_cell_26/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
lstm_26/while/NoOp"9
lstm_26_while_identitylstm_26/while/Identity:output:0"=
lstm_26_while_identity_1!lstm_26/while/Identity_1:output:0"=
lstm_26_while_identity_2!lstm_26/while/Identity_2:output:0"=
lstm_26_while_identity_3!lstm_26/while/Identity_3:output:0"=
lstm_26_while_identity_4!lstm_26/while/Identity_4:output:0"=
lstm_26_while_identity_5!lstm_26/while/Identity_5:output:0"P
%lstm_26_while_lstm_26_strided_slice_1'lstm_26_while_lstm_26_strided_slice_1_0"z
:lstm_26_while_lstm_cell_26_biasadd_readvariableop_resource<lstm_26_while_lstm_cell_26_biasadd_readvariableop_resource_0"|
;lstm_26_while_lstm_cell_26_matmul_1_readvariableop_resource=lstm_26_while_lstm_cell_26_matmul_1_readvariableop_resource_0"x
9lstm_26_while_lstm_cell_26_matmul_readvariableop_resource;lstm_26_while_lstm_cell_26_matmul_readvariableop_resource_0"╚
alstm_26_while_tensorarrayv2read_tensorlistgetitem_lstm_26_tensorarrayunstack_tensorlistfromtensorclstm_26_while_tensorarrayv2read_tensorlistgetitem_lstm_26_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :         В:         В: : : : : 2f
1lstm_26/while/lstm_cell_26/BiasAdd/ReadVariableOp1lstm_26/while/lstm_cell_26/BiasAdd/ReadVariableOp2d
0lstm_26/while/lstm_cell_26/MatMul/ReadVariableOp0lstm_26/while/lstm_cell_26/MatMul/ReadVariableOp2h
2lstm_26/while/lstm_cell_26/MatMul_1/ReadVariableOp2lstm_26/while/lstm_cell_26/MatMul_1/ReadVariableOp: 
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
:         В:.*
(
_output_shapes
:         В:

_output_shapes
: :

_output_shapes
: 
щJ
╓

lstm_27_while_body_40074824,
(lstm_27_while_lstm_27_while_loop_counter2
.lstm_27_while_lstm_27_while_maximum_iterations
lstm_27_while_placeholder
lstm_27_while_placeholder_1
lstm_27_while_placeholder_2
lstm_27_while_placeholder_3+
'lstm_27_while_lstm_27_strided_slice_1_0g
clstm_27_while_tensorarrayv2read_tensorlistgetitem_lstm_27_tensorarrayunstack_tensorlistfromtensor_0O
;lstm_27_while_lstm_cell_27_matmul_readvariableop_resource_0:
ВЁQ
=lstm_27_while_lstm_cell_27_matmul_1_readvariableop_resource_0:
ЬЁK
<lstm_27_while_lstm_cell_27_biasadd_readvariableop_resource_0:	Ё
lstm_27_while_identity
lstm_27_while_identity_1
lstm_27_while_identity_2
lstm_27_while_identity_3
lstm_27_while_identity_4
lstm_27_while_identity_5)
%lstm_27_while_lstm_27_strided_slice_1e
alstm_27_while_tensorarrayv2read_tensorlistgetitem_lstm_27_tensorarrayunstack_tensorlistfromtensorM
9lstm_27_while_lstm_cell_27_matmul_readvariableop_resource:
ВЁO
;lstm_27_while_lstm_cell_27_matmul_1_readvariableop_resource:
ЬЁI
:lstm_27_while_lstm_cell_27_biasadd_readvariableop_resource:	ЁИв1lstm_27/while/lstm_cell_27/BiasAdd/ReadVariableOpв0lstm_27/while/lstm_cell_27/MatMul/ReadVariableOpв2lstm_27/while/lstm_cell_27/MatMul_1/ReadVariableOp╙
?lstm_27/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    В   2A
?lstm_27/while/TensorArrayV2Read/TensorListGetItem/element_shapeД
1lstm_27/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemclstm_27_while_tensorarrayv2read_tensorlistgetitem_lstm_27_tensorarrayunstack_tensorlistfromtensor_0lstm_27_while_placeholderHlstm_27/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:         В*
element_dtype023
1lstm_27/while/TensorArrayV2Read/TensorListGetItemт
0lstm_27/while/lstm_cell_27/MatMul/ReadVariableOpReadVariableOp;lstm_27_while_lstm_cell_27_matmul_readvariableop_resource_0* 
_output_shapes
:
ВЁ*
dtype022
0lstm_27/while/lstm_cell_27/MatMul/ReadVariableOpў
!lstm_27/while/lstm_cell_27/MatMulMatMul8lstm_27/while/TensorArrayV2Read/TensorListGetItem:item:08lstm_27/while/lstm_cell_27/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ё2#
!lstm_27/while/lstm_cell_27/MatMulш
2lstm_27/while/lstm_cell_27/MatMul_1/ReadVariableOpReadVariableOp=lstm_27_while_lstm_cell_27_matmul_1_readvariableop_resource_0* 
_output_shapes
:
ЬЁ*
dtype024
2lstm_27/while/lstm_cell_27/MatMul_1/ReadVariableOpр
#lstm_27/while/lstm_cell_27/MatMul_1MatMullstm_27_while_placeholder_2:lstm_27/while/lstm_cell_27/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ё2%
#lstm_27/while/lstm_cell_27/MatMul_1╪
lstm_27/while/lstm_cell_27/addAddV2+lstm_27/while/lstm_cell_27/MatMul:product:0-lstm_27/while/lstm_cell_27/MatMul_1:product:0*
T0*(
_output_shapes
:         Ё2 
lstm_27/while/lstm_cell_27/addр
1lstm_27/while/lstm_cell_27/BiasAdd/ReadVariableOpReadVariableOp<lstm_27_while_lstm_cell_27_biasadd_readvariableop_resource_0*
_output_shapes	
:Ё*
dtype023
1lstm_27/while/lstm_cell_27/BiasAdd/ReadVariableOpх
"lstm_27/while/lstm_cell_27/BiasAddBiasAdd"lstm_27/while/lstm_cell_27/add:z:09lstm_27/while/lstm_cell_27/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ё2$
"lstm_27/while/lstm_cell_27/BiasAddЪ
*lstm_27/while/lstm_cell_27/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2,
*lstm_27/while/lstm_cell_27/split/split_dimп
 lstm_27/while/lstm_cell_27/splitSplit3lstm_27/while/lstm_cell_27/split/split_dim:output:0+lstm_27/while/lstm_cell_27/BiasAdd:output:0*
T0*d
_output_shapesR
P:         Ь:         Ь:         Ь:         Ь*
	num_split2"
 lstm_27/while/lstm_cell_27/split▒
"lstm_27/while/lstm_cell_27/SigmoidSigmoid)lstm_27/while/lstm_cell_27/split:output:0*
T0*(
_output_shapes
:         Ь2$
"lstm_27/while/lstm_cell_27/Sigmoid╡
$lstm_27/while/lstm_cell_27/Sigmoid_1Sigmoid)lstm_27/while/lstm_cell_27/split:output:1*
T0*(
_output_shapes
:         Ь2&
$lstm_27/while/lstm_cell_27/Sigmoid_1┴
lstm_27/while/lstm_cell_27/mulMul(lstm_27/while/lstm_cell_27/Sigmoid_1:y:0lstm_27_while_placeholder_3*
T0*(
_output_shapes
:         Ь2 
lstm_27/while/lstm_cell_27/mulи
lstm_27/while/lstm_cell_27/ReluRelu)lstm_27/while/lstm_cell_27/split:output:2*
T0*(
_output_shapes
:         Ь2!
lstm_27/while/lstm_cell_27/Relu╒
 lstm_27/while/lstm_cell_27/mul_1Mul&lstm_27/while/lstm_cell_27/Sigmoid:y:0-lstm_27/while/lstm_cell_27/Relu:activations:0*
T0*(
_output_shapes
:         Ь2"
 lstm_27/while/lstm_cell_27/mul_1╩
 lstm_27/while/lstm_cell_27/add_1AddV2"lstm_27/while/lstm_cell_27/mul:z:0$lstm_27/while/lstm_cell_27/mul_1:z:0*
T0*(
_output_shapes
:         Ь2"
 lstm_27/while/lstm_cell_27/add_1╡
$lstm_27/while/lstm_cell_27/Sigmoid_2Sigmoid)lstm_27/while/lstm_cell_27/split:output:3*
T0*(
_output_shapes
:         Ь2&
$lstm_27/while/lstm_cell_27/Sigmoid_2з
!lstm_27/while/lstm_cell_27/Relu_1Relu$lstm_27/while/lstm_cell_27/add_1:z:0*
T0*(
_output_shapes
:         Ь2#
!lstm_27/while/lstm_cell_27/Relu_1┘
 lstm_27/while/lstm_cell_27/mul_2Mul(lstm_27/while/lstm_cell_27/Sigmoid_2:y:0/lstm_27/while/lstm_cell_27/Relu_1:activations:0*
T0*(
_output_shapes
:         Ь2"
 lstm_27/while/lstm_cell_27/mul_2И
2lstm_27/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_27_while_placeholder_1lstm_27_while_placeholder$lstm_27/while/lstm_cell_27/mul_2:z:0*
_output_shapes
: *
element_dtype024
2lstm_27/while/TensorArrayV2Write/TensorListSetIteml
lstm_27/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_27/while/add/yЙ
lstm_27/while/addAddV2lstm_27_while_placeholderlstm_27/while/add/y:output:0*
T0*
_output_shapes
: 2
lstm_27/while/addp
lstm_27/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_27/while/add_1/yЮ
lstm_27/while/add_1AddV2(lstm_27_while_lstm_27_while_loop_counterlstm_27/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lstm_27/while/add_1Л
lstm_27/while/IdentityIdentitylstm_27/while/add_1:z:0^lstm_27/while/NoOp*
T0*
_output_shapes
: 2
lstm_27/while/Identityж
lstm_27/while/Identity_1Identity.lstm_27_while_lstm_27_while_maximum_iterations^lstm_27/while/NoOp*
T0*
_output_shapes
: 2
lstm_27/while/Identity_1Н
lstm_27/while/Identity_2Identitylstm_27/while/add:z:0^lstm_27/while/NoOp*
T0*
_output_shapes
: 2
lstm_27/while/Identity_2║
lstm_27/while/Identity_3IdentityBlstm_27/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_27/while/NoOp*
T0*
_output_shapes
: 2
lstm_27/while/Identity_3о
lstm_27/while/Identity_4Identity$lstm_27/while/lstm_cell_27/mul_2:z:0^lstm_27/while/NoOp*
T0*(
_output_shapes
:         Ь2
lstm_27/while/Identity_4о
lstm_27/while/Identity_5Identity$lstm_27/while/lstm_cell_27/add_1:z:0^lstm_27/while/NoOp*
T0*(
_output_shapes
:         Ь2
lstm_27/while/Identity_5Ж
lstm_27/while/NoOpNoOp2^lstm_27/while/lstm_cell_27/BiasAdd/ReadVariableOp1^lstm_27/while/lstm_cell_27/MatMul/ReadVariableOp3^lstm_27/while/lstm_cell_27/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
lstm_27/while/NoOp"9
lstm_27_while_identitylstm_27/while/Identity:output:0"=
lstm_27_while_identity_1!lstm_27/while/Identity_1:output:0"=
lstm_27_while_identity_2!lstm_27/while/Identity_2:output:0"=
lstm_27_while_identity_3!lstm_27/while/Identity_3:output:0"=
lstm_27_while_identity_4!lstm_27/while/Identity_4:output:0"=
lstm_27_while_identity_5!lstm_27/while/Identity_5:output:0"P
%lstm_27_while_lstm_27_strided_slice_1'lstm_27_while_lstm_27_strided_slice_1_0"z
:lstm_27_while_lstm_cell_27_biasadd_readvariableop_resource<lstm_27_while_lstm_cell_27_biasadd_readvariableop_resource_0"|
;lstm_27_while_lstm_cell_27_matmul_1_readvariableop_resource=lstm_27_while_lstm_cell_27_matmul_1_readvariableop_resource_0"x
9lstm_27_while_lstm_cell_27_matmul_readvariableop_resource;lstm_27_while_lstm_cell_27_matmul_readvariableop_resource_0"╚
alstm_27_while_tensorarrayv2read_tensorlistgetitem_lstm_27_tensorarrayunstack_tensorlistfromtensorclstm_27_while_tensorarrayv2read_tensorlistgetitem_lstm_27_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :         Ь:         Ь: : : : : 2f
1lstm_27/while/lstm_cell_27/BiasAdd/ReadVariableOp1lstm_27/while/lstm_cell_27/BiasAdd/ReadVariableOp2d
0lstm_27/while/lstm_cell_27/MatMul/ReadVariableOp0lstm_27/while/lstm_cell_27/MatMul/ReadVariableOp2h
2lstm_27/while/lstm_cell_27/MatMul_1/ReadVariableOp2lstm_27/while/lstm_cell_27/MatMul_1/ReadVariableOp: 
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
:         Ь:.*
(
_output_shapes
:         Ь:

_output_shapes
: :

_output_shapes
: 
╒!
■
F__inference_dense_13_layer_call_and_return_conditional_losses_40076366

inputs4
!tensordot_readvariableop_resource:	Ь-
biasadd_readvariableop_resource:
identityИвBiasAdd/ReadVariableOpвTensordot/ReadVariableOpЧ
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes
:	Ь*
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
:         Ь2
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
:         Ь: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:T P
,
_output_shapes
:         Ь
 
_user_specified_nameinputs"иL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*┐
serving_defaultл
K
lstm_26_input:
serving_default_lstm_26_input:0         ]@
dense_134
StatefulPartitionedCall:0         tensorflow/serving/predict:О│
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
": 	Ь2dense_13/kernel
:2dense_13/bias
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
.:,	]И2lstm_26/lstm_cell_26/kernel
9:7
ВИ2%lstm_26/lstm_cell_26/recurrent_kernel
(:&И2lstm_26/lstm_cell_26/bias
/:-
ВЁ2lstm_27/lstm_cell_27/kernel
9:7
ЬЁ2%lstm_27/lstm_cell_27/recurrent_kernel
(:&Ё2lstm_27/lstm_cell_27/bias
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
#__inference__wrapped_model_40072058lstm_26_input"Ш
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
·2ў
K__inference_sequential_13_layer_call_and_return_conditional_losses_40074602
K__inference_sequential_13_layer_call_and_return_conditional_losses_40074943
K__inference_sequential_13_layer_call_and_return_conditional_losses_40074227
K__inference_sequential_13_layer_call_and_return_conditional_losses_40074252└
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
О2Л
0__inference_sequential_13_layer_call_fn_40073712
0__inference_sequential_13_layer_call_fn_40074964
0__inference_sequential_13_layer_call_fn_40074985
0__inference_sequential_13_layer_call_fn_40074202└
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
E__inference_lstm_26_layer_call_and_return_conditional_losses_40075136
E__inference_lstm_26_layer_call_and_return_conditional_losses_40075287
E__inference_lstm_26_layer_call_and_return_conditional_losses_40075438
E__inference_lstm_26_layer_call_and_return_conditional_losses_40075589╒
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
*__inference_lstm_26_layer_call_fn_40075600
*__inference_lstm_26_layer_call_fn_40075611
*__inference_lstm_26_layer_call_fn_40075622
*__inference_lstm_26_layer_call_fn_40075633╒
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
H__inference_dropout_26_layer_call_and_return_conditional_losses_40075638
H__inference_dropout_26_layer_call_and_return_conditional_losses_40075650┤
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
-__inference_dropout_26_layer_call_fn_40075655
-__inference_dropout_26_layer_call_fn_40075660┤
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
E__inference_lstm_27_layer_call_and_return_conditional_losses_40075811
E__inference_lstm_27_layer_call_and_return_conditional_losses_40075962
E__inference_lstm_27_layer_call_and_return_conditional_losses_40076113
E__inference_lstm_27_layer_call_and_return_conditional_losses_40076264╒
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
*__inference_lstm_27_layer_call_fn_40076275
*__inference_lstm_27_layer_call_fn_40076286
*__inference_lstm_27_layer_call_fn_40076297
*__inference_lstm_27_layer_call_fn_40076308╒
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
H__inference_dropout_27_layer_call_and_return_conditional_losses_40076313
H__inference_dropout_27_layer_call_and_return_conditional_losses_40076325┤
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
-__inference_dropout_27_layer_call_fn_40076330
-__inference_dropout_27_layer_call_fn_40076335┤
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
Ё2э
F__inference_dense_13_layer_call_and_return_conditional_losses_40076366в
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
╒2╥
+__inference_dense_13_layer_call_fn_40076375в
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
&__inference_signature_wrapper_40074275lstm_26_input"Ф
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
J__inference_lstm_cell_26_layer_call_and_return_conditional_losses_40076407
J__inference_lstm_cell_26_layer_call_and_return_conditional_losses_40076439╛
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
/__inference_lstm_cell_26_layer_call_fn_40076456
/__inference_lstm_cell_26_layer_call_fn_40076473╛
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
J__inference_lstm_cell_27_layer_call_and_return_conditional_losses_40076505
J__inference_lstm_cell_27_layer_call_and_return_conditional_losses_40076537╛
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
/__inference_lstm_cell_27_layer_call_fn_40076554
/__inference_lstm_cell_27_layer_call_fn_40076571╛
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
 ж
#__inference__wrapped_model_40072058&'()*+ !:в7
0в-
+К(
lstm_26_input         ]
к "7к4
2
dense_13&К#
dense_13         п
F__inference_dense_13_layer_call_and_return_conditional_losses_40076366e !4в1
*в'
%К"
inputs         Ь
к ")в&
К
0         
Ъ З
+__inference_dense_13_layer_call_fn_40076375X !4в1
*в'
%К"
inputs         Ь
к "К         ▓
H__inference_dropout_26_layer_call_and_return_conditional_losses_40075638f8в5
.в+
%К"
inputs         В
p 
к "*в'
 К
0         В
Ъ ▓
H__inference_dropout_26_layer_call_and_return_conditional_losses_40075650f8в5
.в+
%К"
inputs         В
p
к "*в'
 К
0         В
Ъ К
-__inference_dropout_26_layer_call_fn_40075655Y8в5
.в+
%К"
inputs         В
p 
к "К         ВК
-__inference_dropout_26_layer_call_fn_40075660Y8в5
.в+
%К"
inputs         В
p
к "К         В▓
H__inference_dropout_27_layer_call_and_return_conditional_losses_40076313f8в5
.в+
%К"
inputs         Ь
p 
к "*в'
 К
0         Ь
Ъ ▓
H__inference_dropout_27_layer_call_and_return_conditional_losses_40076325f8в5
.в+
%К"
inputs         Ь
p
к "*в'
 К
0         Ь
Ъ К
-__inference_dropout_27_layer_call_fn_40076330Y8в5
.в+
%К"
inputs         Ь
p 
к "К         ЬК
-__inference_dropout_27_layer_call_fn_40076335Y8в5
.в+
%К"
inputs         Ь
p
к "К         Ь╒
E__inference_lstm_26_layer_call_and_return_conditional_losses_40075136Л&'(OвL
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
0                  В
Ъ ╒
E__inference_lstm_26_layer_call_and_return_conditional_losses_40075287Л&'(OвL
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
0                  В
Ъ ╗
E__inference_lstm_26_layer_call_and_return_conditional_losses_40075438r&'(?в<
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
0         В
Ъ ╗
E__inference_lstm_26_layer_call_and_return_conditional_losses_40075589r&'(?в<
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
0         В
Ъ м
*__inference_lstm_26_layer_call_fn_40075600~&'(OвL
EвB
4Ъ1
/К,
inputs/0                  ]

 
p 

 
к "&К#                  Вм
*__inference_lstm_26_layer_call_fn_40075611~&'(OвL
EвB
4Ъ1
/К,
inputs/0                  ]

 
p

 
к "&К#                  ВУ
*__inference_lstm_26_layer_call_fn_40075622e&'(?в<
5в2
$К!
inputs         ]

 
p 

 
к "К         ВУ
*__inference_lstm_26_layer_call_fn_40075633e&'(?в<
5в2
$К!
inputs         ]

 
p

 
к "К         В╓
E__inference_lstm_27_layer_call_and_return_conditional_losses_40075811М)*+PвM
FвC
5Ъ2
0К-
inputs/0                  В

 
p 

 
к "3в0
)К&
0                  Ь
Ъ ╓
E__inference_lstm_27_layer_call_and_return_conditional_losses_40075962М)*+PвM
FвC
5Ъ2
0К-
inputs/0                  В

 
p

 
к "3в0
)К&
0                  Ь
Ъ ╝
E__inference_lstm_27_layer_call_and_return_conditional_losses_40076113s)*+@в=
6в3
%К"
inputs         В

 
p 

 
к "*в'
 К
0         Ь
Ъ ╝
E__inference_lstm_27_layer_call_and_return_conditional_losses_40076264s)*+@в=
6в3
%К"
inputs         В

 
p

 
к "*в'
 К
0         Ь
Ъ н
*__inference_lstm_27_layer_call_fn_40076275)*+PвM
FвC
5Ъ2
0К-
inputs/0                  В

 
p 

 
к "&К#                  Ьн
*__inference_lstm_27_layer_call_fn_40076286)*+PвM
FвC
5Ъ2
0К-
inputs/0                  В

 
p

 
к "&К#                  ЬФ
*__inference_lstm_27_layer_call_fn_40076297f)*+@в=
6в3
%К"
inputs         В

 
p 

 
к "К         ЬФ
*__inference_lstm_27_layer_call_fn_40076308f)*+@в=
6в3
%К"
inputs         В

 
p

 
к "К         Ь╤
J__inference_lstm_cell_26_layer_call_and_return_conditional_losses_40076407В&'(Вв
xвu
 К
inputs         ]
MвJ
#К 
states/0         В
#К 
states/1         В
p 
к "vвs
lвi
К
0/0         В
GЪD
 К
0/1/0         В
 К
0/1/1         В
Ъ ╤
J__inference_lstm_cell_26_layer_call_and_return_conditional_losses_40076439В&'(Вв
xвu
 К
inputs         ]
MвJ
#К 
states/0         В
#К 
states/1         В
p
к "vвs
lвi
К
0/0         В
GЪD
 К
0/1/0         В
 К
0/1/1         В
Ъ ж
/__inference_lstm_cell_26_layer_call_fn_40076456Є&'(Вв
xвu
 К
inputs         ]
MвJ
#К 
states/0         В
#К 
states/1         В
p 
к "fвc
К
0         В
CЪ@
К
1/0         В
К
1/1         Вж
/__inference_lstm_cell_26_layer_call_fn_40076473Є&'(Вв
xвu
 К
inputs         ]
MвJ
#К 
states/0         В
#К 
states/1         В
p
к "fвc
К
0         В
CЪ@
К
1/0         В
К
1/1         В╙
J__inference_lstm_cell_27_layer_call_and_return_conditional_losses_40076505Д)*+ДвА
yвv
!К
inputs         В
MвJ
#К 
states/0         Ь
#К 
states/1         Ь
p 
к "vвs
lвi
К
0/0         Ь
GЪD
 К
0/1/0         Ь
 К
0/1/1         Ь
Ъ ╙
J__inference_lstm_cell_27_layer_call_and_return_conditional_losses_40076537Д)*+ДвА
yвv
!К
inputs         В
MвJ
#К 
states/0         Ь
#К 
states/1         Ь
p
к "vвs
lвi
К
0/0         Ь
GЪD
 К
0/1/0         Ь
 К
0/1/1         Ь
Ъ и
/__inference_lstm_cell_27_layer_call_fn_40076554Ї)*+ДвА
yвv
!К
inputs         В
MвJ
#К 
states/0         Ь
#К 
states/1         Ь
p 
к "fвc
К
0         Ь
CЪ@
К
1/0         Ь
К
1/1         Ьи
/__inference_lstm_cell_27_layer_call_fn_40076571Ї)*+ДвА
yвv
!К
inputs         В
MвJ
#К 
states/0         Ь
#К 
states/1         Ь
p
к "fвc
К
0         Ь
CЪ@
К
1/0         Ь
К
1/1         Ь╚
K__inference_sequential_13_layer_call_and_return_conditional_losses_40074227y&'()*+ !Bв?
8в5
+К(
lstm_26_input         ]
p 

 
к ")в&
К
0         
Ъ ╚
K__inference_sequential_13_layer_call_and_return_conditional_losses_40074252y&'()*+ !Bв?
8в5
+К(
lstm_26_input         ]
p

 
к ")в&
К
0         
Ъ ┴
K__inference_sequential_13_layer_call_and_return_conditional_losses_40074602r&'()*+ !;в8
1в.
$К!
inputs         ]
p 

 
к ")в&
К
0         
Ъ ┴
K__inference_sequential_13_layer_call_and_return_conditional_losses_40074943r&'()*+ !;в8
1в.
$К!
inputs         ]
p

 
к ")в&
К
0         
Ъ а
0__inference_sequential_13_layer_call_fn_40073712l&'()*+ !Bв?
8в5
+К(
lstm_26_input         ]
p 

 
к "К         а
0__inference_sequential_13_layer_call_fn_40074202l&'()*+ !Bв?
8в5
+К(
lstm_26_input         ]
p

 
к "К         Щ
0__inference_sequential_13_layer_call_fn_40074964e&'()*+ !;в8
1в.
$К!
inputs         ]
p 

 
к "К         Щ
0__inference_sequential_13_layer_call_fn_40074985e&'()*+ !;в8
1в.
$К!
inputs         ]
p

 
к "К         ╗
&__inference_signature_wrapper_40074275Р&'()*+ !KвH
в 
Aк>
<
lstm_26_input+К(
lstm_26_input         ]"7к4
2
dense_13&К#
dense_13         