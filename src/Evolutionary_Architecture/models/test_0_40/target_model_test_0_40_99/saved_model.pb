ЛХ&
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
И"serve*2.6.02v2.6.0-rc2-32-g919f693420e8¤┴$
z
dense_11/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:m* 
shared_namedense_11/kernel
s
#dense_11/kernel/Read/ReadVariableOpReadVariableOpdense_11/kernel*
_output_shapes

:m*
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
У
lstm_22/lstm_cell_22/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	]Р*,
shared_namelstm_22/lstm_cell_22/kernel
М
/lstm_22/lstm_cell_22/kernel/Read/ReadVariableOpReadVariableOplstm_22/lstm_cell_22/kernel*
_output_shapes
:	]Р*
dtype0
и
%lstm_22/lstm_cell_22/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ДР*6
shared_name'%lstm_22/lstm_cell_22/recurrent_kernel
б
9lstm_22/lstm_cell_22/recurrent_kernel/Read/ReadVariableOpReadVariableOp%lstm_22/lstm_cell_22/recurrent_kernel* 
_output_shapes
:
ДР*
dtype0
Л
lstm_22/lstm_cell_22/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:Р**
shared_namelstm_22/lstm_cell_22/bias
Д
-lstm_22/lstm_cell_22/bias/Read/ReadVariableOpReadVariableOplstm_22/lstm_cell_22/bias*
_output_shapes	
:Р*
dtype0
Ф
lstm_23/lstm_cell_23/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
Д┤*,
shared_namelstm_23/lstm_cell_23/kernel
Н
/lstm_23/lstm_cell_23/kernel/Read/ReadVariableOpReadVariableOplstm_23/lstm_cell_23/kernel* 
_output_shapes
:
Д┤*
dtype0
з
%lstm_23/lstm_cell_23/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	m┤*6
shared_name'%lstm_23/lstm_cell_23/recurrent_kernel
а
9lstm_23/lstm_cell_23/recurrent_kernel/Read/ReadVariableOpReadVariableOp%lstm_23/lstm_cell_23/recurrent_kernel*
_output_shapes
:	m┤*
dtype0
Л
lstm_23/lstm_cell_23/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:┤**
shared_namelstm_23/lstm_cell_23/bias
Д
-lstm_23/lstm_cell_23/bias/Read/ReadVariableOpReadVariableOplstm_23/lstm_cell_23/bias*
_output_shapes	
:┤*
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
	variables
trainable_variables
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
	variables
trainable_variables
regularization_losses
	keras_api
h

 kernel
!bias
"	variables
#trainable_variables
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

,layers
-layer_metrics
.layer_regularization_losses
/non_trainable_variables
0metrics
trainable_variables
	variables
	regularization_losses
 
О
1
state_size

&kernel
'recurrent_kernel
(bias
2	variables
3trainable_variables
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

6states

7layers
8layer_metrics
9layer_regularization_losses
:non_trainable_variables
;metrics
trainable_variables
	variables
regularization_losses
 
 
 
н

<layers
	variables
=layer_metrics
>non_trainable_variables
?metrics
trainable_variables
@layer_regularization_losses
regularization_losses
О
A
state_size

)kernel
*recurrent_kernel
+bias
B	variables
Ctrainable_variables
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

Fstates

Glayers
Hlayer_metrics
Ilayer_regularization_losses
Jnon_trainable_variables
Kmetrics
trainable_variables
	variables
regularization_losses
 
 
 
н

Llayers
	variables
Mlayer_metrics
Nnon_trainable_variables
Ometrics
trainable_variables
Player_regularization_losses
regularization_losses
[Y
VARIABLE_VALUEdense_11/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_11/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

 0
!1

 0
!1
 
н

Qlayers
"	variables
Rlayer_metrics
Snon_trainable_variables
Tmetrics
#trainable_variables
Ulayer_regularization_losses
$regularization_losses
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
#
0
1
2
3
4
 
 
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

Xlayers
2	variables
Ylayer_metrics
Znon_trainable_variables
[metrics
3trainable_variables
\layer_regularization_losses
4regularization_losses
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

]layers
B	variables
^layer_metrics
_non_trainable_variables
`metrics
Ctrainable_variables
alayer_regularization_losses
Dregularization_losses
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
serving_default_lstm_22_inputPlaceholder*+
_output_shapes
:         ]*
dtype0* 
shape:         ]
м
StatefulPartitionedCallStatefulPartitionedCallserving_default_lstm_22_inputlstm_22/lstm_cell_22/kernel%lstm_22/lstm_cell_22/recurrent_kernellstm_22/lstm_cell_22/biaslstm_23/lstm_cell_23/kernel%lstm_23/lstm_cell_23/recurrent_kernellstm_23/lstm_cell_23/biasdense_11/kerneldense_11/bias*
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
&__inference_signature_wrapper_39110309
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
Ч
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
GPU 2J 8В **
f%R#
!__inference__traced_save_39112664
в
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
GPU 2J 8В *-
f(R&
$__inference__traced_restore_39112710Шў#
╧
g
H__inference_dropout_23_layer_call_and_return_conditional_losses_39109776

inputs
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  а?2
dropout/Constw
dropout/MulMulinputsdropout/Const:output:0*
T0*+
_output_shapes
:         m2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape╕
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:         m*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>2
dropout/GreaterEqual/y┬
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:         m2
dropout/GreaterEqualГ
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:         m2
dropout/Cast~
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*+
_output_shapes
:         m2
dropout/Mul_1i
IdentityIdentitydropout/Mul_1:z:0*
T0*+
_output_shapes
:         m2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         m:S O
+
_output_shapes
:         m
 
_user_specified_nameinputs
Й
f
H__inference_dropout_22_layer_call_and_return_conditional_losses_39109522

inputs

identity_1_
IdentityIdentityinputs*
T0*,
_output_shapes
:         Д2

Identityn

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:         Д2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         Д:T P
,
_output_shapes
:         Д
 
_user_specified_nameinputs
╞
└
K__inference_sequential_11_layer_call_and_return_conditional_losses_39110286
lstm_22_input#
lstm_22_39110264:	]Р$
lstm_22_39110266:
ДР
lstm_22_39110268:	Р$
lstm_23_39110272:
Д┤#
lstm_23_39110274:	m┤
lstm_23_39110276:	┤#
dense_11_39110280:m
dense_11_39110282:
identityИв dense_11/StatefulPartitionedCallв"dropout_22/StatefulPartitionedCallв"dropout_23/StatefulPartitionedCallвlstm_22/StatefulPartitionedCallвlstm_23/StatefulPartitionedCall╡
lstm_22/StatefulPartitionedCallStatefulPartitionedCalllstm_22_inputlstm_22_39110264lstm_22_39110266lstm_22_39110268*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         Д*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_lstm_22_layer_call_and_return_conditional_losses_391101392!
lstm_22/StatefulPartitionedCallЫ
"dropout_22/StatefulPartitionedCallStatefulPartitionedCall(lstm_22/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         Д* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_dropout_22_layer_call_and_return_conditional_losses_391099722$
"dropout_22/StatefulPartitionedCall╥
lstm_23/StatefulPartitionedCallStatefulPartitionedCall+dropout_22/StatefulPartitionedCall:output:0lstm_23_39110272lstm_23_39110274lstm_23_39110276*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         m*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_lstm_23_layer_call_and_return_conditional_losses_391099432!
lstm_23/StatefulPartitionedCall┐
"dropout_23/StatefulPartitionedCallStatefulPartitionedCall(lstm_23/StatefulPartitionedCall:output:0#^dropout_22/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         m* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_dropout_23_layer_call_and_return_conditional_losses_391097762$
"dropout_23/StatefulPartitionedCall├
 dense_11/StatefulPartitionedCallStatefulPartitionedCall+dropout_23/StatefulPartitionedCall:output:0dense_11_39110280dense_11_39110282*
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
F__inference_dense_11_layer_call_and_return_conditional_losses_391097202"
 dense_11/StatefulPartitionedCallИ
IdentityIdentity)dense_11/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:         2

Identity 
NoOpNoOp!^dense_11/StatefulPartitionedCall#^dropout_22/StatefulPartitionedCall#^dropout_23/StatefulPartitionedCall ^lstm_22/StatefulPartitionedCall ^lstm_23/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':         ]: : : : : : : : 2D
 dense_11/StatefulPartitionedCall dense_11/StatefulPartitionedCall2H
"dropout_22/StatefulPartitionedCall"dropout_22/StatefulPartitionedCall2H
"dropout_23/StatefulPartitionedCall"dropout_23/StatefulPartitionedCall2B
lstm_22/StatefulPartitionedCalllstm_22/StatefulPartitionedCall2B
lstm_23/StatefulPartitionedCalllstm_23/StatefulPartitionedCall:Z V
+
_output_shapes
:         ]
'
_user_specified_namelstm_22_input
у
═
while_cond_39111085
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_39111085___redundant_placeholder06
2while_while_cond_39111085___redundant_placeholder16
2while_while_cond_39111085___redundant_placeholder26
2while_while_cond_39111085___redundant_placeholder3
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
B: : : : :         Д:         Д: ::::: 
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
:         Д:.*
(
_output_shapes
:         Д:

_output_shapes
: :

_output_shapes
:
Ю?
╘
while_body_39112214
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
3while_lstm_cell_23_matmul_readvariableop_resource_0:
Д┤H
5while_lstm_cell_23_matmul_1_readvariableop_resource_0:	m┤C
4while_lstm_cell_23_biasadd_readvariableop_resource_0:	┤
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
1while_lstm_cell_23_matmul_readvariableop_resource:
Д┤F
3while_lstm_cell_23_matmul_1_readvariableop_resource:	m┤A
2while_lstm_cell_23_biasadd_readvariableop_resource:	┤Ив)while/lstm_cell_23/BiasAdd/ReadVariableOpв(while/lstm_cell_23/MatMul/ReadVariableOpв*while/lstm_cell_23/MatMul_1/ReadVariableOp├
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    Д   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape╘
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:         Д*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem╩
(while/lstm_cell_23/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_23_matmul_readvariableop_resource_0* 
_output_shapes
:
Д┤*
dtype02*
(while/lstm_cell_23/MatMul/ReadVariableOp╫
while/lstm_cell_23/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_23/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ┤2
while/lstm_cell_23/MatMul╧
*while/lstm_cell_23/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_23_matmul_1_readvariableop_resource_0*
_output_shapes
:	m┤*
dtype02,
*while/lstm_cell_23/MatMul_1/ReadVariableOp└
while/lstm_cell_23/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_23/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ┤2
while/lstm_cell_23/MatMul_1╕
while/lstm_cell_23/addAddV2#while/lstm_cell_23/MatMul:product:0%while/lstm_cell_23/MatMul_1:product:0*
T0*(
_output_shapes
:         ┤2
while/lstm_cell_23/add╚
)while/lstm_cell_23/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_23_biasadd_readvariableop_resource_0*
_output_shapes	
:┤*
dtype02+
)while/lstm_cell_23/BiasAdd/ReadVariableOp┼
while/lstm_cell_23/BiasAddBiasAddwhile/lstm_cell_23/add:z:01while/lstm_cell_23/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ┤2
while/lstm_cell_23/BiasAddК
"while/lstm_cell_23/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_23/split/split_dimЛ
while/lstm_cell_23/splitSplit+while/lstm_cell_23/split/split_dim:output:0#while/lstm_cell_23/BiasAdd:output:0*
T0*`
_output_shapesN
L:         m:         m:         m:         m*
	num_split2
while/lstm_cell_23/splitШ
while/lstm_cell_23/SigmoidSigmoid!while/lstm_cell_23/split:output:0*
T0*'
_output_shapes
:         m2
while/lstm_cell_23/SigmoidЬ
while/lstm_cell_23/Sigmoid_1Sigmoid!while/lstm_cell_23/split:output:1*
T0*'
_output_shapes
:         m2
while/lstm_cell_23/Sigmoid_1а
while/lstm_cell_23/mulMul while/lstm_cell_23/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:         m2
while/lstm_cell_23/mulП
while/lstm_cell_23/ReluRelu!while/lstm_cell_23/split:output:2*
T0*'
_output_shapes
:         m2
while/lstm_cell_23/Relu┤
while/lstm_cell_23/mul_1Mulwhile/lstm_cell_23/Sigmoid:y:0%while/lstm_cell_23/Relu:activations:0*
T0*'
_output_shapes
:         m2
while/lstm_cell_23/mul_1й
while/lstm_cell_23/add_1AddV2while/lstm_cell_23/mul:z:0while/lstm_cell_23/mul_1:z:0*
T0*'
_output_shapes
:         m2
while/lstm_cell_23/add_1Ь
while/lstm_cell_23/Sigmoid_2Sigmoid!while/lstm_cell_23/split:output:3*
T0*'
_output_shapes
:         m2
while/lstm_cell_23/Sigmoid_2О
while/lstm_cell_23/Relu_1Reluwhile/lstm_cell_23/add_1:z:0*
T0*'
_output_shapes
:         m2
while/lstm_cell_23/Relu_1╕
while/lstm_cell_23/mul_2Mul while/lstm_cell_23/Sigmoid_2:y:0'while/lstm_cell_23/Relu_1:activations:0*
T0*'
_output_shapes
:         m2
while/lstm_cell_23/mul_2р
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
while/Identity_2Ъ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3Н
while/Identity_4Identitywhile/lstm_cell_23/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:         m2
while/Identity_4Н
while/Identity_5Identitywhile/lstm_cell_23/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:         m2
while/Identity_5▐

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
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         m:         m: : : : : 2V
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
: :-)
'
_output_shapes
:         m:-)
'
_output_shapes
:         m:

_output_shapes
: :

_output_shapes
: 
м

╥
0__inference_sequential_11_layer_call_fn_39110236
lstm_22_input
unknown:	]Р
	unknown_0:
ДР
	unknown_1:	Р
	unknown_2:
Д┤
	unknown_3:	m┤
	unknown_4:	┤
	unknown_5:m
	unknown_6:
identityИвStatefulPartitionedCall╘
StatefulPartitionedCallStatefulPartitionedCalllstm_22_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
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
K__inference_sequential_11_layer_call_and_return_conditional_losses_391101962
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
_user_specified_namelstm_22_input
▀
═
while_cond_39109020
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_39109020___redundant_placeholder06
2while_while_cond_39109020___redundant_placeholder16
2while_while_cond_39109020___redundant_placeholder26
2while_while_cond_39109020___redundant_placeholder3
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
@: : : : :         m:         m: ::::: 
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
:         m:-)
'
_output_shapes
:         m:

_output_shapes
: :

_output_shapes
:
╢
f
-__inference_dropout_23_layer_call_fn_39112369

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
:         m* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_dropout_23_layer_call_and_return_conditional_losses_391097762
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:         m2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         m22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:         m
 
_user_specified_nameinputs
у
═
while_cond_39108180
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_39108180___redundant_placeholder06
2while_while_cond_39108180___redundant_placeholder16
2while_while_cond_39108180___redundant_placeholder26
2while_while_cond_39108180___redundant_placeholder3
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
B: : : : :         Д:         Д: ::::: 
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
:         Д:.*
(
_output_shapes
:         Д:

_output_shapes
: :

_output_shapes
:
д
я
K__inference_sequential_11_layer_call_and_return_conditional_losses_39109727

inputs#
lstm_22_39109510:	]Р$
lstm_22_39109512:
ДР
lstm_22_39109514:	Р$
lstm_23_39109675:
Д┤#
lstm_23_39109677:	m┤
lstm_23_39109679:	┤#
dense_11_39109721:m
dense_11_39109723:
identityИв dense_11/StatefulPartitionedCallвlstm_22/StatefulPartitionedCallвlstm_23/StatefulPartitionedCallо
lstm_22/StatefulPartitionedCallStatefulPartitionedCallinputslstm_22_39109510lstm_22_39109512lstm_22_39109514*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         Д*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_lstm_22_layer_call_and_return_conditional_losses_391095092!
lstm_22/StatefulPartitionedCallГ
dropout_22/PartitionedCallPartitionedCall(lstm_22/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         Д* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_dropout_22_layer_call_and_return_conditional_losses_391095222
dropout_22/PartitionedCall╩
lstm_23/StatefulPartitionedCallStatefulPartitionedCall#dropout_22/PartitionedCall:output:0lstm_23_39109675lstm_23_39109677lstm_23_39109679*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         m*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_lstm_23_layer_call_and_return_conditional_losses_391096742!
lstm_23/StatefulPartitionedCallВ
dropout_23/PartitionedCallPartitionedCall(lstm_23/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         m* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_dropout_23_layer_call_and_return_conditional_losses_391096872
dropout_23/PartitionedCall╗
 dense_11/StatefulPartitionedCallStatefulPartitionedCall#dropout_23/PartitionedCall:output:0dense_11_39109721dense_11_39109723*
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
F__inference_dense_11_layer_call_and_return_conditional_losses_391097202"
 dense_11/StatefulPartitionedCallИ
IdentityIdentity)dense_11/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:         2

Identity╡
NoOpNoOp!^dense_11/StatefulPartitionedCall ^lstm_22/StatefulPartitionedCall ^lstm_23/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':         ]: : : : : : : : 2D
 dense_11/StatefulPartitionedCall dense_11/StatefulPartitionedCall2B
lstm_22/StatefulPartitionedCalllstm_22/StatefulPartitionedCall2B
lstm_23/StatefulPartitionedCalllstm_23/StatefulPartitionedCall:S O
+
_output_shapes
:         ]
 
_user_specified_nameinputs
╫
g
H__inference_dropout_22_layer_call_and_return_conditional_losses_39111684

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
:         Д2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape╣
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:         Д*
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
:         Д2
dropout/GreaterEqualД
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:         Д2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:         Д2
dropout/Mul_1j
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:         Д2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         Д:T P
,
_output_shapes
:         Д
 
_user_specified_nameinputs
░?
╘
while_body_39111539
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_22_matmul_readvariableop_resource_0:	]РI
5while_lstm_cell_22_matmul_1_readvariableop_resource_0:
ДРC
4while_lstm_cell_22_biasadd_readvariableop_resource_0:	Р
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_22_matmul_readvariableop_resource:	]РG
3while_lstm_cell_22_matmul_1_readvariableop_resource:
ДРA
2while_lstm_cell_22_biasadd_readvariableop_resource:	РИв)while/lstm_cell_22/BiasAdd/ReadVariableOpв(while/lstm_cell_22/MatMul/ReadVariableOpв*while/lstm_cell_22/MatMul_1/ReadVariableOp├
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
(while/lstm_cell_22/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_22_matmul_readvariableop_resource_0*
_output_shapes
:	]Р*
dtype02*
(while/lstm_cell_22/MatMul/ReadVariableOp╫
while/lstm_cell_22/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_22/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Р2
while/lstm_cell_22/MatMul╨
*while/lstm_cell_22/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_22_matmul_1_readvariableop_resource_0* 
_output_shapes
:
ДР*
dtype02,
*while/lstm_cell_22/MatMul_1/ReadVariableOp└
while/lstm_cell_22/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_22/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Р2
while/lstm_cell_22/MatMul_1╕
while/lstm_cell_22/addAddV2#while/lstm_cell_22/MatMul:product:0%while/lstm_cell_22/MatMul_1:product:0*
T0*(
_output_shapes
:         Р2
while/lstm_cell_22/add╚
)while/lstm_cell_22/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_22_biasadd_readvariableop_resource_0*
_output_shapes	
:Р*
dtype02+
)while/lstm_cell_22/BiasAdd/ReadVariableOp┼
while/lstm_cell_22/BiasAddBiasAddwhile/lstm_cell_22/add:z:01while/lstm_cell_22/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Р2
while/lstm_cell_22/BiasAddК
"while/lstm_cell_22/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_22/split/split_dimП
while/lstm_cell_22/splitSplit+while/lstm_cell_22/split/split_dim:output:0#while/lstm_cell_22/BiasAdd:output:0*
T0*d
_output_shapesR
P:         Д:         Д:         Д:         Д*
	num_split2
while/lstm_cell_22/splitЩ
while/lstm_cell_22/SigmoidSigmoid!while/lstm_cell_22/split:output:0*
T0*(
_output_shapes
:         Д2
while/lstm_cell_22/SigmoidЭ
while/lstm_cell_22/Sigmoid_1Sigmoid!while/lstm_cell_22/split:output:1*
T0*(
_output_shapes
:         Д2
while/lstm_cell_22/Sigmoid_1б
while/lstm_cell_22/mulMul while/lstm_cell_22/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:         Д2
while/lstm_cell_22/mulР
while/lstm_cell_22/ReluRelu!while/lstm_cell_22/split:output:2*
T0*(
_output_shapes
:         Д2
while/lstm_cell_22/Relu╡
while/lstm_cell_22/mul_1Mulwhile/lstm_cell_22/Sigmoid:y:0%while/lstm_cell_22/Relu:activations:0*
T0*(
_output_shapes
:         Д2
while/lstm_cell_22/mul_1к
while/lstm_cell_22/add_1AddV2while/lstm_cell_22/mul:z:0while/lstm_cell_22/mul_1:z:0*
T0*(
_output_shapes
:         Д2
while/lstm_cell_22/add_1Э
while/lstm_cell_22/Sigmoid_2Sigmoid!while/lstm_cell_22/split:output:3*
T0*(
_output_shapes
:         Д2
while/lstm_cell_22/Sigmoid_2П
while/lstm_cell_22/Relu_1Reluwhile/lstm_cell_22/add_1:z:0*
T0*(
_output_shapes
:         Д2
while/lstm_cell_22/Relu_1╣
while/lstm_cell_22/mul_2Mul while/lstm_cell_22/Sigmoid_2:y:0'while/lstm_cell_22/Relu_1:activations:0*
T0*(
_output_shapes
:         Д2
while/lstm_cell_22/mul_2р
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
while/Identity_2Ъ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3О
while/Identity_4Identitywhile/lstm_cell_22/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:         Д2
while/Identity_4О
while/Identity_5Identitywhile/lstm_cell_22/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:         Д2
while/Identity_5▐

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
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :         Д:         Д: : : : : 2V
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
:         Д:.*
(
_output_shapes
:         Д:

_output_shapes
: :

_output_shapes
: 
Д\
Ю
E__inference_lstm_22_layer_call_and_return_conditional_losses_39109509

inputs>
+lstm_cell_22_matmul_readvariableop_resource:	]РA
-lstm_cell_22_matmul_1_readvariableop_resource:
ДР;
,lstm_cell_22_biasadd_readvariableop_resource:	Р
identityИв#lstm_cell_22/BiasAdd/ReadVariableOpв"lstm_cell_22/MatMul/ReadVariableOpв$lstm_cell_22/MatMul_1/ReadVariableOpвwhileD
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
B :Д2
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
B :Д2
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
:         Д2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :Д2
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
B :Д2
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
:         Д2	
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
"lstm_cell_22/MatMul/ReadVariableOpReadVariableOp+lstm_cell_22_matmul_readvariableop_resource*
_output_shapes
:	]Р*
dtype02$
"lstm_cell_22/MatMul/ReadVariableOpн
lstm_cell_22/MatMulMatMulstrided_slice_2:output:0*lstm_cell_22/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Р2
lstm_cell_22/MatMul╝
$lstm_cell_22/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_22_matmul_1_readvariableop_resource* 
_output_shapes
:
ДР*
dtype02&
$lstm_cell_22/MatMul_1/ReadVariableOpй
lstm_cell_22/MatMul_1MatMulzeros:output:0,lstm_cell_22/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Р2
lstm_cell_22/MatMul_1а
lstm_cell_22/addAddV2lstm_cell_22/MatMul:product:0lstm_cell_22/MatMul_1:product:0*
T0*(
_output_shapes
:         Р2
lstm_cell_22/add┤
#lstm_cell_22/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_22_biasadd_readvariableop_resource*
_output_shapes	
:Р*
dtype02%
#lstm_cell_22/BiasAdd/ReadVariableOpн
lstm_cell_22/BiasAddBiasAddlstm_cell_22/add:z:0+lstm_cell_22/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Р2
lstm_cell_22/BiasAdd~
lstm_cell_22/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_22/split/split_dimў
lstm_cell_22/splitSplit%lstm_cell_22/split/split_dim:output:0lstm_cell_22/BiasAdd:output:0*
T0*d
_output_shapesR
P:         Д:         Д:         Д:         Д*
	num_split2
lstm_cell_22/splitЗ
lstm_cell_22/SigmoidSigmoidlstm_cell_22/split:output:0*
T0*(
_output_shapes
:         Д2
lstm_cell_22/SigmoidЛ
lstm_cell_22/Sigmoid_1Sigmoidlstm_cell_22/split:output:1*
T0*(
_output_shapes
:         Д2
lstm_cell_22/Sigmoid_1М
lstm_cell_22/mulMullstm_cell_22/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:         Д2
lstm_cell_22/mul~
lstm_cell_22/ReluRelulstm_cell_22/split:output:2*
T0*(
_output_shapes
:         Д2
lstm_cell_22/ReluЭ
lstm_cell_22/mul_1Mullstm_cell_22/Sigmoid:y:0lstm_cell_22/Relu:activations:0*
T0*(
_output_shapes
:         Д2
lstm_cell_22/mul_1Т
lstm_cell_22/add_1AddV2lstm_cell_22/mul:z:0lstm_cell_22/mul_1:z:0*
T0*(
_output_shapes
:         Д2
lstm_cell_22/add_1Л
lstm_cell_22/Sigmoid_2Sigmoidlstm_cell_22/split:output:3*
T0*(
_output_shapes
:         Д2
lstm_cell_22/Sigmoid_2}
lstm_cell_22/Relu_1Relulstm_cell_22/add_1:z:0*
T0*(
_output_shapes
:         Д2
lstm_cell_22/Relu_1б
lstm_cell_22/mul_2Mullstm_cell_22/Sigmoid_2:y:0!lstm_cell_22/Relu_1:activations:0*
T0*(
_output_shapes
:         Д2
lstm_cell_22/mul_2П
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    Д   2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_22_matmul_readvariableop_resource-lstm_cell_22_matmul_1_readvariableop_resource,lstm_cell_22_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :         Д:         Д: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_39109425*
condR
while_cond_39109424*M
output_shapes<
:: : : : :         Д:         Д: : : : : *
parallel_iterations 2
while╡
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    Д   22
0TensorArrayV2Stack/TensorListStack/element_shapeщ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:         Д*
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
:         Д*
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
:         Д2
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
:         Д2

Identity╚
NoOpNoOp$^lstm_cell_22/BiasAdd/ReadVariableOp#^lstm_cell_22/MatMul/ReadVariableOp%^lstm_cell_22/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         ]: : : 2J
#lstm_cell_22/BiasAdd/ReadVariableOp#lstm_cell_22/BiasAdd/ReadVariableOp2H
"lstm_cell_22/MatMul/ReadVariableOp"lstm_cell_22/MatMul/ReadVariableOp2L
$lstm_cell_22/MatMul_1/ReadVariableOp$lstm_cell_22/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:         ]
 
_user_specified_nameinputs
м\
а
E__inference_lstm_23_layer_call_and_return_conditional_losses_39111996
inputs_0?
+lstm_cell_23_matmul_readvariableop_resource:
Д┤@
-lstm_cell_23_matmul_1_readvariableop_resource:	m┤;
,lstm_cell_23_biasadd_readvariableop_resource:	┤
identityИв#lstm_cell_23/BiasAdd/ReadVariableOpв"lstm_cell_23/MatMul/ReadVariableOpв$lstm_cell_23/MatMul_1/ReadVariableOpвwhileF
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
value	B :m2
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
value	B :m2
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
:         m2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :m2
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
value	B :m2
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
:         m2	
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
!:                  Д2
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
valueB"    Д   27
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
:         Д*
shrink_axis_mask2
strided_slice_2╢
"lstm_cell_23/MatMul/ReadVariableOpReadVariableOp+lstm_cell_23_matmul_readvariableop_resource* 
_output_shapes
:
Д┤*
dtype02$
"lstm_cell_23/MatMul/ReadVariableOpн
lstm_cell_23/MatMulMatMulstrided_slice_2:output:0*lstm_cell_23/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ┤2
lstm_cell_23/MatMul╗
$lstm_cell_23/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_23_matmul_1_readvariableop_resource*
_output_shapes
:	m┤*
dtype02&
$lstm_cell_23/MatMul_1/ReadVariableOpй
lstm_cell_23/MatMul_1MatMulzeros:output:0,lstm_cell_23/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ┤2
lstm_cell_23/MatMul_1а
lstm_cell_23/addAddV2lstm_cell_23/MatMul:product:0lstm_cell_23/MatMul_1:product:0*
T0*(
_output_shapes
:         ┤2
lstm_cell_23/add┤
#lstm_cell_23/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_23_biasadd_readvariableop_resource*
_output_shapes	
:┤*
dtype02%
#lstm_cell_23/BiasAdd/ReadVariableOpн
lstm_cell_23/BiasAddBiasAddlstm_cell_23/add:z:0+lstm_cell_23/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ┤2
lstm_cell_23/BiasAdd~
lstm_cell_23/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_23/split/split_dimє
lstm_cell_23/splitSplit%lstm_cell_23/split/split_dim:output:0lstm_cell_23/BiasAdd:output:0*
T0*`
_output_shapesN
L:         m:         m:         m:         m*
	num_split2
lstm_cell_23/splitЖ
lstm_cell_23/SigmoidSigmoidlstm_cell_23/split:output:0*
T0*'
_output_shapes
:         m2
lstm_cell_23/SigmoidК
lstm_cell_23/Sigmoid_1Sigmoidlstm_cell_23/split:output:1*
T0*'
_output_shapes
:         m2
lstm_cell_23/Sigmoid_1Л
lstm_cell_23/mulMullstm_cell_23/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:         m2
lstm_cell_23/mul}
lstm_cell_23/ReluRelulstm_cell_23/split:output:2*
T0*'
_output_shapes
:         m2
lstm_cell_23/ReluЬ
lstm_cell_23/mul_1Mullstm_cell_23/Sigmoid:y:0lstm_cell_23/Relu:activations:0*
T0*'
_output_shapes
:         m2
lstm_cell_23/mul_1С
lstm_cell_23/add_1AddV2lstm_cell_23/mul:z:0lstm_cell_23/mul_1:z:0*
T0*'
_output_shapes
:         m2
lstm_cell_23/add_1К
lstm_cell_23/Sigmoid_2Sigmoidlstm_cell_23/split:output:3*
T0*'
_output_shapes
:         m2
lstm_cell_23/Sigmoid_2|
lstm_cell_23/Relu_1Relulstm_cell_23/add_1:z:0*
T0*'
_output_shapes
:         m2
lstm_cell_23/Relu_1а
lstm_cell_23/mul_2Mullstm_cell_23/Sigmoid_2:y:0!lstm_cell_23/Relu_1:activations:0*
T0*'
_output_shapes
:         m2
lstm_cell_23/mul_2П
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    m   2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_23_matmul_readvariableop_resource-lstm_cell_23_matmul_1_readvariableop_resource,lstm_cell_23_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         m:         m: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_39111912*
condR
while_cond_39111911*K
output_shapes:
8: : : : :         m:         m: : : : : *
parallel_iterations 2
while╡
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    m   22
0TensorArrayV2Stack/TensorListStack/element_shapeё
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :                  m*
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
:         m*
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
 :                  m2
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
 :                  m2

Identity╚
NoOpNoOp$^lstm_cell_23/BiasAdd/ReadVariableOp#^lstm_cell_23/MatMul/ReadVariableOp%^lstm_cell_23/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':                  Д: : : 2J
#lstm_cell_23/BiasAdd/ReadVariableOp#lstm_cell_23/BiasAdd/ReadVariableOp2H
"lstm_cell_23/MatMul/ReadVariableOp"lstm_cell_23/MatMul/ReadVariableOp2L
$lstm_cell_23/MatMul_1/ReadVariableOp$lstm_cell_23/MatMul_1/ReadVariableOp2
whilewhile:_ [
5
_output_shapes#
!:                  Д
"
_user_specified_name
inputs/0
Д\
Ю
E__inference_lstm_22_layer_call_and_return_conditional_losses_39111472

inputs>
+lstm_cell_22_matmul_readvariableop_resource:	]РA
-lstm_cell_22_matmul_1_readvariableop_resource:
ДР;
,lstm_cell_22_biasadd_readvariableop_resource:	Р
identityИв#lstm_cell_22/BiasAdd/ReadVariableOpв"lstm_cell_22/MatMul/ReadVariableOpв$lstm_cell_22/MatMul_1/ReadVariableOpвwhileD
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
B :Д2
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
B :Д2
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
:         Д2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :Д2
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
B :Д2
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
:         Д2	
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
"lstm_cell_22/MatMul/ReadVariableOpReadVariableOp+lstm_cell_22_matmul_readvariableop_resource*
_output_shapes
:	]Р*
dtype02$
"lstm_cell_22/MatMul/ReadVariableOpн
lstm_cell_22/MatMulMatMulstrided_slice_2:output:0*lstm_cell_22/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Р2
lstm_cell_22/MatMul╝
$lstm_cell_22/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_22_matmul_1_readvariableop_resource* 
_output_shapes
:
ДР*
dtype02&
$lstm_cell_22/MatMul_1/ReadVariableOpй
lstm_cell_22/MatMul_1MatMulzeros:output:0,lstm_cell_22/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Р2
lstm_cell_22/MatMul_1а
lstm_cell_22/addAddV2lstm_cell_22/MatMul:product:0lstm_cell_22/MatMul_1:product:0*
T0*(
_output_shapes
:         Р2
lstm_cell_22/add┤
#lstm_cell_22/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_22_biasadd_readvariableop_resource*
_output_shapes	
:Р*
dtype02%
#lstm_cell_22/BiasAdd/ReadVariableOpн
lstm_cell_22/BiasAddBiasAddlstm_cell_22/add:z:0+lstm_cell_22/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Р2
lstm_cell_22/BiasAdd~
lstm_cell_22/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_22/split/split_dimў
lstm_cell_22/splitSplit%lstm_cell_22/split/split_dim:output:0lstm_cell_22/BiasAdd:output:0*
T0*d
_output_shapesR
P:         Д:         Д:         Д:         Д*
	num_split2
lstm_cell_22/splitЗ
lstm_cell_22/SigmoidSigmoidlstm_cell_22/split:output:0*
T0*(
_output_shapes
:         Д2
lstm_cell_22/SigmoidЛ
lstm_cell_22/Sigmoid_1Sigmoidlstm_cell_22/split:output:1*
T0*(
_output_shapes
:         Д2
lstm_cell_22/Sigmoid_1М
lstm_cell_22/mulMullstm_cell_22/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:         Д2
lstm_cell_22/mul~
lstm_cell_22/ReluRelulstm_cell_22/split:output:2*
T0*(
_output_shapes
:         Д2
lstm_cell_22/ReluЭ
lstm_cell_22/mul_1Mullstm_cell_22/Sigmoid:y:0lstm_cell_22/Relu:activations:0*
T0*(
_output_shapes
:         Д2
lstm_cell_22/mul_1Т
lstm_cell_22/add_1AddV2lstm_cell_22/mul:z:0lstm_cell_22/mul_1:z:0*
T0*(
_output_shapes
:         Д2
lstm_cell_22/add_1Л
lstm_cell_22/Sigmoid_2Sigmoidlstm_cell_22/split:output:3*
T0*(
_output_shapes
:         Д2
lstm_cell_22/Sigmoid_2}
lstm_cell_22/Relu_1Relulstm_cell_22/add_1:z:0*
T0*(
_output_shapes
:         Д2
lstm_cell_22/Relu_1б
lstm_cell_22/mul_2Mullstm_cell_22/Sigmoid_2:y:0!lstm_cell_22/Relu_1:activations:0*
T0*(
_output_shapes
:         Д2
lstm_cell_22/mul_2П
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    Д   2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_22_matmul_readvariableop_resource-lstm_cell_22_matmul_1_readvariableop_resource,lstm_cell_22_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :         Д:         Д: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_39111388*
condR
while_cond_39111387*M
output_shapes<
:: : : : :         Д:         Д: : : : : *
parallel_iterations 2
while╡
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    Д   22
0TensorArrayV2Stack/TensorListStack/element_shapeщ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:         Д*
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
:         Д*
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
:         Д2
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
:         Д2

Identity╚
NoOpNoOp$^lstm_cell_22/BiasAdd/ReadVariableOp#^lstm_cell_22/MatMul/ReadVariableOp%^lstm_cell_22/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         ]: : : 2J
#lstm_cell_22/BiasAdd/ReadVariableOp#lstm_cell_22/BiasAdd/ReadVariableOp2H
"lstm_cell_22/MatMul/ReadVariableOp"lstm_cell_22/MatMul/ReadVariableOp2L
$lstm_cell_22/MatMul_1/ReadVariableOp$lstm_cell_22/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:         ]
 
_user_specified_nameinputs
░?
╘
while_body_39111237
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_22_matmul_readvariableop_resource_0:	]РI
5while_lstm_cell_22_matmul_1_readvariableop_resource_0:
ДРC
4while_lstm_cell_22_biasadd_readvariableop_resource_0:	Р
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_22_matmul_readvariableop_resource:	]РG
3while_lstm_cell_22_matmul_1_readvariableop_resource:
ДРA
2while_lstm_cell_22_biasadd_readvariableop_resource:	РИв)while/lstm_cell_22/BiasAdd/ReadVariableOpв(while/lstm_cell_22/MatMul/ReadVariableOpв*while/lstm_cell_22/MatMul_1/ReadVariableOp├
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
(while/lstm_cell_22/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_22_matmul_readvariableop_resource_0*
_output_shapes
:	]Р*
dtype02*
(while/lstm_cell_22/MatMul/ReadVariableOp╫
while/lstm_cell_22/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_22/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Р2
while/lstm_cell_22/MatMul╨
*while/lstm_cell_22/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_22_matmul_1_readvariableop_resource_0* 
_output_shapes
:
ДР*
dtype02,
*while/lstm_cell_22/MatMul_1/ReadVariableOp└
while/lstm_cell_22/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_22/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Р2
while/lstm_cell_22/MatMul_1╕
while/lstm_cell_22/addAddV2#while/lstm_cell_22/MatMul:product:0%while/lstm_cell_22/MatMul_1:product:0*
T0*(
_output_shapes
:         Р2
while/lstm_cell_22/add╚
)while/lstm_cell_22/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_22_biasadd_readvariableop_resource_0*
_output_shapes	
:Р*
dtype02+
)while/lstm_cell_22/BiasAdd/ReadVariableOp┼
while/lstm_cell_22/BiasAddBiasAddwhile/lstm_cell_22/add:z:01while/lstm_cell_22/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Р2
while/lstm_cell_22/BiasAddК
"while/lstm_cell_22/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_22/split/split_dimП
while/lstm_cell_22/splitSplit+while/lstm_cell_22/split/split_dim:output:0#while/lstm_cell_22/BiasAdd:output:0*
T0*d
_output_shapesR
P:         Д:         Д:         Д:         Д*
	num_split2
while/lstm_cell_22/splitЩ
while/lstm_cell_22/SigmoidSigmoid!while/lstm_cell_22/split:output:0*
T0*(
_output_shapes
:         Д2
while/lstm_cell_22/SigmoidЭ
while/lstm_cell_22/Sigmoid_1Sigmoid!while/lstm_cell_22/split:output:1*
T0*(
_output_shapes
:         Д2
while/lstm_cell_22/Sigmoid_1б
while/lstm_cell_22/mulMul while/lstm_cell_22/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:         Д2
while/lstm_cell_22/mulР
while/lstm_cell_22/ReluRelu!while/lstm_cell_22/split:output:2*
T0*(
_output_shapes
:         Д2
while/lstm_cell_22/Relu╡
while/lstm_cell_22/mul_1Mulwhile/lstm_cell_22/Sigmoid:y:0%while/lstm_cell_22/Relu:activations:0*
T0*(
_output_shapes
:         Д2
while/lstm_cell_22/mul_1к
while/lstm_cell_22/add_1AddV2while/lstm_cell_22/mul:z:0while/lstm_cell_22/mul_1:z:0*
T0*(
_output_shapes
:         Д2
while/lstm_cell_22/add_1Э
while/lstm_cell_22/Sigmoid_2Sigmoid!while/lstm_cell_22/split:output:3*
T0*(
_output_shapes
:         Д2
while/lstm_cell_22/Sigmoid_2П
while/lstm_cell_22/Relu_1Reluwhile/lstm_cell_22/add_1:z:0*
T0*(
_output_shapes
:         Д2
while/lstm_cell_22/Relu_1╣
while/lstm_cell_22/mul_2Mul while/lstm_cell_22/Sigmoid_2:y:0'while/lstm_cell_22/Relu_1:activations:0*
T0*(
_output_shapes
:         Д2
while/lstm_cell_22/mul_2р
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
while/Identity_2Ъ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3О
while/Identity_4Identitywhile/lstm_cell_22/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:         Д2
while/Identity_4О
while/Identity_5Identitywhile/lstm_cell_22/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:         Д2
while/Identity_5▐

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
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :         Д:         Д: : : : : 2V
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
:         Д:.*
(
_output_shapes
:         Д:

_output_shapes
: :

_output_shapes
: 
▒
╣
K__inference_sequential_11_layer_call_and_return_conditional_losses_39110196

inputs#
lstm_22_39110174:	]Р$
lstm_22_39110176:
ДР
lstm_22_39110178:	Р$
lstm_23_39110182:
Д┤#
lstm_23_39110184:	m┤
lstm_23_39110186:	┤#
dense_11_39110190:m
dense_11_39110192:
identityИв dense_11/StatefulPartitionedCallв"dropout_22/StatefulPartitionedCallв"dropout_23/StatefulPartitionedCallвlstm_22/StatefulPartitionedCallвlstm_23/StatefulPartitionedCallо
lstm_22/StatefulPartitionedCallStatefulPartitionedCallinputslstm_22_39110174lstm_22_39110176lstm_22_39110178*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         Д*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_lstm_22_layer_call_and_return_conditional_losses_391101392!
lstm_22/StatefulPartitionedCallЫ
"dropout_22/StatefulPartitionedCallStatefulPartitionedCall(lstm_22/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         Д* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_dropout_22_layer_call_and_return_conditional_losses_391099722$
"dropout_22/StatefulPartitionedCall╥
lstm_23/StatefulPartitionedCallStatefulPartitionedCall+dropout_22/StatefulPartitionedCall:output:0lstm_23_39110182lstm_23_39110184lstm_23_39110186*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         m*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_lstm_23_layer_call_and_return_conditional_losses_391099432!
lstm_23/StatefulPartitionedCall┐
"dropout_23/StatefulPartitionedCallStatefulPartitionedCall(lstm_23/StatefulPartitionedCall:output:0#^dropout_22/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         m* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_dropout_23_layer_call_and_return_conditional_losses_391097762$
"dropout_23/StatefulPartitionedCall├
 dense_11/StatefulPartitionedCallStatefulPartitionedCall+dropout_23/StatefulPartitionedCall:output:0dense_11_39110190dense_11_39110192*
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
F__inference_dense_11_layer_call_and_return_conditional_losses_391097202"
 dense_11/StatefulPartitionedCallИ
IdentityIdentity)dense_11/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:         2

Identity 
NoOpNoOp!^dense_11/StatefulPartitionedCall#^dropout_22/StatefulPartitionedCall#^dropout_23/StatefulPartitionedCall ^lstm_22/StatefulPartitionedCall ^lstm_23/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':         ]: : : : : : : : 2D
 dense_11/StatefulPartitionedCall dense_11/StatefulPartitionedCall2H
"dropout_22/StatefulPartitionedCall"dropout_22/StatefulPartitionedCall2H
"dropout_23/StatefulPartitionedCall"dropout_23/StatefulPartitionedCall2B
lstm_22/StatefulPartitionedCalllstm_22/StatefulPartitionedCall2B
lstm_23/StatefulPartitionedCalllstm_23/StatefulPartitionedCall:S O
+
_output_shapes
:         ]
 
_user_specified_nameinputs
У
Й
J__inference_lstm_cell_22_layer_call_and_return_conditional_losses_39112441

inputs
states_0
states_11
matmul_readvariableop_resource:	]Р4
 matmul_1_readvariableop_resource:
ДР.
biasadd_readvariableop_resource:	Р
identity

identity_1

identity_2ИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpвMatMul_1/ReadVariableOpО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	]Р*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Р2
MatMulХ
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
ДР*
dtype02
MatMul_1/ReadVariableOp|
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Р2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:         Р2
addН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:Р*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Р2	
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
P:         Д:         Д:         Д:         Д*
	num_split2
split`
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:         Д2	
Sigmoidd
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:         Д2
	Sigmoid_1]
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:         Д2
mulW
ReluRelusplit:output:2*
T0*(
_output_shapes
:         Д2
Relui
mul_1MulSigmoid:y:0Relu:activations:0*
T0*(
_output_shapes
:         Д2
mul_1^
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:         Д2
add_1d
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:         Д2
	Sigmoid_2V
Relu_1Relu	add_1:z:0*
T0*(
_output_shapes
:         Д2
Relu_1m
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*(
_output_shapes
:         Д2
mul_2e
IdentityIdentity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:         Д2

Identityi

Identity_1Identity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:         Д2

Identity_1i

Identity_2Identity	add_1:z:0^NoOp*
T0*(
_output_shapes
:         Д2

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
A:         ]:         Д:         Д: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:         ]
 
_user_specified_nameinputs:RN
(
_output_shapes
:         Д
"
_user_specified_name
states/0:RN
(
_output_shapes
:         Д
"
_user_specified_name
states/1
∙
З
J__inference_lstm_cell_23_layer_call_and_return_conditional_losses_39108943

inputs

states
states_12
matmul_readvariableop_resource:
Д┤3
 matmul_1_readvariableop_resource:	m┤.
biasadd_readvariableop_resource:	┤
identity

identity_1

identity_2ИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpвMatMul_1/ReadVariableOpП
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
Д┤*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ┤2
MatMulФ
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	m┤*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ┤2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:         ┤2
addН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:┤*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ┤2	
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
L:         m:         m:         m:         m*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:         m2	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:         m2
	Sigmoid_1\
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:         m2
mulV
ReluRelusplit:output:2*
T0*'
_output_shapes
:         m2
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:         m2
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:         m2
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:         m2
	Sigmoid_2U
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:         m2
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:         m2
mul_2d
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:         m2

Identityh

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:         m2

Identity_1h

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:         m2

Identity_2Щ
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:         Д:         m:         m: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:P L
(
_output_shapes
:         Д
 
_user_specified_nameinputs:OK
'
_output_shapes
:         m
 
_user_specified_namestates:OK
'
_output_shapes
:         m
 
_user_specified_namestates
╜
∙
/__inference_lstm_cell_23_layer_call_fn_39112588

inputs
states_0
states_1
unknown:
Д┤
	unknown_0:	m┤
	unknown_1:	┤
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
9:         m:         m:         m*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_lstm_cell_23_layer_call_and_return_conditional_losses_391087972
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         m2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:         m2

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:         m2

Identity_2h
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:         Д:         m:         m: : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         Д
 
_user_specified_nameinputs:QM
'
_output_shapes
:         m
"
_user_specified_name
states/0:QM
'
_output_shapes
:         m
"
_user_specified_name
states/1
Е
f
H__inference_dropout_23_layer_call_and_return_conditional_losses_39109687

inputs

identity_1^
IdentityIdentityinputs*
T0*+
_output_shapes
:         m2

Identitym

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:         m2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         m:S O
+
_output_shapes
:         m
 
_user_specified_nameinputs
хJ
╘

lstm_22_while_body_39110376,
(lstm_22_while_lstm_22_while_loop_counter2
.lstm_22_while_lstm_22_while_maximum_iterations
lstm_22_while_placeholder
lstm_22_while_placeholder_1
lstm_22_while_placeholder_2
lstm_22_while_placeholder_3+
'lstm_22_while_lstm_22_strided_slice_1_0g
clstm_22_while_tensorarrayv2read_tensorlistgetitem_lstm_22_tensorarrayunstack_tensorlistfromtensor_0N
;lstm_22_while_lstm_cell_22_matmul_readvariableop_resource_0:	]РQ
=lstm_22_while_lstm_cell_22_matmul_1_readvariableop_resource_0:
ДРK
<lstm_22_while_lstm_cell_22_biasadd_readvariableop_resource_0:	Р
lstm_22_while_identity
lstm_22_while_identity_1
lstm_22_while_identity_2
lstm_22_while_identity_3
lstm_22_while_identity_4
lstm_22_while_identity_5)
%lstm_22_while_lstm_22_strided_slice_1e
alstm_22_while_tensorarrayv2read_tensorlistgetitem_lstm_22_tensorarrayunstack_tensorlistfromtensorL
9lstm_22_while_lstm_cell_22_matmul_readvariableop_resource:	]РO
;lstm_22_while_lstm_cell_22_matmul_1_readvariableop_resource:
ДРI
:lstm_22_while_lstm_cell_22_biasadd_readvariableop_resource:	РИв1lstm_22/while/lstm_cell_22/BiasAdd/ReadVariableOpв0lstm_22/while/lstm_cell_22/MatMul/ReadVariableOpв2lstm_22/while/lstm_cell_22/MatMul_1/ReadVariableOp╙
?lstm_22/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ]   2A
?lstm_22/while/TensorArrayV2Read/TensorListGetItem/element_shapeГ
1lstm_22/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemclstm_22_while_tensorarrayv2read_tensorlistgetitem_lstm_22_tensorarrayunstack_tensorlistfromtensor_0lstm_22_while_placeholderHlstm_22/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         ]*
element_dtype023
1lstm_22/while/TensorArrayV2Read/TensorListGetItemс
0lstm_22/while/lstm_cell_22/MatMul/ReadVariableOpReadVariableOp;lstm_22_while_lstm_cell_22_matmul_readvariableop_resource_0*
_output_shapes
:	]Р*
dtype022
0lstm_22/while/lstm_cell_22/MatMul/ReadVariableOpў
!lstm_22/while/lstm_cell_22/MatMulMatMul8lstm_22/while/TensorArrayV2Read/TensorListGetItem:item:08lstm_22/while/lstm_cell_22/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Р2#
!lstm_22/while/lstm_cell_22/MatMulш
2lstm_22/while/lstm_cell_22/MatMul_1/ReadVariableOpReadVariableOp=lstm_22_while_lstm_cell_22_matmul_1_readvariableop_resource_0* 
_output_shapes
:
ДР*
dtype024
2lstm_22/while/lstm_cell_22/MatMul_1/ReadVariableOpр
#lstm_22/while/lstm_cell_22/MatMul_1MatMullstm_22_while_placeholder_2:lstm_22/while/lstm_cell_22/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Р2%
#lstm_22/while/lstm_cell_22/MatMul_1╪
lstm_22/while/lstm_cell_22/addAddV2+lstm_22/while/lstm_cell_22/MatMul:product:0-lstm_22/while/lstm_cell_22/MatMul_1:product:0*
T0*(
_output_shapes
:         Р2 
lstm_22/while/lstm_cell_22/addр
1lstm_22/while/lstm_cell_22/BiasAdd/ReadVariableOpReadVariableOp<lstm_22_while_lstm_cell_22_biasadd_readvariableop_resource_0*
_output_shapes	
:Р*
dtype023
1lstm_22/while/lstm_cell_22/BiasAdd/ReadVariableOpх
"lstm_22/while/lstm_cell_22/BiasAddBiasAdd"lstm_22/while/lstm_cell_22/add:z:09lstm_22/while/lstm_cell_22/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Р2$
"lstm_22/while/lstm_cell_22/BiasAddЪ
*lstm_22/while/lstm_cell_22/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2,
*lstm_22/while/lstm_cell_22/split/split_dimп
 lstm_22/while/lstm_cell_22/splitSplit3lstm_22/while/lstm_cell_22/split/split_dim:output:0+lstm_22/while/lstm_cell_22/BiasAdd:output:0*
T0*d
_output_shapesR
P:         Д:         Д:         Д:         Д*
	num_split2"
 lstm_22/while/lstm_cell_22/split▒
"lstm_22/while/lstm_cell_22/SigmoidSigmoid)lstm_22/while/lstm_cell_22/split:output:0*
T0*(
_output_shapes
:         Д2$
"lstm_22/while/lstm_cell_22/Sigmoid╡
$lstm_22/while/lstm_cell_22/Sigmoid_1Sigmoid)lstm_22/while/lstm_cell_22/split:output:1*
T0*(
_output_shapes
:         Д2&
$lstm_22/while/lstm_cell_22/Sigmoid_1┴
lstm_22/while/lstm_cell_22/mulMul(lstm_22/while/lstm_cell_22/Sigmoid_1:y:0lstm_22_while_placeholder_3*
T0*(
_output_shapes
:         Д2 
lstm_22/while/lstm_cell_22/mulи
lstm_22/while/lstm_cell_22/ReluRelu)lstm_22/while/lstm_cell_22/split:output:2*
T0*(
_output_shapes
:         Д2!
lstm_22/while/lstm_cell_22/Relu╒
 lstm_22/while/lstm_cell_22/mul_1Mul&lstm_22/while/lstm_cell_22/Sigmoid:y:0-lstm_22/while/lstm_cell_22/Relu:activations:0*
T0*(
_output_shapes
:         Д2"
 lstm_22/while/lstm_cell_22/mul_1╩
 lstm_22/while/lstm_cell_22/add_1AddV2"lstm_22/while/lstm_cell_22/mul:z:0$lstm_22/while/lstm_cell_22/mul_1:z:0*
T0*(
_output_shapes
:         Д2"
 lstm_22/while/lstm_cell_22/add_1╡
$lstm_22/while/lstm_cell_22/Sigmoid_2Sigmoid)lstm_22/while/lstm_cell_22/split:output:3*
T0*(
_output_shapes
:         Д2&
$lstm_22/while/lstm_cell_22/Sigmoid_2з
!lstm_22/while/lstm_cell_22/Relu_1Relu$lstm_22/while/lstm_cell_22/add_1:z:0*
T0*(
_output_shapes
:         Д2#
!lstm_22/while/lstm_cell_22/Relu_1┘
 lstm_22/while/lstm_cell_22/mul_2Mul(lstm_22/while/lstm_cell_22/Sigmoid_2:y:0/lstm_22/while/lstm_cell_22/Relu_1:activations:0*
T0*(
_output_shapes
:         Д2"
 lstm_22/while/lstm_cell_22/mul_2И
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
lstm_22/while/add/yЙ
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
lstm_22/while/add_1/yЮ
lstm_22/while/add_1AddV2(lstm_22_while_lstm_22_while_loop_counterlstm_22/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lstm_22/while/add_1Л
lstm_22/while/IdentityIdentitylstm_22/while/add_1:z:0^lstm_22/while/NoOp*
T0*
_output_shapes
: 2
lstm_22/while/Identityж
lstm_22/while/Identity_1Identity.lstm_22_while_lstm_22_while_maximum_iterations^lstm_22/while/NoOp*
T0*
_output_shapes
: 2
lstm_22/while/Identity_1Н
lstm_22/while/Identity_2Identitylstm_22/while/add:z:0^lstm_22/while/NoOp*
T0*
_output_shapes
: 2
lstm_22/while/Identity_2║
lstm_22/while/Identity_3IdentityBlstm_22/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_22/while/NoOp*
T0*
_output_shapes
: 2
lstm_22/while/Identity_3о
lstm_22/while/Identity_4Identity$lstm_22/while/lstm_cell_22/mul_2:z:0^lstm_22/while/NoOp*
T0*(
_output_shapes
:         Д2
lstm_22/while/Identity_4о
lstm_22/while/Identity_5Identity$lstm_22/while/lstm_cell_22/add_1:z:0^lstm_22/while/NoOp*
T0*(
_output_shapes
:         Д2
lstm_22/while/Identity_5Ж
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
9lstm_22_while_lstm_cell_22_matmul_readvariableop_resource;lstm_22_while_lstm_cell_22_matmul_readvariableop_resource_0"╚
alstm_22_while_tensorarrayv2read_tensorlistgetitem_lstm_22_tensorarrayunstack_tensorlistfromtensorclstm_22_while_tensorarrayv2read_tensorlistgetitem_lstm_22_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :         Д:         Д: : : : : 2f
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
:         Д:.*
(
_output_shapes
:         Д:

_output_shapes
: :

_output_shapes
: 
░?
╘
while_body_39109425
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_22_matmul_readvariableop_resource_0:	]РI
5while_lstm_cell_22_matmul_1_readvariableop_resource_0:
ДРC
4while_lstm_cell_22_biasadd_readvariableop_resource_0:	Р
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_22_matmul_readvariableop_resource:	]РG
3while_lstm_cell_22_matmul_1_readvariableop_resource:
ДРA
2while_lstm_cell_22_biasadd_readvariableop_resource:	РИв)while/lstm_cell_22/BiasAdd/ReadVariableOpв(while/lstm_cell_22/MatMul/ReadVariableOpв*while/lstm_cell_22/MatMul_1/ReadVariableOp├
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
(while/lstm_cell_22/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_22_matmul_readvariableop_resource_0*
_output_shapes
:	]Р*
dtype02*
(while/lstm_cell_22/MatMul/ReadVariableOp╫
while/lstm_cell_22/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_22/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Р2
while/lstm_cell_22/MatMul╨
*while/lstm_cell_22/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_22_matmul_1_readvariableop_resource_0* 
_output_shapes
:
ДР*
dtype02,
*while/lstm_cell_22/MatMul_1/ReadVariableOp└
while/lstm_cell_22/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_22/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Р2
while/lstm_cell_22/MatMul_1╕
while/lstm_cell_22/addAddV2#while/lstm_cell_22/MatMul:product:0%while/lstm_cell_22/MatMul_1:product:0*
T0*(
_output_shapes
:         Р2
while/lstm_cell_22/add╚
)while/lstm_cell_22/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_22_biasadd_readvariableop_resource_0*
_output_shapes	
:Р*
dtype02+
)while/lstm_cell_22/BiasAdd/ReadVariableOp┼
while/lstm_cell_22/BiasAddBiasAddwhile/lstm_cell_22/add:z:01while/lstm_cell_22/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Р2
while/lstm_cell_22/BiasAddК
"while/lstm_cell_22/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_22/split/split_dimП
while/lstm_cell_22/splitSplit+while/lstm_cell_22/split/split_dim:output:0#while/lstm_cell_22/BiasAdd:output:0*
T0*d
_output_shapesR
P:         Д:         Д:         Д:         Д*
	num_split2
while/lstm_cell_22/splitЩ
while/lstm_cell_22/SigmoidSigmoid!while/lstm_cell_22/split:output:0*
T0*(
_output_shapes
:         Д2
while/lstm_cell_22/SigmoidЭ
while/lstm_cell_22/Sigmoid_1Sigmoid!while/lstm_cell_22/split:output:1*
T0*(
_output_shapes
:         Д2
while/lstm_cell_22/Sigmoid_1б
while/lstm_cell_22/mulMul while/lstm_cell_22/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:         Д2
while/lstm_cell_22/mulР
while/lstm_cell_22/ReluRelu!while/lstm_cell_22/split:output:2*
T0*(
_output_shapes
:         Д2
while/lstm_cell_22/Relu╡
while/lstm_cell_22/mul_1Mulwhile/lstm_cell_22/Sigmoid:y:0%while/lstm_cell_22/Relu:activations:0*
T0*(
_output_shapes
:         Д2
while/lstm_cell_22/mul_1к
while/lstm_cell_22/add_1AddV2while/lstm_cell_22/mul:z:0while/lstm_cell_22/mul_1:z:0*
T0*(
_output_shapes
:         Д2
while/lstm_cell_22/add_1Э
while/lstm_cell_22/Sigmoid_2Sigmoid!while/lstm_cell_22/split:output:3*
T0*(
_output_shapes
:         Д2
while/lstm_cell_22/Sigmoid_2П
while/lstm_cell_22/Relu_1Reluwhile/lstm_cell_22/add_1:z:0*
T0*(
_output_shapes
:         Д2
while/lstm_cell_22/Relu_1╣
while/lstm_cell_22/mul_2Mul while/lstm_cell_22/Sigmoid_2:y:0'while/lstm_cell_22/Relu_1:activations:0*
T0*(
_output_shapes
:         Д2
while/lstm_cell_22/mul_2р
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
while/Identity_2Ъ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3О
while/Identity_4Identitywhile/lstm_cell_22/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:         Д2
while/Identity_4О
while/Identity_5Identitywhile/lstm_cell_22/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:         Д2
while/Identity_5▐

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
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :         Д:         Д: : : : : 2V
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
:         Д:.*
(
_output_shapes
:         Д:

_output_shapes
: :

_output_shapes
: 
Й
f
H__inference_dropout_22_layer_call_and_return_conditional_losses_39111672

inputs

identity_1_
IdentityIdentityinputs*
T0*,
_output_shapes
:         Д2

Identityn

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:         Д2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         Д:T P
,
_output_shapes
:         Д
 
_user_specified_nameinputs
▀
═
while_cond_39108810
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_39108810___redundant_placeholder06
2while_while_cond_39108810___redundant_placeholder16
2while_while_cond_39108810___redundant_placeholder26
2while_while_cond_39108810___redundant_placeholder3
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
@: : : : :         m:         m: ::::: 
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
:         m:-)
'
_output_shapes
:         m:

_output_shapes
: :

_output_shapes
:
╟
∙
/__inference_lstm_cell_22_layer_call_fn_39112507

inputs
states_0
states_1
unknown:	]Р
	unknown_0:
ДР
	unknown_1:	Р
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
<:         Д:         Д:         Д*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_lstm_cell_22_layer_call_and_return_conditional_losses_391083132
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         Д2

IdentityА

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:         Д2

Identity_1А

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:         Д2

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
A:         ]:         Д:         Д: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         ]
 
_user_specified_nameinputs:RN
(
_output_shapes
:         Д
"
_user_specified_name
states/0:RN
(
_output_shapes
:         Д
"
_user_specified_name
states/1
Ю?
╘
while_body_39111761
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
3while_lstm_cell_23_matmul_readvariableop_resource_0:
Д┤H
5while_lstm_cell_23_matmul_1_readvariableop_resource_0:	m┤C
4while_lstm_cell_23_biasadd_readvariableop_resource_0:	┤
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
1while_lstm_cell_23_matmul_readvariableop_resource:
Д┤F
3while_lstm_cell_23_matmul_1_readvariableop_resource:	m┤A
2while_lstm_cell_23_biasadd_readvariableop_resource:	┤Ив)while/lstm_cell_23/BiasAdd/ReadVariableOpв(while/lstm_cell_23/MatMul/ReadVariableOpв*while/lstm_cell_23/MatMul_1/ReadVariableOp├
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    Д   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape╘
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:         Д*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem╩
(while/lstm_cell_23/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_23_matmul_readvariableop_resource_0* 
_output_shapes
:
Д┤*
dtype02*
(while/lstm_cell_23/MatMul/ReadVariableOp╫
while/lstm_cell_23/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_23/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ┤2
while/lstm_cell_23/MatMul╧
*while/lstm_cell_23/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_23_matmul_1_readvariableop_resource_0*
_output_shapes
:	m┤*
dtype02,
*while/lstm_cell_23/MatMul_1/ReadVariableOp└
while/lstm_cell_23/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_23/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ┤2
while/lstm_cell_23/MatMul_1╕
while/lstm_cell_23/addAddV2#while/lstm_cell_23/MatMul:product:0%while/lstm_cell_23/MatMul_1:product:0*
T0*(
_output_shapes
:         ┤2
while/lstm_cell_23/add╚
)while/lstm_cell_23/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_23_biasadd_readvariableop_resource_0*
_output_shapes	
:┤*
dtype02+
)while/lstm_cell_23/BiasAdd/ReadVariableOp┼
while/lstm_cell_23/BiasAddBiasAddwhile/lstm_cell_23/add:z:01while/lstm_cell_23/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ┤2
while/lstm_cell_23/BiasAddК
"while/lstm_cell_23/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_23/split/split_dimЛ
while/lstm_cell_23/splitSplit+while/lstm_cell_23/split/split_dim:output:0#while/lstm_cell_23/BiasAdd:output:0*
T0*`
_output_shapesN
L:         m:         m:         m:         m*
	num_split2
while/lstm_cell_23/splitШ
while/lstm_cell_23/SigmoidSigmoid!while/lstm_cell_23/split:output:0*
T0*'
_output_shapes
:         m2
while/lstm_cell_23/SigmoidЬ
while/lstm_cell_23/Sigmoid_1Sigmoid!while/lstm_cell_23/split:output:1*
T0*'
_output_shapes
:         m2
while/lstm_cell_23/Sigmoid_1а
while/lstm_cell_23/mulMul while/lstm_cell_23/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:         m2
while/lstm_cell_23/mulП
while/lstm_cell_23/ReluRelu!while/lstm_cell_23/split:output:2*
T0*'
_output_shapes
:         m2
while/lstm_cell_23/Relu┤
while/lstm_cell_23/mul_1Mulwhile/lstm_cell_23/Sigmoid:y:0%while/lstm_cell_23/Relu:activations:0*
T0*'
_output_shapes
:         m2
while/lstm_cell_23/mul_1й
while/lstm_cell_23/add_1AddV2while/lstm_cell_23/mul:z:0while/lstm_cell_23/mul_1:z:0*
T0*'
_output_shapes
:         m2
while/lstm_cell_23/add_1Ь
while/lstm_cell_23/Sigmoid_2Sigmoid!while/lstm_cell_23/split:output:3*
T0*'
_output_shapes
:         m2
while/lstm_cell_23/Sigmoid_2О
while/lstm_cell_23/Relu_1Reluwhile/lstm_cell_23/add_1:z:0*
T0*'
_output_shapes
:         m2
while/lstm_cell_23/Relu_1╕
while/lstm_cell_23/mul_2Mul while/lstm_cell_23/Sigmoid_2:y:0'while/lstm_cell_23/Relu_1:activations:0*
T0*'
_output_shapes
:         m2
while/lstm_cell_23/mul_2р
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
while/Identity_2Ъ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3Н
while/Identity_4Identitywhile/lstm_cell_23/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:         m2
while/Identity_4Н
while/Identity_5Identitywhile/lstm_cell_23/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:         m2
while/Identity_5▐

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
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         m:         m: : : : : 2V
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
: :-)
'
_output_shapes
:         m:-)
'
_output_shapes
:         m:

_output_shapes
: :

_output_shapes
: 
░?
╘
while_body_39111388
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_22_matmul_readvariableop_resource_0:	]РI
5while_lstm_cell_22_matmul_1_readvariableop_resource_0:
ДРC
4while_lstm_cell_22_biasadd_readvariableop_resource_0:	Р
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_22_matmul_readvariableop_resource:	]РG
3while_lstm_cell_22_matmul_1_readvariableop_resource:
ДРA
2while_lstm_cell_22_biasadd_readvariableop_resource:	РИв)while/lstm_cell_22/BiasAdd/ReadVariableOpв(while/lstm_cell_22/MatMul/ReadVariableOpв*while/lstm_cell_22/MatMul_1/ReadVariableOp├
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
(while/lstm_cell_22/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_22_matmul_readvariableop_resource_0*
_output_shapes
:	]Р*
dtype02*
(while/lstm_cell_22/MatMul/ReadVariableOp╫
while/lstm_cell_22/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_22/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Р2
while/lstm_cell_22/MatMul╨
*while/lstm_cell_22/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_22_matmul_1_readvariableop_resource_0* 
_output_shapes
:
ДР*
dtype02,
*while/lstm_cell_22/MatMul_1/ReadVariableOp└
while/lstm_cell_22/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_22/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Р2
while/lstm_cell_22/MatMul_1╕
while/lstm_cell_22/addAddV2#while/lstm_cell_22/MatMul:product:0%while/lstm_cell_22/MatMul_1:product:0*
T0*(
_output_shapes
:         Р2
while/lstm_cell_22/add╚
)while/lstm_cell_22/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_22_biasadd_readvariableop_resource_0*
_output_shapes	
:Р*
dtype02+
)while/lstm_cell_22/BiasAdd/ReadVariableOp┼
while/lstm_cell_22/BiasAddBiasAddwhile/lstm_cell_22/add:z:01while/lstm_cell_22/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Р2
while/lstm_cell_22/BiasAddК
"while/lstm_cell_22/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_22/split/split_dimП
while/lstm_cell_22/splitSplit+while/lstm_cell_22/split/split_dim:output:0#while/lstm_cell_22/BiasAdd:output:0*
T0*d
_output_shapesR
P:         Д:         Д:         Д:         Д*
	num_split2
while/lstm_cell_22/splitЩ
while/lstm_cell_22/SigmoidSigmoid!while/lstm_cell_22/split:output:0*
T0*(
_output_shapes
:         Д2
while/lstm_cell_22/SigmoidЭ
while/lstm_cell_22/Sigmoid_1Sigmoid!while/lstm_cell_22/split:output:1*
T0*(
_output_shapes
:         Д2
while/lstm_cell_22/Sigmoid_1б
while/lstm_cell_22/mulMul while/lstm_cell_22/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:         Д2
while/lstm_cell_22/mulР
while/lstm_cell_22/ReluRelu!while/lstm_cell_22/split:output:2*
T0*(
_output_shapes
:         Д2
while/lstm_cell_22/Relu╡
while/lstm_cell_22/mul_1Mulwhile/lstm_cell_22/Sigmoid:y:0%while/lstm_cell_22/Relu:activations:0*
T0*(
_output_shapes
:         Д2
while/lstm_cell_22/mul_1к
while/lstm_cell_22/add_1AddV2while/lstm_cell_22/mul:z:0while/lstm_cell_22/mul_1:z:0*
T0*(
_output_shapes
:         Д2
while/lstm_cell_22/add_1Э
while/lstm_cell_22/Sigmoid_2Sigmoid!while/lstm_cell_22/split:output:3*
T0*(
_output_shapes
:         Д2
while/lstm_cell_22/Sigmoid_2П
while/lstm_cell_22/Relu_1Reluwhile/lstm_cell_22/add_1:z:0*
T0*(
_output_shapes
:         Д2
while/lstm_cell_22/Relu_1╣
while/lstm_cell_22/mul_2Mul while/lstm_cell_22/Sigmoid_2:y:0'while/lstm_cell_22/Relu_1:activations:0*
T0*(
_output_shapes
:         Д2
while/lstm_cell_22/mul_2р
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
while/Identity_2Ъ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3О
while/Identity_4Identitywhile/lstm_cell_22/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:         Д2
while/Identity_4О
while/Identity_5Identitywhile/lstm_cell_22/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:         Д2
while/Identity_5▐

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
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :         Д:         Д: : : : : 2V
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
:         Д:.*
(
_output_shapes
:         Д:

_output_shapes
: :

_output_shapes
: 
▀
═
while_cond_39109858
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_39109858___redundant_placeholder06
2while_while_cond_39109858___redundant_placeholder16
2while_while_cond_39109858___redundant_placeholder26
2while_while_cond_39109858___redundant_placeholder3
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
@: : : : :         m:         m: ::::: 
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
:         m:-)
'
_output_shapes
:         m:

_output_shapes
: :

_output_shapes
:
╢
╕
*__inference_lstm_22_layer_call_fn_39111656

inputs
unknown:	]Р
	unknown_0:
ДР
	unknown_1:	Р
identityИвStatefulPartitionedCallЗ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         Д*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_lstm_22_layer_call_and_return_conditional_losses_391095092
StatefulPartitionedCallА
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:         Д2

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
у
═
while_cond_39111236
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_39111236___redundant_placeholder06
2while_while_cond_39111236___redundant_placeholder16
2while_while_cond_39111236___redundant_placeholder26
2while_while_cond_39111236___redundant_placeholder3
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
B: : : : :         Д:         Д: ::::: 
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
:         Д:.*
(
_output_shapes
:         Д:

_output_shapes
: :

_output_shapes
:
░?
╘
while_body_39111086
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_22_matmul_readvariableop_resource_0:	]РI
5while_lstm_cell_22_matmul_1_readvariableop_resource_0:
ДРC
4while_lstm_cell_22_biasadd_readvariableop_resource_0:	Р
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_22_matmul_readvariableop_resource:	]РG
3while_lstm_cell_22_matmul_1_readvariableop_resource:
ДРA
2while_lstm_cell_22_biasadd_readvariableop_resource:	РИв)while/lstm_cell_22/BiasAdd/ReadVariableOpв(while/lstm_cell_22/MatMul/ReadVariableOpв*while/lstm_cell_22/MatMul_1/ReadVariableOp├
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
(while/lstm_cell_22/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_22_matmul_readvariableop_resource_0*
_output_shapes
:	]Р*
dtype02*
(while/lstm_cell_22/MatMul/ReadVariableOp╫
while/lstm_cell_22/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_22/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Р2
while/lstm_cell_22/MatMul╨
*while/lstm_cell_22/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_22_matmul_1_readvariableop_resource_0* 
_output_shapes
:
ДР*
dtype02,
*while/lstm_cell_22/MatMul_1/ReadVariableOp└
while/lstm_cell_22/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_22/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Р2
while/lstm_cell_22/MatMul_1╕
while/lstm_cell_22/addAddV2#while/lstm_cell_22/MatMul:product:0%while/lstm_cell_22/MatMul_1:product:0*
T0*(
_output_shapes
:         Р2
while/lstm_cell_22/add╚
)while/lstm_cell_22/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_22_biasadd_readvariableop_resource_0*
_output_shapes	
:Р*
dtype02+
)while/lstm_cell_22/BiasAdd/ReadVariableOp┼
while/lstm_cell_22/BiasAddBiasAddwhile/lstm_cell_22/add:z:01while/lstm_cell_22/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Р2
while/lstm_cell_22/BiasAddК
"while/lstm_cell_22/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_22/split/split_dimП
while/lstm_cell_22/splitSplit+while/lstm_cell_22/split/split_dim:output:0#while/lstm_cell_22/BiasAdd:output:0*
T0*d
_output_shapesR
P:         Д:         Д:         Д:         Д*
	num_split2
while/lstm_cell_22/splitЩ
while/lstm_cell_22/SigmoidSigmoid!while/lstm_cell_22/split:output:0*
T0*(
_output_shapes
:         Д2
while/lstm_cell_22/SigmoidЭ
while/lstm_cell_22/Sigmoid_1Sigmoid!while/lstm_cell_22/split:output:1*
T0*(
_output_shapes
:         Д2
while/lstm_cell_22/Sigmoid_1б
while/lstm_cell_22/mulMul while/lstm_cell_22/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:         Д2
while/lstm_cell_22/mulР
while/lstm_cell_22/ReluRelu!while/lstm_cell_22/split:output:2*
T0*(
_output_shapes
:         Д2
while/lstm_cell_22/Relu╡
while/lstm_cell_22/mul_1Mulwhile/lstm_cell_22/Sigmoid:y:0%while/lstm_cell_22/Relu:activations:0*
T0*(
_output_shapes
:         Д2
while/lstm_cell_22/mul_1к
while/lstm_cell_22/add_1AddV2while/lstm_cell_22/mul:z:0while/lstm_cell_22/mul_1:z:0*
T0*(
_output_shapes
:         Д2
while/lstm_cell_22/add_1Э
while/lstm_cell_22/Sigmoid_2Sigmoid!while/lstm_cell_22/split:output:3*
T0*(
_output_shapes
:         Д2
while/lstm_cell_22/Sigmoid_2П
while/lstm_cell_22/Relu_1Reluwhile/lstm_cell_22/add_1:z:0*
T0*(
_output_shapes
:         Д2
while/lstm_cell_22/Relu_1╣
while/lstm_cell_22/mul_2Mul while/lstm_cell_22/Sigmoid_2:y:0'while/lstm_cell_22/Relu_1:activations:0*
T0*(
_output_shapes
:         Д2
while/lstm_cell_22/mul_2р
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
while/Identity_2Ъ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3О
while/Identity_4Identitywhile/lstm_cell_22/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:         Д2
while/Identity_4О
while/Identity_5Identitywhile/lstm_cell_22/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:         Д2
while/Identity_5▐

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
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :         Д:         Д: : : : : 2V
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
:         Д:.*
(
_output_shapes
:         Д:

_output_shapes
: :

_output_shapes
: 
╬М
К
K__inference_sequential_11_layer_call_and_return_conditional_losses_39110977

inputsF
3lstm_22_lstm_cell_22_matmul_readvariableop_resource:	]РI
5lstm_22_lstm_cell_22_matmul_1_readvariableop_resource:
ДРC
4lstm_22_lstm_cell_22_biasadd_readvariableop_resource:	РG
3lstm_23_lstm_cell_23_matmul_readvariableop_resource:
Д┤H
5lstm_23_lstm_cell_23_matmul_1_readvariableop_resource:	m┤C
4lstm_23_lstm_cell_23_biasadd_readvariableop_resource:	┤<
*dense_11_tensordot_readvariableop_resource:m6
(dense_11_biasadd_readvariableop_resource:
identityИвdense_11/BiasAdd/ReadVariableOpв!dense_11/Tensordot/ReadVariableOpв+lstm_22/lstm_cell_22/BiasAdd/ReadVariableOpв*lstm_22/lstm_cell_22/MatMul/ReadVariableOpв,lstm_22/lstm_cell_22/MatMul_1/ReadVariableOpвlstm_22/whileв+lstm_23/lstm_cell_23/BiasAdd/ReadVariableOpв*lstm_23/lstm_cell_23/MatMul/ReadVariableOpв,lstm_23/lstm_cell_23/MatMul_1/ReadVariableOpвlstm_23/whileT
lstm_22/ShapeShapeinputs*
T0*
_output_shapes
:2
lstm_22/ShapeД
lstm_22/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_22/strided_slice/stackИ
lstm_22/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_22/strided_slice/stack_1И
lstm_22/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_22/strided_slice/stack_2Т
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
B :Д2
lstm_22/zeros/mul/yМ
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
B :ш2
lstm_22/zeros/Less/yЗ
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
B :Д2
lstm_22/zeros/packed/1г
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
lstm_22/zeros/ConstЦ
lstm_22/zerosFilllstm_22/zeros/packed:output:0lstm_22/zeros/Const:output:0*
T0*(
_output_shapes
:         Д2
lstm_22/zerosq
lstm_22/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :Д2
lstm_22/zeros_1/mul/yТ
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
B :ш2
lstm_22/zeros_1/Less/yП
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
B :Д2
lstm_22/zeros_1/packed/1й
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
lstm_22/zeros_1/ConstЮ
lstm_22/zeros_1Filllstm_22/zeros_1/packed:output:0lstm_22/zeros_1/Const:output:0*
T0*(
_output_shapes
:         Д2
lstm_22/zeros_1Е
lstm_22/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_22/transpose/permТ
lstm_22/transpose	Transposeinputslstm_22/transpose/perm:output:0*
T0*+
_output_shapes
:         ]2
lstm_22/transposeg
lstm_22/Shape_1Shapelstm_22/transpose:y:0*
T0*
_output_shapes
:2
lstm_22/Shape_1И
lstm_22/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_22/strided_slice_1/stackМ
lstm_22/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_22/strided_slice_1/stack_1М
lstm_22/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_22/strided_slice_1/stack_2Ю
lstm_22/strided_slice_1StridedSlicelstm_22/Shape_1:output:0&lstm_22/strided_slice_1/stack:output:0(lstm_22/strided_slice_1/stack_1:output:0(lstm_22/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_22/strided_slice_1Х
#lstm_22/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         2%
#lstm_22/TensorArrayV2/element_shape╥
lstm_22/TensorArrayV2TensorListReserve,lstm_22/TensorArrayV2/element_shape:output:0 lstm_22/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_22/TensorArrayV2╧
=lstm_22/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ]   2?
=lstm_22/TensorArrayUnstack/TensorListFromTensor/element_shapeШ
/lstm_22/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_22/transpose:y:0Flstm_22/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type021
/lstm_22/TensorArrayUnstack/TensorListFromTensorИ
lstm_22/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_22/strided_slice_2/stackМ
lstm_22/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_22/strided_slice_2/stack_1М
lstm_22/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_22/strided_slice_2/stack_2м
lstm_22/strided_slice_2StridedSlicelstm_22/transpose:y:0&lstm_22/strided_slice_2/stack:output:0(lstm_22/strided_slice_2/stack_1:output:0(lstm_22/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         ]*
shrink_axis_mask2
lstm_22/strided_slice_2═
*lstm_22/lstm_cell_22/MatMul/ReadVariableOpReadVariableOp3lstm_22_lstm_cell_22_matmul_readvariableop_resource*
_output_shapes
:	]Р*
dtype02,
*lstm_22/lstm_cell_22/MatMul/ReadVariableOp═
lstm_22/lstm_cell_22/MatMulMatMul lstm_22/strided_slice_2:output:02lstm_22/lstm_cell_22/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Р2
lstm_22/lstm_cell_22/MatMul╘
,lstm_22/lstm_cell_22/MatMul_1/ReadVariableOpReadVariableOp5lstm_22_lstm_cell_22_matmul_1_readvariableop_resource* 
_output_shapes
:
ДР*
dtype02.
,lstm_22/lstm_cell_22/MatMul_1/ReadVariableOp╔
lstm_22/lstm_cell_22/MatMul_1MatMullstm_22/zeros:output:04lstm_22/lstm_cell_22/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Р2
lstm_22/lstm_cell_22/MatMul_1└
lstm_22/lstm_cell_22/addAddV2%lstm_22/lstm_cell_22/MatMul:product:0'lstm_22/lstm_cell_22/MatMul_1:product:0*
T0*(
_output_shapes
:         Р2
lstm_22/lstm_cell_22/add╠
+lstm_22/lstm_cell_22/BiasAdd/ReadVariableOpReadVariableOp4lstm_22_lstm_cell_22_biasadd_readvariableop_resource*
_output_shapes	
:Р*
dtype02-
+lstm_22/lstm_cell_22/BiasAdd/ReadVariableOp═
lstm_22/lstm_cell_22/BiasAddBiasAddlstm_22/lstm_cell_22/add:z:03lstm_22/lstm_cell_22/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Р2
lstm_22/lstm_cell_22/BiasAddО
$lstm_22/lstm_cell_22/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2&
$lstm_22/lstm_cell_22/split/split_dimЧ
lstm_22/lstm_cell_22/splitSplit-lstm_22/lstm_cell_22/split/split_dim:output:0%lstm_22/lstm_cell_22/BiasAdd:output:0*
T0*d
_output_shapesR
P:         Д:         Д:         Д:         Д*
	num_split2
lstm_22/lstm_cell_22/splitЯ
lstm_22/lstm_cell_22/SigmoidSigmoid#lstm_22/lstm_cell_22/split:output:0*
T0*(
_output_shapes
:         Д2
lstm_22/lstm_cell_22/Sigmoidг
lstm_22/lstm_cell_22/Sigmoid_1Sigmoid#lstm_22/lstm_cell_22/split:output:1*
T0*(
_output_shapes
:         Д2 
lstm_22/lstm_cell_22/Sigmoid_1м
lstm_22/lstm_cell_22/mulMul"lstm_22/lstm_cell_22/Sigmoid_1:y:0lstm_22/zeros_1:output:0*
T0*(
_output_shapes
:         Д2
lstm_22/lstm_cell_22/mulЦ
lstm_22/lstm_cell_22/ReluRelu#lstm_22/lstm_cell_22/split:output:2*
T0*(
_output_shapes
:         Д2
lstm_22/lstm_cell_22/Relu╜
lstm_22/lstm_cell_22/mul_1Mul lstm_22/lstm_cell_22/Sigmoid:y:0'lstm_22/lstm_cell_22/Relu:activations:0*
T0*(
_output_shapes
:         Д2
lstm_22/lstm_cell_22/mul_1▓
lstm_22/lstm_cell_22/add_1AddV2lstm_22/lstm_cell_22/mul:z:0lstm_22/lstm_cell_22/mul_1:z:0*
T0*(
_output_shapes
:         Д2
lstm_22/lstm_cell_22/add_1г
lstm_22/lstm_cell_22/Sigmoid_2Sigmoid#lstm_22/lstm_cell_22/split:output:3*
T0*(
_output_shapes
:         Д2 
lstm_22/lstm_cell_22/Sigmoid_2Х
lstm_22/lstm_cell_22/Relu_1Relulstm_22/lstm_cell_22/add_1:z:0*
T0*(
_output_shapes
:         Д2
lstm_22/lstm_cell_22/Relu_1┴
lstm_22/lstm_cell_22/mul_2Mul"lstm_22/lstm_cell_22/Sigmoid_2:y:0)lstm_22/lstm_cell_22/Relu_1:activations:0*
T0*(
_output_shapes
:         Д2
lstm_22/lstm_cell_22/mul_2Я
%lstm_22/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    Д   2'
%lstm_22/TensorArrayV2_1/element_shape╪
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
lstm_22/timeП
 lstm_22/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         2"
 lstm_22/while/maximum_iterationsz
lstm_22/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_22/while/loop_counterЛ
lstm_22/whileWhile#lstm_22/while/loop_counter:output:0)lstm_22/while/maximum_iterations:output:0lstm_22/time:output:0 lstm_22/TensorArrayV2_1:handle:0lstm_22/zeros:output:0lstm_22/zeros_1:output:0 lstm_22/strided_slice_1:output:0?lstm_22/TensorArrayUnstack/TensorListFromTensor:output_handle:03lstm_22_lstm_cell_22_matmul_readvariableop_resource5lstm_22_lstm_cell_22_matmul_1_readvariableop_resource4lstm_22_lstm_cell_22_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :         Д:         Д: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *'
bodyR
lstm_22_while_body_39110703*'
condR
lstm_22_while_cond_39110702*M
output_shapes<
:: : : : :         Д:         Д: : : : : *
parallel_iterations 2
lstm_22/while┼
8lstm_22/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    Д   2:
8lstm_22/TensorArrayV2Stack/TensorListStack/element_shapeЙ
*lstm_22/TensorArrayV2Stack/TensorListStackTensorListStacklstm_22/while:output:3Alstm_22/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:         Д*
element_dtype02,
*lstm_22/TensorArrayV2Stack/TensorListStackС
lstm_22/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         2
lstm_22/strided_slice_3/stackМ
lstm_22/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2!
lstm_22/strided_slice_3/stack_1М
lstm_22/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_22/strided_slice_3/stack_2╦
lstm_22/strided_slice_3StridedSlice3lstm_22/TensorArrayV2Stack/TensorListStack:tensor:0&lstm_22/strided_slice_3/stack:output:0(lstm_22/strided_slice_3/stack_1:output:0(lstm_22/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         Д*
shrink_axis_mask2
lstm_22/strided_slice_3Й
lstm_22/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_22/transpose_1/perm╞
lstm_22/transpose_1	Transpose3lstm_22/TensorArrayV2Stack/TensorListStack:tensor:0!lstm_22/transpose_1/perm:output:0*
T0*,
_output_shapes
:         Д2
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
 *лкк?2
dropout_22/dropout/Constк
dropout_22/dropout/MulMullstm_22/transpose_1:y:0!dropout_22/dropout/Const:output:0*
T0*,
_output_shapes
:         Д2
dropout_22/dropout/Mul{
dropout_22/dropout/ShapeShapelstm_22/transpose_1:y:0*
T0*
_output_shapes
:2
dropout_22/dropout/Shape┌
/dropout_22/dropout/random_uniform/RandomUniformRandomUniform!dropout_22/dropout/Shape:output:0*
T0*,
_output_shapes
:         Д*
dtype021
/dropout_22/dropout/random_uniform/RandomUniformЛ
!dropout_22/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А>2#
!dropout_22/dropout/GreaterEqual/yя
dropout_22/dropout/GreaterEqualGreaterEqual8dropout_22/dropout/random_uniform/RandomUniform:output:0*dropout_22/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:         Д2!
dropout_22/dropout/GreaterEqualе
dropout_22/dropout/CastCast#dropout_22/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:         Д2
dropout_22/dropout/Castл
dropout_22/dropout/Mul_1Muldropout_22/dropout/Mul:z:0dropout_22/dropout/Cast:y:0*
T0*,
_output_shapes
:         Д2
dropout_22/dropout/Mul_1j
lstm_23/ShapeShapedropout_22/dropout/Mul_1:z:0*
T0*
_output_shapes
:2
lstm_23/ShapeД
lstm_23/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_23/strided_slice/stackИ
lstm_23/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_23/strided_slice/stack_1И
lstm_23/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_23/strided_slice/stack_2Т
lstm_23/strided_sliceStridedSlicelstm_23/Shape:output:0$lstm_23/strided_slice/stack:output:0&lstm_23/strided_slice/stack_1:output:0&lstm_23/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_23/strided_slicel
lstm_23/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :m2
lstm_23/zeros/mul/yМ
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
B :ш2
lstm_23/zeros/Less/yЗ
lstm_23/zeros/LessLesslstm_23/zeros/mul:z:0lstm_23/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_23/zeros/Lessr
lstm_23/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :m2
lstm_23/zeros/packed/1г
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
lstm_23/zeros/ConstХ
lstm_23/zerosFilllstm_23/zeros/packed:output:0lstm_23/zeros/Const:output:0*
T0*'
_output_shapes
:         m2
lstm_23/zerosp
lstm_23/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :m2
lstm_23/zeros_1/mul/yТ
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
B :ш2
lstm_23/zeros_1/Less/yП
lstm_23/zeros_1/LessLesslstm_23/zeros_1/mul:z:0lstm_23/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_23/zeros_1/Lessv
lstm_23/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :m2
lstm_23/zeros_1/packed/1й
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
lstm_23/zeros_1/ConstЭ
lstm_23/zeros_1Filllstm_23/zeros_1/packed:output:0lstm_23/zeros_1/Const:output:0*
T0*'
_output_shapes
:         m2
lstm_23/zeros_1Е
lstm_23/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_23/transpose/permй
lstm_23/transpose	Transposedropout_22/dropout/Mul_1:z:0lstm_23/transpose/perm:output:0*
T0*,
_output_shapes
:         Д2
lstm_23/transposeg
lstm_23/Shape_1Shapelstm_23/transpose:y:0*
T0*
_output_shapes
:2
lstm_23/Shape_1И
lstm_23/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_23/strided_slice_1/stackМ
lstm_23/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_23/strided_slice_1/stack_1М
lstm_23/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_23/strided_slice_1/stack_2Ю
lstm_23/strided_slice_1StridedSlicelstm_23/Shape_1:output:0&lstm_23/strided_slice_1/stack:output:0(lstm_23/strided_slice_1/stack_1:output:0(lstm_23/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_23/strided_slice_1Х
#lstm_23/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         2%
#lstm_23/TensorArrayV2/element_shape╥
lstm_23/TensorArrayV2TensorListReserve,lstm_23/TensorArrayV2/element_shape:output:0 lstm_23/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_23/TensorArrayV2╧
=lstm_23/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    Д   2?
=lstm_23/TensorArrayUnstack/TensorListFromTensor/element_shapeШ
/lstm_23/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_23/transpose:y:0Flstm_23/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type021
/lstm_23/TensorArrayUnstack/TensorListFromTensorИ
lstm_23/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_23/strided_slice_2/stackМ
lstm_23/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_23/strided_slice_2/stack_1М
lstm_23/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_23/strided_slice_2/stack_2н
lstm_23/strided_slice_2StridedSlicelstm_23/transpose:y:0&lstm_23/strided_slice_2/stack:output:0(lstm_23/strided_slice_2/stack_1:output:0(lstm_23/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         Д*
shrink_axis_mask2
lstm_23/strided_slice_2╬
*lstm_23/lstm_cell_23/MatMul/ReadVariableOpReadVariableOp3lstm_23_lstm_cell_23_matmul_readvariableop_resource* 
_output_shapes
:
Д┤*
dtype02,
*lstm_23/lstm_cell_23/MatMul/ReadVariableOp═
lstm_23/lstm_cell_23/MatMulMatMul lstm_23/strided_slice_2:output:02lstm_23/lstm_cell_23/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ┤2
lstm_23/lstm_cell_23/MatMul╙
,lstm_23/lstm_cell_23/MatMul_1/ReadVariableOpReadVariableOp5lstm_23_lstm_cell_23_matmul_1_readvariableop_resource*
_output_shapes
:	m┤*
dtype02.
,lstm_23/lstm_cell_23/MatMul_1/ReadVariableOp╔
lstm_23/lstm_cell_23/MatMul_1MatMullstm_23/zeros:output:04lstm_23/lstm_cell_23/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ┤2
lstm_23/lstm_cell_23/MatMul_1└
lstm_23/lstm_cell_23/addAddV2%lstm_23/lstm_cell_23/MatMul:product:0'lstm_23/lstm_cell_23/MatMul_1:product:0*
T0*(
_output_shapes
:         ┤2
lstm_23/lstm_cell_23/add╠
+lstm_23/lstm_cell_23/BiasAdd/ReadVariableOpReadVariableOp4lstm_23_lstm_cell_23_biasadd_readvariableop_resource*
_output_shapes	
:┤*
dtype02-
+lstm_23/lstm_cell_23/BiasAdd/ReadVariableOp═
lstm_23/lstm_cell_23/BiasAddBiasAddlstm_23/lstm_cell_23/add:z:03lstm_23/lstm_cell_23/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ┤2
lstm_23/lstm_cell_23/BiasAddО
$lstm_23/lstm_cell_23/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2&
$lstm_23/lstm_cell_23/split/split_dimУ
lstm_23/lstm_cell_23/splitSplit-lstm_23/lstm_cell_23/split/split_dim:output:0%lstm_23/lstm_cell_23/BiasAdd:output:0*
T0*`
_output_shapesN
L:         m:         m:         m:         m*
	num_split2
lstm_23/lstm_cell_23/splitЮ
lstm_23/lstm_cell_23/SigmoidSigmoid#lstm_23/lstm_cell_23/split:output:0*
T0*'
_output_shapes
:         m2
lstm_23/lstm_cell_23/Sigmoidв
lstm_23/lstm_cell_23/Sigmoid_1Sigmoid#lstm_23/lstm_cell_23/split:output:1*
T0*'
_output_shapes
:         m2 
lstm_23/lstm_cell_23/Sigmoid_1л
lstm_23/lstm_cell_23/mulMul"lstm_23/lstm_cell_23/Sigmoid_1:y:0lstm_23/zeros_1:output:0*
T0*'
_output_shapes
:         m2
lstm_23/lstm_cell_23/mulХ
lstm_23/lstm_cell_23/ReluRelu#lstm_23/lstm_cell_23/split:output:2*
T0*'
_output_shapes
:         m2
lstm_23/lstm_cell_23/Relu╝
lstm_23/lstm_cell_23/mul_1Mul lstm_23/lstm_cell_23/Sigmoid:y:0'lstm_23/lstm_cell_23/Relu:activations:0*
T0*'
_output_shapes
:         m2
lstm_23/lstm_cell_23/mul_1▒
lstm_23/lstm_cell_23/add_1AddV2lstm_23/lstm_cell_23/mul:z:0lstm_23/lstm_cell_23/mul_1:z:0*
T0*'
_output_shapes
:         m2
lstm_23/lstm_cell_23/add_1в
lstm_23/lstm_cell_23/Sigmoid_2Sigmoid#lstm_23/lstm_cell_23/split:output:3*
T0*'
_output_shapes
:         m2 
lstm_23/lstm_cell_23/Sigmoid_2Ф
lstm_23/lstm_cell_23/Relu_1Relulstm_23/lstm_cell_23/add_1:z:0*
T0*'
_output_shapes
:         m2
lstm_23/lstm_cell_23/Relu_1└
lstm_23/lstm_cell_23/mul_2Mul"lstm_23/lstm_cell_23/Sigmoid_2:y:0)lstm_23/lstm_cell_23/Relu_1:activations:0*
T0*'
_output_shapes
:         m2
lstm_23/lstm_cell_23/mul_2Я
%lstm_23/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    m   2'
%lstm_23/TensorArrayV2_1/element_shape╪
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
lstm_23/timeП
 lstm_23/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         2"
 lstm_23/while/maximum_iterationsz
lstm_23/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_23/while/loop_counterЗ
lstm_23/whileWhile#lstm_23/while/loop_counter:output:0)lstm_23/while/maximum_iterations:output:0lstm_23/time:output:0 lstm_23/TensorArrayV2_1:handle:0lstm_23/zeros:output:0lstm_23/zeros_1:output:0 lstm_23/strided_slice_1:output:0?lstm_23/TensorArrayUnstack/TensorListFromTensor:output_handle:03lstm_23_lstm_cell_23_matmul_readvariableop_resource5lstm_23_lstm_cell_23_matmul_1_readvariableop_resource4lstm_23_lstm_cell_23_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         m:         m: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *'
bodyR
lstm_23_while_body_39110858*'
condR
lstm_23_while_cond_39110857*K
output_shapes:
8: : : : :         m:         m: : : : : *
parallel_iterations 2
lstm_23/while┼
8lstm_23/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    m   2:
8lstm_23/TensorArrayV2Stack/TensorListStack/element_shapeИ
*lstm_23/TensorArrayV2Stack/TensorListStackTensorListStacklstm_23/while:output:3Alstm_23/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:         m*
element_dtype02,
*lstm_23/TensorArrayV2Stack/TensorListStackС
lstm_23/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         2
lstm_23/strided_slice_3/stackМ
lstm_23/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2!
lstm_23/strided_slice_3/stack_1М
lstm_23/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_23/strided_slice_3/stack_2╩
lstm_23/strided_slice_3StridedSlice3lstm_23/TensorArrayV2Stack/TensorListStack:tensor:0&lstm_23/strided_slice_3/stack:output:0(lstm_23/strided_slice_3/stack_1:output:0(lstm_23/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         m*
shrink_axis_mask2
lstm_23/strided_slice_3Й
lstm_23/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_23/transpose_1/perm┼
lstm_23/transpose_1	Transpose3lstm_23/TensorArrayV2Stack/TensorListStack:tensor:0!lstm_23/transpose_1/perm:output:0*
T0*+
_output_shapes
:         m2
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
 *  а?2
dropout_23/dropout/Constй
dropout_23/dropout/MulMullstm_23/transpose_1:y:0!dropout_23/dropout/Const:output:0*
T0*+
_output_shapes
:         m2
dropout_23/dropout/Mul{
dropout_23/dropout/ShapeShapelstm_23/transpose_1:y:0*
T0*
_output_shapes
:2
dropout_23/dropout/Shape┘
/dropout_23/dropout/random_uniform/RandomUniformRandomUniform!dropout_23/dropout/Shape:output:0*
T0*+
_output_shapes
:         m*
dtype021
/dropout_23/dropout/random_uniform/RandomUniformЛ
!dropout_23/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>2#
!dropout_23/dropout/GreaterEqual/yю
dropout_23/dropout/GreaterEqualGreaterEqual8dropout_23/dropout/random_uniform/RandomUniform:output:0*dropout_23/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:         m2!
dropout_23/dropout/GreaterEqualд
dropout_23/dropout/CastCast#dropout_23/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:         m2
dropout_23/dropout/Castк
dropout_23/dropout/Mul_1Muldropout_23/dropout/Mul:z:0dropout_23/dropout/Cast:y:0*
T0*+
_output_shapes
:         m2
dropout_23/dropout/Mul_1▒
!dense_11/Tensordot/ReadVariableOpReadVariableOp*dense_11_tensordot_readvariableop_resource*
_output_shapes

:m*
dtype02#
!dense_11/Tensordot/ReadVariableOp|
dense_11/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_11/Tensordot/axesГ
dense_11/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_11/Tensordot/freeА
dense_11/Tensordot/ShapeShapedropout_23/dropout/Mul_1:z:0*
T0*
_output_shapes
:2
dense_11/Tensordot/ShapeЖ
 dense_11/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_11/Tensordot/GatherV2/axis■
dense_11/Tensordot/GatherV2GatherV2!dense_11/Tensordot/Shape:output:0 dense_11/Tensordot/free:output:0)dense_11/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_11/Tensordot/GatherV2К
"dense_11/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"dense_11/Tensordot/GatherV2_1/axisД
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
dense_11/Tensordot/Constд
dense_11/Tensordot/ProdProd$dense_11/Tensordot/GatherV2:output:0!dense_11/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_11/Tensordot/ProdВ
dense_11/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_11/Tensordot/Const_1м
dense_11/Tensordot/Prod_1Prod&dense_11/Tensordot/GatherV2_1:output:0#dense_11/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_11/Tensordot/Prod_1В
dense_11/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2 
dense_11/Tensordot/concat/axis▌
dense_11/Tensordot/concatConcatV2 dense_11/Tensordot/free:output:0 dense_11/Tensordot/axes:output:0'dense_11/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_11/Tensordot/concat░
dense_11/Tensordot/stackPack dense_11/Tensordot/Prod:output:0"dense_11/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_11/Tensordot/stack┴
dense_11/Tensordot/transpose	Transposedropout_23/dropout/Mul_1:z:0"dense_11/Tensordot/concat:output:0*
T0*+
_output_shapes
:         m2
dense_11/Tensordot/transpose├
dense_11/Tensordot/ReshapeReshape dense_11/Tensordot/transpose:y:0!dense_11/Tensordot/stack:output:0*
T0*0
_output_shapes
:                  2
dense_11/Tensordot/Reshape┬
dense_11/Tensordot/MatMulMatMul#dense_11/Tensordot/Reshape:output:0)dense_11/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_11/Tensordot/MatMulВ
dense_11/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
dense_11/Tensordot/Const_2Ж
 dense_11/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_11/Tensordot/concat_1/axisъ
dense_11/Tensordot/concat_1ConcatV2$dense_11/Tensordot/GatherV2:output:0#dense_11/Tensordot/Const_2:output:0)dense_11/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_11/Tensordot/concat_1┤
dense_11/TensordotReshape#dense_11/Tensordot/MatMul:product:0$dense_11/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:         2
dense_11/Tensordotз
dense_11/BiasAdd/ReadVariableOpReadVariableOp(dense_11_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_11/BiasAdd/ReadVariableOpл
dense_11/BiasAddBiasAdddense_11/Tensordot:output:0'dense_11/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:         2
dense_11/BiasAddА
dense_11/SoftmaxSoftmaxdense_11/BiasAdd:output:0*
T0*+
_output_shapes
:         2
dense_11/Softmaxy
IdentityIdentitydense_11/Softmax:softmax:0^NoOp*
T0*+
_output_shapes
:         2

Identity╚
NoOpNoOp ^dense_11/BiasAdd/ReadVariableOp"^dense_11/Tensordot/ReadVariableOp,^lstm_22/lstm_cell_22/BiasAdd/ReadVariableOp+^lstm_22/lstm_cell_22/MatMul/ReadVariableOp-^lstm_22/lstm_cell_22/MatMul_1/ReadVariableOp^lstm_22/while,^lstm_23/lstm_cell_23/BiasAdd/ReadVariableOp+^lstm_23/lstm_cell_23/MatMul/ReadVariableOp-^lstm_23/lstm_cell_23/MatMul_1/ReadVariableOp^lstm_23/while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':         ]: : : : : : : : 2B
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
:         ]
 
_user_specified_nameinputs
Б
Й
J__inference_lstm_cell_23_layer_call_and_return_conditional_losses_39112571

inputs
states_0
states_12
matmul_readvariableop_resource:
Д┤3
 matmul_1_readvariableop_resource:	m┤.
biasadd_readvariableop_resource:	┤
identity

identity_1

identity_2ИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpвMatMul_1/ReadVariableOpП
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
Д┤*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ┤2
MatMulФ
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	m┤*
dtype02
MatMul_1/ReadVariableOp|
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ┤2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:         ┤2
addН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:┤*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ┤2	
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
L:         m:         m:         m:         m*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:         m2	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:         m2
	Sigmoid_1\
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:         m2
mulV
ReluRelusplit:output:2*
T0*'
_output_shapes
:         m2
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:         m2
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:         m2
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:         m2
	Sigmoid_2U
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:         m2
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:         m2
mul_2d
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:         m2

Identityh

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:         m2

Identity_1h

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:         m2

Identity_2Щ
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:         Д:         m:         m: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:P L
(
_output_shapes
:         Д
 
_user_specified_nameinputs:QM
'
_output_shapes
:         m
"
_user_specified_name
states/0:QM
'
_output_shapes
:         m
"
_user_specified_name
states/1
Ю?
╘
while_body_39109859
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
3while_lstm_cell_23_matmul_readvariableop_resource_0:
Д┤H
5while_lstm_cell_23_matmul_1_readvariableop_resource_0:	m┤C
4while_lstm_cell_23_biasadd_readvariableop_resource_0:	┤
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
1while_lstm_cell_23_matmul_readvariableop_resource:
Д┤F
3while_lstm_cell_23_matmul_1_readvariableop_resource:	m┤A
2while_lstm_cell_23_biasadd_readvariableop_resource:	┤Ив)while/lstm_cell_23/BiasAdd/ReadVariableOpв(while/lstm_cell_23/MatMul/ReadVariableOpв*while/lstm_cell_23/MatMul_1/ReadVariableOp├
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    Д   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape╘
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:         Д*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem╩
(while/lstm_cell_23/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_23_matmul_readvariableop_resource_0* 
_output_shapes
:
Д┤*
dtype02*
(while/lstm_cell_23/MatMul/ReadVariableOp╫
while/lstm_cell_23/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_23/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ┤2
while/lstm_cell_23/MatMul╧
*while/lstm_cell_23/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_23_matmul_1_readvariableop_resource_0*
_output_shapes
:	m┤*
dtype02,
*while/lstm_cell_23/MatMul_1/ReadVariableOp└
while/lstm_cell_23/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_23/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ┤2
while/lstm_cell_23/MatMul_1╕
while/lstm_cell_23/addAddV2#while/lstm_cell_23/MatMul:product:0%while/lstm_cell_23/MatMul_1:product:0*
T0*(
_output_shapes
:         ┤2
while/lstm_cell_23/add╚
)while/lstm_cell_23/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_23_biasadd_readvariableop_resource_0*
_output_shapes	
:┤*
dtype02+
)while/lstm_cell_23/BiasAdd/ReadVariableOp┼
while/lstm_cell_23/BiasAddBiasAddwhile/lstm_cell_23/add:z:01while/lstm_cell_23/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ┤2
while/lstm_cell_23/BiasAddК
"while/lstm_cell_23/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_23/split/split_dimЛ
while/lstm_cell_23/splitSplit+while/lstm_cell_23/split/split_dim:output:0#while/lstm_cell_23/BiasAdd:output:0*
T0*`
_output_shapesN
L:         m:         m:         m:         m*
	num_split2
while/lstm_cell_23/splitШ
while/lstm_cell_23/SigmoidSigmoid!while/lstm_cell_23/split:output:0*
T0*'
_output_shapes
:         m2
while/lstm_cell_23/SigmoidЬ
while/lstm_cell_23/Sigmoid_1Sigmoid!while/lstm_cell_23/split:output:1*
T0*'
_output_shapes
:         m2
while/lstm_cell_23/Sigmoid_1а
while/lstm_cell_23/mulMul while/lstm_cell_23/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:         m2
while/lstm_cell_23/mulП
while/lstm_cell_23/ReluRelu!while/lstm_cell_23/split:output:2*
T0*'
_output_shapes
:         m2
while/lstm_cell_23/Relu┤
while/lstm_cell_23/mul_1Mulwhile/lstm_cell_23/Sigmoid:y:0%while/lstm_cell_23/Relu:activations:0*
T0*'
_output_shapes
:         m2
while/lstm_cell_23/mul_1й
while/lstm_cell_23/add_1AddV2while/lstm_cell_23/mul:z:0while/lstm_cell_23/mul_1:z:0*
T0*'
_output_shapes
:         m2
while/lstm_cell_23/add_1Ь
while/lstm_cell_23/Sigmoid_2Sigmoid!while/lstm_cell_23/split:output:3*
T0*'
_output_shapes
:         m2
while/lstm_cell_23/Sigmoid_2О
while/lstm_cell_23/Relu_1Reluwhile/lstm_cell_23/add_1:z:0*
T0*'
_output_shapes
:         m2
while/lstm_cell_23/Relu_1╕
while/lstm_cell_23/mul_2Mul while/lstm_cell_23/Sigmoid_2:y:0'while/lstm_cell_23/Relu_1:activations:0*
T0*'
_output_shapes
:         m2
while/lstm_cell_23/mul_2р
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
while/Identity_2Ъ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3Н
while/Identity_4Identitywhile/lstm_cell_23/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:         m2
while/Identity_4Н
while/Identity_5Identitywhile/lstm_cell_23/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:         m2
while/Identity_5▐

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
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         m:         m: : : : : 2V
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
: :-)
'
_output_shapes
:         m:-)
'
_output_shapes
:         m:

_output_shapes
: :

_output_shapes
: 
у
═
while_cond_39111387
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_39111387___redundant_placeholder06
2while_while_cond_39111387___redundant_placeholder16
2while_while_cond_39111387___redundant_placeholder26
2while_while_cond_39111387___redundant_placeholder3
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
B: : : : :         Д:         Д: ::::: 
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
:         Д:.*
(
_output_shapes
:         Д:

_output_shapes
: :

_output_shapes
:
╦F
О
E__inference_lstm_22_layer_call_and_return_conditional_losses_39108460

inputs(
lstm_cell_22_39108378:	]Р)
lstm_cell_22_39108380:
ДР$
lstm_cell_22_39108382:	Р
identityИв$lstm_cell_22/StatefulPartitionedCallвwhileD
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
B :Д2
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
B :Д2
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
:         Д2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :Д2
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
B :Д2
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
:         Д2	
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
$lstm_cell_22/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_22_39108378lstm_cell_22_39108380lstm_cell_22_39108382*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:         Д:         Д:         Д*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_lstm_cell_22_layer_call_and_return_conditional_losses_391083132&
$lstm_cell_22/StatefulPartitionedCallП
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    Д   2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_22_39108378lstm_cell_22_39108380lstm_cell_22_39108382*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :         Д:         Д: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_39108391*
condR
while_cond_39108390*M
output_shapes<
:: : : : :         Д:         Д: : : : : *
parallel_iterations 2
while╡
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    Д   22
0TensorArrayV2Stack/TensorListStack/element_shapeЄ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:                  Д*
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
:         Д*
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
!:                  Д2
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
!:                  Д2

Identity}
NoOpNoOp%^lstm_cell_22/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  ]: : : 2L
$lstm_cell_22/StatefulPartitionedCall$lstm_cell_22/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :                  ]
 
_user_specified_nameinputs
╡
╕
*__inference_lstm_23_layer_call_fn_39112331

inputs
unknown:
Д┤
	unknown_0:	m┤
	unknown_1:	┤
identityИвStatefulPartitionedCallЖ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         m*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_lstm_23_layer_call_and_return_conditional_losses_391096742
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:         m2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:         Д: : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:         Д
 
_user_specified_nameinputs
р
║
*__inference_lstm_23_layer_call_fn_39112309
inputs_0
unknown:
Д┤
	unknown_0:	m┤
	unknown_1:	┤
identityИвStatefulPartitionedCallС
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :                  m*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_lstm_23_layer_call_and_return_conditional_losses_391088802
StatefulPartitionedCallИ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :                  m2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':                  Д: : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
5
_output_shapes#
!:                  Д
"
_user_specified_name
inputs/0
Д\
Ю
E__inference_lstm_22_layer_call_and_return_conditional_losses_39110139

inputs>
+lstm_cell_22_matmul_readvariableop_resource:	]РA
-lstm_cell_22_matmul_1_readvariableop_resource:
ДР;
,lstm_cell_22_biasadd_readvariableop_resource:	Р
identityИв#lstm_cell_22/BiasAdd/ReadVariableOpв"lstm_cell_22/MatMul/ReadVariableOpв$lstm_cell_22/MatMul_1/ReadVariableOpвwhileD
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
B :Д2
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
B :Д2
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
:         Д2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :Д2
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
B :Д2
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
:         Д2	
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
"lstm_cell_22/MatMul/ReadVariableOpReadVariableOp+lstm_cell_22_matmul_readvariableop_resource*
_output_shapes
:	]Р*
dtype02$
"lstm_cell_22/MatMul/ReadVariableOpн
lstm_cell_22/MatMulMatMulstrided_slice_2:output:0*lstm_cell_22/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Р2
lstm_cell_22/MatMul╝
$lstm_cell_22/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_22_matmul_1_readvariableop_resource* 
_output_shapes
:
ДР*
dtype02&
$lstm_cell_22/MatMul_1/ReadVariableOpй
lstm_cell_22/MatMul_1MatMulzeros:output:0,lstm_cell_22/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Р2
lstm_cell_22/MatMul_1а
lstm_cell_22/addAddV2lstm_cell_22/MatMul:product:0lstm_cell_22/MatMul_1:product:0*
T0*(
_output_shapes
:         Р2
lstm_cell_22/add┤
#lstm_cell_22/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_22_biasadd_readvariableop_resource*
_output_shapes	
:Р*
dtype02%
#lstm_cell_22/BiasAdd/ReadVariableOpн
lstm_cell_22/BiasAddBiasAddlstm_cell_22/add:z:0+lstm_cell_22/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Р2
lstm_cell_22/BiasAdd~
lstm_cell_22/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_22/split/split_dimў
lstm_cell_22/splitSplit%lstm_cell_22/split/split_dim:output:0lstm_cell_22/BiasAdd:output:0*
T0*d
_output_shapesR
P:         Д:         Д:         Д:         Д*
	num_split2
lstm_cell_22/splitЗ
lstm_cell_22/SigmoidSigmoidlstm_cell_22/split:output:0*
T0*(
_output_shapes
:         Д2
lstm_cell_22/SigmoidЛ
lstm_cell_22/Sigmoid_1Sigmoidlstm_cell_22/split:output:1*
T0*(
_output_shapes
:         Д2
lstm_cell_22/Sigmoid_1М
lstm_cell_22/mulMullstm_cell_22/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:         Д2
lstm_cell_22/mul~
lstm_cell_22/ReluRelulstm_cell_22/split:output:2*
T0*(
_output_shapes
:         Д2
lstm_cell_22/ReluЭ
lstm_cell_22/mul_1Mullstm_cell_22/Sigmoid:y:0lstm_cell_22/Relu:activations:0*
T0*(
_output_shapes
:         Д2
lstm_cell_22/mul_1Т
lstm_cell_22/add_1AddV2lstm_cell_22/mul:z:0lstm_cell_22/mul_1:z:0*
T0*(
_output_shapes
:         Д2
lstm_cell_22/add_1Л
lstm_cell_22/Sigmoid_2Sigmoidlstm_cell_22/split:output:3*
T0*(
_output_shapes
:         Д2
lstm_cell_22/Sigmoid_2}
lstm_cell_22/Relu_1Relulstm_cell_22/add_1:z:0*
T0*(
_output_shapes
:         Д2
lstm_cell_22/Relu_1б
lstm_cell_22/mul_2Mullstm_cell_22/Sigmoid_2:y:0!lstm_cell_22/Relu_1:activations:0*
T0*(
_output_shapes
:         Д2
lstm_cell_22/mul_2П
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    Д   2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_22_matmul_readvariableop_resource-lstm_cell_22_matmul_1_readvariableop_resource,lstm_cell_22_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :         Д:         Д: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_39110055*
condR
while_cond_39110054*M
output_shapes<
:: : : : :         Д:         Д: : : : : *
parallel_iterations 2
while╡
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    Д   22
0TensorArrayV2Stack/TensorListStack/element_shapeщ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:         Д*
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
:         Д*
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
:         Д2
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
:         Д2

Identity╚
NoOpNoOp$^lstm_cell_22/BiasAdd/ReadVariableOp#^lstm_cell_22/MatMul/ReadVariableOp%^lstm_cell_22/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         ]: : : 2J
#lstm_cell_22/BiasAdd/ReadVariableOp#lstm_cell_22/BiasAdd/ReadVariableOp2H
"lstm_cell_22/MatMul/ReadVariableOp"lstm_cell_22/MatMul/ReadVariableOp2L
$lstm_cell_22/MatMul_1/ReadVariableOp$lstm_cell_22/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:         ]
 
_user_specified_nameinputs
Ч

╦
0__inference_sequential_11_layer_call_fn_39111019

inputs
unknown:	]Р
	unknown_0:
ДР
	unknown_1:	Р
	unknown_2:
Д┤
	unknown_3:	m┤
	unknown_4:	┤
	unknown_5:m
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
K__inference_sequential_11_layer_call_and_return_conditional_losses_391101962
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
╜
∙
/__inference_lstm_cell_23_layer_call_fn_39112605

inputs
states_0
states_1
unknown:
Д┤
	unknown_0:	m┤
	unknown_1:	┤
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
9:         m:         m:         m*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_lstm_cell_23_layer_call_and_return_conditional_losses_391089432
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         m2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:         m2

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:         m2

Identity_2h
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:         Д:         m:         m: : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         Д
 
_user_specified_nameinputs:QM
'
_output_shapes
:         m
"
_user_specified_name
states/0:QM
'
_output_shapes
:         m
"
_user_specified_name
states/1
И╣
х	
#__inference__wrapped_model_39108092
lstm_22_inputT
Asequential_11_lstm_22_lstm_cell_22_matmul_readvariableop_resource:	]РW
Csequential_11_lstm_22_lstm_cell_22_matmul_1_readvariableop_resource:
ДРQ
Bsequential_11_lstm_22_lstm_cell_22_biasadd_readvariableop_resource:	РU
Asequential_11_lstm_23_lstm_cell_23_matmul_readvariableop_resource:
Д┤V
Csequential_11_lstm_23_lstm_cell_23_matmul_1_readvariableop_resource:	m┤Q
Bsequential_11_lstm_23_lstm_cell_23_biasadd_readvariableop_resource:	┤J
8sequential_11_dense_11_tensordot_readvariableop_resource:mD
6sequential_11_dense_11_biasadd_readvariableop_resource:
identityИв-sequential_11/dense_11/BiasAdd/ReadVariableOpв/sequential_11/dense_11/Tensordot/ReadVariableOpв9sequential_11/lstm_22/lstm_cell_22/BiasAdd/ReadVariableOpв8sequential_11/lstm_22/lstm_cell_22/MatMul/ReadVariableOpв:sequential_11/lstm_22/lstm_cell_22/MatMul_1/ReadVariableOpвsequential_11/lstm_22/whileв9sequential_11/lstm_23/lstm_cell_23/BiasAdd/ReadVariableOpв8sequential_11/lstm_23/lstm_cell_23/MatMul/ReadVariableOpв:sequential_11/lstm_23/lstm_cell_23/MatMul_1/ReadVariableOpвsequential_11/lstm_23/whilew
sequential_11/lstm_22/ShapeShapelstm_22_input*
T0*
_output_shapes
:2
sequential_11/lstm_22/Shapeа
)sequential_11/lstm_22/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)sequential_11/lstm_22/strided_slice/stackд
+sequential_11/lstm_22/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential_11/lstm_22/strided_slice/stack_1д
+sequential_11/lstm_22/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential_11/lstm_22/strided_slice/stack_2ц
#sequential_11/lstm_22/strided_sliceStridedSlice$sequential_11/lstm_22/Shape:output:02sequential_11/lstm_22/strided_slice/stack:output:04sequential_11/lstm_22/strided_slice/stack_1:output:04sequential_11/lstm_22/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#sequential_11/lstm_22/strided_sliceЙ
!sequential_11/lstm_22/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :Д2#
!sequential_11/lstm_22/zeros/mul/y─
sequential_11/lstm_22/zeros/mulMul,sequential_11/lstm_22/strided_slice:output:0*sequential_11/lstm_22/zeros/mul/y:output:0*
T0*
_output_shapes
: 2!
sequential_11/lstm_22/zeros/mulЛ
"sequential_11/lstm_22/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2$
"sequential_11/lstm_22/zeros/Less/y┐
 sequential_11/lstm_22/zeros/LessLess#sequential_11/lstm_22/zeros/mul:z:0+sequential_11/lstm_22/zeros/Less/y:output:0*
T0*
_output_shapes
: 2"
 sequential_11/lstm_22/zeros/LessП
$sequential_11/lstm_22/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :Д2&
$sequential_11/lstm_22/zeros/packed/1█
"sequential_11/lstm_22/zeros/packedPack,sequential_11/lstm_22/strided_slice:output:0-sequential_11/lstm_22/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2$
"sequential_11/lstm_22/zeros/packedЛ
!sequential_11/lstm_22/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!sequential_11/lstm_22/zeros/Const╬
sequential_11/lstm_22/zerosFill+sequential_11/lstm_22/zeros/packed:output:0*sequential_11/lstm_22/zeros/Const:output:0*
T0*(
_output_shapes
:         Д2
sequential_11/lstm_22/zerosН
#sequential_11/lstm_22/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :Д2%
#sequential_11/lstm_22/zeros_1/mul/y╩
!sequential_11/lstm_22/zeros_1/mulMul,sequential_11/lstm_22/strided_slice:output:0,sequential_11/lstm_22/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2#
!sequential_11/lstm_22/zeros_1/mulП
$sequential_11/lstm_22/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2&
$sequential_11/lstm_22/zeros_1/Less/y╟
"sequential_11/lstm_22/zeros_1/LessLess%sequential_11/lstm_22/zeros_1/mul:z:0-sequential_11/lstm_22/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2$
"sequential_11/lstm_22/zeros_1/LessУ
&sequential_11/lstm_22/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :Д2(
&sequential_11/lstm_22/zeros_1/packed/1с
$sequential_11/lstm_22/zeros_1/packedPack,sequential_11/lstm_22/strided_slice:output:0/sequential_11/lstm_22/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2&
$sequential_11/lstm_22/zeros_1/packedП
#sequential_11/lstm_22/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2%
#sequential_11/lstm_22/zeros_1/Const╓
sequential_11/lstm_22/zeros_1Fill-sequential_11/lstm_22/zeros_1/packed:output:0,sequential_11/lstm_22/zeros_1/Const:output:0*
T0*(
_output_shapes
:         Д2
sequential_11/lstm_22/zeros_1б
$sequential_11/lstm_22/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2&
$sequential_11/lstm_22/transpose/perm├
sequential_11/lstm_22/transpose	Transposelstm_22_input-sequential_11/lstm_22/transpose/perm:output:0*
T0*+
_output_shapes
:         ]2!
sequential_11/lstm_22/transposeС
sequential_11/lstm_22/Shape_1Shape#sequential_11/lstm_22/transpose:y:0*
T0*
_output_shapes
:2
sequential_11/lstm_22/Shape_1д
+sequential_11/lstm_22/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2-
+sequential_11/lstm_22/strided_slice_1/stackи
-sequential_11/lstm_22/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_11/lstm_22/strided_slice_1/stack_1и
-sequential_11/lstm_22/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_11/lstm_22/strided_slice_1/stack_2Є
%sequential_11/lstm_22/strided_slice_1StridedSlice&sequential_11/lstm_22/Shape_1:output:04sequential_11/lstm_22/strided_slice_1/stack:output:06sequential_11/lstm_22/strided_slice_1/stack_1:output:06sequential_11/lstm_22/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%sequential_11/lstm_22/strided_slice_1▒
1sequential_11/lstm_22/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         23
1sequential_11/lstm_22/TensorArrayV2/element_shapeК
#sequential_11/lstm_22/TensorArrayV2TensorListReserve:sequential_11/lstm_22/TensorArrayV2/element_shape:output:0.sequential_11/lstm_22/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02%
#sequential_11/lstm_22/TensorArrayV2ы
Ksequential_11/lstm_22/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ]   2M
Ksequential_11/lstm_22/TensorArrayUnstack/TensorListFromTensor/element_shape╨
=sequential_11/lstm_22/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor#sequential_11/lstm_22/transpose:y:0Tsequential_11/lstm_22/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02?
=sequential_11/lstm_22/TensorArrayUnstack/TensorListFromTensorд
+sequential_11/lstm_22/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2-
+sequential_11/lstm_22/strided_slice_2/stackи
-sequential_11/lstm_22/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_11/lstm_22/strided_slice_2/stack_1и
-sequential_11/lstm_22/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_11/lstm_22/strided_slice_2/stack_2А
%sequential_11/lstm_22/strided_slice_2StridedSlice#sequential_11/lstm_22/transpose:y:04sequential_11/lstm_22/strided_slice_2/stack:output:06sequential_11/lstm_22/strided_slice_2/stack_1:output:06sequential_11/lstm_22/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         ]*
shrink_axis_mask2'
%sequential_11/lstm_22/strided_slice_2ў
8sequential_11/lstm_22/lstm_cell_22/MatMul/ReadVariableOpReadVariableOpAsequential_11_lstm_22_lstm_cell_22_matmul_readvariableop_resource*
_output_shapes
:	]Р*
dtype02:
8sequential_11/lstm_22/lstm_cell_22/MatMul/ReadVariableOpЕ
)sequential_11/lstm_22/lstm_cell_22/MatMulMatMul.sequential_11/lstm_22/strided_slice_2:output:0@sequential_11/lstm_22/lstm_cell_22/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Р2+
)sequential_11/lstm_22/lstm_cell_22/MatMul■
:sequential_11/lstm_22/lstm_cell_22/MatMul_1/ReadVariableOpReadVariableOpCsequential_11_lstm_22_lstm_cell_22_matmul_1_readvariableop_resource* 
_output_shapes
:
ДР*
dtype02<
:sequential_11/lstm_22/lstm_cell_22/MatMul_1/ReadVariableOpБ
+sequential_11/lstm_22/lstm_cell_22/MatMul_1MatMul$sequential_11/lstm_22/zeros:output:0Bsequential_11/lstm_22/lstm_cell_22/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Р2-
+sequential_11/lstm_22/lstm_cell_22/MatMul_1°
&sequential_11/lstm_22/lstm_cell_22/addAddV23sequential_11/lstm_22/lstm_cell_22/MatMul:product:05sequential_11/lstm_22/lstm_cell_22/MatMul_1:product:0*
T0*(
_output_shapes
:         Р2(
&sequential_11/lstm_22/lstm_cell_22/addЎ
9sequential_11/lstm_22/lstm_cell_22/BiasAdd/ReadVariableOpReadVariableOpBsequential_11_lstm_22_lstm_cell_22_biasadd_readvariableop_resource*
_output_shapes	
:Р*
dtype02;
9sequential_11/lstm_22/lstm_cell_22/BiasAdd/ReadVariableOpЕ
*sequential_11/lstm_22/lstm_cell_22/BiasAddBiasAdd*sequential_11/lstm_22/lstm_cell_22/add:z:0Asequential_11/lstm_22/lstm_cell_22/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Р2,
*sequential_11/lstm_22/lstm_cell_22/BiasAddк
2sequential_11/lstm_22/lstm_cell_22/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :24
2sequential_11/lstm_22/lstm_cell_22/split/split_dim╧
(sequential_11/lstm_22/lstm_cell_22/splitSplit;sequential_11/lstm_22/lstm_cell_22/split/split_dim:output:03sequential_11/lstm_22/lstm_cell_22/BiasAdd:output:0*
T0*d
_output_shapesR
P:         Д:         Д:         Д:         Д*
	num_split2*
(sequential_11/lstm_22/lstm_cell_22/split╔
*sequential_11/lstm_22/lstm_cell_22/SigmoidSigmoid1sequential_11/lstm_22/lstm_cell_22/split:output:0*
T0*(
_output_shapes
:         Д2,
*sequential_11/lstm_22/lstm_cell_22/Sigmoid═
,sequential_11/lstm_22/lstm_cell_22/Sigmoid_1Sigmoid1sequential_11/lstm_22/lstm_cell_22/split:output:1*
T0*(
_output_shapes
:         Д2.
,sequential_11/lstm_22/lstm_cell_22/Sigmoid_1ф
&sequential_11/lstm_22/lstm_cell_22/mulMul0sequential_11/lstm_22/lstm_cell_22/Sigmoid_1:y:0&sequential_11/lstm_22/zeros_1:output:0*
T0*(
_output_shapes
:         Д2(
&sequential_11/lstm_22/lstm_cell_22/mul└
'sequential_11/lstm_22/lstm_cell_22/ReluRelu1sequential_11/lstm_22/lstm_cell_22/split:output:2*
T0*(
_output_shapes
:         Д2)
'sequential_11/lstm_22/lstm_cell_22/Reluї
(sequential_11/lstm_22/lstm_cell_22/mul_1Mul.sequential_11/lstm_22/lstm_cell_22/Sigmoid:y:05sequential_11/lstm_22/lstm_cell_22/Relu:activations:0*
T0*(
_output_shapes
:         Д2*
(sequential_11/lstm_22/lstm_cell_22/mul_1ъ
(sequential_11/lstm_22/lstm_cell_22/add_1AddV2*sequential_11/lstm_22/lstm_cell_22/mul:z:0,sequential_11/lstm_22/lstm_cell_22/mul_1:z:0*
T0*(
_output_shapes
:         Д2*
(sequential_11/lstm_22/lstm_cell_22/add_1═
,sequential_11/lstm_22/lstm_cell_22/Sigmoid_2Sigmoid1sequential_11/lstm_22/lstm_cell_22/split:output:3*
T0*(
_output_shapes
:         Д2.
,sequential_11/lstm_22/lstm_cell_22/Sigmoid_2┐
)sequential_11/lstm_22/lstm_cell_22/Relu_1Relu,sequential_11/lstm_22/lstm_cell_22/add_1:z:0*
T0*(
_output_shapes
:         Д2+
)sequential_11/lstm_22/lstm_cell_22/Relu_1∙
(sequential_11/lstm_22/lstm_cell_22/mul_2Mul0sequential_11/lstm_22/lstm_cell_22/Sigmoid_2:y:07sequential_11/lstm_22/lstm_cell_22/Relu_1:activations:0*
T0*(
_output_shapes
:         Д2*
(sequential_11/lstm_22/lstm_cell_22/mul_2╗
3sequential_11/lstm_22/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    Д   25
3sequential_11/lstm_22/TensorArrayV2_1/element_shapeР
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
sequential_11/lstm_22/timeл
.sequential_11/lstm_22/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         20
.sequential_11/lstm_22/while/maximum_iterationsЦ
(sequential_11/lstm_22/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2*
(sequential_11/lstm_22/while/loop_counter▌
sequential_11/lstm_22/whileWhile1sequential_11/lstm_22/while/loop_counter:output:07sequential_11/lstm_22/while/maximum_iterations:output:0#sequential_11/lstm_22/time:output:0.sequential_11/lstm_22/TensorArrayV2_1:handle:0$sequential_11/lstm_22/zeros:output:0&sequential_11/lstm_22/zeros_1:output:0.sequential_11/lstm_22/strided_slice_1:output:0Msequential_11/lstm_22/TensorArrayUnstack/TensorListFromTensor:output_handle:0Asequential_11_lstm_22_lstm_cell_22_matmul_readvariableop_resourceCsequential_11_lstm_22_lstm_cell_22_matmul_1_readvariableop_resourceBsequential_11_lstm_22_lstm_cell_22_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :         Д:         Д: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *5
body-R+
)sequential_11_lstm_22_while_body_39107832*5
cond-R+
)sequential_11_lstm_22_while_cond_39107831*M
output_shapes<
:: : : : :         Д:         Д: : : : : *
parallel_iterations 2
sequential_11/lstm_22/whileс
Fsequential_11/lstm_22/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    Д   2H
Fsequential_11/lstm_22/TensorArrayV2Stack/TensorListStack/element_shape┴
8sequential_11/lstm_22/TensorArrayV2Stack/TensorListStackTensorListStack$sequential_11/lstm_22/while:output:3Osequential_11/lstm_22/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:         Д*
element_dtype02:
8sequential_11/lstm_22/TensorArrayV2Stack/TensorListStackн
+sequential_11/lstm_22/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         2-
+sequential_11/lstm_22/strided_slice_3/stackи
-sequential_11/lstm_22/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2/
-sequential_11/lstm_22/strided_slice_3/stack_1и
-sequential_11/lstm_22/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_11/lstm_22/strided_slice_3/stack_2Я
%sequential_11/lstm_22/strided_slice_3StridedSliceAsequential_11/lstm_22/TensorArrayV2Stack/TensorListStack:tensor:04sequential_11/lstm_22/strided_slice_3/stack:output:06sequential_11/lstm_22/strided_slice_3/stack_1:output:06sequential_11/lstm_22/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         Д*
shrink_axis_mask2'
%sequential_11/lstm_22/strided_slice_3е
&sequential_11/lstm_22/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2(
&sequential_11/lstm_22/transpose_1/perm■
!sequential_11/lstm_22/transpose_1	TransposeAsequential_11/lstm_22/TensorArrayV2Stack/TensorListStack:tensor:0/sequential_11/lstm_22/transpose_1/perm:output:0*
T0*,
_output_shapes
:         Д2#
!sequential_11/lstm_22/transpose_1Т
sequential_11/lstm_22/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
sequential_11/lstm_22/runtime░
!sequential_11/dropout_22/IdentityIdentity%sequential_11/lstm_22/transpose_1:y:0*
T0*,
_output_shapes
:         Д2#
!sequential_11/dropout_22/IdentityФ
sequential_11/lstm_23/ShapeShape*sequential_11/dropout_22/Identity:output:0*
T0*
_output_shapes
:2
sequential_11/lstm_23/Shapeа
)sequential_11/lstm_23/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)sequential_11/lstm_23/strided_slice/stackд
+sequential_11/lstm_23/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential_11/lstm_23/strided_slice/stack_1д
+sequential_11/lstm_23/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential_11/lstm_23/strided_slice/stack_2ц
#sequential_11/lstm_23/strided_sliceStridedSlice$sequential_11/lstm_23/Shape:output:02sequential_11/lstm_23/strided_slice/stack:output:04sequential_11/lstm_23/strided_slice/stack_1:output:04sequential_11/lstm_23/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#sequential_11/lstm_23/strided_sliceИ
!sequential_11/lstm_23/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :m2#
!sequential_11/lstm_23/zeros/mul/y─
sequential_11/lstm_23/zeros/mulMul,sequential_11/lstm_23/strided_slice:output:0*sequential_11/lstm_23/zeros/mul/y:output:0*
T0*
_output_shapes
: 2!
sequential_11/lstm_23/zeros/mulЛ
"sequential_11/lstm_23/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2$
"sequential_11/lstm_23/zeros/Less/y┐
 sequential_11/lstm_23/zeros/LessLess#sequential_11/lstm_23/zeros/mul:z:0+sequential_11/lstm_23/zeros/Less/y:output:0*
T0*
_output_shapes
: 2"
 sequential_11/lstm_23/zeros/LessО
$sequential_11/lstm_23/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :m2&
$sequential_11/lstm_23/zeros/packed/1█
"sequential_11/lstm_23/zeros/packedPack,sequential_11/lstm_23/strided_slice:output:0-sequential_11/lstm_23/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2$
"sequential_11/lstm_23/zeros/packedЛ
!sequential_11/lstm_23/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!sequential_11/lstm_23/zeros/Const═
sequential_11/lstm_23/zerosFill+sequential_11/lstm_23/zeros/packed:output:0*sequential_11/lstm_23/zeros/Const:output:0*
T0*'
_output_shapes
:         m2
sequential_11/lstm_23/zerosМ
#sequential_11/lstm_23/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :m2%
#sequential_11/lstm_23/zeros_1/mul/y╩
!sequential_11/lstm_23/zeros_1/mulMul,sequential_11/lstm_23/strided_slice:output:0,sequential_11/lstm_23/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2#
!sequential_11/lstm_23/zeros_1/mulП
$sequential_11/lstm_23/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2&
$sequential_11/lstm_23/zeros_1/Less/y╟
"sequential_11/lstm_23/zeros_1/LessLess%sequential_11/lstm_23/zeros_1/mul:z:0-sequential_11/lstm_23/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2$
"sequential_11/lstm_23/zeros_1/LessТ
&sequential_11/lstm_23/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :m2(
&sequential_11/lstm_23/zeros_1/packed/1с
$sequential_11/lstm_23/zeros_1/packedPack,sequential_11/lstm_23/strided_slice:output:0/sequential_11/lstm_23/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2&
$sequential_11/lstm_23/zeros_1/packedП
#sequential_11/lstm_23/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2%
#sequential_11/lstm_23/zeros_1/Const╒
sequential_11/lstm_23/zeros_1Fill-sequential_11/lstm_23/zeros_1/packed:output:0,sequential_11/lstm_23/zeros_1/Const:output:0*
T0*'
_output_shapes
:         m2
sequential_11/lstm_23/zeros_1б
$sequential_11/lstm_23/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2&
$sequential_11/lstm_23/transpose/permс
sequential_11/lstm_23/transpose	Transpose*sequential_11/dropout_22/Identity:output:0-sequential_11/lstm_23/transpose/perm:output:0*
T0*,
_output_shapes
:         Д2!
sequential_11/lstm_23/transposeС
sequential_11/lstm_23/Shape_1Shape#sequential_11/lstm_23/transpose:y:0*
T0*
_output_shapes
:2
sequential_11/lstm_23/Shape_1д
+sequential_11/lstm_23/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2-
+sequential_11/lstm_23/strided_slice_1/stackи
-sequential_11/lstm_23/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_11/lstm_23/strided_slice_1/stack_1и
-sequential_11/lstm_23/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_11/lstm_23/strided_slice_1/stack_2Є
%sequential_11/lstm_23/strided_slice_1StridedSlice&sequential_11/lstm_23/Shape_1:output:04sequential_11/lstm_23/strided_slice_1/stack:output:06sequential_11/lstm_23/strided_slice_1/stack_1:output:06sequential_11/lstm_23/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%sequential_11/lstm_23/strided_slice_1▒
1sequential_11/lstm_23/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         23
1sequential_11/lstm_23/TensorArrayV2/element_shapeК
#sequential_11/lstm_23/TensorArrayV2TensorListReserve:sequential_11/lstm_23/TensorArrayV2/element_shape:output:0.sequential_11/lstm_23/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02%
#sequential_11/lstm_23/TensorArrayV2ы
Ksequential_11/lstm_23/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    Д   2M
Ksequential_11/lstm_23/TensorArrayUnstack/TensorListFromTensor/element_shape╨
=sequential_11/lstm_23/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor#sequential_11/lstm_23/transpose:y:0Tsequential_11/lstm_23/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02?
=sequential_11/lstm_23/TensorArrayUnstack/TensorListFromTensorд
+sequential_11/lstm_23/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2-
+sequential_11/lstm_23/strided_slice_2/stackи
-sequential_11/lstm_23/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_11/lstm_23/strided_slice_2/stack_1и
-sequential_11/lstm_23/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_11/lstm_23/strided_slice_2/stack_2Б
%sequential_11/lstm_23/strided_slice_2StridedSlice#sequential_11/lstm_23/transpose:y:04sequential_11/lstm_23/strided_slice_2/stack:output:06sequential_11/lstm_23/strided_slice_2/stack_1:output:06sequential_11/lstm_23/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         Д*
shrink_axis_mask2'
%sequential_11/lstm_23/strided_slice_2°
8sequential_11/lstm_23/lstm_cell_23/MatMul/ReadVariableOpReadVariableOpAsequential_11_lstm_23_lstm_cell_23_matmul_readvariableop_resource* 
_output_shapes
:
Д┤*
dtype02:
8sequential_11/lstm_23/lstm_cell_23/MatMul/ReadVariableOpЕ
)sequential_11/lstm_23/lstm_cell_23/MatMulMatMul.sequential_11/lstm_23/strided_slice_2:output:0@sequential_11/lstm_23/lstm_cell_23/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ┤2+
)sequential_11/lstm_23/lstm_cell_23/MatMul¤
:sequential_11/lstm_23/lstm_cell_23/MatMul_1/ReadVariableOpReadVariableOpCsequential_11_lstm_23_lstm_cell_23_matmul_1_readvariableop_resource*
_output_shapes
:	m┤*
dtype02<
:sequential_11/lstm_23/lstm_cell_23/MatMul_1/ReadVariableOpБ
+sequential_11/lstm_23/lstm_cell_23/MatMul_1MatMul$sequential_11/lstm_23/zeros:output:0Bsequential_11/lstm_23/lstm_cell_23/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ┤2-
+sequential_11/lstm_23/lstm_cell_23/MatMul_1°
&sequential_11/lstm_23/lstm_cell_23/addAddV23sequential_11/lstm_23/lstm_cell_23/MatMul:product:05sequential_11/lstm_23/lstm_cell_23/MatMul_1:product:0*
T0*(
_output_shapes
:         ┤2(
&sequential_11/lstm_23/lstm_cell_23/addЎ
9sequential_11/lstm_23/lstm_cell_23/BiasAdd/ReadVariableOpReadVariableOpBsequential_11_lstm_23_lstm_cell_23_biasadd_readvariableop_resource*
_output_shapes	
:┤*
dtype02;
9sequential_11/lstm_23/lstm_cell_23/BiasAdd/ReadVariableOpЕ
*sequential_11/lstm_23/lstm_cell_23/BiasAddBiasAdd*sequential_11/lstm_23/lstm_cell_23/add:z:0Asequential_11/lstm_23/lstm_cell_23/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ┤2,
*sequential_11/lstm_23/lstm_cell_23/BiasAddк
2sequential_11/lstm_23/lstm_cell_23/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :24
2sequential_11/lstm_23/lstm_cell_23/split/split_dim╦
(sequential_11/lstm_23/lstm_cell_23/splitSplit;sequential_11/lstm_23/lstm_cell_23/split/split_dim:output:03sequential_11/lstm_23/lstm_cell_23/BiasAdd:output:0*
T0*`
_output_shapesN
L:         m:         m:         m:         m*
	num_split2*
(sequential_11/lstm_23/lstm_cell_23/split╚
*sequential_11/lstm_23/lstm_cell_23/SigmoidSigmoid1sequential_11/lstm_23/lstm_cell_23/split:output:0*
T0*'
_output_shapes
:         m2,
*sequential_11/lstm_23/lstm_cell_23/Sigmoid╠
,sequential_11/lstm_23/lstm_cell_23/Sigmoid_1Sigmoid1sequential_11/lstm_23/lstm_cell_23/split:output:1*
T0*'
_output_shapes
:         m2.
,sequential_11/lstm_23/lstm_cell_23/Sigmoid_1у
&sequential_11/lstm_23/lstm_cell_23/mulMul0sequential_11/lstm_23/lstm_cell_23/Sigmoid_1:y:0&sequential_11/lstm_23/zeros_1:output:0*
T0*'
_output_shapes
:         m2(
&sequential_11/lstm_23/lstm_cell_23/mul┐
'sequential_11/lstm_23/lstm_cell_23/ReluRelu1sequential_11/lstm_23/lstm_cell_23/split:output:2*
T0*'
_output_shapes
:         m2)
'sequential_11/lstm_23/lstm_cell_23/ReluЇ
(sequential_11/lstm_23/lstm_cell_23/mul_1Mul.sequential_11/lstm_23/lstm_cell_23/Sigmoid:y:05sequential_11/lstm_23/lstm_cell_23/Relu:activations:0*
T0*'
_output_shapes
:         m2*
(sequential_11/lstm_23/lstm_cell_23/mul_1щ
(sequential_11/lstm_23/lstm_cell_23/add_1AddV2*sequential_11/lstm_23/lstm_cell_23/mul:z:0,sequential_11/lstm_23/lstm_cell_23/mul_1:z:0*
T0*'
_output_shapes
:         m2*
(sequential_11/lstm_23/lstm_cell_23/add_1╠
,sequential_11/lstm_23/lstm_cell_23/Sigmoid_2Sigmoid1sequential_11/lstm_23/lstm_cell_23/split:output:3*
T0*'
_output_shapes
:         m2.
,sequential_11/lstm_23/lstm_cell_23/Sigmoid_2╛
)sequential_11/lstm_23/lstm_cell_23/Relu_1Relu,sequential_11/lstm_23/lstm_cell_23/add_1:z:0*
T0*'
_output_shapes
:         m2+
)sequential_11/lstm_23/lstm_cell_23/Relu_1°
(sequential_11/lstm_23/lstm_cell_23/mul_2Mul0sequential_11/lstm_23/lstm_cell_23/Sigmoid_2:y:07sequential_11/lstm_23/lstm_cell_23/Relu_1:activations:0*
T0*'
_output_shapes
:         m2*
(sequential_11/lstm_23/lstm_cell_23/mul_2╗
3sequential_11/lstm_23/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    m   25
3sequential_11/lstm_23/TensorArrayV2_1/element_shapeР
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
sequential_11/lstm_23/timeл
.sequential_11/lstm_23/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         20
.sequential_11/lstm_23/while/maximum_iterationsЦ
(sequential_11/lstm_23/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2*
(sequential_11/lstm_23/while/loop_counter┘
sequential_11/lstm_23/whileWhile1sequential_11/lstm_23/while/loop_counter:output:07sequential_11/lstm_23/while/maximum_iterations:output:0#sequential_11/lstm_23/time:output:0.sequential_11/lstm_23/TensorArrayV2_1:handle:0$sequential_11/lstm_23/zeros:output:0&sequential_11/lstm_23/zeros_1:output:0.sequential_11/lstm_23/strided_slice_1:output:0Msequential_11/lstm_23/TensorArrayUnstack/TensorListFromTensor:output_handle:0Asequential_11_lstm_23_lstm_cell_23_matmul_readvariableop_resourceCsequential_11_lstm_23_lstm_cell_23_matmul_1_readvariableop_resourceBsequential_11_lstm_23_lstm_cell_23_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         m:         m: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *5
body-R+
)sequential_11_lstm_23_while_body_39107980*5
cond-R+
)sequential_11_lstm_23_while_cond_39107979*K
output_shapes:
8: : : : :         m:         m: : : : : *
parallel_iterations 2
sequential_11/lstm_23/whileс
Fsequential_11/lstm_23/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    m   2H
Fsequential_11/lstm_23/TensorArrayV2Stack/TensorListStack/element_shape└
8sequential_11/lstm_23/TensorArrayV2Stack/TensorListStackTensorListStack$sequential_11/lstm_23/while:output:3Osequential_11/lstm_23/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:         m*
element_dtype02:
8sequential_11/lstm_23/TensorArrayV2Stack/TensorListStackн
+sequential_11/lstm_23/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         2-
+sequential_11/lstm_23/strided_slice_3/stackи
-sequential_11/lstm_23/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2/
-sequential_11/lstm_23/strided_slice_3/stack_1и
-sequential_11/lstm_23/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_11/lstm_23/strided_slice_3/stack_2Ю
%sequential_11/lstm_23/strided_slice_3StridedSliceAsequential_11/lstm_23/TensorArrayV2Stack/TensorListStack:tensor:04sequential_11/lstm_23/strided_slice_3/stack:output:06sequential_11/lstm_23/strided_slice_3/stack_1:output:06sequential_11/lstm_23/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         m*
shrink_axis_mask2'
%sequential_11/lstm_23/strided_slice_3е
&sequential_11/lstm_23/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2(
&sequential_11/lstm_23/transpose_1/perm¤
!sequential_11/lstm_23/transpose_1	TransposeAsequential_11/lstm_23/TensorArrayV2Stack/TensorListStack:tensor:0/sequential_11/lstm_23/transpose_1/perm:output:0*
T0*+
_output_shapes
:         m2#
!sequential_11/lstm_23/transpose_1Т
sequential_11/lstm_23/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
sequential_11/lstm_23/runtimeп
!sequential_11/dropout_23/IdentityIdentity%sequential_11/lstm_23/transpose_1:y:0*
T0*+
_output_shapes
:         m2#
!sequential_11/dropout_23/Identity█
/sequential_11/dense_11/Tensordot/ReadVariableOpReadVariableOp8sequential_11_dense_11_tensordot_readvariableop_resource*
_output_shapes

:m*
dtype021
/sequential_11/dense_11/Tensordot/ReadVariableOpШ
%sequential_11/dense_11/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2'
%sequential_11/dense_11/Tensordot/axesЯ
%sequential_11/dense_11/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2'
%sequential_11/dense_11/Tensordot/freeк
&sequential_11/dense_11/Tensordot/ShapeShape*sequential_11/dropout_23/Identity:output:0*
T0*
_output_shapes
:2(
&sequential_11/dense_11/Tensordot/Shapeв
.sequential_11/dense_11/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.sequential_11/dense_11/Tensordot/GatherV2/axis─
)sequential_11/dense_11/Tensordot/GatherV2GatherV2/sequential_11/dense_11/Tensordot/Shape:output:0.sequential_11/dense_11/Tensordot/free:output:07sequential_11/dense_11/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2+
)sequential_11/dense_11/Tensordot/GatherV2ж
0sequential_11/dense_11/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 22
0sequential_11/dense_11/Tensordot/GatherV2_1/axis╩
+sequential_11/dense_11/Tensordot/GatherV2_1GatherV2/sequential_11/dense_11/Tensordot/Shape:output:0.sequential_11/dense_11/Tensordot/axes:output:09sequential_11/dense_11/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2-
+sequential_11/dense_11/Tensordot/GatherV2_1Ъ
&sequential_11/dense_11/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2(
&sequential_11/dense_11/Tensordot/Const▄
%sequential_11/dense_11/Tensordot/ProdProd2sequential_11/dense_11/Tensordot/GatherV2:output:0/sequential_11/dense_11/Tensordot/Const:output:0*
T0*
_output_shapes
: 2'
%sequential_11/dense_11/Tensordot/ProdЮ
(sequential_11/dense_11/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(sequential_11/dense_11/Tensordot/Const_1ф
'sequential_11/dense_11/Tensordot/Prod_1Prod4sequential_11/dense_11/Tensordot/GatherV2_1:output:01sequential_11/dense_11/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2)
'sequential_11/dense_11/Tensordot/Prod_1Ю
,sequential_11/dense_11/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,sequential_11/dense_11/Tensordot/concat/axisг
'sequential_11/dense_11/Tensordot/concatConcatV2.sequential_11/dense_11/Tensordot/free:output:0.sequential_11/dense_11/Tensordot/axes:output:05sequential_11/dense_11/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2)
'sequential_11/dense_11/Tensordot/concatш
&sequential_11/dense_11/Tensordot/stackPack.sequential_11/dense_11/Tensordot/Prod:output:00sequential_11/dense_11/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2(
&sequential_11/dense_11/Tensordot/stack∙
*sequential_11/dense_11/Tensordot/transpose	Transpose*sequential_11/dropout_23/Identity:output:00sequential_11/dense_11/Tensordot/concat:output:0*
T0*+
_output_shapes
:         m2,
*sequential_11/dense_11/Tensordot/transpose√
(sequential_11/dense_11/Tensordot/ReshapeReshape.sequential_11/dense_11/Tensordot/transpose:y:0/sequential_11/dense_11/Tensordot/stack:output:0*
T0*0
_output_shapes
:                  2*
(sequential_11/dense_11/Tensordot/Reshape·
'sequential_11/dense_11/Tensordot/MatMulMatMul1sequential_11/dense_11/Tensordot/Reshape:output:07sequential_11/dense_11/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2)
'sequential_11/dense_11/Tensordot/MatMulЮ
(sequential_11/dense_11/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(sequential_11/dense_11/Tensordot/Const_2в
.sequential_11/dense_11/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.sequential_11/dense_11/Tensordot/concat_1/axis░
)sequential_11/dense_11/Tensordot/concat_1ConcatV22sequential_11/dense_11/Tensordot/GatherV2:output:01sequential_11/dense_11/Tensordot/Const_2:output:07sequential_11/dense_11/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2+
)sequential_11/dense_11/Tensordot/concat_1ь
 sequential_11/dense_11/TensordotReshape1sequential_11/dense_11/Tensordot/MatMul:product:02sequential_11/dense_11/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:         2"
 sequential_11/dense_11/Tensordot╤
-sequential_11/dense_11/BiasAdd/ReadVariableOpReadVariableOp6sequential_11_dense_11_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02/
-sequential_11/dense_11/BiasAdd/ReadVariableOpу
sequential_11/dense_11/BiasAddBiasAdd)sequential_11/dense_11/Tensordot:output:05sequential_11/dense_11/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:         2 
sequential_11/dense_11/BiasAddк
sequential_11/dense_11/SoftmaxSoftmax'sequential_11/dense_11/BiasAdd:output:0*
T0*+
_output_shapes
:         2 
sequential_11/dense_11/SoftmaxЗ
IdentityIdentity(sequential_11/dense_11/Softmax:softmax:0^NoOp*
T0*+
_output_shapes
:         2

Identity╘
NoOpNoOp.^sequential_11/dense_11/BiasAdd/ReadVariableOp0^sequential_11/dense_11/Tensordot/ReadVariableOp:^sequential_11/lstm_22/lstm_cell_22/BiasAdd/ReadVariableOp9^sequential_11/lstm_22/lstm_cell_22/MatMul/ReadVariableOp;^sequential_11/lstm_22/lstm_cell_22/MatMul_1/ReadVariableOp^sequential_11/lstm_22/while:^sequential_11/lstm_23/lstm_cell_23/BiasAdd/ReadVariableOp9^sequential_11/lstm_23/lstm_cell_23/MatMul/ReadVariableOp;^sequential_11/lstm_23/lstm_cell_23/MatMul_1/ReadVariableOp^sequential_11/lstm_23/while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':         ]: : : : : : : : 2^
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
:         ]
'
_user_specified_namelstm_22_input
╫
g
H__inference_dropout_22_layer_call_and_return_conditional_losses_39109972

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
:         Д2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape╣
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:         Д*
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
:         Д2
dropout/GreaterEqualД
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:         Д2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:         Д2
dropout/Mul_1j
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:         Д2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         Д:T P
,
_output_shapes
:         Д
 
_user_specified_nameinputs
∙
З
J__inference_lstm_cell_23_layer_call_and_return_conditional_losses_39108797

inputs

states
states_12
matmul_readvariableop_resource:
Д┤3
 matmul_1_readvariableop_resource:	m┤.
biasadd_readvariableop_resource:	┤
identity

identity_1

identity_2ИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpвMatMul_1/ReadVariableOpП
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
Д┤*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ┤2
MatMulФ
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	m┤*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ┤2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:         ┤2
addН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:┤*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ┤2	
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
L:         m:         m:         m:         m*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:         m2	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:         m2
	Sigmoid_1\
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:         m2
mulV
ReluRelusplit:output:2*
T0*'
_output_shapes
:         m2
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:         m2
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:         m2
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:         m2
	Sigmoid_2U
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:         m2
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:         m2
mul_2d
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:         m2

Identityh

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:         m2

Identity_1h

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:         m2

Identity_2Щ
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:         Д:         m:         m: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:P L
(
_output_shapes
:         Д
 
_user_specified_nameinputs:OK
'
_output_shapes
:         m
 
_user_specified_namestates:OK
'
_output_shapes
:         m
 
_user_specified_namestates
Е
f
H__inference_dropout_23_layer_call_and_return_conditional_losses_39112347

inputs

identity_1^
IdentityIdentityinputs*
T0*+
_output_shapes
:         m2

Identitym

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:         m2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         m:S O
+
_output_shapes
:         m
 
_user_specified_nameinputs
╘

э
lstm_22_while_cond_39110702,
(lstm_22_while_lstm_22_while_loop_counter2
.lstm_22_while_lstm_22_while_maximum_iterations
lstm_22_while_placeholder
lstm_22_while_placeholder_1
lstm_22_while_placeholder_2
lstm_22_while_placeholder_3.
*lstm_22_while_less_lstm_22_strided_slice_1F
Blstm_22_while_lstm_22_while_cond_39110702___redundant_placeholder0F
Blstm_22_while_lstm_22_while_cond_39110702___redundant_placeholder1F
Blstm_22_while_lstm_22_while_cond_39110702___redundant_placeholder2F
Blstm_22_while_lstm_22_while_cond_39110702___redundant_placeholder3
lstm_22_while_identity
Ш
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
B: : : : :         Д:         Д: ::::: 
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
:         Д:.*
(
_output_shapes
:         Д:

_output_shapes
: :

_output_shapes
:
╟
∙
/__inference_lstm_cell_22_layer_call_fn_39112490

inputs
states_0
states_1
unknown:	]Р
	unknown_0:
ДР
	unknown_1:	Р
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
<:         Д:         Д:         Д*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_lstm_cell_22_layer_call_and_return_conditional_losses_391081672
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         Д2

IdentityА

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:         Д2

Identity_1А

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:         Д2

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
A:         ]:         Д:         Д: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         ]
 
_user_specified_nameinputs:RN
(
_output_shapes
:         Д
"
_user_specified_name
states/0:RN
(
_output_shapes
:         Д
"
_user_specified_name
states/1
р
║
*__inference_lstm_23_layer_call_fn_39112320
inputs_0
unknown:
Д┤
	unknown_0:	m┤
	unknown_1:	┤
identityИвStatefulPartitionedCallС
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :                  m*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_lstm_23_layer_call_and_return_conditional_losses_391090902
StatefulPartitionedCallИ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :                  m2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':                  Д: : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
5
_output_shapes#
!:                  Д
"
_user_specified_name
inputs/0
┤∙
К
K__inference_sequential_11_layer_call_and_return_conditional_losses_39110636

inputsF
3lstm_22_lstm_cell_22_matmul_readvariableop_resource:	]РI
5lstm_22_lstm_cell_22_matmul_1_readvariableop_resource:
ДРC
4lstm_22_lstm_cell_22_biasadd_readvariableop_resource:	РG
3lstm_23_lstm_cell_23_matmul_readvariableop_resource:
Д┤H
5lstm_23_lstm_cell_23_matmul_1_readvariableop_resource:	m┤C
4lstm_23_lstm_cell_23_biasadd_readvariableop_resource:	┤<
*dense_11_tensordot_readvariableop_resource:m6
(dense_11_biasadd_readvariableop_resource:
identityИвdense_11/BiasAdd/ReadVariableOpв!dense_11/Tensordot/ReadVariableOpв+lstm_22/lstm_cell_22/BiasAdd/ReadVariableOpв*lstm_22/lstm_cell_22/MatMul/ReadVariableOpв,lstm_22/lstm_cell_22/MatMul_1/ReadVariableOpвlstm_22/whileв+lstm_23/lstm_cell_23/BiasAdd/ReadVariableOpв*lstm_23/lstm_cell_23/MatMul/ReadVariableOpв,lstm_23/lstm_cell_23/MatMul_1/ReadVariableOpвlstm_23/whileT
lstm_22/ShapeShapeinputs*
T0*
_output_shapes
:2
lstm_22/ShapeД
lstm_22/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_22/strided_slice/stackИ
lstm_22/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_22/strided_slice/stack_1И
lstm_22/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_22/strided_slice/stack_2Т
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
B :Д2
lstm_22/zeros/mul/yМ
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
B :ш2
lstm_22/zeros/Less/yЗ
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
B :Д2
lstm_22/zeros/packed/1г
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
lstm_22/zeros/ConstЦ
lstm_22/zerosFilllstm_22/zeros/packed:output:0lstm_22/zeros/Const:output:0*
T0*(
_output_shapes
:         Д2
lstm_22/zerosq
lstm_22/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :Д2
lstm_22/zeros_1/mul/yТ
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
B :ш2
lstm_22/zeros_1/Less/yП
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
B :Д2
lstm_22/zeros_1/packed/1й
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
lstm_22/zeros_1/ConstЮ
lstm_22/zeros_1Filllstm_22/zeros_1/packed:output:0lstm_22/zeros_1/Const:output:0*
T0*(
_output_shapes
:         Д2
lstm_22/zeros_1Е
lstm_22/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_22/transpose/permТ
lstm_22/transpose	Transposeinputslstm_22/transpose/perm:output:0*
T0*+
_output_shapes
:         ]2
lstm_22/transposeg
lstm_22/Shape_1Shapelstm_22/transpose:y:0*
T0*
_output_shapes
:2
lstm_22/Shape_1И
lstm_22/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_22/strided_slice_1/stackМ
lstm_22/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_22/strided_slice_1/stack_1М
lstm_22/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_22/strided_slice_1/stack_2Ю
lstm_22/strided_slice_1StridedSlicelstm_22/Shape_1:output:0&lstm_22/strided_slice_1/stack:output:0(lstm_22/strided_slice_1/stack_1:output:0(lstm_22/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_22/strided_slice_1Х
#lstm_22/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         2%
#lstm_22/TensorArrayV2/element_shape╥
lstm_22/TensorArrayV2TensorListReserve,lstm_22/TensorArrayV2/element_shape:output:0 lstm_22/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_22/TensorArrayV2╧
=lstm_22/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ]   2?
=lstm_22/TensorArrayUnstack/TensorListFromTensor/element_shapeШ
/lstm_22/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_22/transpose:y:0Flstm_22/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type021
/lstm_22/TensorArrayUnstack/TensorListFromTensorИ
lstm_22/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_22/strided_slice_2/stackМ
lstm_22/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_22/strided_slice_2/stack_1М
lstm_22/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_22/strided_slice_2/stack_2м
lstm_22/strided_slice_2StridedSlicelstm_22/transpose:y:0&lstm_22/strided_slice_2/stack:output:0(lstm_22/strided_slice_2/stack_1:output:0(lstm_22/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         ]*
shrink_axis_mask2
lstm_22/strided_slice_2═
*lstm_22/lstm_cell_22/MatMul/ReadVariableOpReadVariableOp3lstm_22_lstm_cell_22_matmul_readvariableop_resource*
_output_shapes
:	]Р*
dtype02,
*lstm_22/lstm_cell_22/MatMul/ReadVariableOp═
lstm_22/lstm_cell_22/MatMulMatMul lstm_22/strided_slice_2:output:02lstm_22/lstm_cell_22/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Р2
lstm_22/lstm_cell_22/MatMul╘
,lstm_22/lstm_cell_22/MatMul_1/ReadVariableOpReadVariableOp5lstm_22_lstm_cell_22_matmul_1_readvariableop_resource* 
_output_shapes
:
ДР*
dtype02.
,lstm_22/lstm_cell_22/MatMul_1/ReadVariableOp╔
lstm_22/lstm_cell_22/MatMul_1MatMullstm_22/zeros:output:04lstm_22/lstm_cell_22/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Р2
lstm_22/lstm_cell_22/MatMul_1└
lstm_22/lstm_cell_22/addAddV2%lstm_22/lstm_cell_22/MatMul:product:0'lstm_22/lstm_cell_22/MatMul_1:product:0*
T0*(
_output_shapes
:         Р2
lstm_22/lstm_cell_22/add╠
+lstm_22/lstm_cell_22/BiasAdd/ReadVariableOpReadVariableOp4lstm_22_lstm_cell_22_biasadd_readvariableop_resource*
_output_shapes	
:Р*
dtype02-
+lstm_22/lstm_cell_22/BiasAdd/ReadVariableOp═
lstm_22/lstm_cell_22/BiasAddBiasAddlstm_22/lstm_cell_22/add:z:03lstm_22/lstm_cell_22/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Р2
lstm_22/lstm_cell_22/BiasAddО
$lstm_22/lstm_cell_22/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2&
$lstm_22/lstm_cell_22/split/split_dimЧ
lstm_22/lstm_cell_22/splitSplit-lstm_22/lstm_cell_22/split/split_dim:output:0%lstm_22/lstm_cell_22/BiasAdd:output:0*
T0*d
_output_shapesR
P:         Д:         Д:         Д:         Д*
	num_split2
lstm_22/lstm_cell_22/splitЯ
lstm_22/lstm_cell_22/SigmoidSigmoid#lstm_22/lstm_cell_22/split:output:0*
T0*(
_output_shapes
:         Д2
lstm_22/lstm_cell_22/Sigmoidг
lstm_22/lstm_cell_22/Sigmoid_1Sigmoid#lstm_22/lstm_cell_22/split:output:1*
T0*(
_output_shapes
:         Д2 
lstm_22/lstm_cell_22/Sigmoid_1м
lstm_22/lstm_cell_22/mulMul"lstm_22/lstm_cell_22/Sigmoid_1:y:0lstm_22/zeros_1:output:0*
T0*(
_output_shapes
:         Д2
lstm_22/lstm_cell_22/mulЦ
lstm_22/lstm_cell_22/ReluRelu#lstm_22/lstm_cell_22/split:output:2*
T0*(
_output_shapes
:         Д2
lstm_22/lstm_cell_22/Relu╜
lstm_22/lstm_cell_22/mul_1Mul lstm_22/lstm_cell_22/Sigmoid:y:0'lstm_22/lstm_cell_22/Relu:activations:0*
T0*(
_output_shapes
:         Д2
lstm_22/lstm_cell_22/mul_1▓
lstm_22/lstm_cell_22/add_1AddV2lstm_22/lstm_cell_22/mul:z:0lstm_22/lstm_cell_22/mul_1:z:0*
T0*(
_output_shapes
:         Д2
lstm_22/lstm_cell_22/add_1г
lstm_22/lstm_cell_22/Sigmoid_2Sigmoid#lstm_22/lstm_cell_22/split:output:3*
T0*(
_output_shapes
:         Д2 
lstm_22/lstm_cell_22/Sigmoid_2Х
lstm_22/lstm_cell_22/Relu_1Relulstm_22/lstm_cell_22/add_1:z:0*
T0*(
_output_shapes
:         Д2
lstm_22/lstm_cell_22/Relu_1┴
lstm_22/lstm_cell_22/mul_2Mul"lstm_22/lstm_cell_22/Sigmoid_2:y:0)lstm_22/lstm_cell_22/Relu_1:activations:0*
T0*(
_output_shapes
:         Д2
lstm_22/lstm_cell_22/mul_2Я
%lstm_22/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    Д   2'
%lstm_22/TensorArrayV2_1/element_shape╪
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
lstm_22/timeП
 lstm_22/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         2"
 lstm_22/while/maximum_iterationsz
lstm_22/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_22/while/loop_counterЛ
lstm_22/whileWhile#lstm_22/while/loop_counter:output:0)lstm_22/while/maximum_iterations:output:0lstm_22/time:output:0 lstm_22/TensorArrayV2_1:handle:0lstm_22/zeros:output:0lstm_22/zeros_1:output:0 lstm_22/strided_slice_1:output:0?lstm_22/TensorArrayUnstack/TensorListFromTensor:output_handle:03lstm_22_lstm_cell_22_matmul_readvariableop_resource5lstm_22_lstm_cell_22_matmul_1_readvariableop_resource4lstm_22_lstm_cell_22_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :         Д:         Д: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *'
bodyR
lstm_22_while_body_39110376*'
condR
lstm_22_while_cond_39110375*M
output_shapes<
:: : : : :         Д:         Д: : : : : *
parallel_iterations 2
lstm_22/while┼
8lstm_22/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    Д   2:
8lstm_22/TensorArrayV2Stack/TensorListStack/element_shapeЙ
*lstm_22/TensorArrayV2Stack/TensorListStackTensorListStacklstm_22/while:output:3Alstm_22/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:         Д*
element_dtype02,
*lstm_22/TensorArrayV2Stack/TensorListStackС
lstm_22/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         2
lstm_22/strided_slice_3/stackМ
lstm_22/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2!
lstm_22/strided_slice_3/stack_1М
lstm_22/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_22/strided_slice_3/stack_2╦
lstm_22/strided_slice_3StridedSlice3lstm_22/TensorArrayV2Stack/TensorListStack:tensor:0&lstm_22/strided_slice_3/stack:output:0(lstm_22/strided_slice_3/stack_1:output:0(lstm_22/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         Д*
shrink_axis_mask2
lstm_22/strided_slice_3Й
lstm_22/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_22/transpose_1/perm╞
lstm_22/transpose_1	Transpose3lstm_22/TensorArrayV2Stack/TensorListStack:tensor:0!lstm_22/transpose_1/perm:output:0*
T0*,
_output_shapes
:         Д2
lstm_22/transpose_1v
lstm_22/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_22/runtimeЖ
dropout_22/IdentityIdentitylstm_22/transpose_1:y:0*
T0*,
_output_shapes
:         Д2
dropout_22/Identityj
lstm_23/ShapeShapedropout_22/Identity:output:0*
T0*
_output_shapes
:2
lstm_23/ShapeД
lstm_23/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_23/strided_slice/stackИ
lstm_23/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_23/strided_slice/stack_1И
lstm_23/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_23/strided_slice/stack_2Т
lstm_23/strided_sliceStridedSlicelstm_23/Shape:output:0$lstm_23/strided_slice/stack:output:0&lstm_23/strided_slice/stack_1:output:0&lstm_23/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_23/strided_slicel
lstm_23/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :m2
lstm_23/zeros/mul/yМ
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
B :ш2
lstm_23/zeros/Less/yЗ
lstm_23/zeros/LessLesslstm_23/zeros/mul:z:0lstm_23/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_23/zeros/Lessr
lstm_23/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :m2
lstm_23/zeros/packed/1г
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
lstm_23/zeros/ConstХ
lstm_23/zerosFilllstm_23/zeros/packed:output:0lstm_23/zeros/Const:output:0*
T0*'
_output_shapes
:         m2
lstm_23/zerosp
lstm_23/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :m2
lstm_23/zeros_1/mul/yТ
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
B :ш2
lstm_23/zeros_1/Less/yП
lstm_23/zeros_1/LessLesslstm_23/zeros_1/mul:z:0lstm_23/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_23/zeros_1/Lessv
lstm_23/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :m2
lstm_23/zeros_1/packed/1й
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
lstm_23/zeros_1/ConstЭ
lstm_23/zeros_1Filllstm_23/zeros_1/packed:output:0lstm_23/zeros_1/Const:output:0*
T0*'
_output_shapes
:         m2
lstm_23/zeros_1Е
lstm_23/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_23/transpose/permй
lstm_23/transpose	Transposedropout_22/Identity:output:0lstm_23/transpose/perm:output:0*
T0*,
_output_shapes
:         Д2
lstm_23/transposeg
lstm_23/Shape_1Shapelstm_23/transpose:y:0*
T0*
_output_shapes
:2
lstm_23/Shape_1И
lstm_23/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_23/strided_slice_1/stackМ
lstm_23/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_23/strided_slice_1/stack_1М
lstm_23/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_23/strided_slice_1/stack_2Ю
lstm_23/strided_slice_1StridedSlicelstm_23/Shape_1:output:0&lstm_23/strided_slice_1/stack:output:0(lstm_23/strided_slice_1/stack_1:output:0(lstm_23/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_23/strided_slice_1Х
#lstm_23/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         2%
#lstm_23/TensorArrayV2/element_shape╥
lstm_23/TensorArrayV2TensorListReserve,lstm_23/TensorArrayV2/element_shape:output:0 lstm_23/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_23/TensorArrayV2╧
=lstm_23/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    Д   2?
=lstm_23/TensorArrayUnstack/TensorListFromTensor/element_shapeШ
/lstm_23/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_23/transpose:y:0Flstm_23/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type021
/lstm_23/TensorArrayUnstack/TensorListFromTensorИ
lstm_23/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_23/strided_slice_2/stackМ
lstm_23/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_23/strided_slice_2/stack_1М
lstm_23/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_23/strided_slice_2/stack_2н
lstm_23/strided_slice_2StridedSlicelstm_23/transpose:y:0&lstm_23/strided_slice_2/stack:output:0(lstm_23/strided_slice_2/stack_1:output:0(lstm_23/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         Д*
shrink_axis_mask2
lstm_23/strided_slice_2╬
*lstm_23/lstm_cell_23/MatMul/ReadVariableOpReadVariableOp3lstm_23_lstm_cell_23_matmul_readvariableop_resource* 
_output_shapes
:
Д┤*
dtype02,
*lstm_23/lstm_cell_23/MatMul/ReadVariableOp═
lstm_23/lstm_cell_23/MatMulMatMul lstm_23/strided_slice_2:output:02lstm_23/lstm_cell_23/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ┤2
lstm_23/lstm_cell_23/MatMul╙
,lstm_23/lstm_cell_23/MatMul_1/ReadVariableOpReadVariableOp5lstm_23_lstm_cell_23_matmul_1_readvariableop_resource*
_output_shapes
:	m┤*
dtype02.
,lstm_23/lstm_cell_23/MatMul_1/ReadVariableOp╔
lstm_23/lstm_cell_23/MatMul_1MatMullstm_23/zeros:output:04lstm_23/lstm_cell_23/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ┤2
lstm_23/lstm_cell_23/MatMul_1└
lstm_23/lstm_cell_23/addAddV2%lstm_23/lstm_cell_23/MatMul:product:0'lstm_23/lstm_cell_23/MatMul_1:product:0*
T0*(
_output_shapes
:         ┤2
lstm_23/lstm_cell_23/add╠
+lstm_23/lstm_cell_23/BiasAdd/ReadVariableOpReadVariableOp4lstm_23_lstm_cell_23_biasadd_readvariableop_resource*
_output_shapes	
:┤*
dtype02-
+lstm_23/lstm_cell_23/BiasAdd/ReadVariableOp═
lstm_23/lstm_cell_23/BiasAddBiasAddlstm_23/lstm_cell_23/add:z:03lstm_23/lstm_cell_23/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ┤2
lstm_23/lstm_cell_23/BiasAddО
$lstm_23/lstm_cell_23/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2&
$lstm_23/lstm_cell_23/split/split_dimУ
lstm_23/lstm_cell_23/splitSplit-lstm_23/lstm_cell_23/split/split_dim:output:0%lstm_23/lstm_cell_23/BiasAdd:output:0*
T0*`
_output_shapesN
L:         m:         m:         m:         m*
	num_split2
lstm_23/lstm_cell_23/splitЮ
lstm_23/lstm_cell_23/SigmoidSigmoid#lstm_23/lstm_cell_23/split:output:0*
T0*'
_output_shapes
:         m2
lstm_23/lstm_cell_23/Sigmoidв
lstm_23/lstm_cell_23/Sigmoid_1Sigmoid#lstm_23/lstm_cell_23/split:output:1*
T0*'
_output_shapes
:         m2 
lstm_23/lstm_cell_23/Sigmoid_1л
lstm_23/lstm_cell_23/mulMul"lstm_23/lstm_cell_23/Sigmoid_1:y:0lstm_23/zeros_1:output:0*
T0*'
_output_shapes
:         m2
lstm_23/lstm_cell_23/mulХ
lstm_23/lstm_cell_23/ReluRelu#lstm_23/lstm_cell_23/split:output:2*
T0*'
_output_shapes
:         m2
lstm_23/lstm_cell_23/Relu╝
lstm_23/lstm_cell_23/mul_1Mul lstm_23/lstm_cell_23/Sigmoid:y:0'lstm_23/lstm_cell_23/Relu:activations:0*
T0*'
_output_shapes
:         m2
lstm_23/lstm_cell_23/mul_1▒
lstm_23/lstm_cell_23/add_1AddV2lstm_23/lstm_cell_23/mul:z:0lstm_23/lstm_cell_23/mul_1:z:0*
T0*'
_output_shapes
:         m2
lstm_23/lstm_cell_23/add_1в
lstm_23/lstm_cell_23/Sigmoid_2Sigmoid#lstm_23/lstm_cell_23/split:output:3*
T0*'
_output_shapes
:         m2 
lstm_23/lstm_cell_23/Sigmoid_2Ф
lstm_23/lstm_cell_23/Relu_1Relulstm_23/lstm_cell_23/add_1:z:0*
T0*'
_output_shapes
:         m2
lstm_23/lstm_cell_23/Relu_1└
lstm_23/lstm_cell_23/mul_2Mul"lstm_23/lstm_cell_23/Sigmoid_2:y:0)lstm_23/lstm_cell_23/Relu_1:activations:0*
T0*'
_output_shapes
:         m2
lstm_23/lstm_cell_23/mul_2Я
%lstm_23/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    m   2'
%lstm_23/TensorArrayV2_1/element_shape╪
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
lstm_23/timeП
 lstm_23/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         2"
 lstm_23/while/maximum_iterationsz
lstm_23/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_23/while/loop_counterЗ
lstm_23/whileWhile#lstm_23/while/loop_counter:output:0)lstm_23/while/maximum_iterations:output:0lstm_23/time:output:0 lstm_23/TensorArrayV2_1:handle:0lstm_23/zeros:output:0lstm_23/zeros_1:output:0 lstm_23/strided_slice_1:output:0?lstm_23/TensorArrayUnstack/TensorListFromTensor:output_handle:03lstm_23_lstm_cell_23_matmul_readvariableop_resource5lstm_23_lstm_cell_23_matmul_1_readvariableop_resource4lstm_23_lstm_cell_23_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         m:         m: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *'
bodyR
lstm_23_while_body_39110524*'
condR
lstm_23_while_cond_39110523*K
output_shapes:
8: : : : :         m:         m: : : : : *
parallel_iterations 2
lstm_23/while┼
8lstm_23/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    m   2:
8lstm_23/TensorArrayV2Stack/TensorListStack/element_shapeИ
*lstm_23/TensorArrayV2Stack/TensorListStackTensorListStacklstm_23/while:output:3Alstm_23/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:         m*
element_dtype02,
*lstm_23/TensorArrayV2Stack/TensorListStackС
lstm_23/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         2
lstm_23/strided_slice_3/stackМ
lstm_23/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2!
lstm_23/strided_slice_3/stack_1М
lstm_23/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_23/strided_slice_3/stack_2╩
lstm_23/strided_slice_3StridedSlice3lstm_23/TensorArrayV2Stack/TensorListStack:tensor:0&lstm_23/strided_slice_3/stack:output:0(lstm_23/strided_slice_3/stack_1:output:0(lstm_23/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         m*
shrink_axis_mask2
lstm_23/strided_slice_3Й
lstm_23/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_23/transpose_1/perm┼
lstm_23/transpose_1	Transpose3lstm_23/TensorArrayV2Stack/TensorListStack:tensor:0!lstm_23/transpose_1/perm:output:0*
T0*+
_output_shapes
:         m2
lstm_23/transpose_1v
lstm_23/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_23/runtimeЕ
dropout_23/IdentityIdentitylstm_23/transpose_1:y:0*
T0*+
_output_shapes
:         m2
dropout_23/Identity▒
!dense_11/Tensordot/ReadVariableOpReadVariableOp*dense_11_tensordot_readvariableop_resource*
_output_shapes

:m*
dtype02#
!dense_11/Tensordot/ReadVariableOp|
dense_11/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_11/Tensordot/axesГ
dense_11/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_11/Tensordot/freeА
dense_11/Tensordot/ShapeShapedropout_23/Identity:output:0*
T0*
_output_shapes
:2
dense_11/Tensordot/ShapeЖ
 dense_11/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_11/Tensordot/GatherV2/axis■
dense_11/Tensordot/GatherV2GatherV2!dense_11/Tensordot/Shape:output:0 dense_11/Tensordot/free:output:0)dense_11/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_11/Tensordot/GatherV2К
"dense_11/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"dense_11/Tensordot/GatherV2_1/axisД
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
dense_11/Tensordot/Constд
dense_11/Tensordot/ProdProd$dense_11/Tensordot/GatherV2:output:0!dense_11/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_11/Tensordot/ProdВ
dense_11/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_11/Tensordot/Const_1м
dense_11/Tensordot/Prod_1Prod&dense_11/Tensordot/GatherV2_1:output:0#dense_11/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_11/Tensordot/Prod_1В
dense_11/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2 
dense_11/Tensordot/concat/axis▌
dense_11/Tensordot/concatConcatV2 dense_11/Tensordot/free:output:0 dense_11/Tensordot/axes:output:0'dense_11/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_11/Tensordot/concat░
dense_11/Tensordot/stackPack dense_11/Tensordot/Prod:output:0"dense_11/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_11/Tensordot/stack┴
dense_11/Tensordot/transpose	Transposedropout_23/Identity:output:0"dense_11/Tensordot/concat:output:0*
T0*+
_output_shapes
:         m2
dense_11/Tensordot/transpose├
dense_11/Tensordot/ReshapeReshape dense_11/Tensordot/transpose:y:0!dense_11/Tensordot/stack:output:0*
T0*0
_output_shapes
:                  2
dense_11/Tensordot/Reshape┬
dense_11/Tensordot/MatMulMatMul#dense_11/Tensordot/Reshape:output:0)dense_11/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_11/Tensordot/MatMulВ
dense_11/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
dense_11/Tensordot/Const_2Ж
 dense_11/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_11/Tensordot/concat_1/axisъ
dense_11/Tensordot/concat_1ConcatV2$dense_11/Tensordot/GatherV2:output:0#dense_11/Tensordot/Const_2:output:0)dense_11/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_11/Tensordot/concat_1┤
dense_11/TensordotReshape#dense_11/Tensordot/MatMul:product:0$dense_11/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:         2
dense_11/Tensordotз
dense_11/BiasAdd/ReadVariableOpReadVariableOp(dense_11_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_11/BiasAdd/ReadVariableOpл
dense_11/BiasAddBiasAdddense_11/Tensordot:output:0'dense_11/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:         2
dense_11/BiasAddА
dense_11/SoftmaxSoftmaxdense_11/BiasAdd:output:0*
T0*+
_output_shapes
:         2
dense_11/Softmaxy
IdentityIdentitydense_11/Softmax:softmax:0^NoOp*
T0*+
_output_shapes
:         2

Identity╚
NoOpNoOp ^dense_11/BiasAdd/ReadVariableOp"^dense_11/Tensordot/ReadVariableOp,^lstm_22/lstm_cell_22/BiasAdd/ReadVariableOp+^lstm_22/lstm_cell_22/MatMul/ReadVariableOp-^lstm_22/lstm_cell_22/MatMul_1/ReadVariableOp^lstm_22/while,^lstm_23/lstm_cell_23/BiasAdd/ReadVariableOp+^lstm_23/lstm_cell_23/MatMul/ReadVariableOp-^lstm_23/lstm_cell_23/MatMul_1/ReadVariableOp^lstm_23/while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':         ]: : : : : : : : 2B
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
:         ]
 
_user_specified_nameinputs
╨

э
lstm_23_while_cond_39110523,
(lstm_23_while_lstm_23_while_loop_counter2
.lstm_23_while_lstm_23_while_maximum_iterations
lstm_23_while_placeholder
lstm_23_while_placeholder_1
lstm_23_while_placeholder_2
lstm_23_while_placeholder_3.
*lstm_23_while_less_lstm_23_strided_slice_1F
Blstm_23_while_lstm_23_while_cond_39110523___redundant_placeholder0F
Blstm_23_while_lstm_23_while_cond_39110523___redundant_placeholder1F
Blstm_23_while_lstm_23_while_cond_39110523___redundant_placeholder2F
Blstm_23_while_lstm_23_while_cond_39110523___redundant_placeholder3
lstm_23_while_identity
Ш
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
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :         m:         m: ::::: 
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
:         m:-)
'
_output_shapes
:         m:

_output_shapes
: :

_output_shapes
:
Ю?
╘
while_body_39112063
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
3while_lstm_cell_23_matmul_readvariableop_resource_0:
Д┤H
5while_lstm_cell_23_matmul_1_readvariableop_resource_0:	m┤C
4while_lstm_cell_23_biasadd_readvariableop_resource_0:	┤
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
1while_lstm_cell_23_matmul_readvariableop_resource:
Д┤F
3while_lstm_cell_23_matmul_1_readvariableop_resource:	m┤A
2while_lstm_cell_23_biasadd_readvariableop_resource:	┤Ив)while/lstm_cell_23/BiasAdd/ReadVariableOpв(while/lstm_cell_23/MatMul/ReadVariableOpв*while/lstm_cell_23/MatMul_1/ReadVariableOp├
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    Д   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape╘
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:         Д*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem╩
(while/lstm_cell_23/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_23_matmul_readvariableop_resource_0* 
_output_shapes
:
Д┤*
dtype02*
(while/lstm_cell_23/MatMul/ReadVariableOp╫
while/lstm_cell_23/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_23/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ┤2
while/lstm_cell_23/MatMul╧
*while/lstm_cell_23/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_23_matmul_1_readvariableop_resource_0*
_output_shapes
:	m┤*
dtype02,
*while/lstm_cell_23/MatMul_1/ReadVariableOp└
while/lstm_cell_23/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_23/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ┤2
while/lstm_cell_23/MatMul_1╕
while/lstm_cell_23/addAddV2#while/lstm_cell_23/MatMul:product:0%while/lstm_cell_23/MatMul_1:product:0*
T0*(
_output_shapes
:         ┤2
while/lstm_cell_23/add╚
)while/lstm_cell_23/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_23_biasadd_readvariableop_resource_0*
_output_shapes	
:┤*
dtype02+
)while/lstm_cell_23/BiasAdd/ReadVariableOp┼
while/lstm_cell_23/BiasAddBiasAddwhile/lstm_cell_23/add:z:01while/lstm_cell_23/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ┤2
while/lstm_cell_23/BiasAddК
"while/lstm_cell_23/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_23/split/split_dimЛ
while/lstm_cell_23/splitSplit+while/lstm_cell_23/split/split_dim:output:0#while/lstm_cell_23/BiasAdd:output:0*
T0*`
_output_shapesN
L:         m:         m:         m:         m*
	num_split2
while/lstm_cell_23/splitШ
while/lstm_cell_23/SigmoidSigmoid!while/lstm_cell_23/split:output:0*
T0*'
_output_shapes
:         m2
while/lstm_cell_23/SigmoidЬ
while/lstm_cell_23/Sigmoid_1Sigmoid!while/lstm_cell_23/split:output:1*
T0*'
_output_shapes
:         m2
while/lstm_cell_23/Sigmoid_1а
while/lstm_cell_23/mulMul while/lstm_cell_23/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:         m2
while/lstm_cell_23/mulП
while/lstm_cell_23/ReluRelu!while/lstm_cell_23/split:output:2*
T0*'
_output_shapes
:         m2
while/lstm_cell_23/Relu┤
while/lstm_cell_23/mul_1Mulwhile/lstm_cell_23/Sigmoid:y:0%while/lstm_cell_23/Relu:activations:0*
T0*'
_output_shapes
:         m2
while/lstm_cell_23/mul_1й
while/lstm_cell_23/add_1AddV2while/lstm_cell_23/mul:z:0while/lstm_cell_23/mul_1:z:0*
T0*'
_output_shapes
:         m2
while/lstm_cell_23/add_1Ь
while/lstm_cell_23/Sigmoid_2Sigmoid!while/lstm_cell_23/split:output:3*
T0*'
_output_shapes
:         m2
while/lstm_cell_23/Sigmoid_2О
while/lstm_cell_23/Relu_1Reluwhile/lstm_cell_23/add_1:z:0*
T0*'
_output_shapes
:         m2
while/lstm_cell_23/Relu_1╕
while/lstm_cell_23/mul_2Mul while/lstm_cell_23/Sigmoid_2:y:0'while/lstm_cell_23/Relu_1:activations:0*
T0*'
_output_shapes
:         m2
while/lstm_cell_23/mul_2р
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
while/Identity_2Ъ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3Н
while/Identity_4Identitywhile/lstm_cell_23/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:         m2
while/Identity_4Н
while/Identity_5Identitywhile/lstm_cell_23/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:         m2
while/Identity_5▐

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
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         m:         m: : : : : 2V
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
: :-)
'
_output_shapes
:         m:-)
'
_output_shapes
:         m:

_output_shapes
: :

_output_shapes
: 
╪
I
-__inference_dropout_22_layer_call_fn_39111689

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
:         Д* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_dropout_22_layer_call_and_return_conditional_losses_391095222
PartitionedCallq
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:         Д2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         Д:T P
,
_output_shapes
:         Д
 
_user_specified_nameinputs
╨!
¤
F__inference_dense_11_layer_call_and_return_conditional_losses_39112400

inputs3
!tensordot_readvariableop_resource:m-
biasadd_readvariableop_resource:
identityИвBiasAdd/ReadVariableOpвTensordot/ReadVariableOpЦ
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:m*
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
Tensordot/stackР
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:         m2
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
_construction_contextkEagerRuntime*.
_input_shapes
:         m: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:S O
+
_output_shapes
:         m
 
_user_specified_nameinputs
╥^
Ц
)sequential_11_lstm_23_while_body_39107980H
Dsequential_11_lstm_23_while_sequential_11_lstm_23_while_loop_counterN
Jsequential_11_lstm_23_while_sequential_11_lstm_23_while_maximum_iterations+
'sequential_11_lstm_23_while_placeholder-
)sequential_11_lstm_23_while_placeholder_1-
)sequential_11_lstm_23_while_placeholder_2-
)sequential_11_lstm_23_while_placeholder_3G
Csequential_11_lstm_23_while_sequential_11_lstm_23_strided_slice_1_0Г
sequential_11_lstm_23_while_tensorarrayv2read_tensorlistgetitem_sequential_11_lstm_23_tensorarrayunstack_tensorlistfromtensor_0]
Isequential_11_lstm_23_while_lstm_cell_23_matmul_readvariableop_resource_0:
Д┤^
Ksequential_11_lstm_23_while_lstm_cell_23_matmul_1_readvariableop_resource_0:	m┤Y
Jsequential_11_lstm_23_while_lstm_cell_23_biasadd_readvariableop_resource_0:	┤(
$sequential_11_lstm_23_while_identity*
&sequential_11_lstm_23_while_identity_1*
&sequential_11_lstm_23_while_identity_2*
&sequential_11_lstm_23_while_identity_3*
&sequential_11_lstm_23_while_identity_4*
&sequential_11_lstm_23_while_identity_5E
Asequential_11_lstm_23_while_sequential_11_lstm_23_strided_slice_1Б
}sequential_11_lstm_23_while_tensorarrayv2read_tensorlistgetitem_sequential_11_lstm_23_tensorarrayunstack_tensorlistfromtensor[
Gsequential_11_lstm_23_while_lstm_cell_23_matmul_readvariableop_resource:
Д┤\
Isequential_11_lstm_23_while_lstm_cell_23_matmul_1_readvariableop_resource:	m┤W
Hsequential_11_lstm_23_while_lstm_cell_23_biasadd_readvariableop_resource:	┤Ив?sequential_11/lstm_23/while/lstm_cell_23/BiasAdd/ReadVariableOpв>sequential_11/lstm_23/while/lstm_cell_23/MatMul/ReadVariableOpв@sequential_11/lstm_23/while/lstm_cell_23/MatMul_1/ReadVariableOpя
Msequential_11/lstm_23/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    Д   2O
Msequential_11/lstm_23/while/TensorArrayV2Read/TensorListGetItem/element_shape╪
?sequential_11/lstm_23/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemsequential_11_lstm_23_while_tensorarrayv2read_tensorlistgetitem_sequential_11_lstm_23_tensorarrayunstack_tensorlistfromtensor_0'sequential_11_lstm_23_while_placeholderVsequential_11/lstm_23/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:         Д*
element_dtype02A
?sequential_11/lstm_23/while/TensorArrayV2Read/TensorListGetItemМ
>sequential_11/lstm_23/while/lstm_cell_23/MatMul/ReadVariableOpReadVariableOpIsequential_11_lstm_23_while_lstm_cell_23_matmul_readvariableop_resource_0* 
_output_shapes
:
Д┤*
dtype02@
>sequential_11/lstm_23/while/lstm_cell_23/MatMul/ReadVariableOpп
/sequential_11/lstm_23/while/lstm_cell_23/MatMulMatMulFsequential_11/lstm_23/while/TensorArrayV2Read/TensorListGetItem:item:0Fsequential_11/lstm_23/while/lstm_cell_23/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ┤21
/sequential_11/lstm_23/while/lstm_cell_23/MatMulС
@sequential_11/lstm_23/while/lstm_cell_23/MatMul_1/ReadVariableOpReadVariableOpKsequential_11_lstm_23_while_lstm_cell_23_matmul_1_readvariableop_resource_0*
_output_shapes
:	m┤*
dtype02B
@sequential_11/lstm_23/while/lstm_cell_23/MatMul_1/ReadVariableOpШ
1sequential_11/lstm_23/while/lstm_cell_23/MatMul_1MatMul)sequential_11_lstm_23_while_placeholder_2Hsequential_11/lstm_23/while/lstm_cell_23/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ┤23
1sequential_11/lstm_23/while/lstm_cell_23/MatMul_1Р
,sequential_11/lstm_23/while/lstm_cell_23/addAddV29sequential_11/lstm_23/while/lstm_cell_23/MatMul:product:0;sequential_11/lstm_23/while/lstm_cell_23/MatMul_1:product:0*
T0*(
_output_shapes
:         ┤2.
,sequential_11/lstm_23/while/lstm_cell_23/addК
?sequential_11/lstm_23/while/lstm_cell_23/BiasAdd/ReadVariableOpReadVariableOpJsequential_11_lstm_23_while_lstm_cell_23_biasadd_readvariableop_resource_0*
_output_shapes	
:┤*
dtype02A
?sequential_11/lstm_23/while/lstm_cell_23/BiasAdd/ReadVariableOpЭ
0sequential_11/lstm_23/while/lstm_cell_23/BiasAddBiasAdd0sequential_11/lstm_23/while/lstm_cell_23/add:z:0Gsequential_11/lstm_23/while/lstm_cell_23/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ┤22
0sequential_11/lstm_23/while/lstm_cell_23/BiasAdd╢
8sequential_11/lstm_23/while/lstm_cell_23/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2:
8sequential_11/lstm_23/while/lstm_cell_23/split/split_dimу
.sequential_11/lstm_23/while/lstm_cell_23/splitSplitAsequential_11/lstm_23/while/lstm_cell_23/split/split_dim:output:09sequential_11/lstm_23/while/lstm_cell_23/BiasAdd:output:0*
T0*`
_output_shapesN
L:         m:         m:         m:         m*
	num_split20
.sequential_11/lstm_23/while/lstm_cell_23/split┌
0sequential_11/lstm_23/while/lstm_cell_23/SigmoidSigmoid7sequential_11/lstm_23/while/lstm_cell_23/split:output:0*
T0*'
_output_shapes
:         m22
0sequential_11/lstm_23/while/lstm_cell_23/Sigmoid▐
2sequential_11/lstm_23/while/lstm_cell_23/Sigmoid_1Sigmoid7sequential_11/lstm_23/while/lstm_cell_23/split:output:1*
T0*'
_output_shapes
:         m24
2sequential_11/lstm_23/while/lstm_cell_23/Sigmoid_1°
,sequential_11/lstm_23/while/lstm_cell_23/mulMul6sequential_11/lstm_23/while/lstm_cell_23/Sigmoid_1:y:0)sequential_11_lstm_23_while_placeholder_3*
T0*'
_output_shapes
:         m2.
,sequential_11/lstm_23/while/lstm_cell_23/mul╤
-sequential_11/lstm_23/while/lstm_cell_23/ReluRelu7sequential_11/lstm_23/while/lstm_cell_23/split:output:2*
T0*'
_output_shapes
:         m2/
-sequential_11/lstm_23/while/lstm_cell_23/ReluМ
.sequential_11/lstm_23/while/lstm_cell_23/mul_1Mul4sequential_11/lstm_23/while/lstm_cell_23/Sigmoid:y:0;sequential_11/lstm_23/while/lstm_cell_23/Relu:activations:0*
T0*'
_output_shapes
:         m20
.sequential_11/lstm_23/while/lstm_cell_23/mul_1Б
.sequential_11/lstm_23/while/lstm_cell_23/add_1AddV20sequential_11/lstm_23/while/lstm_cell_23/mul:z:02sequential_11/lstm_23/while/lstm_cell_23/mul_1:z:0*
T0*'
_output_shapes
:         m20
.sequential_11/lstm_23/while/lstm_cell_23/add_1▐
2sequential_11/lstm_23/while/lstm_cell_23/Sigmoid_2Sigmoid7sequential_11/lstm_23/while/lstm_cell_23/split:output:3*
T0*'
_output_shapes
:         m24
2sequential_11/lstm_23/while/lstm_cell_23/Sigmoid_2╨
/sequential_11/lstm_23/while/lstm_cell_23/Relu_1Relu2sequential_11/lstm_23/while/lstm_cell_23/add_1:z:0*
T0*'
_output_shapes
:         m21
/sequential_11/lstm_23/while/lstm_cell_23/Relu_1Р
.sequential_11/lstm_23/while/lstm_cell_23/mul_2Mul6sequential_11/lstm_23/while/lstm_cell_23/Sigmoid_2:y:0=sequential_11/lstm_23/while/lstm_cell_23/Relu_1:activations:0*
T0*'
_output_shapes
:         m20
.sequential_11/lstm_23/while/lstm_cell_23/mul_2╬
@sequential_11/lstm_23/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem)sequential_11_lstm_23_while_placeholder_1'sequential_11_lstm_23_while_placeholder2sequential_11/lstm_23/while/lstm_cell_23/mul_2:z:0*
_output_shapes
: *
element_dtype02B
@sequential_11/lstm_23/while/TensorArrayV2Write/TensorListSetItemИ
!sequential_11/lstm_23/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2#
!sequential_11/lstm_23/while/add/y┴
sequential_11/lstm_23/while/addAddV2'sequential_11_lstm_23_while_placeholder*sequential_11/lstm_23/while/add/y:output:0*
T0*
_output_shapes
: 2!
sequential_11/lstm_23/while/addМ
#sequential_11/lstm_23/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2%
#sequential_11/lstm_23/while/add_1/yф
!sequential_11/lstm_23/while/add_1AddV2Dsequential_11_lstm_23_while_sequential_11_lstm_23_while_loop_counter,sequential_11/lstm_23/while/add_1/y:output:0*
T0*
_output_shapes
: 2#
!sequential_11/lstm_23/while/add_1├
$sequential_11/lstm_23/while/IdentityIdentity%sequential_11/lstm_23/while/add_1:z:0!^sequential_11/lstm_23/while/NoOp*
T0*
_output_shapes
: 2&
$sequential_11/lstm_23/while/Identityь
&sequential_11/lstm_23/while/Identity_1IdentityJsequential_11_lstm_23_while_sequential_11_lstm_23_while_maximum_iterations!^sequential_11/lstm_23/while/NoOp*
T0*
_output_shapes
: 2(
&sequential_11/lstm_23/while/Identity_1┼
&sequential_11/lstm_23/while/Identity_2Identity#sequential_11/lstm_23/while/add:z:0!^sequential_11/lstm_23/while/NoOp*
T0*
_output_shapes
: 2(
&sequential_11/lstm_23/while/Identity_2Є
&sequential_11/lstm_23/while/Identity_3IdentityPsequential_11/lstm_23/while/TensorArrayV2Write/TensorListSetItem:output_handle:0!^sequential_11/lstm_23/while/NoOp*
T0*
_output_shapes
: 2(
&sequential_11/lstm_23/while/Identity_3х
&sequential_11/lstm_23/while/Identity_4Identity2sequential_11/lstm_23/while/lstm_cell_23/mul_2:z:0!^sequential_11/lstm_23/while/NoOp*
T0*'
_output_shapes
:         m2(
&sequential_11/lstm_23/while/Identity_4х
&sequential_11/lstm_23/while/Identity_5Identity2sequential_11/lstm_23/while/lstm_cell_23/add_1:z:0!^sequential_11/lstm_23/while/NoOp*
T0*'
_output_shapes
:         m2(
&sequential_11/lstm_23/while/Identity_5╠
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
&sequential_11_lstm_23_while_identity_5/sequential_11/lstm_23/while/Identity_5:output:0"Ц
Hsequential_11_lstm_23_while_lstm_cell_23_biasadd_readvariableop_resourceJsequential_11_lstm_23_while_lstm_cell_23_biasadd_readvariableop_resource_0"Ш
Isequential_11_lstm_23_while_lstm_cell_23_matmul_1_readvariableop_resourceKsequential_11_lstm_23_while_lstm_cell_23_matmul_1_readvariableop_resource_0"Ф
Gsequential_11_lstm_23_while_lstm_cell_23_matmul_readvariableop_resourceIsequential_11_lstm_23_while_lstm_cell_23_matmul_readvariableop_resource_0"И
Asequential_11_lstm_23_while_sequential_11_lstm_23_strided_slice_1Csequential_11_lstm_23_while_sequential_11_lstm_23_strided_slice_1_0"А
}sequential_11_lstm_23_while_tensorarrayv2read_tensorlistgetitem_sequential_11_lstm_23_tensorarrayunstack_tensorlistfromtensorsequential_11_lstm_23_while_tensorarrayv2read_tensorlistgetitem_sequential_11_lstm_23_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         m:         m: : : : : 2В
?sequential_11/lstm_23/while/lstm_cell_23/BiasAdd/ReadVariableOp?sequential_11/lstm_23/while/lstm_cell_23/BiasAdd/ReadVariableOp2А
>sequential_11/lstm_23/while/lstm_cell_23/MatMul/ReadVariableOp>sequential_11/lstm_23/while/lstm_cell_23/MatMul/ReadVariableOp2Д
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
: :-)
'
_output_shapes
:         m:-)
'
_output_shapes
:         m:

_output_shapes
: :

_output_shapes
: 
р
║
*__inference_lstm_22_layer_call_fn_39111634
inputs_0
unknown:	]Р
	unknown_0:
ДР
	unknown_1:	Р
identityИвStatefulPartitionedCallТ
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:                  Д*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_lstm_22_layer_call_and_return_conditional_losses_391082502
StatefulPartitionedCallЙ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:                  Д2

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
E__inference_lstm_22_layer_call_and_return_conditional_losses_39111321
inputs_0>
+lstm_cell_22_matmul_readvariableop_resource:	]РA
-lstm_cell_22_matmul_1_readvariableop_resource:
ДР;
,lstm_cell_22_biasadd_readvariableop_resource:	Р
identityИв#lstm_cell_22/BiasAdd/ReadVariableOpв"lstm_cell_22/MatMul/ReadVariableOpв$lstm_cell_22/MatMul_1/ReadVariableOpвwhileF
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
B :Д2
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
B :Д2
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
:         Д2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :Д2
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
B :Д2
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
:         Д2	
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
"lstm_cell_22/MatMul/ReadVariableOpReadVariableOp+lstm_cell_22_matmul_readvariableop_resource*
_output_shapes
:	]Р*
dtype02$
"lstm_cell_22/MatMul/ReadVariableOpн
lstm_cell_22/MatMulMatMulstrided_slice_2:output:0*lstm_cell_22/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Р2
lstm_cell_22/MatMul╝
$lstm_cell_22/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_22_matmul_1_readvariableop_resource* 
_output_shapes
:
ДР*
dtype02&
$lstm_cell_22/MatMul_1/ReadVariableOpй
lstm_cell_22/MatMul_1MatMulzeros:output:0,lstm_cell_22/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Р2
lstm_cell_22/MatMul_1а
lstm_cell_22/addAddV2lstm_cell_22/MatMul:product:0lstm_cell_22/MatMul_1:product:0*
T0*(
_output_shapes
:         Р2
lstm_cell_22/add┤
#lstm_cell_22/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_22_biasadd_readvariableop_resource*
_output_shapes	
:Р*
dtype02%
#lstm_cell_22/BiasAdd/ReadVariableOpн
lstm_cell_22/BiasAddBiasAddlstm_cell_22/add:z:0+lstm_cell_22/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Р2
lstm_cell_22/BiasAdd~
lstm_cell_22/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_22/split/split_dimў
lstm_cell_22/splitSplit%lstm_cell_22/split/split_dim:output:0lstm_cell_22/BiasAdd:output:0*
T0*d
_output_shapesR
P:         Д:         Д:         Д:         Д*
	num_split2
lstm_cell_22/splitЗ
lstm_cell_22/SigmoidSigmoidlstm_cell_22/split:output:0*
T0*(
_output_shapes
:         Д2
lstm_cell_22/SigmoidЛ
lstm_cell_22/Sigmoid_1Sigmoidlstm_cell_22/split:output:1*
T0*(
_output_shapes
:         Д2
lstm_cell_22/Sigmoid_1М
lstm_cell_22/mulMullstm_cell_22/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:         Д2
lstm_cell_22/mul~
lstm_cell_22/ReluRelulstm_cell_22/split:output:2*
T0*(
_output_shapes
:         Д2
lstm_cell_22/ReluЭ
lstm_cell_22/mul_1Mullstm_cell_22/Sigmoid:y:0lstm_cell_22/Relu:activations:0*
T0*(
_output_shapes
:         Д2
lstm_cell_22/mul_1Т
lstm_cell_22/add_1AddV2lstm_cell_22/mul:z:0lstm_cell_22/mul_1:z:0*
T0*(
_output_shapes
:         Д2
lstm_cell_22/add_1Л
lstm_cell_22/Sigmoid_2Sigmoidlstm_cell_22/split:output:3*
T0*(
_output_shapes
:         Д2
lstm_cell_22/Sigmoid_2}
lstm_cell_22/Relu_1Relulstm_cell_22/add_1:z:0*
T0*(
_output_shapes
:         Д2
lstm_cell_22/Relu_1б
lstm_cell_22/mul_2Mullstm_cell_22/Sigmoid_2:y:0!lstm_cell_22/Relu_1:activations:0*
T0*(
_output_shapes
:         Д2
lstm_cell_22/mul_2П
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    Д   2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_22_matmul_readvariableop_resource-lstm_cell_22_matmul_1_readvariableop_resource,lstm_cell_22_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :         Д:         Д: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_39111237*
condR
while_cond_39111236*M
output_shapes<
:: : : : :         Д:         Д: : : : : *
parallel_iterations 2
while╡
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    Д   22
0TensorArrayV2Stack/TensorListStack/element_shapeЄ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:                  Д*
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
:         Д*
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
!:                  Д2
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
!:                  Д2

Identity╚
NoOpNoOp$^lstm_cell_22/BiasAdd/ReadVariableOp#^lstm_cell_22/MatMul/ReadVariableOp%^lstm_cell_22/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  ]: : : 2J
#lstm_cell_22/BiasAdd/ReadVariableOp#lstm_cell_22/BiasAdd/ReadVariableOp2H
"lstm_cell_22/MatMul/ReadVariableOp"lstm_cell_22/MatMul/ReadVariableOp2L
$lstm_cell_22/MatMul_1/ReadVariableOp$lstm_cell_22/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :                  ]
"
_user_specified_name
inputs/0
╢
╕
*__inference_lstm_22_layer_call_fn_39111667

inputs
unknown:	]Р
	unknown_0:
ДР
	unknown_1:	Р
identityИвStatefulPartitionedCallЗ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         Д*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_lstm_22_layer_call_and_return_conditional_losses_391101392
StatefulPartitionedCallА
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:         Д2

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
э[
Ю
E__inference_lstm_23_layer_call_and_return_conditional_losses_39112147

inputs?
+lstm_cell_23_matmul_readvariableop_resource:
Д┤@
-lstm_cell_23_matmul_1_readvariableop_resource:	m┤;
,lstm_cell_23_biasadd_readvariableop_resource:	┤
identityИв#lstm_cell_23/BiasAdd/ReadVariableOpв"lstm_cell_23/MatMul/ReadVariableOpв$lstm_cell_23/MatMul_1/ReadVariableOpвwhileD
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
value	B :m2
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
value	B :m2
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
:         m2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :m2
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
value	B :m2
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
:         m2	
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
:         Д2
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
valueB"    Д   27
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
:         Д*
shrink_axis_mask2
strided_slice_2╢
"lstm_cell_23/MatMul/ReadVariableOpReadVariableOp+lstm_cell_23_matmul_readvariableop_resource* 
_output_shapes
:
Д┤*
dtype02$
"lstm_cell_23/MatMul/ReadVariableOpн
lstm_cell_23/MatMulMatMulstrided_slice_2:output:0*lstm_cell_23/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ┤2
lstm_cell_23/MatMul╗
$lstm_cell_23/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_23_matmul_1_readvariableop_resource*
_output_shapes
:	m┤*
dtype02&
$lstm_cell_23/MatMul_1/ReadVariableOpй
lstm_cell_23/MatMul_1MatMulzeros:output:0,lstm_cell_23/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ┤2
lstm_cell_23/MatMul_1а
lstm_cell_23/addAddV2lstm_cell_23/MatMul:product:0lstm_cell_23/MatMul_1:product:0*
T0*(
_output_shapes
:         ┤2
lstm_cell_23/add┤
#lstm_cell_23/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_23_biasadd_readvariableop_resource*
_output_shapes	
:┤*
dtype02%
#lstm_cell_23/BiasAdd/ReadVariableOpн
lstm_cell_23/BiasAddBiasAddlstm_cell_23/add:z:0+lstm_cell_23/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ┤2
lstm_cell_23/BiasAdd~
lstm_cell_23/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_23/split/split_dimє
lstm_cell_23/splitSplit%lstm_cell_23/split/split_dim:output:0lstm_cell_23/BiasAdd:output:0*
T0*`
_output_shapesN
L:         m:         m:         m:         m*
	num_split2
lstm_cell_23/splitЖ
lstm_cell_23/SigmoidSigmoidlstm_cell_23/split:output:0*
T0*'
_output_shapes
:         m2
lstm_cell_23/SigmoidК
lstm_cell_23/Sigmoid_1Sigmoidlstm_cell_23/split:output:1*
T0*'
_output_shapes
:         m2
lstm_cell_23/Sigmoid_1Л
lstm_cell_23/mulMullstm_cell_23/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:         m2
lstm_cell_23/mul}
lstm_cell_23/ReluRelulstm_cell_23/split:output:2*
T0*'
_output_shapes
:         m2
lstm_cell_23/ReluЬ
lstm_cell_23/mul_1Mullstm_cell_23/Sigmoid:y:0lstm_cell_23/Relu:activations:0*
T0*'
_output_shapes
:         m2
lstm_cell_23/mul_1С
lstm_cell_23/add_1AddV2lstm_cell_23/mul:z:0lstm_cell_23/mul_1:z:0*
T0*'
_output_shapes
:         m2
lstm_cell_23/add_1К
lstm_cell_23/Sigmoid_2Sigmoidlstm_cell_23/split:output:3*
T0*'
_output_shapes
:         m2
lstm_cell_23/Sigmoid_2|
lstm_cell_23/Relu_1Relulstm_cell_23/add_1:z:0*
T0*'
_output_shapes
:         m2
lstm_cell_23/Relu_1а
lstm_cell_23/mul_2Mullstm_cell_23/Sigmoid_2:y:0!lstm_cell_23/Relu_1:activations:0*
T0*'
_output_shapes
:         m2
lstm_cell_23/mul_2П
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    m   2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_23_matmul_readvariableop_resource-lstm_cell_23_matmul_1_readvariableop_resource,lstm_cell_23_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         m:         m: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_39112063*
condR
while_cond_39112062*K
output_shapes:
8: : : : :         m:         m: : : : : *
parallel_iterations 2
while╡
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    m   22
0TensorArrayV2Stack/TensorListStack/element_shapeш
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:         m*
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
:         m*
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
:         m2
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
:         m2

Identity╚
NoOpNoOp$^lstm_cell_23/BiasAdd/ReadVariableOp#^lstm_cell_23/MatMul/ReadVariableOp%^lstm_cell_23/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:         Д: : : 2J
#lstm_cell_23/BiasAdd/ReadVariableOp#lstm_cell_23/BiasAdd/ReadVariableOp2H
"lstm_cell_23/MatMul/ReadVariableOp"lstm_cell_23/MatMul/ReadVariableOp2L
$lstm_cell_23/MatMul_1/ReadVariableOp$lstm_cell_23/MatMul_1/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:         Д
 
_user_specified_nameinputs
у
═
while_cond_39110054
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_39110054___redundant_placeholder06
2while_while_cond_39110054___redundant_placeholder16
2while_while_cond_39110054___redundant_placeholder26
2while_while_cond_39110054___redundant_placeholder3
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
B: : : : :         Д:         Д: ::::: 
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
:         Д:.*
(
_output_shapes
:         Д:

_output_shapes
: :

_output_shapes
:
э[
Ю
E__inference_lstm_23_layer_call_and_return_conditional_losses_39112298

inputs?
+lstm_cell_23_matmul_readvariableop_resource:
Д┤@
-lstm_cell_23_matmul_1_readvariableop_resource:	m┤;
,lstm_cell_23_biasadd_readvariableop_resource:	┤
identityИв#lstm_cell_23/BiasAdd/ReadVariableOpв"lstm_cell_23/MatMul/ReadVariableOpв$lstm_cell_23/MatMul_1/ReadVariableOpвwhileD
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
value	B :m2
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
value	B :m2
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
:         m2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :m2
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
value	B :m2
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
:         m2	
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
:         Д2
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
valueB"    Д   27
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
:         Д*
shrink_axis_mask2
strided_slice_2╢
"lstm_cell_23/MatMul/ReadVariableOpReadVariableOp+lstm_cell_23_matmul_readvariableop_resource* 
_output_shapes
:
Д┤*
dtype02$
"lstm_cell_23/MatMul/ReadVariableOpн
lstm_cell_23/MatMulMatMulstrided_slice_2:output:0*lstm_cell_23/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ┤2
lstm_cell_23/MatMul╗
$lstm_cell_23/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_23_matmul_1_readvariableop_resource*
_output_shapes
:	m┤*
dtype02&
$lstm_cell_23/MatMul_1/ReadVariableOpй
lstm_cell_23/MatMul_1MatMulzeros:output:0,lstm_cell_23/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ┤2
lstm_cell_23/MatMul_1а
lstm_cell_23/addAddV2lstm_cell_23/MatMul:product:0lstm_cell_23/MatMul_1:product:0*
T0*(
_output_shapes
:         ┤2
lstm_cell_23/add┤
#lstm_cell_23/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_23_biasadd_readvariableop_resource*
_output_shapes	
:┤*
dtype02%
#lstm_cell_23/BiasAdd/ReadVariableOpн
lstm_cell_23/BiasAddBiasAddlstm_cell_23/add:z:0+lstm_cell_23/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ┤2
lstm_cell_23/BiasAdd~
lstm_cell_23/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_23/split/split_dimє
lstm_cell_23/splitSplit%lstm_cell_23/split/split_dim:output:0lstm_cell_23/BiasAdd:output:0*
T0*`
_output_shapesN
L:         m:         m:         m:         m*
	num_split2
lstm_cell_23/splitЖ
lstm_cell_23/SigmoidSigmoidlstm_cell_23/split:output:0*
T0*'
_output_shapes
:         m2
lstm_cell_23/SigmoidК
lstm_cell_23/Sigmoid_1Sigmoidlstm_cell_23/split:output:1*
T0*'
_output_shapes
:         m2
lstm_cell_23/Sigmoid_1Л
lstm_cell_23/mulMullstm_cell_23/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:         m2
lstm_cell_23/mul}
lstm_cell_23/ReluRelulstm_cell_23/split:output:2*
T0*'
_output_shapes
:         m2
lstm_cell_23/ReluЬ
lstm_cell_23/mul_1Mullstm_cell_23/Sigmoid:y:0lstm_cell_23/Relu:activations:0*
T0*'
_output_shapes
:         m2
lstm_cell_23/mul_1С
lstm_cell_23/add_1AddV2lstm_cell_23/mul:z:0lstm_cell_23/mul_1:z:0*
T0*'
_output_shapes
:         m2
lstm_cell_23/add_1К
lstm_cell_23/Sigmoid_2Sigmoidlstm_cell_23/split:output:3*
T0*'
_output_shapes
:         m2
lstm_cell_23/Sigmoid_2|
lstm_cell_23/Relu_1Relulstm_cell_23/add_1:z:0*
T0*'
_output_shapes
:         m2
lstm_cell_23/Relu_1а
lstm_cell_23/mul_2Mullstm_cell_23/Sigmoid_2:y:0!lstm_cell_23/Relu_1:activations:0*
T0*'
_output_shapes
:         m2
lstm_cell_23/mul_2П
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    m   2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_23_matmul_readvariableop_resource-lstm_cell_23_matmul_1_readvariableop_resource,lstm_cell_23_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         m:         m: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_39112214*
condR
while_cond_39112213*K
output_shapes:
8: : : : :         m:         m: : : : : *
parallel_iterations 2
while╡
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    m   22
0TensorArrayV2Stack/TensorListStack/element_shapeш
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:         m*
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
:         m*
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
:         m2
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
:         m2

Identity╚
NoOpNoOp$^lstm_cell_23/BiasAdd/ReadVariableOp#^lstm_cell_23/MatMul/ReadVariableOp%^lstm_cell_23/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:         Д: : : 2J
#lstm_cell_23/BiasAdd/ReadVariableOp#lstm_cell_23/BiasAdd/ReadVariableOp2H
"lstm_cell_23/MatMul/ReadVariableOp"lstm_cell_23/MatMul/ReadVariableOp2L
$lstm_cell_23/MatMul_1/ReadVariableOp$lstm_cell_23/MatMul_1/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:         Д
 
_user_specified_nameinputs
м

╥
0__inference_sequential_11_layer_call_fn_39109746
lstm_22_input
unknown:	]Р
	unknown_0:
ДР
	unknown_1:	Р
	unknown_2:
Д┤
	unknown_3:	m┤
	unknown_4:	┤
	unknown_5:m
	unknown_6:
identityИвStatefulPartitionedCall╘
StatefulPartitionedCallStatefulPartitionedCalllstm_22_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
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
K__inference_sequential_11_layer_call_and_return_conditional_losses_391097272
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
_user_specified_namelstm_22_input
Л
З
J__inference_lstm_cell_22_layer_call_and_return_conditional_losses_39108313

inputs

states
states_11
matmul_readvariableop_resource:	]Р4
 matmul_1_readvariableop_resource:
ДР.
biasadd_readvariableop_resource:	Р
identity

identity_1

identity_2ИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpвMatMul_1/ReadVariableOpО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	]Р*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Р2
MatMulХ
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
ДР*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Р2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:         Р2
addН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:Р*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Р2	
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
P:         Д:         Д:         Д:         Д*
	num_split2
split`
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:         Д2	
Sigmoidd
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:         Д2
	Sigmoid_1]
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:         Д2
mulW
ReluRelusplit:output:2*
T0*(
_output_shapes
:         Д2
Relui
mul_1MulSigmoid:y:0Relu:activations:0*
T0*(
_output_shapes
:         Д2
mul_1^
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:         Д2
add_1d
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:         Д2
	Sigmoid_2V
Relu_1Relu	add_1:z:0*
T0*(
_output_shapes
:         Д2
Relu_1m
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*(
_output_shapes
:         Д2
mul_2e
IdentityIdentity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:         Д2

Identityi

Identity_1Identity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:         Д2

Identity_1i

Identity_2Identity	add_1:z:0^NoOp*
T0*(
_output_shapes
:         Д2

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
A:         ]:         Д:         Д: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:         ]
 
_user_specified_nameinputs:PL
(
_output_shapes
:         Д
 
_user_specified_namestates:PL
(
_output_shapes
:         Д
 
_user_specified_namestates
у
═
while_cond_39108390
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_39108390___redundant_placeholder06
2while_while_cond_39108390___redundant_placeholder16
2while_while_cond_39108390___redundant_placeholder26
2while_while_cond_39108390___redundant_placeholder3
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
B: : : : :         Д:         Д: ::::: 
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
:         Д:.*
(
_output_shapes
:         Д:

_output_shapes
: :

_output_shapes
:
у
═
while_cond_39109424
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_39109424___redundant_placeholder06
2while_while_cond_39109424___redundant_placeholder16
2while_while_cond_39109424___redundant_placeholder26
2while_while_cond_39109424___redundant_placeholder3
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
B: : : : :         Д:         Д: ::::: 
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
:         Д:.*
(
_output_shapes
:         Д:

_output_shapes
: :

_output_shapes
:
Л
З
J__inference_lstm_cell_22_layer_call_and_return_conditional_losses_39108167

inputs

states
states_11
matmul_readvariableop_resource:	]Р4
 matmul_1_readvariableop_resource:
ДР.
biasadd_readvariableop_resource:	Р
identity

identity_1

identity_2ИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpвMatMul_1/ReadVariableOpО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	]Р*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Р2
MatMulХ
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
ДР*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Р2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:         Р2
addН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:Р*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Р2	
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
P:         Д:         Д:         Д:         Д*
	num_split2
split`
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:         Д2	
Sigmoidd
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:         Д2
	Sigmoid_1]
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:         Д2
mulW
ReluRelusplit:output:2*
T0*(
_output_shapes
:         Д2
Relui
mul_1MulSigmoid:y:0Relu:activations:0*
T0*(
_output_shapes
:         Д2
mul_1^
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:         Д2
add_1d
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:         Д2
	Sigmoid_2V
Relu_1Relu	add_1:z:0*
T0*(
_output_shapes
:         Д2
Relu_1m
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*(
_output_shapes
:         Д2
mul_2e
IdentityIdentity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:         Д2

Identityi

Identity_1Identity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:         Д2

Identity_1i

Identity_2Identity	add_1:z:0^NoOp*
T0*(
_output_shapes
:         Д2

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
A:         ]:         Д:         Д: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:         ]
 
_user_specified_nameinputs:PL
(
_output_shapes
:         Д
 
_user_specified_namestates:PL
(
_output_shapes
:         Д
 
_user_specified_namestates
▀
═
while_cond_39109589
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_39109589___redundant_placeholder06
2while_while_cond_39109589___redundant_placeholder16
2while_while_cond_39109589___redundant_placeholder26
2while_while_cond_39109589___redundant_placeholder3
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
@: : : : :         m:         m: ::::: 
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
:         m:-)
'
_output_shapes
:         m:

_output_shapes
: :

_output_shapes
:
╗
f
-__inference_dropout_22_layer_call_fn_39111694

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
:         Д* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_dropout_22_layer_call_and_return_conditional_losses_391099722
StatefulPartitionedCallА
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:         Д2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         Д22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:         Д
 
_user_specified_nameinputs
У
Й
J__inference_lstm_cell_22_layer_call_and_return_conditional_losses_39112473

inputs
states_0
states_11
matmul_readvariableop_resource:	]Р4
 matmul_1_readvariableop_resource:
ДР.
biasadd_readvariableop_resource:	Р
identity

identity_1

identity_2ИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpвMatMul_1/ReadVariableOpО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	]Р*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Р2
MatMulХ
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
ДР*
dtype02
MatMul_1/ReadVariableOp|
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Р2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:         Р2
addН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:Р*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Р2	
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
P:         Д:         Д:         Д:         Д*
	num_split2
split`
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:         Д2	
Sigmoidd
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:         Д2
	Sigmoid_1]
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:         Д2
mulW
ReluRelusplit:output:2*
T0*(
_output_shapes
:         Д2
Relui
mul_1MulSigmoid:y:0Relu:activations:0*
T0*(
_output_shapes
:         Д2
mul_1^
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:         Д2
add_1d
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:         Д2
	Sigmoid_2V
Relu_1Relu	add_1:z:0*
T0*(
_output_shapes
:         Д2
Relu_1m
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*(
_output_shapes
:         Д2
mul_2e
IdentityIdentity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:         Д2

Identityi

Identity_1Identity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:         Д2

Identity_1i

Identity_2Identity	add_1:z:0^NoOp*
T0*(
_output_shapes
:         Д2

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
A:         ]:         Д:         Д: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:         ]
 
_user_specified_nameinputs:RN
(
_output_shapes
:         Д
"
_user_specified_name
states/0:RN
(
_output_shapes
:         Д
"
_user_specified_name
states/1
¤%
є
while_body_39109021
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_01
while_lstm_cell_23_39109045_0:
Д┤0
while_lstm_cell_23_39109047_0:	m┤,
while_lstm_cell_23_39109049_0:	┤
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor/
while_lstm_cell_23_39109045:
Д┤.
while_lstm_cell_23_39109047:	m┤*
while_lstm_cell_23_39109049:	┤Ив*while/lstm_cell_23/StatefulPartitionedCall├
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    Д   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape╘
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:         Д*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemщ
*while/lstm_cell_23/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_23_39109045_0while_lstm_cell_23_39109047_0while_lstm_cell_23_39109049_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:         m:         m:         m*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_lstm_cell_23_layer_call_and_return_conditional_losses_391089432,
*while/lstm_cell_23/StatefulPartitionedCallў
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
while/Identity_2Ъ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3д
while/Identity_4Identity3while/lstm_cell_23/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:         m2
while/Identity_4д
while/Identity_5Identity3while/lstm_cell_23/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:         m2
while/Identity_5З

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
while_lstm_cell_23_39109045while_lstm_cell_23_39109045_0"<
while_lstm_cell_23_39109047while_lstm_cell_23_39109047_0"<
while_lstm_cell_23_39109049while_lstm_cell_23_39109049_0"0
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         m:         m: : : : : 2X
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
: :-)
'
_output_shapes
:         m:-)
'
_output_shapes
:         m:

_output_shapes
: :

_output_shapes
: 
╘
I
-__inference_dropout_23_layer_call_fn_39112364

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
:         m* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_dropout_23_layer_call_and_return_conditional_losses_391096872
PartitionedCallp
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:         m2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         m:S O
+
_output_shapes
:         m
 
_user_specified_nameinputs
╦F
О
E__inference_lstm_22_layer_call_and_return_conditional_losses_39108250

inputs(
lstm_cell_22_39108168:	]Р)
lstm_cell_22_39108170:
ДР$
lstm_cell_22_39108172:	Р
identityИв$lstm_cell_22/StatefulPartitionedCallвwhileD
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
B :Д2
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
B :Д2
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
:         Д2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :Д2
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
B :Д2
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
:         Д2	
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
$lstm_cell_22/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_22_39108168lstm_cell_22_39108170lstm_cell_22_39108172*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:         Д:         Д:         Д*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_lstm_cell_22_layer_call_and_return_conditional_losses_391081672&
$lstm_cell_22/StatefulPartitionedCallП
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    Д   2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_22_39108168lstm_cell_22_39108170lstm_cell_22_39108172*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :         Д:         Д: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_39108181*
condR
while_cond_39108180*M
output_shapes<
:: : : : :         Д:         Д: : : : : *
parallel_iterations 2
while╡
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    Д   22
0TensorArrayV2Stack/TensorListStack/element_shapeЄ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:                  Д*
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
:         Д*
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
!:                  Д2
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
!:                  Д2

Identity}
NoOpNoOp%^lstm_cell_22/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  ]: : : 2L
$lstm_cell_22/StatefulPartitionedCall$lstm_cell_22/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :                  ]
 
_user_specified_nameinputs
¤%
є
while_body_39108811
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_01
while_lstm_cell_23_39108835_0:
Д┤0
while_lstm_cell_23_39108837_0:	m┤,
while_lstm_cell_23_39108839_0:	┤
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor/
while_lstm_cell_23_39108835:
Д┤.
while_lstm_cell_23_39108837:	m┤*
while_lstm_cell_23_39108839:	┤Ив*while/lstm_cell_23/StatefulPartitionedCall├
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    Д   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape╘
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:         Д*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemщ
*while/lstm_cell_23/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_23_39108835_0while_lstm_cell_23_39108837_0while_lstm_cell_23_39108839_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:         m:         m:         m*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_lstm_cell_23_layer_call_and_return_conditional_losses_391087972,
*while/lstm_cell_23/StatefulPartitionedCallў
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
while/Identity_2Ъ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3д
while/Identity_4Identity3while/lstm_cell_23/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:         m2
while/Identity_4д
while/Identity_5Identity3while/lstm_cell_23/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:         m2
while/Identity_5З

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
while_lstm_cell_23_39108835while_lstm_cell_23_39108835_0"<
while_lstm_cell_23_39108837while_lstm_cell_23_39108837_0"<
while_lstm_cell_23_39108839while_lstm_cell_23_39108839_0"0
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         m:         m: : : : : 2X
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
: :-)
'
_output_shapes
:         m:-)
'
_output_shapes
:         m:

_output_shapes
: :

_output_shapes
: 
╠7
▌
$__inference__traced_restore_39112710
file_prefix2
 assignvariableop_dense_11_kernel:m.
 assignvariableop_1_dense_11_bias:A
.assignvariableop_2_lstm_22_lstm_cell_22_kernel:	]РL
8assignvariableop_3_lstm_22_lstm_cell_22_recurrent_kernel:
ДР;
,assignvariableop_4_lstm_22_lstm_cell_22_bias:	РB
.assignvariableop_5_lstm_23_lstm_cell_23_kernel:
Д┤K
8assignvariableop_6_lstm_23_lstm_cell_23_recurrent_kernel:	m┤;
,assignvariableop_7_lstm_23_lstm_cell_23_bias:	┤"
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
AssignVariableOpAssignVariableOp assignvariableop_dense_11_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1е
AssignVariableOp_1AssignVariableOp assignvariableop_1_dense_11_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2│
AssignVariableOp_2AssignVariableOp.assignvariableop_2_lstm_22_lstm_cell_22_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3╜
AssignVariableOp_3AssignVariableOp8assignvariableop_3_lstm_22_lstm_cell_22_recurrent_kernelIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4▒
AssignVariableOp_4AssignVariableOp,assignvariableop_4_lstm_22_lstm_cell_22_biasIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5│
AssignVariableOp_5AssignVariableOp.assignvariableop_5_lstm_23_lstm_cell_23_kernelIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6╜
AssignVariableOp_6AssignVariableOp8assignvariableop_6_lstm_23_lstm_cell_23_recurrent_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7▒
AssignVariableOp_7AssignVariableOp,assignvariableop_7_lstm_23_lstm_cell_23_biasIdentity_7:output:0"/device:CPU:0*
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
э[
Ю
E__inference_lstm_23_layer_call_and_return_conditional_losses_39109674

inputs?
+lstm_cell_23_matmul_readvariableop_resource:
Д┤@
-lstm_cell_23_matmul_1_readvariableop_resource:	m┤;
,lstm_cell_23_biasadd_readvariableop_resource:	┤
identityИв#lstm_cell_23/BiasAdd/ReadVariableOpв"lstm_cell_23/MatMul/ReadVariableOpв$lstm_cell_23/MatMul_1/ReadVariableOpвwhileD
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
value	B :m2
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
value	B :m2
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
:         m2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :m2
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
value	B :m2
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
:         m2	
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
:         Д2
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
valueB"    Д   27
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
:         Д*
shrink_axis_mask2
strided_slice_2╢
"lstm_cell_23/MatMul/ReadVariableOpReadVariableOp+lstm_cell_23_matmul_readvariableop_resource* 
_output_shapes
:
Д┤*
dtype02$
"lstm_cell_23/MatMul/ReadVariableOpн
lstm_cell_23/MatMulMatMulstrided_slice_2:output:0*lstm_cell_23/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ┤2
lstm_cell_23/MatMul╗
$lstm_cell_23/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_23_matmul_1_readvariableop_resource*
_output_shapes
:	m┤*
dtype02&
$lstm_cell_23/MatMul_1/ReadVariableOpй
lstm_cell_23/MatMul_1MatMulzeros:output:0,lstm_cell_23/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ┤2
lstm_cell_23/MatMul_1а
lstm_cell_23/addAddV2lstm_cell_23/MatMul:product:0lstm_cell_23/MatMul_1:product:0*
T0*(
_output_shapes
:         ┤2
lstm_cell_23/add┤
#lstm_cell_23/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_23_biasadd_readvariableop_resource*
_output_shapes	
:┤*
dtype02%
#lstm_cell_23/BiasAdd/ReadVariableOpн
lstm_cell_23/BiasAddBiasAddlstm_cell_23/add:z:0+lstm_cell_23/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ┤2
lstm_cell_23/BiasAdd~
lstm_cell_23/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_23/split/split_dimє
lstm_cell_23/splitSplit%lstm_cell_23/split/split_dim:output:0lstm_cell_23/BiasAdd:output:0*
T0*`
_output_shapesN
L:         m:         m:         m:         m*
	num_split2
lstm_cell_23/splitЖ
lstm_cell_23/SigmoidSigmoidlstm_cell_23/split:output:0*
T0*'
_output_shapes
:         m2
lstm_cell_23/SigmoidК
lstm_cell_23/Sigmoid_1Sigmoidlstm_cell_23/split:output:1*
T0*'
_output_shapes
:         m2
lstm_cell_23/Sigmoid_1Л
lstm_cell_23/mulMullstm_cell_23/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:         m2
lstm_cell_23/mul}
lstm_cell_23/ReluRelulstm_cell_23/split:output:2*
T0*'
_output_shapes
:         m2
lstm_cell_23/ReluЬ
lstm_cell_23/mul_1Mullstm_cell_23/Sigmoid:y:0lstm_cell_23/Relu:activations:0*
T0*'
_output_shapes
:         m2
lstm_cell_23/mul_1С
lstm_cell_23/add_1AddV2lstm_cell_23/mul:z:0lstm_cell_23/mul_1:z:0*
T0*'
_output_shapes
:         m2
lstm_cell_23/add_1К
lstm_cell_23/Sigmoid_2Sigmoidlstm_cell_23/split:output:3*
T0*'
_output_shapes
:         m2
lstm_cell_23/Sigmoid_2|
lstm_cell_23/Relu_1Relulstm_cell_23/add_1:z:0*
T0*'
_output_shapes
:         m2
lstm_cell_23/Relu_1а
lstm_cell_23/mul_2Mullstm_cell_23/Sigmoid_2:y:0!lstm_cell_23/Relu_1:activations:0*
T0*'
_output_shapes
:         m2
lstm_cell_23/mul_2П
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    m   2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_23_matmul_readvariableop_resource-lstm_cell_23_matmul_1_readvariableop_resource,lstm_cell_23_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         m:         m: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_39109590*
condR
while_cond_39109589*K
output_shapes:
8: : : : :         m:         m: : : : : *
parallel_iterations 2
while╡
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    m   22
0TensorArrayV2Stack/TensorListStack/element_shapeш
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:         m*
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
:         m*
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
:         m2
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
:         m2

Identity╚
NoOpNoOp$^lstm_cell_23/BiasAdd/ReadVariableOp#^lstm_cell_23/MatMul/ReadVariableOp%^lstm_cell_23/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:         Д: : : 2J
#lstm_cell_23/BiasAdd/ReadVariableOp#lstm_cell_23/BiasAdd/ReadVariableOp2H
"lstm_cell_23/MatMul/ReadVariableOp"lstm_cell_23/MatMul/ReadVariableOp2L
$lstm_cell_23/MatMul_1/ReadVariableOp$lstm_cell_23/MatMul_1/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:         Д
 
_user_specified_nameinputs
▀
═
while_cond_39111911
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_39111911___redundant_placeholder06
2while_while_cond_39111911___redundant_placeholder16
2while_while_cond_39111911___redundant_placeholder26
2while_while_cond_39111911___redundant_placeholder3
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
@: : : : :         m:         m: ::::: 
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
:         m:-)
'
_output_shapes
:         m:

_output_shapes
: :

_output_shapes
:
╣
Ў
K__inference_sequential_11_layer_call_and_return_conditional_losses_39110261
lstm_22_input#
lstm_22_39110239:	]Р$
lstm_22_39110241:
ДР
lstm_22_39110243:	Р$
lstm_23_39110247:
Д┤#
lstm_23_39110249:	m┤
lstm_23_39110251:	┤#
dense_11_39110255:m
dense_11_39110257:
identityИв dense_11/StatefulPartitionedCallвlstm_22/StatefulPartitionedCallвlstm_23/StatefulPartitionedCall╡
lstm_22/StatefulPartitionedCallStatefulPartitionedCalllstm_22_inputlstm_22_39110239lstm_22_39110241lstm_22_39110243*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         Д*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_lstm_22_layer_call_and_return_conditional_losses_391095092!
lstm_22/StatefulPartitionedCallГ
dropout_22/PartitionedCallPartitionedCall(lstm_22/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         Д* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_dropout_22_layer_call_and_return_conditional_losses_391095222
dropout_22/PartitionedCall╩
lstm_23/StatefulPartitionedCallStatefulPartitionedCall#dropout_22/PartitionedCall:output:0lstm_23_39110247lstm_23_39110249lstm_23_39110251*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         m*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_lstm_23_layer_call_and_return_conditional_losses_391096742!
lstm_23/StatefulPartitionedCallВ
dropout_23/PartitionedCallPartitionedCall(lstm_23/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         m* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_dropout_23_layer_call_and_return_conditional_losses_391096872
dropout_23/PartitionedCall╗
 dense_11/StatefulPartitionedCallStatefulPartitionedCall#dropout_23/PartitionedCall:output:0dense_11_39110255dense_11_39110257*
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
F__inference_dense_11_layer_call_and_return_conditional_losses_391097202"
 dense_11/StatefulPartitionedCallИ
IdentityIdentity)dense_11/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:         2

Identity╡
NoOpNoOp!^dense_11/StatefulPartitionedCall ^lstm_22/StatefulPartitionedCall ^lstm_23/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':         ]: : : : : : : : 2D
 dense_11/StatefulPartitionedCall dense_11/StatefulPartitionedCall2B
lstm_22/StatefulPartitionedCalllstm_22/StatefulPartitionedCall2B
lstm_23/StatefulPartitionedCalllstm_23/StatefulPartitionedCall:Z V
+
_output_shapes
:         ]
'
_user_specified_namelstm_22_input
Ю?
╘
while_body_39111912
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
3while_lstm_cell_23_matmul_readvariableop_resource_0:
Д┤H
5while_lstm_cell_23_matmul_1_readvariableop_resource_0:	m┤C
4while_lstm_cell_23_biasadd_readvariableop_resource_0:	┤
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
1while_lstm_cell_23_matmul_readvariableop_resource:
Д┤F
3while_lstm_cell_23_matmul_1_readvariableop_resource:	m┤A
2while_lstm_cell_23_biasadd_readvariableop_resource:	┤Ив)while/lstm_cell_23/BiasAdd/ReadVariableOpв(while/lstm_cell_23/MatMul/ReadVariableOpв*while/lstm_cell_23/MatMul_1/ReadVariableOp├
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    Д   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape╘
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:         Д*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem╩
(while/lstm_cell_23/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_23_matmul_readvariableop_resource_0* 
_output_shapes
:
Д┤*
dtype02*
(while/lstm_cell_23/MatMul/ReadVariableOp╫
while/lstm_cell_23/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_23/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ┤2
while/lstm_cell_23/MatMul╧
*while/lstm_cell_23/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_23_matmul_1_readvariableop_resource_0*
_output_shapes
:	m┤*
dtype02,
*while/lstm_cell_23/MatMul_1/ReadVariableOp└
while/lstm_cell_23/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_23/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ┤2
while/lstm_cell_23/MatMul_1╕
while/lstm_cell_23/addAddV2#while/lstm_cell_23/MatMul:product:0%while/lstm_cell_23/MatMul_1:product:0*
T0*(
_output_shapes
:         ┤2
while/lstm_cell_23/add╚
)while/lstm_cell_23/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_23_biasadd_readvariableop_resource_0*
_output_shapes	
:┤*
dtype02+
)while/lstm_cell_23/BiasAdd/ReadVariableOp┼
while/lstm_cell_23/BiasAddBiasAddwhile/lstm_cell_23/add:z:01while/lstm_cell_23/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ┤2
while/lstm_cell_23/BiasAddК
"while/lstm_cell_23/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_23/split/split_dimЛ
while/lstm_cell_23/splitSplit+while/lstm_cell_23/split/split_dim:output:0#while/lstm_cell_23/BiasAdd:output:0*
T0*`
_output_shapesN
L:         m:         m:         m:         m*
	num_split2
while/lstm_cell_23/splitШ
while/lstm_cell_23/SigmoidSigmoid!while/lstm_cell_23/split:output:0*
T0*'
_output_shapes
:         m2
while/lstm_cell_23/SigmoidЬ
while/lstm_cell_23/Sigmoid_1Sigmoid!while/lstm_cell_23/split:output:1*
T0*'
_output_shapes
:         m2
while/lstm_cell_23/Sigmoid_1а
while/lstm_cell_23/mulMul while/lstm_cell_23/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:         m2
while/lstm_cell_23/mulП
while/lstm_cell_23/ReluRelu!while/lstm_cell_23/split:output:2*
T0*'
_output_shapes
:         m2
while/lstm_cell_23/Relu┤
while/lstm_cell_23/mul_1Mulwhile/lstm_cell_23/Sigmoid:y:0%while/lstm_cell_23/Relu:activations:0*
T0*'
_output_shapes
:         m2
while/lstm_cell_23/mul_1й
while/lstm_cell_23/add_1AddV2while/lstm_cell_23/mul:z:0while/lstm_cell_23/mul_1:z:0*
T0*'
_output_shapes
:         m2
while/lstm_cell_23/add_1Ь
while/lstm_cell_23/Sigmoid_2Sigmoid!while/lstm_cell_23/split:output:3*
T0*'
_output_shapes
:         m2
while/lstm_cell_23/Sigmoid_2О
while/lstm_cell_23/Relu_1Reluwhile/lstm_cell_23/add_1:z:0*
T0*'
_output_shapes
:         m2
while/lstm_cell_23/Relu_1╕
while/lstm_cell_23/mul_2Mul while/lstm_cell_23/Sigmoid_2:y:0'while/lstm_cell_23/Relu_1:activations:0*
T0*'
_output_shapes
:         m2
while/lstm_cell_23/mul_2р
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
while/Identity_2Ъ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3Н
while/Identity_4Identitywhile/lstm_cell_23/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:         m2
while/Identity_4Н
while/Identity_5Identitywhile/lstm_cell_23/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:         m2
while/Identity_5▐

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
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         m:         m: : : : : 2V
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
: :-)
'
_output_shapes
:         m:-)
'
_output_shapes
:         m:

_output_shapes
: :

_output_shapes
: 
Ї%
р
!__inference__traced_save_39112664
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
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_dense_11_kernel_read_readvariableop(savev2_dense_11_bias_read_readvariableop6savev2_lstm_22_lstm_cell_22_kernel_read_readvariableop@savev2_lstm_22_lstm_cell_22_recurrent_kernel_read_readvariableop4savev2_lstm_22_lstm_cell_22_bias_read_readvariableop6savev2_lstm_23_lstm_cell_23_kernel_read_readvariableop@savev2_lstm_23_lstm_cell_23_recurrent_kernel_read_readvariableop4savev2_lstm_23_lstm_cell_23_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableopsavev2_const"/device:CPU:0*
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
X: :m::	]Р:
ДР:Р:
Д┤:	m┤:┤: : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:m: 

_output_shapes
::%!

_output_shapes
:	]Р:&"
 
_output_shapes
:
ДР:!

_output_shapes	
:Р:&"
 
_output_shapes
:
Д┤:%!

_output_shapes
:	m┤:!

_output_shapes	
:┤:	
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
▀
═
while_cond_39112213
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_39112213___redundant_placeholder06
2while_while_cond_39112213___redundant_placeholder16
2while_while_cond_39112213___redundant_placeholder26
2while_while_cond_39112213___redundant_placeholder3
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
@: : : : :         m:         m: ::::: 
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
:         m:-)
'
_output_shapes
:         m:

_output_shapes
: :

_output_shapes
:
├\
а
E__inference_lstm_22_layer_call_and_return_conditional_losses_39111170
inputs_0>
+lstm_cell_22_matmul_readvariableop_resource:	]РA
-lstm_cell_22_matmul_1_readvariableop_resource:
ДР;
,lstm_cell_22_biasadd_readvariableop_resource:	Р
identityИв#lstm_cell_22/BiasAdd/ReadVariableOpв"lstm_cell_22/MatMul/ReadVariableOpв$lstm_cell_22/MatMul_1/ReadVariableOpвwhileF
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
B :Д2
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
B :Д2
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
:         Д2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :Д2
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
B :Д2
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
:         Д2	
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
"lstm_cell_22/MatMul/ReadVariableOpReadVariableOp+lstm_cell_22_matmul_readvariableop_resource*
_output_shapes
:	]Р*
dtype02$
"lstm_cell_22/MatMul/ReadVariableOpн
lstm_cell_22/MatMulMatMulstrided_slice_2:output:0*lstm_cell_22/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Р2
lstm_cell_22/MatMul╝
$lstm_cell_22/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_22_matmul_1_readvariableop_resource* 
_output_shapes
:
ДР*
dtype02&
$lstm_cell_22/MatMul_1/ReadVariableOpй
lstm_cell_22/MatMul_1MatMulzeros:output:0,lstm_cell_22/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Р2
lstm_cell_22/MatMul_1а
lstm_cell_22/addAddV2lstm_cell_22/MatMul:product:0lstm_cell_22/MatMul_1:product:0*
T0*(
_output_shapes
:         Р2
lstm_cell_22/add┤
#lstm_cell_22/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_22_biasadd_readvariableop_resource*
_output_shapes	
:Р*
dtype02%
#lstm_cell_22/BiasAdd/ReadVariableOpн
lstm_cell_22/BiasAddBiasAddlstm_cell_22/add:z:0+lstm_cell_22/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Р2
lstm_cell_22/BiasAdd~
lstm_cell_22/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_22/split/split_dimў
lstm_cell_22/splitSplit%lstm_cell_22/split/split_dim:output:0lstm_cell_22/BiasAdd:output:0*
T0*d
_output_shapesR
P:         Д:         Д:         Д:         Д*
	num_split2
lstm_cell_22/splitЗ
lstm_cell_22/SigmoidSigmoidlstm_cell_22/split:output:0*
T0*(
_output_shapes
:         Д2
lstm_cell_22/SigmoidЛ
lstm_cell_22/Sigmoid_1Sigmoidlstm_cell_22/split:output:1*
T0*(
_output_shapes
:         Д2
lstm_cell_22/Sigmoid_1М
lstm_cell_22/mulMullstm_cell_22/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:         Д2
lstm_cell_22/mul~
lstm_cell_22/ReluRelulstm_cell_22/split:output:2*
T0*(
_output_shapes
:         Д2
lstm_cell_22/ReluЭ
lstm_cell_22/mul_1Mullstm_cell_22/Sigmoid:y:0lstm_cell_22/Relu:activations:0*
T0*(
_output_shapes
:         Д2
lstm_cell_22/mul_1Т
lstm_cell_22/add_1AddV2lstm_cell_22/mul:z:0lstm_cell_22/mul_1:z:0*
T0*(
_output_shapes
:         Д2
lstm_cell_22/add_1Л
lstm_cell_22/Sigmoid_2Sigmoidlstm_cell_22/split:output:3*
T0*(
_output_shapes
:         Д2
lstm_cell_22/Sigmoid_2}
lstm_cell_22/Relu_1Relulstm_cell_22/add_1:z:0*
T0*(
_output_shapes
:         Д2
lstm_cell_22/Relu_1б
lstm_cell_22/mul_2Mullstm_cell_22/Sigmoid_2:y:0!lstm_cell_22/Relu_1:activations:0*
T0*(
_output_shapes
:         Д2
lstm_cell_22/mul_2П
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    Д   2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_22_matmul_readvariableop_resource-lstm_cell_22_matmul_1_readvariableop_resource,lstm_cell_22_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :         Д:         Д: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_39111086*
condR
while_cond_39111085*M
output_shapes<
:: : : : :         Д:         Д: : : : : *
parallel_iterations 2
while╡
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    Д   22
0TensorArrayV2Stack/TensorListStack/element_shapeЄ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:                  Д*
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
:         Д*
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
!:                  Д2
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
!:                  Д2

Identity╚
NoOpNoOp$^lstm_cell_22/BiasAdd/ReadVariableOp#^lstm_cell_22/MatMul/ReadVariableOp%^lstm_cell_22/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  ]: : : 2J
#lstm_cell_22/BiasAdd/ReadVariableOp#lstm_cell_22/BiasAdd/ReadVariableOp2H
"lstm_cell_22/MatMul/ReadVariableOp"lstm_cell_22/MatMul/ReadVariableOp2L
$lstm_cell_22/MatMul_1/ReadVariableOp$lstm_cell_22/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :                  ]
"
_user_specified_name
inputs/0
Б
Й
J__inference_lstm_cell_23_layer_call_and_return_conditional_losses_39112539

inputs
states_0
states_12
matmul_readvariableop_resource:
Д┤3
 matmul_1_readvariableop_resource:	m┤.
biasadd_readvariableop_resource:	┤
identity

identity_1

identity_2ИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpвMatMul_1/ReadVariableOpП
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
Д┤*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ┤2
MatMulФ
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	m┤*
dtype02
MatMul_1/ReadVariableOp|
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ┤2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:         ┤2
addН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:┤*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ┤2	
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
L:         m:         m:         m:         m*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:         m2	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:         m2
	Sigmoid_1\
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:         m2
mulV
ReluRelusplit:output:2*
T0*'
_output_shapes
:         m2
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:         m2
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:         m2
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:         m2
	Sigmoid_2U
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:         m2
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:         m2
mul_2d
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:         m2

Identityh

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:         m2

Identity_1h

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:         m2

Identity_2Щ
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:         Д:         m:         m: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:P L
(
_output_shapes
:         Д
 
_user_specified_nameinputs:QM
'
_output_shapes
:         m
"
_user_specified_name
states/0:QM
'
_output_shapes
:         m
"
_user_specified_name
states/1
Е&
є
while_body_39108181
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_00
while_lstm_cell_22_39108205_0:	]Р1
while_lstm_cell_22_39108207_0:
ДР,
while_lstm_cell_22_39108209_0:	Р
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor.
while_lstm_cell_22_39108205:	]Р/
while_lstm_cell_22_39108207:
ДР*
while_lstm_cell_22_39108209:	РИв*while/lstm_cell_22/StatefulPartitionedCall├
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
*while/lstm_cell_22/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_22_39108205_0while_lstm_cell_22_39108207_0while_lstm_cell_22_39108209_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:         Д:         Д:         Д*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_lstm_cell_22_layer_call_and_return_conditional_losses_391081672,
*while/lstm_cell_22/StatefulPartitionedCallў
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
while/Identity_2Ъ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3е
while/Identity_4Identity3while/lstm_cell_22/StatefulPartitionedCall:output:1^while/NoOp*
T0*(
_output_shapes
:         Д2
while/Identity_4е
while/Identity_5Identity3while/lstm_cell_22/StatefulPartitionedCall:output:2^while/NoOp*
T0*(
_output_shapes
:         Д2
while/Identity_5З

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
while_lstm_cell_22_39108205while_lstm_cell_22_39108205_0"<
while_lstm_cell_22_39108207while_lstm_cell_22_39108207_0"<
while_lstm_cell_22_39108209while_lstm_cell_22_39108209_0"0
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :         Д:         Д: : : : : 2X
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
:         Д:.*
(
_output_shapes
:         Д:

_output_shapes
: :

_output_shapes
: 
░?
╘
while_body_39110055
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_22_matmul_readvariableop_resource_0:	]РI
5while_lstm_cell_22_matmul_1_readvariableop_resource_0:
ДРC
4while_lstm_cell_22_biasadd_readvariableop_resource_0:	Р
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_22_matmul_readvariableop_resource:	]РG
3while_lstm_cell_22_matmul_1_readvariableop_resource:
ДРA
2while_lstm_cell_22_biasadd_readvariableop_resource:	РИв)while/lstm_cell_22/BiasAdd/ReadVariableOpв(while/lstm_cell_22/MatMul/ReadVariableOpв*while/lstm_cell_22/MatMul_1/ReadVariableOp├
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
(while/lstm_cell_22/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_22_matmul_readvariableop_resource_0*
_output_shapes
:	]Р*
dtype02*
(while/lstm_cell_22/MatMul/ReadVariableOp╫
while/lstm_cell_22/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_22/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Р2
while/lstm_cell_22/MatMul╨
*while/lstm_cell_22/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_22_matmul_1_readvariableop_resource_0* 
_output_shapes
:
ДР*
dtype02,
*while/lstm_cell_22/MatMul_1/ReadVariableOp└
while/lstm_cell_22/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_22/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Р2
while/lstm_cell_22/MatMul_1╕
while/lstm_cell_22/addAddV2#while/lstm_cell_22/MatMul:product:0%while/lstm_cell_22/MatMul_1:product:0*
T0*(
_output_shapes
:         Р2
while/lstm_cell_22/add╚
)while/lstm_cell_22/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_22_biasadd_readvariableop_resource_0*
_output_shapes	
:Р*
dtype02+
)while/lstm_cell_22/BiasAdd/ReadVariableOp┼
while/lstm_cell_22/BiasAddBiasAddwhile/lstm_cell_22/add:z:01while/lstm_cell_22/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Р2
while/lstm_cell_22/BiasAddК
"while/lstm_cell_22/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_22/split/split_dimП
while/lstm_cell_22/splitSplit+while/lstm_cell_22/split/split_dim:output:0#while/lstm_cell_22/BiasAdd:output:0*
T0*d
_output_shapesR
P:         Д:         Д:         Д:         Д*
	num_split2
while/lstm_cell_22/splitЩ
while/lstm_cell_22/SigmoidSigmoid!while/lstm_cell_22/split:output:0*
T0*(
_output_shapes
:         Д2
while/lstm_cell_22/SigmoidЭ
while/lstm_cell_22/Sigmoid_1Sigmoid!while/lstm_cell_22/split:output:1*
T0*(
_output_shapes
:         Д2
while/lstm_cell_22/Sigmoid_1б
while/lstm_cell_22/mulMul while/lstm_cell_22/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:         Д2
while/lstm_cell_22/mulР
while/lstm_cell_22/ReluRelu!while/lstm_cell_22/split:output:2*
T0*(
_output_shapes
:         Д2
while/lstm_cell_22/Relu╡
while/lstm_cell_22/mul_1Mulwhile/lstm_cell_22/Sigmoid:y:0%while/lstm_cell_22/Relu:activations:0*
T0*(
_output_shapes
:         Д2
while/lstm_cell_22/mul_1к
while/lstm_cell_22/add_1AddV2while/lstm_cell_22/mul:z:0while/lstm_cell_22/mul_1:z:0*
T0*(
_output_shapes
:         Д2
while/lstm_cell_22/add_1Э
while/lstm_cell_22/Sigmoid_2Sigmoid!while/lstm_cell_22/split:output:3*
T0*(
_output_shapes
:         Д2
while/lstm_cell_22/Sigmoid_2П
while/lstm_cell_22/Relu_1Reluwhile/lstm_cell_22/add_1:z:0*
T0*(
_output_shapes
:         Д2
while/lstm_cell_22/Relu_1╣
while/lstm_cell_22/mul_2Mul while/lstm_cell_22/Sigmoid_2:y:0'while/lstm_cell_22/Relu_1:activations:0*
T0*(
_output_shapes
:         Д2
while/lstm_cell_22/mul_2р
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
while/Identity_2Ъ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3О
while/Identity_4Identitywhile/lstm_cell_22/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:         Д2
while/Identity_4О
while/Identity_5Identitywhile/lstm_cell_22/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:         Д2
while/Identity_5▐

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
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :         Д:         Д: : : : : 2V
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
:         Д:.*
(
_output_shapes
:         Д:

_output_shapes
: :

_output_shapes
: 
╘

э
lstm_22_while_cond_39110375,
(lstm_22_while_lstm_22_while_loop_counter2
.lstm_22_while_lstm_22_while_maximum_iterations
lstm_22_while_placeholder
lstm_22_while_placeholder_1
lstm_22_while_placeholder_2
lstm_22_while_placeholder_3.
*lstm_22_while_less_lstm_22_strided_slice_1F
Blstm_22_while_lstm_22_while_cond_39110375___redundant_placeholder0F
Blstm_22_while_lstm_22_while_cond_39110375___redundant_placeholder1F
Blstm_22_while_lstm_22_while_cond_39110375___redundant_placeholder2F
Blstm_22_while_lstm_22_while_cond_39110375___redundant_placeholder3
lstm_22_while_identity
Ш
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
B: : : : :         Д:         Д: ::::: 
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
:         Д:.*
(
_output_shapes
:         Д:

_output_shapes
: :

_output_shapes
:
э[
Ю
E__inference_lstm_23_layer_call_and_return_conditional_losses_39109943

inputs?
+lstm_cell_23_matmul_readvariableop_resource:
Д┤@
-lstm_cell_23_matmul_1_readvariableop_resource:	m┤;
,lstm_cell_23_biasadd_readvariableop_resource:	┤
identityИв#lstm_cell_23/BiasAdd/ReadVariableOpв"lstm_cell_23/MatMul/ReadVariableOpв$lstm_cell_23/MatMul_1/ReadVariableOpвwhileD
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
value	B :m2
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
value	B :m2
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
:         m2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :m2
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
value	B :m2
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
:         m2	
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
:         Д2
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
valueB"    Д   27
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
:         Д*
shrink_axis_mask2
strided_slice_2╢
"lstm_cell_23/MatMul/ReadVariableOpReadVariableOp+lstm_cell_23_matmul_readvariableop_resource* 
_output_shapes
:
Д┤*
dtype02$
"lstm_cell_23/MatMul/ReadVariableOpн
lstm_cell_23/MatMulMatMulstrided_slice_2:output:0*lstm_cell_23/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ┤2
lstm_cell_23/MatMul╗
$lstm_cell_23/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_23_matmul_1_readvariableop_resource*
_output_shapes
:	m┤*
dtype02&
$lstm_cell_23/MatMul_1/ReadVariableOpй
lstm_cell_23/MatMul_1MatMulzeros:output:0,lstm_cell_23/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ┤2
lstm_cell_23/MatMul_1а
lstm_cell_23/addAddV2lstm_cell_23/MatMul:product:0lstm_cell_23/MatMul_1:product:0*
T0*(
_output_shapes
:         ┤2
lstm_cell_23/add┤
#lstm_cell_23/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_23_biasadd_readvariableop_resource*
_output_shapes	
:┤*
dtype02%
#lstm_cell_23/BiasAdd/ReadVariableOpн
lstm_cell_23/BiasAddBiasAddlstm_cell_23/add:z:0+lstm_cell_23/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ┤2
lstm_cell_23/BiasAdd~
lstm_cell_23/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_23/split/split_dimє
lstm_cell_23/splitSplit%lstm_cell_23/split/split_dim:output:0lstm_cell_23/BiasAdd:output:0*
T0*`
_output_shapesN
L:         m:         m:         m:         m*
	num_split2
lstm_cell_23/splitЖ
lstm_cell_23/SigmoidSigmoidlstm_cell_23/split:output:0*
T0*'
_output_shapes
:         m2
lstm_cell_23/SigmoidК
lstm_cell_23/Sigmoid_1Sigmoidlstm_cell_23/split:output:1*
T0*'
_output_shapes
:         m2
lstm_cell_23/Sigmoid_1Л
lstm_cell_23/mulMullstm_cell_23/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:         m2
lstm_cell_23/mul}
lstm_cell_23/ReluRelulstm_cell_23/split:output:2*
T0*'
_output_shapes
:         m2
lstm_cell_23/ReluЬ
lstm_cell_23/mul_1Mullstm_cell_23/Sigmoid:y:0lstm_cell_23/Relu:activations:0*
T0*'
_output_shapes
:         m2
lstm_cell_23/mul_1С
lstm_cell_23/add_1AddV2lstm_cell_23/mul:z:0lstm_cell_23/mul_1:z:0*
T0*'
_output_shapes
:         m2
lstm_cell_23/add_1К
lstm_cell_23/Sigmoid_2Sigmoidlstm_cell_23/split:output:3*
T0*'
_output_shapes
:         m2
lstm_cell_23/Sigmoid_2|
lstm_cell_23/Relu_1Relulstm_cell_23/add_1:z:0*
T0*'
_output_shapes
:         m2
lstm_cell_23/Relu_1а
lstm_cell_23/mul_2Mullstm_cell_23/Sigmoid_2:y:0!lstm_cell_23/Relu_1:activations:0*
T0*'
_output_shapes
:         m2
lstm_cell_23/mul_2П
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    m   2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_23_matmul_readvariableop_resource-lstm_cell_23_matmul_1_readvariableop_resource,lstm_cell_23_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         m:         m: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_39109859*
condR
while_cond_39109858*K
output_shapes:
8: : : : :         m:         m: : : : : *
parallel_iterations 2
while╡
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    m   22
0TensorArrayV2Stack/TensorListStack/element_shapeш
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:         m*
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
:         m*
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
:         m2
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
:         m2

Identity╚
NoOpNoOp$^lstm_cell_23/BiasAdd/ReadVariableOp#^lstm_cell_23/MatMul/ReadVariableOp%^lstm_cell_23/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:         Д: : : 2J
#lstm_cell_23/BiasAdd/ReadVariableOp#lstm_cell_23/BiasAdd/ReadVariableOp2H
"lstm_cell_23/MatMul/ReadVariableOp"lstm_cell_23/MatMul/ReadVariableOp2L
$lstm_cell_23/MatMul_1/ReadVariableOp$lstm_cell_23/MatMul_1/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:         Д
 
_user_specified_nameinputs
хJ
╘

lstm_22_while_body_39110703,
(lstm_22_while_lstm_22_while_loop_counter2
.lstm_22_while_lstm_22_while_maximum_iterations
lstm_22_while_placeholder
lstm_22_while_placeholder_1
lstm_22_while_placeholder_2
lstm_22_while_placeholder_3+
'lstm_22_while_lstm_22_strided_slice_1_0g
clstm_22_while_tensorarrayv2read_tensorlistgetitem_lstm_22_tensorarrayunstack_tensorlistfromtensor_0N
;lstm_22_while_lstm_cell_22_matmul_readvariableop_resource_0:	]РQ
=lstm_22_while_lstm_cell_22_matmul_1_readvariableop_resource_0:
ДРK
<lstm_22_while_lstm_cell_22_biasadd_readvariableop_resource_0:	Р
lstm_22_while_identity
lstm_22_while_identity_1
lstm_22_while_identity_2
lstm_22_while_identity_3
lstm_22_while_identity_4
lstm_22_while_identity_5)
%lstm_22_while_lstm_22_strided_slice_1e
alstm_22_while_tensorarrayv2read_tensorlistgetitem_lstm_22_tensorarrayunstack_tensorlistfromtensorL
9lstm_22_while_lstm_cell_22_matmul_readvariableop_resource:	]РO
;lstm_22_while_lstm_cell_22_matmul_1_readvariableop_resource:
ДРI
:lstm_22_while_lstm_cell_22_biasadd_readvariableop_resource:	РИв1lstm_22/while/lstm_cell_22/BiasAdd/ReadVariableOpв0lstm_22/while/lstm_cell_22/MatMul/ReadVariableOpв2lstm_22/while/lstm_cell_22/MatMul_1/ReadVariableOp╙
?lstm_22/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ]   2A
?lstm_22/while/TensorArrayV2Read/TensorListGetItem/element_shapeГ
1lstm_22/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemclstm_22_while_tensorarrayv2read_tensorlistgetitem_lstm_22_tensorarrayunstack_tensorlistfromtensor_0lstm_22_while_placeholderHlstm_22/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         ]*
element_dtype023
1lstm_22/while/TensorArrayV2Read/TensorListGetItemс
0lstm_22/while/lstm_cell_22/MatMul/ReadVariableOpReadVariableOp;lstm_22_while_lstm_cell_22_matmul_readvariableop_resource_0*
_output_shapes
:	]Р*
dtype022
0lstm_22/while/lstm_cell_22/MatMul/ReadVariableOpў
!lstm_22/while/lstm_cell_22/MatMulMatMul8lstm_22/while/TensorArrayV2Read/TensorListGetItem:item:08lstm_22/while/lstm_cell_22/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Р2#
!lstm_22/while/lstm_cell_22/MatMulш
2lstm_22/while/lstm_cell_22/MatMul_1/ReadVariableOpReadVariableOp=lstm_22_while_lstm_cell_22_matmul_1_readvariableop_resource_0* 
_output_shapes
:
ДР*
dtype024
2lstm_22/while/lstm_cell_22/MatMul_1/ReadVariableOpр
#lstm_22/while/lstm_cell_22/MatMul_1MatMullstm_22_while_placeholder_2:lstm_22/while/lstm_cell_22/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Р2%
#lstm_22/while/lstm_cell_22/MatMul_1╪
lstm_22/while/lstm_cell_22/addAddV2+lstm_22/while/lstm_cell_22/MatMul:product:0-lstm_22/while/lstm_cell_22/MatMul_1:product:0*
T0*(
_output_shapes
:         Р2 
lstm_22/while/lstm_cell_22/addр
1lstm_22/while/lstm_cell_22/BiasAdd/ReadVariableOpReadVariableOp<lstm_22_while_lstm_cell_22_biasadd_readvariableop_resource_0*
_output_shapes	
:Р*
dtype023
1lstm_22/while/lstm_cell_22/BiasAdd/ReadVariableOpх
"lstm_22/while/lstm_cell_22/BiasAddBiasAdd"lstm_22/while/lstm_cell_22/add:z:09lstm_22/while/lstm_cell_22/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Р2$
"lstm_22/while/lstm_cell_22/BiasAddЪ
*lstm_22/while/lstm_cell_22/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2,
*lstm_22/while/lstm_cell_22/split/split_dimп
 lstm_22/while/lstm_cell_22/splitSplit3lstm_22/while/lstm_cell_22/split/split_dim:output:0+lstm_22/while/lstm_cell_22/BiasAdd:output:0*
T0*d
_output_shapesR
P:         Д:         Д:         Д:         Д*
	num_split2"
 lstm_22/while/lstm_cell_22/split▒
"lstm_22/while/lstm_cell_22/SigmoidSigmoid)lstm_22/while/lstm_cell_22/split:output:0*
T0*(
_output_shapes
:         Д2$
"lstm_22/while/lstm_cell_22/Sigmoid╡
$lstm_22/while/lstm_cell_22/Sigmoid_1Sigmoid)lstm_22/while/lstm_cell_22/split:output:1*
T0*(
_output_shapes
:         Д2&
$lstm_22/while/lstm_cell_22/Sigmoid_1┴
lstm_22/while/lstm_cell_22/mulMul(lstm_22/while/lstm_cell_22/Sigmoid_1:y:0lstm_22_while_placeholder_3*
T0*(
_output_shapes
:         Д2 
lstm_22/while/lstm_cell_22/mulи
lstm_22/while/lstm_cell_22/ReluRelu)lstm_22/while/lstm_cell_22/split:output:2*
T0*(
_output_shapes
:         Д2!
lstm_22/while/lstm_cell_22/Relu╒
 lstm_22/while/lstm_cell_22/mul_1Mul&lstm_22/while/lstm_cell_22/Sigmoid:y:0-lstm_22/while/lstm_cell_22/Relu:activations:0*
T0*(
_output_shapes
:         Д2"
 lstm_22/while/lstm_cell_22/mul_1╩
 lstm_22/while/lstm_cell_22/add_1AddV2"lstm_22/while/lstm_cell_22/mul:z:0$lstm_22/while/lstm_cell_22/mul_1:z:0*
T0*(
_output_shapes
:         Д2"
 lstm_22/while/lstm_cell_22/add_1╡
$lstm_22/while/lstm_cell_22/Sigmoid_2Sigmoid)lstm_22/while/lstm_cell_22/split:output:3*
T0*(
_output_shapes
:         Д2&
$lstm_22/while/lstm_cell_22/Sigmoid_2з
!lstm_22/while/lstm_cell_22/Relu_1Relu$lstm_22/while/lstm_cell_22/add_1:z:0*
T0*(
_output_shapes
:         Д2#
!lstm_22/while/lstm_cell_22/Relu_1┘
 lstm_22/while/lstm_cell_22/mul_2Mul(lstm_22/while/lstm_cell_22/Sigmoid_2:y:0/lstm_22/while/lstm_cell_22/Relu_1:activations:0*
T0*(
_output_shapes
:         Д2"
 lstm_22/while/lstm_cell_22/mul_2И
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
lstm_22/while/add/yЙ
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
lstm_22/while/add_1/yЮ
lstm_22/while/add_1AddV2(lstm_22_while_lstm_22_while_loop_counterlstm_22/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lstm_22/while/add_1Л
lstm_22/while/IdentityIdentitylstm_22/while/add_1:z:0^lstm_22/while/NoOp*
T0*
_output_shapes
: 2
lstm_22/while/Identityж
lstm_22/while/Identity_1Identity.lstm_22_while_lstm_22_while_maximum_iterations^lstm_22/while/NoOp*
T0*
_output_shapes
: 2
lstm_22/while/Identity_1Н
lstm_22/while/Identity_2Identitylstm_22/while/add:z:0^lstm_22/while/NoOp*
T0*
_output_shapes
: 2
lstm_22/while/Identity_2║
lstm_22/while/Identity_3IdentityBlstm_22/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_22/while/NoOp*
T0*
_output_shapes
: 2
lstm_22/while/Identity_3о
lstm_22/while/Identity_4Identity$lstm_22/while/lstm_cell_22/mul_2:z:0^lstm_22/while/NoOp*
T0*(
_output_shapes
:         Д2
lstm_22/while/Identity_4о
lstm_22/while/Identity_5Identity$lstm_22/while/lstm_cell_22/add_1:z:0^lstm_22/while/NoOp*
T0*(
_output_shapes
:         Д2
lstm_22/while/Identity_5Ж
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
9lstm_22_while_lstm_cell_22_matmul_readvariableop_resource;lstm_22_while_lstm_cell_22_matmul_readvariableop_resource_0"╚
alstm_22_while_tensorarrayv2read_tensorlistgetitem_lstm_22_tensorarrayunstack_tensorlistfromtensorclstm_22_while_tensorarrayv2read_tensorlistgetitem_lstm_22_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :         Д:         Д: : : : : 2f
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
:         Д:.*
(
_output_shapes
:         Д:

_output_shapes
: :

_output_shapes
: 
▀
═
while_cond_39112062
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_39112062___redundant_placeholder06
2while_while_cond_39112062___redundant_placeholder16
2while_while_cond_39112062___redundant_placeholder26
2while_while_cond_39112062___redundant_placeholder3
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
@: : : : :         m:         m: ::::: 
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
:         m:-)
'
_output_shapes
:         m:

_output_shapes
: :

_output_shapes
:
╛F
О
E__inference_lstm_23_layer_call_and_return_conditional_losses_39108880

inputs)
lstm_cell_23_39108798:
Д┤(
lstm_cell_23_39108800:	m┤$
lstm_cell_23_39108802:	┤
identityИв$lstm_cell_23/StatefulPartitionedCallвwhileD
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
value	B :m2
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
value	B :m2
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
:         m2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :m2
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
value	B :m2
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
:         m2	
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
!:                  Д2
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
valueB"    Д   27
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
:         Д*
shrink_axis_mask2
strided_slice_2е
$lstm_cell_23/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_23_39108798lstm_cell_23_39108800lstm_cell_23_39108802*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:         m:         m:         m*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_lstm_cell_23_layer_call_and_return_conditional_losses_391087972&
$lstm_cell_23/StatefulPartitionedCallП
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    m   2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_23_39108798lstm_cell_23_39108800lstm_cell_23_39108802*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         m:         m: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_39108811*
condR
while_cond_39108810*K
output_shapes:
8: : : : :         m:         m: : : : : *
parallel_iterations 2
while╡
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    m   22
0TensorArrayV2Stack/TensorListStack/element_shapeё
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :                  m*
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
:         m*
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
 :                  m2
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
 :                  m2

Identity}
NoOpNoOp%^lstm_cell_23/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':                  Д: : : 2L
$lstm_cell_23/StatefulPartitionedCall$lstm_cell_23/StatefulPartitionedCall2
whilewhile:] Y
5
_output_shapes#
!:                  Д
 
_user_specified_nameinputs
╨

э
lstm_23_while_cond_39110857,
(lstm_23_while_lstm_23_while_loop_counter2
.lstm_23_while_lstm_23_while_maximum_iterations
lstm_23_while_placeholder
lstm_23_while_placeholder_1
lstm_23_while_placeholder_2
lstm_23_while_placeholder_3.
*lstm_23_while_less_lstm_23_strided_slice_1F
Blstm_23_while_lstm_23_while_cond_39110857___redundant_placeholder0F
Blstm_23_while_lstm_23_while_cond_39110857___redundant_placeholder1F
Blstm_23_while_lstm_23_while_cond_39110857___redundant_placeholder2F
Blstm_23_while_lstm_23_while_cond_39110857___redundant_placeholder3
lstm_23_while_identity
Ш
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
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :         m:         m: ::::: 
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
:         m:-)
'
_output_shapes
:         m:

_output_shapes
: :

_output_shapes
:
·	
╚
&__inference_signature_wrapper_39110309
lstm_22_input
unknown:	]Р
	unknown_0:
ДР
	unknown_1:	Р
	unknown_2:
Д┤
	unknown_3:	m┤
	unknown_4:	┤
	unknown_5:m
	unknown_6:
identityИвStatefulPartitionedCallм
StatefulPartitionedCallStatefulPartitionedCalllstm_22_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
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
#__inference__wrapped_model_391080922
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
_user_specified_namelstm_22_input
∙
Е
)sequential_11_lstm_22_while_cond_39107831H
Dsequential_11_lstm_22_while_sequential_11_lstm_22_while_loop_counterN
Jsequential_11_lstm_22_while_sequential_11_lstm_22_while_maximum_iterations+
'sequential_11_lstm_22_while_placeholder-
)sequential_11_lstm_22_while_placeholder_1-
)sequential_11_lstm_22_while_placeholder_2-
)sequential_11_lstm_22_while_placeholder_3J
Fsequential_11_lstm_22_while_less_sequential_11_lstm_22_strided_slice_1b
^sequential_11_lstm_22_while_sequential_11_lstm_22_while_cond_39107831___redundant_placeholder0b
^sequential_11_lstm_22_while_sequential_11_lstm_22_while_cond_39107831___redundant_placeholder1b
^sequential_11_lstm_22_while_sequential_11_lstm_22_while_cond_39107831___redundant_placeholder2b
^sequential_11_lstm_22_while_sequential_11_lstm_22_while_cond_39107831___redundant_placeholder3(
$sequential_11_lstm_22_while_identity
▐
 sequential_11/lstm_22/while/LessLess'sequential_11_lstm_22_while_placeholderFsequential_11_lstm_22_while_less_sequential_11_lstm_22_strided_slice_1*
T0*
_output_shapes
: 2"
 sequential_11/lstm_22/while/LessЯ
$sequential_11/lstm_22/while/IdentityIdentity$sequential_11/lstm_22/while/Less:z:0*
T0
*
_output_shapes
: 2&
$sequential_11/lstm_22/while/Identity"U
$sequential_11_lstm_22_while_identity-sequential_11/lstm_22/while/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :         Д:         Д: ::::: 
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
:         Д:.*
(
_output_shapes
:         Д:

_output_shapes
: :

_output_shapes
:
м\
а
E__inference_lstm_23_layer_call_and_return_conditional_losses_39111845
inputs_0?
+lstm_cell_23_matmul_readvariableop_resource:
Д┤@
-lstm_cell_23_matmul_1_readvariableop_resource:	m┤;
,lstm_cell_23_biasadd_readvariableop_resource:	┤
identityИв#lstm_cell_23/BiasAdd/ReadVariableOpв"lstm_cell_23/MatMul/ReadVariableOpв$lstm_cell_23/MatMul_1/ReadVariableOpвwhileF
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
value	B :m2
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
value	B :m2
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
:         m2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :m2
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
value	B :m2
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
:         m2	
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
!:                  Д2
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
valueB"    Д   27
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
:         Д*
shrink_axis_mask2
strided_slice_2╢
"lstm_cell_23/MatMul/ReadVariableOpReadVariableOp+lstm_cell_23_matmul_readvariableop_resource* 
_output_shapes
:
Д┤*
dtype02$
"lstm_cell_23/MatMul/ReadVariableOpн
lstm_cell_23/MatMulMatMulstrided_slice_2:output:0*lstm_cell_23/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ┤2
lstm_cell_23/MatMul╗
$lstm_cell_23/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_23_matmul_1_readvariableop_resource*
_output_shapes
:	m┤*
dtype02&
$lstm_cell_23/MatMul_1/ReadVariableOpй
lstm_cell_23/MatMul_1MatMulzeros:output:0,lstm_cell_23/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ┤2
lstm_cell_23/MatMul_1а
lstm_cell_23/addAddV2lstm_cell_23/MatMul:product:0lstm_cell_23/MatMul_1:product:0*
T0*(
_output_shapes
:         ┤2
lstm_cell_23/add┤
#lstm_cell_23/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_23_biasadd_readvariableop_resource*
_output_shapes	
:┤*
dtype02%
#lstm_cell_23/BiasAdd/ReadVariableOpн
lstm_cell_23/BiasAddBiasAddlstm_cell_23/add:z:0+lstm_cell_23/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ┤2
lstm_cell_23/BiasAdd~
lstm_cell_23/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_23/split/split_dimє
lstm_cell_23/splitSplit%lstm_cell_23/split/split_dim:output:0lstm_cell_23/BiasAdd:output:0*
T0*`
_output_shapesN
L:         m:         m:         m:         m*
	num_split2
lstm_cell_23/splitЖ
lstm_cell_23/SigmoidSigmoidlstm_cell_23/split:output:0*
T0*'
_output_shapes
:         m2
lstm_cell_23/SigmoidК
lstm_cell_23/Sigmoid_1Sigmoidlstm_cell_23/split:output:1*
T0*'
_output_shapes
:         m2
lstm_cell_23/Sigmoid_1Л
lstm_cell_23/mulMullstm_cell_23/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:         m2
lstm_cell_23/mul}
lstm_cell_23/ReluRelulstm_cell_23/split:output:2*
T0*'
_output_shapes
:         m2
lstm_cell_23/ReluЬ
lstm_cell_23/mul_1Mullstm_cell_23/Sigmoid:y:0lstm_cell_23/Relu:activations:0*
T0*'
_output_shapes
:         m2
lstm_cell_23/mul_1С
lstm_cell_23/add_1AddV2lstm_cell_23/mul:z:0lstm_cell_23/mul_1:z:0*
T0*'
_output_shapes
:         m2
lstm_cell_23/add_1К
lstm_cell_23/Sigmoid_2Sigmoidlstm_cell_23/split:output:3*
T0*'
_output_shapes
:         m2
lstm_cell_23/Sigmoid_2|
lstm_cell_23/Relu_1Relulstm_cell_23/add_1:z:0*
T0*'
_output_shapes
:         m2
lstm_cell_23/Relu_1а
lstm_cell_23/mul_2Mullstm_cell_23/Sigmoid_2:y:0!lstm_cell_23/Relu_1:activations:0*
T0*'
_output_shapes
:         m2
lstm_cell_23/mul_2П
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    m   2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_23_matmul_readvariableop_resource-lstm_cell_23_matmul_1_readvariableop_resource,lstm_cell_23_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         m:         m: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_39111761*
condR
while_cond_39111760*K
output_shapes:
8: : : : :         m:         m: : : : : *
parallel_iterations 2
while╡
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    m   22
0TensorArrayV2Stack/TensorListStack/element_shapeё
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :                  m*
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
:         m*
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
 :                  m2
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
 :                  m2

Identity╚
NoOpNoOp$^lstm_cell_23/BiasAdd/ReadVariableOp#^lstm_cell_23/MatMul/ReadVariableOp%^lstm_cell_23/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':                  Д: : : 2J
#lstm_cell_23/BiasAdd/ReadVariableOp#lstm_cell_23/BiasAdd/ReadVariableOp2H
"lstm_cell_23/MatMul/ReadVariableOp"lstm_cell_23/MatMul/ReadVariableOp2L
$lstm_cell_23/MatMul_1/ReadVariableOp$lstm_cell_23/MatMul_1/ReadVariableOp2
whilewhile:_ [
5
_output_shapes#
!:                  Д
"
_user_specified_name
inputs/0
ї
Е
)sequential_11_lstm_23_while_cond_39107979H
Dsequential_11_lstm_23_while_sequential_11_lstm_23_while_loop_counterN
Jsequential_11_lstm_23_while_sequential_11_lstm_23_while_maximum_iterations+
'sequential_11_lstm_23_while_placeholder-
)sequential_11_lstm_23_while_placeholder_1-
)sequential_11_lstm_23_while_placeholder_2-
)sequential_11_lstm_23_while_placeholder_3J
Fsequential_11_lstm_23_while_less_sequential_11_lstm_23_strided_slice_1b
^sequential_11_lstm_23_while_sequential_11_lstm_23_while_cond_39107979___redundant_placeholder0b
^sequential_11_lstm_23_while_sequential_11_lstm_23_while_cond_39107979___redundant_placeholder1b
^sequential_11_lstm_23_while_sequential_11_lstm_23_while_cond_39107979___redundant_placeholder2b
^sequential_11_lstm_23_while_sequential_11_lstm_23_while_cond_39107979___redundant_placeholder3(
$sequential_11_lstm_23_while_identity
▐
 sequential_11/lstm_23/while/LessLess'sequential_11_lstm_23_while_placeholderFsequential_11_lstm_23_while_less_sequential_11_lstm_23_strided_slice_1*
T0*
_output_shapes
: 2"
 sequential_11/lstm_23/while/LessЯ
$sequential_11/lstm_23/while/IdentityIdentity$sequential_11/lstm_23/while/Less:z:0*
T0
*
_output_shapes
: 2&
$sequential_11/lstm_23/while/Identity"U
$sequential_11_lstm_23_while_identity-sequential_11/lstm_23/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :         m:         m: ::::: 
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
:         m:-)
'
_output_shapes
:         m:

_output_shapes
: :

_output_shapes
:
╨!
¤
F__inference_dense_11_layer_call_and_return_conditional_losses_39109720

inputs3
!tensordot_readvariableop_resource:m-
biasadd_readvariableop_resource:
identityИвBiasAdd/ReadVariableOpвTensordot/ReadVariableOpЦ
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:m*
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
Tensordot/stackР
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:         m2
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
_construction_contextkEagerRuntime*.
_input_shapes
:         m: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:S O
+
_output_shapes
:         m
 
_user_specified_nameinputs
Е&
є
while_body_39108391
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_00
while_lstm_cell_22_39108415_0:	]Р1
while_lstm_cell_22_39108417_0:
ДР,
while_lstm_cell_22_39108419_0:	Р
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor.
while_lstm_cell_22_39108415:	]Р/
while_lstm_cell_22_39108417:
ДР*
while_lstm_cell_22_39108419:	РИв*while/lstm_cell_22/StatefulPartitionedCall├
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
*while/lstm_cell_22/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_22_39108415_0while_lstm_cell_22_39108417_0while_lstm_cell_22_39108419_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:         Д:         Д:         Д*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_lstm_cell_22_layer_call_and_return_conditional_losses_391083132,
*while/lstm_cell_22/StatefulPartitionedCallў
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
while/Identity_2Ъ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3е
while/Identity_4Identity3while/lstm_cell_22/StatefulPartitionedCall:output:1^while/NoOp*
T0*(
_output_shapes
:         Д2
while/Identity_4е
while/Identity_5Identity3while/lstm_cell_22/StatefulPartitionedCall:output:2^while/NoOp*
T0*(
_output_shapes
:         Д2
while/Identity_5З

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
while_lstm_cell_22_39108415while_lstm_cell_22_39108415_0"<
while_lstm_cell_22_39108417while_lstm_cell_22_39108417_0"<
while_lstm_cell_22_39108419while_lstm_cell_22_39108419_0"0
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :         Д:         Д: : : : : 2X
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
:         Д:.*
(
_output_shapes
:         Д:

_output_shapes
: :

_output_shapes
: 
у
═
while_cond_39111538
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_39111538___redundant_placeholder06
2while_while_cond_39111538___redundant_placeholder16
2while_while_cond_39111538___redundant_placeholder26
2while_while_cond_39111538___redundant_placeholder3
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
B: : : : :         Д:         Д: ::::: 
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
:         Д:.*
(
_output_shapes
:         Д:

_output_shapes
: :

_output_shapes
:
▀
═
while_cond_39111760
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_39111760___redundant_placeholder06
2while_while_cond_39111760___redundant_placeholder16
2while_while_cond_39111760___redundant_placeholder26
2while_while_cond_39111760___redundant_placeholder3
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
@: : : : :         m:         m: ::::: 
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
:         m:-)
'
_output_shapes
:         m:

_output_shapes
: :

_output_shapes
:
╙J
╘

lstm_23_while_body_39110858,
(lstm_23_while_lstm_23_while_loop_counter2
.lstm_23_while_lstm_23_while_maximum_iterations
lstm_23_while_placeholder
lstm_23_while_placeholder_1
lstm_23_while_placeholder_2
lstm_23_while_placeholder_3+
'lstm_23_while_lstm_23_strided_slice_1_0g
clstm_23_while_tensorarrayv2read_tensorlistgetitem_lstm_23_tensorarrayunstack_tensorlistfromtensor_0O
;lstm_23_while_lstm_cell_23_matmul_readvariableop_resource_0:
Д┤P
=lstm_23_while_lstm_cell_23_matmul_1_readvariableop_resource_0:	m┤K
<lstm_23_while_lstm_cell_23_biasadd_readvariableop_resource_0:	┤
lstm_23_while_identity
lstm_23_while_identity_1
lstm_23_while_identity_2
lstm_23_while_identity_3
lstm_23_while_identity_4
lstm_23_while_identity_5)
%lstm_23_while_lstm_23_strided_slice_1e
alstm_23_while_tensorarrayv2read_tensorlistgetitem_lstm_23_tensorarrayunstack_tensorlistfromtensorM
9lstm_23_while_lstm_cell_23_matmul_readvariableop_resource:
Д┤N
;lstm_23_while_lstm_cell_23_matmul_1_readvariableop_resource:	m┤I
:lstm_23_while_lstm_cell_23_biasadd_readvariableop_resource:	┤Ив1lstm_23/while/lstm_cell_23/BiasAdd/ReadVariableOpв0lstm_23/while/lstm_cell_23/MatMul/ReadVariableOpв2lstm_23/while/lstm_cell_23/MatMul_1/ReadVariableOp╙
?lstm_23/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    Д   2A
?lstm_23/while/TensorArrayV2Read/TensorListGetItem/element_shapeД
1lstm_23/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemclstm_23_while_tensorarrayv2read_tensorlistgetitem_lstm_23_tensorarrayunstack_tensorlistfromtensor_0lstm_23_while_placeholderHlstm_23/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:         Д*
element_dtype023
1lstm_23/while/TensorArrayV2Read/TensorListGetItemт
0lstm_23/while/lstm_cell_23/MatMul/ReadVariableOpReadVariableOp;lstm_23_while_lstm_cell_23_matmul_readvariableop_resource_0* 
_output_shapes
:
Д┤*
dtype022
0lstm_23/while/lstm_cell_23/MatMul/ReadVariableOpў
!lstm_23/while/lstm_cell_23/MatMulMatMul8lstm_23/while/TensorArrayV2Read/TensorListGetItem:item:08lstm_23/while/lstm_cell_23/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ┤2#
!lstm_23/while/lstm_cell_23/MatMulч
2lstm_23/while/lstm_cell_23/MatMul_1/ReadVariableOpReadVariableOp=lstm_23_while_lstm_cell_23_matmul_1_readvariableop_resource_0*
_output_shapes
:	m┤*
dtype024
2lstm_23/while/lstm_cell_23/MatMul_1/ReadVariableOpр
#lstm_23/while/lstm_cell_23/MatMul_1MatMullstm_23_while_placeholder_2:lstm_23/while/lstm_cell_23/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ┤2%
#lstm_23/while/lstm_cell_23/MatMul_1╪
lstm_23/while/lstm_cell_23/addAddV2+lstm_23/while/lstm_cell_23/MatMul:product:0-lstm_23/while/lstm_cell_23/MatMul_1:product:0*
T0*(
_output_shapes
:         ┤2 
lstm_23/while/lstm_cell_23/addр
1lstm_23/while/lstm_cell_23/BiasAdd/ReadVariableOpReadVariableOp<lstm_23_while_lstm_cell_23_biasadd_readvariableop_resource_0*
_output_shapes	
:┤*
dtype023
1lstm_23/while/lstm_cell_23/BiasAdd/ReadVariableOpх
"lstm_23/while/lstm_cell_23/BiasAddBiasAdd"lstm_23/while/lstm_cell_23/add:z:09lstm_23/while/lstm_cell_23/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ┤2$
"lstm_23/while/lstm_cell_23/BiasAddЪ
*lstm_23/while/lstm_cell_23/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2,
*lstm_23/while/lstm_cell_23/split/split_dimл
 lstm_23/while/lstm_cell_23/splitSplit3lstm_23/while/lstm_cell_23/split/split_dim:output:0+lstm_23/while/lstm_cell_23/BiasAdd:output:0*
T0*`
_output_shapesN
L:         m:         m:         m:         m*
	num_split2"
 lstm_23/while/lstm_cell_23/split░
"lstm_23/while/lstm_cell_23/SigmoidSigmoid)lstm_23/while/lstm_cell_23/split:output:0*
T0*'
_output_shapes
:         m2$
"lstm_23/while/lstm_cell_23/Sigmoid┤
$lstm_23/while/lstm_cell_23/Sigmoid_1Sigmoid)lstm_23/while/lstm_cell_23/split:output:1*
T0*'
_output_shapes
:         m2&
$lstm_23/while/lstm_cell_23/Sigmoid_1└
lstm_23/while/lstm_cell_23/mulMul(lstm_23/while/lstm_cell_23/Sigmoid_1:y:0lstm_23_while_placeholder_3*
T0*'
_output_shapes
:         m2 
lstm_23/while/lstm_cell_23/mulз
lstm_23/while/lstm_cell_23/ReluRelu)lstm_23/while/lstm_cell_23/split:output:2*
T0*'
_output_shapes
:         m2!
lstm_23/while/lstm_cell_23/Relu╘
 lstm_23/while/lstm_cell_23/mul_1Mul&lstm_23/while/lstm_cell_23/Sigmoid:y:0-lstm_23/while/lstm_cell_23/Relu:activations:0*
T0*'
_output_shapes
:         m2"
 lstm_23/while/lstm_cell_23/mul_1╔
 lstm_23/while/lstm_cell_23/add_1AddV2"lstm_23/while/lstm_cell_23/mul:z:0$lstm_23/while/lstm_cell_23/mul_1:z:0*
T0*'
_output_shapes
:         m2"
 lstm_23/while/lstm_cell_23/add_1┤
$lstm_23/while/lstm_cell_23/Sigmoid_2Sigmoid)lstm_23/while/lstm_cell_23/split:output:3*
T0*'
_output_shapes
:         m2&
$lstm_23/while/lstm_cell_23/Sigmoid_2ж
!lstm_23/while/lstm_cell_23/Relu_1Relu$lstm_23/while/lstm_cell_23/add_1:z:0*
T0*'
_output_shapes
:         m2#
!lstm_23/while/lstm_cell_23/Relu_1╪
 lstm_23/while/lstm_cell_23/mul_2Mul(lstm_23/while/lstm_cell_23/Sigmoid_2:y:0/lstm_23/while/lstm_cell_23/Relu_1:activations:0*
T0*'
_output_shapes
:         m2"
 lstm_23/while/lstm_cell_23/mul_2И
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
lstm_23/while/add/yЙ
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
lstm_23/while/add_1/yЮ
lstm_23/while/add_1AddV2(lstm_23_while_lstm_23_while_loop_counterlstm_23/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lstm_23/while/add_1Л
lstm_23/while/IdentityIdentitylstm_23/while/add_1:z:0^lstm_23/while/NoOp*
T0*
_output_shapes
: 2
lstm_23/while/Identityж
lstm_23/while/Identity_1Identity.lstm_23_while_lstm_23_while_maximum_iterations^lstm_23/while/NoOp*
T0*
_output_shapes
: 2
lstm_23/while/Identity_1Н
lstm_23/while/Identity_2Identitylstm_23/while/add:z:0^lstm_23/while/NoOp*
T0*
_output_shapes
: 2
lstm_23/while/Identity_2║
lstm_23/while/Identity_3IdentityBlstm_23/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_23/while/NoOp*
T0*
_output_shapes
: 2
lstm_23/while/Identity_3н
lstm_23/while/Identity_4Identity$lstm_23/while/lstm_cell_23/mul_2:z:0^lstm_23/while/NoOp*
T0*'
_output_shapes
:         m2
lstm_23/while/Identity_4н
lstm_23/while/Identity_5Identity$lstm_23/while/lstm_cell_23/add_1:z:0^lstm_23/while/NoOp*
T0*'
_output_shapes
:         m2
lstm_23/while/Identity_5Ж
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
9lstm_23_while_lstm_cell_23_matmul_readvariableop_resource;lstm_23_while_lstm_cell_23_matmul_readvariableop_resource_0"╚
alstm_23_while_tensorarrayv2read_tensorlistgetitem_lstm_23_tensorarrayunstack_tensorlistfromtensorclstm_23_while_tensorarrayv2read_tensorlistgetitem_lstm_23_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         m:         m: : : : : 2f
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
: :-)
'
_output_shapes
:         m:-)
'
_output_shapes
:         m:

_output_shapes
: :

_output_shapes
: 
Ч

╦
0__inference_sequential_11_layer_call_fn_39110998

inputs
unknown:	]Р
	unknown_0:
ДР
	unknown_1:	Р
	unknown_2:
Д┤
	unknown_3:	m┤
	unknown_4:	┤
	unknown_5:m
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
K__inference_sequential_11_layer_call_and_return_conditional_losses_391097272
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
Е
Ш
+__inference_dense_11_layer_call_fn_39112409

inputs
unknown:m
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
F__inference_dense_11_layer_call_and_return_conditional_losses_391097202
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
_construction_contextkEagerRuntime*.
_input_shapes
:         m: : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:         m
 
_user_specified_nameinputs
Д\
Ю
E__inference_lstm_22_layer_call_and_return_conditional_losses_39111623

inputs>
+lstm_cell_22_matmul_readvariableop_resource:	]РA
-lstm_cell_22_matmul_1_readvariableop_resource:
ДР;
,lstm_cell_22_biasadd_readvariableop_resource:	Р
identityИв#lstm_cell_22/BiasAdd/ReadVariableOpв"lstm_cell_22/MatMul/ReadVariableOpв$lstm_cell_22/MatMul_1/ReadVariableOpвwhileD
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
B :Д2
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
B :Д2
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
:         Д2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :Д2
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
B :Д2
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
:         Д2	
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
"lstm_cell_22/MatMul/ReadVariableOpReadVariableOp+lstm_cell_22_matmul_readvariableop_resource*
_output_shapes
:	]Р*
dtype02$
"lstm_cell_22/MatMul/ReadVariableOpн
lstm_cell_22/MatMulMatMulstrided_slice_2:output:0*lstm_cell_22/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Р2
lstm_cell_22/MatMul╝
$lstm_cell_22/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_22_matmul_1_readvariableop_resource* 
_output_shapes
:
ДР*
dtype02&
$lstm_cell_22/MatMul_1/ReadVariableOpй
lstm_cell_22/MatMul_1MatMulzeros:output:0,lstm_cell_22/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Р2
lstm_cell_22/MatMul_1а
lstm_cell_22/addAddV2lstm_cell_22/MatMul:product:0lstm_cell_22/MatMul_1:product:0*
T0*(
_output_shapes
:         Р2
lstm_cell_22/add┤
#lstm_cell_22/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_22_biasadd_readvariableop_resource*
_output_shapes	
:Р*
dtype02%
#lstm_cell_22/BiasAdd/ReadVariableOpн
lstm_cell_22/BiasAddBiasAddlstm_cell_22/add:z:0+lstm_cell_22/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Р2
lstm_cell_22/BiasAdd~
lstm_cell_22/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_22/split/split_dimў
lstm_cell_22/splitSplit%lstm_cell_22/split/split_dim:output:0lstm_cell_22/BiasAdd:output:0*
T0*d
_output_shapesR
P:         Д:         Д:         Д:         Д*
	num_split2
lstm_cell_22/splitЗ
lstm_cell_22/SigmoidSigmoidlstm_cell_22/split:output:0*
T0*(
_output_shapes
:         Д2
lstm_cell_22/SigmoidЛ
lstm_cell_22/Sigmoid_1Sigmoidlstm_cell_22/split:output:1*
T0*(
_output_shapes
:         Д2
lstm_cell_22/Sigmoid_1М
lstm_cell_22/mulMullstm_cell_22/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:         Д2
lstm_cell_22/mul~
lstm_cell_22/ReluRelulstm_cell_22/split:output:2*
T0*(
_output_shapes
:         Д2
lstm_cell_22/ReluЭ
lstm_cell_22/mul_1Mullstm_cell_22/Sigmoid:y:0lstm_cell_22/Relu:activations:0*
T0*(
_output_shapes
:         Д2
lstm_cell_22/mul_1Т
lstm_cell_22/add_1AddV2lstm_cell_22/mul:z:0lstm_cell_22/mul_1:z:0*
T0*(
_output_shapes
:         Д2
lstm_cell_22/add_1Л
lstm_cell_22/Sigmoid_2Sigmoidlstm_cell_22/split:output:3*
T0*(
_output_shapes
:         Д2
lstm_cell_22/Sigmoid_2}
lstm_cell_22/Relu_1Relulstm_cell_22/add_1:z:0*
T0*(
_output_shapes
:         Д2
lstm_cell_22/Relu_1б
lstm_cell_22/mul_2Mullstm_cell_22/Sigmoid_2:y:0!lstm_cell_22/Relu_1:activations:0*
T0*(
_output_shapes
:         Д2
lstm_cell_22/mul_2П
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    Д   2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_22_matmul_readvariableop_resource-lstm_cell_22_matmul_1_readvariableop_resource,lstm_cell_22_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :         Д:         Д: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_39111539*
condR
while_cond_39111538*M
output_shapes<
:: : : : :         Д:         Д: : : : : *
parallel_iterations 2
while╡
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    Д   22
0TensorArrayV2Stack/TensorListStack/element_shapeщ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:         Д*
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
:         Д*
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
:         Д2
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
:         Д2

Identity╚
NoOpNoOp$^lstm_cell_22/BiasAdd/ReadVariableOp#^lstm_cell_22/MatMul/ReadVariableOp%^lstm_cell_22/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         ]: : : 2J
#lstm_cell_22/BiasAdd/ReadVariableOp#lstm_cell_22/BiasAdd/ReadVariableOp2H
"lstm_cell_22/MatMul/ReadVariableOp"lstm_cell_22/MatMul/ReadVariableOp2L
$lstm_cell_22/MatMul_1/ReadVariableOp$lstm_cell_22/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:         ]
 
_user_specified_nameinputs
╙J
╘

lstm_23_while_body_39110524,
(lstm_23_while_lstm_23_while_loop_counter2
.lstm_23_while_lstm_23_while_maximum_iterations
lstm_23_while_placeholder
lstm_23_while_placeholder_1
lstm_23_while_placeholder_2
lstm_23_while_placeholder_3+
'lstm_23_while_lstm_23_strided_slice_1_0g
clstm_23_while_tensorarrayv2read_tensorlistgetitem_lstm_23_tensorarrayunstack_tensorlistfromtensor_0O
;lstm_23_while_lstm_cell_23_matmul_readvariableop_resource_0:
Д┤P
=lstm_23_while_lstm_cell_23_matmul_1_readvariableop_resource_0:	m┤K
<lstm_23_while_lstm_cell_23_biasadd_readvariableop_resource_0:	┤
lstm_23_while_identity
lstm_23_while_identity_1
lstm_23_while_identity_2
lstm_23_while_identity_3
lstm_23_while_identity_4
lstm_23_while_identity_5)
%lstm_23_while_lstm_23_strided_slice_1e
alstm_23_while_tensorarrayv2read_tensorlistgetitem_lstm_23_tensorarrayunstack_tensorlistfromtensorM
9lstm_23_while_lstm_cell_23_matmul_readvariableop_resource:
Д┤N
;lstm_23_while_lstm_cell_23_matmul_1_readvariableop_resource:	m┤I
:lstm_23_while_lstm_cell_23_biasadd_readvariableop_resource:	┤Ив1lstm_23/while/lstm_cell_23/BiasAdd/ReadVariableOpв0lstm_23/while/lstm_cell_23/MatMul/ReadVariableOpв2lstm_23/while/lstm_cell_23/MatMul_1/ReadVariableOp╙
?lstm_23/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    Д   2A
?lstm_23/while/TensorArrayV2Read/TensorListGetItem/element_shapeД
1lstm_23/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemclstm_23_while_tensorarrayv2read_tensorlistgetitem_lstm_23_tensorarrayunstack_tensorlistfromtensor_0lstm_23_while_placeholderHlstm_23/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:         Д*
element_dtype023
1lstm_23/while/TensorArrayV2Read/TensorListGetItemт
0lstm_23/while/lstm_cell_23/MatMul/ReadVariableOpReadVariableOp;lstm_23_while_lstm_cell_23_matmul_readvariableop_resource_0* 
_output_shapes
:
Д┤*
dtype022
0lstm_23/while/lstm_cell_23/MatMul/ReadVariableOpў
!lstm_23/while/lstm_cell_23/MatMulMatMul8lstm_23/while/TensorArrayV2Read/TensorListGetItem:item:08lstm_23/while/lstm_cell_23/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ┤2#
!lstm_23/while/lstm_cell_23/MatMulч
2lstm_23/while/lstm_cell_23/MatMul_1/ReadVariableOpReadVariableOp=lstm_23_while_lstm_cell_23_matmul_1_readvariableop_resource_0*
_output_shapes
:	m┤*
dtype024
2lstm_23/while/lstm_cell_23/MatMul_1/ReadVariableOpр
#lstm_23/while/lstm_cell_23/MatMul_1MatMullstm_23_while_placeholder_2:lstm_23/while/lstm_cell_23/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ┤2%
#lstm_23/while/lstm_cell_23/MatMul_1╪
lstm_23/while/lstm_cell_23/addAddV2+lstm_23/while/lstm_cell_23/MatMul:product:0-lstm_23/while/lstm_cell_23/MatMul_1:product:0*
T0*(
_output_shapes
:         ┤2 
lstm_23/while/lstm_cell_23/addр
1lstm_23/while/lstm_cell_23/BiasAdd/ReadVariableOpReadVariableOp<lstm_23_while_lstm_cell_23_biasadd_readvariableop_resource_0*
_output_shapes	
:┤*
dtype023
1lstm_23/while/lstm_cell_23/BiasAdd/ReadVariableOpх
"lstm_23/while/lstm_cell_23/BiasAddBiasAdd"lstm_23/while/lstm_cell_23/add:z:09lstm_23/while/lstm_cell_23/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ┤2$
"lstm_23/while/lstm_cell_23/BiasAddЪ
*lstm_23/while/lstm_cell_23/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2,
*lstm_23/while/lstm_cell_23/split/split_dimл
 lstm_23/while/lstm_cell_23/splitSplit3lstm_23/while/lstm_cell_23/split/split_dim:output:0+lstm_23/while/lstm_cell_23/BiasAdd:output:0*
T0*`
_output_shapesN
L:         m:         m:         m:         m*
	num_split2"
 lstm_23/while/lstm_cell_23/split░
"lstm_23/while/lstm_cell_23/SigmoidSigmoid)lstm_23/while/lstm_cell_23/split:output:0*
T0*'
_output_shapes
:         m2$
"lstm_23/while/lstm_cell_23/Sigmoid┤
$lstm_23/while/lstm_cell_23/Sigmoid_1Sigmoid)lstm_23/while/lstm_cell_23/split:output:1*
T0*'
_output_shapes
:         m2&
$lstm_23/while/lstm_cell_23/Sigmoid_1└
lstm_23/while/lstm_cell_23/mulMul(lstm_23/while/lstm_cell_23/Sigmoid_1:y:0lstm_23_while_placeholder_3*
T0*'
_output_shapes
:         m2 
lstm_23/while/lstm_cell_23/mulз
lstm_23/while/lstm_cell_23/ReluRelu)lstm_23/while/lstm_cell_23/split:output:2*
T0*'
_output_shapes
:         m2!
lstm_23/while/lstm_cell_23/Relu╘
 lstm_23/while/lstm_cell_23/mul_1Mul&lstm_23/while/lstm_cell_23/Sigmoid:y:0-lstm_23/while/lstm_cell_23/Relu:activations:0*
T0*'
_output_shapes
:         m2"
 lstm_23/while/lstm_cell_23/mul_1╔
 lstm_23/while/lstm_cell_23/add_1AddV2"lstm_23/while/lstm_cell_23/mul:z:0$lstm_23/while/lstm_cell_23/mul_1:z:0*
T0*'
_output_shapes
:         m2"
 lstm_23/while/lstm_cell_23/add_1┤
$lstm_23/while/lstm_cell_23/Sigmoid_2Sigmoid)lstm_23/while/lstm_cell_23/split:output:3*
T0*'
_output_shapes
:         m2&
$lstm_23/while/lstm_cell_23/Sigmoid_2ж
!lstm_23/while/lstm_cell_23/Relu_1Relu$lstm_23/while/lstm_cell_23/add_1:z:0*
T0*'
_output_shapes
:         m2#
!lstm_23/while/lstm_cell_23/Relu_1╪
 lstm_23/while/lstm_cell_23/mul_2Mul(lstm_23/while/lstm_cell_23/Sigmoid_2:y:0/lstm_23/while/lstm_cell_23/Relu_1:activations:0*
T0*'
_output_shapes
:         m2"
 lstm_23/while/lstm_cell_23/mul_2И
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
lstm_23/while/add/yЙ
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
lstm_23/while/add_1/yЮ
lstm_23/while/add_1AddV2(lstm_23_while_lstm_23_while_loop_counterlstm_23/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lstm_23/while/add_1Л
lstm_23/while/IdentityIdentitylstm_23/while/add_1:z:0^lstm_23/while/NoOp*
T0*
_output_shapes
: 2
lstm_23/while/Identityж
lstm_23/while/Identity_1Identity.lstm_23_while_lstm_23_while_maximum_iterations^lstm_23/while/NoOp*
T0*
_output_shapes
: 2
lstm_23/while/Identity_1Н
lstm_23/while/Identity_2Identitylstm_23/while/add:z:0^lstm_23/while/NoOp*
T0*
_output_shapes
: 2
lstm_23/while/Identity_2║
lstm_23/while/Identity_3IdentityBlstm_23/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_23/while/NoOp*
T0*
_output_shapes
: 2
lstm_23/while/Identity_3н
lstm_23/while/Identity_4Identity$lstm_23/while/lstm_cell_23/mul_2:z:0^lstm_23/while/NoOp*
T0*'
_output_shapes
:         m2
lstm_23/while/Identity_4н
lstm_23/while/Identity_5Identity$lstm_23/while/lstm_cell_23/add_1:z:0^lstm_23/while/NoOp*
T0*'
_output_shapes
:         m2
lstm_23/while/Identity_5Ж
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
9lstm_23_while_lstm_cell_23_matmul_readvariableop_resource;lstm_23_while_lstm_cell_23_matmul_readvariableop_resource_0"╚
alstm_23_while_tensorarrayv2read_tensorlistgetitem_lstm_23_tensorarrayunstack_tensorlistfromtensorclstm_23_while_tensorarrayv2read_tensorlistgetitem_lstm_23_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         m:         m: : : : : 2f
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
: :-)
'
_output_shapes
:         m:-)
'
_output_shapes
:         m:

_output_shapes
: :

_output_shapes
: 
р
║
*__inference_lstm_22_layer_call_fn_39111645
inputs_0
unknown:	]Р
	unknown_0:
ДР
	unknown_1:	Р
identityИвStatefulPartitionedCallТ
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:                  Д*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_lstm_22_layer_call_and_return_conditional_losses_391084602
StatefulPartitionedCallЙ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:                  Д2

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
╡
╕
*__inference_lstm_23_layer_call_fn_39112342

inputs
unknown:
Д┤
	unknown_0:	m┤
	unknown_1:	┤
identityИвStatefulPartitionedCallЖ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         m*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_lstm_23_layer_call_and_return_conditional_losses_391099432
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:         m2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:         Д: : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:         Д
 
_user_specified_nameinputs
╧
g
H__inference_dropout_23_layer_call_and_return_conditional_losses_39112359

inputs
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  а?2
dropout/Constw
dropout/MulMulinputsdropout/Const:output:0*
T0*+
_output_shapes
:         m2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape╕
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:         m*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>2
dropout/GreaterEqual/y┬
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:         m2
dropout/GreaterEqualГ
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:         m2
dropout/Cast~
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*+
_output_shapes
:         m2
dropout/Mul_1i
IdentityIdentitydropout/Mul_1:z:0*
T0*+
_output_shapes
:         m2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         m:S O
+
_output_shapes
:         m
 
_user_specified_nameinputs
╛F
О
E__inference_lstm_23_layer_call_and_return_conditional_losses_39109090

inputs)
lstm_cell_23_39109008:
Д┤(
lstm_cell_23_39109010:	m┤$
lstm_cell_23_39109012:	┤
identityИв$lstm_cell_23/StatefulPartitionedCallвwhileD
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
value	B :m2
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
value	B :m2
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
:         m2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :m2
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
value	B :m2
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
:         m2	
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
!:                  Д2
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
valueB"    Д   27
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
:         Д*
shrink_axis_mask2
strided_slice_2е
$lstm_cell_23/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_23_39109008lstm_cell_23_39109010lstm_cell_23_39109012*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:         m:         m:         m*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_lstm_cell_23_layer_call_and_return_conditional_losses_391089432&
$lstm_cell_23/StatefulPartitionedCallП
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    m   2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_23_39109008lstm_cell_23_39109010lstm_cell_23_39109012*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         m:         m: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_39109021*
condR
while_cond_39109020*K
output_shapes:
8: : : : :         m:         m: : : : : *
parallel_iterations 2
while╡
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    m   22
0TensorArrayV2Stack/TensorListStack/element_shapeё
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :                  m*
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
:         m*
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
 :                  m2
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
 :                  m2

Identity}
NoOpNoOp%^lstm_cell_23/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':                  Д: : : 2L
$lstm_cell_23/StatefulPartitionedCall$lstm_cell_23/StatefulPartitionedCall2
whilewhile:] Y
5
_output_shapes#
!:                  Д
 
_user_specified_nameinputs
Ю?
╘
while_body_39109590
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
3while_lstm_cell_23_matmul_readvariableop_resource_0:
Д┤H
5while_lstm_cell_23_matmul_1_readvariableop_resource_0:	m┤C
4while_lstm_cell_23_biasadd_readvariableop_resource_0:	┤
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
1while_lstm_cell_23_matmul_readvariableop_resource:
Д┤F
3while_lstm_cell_23_matmul_1_readvariableop_resource:	m┤A
2while_lstm_cell_23_biasadd_readvariableop_resource:	┤Ив)while/lstm_cell_23/BiasAdd/ReadVariableOpв(while/lstm_cell_23/MatMul/ReadVariableOpв*while/lstm_cell_23/MatMul_1/ReadVariableOp├
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    Д   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape╘
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:         Д*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem╩
(while/lstm_cell_23/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_23_matmul_readvariableop_resource_0* 
_output_shapes
:
Д┤*
dtype02*
(while/lstm_cell_23/MatMul/ReadVariableOp╫
while/lstm_cell_23/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_23/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ┤2
while/lstm_cell_23/MatMul╧
*while/lstm_cell_23/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_23_matmul_1_readvariableop_resource_0*
_output_shapes
:	m┤*
dtype02,
*while/lstm_cell_23/MatMul_1/ReadVariableOp└
while/lstm_cell_23/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_23/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ┤2
while/lstm_cell_23/MatMul_1╕
while/lstm_cell_23/addAddV2#while/lstm_cell_23/MatMul:product:0%while/lstm_cell_23/MatMul_1:product:0*
T0*(
_output_shapes
:         ┤2
while/lstm_cell_23/add╚
)while/lstm_cell_23/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_23_biasadd_readvariableop_resource_0*
_output_shapes	
:┤*
dtype02+
)while/lstm_cell_23/BiasAdd/ReadVariableOp┼
while/lstm_cell_23/BiasAddBiasAddwhile/lstm_cell_23/add:z:01while/lstm_cell_23/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ┤2
while/lstm_cell_23/BiasAddК
"while/lstm_cell_23/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_23/split/split_dimЛ
while/lstm_cell_23/splitSplit+while/lstm_cell_23/split/split_dim:output:0#while/lstm_cell_23/BiasAdd:output:0*
T0*`
_output_shapesN
L:         m:         m:         m:         m*
	num_split2
while/lstm_cell_23/splitШ
while/lstm_cell_23/SigmoidSigmoid!while/lstm_cell_23/split:output:0*
T0*'
_output_shapes
:         m2
while/lstm_cell_23/SigmoidЬ
while/lstm_cell_23/Sigmoid_1Sigmoid!while/lstm_cell_23/split:output:1*
T0*'
_output_shapes
:         m2
while/lstm_cell_23/Sigmoid_1а
while/lstm_cell_23/mulMul while/lstm_cell_23/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:         m2
while/lstm_cell_23/mulП
while/lstm_cell_23/ReluRelu!while/lstm_cell_23/split:output:2*
T0*'
_output_shapes
:         m2
while/lstm_cell_23/Relu┤
while/lstm_cell_23/mul_1Mulwhile/lstm_cell_23/Sigmoid:y:0%while/lstm_cell_23/Relu:activations:0*
T0*'
_output_shapes
:         m2
while/lstm_cell_23/mul_1й
while/lstm_cell_23/add_1AddV2while/lstm_cell_23/mul:z:0while/lstm_cell_23/mul_1:z:0*
T0*'
_output_shapes
:         m2
while/lstm_cell_23/add_1Ь
while/lstm_cell_23/Sigmoid_2Sigmoid!while/lstm_cell_23/split:output:3*
T0*'
_output_shapes
:         m2
while/lstm_cell_23/Sigmoid_2О
while/lstm_cell_23/Relu_1Reluwhile/lstm_cell_23/add_1:z:0*
T0*'
_output_shapes
:         m2
while/lstm_cell_23/Relu_1╕
while/lstm_cell_23/mul_2Mul while/lstm_cell_23/Sigmoid_2:y:0'while/lstm_cell_23/Relu_1:activations:0*
T0*'
_output_shapes
:         m2
while/lstm_cell_23/mul_2р
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
while/Identity_2Ъ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3Н
while/Identity_4Identitywhile/lstm_cell_23/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:         m2
while/Identity_4Н
while/Identity_5Identitywhile/lstm_cell_23/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:         m2
while/Identity_5▐

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
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         m:         m: : : : : 2V
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
: :-)
'
_output_shapes
:         m:-)
'
_output_shapes
:         m:

_output_shapes
: :

_output_shapes
: 
ф^
Ц
)sequential_11_lstm_22_while_body_39107832H
Dsequential_11_lstm_22_while_sequential_11_lstm_22_while_loop_counterN
Jsequential_11_lstm_22_while_sequential_11_lstm_22_while_maximum_iterations+
'sequential_11_lstm_22_while_placeholder-
)sequential_11_lstm_22_while_placeholder_1-
)sequential_11_lstm_22_while_placeholder_2-
)sequential_11_lstm_22_while_placeholder_3G
Csequential_11_lstm_22_while_sequential_11_lstm_22_strided_slice_1_0Г
sequential_11_lstm_22_while_tensorarrayv2read_tensorlistgetitem_sequential_11_lstm_22_tensorarrayunstack_tensorlistfromtensor_0\
Isequential_11_lstm_22_while_lstm_cell_22_matmul_readvariableop_resource_0:	]Р_
Ksequential_11_lstm_22_while_lstm_cell_22_matmul_1_readvariableop_resource_0:
ДРY
Jsequential_11_lstm_22_while_lstm_cell_22_biasadd_readvariableop_resource_0:	Р(
$sequential_11_lstm_22_while_identity*
&sequential_11_lstm_22_while_identity_1*
&sequential_11_lstm_22_while_identity_2*
&sequential_11_lstm_22_while_identity_3*
&sequential_11_lstm_22_while_identity_4*
&sequential_11_lstm_22_while_identity_5E
Asequential_11_lstm_22_while_sequential_11_lstm_22_strided_slice_1Б
}sequential_11_lstm_22_while_tensorarrayv2read_tensorlistgetitem_sequential_11_lstm_22_tensorarrayunstack_tensorlistfromtensorZ
Gsequential_11_lstm_22_while_lstm_cell_22_matmul_readvariableop_resource:	]Р]
Isequential_11_lstm_22_while_lstm_cell_22_matmul_1_readvariableop_resource:
ДРW
Hsequential_11_lstm_22_while_lstm_cell_22_biasadd_readvariableop_resource:	РИв?sequential_11/lstm_22/while/lstm_cell_22/BiasAdd/ReadVariableOpв>sequential_11/lstm_22/while/lstm_cell_22/MatMul/ReadVariableOpв@sequential_11/lstm_22/while/lstm_cell_22/MatMul_1/ReadVariableOpя
Msequential_11/lstm_22/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ]   2O
Msequential_11/lstm_22/while/TensorArrayV2Read/TensorListGetItem/element_shape╫
?sequential_11/lstm_22/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemsequential_11_lstm_22_while_tensorarrayv2read_tensorlistgetitem_sequential_11_lstm_22_tensorarrayunstack_tensorlistfromtensor_0'sequential_11_lstm_22_while_placeholderVsequential_11/lstm_22/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         ]*
element_dtype02A
?sequential_11/lstm_22/while/TensorArrayV2Read/TensorListGetItemЛ
>sequential_11/lstm_22/while/lstm_cell_22/MatMul/ReadVariableOpReadVariableOpIsequential_11_lstm_22_while_lstm_cell_22_matmul_readvariableop_resource_0*
_output_shapes
:	]Р*
dtype02@
>sequential_11/lstm_22/while/lstm_cell_22/MatMul/ReadVariableOpп
/sequential_11/lstm_22/while/lstm_cell_22/MatMulMatMulFsequential_11/lstm_22/while/TensorArrayV2Read/TensorListGetItem:item:0Fsequential_11/lstm_22/while/lstm_cell_22/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Р21
/sequential_11/lstm_22/while/lstm_cell_22/MatMulТ
@sequential_11/lstm_22/while/lstm_cell_22/MatMul_1/ReadVariableOpReadVariableOpKsequential_11_lstm_22_while_lstm_cell_22_matmul_1_readvariableop_resource_0* 
_output_shapes
:
ДР*
dtype02B
@sequential_11/lstm_22/while/lstm_cell_22/MatMul_1/ReadVariableOpШ
1sequential_11/lstm_22/while/lstm_cell_22/MatMul_1MatMul)sequential_11_lstm_22_while_placeholder_2Hsequential_11/lstm_22/while/lstm_cell_22/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Р23
1sequential_11/lstm_22/while/lstm_cell_22/MatMul_1Р
,sequential_11/lstm_22/while/lstm_cell_22/addAddV29sequential_11/lstm_22/while/lstm_cell_22/MatMul:product:0;sequential_11/lstm_22/while/lstm_cell_22/MatMul_1:product:0*
T0*(
_output_shapes
:         Р2.
,sequential_11/lstm_22/while/lstm_cell_22/addК
?sequential_11/lstm_22/while/lstm_cell_22/BiasAdd/ReadVariableOpReadVariableOpJsequential_11_lstm_22_while_lstm_cell_22_biasadd_readvariableop_resource_0*
_output_shapes	
:Р*
dtype02A
?sequential_11/lstm_22/while/lstm_cell_22/BiasAdd/ReadVariableOpЭ
0sequential_11/lstm_22/while/lstm_cell_22/BiasAddBiasAdd0sequential_11/lstm_22/while/lstm_cell_22/add:z:0Gsequential_11/lstm_22/while/lstm_cell_22/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Р22
0sequential_11/lstm_22/while/lstm_cell_22/BiasAdd╢
8sequential_11/lstm_22/while/lstm_cell_22/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2:
8sequential_11/lstm_22/while/lstm_cell_22/split/split_dimч
.sequential_11/lstm_22/while/lstm_cell_22/splitSplitAsequential_11/lstm_22/while/lstm_cell_22/split/split_dim:output:09sequential_11/lstm_22/while/lstm_cell_22/BiasAdd:output:0*
T0*d
_output_shapesR
P:         Д:         Д:         Д:         Д*
	num_split20
.sequential_11/lstm_22/while/lstm_cell_22/split█
0sequential_11/lstm_22/while/lstm_cell_22/SigmoidSigmoid7sequential_11/lstm_22/while/lstm_cell_22/split:output:0*
T0*(
_output_shapes
:         Д22
0sequential_11/lstm_22/while/lstm_cell_22/Sigmoid▀
2sequential_11/lstm_22/while/lstm_cell_22/Sigmoid_1Sigmoid7sequential_11/lstm_22/while/lstm_cell_22/split:output:1*
T0*(
_output_shapes
:         Д24
2sequential_11/lstm_22/while/lstm_cell_22/Sigmoid_1∙
,sequential_11/lstm_22/while/lstm_cell_22/mulMul6sequential_11/lstm_22/while/lstm_cell_22/Sigmoid_1:y:0)sequential_11_lstm_22_while_placeholder_3*
T0*(
_output_shapes
:         Д2.
,sequential_11/lstm_22/while/lstm_cell_22/mul╥
-sequential_11/lstm_22/while/lstm_cell_22/ReluRelu7sequential_11/lstm_22/while/lstm_cell_22/split:output:2*
T0*(
_output_shapes
:         Д2/
-sequential_11/lstm_22/while/lstm_cell_22/ReluН
.sequential_11/lstm_22/while/lstm_cell_22/mul_1Mul4sequential_11/lstm_22/while/lstm_cell_22/Sigmoid:y:0;sequential_11/lstm_22/while/lstm_cell_22/Relu:activations:0*
T0*(
_output_shapes
:         Д20
.sequential_11/lstm_22/while/lstm_cell_22/mul_1В
.sequential_11/lstm_22/while/lstm_cell_22/add_1AddV20sequential_11/lstm_22/while/lstm_cell_22/mul:z:02sequential_11/lstm_22/while/lstm_cell_22/mul_1:z:0*
T0*(
_output_shapes
:         Д20
.sequential_11/lstm_22/while/lstm_cell_22/add_1▀
2sequential_11/lstm_22/while/lstm_cell_22/Sigmoid_2Sigmoid7sequential_11/lstm_22/while/lstm_cell_22/split:output:3*
T0*(
_output_shapes
:         Д24
2sequential_11/lstm_22/while/lstm_cell_22/Sigmoid_2╤
/sequential_11/lstm_22/while/lstm_cell_22/Relu_1Relu2sequential_11/lstm_22/while/lstm_cell_22/add_1:z:0*
T0*(
_output_shapes
:         Д21
/sequential_11/lstm_22/while/lstm_cell_22/Relu_1С
.sequential_11/lstm_22/while/lstm_cell_22/mul_2Mul6sequential_11/lstm_22/while/lstm_cell_22/Sigmoid_2:y:0=sequential_11/lstm_22/while/lstm_cell_22/Relu_1:activations:0*
T0*(
_output_shapes
:         Д20
.sequential_11/lstm_22/while/lstm_cell_22/mul_2╬
@sequential_11/lstm_22/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem)sequential_11_lstm_22_while_placeholder_1'sequential_11_lstm_22_while_placeholder2sequential_11/lstm_22/while/lstm_cell_22/mul_2:z:0*
_output_shapes
: *
element_dtype02B
@sequential_11/lstm_22/while/TensorArrayV2Write/TensorListSetItemИ
!sequential_11/lstm_22/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2#
!sequential_11/lstm_22/while/add/y┴
sequential_11/lstm_22/while/addAddV2'sequential_11_lstm_22_while_placeholder*sequential_11/lstm_22/while/add/y:output:0*
T0*
_output_shapes
: 2!
sequential_11/lstm_22/while/addМ
#sequential_11/lstm_22/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2%
#sequential_11/lstm_22/while/add_1/yф
!sequential_11/lstm_22/while/add_1AddV2Dsequential_11_lstm_22_while_sequential_11_lstm_22_while_loop_counter,sequential_11/lstm_22/while/add_1/y:output:0*
T0*
_output_shapes
: 2#
!sequential_11/lstm_22/while/add_1├
$sequential_11/lstm_22/while/IdentityIdentity%sequential_11/lstm_22/while/add_1:z:0!^sequential_11/lstm_22/while/NoOp*
T0*
_output_shapes
: 2&
$sequential_11/lstm_22/while/Identityь
&sequential_11/lstm_22/while/Identity_1IdentityJsequential_11_lstm_22_while_sequential_11_lstm_22_while_maximum_iterations!^sequential_11/lstm_22/while/NoOp*
T0*
_output_shapes
: 2(
&sequential_11/lstm_22/while/Identity_1┼
&sequential_11/lstm_22/while/Identity_2Identity#sequential_11/lstm_22/while/add:z:0!^sequential_11/lstm_22/while/NoOp*
T0*
_output_shapes
: 2(
&sequential_11/lstm_22/while/Identity_2Є
&sequential_11/lstm_22/while/Identity_3IdentityPsequential_11/lstm_22/while/TensorArrayV2Write/TensorListSetItem:output_handle:0!^sequential_11/lstm_22/while/NoOp*
T0*
_output_shapes
: 2(
&sequential_11/lstm_22/while/Identity_3ц
&sequential_11/lstm_22/while/Identity_4Identity2sequential_11/lstm_22/while/lstm_cell_22/mul_2:z:0!^sequential_11/lstm_22/while/NoOp*
T0*(
_output_shapes
:         Д2(
&sequential_11/lstm_22/while/Identity_4ц
&sequential_11/lstm_22/while/Identity_5Identity2sequential_11/lstm_22/while/lstm_cell_22/add_1:z:0!^sequential_11/lstm_22/while/NoOp*
T0*(
_output_shapes
:         Д2(
&sequential_11/lstm_22/while/Identity_5╠
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
&sequential_11_lstm_22_while_identity_5/sequential_11/lstm_22/while/Identity_5:output:0"Ц
Hsequential_11_lstm_22_while_lstm_cell_22_biasadd_readvariableop_resourceJsequential_11_lstm_22_while_lstm_cell_22_biasadd_readvariableop_resource_0"Ш
Isequential_11_lstm_22_while_lstm_cell_22_matmul_1_readvariableop_resourceKsequential_11_lstm_22_while_lstm_cell_22_matmul_1_readvariableop_resource_0"Ф
Gsequential_11_lstm_22_while_lstm_cell_22_matmul_readvariableop_resourceIsequential_11_lstm_22_while_lstm_cell_22_matmul_readvariableop_resource_0"И
Asequential_11_lstm_22_while_sequential_11_lstm_22_strided_slice_1Csequential_11_lstm_22_while_sequential_11_lstm_22_strided_slice_1_0"А
}sequential_11_lstm_22_while_tensorarrayv2read_tensorlistgetitem_sequential_11_lstm_22_tensorarrayunstack_tensorlistfromtensorsequential_11_lstm_22_while_tensorarrayv2read_tensorlistgetitem_sequential_11_lstm_22_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :         Д:         Д: : : : : 2В
?sequential_11/lstm_22/while/lstm_cell_22/BiasAdd/ReadVariableOp?sequential_11/lstm_22/while/lstm_cell_22/BiasAdd/ReadVariableOp2А
>sequential_11/lstm_22/while/lstm_cell_22/MatMul/ReadVariableOp>sequential_11/lstm_22/while/lstm_cell_22/MatMul/ReadVariableOp2Д
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
:         Д:.*
(
_output_shapes
:         Д:

_output_shapes
: :

_output_shapes
: "иL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*┐
serving_defaultл
K
lstm_22_input:
serving_default_lstm_22_input:0         ]@
dense_114
StatefulPartitionedCall:0         tensorflow/serving/predict:т▓
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
	variables
trainable_variables
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
	variables
trainable_variables
regularization_losses
	keras_api
*t&call_and_return_all_conditional_losses
u__call__"
_tf_keras_layer
╗

 kernel
!bias
"	variables
#trainable_variables
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

,layers
-layer_metrics
.layer_regularization_losses
/non_trainable_variables
0metrics
trainable_variables
	variables
	regularization_losses
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
2	variables
3trainable_variables
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

6states

7layers
8layer_metrics
9layer_regularization_losses
:non_trainable_variables
;metrics
trainable_variables
	variables
regularization_losses
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
	variables
=layer_metrics
>non_trainable_variables
?metrics
trainable_variables
@layer_regularization_losses
regularization_losses
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
B	variables
Ctrainable_variables
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

Fstates

Glayers
Hlayer_metrics
Ilayer_regularization_losses
Jnon_trainable_variables
Kmetrics
trainable_variables
	variables
regularization_losses
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
	variables
Mlayer_metrics
Nnon_trainable_variables
Ometrics
trainable_variables
Player_regularization_losses
regularization_losses
u__call__
*t&call_and_return_all_conditional_losses
&t"call_and_return_conditional_losses"
_generic_user_object
!:m2dense_11/kernel
:2dense_11/bias
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

Qlayers
"	variables
Rlayer_metrics
Snon_trainable_variables
Tmetrics
#trainable_variables
Ulayer_regularization_losses
$regularization_losses
w__call__
*v&call_and_return_all_conditional_losses
&v"call_and_return_conditional_losses"
_generic_user_object
.:,	]Р2lstm_22/lstm_cell_22/kernel
9:7
ДР2%lstm_22/lstm_cell_22/recurrent_kernel
(:&Р2lstm_22/lstm_cell_22/bias
/:-
Д┤2lstm_23/lstm_cell_23/kernel
8:6	m┤2%lstm_23/lstm_cell_23/recurrent_kernel
(:&┤2lstm_23/lstm_cell_23/bias
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

Xlayers
2	variables
Ylayer_metrics
Znon_trainable_variables
[metrics
3trainable_variables
\layer_regularization_losses
4regularization_losses
z__call__
*y&call_and_return_all_conditional_losses
&y"call_and_return_conditional_losses"
_generic_user_object
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

]layers
B	variables
^layer_metrics
_non_trainable_variables
`metrics
Ctrainable_variables
alayer_regularization_losses
Dregularization_losses
|__call__
*{&call_and_return_all_conditional_losses
&{"call_and_return_conditional_losses"
_generic_user_object
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
#__inference__wrapped_model_39108092lstm_22_input"Ш
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
K__inference_sequential_11_layer_call_and_return_conditional_losses_39110636
K__inference_sequential_11_layer_call_and_return_conditional_losses_39110977
K__inference_sequential_11_layer_call_and_return_conditional_losses_39110261
K__inference_sequential_11_layer_call_and_return_conditional_losses_39110286└
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
0__inference_sequential_11_layer_call_fn_39109746
0__inference_sequential_11_layer_call_fn_39110998
0__inference_sequential_11_layer_call_fn_39111019
0__inference_sequential_11_layer_call_fn_39110236└
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
E__inference_lstm_22_layer_call_and_return_conditional_losses_39111170
E__inference_lstm_22_layer_call_and_return_conditional_losses_39111321
E__inference_lstm_22_layer_call_and_return_conditional_losses_39111472
E__inference_lstm_22_layer_call_and_return_conditional_losses_39111623╒
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
*__inference_lstm_22_layer_call_fn_39111634
*__inference_lstm_22_layer_call_fn_39111645
*__inference_lstm_22_layer_call_fn_39111656
*__inference_lstm_22_layer_call_fn_39111667╒
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
H__inference_dropout_22_layer_call_and_return_conditional_losses_39111672
H__inference_dropout_22_layer_call_and_return_conditional_losses_39111684┤
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
-__inference_dropout_22_layer_call_fn_39111689
-__inference_dropout_22_layer_call_fn_39111694┤
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
E__inference_lstm_23_layer_call_and_return_conditional_losses_39111845
E__inference_lstm_23_layer_call_and_return_conditional_losses_39111996
E__inference_lstm_23_layer_call_and_return_conditional_losses_39112147
E__inference_lstm_23_layer_call_and_return_conditional_losses_39112298╒
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
*__inference_lstm_23_layer_call_fn_39112309
*__inference_lstm_23_layer_call_fn_39112320
*__inference_lstm_23_layer_call_fn_39112331
*__inference_lstm_23_layer_call_fn_39112342╒
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
H__inference_dropout_23_layer_call_and_return_conditional_losses_39112347
H__inference_dropout_23_layer_call_and_return_conditional_losses_39112359┤
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
-__inference_dropout_23_layer_call_fn_39112364
-__inference_dropout_23_layer_call_fn_39112369┤
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
F__inference_dense_11_layer_call_and_return_conditional_losses_39112400в
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
+__inference_dense_11_layer_call_fn_39112409в
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
&__inference_signature_wrapper_39110309lstm_22_input"Ф
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
J__inference_lstm_cell_22_layer_call_and_return_conditional_losses_39112441
J__inference_lstm_cell_22_layer_call_and_return_conditional_losses_39112473╛
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
/__inference_lstm_cell_22_layer_call_fn_39112490
/__inference_lstm_cell_22_layer_call_fn_39112507╛
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
J__inference_lstm_cell_23_layer_call_and_return_conditional_losses_39112539
J__inference_lstm_cell_23_layer_call_and_return_conditional_losses_39112571╛
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
/__inference_lstm_cell_23_layer_call_fn_39112588
/__inference_lstm_cell_23_layer_call_fn_39112605╛
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
#__inference__wrapped_model_39108092&'()*+ !:в7
0в-
+К(
lstm_22_input         ]
к "7к4
2
dense_11&К#
dense_11         о
F__inference_dense_11_layer_call_and_return_conditional_losses_39112400d !3в0
)в&
$К!
inputs         m
к ")в&
К
0         
Ъ Ж
+__inference_dense_11_layer_call_fn_39112409W !3в0
)в&
$К!
inputs         m
к "К         ▓
H__inference_dropout_22_layer_call_and_return_conditional_losses_39111672f8в5
.в+
%К"
inputs         Д
p 
к "*в'
 К
0         Д
Ъ ▓
H__inference_dropout_22_layer_call_and_return_conditional_losses_39111684f8в5
.в+
%К"
inputs         Д
p
к "*в'
 К
0         Д
Ъ К
-__inference_dropout_22_layer_call_fn_39111689Y8в5
.в+
%К"
inputs         Д
p 
к "К         ДК
-__inference_dropout_22_layer_call_fn_39111694Y8в5
.в+
%К"
inputs         Д
p
к "К         Д░
H__inference_dropout_23_layer_call_and_return_conditional_losses_39112347d7в4
-в*
$К!
inputs         m
p 
к ")в&
К
0         m
Ъ ░
H__inference_dropout_23_layer_call_and_return_conditional_losses_39112359d7в4
-в*
$К!
inputs         m
p
к ")в&
К
0         m
Ъ И
-__inference_dropout_23_layer_call_fn_39112364W7в4
-в*
$К!
inputs         m
p 
к "К         mИ
-__inference_dropout_23_layer_call_fn_39112369W7в4
-в*
$К!
inputs         m
p
к "К         m╒
E__inference_lstm_22_layer_call_and_return_conditional_losses_39111170Л&'(OвL
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
0                  Д
Ъ ╒
E__inference_lstm_22_layer_call_and_return_conditional_losses_39111321Л&'(OвL
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
0                  Д
Ъ ╗
E__inference_lstm_22_layer_call_and_return_conditional_losses_39111472r&'(?в<
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
0         Д
Ъ ╗
E__inference_lstm_22_layer_call_and_return_conditional_losses_39111623r&'(?в<
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
0         Д
Ъ м
*__inference_lstm_22_layer_call_fn_39111634~&'(OвL
EвB
4Ъ1
/К,
inputs/0                  ]

 
p 

 
к "&К#                  Дм
*__inference_lstm_22_layer_call_fn_39111645~&'(OвL
EвB
4Ъ1
/К,
inputs/0                  ]

 
p

 
к "&К#                  ДУ
*__inference_lstm_22_layer_call_fn_39111656e&'(?в<
5в2
$К!
inputs         ]

 
p 

 
к "К         ДУ
*__inference_lstm_22_layer_call_fn_39111667e&'(?в<
5в2
$К!
inputs         ]

 
p

 
к "К         Д╒
E__inference_lstm_23_layer_call_and_return_conditional_losses_39111845Л)*+PвM
FвC
5Ъ2
0К-
inputs/0                  Д

 
p 

 
к "2в/
(К%
0                  m
Ъ ╒
E__inference_lstm_23_layer_call_and_return_conditional_losses_39111996Л)*+PвM
FвC
5Ъ2
0К-
inputs/0                  Д

 
p

 
к "2в/
(К%
0                  m
Ъ ╗
E__inference_lstm_23_layer_call_and_return_conditional_losses_39112147r)*+@в=
6в3
%К"
inputs         Д

 
p 

 
к ")в&
К
0         m
Ъ ╗
E__inference_lstm_23_layer_call_and_return_conditional_losses_39112298r)*+@в=
6в3
%К"
inputs         Д

 
p

 
к ")в&
К
0         m
Ъ м
*__inference_lstm_23_layer_call_fn_39112309~)*+PвM
FвC
5Ъ2
0К-
inputs/0                  Д

 
p 

 
к "%К"                  mм
*__inference_lstm_23_layer_call_fn_39112320~)*+PвM
FвC
5Ъ2
0К-
inputs/0                  Д

 
p

 
к "%К"                  mУ
*__inference_lstm_23_layer_call_fn_39112331e)*+@в=
6в3
%К"
inputs         Д

 
p 

 
к "К         mУ
*__inference_lstm_23_layer_call_fn_39112342e)*+@в=
6в3
%К"
inputs         Д

 
p

 
к "К         m╤
J__inference_lstm_cell_22_layer_call_and_return_conditional_losses_39112441В&'(Вв
xвu
 К
inputs         ]
MвJ
#К 
states/0         Д
#К 
states/1         Д
p 
к "vвs
lвi
К
0/0         Д
GЪD
 К
0/1/0         Д
 К
0/1/1         Д
Ъ ╤
J__inference_lstm_cell_22_layer_call_and_return_conditional_losses_39112473В&'(Вв
xвu
 К
inputs         ]
MвJ
#К 
states/0         Д
#К 
states/1         Д
p
к "vвs
lвi
К
0/0         Д
GЪD
 К
0/1/0         Д
 К
0/1/1         Д
Ъ ж
/__inference_lstm_cell_22_layer_call_fn_39112490Є&'(Вв
xвu
 К
inputs         ]
MвJ
#К 
states/0         Д
#К 
states/1         Д
p 
к "fвc
К
0         Д
CЪ@
К
1/0         Д
К
1/1         Дж
/__inference_lstm_cell_22_layer_call_fn_39112507Є&'(Вв
xвu
 К
inputs         ]
MвJ
#К 
states/0         Д
#К 
states/1         Д
p
к "fвc
К
0         Д
CЪ@
К
1/0         Д
К
1/1         Д═
J__inference_lstm_cell_23_layer_call_and_return_conditional_losses_39112539■)*+Бв~
wвt
!К
inputs         Д
KвH
"К
states/0         m
"К
states/1         m
p 
к "sвp
iвf
К
0/0         m
EЪB
К
0/1/0         m
К
0/1/1         m
Ъ ═
J__inference_lstm_cell_23_layer_call_and_return_conditional_losses_39112571■)*+Бв~
wвt
!К
inputs         Д
KвH
"К
states/0         m
"К
states/1         m
p
к "sвp
iвf
К
0/0         m
EЪB
К
0/1/0         m
К
0/1/1         m
Ъ в
/__inference_lstm_cell_23_layer_call_fn_39112588ю)*+Бв~
wвt
!К
inputs         Д
KвH
"К
states/0         m
"К
states/1         m
p 
к "cв`
К
0         m
AЪ>
К
1/0         m
К
1/1         mв
/__inference_lstm_cell_23_layer_call_fn_39112605ю)*+Бв~
wвt
!К
inputs         Д
KвH
"К
states/0         m
"К
states/1         m
p
к "cв`
К
0         m
AЪ>
К
1/0         m
К
1/1         m╚
K__inference_sequential_11_layer_call_and_return_conditional_losses_39110261y&'()*+ !Bв?
8в5
+К(
lstm_22_input         ]
p 

 
к ")в&
К
0         
Ъ ╚
K__inference_sequential_11_layer_call_and_return_conditional_losses_39110286y&'()*+ !Bв?
8в5
+К(
lstm_22_input         ]
p

 
к ")в&
К
0         
Ъ ┴
K__inference_sequential_11_layer_call_and_return_conditional_losses_39110636r&'()*+ !;в8
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
K__inference_sequential_11_layer_call_and_return_conditional_losses_39110977r&'()*+ !;в8
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
0__inference_sequential_11_layer_call_fn_39109746l&'()*+ !Bв?
8в5
+К(
lstm_22_input         ]
p 

 
к "К         а
0__inference_sequential_11_layer_call_fn_39110236l&'()*+ !Bв?
8в5
+К(
lstm_22_input         ]
p

 
к "К         Щ
0__inference_sequential_11_layer_call_fn_39110998e&'()*+ !;в8
1в.
$К!
inputs         ]
p 

 
к "К         Щ
0__inference_sequential_11_layer_call_fn_39111019e&'()*+ !;в8
1в.
$К!
inputs         ]
p

 
к "К         ╗
&__inference_signature_wrapper_39110309Р&'()*+ !KвH
в 
Aк>
<
lstm_22_input+К(
lstm_22_input         ]"7к4
2
dense_11&К#
dense_11         