&
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
"serve*2.6.02v2.6.0-rc2-32-g919f693420e8·$
x
dense_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:r*
shared_namedense_7/kernel
q
"dense_7/kernel/Read/ReadVariableOpReadVariableOpdense_7/kernel*
_output_shapes

:r*
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

lstm_14/lstm_cell_14/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	] *,
shared_namelstm_14/lstm_cell_14/kernel

/lstm_14/lstm_cell_14/kernel/Read/ReadVariableOpReadVariableOplstm_14/lstm_cell_14/kernel*
_output_shapes
:	] *
dtype0
¨
%lstm_14/lstm_cell_14/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
 *6
shared_name'%lstm_14/lstm_cell_14/recurrent_kernel
¡
9lstm_14/lstm_cell_14/recurrent_kernel/Read/ReadVariableOpReadVariableOp%lstm_14/lstm_cell_14/recurrent_kernel* 
_output_shapes
:
 *
dtype0

lstm_14/lstm_cell_14/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: **
shared_namelstm_14/lstm_cell_14/bias

-lstm_14/lstm_cell_14/bias/Read/ReadVariableOpReadVariableOplstm_14/lstm_cell_14/bias*
_output_shapes	
: *
dtype0

lstm_15/lstm_cell_15/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
È*,
shared_namelstm_15/lstm_cell_15/kernel

/lstm_15/lstm_cell_15/kernel/Read/ReadVariableOpReadVariableOplstm_15/lstm_cell_15/kernel* 
_output_shapes
:
È*
dtype0
§
%lstm_15/lstm_cell_15/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	rÈ*6
shared_name'%lstm_15/lstm_cell_15/recurrent_kernel
 
9lstm_15/lstm_cell_15/recurrent_kernel/Read/ReadVariableOpReadVariableOp%lstm_15/lstm_cell_15/recurrent_kernel*
_output_shapes
:	rÈ*
dtype0

lstm_15/lstm_cell_15/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:È**
shared_namelstm_15/lstm_cell_15/bias

-lstm_15/lstm_cell_15/bias/Read/ReadVariableOpReadVariableOplstm_15/lstm_cell_15/bias*
_output_shapes	
:È*
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
¥"
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*à!
valueÖ!BÓ! BÌ!
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
,layer_metrics
trainable_variables

-layers
.layer_regularization_losses
/metrics
regularization_losses
0non_trainable_variables
		variables
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
6layer_metrics

7states
trainable_variables

8layers
9layer_regularization_losses
:metrics
regularization_losses
;non_trainable_variables
	variables
 
 
 
­
<layer_metrics
trainable_variables

=layers
>layer_regularization_losses
?metrics
regularization_losses
@non_trainable_variables
	variables
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
Flayer_metrics

Gstates
trainable_variables

Hlayers
Ilayer_regularization_losses
Jmetrics
regularization_losses
Knon_trainable_variables
	variables
 
 
 
­
Llayer_metrics
trainable_variables

Mlayers
Nlayer_regularization_losses
Ometrics
regularization_losses
Pnon_trainable_variables
	variables
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
­
Qlayer_metrics
"trainable_variables

Rlayers
Slayer_regularization_losses
Tmetrics
#regularization_losses
Unon_trainable_variables
$	variables
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
Xlayer_metrics
2trainable_variables

Ylayers
Zlayer_regularization_losses
[metrics
3regularization_losses
\non_trainable_variables
4	variables
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
 

)0
*1
+2
­
]layer_metrics
Btrainable_variables

^layers
_layer_regularization_losses
`metrics
Cregularization_losses
anon_trainable_variables
D	variables
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

serving_default_lstm_14_inputPlaceholder*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]*
dtype0* 
shape:ÿÿÿÿÿÿÿÿÿ]
ª
StatefulPartitionedCallStatefulPartitionedCallserving_default_lstm_14_inputlstm_14/lstm_cell_14/kernel%lstm_14/lstm_cell_14/recurrent_kernellstm_14/lstm_cell_14/biaslstm_15/lstm_cell_15/kernel%lstm_15/lstm_cell_15/recurrent_kernellstm_15/lstm_cell_15/biasdense_7/kerneldense_7/bias*
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
&__inference_signature_wrapper_26077161
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 

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
GPU 2J 8 **
f%R#
!__inference__traced_save_26079516
 
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
GPU 2J 8 *-
f(R&
$__inference__traced_restore_26079562Èì#
Ô

í
lstm_14_while_cond_26077227,
(lstm_14_while_lstm_14_while_loop_counter2
.lstm_14_while_lstm_14_while_maximum_iterations
lstm_14_while_placeholder
lstm_14_while_placeholder_1
lstm_14_while_placeholder_2
lstm_14_while_placeholder_3.
*lstm_14_while_less_lstm_14_strided_slice_1F
Blstm_14_while_lstm_14_while_cond_26077227___redundant_placeholder0F
Blstm_14_while_lstm_14_while_cond_26077227___redundant_placeholder1F
Blstm_14_while_lstm_14_while_cond_26077227___redundant_placeholder2F
Blstm_14_while_lstm_14_while_cond_26077227___redundant_placeholder3
lstm_14_while_identity

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
B: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 

_output_shapes
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
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
ã
Í
while_cond_26076276
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_26076276___redundant_placeholder06
2while_while_cond_26076276___redundant_placeholder16
2while_while_cond_26076276___redundant_placeholder26
2while_while_cond_26076276___redundant_placeholder3
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
B: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 

_output_shapes
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
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:

f
H__inference_dropout_14_layer_call_and_return_conditional_losses_26076374

inputs

identity_1_
IdentityIdentityinputs*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityn

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
°?
Ô
while_body_26076277
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_14_matmul_readvariableop_resource_0:	] I
5while_lstm_cell_14_matmul_1_readvariableop_resource_0:
 C
4while_lstm_cell_14_biasadd_readvariableop_resource_0:	 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_14_matmul_readvariableop_resource:	] G
3while_lstm_cell_14_matmul_1_readvariableop_resource:
 A
2while_lstm_cell_14_biasadd_readvariableop_resource:	 ¢)while/lstm_cell_14/BiasAdd/ReadVariableOp¢(while/lstm_cell_14/MatMul/ReadVariableOp¢*while/lstm_cell_14/MatMul_1/ReadVariableOpÃ
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
(while/lstm_cell_14/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_14_matmul_readvariableop_resource_0*
_output_shapes
:	] *
dtype02*
(while/lstm_cell_14/MatMul/ReadVariableOp×
while/lstm_cell_14/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_14/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/lstm_cell_14/MatMulÐ
*while/lstm_cell_14/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_14_matmul_1_readvariableop_resource_0* 
_output_shapes
:
 *
dtype02,
*while/lstm_cell_14/MatMul_1/ReadVariableOpÀ
while/lstm_cell_14/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_14/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/lstm_cell_14/MatMul_1¸
while/lstm_cell_14/addAddV2#while/lstm_cell_14/MatMul:product:0%while/lstm_cell_14/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/lstm_cell_14/addÈ
)while/lstm_cell_14/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_14_biasadd_readvariableop_resource_0*
_output_shapes	
: *
dtype02+
)while/lstm_cell_14/BiasAdd/ReadVariableOpÅ
while/lstm_cell_14/BiasAddBiasAddwhile/lstm_cell_14/add:z:01while/lstm_cell_14/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/lstm_cell_14/BiasAdd
"while/lstm_cell_14/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_14/split/split_dim
while/lstm_cell_14/splitSplit+while/lstm_cell_14/split/split_dim:output:0#while/lstm_cell_14/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2
while/lstm_cell_14/split
while/lstm_cell_14/SigmoidSigmoid!while/lstm_cell_14/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_14/Sigmoid
while/lstm_cell_14/Sigmoid_1Sigmoid!while/lstm_cell_14/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_14/Sigmoid_1¡
while/lstm_cell_14/mulMul while/lstm_cell_14/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_14/mul
while/lstm_cell_14/ReluRelu!while/lstm_cell_14/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_14/Reluµ
while/lstm_cell_14/mul_1Mulwhile/lstm_cell_14/Sigmoid:y:0%while/lstm_cell_14/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_14/mul_1ª
while/lstm_cell_14/add_1AddV2while/lstm_cell_14/mul:z:0while/lstm_cell_14/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_14/add_1
while/lstm_cell_14/Sigmoid_2Sigmoid!while/lstm_cell_14/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_14/Sigmoid_2
while/lstm_cell_14/Relu_1Reluwhile/lstm_cell_14/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_14/Relu_1¹
while/lstm_cell_14/mul_2Mul while/lstm_cell_14/Sigmoid_2:y:0'while/lstm_cell_14/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_14/mul_2à
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
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_14/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_14/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/Identity_5Þ

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
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2V
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
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
°?
Ô
while_body_26078391
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_14_matmul_readvariableop_resource_0:	] I
5while_lstm_cell_14_matmul_1_readvariableop_resource_0:
 C
4while_lstm_cell_14_biasadd_readvariableop_resource_0:	 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_14_matmul_readvariableop_resource:	] G
3while_lstm_cell_14_matmul_1_readvariableop_resource:
 A
2while_lstm_cell_14_biasadd_readvariableop_resource:	 ¢)while/lstm_cell_14/BiasAdd/ReadVariableOp¢(while/lstm_cell_14/MatMul/ReadVariableOp¢*while/lstm_cell_14/MatMul_1/ReadVariableOpÃ
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
(while/lstm_cell_14/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_14_matmul_readvariableop_resource_0*
_output_shapes
:	] *
dtype02*
(while/lstm_cell_14/MatMul/ReadVariableOp×
while/lstm_cell_14/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_14/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/lstm_cell_14/MatMulÐ
*while/lstm_cell_14/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_14_matmul_1_readvariableop_resource_0* 
_output_shapes
:
 *
dtype02,
*while/lstm_cell_14/MatMul_1/ReadVariableOpÀ
while/lstm_cell_14/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_14/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/lstm_cell_14/MatMul_1¸
while/lstm_cell_14/addAddV2#while/lstm_cell_14/MatMul:product:0%while/lstm_cell_14/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/lstm_cell_14/addÈ
)while/lstm_cell_14/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_14_biasadd_readvariableop_resource_0*
_output_shapes	
: *
dtype02+
)while/lstm_cell_14/BiasAdd/ReadVariableOpÅ
while/lstm_cell_14/BiasAddBiasAddwhile/lstm_cell_14/add:z:01while/lstm_cell_14/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/lstm_cell_14/BiasAdd
"while/lstm_cell_14/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_14/split/split_dim
while/lstm_cell_14/splitSplit+while/lstm_cell_14/split/split_dim:output:0#while/lstm_cell_14/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2
while/lstm_cell_14/split
while/lstm_cell_14/SigmoidSigmoid!while/lstm_cell_14/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_14/Sigmoid
while/lstm_cell_14/Sigmoid_1Sigmoid!while/lstm_cell_14/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_14/Sigmoid_1¡
while/lstm_cell_14/mulMul while/lstm_cell_14/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_14/mul
while/lstm_cell_14/ReluRelu!while/lstm_cell_14/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_14/Reluµ
while/lstm_cell_14/mul_1Mulwhile/lstm_cell_14/Sigmoid:y:0%while/lstm_cell_14/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_14/mul_1ª
while/lstm_cell_14/add_1AddV2while/lstm_cell_14/mul:z:0while/lstm_cell_14/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_14/add_1
while/lstm_cell_14/Sigmoid_2Sigmoid!while/lstm_cell_14/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_14/Sigmoid_2
while/lstm_cell_14/Relu_1Reluwhile/lstm_cell_14/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_14/Relu_1¹
while/lstm_cell_14/mul_2Mul while/lstm_cell_14/Sigmoid_2:y:0'while/lstm_cell_14/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_14/mul_2à
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
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_14/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_14/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/Identity_5Þ

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
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2V
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
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
¾F

E__inference_lstm_15_layer_call_and_return_conditional_losses_26075732

inputs)
lstm_cell_15_26075650:
È(
lstm_cell_15_26075652:	rÈ$
lstm_cell_15_26075654:	È
identity¢$lstm_cell_15/StatefulPartitionedCall¢whileD
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
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :r2
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

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :r2
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
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :r2
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
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :r2
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
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2	
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
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
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
valueB"ÿÿÿÿ  27
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
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
strided_slice_2¥
$lstm_cell_15/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_15_26075650lstm_cell_15_26075652lstm_cell_15_26075654*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿr:ÿÿÿÿÿÿÿÿÿr:ÿÿÿÿÿÿÿÿÿr*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_lstm_cell_15_layer_call_and_return_conditional_losses_260756492&
$lstm_cell_15/StatefulPartitionedCall
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿr   2
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
while/loop_counterÊ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_15_26075650lstm_cell_15_26075652lstm_cell_15_26075654*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿr:ÿÿÿÿÿÿÿÿÿr: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_26075663*
condR
while_cond_26075662*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿr:ÿÿÿÿÿÿÿÿÿr: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿr   22
0TensorArrayV2Stack/TensorListStack/element_shapeñ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿr*
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
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm®
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿr2
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
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿr2

Identity}
NoOpNoOp%^lstm_cell_15/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 2L
$lstm_cell_15/StatefulPartitionedCall$lstm_cell_15/StatefulPartitionedCall2
whilewhile:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¤
µ
J__inference_sequential_7_layer_call_and_return_conditional_losses_26077048

inputs#
lstm_14_26077026:	] $
lstm_14_26077028:
 
lstm_14_26077030:	 $
lstm_15_26077034:
È#
lstm_15_26077036:	rÈ
lstm_15_26077038:	È"
dense_7_26077042:r
dense_7_26077044:
identity¢dense_7/StatefulPartitionedCall¢"dropout_14/StatefulPartitionedCall¢"dropout_15/StatefulPartitionedCall¢lstm_14/StatefulPartitionedCall¢lstm_15/StatefulPartitionedCall®
lstm_14/StatefulPartitionedCallStatefulPartitionedCallinputslstm_14_26077026lstm_14_26077028lstm_14_26077030*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_lstm_14_layer_call_and_return_conditional_losses_260769912!
lstm_14/StatefulPartitionedCall
"dropout_14/StatefulPartitionedCallStatefulPartitionedCall(lstm_14/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_dropout_14_layer_call_and_return_conditional_losses_260768242$
"dropout_14/StatefulPartitionedCallÒ
lstm_15/StatefulPartitionedCallStatefulPartitionedCall+dropout_14/StatefulPartitionedCall:output:0lstm_15_26077034lstm_15_26077036lstm_15_26077038*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿr*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_lstm_15_layer_call_and_return_conditional_losses_260767952!
lstm_15/StatefulPartitionedCall¿
"dropout_15/StatefulPartitionedCallStatefulPartitionedCall(lstm_15/StatefulPartitionedCall:output:0#^dropout_14/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿr* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_dropout_15_layer_call_and_return_conditional_losses_260766282$
"dropout_15/StatefulPartitionedCall¾
dense_7/StatefulPartitionedCallStatefulPartitionedCall+dropout_15/StatefulPartitionedCall:output:0dense_7_26077042dense_7_26077044*
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
GPU 2J 8 *N
fIRG
E__inference_dense_7_layer_call_and_return_conditional_losses_260765722!
dense_7/StatefulPartitionedCall
IdentityIdentity(dense_7/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityþ
NoOpNoOp ^dense_7/StatefulPartitionedCall#^dropout_14/StatefulPartitionedCall#^dropout_15/StatefulPartitionedCall ^lstm_14/StatefulPartitionedCall ^lstm_15/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ]: : : : : : : : 2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall2H
"dropout_14/StatefulPartitionedCall"dropout_14/StatefulPartitionedCall2H
"dropout_15/StatefulPartitionedCall"dropout_15/StatefulPartitionedCall2B
lstm_14/StatefulPartitionedCalllstm_14/StatefulPartitionedCall2B
lstm_15/StatefulPartitionedCalllstm_15/StatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
 
_user_specified_nameinputs
ð%
Þ
!__inference__traced_save_26079516
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
SaveV2/shape_and_slices
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0)savev2_dense_7_kernel_read_readvariableop'savev2_dense_7_bias_read_readvariableop6savev2_lstm_14_lstm_cell_14_kernel_read_readvariableop@savev2_lstm_14_lstm_cell_14_recurrent_kernel_read_readvariableop4savev2_lstm_14_lstm_cell_14_bias_read_readvariableop6savev2_lstm_15_lstm_cell_15_kernel_read_readvariableop@savev2_lstm_15_lstm_cell_15_recurrent_kernel_read_readvariableop4savev2_lstm_15_lstm_cell_15_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableopsavev2_const"/device:CPU:0*
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

identity_1Identity_1:output:0*k
_input_shapesZ
X: :r::	] :
 : :
È:	rÈ:È: : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:r: 

_output_shapes
::%!

_output_shapes
:	] :&"
 
_output_shapes
:
 :!

_output_shapes	
: :&"
 
_output_shapes
:
È:%!

_output_shapes
:	rÈ:!

_output_shapes	
:È:	
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
ã
Í
while_cond_26075242
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_26075242___redundant_placeholder06
2while_while_cond_26075242___redundant_placeholder16
2while_while_cond_26075242___redundant_placeholder26
2while_while_cond_26075242___redundant_placeholder3
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
B: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 

_output_shapes
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
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
¬\
 
E__inference_lstm_15_layer_call_and_return_conditional_losses_26078848
inputs_0?
+lstm_cell_15_matmul_readvariableop_resource:
È@
-lstm_cell_15_matmul_1_readvariableop_resource:	rÈ;
,lstm_cell_15_biasadd_readvariableop_resource:	È
identity¢#lstm_cell_15/BiasAdd/ReadVariableOp¢"lstm_cell_15/MatMul/ReadVariableOp¢$lstm_cell_15/MatMul_1/ReadVariableOp¢whileF
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
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :r2
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

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :r2
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
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :r2
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
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :r2
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
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2	
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
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
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
valueB"ÿÿÿÿ  27
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
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
strided_slice_2¶
"lstm_cell_15/MatMul/ReadVariableOpReadVariableOp+lstm_cell_15_matmul_readvariableop_resource* 
_output_shapes
:
È*
dtype02$
"lstm_cell_15/MatMul/ReadVariableOp­
lstm_cell_15/MatMulMatMulstrided_slice_2:output:0*lstm_cell_15/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell_15/MatMul»
$lstm_cell_15/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_15_matmul_1_readvariableop_resource*
_output_shapes
:	rÈ*
dtype02&
$lstm_cell_15/MatMul_1/ReadVariableOp©
lstm_cell_15/MatMul_1MatMulzeros:output:0,lstm_cell_15/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell_15/MatMul_1 
lstm_cell_15/addAddV2lstm_cell_15/MatMul:product:0lstm_cell_15/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell_15/add´
#lstm_cell_15/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_15_biasadd_readvariableop_resource*
_output_shapes	
:È*
dtype02%
#lstm_cell_15/BiasAdd/ReadVariableOp­
lstm_cell_15/BiasAddBiasAddlstm_cell_15/add:z:0+lstm_cell_15/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell_15/BiasAdd~
lstm_cell_15/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_15/split/split_dimó
lstm_cell_15/splitSplit%lstm_cell_15/split/split_dim:output:0lstm_cell_15/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿr:ÿÿÿÿÿÿÿÿÿr:ÿÿÿÿÿÿÿÿÿr:ÿÿÿÿÿÿÿÿÿr*
	num_split2
lstm_cell_15/split
lstm_cell_15/SigmoidSigmoidlstm_cell_15/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2
lstm_cell_15/Sigmoid
lstm_cell_15/Sigmoid_1Sigmoidlstm_cell_15/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2
lstm_cell_15/Sigmoid_1
lstm_cell_15/mulMullstm_cell_15/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2
lstm_cell_15/mul}
lstm_cell_15/ReluRelulstm_cell_15/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2
lstm_cell_15/Relu
lstm_cell_15/mul_1Mullstm_cell_15/Sigmoid:y:0lstm_cell_15/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2
lstm_cell_15/mul_1
lstm_cell_15/add_1AddV2lstm_cell_15/mul:z:0lstm_cell_15/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2
lstm_cell_15/add_1
lstm_cell_15/Sigmoid_2Sigmoidlstm_cell_15/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2
lstm_cell_15/Sigmoid_2|
lstm_cell_15/Relu_1Relulstm_cell_15/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2
lstm_cell_15/Relu_1 
lstm_cell_15/mul_2Mullstm_cell_15/Sigmoid_2:y:0!lstm_cell_15/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2
lstm_cell_15/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿr   2
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
while/loop_counter
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_15_matmul_readvariableop_resource-lstm_cell_15_matmul_1_readvariableop_resource,lstm_cell_15_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿr:ÿÿÿÿÿÿÿÿÿr: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_26078764*
condR
while_cond_26078763*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿr:ÿÿÿÿÿÿÿÿÿr: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿr   22
0TensorArrayV2Stack/TensorListStack/element_shapeñ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿr*
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
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm®
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿr2
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
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿr2

IdentityÈ
NoOpNoOp$^lstm_cell_15/BiasAdd/ReadVariableOp#^lstm_cell_15/MatMul/ReadVariableOp%^lstm_cell_15/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 2J
#lstm_cell_15/BiasAdd/ReadVariableOp#lstm_cell_15/BiasAdd/ReadVariableOp2H
"lstm_cell_15/MatMul/ReadVariableOp"lstm_cell_15/MatMul/ReadVariableOp2L
$lstm_cell_15/MatMul_1/ReadVariableOp$lstm_cell_15/MatMul_1/ReadVariableOp2
whilewhile:_ [
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0
ß
Í
while_cond_26076710
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_26076710___redundant_placeholder06
2while_while_cond_26076710___redundant_placeholder16
2while_while_cond_26076710___redundant_placeholder26
2while_while_cond_26076710___redundant_placeholder3
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
@: : : : :ÿÿÿÿÿÿÿÿÿr:ÿÿÿÿÿÿÿÿÿr: ::::: 

_output_shapes
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
:ÿÿÿÿÿÿÿÿÿr:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr:

_output_shapes
: :

_output_shapes
:
×
g
H__inference_dropout_14_layer_call_and_return_conditional_losses_26076824

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
:ÿÿÿÿÿÿÿÿÿ2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape¹
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
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
:ÿÿÿÿÿÿÿÿÿ2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/Mul_1j
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ã\
 
E__inference_lstm_14_layer_call_and_return_conditional_losses_26078022
inputs_0>
+lstm_cell_14_matmul_readvariableop_resource:	] A
-lstm_cell_14_matmul_1_readvariableop_resource:
 ;
,lstm_cell_14_biasadd_readvariableop_resource:	 
identity¢#lstm_cell_14/BiasAdd/ReadVariableOp¢"lstm_cell_14/MatMul/ReadVariableOp¢$lstm_cell_14/MatMul_1/ReadVariableOp¢whileF
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
B :2
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
B :2
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
:ÿÿÿÿÿÿÿÿÿ2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :2
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
B :2
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
:ÿÿÿÿÿÿÿÿÿ2	
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
"lstm_cell_14/MatMul/ReadVariableOpReadVariableOp+lstm_cell_14_matmul_readvariableop_resource*
_output_shapes
:	] *
dtype02$
"lstm_cell_14/MatMul/ReadVariableOp­
lstm_cell_14/MatMulMatMulstrided_slice_2:output:0*lstm_cell_14/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
lstm_cell_14/MatMul¼
$lstm_cell_14/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_14_matmul_1_readvariableop_resource* 
_output_shapes
:
 *
dtype02&
$lstm_cell_14/MatMul_1/ReadVariableOp©
lstm_cell_14/MatMul_1MatMulzeros:output:0,lstm_cell_14/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
lstm_cell_14/MatMul_1 
lstm_cell_14/addAddV2lstm_cell_14/MatMul:product:0lstm_cell_14/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
lstm_cell_14/add´
#lstm_cell_14/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_14_biasadd_readvariableop_resource*
_output_shapes	
: *
dtype02%
#lstm_cell_14/BiasAdd/ReadVariableOp­
lstm_cell_14/BiasAddBiasAddlstm_cell_14/add:z:0+lstm_cell_14/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
lstm_cell_14/BiasAdd~
lstm_cell_14/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_14/split/split_dim÷
lstm_cell_14/splitSplit%lstm_cell_14/split/split_dim:output:0lstm_cell_14/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2
lstm_cell_14/split
lstm_cell_14/SigmoidSigmoidlstm_cell_14/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_14/Sigmoid
lstm_cell_14/Sigmoid_1Sigmoidlstm_cell_14/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_14/Sigmoid_1
lstm_cell_14/mulMullstm_cell_14/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_14/mul~
lstm_cell_14/ReluRelulstm_cell_14/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_14/Relu
lstm_cell_14/mul_1Mullstm_cell_14/Sigmoid:y:0lstm_cell_14/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_14/mul_1
lstm_cell_14/add_1AddV2lstm_cell_14/mul:z:0lstm_cell_14/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_14/add_1
lstm_cell_14/Sigmoid_2Sigmoidlstm_cell_14/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_14/Sigmoid_2}
lstm_cell_14/Relu_1Relulstm_cell_14/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_14/Relu_1¡
lstm_cell_14/mul_2Mullstm_cell_14/Sigmoid_2:y:0!lstm_cell_14/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_14/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ  2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_14_matmul_readvariableop_resource-lstm_cell_14_matmul_1_readvariableop_resource,lstm_cell_14_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_26077938*
condR
while_cond_26077937*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ  22
0TensorArrayV2Stack/TensorListStack/element_shapeò
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
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
:ÿÿÿÿÿÿÿÿÿ*
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
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
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
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

IdentityÈ
NoOpNoOp$^lstm_cell_14/BiasAdd/ReadVariableOp#^lstm_cell_14/MatMul/ReadVariableOp%^lstm_cell_14/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ]: : : 2J
#lstm_cell_14/BiasAdd/ReadVariableOp#lstm_cell_14/BiasAdd/ReadVariableOp2H
"lstm_cell_14/MatMul/ReadVariableOp"lstm_cell_14/MatMul/ReadVariableOp2L
$lstm_cell_14/MatMul_1/ReadVariableOp$lstm_cell_14/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ]
"
_user_specified_name
inputs/0
µ
¸
*__inference_lstm_15_layer_call_fn_26079183

inputs
unknown:
È
	unknown_0:	rÈ
	unknown_1:	È
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿr*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_lstm_15_layer_call_and_return_conditional_losses_260765262
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:ÿÿÿÿÿÿÿÿÿ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ï
g
H__inference_dropout_15_layer_call_and_return_conditional_losses_26079211

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/Constw
dropout/MulMulinputsdropout/Const:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape¸
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿr*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>2
dropout/GreaterEqual/yÂ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2
dropout/Cast~
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2
dropout/Mul_1i
IdentityIdentitydropout/Mul_1:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿr:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
 
_user_specified_nameinputs


Ê
/__inference_sequential_7_layer_call_fn_26077850

inputs
unknown:	] 
	unknown_0:
 
	unknown_1:	 
	unknown_2:
È
	unknown_3:	rÈ
	unknown_4:	È
	unknown_5:r
	unknown_6:
identity¢StatefulPartitionedCallÌ
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
GPU 2J 8 *S
fNRL
J__inference_sequential_7_layer_call_and_return_conditional_losses_260765792
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
í[

E__inference_lstm_15_layer_call_and_return_conditional_losses_26076526

inputs?
+lstm_cell_15_matmul_readvariableop_resource:
È@
-lstm_cell_15_matmul_1_readvariableop_resource:	rÈ;
,lstm_cell_15_biasadd_readvariableop_resource:	È
identity¢#lstm_cell_15/BiasAdd/ReadVariableOp¢"lstm_cell_15/MatMul/ReadVariableOp¢$lstm_cell_15/MatMul_1/ReadVariableOp¢whileD
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
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :r2
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

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :r2
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
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :r2
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
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :r2
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
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2	
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
:ÿÿÿÿÿÿÿÿÿ2
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
valueB"ÿÿÿÿ  27
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
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
strided_slice_2¶
"lstm_cell_15/MatMul/ReadVariableOpReadVariableOp+lstm_cell_15_matmul_readvariableop_resource* 
_output_shapes
:
È*
dtype02$
"lstm_cell_15/MatMul/ReadVariableOp­
lstm_cell_15/MatMulMatMulstrided_slice_2:output:0*lstm_cell_15/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell_15/MatMul»
$lstm_cell_15/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_15_matmul_1_readvariableop_resource*
_output_shapes
:	rÈ*
dtype02&
$lstm_cell_15/MatMul_1/ReadVariableOp©
lstm_cell_15/MatMul_1MatMulzeros:output:0,lstm_cell_15/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell_15/MatMul_1 
lstm_cell_15/addAddV2lstm_cell_15/MatMul:product:0lstm_cell_15/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell_15/add´
#lstm_cell_15/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_15_biasadd_readvariableop_resource*
_output_shapes	
:È*
dtype02%
#lstm_cell_15/BiasAdd/ReadVariableOp­
lstm_cell_15/BiasAddBiasAddlstm_cell_15/add:z:0+lstm_cell_15/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell_15/BiasAdd~
lstm_cell_15/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_15/split/split_dimó
lstm_cell_15/splitSplit%lstm_cell_15/split/split_dim:output:0lstm_cell_15/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿr:ÿÿÿÿÿÿÿÿÿr:ÿÿÿÿÿÿÿÿÿr:ÿÿÿÿÿÿÿÿÿr*
	num_split2
lstm_cell_15/split
lstm_cell_15/SigmoidSigmoidlstm_cell_15/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2
lstm_cell_15/Sigmoid
lstm_cell_15/Sigmoid_1Sigmoidlstm_cell_15/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2
lstm_cell_15/Sigmoid_1
lstm_cell_15/mulMullstm_cell_15/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2
lstm_cell_15/mul}
lstm_cell_15/ReluRelulstm_cell_15/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2
lstm_cell_15/Relu
lstm_cell_15/mul_1Mullstm_cell_15/Sigmoid:y:0lstm_cell_15/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2
lstm_cell_15/mul_1
lstm_cell_15/add_1AddV2lstm_cell_15/mul:z:0lstm_cell_15/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2
lstm_cell_15/add_1
lstm_cell_15/Sigmoid_2Sigmoidlstm_cell_15/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2
lstm_cell_15/Sigmoid_2|
lstm_cell_15/Relu_1Relulstm_cell_15/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2
lstm_cell_15/Relu_1 
lstm_cell_15/mul_2Mullstm_cell_15/Sigmoid_2:y:0!lstm_cell_15/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2
lstm_cell_15/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿr   2
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
while/loop_counter
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_15_matmul_readvariableop_resource-lstm_cell_15_matmul_1_readvariableop_resource,lstm_cell_15_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿr:ÿÿÿÿÿÿÿÿÿr: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_26076442*
condR
while_cond_26076441*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿr:ÿÿÿÿÿÿÿÿÿr: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿr   22
0TensorArrayV2Stack/TensorListStack/element_shapeè
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿr*
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
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm¥
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2
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
:ÿÿÿÿÿÿÿÿÿr2

IdentityÈ
NoOpNoOp$^lstm_cell_15/BiasAdd/ReadVariableOp#^lstm_cell_15/MatMul/ReadVariableOp%^lstm_cell_15/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:ÿÿÿÿÿÿÿÿÿ: : : 2J
#lstm_cell_15/BiasAdd/ReadVariableOp#lstm_cell_15/BiasAdd/ReadVariableOp2H
"lstm_cell_15/MatMul/ReadVariableOp"lstm_cell_15/MatMul/ReadVariableOp2L
$lstm_cell_15/MatMul_1/ReadVariableOp$lstm_cell_15/MatMul_1/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ù

J__inference_lstm_cell_15_layer_call_and_return_conditional_losses_26075649

inputs

states
states_12
matmul_readvariableop_resource:
È3
 matmul_1_readvariableop_resource:	rÈ.
biasadd_readvariableop_resource:	È
identity

identity_1

identity_2¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
È*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
MatMul
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	rÈ*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
add
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:È*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2	
BiasAddd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim¿
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿr:ÿÿÿÿÿÿÿÿÿr:ÿÿÿÿÿÿÿÿÿr:ÿÿÿÿÿÿÿÿÿr*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2
	Sigmoid_1\
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2
mulV
ReluRelusplit:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2
	Sigmoid_2U
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2
mul_2d
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2

Identityh

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2

Identity_1h

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2

Identity_2
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
@:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿr:ÿÿÿÿÿÿÿÿÿr: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
 
_user_specified_namestates:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
 
_user_specified_namestates
¹
¼
J__inference_sequential_7_layer_call_and_return_conditional_losses_26077138
lstm_14_input#
lstm_14_26077116:	] $
lstm_14_26077118:
 
lstm_14_26077120:	 $
lstm_15_26077124:
È#
lstm_15_26077126:	rÈ
lstm_15_26077128:	È"
dense_7_26077132:r
dense_7_26077134:
identity¢dense_7/StatefulPartitionedCall¢"dropout_14/StatefulPartitionedCall¢"dropout_15/StatefulPartitionedCall¢lstm_14/StatefulPartitionedCall¢lstm_15/StatefulPartitionedCallµ
lstm_14/StatefulPartitionedCallStatefulPartitionedCalllstm_14_inputlstm_14_26077116lstm_14_26077118lstm_14_26077120*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_lstm_14_layer_call_and_return_conditional_losses_260769912!
lstm_14/StatefulPartitionedCall
"dropout_14/StatefulPartitionedCallStatefulPartitionedCall(lstm_14/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_dropout_14_layer_call_and_return_conditional_losses_260768242$
"dropout_14/StatefulPartitionedCallÒ
lstm_15/StatefulPartitionedCallStatefulPartitionedCall+dropout_14/StatefulPartitionedCall:output:0lstm_15_26077124lstm_15_26077126lstm_15_26077128*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿr*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_lstm_15_layer_call_and_return_conditional_losses_260767952!
lstm_15/StatefulPartitionedCall¿
"dropout_15/StatefulPartitionedCallStatefulPartitionedCall(lstm_15/StatefulPartitionedCall:output:0#^dropout_14/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿr* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_dropout_15_layer_call_and_return_conditional_losses_260766282$
"dropout_15/StatefulPartitionedCall¾
dense_7/StatefulPartitionedCallStatefulPartitionedCall+dropout_15/StatefulPartitionedCall:output:0dense_7_26077132dense_7_26077134*
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
GPU 2J 8 *N
fIRG
E__inference_dense_7_layer_call_and_return_conditional_losses_260765722!
dense_7/StatefulPartitionedCall
IdentityIdentity(dense_7/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityþ
NoOpNoOp ^dense_7/StatefulPartitionedCall#^dropout_14/StatefulPartitionedCall#^dropout_15/StatefulPartitionedCall ^lstm_14/StatefulPartitionedCall ^lstm_15/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ]: : : : : : : : 2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall2H
"dropout_14/StatefulPartitionedCall"dropout_14/StatefulPartitionedCall2H
"dropout_15/StatefulPartitionedCall"dropout_15/StatefulPartitionedCall2B
lstm_14/StatefulPartitionedCalllstm_14/StatefulPartitionedCall2B
lstm_15/StatefulPartitionedCalllstm_15/StatefulPartitionedCall:Z V
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
'
_user_specified_namelstm_14_input
\

E__inference_lstm_14_layer_call_and_return_conditional_losses_26076991

inputs>
+lstm_cell_14_matmul_readvariableop_resource:	] A
-lstm_cell_14_matmul_1_readvariableop_resource:
 ;
,lstm_cell_14_biasadd_readvariableop_resource:	 
identity¢#lstm_cell_14/BiasAdd/ReadVariableOp¢"lstm_cell_14/MatMul/ReadVariableOp¢$lstm_cell_14/MatMul_1/ReadVariableOp¢whileD
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
B :2
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
B :2
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
:ÿÿÿÿÿÿÿÿÿ2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :2
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
B :2
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
:ÿÿÿÿÿÿÿÿÿ2	
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
"lstm_cell_14/MatMul/ReadVariableOpReadVariableOp+lstm_cell_14_matmul_readvariableop_resource*
_output_shapes
:	] *
dtype02$
"lstm_cell_14/MatMul/ReadVariableOp­
lstm_cell_14/MatMulMatMulstrided_slice_2:output:0*lstm_cell_14/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
lstm_cell_14/MatMul¼
$lstm_cell_14/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_14_matmul_1_readvariableop_resource* 
_output_shapes
:
 *
dtype02&
$lstm_cell_14/MatMul_1/ReadVariableOp©
lstm_cell_14/MatMul_1MatMulzeros:output:0,lstm_cell_14/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
lstm_cell_14/MatMul_1 
lstm_cell_14/addAddV2lstm_cell_14/MatMul:product:0lstm_cell_14/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
lstm_cell_14/add´
#lstm_cell_14/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_14_biasadd_readvariableop_resource*
_output_shapes	
: *
dtype02%
#lstm_cell_14/BiasAdd/ReadVariableOp­
lstm_cell_14/BiasAddBiasAddlstm_cell_14/add:z:0+lstm_cell_14/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
lstm_cell_14/BiasAdd~
lstm_cell_14/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_14/split/split_dim÷
lstm_cell_14/splitSplit%lstm_cell_14/split/split_dim:output:0lstm_cell_14/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2
lstm_cell_14/split
lstm_cell_14/SigmoidSigmoidlstm_cell_14/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_14/Sigmoid
lstm_cell_14/Sigmoid_1Sigmoidlstm_cell_14/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_14/Sigmoid_1
lstm_cell_14/mulMullstm_cell_14/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_14/mul~
lstm_cell_14/ReluRelulstm_cell_14/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_14/Relu
lstm_cell_14/mul_1Mullstm_cell_14/Sigmoid:y:0lstm_cell_14/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_14/mul_1
lstm_cell_14/add_1AddV2lstm_cell_14/mul:z:0lstm_cell_14/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_14/add_1
lstm_cell_14/Sigmoid_2Sigmoidlstm_cell_14/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_14/Sigmoid_2}
lstm_cell_14/Relu_1Relulstm_cell_14/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_14/Relu_1¡
lstm_cell_14/mul_2Mullstm_cell_14/Sigmoid_2:y:0!lstm_cell_14/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_14/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ  2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_14_matmul_readvariableop_resource-lstm_cell_14_matmul_1_readvariableop_resource,lstm_cell_14_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_26076907*
condR
while_cond_26076906*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ  22
0TensorArrayV2Stack/TensorListStack/element_shapeé
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
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
:ÿÿÿÿÿÿÿÿÿ*
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
:ÿÿÿÿÿÿÿÿÿ2
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
:ÿÿÿÿÿÿÿÿÿ2

IdentityÈ
NoOpNoOp$^lstm_cell_14/BiasAdd/ReadVariableOp#^lstm_cell_14/MatMul/ReadVariableOp%^lstm_cell_14/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ]: : : 2J
#lstm_cell_14/BiasAdd/ReadVariableOp#lstm_cell_14/BiasAdd/ReadVariableOp2H
"lstm_cell_14/MatMul/ReadVariableOp"lstm_cell_14/MatMul/ReadVariableOp2L
$lstm_cell_14/MatMul_1/ReadVariableOp$lstm_cell_14/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
 
_user_specified_nameinputs
Ø
I
-__inference_dropout_14_layer_call_fn_26078541

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
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_dropout_14_layer_call_and_return_conditional_losses_260763742
PartitionedCallq
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
\

E__inference_lstm_14_layer_call_and_return_conditional_losses_26078475

inputs>
+lstm_cell_14_matmul_readvariableop_resource:	] A
-lstm_cell_14_matmul_1_readvariableop_resource:
 ;
,lstm_cell_14_biasadd_readvariableop_resource:	 
identity¢#lstm_cell_14/BiasAdd/ReadVariableOp¢"lstm_cell_14/MatMul/ReadVariableOp¢$lstm_cell_14/MatMul_1/ReadVariableOp¢whileD
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
B :2
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
B :2
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
:ÿÿÿÿÿÿÿÿÿ2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :2
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
B :2
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
:ÿÿÿÿÿÿÿÿÿ2	
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
"lstm_cell_14/MatMul/ReadVariableOpReadVariableOp+lstm_cell_14_matmul_readvariableop_resource*
_output_shapes
:	] *
dtype02$
"lstm_cell_14/MatMul/ReadVariableOp­
lstm_cell_14/MatMulMatMulstrided_slice_2:output:0*lstm_cell_14/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
lstm_cell_14/MatMul¼
$lstm_cell_14/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_14_matmul_1_readvariableop_resource* 
_output_shapes
:
 *
dtype02&
$lstm_cell_14/MatMul_1/ReadVariableOp©
lstm_cell_14/MatMul_1MatMulzeros:output:0,lstm_cell_14/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
lstm_cell_14/MatMul_1 
lstm_cell_14/addAddV2lstm_cell_14/MatMul:product:0lstm_cell_14/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
lstm_cell_14/add´
#lstm_cell_14/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_14_biasadd_readvariableop_resource*
_output_shapes	
: *
dtype02%
#lstm_cell_14/BiasAdd/ReadVariableOp­
lstm_cell_14/BiasAddBiasAddlstm_cell_14/add:z:0+lstm_cell_14/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
lstm_cell_14/BiasAdd~
lstm_cell_14/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_14/split/split_dim÷
lstm_cell_14/splitSplit%lstm_cell_14/split/split_dim:output:0lstm_cell_14/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2
lstm_cell_14/split
lstm_cell_14/SigmoidSigmoidlstm_cell_14/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_14/Sigmoid
lstm_cell_14/Sigmoid_1Sigmoidlstm_cell_14/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_14/Sigmoid_1
lstm_cell_14/mulMullstm_cell_14/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_14/mul~
lstm_cell_14/ReluRelulstm_cell_14/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_14/Relu
lstm_cell_14/mul_1Mullstm_cell_14/Sigmoid:y:0lstm_cell_14/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_14/mul_1
lstm_cell_14/add_1AddV2lstm_cell_14/mul:z:0lstm_cell_14/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_14/add_1
lstm_cell_14/Sigmoid_2Sigmoidlstm_cell_14/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_14/Sigmoid_2}
lstm_cell_14/Relu_1Relulstm_cell_14/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_14/Relu_1¡
lstm_cell_14/mul_2Mullstm_cell_14/Sigmoid_2:y:0!lstm_cell_14/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_14/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ  2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_14_matmul_readvariableop_resource-lstm_cell_14_matmul_1_readvariableop_resource,lstm_cell_14_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_26078391*
condR
while_cond_26078390*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ  22
0TensorArrayV2Stack/TensorListStack/element_shapeé
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
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
:ÿÿÿÿÿÿÿÿÿ*
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
:ÿÿÿÿÿÿÿÿÿ2
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
:ÿÿÿÿÿÿÿÿÿ2

IdentityÈ
NoOpNoOp$^lstm_cell_14/BiasAdd/ReadVariableOp#^lstm_cell_14/MatMul/ReadVariableOp%^lstm_cell_14/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ]: : : 2J
#lstm_cell_14/BiasAdd/ReadVariableOp#lstm_cell_14/BiasAdd/ReadVariableOp2H
"lstm_cell_14/MatMul/ReadVariableOp"lstm_cell_14/MatMul/ReadVariableOp2L
$lstm_cell_14/MatMul_1/ReadVariableOp$lstm_cell_14/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
 
_user_specified_nameinputs
Ô

í
lstm_14_while_cond_26077554,
(lstm_14_while_lstm_14_while_loop_counter2
.lstm_14_while_lstm_14_while_maximum_iterations
lstm_14_while_placeholder
lstm_14_while_placeholder_1
lstm_14_while_placeholder_2
lstm_14_while_placeholder_3.
*lstm_14_while_less_lstm_14_strided_slice_1F
Blstm_14_while_lstm_14_while_cond_26077554___redundant_placeholder0F
Blstm_14_while_lstm_14_while_cond_26077554___redundant_placeholder1F
Blstm_14_while_lstm_14_while_cond_26077554___redundant_placeholder2F
Blstm_14_while_lstm_14_while_cond_26077554___redundant_placeholder3
lstm_14_while_identity

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
B: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 

_output_shapes
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
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
ß
Í
while_cond_26079065
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_26079065___redundant_placeholder06
2while_while_cond_26079065___redundant_placeholder16
2while_while_cond_26079065___redundant_placeholder26
2while_while_cond_26079065___redundant_placeholder3
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
@: : : : :ÿÿÿÿÿÿÿÿÿr:ÿÿÿÿÿÿÿÿÿr: ::::: 

_output_shapes
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
:ÿÿÿÿÿÿÿÿÿr:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr:

_output_shapes
: :

_output_shapes
:
°?
Ô
while_body_26078089
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_14_matmul_readvariableop_resource_0:	] I
5while_lstm_cell_14_matmul_1_readvariableop_resource_0:
 C
4while_lstm_cell_14_biasadd_readvariableop_resource_0:	 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_14_matmul_readvariableop_resource:	] G
3while_lstm_cell_14_matmul_1_readvariableop_resource:
 A
2while_lstm_cell_14_biasadd_readvariableop_resource:	 ¢)while/lstm_cell_14/BiasAdd/ReadVariableOp¢(while/lstm_cell_14/MatMul/ReadVariableOp¢*while/lstm_cell_14/MatMul_1/ReadVariableOpÃ
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
(while/lstm_cell_14/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_14_matmul_readvariableop_resource_0*
_output_shapes
:	] *
dtype02*
(while/lstm_cell_14/MatMul/ReadVariableOp×
while/lstm_cell_14/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_14/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/lstm_cell_14/MatMulÐ
*while/lstm_cell_14/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_14_matmul_1_readvariableop_resource_0* 
_output_shapes
:
 *
dtype02,
*while/lstm_cell_14/MatMul_1/ReadVariableOpÀ
while/lstm_cell_14/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_14/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/lstm_cell_14/MatMul_1¸
while/lstm_cell_14/addAddV2#while/lstm_cell_14/MatMul:product:0%while/lstm_cell_14/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/lstm_cell_14/addÈ
)while/lstm_cell_14/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_14_biasadd_readvariableop_resource_0*
_output_shapes	
: *
dtype02+
)while/lstm_cell_14/BiasAdd/ReadVariableOpÅ
while/lstm_cell_14/BiasAddBiasAddwhile/lstm_cell_14/add:z:01while/lstm_cell_14/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/lstm_cell_14/BiasAdd
"while/lstm_cell_14/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_14/split/split_dim
while/lstm_cell_14/splitSplit+while/lstm_cell_14/split/split_dim:output:0#while/lstm_cell_14/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2
while/lstm_cell_14/split
while/lstm_cell_14/SigmoidSigmoid!while/lstm_cell_14/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_14/Sigmoid
while/lstm_cell_14/Sigmoid_1Sigmoid!while/lstm_cell_14/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_14/Sigmoid_1¡
while/lstm_cell_14/mulMul while/lstm_cell_14/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_14/mul
while/lstm_cell_14/ReluRelu!while/lstm_cell_14/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_14/Reluµ
while/lstm_cell_14/mul_1Mulwhile/lstm_cell_14/Sigmoid:y:0%while/lstm_cell_14/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_14/mul_1ª
while/lstm_cell_14/add_1AddV2while/lstm_cell_14/mul:z:0while/lstm_cell_14/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_14/add_1
while/lstm_cell_14/Sigmoid_2Sigmoid!while/lstm_cell_14/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_14/Sigmoid_2
while/lstm_cell_14/Relu_1Reluwhile/lstm_cell_14/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_14/Relu_1¹
while/lstm_cell_14/mul_2Mul while/lstm_cell_14/Sigmoid_2:y:0'while/lstm_cell_14/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_14/mul_2à
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
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_14/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_14/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/Identity_5Þ

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
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2V
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
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
Öø

J__inference_sequential_7_layer_call_and_return_conditional_losses_26077488

inputsF
3lstm_14_lstm_cell_14_matmul_readvariableop_resource:	] I
5lstm_14_lstm_cell_14_matmul_1_readvariableop_resource:
 C
4lstm_14_lstm_cell_14_biasadd_readvariableop_resource:	 G
3lstm_15_lstm_cell_15_matmul_readvariableop_resource:
ÈH
5lstm_15_lstm_cell_15_matmul_1_readvariableop_resource:	rÈC
4lstm_15_lstm_cell_15_biasadd_readvariableop_resource:	È;
)dense_7_tensordot_readvariableop_resource:r5
'dense_7_biasadd_readvariableop_resource:
identity¢dense_7/BiasAdd/ReadVariableOp¢ dense_7/Tensordot/ReadVariableOp¢+lstm_14/lstm_cell_14/BiasAdd/ReadVariableOp¢*lstm_14/lstm_cell_14/MatMul/ReadVariableOp¢,lstm_14/lstm_cell_14/MatMul_1/ReadVariableOp¢lstm_14/while¢+lstm_15/lstm_cell_15/BiasAdd/ReadVariableOp¢*lstm_15/lstm_cell_15/MatMul/ReadVariableOp¢,lstm_15/lstm_cell_15/MatMul_1/ReadVariableOp¢lstm_15/whileT
lstm_14/ShapeShapeinputs*
T0*
_output_shapes
:2
lstm_14/Shape
lstm_14/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_14/strided_slice/stack
lstm_14/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_14/strided_slice/stack_1
lstm_14/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_14/strided_slice/stack_2
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
B :2
lstm_14/zeros/mul/y
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
B :è2
lstm_14/zeros/Less/y
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
B :2
lstm_14/zeros/packed/1£
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
lstm_14/zeros/Const
lstm_14/zerosFilllstm_14/zeros/packed:output:0lstm_14/zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_14/zerosq
lstm_14/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :2
lstm_14/zeros_1/mul/y
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
B :è2
lstm_14/zeros_1/Less/y
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
B :2
lstm_14/zeros_1/packed/1©
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
lstm_14/zeros_1/Const
lstm_14/zeros_1Filllstm_14/zeros_1/packed:output:0lstm_14/zeros_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_14/zeros_1
lstm_14/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_14/transpose/perm
lstm_14/transpose	Transposeinputslstm_14/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]2
lstm_14/transposeg
lstm_14/Shape_1Shapelstm_14/transpose:y:0*
T0*
_output_shapes
:2
lstm_14/Shape_1
lstm_14/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_14/strided_slice_1/stack
lstm_14/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_14/strided_slice_1/stack_1
lstm_14/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_14/strided_slice_1/stack_2
lstm_14/strided_slice_1StridedSlicelstm_14/Shape_1:output:0&lstm_14/strided_slice_1/stack:output:0(lstm_14/strided_slice_1/stack_1:output:0(lstm_14/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_14/strided_slice_1
#lstm_14/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2%
#lstm_14/TensorArrayV2/element_shapeÒ
lstm_14/TensorArrayV2TensorListReserve,lstm_14/TensorArrayV2/element_shape:output:0 lstm_14/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_14/TensorArrayV2Ï
=lstm_14/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ]   2?
=lstm_14/TensorArrayUnstack/TensorListFromTensor/element_shape
/lstm_14/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_14/transpose:y:0Flstm_14/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type021
/lstm_14/TensorArrayUnstack/TensorListFromTensor
lstm_14/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_14/strided_slice_2/stack
lstm_14/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_14/strided_slice_2/stack_1
lstm_14/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_14/strided_slice_2/stack_2¬
lstm_14/strided_slice_2StridedSlicelstm_14/transpose:y:0&lstm_14/strided_slice_2/stack:output:0(lstm_14/strided_slice_2/stack_1:output:0(lstm_14/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]*
shrink_axis_mask2
lstm_14/strided_slice_2Í
*lstm_14/lstm_cell_14/MatMul/ReadVariableOpReadVariableOp3lstm_14_lstm_cell_14_matmul_readvariableop_resource*
_output_shapes
:	] *
dtype02,
*lstm_14/lstm_cell_14/MatMul/ReadVariableOpÍ
lstm_14/lstm_cell_14/MatMulMatMul lstm_14/strided_slice_2:output:02lstm_14/lstm_cell_14/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
lstm_14/lstm_cell_14/MatMulÔ
,lstm_14/lstm_cell_14/MatMul_1/ReadVariableOpReadVariableOp5lstm_14_lstm_cell_14_matmul_1_readvariableop_resource* 
_output_shapes
:
 *
dtype02.
,lstm_14/lstm_cell_14/MatMul_1/ReadVariableOpÉ
lstm_14/lstm_cell_14/MatMul_1MatMullstm_14/zeros:output:04lstm_14/lstm_cell_14/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
lstm_14/lstm_cell_14/MatMul_1À
lstm_14/lstm_cell_14/addAddV2%lstm_14/lstm_cell_14/MatMul:product:0'lstm_14/lstm_cell_14/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
lstm_14/lstm_cell_14/addÌ
+lstm_14/lstm_cell_14/BiasAdd/ReadVariableOpReadVariableOp4lstm_14_lstm_cell_14_biasadd_readvariableop_resource*
_output_shapes	
: *
dtype02-
+lstm_14/lstm_cell_14/BiasAdd/ReadVariableOpÍ
lstm_14/lstm_cell_14/BiasAddBiasAddlstm_14/lstm_cell_14/add:z:03lstm_14/lstm_cell_14/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
lstm_14/lstm_cell_14/BiasAdd
$lstm_14/lstm_cell_14/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2&
$lstm_14/lstm_cell_14/split/split_dim
lstm_14/lstm_cell_14/splitSplit-lstm_14/lstm_cell_14/split/split_dim:output:0%lstm_14/lstm_cell_14/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2
lstm_14/lstm_cell_14/split
lstm_14/lstm_cell_14/SigmoidSigmoid#lstm_14/lstm_cell_14/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_14/lstm_cell_14/Sigmoid£
lstm_14/lstm_cell_14/Sigmoid_1Sigmoid#lstm_14/lstm_cell_14/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
lstm_14/lstm_cell_14/Sigmoid_1¬
lstm_14/lstm_cell_14/mulMul"lstm_14/lstm_cell_14/Sigmoid_1:y:0lstm_14/zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_14/lstm_cell_14/mul
lstm_14/lstm_cell_14/ReluRelu#lstm_14/lstm_cell_14/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_14/lstm_cell_14/Relu½
lstm_14/lstm_cell_14/mul_1Mul lstm_14/lstm_cell_14/Sigmoid:y:0'lstm_14/lstm_cell_14/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_14/lstm_cell_14/mul_1²
lstm_14/lstm_cell_14/add_1AddV2lstm_14/lstm_cell_14/mul:z:0lstm_14/lstm_cell_14/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_14/lstm_cell_14/add_1£
lstm_14/lstm_cell_14/Sigmoid_2Sigmoid#lstm_14/lstm_cell_14/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
lstm_14/lstm_cell_14/Sigmoid_2
lstm_14/lstm_cell_14/Relu_1Relulstm_14/lstm_cell_14/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_14/lstm_cell_14/Relu_1Á
lstm_14/lstm_cell_14/mul_2Mul"lstm_14/lstm_cell_14/Sigmoid_2:y:0)lstm_14/lstm_cell_14/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_14/lstm_cell_14/mul_2
%lstm_14/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ  2'
%lstm_14/TensorArrayV2_1/element_shapeØ
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
lstm_14/time
 lstm_14/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2"
 lstm_14/while/maximum_iterationsz
lstm_14/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_14/while/loop_counter
lstm_14/whileWhile#lstm_14/while/loop_counter:output:0)lstm_14/while/maximum_iterations:output:0lstm_14/time:output:0 lstm_14/TensorArrayV2_1:handle:0lstm_14/zeros:output:0lstm_14/zeros_1:output:0 lstm_14/strided_slice_1:output:0?lstm_14/TensorArrayUnstack/TensorListFromTensor:output_handle:03lstm_14_lstm_cell_14_matmul_readvariableop_resource5lstm_14_lstm_cell_14_matmul_1_readvariableop_resource4lstm_14_lstm_cell_14_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *'
bodyR
lstm_14_while_body_26077228*'
condR
lstm_14_while_cond_26077227*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 2
lstm_14/whileÅ
8lstm_14/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ  2:
8lstm_14/TensorArrayV2Stack/TensorListStack/element_shape
*lstm_14/TensorArrayV2Stack/TensorListStackTensorListStacklstm_14/while:output:3Alstm_14/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype02,
*lstm_14/TensorArrayV2Stack/TensorListStack
lstm_14/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
lstm_14/strided_slice_3/stack
lstm_14/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2!
lstm_14/strided_slice_3/stack_1
lstm_14/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_14/strided_slice_3/stack_2Ë
lstm_14/strided_slice_3StridedSlice3lstm_14/TensorArrayV2Stack/TensorListStack:tensor:0&lstm_14/strided_slice_3/stack:output:0(lstm_14/strided_slice_3/stack_1:output:0(lstm_14/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
lstm_14/strided_slice_3
lstm_14/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_14/transpose_1/permÆ
lstm_14/transpose_1	Transpose3lstm_14/TensorArrayV2Stack/TensorListStack:tensor:0!lstm_14/transpose_1/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_14/transpose_1v
lstm_14/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_14/runtime
dropout_14/IdentityIdentitylstm_14/transpose_1:y:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout_14/Identityj
lstm_15/ShapeShapedropout_14/Identity:output:0*
T0*
_output_shapes
:2
lstm_15/Shape
lstm_15/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_15/strided_slice/stack
lstm_15/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_15/strided_slice/stack_1
lstm_15/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_15/strided_slice/stack_2
lstm_15/strided_sliceStridedSlicelstm_15/Shape:output:0$lstm_15/strided_slice/stack:output:0&lstm_15/strided_slice/stack_1:output:0&lstm_15/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_15/strided_slicel
lstm_15/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :r2
lstm_15/zeros/mul/y
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
B :è2
lstm_15/zeros/Less/y
lstm_15/zeros/LessLesslstm_15/zeros/mul:z:0lstm_15/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_15/zeros/Lessr
lstm_15/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :r2
lstm_15/zeros/packed/1£
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
lstm_15/zeros/Const
lstm_15/zerosFilllstm_15/zeros/packed:output:0lstm_15/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2
lstm_15/zerosp
lstm_15/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :r2
lstm_15/zeros_1/mul/y
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
B :è2
lstm_15/zeros_1/Less/y
lstm_15/zeros_1/LessLesslstm_15/zeros_1/mul:z:0lstm_15/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_15/zeros_1/Lessv
lstm_15/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :r2
lstm_15/zeros_1/packed/1©
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
lstm_15/zeros_1/Const
lstm_15/zeros_1Filllstm_15/zeros_1/packed:output:0lstm_15/zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2
lstm_15/zeros_1
lstm_15/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_15/transpose/perm©
lstm_15/transpose	Transposedropout_14/Identity:output:0lstm_15/transpose/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_15/transposeg
lstm_15/Shape_1Shapelstm_15/transpose:y:0*
T0*
_output_shapes
:2
lstm_15/Shape_1
lstm_15/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_15/strided_slice_1/stack
lstm_15/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_15/strided_slice_1/stack_1
lstm_15/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_15/strided_slice_1/stack_2
lstm_15/strided_slice_1StridedSlicelstm_15/Shape_1:output:0&lstm_15/strided_slice_1/stack:output:0(lstm_15/strided_slice_1/stack_1:output:0(lstm_15/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_15/strided_slice_1
#lstm_15/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2%
#lstm_15/TensorArrayV2/element_shapeÒ
lstm_15/TensorArrayV2TensorListReserve,lstm_15/TensorArrayV2/element_shape:output:0 lstm_15/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_15/TensorArrayV2Ï
=lstm_15/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ  2?
=lstm_15/TensorArrayUnstack/TensorListFromTensor/element_shape
/lstm_15/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_15/transpose:y:0Flstm_15/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type021
/lstm_15/TensorArrayUnstack/TensorListFromTensor
lstm_15/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_15/strided_slice_2/stack
lstm_15/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_15/strided_slice_2/stack_1
lstm_15/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_15/strided_slice_2/stack_2­
lstm_15/strided_slice_2StridedSlicelstm_15/transpose:y:0&lstm_15/strided_slice_2/stack:output:0(lstm_15/strided_slice_2/stack_1:output:0(lstm_15/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
lstm_15/strided_slice_2Î
*lstm_15/lstm_cell_15/MatMul/ReadVariableOpReadVariableOp3lstm_15_lstm_cell_15_matmul_readvariableop_resource* 
_output_shapes
:
È*
dtype02,
*lstm_15/lstm_cell_15/MatMul/ReadVariableOpÍ
lstm_15/lstm_cell_15/MatMulMatMul lstm_15/strided_slice_2:output:02lstm_15/lstm_cell_15/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_15/lstm_cell_15/MatMulÓ
,lstm_15/lstm_cell_15/MatMul_1/ReadVariableOpReadVariableOp5lstm_15_lstm_cell_15_matmul_1_readvariableop_resource*
_output_shapes
:	rÈ*
dtype02.
,lstm_15/lstm_cell_15/MatMul_1/ReadVariableOpÉ
lstm_15/lstm_cell_15/MatMul_1MatMullstm_15/zeros:output:04lstm_15/lstm_cell_15/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_15/lstm_cell_15/MatMul_1À
lstm_15/lstm_cell_15/addAddV2%lstm_15/lstm_cell_15/MatMul:product:0'lstm_15/lstm_cell_15/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_15/lstm_cell_15/addÌ
+lstm_15/lstm_cell_15/BiasAdd/ReadVariableOpReadVariableOp4lstm_15_lstm_cell_15_biasadd_readvariableop_resource*
_output_shapes	
:È*
dtype02-
+lstm_15/lstm_cell_15/BiasAdd/ReadVariableOpÍ
lstm_15/lstm_cell_15/BiasAddBiasAddlstm_15/lstm_cell_15/add:z:03lstm_15/lstm_cell_15/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_15/lstm_cell_15/BiasAdd
$lstm_15/lstm_cell_15/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2&
$lstm_15/lstm_cell_15/split/split_dim
lstm_15/lstm_cell_15/splitSplit-lstm_15/lstm_cell_15/split/split_dim:output:0%lstm_15/lstm_cell_15/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿr:ÿÿÿÿÿÿÿÿÿr:ÿÿÿÿÿÿÿÿÿr:ÿÿÿÿÿÿÿÿÿr*
	num_split2
lstm_15/lstm_cell_15/split
lstm_15/lstm_cell_15/SigmoidSigmoid#lstm_15/lstm_cell_15/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2
lstm_15/lstm_cell_15/Sigmoid¢
lstm_15/lstm_cell_15/Sigmoid_1Sigmoid#lstm_15/lstm_cell_15/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2 
lstm_15/lstm_cell_15/Sigmoid_1«
lstm_15/lstm_cell_15/mulMul"lstm_15/lstm_cell_15/Sigmoid_1:y:0lstm_15/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2
lstm_15/lstm_cell_15/mul
lstm_15/lstm_cell_15/ReluRelu#lstm_15/lstm_cell_15/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2
lstm_15/lstm_cell_15/Relu¼
lstm_15/lstm_cell_15/mul_1Mul lstm_15/lstm_cell_15/Sigmoid:y:0'lstm_15/lstm_cell_15/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2
lstm_15/lstm_cell_15/mul_1±
lstm_15/lstm_cell_15/add_1AddV2lstm_15/lstm_cell_15/mul:z:0lstm_15/lstm_cell_15/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2
lstm_15/lstm_cell_15/add_1¢
lstm_15/lstm_cell_15/Sigmoid_2Sigmoid#lstm_15/lstm_cell_15/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2 
lstm_15/lstm_cell_15/Sigmoid_2
lstm_15/lstm_cell_15/Relu_1Relulstm_15/lstm_cell_15/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2
lstm_15/lstm_cell_15/Relu_1À
lstm_15/lstm_cell_15/mul_2Mul"lstm_15/lstm_cell_15/Sigmoid_2:y:0)lstm_15/lstm_cell_15/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2
lstm_15/lstm_cell_15/mul_2
%lstm_15/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿr   2'
%lstm_15/TensorArrayV2_1/element_shapeØ
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
lstm_15/time
 lstm_15/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2"
 lstm_15/while/maximum_iterationsz
lstm_15/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_15/while/loop_counter
lstm_15/whileWhile#lstm_15/while/loop_counter:output:0)lstm_15/while/maximum_iterations:output:0lstm_15/time:output:0 lstm_15/TensorArrayV2_1:handle:0lstm_15/zeros:output:0lstm_15/zeros_1:output:0 lstm_15/strided_slice_1:output:0?lstm_15/TensorArrayUnstack/TensorListFromTensor:output_handle:03lstm_15_lstm_cell_15_matmul_readvariableop_resource5lstm_15_lstm_cell_15_matmul_1_readvariableop_resource4lstm_15_lstm_cell_15_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿr:ÿÿÿÿÿÿÿÿÿr: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *'
bodyR
lstm_15_while_body_26077376*'
condR
lstm_15_while_cond_26077375*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿr:ÿÿÿÿÿÿÿÿÿr: : : : : *
parallel_iterations 2
lstm_15/whileÅ
8lstm_15/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿr   2:
8lstm_15/TensorArrayV2Stack/TensorListStack/element_shape
*lstm_15/TensorArrayV2Stack/TensorListStackTensorListStacklstm_15/while:output:3Alstm_15/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿr*
element_dtype02,
*lstm_15/TensorArrayV2Stack/TensorListStack
lstm_15/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
lstm_15/strided_slice_3/stack
lstm_15/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2!
lstm_15/strided_slice_3/stack_1
lstm_15/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_15/strided_slice_3/stack_2Ê
lstm_15/strided_slice_3StridedSlice3lstm_15/TensorArrayV2Stack/TensorListStack:tensor:0&lstm_15/strided_slice_3/stack:output:0(lstm_15/strided_slice_3/stack_1:output:0(lstm_15/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr*
shrink_axis_mask2
lstm_15/strided_slice_3
lstm_15/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_15/transpose_1/permÅ
lstm_15/transpose_1	Transpose3lstm_15/TensorArrayV2Stack/TensorListStack:tensor:0!lstm_15/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2
lstm_15/transpose_1v
lstm_15/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_15/runtime
dropout_15/IdentityIdentitylstm_15/transpose_1:y:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2
dropout_15/Identity®
 dense_7/Tensordot/ReadVariableOpReadVariableOp)dense_7_tensordot_readvariableop_resource*
_output_shapes

:r*
dtype02"
 dense_7/Tensordot/ReadVariableOpz
dense_7/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_7/Tensordot/axes
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
dense_7/Tensordot/Shape
dense_7/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_7/Tensordot/GatherV2/axisù
dense_7/Tensordot/GatherV2GatherV2 dense_7/Tensordot/Shape:output:0dense_7/Tensordot/free:output:0(dense_7/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_7/Tensordot/GatherV2
!dense_7/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_7/Tensordot/GatherV2_1/axisÿ
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
dense_7/Tensordot/Const 
dense_7/Tensordot/ProdProd#dense_7/Tensordot/GatherV2:output:0 dense_7/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_7/Tensordot/Prod
dense_7/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_7/Tensordot/Const_1¨
dense_7/Tensordot/Prod_1Prod%dense_7/Tensordot/GatherV2_1:output:0"dense_7/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_7/Tensordot/Prod_1
dense_7/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense_7/Tensordot/concat/axisØ
dense_7/Tensordot/concatConcatV2dense_7/Tensordot/free:output:0dense_7/Tensordot/axes:output:0&dense_7/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_7/Tensordot/concat¬
dense_7/Tensordot/stackPackdense_7/Tensordot/Prod:output:0!dense_7/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_7/Tensordot/stack¾
dense_7/Tensordot/transpose	Transposedropout_15/Identity:output:0!dense_7/Tensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2
dense_7/Tensordot/transpose¿
dense_7/Tensordot/ReshapeReshapedense_7/Tensordot/transpose:y:0 dense_7/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
dense_7/Tensordot/Reshape¾
dense_7/Tensordot/MatMulMatMul"dense_7/Tensordot/Reshape:output:0(dense_7/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_7/Tensordot/MatMul
dense_7/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
dense_7/Tensordot/Const_2
dense_7/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_7/Tensordot/concat_1/axiså
dense_7/Tensordot/concat_1ConcatV2#dense_7/Tensordot/GatherV2:output:0"dense_7/Tensordot/Const_2:output:0(dense_7/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_7/Tensordot/concat_1°
dense_7/TensordotReshape"dense_7/Tensordot/MatMul:product:0#dense_7/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_7/Tensordot¤
dense_7/BiasAdd/ReadVariableOpReadVariableOp'dense_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_7/BiasAdd/ReadVariableOp§
dense_7/BiasAddBiasAdddense_7/Tensordot:output:0&dense_7/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_7/BiasAdd}
dense_7/SoftmaxSoftmaxdense_7/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_7/Softmaxx
IdentityIdentitydense_7/Softmax:softmax:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

IdentityÆ
NoOpNoOp^dense_7/BiasAdd/ReadVariableOp!^dense_7/Tensordot/ReadVariableOp,^lstm_14/lstm_cell_14/BiasAdd/ReadVariableOp+^lstm_14/lstm_cell_14/MatMul/ReadVariableOp-^lstm_14/lstm_cell_14/MatMul_1/ReadVariableOp^lstm_14/while,^lstm_15/lstm_cell_15/BiasAdd/ReadVariableOp+^lstm_15/lstm_cell_15/MatMul/ReadVariableOp-^lstm_15/lstm_cell_15/MatMul_1/ReadVariableOp^lstm_15/while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ]: : : : : : : : 2@
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
:ÿÿÿÿÿÿÿÿÿ]
 
_user_specified_nameinputs
ú	
È
&__inference_signature_wrapper_26077161
lstm_14_input
unknown:	] 
	unknown_0:
 
	unknown_1:	 
	unknown_2:
È
	unknown_3:	rÈ
	unknown_4:	È
	unknown_5:r
	unknown_6:
identity¢StatefulPartitionedCall¬
StatefulPartitionedCallStatefulPartitionedCalllstm_14_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
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
#__inference__wrapped_model_260749442
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
_user_specified_namelstm_14_input
Û
ñ
(sequential_7_lstm_14_while_cond_26074683F
Bsequential_7_lstm_14_while_sequential_7_lstm_14_while_loop_counterL
Hsequential_7_lstm_14_while_sequential_7_lstm_14_while_maximum_iterations*
&sequential_7_lstm_14_while_placeholder,
(sequential_7_lstm_14_while_placeholder_1,
(sequential_7_lstm_14_while_placeholder_2,
(sequential_7_lstm_14_while_placeholder_3H
Dsequential_7_lstm_14_while_less_sequential_7_lstm_14_strided_slice_1`
\sequential_7_lstm_14_while_sequential_7_lstm_14_while_cond_26074683___redundant_placeholder0`
\sequential_7_lstm_14_while_sequential_7_lstm_14_while_cond_26074683___redundant_placeholder1`
\sequential_7_lstm_14_while_sequential_7_lstm_14_while_cond_26074683___redundant_placeholder2`
\sequential_7_lstm_14_while_sequential_7_lstm_14_while_cond_26074683___redundant_placeholder3'
#sequential_7_lstm_14_while_identity
Ù
sequential_7/lstm_14/while/LessLess&sequential_7_lstm_14_while_placeholderDsequential_7_lstm_14_while_less_sequential_7_lstm_14_strided_slice_1*
T0*
_output_shapes
: 2!
sequential_7/lstm_14/while/Less
#sequential_7/lstm_14/while/IdentityIdentity#sequential_7/lstm_14/while/Less:z:0*
T0
*
_output_shapes
: 2%
#sequential_7/lstm_14/while/Identity"S
#sequential_7_lstm_14_while_identity,sequential_7/lstm_14/while/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 

_output_shapes
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
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
ª

Ñ
/__inference_sequential_7_layer_call_fn_26076598
lstm_14_input
unknown:	] 
	unknown_0:
 
	unknown_1:	 
	unknown_2:
È
	unknown_3:	rÈ
	unknown_4:	È
	unknown_5:r
	unknown_6:
identity¢StatefulPartitionedCallÓ
StatefulPartitionedCallStatefulPartitionedCalllstm_14_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
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
GPU 2J 8 *S
fNRL
J__inference_sequential_7_layer_call_and_return_conditional_losses_260765792
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
_user_specified_namelstm_14_input
ý%
ó
while_body_26075873
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_01
while_lstm_cell_15_26075897_0:
È0
while_lstm_cell_15_26075899_0:	rÈ,
while_lstm_cell_15_26075901_0:	È
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor/
while_lstm_cell_15_26075897:
È.
while_lstm_cell_15_26075899:	rÈ*
while_lstm_cell_15_26075901:	È¢*while/lstm_cell_15/StatefulPartitionedCallÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ  29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÔ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemé
*while/lstm_cell_15/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_15_26075897_0while_lstm_cell_15_26075899_0while_lstm_cell_15_26075901_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿr:ÿÿÿÿÿÿÿÿÿr:ÿÿÿÿÿÿÿÿÿr*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_lstm_cell_15_layer_call_and_return_conditional_losses_260757952,
*while/lstm_cell_15/StatefulPartitionedCall÷
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
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3¤
while/Identity_4Identity3while/lstm_cell_15/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2
while/Identity_4¤
while/Identity_5Identity3while/lstm_cell_15/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2
while/Identity_5

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
while_lstm_cell_15_26075897while_lstm_cell_15_26075897_0"<
while_lstm_cell_15_26075899while_lstm_cell_15_26075899_0"<
while_lstm_cell_15_26075901while_lstm_cell_15_26075901_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿr:ÿÿÿÿÿÿÿÿÿr: : : : : 2X
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
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr:

_output_shapes
: :

_output_shapes
: 
È7
Û
$__inference__traced_restore_26079562
file_prefix1
assignvariableop_dense_7_kernel:r-
assignvariableop_1_dense_7_bias:A
.assignvariableop_2_lstm_14_lstm_cell_14_kernel:	] L
8assignvariableop_3_lstm_14_lstm_cell_14_recurrent_kernel:
 ;
,assignvariableop_4_lstm_14_lstm_cell_14_bias:	 B
.assignvariableop_5_lstm_15_lstm_cell_15_kernel:
ÈK
8assignvariableop_6_lstm_15_lstm_cell_15_recurrent_kernel:	rÈ;
,assignvariableop_7_lstm_15_lstm_cell_15_bias:	È"
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

Identity
AssignVariableOpAssignVariableOpassignvariableop_dense_7_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1¤
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_7_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2³
AssignVariableOp_2AssignVariableOp.assignvariableop_2_lstm_14_lstm_cell_14_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3½
AssignVariableOp_3AssignVariableOp8assignvariableop_3_lstm_14_lstm_cell_14_recurrent_kernelIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4±
AssignVariableOp_4AssignVariableOp,assignvariableop_4_lstm_14_lstm_cell_14_biasIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5³
AssignVariableOp_5AssignVariableOp.assignvariableop_5_lstm_15_lstm_cell_15_kernelIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6½
AssignVariableOp_6AssignVariableOp8assignvariableop_6_lstm_15_lstm_cell_15_recurrent_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7±
AssignVariableOp_7AssignVariableOp,assignvariableop_7_lstm_15_lstm_cell_15_biasIdentity_7:output:0"/device:CPU:0*
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

ë
J__inference_sequential_7_layer_call_and_return_conditional_losses_26076579

inputs#
lstm_14_26076362:	] $
lstm_14_26076364:
 
lstm_14_26076366:	 $
lstm_15_26076527:
È#
lstm_15_26076529:	rÈ
lstm_15_26076531:	È"
dense_7_26076573:r
dense_7_26076575:
identity¢dense_7/StatefulPartitionedCall¢lstm_14/StatefulPartitionedCall¢lstm_15/StatefulPartitionedCall®
lstm_14/StatefulPartitionedCallStatefulPartitionedCallinputslstm_14_26076362lstm_14_26076364lstm_14_26076366*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_lstm_14_layer_call_and_return_conditional_losses_260763612!
lstm_14/StatefulPartitionedCall
dropout_14/PartitionedCallPartitionedCall(lstm_14/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_dropout_14_layer_call_and_return_conditional_losses_260763742
dropout_14/PartitionedCallÊ
lstm_15/StatefulPartitionedCallStatefulPartitionedCall#dropout_14/PartitionedCall:output:0lstm_15_26076527lstm_15_26076529lstm_15_26076531*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿr*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_lstm_15_layer_call_and_return_conditional_losses_260765262!
lstm_15/StatefulPartitionedCall
dropout_15/PartitionedCallPartitionedCall(lstm_15/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿr* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_dropout_15_layer_call_and_return_conditional_losses_260765392
dropout_15/PartitionedCall¶
dense_7/StatefulPartitionedCallStatefulPartitionedCall#dropout_15/PartitionedCall:output:0dense_7_26076573dense_7_26076575*
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
GPU 2J 8 *N
fIRG
E__inference_dense_7_layer_call_and_return_conditional_losses_260765722!
dense_7/StatefulPartitionedCall
IdentityIdentity(dense_7/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity´
NoOpNoOp ^dense_7/StatefulPartitionedCall ^lstm_14/StatefulPartitionedCall ^lstm_15/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ]: : : : : : : : 2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall2B
lstm_14/StatefulPartitionedCalllstm_14/StatefulPartitionedCall2B
lstm_15/StatefulPartitionedCalllstm_15/StatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
 
_user_specified_nameinputs

f
H__inference_dropout_15_layer_call_and_return_conditional_losses_26079199

inputs

identity_1^
IdentityIdentityinputs*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2

Identitym

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿr:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
 
_user_specified_nameinputs
Ï!
ü
E__inference_dense_7_layer_call_and_return_conditional_losses_26079252

inputs3
!tensordot_readvariableop_resource:r-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢Tensordot/ReadVariableOp
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:r*
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
Tensordot/stack
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2
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
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿr: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
 
_user_specified_nameinputs
¶
f
-__inference_dropout_15_layer_call_fn_26079221

inputs
identity¢StatefulPartitionedCallâ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿr* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_dropout_15_layer_call_and_return_conditional_losses_260766282
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿr22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
 
_user_specified_nameinputs
ã
Í
while_cond_26078390
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_26078390___redundant_placeholder06
2while_while_cond_26078390___redundant_placeholder16
2while_while_cond_26078390___redundant_placeholder26
2while_while_cond_26078390___redundant_placeholder3
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
B: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 

_output_shapes
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
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:

f
H__inference_dropout_15_layer_call_and_return_conditional_losses_26076539

inputs

identity_1^
IdentityIdentityinputs*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2

Identitym

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿr:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
 
_user_specified_nameinputs
×
g
H__inference_dropout_14_layer_call_and_return_conditional_losses_26078536

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
:ÿÿÿÿÿÿÿÿÿ2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape¹
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
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
:ÿÿÿÿÿÿÿÿÿ2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/Mul_1j
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ã
Í
while_cond_26077937
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_26077937___redundant_placeholder06
2while_while_cond_26077937___redundant_placeholder16
2while_while_cond_26077937___redundant_placeholder26
2while_while_cond_26077937___redundant_placeholder3
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
B: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 

_output_shapes
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
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
ß
Í
while_cond_26075872
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_26075872___redundant_placeholder06
2while_while_cond_26075872___redundant_placeholder16
2while_while_cond_26075872___redundant_placeholder26
2while_while_cond_26075872___redundant_placeholder3
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
@: : : : :ÿÿÿÿÿÿÿÿÿr:ÿÿÿÿÿÿÿÿÿr: ::::: 

_output_shapes
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
:ÿÿÿÿÿÿÿÿÿr:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr:

_output_shapes
: :

_output_shapes
:
&
ó
while_body_26075033
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_00
while_lstm_cell_14_26075057_0:	] 1
while_lstm_cell_14_26075059_0:
 ,
while_lstm_cell_14_26075061_0:	 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor.
while_lstm_cell_14_26075057:	] /
while_lstm_cell_14_26075059:
 *
while_lstm_cell_14_26075061:	 ¢*while/lstm_cell_14/StatefulPartitionedCallÃ
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
*while/lstm_cell_14/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_14_26075057_0while_lstm_cell_14_26075059_0while_lstm_cell_14_26075061_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_lstm_cell_14_layer_call_and_return_conditional_losses_260750192,
*while/lstm_cell_14/StatefulPartitionedCall÷
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
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3¥
while/Identity_4Identity3while/lstm_cell_14/StatefulPartitionedCall:output:1^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/Identity_4¥
while/Identity_5Identity3while/lstm_cell_14/StatefulPartitionedCall:output:2^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/Identity_5

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
while_lstm_cell_14_26075057while_lstm_cell_14_26075057_0"<
while_lstm_cell_14_26075059while_lstm_cell_14_26075059_0"<
while_lstm_cell_14_26075061while_lstm_cell_14_26075061_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2X
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
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
Ô
I
-__inference_dropout_15_layer_call_fn_26079216

inputs
identityÊ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿr* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_dropout_15_layer_call_and_return_conditional_losses_260765392
PartitionedCallp
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿr:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
 
_user_specified_nameinputs
ã
Í
while_cond_26076906
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_26076906___redundant_placeholder06
2while_while_cond_26076906___redundant_placeholder16
2while_while_cond_26076906___redundant_placeholder26
2while_while_cond_26076906___redundant_placeholder3
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
B: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 

_output_shapes
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
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
¬\
 
E__inference_lstm_15_layer_call_and_return_conditional_losses_26078697
inputs_0?
+lstm_cell_15_matmul_readvariableop_resource:
È@
-lstm_cell_15_matmul_1_readvariableop_resource:	rÈ;
,lstm_cell_15_biasadd_readvariableop_resource:	È
identity¢#lstm_cell_15/BiasAdd/ReadVariableOp¢"lstm_cell_15/MatMul/ReadVariableOp¢$lstm_cell_15/MatMul_1/ReadVariableOp¢whileF
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
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :r2
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

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :r2
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
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :r2
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
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :r2
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
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2	
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
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
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
valueB"ÿÿÿÿ  27
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
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
strided_slice_2¶
"lstm_cell_15/MatMul/ReadVariableOpReadVariableOp+lstm_cell_15_matmul_readvariableop_resource* 
_output_shapes
:
È*
dtype02$
"lstm_cell_15/MatMul/ReadVariableOp­
lstm_cell_15/MatMulMatMulstrided_slice_2:output:0*lstm_cell_15/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell_15/MatMul»
$lstm_cell_15/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_15_matmul_1_readvariableop_resource*
_output_shapes
:	rÈ*
dtype02&
$lstm_cell_15/MatMul_1/ReadVariableOp©
lstm_cell_15/MatMul_1MatMulzeros:output:0,lstm_cell_15/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell_15/MatMul_1 
lstm_cell_15/addAddV2lstm_cell_15/MatMul:product:0lstm_cell_15/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell_15/add´
#lstm_cell_15/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_15_biasadd_readvariableop_resource*
_output_shapes	
:È*
dtype02%
#lstm_cell_15/BiasAdd/ReadVariableOp­
lstm_cell_15/BiasAddBiasAddlstm_cell_15/add:z:0+lstm_cell_15/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell_15/BiasAdd~
lstm_cell_15/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_15/split/split_dimó
lstm_cell_15/splitSplit%lstm_cell_15/split/split_dim:output:0lstm_cell_15/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿr:ÿÿÿÿÿÿÿÿÿr:ÿÿÿÿÿÿÿÿÿr:ÿÿÿÿÿÿÿÿÿr*
	num_split2
lstm_cell_15/split
lstm_cell_15/SigmoidSigmoidlstm_cell_15/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2
lstm_cell_15/Sigmoid
lstm_cell_15/Sigmoid_1Sigmoidlstm_cell_15/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2
lstm_cell_15/Sigmoid_1
lstm_cell_15/mulMullstm_cell_15/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2
lstm_cell_15/mul}
lstm_cell_15/ReluRelulstm_cell_15/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2
lstm_cell_15/Relu
lstm_cell_15/mul_1Mullstm_cell_15/Sigmoid:y:0lstm_cell_15/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2
lstm_cell_15/mul_1
lstm_cell_15/add_1AddV2lstm_cell_15/mul:z:0lstm_cell_15/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2
lstm_cell_15/add_1
lstm_cell_15/Sigmoid_2Sigmoidlstm_cell_15/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2
lstm_cell_15/Sigmoid_2|
lstm_cell_15/Relu_1Relulstm_cell_15/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2
lstm_cell_15/Relu_1 
lstm_cell_15/mul_2Mullstm_cell_15/Sigmoid_2:y:0!lstm_cell_15/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2
lstm_cell_15/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿr   2
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
while/loop_counter
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_15_matmul_readvariableop_resource-lstm_cell_15_matmul_1_readvariableop_resource,lstm_cell_15_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿr:ÿÿÿÿÿÿÿÿÿr: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_26078613*
condR
while_cond_26078612*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿr:ÿÿÿÿÿÿÿÿÿr: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿr   22
0TensorArrayV2Stack/TensorListStack/element_shapeñ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿr*
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
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm®
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿr2
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
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿr2

IdentityÈ
NoOpNoOp$^lstm_cell_15/BiasAdd/ReadVariableOp#^lstm_cell_15/MatMul/ReadVariableOp%^lstm_cell_15/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 2J
#lstm_cell_15/BiasAdd/ReadVariableOp#lstm_cell_15/BiasAdd/ReadVariableOp2H
"lstm_cell_15/MatMul/ReadVariableOp"lstm_cell_15/MatMul/ReadVariableOp2L
$lstm_cell_15/MatMul_1/ReadVariableOp$lstm_cell_15/MatMul_1/ReadVariableOp2
whilewhile:_ [
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0
åJ
Ô

lstm_14_while_body_26077228,
(lstm_14_while_lstm_14_while_loop_counter2
.lstm_14_while_lstm_14_while_maximum_iterations
lstm_14_while_placeholder
lstm_14_while_placeholder_1
lstm_14_while_placeholder_2
lstm_14_while_placeholder_3+
'lstm_14_while_lstm_14_strided_slice_1_0g
clstm_14_while_tensorarrayv2read_tensorlistgetitem_lstm_14_tensorarrayunstack_tensorlistfromtensor_0N
;lstm_14_while_lstm_cell_14_matmul_readvariableop_resource_0:	] Q
=lstm_14_while_lstm_cell_14_matmul_1_readvariableop_resource_0:
 K
<lstm_14_while_lstm_cell_14_biasadd_readvariableop_resource_0:	 
lstm_14_while_identity
lstm_14_while_identity_1
lstm_14_while_identity_2
lstm_14_while_identity_3
lstm_14_while_identity_4
lstm_14_while_identity_5)
%lstm_14_while_lstm_14_strided_slice_1e
alstm_14_while_tensorarrayv2read_tensorlistgetitem_lstm_14_tensorarrayunstack_tensorlistfromtensorL
9lstm_14_while_lstm_cell_14_matmul_readvariableop_resource:	] O
;lstm_14_while_lstm_cell_14_matmul_1_readvariableop_resource:
 I
:lstm_14_while_lstm_cell_14_biasadd_readvariableop_resource:	 ¢1lstm_14/while/lstm_cell_14/BiasAdd/ReadVariableOp¢0lstm_14/while/lstm_cell_14/MatMul/ReadVariableOp¢2lstm_14/while/lstm_cell_14/MatMul_1/ReadVariableOpÓ
?lstm_14/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ]   2A
?lstm_14/while/TensorArrayV2Read/TensorListGetItem/element_shape
1lstm_14/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemclstm_14_while_tensorarrayv2read_tensorlistgetitem_lstm_14_tensorarrayunstack_tensorlistfromtensor_0lstm_14_while_placeholderHlstm_14/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]*
element_dtype023
1lstm_14/while/TensorArrayV2Read/TensorListGetItemá
0lstm_14/while/lstm_cell_14/MatMul/ReadVariableOpReadVariableOp;lstm_14_while_lstm_cell_14_matmul_readvariableop_resource_0*
_output_shapes
:	] *
dtype022
0lstm_14/while/lstm_cell_14/MatMul/ReadVariableOp÷
!lstm_14/while/lstm_cell_14/MatMulMatMul8lstm_14/while/TensorArrayV2Read/TensorListGetItem:item:08lstm_14/while/lstm_cell_14/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!lstm_14/while/lstm_cell_14/MatMulè
2lstm_14/while/lstm_cell_14/MatMul_1/ReadVariableOpReadVariableOp=lstm_14_while_lstm_cell_14_matmul_1_readvariableop_resource_0* 
_output_shapes
:
 *
dtype024
2lstm_14/while/lstm_cell_14/MatMul_1/ReadVariableOpà
#lstm_14/while/lstm_cell_14/MatMul_1MatMullstm_14_while_placeholder_2:lstm_14/while/lstm_cell_14/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2%
#lstm_14/while/lstm_cell_14/MatMul_1Ø
lstm_14/while/lstm_cell_14/addAddV2+lstm_14/while/lstm_cell_14/MatMul:product:0-lstm_14/while/lstm_cell_14/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2 
lstm_14/while/lstm_cell_14/addà
1lstm_14/while/lstm_cell_14/BiasAdd/ReadVariableOpReadVariableOp<lstm_14_while_lstm_cell_14_biasadd_readvariableop_resource_0*
_output_shapes	
: *
dtype023
1lstm_14/while/lstm_cell_14/BiasAdd/ReadVariableOpå
"lstm_14/while/lstm_cell_14/BiasAddBiasAdd"lstm_14/while/lstm_cell_14/add:z:09lstm_14/while/lstm_cell_14/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2$
"lstm_14/while/lstm_cell_14/BiasAdd
*lstm_14/while/lstm_cell_14/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2,
*lstm_14/while/lstm_cell_14/split/split_dim¯
 lstm_14/while/lstm_cell_14/splitSplit3lstm_14/while/lstm_cell_14/split/split_dim:output:0+lstm_14/while/lstm_cell_14/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2"
 lstm_14/while/lstm_cell_14/split±
"lstm_14/while/lstm_cell_14/SigmoidSigmoid)lstm_14/while/lstm_cell_14/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2$
"lstm_14/while/lstm_cell_14/Sigmoidµ
$lstm_14/while/lstm_cell_14/Sigmoid_1Sigmoid)lstm_14/while/lstm_cell_14/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$lstm_14/while/lstm_cell_14/Sigmoid_1Á
lstm_14/while/lstm_cell_14/mulMul(lstm_14/while/lstm_cell_14/Sigmoid_1:y:0lstm_14_while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
lstm_14/while/lstm_cell_14/mul¨
lstm_14/while/lstm_cell_14/ReluRelu)lstm_14/while/lstm_cell_14/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
lstm_14/while/lstm_cell_14/ReluÕ
 lstm_14/while/lstm_cell_14/mul_1Mul&lstm_14/while/lstm_cell_14/Sigmoid:y:0-lstm_14/while/lstm_cell_14/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 lstm_14/while/lstm_cell_14/mul_1Ê
 lstm_14/while/lstm_cell_14/add_1AddV2"lstm_14/while/lstm_cell_14/mul:z:0$lstm_14/while/lstm_cell_14/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 lstm_14/while/lstm_cell_14/add_1µ
$lstm_14/while/lstm_cell_14/Sigmoid_2Sigmoid)lstm_14/while/lstm_cell_14/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$lstm_14/while/lstm_cell_14/Sigmoid_2§
!lstm_14/while/lstm_cell_14/Relu_1Relu$lstm_14/while/lstm_cell_14/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!lstm_14/while/lstm_cell_14/Relu_1Ù
 lstm_14/while/lstm_cell_14/mul_2Mul(lstm_14/while/lstm_cell_14/Sigmoid_2:y:0/lstm_14/while/lstm_cell_14/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 lstm_14/while/lstm_cell_14/mul_2
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
lstm_14/while/add/y
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
lstm_14/while/add_1/y
lstm_14/while/add_1AddV2(lstm_14_while_lstm_14_while_loop_counterlstm_14/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lstm_14/while/add_1
lstm_14/while/IdentityIdentitylstm_14/while/add_1:z:0^lstm_14/while/NoOp*
T0*
_output_shapes
: 2
lstm_14/while/Identity¦
lstm_14/while/Identity_1Identity.lstm_14_while_lstm_14_while_maximum_iterations^lstm_14/while/NoOp*
T0*
_output_shapes
: 2
lstm_14/while/Identity_1
lstm_14/while/Identity_2Identitylstm_14/while/add:z:0^lstm_14/while/NoOp*
T0*
_output_shapes
: 2
lstm_14/while/Identity_2º
lstm_14/while/Identity_3IdentityBlstm_14/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_14/while/NoOp*
T0*
_output_shapes
: 2
lstm_14/while/Identity_3®
lstm_14/while/Identity_4Identity$lstm_14/while/lstm_cell_14/mul_2:z:0^lstm_14/while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_14/while/Identity_4®
lstm_14/while/Identity_5Identity$lstm_14/while/lstm_cell_14/add_1:z:0^lstm_14/while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_14/while/Identity_5
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
9lstm_14_while_lstm_cell_14_matmul_readvariableop_resource;lstm_14_while_lstm_cell_14_matmul_readvariableop_resource_0"È
alstm_14_while_tensorarrayv2read_tensorlistgetitem_lstm_14_tensorarrayunstack_tensorlistfromtensorclstm_14_while_tensorarrayv2read_tensorlistgetitem_lstm_14_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2f
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
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
\

E__inference_lstm_14_layer_call_and_return_conditional_losses_26076361

inputs>
+lstm_cell_14_matmul_readvariableop_resource:	] A
-lstm_cell_14_matmul_1_readvariableop_resource:
 ;
,lstm_cell_14_biasadd_readvariableop_resource:	 
identity¢#lstm_cell_14/BiasAdd/ReadVariableOp¢"lstm_cell_14/MatMul/ReadVariableOp¢$lstm_cell_14/MatMul_1/ReadVariableOp¢whileD
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
B :2
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
B :2
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
:ÿÿÿÿÿÿÿÿÿ2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :2
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
B :2
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
:ÿÿÿÿÿÿÿÿÿ2	
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
"lstm_cell_14/MatMul/ReadVariableOpReadVariableOp+lstm_cell_14_matmul_readvariableop_resource*
_output_shapes
:	] *
dtype02$
"lstm_cell_14/MatMul/ReadVariableOp­
lstm_cell_14/MatMulMatMulstrided_slice_2:output:0*lstm_cell_14/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
lstm_cell_14/MatMul¼
$lstm_cell_14/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_14_matmul_1_readvariableop_resource* 
_output_shapes
:
 *
dtype02&
$lstm_cell_14/MatMul_1/ReadVariableOp©
lstm_cell_14/MatMul_1MatMulzeros:output:0,lstm_cell_14/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
lstm_cell_14/MatMul_1 
lstm_cell_14/addAddV2lstm_cell_14/MatMul:product:0lstm_cell_14/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
lstm_cell_14/add´
#lstm_cell_14/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_14_biasadd_readvariableop_resource*
_output_shapes	
: *
dtype02%
#lstm_cell_14/BiasAdd/ReadVariableOp­
lstm_cell_14/BiasAddBiasAddlstm_cell_14/add:z:0+lstm_cell_14/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
lstm_cell_14/BiasAdd~
lstm_cell_14/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_14/split/split_dim÷
lstm_cell_14/splitSplit%lstm_cell_14/split/split_dim:output:0lstm_cell_14/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2
lstm_cell_14/split
lstm_cell_14/SigmoidSigmoidlstm_cell_14/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_14/Sigmoid
lstm_cell_14/Sigmoid_1Sigmoidlstm_cell_14/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_14/Sigmoid_1
lstm_cell_14/mulMullstm_cell_14/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_14/mul~
lstm_cell_14/ReluRelulstm_cell_14/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_14/Relu
lstm_cell_14/mul_1Mullstm_cell_14/Sigmoid:y:0lstm_cell_14/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_14/mul_1
lstm_cell_14/add_1AddV2lstm_cell_14/mul:z:0lstm_cell_14/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_14/add_1
lstm_cell_14/Sigmoid_2Sigmoidlstm_cell_14/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_14/Sigmoid_2}
lstm_cell_14/Relu_1Relulstm_cell_14/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_14/Relu_1¡
lstm_cell_14/mul_2Mullstm_cell_14/Sigmoid_2:y:0!lstm_cell_14/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_14/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ  2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_14_matmul_readvariableop_resource-lstm_cell_14_matmul_1_readvariableop_resource,lstm_cell_14_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_26076277*
condR
while_cond_26076276*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ  22
0TensorArrayV2Stack/TensorListStack/element_shapeé
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
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
:ÿÿÿÿÿÿÿÿÿ*
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
:ÿÿÿÿÿÿÿÿÿ2
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
:ÿÿÿÿÿÿÿÿÿ2

IdentityÈ
NoOpNoOp$^lstm_cell_14/BiasAdd/ReadVariableOp#^lstm_cell_14/MatMul/ReadVariableOp%^lstm_cell_14/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ]: : : 2J
#lstm_cell_14/BiasAdd/ReadVariableOp#lstm_cell_14/BiasAdd/ReadVariableOp2H
"lstm_cell_14/MatMul/ReadVariableOp"lstm_cell_14/MatMul/ReadVariableOp2L
$lstm_cell_14/MatMul_1/ReadVariableOp$lstm_cell_14/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
 
_user_specified_nameinputs
ÓJ
Ô

lstm_15_while_body_26077376,
(lstm_15_while_lstm_15_while_loop_counter2
.lstm_15_while_lstm_15_while_maximum_iterations
lstm_15_while_placeholder
lstm_15_while_placeholder_1
lstm_15_while_placeholder_2
lstm_15_while_placeholder_3+
'lstm_15_while_lstm_15_strided_slice_1_0g
clstm_15_while_tensorarrayv2read_tensorlistgetitem_lstm_15_tensorarrayunstack_tensorlistfromtensor_0O
;lstm_15_while_lstm_cell_15_matmul_readvariableop_resource_0:
ÈP
=lstm_15_while_lstm_cell_15_matmul_1_readvariableop_resource_0:	rÈK
<lstm_15_while_lstm_cell_15_biasadd_readvariableop_resource_0:	È
lstm_15_while_identity
lstm_15_while_identity_1
lstm_15_while_identity_2
lstm_15_while_identity_3
lstm_15_while_identity_4
lstm_15_while_identity_5)
%lstm_15_while_lstm_15_strided_slice_1e
alstm_15_while_tensorarrayv2read_tensorlistgetitem_lstm_15_tensorarrayunstack_tensorlistfromtensorM
9lstm_15_while_lstm_cell_15_matmul_readvariableop_resource:
ÈN
;lstm_15_while_lstm_cell_15_matmul_1_readvariableop_resource:	rÈI
:lstm_15_while_lstm_cell_15_biasadd_readvariableop_resource:	È¢1lstm_15/while/lstm_cell_15/BiasAdd/ReadVariableOp¢0lstm_15/while/lstm_cell_15/MatMul/ReadVariableOp¢2lstm_15/while/lstm_cell_15/MatMul_1/ReadVariableOpÓ
?lstm_15/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ  2A
?lstm_15/while/TensorArrayV2Read/TensorListGetItem/element_shape
1lstm_15/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemclstm_15_while_tensorarrayv2read_tensorlistgetitem_lstm_15_tensorarrayunstack_tensorlistfromtensor_0lstm_15_while_placeholderHlstm_15/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype023
1lstm_15/while/TensorArrayV2Read/TensorListGetItemâ
0lstm_15/while/lstm_cell_15/MatMul/ReadVariableOpReadVariableOp;lstm_15_while_lstm_cell_15_matmul_readvariableop_resource_0* 
_output_shapes
:
È*
dtype022
0lstm_15/while/lstm_cell_15/MatMul/ReadVariableOp÷
!lstm_15/while/lstm_cell_15/MatMulMatMul8lstm_15/while/TensorArrayV2Read/TensorListGetItem:item:08lstm_15/while/lstm_cell_15/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2#
!lstm_15/while/lstm_cell_15/MatMulç
2lstm_15/while/lstm_cell_15/MatMul_1/ReadVariableOpReadVariableOp=lstm_15_while_lstm_cell_15_matmul_1_readvariableop_resource_0*
_output_shapes
:	rÈ*
dtype024
2lstm_15/while/lstm_cell_15/MatMul_1/ReadVariableOpà
#lstm_15/while/lstm_cell_15/MatMul_1MatMullstm_15_while_placeholder_2:lstm_15/while/lstm_cell_15/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2%
#lstm_15/while/lstm_cell_15/MatMul_1Ø
lstm_15/while/lstm_cell_15/addAddV2+lstm_15/while/lstm_cell_15/MatMul:product:0-lstm_15/while/lstm_cell_15/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2 
lstm_15/while/lstm_cell_15/addà
1lstm_15/while/lstm_cell_15/BiasAdd/ReadVariableOpReadVariableOp<lstm_15_while_lstm_cell_15_biasadd_readvariableop_resource_0*
_output_shapes	
:È*
dtype023
1lstm_15/while/lstm_cell_15/BiasAdd/ReadVariableOpå
"lstm_15/while/lstm_cell_15/BiasAddBiasAdd"lstm_15/while/lstm_cell_15/add:z:09lstm_15/while/lstm_cell_15/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2$
"lstm_15/while/lstm_cell_15/BiasAdd
*lstm_15/while/lstm_cell_15/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2,
*lstm_15/while/lstm_cell_15/split/split_dim«
 lstm_15/while/lstm_cell_15/splitSplit3lstm_15/while/lstm_cell_15/split/split_dim:output:0+lstm_15/while/lstm_cell_15/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿr:ÿÿÿÿÿÿÿÿÿr:ÿÿÿÿÿÿÿÿÿr:ÿÿÿÿÿÿÿÿÿr*
	num_split2"
 lstm_15/while/lstm_cell_15/split°
"lstm_15/while/lstm_cell_15/SigmoidSigmoid)lstm_15/while/lstm_cell_15/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2$
"lstm_15/while/lstm_cell_15/Sigmoid´
$lstm_15/while/lstm_cell_15/Sigmoid_1Sigmoid)lstm_15/while/lstm_cell_15/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2&
$lstm_15/while/lstm_cell_15/Sigmoid_1À
lstm_15/while/lstm_cell_15/mulMul(lstm_15/while/lstm_cell_15/Sigmoid_1:y:0lstm_15_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2 
lstm_15/while/lstm_cell_15/mul§
lstm_15/while/lstm_cell_15/ReluRelu)lstm_15/while/lstm_cell_15/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2!
lstm_15/while/lstm_cell_15/ReluÔ
 lstm_15/while/lstm_cell_15/mul_1Mul&lstm_15/while/lstm_cell_15/Sigmoid:y:0-lstm_15/while/lstm_cell_15/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2"
 lstm_15/while/lstm_cell_15/mul_1É
 lstm_15/while/lstm_cell_15/add_1AddV2"lstm_15/while/lstm_cell_15/mul:z:0$lstm_15/while/lstm_cell_15/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2"
 lstm_15/while/lstm_cell_15/add_1´
$lstm_15/while/lstm_cell_15/Sigmoid_2Sigmoid)lstm_15/while/lstm_cell_15/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2&
$lstm_15/while/lstm_cell_15/Sigmoid_2¦
!lstm_15/while/lstm_cell_15/Relu_1Relu$lstm_15/while/lstm_cell_15/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2#
!lstm_15/while/lstm_cell_15/Relu_1Ø
 lstm_15/while/lstm_cell_15/mul_2Mul(lstm_15/while/lstm_cell_15/Sigmoid_2:y:0/lstm_15/while/lstm_cell_15/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2"
 lstm_15/while/lstm_cell_15/mul_2
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
lstm_15/while/add/y
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
lstm_15/while/add_1/y
lstm_15/while/add_1AddV2(lstm_15_while_lstm_15_while_loop_counterlstm_15/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lstm_15/while/add_1
lstm_15/while/IdentityIdentitylstm_15/while/add_1:z:0^lstm_15/while/NoOp*
T0*
_output_shapes
: 2
lstm_15/while/Identity¦
lstm_15/while/Identity_1Identity.lstm_15_while_lstm_15_while_maximum_iterations^lstm_15/while/NoOp*
T0*
_output_shapes
: 2
lstm_15/while/Identity_1
lstm_15/while/Identity_2Identitylstm_15/while/add:z:0^lstm_15/while/NoOp*
T0*
_output_shapes
: 2
lstm_15/while/Identity_2º
lstm_15/while/Identity_3IdentityBlstm_15/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_15/while/NoOp*
T0*
_output_shapes
: 2
lstm_15/while/Identity_3­
lstm_15/while/Identity_4Identity$lstm_15/while/lstm_cell_15/mul_2:z:0^lstm_15/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2
lstm_15/while/Identity_4­
lstm_15/while/Identity_5Identity$lstm_15/while/lstm_cell_15/add_1:z:0^lstm_15/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2
lstm_15/while/Identity_5
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
9lstm_15_while_lstm_cell_15_matmul_readvariableop_resource;lstm_15_while_lstm_cell_15_matmul_readvariableop_resource_0"È
alstm_15_while_tensorarrayv2read_tensorlistgetitem_lstm_15_tensorarrayunstack_tensorlistfromtensorclstm_15_while_tensorarrayv2read_tensorlistgetitem_lstm_15_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿr:ÿÿÿÿÿÿÿÿÿr: : : : : 2f
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
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr:

_output_shapes
: :

_output_shapes
: 
ã
Í
while_cond_26078088
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_26078088___redundant_placeholder06
2while_while_cond_26078088___redundant_placeholder16
2while_while_cond_26078088___redundant_placeholder26
2while_while_cond_26078088___redundant_placeholder3
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
B: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 

_output_shapes
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
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
&
ó
while_body_26075243
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_00
while_lstm_cell_14_26075267_0:	] 1
while_lstm_cell_14_26075269_0:
 ,
while_lstm_cell_14_26075271_0:	 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor.
while_lstm_cell_14_26075267:	] /
while_lstm_cell_14_26075269:
 *
while_lstm_cell_14_26075271:	 ¢*while/lstm_cell_14/StatefulPartitionedCallÃ
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
*while/lstm_cell_14/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_14_26075267_0while_lstm_cell_14_26075269_0while_lstm_cell_14_26075271_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_lstm_cell_14_layer_call_and_return_conditional_losses_260751652,
*while/lstm_cell_14/StatefulPartitionedCall÷
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
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3¥
while/Identity_4Identity3while/lstm_cell_14/StatefulPartitionedCall:output:1^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/Identity_4¥
while/Identity_5Identity3while/lstm_cell_14/StatefulPartitionedCall:output:2^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/Identity_5

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
while_lstm_cell_14_26075267while_lstm_cell_14_26075267_0"<
while_lstm_cell_14_26075269while_lstm_cell_14_26075269_0"<
while_lstm_cell_14_26075271while_lstm_cell_14_26075271_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2X
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
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 


Ê
/__inference_sequential_7_layer_call_fn_26077871

inputs
unknown:	] 
	unknown_0:
 
	unknown_1:	 
	unknown_2:
È
	unknown_3:	rÈ
	unknown_4:	È
	unknown_5:r
	unknown_6:
identity¢StatefulPartitionedCallÌ
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
GPU 2J 8 *S
fNRL
J__inference_sequential_7_layer_call_and_return_conditional_losses_260770482
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


*__inference_dense_7_layer_call_fn_26079261

inputs
unknown:r
	unknown_0:
identity¢StatefulPartitionedCallù
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
GPU 2J 8 *N
fIRG
E__inference_dense_7_layer_call_and_return_conditional_losses_260765722
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
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿr: : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
 
_user_specified_nameinputs
Ã\
 
E__inference_lstm_14_layer_call_and_return_conditional_losses_26078173
inputs_0>
+lstm_cell_14_matmul_readvariableop_resource:	] A
-lstm_cell_14_matmul_1_readvariableop_resource:
 ;
,lstm_cell_14_biasadd_readvariableop_resource:	 
identity¢#lstm_cell_14/BiasAdd/ReadVariableOp¢"lstm_cell_14/MatMul/ReadVariableOp¢$lstm_cell_14/MatMul_1/ReadVariableOp¢whileF
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
B :2
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
B :2
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
:ÿÿÿÿÿÿÿÿÿ2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :2
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
B :2
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
:ÿÿÿÿÿÿÿÿÿ2	
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
"lstm_cell_14/MatMul/ReadVariableOpReadVariableOp+lstm_cell_14_matmul_readvariableop_resource*
_output_shapes
:	] *
dtype02$
"lstm_cell_14/MatMul/ReadVariableOp­
lstm_cell_14/MatMulMatMulstrided_slice_2:output:0*lstm_cell_14/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
lstm_cell_14/MatMul¼
$lstm_cell_14/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_14_matmul_1_readvariableop_resource* 
_output_shapes
:
 *
dtype02&
$lstm_cell_14/MatMul_1/ReadVariableOp©
lstm_cell_14/MatMul_1MatMulzeros:output:0,lstm_cell_14/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
lstm_cell_14/MatMul_1 
lstm_cell_14/addAddV2lstm_cell_14/MatMul:product:0lstm_cell_14/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
lstm_cell_14/add´
#lstm_cell_14/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_14_biasadd_readvariableop_resource*
_output_shapes	
: *
dtype02%
#lstm_cell_14/BiasAdd/ReadVariableOp­
lstm_cell_14/BiasAddBiasAddlstm_cell_14/add:z:0+lstm_cell_14/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
lstm_cell_14/BiasAdd~
lstm_cell_14/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_14/split/split_dim÷
lstm_cell_14/splitSplit%lstm_cell_14/split/split_dim:output:0lstm_cell_14/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2
lstm_cell_14/split
lstm_cell_14/SigmoidSigmoidlstm_cell_14/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_14/Sigmoid
lstm_cell_14/Sigmoid_1Sigmoidlstm_cell_14/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_14/Sigmoid_1
lstm_cell_14/mulMullstm_cell_14/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_14/mul~
lstm_cell_14/ReluRelulstm_cell_14/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_14/Relu
lstm_cell_14/mul_1Mullstm_cell_14/Sigmoid:y:0lstm_cell_14/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_14/mul_1
lstm_cell_14/add_1AddV2lstm_cell_14/mul:z:0lstm_cell_14/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_14/add_1
lstm_cell_14/Sigmoid_2Sigmoidlstm_cell_14/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_14/Sigmoid_2}
lstm_cell_14/Relu_1Relulstm_cell_14/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_14/Relu_1¡
lstm_cell_14/mul_2Mullstm_cell_14/Sigmoid_2:y:0!lstm_cell_14/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_14/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ  2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_14_matmul_readvariableop_resource-lstm_cell_14_matmul_1_readvariableop_resource,lstm_cell_14_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_26078089*
condR
while_cond_26078088*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ  22
0TensorArrayV2Stack/TensorListStack/element_shapeò
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
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
:ÿÿÿÿÿÿÿÿÿ*
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
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
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
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

IdentityÈ
NoOpNoOp$^lstm_cell_14/BiasAdd/ReadVariableOp#^lstm_cell_14/MatMul/ReadVariableOp%^lstm_cell_14/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ]: : : 2J
#lstm_cell_14/BiasAdd/ReadVariableOp#lstm_cell_14/BiasAdd/ReadVariableOp2H
"lstm_cell_14/MatMul/ReadVariableOp"lstm_cell_14/MatMul/ReadVariableOp2L
$lstm_cell_14/MatMul_1/ReadVariableOp$lstm_cell_14/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ]
"
_user_specified_name
inputs/0
?
Ô
while_body_26076442
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
3while_lstm_cell_15_matmul_readvariableop_resource_0:
ÈH
5while_lstm_cell_15_matmul_1_readvariableop_resource_0:	rÈC
4while_lstm_cell_15_biasadd_readvariableop_resource_0:	È
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
1while_lstm_cell_15_matmul_readvariableop_resource:
ÈF
3while_lstm_cell_15_matmul_1_readvariableop_resource:	rÈA
2while_lstm_cell_15_biasadd_readvariableop_resource:	È¢)while/lstm_cell_15/BiasAdd/ReadVariableOp¢(while/lstm_cell_15/MatMul/ReadVariableOp¢*while/lstm_cell_15/MatMul_1/ReadVariableOpÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ  29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÔ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemÊ
(while/lstm_cell_15/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_15_matmul_readvariableop_resource_0* 
_output_shapes
:
È*
dtype02*
(while/lstm_cell_15/MatMul/ReadVariableOp×
while/lstm_cell_15/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_15/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell_15/MatMulÏ
*while/lstm_cell_15/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_15_matmul_1_readvariableop_resource_0*
_output_shapes
:	rÈ*
dtype02,
*while/lstm_cell_15/MatMul_1/ReadVariableOpÀ
while/lstm_cell_15/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_15/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell_15/MatMul_1¸
while/lstm_cell_15/addAddV2#while/lstm_cell_15/MatMul:product:0%while/lstm_cell_15/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell_15/addÈ
)while/lstm_cell_15/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_15_biasadd_readvariableop_resource_0*
_output_shapes	
:È*
dtype02+
)while/lstm_cell_15/BiasAdd/ReadVariableOpÅ
while/lstm_cell_15/BiasAddBiasAddwhile/lstm_cell_15/add:z:01while/lstm_cell_15/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell_15/BiasAdd
"while/lstm_cell_15/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_15/split/split_dim
while/lstm_cell_15/splitSplit+while/lstm_cell_15/split/split_dim:output:0#while/lstm_cell_15/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿr:ÿÿÿÿÿÿÿÿÿr:ÿÿÿÿÿÿÿÿÿr:ÿÿÿÿÿÿÿÿÿr*
	num_split2
while/lstm_cell_15/split
while/lstm_cell_15/SigmoidSigmoid!while/lstm_cell_15/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2
while/lstm_cell_15/Sigmoid
while/lstm_cell_15/Sigmoid_1Sigmoid!while/lstm_cell_15/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2
while/lstm_cell_15/Sigmoid_1 
while/lstm_cell_15/mulMul while/lstm_cell_15/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2
while/lstm_cell_15/mul
while/lstm_cell_15/ReluRelu!while/lstm_cell_15/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2
while/lstm_cell_15/Relu´
while/lstm_cell_15/mul_1Mulwhile/lstm_cell_15/Sigmoid:y:0%while/lstm_cell_15/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2
while/lstm_cell_15/mul_1©
while/lstm_cell_15/add_1AddV2while/lstm_cell_15/mul:z:0while/lstm_cell_15/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2
while/lstm_cell_15/add_1
while/lstm_cell_15/Sigmoid_2Sigmoid!while/lstm_cell_15/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2
while/lstm_cell_15/Sigmoid_2
while/lstm_cell_15/Relu_1Reluwhile/lstm_cell_15/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2
while/lstm_cell_15/Relu_1¸
while/lstm_cell_15/mul_2Mul while/lstm_cell_15/Sigmoid_2:y:0'while/lstm_cell_15/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2
while/lstm_cell_15/mul_2à
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
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_15/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_15/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2
while/Identity_5Þ

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
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿr:ÿÿÿÿÿÿÿÿÿr: : : : : 2V
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
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr:

_output_shapes
: :

_output_shapes
: 
?
Ô
while_body_26078915
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
3while_lstm_cell_15_matmul_readvariableop_resource_0:
ÈH
5while_lstm_cell_15_matmul_1_readvariableop_resource_0:	rÈC
4while_lstm_cell_15_biasadd_readvariableop_resource_0:	È
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
1while_lstm_cell_15_matmul_readvariableop_resource:
ÈF
3while_lstm_cell_15_matmul_1_readvariableop_resource:	rÈA
2while_lstm_cell_15_biasadd_readvariableop_resource:	È¢)while/lstm_cell_15/BiasAdd/ReadVariableOp¢(while/lstm_cell_15/MatMul/ReadVariableOp¢*while/lstm_cell_15/MatMul_1/ReadVariableOpÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ  29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÔ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemÊ
(while/lstm_cell_15/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_15_matmul_readvariableop_resource_0* 
_output_shapes
:
È*
dtype02*
(while/lstm_cell_15/MatMul/ReadVariableOp×
while/lstm_cell_15/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_15/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell_15/MatMulÏ
*while/lstm_cell_15/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_15_matmul_1_readvariableop_resource_0*
_output_shapes
:	rÈ*
dtype02,
*while/lstm_cell_15/MatMul_1/ReadVariableOpÀ
while/lstm_cell_15/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_15/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell_15/MatMul_1¸
while/lstm_cell_15/addAddV2#while/lstm_cell_15/MatMul:product:0%while/lstm_cell_15/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell_15/addÈ
)while/lstm_cell_15/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_15_biasadd_readvariableop_resource_0*
_output_shapes	
:È*
dtype02+
)while/lstm_cell_15/BiasAdd/ReadVariableOpÅ
while/lstm_cell_15/BiasAddBiasAddwhile/lstm_cell_15/add:z:01while/lstm_cell_15/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell_15/BiasAdd
"while/lstm_cell_15/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_15/split/split_dim
while/lstm_cell_15/splitSplit+while/lstm_cell_15/split/split_dim:output:0#while/lstm_cell_15/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿr:ÿÿÿÿÿÿÿÿÿr:ÿÿÿÿÿÿÿÿÿr:ÿÿÿÿÿÿÿÿÿr*
	num_split2
while/lstm_cell_15/split
while/lstm_cell_15/SigmoidSigmoid!while/lstm_cell_15/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2
while/lstm_cell_15/Sigmoid
while/lstm_cell_15/Sigmoid_1Sigmoid!while/lstm_cell_15/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2
while/lstm_cell_15/Sigmoid_1 
while/lstm_cell_15/mulMul while/lstm_cell_15/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2
while/lstm_cell_15/mul
while/lstm_cell_15/ReluRelu!while/lstm_cell_15/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2
while/lstm_cell_15/Relu´
while/lstm_cell_15/mul_1Mulwhile/lstm_cell_15/Sigmoid:y:0%while/lstm_cell_15/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2
while/lstm_cell_15/mul_1©
while/lstm_cell_15/add_1AddV2while/lstm_cell_15/mul:z:0while/lstm_cell_15/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2
while/lstm_cell_15/add_1
while/lstm_cell_15/Sigmoid_2Sigmoid!while/lstm_cell_15/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2
while/lstm_cell_15/Sigmoid_2
while/lstm_cell_15/Relu_1Reluwhile/lstm_cell_15/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2
while/lstm_cell_15/Relu_1¸
while/lstm_cell_15/mul_2Mul while/lstm_cell_15/Sigmoid_2:y:0'while/lstm_cell_15/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2
while/lstm_cell_15/mul_2à
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
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_15/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_15/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2
while/Identity_5Þ

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
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿr:ÿÿÿÿÿÿÿÿÿr: : : : : 2V
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
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr:

_output_shapes
: :

_output_shapes
: 
¬
ò
J__inference_sequential_7_layer_call_and_return_conditional_losses_26077113
lstm_14_input#
lstm_14_26077091:	] $
lstm_14_26077093:
 
lstm_14_26077095:	 $
lstm_15_26077099:
È#
lstm_15_26077101:	rÈ
lstm_15_26077103:	È"
dense_7_26077107:r
dense_7_26077109:
identity¢dense_7/StatefulPartitionedCall¢lstm_14/StatefulPartitionedCall¢lstm_15/StatefulPartitionedCallµ
lstm_14/StatefulPartitionedCallStatefulPartitionedCalllstm_14_inputlstm_14_26077091lstm_14_26077093lstm_14_26077095*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_lstm_14_layer_call_and_return_conditional_losses_260763612!
lstm_14/StatefulPartitionedCall
dropout_14/PartitionedCallPartitionedCall(lstm_14/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_dropout_14_layer_call_and_return_conditional_losses_260763742
dropout_14/PartitionedCallÊ
lstm_15/StatefulPartitionedCallStatefulPartitionedCall#dropout_14/PartitionedCall:output:0lstm_15_26077099lstm_15_26077101lstm_15_26077103*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿr*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_lstm_15_layer_call_and_return_conditional_losses_260765262!
lstm_15/StatefulPartitionedCall
dropout_15/PartitionedCallPartitionedCall(lstm_15/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿr* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_dropout_15_layer_call_and_return_conditional_losses_260765392
dropout_15/PartitionedCall¶
dense_7/StatefulPartitionedCallStatefulPartitionedCall#dropout_15/PartitionedCall:output:0dense_7_26077107dense_7_26077109*
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
GPU 2J 8 *N
fIRG
E__inference_dense_7_layer_call_and_return_conditional_losses_260765722!
dense_7/StatefulPartitionedCall
IdentityIdentity(dense_7/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity´
NoOpNoOp ^dense_7/StatefulPartitionedCall ^lstm_14/StatefulPartitionedCall ^lstm_15/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ]: : : : : : : : 2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall2B
lstm_14/StatefulPartitionedCalllstm_14/StatefulPartitionedCall2B
lstm_15/StatefulPartitionedCalllstm_15/StatefulPartitionedCall:Z V
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
'
_user_specified_namelstm_14_input


J__inference_lstm_cell_14_layer_call_and_return_conditional_losses_26075019

inputs

states
states_11
matmul_readvariableop_resource:	] 4
 matmul_1_readvariableop_resource:
 .
biasadd_readvariableop_resource:	 
identity

identity_1

identity_2¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	] *
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
MatMul
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
 *
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
add
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
: *
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2	
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
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2
split`
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
Sigmoidd
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	Sigmoid_1]
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mulW
ReluRelusplit:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relui
mul_1MulSigmoid:y:0Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mul_1^
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
add_1d
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	Sigmoid_2V
Relu_1Relu	add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu_1m
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mul_2e
IdentityIdentity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityi

Identity_1Identity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1i

Identity_2Identity	add_1:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

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
A:ÿÿÿÿÿÿÿÿÿ]:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
 
_user_specified_nameinputs:PL
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_namestates:PL
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_namestates
ã
Í
while_cond_26075032
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_26075032___redundant_placeholder06
2while_while_cond_26075032___redundant_placeholder16
2while_while_cond_26075032___redundant_placeholder26
2while_while_cond_26075032___redundant_placeholder3
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
B: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 

_output_shapes
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
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
»
f
-__inference_dropout_14_layer_call_fn_26078546

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
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_dropout_14_layer_call_and_return_conditional_losses_260768242
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
µ
¸
*__inference_lstm_15_layer_call_fn_26079194

inputs
unknown:
È
	unknown_0:	rÈ
	unknown_1:	È
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿr*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_lstm_15_layer_call_and_return_conditional_losses_260767952
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:ÿÿÿÿÿÿÿÿÿ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ß
Í
while_cond_26078914
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_26078914___redundant_placeholder06
2while_while_cond_26078914___redundant_placeholder16
2while_while_cond_26078914___redundant_placeholder26
2while_while_cond_26078914___redundant_placeholder3
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
@: : : : :ÿÿÿÿÿÿÿÿÿr:ÿÿÿÿÿÿÿÿÿr: ::::: 

_output_shapes
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
:ÿÿÿÿÿÿÿÿÿr:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr:

_output_shapes
: :

_output_shapes
:
í[

E__inference_lstm_15_layer_call_and_return_conditional_losses_26078999

inputs?
+lstm_cell_15_matmul_readvariableop_resource:
È@
-lstm_cell_15_matmul_1_readvariableop_resource:	rÈ;
,lstm_cell_15_biasadd_readvariableop_resource:	È
identity¢#lstm_cell_15/BiasAdd/ReadVariableOp¢"lstm_cell_15/MatMul/ReadVariableOp¢$lstm_cell_15/MatMul_1/ReadVariableOp¢whileD
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
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :r2
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

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :r2
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
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :r2
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
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :r2
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
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2	
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
:ÿÿÿÿÿÿÿÿÿ2
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
valueB"ÿÿÿÿ  27
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
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
strided_slice_2¶
"lstm_cell_15/MatMul/ReadVariableOpReadVariableOp+lstm_cell_15_matmul_readvariableop_resource* 
_output_shapes
:
È*
dtype02$
"lstm_cell_15/MatMul/ReadVariableOp­
lstm_cell_15/MatMulMatMulstrided_slice_2:output:0*lstm_cell_15/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell_15/MatMul»
$lstm_cell_15/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_15_matmul_1_readvariableop_resource*
_output_shapes
:	rÈ*
dtype02&
$lstm_cell_15/MatMul_1/ReadVariableOp©
lstm_cell_15/MatMul_1MatMulzeros:output:0,lstm_cell_15/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell_15/MatMul_1 
lstm_cell_15/addAddV2lstm_cell_15/MatMul:product:0lstm_cell_15/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell_15/add´
#lstm_cell_15/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_15_biasadd_readvariableop_resource*
_output_shapes	
:È*
dtype02%
#lstm_cell_15/BiasAdd/ReadVariableOp­
lstm_cell_15/BiasAddBiasAddlstm_cell_15/add:z:0+lstm_cell_15/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell_15/BiasAdd~
lstm_cell_15/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_15/split/split_dimó
lstm_cell_15/splitSplit%lstm_cell_15/split/split_dim:output:0lstm_cell_15/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿr:ÿÿÿÿÿÿÿÿÿr:ÿÿÿÿÿÿÿÿÿr:ÿÿÿÿÿÿÿÿÿr*
	num_split2
lstm_cell_15/split
lstm_cell_15/SigmoidSigmoidlstm_cell_15/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2
lstm_cell_15/Sigmoid
lstm_cell_15/Sigmoid_1Sigmoidlstm_cell_15/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2
lstm_cell_15/Sigmoid_1
lstm_cell_15/mulMullstm_cell_15/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2
lstm_cell_15/mul}
lstm_cell_15/ReluRelulstm_cell_15/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2
lstm_cell_15/Relu
lstm_cell_15/mul_1Mullstm_cell_15/Sigmoid:y:0lstm_cell_15/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2
lstm_cell_15/mul_1
lstm_cell_15/add_1AddV2lstm_cell_15/mul:z:0lstm_cell_15/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2
lstm_cell_15/add_1
lstm_cell_15/Sigmoid_2Sigmoidlstm_cell_15/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2
lstm_cell_15/Sigmoid_2|
lstm_cell_15/Relu_1Relulstm_cell_15/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2
lstm_cell_15/Relu_1 
lstm_cell_15/mul_2Mullstm_cell_15/Sigmoid_2:y:0!lstm_cell_15/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2
lstm_cell_15/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿr   2
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
while/loop_counter
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_15_matmul_readvariableop_resource-lstm_cell_15_matmul_1_readvariableop_resource,lstm_cell_15_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿr:ÿÿÿÿÿÿÿÿÿr: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_26078915*
condR
while_cond_26078914*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿr:ÿÿÿÿÿÿÿÿÿr: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿr   22
0TensorArrayV2Stack/TensorListStack/element_shapeè
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿr*
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
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm¥
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2
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
:ÿÿÿÿÿÿÿÿÿr2

IdentityÈ
NoOpNoOp$^lstm_cell_15/BiasAdd/ReadVariableOp#^lstm_cell_15/MatMul/ReadVariableOp%^lstm_cell_15/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:ÿÿÿÿÿÿÿÿÿ: : : 2J
#lstm_cell_15/BiasAdd/ReadVariableOp#lstm_cell_15/BiasAdd/ReadVariableOp2H
"lstm_cell_15/MatMul/ReadVariableOp"lstm_cell_15/MatMul/ReadVariableOp2L
$lstm_cell_15/MatMul_1/ReadVariableOp$lstm_cell_15/MatMul_1/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¶
¸
*__inference_lstm_14_layer_call_fn_26078508

inputs
unknown:	] 
	unknown_0:
 
	unknown_1:	 
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_lstm_14_layer_call_and_return_conditional_losses_260763612
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

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
°?
Ô
while_body_26078240
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_14_matmul_readvariableop_resource_0:	] I
5while_lstm_cell_14_matmul_1_readvariableop_resource_0:
 C
4while_lstm_cell_14_biasadd_readvariableop_resource_0:	 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_14_matmul_readvariableop_resource:	] G
3while_lstm_cell_14_matmul_1_readvariableop_resource:
 A
2while_lstm_cell_14_biasadd_readvariableop_resource:	 ¢)while/lstm_cell_14/BiasAdd/ReadVariableOp¢(while/lstm_cell_14/MatMul/ReadVariableOp¢*while/lstm_cell_14/MatMul_1/ReadVariableOpÃ
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
(while/lstm_cell_14/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_14_matmul_readvariableop_resource_0*
_output_shapes
:	] *
dtype02*
(while/lstm_cell_14/MatMul/ReadVariableOp×
while/lstm_cell_14/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_14/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/lstm_cell_14/MatMulÐ
*while/lstm_cell_14/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_14_matmul_1_readvariableop_resource_0* 
_output_shapes
:
 *
dtype02,
*while/lstm_cell_14/MatMul_1/ReadVariableOpÀ
while/lstm_cell_14/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_14/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/lstm_cell_14/MatMul_1¸
while/lstm_cell_14/addAddV2#while/lstm_cell_14/MatMul:product:0%while/lstm_cell_14/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/lstm_cell_14/addÈ
)while/lstm_cell_14/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_14_biasadd_readvariableop_resource_0*
_output_shapes	
: *
dtype02+
)while/lstm_cell_14/BiasAdd/ReadVariableOpÅ
while/lstm_cell_14/BiasAddBiasAddwhile/lstm_cell_14/add:z:01while/lstm_cell_14/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/lstm_cell_14/BiasAdd
"while/lstm_cell_14/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_14/split/split_dim
while/lstm_cell_14/splitSplit+while/lstm_cell_14/split/split_dim:output:0#while/lstm_cell_14/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2
while/lstm_cell_14/split
while/lstm_cell_14/SigmoidSigmoid!while/lstm_cell_14/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_14/Sigmoid
while/lstm_cell_14/Sigmoid_1Sigmoid!while/lstm_cell_14/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_14/Sigmoid_1¡
while/lstm_cell_14/mulMul while/lstm_cell_14/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_14/mul
while/lstm_cell_14/ReluRelu!while/lstm_cell_14/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_14/Reluµ
while/lstm_cell_14/mul_1Mulwhile/lstm_cell_14/Sigmoid:y:0%while/lstm_cell_14/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_14/mul_1ª
while/lstm_cell_14/add_1AddV2while/lstm_cell_14/mul:z:0while/lstm_cell_14/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_14/add_1
while/lstm_cell_14/Sigmoid_2Sigmoid!while/lstm_cell_14/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_14/Sigmoid_2
while/lstm_cell_14/Relu_1Reluwhile/lstm_cell_14/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_14/Relu_1¹
while/lstm_cell_14/mul_2Mul while/lstm_cell_14/Sigmoid_2:y:0'while/lstm_cell_14/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_14/mul_2à
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
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_14/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_14/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/Identity_5Þ

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
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2V
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
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
×
ñ
(sequential_7_lstm_15_while_cond_26074831F
Bsequential_7_lstm_15_while_sequential_7_lstm_15_while_loop_counterL
Hsequential_7_lstm_15_while_sequential_7_lstm_15_while_maximum_iterations*
&sequential_7_lstm_15_while_placeholder,
(sequential_7_lstm_15_while_placeholder_1,
(sequential_7_lstm_15_while_placeholder_2,
(sequential_7_lstm_15_while_placeholder_3H
Dsequential_7_lstm_15_while_less_sequential_7_lstm_15_strided_slice_1`
\sequential_7_lstm_15_while_sequential_7_lstm_15_while_cond_26074831___redundant_placeholder0`
\sequential_7_lstm_15_while_sequential_7_lstm_15_while_cond_26074831___redundant_placeholder1`
\sequential_7_lstm_15_while_sequential_7_lstm_15_while_cond_26074831___redundant_placeholder2`
\sequential_7_lstm_15_while_sequential_7_lstm_15_while_cond_26074831___redundant_placeholder3'
#sequential_7_lstm_15_while_identity
Ù
sequential_7/lstm_15/while/LessLess&sequential_7_lstm_15_while_placeholderDsequential_7_lstm_15_while_less_sequential_7_lstm_15_strided_slice_1*
T0*
_output_shapes
: 2!
sequential_7/lstm_15/while/Less
#sequential_7/lstm_15/while/IdentityIdentity#sequential_7/lstm_15/while/Less:z:0*
T0
*
_output_shapes
: 2%
#sequential_7/lstm_15/while/Identity"S
#sequential_7_lstm_15_while_identity,sequential_7/lstm_15/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿr:ÿÿÿÿÿÿÿÿÿr: ::::: 

_output_shapes
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
:ÿÿÿÿÿÿÿÿÿr:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr:

_output_shapes
: :

_output_shapes
:
Ð

í
lstm_15_while_cond_26077375,
(lstm_15_while_lstm_15_while_loop_counter2
.lstm_15_while_lstm_15_while_maximum_iterations
lstm_15_while_placeholder
lstm_15_while_placeholder_1
lstm_15_while_placeholder_2
lstm_15_while_placeholder_3.
*lstm_15_while_less_lstm_15_strided_slice_1F
Blstm_15_while_lstm_15_while_cond_26077375___redundant_placeholder0F
Blstm_15_while_lstm_15_while_cond_26077375___redundant_placeholder1F
Blstm_15_while_lstm_15_while_cond_26077375___redundant_placeholder2F
Blstm_15_while_lstm_15_while_cond_26077375___redundant_placeholder3
lstm_15_while_identity

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
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿr:ÿÿÿÿÿÿÿÿÿr: ::::: 

_output_shapes
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
:ÿÿÿÿÿÿÿÿÿr:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr:

_output_shapes
: :

_output_shapes
:
?
Ô
while_body_26078764
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
3while_lstm_cell_15_matmul_readvariableop_resource_0:
ÈH
5while_lstm_cell_15_matmul_1_readvariableop_resource_0:	rÈC
4while_lstm_cell_15_biasadd_readvariableop_resource_0:	È
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
1while_lstm_cell_15_matmul_readvariableop_resource:
ÈF
3while_lstm_cell_15_matmul_1_readvariableop_resource:	rÈA
2while_lstm_cell_15_biasadd_readvariableop_resource:	È¢)while/lstm_cell_15/BiasAdd/ReadVariableOp¢(while/lstm_cell_15/MatMul/ReadVariableOp¢*while/lstm_cell_15/MatMul_1/ReadVariableOpÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ  29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÔ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemÊ
(while/lstm_cell_15/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_15_matmul_readvariableop_resource_0* 
_output_shapes
:
È*
dtype02*
(while/lstm_cell_15/MatMul/ReadVariableOp×
while/lstm_cell_15/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_15/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell_15/MatMulÏ
*while/lstm_cell_15/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_15_matmul_1_readvariableop_resource_0*
_output_shapes
:	rÈ*
dtype02,
*while/lstm_cell_15/MatMul_1/ReadVariableOpÀ
while/lstm_cell_15/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_15/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell_15/MatMul_1¸
while/lstm_cell_15/addAddV2#while/lstm_cell_15/MatMul:product:0%while/lstm_cell_15/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell_15/addÈ
)while/lstm_cell_15/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_15_biasadd_readvariableop_resource_0*
_output_shapes	
:È*
dtype02+
)while/lstm_cell_15/BiasAdd/ReadVariableOpÅ
while/lstm_cell_15/BiasAddBiasAddwhile/lstm_cell_15/add:z:01while/lstm_cell_15/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell_15/BiasAdd
"while/lstm_cell_15/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_15/split/split_dim
while/lstm_cell_15/splitSplit+while/lstm_cell_15/split/split_dim:output:0#while/lstm_cell_15/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿr:ÿÿÿÿÿÿÿÿÿr:ÿÿÿÿÿÿÿÿÿr:ÿÿÿÿÿÿÿÿÿr*
	num_split2
while/lstm_cell_15/split
while/lstm_cell_15/SigmoidSigmoid!while/lstm_cell_15/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2
while/lstm_cell_15/Sigmoid
while/lstm_cell_15/Sigmoid_1Sigmoid!while/lstm_cell_15/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2
while/lstm_cell_15/Sigmoid_1 
while/lstm_cell_15/mulMul while/lstm_cell_15/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2
while/lstm_cell_15/mul
while/lstm_cell_15/ReluRelu!while/lstm_cell_15/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2
while/lstm_cell_15/Relu´
while/lstm_cell_15/mul_1Mulwhile/lstm_cell_15/Sigmoid:y:0%while/lstm_cell_15/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2
while/lstm_cell_15/mul_1©
while/lstm_cell_15/add_1AddV2while/lstm_cell_15/mul:z:0while/lstm_cell_15/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2
while/lstm_cell_15/add_1
while/lstm_cell_15/Sigmoid_2Sigmoid!while/lstm_cell_15/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2
while/lstm_cell_15/Sigmoid_2
while/lstm_cell_15/Relu_1Reluwhile/lstm_cell_15/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2
while/lstm_cell_15/Relu_1¸
while/lstm_cell_15/mul_2Mul while/lstm_cell_15/Sigmoid_2:y:0'while/lstm_cell_15/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2
while/lstm_cell_15/mul_2à
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
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_15/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_15/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2
while/Identity_5Þ

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
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿr:ÿÿÿÿÿÿÿÿÿr: : : : : 2V
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
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr:

_output_shapes
: :

_output_shapes
: 
Ï!
ü
E__inference_dense_7_layer_call_and_return_conditional_losses_26076572

inputs3
!tensordot_readvariableop_resource:r-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢Tensordot/ReadVariableOp
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:r*
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
Tensordot/stack
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2
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
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿr: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
 
_user_specified_nameinputs
ß
Í
while_cond_26078612
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_26078612___redundant_placeholder06
2while_while_cond_26078612___redundant_placeholder16
2while_while_cond_26078612___redundant_placeholder26
2while_while_cond_26078612___redundant_placeholder3
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
@: : : : :ÿÿÿÿÿÿÿÿÿr:ÿÿÿÿÿÿÿÿÿr: ::::: 

_output_shapes
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
:ÿÿÿÿÿÿÿÿÿr:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr:

_output_shapes
: :

_output_shapes
:
¾F

E__inference_lstm_15_layer_call_and_return_conditional_losses_26075942

inputs)
lstm_cell_15_26075860:
È(
lstm_cell_15_26075862:	rÈ$
lstm_cell_15_26075864:	È
identity¢$lstm_cell_15/StatefulPartitionedCall¢whileD
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
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :r2
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

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :r2
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
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :r2
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
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :r2
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
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2	
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
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
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
valueB"ÿÿÿÿ  27
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
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
strided_slice_2¥
$lstm_cell_15/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_15_26075860lstm_cell_15_26075862lstm_cell_15_26075864*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿr:ÿÿÿÿÿÿÿÿÿr:ÿÿÿÿÿÿÿÿÿr*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_lstm_cell_15_layer_call_and_return_conditional_losses_260757952&
$lstm_cell_15/StatefulPartitionedCall
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿr   2
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
while/loop_counterÊ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_15_26075860lstm_cell_15_26075862lstm_cell_15_26075864*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿr:ÿÿÿÿÿÿÿÿÿr: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_26075873*
condR
while_cond_26075872*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿr:ÿÿÿÿÿÿÿÿÿr: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿr   22
0TensorArrayV2Stack/TensorListStack/element_shapeñ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿr*
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
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm®
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿr2
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
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿr2

Identity}
NoOpNoOp%^lstm_cell_15/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 2L
$lstm_cell_15/StatefulPartitionedCall$lstm_cell_15/StatefulPartitionedCall2
whilewhile:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
è³
Ï	
#__inference__wrapped_model_26074944
lstm_14_inputS
@sequential_7_lstm_14_lstm_cell_14_matmul_readvariableop_resource:	] V
Bsequential_7_lstm_14_lstm_cell_14_matmul_1_readvariableop_resource:
 P
Asequential_7_lstm_14_lstm_cell_14_biasadd_readvariableop_resource:	 T
@sequential_7_lstm_15_lstm_cell_15_matmul_readvariableop_resource:
ÈU
Bsequential_7_lstm_15_lstm_cell_15_matmul_1_readvariableop_resource:	rÈP
Asequential_7_lstm_15_lstm_cell_15_biasadd_readvariableop_resource:	ÈH
6sequential_7_dense_7_tensordot_readvariableop_resource:rB
4sequential_7_dense_7_biasadd_readvariableop_resource:
identity¢+sequential_7/dense_7/BiasAdd/ReadVariableOp¢-sequential_7/dense_7/Tensordot/ReadVariableOp¢8sequential_7/lstm_14/lstm_cell_14/BiasAdd/ReadVariableOp¢7sequential_7/lstm_14/lstm_cell_14/MatMul/ReadVariableOp¢9sequential_7/lstm_14/lstm_cell_14/MatMul_1/ReadVariableOp¢sequential_7/lstm_14/while¢8sequential_7/lstm_15/lstm_cell_15/BiasAdd/ReadVariableOp¢7sequential_7/lstm_15/lstm_cell_15/MatMul/ReadVariableOp¢9sequential_7/lstm_15/lstm_cell_15/MatMul_1/ReadVariableOp¢sequential_7/lstm_15/whileu
sequential_7/lstm_14/ShapeShapelstm_14_input*
T0*
_output_shapes
:2
sequential_7/lstm_14/Shape
(sequential_7/lstm_14/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(sequential_7/lstm_14/strided_slice/stack¢
*sequential_7/lstm_14/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*sequential_7/lstm_14/strided_slice/stack_1¢
*sequential_7/lstm_14/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*sequential_7/lstm_14/strided_slice/stack_2à
"sequential_7/lstm_14/strided_sliceStridedSlice#sequential_7/lstm_14/Shape:output:01sequential_7/lstm_14/strided_slice/stack:output:03sequential_7/lstm_14/strided_slice/stack_1:output:03sequential_7/lstm_14/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"sequential_7/lstm_14/strided_slice
 sequential_7/lstm_14/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :2"
 sequential_7/lstm_14/zeros/mul/yÀ
sequential_7/lstm_14/zeros/mulMul+sequential_7/lstm_14/strided_slice:output:0)sequential_7/lstm_14/zeros/mul/y:output:0*
T0*
_output_shapes
: 2 
sequential_7/lstm_14/zeros/mul
!sequential_7/lstm_14/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2#
!sequential_7/lstm_14/zeros/Less/y»
sequential_7/lstm_14/zeros/LessLess"sequential_7/lstm_14/zeros/mul:z:0*sequential_7/lstm_14/zeros/Less/y:output:0*
T0*
_output_shapes
: 2!
sequential_7/lstm_14/zeros/Less
#sequential_7/lstm_14/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :2%
#sequential_7/lstm_14/zeros/packed/1×
!sequential_7/lstm_14/zeros/packedPack+sequential_7/lstm_14/strided_slice:output:0,sequential_7/lstm_14/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2#
!sequential_7/lstm_14/zeros/packed
 sequential_7/lstm_14/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 sequential_7/lstm_14/zeros/ConstÊ
sequential_7/lstm_14/zerosFill*sequential_7/lstm_14/zeros/packed:output:0)sequential_7/lstm_14/zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential_7/lstm_14/zeros
"sequential_7/lstm_14/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :2$
"sequential_7/lstm_14/zeros_1/mul/yÆ
 sequential_7/lstm_14/zeros_1/mulMul+sequential_7/lstm_14/strided_slice:output:0+sequential_7/lstm_14/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2"
 sequential_7/lstm_14/zeros_1/mul
#sequential_7/lstm_14/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2%
#sequential_7/lstm_14/zeros_1/Less/yÃ
!sequential_7/lstm_14/zeros_1/LessLess$sequential_7/lstm_14/zeros_1/mul:z:0,sequential_7/lstm_14/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2#
!sequential_7/lstm_14/zeros_1/Less
%sequential_7/lstm_14/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :2'
%sequential_7/lstm_14/zeros_1/packed/1Ý
#sequential_7/lstm_14/zeros_1/packedPack+sequential_7/lstm_14/strided_slice:output:0.sequential_7/lstm_14/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2%
#sequential_7/lstm_14/zeros_1/packed
"sequential_7/lstm_14/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"sequential_7/lstm_14/zeros_1/ConstÒ
sequential_7/lstm_14/zeros_1Fill,sequential_7/lstm_14/zeros_1/packed:output:0+sequential_7/lstm_14/zeros_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential_7/lstm_14/zeros_1
#sequential_7/lstm_14/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2%
#sequential_7/lstm_14/transpose/permÀ
sequential_7/lstm_14/transpose	Transposelstm_14_input,sequential_7/lstm_14/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]2 
sequential_7/lstm_14/transpose
sequential_7/lstm_14/Shape_1Shape"sequential_7/lstm_14/transpose:y:0*
T0*
_output_shapes
:2
sequential_7/lstm_14/Shape_1¢
*sequential_7/lstm_14/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2,
*sequential_7/lstm_14/strided_slice_1/stack¦
,sequential_7/lstm_14/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_7/lstm_14/strided_slice_1/stack_1¦
,sequential_7/lstm_14/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_7/lstm_14/strided_slice_1/stack_2ì
$sequential_7/lstm_14/strided_slice_1StridedSlice%sequential_7/lstm_14/Shape_1:output:03sequential_7/lstm_14/strided_slice_1/stack:output:05sequential_7/lstm_14/strided_slice_1/stack_1:output:05sequential_7/lstm_14/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2&
$sequential_7/lstm_14/strided_slice_1¯
0sequential_7/lstm_14/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ22
0sequential_7/lstm_14/TensorArrayV2/element_shape
"sequential_7/lstm_14/TensorArrayV2TensorListReserve9sequential_7/lstm_14/TensorArrayV2/element_shape:output:0-sequential_7/lstm_14/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02$
"sequential_7/lstm_14/TensorArrayV2é
Jsequential_7/lstm_14/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ]   2L
Jsequential_7/lstm_14/TensorArrayUnstack/TensorListFromTensor/element_shapeÌ
<sequential_7/lstm_14/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor"sequential_7/lstm_14/transpose:y:0Ssequential_7/lstm_14/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02>
<sequential_7/lstm_14/TensorArrayUnstack/TensorListFromTensor¢
*sequential_7/lstm_14/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2,
*sequential_7/lstm_14/strided_slice_2/stack¦
,sequential_7/lstm_14/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_7/lstm_14/strided_slice_2/stack_1¦
,sequential_7/lstm_14/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_7/lstm_14/strided_slice_2/stack_2ú
$sequential_7/lstm_14/strided_slice_2StridedSlice"sequential_7/lstm_14/transpose:y:03sequential_7/lstm_14/strided_slice_2/stack:output:05sequential_7/lstm_14/strided_slice_2/stack_1:output:05sequential_7/lstm_14/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]*
shrink_axis_mask2&
$sequential_7/lstm_14/strided_slice_2ô
7sequential_7/lstm_14/lstm_cell_14/MatMul/ReadVariableOpReadVariableOp@sequential_7_lstm_14_lstm_cell_14_matmul_readvariableop_resource*
_output_shapes
:	] *
dtype029
7sequential_7/lstm_14/lstm_cell_14/MatMul/ReadVariableOp
(sequential_7/lstm_14/lstm_cell_14/MatMulMatMul-sequential_7/lstm_14/strided_slice_2:output:0?sequential_7/lstm_14/lstm_cell_14/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2*
(sequential_7/lstm_14/lstm_cell_14/MatMulû
9sequential_7/lstm_14/lstm_cell_14/MatMul_1/ReadVariableOpReadVariableOpBsequential_7_lstm_14_lstm_cell_14_matmul_1_readvariableop_resource* 
_output_shapes
:
 *
dtype02;
9sequential_7/lstm_14/lstm_cell_14/MatMul_1/ReadVariableOpý
*sequential_7/lstm_14/lstm_cell_14/MatMul_1MatMul#sequential_7/lstm_14/zeros:output:0Asequential_7/lstm_14/lstm_cell_14/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2,
*sequential_7/lstm_14/lstm_cell_14/MatMul_1ô
%sequential_7/lstm_14/lstm_cell_14/addAddV22sequential_7/lstm_14/lstm_cell_14/MatMul:product:04sequential_7/lstm_14/lstm_cell_14/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2'
%sequential_7/lstm_14/lstm_cell_14/addó
8sequential_7/lstm_14/lstm_cell_14/BiasAdd/ReadVariableOpReadVariableOpAsequential_7_lstm_14_lstm_cell_14_biasadd_readvariableop_resource*
_output_shapes	
: *
dtype02:
8sequential_7/lstm_14/lstm_cell_14/BiasAdd/ReadVariableOp
)sequential_7/lstm_14/lstm_cell_14/BiasAddBiasAdd)sequential_7/lstm_14/lstm_cell_14/add:z:0@sequential_7/lstm_14/lstm_cell_14/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2+
)sequential_7/lstm_14/lstm_cell_14/BiasAdd¨
1sequential_7/lstm_14/lstm_cell_14/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :23
1sequential_7/lstm_14/lstm_cell_14/split/split_dimË
'sequential_7/lstm_14/lstm_cell_14/splitSplit:sequential_7/lstm_14/lstm_cell_14/split/split_dim:output:02sequential_7/lstm_14/lstm_cell_14/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2)
'sequential_7/lstm_14/lstm_cell_14/splitÆ
)sequential_7/lstm_14/lstm_cell_14/SigmoidSigmoid0sequential_7/lstm_14/lstm_cell_14/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2+
)sequential_7/lstm_14/lstm_cell_14/SigmoidÊ
+sequential_7/lstm_14/lstm_cell_14/Sigmoid_1Sigmoid0sequential_7/lstm_14/lstm_cell_14/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2-
+sequential_7/lstm_14/lstm_cell_14/Sigmoid_1à
%sequential_7/lstm_14/lstm_cell_14/mulMul/sequential_7/lstm_14/lstm_cell_14/Sigmoid_1:y:0%sequential_7/lstm_14/zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%sequential_7/lstm_14/lstm_cell_14/mul½
&sequential_7/lstm_14/lstm_cell_14/ReluRelu0sequential_7/lstm_14/lstm_cell_14/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&sequential_7/lstm_14/lstm_cell_14/Reluñ
'sequential_7/lstm_14/lstm_cell_14/mul_1Mul-sequential_7/lstm_14/lstm_cell_14/Sigmoid:y:04sequential_7/lstm_14/lstm_cell_14/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'sequential_7/lstm_14/lstm_cell_14/mul_1æ
'sequential_7/lstm_14/lstm_cell_14/add_1AddV2)sequential_7/lstm_14/lstm_cell_14/mul:z:0+sequential_7/lstm_14/lstm_cell_14/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'sequential_7/lstm_14/lstm_cell_14/add_1Ê
+sequential_7/lstm_14/lstm_cell_14/Sigmoid_2Sigmoid0sequential_7/lstm_14/lstm_cell_14/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2-
+sequential_7/lstm_14/lstm_cell_14/Sigmoid_2¼
(sequential_7/lstm_14/lstm_cell_14/Relu_1Relu+sequential_7/lstm_14/lstm_cell_14/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
(sequential_7/lstm_14/lstm_cell_14/Relu_1õ
'sequential_7/lstm_14/lstm_cell_14/mul_2Mul/sequential_7/lstm_14/lstm_cell_14/Sigmoid_2:y:06sequential_7/lstm_14/lstm_cell_14/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'sequential_7/lstm_14/lstm_cell_14/mul_2¹
2sequential_7/lstm_14/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ  24
2sequential_7/lstm_14/TensorArrayV2_1/element_shape
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
sequential_7/lstm_14/time©
-sequential_7/lstm_14/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2/
-sequential_7/lstm_14/while/maximum_iterations
'sequential_7/lstm_14/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2)
'sequential_7/lstm_14/while/loop_counterÎ
sequential_7/lstm_14/whileWhile0sequential_7/lstm_14/while/loop_counter:output:06sequential_7/lstm_14/while/maximum_iterations:output:0"sequential_7/lstm_14/time:output:0-sequential_7/lstm_14/TensorArrayV2_1:handle:0#sequential_7/lstm_14/zeros:output:0%sequential_7/lstm_14/zeros_1:output:0-sequential_7/lstm_14/strided_slice_1:output:0Lsequential_7/lstm_14/TensorArrayUnstack/TensorListFromTensor:output_handle:0@sequential_7_lstm_14_lstm_cell_14_matmul_readvariableop_resourceBsequential_7_lstm_14_lstm_cell_14_matmul_1_readvariableop_resourceAsequential_7_lstm_14_lstm_cell_14_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *4
body,R*
(sequential_7_lstm_14_while_body_26074684*4
cond,R*
(sequential_7_lstm_14_while_cond_26074683*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 2
sequential_7/lstm_14/whileß
Esequential_7/lstm_14/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ  2G
Esequential_7/lstm_14/TensorArrayV2Stack/TensorListStack/element_shape½
7sequential_7/lstm_14/TensorArrayV2Stack/TensorListStackTensorListStack#sequential_7/lstm_14/while:output:3Nsequential_7/lstm_14/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype029
7sequential_7/lstm_14/TensorArrayV2Stack/TensorListStack«
*sequential_7/lstm_14/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2,
*sequential_7/lstm_14/strided_slice_3/stack¦
,sequential_7/lstm_14/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2.
,sequential_7/lstm_14/strided_slice_3/stack_1¦
,sequential_7/lstm_14/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_7/lstm_14/strided_slice_3/stack_2
$sequential_7/lstm_14/strided_slice_3StridedSlice@sequential_7/lstm_14/TensorArrayV2Stack/TensorListStack:tensor:03sequential_7/lstm_14/strided_slice_3/stack:output:05sequential_7/lstm_14/strided_slice_3/stack_1:output:05sequential_7/lstm_14/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2&
$sequential_7/lstm_14/strided_slice_3£
%sequential_7/lstm_14/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2'
%sequential_7/lstm_14/transpose_1/permú
 sequential_7/lstm_14/transpose_1	Transpose@sequential_7/lstm_14/TensorArrayV2Stack/TensorListStack:tensor:0.sequential_7/lstm_14/transpose_1/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 sequential_7/lstm_14/transpose_1
sequential_7/lstm_14/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
sequential_7/lstm_14/runtime­
 sequential_7/dropout_14/IdentityIdentity$sequential_7/lstm_14/transpose_1:y:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 sequential_7/dropout_14/Identity
sequential_7/lstm_15/ShapeShape)sequential_7/dropout_14/Identity:output:0*
T0*
_output_shapes
:2
sequential_7/lstm_15/Shape
(sequential_7/lstm_15/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(sequential_7/lstm_15/strided_slice/stack¢
*sequential_7/lstm_15/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*sequential_7/lstm_15/strided_slice/stack_1¢
*sequential_7/lstm_15/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*sequential_7/lstm_15/strided_slice/stack_2à
"sequential_7/lstm_15/strided_sliceStridedSlice#sequential_7/lstm_15/Shape:output:01sequential_7/lstm_15/strided_slice/stack:output:03sequential_7/lstm_15/strided_slice/stack_1:output:03sequential_7/lstm_15/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"sequential_7/lstm_15/strided_slice
 sequential_7/lstm_15/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :r2"
 sequential_7/lstm_15/zeros/mul/yÀ
sequential_7/lstm_15/zeros/mulMul+sequential_7/lstm_15/strided_slice:output:0)sequential_7/lstm_15/zeros/mul/y:output:0*
T0*
_output_shapes
: 2 
sequential_7/lstm_15/zeros/mul
!sequential_7/lstm_15/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2#
!sequential_7/lstm_15/zeros/Less/y»
sequential_7/lstm_15/zeros/LessLess"sequential_7/lstm_15/zeros/mul:z:0*sequential_7/lstm_15/zeros/Less/y:output:0*
T0*
_output_shapes
: 2!
sequential_7/lstm_15/zeros/Less
#sequential_7/lstm_15/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :r2%
#sequential_7/lstm_15/zeros/packed/1×
!sequential_7/lstm_15/zeros/packedPack+sequential_7/lstm_15/strided_slice:output:0,sequential_7/lstm_15/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2#
!sequential_7/lstm_15/zeros/packed
 sequential_7/lstm_15/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 sequential_7/lstm_15/zeros/ConstÉ
sequential_7/lstm_15/zerosFill*sequential_7/lstm_15/zeros/packed:output:0)sequential_7/lstm_15/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2
sequential_7/lstm_15/zeros
"sequential_7/lstm_15/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :r2$
"sequential_7/lstm_15/zeros_1/mul/yÆ
 sequential_7/lstm_15/zeros_1/mulMul+sequential_7/lstm_15/strided_slice:output:0+sequential_7/lstm_15/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2"
 sequential_7/lstm_15/zeros_1/mul
#sequential_7/lstm_15/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2%
#sequential_7/lstm_15/zeros_1/Less/yÃ
!sequential_7/lstm_15/zeros_1/LessLess$sequential_7/lstm_15/zeros_1/mul:z:0,sequential_7/lstm_15/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2#
!sequential_7/lstm_15/zeros_1/Less
%sequential_7/lstm_15/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :r2'
%sequential_7/lstm_15/zeros_1/packed/1Ý
#sequential_7/lstm_15/zeros_1/packedPack+sequential_7/lstm_15/strided_slice:output:0.sequential_7/lstm_15/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2%
#sequential_7/lstm_15/zeros_1/packed
"sequential_7/lstm_15/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"sequential_7/lstm_15/zeros_1/ConstÑ
sequential_7/lstm_15/zeros_1Fill,sequential_7/lstm_15/zeros_1/packed:output:0+sequential_7/lstm_15/zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2
sequential_7/lstm_15/zeros_1
#sequential_7/lstm_15/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2%
#sequential_7/lstm_15/transpose/permÝ
sequential_7/lstm_15/transpose	Transpose)sequential_7/dropout_14/Identity:output:0,sequential_7/lstm_15/transpose/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
sequential_7/lstm_15/transpose
sequential_7/lstm_15/Shape_1Shape"sequential_7/lstm_15/transpose:y:0*
T0*
_output_shapes
:2
sequential_7/lstm_15/Shape_1¢
*sequential_7/lstm_15/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2,
*sequential_7/lstm_15/strided_slice_1/stack¦
,sequential_7/lstm_15/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_7/lstm_15/strided_slice_1/stack_1¦
,sequential_7/lstm_15/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_7/lstm_15/strided_slice_1/stack_2ì
$sequential_7/lstm_15/strided_slice_1StridedSlice%sequential_7/lstm_15/Shape_1:output:03sequential_7/lstm_15/strided_slice_1/stack:output:05sequential_7/lstm_15/strided_slice_1/stack_1:output:05sequential_7/lstm_15/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2&
$sequential_7/lstm_15/strided_slice_1¯
0sequential_7/lstm_15/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ22
0sequential_7/lstm_15/TensorArrayV2/element_shape
"sequential_7/lstm_15/TensorArrayV2TensorListReserve9sequential_7/lstm_15/TensorArrayV2/element_shape:output:0-sequential_7/lstm_15/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02$
"sequential_7/lstm_15/TensorArrayV2é
Jsequential_7/lstm_15/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ  2L
Jsequential_7/lstm_15/TensorArrayUnstack/TensorListFromTensor/element_shapeÌ
<sequential_7/lstm_15/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor"sequential_7/lstm_15/transpose:y:0Ssequential_7/lstm_15/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02>
<sequential_7/lstm_15/TensorArrayUnstack/TensorListFromTensor¢
*sequential_7/lstm_15/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2,
*sequential_7/lstm_15/strided_slice_2/stack¦
,sequential_7/lstm_15/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_7/lstm_15/strided_slice_2/stack_1¦
,sequential_7/lstm_15/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_7/lstm_15/strided_slice_2/stack_2û
$sequential_7/lstm_15/strided_slice_2StridedSlice"sequential_7/lstm_15/transpose:y:03sequential_7/lstm_15/strided_slice_2/stack:output:05sequential_7/lstm_15/strided_slice_2/stack_1:output:05sequential_7/lstm_15/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2&
$sequential_7/lstm_15/strided_slice_2õ
7sequential_7/lstm_15/lstm_cell_15/MatMul/ReadVariableOpReadVariableOp@sequential_7_lstm_15_lstm_cell_15_matmul_readvariableop_resource* 
_output_shapes
:
È*
dtype029
7sequential_7/lstm_15/lstm_cell_15/MatMul/ReadVariableOp
(sequential_7/lstm_15/lstm_cell_15/MatMulMatMul-sequential_7/lstm_15/strided_slice_2:output:0?sequential_7/lstm_15/lstm_cell_15/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2*
(sequential_7/lstm_15/lstm_cell_15/MatMulú
9sequential_7/lstm_15/lstm_cell_15/MatMul_1/ReadVariableOpReadVariableOpBsequential_7_lstm_15_lstm_cell_15_matmul_1_readvariableop_resource*
_output_shapes
:	rÈ*
dtype02;
9sequential_7/lstm_15/lstm_cell_15/MatMul_1/ReadVariableOpý
*sequential_7/lstm_15/lstm_cell_15/MatMul_1MatMul#sequential_7/lstm_15/zeros:output:0Asequential_7/lstm_15/lstm_cell_15/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2,
*sequential_7/lstm_15/lstm_cell_15/MatMul_1ô
%sequential_7/lstm_15/lstm_cell_15/addAddV22sequential_7/lstm_15/lstm_cell_15/MatMul:product:04sequential_7/lstm_15/lstm_cell_15/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2'
%sequential_7/lstm_15/lstm_cell_15/addó
8sequential_7/lstm_15/lstm_cell_15/BiasAdd/ReadVariableOpReadVariableOpAsequential_7_lstm_15_lstm_cell_15_biasadd_readvariableop_resource*
_output_shapes	
:È*
dtype02:
8sequential_7/lstm_15/lstm_cell_15/BiasAdd/ReadVariableOp
)sequential_7/lstm_15/lstm_cell_15/BiasAddBiasAdd)sequential_7/lstm_15/lstm_cell_15/add:z:0@sequential_7/lstm_15/lstm_cell_15/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2+
)sequential_7/lstm_15/lstm_cell_15/BiasAdd¨
1sequential_7/lstm_15/lstm_cell_15/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :23
1sequential_7/lstm_15/lstm_cell_15/split/split_dimÇ
'sequential_7/lstm_15/lstm_cell_15/splitSplit:sequential_7/lstm_15/lstm_cell_15/split/split_dim:output:02sequential_7/lstm_15/lstm_cell_15/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿr:ÿÿÿÿÿÿÿÿÿr:ÿÿÿÿÿÿÿÿÿr:ÿÿÿÿÿÿÿÿÿr*
	num_split2)
'sequential_7/lstm_15/lstm_cell_15/splitÅ
)sequential_7/lstm_15/lstm_cell_15/SigmoidSigmoid0sequential_7/lstm_15/lstm_cell_15/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2+
)sequential_7/lstm_15/lstm_cell_15/SigmoidÉ
+sequential_7/lstm_15/lstm_cell_15/Sigmoid_1Sigmoid0sequential_7/lstm_15/lstm_cell_15/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2-
+sequential_7/lstm_15/lstm_cell_15/Sigmoid_1ß
%sequential_7/lstm_15/lstm_cell_15/mulMul/sequential_7/lstm_15/lstm_cell_15/Sigmoid_1:y:0%sequential_7/lstm_15/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2'
%sequential_7/lstm_15/lstm_cell_15/mul¼
&sequential_7/lstm_15/lstm_cell_15/ReluRelu0sequential_7/lstm_15/lstm_cell_15/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2(
&sequential_7/lstm_15/lstm_cell_15/Reluð
'sequential_7/lstm_15/lstm_cell_15/mul_1Mul-sequential_7/lstm_15/lstm_cell_15/Sigmoid:y:04sequential_7/lstm_15/lstm_cell_15/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2)
'sequential_7/lstm_15/lstm_cell_15/mul_1å
'sequential_7/lstm_15/lstm_cell_15/add_1AddV2)sequential_7/lstm_15/lstm_cell_15/mul:z:0+sequential_7/lstm_15/lstm_cell_15/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2)
'sequential_7/lstm_15/lstm_cell_15/add_1É
+sequential_7/lstm_15/lstm_cell_15/Sigmoid_2Sigmoid0sequential_7/lstm_15/lstm_cell_15/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2-
+sequential_7/lstm_15/lstm_cell_15/Sigmoid_2»
(sequential_7/lstm_15/lstm_cell_15/Relu_1Relu+sequential_7/lstm_15/lstm_cell_15/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2*
(sequential_7/lstm_15/lstm_cell_15/Relu_1ô
'sequential_7/lstm_15/lstm_cell_15/mul_2Mul/sequential_7/lstm_15/lstm_cell_15/Sigmoid_2:y:06sequential_7/lstm_15/lstm_cell_15/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2)
'sequential_7/lstm_15/lstm_cell_15/mul_2¹
2sequential_7/lstm_15/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿr   24
2sequential_7/lstm_15/TensorArrayV2_1/element_shape
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
sequential_7/lstm_15/time©
-sequential_7/lstm_15/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2/
-sequential_7/lstm_15/while/maximum_iterations
'sequential_7/lstm_15/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2)
'sequential_7/lstm_15/while/loop_counterÊ
sequential_7/lstm_15/whileWhile0sequential_7/lstm_15/while/loop_counter:output:06sequential_7/lstm_15/while/maximum_iterations:output:0"sequential_7/lstm_15/time:output:0-sequential_7/lstm_15/TensorArrayV2_1:handle:0#sequential_7/lstm_15/zeros:output:0%sequential_7/lstm_15/zeros_1:output:0-sequential_7/lstm_15/strided_slice_1:output:0Lsequential_7/lstm_15/TensorArrayUnstack/TensorListFromTensor:output_handle:0@sequential_7_lstm_15_lstm_cell_15_matmul_readvariableop_resourceBsequential_7_lstm_15_lstm_cell_15_matmul_1_readvariableop_resourceAsequential_7_lstm_15_lstm_cell_15_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿr:ÿÿÿÿÿÿÿÿÿr: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *4
body,R*
(sequential_7_lstm_15_while_body_26074832*4
cond,R*
(sequential_7_lstm_15_while_cond_26074831*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿr:ÿÿÿÿÿÿÿÿÿr: : : : : *
parallel_iterations 2
sequential_7/lstm_15/whileß
Esequential_7/lstm_15/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿr   2G
Esequential_7/lstm_15/TensorArrayV2Stack/TensorListStack/element_shape¼
7sequential_7/lstm_15/TensorArrayV2Stack/TensorListStackTensorListStack#sequential_7/lstm_15/while:output:3Nsequential_7/lstm_15/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿr*
element_dtype029
7sequential_7/lstm_15/TensorArrayV2Stack/TensorListStack«
*sequential_7/lstm_15/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2,
*sequential_7/lstm_15/strided_slice_3/stack¦
,sequential_7/lstm_15/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2.
,sequential_7/lstm_15/strided_slice_3/stack_1¦
,sequential_7/lstm_15/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_7/lstm_15/strided_slice_3/stack_2
$sequential_7/lstm_15/strided_slice_3StridedSlice@sequential_7/lstm_15/TensorArrayV2Stack/TensorListStack:tensor:03sequential_7/lstm_15/strided_slice_3/stack:output:05sequential_7/lstm_15/strided_slice_3/stack_1:output:05sequential_7/lstm_15/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr*
shrink_axis_mask2&
$sequential_7/lstm_15/strided_slice_3£
%sequential_7/lstm_15/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2'
%sequential_7/lstm_15/transpose_1/permù
 sequential_7/lstm_15/transpose_1	Transpose@sequential_7/lstm_15/TensorArrayV2Stack/TensorListStack:tensor:0.sequential_7/lstm_15/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2"
 sequential_7/lstm_15/transpose_1
sequential_7/lstm_15/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
sequential_7/lstm_15/runtime¬
 sequential_7/dropout_15/IdentityIdentity$sequential_7/lstm_15/transpose_1:y:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2"
 sequential_7/dropout_15/IdentityÕ
-sequential_7/dense_7/Tensordot/ReadVariableOpReadVariableOp6sequential_7_dense_7_tensordot_readvariableop_resource*
_output_shapes

:r*
dtype02/
-sequential_7/dense_7/Tensordot/ReadVariableOp
#sequential_7/dense_7/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2%
#sequential_7/dense_7/Tensordot/axes
#sequential_7/dense_7/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2%
#sequential_7/dense_7/Tensordot/free¥
$sequential_7/dense_7/Tensordot/ShapeShape)sequential_7/dropout_15/Identity:output:0*
T0*
_output_shapes
:2&
$sequential_7/dense_7/Tensordot/Shape
,sequential_7/dense_7/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,sequential_7/dense_7/Tensordot/GatherV2/axisº
'sequential_7/dense_7/Tensordot/GatherV2GatherV2-sequential_7/dense_7/Tensordot/Shape:output:0,sequential_7/dense_7/Tensordot/free:output:05sequential_7/dense_7/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2)
'sequential_7/dense_7/Tensordot/GatherV2¢
.sequential_7/dense_7/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.sequential_7/dense_7/Tensordot/GatherV2_1/axisÀ
)sequential_7/dense_7/Tensordot/GatherV2_1GatherV2-sequential_7/dense_7/Tensordot/Shape:output:0,sequential_7/dense_7/Tensordot/axes:output:07sequential_7/dense_7/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2+
)sequential_7/dense_7/Tensordot/GatherV2_1
$sequential_7/dense_7/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2&
$sequential_7/dense_7/Tensordot/ConstÔ
#sequential_7/dense_7/Tensordot/ProdProd0sequential_7/dense_7/Tensordot/GatherV2:output:0-sequential_7/dense_7/Tensordot/Const:output:0*
T0*
_output_shapes
: 2%
#sequential_7/dense_7/Tensordot/Prod
&sequential_7/dense_7/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2(
&sequential_7/dense_7/Tensordot/Const_1Ü
%sequential_7/dense_7/Tensordot/Prod_1Prod2sequential_7/dense_7/Tensordot/GatherV2_1:output:0/sequential_7/dense_7/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2'
%sequential_7/dense_7/Tensordot/Prod_1
*sequential_7/dense_7/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2,
*sequential_7/dense_7/Tensordot/concat/axis
%sequential_7/dense_7/Tensordot/concatConcatV2,sequential_7/dense_7/Tensordot/free:output:0,sequential_7/dense_7/Tensordot/axes:output:03sequential_7/dense_7/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2'
%sequential_7/dense_7/Tensordot/concatà
$sequential_7/dense_7/Tensordot/stackPack,sequential_7/dense_7/Tensordot/Prod:output:0.sequential_7/dense_7/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2&
$sequential_7/dense_7/Tensordot/stackò
(sequential_7/dense_7/Tensordot/transpose	Transpose)sequential_7/dropout_15/Identity:output:0.sequential_7/dense_7/Tensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2*
(sequential_7/dense_7/Tensordot/transposeó
&sequential_7/dense_7/Tensordot/ReshapeReshape,sequential_7/dense_7/Tensordot/transpose:y:0-sequential_7/dense_7/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2(
&sequential_7/dense_7/Tensordot/Reshapeò
%sequential_7/dense_7/Tensordot/MatMulMatMul/sequential_7/dense_7/Tensordot/Reshape:output:05sequential_7/dense_7/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%sequential_7/dense_7/Tensordot/MatMul
&sequential_7/dense_7/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&sequential_7/dense_7/Tensordot/Const_2
,sequential_7/dense_7/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,sequential_7/dense_7/Tensordot/concat_1/axis¦
'sequential_7/dense_7/Tensordot/concat_1ConcatV20sequential_7/dense_7/Tensordot/GatherV2:output:0/sequential_7/dense_7/Tensordot/Const_2:output:05sequential_7/dense_7/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2)
'sequential_7/dense_7/Tensordot/concat_1ä
sequential_7/dense_7/TensordotReshape/sequential_7/dense_7/Tensordot/MatMul:product:00sequential_7/dense_7/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
sequential_7/dense_7/TensordotË
+sequential_7/dense_7/BiasAdd/ReadVariableOpReadVariableOp4sequential_7_dense_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02-
+sequential_7/dense_7/BiasAdd/ReadVariableOpÛ
sequential_7/dense_7/BiasAddBiasAdd'sequential_7/dense_7/Tensordot:output:03sequential_7/dense_7/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential_7/dense_7/BiasAdd¤
sequential_7/dense_7/SoftmaxSoftmax%sequential_7/dense_7/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential_7/dense_7/Softmax
IdentityIdentity&sequential_7/dense_7/Softmax:softmax:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

IdentityÈ
NoOpNoOp,^sequential_7/dense_7/BiasAdd/ReadVariableOp.^sequential_7/dense_7/Tensordot/ReadVariableOp9^sequential_7/lstm_14/lstm_cell_14/BiasAdd/ReadVariableOp8^sequential_7/lstm_14/lstm_cell_14/MatMul/ReadVariableOp:^sequential_7/lstm_14/lstm_cell_14/MatMul_1/ReadVariableOp^sequential_7/lstm_14/while9^sequential_7/lstm_15/lstm_cell_15/BiasAdd/ReadVariableOp8^sequential_7/lstm_15/lstm_cell_15/MatMul/ReadVariableOp:^sequential_7/lstm_15/lstm_cell_15/MatMul_1/ReadVariableOp^sequential_7/lstm_15/while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ]: : : : : : : : 2Z
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
:ÿÿÿÿÿÿÿÿÿ]
'
_user_specified_namelstm_14_input


J__inference_lstm_cell_15_layer_call_and_return_conditional_losses_26079423

inputs
states_0
states_12
matmul_readvariableop_resource:
È3
 matmul_1_readvariableop_resource:	rÈ.
biasadd_readvariableop_resource:	È
identity

identity_1

identity_2¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
È*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
MatMul
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	rÈ*
dtype02
MatMul_1/ReadVariableOp|
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
add
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:È*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2	
BiasAddd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim¿
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿr:ÿÿÿÿÿÿÿÿÿr:ÿÿÿÿÿÿÿÿÿr:ÿÿÿÿÿÿÿÿÿr*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2
	Sigmoid_1\
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2
mulV
ReluRelusplit:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2
	Sigmoid_2U
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2
mul_2d
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2

Identityh

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2

Identity_1h

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2

Identity_2
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
@:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿr:ÿÿÿÿÿÿÿÿÿr: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
"
_user_specified_name
states/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
"
_user_specified_name
states/1
¶
¸
*__inference_lstm_14_layer_call_fn_26078519

inputs
unknown:	] 
	unknown_0:
 
	unknown_1:	 
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_lstm_14_layer_call_and_return_conditional_losses_260769912
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

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
?
Ô
while_body_26079066
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
3while_lstm_cell_15_matmul_readvariableop_resource_0:
ÈH
5while_lstm_cell_15_matmul_1_readvariableop_resource_0:	rÈC
4while_lstm_cell_15_biasadd_readvariableop_resource_0:	È
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
1while_lstm_cell_15_matmul_readvariableop_resource:
ÈF
3while_lstm_cell_15_matmul_1_readvariableop_resource:	rÈA
2while_lstm_cell_15_biasadd_readvariableop_resource:	È¢)while/lstm_cell_15/BiasAdd/ReadVariableOp¢(while/lstm_cell_15/MatMul/ReadVariableOp¢*while/lstm_cell_15/MatMul_1/ReadVariableOpÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ  29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÔ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemÊ
(while/lstm_cell_15/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_15_matmul_readvariableop_resource_0* 
_output_shapes
:
È*
dtype02*
(while/lstm_cell_15/MatMul/ReadVariableOp×
while/lstm_cell_15/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_15/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell_15/MatMulÏ
*while/lstm_cell_15/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_15_matmul_1_readvariableop_resource_0*
_output_shapes
:	rÈ*
dtype02,
*while/lstm_cell_15/MatMul_1/ReadVariableOpÀ
while/lstm_cell_15/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_15/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell_15/MatMul_1¸
while/lstm_cell_15/addAddV2#while/lstm_cell_15/MatMul:product:0%while/lstm_cell_15/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell_15/addÈ
)while/lstm_cell_15/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_15_biasadd_readvariableop_resource_0*
_output_shapes	
:È*
dtype02+
)while/lstm_cell_15/BiasAdd/ReadVariableOpÅ
while/lstm_cell_15/BiasAddBiasAddwhile/lstm_cell_15/add:z:01while/lstm_cell_15/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell_15/BiasAdd
"while/lstm_cell_15/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_15/split/split_dim
while/lstm_cell_15/splitSplit+while/lstm_cell_15/split/split_dim:output:0#while/lstm_cell_15/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿr:ÿÿÿÿÿÿÿÿÿr:ÿÿÿÿÿÿÿÿÿr:ÿÿÿÿÿÿÿÿÿr*
	num_split2
while/lstm_cell_15/split
while/lstm_cell_15/SigmoidSigmoid!while/lstm_cell_15/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2
while/lstm_cell_15/Sigmoid
while/lstm_cell_15/Sigmoid_1Sigmoid!while/lstm_cell_15/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2
while/lstm_cell_15/Sigmoid_1 
while/lstm_cell_15/mulMul while/lstm_cell_15/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2
while/lstm_cell_15/mul
while/lstm_cell_15/ReluRelu!while/lstm_cell_15/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2
while/lstm_cell_15/Relu´
while/lstm_cell_15/mul_1Mulwhile/lstm_cell_15/Sigmoid:y:0%while/lstm_cell_15/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2
while/lstm_cell_15/mul_1©
while/lstm_cell_15/add_1AddV2while/lstm_cell_15/mul:z:0while/lstm_cell_15/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2
while/lstm_cell_15/add_1
while/lstm_cell_15/Sigmoid_2Sigmoid!while/lstm_cell_15/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2
while/lstm_cell_15/Sigmoid_2
while/lstm_cell_15/Relu_1Reluwhile/lstm_cell_15/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2
while/lstm_cell_15/Relu_1¸
while/lstm_cell_15/mul_2Mul while/lstm_cell_15/Sigmoid_2:y:0'while/lstm_cell_15/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2
while/lstm_cell_15/mul_2à
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
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_15/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_15/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2
while/Identity_5Þ

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
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿr:ÿÿÿÿÿÿÿÿÿr: : : : : 2V
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
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr:

_output_shapes
: :

_output_shapes
: 
í[

E__inference_lstm_15_layer_call_and_return_conditional_losses_26079150

inputs?
+lstm_cell_15_matmul_readvariableop_resource:
È@
-lstm_cell_15_matmul_1_readvariableop_resource:	rÈ;
,lstm_cell_15_biasadd_readvariableop_resource:	È
identity¢#lstm_cell_15/BiasAdd/ReadVariableOp¢"lstm_cell_15/MatMul/ReadVariableOp¢$lstm_cell_15/MatMul_1/ReadVariableOp¢whileD
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
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :r2
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

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :r2
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
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :r2
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
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :r2
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
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2	
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
:ÿÿÿÿÿÿÿÿÿ2
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
valueB"ÿÿÿÿ  27
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
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
strided_slice_2¶
"lstm_cell_15/MatMul/ReadVariableOpReadVariableOp+lstm_cell_15_matmul_readvariableop_resource* 
_output_shapes
:
È*
dtype02$
"lstm_cell_15/MatMul/ReadVariableOp­
lstm_cell_15/MatMulMatMulstrided_slice_2:output:0*lstm_cell_15/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell_15/MatMul»
$lstm_cell_15/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_15_matmul_1_readvariableop_resource*
_output_shapes
:	rÈ*
dtype02&
$lstm_cell_15/MatMul_1/ReadVariableOp©
lstm_cell_15/MatMul_1MatMulzeros:output:0,lstm_cell_15/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell_15/MatMul_1 
lstm_cell_15/addAddV2lstm_cell_15/MatMul:product:0lstm_cell_15/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell_15/add´
#lstm_cell_15/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_15_biasadd_readvariableop_resource*
_output_shapes	
:È*
dtype02%
#lstm_cell_15/BiasAdd/ReadVariableOp­
lstm_cell_15/BiasAddBiasAddlstm_cell_15/add:z:0+lstm_cell_15/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell_15/BiasAdd~
lstm_cell_15/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_15/split/split_dimó
lstm_cell_15/splitSplit%lstm_cell_15/split/split_dim:output:0lstm_cell_15/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿr:ÿÿÿÿÿÿÿÿÿr:ÿÿÿÿÿÿÿÿÿr:ÿÿÿÿÿÿÿÿÿr*
	num_split2
lstm_cell_15/split
lstm_cell_15/SigmoidSigmoidlstm_cell_15/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2
lstm_cell_15/Sigmoid
lstm_cell_15/Sigmoid_1Sigmoidlstm_cell_15/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2
lstm_cell_15/Sigmoid_1
lstm_cell_15/mulMullstm_cell_15/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2
lstm_cell_15/mul}
lstm_cell_15/ReluRelulstm_cell_15/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2
lstm_cell_15/Relu
lstm_cell_15/mul_1Mullstm_cell_15/Sigmoid:y:0lstm_cell_15/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2
lstm_cell_15/mul_1
lstm_cell_15/add_1AddV2lstm_cell_15/mul:z:0lstm_cell_15/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2
lstm_cell_15/add_1
lstm_cell_15/Sigmoid_2Sigmoidlstm_cell_15/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2
lstm_cell_15/Sigmoid_2|
lstm_cell_15/Relu_1Relulstm_cell_15/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2
lstm_cell_15/Relu_1 
lstm_cell_15/mul_2Mullstm_cell_15/Sigmoid_2:y:0!lstm_cell_15/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2
lstm_cell_15/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿr   2
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
while/loop_counter
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_15_matmul_readvariableop_resource-lstm_cell_15_matmul_1_readvariableop_resource,lstm_cell_15_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿr:ÿÿÿÿÿÿÿÿÿr: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_26079066*
condR
while_cond_26079065*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿr:ÿÿÿÿÿÿÿÿÿr: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿr   22
0TensorArrayV2Stack/TensorListStack/element_shapeè
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿr*
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
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm¥
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2
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
:ÿÿÿÿÿÿÿÿÿr2

IdentityÈ
NoOpNoOp$^lstm_cell_15/BiasAdd/ReadVariableOp#^lstm_cell_15/MatMul/ReadVariableOp%^lstm_cell_15/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:ÿÿÿÿÿÿÿÿÿ: : : 2J
#lstm_cell_15/BiasAdd/ReadVariableOp#lstm_cell_15/BiasAdd/ReadVariableOp2H
"lstm_cell_15/MatMul/ReadVariableOp"lstm_cell_15/MatMul/ReadVariableOp2L
$lstm_cell_15/MatMul_1/ReadVariableOp$lstm_cell_15/MatMul_1/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ª

Ñ
/__inference_sequential_7_layer_call_fn_26077088
lstm_14_input
unknown:	] 
	unknown_0:
 
	unknown_1:	 
	unknown_2:
È
	unknown_3:	rÈ
	unknown_4:	È
	unknown_5:r
	unknown_6:
identity¢StatefulPartitionedCallÓ
StatefulPartitionedCallStatefulPartitionedCalllstm_14_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
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
GPU 2J 8 *S
fNRL
J__inference_sequential_7_layer_call_and_return_conditional_losses_260770482
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
_user_specified_namelstm_14_input
à
º
*__inference_lstm_15_layer_call_fn_26079172
inputs_0
unknown:
È
	unknown_0:	rÈ
	unknown_1:	È
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿr*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_lstm_15_layer_call_and_return_conditional_losses_260759422
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿr2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0


J__inference_lstm_cell_14_layer_call_and_return_conditional_losses_26079293

inputs
states_0
states_11
matmul_readvariableop_resource:	] 4
 matmul_1_readvariableop_resource:
 .
biasadd_readvariableop_resource:	 
identity

identity_1

identity_2¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	] *
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
MatMul
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
 *
dtype02
MatMul_1/ReadVariableOp|
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
add
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
: *
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2	
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
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2
split`
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
Sigmoidd
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	Sigmoid_1]
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mulW
ReluRelusplit:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relui
mul_1MulSigmoid:y:0Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mul_1^
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
add_1d
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	Sigmoid_2V
Relu_1Relu	add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu_1m
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mul_2e
IdentityIdentity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityi

Identity_1Identity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1i

Identity_2Identity	add_1:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

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
A:ÿÿÿÿÿÿÿÿÿ]:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
 
_user_specified_nameinputs:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
states/0:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
states/1
ð

J__inference_sequential_7_layer_call_and_return_conditional_losses_26077829

inputsF
3lstm_14_lstm_cell_14_matmul_readvariableop_resource:	] I
5lstm_14_lstm_cell_14_matmul_1_readvariableop_resource:
 C
4lstm_14_lstm_cell_14_biasadd_readvariableop_resource:	 G
3lstm_15_lstm_cell_15_matmul_readvariableop_resource:
ÈH
5lstm_15_lstm_cell_15_matmul_1_readvariableop_resource:	rÈC
4lstm_15_lstm_cell_15_biasadd_readvariableop_resource:	È;
)dense_7_tensordot_readvariableop_resource:r5
'dense_7_biasadd_readvariableop_resource:
identity¢dense_7/BiasAdd/ReadVariableOp¢ dense_7/Tensordot/ReadVariableOp¢+lstm_14/lstm_cell_14/BiasAdd/ReadVariableOp¢*lstm_14/lstm_cell_14/MatMul/ReadVariableOp¢,lstm_14/lstm_cell_14/MatMul_1/ReadVariableOp¢lstm_14/while¢+lstm_15/lstm_cell_15/BiasAdd/ReadVariableOp¢*lstm_15/lstm_cell_15/MatMul/ReadVariableOp¢,lstm_15/lstm_cell_15/MatMul_1/ReadVariableOp¢lstm_15/whileT
lstm_14/ShapeShapeinputs*
T0*
_output_shapes
:2
lstm_14/Shape
lstm_14/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_14/strided_slice/stack
lstm_14/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_14/strided_slice/stack_1
lstm_14/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_14/strided_slice/stack_2
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
B :2
lstm_14/zeros/mul/y
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
B :è2
lstm_14/zeros/Less/y
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
B :2
lstm_14/zeros/packed/1£
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
lstm_14/zeros/Const
lstm_14/zerosFilllstm_14/zeros/packed:output:0lstm_14/zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_14/zerosq
lstm_14/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :2
lstm_14/zeros_1/mul/y
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
B :è2
lstm_14/zeros_1/Less/y
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
B :2
lstm_14/zeros_1/packed/1©
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
lstm_14/zeros_1/Const
lstm_14/zeros_1Filllstm_14/zeros_1/packed:output:0lstm_14/zeros_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_14/zeros_1
lstm_14/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_14/transpose/perm
lstm_14/transpose	Transposeinputslstm_14/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]2
lstm_14/transposeg
lstm_14/Shape_1Shapelstm_14/transpose:y:0*
T0*
_output_shapes
:2
lstm_14/Shape_1
lstm_14/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_14/strided_slice_1/stack
lstm_14/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_14/strided_slice_1/stack_1
lstm_14/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_14/strided_slice_1/stack_2
lstm_14/strided_slice_1StridedSlicelstm_14/Shape_1:output:0&lstm_14/strided_slice_1/stack:output:0(lstm_14/strided_slice_1/stack_1:output:0(lstm_14/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_14/strided_slice_1
#lstm_14/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2%
#lstm_14/TensorArrayV2/element_shapeÒ
lstm_14/TensorArrayV2TensorListReserve,lstm_14/TensorArrayV2/element_shape:output:0 lstm_14/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_14/TensorArrayV2Ï
=lstm_14/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ]   2?
=lstm_14/TensorArrayUnstack/TensorListFromTensor/element_shape
/lstm_14/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_14/transpose:y:0Flstm_14/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type021
/lstm_14/TensorArrayUnstack/TensorListFromTensor
lstm_14/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_14/strided_slice_2/stack
lstm_14/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_14/strided_slice_2/stack_1
lstm_14/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_14/strided_slice_2/stack_2¬
lstm_14/strided_slice_2StridedSlicelstm_14/transpose:y:0&lstm_14/strided_slice_2/stack:output:0(lstm_14/strided_slice_2/stack_1:output:0(lstm_14/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]*
shrink_axis_mask2
lstm_14/strided_slice_2Í
*lstm_14/lstm_cell_14/MatMul/ReadVariableOpReadVariableOp3lstm_14_lstm_cell_14_matmul_readvariableop_resource*
_output_shapes
:	] *
dtype02,
*lstm_14/lstm_cell_14/MatMul/ReadVariableOpÍ
lstm_14/lstm_cell_14/MatMulMatMul lstm_14/strided_slice_2:output:02lstm_14/lstm_cell_14/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
lstm_14/lstm_cell_14/MatMulÔ
,lstm_14/lstm_cell_14/MatMul_1/ReadVariableOpReadVariableOp5lstm_14_lstm_cell_14_matmul_1_readvariableop_resource* 
_output_shapes
:
 *
dtype02.
,lstm_14/lstm_cell_14/MatMul_1/ReadVariableOpÉ
lstm_14/lstm_cell_14/MatMul_1MatMullstm_14/zeros:output:04lstm_14/lstm_cell_14/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
lstm_14/lstm_cell_14/MatMul_1À
lstm_14/lstm_cell_14/addAddV2%lstm_14/lstm_cell_14/MatMul:product:0'lstm_14/lstm_cell_14/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
lstm_14/lstm_cell_14/addÌ
+lstm_14/lstm_cell_14/BiasAdd/ReadVariableOpReadVariableOp4lstm_14_lstm_cell_14_biasadd_readvariableop_resource*
_output_shapes	
: *
dtype02-
+lstm_14/lstm_cell_14/BiasAdd/ReadVariableOpÍ
lstm_14/lstm_cell_14/BiasAddBiasAddlstm_14/lstm_cell_14/add:z:03lstm_14/lstm_cell_14/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
lstm_14/lstm_cell_14/BiasAdd
$lstm_14/lstm_cell_14/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2&
$lstm_14/lstm_cell_14/split/split_dim
lstm_14/lstm_cell_14/splitSplit-lstm_14/lstm_cell_14/split/split_dim:output:0%lstm_14/lstm_cell_14/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2
lstm_14/lstm_cell_14/split
lstm_14/lstm_cell_14/SigmoidSigmoid#lstm_14/lstm_cell_14/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_14/lstm_cell_14/Sigmoid£
lstm_14/lstm_cell_14/Sigmoid_1Sigmoid#lstm_14/lstm_cell_14/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
lstm_14/lstm_cell_14/Sigmoid_1¬
lstm_14/lstm_cell_14/mulMul"lstm_14/lstm_cell_14/Sigmoid_1:y:0lstm_14/zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_14/lstm_cell_14/mul
lstm_14/lstm_cell_14/ReluRelu#lstm_14/lstm_cell_14/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_14/lstm_cell_14/Relu½
lstm_14/lstm_cell_14/mul_1Mul lstm_14/lstm_cell_14/Sigmoid:y:0'lstm_14/lstm_cell_14/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_14/lstm_cell_14/mul_1²
lstm_14/lstm_cell_14/add_1AddV2lstm_14/lstm_cell_14/mul:z:0lstm_14/lstm_cell_14/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_14/lstm_cell_14/add_1£
lstm_14/lstm_cell_14/Sigmoid_2Sigmoid#lstm_14/lstm_cell_14/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
lstm_14/lstm_cell_14/Sigmoid_2
lstm_14/lstm_cell_14/Relu_1Relulstm_14/lstm_cell_14/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_14/lstm_cell_14/Relu_1Á
lstm_14/lstm_cell_14/mul_2Mul"lstm_14/lstm_cell_14/Sigmoid_2:y:0)lstm_14/lstm_cell_14/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_14/lstm_cell_14/mul_2
%lstm_14/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ  2'
%lstm_14/TensorArrayV2_1/element_shapeØ
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
lstm_14/time
 lstm_14/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2"
 lstm_14/while/maximum_iterationsz
lstm_14/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_14/while/loop_counter
lstm_14/whileWhile#lstm_14/while/loop_counter:output:0)lstm_14/while/maximum_iterations:output:0lstm_14/time:output:0 lstm_14/TensorArrayV2_1:handle:0lstm_14/zeros:output:0lstm_14/zeros_1:output:0 lstm_14/strided_slice_1:output:0?lstm_14/TensorArrayUnstack/TensorListFromTensor:output_handle:03lstm_14_lstm_cell_14_matmul_readvariableop_resource5lstm_14_lstm_cell_14_matmul_1_readvariableop_resource4lstm_14_lstm_cell_14_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *'
bodyR
lstm_14_while_body_26077555*'
condR
lstm_14_while_cond_26077554*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 2
lstm_14/whileÅ
8lstm_14/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ  2:
8lstm_14/TensorArrayV2Stack/TensorListStack/element_shape
*lstm_14/TensorArrayV2Stack/TensorListStackTensorListStacklstm_14/while:output:3Alstm_14/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype02,
*lstm_14/TensorArrayV2Stack/TensorListStack
lstm_14/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
lstm_14/strided_slice_3/stack
lstm_14/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2!
lstm_14/strided_slice_3/stack_1
lstm_14/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_14/strided_slice_3/stack_2Ë
lstm_14/strided_slice_3StridedSlice3lstm_14/TensorArrayV2Stack/TensorListStack:tensor:0&lstm_14/strided_slice_3/stack:output:0(lstm_14/strided_slice_3/stack_1:output:0(lstm_14/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
lstm_14/strided_slice_3
lstm_14/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_14/transpose_1/permÆ
lstm_14/transpose_1	Transpose3lstm_14/TensorArrayV2Stack/TensorListStack:tensor:0!lstm_14/transpose_1/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
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
 *«ªª?2
dropout_14/dropout/Constª
dropout_14/dropout/MulMullstm_14/transpose_1:y:0!dropout_14/dropout/Const:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout_14/dropout/Mul{
dropout_14/dropout/ShapeShapelstm_14/transpose_1:y:0*
T0*
_output_shapes
:2
dropout_14/dropout/ShapeÚ
/dropout_14/dropout/random_uniform/RandomUniformRandomUniform!dropout_14/dropout/Shape:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype021
/dropout_14/dropout/random_uniform/RandomUniform
!dropout_14/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  >2#
!dropout_14/dropout/GreaterEqual/yï
dropout_14/dropout/GreaterEqualGreaterEqual8dropout_14/dropout/random_uniform/RandomUniform:output:0*dropout_14/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
dropout_14/dropout/GreaterEqual¥
dropout_14/dropout/CastCast#dropout_14/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout_14/dropout/Cast«
dropout_14/dropout/Mul_1Muldropout_14/dropout/Mul:z:0dropout_14/dropout/Cast:y:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout_14/dropout/Mul_1j
lstm_15/ShapeShapedropout_14/dropout/Mul_1:z:0*
T0*
_output_shapes
:2
lstm_15/Shape
lstm_15/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_15/strided_slice/stack
lstm_15/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_15/strided_slice/stack_1
lstm_15/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_15/strided_slice/stack_2
lstm_15/strided_sliceStridedSlicelstm_15/Shape:output:0$lstm_15/strided_slice/stack:output:0&lstm_15/strided_slice/stack_1:output:0&lstm_15/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_15/strided_slicel
lstm_15/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :r2
lstm_15/zeros/mul/y
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
B :è2
lstm_15/zeros/Less/y
lstm_15/zeros/LessLesslstm_15/zeros/mul:z:0lstm_15/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_15/zeros/Lessr
lstm_15/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :r2
lstm_15/zeros/packed/1£
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
lstm_15/zeros/Const
lstm_15/zerosFilllstm_15/zeros/packed:output:0lstm_15/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2
lstm_15/zerosp
lstm_15/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :r2
lstm_15/zeros_1/mul/y
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
B :è2
lstm_15/zeros_1/Less/y
lstm_15/zeros_1/LessLesslstm_15/zeros_1/mul:z:0lstm_15/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_15/zeros_1/Lessv
lstm_15/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :r2
lstm_15/zeros_1/packed/1©
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
lstm_15/zeros_1/Const
lstm_15/zeros_1Filllstm_15/zeros_1/packed:output:0lstm_15/zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2
lstm_15/zeros_1
lstm_15/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_15/transpose/perm©
lstm_15/transpose	Transposedropout_14/dropout/Mul_1:z:0lstm_15/transpose/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_15/transposeg
lstm_15/Shape_1Shapelstm_15/transpose:y:0*
T0*
_output_shapes
:2
lstm_15/Shape_1
lstm_15/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_15/strided_slice_1/stack
lstm_15/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_15/strided_slice_1/stack_1
lstm_15/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_15/strided_slice_1/stack_2
lstm_15/strided_slice_1StridedSlicelstm_15/Shape_1:output:0&lstm_15/strided_slice_1/stack:output:0(lstm_15/strided_slice_1/stack_1:output:0(lstm_15/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_15/strided_slice_1
#lstm_15/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2%
#lstm_15/TensorArrayV2/element_shapeÒ
lstm_15/TensorArrayV2TensorListReserve,lstm_15/TensorArrayV2/element_shape:output:0 lstm_15/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_15/TensorArrayV2Ï
=lstm_15/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ  2?
=lstm_15/TensorArrayUnstack/TensorListFromTensor/element_shape
/lstm_15/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_15/transpose:y:0Flstm_15/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type021
/lstm_15/TensorArrayUnstack/TensorListFromTensor
lstm_15/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_15/strided_slice_2/stack
lstm_15/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_15/strided_slice_2/stack_1
lstm_15/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_15/strided_slice_2/stack_2­
lstm_15/strided_slice_2StridedSlicelstm_15/transpose:y:0&lstm_15/strided_slice_2/stack:output:0(lstm_15/strided_slice_2/stack_1:output:0(lstm_15/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
lstm_15/strided_slice_2Î
*lstm_15/lstm_cell_15/MatMul/ReadVariableOpReadVariableOp3lstm_15_lstm_cell_15_matmul_readvariableop_resource* 
_output_shapes
:
È*
dtype02,
*lstm_15/lstm_cell_15/MatMul/ReadVariableOpÍ
lstm_15/lstm_cell_15/MatMulMatMul lstm_15/strided_slice_2:output:02lstm_15/lstm_cell_15/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_15/lstm_cell_15/MatMulÓ
,lstm_15/lstm_cell_15/MatMul_1/ReadVariableOpReadVariableOp5lstm_15_lstm_cell_15_matmul_1_readvariableop_resource*
_output_shapes
:	rÈ*
dtype02.
,lstm_15/lstm_cell_15/MatMul_1/ReadVariableOpÉ
lstm_15/lstm_cell_15/MatMul_1MatMullstm_15/zeros:output:04lstm_15/lstm_cell_15/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_15/lstm_cell_15/MatMul_1À
lstm_15/lstm_cell_15/addAddV2%lstm_15/lstm_cell_15/MatMul:product:0'lstm_15/lstm_cell_15/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_15/lstm_cell_15/addÌ
+lstm_15/lstm_cell_15/BiasAdd/ReadVariableOpReadVariableOp4lstm_15_lstm_cell_15_biasadd_readvariableop_resource*
_output_shapes	
:È*
dtype02-
+lstm_15/lstm_cell_15/BiasAdd/ReadVariableOpÍ
lstm_15/lstm_cell_15/BiasAddBiasAddlstm_15/lstm_cell_15/add:z:03lstm_15/lstm_cell_15/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_15/lstm_cell_15/BiasAdd
$lstm_15/lstm_cell_15/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2&
$lstm_15/lstm_cell_15/split/split_dim
lstm_15/lstm_cell_15/splitSplit-lstm_15/lstm_cell_15/split/split_dim:output:0%lstm_15/lstm_cell_15/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿr:ÿÿÿÿÿÿÿÿÿr:ÿÿÿÿÿÿÿÿÿr:ÿÿÿÿÿÿÿÿÿr*
	num_split2
lstm_15/lstm_cell_15/split
lstm_15/lstm_cell_15/SigmoidSigmoid#lstm_15/lstm_cell_15/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2
lstm_15/lstm_cell_15/Sigmoid¢
lstm_15/lstm_cell_15/Sigmoid_1Sigmoid#lstm_15/lstm_cell_15/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2 
lstm_15/lstm_cell_15/Sigmoid_1«
lstm_15/lstm_cell_15/mulMul"lstm_15/lstm_cell_15/Sigmoid_1:y:0lstm_15/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2
lstm_15/lstm_cell_15/mul
lstm_15/lstm_cell_15/ReluRelu#lstm_15/lstm_cell_15/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2
lstm_15/lstm_cell_15/Relu¼
lstm_15/lstm_cell_15/mul_1Mul lstm_15/lstm_cell_15/Sigmoid:y:0'lstm_15/lstm_cell_15/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2
lstm_15/lstm_cell_15/mul_1±
lstm_15/lstm_cell_15/add_1AddV2lstm_15/lstm_cell_15/mul:z:0lstm_15/lstm_cell_15/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2
lstm_15/lstm_cell_15/add_1¢
lstm_15/lstm_cell_15/Sigmoid_2Sigmoid#lstm_15/lstm_cell_15/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2 
lstm_15/lstm_cell_15/Sigmoid_2
lstm_15/lstm_cell_15/Relu_1Relulstm_15/lstm_cell_15/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2
lstm_15/lstm_cell_15/Relu_1À
lstm_15/lstm_cell_15/mul_2Mul"lstm_15/lstm_cell_15/Sigmoid_2:y:0)lstm_15/lstm_cell_15/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2
lstm_15/lstm_cell_15/mul_2
%lstm_15/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿr   2'
%lstm_15/TensorArrayV2_1/element_shapeØ
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
lstm_15/time
 lstm_15/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2"
 lstm_15/while/maximum_iterationsz
lstm_15/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_15/while/loop_counter
lstm_15/whileWhile#lstm_15/while/loop_counter:output:0)lstm_15/while/maximum_iterations:output:0lstm_15/time:output:0 lstm_15/TensorArrayV2_1:handle:0lstm_15/zeros:output:0lstm_15/zeros_1:output:0 lstm_15/strided_slice_1:output:0?lstm_15/TensorArrayUnstack/TensorListFromTensor:output_handle:03lstm_15_lstm_cell_15_matmul_readvariableop_resource5lstm_15_lstm_cell_15_matmul_1_readvariableop_resource4lstm_15_lstm_cell_15_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿr:ÿÿÿÿÿÿÿÿÿr: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *'
bodyR
lstm_15_while_body_26077710*'
condR
lstm_15_while_cond_26077709*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿr:ÿÿÿÿÿÿÿÿÿr: : : : : *
parallel_iterations 2
lstm_15/whileÅ
8lstm_15/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿr   2:
8lstm_15/TensorArrayV2Stack/TensorListStack/element_shape
*lstm_15/TensorArrayV2Stack/TensorListStackTensorListStacklstm_15/while:output:3Alstm_15/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿr*
element_dtype02,
*lstm_15/TensorArrayV2Stack/TensorListStack
lstm_15/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
lstm_15/strided_slice_3/stack
lstm_15/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2!
lstm_15/strided_slice_3/stack_1
lstm_15/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_15/strided_slice_3/stack_2Ê
lstm_15/strided_slice_3StridedSlice3lstm_15/TensorArrayV2Stack/TensorListStack:tensor:0&lstm_15/strided_slice_3/stack:output:0(lstm_15/strided_slice_3/stack_1:output:0(lstm_15/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr*
shrink_axis_mask2
lstm_15/strided_slice_3
lstm_15/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_15/transpose_1/permÅ
lstm_15/transpose_1	Transpose3lstm_15/TensorArrayV2Stack/TensorListStack:tensor:0!lstm_15/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2
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
 *   ?2
dropout_15/dropout/Const©
dropout_15/dropout/MulMullstm_15/transpose_1:y:0!dropout_15/dropout/Const:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2
dropout_15/dropout/Mul{
dropout_15/dropout/ShapeShapelstm_15/transpose_1:y:0*
T0*
_output_shapes
:2
dropout_15/dropout/ShapeÙ
/dropout_15/dropout/random_uniform/RandomUniformRandomUniform!dropout_15/dropout/Shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿr*
dtype021
/dropout_15/dropout/random_uniform/RandomUniform
!dropout_15/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>2#
!dropout_15/dropout/GreaterEqual/yî
dropout_15/dropout/GreaterEqualGreaterEqual8dropout_15/dropout/random_uniform/RandomUniform:output:0*dropout_15/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2!
dropout_15/dropout/GreaterEqual¤
dropout_15/dropout/CastCast#dropout_15/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2
dropout_15/dropout/Castª
dropout_15/dropout/Mul_1Muldropout_15/dropout/Mul:z:0dropout_15/dropout/Cast:y:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2
dropout_15/dropout/Mul_1®
 dense_7/Tensordot/ReadVariableOpReadVariableOp)dense_7_tensordot_readvariableop_resource*
_output_shapes

:r*
dtype02"
 dense_7/Tensordot/ReadVariableOpz
dense_7/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_7/Tensordot/axes
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
dense_7/Tensordot/Shape
dense_7/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_7/Tensordot/GatherV2/axisù
dense_7/Tensordot/GatherV2GatherV2 dense_7/Tensordot/Shape:output:0dense_7/Tensordot/free:output:0(dense_7/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_7/Tensordot/GatherV2
!dense_7/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_7/Tensordot/GatherV2_1/axisÿ
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
dense_7/Tensordot/Const 
dense_7/Tensordot/ProdProd#dense_7/Tensordot/GatherV2:output:0 dense_7/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_7/Tensordot/Prod
dense_7/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_7/Tensordot/Const_1¨
dense_7/Tensordot/Prod_1Prod%dense_7/Tensordot/GatherV2_1:output:0"dense_7/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_7/Tensordot/Prod_1
dense_7/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense_7/Tensordot/concat/axisØ
dense_7/Tensordot/concatConcatV2dense_7/Tensordot/free:output:0dense_7/Tensordot/axes:output:0&dense_7/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_7/Tensordot/concat¬
dense_7/Tensordot/stackPackdense_7/Tensordot/Prod:output:0!dense_7/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_7/Tensordot/stack¾
dense_7/Tensordot/transpose	Transposedropout_15/dropout/Mul_1:z:0!dense_7/Tensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2
dense_7/Tensordot/transpose¿
dense_7/Tensordot/ReshapeReshapedense_7/Tensordot/transpose:y:0 dense_7/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
dense_7/Tensordot/Reshape¾
dense_7/Tensordot/MatMulMatMul"dense_7/Tensordot/Reshape:output:0(dense_7/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_7/Tensordot/MatMul
dense_7/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
dense_7/Tensordot/Const_2
dense_7/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_7/Tensordot/concat_1/axiså
dense_7/Tensordot/concat_1ConcatV2#dense_7/Tensordot/GatherV2:output:0"dense_7/Tensordot/Const_2:output:0(dense_7/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_7/Tensordot/concat_1°
dense_7/TensordotReshape"dense_7/Tensordot/MatMul:product:0#dense_7/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_7/Tensordot¤
dense_7/BiasAdd/ReadVariableOpReadVariableOp'dense_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_7/BiasAdd/ReadVariableOp§
dense_7/BiasAddBiasAdddense_7/Tensordot:output:0&dense_7/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_7/BiasAdd}
dense_7/SoftmaxSoftmaxdense_7/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_7/Softmaxx
IdentityIdentitydense_7/Softmax:softmax:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

IdentityÆ
NoOpNoOp^dense_7/BiasAdd/ReadVariableOp!^dense_7/Tensordot/ReadVariableOp,^lstm_14/lstm_cell_14/BiasAdd/ReadVariableOp+^lstm_14/lstm_cell_14/MatMul/ReadVariableOp-^lstm_14/lstm_cell_14/MatMul_1/ReadVariableOp^lstm_14/while,^lstm_15/lstm_cell_15/BiasAdd/ReadVariableOp+^lstm_15/lstm_cell_15/MatMul/ReadVariableOp-^lstm_15/lstm_cell_15/MatMul_1/ReadVariableOp^lstm_15/while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ]: : : : : : : : 2@
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
:ÿÿÿÿÿÿÿÿÿ]
 
_user_specified_nameinputs
¬]
õ
(sequential_7_lstm_14_while_body_26074684F
Bsequential_7_lstm_14_while_sequential_7_lstm_14_while_loop_counterL
Hsequential_7_lstm_14_while_sequential_7_lstm_14_while_maximum_iterations*
&sequential_7_lstm_14_while_placeholder,
(sequential_7_lstm_14_while_placeholder_1,
(sequential_7_lstm_14_while_placeholder_2,
(sequential_7_lstm_14_while_placeholder_3E
Asequential_7_lstm_14_while_sequential_7_lstm_14_strided_slice_1_0
}sequential_7_lstm_14_while_tensorarrayv2read_tensorlistgetitem_sequential_7_lstm_14_tensorarrayunstack_tensorlistfromtensor_0[
Hsequential_7_lstm_14_while_lstm_cell_14_matmul_readvariableop_resource_0:	] ^
Jsequential_7_lstm_14_while_lstm_cell_14_matmul_1_readvariableop_resource_0:
 X
Isequential_7_lstm_14_while_lstm_cell_14_biasadd_readvariableop_resource_0:	 '
#sequential_7_lstm_14_while_identity)
%sequential_7_lstm_14_while_identity_1)
%sequential_7_lstm_14_while_identity_2)
%sequential_7_lstm_14_while_identity_3)
%sequential_7_lstm_14_while_identity_4)
%sequential_7_lstm_14_while_identity_5C
?sequential_7_lstm_14_while_sequential_7_lstm_14_strided_slice_1
{sequential_7_lstm_14_while_tensorarrayv2read_tensorlistgetitem_sequential_7_lstm_14_tensorarrayunstack_tensorlistfromtensorY
Fsequential_7_lstm_14_while_lstm_cell_14_matmul_readvariableop_resource:	] \
Hsequential_7_lstm_14_while_lstm_cell_14_matmul_1_readvariableop_resource:
 V
Gsequential_7_lstm_14_while_lstm_cell_14_biasadd_readvariableop_resource:	 ¢>sequential_7/lstm_14/while/lstm_cell_14/BiasAdd/ReadVariableOp¢=sequential_7/lstm_14/while/lstm_cell_14/MatMul/ReadVariableOp¢?sequential_7/lstm_14/while/lstm_cell_14/MatMul_1/ReadVariableOpí
Lsequential_7/lstm_14/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ]   2N
Lsequential_7/lstm_14/while/TensorArrayV2Read/TensorListGetItem/element_shapeÑ
>sequential_7/lstm_14/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem}sequential_7_lstm_14_while_tensorarrayv2read_tensorlistgetitem_sequential_7_lstm_14_tensorarrayunstack_tensorlistfromtensor_0&sequential_7_lstm_14_while_placeholderUsequential_7/lstm_14/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]*
element_dtype02@
>sequential_7/lstm_14/while/TensorArrayV2Read/TensorListGetItem
=sequential_7/lstm_14/while/lstm_cell_14/MatMul/ReadVariableOpReadVariableOpHsequential_7_lstm_14_while_lstm_cell_14_matmul_readvariableop_resource_0*
_output_shapes
:	] *
dtype02?
=sequential_7/lstm_14/while/lstm_cell_14/MatMul/ReadVariableOp«
.sequential_7/lstm_14/while/lstm_cell_14/MatMulMatMulEsequential_7/lstm_14/while/TensorArrayV2Read/TensorListGetItem:item:0Esequential_7/lstm_14/while/lstm_cell_14/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 20
.sequential_7/lstm_14/while/lstm_cell_14/MatMul
?sequential_7/lstm_14/while/lstm_cell_14/MatMul_1/ReadVariableOpReadVariableOpJsequential_7_lstm_14_while_lstm_cell_14_matmul_1_readvariableop_resource_0* 
_output_shapes
:
 *
dtype02A
?sequential_7/lstm_14/while/lstm_cell_14/MatMul_1/ReadVariableOp
0sequential_7/lstm_14/while/lstm_cell_14/MatMul_1MatMul(sequential_7_lstm_14_while_placeholder_2Gsequential_7/lstm_14/while/lstm_cell_14/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 22
0sequential_7/lstm_14/while/lstm_cell_14/MatMul_1
+sequential_7/lstm_14/while/lstm_cell_14/addAddV28sequential_7/lstm_14/while/lstm_cell_14/MatMul:product:0:sequential_7/lstm_14/while/lstm_cell_14/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2-
+sequential_7/lstm_14/while/lstm_cell_14/add
>sequential_7/lstm_14/while/lstm_cell_14/BiasAdd/ReadVariableOpReadVariableOpIsequential_7_lstm_14_while_lstm_cell_14_biasadd_readvariableop_resource_0*
_output_shapes	
: *
dtype02@
>sequential_7/lstm_14/while/lstm_cell_14/BiasAdd/ReadVariableOp
/sequential_7/lstm_14/while/lstm_cell_14/BiasAddBiasAdd/sequential_7/lstm_14/while/lstm_cell_14/add:z:0Fsequential_7/lstm_14/while/lstm_cell_14/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 21
/sequential_7/lstm_14/while/lstm_cell_14/BiasAdd´
7sequential_7/lstm_14/while/lstm_cell_14/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :29
7sequential_7/lstm_14/while/lstm_cell_14/split/split_dimã
-sequential_7/lstm_14/while/lstm_cell_14/splitSplit@sequential_7/lstm_14/while/lstm_cell_14/split/split_dim:output:08sequential_7/lstm_14/while/lstm_cell_14/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2/
-sequential_7/lstm_14/while/lstm_cell_14/splitØ
/sequential_7/lstm_14/while/lstm_cell_14/SigmoidSigmoid6sequential_7/lstm_14/while/lstm_cell_14/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ21
/sequential_7/lstm_14/while/lstm_cell_14/SigmoidÜ
1sequential_7/lstm_14/while/lstm_cell_14/Sigmoid_1Sigmoid6sequential_7/lstm_14/while/lstm_cell_14/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ23
1sequential_7/lstm_14/while/lstm_cell_14/Sigmoid_1õ
+sequential_7/lstm_14/while/lstm_cell_14/mulMul5sequential_7/lstm_14/while/lstm_cell_14/Sigmoid_1:y:0(sequential_7_lstm_14_while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2-
+sequential_7/lstm_14/while/lstm_cell_14/mulÏ
,sequential_7/lstm_14/while/lstm_cell_14/ReluRelu6sequential_7/lstm_14/while/lstm_cell_14/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2.
,sequential_7/lstm_14/while/lstm_cell_14/Relu
-sequential_7/lstm_14/while/lstm_cell_14/mul_1Mul3sequential_7/lstm_14/while/lstm_cell_14/Sigmoid:y:0:sequential_7/lstm_14/while/lstm_cell_14/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2/
-sequential_7/lstm_14/while/lstm_cell_14/mul_1þ
-sequential_7/lstm_14/while/lstm_cell_14/add_1AddV2/sequential_7/lstm_14/while/lstm_cell_14/mul:z:01sequential_7/lstm_14/while/lstm_cell_14/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2/
-sequential_7/lstm_14/while/lstm_cell_14/add_1Ü
1sequential_7/lstm_14/while/lstm_cell_14/Sigmoid_2Sigmoid6sequential_7/lstm_14/while/lstm_cell_14/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ23
1sequential_7/lstm_14/while/lstm_cell_14/Sigmoid_2Î
.sequential_7/lstm_14/while/lstm_cell_14/Relu_1Relu1sequential_7/lstm_14/while/lstm_cell_14/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ20
.sequential_7/lstm_14/while/lstm_cell_14/Relu_1
-sequential_7/lstm_14/while/lstm_cell_14/mul_2Mul5sequential_7/lstm_14/while/lstm_cell_14/Sigmoid_2:y:0<sequential_7/lstm_14/while/lstm_cell_14/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2/
-sequential_7/lstm_14/while/lstm_cell_14/mul_2É
?sequential_7/lstm_14/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem(sequential_7_lstm_14_while_placeholder_1&sequential_7_lstm_14_while_placeholder1sequential_7/lstm_14/while/lstm_cell_14/mul_2:z:0*
_output_shapes
: *
element_dtype02A
?sequential_7/lstm_14/while/TensorArrayV2Write/TensorListSetItem
 sequential_7/lstm_14/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2"
 sequential_7/lstm_14/while/add/y½
sequential_7/lstm_14/while/addAddV2&sequential_7_lstm_14_while_placeholder)sequential_7/lstm_14/while/add/y:output:0*
T0*
_output_shapes
: 2 
sequential_7/lstm_14/while/add
"sequential_7/lstm_14/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2$
"sequential_7/lstm_14/while/add_1/yß
 sequential_7/lstm_14/while/add_1AddV2Bsequential_7_lstm_14_while_sequential_7_lstm_14_while_loop_counter+sequential_7/lstm_14/while/add_1/y:output:0*
T0*
_output_shapes
: 2"
 sequential_7/lstm_14/while/add_1¿
#sequential_7/lstm_14/while/IdentityIdentity$sequential_7/lstm_14/while/add_1:z:0 ^sequential_7/lstm_14/while/NoOp*
T0*
_output_shapes
: 2%
#sequential_7/lstm_14/while/Identityç
%sequential_7/lstm_14/while/Identity_1IdentityHsequential_7_lstm_14_while_sequential_7_lstm_14_while_maximum_iterations ^sequential_7/lstm_14/while/NoOp*
T0*
_output_shapes
: 2'
%sequential_7/lstm_14/while/Identity_1Á
%sequential_7/lstm_14/while/Identity_2Identity"sequential_7/lstm_14/while/add:z:0 ^sequential_7/lstm_14/while/NoOp*
T0*
_output_shapes
: 2'
%sequential_7/lstm_14/while/Identity_2î
%sequential_7/lstm_14/while/Identity_3IdentityOsequential_7/lstm_14/while/TensorArrayV2Write/TensorListSetItem:output_handle:0 ^sequential_7/lstm_14/while/NoOp*
T0*
_output_shapes
: 2'
%sequential_7/lstm_14/while/Identity_3â
%sequential_7/lstm_14/while/Identity_4Identity1sequential_7/lstm_14/while/lstm_cell_14/mul_2:z:0 ^sequential_7/lstm_14/while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%sequential_7/lstm_14/while/Identity_4â
%sequential_7/lstm_14/while/Identity_5Identity1sequential_7/lstm_14/while/lstm_cell_14/add_1:z:0 ^sequential_7/lstm_14/while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%sequential_7/lstm_14/while/Identity_5Ç
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
%sequential_7_lstm_14_while_identity_5.sequential_7/lstm_14/while/Identity_5:output:0"
Gsequential_7_lstm_14_while_lstm_cell_14_biasadd_readvariableop_resourceIsequential_7_lstm_14_while_lstm_cell_14_biasadd_readvariableop_resource_0"
Hsequential_7_lstm_14_while_lstm_cell_14_matmul_1_readvariableop_resourceJsequential_7_lstm_14_while_lstm_cell_14_matmul_1_readvariableop_resource_0"
Fsequential_7_lstm_14_while_lstm_cell_14_matmul_readvariableop_resourceHsequential_7_lstm_14_while_lstm_cell_14_matmul_readvariableop_resource_0"
?sequential_7_lstm_14_while_sequential_7_lstm_14_strided_slice_1Asequential_7_lstm_14_while_sequential_7_lstm_14_strided_slice_1_0"ü
{sequential_7_lstm_14_while_tensorarrayv2read_tensorlistgetitem_sequential_7_lstm_14_tensorarrayunstack_tensorlistfromtensor}sequential_7_lstm_14_while_tensorarrayv2read_tensorlistgetitem_sequential_7_lstm_14_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2
>sequential_7/lstm_14/while/lstm_cell_14/BiasAdd/ReadVariableOp>sequential_7/lstm_14/while/lstm_cell_14/BiasAdd/ReadVariableOp2~
=sequential_7/lstm_14/while/lstm_cell_14/MatMul/ReadVariableOp=sequential_7/lstm_14/while/lstm_cell_14/MatMul/ReadVariableOp2
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
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 


J__inference_lstm_cell_14_layer_call_and_return_conditional_losses_26079325

inputs
states_0
states_11
matmul_readvariableop_resource:	] 4
 matmul_1_readvariableop_resource:
 .
biasadd_readvariableop_resource:	 
identity

identity_1

identity_2¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	] *
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
MatMul
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
 *
dtype02
MatMul_1/ReadVariableOp|
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
add
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
: *
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2	
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
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2
split`
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
Sigmoidd
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	Sigmoid_1]
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mulW
ReluRelusplit:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relui
mul_1MulSigmoid:y:0Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mul_1^
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
add_1d
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	Sigmoid_2V
Relu_1Relu	add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu_1m
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mul_2e
IdentityIdentity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityi

Identity_1Identity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1i

Identity_2Identity	add_1:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

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
A:ÿÿÿÿÿÿÿÿÿ]:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
 
_user_specified_nameinputs:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
states/0:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
states/1
\

E__inference_lstm_14_layer_call_and_return_conditional_losses_26078324

inputs>
+lstm_cell_14_matmul_readvariableop_resource:	] A
-lstm_cell_14_matmul_1_readvariableop_resource:
 ;
,lstm_cell_14_biasadd_readvariableop_resource:	 
identity¢#lstm_cell_14/BiasAdd/ReadVariableOp¢"lstm_cell_14/MatMul/ReadVariableOp¢$lstm_cell_14/MatMul_1/ReadVariableOp¢whileD
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
B :2
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
B :2
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
:ÿÿÿÿÿÿÿÿÿ2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :2
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
B :2
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
:ÿÿÿÿÿÿÿÿÿ2	
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
"lstm_cell_14/MatMul/ReadVariableOpReadVariableOp+lstm_cell_14_matmul_readvariableop_resource*
_output_shapes
:	] *
dtype02$
"lstm_cell_14/MatMul/ReadVariableOp­
lstm_cell_14/MatMulMatMulstrided_slice_2:output:0*lstm_cell_14/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
lstm_cell_14/MatMul¼
$lstm_cell_14/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_14_matmul_1_readvariableop_resource* 
_output_shapes
:
 *
dtype02&
$lstm_cell_14/MatMul_1/ReadVariableOp©
lstm_cell_14/MatMul_1MatMulzeros:output:0,lstm_cell_14/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
lstm_cell_14/MatMul_1 
lstm_cell_14/addAddV2lstm_cell_14/MatMul:product:0lstm_cell_14/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
lstm_cell_14/add´
#lstm_cell_14/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_14_biasadd_readvariableop_resource*
_output_shapes	
: *
dtype02%
#lstm_cell_14/BiasAdd/ReadVariableOp­
lstm_cell_14/BiasAddBiasAddlstm_cell_14/add:z:0+lstm_cell_14/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
lstm_cell_14/BiasAdd~
lstm_cell_14/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_14/split/split_dim÷
lstm_cell_14/splitSplit%lstm_cell_14/split/split_dim:output:0lstm_cell_14/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2
lstm_cell_14/split
lstm_cell_14/SigmoidSigmoidlstm_cell_14/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_14/Sigmoid
lstm_cell_14/Sigmoid_1Sigmoidlstm_cell_14/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_14/Sigmoid_1
lstm_cell_14/mulMullstm_cell_14/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_14/mul~
lstm_cell_14/ReluRelulstm_cell_14/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_14/Relu
lstm_cell_14/mul_1Mullstm_cell_14/Sigmoid:y:0lstm_cell_14/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_14/mul_1
lstm_cell_14/add_1AddV2lstm_cell_14/mul:z:0lstm_cell_14/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_14/add_1
lstm_cell_14/Sigmoid_2Sigmoidlstm_cell_14/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_14/Sigmoid_2}
lstm_cell_14/Relu_1Relulstm_cell_14/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_14/Relu_1¡
lstm_cell_14/mul_2Mullstm_cell_14/Sigmoid_2:y:0!lstm_cell_14/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_14/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ  2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_14_matmul_readvariableop_resource-lstm_cell_14_matmul_1_readvariableop_resource,lstm_cell_14_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_26078240*
condR
while_cond_26078239*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ  22
0TensorArrayV2Stack/TensorListStack/element_shapeé
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
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
:ÿÿÿÿÿÿÿÿÿ*
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
:ÿÿÿÿÿÿÿÿÿ2
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
:ÿÿÿÿÿÿÿÿÿ2

IdentityÈ
NoOpNoOp$^lstm_cell_14/BiasAdd/ReadVariableOp#^lstm_cell_14/MatMul/ReadVariableOp%^lstm_cell_14/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ]: : : 2J
#lstm_cell_14/BiasAdd/ReadVariableOp#lstm_cell_14/BiasAdd/ReadVariableOp2H
"lstm_cell_14/MatMul/ReadVariableOp"lstm_cell_14/MatMul/ReadVariableOp2L
$lstm_cell_14/MatMul_1/ReadVariableOp$lstm_cell_14/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
 
_user_specified_nameinputs
ù

J__inference_lstm_cell_15_layer_call_and_return_conditional_losses_26075795

inputs

states
states_12
matmul_readvariableop_resource:
È3
 matmul_1_readvariableop_resource:	rÈ.
biasadd_readvariableop_resource:	È
identity

identity_1

identity_2¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
È*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
MatMul
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	rÈ*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
add
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:È*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2	
BiasAddd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim¿
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿr:ÿÿÿÿÿÿÿÿÿr:ÿÿÿÿÿÿÿÿÿr:ÿÿÿÿÿÿÿÿÿr*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2
	Sigmoid_1\
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2
mulV
ReluRelusplit:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2
	Sigmoid_2U
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2
mul_2d
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2

Identityh

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2

Identity_1h

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2

Identity_2
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
@:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿr:ÿÿÿÿÿÿÿÿÿr: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
 
_user_specified_namestates:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
 
_user_specified_namestates
Ï
g
H__inference_dropout_15_layer_call_and_return_conditional_losses_26076628

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/Constw
dropout/MulMulinputsdropout/Const:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape¸
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿr*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>2
dropout/GreaterEqual/yÂ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2
dropout/Cast~
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2
dropout/Mul_1i
IdentityIdentitydropout/Mul_1:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿr:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
 
_user_specified_nameinputs

f
H__inference_dropout_14_layer_call_and_return_conditional_losses_26078524

inputs

identity_1_
IdentityIdentityinputs*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityn

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
à
º
*__inference_lstm_15_layer_call_fn_26079161
inputs_0
unknown:
È
	unknown_0:	rÈ
	unknown_1:	È
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿr*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_lstm_15_layer_call_and_return_conditional_losses_260757322
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿr2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0
ý%
ó
while_body_26075663
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_01
while_lstm_cell_15_26075687_0:
È0
while_lstm_cell_15_26075689_0:	rÈ,
while_lstm_cell_15_26075691_0:	È
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor/
while_lstm_cell_15_26075687:
È.
while_lstm_cell_15_26075689:	rÈ*
while_lstm_cell_15_26075691:	È¢*while/lstm_cell_15/StatefulPartitionedCallÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ  29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÔ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemé
*while/lstm_cell_15/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_15_26075687_0while_lstm_cell_15_26075689_0while_lstm_cell_15_26075691_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿr:ÿÿÿÿÿÿÿÿÿr:ÿÿÿÿÿÿÿÿÿr*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_lstm_cell_15_layer_call_and_return_conditional_losses_260756492,
*while/lstm_cell_15/StatefulPartitionedCall÷
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
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3¤
while/Identity_4Identity3while/lstm_cell_15/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2
while/Identity_4¤
while/Identity_5Identity3while/lstm_cell_15/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2
while/Identity_5

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
while_lstm_cell_15_26075687while_lstm_cell_15_26075687_0"<
while_lstm_cell_15_26075689while_lstm_cell_15_26075689_0"<
while_lstm_cell_15_26075691while_lstm_cell_15_26075691_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿr:ÿÿÿÿÿÿÿÿÿr: : : : : 2X
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
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr:

_output_shapes
: :

_output_shapes
: 
ã
Í
while_cond_26078239
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_26078239___redundant_placeholder06
2while_while_cond_26078239___redundant_placeholder16
2while_while_cond_26078239___redundant_placeholder26
2while_while_cond_26078239___redundant_placeholder3
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
B: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 

_output_shapes
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
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
ÓJ
Ô

lstm_15_while_body_26077710,
(lstm_15_while_lstm_15_while_loop_counter2
.lstm_15_while_lstm_15_while_maximum_iterations
lstm_15_while_placeholder
lstm_15_while_placeholder_1
lstm_15_while_placeholder_2
lstm_15_while_placeholder_3+
'lstm_15_while_lstm_15_strided_slice_1_0g
clstm_15_while_tensorarrayv2read_tensorlistgetitem_lstm_15_tensorarrayunstack_tensorlistfromtensor_0O
;lstm_15_while_lstm_cell_15_matmul_readvariableop_resource_0:
ÈP
=lstm_15_while_lstm_cell_15_matmul_1_readvariableop_resource_0:	rÈK
<lstm_15_while_lstm_cell_15_biasadd_readvariableop_resource_0:	È
lstm_15_while_identity
lstm_15_while_identity_1
lstm_15_while_identity_2
lstm_15_while_identity_3
lstm_15_while_identity_4
lstm_15_while_identity_5)
%lstm_15_while_lstm_15_strided_slice_1e
alstm_15_while_tensorarrayv2read_tensorlistgetitem_lstm_15_tensorarrayunstack_tensorlistfromtensorM
9lstm_15_while_lstm_cell_15_matmul_readvariableop_resource:
ÈN
;lstm_15_while_lstm_cell_15_matmul_1_readvariableop_resource:	rÈI
:lstm_15_while_lstm_cell_15_biasadd_readvariableop_resource:	È¢1lstm_15/while/lstm_cell_15/BiasAdd/ReadVariableOp¢0lstm_15/while/lstm_cell_15/MatMul/ReadVariableOp¢2lstm_15/while/lstm_cell_15/MatMul_1/ReadVariableOpÓ
?lstm_15/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ  2A
?lstm_15/while/TensorArrayV2Read/TensorListGetItem/element_shape
1lstm_15/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemclstm_15_while_tensorarrayv2read_tensorlistgetitem_lstm_15_tensorarrayunstack_tensorlistfromtensor_0lstm_15_while_placeholderHlstm_15/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype023
1lstm_15/while/TensorArrayV2Read/TensorListGetItemâ
0lstm_15/while/lstm_cell_15/MatMul/ReadVariableOpReadVariableOp;lstm_15_while_lstm_cell_15_matmul_readvariableop_resource_0* 
_output_shapes
:
È*
dtype022
0lstm_15/while/lstm_cell_15/MatMul/ReadVariableOp÷
!lstm_15/while/lstm_cell_15/MatMulMatMul8lstm_15/while/TensorArrayV2Read/TensorListGetItem:item:08lstm_15/while/lstm_cell_15/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2#
!lstm_15/while/lstm_cell_15/MatMulç
2lstm_15/while/lstm_cell_15/MatMul_1/ReadVariableOpReadVariableOp=lstm_15_while_lstm_cell_15_matmul_1_readvariableop_resource_0*
_output_shapes
:	rÈ*
dtype024
2lstm_15/while/lstm_cell_15/MatMul_1/ReadVariableOpà
#lstm_15/while/lstm_cell_15/MatMul_1MatMullstm_15_while_placeholder_2:lstm_15/while/lstm_cell_15/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2%
#lstm_15/while/lstm_cell_15/MatMul_1Ø
lstm_15/while/lstm_cell_15/addAddV2+lstm_15/while/lstm_cell_15/MatMul:product:0-lstm_15/while/lstm_cell_15/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2 
lstm_15/while/lstm_cell_15/addà
1lstm_15/while/lstm_cell_15/BiasAdd/ReadVariableOpReadVariableOp<lstm_15_while_lstm_cell_15_biasadd_readvariableop_resource_0*
_output_shapes	
:È*
dtype023
1lstm_15/while/lstm_cell_15/BiasAdd/ReadVariableOpå
"lstm_15/while/lstm_cell_15/BiasAddBiasAdd"lstm_15/while/lstm_cell_15/add:z:09lstm_15/while/lstm_cell_15/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2$
"lstm_15/while/lstm_cell_15/BiasAdd
*lstm_15/while/lstm_cell_15/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2,
*lstm_15/while/lstm_cell_15/split/split_dim«
 lstm_15/while/lstm_cell_15/splitSplit3lstm_15/while/lstm_cell_15/split/split_dim:output:0+lstm_15/while/lstm_cell_15/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿr:ÿÿÿÿÿÿÿÿÿr:ÿÿÿÿÿÿÿÿÿr:ÿÿÿÿÿÿÿÿÿr*
	num_split2"
 lstm_15/while/lstm_cell_15/split°
"lstm_15/while/lstm_cell_15/SigmoidSigmoid)lstm_15/while/lstm_cell_15/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2$
"lstm_15/while/lstm_cell_15/Sigmoid´
$lstm_15/while/lstm_cell_15/Sigmoid_1Sigmoid)lstm_15/while/lstm_cell_15/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2&
$lstm_15/while/lstm_cell_15/Sigmoid_1À
lstm_15/while/lstm_cell_15/mulMul(lstm_15/while/lstm_cell_15/Sigmoid_1:y:0lstm_15_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2 
lstm_15/while/lstm_cell_15/mul§
lstm_15/while/lstm_cell_15/ReluRelu)lstm_15/while/lstm_cell_15/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2!
lstm_15/while/lstm_cell_15/ReluÔ
 lstm_15/while/lstm_cell_15/mul_1Mul&lstm_15/while/lstm_cell_15/Sigmoid:y:0-lstm_15/while/lstm_cell_15/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2"
 lstm_15/while/lstm_cell_15/mul_1É
 lstm_15/while/lstm_cell_15/add_1AddV2"lstm_15/while/lstm_cell_15/mul:z:0$lstm_15/while/lstm_cell_15/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2"
 lstm_15/while/lstm_cell_15/add_1´
$lstm_15/while/lstm_cell_15/Sigmoid_2Sigmoid)lstm_15/while/lstm_cell_15/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2&
$lstm_15/while/lstm_cell_15/Sigmoid_2¦
!lstm_15/while/lstm_cell_15/Relu_1Relu$lstm_15/while/lstm_cell_15/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2#
!lstm_15/while/lstm_cell_15/Relu_1Ø
 lstm_15/while/lstm_cell_15/mul_2Mul(lstm_15/while/lstm_cell_15/Sigmoid_2:y:0/lstm_15/while/lstm_cell_15/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2"
 lstm_15/while/lstm_cell_15/mul_2
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
lstm_15/while/add/y
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
lstm_15/while/add_1/y
lstm_15/while/add_1AddV2(lstm_15_while_lstm_15_while_loop_counterlstm_15/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lstm_15/while/add_1
lstm_15/while/IdentityIdentitylstm_15/while/add_1:z:0^lstm_15/while/NoOp*
T0*
_output_shapes
: 2
lstm_15/while/Identity¦
lstm_15/while/Identity_1Identity.lstm_15_while_lstm_15_while_maximum_iterations^lstm_15/while/NoOp*
T0*
_output_shapes
: 2
lstm_15/while/Identity_1
lstm_15/while/Identity_2Identitylstm_15/while/add:z:0^lstm_15/while/NoOp*
T0*
_output_shapes
: 2
lstm_15/while/Identity_2º
lstm_15/while/Identity_3IdentityBlstm_15/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_15/while/NoOp*
T0*
_output_shapes
: 2
lstm_15/while/Identity_3­
lstm_15/while/Identity_4Identity$lstm_15/while/lstm_cell_15/mul_2:z:0^lstm_15/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2
lstm_15/while/Identity_4­
lstm_15/while/Identity_5Identity$lstm_15/while/lstm_cell_15/add_1:z:0^lstm_15/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2
lstm_15/while/Identity_5
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
9lstm_15_while_lstm_cell_15_matmul_readvariableop_resource;lstm_15_while_lstm_cell_15_matmul_readvariableop_resource_0"È
alstm_15_while_tensorarrayv2read_tensorlistgetitem_lstm_15_tensorarrayunstack_tensorlistfromtensorclstm_15_while_tensorarrayv2read_tensorlistgetitem_lstm_15_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿr:ÿÿÿÿÿÿÿÿÿr: : : : : 2f
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
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr:

_output_shapes
: :

_output_shapes
: 
ËF

E__inference_lstm_14_layer_call_and_return_conditional_losses_26075312

inputs(
lstm_cell_14_26075230:	] )
lstm_cell_14_26075232:
 $
lstm_cell_14_26075234:	 
identity¢$lstm_cell_14/StatefulPartitionedCall¢whileD
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
B :2
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
B :2
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
:ÿÿÿÿÿÿÿÿÿ2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :2
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
B :2
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
:ÿÿÿÿÿÿÿÿÿ2	
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
$lstm_cell_14/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_14_26075230lstm_cell_14_26075232lstm_cell_14_26075234*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_lstm_cell_14_layer_call_and_return_conditional_losses_260751652&
$lstm_cell_14/StatefulPartitionedCall
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ  2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_14_26075230lstm_cell_14_26075232lstm_cell_14_26075234*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_26075243*
condR
while_cond_26075242*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ  22
0TensorArrayV2Stack/TensorListStack/element_shapeò
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
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
:ÿÿÿÿÿÿÿÿÿ*
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
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
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
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity}
NoOpNoOp%^lstm_cell_14/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ]: : : 2L
$lstm_cell_14/StatefulPartitionedCall$lstm_cell_14/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ]
 
_user_specified_nameinputs
ß
Í
while_cond_26076441
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_26076441___redundant_placeholder06
2while_while_cond_26076441___redundant_placeholder16
2while_while_cond_26076441___redundant_placeholder26
2while_while_cond_26076441___redundant_placeholder3
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
@: : : : :ÿÿÿÿÿÿÿÿÿr:ÿÿÿÿÿÿÿÿÿr: ::::: 

_output_shapes
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
:ÿÿÿÿÿÿÿÿÿr:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr:

_output_shapes
: :

_output_shapes
:
í[

E__inference_lstm_15_layer_call_and_return_conditional_losses_26076795

inputs?
+lstm_cell_15_matmul_readvariableop_resource:
È@
-lstm_cell_15_matmul_1_readvariableop_resource:	rÈ;
,lstm_cell_15_biasadd_readvariableop_resource:	È
identity¢#lstm_cell_15/BiasAdd/ReadVariableOp¢"lstm_cell_15/MatMul/ReadVariableOp¢$lstm_cell_15/MatMul_1/ReadVariableOp¢whileD
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
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :r2
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

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :r2
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
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :r2
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
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :r2
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
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2	
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
:ÿÿÿÿÿÿÿÿÿ2
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
valueB"ÿÿÿÿ  27
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
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
strided_slice_2¶
"lstm_cell_15/MatMul/ReadVariableOpReadVariableOp+lstm_cell_15_matmul_readvariableop_resource* 
_output_shapes
:
È*
dtype02$
"lstm_cell_15/MatMul/ReadVariableOp­
lstm_cell_15/MatMulMatMulstrided_slice_2:output:0*lstm_cell_15/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell_15/MatMul»
$lstm_cell_15/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_15_matmul_1_readvariableop_resource*
_output_shapes
:	rÈ*
dtype02&
$lstm_cell_15/MatMul_1/ReadVariableOp©
lstm_cell_15/MatMul_1MatMulzeros:output:0,lstm_cell_15/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell_15/MatMul_1 
lstm_cell_15/addAddV2lstm_cell_15/MatMul:product:0lstm_cell_15/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell_15/add´
#lstm_cell_15/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_15_biasadd_readvariableop_resource*
_output_shapes	
:È*
dtype02%
#lstm_cell_15/BiasAdd/ReadVariableOp­
lstm_cell_15/BiasAddBiasAddlstm_cell_15/add:z:0+lstm_cell_15/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell_15/BiasAdd~
lstm_cell_15/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_15/split/split_dimó
lstm_cell_15/splitSplit%lstm_cell_15/split/split_dim:output:0lstm_cell_15/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿr:ÿÿÿÿÿÿÿÿÿr:ÿÿÿÿÿÿÿÿÿr:ÿÿÿÿÿÿÿÿÿr*
	num_split2
lstm_cell_15/split
lstm_cell_15/SigmoidSigmoidlstm_cell_15/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2
lstm_cell_15/Sigmoid
lstm_cell_15/Sigmoid_1Sigmoidlstm_cell_15/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2
lstm_cell_15/Sigmoid_1
lstm_cell_15/mulMullstm_cell_15/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2
lstm_cell_15/mul}
lstm_cell_15/ReluRelulstm_cell_15/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2
lstm_cell_15/Relu
lstm_cell_15/mul_1Mullstm_cell_15/Sigmoid:y:0lstm_cell_15/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2
lstm_cell_15/mul_1
lstm_cell_15/add_1AddV2lstm_cell_15/mul:z:0lstm_cell_15/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2
lstm_cell_15/add_1
lstm_cell_15/Sigmoid_2Sigmoidlstm_cell_15/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2
lstm_cell_15/Sigmoid_2|
lstm_cell_15/Relu_1Relulstm_cell_15/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2
lstm_cell_15/Relu_1 
lstm_cell_15/mul_2Mullstm_cell_15/Sigmoid_2:y:0!lstm_cell_15/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2
lstm_cell_15/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿr   2
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
while/loop_counter
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_15_matmul_readvariableop_resource-lstm_cell_15_matmul_1_readvariableop_resource,lstm_cell_15_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿr:ÿÿÿÿÿÿÿÿÿr: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_26076711*
condR
while_cond_26076710*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿr:ÿÿÿÿÿÿÿÿÿr: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿr   22
0TensorArrayV2Stack/TensorListStack/element_shapeè
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿr*
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
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm¥
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2
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
:ÿÿÿÿÿÿÿÿÿr2

IdentityÈ
NoOpNoOp$^lstm_cell_15/BiasAdd/ReadVariableOp#^lstm_cell_15/MatMul/ReadVariableOp%^lstm_cell_15/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:ÿÿÿÿÿÿÿÿÿ: : : 2J
#lstm_cell_15/BiasAdd/ReadVariableOp#lstm_cell_15/BiasAdd/ReadVariableOp2H
"lstm_cell_15/MatMul/ReadVariableOp"lstm_cell_15/MatMul/ReadVariableOp2L
$lstm_cell_15/MatMul_1/ReadVariableOp$lstm_cell_15/MatMul_1/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


J__inference_lstm_cell_15_layer_call_and_return_conditional_losses_26079391

inputs
states_0
states_12
matmul_readvariableop_resource:
È3
 matmul_1_readvariableop_resource:	rÈ.
biasadd_readvariableop_resource:	È
identity

identity_1

identity_2¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
È*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
MatMul
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	rÈ*
dtype02
MatMul_1/ReadVariableOp|
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
add
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:È*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2	
BiasAddd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim¿
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿr:ÿÿÿÿÿÿÿÿÿr:ÿÿÿÿÿÿÿÿÿr:ÿÿÿÿÿÿÿÿÿr*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2
	Sigmoid_1\
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2
mulV
ReluRelusplit:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2
	Sigmoid_2U
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2
mul_2d
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2

Identityh

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2

Identity_1h

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2

Identity_2
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
@:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿr:ÿÿÿÿÿÿÿÿÿr: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
"
_user_specified_name
states/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
"
_user_specified_name
states/1
à
º
*__inference_lstm_14_layer_call_fn_26078486
inputs_0
unknown:	] 
	unknown_0:
 
	unknown_1:	 
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_lstm_14_layer_call_and_return_conditional_losses_260751022
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

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
Ç
ù
/__inference_lstm_cell_14_layer_call_fn_26079342

inputs
states_0
states_1
unknown:	] 
	unknown_0:
 
	unknown_1:	 
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
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_lstm_cell_14_layer_call_and_return_conditional_losses_260750192
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

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
A:ÿÿÿÿÿÿÿÿÿ]:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
 
_user_specified_nameinputs:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
states/0:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
states/1
?
Ô
while_body_26078613
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
3while_lstm_cell_15_matmul_readvariableop_resource_0:
ÈH
5while_lstm_cell_15_matmul_1_readvariableop_resource_0:	rÈC
4while_lstm_cell_15_biasadd_readvariableop_resource_0:	È
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
1while_lstm_cell_15_matmul_readvariableop_resource:
ÈF
3while_lstm_cell_15_matmul_1_readvariableop_resource:	rÈA
2while_lstm_cell_15_biasadd_readvariableop_resource:	È¢)while/lstm_cell_15/BiasAdd/ReadVariableOp¢(while/lstm_cell_15/MatMul/ReadVariableOp¢*while/lstm_cell_15/MatMul_1/ReadVariableOpÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ  29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÔ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemÊ
(while/lstm_cell_15/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_15_matmul_readvariableop_resource_0* 
_output_shapes
:
È*
dtype02*
(while/lstm_cell_15/MatMul/ReadVariableOp×
while/lstm_cell_15/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_15/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell_15/MatMulÏ
*while/lstm_cell_15/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_15_matmul_1_readvariableop_resource_0*
_output_shapes
:	rÈ*
dtype02,
*while/lstm_cell_15/MatMul_1/ReadVariableOpÀ
while/lstm_cell_15/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_15/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell_15/MatMul_1¸
while/lstm_cell_15/addAddV2#while/lstm_cell_15/MatMul:product:0%while/lstm_cell_15/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell_15/addÈ
)while/lstm_cell_15/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_15_biasadd_readvariableop_resource_0*
_output_shapes	
:È*
dtype02+
)while/lstm_cell_15/BiasAdd/ReadVariableOpÅ
while/lstm_cell_15/BiasAddBiasAddwhile/lstm_cell_15/add:z:01while/lstm_cell_15/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell_15/BiasAdd
"while/lstm_cell_15/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_15/split/split_dim
while/lstm_cell_15/splitSplit+while/lstm_cell_15/split/split_dim:output:0#while/lstm_cell_15/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿr:ÿÿÿÿÿÿÿÿÿr:ÿÿÿÿÿÿÿÿÿr:ÿÿÿÿÿÿÿÿÿr*
	num_split2
while/lstm_cell_15/split
while/lstm_cell_15/SigmoidSigmoid!while/lstm_cell_15/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2
while/lstm_cell_15/Sigmoid
while/lstm_cell_15/Sigmoid_1Sigmoid!while/lstm_cell_15/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2
while/lstm_cell_15/Sigmoid_1 
while/lstm_cell_15/mulMul while/lstm_cell_15/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2
while/lstm_cell_15/mul
while/lstm_cell_15/ReluRelu!while/lstm_cell_15/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2
while/lstm_cell_15/Relu´
while/lstm_cell_15/mul_1Mulwhile/lstm_cell_15/Sigmoid:y:0%while/lstm_cell_15/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2
while/lstm_cell_15/mul_1©
while/lstm_cell_15/add_1AddV2while/lstm_cell_15/mul:z:0while/lstm_cell_15/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2
while/lstm_cell_15/add_1
while/lstm_cell_15/Sigmoid_2Sigmoid!while/lstm_cell_15/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2
while/lstm_cell_15/Sigmoid_2
while/lstm_cell_15/Relu_1Reluwhile/lstm_cell_15/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2
while/lstm_cell_15/Relu_1¸
while/lstm_cell_15/mul_2Mul while/lstm_cell_15/Sigmoid_2:y:0'while/lstm_cell_15/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2
while/lstm_cell_15/mul_2à
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
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_15/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_15/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2
while/Identity_5Þ

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
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿr:ÿÿÿÿÿÿÿÿÿr: : : : : 2V
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
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr:

_output_shapes
: :

_output_shapes
: 
?
Ô
while_body_26076711
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
3while_lstm_cell_15_matmul_readvariableop_resource_0:
ÈH
5while_lstm_cell_15_matmul_1_readvariableop_resource_0:	rÈC
4while_lstm_cell_15_biasadd_readvariableop_resource_0:	È
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
1while_lstm_cell_15_matmul_readvariableop_resource:
ÈF
3while_lstm_cell_15_matmul_1_readvariableop_resource:	rÈA
2while_lstm_cell_15_biasadd_readvariableop_resource:	È¢)while/lstm_cell_15/BiasAdd/ReadVariableOp¢(while/lstm_cell_15/MatMul/ReadVariableOp¢*while/lstm_cell_15/MatMul_1/ReadVariableOpÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ  29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÔ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemÊ
(while/lstm_cell_15/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_15_matmul_readvariableop_resource_0* 
_output_shapes
:
È*
dtype02*
(while/lstm_cell_15/MatMul/ReadVariableOp×
while/lstm_cell_15/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_15/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell_15/MatMulÏ
*while/lstm_cell_15/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_15_matmul_1_readvariableop_resource_0*
_output_shapes
:	rÈ*
dtype02,
*while/lstm_cell_15/MatMul_1/ReadVariableOpÀ
while/lstm_cell_15/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_15/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell_15/MatMul_1¸
while/lstm_cell_15/addAddV2#while/lstm_cell_15/MatMul:product:0%while/lstm_cell_15/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell_15/addÈ
)while/lstm_cell_15/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_15_biasadd_readvariableop_resource_0*
_output_shapes	
:È*
dtype02+
)while/lstm_cell_15/BiasAdd/ReadVariableOpÅ
while/lstm_cell_15/BiasAddBiasAddwhile/lstm_cell_15/add:z:01while/lstm_cell_15/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell_15/BiasAdd
"while/lstm_cell_15/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_15/split/split_dim
while/lstm_cell_15/splitSplit+while/lstm_cell_15/split/split_dim:output:0#while/lstm_cell_15/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿr:ÿÿÿÿÿÿÿÿÿr:ÿÿÿÿÿÿÿÿÿr:ÿÿÿÿÿÿÿÿÿr*
	num_split2
while/lstm_cell_15/split
while/lstm_cell_15/SigmoidSigmoid!while/lstm_cell_15/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2
while/lstm_cell_15/Sigmoid
while/lstm_cell_15/Sigmoid_1Sigmoid!while/lstm_cell_15/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2
while/lstm_cell_15/Sigmoid_1 
while/lstm_cell_15/mulMul while/lstm_cell_15/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2
while/lstm_cell_15/mul
while/lstm_cell_15/ReluRelu!while/lstm_cell_15/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2
while/lstm_cell_15/Relu´
while/lstm_cell_15/mul_1Mulwhile/lstm_cell_15/Sigmoid:y:0%while/lstm_cell_15/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2
while/lstm_cell_15/mul_1©
while/lstm_cell_15/add_1AddV2while/lstm_cell_15/mul:z:0while/lstm_cell_15/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2
while/lstm_cell_15/add_1
while/lstm_cell_15/Sigmoid_2Sigmoid!while/lstm_cell_15/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2
while/lstm_cell_15/Sigmoid_2
while/lstm_cell_15/Relu_1Reluwhile/lstm_cell_15/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2
while/lstm_cell_15/Relu_1¸
while/lstm_cell_15/mul_2Mul while/lstm_cell_15/Sigmoid_2:y:0'while/lstm_cell_15/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2
while/lstm_cell_15/mul_2à
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
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_15/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_15/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2
while/Identity_5Þ

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
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿr:ÿÿÿÿÿÿÿÿÿr: : : : : 2V
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
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr:

_output_shapes
: :

_output_shapes
: 
Ç
ù
/__inference_lstm_cell_14_layer_call_fn_26079359

inputs
states_0
states_1
unknown:	] 
	unknown_0:
 
	unknown_1:	 
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
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_lstm_cell_14_layer_call_and_return_conditional_losses_260751652
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

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
A:ÿÿÿÿÿÿÿÿÿ]:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
 
_user_specified_nameinputs:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
states/0:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
states/1
½
ù
/__inference_lstm_cell_15_layer_call_fn_26079440

inputs
states_0
states_1
unknown:
È
	unknown_0:	rÈ
	unknown_1:	È
identity

identity_1

identity_2¢StatefulPartitionedCallÅ
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿr:ÿÿÿÿÿÿÿÿÿr:ÿÿÿÿÿÿÿÿÿr*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_lstm_cell_15_layer_call_and_return_conditional_losses_260756492
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2

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
@:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿr:ÿÿÿÿÿÿÿÿÿr: : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
"
_user_specified_name
states/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
"
_user_specified_name
states/1
Ð

í
lstm_15_while_cond_26077709,
(lstm_15_while_lstm_15_while_loop_counter2
.lstm_15_while_lstm_15_while_maximum_iterations
lstm_15_while_placeholder
lstm_15_while_placeholder_1
lstm_15_while_placeholder_2
lstm_15_while_placeholder_3.
*lstm_15_while_less_lstm_15_strided_slice_1F
Blstm_15_while_lstm_15_while_cond_26077709___redundant_placeholder0F
Blstm_15_while_lstm_15_while_cond_26077709___redundant_placeholder1F
Blstm_15_while_lstm_15_while_cond_26077709___redundant_placeholder2F
Blstm_15_while_lstm_15_while_cond_26077709___redundant_placeholder3
lstm_15_while_identity

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
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿr:ÿÿÿÿÿÿÿÿÿr: ::::: 

_output_shapes
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
:ÿÿÿÿÿÿÿÿÿr:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr:

_output_shapes
: :

_output_shapes
:
°?
Ô
while_body_26076907
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_14_matmul_readvariableop_resource_0:	] I
5while_lstm_cell_14_matmul_1_readvariableop_resource_0:
 C
4while_lstm_cell_14_biasadd_readvariableop_resource_0:	 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_14_matmul_readvariableop_resource:	] G
3while_lstm_cell_14_matmul_1_readvariableop_resource:
 A
2while_lstm_cell_14_biasadd_readvariableop_resource:	 ¢)while/lstm_cell_14/BiasAdd/ReadVariableOp¢(while/lstm_cell_14/MatMul/ReadVariableOp¢*while/lstm_cell_14/MatMul_1/ReadVariableOpÃ
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
(while/lstm_cell_14/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_14_matmul_readvariableop_resource_0*
_output_shapes
:	] *
dtype02*
(while/lstm_cell_14/MatMul/ReadVariableOp×
while/lstm_cell_14/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_14/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/lstm_cell_14/MatMulÐ
*while/lstm_cell_14/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_14_matmul_1_readvariableop_resource_0* 
_output_shapes
:
 *
dtype02,
*while/lstm_cell_14/MatMul_1/ReadVariableOpÀ
while/lstm_cell_14/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_14/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/lstm_cell_14/MatMul_1¸
while/lstm_cell_14/addAddV2#while/lstm_cell_14/MatMul:product:0%while/lstm_cell_14/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/lstm_cell_14/addÈ
)while/lstm_cell_14/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_14_biasadd_readvariableop_resource_0*
_output_shapes	
: *
dtype02+
)while/lstm_cell_14/BiasAdd/ReadVariableOpÅ
while/lstm_cell_14/BiasAddBiasAddwhile/lstm_cell_14/add:z:01while/lstm_cell_14/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/lstm_cell_14/BiasAdd
"while/lstm_cell_14/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_14/split/split_dim
while/lstm_cell_14/splitSplit+while/lstm_cell_14/split/split_dim:output:0#while/lstm_cell_14/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2
while/lstm_cell_14/split
while/lstm_cell_14/SigmoidSigmoid!while/lstm_cell_14/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_14/Sigmoid
while/lstm_cell_14/Sigmoid_1Sigmoid!while/lstm_cell_14/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_14/Sigmoid_1¡
while/lstm_cell_14/mulMul while/lstm_cell_14/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_14/mul
while/lstm_cell_14/ReluRelu!while/lstm_cell_14/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_14/Reluµ
while/lstm_cell_14/mul_1Mulwhile/lstm_cell_14/Sigmoid:y:0%while/lstm_cell_14/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_14/mul_1ª
while/lstm_cell_14/add_1AddV2while/lstm_cell_14/mul:z:0while/lstm_cell_14/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_14/add_1
while/lstm_cell_14/Sigmoid_2Sigmoid!while/lstm_cell_14/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_14/Sigmoid_2
while/lstm_cell_14/Relu_1Reluwhile/lstm_cell_14/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_14/Relu_1¹
while/lstm_cell_14/mul_2Mul while/lstm_cell_14/Sigmoid_2:y:0'while/lstm_cell_14/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_14/mul_2à
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
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_14/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_14/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/Identity_5Þ

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
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2V
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
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
°?
Ô
while_body_26077938
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_14_matmul_readvariableop_resource_0:	] I
5while_lstm_cell_14_matmul_1_readvariableop_resource_0:
 C
4while_lstm_cell_14_biasadd_readvariableop_resource_0:	 
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_14_matmul_readvariableop_resource:	] G
3while_lstm_cell_14_matmul_1_readvariableop_resource:
 A
2while_lstm_cell_14_biasadd_readvariableop_resource:	 ¢)while/lstm_cell_14/BiasAdd/ReadVariableOp¢(while/lstm_cell_14/MatMul/ReadVariableOp¢*while/lstm_cell_14/MatMul_1/ReadVariableOpÃ
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
(while/lstm_cell_14/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_14_matmul_readvariableop_resource_0*
_output_shapes
:	] *
dtype02*
(while/lstm_cell_14/MatMul/ReadVariableOp×
while/lstm_cell_14/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_14/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/lstm_cell_14/MatMulÐ
*while/lstm_cell_14/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_14_matmul_1_readvariableop_resource_0* 
_output_shapes
:
 *
dtype02,
*while/lstm_cell_14/MatMul_1/ReadVariableOpÀ
while/lstm_cell_14/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_14/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/lstm_cell_14/MatMul_1¸
while/lstm_cell_14/addAddV2#while/lstm_cell_14/MatMul:product:0%while/lstm_cell_14/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/lstm_cell_14/addÈ
)while/lstm_cell_14/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_14_biasadd_readvariableop_resource_0*
_output_shapes	
: *
dtype02+
)while/lstm_cell_14/BiasAdd/ReadVariableOpÅ
while/lstm_cell_14/BiasAddBiasAddwhile/lstm_cell_14/add:z:01while/lstm_cell_14/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
while/lstm_cell_14/BiasAdd
"while/lstm_cell_14/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_14/split/split_dim
while/lstm_cell_14/splitSplit+while/lstm_cell_14/split/split_dim:output:0#while/lstm_cell_14/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2
while/lstm_cell_14/split
while/lstm_cell_14/SigmoidSigmoid!while/lstm_cell_14/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_14/Sigmoid
while/lstm_cell_14/Sigmoid_1Sigmoid!while/lstm_cell_14/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_14/Sigmoid_1¡
while/lstm_cell_14/mulMul while/lstm_cell_14/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_14/mul
while/lstm_cell_14/ReluRelu!while/lstm_cell_14/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_14/Reluµ
while/lstm_cell_14/mul_1Mulwhile/lstm_cell_14/Sigmoid:y:0%while/lstm_cell_14/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_14/mul_1ª
while/lstm_cell_14/add_1AddV2while/lstm_cell_14/mul:z:0while/lstm_cell_14/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_14/add_1
while/lstm_cell_14/Sigmoid_2Sigmoid!while/lstm_cell_14/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_14/Sigmoid_2
while/lstm_cell_14/Relu_1Reluwhile/lstm_cell_14/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_14/Relu_1¹
while/lstm_cell_14/mul_2Mul while/lstm_cell_14/Sigmoid_2:y:0'while/lstm_cell_14/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_14/mul_2à
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
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_14/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_14/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/Identity_5Þ

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
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2V
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
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
½
ù
/__inference_lstm_cell_15_layer_call_fn_26079457

inputs
states_0
states_1
unknown:
È
	unknown_0:	rÈ
	unknown_1:	È
identity

identity_1

identity_2¢StatefulPartitionedCallÅ
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿr:ÿÿÿÿÿÿÿÿÿr:ÿÿÿÿÿÿÿÿÿr*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_lstm_cell_15_layer_call_and_return_conditional_losses_260757952
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2

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
@:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿr:ÿÿÿÿÿÿÿÿÿr: : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
"
_user_specified_name
states/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
"
_user_specified_name
states/1
ß
Í
while_cond_26075662
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_26075662___redundant_placeholder06
2while_while_cond_26075662___redundant_placeholder16
2while_while_cond_26075662___redundant_placeholder26
2while_while_cond_26075662___redundant_placeholder3
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
@: : : : :ÿÿÿÿÿÿÿÿÿr:ÿÿÿÿÿÿÿÿÿr: ::::: 

_output_shapes
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
:ÿÿÿÿÿÿÿÿÿr:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr:

_output_shapes
: :

_output_shapes
:
]
õ
(sequential_7_lstm_15_while_body_26074832F
Bsequential_7_lstm_15_while_sequential_7_lstm_15_while_loop_counterL
Hsequential_7_lstm_15_while_sequential_7_lstm_15_while_maximum_iterations*
&sequential_7_lstm_15_while_placeholder,
(sequential_7_lstm_15_while_placeholder_1,
(sequential_7_lstm_15_while_placeholder_2,
(sequential_7_lstm_15_while_placeholder_3E
Asequential_7_lstm_15_while_sequential_7_lstm_15_strided_slice_1_0
}sequential_7_lstm_15_while_tensorarrayv2read_tensorlistgetitem_sequential_7_lstm_15_tensorarrayunstack_tensorlistfromtensor_0\
Hsequential_7_lstm_15_while_lstm_cell_15_matmul_readvariableop_resource_0:
È]
Jsequential_7_lstm_15_while_lstm_cell_15_matmul_1_readvariableop_resource_0:	rÈX
Isequential_7_lstm_15_while_lstm_cell_15_biasadd_readvariableop_resource_0:	È'
#sequential_7_lstm_15_while_identity)
%sequential_7_lstm_15_while_identity_1)
%sequential_7_lstm_15_while_identity_2)
%sequential_7_lstm_15_while_identity_3)
%sequential_7_lstm_15_while_identity_4)
%sequential_7_lstm_15_while_identity_5C
?sequential_7_lstm_15_while_sequential_7_lstm_15_strided_slice_1
{sequential_7_lstm_15_while_tensorarrayv2read_tensorlistgetitem_sequential_7_lstm_15_tensorarrayunstack_tensorlistfromtensorZ
Fsequential_7_lstm_15_while_lstm_cell_15_matmul_readvariableop_resource:
È[
Hsequential_7_lstm_15_while_lstm_cell_15_matmul_1_readvariableop_resource:	rÈV
Gsequential_7_lstm_15_while_lstm_cell_15_biasadd_readvariableop_resource:	È¢>sequential_7/lstm_15/while/lstm_cell_15/BiasAdd/ReadVariableOp¢=sequential_7/lstm_15/while/lstm_cell_15/MatMul/ReadVariableOp¢?sequential_7/lstm_15/while/lstm_cell_15/MatMul_1/ReadVariableOpí
Lsequential_7/lstm_15/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ  2N
Lsequential_7/lstm_15/while/TensorArrayV2Read/TensorListGetItem/element_shapeÒ
>sequential_7/lstm_15/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem}sequential_7_lstm_15_while_tensorarrayv2read_tensorlistgetitem_sequential_7_lstm_15_tensorarrayunstack_tensorlistfromtensor_0&sequential_7_lstm_15_while_placeholderUsequential_7/lstm_15/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype02@
>sequential_7/lstm_15/while/TensorArrayV2Read/TensorListGetItem
=sequential_7/lstm_15/while/lstm_cell_15/MatMul/ReadVariableOpReadVariableOpHsequential_7_lstm_15_while_lstm_cell_15_matmul_readvariableop_resource_0* 
_output_shapes
:
È*
dtype02?
=sequential_7/lstm_15/while/lstm_cell_15/MatMul/ReadVariableOp«
.sequential_7/lstm_15/while/lstm_cell_15/MatMulMatMulEsequential_7/lstm_15/while/TensorArrayV2Read/TensorListGetItem:item:0Esequential_7/lstm_15/while/lstm_cell_15/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ20
.sequential_7/lstm_15/while/lstm_cell_15/MatMul
?sequential_7/lstm_15/while/lstm_cell_15/MatMul_1/ReadVariableOpReadVariableOpJsequential_7_lstm_15_while_lstm_cell_15_matmul_1_readvariableop_resource_0*
_output_shapes
:	rÈ*
dtype02A
?sequential_7/lstm_15/while/lstm_cell_15/MatMul_1/ReadVariableOp
0sequential_7/lstm_15/while/lstm_cell_15/MatMul_1MatMul(sequential_7_lstm_15_while_placeholder_2Gsequential_7/lstm_15/while/lstm_cell_15/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ22
0sequential_7/lstm_15/while/lstm_cell_15/MatMul_1
+sequential_7/lstm_15/while/lstm_cell_15/addAddV28sequential_7/lstm_15/while/lstm_cell_15/MatMul:product:0:sequential_7/lstm_15/while/lstm_cell_15/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2-
+sequential_7/lstm_15/while/lstm_cell_15/add
>sequential_7/lstm_15/while/lstm_cell_15/BiasAdd/ReadVariableOpReadVariableOpIsequential_7_lstm_15_while_lstm_cell_15_biasadd_readvariableop_resource_0*
_output_shapes	
:È*
dtype02@
>sequential_7/lstm_15/while/lstm_cell_15/BiasAdd/ReadVariableOp
/sequential_7/lstm_15/while/lstm_cell_15/BiasAddBiasAdd/sequential_7/lstm_15/while/lstm_cell_15/add:z:0Fsequential_7/lstm_15/while/lstm_cell_15/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ21
/sequential_7/lstm_15/while/lstm_cell_15/BiasAdd´
7sequential_7/lstm_15/while/lstm_cell_15/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :29
7sequential_7/lstm_15/while/lstm_cell_15/split/split_dimß
-sequential_7/lstm_15/while/lstm_cell_15/splitSplit@sequential_7/lstm_15/while/lstm_cell_15/split/split_dim:output:08sequential_7/lstm_15/while/lstm_cell_15/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿr:ÿÿÿÿÿÿÿÿÿr:ÿÿÿÿÿÿÿÿÿr:ÿÿÿÿÿÿÿÿÿr*
	num_split2/
-sequential_7/lstm_15/while/lstm_cell_15/split×
/sequential_7/lstm_15/while/lstm_cell_15/SigmoidSigmoid6sequential_7/lstm_15/while/lstm_cell_15/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr21
/sequential_7/lstm_15/while/lstm_cell_15/SigmoidÛ
1sequential_7/lstm_15/while/lstm_cell_15/Sigmoid_1Sigmoid6sequential_7/lstm_15/while/lstm_cell_15/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr23
1sequential_7/lstm_15/while/lstm_cell_15/Sigmoid_1ô
+sequential_7/lstm_15/while/lstm_cell_15/mulMul5sequential_7/lstm_15/while/lstm_cell_15/Sigmoid_1:y:0(sequential_7_lstm_15_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2-
+sequential_7/lstm_15/while/lstm_cell_15/mulÎ
,sequential_7/lstm_15/while/lstm_cell_15/ReluRelu6sequential_7/lstm_15/while/lstm_cell_15/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2.
,sequential_7/lstm_15/while/lstm_cell_15/Relu
-sequential_7/lstm_15/while/lstm_cell_15/mul_1Mul3sequential_7/lstm_15/while/lstm_cell_15/Sigmoid:y:0:sequential_7/lstm_15/while/lstm_cell_15/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2/
-sequential_7/lstm_15/while/lstm_cell_15/mul_1ý
-sequential_7/lstm_15/while/lstm_cell_15/add_1AddV2/sequential_7/lstm_15/while/lstm_cell_15/mul:z:01sequential_7/lstm_15/while/lstm_cell_15/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2/
-sequential_7/lstm_15/while/lstm_cell_15/add_1Û
1sequential_7/lstm_15/while/lstm_cell_15/Sigmoid_2Sigmoid6sequential_7/lstm_15/while/lstm_cell_15/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr23
1sequential_7/lstm_15/while/lstm_cell_15/Sigmoid_2Í
.sequential_7/lstm_15/while/lstm_cell_15/Relu_1Relu1sequential_7/lstm_15/while/lstm_cell_15/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr20
.sequential_7/lstm_15/while/lstm_cell_15/Relu_1
-sequential_7/lstm_15/while/lstm_cell_15/mul_2Mul5sequential_7/lstm_15/while/lstm_cell_15/Sigmoid_2:y:0<sequential_7/lstm_15/while/lstm_cell_15/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2/
-sequential_7/lstm_15/while/lstm_cell_15/mul_2É
?sequential_7/lstm_15/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem(sequential_7_lstm_15_while_placeholder_1&sequential_7_lstm_15_while_placeholder1sequential_7/lstm_15/while/lstm_cell_15/mul_2:z:0*
_output_shapes
: *
element_dtype02A
?sequential_7/lstm_15/while/TensorArrayV2Write/TensorListSetItem
 sequential_7/lstm_15/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2"
 sequential_7/lstm_15/while/add/y½
sequential_7/lstm_15/while/addAddV2&sequential_7_lstm_15_while_placeholder)sequential_7/lstm_15/while/add/y:output:0*
T0*
_output_shapes
: 2 
sequential_7/lstm_15/while/add
"sequential_7/lstm_15/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2$
"sequential_7/lstm_15/while/add_1/yß
 sequential_7/lstm_15/while/add_1AddV2Bsequential_7_lstm_15_while_sequential_7_lstm_15_while_loop_counter+sequential_7/lstm_15/while/add_1/y:output:0*
T0*
_output_shapes
: 2"
 sequential_7/lstm_15/while/add_1¿
#sequential_7/lstm_15/while/IdentityIdentity$sequential_7/lstm_15/while/add_1:z:0 ^sequential_7/lstm_15/while/NoOp*
T0*
_output_shapes
: 2%
#sequential_7/lstm_15/while/Identityç
%sequential_7/lstm_15/while/Identity_1IdentityHsequential_7_lstm_15_while_sequential_7_lstm_15_while_maximum_iterations ^sequential_7/lstm_15/while/NoOp*
T0*
_output_shapes
: 2'
%sequential_7/lstm_15/while/Identity_1Á
%sequential_7/lstm_15/while/Identity_2Identity"sequential_7/lstm_15/while/add:z:0 ^sequential_7/lstm_15/while/NoOp*
T0*
_output_shapes
: 2'
%sequential_7/lstm_15/while/Identity_2î
%sequential_7/lstm_15/while/Identity_3IdentityOsequential_7/lstm_15/while/TensorArrayV2Write/TensorListSetItem:output_handle:0 ^sequential_7/lstm_15/while/NoOp*
T0*
_output_shapes
: 2'
%sequential_7/lstm_15/while/Identity_3á
%sequential_7/lstm_15/while/Identity_4Identity1sequential_7/lstm_15/while/lstm_cell_15/mul_2:z:0 ^sequential_7/lstm_15/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2'
%sequential_7/lstm_15/while/Identity_4á
%sequential_7/lstm_15/while/Identity_5Identity1sequential_7/lstm_15/while/lstm_cell_15/add_1:z:0 ^sequential_7/lstm_15/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr2'
%sequential_7/lstm_15/while/Identity_5Ç
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
%sequential_7_lstm_15_while_identity_5.sequential_7/lstm_15/while/Identity_5:output:0"
Gsequential_7_lstm_15_while_lstm_cell_15_biasadd_readvariableop_resourceIsequential_7_lstm_15_while_lstm_cell_15_biasadd_readvariableop_resource_0"
Hsequential_7_lstm_15_while_lstm_cell_15_matmul_1_readvariableop_resourceJsequential_7_lstm_15_while_lstm_cell_15_matmul_1_readvariableop_resource_0"
Fsequential_7_lstm_15_while_lstm_cell_15_matmul_readvariableop_resourceHsequential_7_lstm_15_while_lstm_cell_15_matmul_readvariableop_resource_0"
?sequential_7_lstm_15_while_sequential_7_lstm_15_strided_slice_1Asequential_7_lstm_15_while_sequential_7_lstm_15_strided_slice_1_0"ü
{sequential_7_lstm_15_while_tensorarrayv2read_tensorlistgetitem_sequential_7_lstm_15_tensorarrayunstack_tensorlistfromtensor}sequential_7_lstm_15_while_tensorarrayv2read_tensorlistgetitem_sequential_7_lstm_15_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿr:ÿÿÿÿÿÿÿÿÿr: : : : : 2
>sequential_7/lstm_15/while/lstm_cell_15/BiasAdd/ReadVariableOp>sequential_7/lstm_15/while/lstm_cell_15/BiasAdd/ReadVariableOp2~
=sequential_7/lstm_15/while/lstm_cell_15/MatMul/ReadVariableOp=sequential_7/lstm_15/while/lstm_cell_15/MatMul/ReadVariableOp2
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
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr:

_output_shapes
: :

_output_shapes
: 
åJ
Ô

lstm_14_while_body_26077555,
(lstm_14_while_lstm_14_while_loop_counter2
.lstm_14_while_lstm_14_while_maximum_iterations
lstm_14_while_placeholder
lstm_14_while_placeholder_1
lstm_14_while_placeholder_2
lstm_14_while_placeholder_3+
'lstm_14_while_lstm_14_strided_slice_1_0g
clstm_14_while_tensorarrayv2read_tensorlistgetitem_lstm_14_tensorarrayunstack_tensorlistfromtensor_0N
;lstm_14_while_lstm_cell_14_matmul_readvariableop_resource_0:	] Q
=lstm_14_while_lstm_cell_14_matmul_1_readvariableop_resource_0:
 K
<lstm_14_while_lstm_cell_14_biasadd_readvariableop_resource_0:	 
lstm_14_while_identity
lstm_14_while_identity_1
lstm_14_while_identity_2
lstm_14_while_identity_3
lstm_14_while_identity_4
lstm_14_while_identity_5)
%lstm_14_while_lstm_14_strided_slice_1e
alstm_14_while_tensorarrayv2read_tensorlistgetitem_lstm_14_tensorarrayunstack_tensorlistfromtensorL
9lstm_14_while_lstm_cell_14_matmul_readvariableop_resource:	] O
;lstm_14_while_lstm_cell_14_matmul_1_readvariableop_resource:
 I
:lstm_14_while_lstm_cell_14_biasadd_readvariableop_resource:	 ¢1lstm_14/while/lstm_cell_14/BiasAdd/ReadVariableOp¢0lstm_14/while/lstm_cell_14/MatMul/ReadVariableOp¢2lstm_14/while/lstm_cell_14/MatMul_1/ReadVariableOpÓ
?lstm_14/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ]   2A
?lstm_14/while/TensorArrayV2Read/TensorListGetItem/element_shape
1lstm_14/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemclstm_14_while_tensorarrayv2read_tensorlistgetitem_lstm_14_tensorarrayunstack_tensorlistfromtensor_0lstm_14_while_placeholderHlstm_14/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]*
element_dtype023
1lstm_14/while/TensorArrayV2Read/TensorListGetItemá
0lstm_14/while/lstm_cell_14/MatMul/ReadVariableOpReadVariableOp;lstm_14_while_lstm_cell_14_matmul_readvariableop_resource_0*
_output_shapes
:	] *
dtype022
0lstm_14/while/lstm_cell_14/MatMul/ReadVariableOp÷
!lstm_14/while/lstm_cell_14/MatMulMatMul8lstm_14/while/TensorArrayV2Read/TensorListGetItem:item:08lstm_14/while/lstm_cell_14/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!lstm_14/while/lstm_cell_14/MatMulè
2lstm_14/while/lstm_cell_14/MatMul_1/ReadVariableOpReadVariableOp=lstm_14_while_lstm_cell_14_matmul_1_readvariableop_resource_0* 
_output_shapes
:
 *
dtype024
2lstm_14/while/lstm_cell_14/MatMul_1/ReadVariableOpà
#lstm_14/while/lstm_cell_14/MatMul_1MatMullstm_14_while_placeholder_2:lstm_14/while/lstm_cell_14/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2%
#lstm_14/while/lstm_cell_14/MatMul_1Ø
lstm_14/while/lstm_cell_14/addAddV2+lstm_14/while/lstm_cell_14/MatMul:product:0-lstm_14/while/lstm_cell_14/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2 
lstm_14/while/lstm_cell_14/addà
1lstm_14/while/lstm_cell_14/BiasAdd/ReadVariableOpReadVariableOp<lstm_14_while_lstm_cell_14_biasadd_readvariableop_resource_0*
_output_shapes	
: *
dtype023
1lstm_14/while/lstm_cell_14/BiasAdd/ReadVariableOpå
"lstm_14/while/lstm_cell_14/BiasAddBiasAdd"lstm_14/while/lstm_cell_14/add:z:09lstm_14/while/lstm_cell_14/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2$
"lstm_14/while/lstm_cell_14/BiasAdd
*lstm_14/while/lstm_cell_14/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2,
*lstm_14/while/lstm_cell_14/split/split_dim¯
 lstm_14/while/lstm_cell_14/splitSplit3lstm_14/while/lstm_cell_14/split/split_dim:output:0+lstm_14/while/lstm_cell_14/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2"
 lstm_14/while/lstm_cell_14/split±
"lstm_14/while/lstm_cell_14/SigmoidSigmoid)lstm_14/while/lstm_cell_14/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2$
"lstm_14/while/lstm_cell_14/Sigmoidµ
$lstm_14/while/lstm_cell_14/Sigmoid_1Sigmoid)lstm_14/while/lstm_cell_14/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$lstm_14/while/lstm_cell_14/Sigmoid_1Á
lstm_14/while/lstm_cell_14/mulMul(lstm_14/while/lstm_cell_14/Sigmoid_1:y:0lstm_14_while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
lstm_14/while/lstm_cell_14/mul¨
lstm_14/while/lstm_cell_14/ReluRelu)lstm_14/while/lstm_cell_14/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
lstm_14/while/lstm_cell_14/ReluÕ
 lstm_14/while/lstm_cell_14/mul_1Mul&lstm_14/while/lstm_cell_14/Sigmoid:y:0-lstm_14/while/lstm_cell_14/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 lstm_14/while/lstm_cell_14/mul_1Ê
 lstm_14/while/lstm_cell_14/add_1AddV2"lstm_14/while/lstm_cell_14/mul:z:0$lstm_14/while/lstm_cell_14/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 lstm_14/while/lstm_cell_14/add_1µ
$lstm_14/while/lstm_cell_14/Sigmoid_2Sigmoid)lstm_14/while/lstm_cell_14/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$lstm_14/while/lstm_cell_14/Sigmoid_2§
!lstm_14/while/lstm_cell_14/Relu_1Relu$lstm_14/while/lstm_cell_14/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!lstm_14/while/lstm_cell_14/Relu_1Ù
 lstm_14/while/lstm_cell_14/mul_2Mul(lstm_14/while/lstm_cell_14/Sigmoid_2:y:0/lstm_14/while/lstm_cell_14/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 lstm_14/while/lstm_cell_14/mul_2
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
lstm_14/while/add/y
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
lstm_14/while/add_1/y
lstm_14/while/add_1AddV2(lstm_14_while_lstm_14_while_loop_counterlstm_14/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lstm_14/while/add_1
lstm_14/while/IdentityIdentitylstm_14/while/add_1:z:0^lstm_14/while/NoOp*
T0*
_output_shapes
: 2
lstm_14/while/Identity¦
lstm_14/while/Identity_1Identity.lstm_14_while_lstm_14_while_maximum_iterations^lstm_14/while/NoOp*
T0*
_output_shapes
: 2
lstm_14/while/Identity_1
lstm_14/while/Identity_2Identitylstm_14/while/add:z:0^lstm_14/while/NoOp*
T0*
_output_shapes
: 2
lstm_14/while/Identity_2º
lstm_14/while/Identity_3IdentityBlstm_14/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_14/while/NoOp*
T0*
_output_shapes
: 2
lstm_14/while/Identity_3®
lstm_14/while/Identity_4Identity$lstm_14/while/lstm_cell_14/mul_2:z:0^lstm_14/while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_14/while/Identity_4®
lstm_14/while/Identity_5Identity$lstm_14/while/lstm_cell_14/add_1:z:0^lstm_14/while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_14/while/Identity_5
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
9lstm_14_while_lstm_cell_14_matmul_readvariableop_resource;lstm_14_while_lstm_cell_14_matmul_readvariableop_resource_0"È
alstm_14_while_tensorarrayv2read_tensorlistgetitem_lstm_14_tensorarrayunstack_tensorlistfromtensorclstm_14_while_tensorarrayv2read_tensorlistgetitem_lstm_14_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2f
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
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
ËF

E__inference_lstm_14_layer_call_and_return_conditional_losses_26075102

inputs(
lstm_cell_14_26075020:	] )
lstm_cell_14_26075022:
 $
lstm_cell_14_26075024:	 
identity¢$lstm_cell_14/StatefulPartitionedCall¢whileD
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
B :2
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
B :2
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
:ÿÿÿÿÿÿÿÿÿ2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :2
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
B :2
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
:ÿÿÿÿÿÿÿÿÿ2	
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
$lstm_cell_14/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_14_26075020lstm_cell_14_26075022lstm_cell_14_26075024*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_lstm_cell_14_layer_call_and_return_conditional_losses_260750192&
$lstm_cell_14/StatefulPartitionedCall
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ  2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_14_26075020lstm_cell_14_26075022lstm_cell_14_26075024*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_26075033*
condR
while_cond_26075032*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ  22
0TensorArrayV2Stack/TensorListStack/element_shapeò
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
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
:ÿÿÿÿÿÿÿÿÿ*
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
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
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
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity}
NoOpNoOp%^lstm_cell_14/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ]: : : 2L
$lstm_cell_14/StatefulPartitionedCall$lstm_cell_14/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ]
 
_user_specified_nameinputs
ß
Í
while_cond_26078763
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_26078763___redundant_placeholder06
2while_while_cond_26078763___redundant_placeholder16
2while_while_cond_26078763___redundant_placeholder26
2while_while_cond_26078763___redundant_placeholder3
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
@: : : : :ÿÿÿÿÿÿÿÿÿr:ÿÿÿÿÿÿÿÿÿr: ::::: 

_output_shapes
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
:ÿÿÿÿÿÿÿÿÿr:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr:

_output_shapes
: :

_output_shapes
:
à
º
*__inference_lstm_14_layer_call_fn_26078497
inputs_0
unknown:	] 
	unknown_0:
 
	unknown_1:	 
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_lstm_14_layer_call_and_return_conditional_losses_260753122
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

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


J__inference_lstm_cell_14_layer_call_and_return_conditional_losses_26075165

inputs

states
states_11
matmul_readvariableop_resource:	] 4
 matmul_1_readvariableop_resource:
 .
biasadd_readvariableop_resource:	 
identity

identity_1

identity_2¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	] *
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
MatMul
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
 *
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
add
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
: *
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2	
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
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2
split`
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
Sigmoidd
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	Sigmoid_1]
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mulW
ReluRelusplit:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relui
mul_1MulSigmoid:y:0Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mul_1^
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
add_1d
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	Sigmoid_2V
Relu_1Relu	add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu_1m
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mul_2e
IdentityIdentity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityi

Identity_1Identity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1i

Identity_2Identity	add_1:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

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
A:ÿÿÿÿÿÿÿÿÿ]:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
 
_user_specified_nameinputs:PL
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_namestates:PL
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_namestates"¨L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*¾
serving_defaultª
K
lstm_14_input:
serving_default_lstm_14_input:0ÿÿÿÿÿÿÿÿÿ]?
dense_74
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:È²
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
k_default_save_signature
*l&call_and_return_all_conditional_losses
m__call__"
_tf_keras_sequential
Ã
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
¥
trainable_variables
regularization_losses
	variables
	keras_api
*p&call_and_return_all_conditional_losses
q__call__"
_tf_keras_layer
Ã
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
¥
trainable_variables
regularization_losses
	variables
	keras_api
*t&call_and_return_all_conditional_losses
u__call__"
_tf_keras_layer
»

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
Ê
,layer_metrics
trainable_variables

-layers
.layer_regularization_losses
/metrics
regularization_losses
0non_trainable_variables
		variables
m__call__
k_default_save_signature
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
¹
6layer_metrics

7states
trainable_variables

8layers
9layer_regularization_losses
:metrics
regularization_losses
;non_trainable_variables
	variables
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
­
<layer_metrics
trainable_variables

=layers
>layer_regularization_losses
?metrics
regularization_losses
@non_trainable_variables
	variables
q__call__
*p&call_and_return_all_conditional_losses
&p"call_and_return_conditional_losses"
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
¹
Flayer_metrics

Gstates
trainable_variables

Hlayers
Ilayer_regularization_losses
Jmetrics
regularization_losses
Knon_trainable_variables
	variables
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
­
Llayer_metrics
trainable_variables

Mlayers
Nlayer_regularization_losses
Ometrics
regularization_losses
Pnon_trainable_variables
	variables
u__call__
*t&call_and_return_all_conditional_losses
&t"call_and_return_conditional_losses"
_generic_user_object
 :r2dense_7/kernel
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
­
Qlayer_metrics
"trainable_variables

Rlayers
Slayer_regularization_losses
Tmetrics
#regularization_losses
Unon_trainable_variables
$	variables
w__call__
*v&call_and_return_all_conditional_losses
&v"call_and_return_conditional_losses"
_generic_user_object
.:,	] 2lstm_14/lstm_cell_14/kernel
9:7
 2%lstm_14/lstm_cell_14/recurrent_kernel
(:& 2lstm_14/lstm_cell_14/bias
/:-
È2lstm_15/lstm_cell_15/kernel
8:6	rÈ2%lstm_15/lstm_cell_15/recurrent_kernel
(:&È2lstm_15/lstm_cell_15/bias
 "
trackable_dict_wrapper
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
Xlayer_metrics
2trainable_variables

Ylayers
Zlayer_regularization_losses
[metrics
3regularization_losses
\non_trainable_variables
4	variables
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
 "
trackable_list_wrapper
5
)0
*1
+2"
trackable_list_wrapper
­
]layer_metrics
Btrainable_variables

^layers
_layer_regularization_losses
`metrics
Cregularization_losses
anon_trainable_variables
D	variables
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
ÔBÑ
#__inference__wrapped_model_26074944lstm_14_input"
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
ö2ó
J__inference_sequential_7_layer_call_and_return_conditional_losses_26077488
J__inference_sequential_7_layer_call_and_return_conditional_losses_26077829
J__inference_sequential_7_layer_call_and_return_conditional_losses_26077113
J__inference_sequential_7_layer_call_and_return_conditional_losses_26077138À
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
2
/__inference_sequential_7_layer_call_fn_26076598
/__inference_sequential_7_layer_call_fn_26077850
/__inference_sequential_7_layer_call_fn_26077871
/__inference_sequential_7_layer_call_fn_26077088À
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
÷2ô
E__inference_lstm_14_layer_call_and_return_conditional_losses_26078022
E__inference_lstm_14_layer_call_and_return_conditional_losses_26078173
E__inference_lstm_14_layer_call_and_return_conditional_losses_26078324
E__inference_lstm_14_layer_call_and_return_conditional_losses_26078475Õ
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
2
*__inference_lstm_14_layer_call_fn_26078486
*__inference_lstm_14_layer_call_fn_26078497
*__inference_lstm_14_layer_call_fn_26078508
*__inference_lstm_14_layer_call_fn_26078519Õ
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
Î2Ë
H__inference_dropout_14_layer_call_and_return_conditional_losses_26078524
H__inference_dropout_14_layer_call_and_return_conditional_losses_26078536´
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
2
-__inference_dropout_14_layer_call_fn_26078541
-__inference_dropout_14_layer_call_fn_26078546´
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
÷2ô
E__inference_lstm_15_layer_call_and_return_conditional_losses_26078697
E__inference_lstm_15_layer_call_and_return_conditional_losses_26078848
E__inference_lstm_15_layer_call_and_return_conditional_losses_26078999
E__inference_lstm_15_layer_call_and_return_conditional_losses_26079150Õ
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
2
*__inference_lstm_15_layer_call_fn_26079161
*__inference_lstm_15_layer_call_fn_26079172
*__inference_lstm_15_layer_call_fn_26079183
*__inference_lstm_15_layer_call_fn_26079194Õ
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
Î2Ë
H__inference_dropout_15_layer_call_and_return_conditional_losses_26079199
H__inference_dropout_15_layer_call_and_return_conditional_losses_26079211´
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
2
-__inference_dropout_15_layer_call_fn_26079216
-__inference_dropout_15_layer_call_fn_26079221´
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
ï2ì
E__inference_dense_7_layer_call_and_return_conditional_losses_26079252¢
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
Ô2Ñ
*__inference_dense_7_layer_call_fn_26079261¢
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
&__inference_signature_wrapper_26077161lstm_14_input"
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
Ü2Ù
J__inference_lstm_cell_14_layer_call_and_return_conditional_losses_26079293
J__inference_lstm_cell_14_layer_call_and_return_conditional_losses_26079325¾
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
/__inference_lstm_cell_14_layer_call_fn_26079342
/__inference_lstm_cell_14_layer_call_fn_26079359¾
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
J__inference_lstm_cell_15_layer_call_and_return_conditional_losses_26079391
J__inference_lstm_cell_15_layer_call_and_return_conditional_losses_26079423¾
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
/__inference_lstm_cell_15_layer_call_fn_26079440
/__inference_lstm_cell_15_layer_call_fn_26079457¾
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
 ¤
#__inference__wrapped_model_26074944}&'()*+ !:¢7
0¢-
+(
lstm_14_inputÿÿÿÿÿÿÿÿÿ]
ª "5ª2
0
dense_7%"
dense_7ÿÿÿÿÿÿÿÿÿ­
E__inference_dense_7_layer_call_and_return_conditional_losses_26079252d !3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿr
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ
 
*__inference_dense_7_layer_call_fn_26079261W !3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿr
ª "ÿÿÿÿÿÿÿÿÿ²
H__inference_dropout_14_layer_call_and_return_conditional_losses_26078524f8¢5
.¢+
%"
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "*¢'
 
0ÿÿÿÿÿÿÿÿÿ
 ²
H__inference_dropout_14_layer_call_and_return_conditional_losses_26078536f8¢5
.¢+
%"
inputsÿÿÿÿÿÿÿÿÿ
p
ª "*¢'
 
0ÿÿÿÿÿÿÿÿÿ
 
-__inference_dropout_14_layer_call_fn_26078541Y8¢5
.¢+
%"
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "ÿÿÿÿÿÿÿÿÿ
-__inference_dropout_14_layer_call_fn_26078546Y8¢5
.¢+
%"
inputsÿÿÿÿÿÿÿÿÿ
p
ª "ÿÿÿÿÿÿÿÿÿ°
H__inference_dropout_15_layer_call_and_return_conditional_losses_26079199d7¢4
-¢*
$!
inputsÿÿÿÿÿÿÿÿÿr
p 
ª ")¢&

0ÿÿÿÿÿÿÿÿÿr
 °
H__inference_dropout_15_layer_call_and_return_conditional_losses_26079211d7¢4
-¢*
$!
inputsÿÿÿÿÿÿÿÿÿr
p
ª ")¢&

0ÿÿÿÿÿÿÿÿÿr
 
-__inference_dropout_15_layer_call_fn_26079216W7¢4
-¢*
$!
inputsÿÿÿÿÿÿÿÿÿr
p 
ª "ÿÿÿÿÿÿÿÿÿr
-__inference_dropout_15_layer_call_fn_26079221W7¢4
-¢*
$!
inputsÿÿÿÿÿÿÿÿÿr
p
ª "ÿÿÿÿÿÿÿÿÿrÕ
E__inference_lstm_14_layer_call_and_return_conditional_losses_26078022&'(O¢L
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
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Õ
E__inference_lstm_14_layer_call_and_return_conditional_losses_26078173&'(O¢L
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
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 »
E__inference_lstm_14_layer_call_and_return_conditional_losses_26078324r&'(?¢<
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
0ÿÿÿÿÿÿÿÿÿ
 »
E__inference_lstm_14_layer_call_and_return_conditional_losses_26078475r&'(?¢<
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
0ÿÿÿÿÿÿÿÿÿ
 ¬
*__inference_lstm_14_layer_call_fn_26078486~&'(O¢L
E¢B
41
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ]

 
p 

 
ª "&#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬
*__inference_lstm_14_layer_call_fn_26078497~&'(O¢L
E¢B
41
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ]

 
p

 
ª "&#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
*__inference_lstm_14_layer_call_fn_26078508e&'(?¢<
5¢2
$!
inputsÿÿÿÿÿÿÿÿÿ]

 
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
*__inference_lstm_14_layer_call_fn_26078519e&'(?¢<
5¢2
$!
inputsÿÿÿÿÿÿÿÿÿ]

 
p

 
ª "ÿÿÿÿÿÿÿÿÿÕ
E__inference_lstm_15_layer_call_and_return_conditional_losses_26078697)*+P¢M
F¢C
52
0-
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
p 

 
ª "2¢/
(%
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿr
 Õ
E__inference_lstm_15_layer_call_and_return_conditional_losses_26078848)*+P¢M
F¢C
52
0-
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
p

 
ª "2¢/
(%
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿr
 »
E__inference_lstm_15_layer_call_and_return_conditional_losses_26078999r)*+@¢=
6¢3
%"
inputsÿÿÿÿÿÿÿÿÿ

 
p 

 
ª ")¢&

0ÿÿÿÿÿÿÿÿÿr
 »
E__inference_lstm_15_layer_call_and_return_conditional_losses_26079150r)*+@¢=
6¢3
%"
inputsÿÿÿÿÿÿÿÿÿ

 
p

 
ª ")¢&

0ÿÿÿÿÿÿÿÿÿr
 ¬
*__inference_lstm_15_layer_call_fn_26079161~)*+P¢M
F¢C
52
0-
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
p 

 
ª "%"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿr¬
*__inference_lstm_15_layer_call_fn_26079172~)*+P¢M
F¢C
52
0-
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
p

 
ª "%"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿr
*__inference_lstm_15_layer_call_fn_26079183e)*+@¢=
6¢3
%"
inputsÿÿÿÿÿÿÿÿÿ

 
p 

 
ª "ÿÿÿÿÿÿÿÿÿr
*__inference_lstm_15_layer_call_fn_26079194e)*+@¢=
6¢3
%"
inputsÿÿÿÿÿÿÿÿÿ

 
p

 
ª "ÿÿÿÿÿÿÿÿÿrÑ
J__inference_lstm_cell_14_layer_call_and_return_conditional_losses_26079293&'(¢
x¢u
 
inputsÿÿÿÿÿÿÿÿÿ]
M¢J
# 
states/0ÿÿÿÿÿÿÿÿÿ
# 
states/1ÿÿÿÿÿÿÿÿÿ
p 
ª "v¢s
l¢i

0/0ÿÿÿÿÿÿÿÿÿ
GD
 
0/1/0ÿÿÿÿÿÿÿÿÿ
 
0/1/1ÿÿÿÿÿÿÿÿÿ
 Ñ
J__inference_lstm_cell_14_layer_call_and_return_conditional_losses_26079325&'(¢
x¢u
 
inputsÿÿÿÿÿÿÿÿÿ]
M¢J
# 
states/0ÿÿÿÿÿÿÿÿÿ
# 
states/1ÿÿÿÿÿÿÿÿÿ
p
ª "v¢s
l¢i

0/0ÿÿÿÿÿÿÿÿÿ
GD
 
0/1/0ÿÿÿÿÿÿÿÿÿ
 
0/1/1ÿÿÿÿÿÿÿÿÿ
 ¦
/__inference_lstm_cell_14_layer_call_fn_26079342ò&'(¢
x¢u
 
inputsÿÿÿÿÿÿÿÿÿ]
M¢J
# 
states/0ÿÿÿÿÿÿÿÿÿ
# 
states/1ÿÿÿÿÿÿÿÿÿ
p 
ª "f¢c

0ÿÿÿÿÿÿÿÿÿ
C@

1/0ÿÿÿÿÿÿÿÿÿ

1/1ÿÿÿÿÿÿÿÿÿ¦
/__inference_lstm_cell_14_layer_call_fn_26079359ò&'(¢
x¢u
 
inputsÿÿÿÿÿÿÿÿÿ]
M¢J
# 
states/0ÿÿÿÿÿÿÿÿÿ
# 
states/1ÿÿÿÿÿÿÿÿÿ
p
ª "f¢c

0ÿÿÿÿÿÿÿÿÿ
C@

1/0ÿÿÿÿÿÿÿÿÿ

1/1ÿÿÿÿÿÿÿÿÿÍ
J__inference_lstm_cell_15_layer_call_and_return_conditional_losses_26079391þ)*+¢~
w¢t
!
inputsÿÿÿÿÿÿÿÿÿ
K¢H
"
states/0ÿÿÿÿÿÿÿÿÿr
"
states/1ÿÿÿÿÿÿÿÿÿr
p 
ª "s¢p
i¢f

0/0ÿÿÿÿÿÿÿÿÿr
EB

0/1/0ÿÿÿÿÿÿÿÿÿr

0/1/1ÿÿÿÿÿÿÿÿÿr
 Í
J__inference_lstm_cell_15_layer_call_and_return_conditional_losses_26079423þ)*+¢~
w¢t
!
inputsÿÿÿÿÿÿÿÿÿ
K¢H
"
states/0ÿÿÿÿÿÿÿÿÿr
"
states/1ÿÿÿÿÿÿÿÿÿr
p
ª "s¢p
i¢f

0/0ÿÿÿÿÿÿÿÿÿr
EB

0/1/0ÿÿÿÿÿÿÿÿÿr

0/1/1ÿÿÿÿÿÿÿÿÿr
 ¢
/__inference_lstm_cell_15_layer_call_fn_26079440î)*+¢~
w¢t
!
inputsÿÿÿÿÿÿÿÿÿ
K¢H
"
states/0ÿÿÿÿÿÿÿÿÿr
"
states/1ÿÿÿÿÿÿÿÿÿr
p 
ª "c¢`

0ÿÿÿÿÿÿÿÿÿr
A>

1/0ÿÿÿÿÿÿÿÿÿr

1/1ÿÿÿÿÿÿÿÿÿr¢
/__inference_lstm_cell_15_layer_call_fn_26079457î)*+¢~
w¢t
!
inputsÿÿÿÿÿÿÿÿÿ
K¢H
"
states/0ÿÿÿÿÿÿÿÿÿr
"
states/1ÿÿÿÿÿÿÿÿÿr
p
ª "c¢`

0ÿÿÿÿÿÿÿÿÿr
A>

1/0ÿÿÿÿÿÿÿÿÿr

1/1ÿÿÿÿÿÿÿÿÿrÇ
J__inference_sequential_7_layer_call_and_return_conditional_losses_26077113y&'()*+ !B¢?
8¢5
+(
lstm_14_inputÿÿÿÿÿÿÿÿÿ]
p 

 
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ
 Ç
J__inference_sequential_7_layer_call_and_return_conditional_losses_26077138y&'()*+ !B¢?
8¢5
+(
lstm_14_inputÿÿÿÿÿÿÿÿÿ]
p

 
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ
 À
J__inference_sequential_7_layer_call_and_return_conditional_losses_26077488r&'()*+ !;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿ]
p 

 
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ
 À
J__inference_sequential_7_layer_call_and_return_conditional_losses_26077829r&'()*+ !;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿ]
p

 
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ
 
/__inference_sequential_7_layer_call_fn_26076598l&'()*+ !B¢?
8¢5
+(
lstm_14_inputÿÿÿÿÿÿÿÿÿ]
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
/__inference_sequential_7_layer_call_fn_26077088l&'()*+ !B¢?
8¢5
+(
lstm_14_inputÿÿÿÿÿÿÿÿÿ]
p

 
ª "ÿÿÿÿÿÿÿÿÿ
/__inference_sequential_7_layer_call_fn_26077850e&'()*+ !;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿ]
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
/__inference_sequential_7_layer_call_fn_26077871e&'()*+ !;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿ]
p

 
ª "ÿÿÿÿÿÿÿÿÿ¹
&__inference_signature_wrapper_26077161&'()*+ !K¢H
¢ 
Aª>
<
lstm_14_input+(
lstm_14_inputÿÿÿÿÿÿÿÿÿ]"5ª2
0
dense_7%"
dense_7ÿÿÿÿÿÿÿÿÿ