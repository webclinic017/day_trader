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
dense_13/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	æ* 
shared_namedense_13/kernel
t
#dense_13/kernel/Read/ReadVariableOpReadVariableOpdense_13/kernel*
_output_shapes
:	æ*
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

lstm_26/lstm_cell_26/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	]è
*,
shared_namelstm_26/lstm_cell_26/kernel

/lstm_26/lstm_cell_26/kernel/Read/ReadVariableOpReadVariableOplstm_26/lstm_cell_26/kernel*
_output_shapes
:	]è
*
dtype0
¨
%lstm_26/lstm_cell_26/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
Úè
*6
shared_name'%lstm_26/lstm_cell_26/recurrent_kernel
¡
9lstm_26/lstm_cell_26/recurrent_kernel/Read/ReadVariableOpReadVariableOp%lstm_26/lstm_cell_26/recurrent_kernel* 
_output_shapes
:
Úè
*
dtype0

lstm_26/lstm_cell_26/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:è
**
shared_namelstm_26/lstm_cell_26/bias

-lstm_26/lstm_cell_26/bias/Read/ReadVariableOpReadVariableOplstm_26/lstm_cell_26/bias*
_output_shapes	
:è
*
dtype0

lstm_27/lstm_cell_27/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
Ú*,
shared_namelstm_27/lstm_cell_27/kernel

/lstm_27/lstm_cell_27/kernel/Read/ReadVariableOpReadVariableOplstm_27/lstm_cell_27/kernel* 
_output_shapes
:
Ú*
dtype0
¨
%lstm_27/lstm_cell_27/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
æ*6
shared_name'%lstm_27/lstm_cell_27/recurrent_kernel
¡
9lstm_27/lstm_cell_27/recurrent_kernel/Read/ReadVariableOpReadVariableOp%lstm_27/lstm_cell_27/recurrent_kernel* 
_output_shapes
:
æ*
dtype0

lstm_27/lstm_cell_27/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_namelstm_27/lstm_cell_27/bias

-lstm_27/lstm_cell_27/bias/Read/ReadVariableOpReadVariableOplstm_27/lstm_cell_27/bias*
_output_shapes	
:*
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
serving_default_lstm_26_inputPlaceholder*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]*
dtype0* 
shape:ÿÿÿÿÿÿÿÿÿ]
¬
StatefulPartitionedCallStatefulPartitionedCallserving_default_lstm_26_inputlstm_26/lstm_cell_26/kernel%lstm_26/lstm_cell_26/recurrent_kernellstm_26/lstm_cell_26/biaslstm_27/lstm_cell_27/kernel%lstm_27/lstm_cell_27/recurrent_kernellstm_27/lstm_cell_27/biasdense_13/kerneldense_13/bias*
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
&__inference_signature_wrapper_39974499
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 

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
GPU 2J 8 **
f%R#
!__inference__traced_save_39976854
¢
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
GPU 2J 8 *-
f(R&
$__inference__traced_restore_39976900¤ý#
´?
Ö
while_body_39976297
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
3while_lstm_cell_27_matmul_readvariableop_resource_0:
ÚI
5while_lstm_cell_27_matmul_1_readvariableop_resource_0:
æC
4while_lstm_cell_27_biasadd_readvariableop_resource_0:	
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
1while_lstm_cell_27_matmul_readvariableop_resource:
ÚG
3while_lstm_cell_27_matmul_1_readvariableop_resource:
æA
2while_lstm_cell_27_biasadd_readvariableop_resource:	¢)while/lstm_cell_27/BiasAdd/ReadVariableOp¢(while/lstm_cell_27/MatMul/ReadVariableOp¢*while/lstm_cell_27/MatMul_1/ReadVariableOpÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿZ  29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÔ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemÊ
(while/lstm_cell_27/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_27_matmul_readvariableop_resource_0* 
_output_shapes
:
Ú*
dtype02*
(while/lstm_cell_27/MatMul/ReadVariableOp×
while/lstm_cell_27/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_27/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_27/MatMulÐ
*while/lstm_cell_27/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_27_matmul_1_readvariableop_resource_0* 
_output_shapes
:
æ*
dtype02,
*while/lstm_cell_27/MatMul_1/ReadVariableOpÀ
while/lstm_cell_27/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_27/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_27/MatMul_1¸
while/lstm_cell_27/addAddV2#while/lstm_cell_27/MatMul:product:0%while/lstm_cell_27/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_27/addÈ
)while/lstm_cell_27/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_27_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02+
)while/lstm_cell_27/BiasAdd/ReadVariableOpÅ
while/lstm_cell_27/BiasAddBiasAddwhile/lstm_cell_27/add:z:01while/lstm_cell_27/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_27/BiasAdd
"while/lstm_cell_27/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_27/split/split_dim
while/lstm_cell_27/splitSplit+while/lstm_cell_27/split/split_dim:output:0#while/lstm_cell_27/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿæ:ÿÿÿÿÿÿÿÿÿæ:ÿÿÿÿÿÿÿÿÿæ:ÿÿÿÿÿÿÿÿÿæ*
	num_split2
while/lstm_cell_27/split
while/lstm_cell_27/SigmoidSigmoid!while/lstm_cell_27/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2
while/lstm_cell_27/Sigmoid
while/lstm_cell_27/Sigmoid_1Sigmoid!while/lstm_cell_27/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2
while/lstm_cell_27/Sigmoid_1¡
while/lstm_cell_27/mulMul while/lstm_cell_27/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2
while/lstm_cell_27/mul
while/lstm_cell_27/ReluRelu!while/lstm_cell_27/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2
while/lstm_cell_27/Reluµ
while/lstm_cell_27/mul_1Mulwhile/lstm_cell_27/Sigmoid:y:0%while/lstm_cell_27/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2
while/lstm_cell_27/mul_1ª
while/lstm_cell_27/add_1AddV2while/lstm_cell_27/mul:z:0while/lstm_cell_27/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2
while/lstm_cell_27/add_1
while/lstm_cell_27/Sigmoid_2Sigmoid!while/lstm_cell_27/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2
while/lstm_cell_27/Sigmoid_2
while/lstm_cell_27/Relu_1Reluwhile/lstm_cell_27/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2
while/lstm_cell_27/Relu_1¹
while/lstm_cell_27/mul_2Mul while/lstm_cell_27/Sigmoid_2:y:0'while/lstm_cell_27/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2
while/lstm_cell_27/mul_2à
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
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_27/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_27/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2
while/Identity_5Þ

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
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿæ:ÿÿÿÿÿÿÿÿÿæ: : : : : 2V
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
:ÿÿÿÿÿÿÿÿÿæ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ:

_output_shapes
: :

_output_shapes
: 
Ô

í
lstm_26_while_cond_39974607,
(lstm_26_while_lstm_26_while_loop_counter2
.lstm_26_while_lstm_26_while_maximum_iterations
lstm_26_while_placeholder
lstm_26_while_placeholder_1
lstm_26_while_placeholder_2
lstm_26_while_placeholder_3.
*lstm_26_while_less_lstm_26_strided_slice_1F
Blstm_26_while_lstm_26_while_cond_39974607___redundant_placeholder0F
Blstm_26_while_lstm_26_while_cond_39974607___redundant_placeholder1F
Blstm_26_while_lstm_26_while_cond_39974607___redundant_placeholder2F
Blstm_26_while_lstm_26_while_cond_39974607___redundant_placeholder3
lstm_26_while_identity

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
B: : : : :ÿÿÿÿÿÿÿÿÿÚ:ÿÿÿÿÿÿÿÿÿÚ: ::::: 

_output_shapes
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
:ÿÿÿÿÿÿÿÿÿÚ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ:

_output_shapes
: :

_output_shapes
:
ü	
Ê
&__inference_signature_wrapper_39974499
lstm_26_input
unknown:	]è

	unknown_0:
Úè

	unknown_1:	è

	unknown_2:
Ú
	unknown_3:
æ
	unknown_4:	
	unknown_5:	æ
	unknown_6:
identity¢StatefulPartitionedCall¬
StatefulPartitionedCallStatefulPartitionedCalllstm_26_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
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
#__inference__wrapped_model_399722822
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
_user_specified_namelstm_26_input
ã
Í
while_cond_39975994
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_39975994___redundant_placeholder06
2while_while_cond_39975994___redundant_placeholder16
2while_while_cond_39975994___redundant_placeholder26
2while_while_cond_39975994___redundant_placeholder3
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
B: : : : :ÿÿÿÿÿÿÿÿÿæ:ÿÿÿÿÿÿÿÿÿæ: ::::: 

_output_shapes
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
:ÿÿÿÿÿÿÿÿÿæ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ:

_output_shapes
: :

_output_shapes
:
\

E__inference_lstm_27_layer_call_and_return_conditional_losses_39976381

inputs?
+lstm_cell_27_matmul_readvariableop_resource:
ÚA
-lstm_cell_27_matmul_1_readvariableop_resource:
æ;
,lstm_cell_27_biasadd_readvariableop_resource:	
identity¢#lstm_cell_27/BiasAdd/ReadVariableOp¢"lstm_cell_27/MatMul/ReadVariableOp¢$lstm_cell_27/MatMul_1/ReadVariableOp¢whileD
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
B :æ2
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
B :æ2
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
:ÿÿÿÿÿÿÿÿÿæ2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :æ2
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
B :æ2
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
:ÿÿÿÿÿÿÿÿÿæ2	
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
:ÿÿÿÿÿÿÿÿÿÚ2
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
valueB"ÿÿÿÿZ  27
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
:ÿÿÿÿÿÿÿÿÿÚ*
shrink_axis_mask2
strided_slice_2¶
"lstm_cell_27/MatMul/ReadVariableOpReadVariableOp+lstm_cell_27_matmul_readvariableop_resource* 
_output_shapes
:
Ú*
dtype02$
"lstm_cell_27/MatMul/ReadVariableOp­
lstm_cell_27/MatMulMatMulstrided_slice_2:output:0*lstm_cell_27/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_27/MatMul¼
$lstm_cell_27/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_27_matmul_1_readvariableop_resource* 
_output_shapes
:
æ*
dtype02&
$lstm_cell_27/MatMul_1/ReadVariableOp©
lstm_cell_27/MatMul_1MatMulzeros:output:0,lstm_cell_27/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_27/MatMul_1 
lstm_cell_27/addAddV2lstm_cell_27/MatMul:product:0lstm_cell_27/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_27/add´
#lstm_cell_27/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_27_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02%
#lstm_cell_27/BiasAdd/ReadVariableOp­
lstm_cell_27/BiasAddBiasAddlstm_cell_27/add:z:0+lstm_cell_27/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_27/BiasAdd~
lstm_cell_27/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_27/split/split_dim÷
lstm_cell_27/splitSplit%lstm_cell_27/split/split_dim:output:0lstm_cell_27/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿæ:ÿÿÿÿÿÿÿÿÿæ:ÿÿÿÿÿÿÿÿÿæ:ÿÿÿÿÿÿÿÿÿæ*
	num_split2
lstm_cell_27/split
lstm_cell_27/SigmoidSigmoidlstm_cell_27/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2
lstm_cell_27/Sigmoid
lstm_cell_27/Sigmoid_1Sigmoidlstm_cell_27/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2
lstm_cell_27/Sigmoid_1
lstm_cell_27/mulMullstm_cell_27/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2
lstm_cell_27/mul~
lstm_cell_27/ReluRelulstm_cell_27/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2
lstm_cell_27/Relu
lstm_cell_27/mul_1Mullstm_cell_27/Sigmoid:y:0lstm_cell_27/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2
lstm_cell_27/mul_1
lstm_cell_27/add_1AddV2lstm_cell_27/mul:z:0lstm_cell_27/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2
lstm_cell_27/add_1
lstm_cell_27/Sigmoid_2Sigmoidlstm_cell_27/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2
lstm_cell_27/Sigmoid_2}
lstm_cell_27/Relu_1Relulstm_cell_27/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2
lstm_cell_27/Relu_1¡
lstm_cell_27/mul_2Mullstm_cell_27/Sigmoid_2:y:0!lstm_cell_27/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2
lstm_cell_27/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿf  2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_27_matmul_readvariableop_resource-lstm_cell_27_matmul_1_readvariableop_resource,lstm_cell_27_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿæ:ÿÿÿÿÿÿÿÿÿæ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_39976297*
condR
while_cond_39976296*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿæ:ÿÿÿÿÿÿÿÿÿæ: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿf  22
0TensorArrayV2Stack/TensorListStack/element_shapeé
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ*
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
:ÿÿÿÿÿÿÿÿÿæ*
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
:ÿÿÿÿÿÿÿÿÿæ2
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
:ÿÿÿÿÿÿÿÿÿæ2

IdentityÈ
NoOpNoOp$^lstm_cell_27/BiasAdd/ReadVariableOp#^lstm_cell_27/MatMul/ReadVariableOp%^lstm_cell_27/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:ÿÿÿÿÿÿÿÿÿÚ: : : 2J
#lstm_cell_27/BiasAdd/ReadVariableOp#lstm_cell_27/BiasAdd/ReadVariableOp2H
"lstm_cell_27/MatMul/ReadVariableOp"lstm_cell_27/MatMul/ReadVariableOp2L
$lstm_cell_27/MatMul_1/ReadVariableOp$lstm_cell_27/MatMul_1/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ
 
_user_specified_nameinputs
»
f
-__inference_dropout_26_layer_call_fn_39975867

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
:ÿÿÿÿÿÿÿÿÿÚ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_dropout_26_layer_call_and_return_conditional_losses_399741622
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿÚ22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ
 
_user_specified_nameinputs
ã
Í
while_cond_39973614
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_39973614___redundant_placeholder06
2while_while_cond_39973614___redundant_placeholder16
2while_while_cond_39973614___redundant_placeholder26
2while_while_cond_39973614___redundant_placeholder3
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
B: : : : :ÿÿÿÿÿÿÿÿÿÚ:ÿÿÿÿÿÿÿÿÿÚ: ::::: 

_output_shapes
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
:ÿÿÿÿÿÿÿÿÿÚ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ:

_output_shapes
: :

_output_shapes
:
°?
Ô
while_body_39975622
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_26_matmul_readvariableop_resource_0:	]è
I
5while_lstm_cell_26_matmul_1_readvariableop_resource_0:
Úè
C
4while_lstm_cell_26_biasadd_readvariableop_resource_0:	è

while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_26_matmul_readvariableop_resource:	]è
G
3while_lstm_cell_26_matmul_1_readvariableop_resource:
Úè
A
2while_lstm_cell_26_biasadd_readvariableop_resource:	è
¢)while/lstm_cell_26/BiasAdd/ReadVariableOp¢(while/lstm_cell_26/MatMul/ReadVariableOp¢*while/lstm_cell_26/MatMul_1/ReadVariableOpÃ
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
(while/lstm_cell_26/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_26_matmul_readvariableop_resource_0*
_output_shapes
:	]è
*
dtype02*
(while/lstm_cell_26/MatMul/ReadVariableOp×
while/lstm_cell_26/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_26/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè
2
while/lstm_cell_26/MatMulÐ
*while/lstm_cell_26/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_26_matmul_1_readvariableop_resource_0* 
_output_shapes
:
Úè
*
dtype02,
*while/lstm_cell_26/MatMul_1/ReadVariableOpÀ
while/lstm_cell_26/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_26/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè
2
while/lstm_cell_26/MatMul_1¸
while/lstm_cell_26/addAddV2#while/lstm_cell_26/MatMul:product:0%while/lstm_cell_26/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè
2
while/lstm_cell_26/addÈ
)while/lstm_cell_26/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_26_biasadd_readvariableop_resource_0*
_output_shapes	
:è
*
dtype02+
)while/lstm_cell_26/BiasAdd/ReadVariableOpÅ
while/lstm_cell_26/BiasAddBiasAddwhile/lstm_cell_26/add:z:01while/lstm_cell_26/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè
2
while/lstm_cell_26/BiasAdd
"while/lstm_cell_26/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_26/split/split_dim
while/lstm_cell_26/splitSplit+while/lstm_cell_26/split/split_dim:output:0#while/lstm_cell_26/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿÚ:ÿÿÿÿÿÿÿÿÿÚ:ÿÿÿÿÿÿÿÿÿÚ:ÿÿÿÿÿÿÿÿÿÚ*
	num_split2
while/lstm_cell_26/split
while/lstm_cell_26/SigmoidSigmoid!while/lstm_cell_26/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
while/lstm_cell_26/Sigmoid
while/lstm_cell_26/Sigmoid_1Sigmoid!while/lstm_cell_26/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
while/lstm_cell_26/Sigmoid_1¡
while/lstm_cell_26/mulMul while/lstm_cell_26/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
while/lstm_cell_26/mul
while/lstm_cell_26/ReluRelu!while/lstm_cell_26/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
while/lstm_cell_26/Reluµ
while/lstm_cell_26/mul_1Mulwhile/lstm_cell_26/Sigmoid:y:0%while/lstm_cell_26/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
while/lstm_cell_26/mul_1ª
while/lstm_cell_26/add_1AddV2while/lstm_cell_26/mul:z:0while/lstm_cell_26/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
while/lstm_cell_26/add_1
while/lstm_cell_26/Sigmoid_2Sigmoid!while/lstm_cell_26/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
while/lstm_cell_26/Sigmoid_2
while/lstm_cell_26/Relu_1Reluwhile/lstm_cell_26/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
while/lstm_cell_26/Relu_1¹
while/lstm_cell_26/mul_2Mul while/lstm_cell_26/Sigmoid_2:y:0'while/lstm_cell_26/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
while/lstm_cell_26/mul_2à
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
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_26/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_26/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
while/Identity_5Þ

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
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿÚ:ÿÿÿÿÿÿÿÿÿÚ: : : : : 2V
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
:ÿÿÿÿÿÿÿÿÿÚ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ:

_output_shapes
: :

_output_shapes
: 
ã
Í
while_cond_39974048
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_39974048___redundant_placeholder06
2while_while_cond_39974048___redundant_placeholder16
2while_while_cond_39974048___redundant_placeholder26
2while_while_cond_39974048___redundant_placeholder3
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
B: : : : :ÿÿÿÿÿÿÿÿÿæ:ÿÿÿÿÿÿÿÿÿæ: ::::: 

_output_shapes
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
:ÿÿÿÿÿÿÿÿÿæ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ:

_output_shapes
: :

_output_shapes
:
ã
Í
while_cond_39975470
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_39975470___redundant_placeholder06
2while_while_cond_39975470___redundant_placeholder16
2while_while_cond_39975470___redundant_placeholder26
2while_while_cond_39975470___redundant_placeholder3
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
B: : : : :ÿÿÿÿÿÿÿÿÿÚ:ÿÿÿÿÿÿÿÿÿÚ: ::::: 

_output_shapes
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
:ÿÿÿÿÿÿÿÿÿÚ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ:

_output_shapes
: :

_output_shapes
:


J__inference_lstm_cell_26_layer_call_and_return_conditional_losses_39972357

inputs

states
states_11
matmul_readvariableop_resource:	]è
4
 matmul_1_readvariableop_resource:
Úè
.
biasadd_readvariableop_resource:	è

identity

identity_1

identity_2¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	]è
*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè
2
MatMul
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
Úè
*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè
2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè
2
add
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:è
*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè
2	
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
P:ÿÿÿÿÿÿÿÿÿÚ:ÿÿÿÿÿÿÿÿÿÚ:ÿÿÿÿÿÿÿÿÿÚ:ÿÿÿÿÿÿÿÿÿÚ*
	num_split2
split`
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2	
Sigmoidd
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
	Sigmoid_1]
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
mulW
ReluRelusplit:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
Relui
mul_1MulSigmoid:y:0Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
mul_1^
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
add_1d
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
	Sigmoid_2V
Relu_1Relu	add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
Relu_1m
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
mul_2e
IdentityIdentity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2

Identityi

Identity_1Identity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2

Identity_1i

Identity_2Identity	add_1:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2

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
A:ÿÿÿÿÿÿÿÿÿ]:ÿÿÿÿÿÿÿÿÿÚ:ÿÿÿÿÿÿÿÿÿÚ: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
 
_user_specified_nameinputs:PL
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ
 
_user_specified_namestates:PL
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ
 
_user_specified_namestates
Î7
ß
$__inference__traced_restore_39976900
file_prefix3
 assignvariableop_dense_13_kernel:	æ.
 assignvariableop_1_dense_13_bias:A
.assignvariableop_2_lstm_26_lstm_cell_26_kernel:	]è
L
8assignvariableop_3_lstm_26_lstm_cell_26_recurrent_kernel:
Úè
;
,assignvariableop_4_lstm_26_lstm_cell_26_bias:	è
B
.assignvariableop_5_lstm_27_lstm_cell_27_kernel:
ÚL
8assignvariableop_6_lstm_27_lstm_cell_27_recurrent_kernel:
æ;
,assignvariableop_7_lstm_27_lstm_cell_27_bias:	"
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
AssignVariableOpAssignVariableOp assignvariableop_dense_13_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1¥
AssignVariableOp_1AssignVariableOp assignvariableop_1_dense_13_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2³
AssignVariableOp_2AssignVariableOp.assignvariableop_2_lstm_26_lstm_cell_26_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3½
AssignVariableOp_3AssignVariableOp8assignvariableop_3_lstm_26_lstm_cell_26_recurrent_kernelIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4±
AssignVariableOp_4AssignVariableOp,assignvariableop_4_lstm_26_lstm_cell_26_biasIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5³
AssignVariableOp_5AssignVariableOp.assignvariableop_5_lstm_27_lstm_cell_27_kernelIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6½
AssignVariableOp_6AssignVariableOp8assignvariableop_6_lstm_27_lstm_cell_27_recurrent_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7±
AssignVariableOp_7AssignVariableOp,assignvariableop_7_lstm_27_lstm_cell_27_biasIdentity_7:output:0"/device:CPU:0*
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
ã
Í
while_cond_39974244
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_39974244___redundant_placeholder06
2while_while_cond_39974244___redundant_placeholder16
2while_while_cond_39974244___redundant_placeholder26
2while_while_cond_39974244___redundant_placeholder3
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
B: : : : :ÿÿÿÿÿÿÿÿÿÚ:ÿÿÿÿÿÿÿÿÿÚ: ::::: 

_output_shapes
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
:ÿÿÿÿÿÿÿÿÿÚ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ:

_output_shapes
: :

_output_shapes
:

f
H__inference_dropout_27_layer_call_and_return_conditional_losses_39973877

inputs

identity_1_
IdentityIdentityinputs*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2

Identityn

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿæ:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ
 
_user_specified_nameinputs
É\
¡
E__inference_lstm_27_layer_call_and_return_conditional_losses_39976230
inputs_0?
+lstm_cell_27_matmul_readvariableop_resource:
ÚA
-lstm_cell_27_matmul_1_readvariableop_resource:
æ;
,lstm_cell_27_biasadd_readvariableop_resource:	
identity¢#lstm_cell_27/BiasAdd/ReadVariableOp¢"lstm_cell_27/MatMul/ReadVariableOp¢$lstm_cell_27/MatMul_1/ReadVariableOp¢whileF
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
B :æ2
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
B :æ2
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
:ÿÿÿÿÿÿÿÿÿæ2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :æ2
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
B :æ2
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
:ÿÿÿÿÿÿÿÿÿæ2	
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
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÚ2
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
valueB"ÿÿÿÿZ  27
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
:ÿÿÿÿÿÿÿÿÿÚ*
shrink_axis_mask2
strided_slice_2¶
"lstm_cell_27/MatMul/ReadVariableOpReadVariableOp+lstm_cell_27_matmul_readvariableop_resource* 
_output_shapes
:
Ú*
dtype02$
"lstm_cell_27/MatMul/ReadVariableOp­
lstm_cell_27/MatMulMatMulstrided_slice_2:output:0*lstm_cell_27/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_27/MatMul¼
$lstm_cell_27/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_27_matmul_1_readvariableop_resource* 
_output_shapes
:
æ*
dtype02&
$lstm_cell_27/MatMul_1/ReadVariableOp©
lstm_cell_27/MatMul_1MatMulzeros:output:0,lstm_cell_27/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_27/MatMul_1 
lstm_cell_27/addAddV2lstm_cell_27/MatMul:product:0lstm_cell_27/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_27/add´
#lstm_cell_27/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_27_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02%
#lstm_cell_27/BiasAdd/ReadVariableOp­
lstm_cell_27/BiasAddBiasAddlstm_cell_27/add:z:0+lstm_cell_27/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_27/BiasAdd~
lstm_cell_27/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_27/split/split_dim÷
lstm_cell_27/splitSplit%lstm_cell_27/split/split_dim:output:0lstm_cell_27/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿæ:ÿÿÿÿÿÿÿÿÿæ:ÿÿÿÿÿÿÿÿÿæ:ÿÿÿÿÿÿÿÿÿæ*
	num_split2
lstm_cell_27/split
lstm_cell_27/SigmoidSigmoidlstm_cell_27/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2
lstm_cell_27/Sigmoid
lstm_cell_27/Sigmoid_1Sigmoidlstm_cell_27/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2
lstm_cell_27/Sigmoid_1
lstm_cell_27/mulMullstm_cell_27/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2
lstm_cell_27/mul~
lstm_cell_27/ReluRelulstm_cell_27/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2
lstm_cell_27/Relu
lstm_cell_27/mul_1Mullstm_cell_27/Sigmoid:y:0lstm_cell_27/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2
lstm_cell_27/mul_1
lstm_cell_27/add_1AddV2lstm_cell_27/mul:z:0lstm_cell_27/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2
lstm_cell_27/add_1
lstm_cell_27/Sigmoid_2Sigmoidlstm_cell_27/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2
lstm_cell_27/Sigmoid_2}
lstm_cell_27/Relu_1Relulstm_cell_27/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2
lstm_cell_27/Relu_1¡
lstm_cell_27/mul_2Mullstm_cell_27/Sigmoid_2:y:0!lstm_cell_27/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2
lstm_cell_27/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿf  2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_27_matmul_readvariableop_resource-lstm_cell_27_matmul_1_readvariableop_resource,lstm_cell_27_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿæ:ÿÿÿÿÿÿÿÿÿæ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_39976146*
condR
while_cond_39976145*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿæ:ÿÿÿÿÿÿÿÿÿæ: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿf  22
0TensorArrayV2Stack/TensorListStack/element_shapeò
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿæ*
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
:ÿÿÿÿÿÿÿÿÿæ*
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
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿæ2
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
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿæ2

IdentityÈ
NoOpNoOp$^lstm_cell_27/BiasAdd/ReadVariableOp#^lstm_cell_27/MatMul/ReadVariableOp%^lstm_cell_27/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÚ: : : 2J
#lstm_cell_27/BiasAdd/ReadVariableOp#lstm_cell_27/BiasAdd/ReadVariableOp2H
"lstm_cell_27/MatMul/ReadVariableOp"lstm_cell_27/MatMul/ReadVariableOp2L
$lstm_cell_27/MatMul_1/ReadVariableOp$lstm_cell_27/MatMul_1/ReadVariableOp2
whilewhile:_ [
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÚ
"
_user_specified_name
inputs/0


J__inference_lstm_cell_26_layer_call_and_return_conditional_losses_39976665

inputs
states_0
states_11
matmul_readvariableop_resource:	]è
4
 matmul_1_readvariableop_resource:
Úè
.
biasadd_readvariableop_resource:	è

identity

identity_1

identity_2¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	]è
*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè
2
MatMul
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
Úè
*
dtype02
MatMul_1/ReadVariableOp|
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè
2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè
2
add
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:è
*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè
2	
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
P:ÿÿÿÿÿÿÿÿÿÚ:ÿÿÿÿÿÿÿÿÿÚ:ÿÿÿÿÿÿÿÿÿÚ:ÿÿÿÿÿÿÿÿÿÚ*
	num_split2
split`
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2	
Sigmoidd
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
	Sigmoid_1]
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
mulW
ReluRelusplit:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
Relui
mul_1MulSigmoid:y:0Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
mul_1^
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
add_1d
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
	Sigmoid_2V
Relu_1Relu	add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
Relu_1m
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
mul_2e
IdentityIdentity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2

Identityi

Identity_1Identity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2

Identity_1i

Identity_2Identity	add_1:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2

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
A:ÿÿÿÿÿÿÿÿÿ]:ÿÿÿÿÿÿÿÿÿÚ:ÿÿÿÿÿÿÿÿÿÚ: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
 
_user_specified_nameinputs:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ
"
_user_specified_name
states/0:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ
"
_user_specified_name
states/1

f
H__inference_dropout_26_layer_call_and_return_conditional_losses_39973712

inputs

identity_1_
IdentityIdentityinputs*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2

Identityn

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿÚ:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ
 
_user_specified_nameinputs
°?
Ô
while_body_39975471
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_26_matmul_readvariableop_resource_0:	]è
I
5while_lstm_cell_26_matmul_1_readvariableop_resource_0:
Úè
C
4while_lstm_cell_26_biasadd_readvariableop_resource_0:	è

while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_26_matmul_readvariableop_resource:	]è
G
3while_lstm_cell_26_matmul_1_readvariableop_resource:
Úè
A
2while_lstm_cell_26_biasadd_readvariableop_resource:	è
¢)while/lstm_cell_26/BiasAdd/ReadVariableOp¢(while/lstm_cell_26/MatMul/ReadVariableOp¢*while/lstm_cell_26/MatMul_1/ReadVariableOpÃ
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
(while/lstm_cell_26/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_26_matmul_readvariableop_resource_0*
_output_shapes
:	]è
*
dtype02*
(while/lstm_cell_26/MatMul/ReadVariableOp×
while/lstm_cell_26/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_26/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè
2
while/lstm_cell_26/MatMulÐ
*while/lstm_cell_26/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_26_matmul_1_readvariableop_resource_0* 
_output_shapes
:
Úè
*
dtype02,
*while/lstm_cell_26/MatMul_1/ReadVariableOpÀ
while/lstm_cell_26/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_26/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè
2
while/lstm_cell_26/MatMul_1¸
while/lstm_cell_26/addAddV2#while/lstm_cell_26/MatMul:product:0%while/lstm_cell_26/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè
2
while/lstm_cell_26/addÈ
)while/lstm_cell_26/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_26_biasadd_readvariableop_resource_0*
_output_shapes	
:è
*
dtype02+
)while/lstm_cell_26/BiasAdd/ReadVariableOpÅ
while/lstm_cell_26/BiasAddBiasAddwhile/lstm_cell_26/add:z:01while/lstm_cell_26/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè
2
while/lstm_cell_26/BiasAdd
"while/lstm_cell_26/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_26/split/split_dim
while/lstm_cell_26/splitSplit+while/lstm_cell_26/split/split_dim:output:0#while/lstm_cell_26/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿÚ:ÿÿÿÿÿÿÿÿÿÚ:ÿÿÿÿÿÿÿÿÿÚ:ÿÿÿÿÿÿÿÿÿÚ*
	num_split2
while/lstm_cell_26/split
while/lstm_cell_26/SigmoidSigmoid!while/lstm_cell_26/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
while/lstm_cell_26/Sigmoid
while/lstm_cell_26/Sigmoid_1Sigmoid!while/lstm_cell_26/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
while/lstm_cell_26/Sigmoid_1¡
while/lstm_cell_26/mulMul while/lstm_cell_26/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
while/lstm_cell_26/mul
while/lstm_cell_26/ReluRelu!while/lstm_cell_26/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
while/lstm_cell_26/Reluµ
while/lstm_cell_26/mul_1Mulwhile/lstm_cell_26/Sigmoid:y:0%while/lstm_cell_26/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
while/lstm_cell_26/mul_1ª
while/lstm_cell_26/add_1AddV2while/lstm_cell_26/mul:z:0while/lstm_cell_26/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
while/lstm_cell_26/add_1
while/lstm_cell_26/Sigmoid_2Sigmoid!while/lstm_cell_26/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
while/lstm_cell_26/Sigmoid_2
while/lstm_cell_26/Relu_1Reluwhile/lstm_cell_26/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
while/lstm_cell_26/Relu_1¹
while/lstm_cell_26/mul_2Mul while/lstm_cell_26/Sigmoid_2:y:0'while/lstm_cell_26/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
while/lstm_cell_26/mul_2à
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
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_26/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_26/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
while/Identity_5Þ

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
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿÚ:ÿÿÿÿÿÿÿÿÿÚ: : : : : 2V
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
:ÿÿÿÿÿÿÿÿÿÚ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ:

_output_shapes
: :

_output_shapes
: 
»
f
-__inference_dropout_27_layer_call_fn_39976542

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
:ÿÿÿÿÿÿÿÿÿæ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_dropout_27_layer_call_and_return_conditional_losses_399739662
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿæ22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ
 
_user_specified_nameinputs
Ê
ú
/__inference_lstm_cell_27_layer_call_fn_39976714

inputs
states_0
states_1
unknown:
Ú
	unknown_0:
æ
	unknown_1:	
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
<:ÿÿÿÿÿÿÿÿÿæ:ÿÿÿÿÿÿÿÿÿæ:ÿÿÿÿÿÿÿÿÿæ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_lstm_cell_27_layer_call_and_return_conditional_losses_399729872
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2

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
B:ÿÿÿÿÿÿÿÿÿÚ:ÿÿÿÿÿÿÿÿÿæ:ÿÿÿÿÿÿÿÿÿæ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ
 
_user_specified_nameinputs:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ
"
_user_specified_name
states/0:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ
"
_user_specified_name
states/1
Ø
I
-__inference_dropout_26_layer_call_fn_39975862

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
:ÿÿÿÿÿÿÿÿÿÚ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_dropout_26_layer_call_and_return_conditional_losses_399737122
PartitionedCallq
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿÚ:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ
 
_user_specified_nameinputs
&
ó
while_body_39972371
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_00
while_lstm_cell_26_39972395_0:	]è
1
while_lstm_cell_26_39972397_0:
Úè
,
while_lstm_cell_26_39972399_0:	è

while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor.
while_lstm_cell_26_39972395:	]è
/
while_lstm_cell_26_39972397:
Úè
*
while_lstm_cell_26_39972399:	è
¢*while/lstm_cell_26/StatefulPartitionedCallÃ
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
*while/lstm_cell_26/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_26_39972395_0while_lstm_cell_26_39972397_0while_lstm_cell_26_39972399_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿÚ:ÿÿÿÿÿÿÿÿÿÚ:ÿÿÿÿÿÿÿÿÿÚ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_lstm_cell_26_layer_call_and_return_conditional_losses_399723572,
*while/lstm_cell_26/StatefulPartitionedCall÷
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
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3¥
while/Identity_4Identity3while/lstm_cell_26/StatefulPartitionedCall:output:1^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
while/Identity_4¥
while/Identity_5Identity3while/lstm_cell_26/StatefulPartitionedCall:output:2^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
while/Identity_5

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
while_lstm_cell_26_39972395while_lstm_cell_26_39972395_0"<
while_lstm_cell_26_39972397while_lstm_cell_26_39972397_0"<
while_lstm_cell_26_39972399while_lstm_cell_26_39972399_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿÚ:ÿÿÿÿÿÿÿÿÿÚ: : : : : 2X
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
:ÿÿÿÿÿÿÿÿÿÚ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ:

_output_shapes
: :

_output_shapes
: 
É\
¡
E__inference_lstm_27_layer_call_and_return_conditional_losses_39976079
inputs_0?
+lstm_cell_27_matmul_readvariableop_resource:
ÚA
-lstm_cell_27_matmul_1_readvariableop_resource:
æ;
,lstm_cell_27_biasadd_readvariableop_resource:	
identity¢#lstm_cell_27/BiasAdd/ReadVariableOp¢"lstm_cell_27/MatMul/ReadVariableOp¢$lstm_cell_27/MatMul_1/ReadVariableOp¢whileF
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
B :æ2
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
B :æ2
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
:ÿÿÿÿÿÿÿÿÿæ2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :æ2
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
B :æ2
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
:ÿÿÿÿÿÿÿÿÿæ2	
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
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÚ2
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
valueB"ÿÿÿÿZ  27
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
:ÿÿÿÿÿÿÿÿÿÚ*
shrink_axis_mask2
strided_slice_2¶
"lstm_cell_27/MatMul/ReadVariableOpReadVariableOp+lstm_cell_27_matmul_readvariableop_resource* 
_output_shapes
:
Ú*
dtype02$
"lstm_cell_27/MatMul/ReadVariableOp­
lstm_cell_27/MatMulMatMulstrided_slice_2:output:0*lstm_cell_27/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_27/MatMul¼
$lstm_cell_27/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_27_matmul_1_readvariableop_resource* 
_output_shapes
:
æ*
dtype02&
$lstm_cell_27/MatMul_1/ReadVariableOp©
lstm_cell_27/MatMul_1MatMulzeros:output:0,lstm_cell_27/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_27/MatMul_1 
lstm_cell_27/addAddV2lstm_cell_27/MatMul:product:0lstm_cell_27/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_27/add´
#lstm_cell_27/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_27_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02%
#lstm_cell_27/BiasAdd/ReadVariableOp­
lstm_cell_27/BiasAddBiasAddlstm_cell_27/add:z:0+lstm_cell_27/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_27/BiasAdd~
lstm_cell_27/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_27/split/split_dim÷
lstm_cell_27/splitSplit%lstm_cell_27/split/split_dim:output:0lstm_cell_27/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿæ:ÿÿÿÿÿÿÿÿÿæ:ÿÿÿÿÿÿÿÿÿæ:ÿÿÿÿÿÿÿÿÿæ*
	num_split2
lstm_cell_27/split
lstm_cell_27/SigmoidSigmoidlstm_cell_27/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2
lstm_cell_27/Sigmoid
lstm_cell_27/Sigmoid_1Sigmoidlstm_cell_27/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2
lstm_cell_27/Sigmoid_1
lstm_cell_27/mulMullstm_cell_27/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2
lstm_cell_27/mul~
lstm_cell_27/ReluRelulstm_cell_27/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2
lstm_cell_27/Relu
lstm_cell_27/mul_1Mullstm_cell_27/Sigmoid:y:0lstm_cell_27/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2
lstm_cell_27/mul_1
lstm_cell_27/add_1AddV2lstm_cell_27/mul:z:0lstm_cell_27/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2
lstm_cell_27/add_1
lstm_cell_27/Sigmoid_2Sigmoidlstm_cell_27/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2
lstm_cell_27/Sigmoid_2}
lstm_cell_27/Relu_1Relulstm_cell_27/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2
lstm_cell_27/Relu_1¡
lstm_cell_27/mul_2Mullstm_cell_27/Sigmoid_2:y:0!lstm_cell_27/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2
lstm_cell_27/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿf  2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_27_matmul_readvariableop_resource-lstm_cell_27_matmul_1_readvariableop_resource,lstm_cell_27_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿæ:ÿÿÿÿÿÿÿÿÿæ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_39975995*
condR
while_cond_39975994*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿæ:ÿÿÿÿÿÿÿÿÿæ: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿf  22
0TensorArrayV2Stack/TensorListStack/element_shapeò
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿæ*
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
:ÿÿÿÿÿÿÿÿÿæ*
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
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿæ2
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
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿæ2

IdentityÈ
NoOpNoOp$^lstm_cell_27/BiasAdd/ReadVariableOp#^lstm_cell_27/MatMul/ReadVariableOp%^lstm_cell_27/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÚ: : : 2J
#lstm_cell_27/BiasAdd/ReadVariableOp#lstm_cell_27/BiasAdd/ReadVariableOp2H
"lstm_cell_27/MatMul/ReadVariableOp"lstm_cell_27/MatMul/ReadVariableOp2L
$lstm_cell_27/MatMul_1/ReadVariableOp$lstm_cell_27/MatMul_1/ReadVariableOp2
whilewhile:_ [
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÚ
"
_user_specified_name
inputs/0
®

Ô
0__inference_sequential_13_layer_call_fn_39974426
lstm_26_input
unknown:	]è

	unknown_0:
Úè

	unknown_1:	è

	unknown_2:
Ú
	unknown_3:
æ
	unknown_4:	
	unknown_5:	æ
	unknown_6:
identity¢StatefulPartitionedCallÔ
StatefulPartitionedCallStatefulPartitionedCalllstm_26_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
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
K__inference_sequential_13_layer_call_and_return_conditional_losses_399743862
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
_user_specified_namelstm_26_input


J__inference_lstm_cell_26_layer_call_and_return_conditional_losses_39976697

inputs
states_0
states_11
matmul_readvariableop_resource:	]è
4
 matmul_1_readvariableop_resource:
Úè
.
biasadd_readvariableop_resource:	è

identity

identity_1

identity_2¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	]è
*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè
2
MatMul
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
Úè
*
dtype02
MatMul_1/ReadVariableOp|
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè
2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè
2
add
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:è
*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè
2	
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
P:ÿÿÿÿÿÿÿÿÿÚ:ÿÿÿÿÿÿÿÿÿÚ:ÿÿÿÿÿÿÿÿÿÚ:ÿÿÿÿÿÿÿÿÿÚ*
	num_split2
split`
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2	
Sigmoidd
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
	Sigmoid_1]
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
mulW
ReluRelusplit:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
Relui
mul_1MulSigmoid:y:0Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
mul_1^
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
add_1d
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
	Sigmoid_2V
Relu_1Relu	add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
Relu_1m
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
mul_2e
IdentityIdentity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2

Identityi

Identity_1Identity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2

Identity_1i

Identity_2Identity	add_1:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2

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
A:ÿÿÿÿÿÿÿÿÿ]:ÿÿÿÿÿÿÿÿÿÚ:ÿÿÿÿÿÿÿÿÿÚ: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
 
_user_specified_nameinputs:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ
"
_user_specified_name
states/0:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ
"
_user_specified_name
states/1
\

E__inference_lstm_27_layer_call_and_return_conditional_losses_39976532

inputs?
+lstm_cell_27_matmul_readvariableop_resource:
ÚA
-lstm_cell_27_matmul_1_readvariableop_resource:
æ;
,lstm_cell_27_biasadd_readvariableop_resource:	
identity¢#lstm_cell_27/BiasAdd/ReadVariableOp¢"lstm_cell_27/MatMul/ReadVariableOp¢$lstm_cell_27/MatMul_1/ReadVariableOp¢whileD
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
B :æ2
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
B :æ2
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
:ÿÿÿÿÿÿÿÿÿæ2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :æ2
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
B :æ2
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
:ÿÿÿÿÿÿÿÿÿæ2	
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
:ÿÿÿÿÿÿÿÿÿÚ2
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
valueB"ÿÿÿÿZ  27
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
:ÿÿÿÿÿÿÿÿÿÚ*
shrink_axis_mask2
strided_slice_2¶
"lstm_cell_27/MatMul/ReadVariableOpReadVariableOp+lstm_cell_27_matmul_readvariableop_resource* 
_output_shapes
:
Ú*
dtype02$
"lstm_cell_27/MatMul/ReadVariableOp­
lstm_cell_27/MatMulMatMulstrided_slice_2:output:0*lstm_cell_27/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_27/MatMul¼
$lstm_cell_27/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_27_matmul_1_readvariableop_resource* 
_output_shapes
:
æ*
dtype02&
$lstm_cell_27/MatMul_1/ReadVariableOp©
lstm_cell_27/MatMul_1MatMulzeros:output:0,lstm_cell_27/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_27/MatMul_1 
lstm_cell_27/addAddV2lstm_cell_27/MatMul:product:0lstm_cell_27/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_27/add´
#lstm_cell_27/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_27_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02%
#lstm_cell_27/BiasAdd/ReadVariableOp­
lstm_cell_27/BiasAddBiasAddlstm_cell_27/add:z:0+lstm_cell_27/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_27/BiasAdd~
lstm_cell_27/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_27/split/split_dim÷
lstm_cell_27/splitSplit%lstm_cell_27/split/split_dim:output:0lstm_cell_27/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿæ:ÿÿÿÿÿÿÿÿÿæ:ÿÿÿÿÿÿÿÿÿæ:ÿÿÿÿÿÿÿÿÿæ*
	num_split2
lstm_cell_27/split
lstm_cell_27/SigmoidSigmoidlstm_cell_27/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2
lstm_cell_27/Sigmoid
lstm_cell_27/Sigmoid_1Sigmoidlstm_cell_27/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2
lstm_cell_27/Sigmoid_1
lstm_cell_27/mulMullstm_cell_27/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2
lstm_cell_27/mul~
lstm_cell_27/ReluRelulstm_cell_27/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2
lstm_cell_27/Relu
lstm_cell_27/mul_1Mullstm_cell_27/Sigmoid:y:0lstm_cell_27/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2
lstm_cell_27/mul_1
lstm_cell_27/add_1AddV2lstm_cell_27/mul:z:0lstm_cell_27/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2
lstm_cell_27/add_1
lstm_cell_27/Sigmoid_2Sigmoidlstm_cell_27/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2
lstm_cell_27/Sigmoid_2}
lstm_cell_27/Relu_1Relulstm_cell_27/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2
lstm_cell_27/Relu_1¡
lstm_cell_27/mul_2Mullstm_cell_27/Sigmoid_2:y:0!lstm_cell_27/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2
lstm_cell_27/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿf  2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_27_matmul_readvariableop_resource-lstm_cell_27_matmul_1_readvariableop_resource,lstm_cell_27_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿæ:ÿÿÿÿÿÿÿÿÿæ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_39976448*
condR
while_cond_39976447*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿæ:ÿÿÿÿÿÿÿÿÿæ: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿf  22
0TensorArrayV2Stack/TensorListStack/element_shapeé
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ*
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
:ÿÿÿÿÿÿÿÿÿæ*
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
:ÿÿÿÿÿÿÿÿÿæ2
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
:ÿÿÿÿÿÿÿÿÿæ2

IdentityÈ
NoOpNoOp$^lstm_cell_27/BiasAdd/ReadVariableOp#^lstm_cell_27/MatMul/ReadVariableOp%^lstm_cell_27/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:ÿÿÿÿÿÿÿÿÿÚ: : : 2J
#lstm_cell_27/BiasAdd/ReadVariableOp#lstm_cell_27/BiasAdd/ReadVariableOp2H
"lstm_cell_27/MatMul/ReadVariableOp"lstm_cell_27/MatMul/ReadVariableOp2L
$lstm_cell_27/MatMul_1/ReadVariableOp$lstm_cell_27/MatMul_1/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ
 
_user_specified_nameinputs
\

E__inference_lstm_26_layer_call_and_return_conditional_losses_39974329

inputs>
+lstm_cell_26_matmul_readvariableop_resource:	]è
A
-lstm_cell_26_matmul_1_readvariableop_resource:
Úè
;
,lstm_cell_26_biasadd_readvariableop_resource:	è

identity¢#lstm_cell_26/BiasAdd/ReadVariableOp¢"lstm_cell_26/MatMul/ReadVariableOp¢$lstm_cell_26/MatMul_1/ReadVariableOp¢whileD
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
B :Ú2
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
B :Ú2
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
:ÿÿÿÿÿÿÿÿÿÚ2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :Ú2
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
B :Ú2
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
:ÿÿÿÿÿÿÿÿÿÚ2	
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
"lstm_cell_26/MatMul/ReadVariableOpReadVariableOp+lstm_cell_26_matmul_readvariableop_resource*
_output_shapes
:	]è
*
dtype02$
"lstm_cell_26/MatMul/ReadVariableOp­
lstm_cell_26/MatMulMatMulstrided_slice_2:output:0*lstm_cell_26/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè
2
lstm_cell_26/MatMul¼
$lstm_cell_26/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_26_matmul_1_readvariableop_resource* 
_output_shapes
:
Úè
*
dtype02&
$lstm_cell_26/MatMul_1/ReadVariableOp©
lstm_cell_26/MatMul_1MatMulzeros:output:0,lstm_cell_26/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè
2
lstm_cell_26/MatMul_1 
lstm_cell_26/addAddV2lstm_cell_26/MatMul:product:0lstm_cell_26/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè
2
lstm_cell_26/add´
#lstm_cell_26/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_26_biasadd_readvariableop_resource*
_output_shapes	
:è
*
dtype02%
#lstm_cell_26/BiasAdd/ReadVariableOp­
lstm_cell_26/BiasAddBiasAddlstm_cell_26/add:z:0+lstm_cell_26/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè
2
lstm_cell_26/BiasAdd~
lstm_cell_26/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_26/split/split_dim÷
lstm_cell_26/splitSplit%lstm_cell_26/split/split_dim:output:0lstm_cell_26/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿÚ:ÿÿÿÿÿÿÿÿÿÚ:ÿÿÿÿÿÿÿÿÿÚ:ÿÿÿÿÿÿÿÿÿÚ*
	num_split2
lstm_cell_26/split
lstm_cell_26/SigmoidSigmoidlstm_cell_26/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
lstm_cell_26/Sigmoid
lstm_cell_26/Sigmoid_1Sigmoidlstm_cell_26/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
lstm_cell_26/Sigmoid_1
lstm_cell_26/mulMullstm_cell_26/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
lstm_cell_26/mul~
lstm_cell_26/ReluRelulstm_cell_26/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
lstm_cell_26/Relu
lstm_cell_26/mul_1Mullstm_cell_26/Sigmoid:y:0lstm_cell_26/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
lstm_cell_26/mul_1
lstm_cell_26/add_1AddV2lstm_cell_26/mul:z:0lstm_cell_26/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
lstm_cell_26/add_1
lstm_cell_26/Sigmoid_2Sigmoidlstm_cell_26/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
lstm_cell_26/Sigmoid_2}
lstm_cell_26/Relu_1Relulstm_cell_26/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
lstm_cell_26/Relu_1¡
lstm_cell_26/mul_2Mullstm_cell_26/Sigmoid_2:y:0!lstm_cell_26/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
lstm_cell_26/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿZ  2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_26_matmul_readvariableop_resource-lstm_cell_26_matmul_1_readvariableop_resource,lstm_cell_26_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿÚ:ÿÿÿÿÿÿÿÿÿÚ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_39974245*
condR
while_cond_39974244*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿÚ:ÿÿÿÿÿÿÿÿÿÚ: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿZ  22
0TensorArrayV2Stack/TensorListStack/element_shapeé
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ*
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
:ÿÿÿÿÿÿÿÿÿÚ*
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
:ÿÿÿÿÿÿÿÿÿÚ2
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
:ÿÿÿÿÿÿÿÿÿÚ2

IdentityÈ
NoOpNoOp$^lstm_cell_26/BiasAdd/ReadVariableOp#^lstm_cell_26/MatMul/ReadVariableOp%^lstm_cell_26/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ]: : : 2J
#lstm_cell_26/BiasAdd/ReadVariableOp#lstm_cell_26/BiasAdd/ReadVariableOp2H
"lstm_cell_26/MatMul/ReadVariableOp"lstm_cell_26/MatMul/ReadVariableOp2L
$lstm_cell_26/MatMul_1/ReadVariableOp$lstm_cell_26/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
 
_user_specified_nameinputs
°?
Ô
while_body_39975320
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_26_matmul_readvariableop_resource_0:	]è
I
5while_lstm_cell_26_matmul_1_readvariableop_resource_0:
Úè
C
4while_lstm_cell_26_biasadd_readvariableop_resource_0:	è

while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_26_matmul_readvariableop_resource:	]è
G
3while_lstm_cell_26_matmul_1_readvariableop_resource:
Úè
A
2while_lstm_cell_26_biasadd_readvariableop_resource:	è
¢)while/lstm_cell_26/BiasAdd/ReadVariableOp¢(while/lstm_cell_26/MatMul/ReadVariableOp¢*while/lstm_cell_26/MatMul_1/ReadVariableOpÃ
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
(while/lstm_cell_26/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_26_matmul_readvariableop_resource_0*
_output_shapes
:	]è
*
dtype02*
(while/lstm_cell_26/MatMul/ReadVariableOp×
while/lstm_cell_26/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_26/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè
2
while/lstm_cell_26/MatMulÐ
*while/lstm_cell_26/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_26_matmul_1_readvariableop_resource_0* 
_output_shapes
:
Úè
*
dtype02,
*while/lstm_cell_26/MatMul_1/ReadVariableOpÀ
while/lstm_cell_26/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_26/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè
2
while/lstm_cell_26/MatMul_1¸
while/lstm_cell_26/addAddV2#while/lstm_cell_26/MatMul:product:0%while/lstm_cell_26/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè
2
while/lstm_cell_26/addÈ
)while/lstm_cell_26/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_26_biasadd_readvariableop_resource_0*
_output_shapes	
:è
*
dtype02+
)while/lstm_cell_26/BiasAdd/ReadVariableOpÅ
while/lstm_cell_26/BiasAddBiasAddwhile/lstm_cell_26/add:z:01while/lstm_cell_26/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè
2
while/lstm_cell_26/BiasAdd
"while/lstm_cell_26/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_26/split/split_dim
while/lstm_cell_26/splitSplit+while/lstm_cell_26/split/split_dim:output:0#while/lstm_cell_26/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿÚ:ÿÿÿÿÿÿÿÿÿÚ:ÿÿÿÿÿÿÿÿÿÚ:ÿÿÿÿÿÿÿÿÿÚ*
	num_split2
while/lstm_cell_26/split
while/lstm_cell_26/SigmoidSigmoid!while/lstm_cell_26/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
while/lstm_cell_26/Sigmoid
while/lstm_cell_26/Sigmoid_1Sigmoid!while/lstm_cell_26/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
while/lstm_cell_26/Sigmoid_1¡
while/lstm_cell_26/mulMul while/lstm_cell_26/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
while/lstm_cell_26/mul
while/lstm_cell_26/ReluRelu!while/lstm_cell_26/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
while/lstm_cell_26/Reluµ
while/lstm_cell_26/mul_1Mulwhile/lstm_cell_26/Sigmoid:y:0%while/lstm_cell_26/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
while/lstm_cell_26/mul_1ª
while/lstm_cell_26/add_1AddV2while/lstm_cell_26/mul:z:0while/lstm_cell_26/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
while/lstm_cell_26/add_1
while/lstm_cell_26/Sigmoid_2Sigmoid!while/lstm_cell_26/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
while/lstm_cell_26/Sigmoid_2
while/lstm_cell_26/Relu_1Reluwhile/lstm_cell_26/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
while/lstm_cell_26/Relu_1¹
while/lstm_cell_26/mul_2Mul while/lstm_cell_26/Sigmoid_2:y:0'while/lstm_cell_26/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
while/lstm_cell_26/mul_2à
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
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_26/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_26/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
while/Identity_5Þ

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
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿÚ:ÿÿÿÿÿÿÿÿÿÚ: : : : : 2V
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
:ÿÿÿÿÿÿÿÿÿÚ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ:

_output_shapes
: :

_output_shapes
: 
×
g
H__inference_dropout_27_layer_call_and_return_conditional_losses_39973966

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
:ÿÿÿÿÿÿÿÿÿæ2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape¹
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ*
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
:ÿÿÿÿÿÿÿÿÿæ2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2
dropout/Mul_1j
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿæ:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ
 
_user_specified_nameinputs
\

E__inference_lstm_26_layer_call_and_return_conditional_losses_39973699

inputs>
+lstm_cell_26_matmul_readvariableop_resource:	]è
A
-lstm_cell_26_matmul_1_readvariableop_resource:
Úè
;
,lstm_cell_26_biasadd_readvariableop_resource:	è

identity¢#lstm_cell_26/BiasAdd/ReadVariableOp¢"lstm_cell_26/MatMul/ReadVariableOp¢$lstm_cell_26/MatMul_1/ReadVariableOp¢whileD
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
B :Ú2
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
B :Ú2
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
:ÿÿÿÿÿÿÿÿÿÚ2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :Ú2
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
B :Ú2
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
:ÿÿÿÿÿÿÿÿÿÚ2	
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
"lstm_cell_26/MatMul/ReadVariableOpReadVariableOp+lstm_cell_26_matmul_readvariableop_resource*
_output_shapes
:	]è
*
dtype02$
"lstm_cell_26/MatMul/ReadVariableOp­
lstm_cell_26/MatMulMatMulstrided_slice_2:output:0*lstm_cell_26/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè
2
lstm_cell_26/MatMul¼
$lstm_cell_26/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_26_matmul_1_readvariableop_resource* 
_output_shapes
:
Úè
*
dtype02&
$lstm_cell_26/MatMul_1/ReadVariableOp©
lstm_cell_26/MatMul_1MatMulzeros:output:0,lstm_cell_26/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè
2
lstm_cell_26/MatMul_1 
lstm_cell_26/addAddV2lstm_cell_26/MatMul:product:0lstm_cell_26/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè
2
lstm_cell_26/add´
#lstm_cell_26/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_26_biasadd_readvariableop_resource*
_output_shapes	
:è
*
dtype02%
#lstm_cell_26/BiasAdd/ReadVariableOp­
lstm_cell_26/BiasAddBiasAddlstm_cell_26/add:z:0+lstm_cell_26/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè
2
lstm_cell_26/BiasAdd~
lstm_cell_26/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_26/split/split_dim÷
lstm_cell_26/splitSplit%lstm_cell_26/split/split_dim:output:0lstm_cell_26/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿÚ:ÿÿÿÿÿÿÿÿÿÚ:ÿÿÿÿÿÿÿÿÿÚ:ÿÿÿÿÿÿÿÿÿÚ*
	num_split2
lstm_cell_26/split
lstm_cell_26/SigmoidSigmoidlstm_cell_26/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
lstm_cell_26/Sigmoid
lstm_cell_26/Sigmoid_1Sigmoidlstm_cell_26/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
lstm_cell_26/Sigmoid_1
lstm_cell_26/mulMullstm_cell_26/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
lstm_cell_26/mul~
lstm_cell_26/ReluRelulstm_cell_26/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
lstm_cell_26/Relu
lstm_cell_26/mul_1Mullstm_cell_26/Sigmoid:y:0lstm_cell_26/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
lstm_cell_26/mul_1
lstm_cell_26/add_1AddV2lstm_cell_26/mul:z:0lstm_cell_26/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
lstm_cell_26/add_1
lstm_cell_26/Sigmoid_2Sigmoidlstm_cell_26/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
lstm_cell_26/Sigmoid_2}
lstm_cell_26/Relu_1Relulstm_cell_26/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
lstm_cell_26/Relu_1¡
lstm_cell_26/mul_2Mullstm_cell_26/Sigmoid_2:y:0!lstm_cell_26/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
lstm_cell_26/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿZ  2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_26_matmul_readvariableop_resource-lstm_cell_26_matmul_1_readvariableop_resource,lstm_cell_26_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿÚ:ÿÿÿÿÿÿÿÿÿÚ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_39973615*
condR
while_cond_39973614*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿÚ:ÿÿÿÿÿÿÿÿÿÚ: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿZ  22
0TensorArrayV2Stack/TensorListStack/element_shapeé
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ*
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
:ÿÿÿÿÿÿÿÿÿÚ*
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
:ÿÿÿÿÿÿÿÿÿÚ2
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
:ÿÿÿÿÿÿÿÿÿÚ2

IdentityÈ
NoOpNoOp$^lstm_cell_26/BiasAdd/ReadVariableOp#^lstm_cell_26/MatMul/ReadVariableOp%^lstm_cell_26/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ]: : : 2J
#lstm_cell_26/BiasAdd/ReadVariableOp#lstm_cell_26/BiasAdd/ReadVariableOp2H
"lstm_cell_26/MatMul/ReadVariableOp"lstm_cell_26/MatMul/ReadVariableOp2L
$lstm_cell_26/MatMul_1/ReadVariableOp$lstm_cell_26/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
 
_user_specified_nameinputs
¨¹
ç	
#__inference__wrapped_model_39972282
lstm_26_inputT
Asequential_13_lstm_26_lstm_cell_26_matmul_readvariableop_resource:	]è
W
Csequential_13_lstm_26_lstm_cell_26_matmul_1_readvariableop_resource:
Úè
Q
Bsequential_13_lstm_26_lstm_cell_26_biasadd_readvariableop_resource:	è
U
Asequential_13_lstm_27_lstm_cell_27_matmul_readvariableop_resource:
ÚW
Csequential_13_lstm_27_lstm_cell_27_matmul_1_readvariableop_resource:
æQ
Bsequential_13_lstm_27_lstm_cell_27_biasadd_readvariableop_resource:	K
8sequential_13_dense_13_tensordot_readvariableop_resource:	æD
6sequential_13_dense_13_biasadd_readvariableop_resource:
identity¢-sequential_13/dense_13/BiasAdd/ReadVariableOp¢/sequential_13/dense_13/Tensordot/ReadVariableOp¢9sequential_13/lstm_26/lstm_cell_26/BiasAdd/ReadVariableOp¢8sequential_13/lstm_26/lstm_cell_26/MatMul/ReadVariableOp¢:sequential_13/lstm_26/lstm_cell_26/MatMul_1/ReadVariableOp¢sequential_13/lstm_26/while¢9sequential_13/lstm_27/lstm_cell_27/BiasAdd/ReadVariableOp¢8sequential_13/lstm_27/lstm_cell_27/MatMul/ReadVariableOp¢:sequential_13/lstm_27/lstm_cell_27/MatMul_1/ReadVariableOp¢sequential_13/lstm_27/whilew
sequential_13/lstm_26/ShapeShapelstm_26_input*
T0*
_output_shapes
:2
sequential_13/lstm_26/Shape 
)sequential_13/lstm_26/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)sequential_13/lstm_26/strided_slice/stack¤
+sequential_13/lstm_26/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential_13/lstm_26/strided_slice/stack_1¤
+sequential_13/lstm_26/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential_13/lstm_26/strided_slice/stack_2æ
#sequential_13/lstm_26/strided_sliceStridedSlice$sequential_13/lstm_26/Shape:output:02sequential_13/lstm_26/strided_slice/stack:output:04sequential_13/lstm_26/strided_slice/stack_1:output:04sequential_13/lstm_26/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#sequential_13/lstm_26/strided_slice
!sequential_13/lstm_26/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :Ú2#
!sequential_13/lstm_26/zeros/mul/yÄ
sequential_13/lstm_26/zeros/mulMul,sequential_13/lstm_26/strided_slice:output:0*sequential_13/lstm_26/zeros/mul/y:output:0*
T0*
_output_shapes
: 2!
sequential_13/lstm_26/zeros/mul
"sequential_13/lstm_26/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2$
"sequential_13/lstm_26/zeros/Less/y¿
 sequential_13/lstm_26/zeros/LessLess#sequential_13/lstm_26/zeros/mul:z:0+sequential_13/lstm_26/zeros/Less/y:output:0*
T0*
_output_shapes
: 2"
 sequential_13/lstm_26/zeros/Less
$sequential_13/lstm_26/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :Ú2&
$sequential_13/lstm_26/zeros/packed/1Û
"sequential_13/lstm_26/zeros/packedPack,sequential_13/lstm_26/strided_slice:output:0-sequential_13/lstm_26/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2$
"sequential_13/lstm_26/zeros/packed
!sequential_13/lstm_26/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!sequential_13/lstm_26/zeros/ConstÎ
sequential_13/lstm_26/zerosFill+sequential_13/lstm_26/zeros/packed:output:0*sequential_13/lstm_26/zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
sequential_13/lstm_26/zeros
#sequential_13/lstm_26/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :Ú2%
#sequential_13/lstm_26/zeros_1/mul/yÊ
!sequential_13/lstm_26/zeros_1/mulMul,sequential_13/lstm_26/strided_slice:output:0,sequential_13/lstm_26/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2#
!sequential_13/lstm_26/zeros_1/mul
$sequential_13/lstm_26/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2&
$sequential_13/lstm_26/zeros_1/Less/yÇ
"sequential_13/lstm_26/zeros_1/LessLess%sequential_13/lstm_26/zeros_1/mul:z:0-sequential_13/lstm_26/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2$
"sequential_13/lstm_26/zeros_1/Less
&sequential_13/lstm_26/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :Ú2(
&sequential_13/lstm_26/zeros_1/packed/1á
$sequential_13/lstm_26/zeros_1/packedPack,sequential_13/lstm_26/strided_slice:output:0/sequential_13/lstm_26/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2&
$sequential_13/lstm_26/zeros_1/packed
#sequential_13/lstm_26/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2%
#sequential_13/lstm_26/zeros_1/ConstÖ
sequential_13/lstm_26/zeros_1Fill-sequential_13/lstm_26/zeros_1/packed:output:0,sequential_13/lstm_26/zeros_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
sequential_13/lstm_26/zeros_1¡
$sequential_13/lstm_26/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2&
$sequential_13/lstm_26/transpose/permÃ
sequential_13/lstm_26/transpose	Transposelstm_26_input-sequential_13/lstm_26/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]2!
sequential_13/lstm_26/transpose
sequential_13/lstm_26/Shape_1Shape#sequential_13/lstm_26/transpose:y:0*
T0*
_output_shapes
:2
sequential_13/lstm_26/Shape_1¤
+sequential_13/lstm_26/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2-
+sequential_13/lstm_26/strided_slice_1/stack¨
-sequential_13/lstm_26/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_13/lstm_26/strided_slice_1/stack_1¨
-sequential_13/lstm_26/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_13/lstm_26/strided_slice_1/stack_2ò
%sequential_13/lstm_26/strided_slice_1StridedSlice&sequential_13/lstm_26/Shape_1:output:04sequential_13/lstm_26/strided_slice_1/stack:output:06sequential_13/lstm_26/strided_slice_1/stack_1:output:06sequential_13/lstm_26/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%sequential_13/lstm_26/strided_slice_1±
1sequential_13/lstm_26/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ23
1sequential_13/lstm_26/TensorArrayV2/element_shape
#sequential_13/lstm_26/TensorArrayV2TensorListReserve:sequential_13/lstm_26/TensorArrayV2/element_shape:output:0.sequential_13/lstm_26/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02%
#sequential_13/lstm_26/TensorArrayV2ë
Ksequential_13/lstm_26/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ]   2M
Ksequential_13/lstm_26/TensorArrayUnstack/TensorListFromTensor/element_shapeÐ
=sequential_13/lstm_26/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor#sequential_13/lstm_26/transpose:y:0Tsequential_13/lstm_26/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02?
=sequential_13/lstm_26/TensorArrayUnstack/TensorListFromTensor¤
+sequential_13/lstm_26/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2-
+sequential_13/lstm_26/strided_slice_2/stack¨
-sequential_13/lstm_26/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_13/lstm_26/strided_slice_2/stack_1¨
-sequential_13/lstm_26/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_13/lstm_26/strided_slice_2/stack_2
%sequential_13/lstm_26/strided_slice_2StridedSlice#sequential_13/lstm_26/transpose:y:04sequential_13/lstm_26/strided_slice_2/stack:output:06sequential_13/lstm_26/strided_slice_2/stack_1:output:06sequential_13/lstm_26/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]*
shrink_axis_mask2'
%sequential_13/lstm_26/strided_slice_2÷
8sequential_13/lstm_26/lstm_cell_26/MatMul/ReadVariableOpReadVariableOpAsequential_13_lstm_26_lstm_cell_26_matmul_readvariableop_resource*
_output_shapes
:	]è
*
dtype02:
8sequential_13/lstm_26/lstm_cell_26/MatMul/ReadVariableOp
)sequential_13/lstm_26/lstm_cell_26/MatMulMatMul.sequential_13/lstm_26/strided_slice_2:output:0@sequential_13/lstm_26/lstm_cell_26/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè
2+
)sequential_13/lstm_26/lstm_cell_26/MatMulþ
:sequential_13/lstm_26/lstm_cell_26/MatMul_1/ReadVariableOpReadVariableOpCsequential_13_lstm_26_lstm_cell_26_matmul_1_readvariableop_resource* 
_output_shapes
:
Úè
*
dtype02<
:sequential_13/lstm_26/lstm_cell_26/MatMul_1/ReadVariableOp
+sequential_13/lstm_26/lstm_cell_26/MatMul_1MatMul$sequential_13/lstm_26/zeros:output:0Bsequential_13/lstm_26/lstm_cell_26/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè
2-
+sequential_13/lstm_26/lstm_cell_26/MatMul_1ø
&sequential_13/lstm_26/lstm_cell_26/addAddV23sequential_13/lstm_26/lstm_cell_26/MatMul:product:05sequential_13/lstm_26/lstm_cell_26/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè
2(
&sequential_13/lstm_26/lstm_cell_26/addö
9sequential_13/lstm_26/lstm_cell_26/BiasAdd/ReadVariableOpReadVariableOpBsequential_13_lstm_26_lstm_cell_26_biasadd_readvariableop_resource*
_output_shapes	
:è
*
dtype02;
9sequential_13/lstm_26/lstm_cell_26/BiasAdd/ReadVariableOp
*sequential_13/lstm_26/lstm_cell_26/BiasAddBiasAdd*sequential_13/lstm_26/lstm_cell_26/add:z:0Asequential_13/lstm_26/lstm_cell_26/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè
2,
*sequential_13/lstm_26/lstm_cell_26/BiasAddª
2sequential_13/lstm_26/lstm_cell_26/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :24
2sequential_13/lstm_26/lstm_cell_26/split/split_dimÏ
(sequential_13/lstm_26/lstm_cell_26/splitSplit;sequential_13/lstm_26/lstm_cell_26/split/split_dim:output:03sequential_13/lstm_26/lstm_cell_26/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿÚ:ÿÿÿÿÿÿÿÿÿÚ:ÿÿÿÿÿÿÿÿÿÚ:ÿÿÿÿÿÿÿÿÿÚ*
	num_split2*
(sequential_13/lstm_26/lstm_cell_26/splitÉ
*sequential_13/lstm_26/lstm_cell_26/SigmoidSigmoid1sequential_13/lstm_26/lstm_cell_26/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2,
*sequential_13/lstm_26/lstm_cell_26/SigmoidÍ
,sequential_13/lstm_26/lstm_cell_26/Sigmoid_1Sigmoid1sequential_13/lstm_26/lstm_cell_26/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2.
,sequential_13/lstm_26/lstm_cell_26/Sigmoid_1ä
&sequential_13/lstm_26/lstm_cell_26/mulMul0sequential_13/lstm_26/lstm_cell_26/Sigmoid_1:y:0&sequential_13/lstm_26/zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2(
&sequential_13/lstm_26/lstm_cell_26/mulÀ
'sequential_13/lstm_26/lstm_cell_26/ReluRelu1sequential_13/lstm_26/lstm_cell_26/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2)
'sequential_13/lstm_26/lstm_cell_26/Reluõ
(sequential_13/lstm_26/lstm_cell_26/mul_1Mul.sequential_13/lstm_26/lstm_cell_26/Sigmoid:y:05sequential_13/lstm_26/lstm_cell_26/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2*
(sequential_13/lstm_26/lstm_cell_26/mul_1ê
(sequential_13/lstm_26/lstm_cell_26/add_1AddV2*sequential_13/lstm_26/lstm_cell_26/mul:z:0,sequential_13/lstm_26/lstm_cell_26/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2*
(sequential_13/lstm_26/lstm_cell_26/add_1Í
,sequential_13/lstm_26/lstm_cell_26/Sigmoid_2Sigmoid1sequential_13/lstm_26/lstm_cell_26/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2.
,sequential_13/lstm_26/lstm_cell_26/Sigmoid_2¿
)sequential_13/lstm_26/lstm_cell_26/Relu_1Relu,sequential_13/lstm_26/lstm_cell_26/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2+
)sequential_13/lstm_26/lstm_cell_26/Relu_1ù
(sequential_13/lstm_26/lstm_cell_26/mul_2Mul0sequential_13/lstm_26/lstm_cell_26/Sigmoid_2:y:07sequential_13/lstm_26/lstm_cell_26/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2*
(sequential_13/lstm_26/lstm_cell_26/mul_2»
3sequential_13/lstm_26/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿZ  25
3sequential_13/lstm_26/TensorArrayV2_1/element_shape
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
sequential_13/lstm_26/time«
.sequential_13/lstm_26/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ20
.sequential_13/lstm_26/while/maximum_iterations
(sequential_13/lstm_26/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2*
(sequential_13/lstm_26/while/loop_counterÝ
sequential_13/lstm_26/whileWhile1sequential_13/lstm_26/while/loop_counter:output:07sequential_13/lstm_26/while/maximum_iterations:output:0#sequential_13/lstm_26/time:output:0.sequential_13/lstm_26/TensorArrayV2_1:handle:0$sequential_13/lstm_26/zeros:output:0&sequential_13/lstm_26/zeros_1:output:0.sequential_13/lstm_26/strided_slice_1:output:0Msequential_13/lstm_26/TensorArrayUnstack/TensorListFromTensor:output_handle:0Asequential_13_lstm_26_lstm_cell_26_matmul_readvariableop_resourceCsequential_13_lstm_26_lstm_cell_26_matmul_1_readvariableop_resourceBsequential_13_lstm_26_lstm_cell_26_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿÚ:ÿÿÿÿÿÿÿÿÿÚ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *5
body-R+
)sequential_13_lstm_26_while_body_39972022*5
cond-R+
)sequential_13_lstm_26_while_cond_39972021*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿÚ:ÿÿÿÿÿÿÿÿÿÚ: : : : : *
parallel_iterations 2
sequential_13/lstm_26/whileá
Fsequential_13/lstm_26/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿZ  2H
Fsequential_13/lstm_26/TensorArrayV2Stack/TensorListStack/element_shapeÁ
8sequential_13/lstm_26/TensorArrayV2Stack/TensorListStackTensorListStack$sequential_13/lstm_26/while:output:3Osequential_13/lstm_26/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ*
element_dtype02:
8sequential_13/lstm_26/TensorArrayV2Stack/TensorListStack­
+sequential_13/lstm_26/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2-
+sequential_13/lstm_26/strided_slice_3/stack¨
-sequential_13/lstm_26/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2/
-sequential_13/lstm_26/strided_slice_3/stack_1¨
-sequential_13/lstm_26/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_13/lstm_26/strided_slice_3/stack_2
%sequential_13/lstm_26/strided_slice_3StridedSliceAsequential_13/lstm_26/TensorArrayV2Stack/TensorListStack:tensor:04sequential_13/lstm_26/strided_slice_3/stack:output:06sequential_13/lstm_26/strided_slice_3/stack_1:output:06sequential_13/lstm_26/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ*
shrink_axis_mask2'
%sequential_13/lstm_26/strided_slice_3¥
&sequential_13/lstm_26/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2(
&sequential_13/lstm_26/transpose_1/permþ
!sequential_13/lstm_26/transpose_1	TransposeAsequential_13/lstm_26/TensorArrayV2Stack/TensorListStack:tensor:0/sequential_13/lstm_26/transpose_1/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2#
!sequential_13/lstm_26/transpose_1
sequential_13/lstm_26/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
sequential_13/lstm_26/runtime°
!sequential_13/dropout_26/IdentityIdentity%sequential_13/lstm_26/transpose_1:y:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2#
!sequential_13/dropout_26/Identity
sequential_13/lstm_27/ShapeShape*sequential_13/dropout_26/Identity:output:0*
T0*
_output_shapes
:2
sequential_13/lstm_27/Shape 
)sequential_13/lstm_27/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)sequential_13/lstm_27/strided_slice/stack¤
+sequential_13/lstm_27/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential_13/lstm_27/strided_slice/stack_1¤
+sequential_13/lstm_27/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential_13/lstm_27/strided_slice/stack_2æ
#sequential_13/lstm_27/strided_sliceStridedSlice$sequential_13/lstm_27/Shape:output:02sequential_13/lstm_27/strided_slice/stack:output:04sequential_13/lstm_27/strided_slice/stack_1:output:04sequential_13/lstm_27/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#sequential_13/lstm_27/strided_slice
!sequential_13/lstm_27/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :æ2#
!sequential_13/lstm_27/zeros/mul/yÄ
sequential_13/lstm_27/zeros/mulMul,sequential_13/lstm_27/strided_slice:output:0*sequential_13/lstm_27/zeros/mul/y:output:0*
T0*
_output_shapes
: 2!
sequential_13/lstm_27/zeros/mul
"sequential_13/lstm_27/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2$
"sequential_13/lstm_27/zeros/Less/y¿
 sequential_13/lstm_27/zeros/LessLess#sequential_13/lstm_27/zeros/mul:z:0+sequential_13/lstm_27/zeros/Less/y:output:0*
T0*
_output_shapes
: 2"
 sequential_13/lstm_27/zeros/Less
$sequential_13/lstm_27/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :æ2&
$sequential_13/lstm_27/zeros/packed/1Û
"sequential_13/lstm_27/zeros/packedPack,sequential_13/lstm_27/strided_slice:output:0-sequential_13/lstm_27/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2$
"sequential_13/lstm_27/zeros/packed
!sequential_13/lstm_27/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!sequential_13/lstm_27/zeros/ConstÎ
sequential_13/lstm_27/zerosFill+sequential_13/lstm_27/zeros/packed:output:0*sequential_13/lstm_27/zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2
sequential_13/lstm_27/zeros
#sequential_13/lstm_27/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :æ2%
#sequential_13/lstm_27/zeros_1/mul/yÊ
!sequential_13/lstm_27/zeros_1/mulMul,sequential_13/lstm_27/strided_slice:output:0,sequential_13/lstm_27/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2#
!sequential_13/lstm_27/zeros_1/mul
$sequential_13/lstm_27/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2&
$sequential_13/lstm_27/zeros_1/Less/yÇ
"sequential_13/lstm_27/zeros_1/LessLess%sequential_13/lstm_27/zeros_1/mul:z:0-sequential_13/lstm_27/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2$
"sequential_13/lstm_27/zeros_1/Less
&sequential_13/lstm_27/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :æ2(
&sequential_13/lstm_27/zeros_1/packed/1á
$sequential_13/lstm_27/zeros_1/packedPack,sequential_13/lstm_27/strided_slice:output:0/sequential_13/lstm_27/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2&
$sequential_13/lstm_27/zeros_1/packed
#sequential_13/lstm_27/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2%
#sequential_13/lstm_27/zeros_1/ConstÖ
sequential_13/lstm_27/zeros_1Fill-sequential_13/lstm_27/zeros_1/packed:output:0,sequential_13/lstm_27/zeros_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2
sequential_13/lstm_27/zeros_1¡
$sequential_13/lstm_27/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2&
$sequential_13/lstm_27/transpose/permá
sequential_13/lstm_27/transpose	Transpose*sequential_13/dropout_26/Identity:output:0-sequential_13/lstm_27/transpose/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2!
sequential_13/lstm_27/transpose
sequential_13/lstm_27/Shape_1Shape#sequential_13/lstm_27/transpose:y:0*
T0*
_output_shapes
:2
sequential_13/lstm_27/Shape_1¤
+sequential_13/lstm_27/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2-
+sequential_13/lstm_27/strided_slice_1/stack¨
-sequential_13/lstm_27/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_13/lstm_27/strided_slice_1/stack_1¨
-sequential_13/lstm_27/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_13/lstm_27/strided_slice_1/stack_2ò
%sequential_13/lstm_27/strided_slice_1StridedSlice&sequential_13/lstm_27/Shape_1:output:04sequential_13/lstm_27/strided_slice_1/stack:output:06sequential_13/lstm_27/strided_slice_1/stack_1:output:06sequential_13/lstm_27/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%sequential_13/lstm_27/strided_slice_1±
1sequential_13/lstm_27/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ23
1sequential_13/lstm_27/TensorArrayV2/element_shape
#sequential_13/lstm_27/TensorArrayV2TensorListReserve:sequential_13/lstm_27/TensorArrayV2/element_shape:output:0.sequential_13/lstm_27/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02%
#sequential_13/lstm_27/TensorArrayV2ë
Ksequential_13/lstm_27/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿZ  2M
Ksequential_13/lstm_27/TensorArrayUnstack/TensorListFromTensor/element_shapeÐ
=sequential_13/lstm_27/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor#sequential_13/lstm_27/transpose:y:0Tsequential_13/lstm_27/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02?
=sequential_13/lstm_27/TensorArrayUnstack/TensorListFromTensor¤
+sequential_13/lstm_27/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2-
+sequential_13/lstm_27/strided_slice_2/stack¨
-sequential_13/lstm_27/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_13/lstm_27/strided_slice_2/stack_1¨
-sequential_13/lstm_27/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_13/lstm_27/strided_slice_2/stack_2
%sequential_13/lstm_27/strided_slice_2StridedSlice#sequential_13/lstm_27/transpose:y:04sequential_13/lstm_27/strided_slice_2/stack:output:06sequential_13/lstm_27/strided_slice_2/stack_1:output:06sequential_13/lstm_27/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ*
shrink_axis_mask2'
%sequential_13/lstm_27/strided_slice_2ø
8sequential_13/lstm_27/lstm_cell_27/MatMul/ReadVariableOpReadVariableOpAsequential_13_lstm_27_lstm_cell_27_matmul_readvariableop_resource* 
_output_shapes
:
Ú*
dtype02:
8sequential_13/lstm_27/lstm_cell_27/MatMul/ReadVariableOp
)sequential_13/lstm_27/lstm_cell_27/MatMulMatMul.sequential_13/lstm_27/strided_slice_2:output:0@sequential_13/lstm_27/lstm_cell_27/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2+
)sequential_13/lstm_27/lstm_cell_27/MatMulþ
:sequential_13/lstm_27/lstm_cell_27/MatMul_1/ReadVariableOpReadVariableOpCsequential_13_lstm_27_lstm_cell_27_matmul_1_readvariableop_resource* 
_output_shapes
:
æ*
dtype02<
:sequential_13/lstm_27/lstm_cell_27/MatMul_1/ReadVariableOp
+sequential_13/lstm_27/lstm_cell_27/MatMul_1MatMul$sequential_13/lstm_27/zeros:output:0Bsequential_13/lstm_27/lstm_cell_27/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2-
+sequential_13/lstm_27/lstm_cell_27/MatMul_1ø
&sequential_13/lstm_27/lstm_cell_27/addAddV23sequential_13/lstm_27/lstm_cell_27/MatMul:product:05sequential_13/lstm_27/lstm_cell_27/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&sequential_13/lstm_27/lstm_cell_27/addö
9sequential_13/lstm_27/lstm_cell_27/BiasAdd/ReadVariableOpReadVariableOpBsequential_13_lstm_27_lstm_cell_27_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02;
9sequential_13/lstm_27/lstm_cell_27/BiasAdd/ReadVariableOp
*sequential_13/lstm_27/lstm_cell_27/BiasAddBiasAdd*sequential_13/lstm_27/lstm_cell_27/add:z:0Asequential_13/lstm_27/lstm_cell_27/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2,
*sequential_13/lstm_27/lstm_cell_27/BiasAddª
2sequential_13/lstm_27/lstm_cell_27/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :24
2sequential_13/lstm_27/lstm_cell_27/split/split_dimÏ
(sequential_13/lstm_27/lstm_cell_27/splitSplit;sequential_13/lstm_27/lstm_cell_27/split/split_dim:output:03sequential_13/lstm_27/lstm_cell_27/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿæ:ÿÿÿÿÿÿÿÿÿæ:ÿÿÿÿÿÿÿÿÿæ:ÿÿÿÿÿÿÿÿÿæ*
	num_split2*
(sequential_13/lstm_27/lstm_cell_27/splitÉ
*sequential_13/lstm_27/lstm_cell_27/SigmoidSigmoid1sequential_13/lstm_27/lstm_cell_27/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2,
*sequential_13/lstm_27/lstm_cell_27/SigmoidÍ
,sequential_13/lstm_27/lstm_cell_27/Sigmoid_1Sigmoid1sequential_13/lstm_27/lstm_cell_27/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2.
,sequential_13/lstm_27/lstm_cell_27/Sigmoid_1ä
&sequential_13/lstm_27/lstm_cell_27/mulMul0sequential_13/lstm_27/lstm_cell_27/Sigmoid_1:y:0&sequential_13/lstm_27/zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2(
&sequential_13/lstm_27/lstm_cell_27/mulÀ
'sequential_13/lstm_27/lstm_cell_27/ReluRelu1sequential_13/lstm_27/lstm_cell_27/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2)
'sequential_13/lstm_27/lstm_cell_27/Reluõ
(sequential_13/lstm_27/lstm_cell_27/mul_1Mul.sequential_13/lstm_27/lstm_cell_27/Sigmoid:y:05sequential_13/lstm_27/lstm_cell_27/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2*
(sequential_13/lstm_27/lstm_cell_27/mul_1ê
(sequential_13/lstm_27/lstm_cell_27/add_1AddV2*sequential_13/lstm_27/lstm_cell_27/mul:z:0,sequential_13/lstm_27/lstm_cell_27/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2*
(sequential_13/lstm_27/lstm_cell_27/add_1Í
,sequential_13/lstm_27/lstm_cell_27/Sigmoid_2Sigmoid1sequential_13/lstm_27/lstm_cell_27/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2.
,sequential_13/lstm_27/lstm_cell_27/Sigmoid_2¿
)sequential_13/lstm_27/lstm_cell_27/Relu_1Relu,sequential_13/lstm_27/lstm_cell_27/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2+
)sequential_13/lstm_27/lstm_cell_27/Relu_1ù
(sequential_13/lstm_27/lstm_cell_27/mul_2Mul0sequential_13/lstm_27/lstm_cell_27/Sigmoid_2:y:07sequential_13/lstm_27/lstm_cell_27/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2*
(sequential_13/lstm_27/lstm_cell_27/mul_2»
3sequential_13/lstm_27/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿf  25
3sequential_13/lstm_27/TensorArrayV2_1/element_shape
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
sequential_13/lstm_27/time«
.sequential_13/lstm_27/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ20
.sequential_13/lstm_27/while/maximum_iterations
(sequential_13/lstm_27/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2*
(sequential_13/lstm_27/while/loop_counterÝ
sequential_13/lstm_27/whileWhile1sequential_13/lstm_27/while/loop_counter:output:07sequential_13/lstm_27/while/maximum_iterations:output:0#sequential_13/lstm_27/time:output:0.sequential_13/lstm_27/TensorArrayV2_1:handle:0$sequential_13/lstm_27/zeros:output:0&sequential_13/lstm_27/zeros_1:output:0.sequential_13/lstm_27/strided_slice_1:output:0Msequential_13/lstm_27/TensorArrayUnstack/TensorListFromTensor:output_handle:0Asequential_13_lstm_27_lstm_cell_27_matmul_readvariableop_resourceCsequential_13_lstm_27_lstm_cell_27_matmul_1_readvariableop_resourceBsequential_13_lstm_27_lstm_cell_27_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿæ:ÿÿÿÿÿÿÿÿÿæ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *5
body-R+
)sequential_13_lstm_27_while_body_39972170*5
cond-R+
)sequential_13_lstm_27_while_cond_39972169*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿæ:ÿÿÿÿÿÿÿÿÿæ: : : : : *
parallel_iterations 2
sequential_13/lstm_27/whileá
Fsequential_13/lstm_27/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿf  2H
Fsequential_13/lstm_27/TensorArrayV2Stack/TensorListStack/element_shapeÁ
8sequential_13/lstm_27/TensorArrayV2Stack/TensorListStackTensorListStack$sequential_13/lstm_27/while:output:3Osequential_13/lstm_27/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ*
element_dtype02:
8sequential_13/lstm_27/TensorArrayV2Stack/TensorListStack­
+sequential_13/lstm_27/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2-
+sequential_13/lstm_27/strided_slice_3/stack¨
-sequential_13/lstm_27/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2/
-sequential_13/lstm_27/strided_slice_3/stack_1¨
-sequential_13/lstm_27/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_13/lstm_27/strided_slice_3/stack_2
%sequential_13/lstm_27/strided_slice_3StridedSliceAsequential_13/lstm_27/TensorArrayV2Stack/TensorListStack:tensor:04sequential_13/lstm_27/strided_slice_3/stack:output:06sequential_13/lstm_27/strided_slice_3/stack_1:output:06sequential_13/lstm_27/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ*
shrink_axis_mask2'
%sequential_13/lstm_27/strided_slice_3¥
&sequential_13/lstm_27/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2(
&sequential_13/lstm_27/transpose_1/permþ
!sequential_13/lstm_27/transpose_1	TransposeAsequential_13/lstm_27/TensorArrayV2Stack/TensorListStack:tensor:0/sequential_13/lstm_27/transpose_1/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2#
!sequential_13/lstm_27/transpose_1
sequential_13/lstm_27/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
sequential_13/lstm_27/runtime°
!sequential_13/dropout_27/IdentityIdentity%sequential_13/lstm_27/transpose_1:y:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2#
!sequential_13/dropout_27/IdentityÜ
/sequential_13/dense_13/Tensordot/ReadVariableOpReadVariableOp8sequential_13_dense_13_tensordot_readvariableop_resource*
_output_shapes
:	æ*
dtype021
/sequential_13/dense_13/Tensordot/ReadVariableOp
%sequential_13/dense_13/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2'
%sequential_13/dense_13/Tensordot/axes
%sequential_13/dense_13/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2'
%sequential_13/dense_13/Tensordot/freeª
&sequential_13/dense_13/Tensordot/ShapeShape*sequential_13/dropout_27/Identity:output:0*
T0*
_output_shapes
:2(
&sequential_13/dense_13/Tensordot/Shape¢
.sequential_13/dense_13/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.sequential_13/dense_13/Tensordot/GatherV2/axisÄ
)sequential_13/dense_13/Tensordot/GatherV2GatherV2/sequential_13/dense_13/Tensordot/Shape:output:0.sequential_13/dense_13/Tensordot/free:output:07sequential_13/dense_13/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2+
)sequential_13/dense_13/Tensordot/GatherV2¦
0sequential_13/dense_13/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 22
0sequential_13/dense_13/Tensordot/GatherV2_1/axisÊ
+sequential_13/dense_13/Tensordot/GatherV2_1GatherV2/sequential_13/dense_13/Tensordot/Shape:output:0.sequential_13/dense_13/Tensordot/axes:output:09sequential_13/dense_13/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2-
+sequential_13/dense_13/Tensordot/GatherV2_1
&sequential_13/dense_13/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2(
&sequential_13/dense_13/Tensordot/ConstÜ
%sequential_13/dense_13/Tensordot/ProdProd2sequential_13/dense_13/Tensordot/GatherV2:output:0/sequential_13/dense_13/Tensordot/Const:output:0*
T0*
_output_shapes
: 2'
%sequential_13/dense_13/Tensordot/Prod
(sequential_13/dense_13/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(sequential_13/dense_13/Tensordot/Const_1ä
'sequential_13/dense_13/Tensordot/Prod_1Prod4sequential_13/dense_13/Tensordot/GatherV2_1:output:01sequential_13/dense_13/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2)
'sequential_13/dense_13/Tensordot/Prod_1
,sequential_13/dense_13/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,sequential_13/dense_13/Tensordot/concat/axis£
'sequential_13/dense_13/Tensordot/concatConcatV2.sequential_13/dense_13/Tensordot/free:output:0.sequential_13/dense_13/Tensordot/axes:output:05sequential_13/dense_13/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2)
'sequential_13/dense_13/Tensordot/concatè
&sequential_13/dense_13/Tensordot/stackPack.sequential_13/dense_13/Tensordot/Prod:output:00sequential_13/dense_13/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2(
&sequential_13/dense_13/Tensordot/stackú
*sequential_13/dense_13/Tensordot/transpose	Transpose*sequential_13/dropout_27/Identity:output:00sequential_13/dense_13/Tensordot/concat:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2,
*sequential_13/dense_13/Tensordot/transposeû
(sequential_13/dense_13/Tensordot/ReshapeReshape.sequential_13/dense_13/Tensordot/transpose:y:0/sequential_13/dense_13/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2*
(sequential_13/dense_13/Tensordot/Reshapeú
'sequential_13/dense_13/Tensordot/MatMulMatMul1sequential_13/dense_13/Tensordot/Reshape:output:07sequential_13/dense_13/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'sequential_13/dense_13/Tensordot/MatMul
(sequential_13/dense_13/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(sequential_13/dense_13/Tensordot/Const_2¢
.sequential_13/dense_13/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.sequential_13/dense_13/Tensordot/concat_1/axis°
)sequential_13/dense_13/Tensordot/concat_1ConcatV22sequential_13/dense_13/Tensordot/GatherV2:output:01sequential_13/dense_13/Tensordot/Const_2:output:07sequential_13/dense_13/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2+
)sequential_13/dense_13/Tensordot/concat_1ì
 sequential_13/dense_13/TensordotReshape1sequential_13/dense_13/Tensordot/MatMul:product:02sequential_13/dense_13/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 sequential_13/dense_13/TensordotÑ
-sequential_13/dense_13/BiasAdd/ReadVariableOpReadVariableOp6sequential_13_dense_13_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02/
-sequential_13/dense_13/BiasAdd/ReadVariableOpã
sequential_13/dense_13/BiasAddBiasAdd)sequential_13/dense_13/Tensordot:output:05sequential_13/dense_13/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
sequential_13/dense_13/BiasAddª
sequential_13/dense_13/SoftmaxSoftmax'sequential_13/dense_13/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
sequential_13/dense_13/Softmax
IdentityIdentity(sequential_13/dense_13/Softmax:softmax:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

IdentityÔ
NoOpNoOp.^sequential_13/dense_13/BiasAdd/ReadVariableOp0^sequential_13/dense_13/Tensordot/ReadVariableOp:^sequential_13/lstm_26/lstm_cell_26/BiasAdd/ReadVariableOp9^sequential_13/lstm_26/lstm_cell_26/MatMul/ReadVariableOp;^sequential_13/lstm_26/lstm_cell_26/MatMul_1/ReadVariableOp^sequential_13/lstm_26/while:^sequential_13/lstm_27/lstm_cell_27/BiasAdd/ReadVariableOp9^sequential_13/lstm_27/lstm_cell_27/MatMul/ReadVariableOp;^sequential_13/lstm_27/lstm_cell_27/MatMul_1/ReadVariableOp^sequential_13/lstm_27/while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ]: : : : : : : : 2^
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
:ÿÿÿÿÿÿÿÿÿ]
'
_user_specified_namelstm_26_input
\

E__inference_lstm_26_layer_call_and_return_conditional_losses_39975857

inputs>
+lstm_cell_26_matmul_readvariableop_resource:	]è
A
-lstm_cell_26_matmul_1_readvariableop_resource:
Úè
;
,lstm_cell_26_biasadd_readvariableop_resource:	è

identity¢#lstm_cell_26/BiasAdd/ReadVariableOp¢"lstm_cell_26/MatMul/ReadVariableOp¢$lstm_cell_26/MatMul_1/ReadVariableOp¢whileD
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
B :Ú2
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
B :Ú2
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
:ÿÿÿÿÿÿÿÿÿÚ2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :Ú2
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
B :Ú2
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
:ÿÿÿÿÿÿÿÿÿÚ2	
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
"lstm_cell_26/MatMul/ReadVariableOpReadVariableOp+lstm_cell_26_matmul_readvariableop_resource*
_output_shapes
:	]è
*
dtype02$
"lstm_cell_26/MatMul/ReadVariableOp­
lstm_cell_26/MatMulMatMulstrided_slice_2:output:0*lstm_cell_26/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè
2
lstm_cell_26/MatMul¼
$lstm_cell_26/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_26_matmul_1_readvariableop_resource* 
_output_shapes
:
Úè
*
dtype02&
$lstm_cell_26/MatMul_1/ReadVariableOp©
lstm_cell_26/MatMul_1MatMulzeros:output:0,lstm_cell_26/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè
2
lstm_cell_26/MatMul_1 
lstm_cell_26/addAddV2lstm_cell_26/MatMul:product:0lstm_cell_26/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè
2
lstm_cell_26/add´
#lstm_cell_26/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_26_biasadd_readvariableop_resource*
_output_shapes	
:è
*
dtype02%
#lstm_cell_26/BiasAdd/ReadVariableOp­
lstm_cell_26/BiasAddBiasAddlstm_cell_26/add:z:0+lstm_cell_26/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè
2
lstm_cell_26/BiasAdd~
lstm_cell_26/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_26/split/split_dim÷
lstm_cell_26/splitSplit%lstm_cell_26/split/split_dim:output:0lstm_cell_26/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿÚ:ÿÿÿÿÿÿÿÿÿÚ:ÿÿÿÿÿÿÿÿÿÚ:ÿÿÿÿÿÿÿÿÿÚ*
	num_split2
lstm_cell_26/split
lstm_cell_26/SigmoidSigmoidlstm_cell_26/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
lstm_cell_26/Sigmoid
lstm_cell_26/Sigmoid_1Sigmoidlstm_cell_26/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
lstm_cell_26/Sigmoid_1
lstm_cell_26/mulMullstm_cell_26/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
lstm_cell_26/mul~
lstm_cell_26/ReluRelulstm_cell_26/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
lstm_cell_26/Relu
lstm_cell_26/mul_1Mullstm_cell_26/Sigmoid:y:0lstm_cell_26/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
lstm_cell_26/mul_1
lstm_cell_26/add_1AddV2lstm_cell_26/mul:z:0lstm_cell_26/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
lstm_cell_26/add_1
lstm_cell_26/Sigmoid_2Sigmoidlstm_cell_26/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
lstm_cell_26/Sigmoid_2}
lstm_cell_26/Relu_1Relulstm_cell_26/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
lstm_cell_26/Relu_1¡
lstm_cell_26/mul_2Mullstm_cell_26/Sigmoid_2:y:0!lstm_cell_26/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
lstm_cell_26/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿZ  2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_26_matmul_readvariableop_resource-lstm_cell_26_matmul_1_readvariableop_resource,lstm_cell_26_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿÚ:ÿÿÿÿÿÿÿÿÿÚ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_39975773*
condR
while_cond_39975772*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿÚ:ÿÿÿÿÿÿÿÿÿÚ: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿZ  22
0TensorArrayV2Stack/TensorListStack/element_shapeé
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ*
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
:ÿÿÿÿÿÿÿÿÿÚ*
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
:ÿÿÿÿÿÿÿÿÿÚ2
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
:ÿÿÿÿÿÿÿÿÿÚ2

IdentityÈ
NoOpNoOp$^lstm_cell_26/BiasAdd/ReadVariableOp#^lstm_cell_26/MatMul/ReadVariableOp%^lstm_cell_26/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ]: : : 2J
#lstm_cell_26/BiasAdd/ReadVariableOp#lstm_cell_26/BiasAdd/ReadVariableOp2H
"lstm_cell_26/MatMul/ReadVariableOp"lstm_cell_26/MatMul/ReadVariableOp2L
$lstm_cell_26/MatMul_1/ReadVariableOp$lstm_cell_26/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
 
_user_specified_nameinputs
´?
Ö
while_body_39974049
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
3while_lstm_cell_27_matmul_readvariableop_resource_0:
ÚI
5while_lstm_cell_27_matmul_1_readvariableop_resource_0:
æC
4while_lstm_cell_27_biasadd_readvariableop_resource_0:	
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
1while_lstm_cell_27_matmul_readvariableop_resource:
ÚG
3while_lstm_cell_27_matmul_1_readvariableop_resource:
æA
2while_lstm_cell_27_biasadd_readvariableop_resource:	¢)while/lstm_cell_27/BiasAdd/ReadVariableOp¢(while/lstm_cell_27/MatMul/ReadVariableOp¢*while/lstm_cell_27/MatMul_1/ReadVariableOpÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿZ  29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÔ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemÊ
(while/lstm_cell_27/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_27_matmul_readvariableop_resource_0* 
_output_shapes
:
Ú*
dtype02*
(while/lstm_cell_27/MatMul/ReadVariableOp×
while/lstm_cell_27/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_27/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_27/MatMulÐ
*while/lstm_cell_27/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_27_matmul_1_readvariableop_resource_0* 
_output_shapes
:
æ*
dtype02,
*while/lstm_cell_27/MatMul_1/ReadVariableOpÀ
while/lstm_cell_27/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_27/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_27/MatMul_1¸
while/lstm_cell_27/addAddV2#while/lstm_cell_27/MatMul:product:0%while/lstm_cell_27/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_27/addÈ
)while/lstm_cell_27/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_27_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02+
)while/lstm_cell_27/BiasAdd/ReadVariableOpÅ
while/lstm_cell_27/BiasAddBiasAddwhile/lstm_cell_27/add:z:01while/lstm_cell_27/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_27/BiasAdd
"while/lstm_cell_27/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_27/split/split_dim
while/lstm_cell_27/splitSplit+while/lstm_cell_27/split/split_dim:output:0#while/lstm_cell_27/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿæ:ÿÿÿÿÿÿÿÿÿæ:ÿÿÿÿÿÿÿÿÿæ:ÿÿÿÿÿÿÿÿÿæ*
	num_split2
while/lstm_cell_27/split
while/lstm_cell_27/SigmoidSigmoid!while/lstm_cell_27/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2
while/lstm_cell_27/Sigmoid
while/lstm_cell_27/Sigmoid_1Sigmoid!while/lstm_cell_27/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2
while/lstm_cell_27/Sigmoid_1¡
while/lstm_cell_27/mulMul while/lstm_cell_27/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2
while/lstm_cell_27/mul
while/lstm_cell_27/ReluRelu!while/lstm_cell_27/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2
while/lstm_cell_27/Reluµ
while/lstm_cell_27/mul_1Mulwhile/lstm_cell_27/Sigmoid:y:0%while/lstm_cell_27/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2
while/lstm_cell_27/mul_1ª
while/lstm_cell_27/add_1AddV2while/lstm_cell_27/mul:z:0while/lstm_cell_27/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2
while/lstm_cell_27/add_1
while/lstm_cell_27/Sigmoid_2Sigmoid!while/lstm_cell_27/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2
while/lstm_cell_27/Sigmoid_2
while/lstm_cell_27/Relu_1Reluwhile/lstm_cell_27/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2
while/lstm_cell_27/Relu_1¹
while/lstm_cell_27/mul_2Mul while/lstm_cell_27/Sigmoid_2:y:0'while/lstm_cell_27/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2
while/lstm_cell_27/mul_2à
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
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_27/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_27/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2
while/Identity_5Þ

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
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿæ:ÿÿÿÿÿÿÿÿÿæ: : : : : 2V
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
:ÿÿÿÿÿÿÿÿÿæ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ:

_output_shapes
: :

_output_shapes
: 
éJ
Ö

lstm_27_while_body_39974756,
(lstm_27_while_lstm_27_while_loop_counter2
.lstm_27_while_lstm_27_while_maximum_iterations
lstm_27_while_placeholder
lstm_27_while_placeholder_1
lstm_27_while_placeholder_2
lstm_27_while_placeholder_3+
'lstm_27_while_lstm_27_strided_slice_1_0g
clstm_27_while_tensorarrayv2read_tensorlistgetitem_lstm_27_tensorarrayunstack_tensorlistfromtensor_0O
;lstm_27_while_lstm_cell_27_matmul_readvariableop_resource_0:
ÚQ
=lstm_27_while_lstm_cell_27_matmul_1_readvariableop_resource_0:
æK
<lstm_27_while_lstm_cell_27_biasadd_readvariableop_resource_0:	
lstm_27_while_identity
lstm_27_while_identity_1
lstm_27_while_identity_2
lstm_27_while_identity_3
lstm_27_while_identity_4
lstm_27_while_identity_5)
%lstm_27_while_lstm_27_strided_slice_1e
alstm_27_while_tensorarrayv2read_tensorlistgetitem_lstm_27_tensorarrayunstack_tensorlistfromtensorM
9lstm_27_while_lstm_cell_27_matmul_readvariableop_resource:
ÚO
;lstm_27_while_lstm_cell_27_matmul_1_readvariableop_resource:
æI
:lstm_27_while_lstm_cell_27_biasadd_readvariableop_resource:	¢1lstm_27/while/lstm_cell_27/BiasAdd/ReadVariableOp¢0lstm_27/while/lstm_cell_27/MatMul/ReadVariableOp¢2lstm_27/while/lstm_cell_27/MatMul_1/ReadVariableOpÓ
?lstm_27/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿZ  2A
?lstm_27/while/TensorArrayV2Read/TensorListGetItem/element_shape
1lstm_27/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemclstm_27_while_tensorarrayv2read_tensorlistgetitem_lstm_27_tensorarrayunstack_tensorlistfromtensor_0lstm_27_while_placeholderHlstm_27/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ*
element_dtype023
1lstm_27/while/TensorArrayV2Read/TensorListGetItemâ
0lstm_27/while/lstm_cell_27/MatMul/ReadVariableOpReadVariableOp;lstm_27_while_lstm_cell_27_matmul_readvariableop_resource_0* 
_output_shapes
:
Ú*
dtype022
0lstm_27/while/lstm_cell_27/MatMul/ReadVariableOp÷
!lstm_27/while/lstm_cell_27/MatMulMatMul8lstm_27/while/TensorArrayV2Read/TensorListGetItem:item:08lstm_27/while/lstm_cell_27/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!lstm_27/while/lstm_cell_27/MatMulè
2lstm_27/while/lstm_cell_27/MatMul_1/ReadVariableOpReadVariableOp=lstm_27_while_lstm_cell_27_matmul_1_readvariableop_resource_0* 
_output_shapes
:
æ*
dtype024
2lstm_27/while/lstm_cell_27/MatMul_1/ReadVariableOpà
#lstm_27/while/lstm_cell_27/MatMul_1MatMullstm_27_while_placeholder_2:lstm_27/while/lstm_cell_27/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2%
#lstm_27/while/lstm_cell_27/MatMul_1Ø
lstm_27/while/lstm_cell_27/addAddV2+lstm_27/while/lstm_cell_27/MatMul:product:0-lstm_27/while/lstm_cell_27/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
lstm_27/while/lstm_cell_27/addà
1lstm_27/while/lstm_cell_27/BiasAdd/ReadVariableOpReadVariableOp<lstm_27_while_lstm_cell_27_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype023
1lstm_27/while/lstm_cell_27/BiasAdd/ReadVariableOpå
"lstm_27/while/lstm_cell_27/BiasAddBiasAdd"lstm_27/while/lstm_cell_27/add:z:09lstm_27/while/lstm_cell_27/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2$
"lstm_27/while/lstm_cell_27/BiasAdd
*lstm_27/while/lstm_cell_27/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2,
*lstm_27/while/lstm_cell_27/split/split_dim¯
 lstm_27/while/lstm_cell_27/splitSplit3lstm_27/while/lstm_cell_27/split/split_dim:output:0+lstm_27/while/lstm_cell_27/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿæ:ÿÿÿÿÿÿÿÿÿæ:ÿÿÿÿÿÿÿÿÿæ:ÿÿÿÿÿÿÿÿÿæ*
	num_split2"
 lstm_27/while/lstm_cell_27/split±
"lstm_27/while/lstm_cell_27/SigmoidSigmoid)lstm_27/while/lstm_cell_27/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2$
"lstm_27/while/lstm_cell_27/Sigmoidµ
$lstm_27/while/lstm_cell_27/Sigmoid_1Sigmoid)lstm_27/while/lstm_cell_27/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2&
$lstm_27/while/lstm_cell_27/Sigmoid_1Á
lstm_27/while/lstm_cell_27/mulMul(lstm_27/while/lstm_cell_27/Sigmoid_1:y:0lstm_27_while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2 
lstm_27/while/lstm_cell_27/mul¨
lstm_27/while/lstm_cell_27/ReluRelu)lstm_27/while/lstm_cell_27/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2!
lstm_27/while/lstm_cell_27/ReluÕ
 lstm_27/while/lstm_cell_27/mul_1Mul&lstm_27/while/lstm_cell_27/Sigmoid:y:0-lstm_27/while/lstm_cell_27/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2"
 lstm_27/while/lstm_cell_27/mul_1Ê
 lstm_27/while/lstm_cell_27/add_1AddV2"lstm_27/while/lstm_cell_27/mul:z:0$lstm_27/while/lstm_cell_27/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2"
 lstm_27/while/lstm_cell_27/add_1µ
$lstm_27/while/lstm_cell_27/Sigmoid_2Sigmoid)lstm_27/while/lstm_cell_27/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2&
$lstm_27/while/lstm_cell_27/Sigmoid_2§
!lstm_27/while/lstm_cell_27/Relu_1Relu$lstm_27/while/lstm_cell_27/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2#
!lstm_27/while/lstm_cell_27/Relu_1Ù
 lstm_27/while/lstm_cell_27/mul_2Mul(lstm_27/while/lstm_cell_27/Sigmoid_2:y:0/lstm_27/while/lstm_cell_27/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2"
 lstm_27/while/lstm_cell_27/mul_2
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
lstm_27/while/add/y
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
lstm_27/while/add_1/y
lstm_27/while/add_1AddV2(lstm_27_while_lstm_27_while_loop_counterlstm_27/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lstm_27/while/add_1
lstm_27/while/IdentityIdentitylstm_27/while/add_1:z:0^lstm_27/while/NoOp*
T0*
_output_shapes
: 2
lstm_27/while/Identity¦
lstm_27/while/Identity_1Identity.lstm_27_while_lstm_27_while_maximum_iterations^lstm_27/while/NoOp*
T0*
_output_shapes
: 2
lstm_27/while/Identity_1
lstm_27/while/Identity_2Identitylstm_27/while/add:z:0^lstm_27/while/NoOp*
T0*
_output_shapes
: 2
lstm_27/while/Identity_2º
lstm_27/while/Identity_3IdentityBlstm_27/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_27/while/NoOp*
T0*
_output_shapes
: 2
lstm_27/while/Identity_3®
lstm_27/while/Identity_4Identity$lstm_27/while/lstm_cell_27/mul_2:z:0^lstm_27/while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2
lstm_27/while/Identity_4®
lstm_27/while/Identity_5Identity$lstm_27/while/lstm_cell_27/add_1:z:0^lstm_27/while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2
lstm_27/while/Identity_5
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
9lstm_27_while_lstm_cell_27_matmul_readvariableop_resource;lstm_27_while_lstm_cell_27_matmul_readvariableop_resource_0"È
alstm_27_while_tensorarrayv2read_tensorlistgetitem_lstm_27_tensorarrayunstack_tensorlistfromtensorclstm_27_while_tensorarrayv2read_tensorlistgetitem_lstm_27_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿæ:ÿÿÿÿÿÿÿÿÿæ: : : : : 2f
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
:ÿÿÿÿÿÿÿÿÿæ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ:

_output_shapes
: :

_output_shapes
: 
Ø
I
-__inference_dropout_27_layer_call_fn_39976537

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
:ÿÿÿÿÿÿÿÿÿæ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_dropout_27_layer_call_and_return_conditional_losses_399738772
PartitionedCallq
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿæ:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ
 
_user_specified_nameinputs
×
g
H__inference_dropout_27_layer_call_and_return_conditional_losses_39976559

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
:ÿÿÿÿÿÿÿÿÿæ2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape¹
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ*
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
:ÿÿÿÿÿÿÿÿÿæ2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2
dropout/Mul_1j
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿæ:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ
 
_user_specified_nameinputs
ã
»
*__inference_lstm_27_layer_call_fn_39975906
inputs_0
unknown:
Ú
	unknown_0:
æ
	unknown_1:	
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿæ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_lstm_27_layer_call_and_return_conditional_losses_399732802
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿæ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÚ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÚ
"
_user_specified_name
inputs/0
ã
Í
while_cond_39976145
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_39976145___redundant_placeholder06
2while_while_cond_39976145___redundant_placeholder16
2while_while_cond_39976145___redundant_placeholder26
2while_while_cond_39976145___redundant_placeholder3
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
B: : : : :ÿÿÿÿÿÿÿÿÿæ:ÿÿÿÿÿÿÿÿÿæ: ::::: 

_output_shapes
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
:ÿÿÿÿÿÿÿÿÿæ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ:

_output_shapes
: :

_output_shapes
:
°?
Ô
while_body_39974245
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_26_matmul_readvariableop_resource_0:	]è
I
5while_lstm_cell_26_matmul_1_readvariableop_resource_0:
Úè
C
4while_lstm_cell_26_biasadd_readvariableop_resource_0:	è

while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_26_matmul_readvariableop_resource:	]è
G
3while_lstm_cell_26_matmul_1_readvariableop_resource:
Úè
A
2while_lstm_cell_26_biasadd_readvariableop_resource:	è
¢)while/lstm_cell_26/BiasAdd/ReadVariableOp¢(while/lstm_cell_26/MatMul/ReadVariableOp¢*while/lstm_cell_26/MatMul_1/ReadVariableOpÃ
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
(while/lstm_cell_26/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_26_matmul_readvariableop_resource_0*
_output_shapes
:	]è
*
dtype02*
(while/lstm_cell_26/MatMul/ReadVariableOp×
while/lstm_cell_26/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_26/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè
2
while/lstm_cell_26/MatMulÐ
*while/lstm_cell_26/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_26_matmul_1_readvariableop_resource_0* 
_output_shapes
:
Úè
*
dtype02,
*while/lstm_cell_26/MatMul_1/ReadVariableOpÀ
while/lstm_cell_26/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_26/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè
2
while/lstm_cell_26/MatMul_1¸
while/lstm_cell_26/addAddV2#while/lstm_cell_26/MatMul:product:0%while/lstm_cell_26/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè
2
while/lstm_cell_26/addÈ
)while/lstm_cell_26/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_26_biasadd_readvariableop_resource_0*
_output_shapes	
:è
*
dtype02+
)while/lstm_cell_26/BiasAdd/ReadVariableOpÅ
while/lstm_cell_26/BiasAddBiasAddwhile/lstm_cell_26/add:z:01while/lstm_cell_26/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè
2
while/lstm_cell_26/BiasAdd
"while/lstm_cell_26/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_26/split/split_dim
while/lstm_cell_26/splitSplit+while/lstm_cell_26/split/split_dim:output:0#while/lstm_cell_26/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿÚ:ÿÿÿÿÿÿÿÿÿÚ:ÿÿÿÿÿÿÿÿÿÚ:ÿÿÿÿÿÿÿÿÿÚ*
	num_split2
while/lstm_cell_26/split
while/lstm_cell_26/SigmoidSigmoid!while/lstm_cell_26/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
while/lstm_cell_26/Sigmoid
while/lstm_cell_26/Sigmoid_1Sigmoid!while/lstm_cell_26/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
while/lstm_cell_26/Sigmoid_1¡
while/lstm_cell_26/mulMul while/lstm_cell_26/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
while/lstm_cell_26/mul
while/lstm_cell_26/ReluRelu!while/lstm_cell_26/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
while/lstm_cell_26/Reluµ
while/lstm_cell_26/mul_1Mulwhile/lstm_cell_26/Sigmoid:y:0%while/lstm_cell_26/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
while/lstm_cell_26/mul_1ª
while/lstm_cell_26/add_1AddV2while/lstm_cell_26/mul:z:0while/lstm_cell_26/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
while/lstm_cell_26/add_1
while/lstm_cell_26/Sigmoid_2Sigmoid!while/lstm_cell_26/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
while/lstm_cell_26/Sigmoid_2
while/lstm_cell_26/Relu_1Reluwhile/lstm_cell_26/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
while/lstm_cell_26/Relu_1¹
while/lstm_cell_26/mul_2Mul while/lstm_cell_26/Sigmoid_2:y:0'while/lstm_cell_26/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
while/lstm_cell_26/mul_2à
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
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_26/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_26/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
while/Identity_5Þ

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
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿÚ:ÿÿÿÿÿÿÿÿÿÚ: : : : : 2V
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
:ÿÿÿÿÿÿÿÿÿÚ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ:

_output_shapes
: :

_output_shapes
: 


J__inference_lstm_cell_27_layer_call_and_return_conditional_losses_39976763

inputs
states_0
states_12
matmul_readvariableop_resource:
Ú4
 matmul_1_readvariableop_resource:
æ.
biasadd_readvariableop_resource:	
identity

identity_1

identity_2¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
Ú*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
æ*
dtype02
MatMul_1/ReadVariableOp|
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
add
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
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
P:ÿÿÿÿÿÿÿÿÿæ:ÿÿÿÿÿÿÿÿÿæ:ÿÿÿÿÿÿÿÿÿæ:ÿÿÿÿÿÿÿÿÿæ*
	num_split2
split`
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2	
Sigmoidd
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2
	Sigmoid_1]
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2
mulW
ReluRelusplit:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2
Relui
mul_1MulSigmoid:y:0Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2
mul_1^
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2
add_1d
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2
	Sigmoid_2V
Relu_1Relu	add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2
Relu_1m
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2
mul_2e
IdentityIdentity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2

Identityi

Identity_1Identity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2

Identity_1i

Identity_2Identity	add_1:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2

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
B:ÿÿÿÿÿÿÿÿÿÚ:ÿÿÿÿÿÿÿÿÿæ:ÿÿÿÿÿÿÿÿÿæ: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ
 
_user_specified_nameinputs:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ
"
_user_specified_name
states/0:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ
"
_user_specified_name
states/1
Ã\
 
E__inference_lstm_26_layer_call_and_return_conditional_losses_39975404
inputs_0>
+lstm_cell_26_matmul_readvariableop_resource:	]è
A
-lstm_cell_26_matmul_1_readvariableop_resource:
Úè
;
,lstm_cell_26_biasadd_readvariableop_resource:	è

identity¢#lstm_cell_26/BiasAdd/ReadVariableOp¢"lstm_cell_26/MatMul/ReadVariableOp¢$lstm_cell_26/MatMul_1/ReadVariableOp¢whileF
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
B :Ú2
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
B :Ú2
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
:ÿÿÿÿÿÿÿÿÿÚ2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :Ú2
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
B :Ú2
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
:ÿÿÿÿÿÿÿÿÿÚ2	
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
"lstm_cell_26/MatMul/ReadVariableOpReadVariableOp+lstm_cell_26_matmul_readvariableop_resource*
_output_shapes
:	]è
*
dtype02$
"lstm_cell_26/MatMul/ReadVariableOp­
lstm_cell_26/MatMulMatMulstrided_slice_2:output:0*lstm_cell_26/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè
2
lstm_cell_26/MatMul¼
$lstm_cell_26/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_26_matmul_1_readvariableop_resource* 
_output_shapes
:
Úè
*
dtype02&
$lstm_cell_26/MatMul_1/ReadVariableOp©
lstm_cell_26/MatMul_1MatMulzeros:output:0,lstm_cell_26/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè
2
lstm_cell_26/MatMul_1 
lstm_cell_26/addAddV2lstm_cell_26/MatMul:product:0lstm_cell_26/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè
2
lstm_cell_26/add´
#lstm_cell_26/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_26_biasadd_readvariableop_resource*
_output_shapes	
:è
*
dtype02%
#lstm_cell_26/BiasAdd/ReadVariableOp­
lstm_cell_26/BiasAddBiasAddlstm_cell_26/add:z:0+lstm_cell_26/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè
2
lstm_cell_26/BiasAdd~
lstm_cell_26/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_26/split/split_dim÷
lstm_cell_26/splitSplit%lstm_cell_26/split/split_dim:output:0lstm_cell_26/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿÚ:ÿÿÿÿÿÿÿÿÿÚ:ÿÿÿÿÿÿÿÿÿÚ:ÿÿÿÿÿÿÿÿÿÚ*
	num_split2
lstm_cell_26/split
lstm_cell_26/SigmoidSigmoidlstm_cell_26/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
lstm_cell_26/Sigmoid
lstm_cell_26/Sigmoid_1Sigmoidlstm_cell_26/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
lstm_cell_26/Sigmoid_1
lstm_cell_26/mulMullstm_cell_26/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
lstm_cell_26/mul~
lstm_cell_26/ReluRelulstm_cell_26/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
lstm_cell_26/Relu
lstm_cell_26/mul_1Mullstm_cell_26/Sigmoid:y:0lstm_cell_26/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
lstm_cell_26/mul_1
lstm_cell_26/add_1AddV2lstm_cell_26/mul:z:0lstm_cell_26/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
lstm_cell_26/add_1
lstm_cell_26/Sigmoid_2Sigmoidlstm_cell_26/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
lstm_cell_26/Sigmoid_2}
lstm_cell_26/Relu_1Relulstm_cell_26/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
lstm_cell_26/Relu_1¡
lstm_cell_26/mul_2Mullstm_cell_26/Sigmoid_2:y:0!lstm_cell_26/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
lstm_cell_26/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿZ  2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_26_matmul_readvariableop_resource-lstm_cell_26_matmul_1_readvariableop_resource,lstm_cell_26_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿÚ:ÿÿÿÿÿÿÿÿÿÚ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_39975320*
condR
while_cond_39975319*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿÚ:ÿÿÿÿÿÿÿÿÿÚ: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿZ  22
0TensorArrayV2Stack/TensorListStack/element_shapeò
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÚ*
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
:ÿÿÿÿÿÿÿÿÿÚ*
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
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÚ2
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
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÚ2

IdentityÈ
NoOpNoOp$^lstm_cell_26/BiasAdd/ReadVariableOp#^lstm_cell_26/MatMul/ReadVariableOp%^lstm_cell_26/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ]: : : 2J
#lstm_cell_26/BiasAdd/ReadVariableOp#lstm_cell_26/BiasAdd/ReadVariableOp2H
"lstm_cell_26/MatMul/ReadVariableOp"lstm_cell_26/MatMul/ReadVariableOp2L
$lstm_cell_26/MatMul_1/ReadVariableOp$lstm_cell_26/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ]
"
_user_specified_name
inputs/0
ã
Í
while_cond_39973000
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_39973000___redundant_placeholder06
2while_while_cond_39973000___redundant_placeholder16
2while_while_cond_39973000___redundant_placeholder26
2while_while_cond_39973000___redundant_placeholder3
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
B: : : : :ÿÿÿÿÿÿÿÿÿæ:ÿÿÿÿÿÿÿÿÿæ: ::::: 

_output_shapes
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
:ÿÿÿÿÿÿÿÿÿæ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ:

_output_shapes
: :

_output_shapes
:
Ê
Â
K__inference_sequential_13_layer_call_and_return_conditional_losses_39974476
lstm_26_input#
lstm_26_39974454:	]è
$
lstm_26_39974456:
Úè

lstm_26_39974458:	è
$
lstm_27_39974462:
Ú$
lstm_27_39974464:
æ
lstm_27_39974466:	$
dense_13_39974470:	æ
dense_13_39974472:
identity¢ dense_13/StatefulPartitionedCall¢"dropout_26/StatefulPartitionedCall¢"dropout_27/StatefulPartitionedCall¢lstm_26/StatefulPartitionedCall¢lstm_27/StatefulPartitionedCallµ
lstm_26/StatefulPartitionedCallStatefulPartitionedCalllstm_26_inputlstm_26_39974454lstm_26_39974456lstm_26_39974458*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_lstm_26_layer_call_and_return_conditional_losses_399743292!
lstm_26/StatefulPartitionedCall
"dropout_26/StatefulPartitionedCallStatefulPartitionedCall(lstm_26/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_dropout_26_layer_call_and_return_conditional_losses_399741622$
"dropout_26/StatefulPartitionedCallÓ
lstm_27/StatefulPartitionedCallStatefulPartitionedCall+dropout_26/StatefulPartitionedCall:output:0lstm_27_39974462lstm_27_39974464lstm_27_39974466*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_lstm_27_layer_call_and_return_conditional_losses_399741332!
lstm_27/StatefulPartitionedCallÀ
"dropout_27/StatefulPartitionedCallStatefulPartitionedCall(lstm_27/StatefulPartitionedCall:output:0#^dropout_26/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_dropout_27_layer_call_and_return_conditional_losses_399739662$
"dropout_27/StatefulPartitionedCallÃ
 dense_13/StatefulPartitionedCallStatefulPartitionedCall+dropout_27/StatefulPartitionedCall:output:0dense_13_39974470dense_13_39974472*
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
F__inference_dense_13_layer_call_and_return_conditional_losses_399739102"
 dense_13/StatefulPartitionedCall
IdentityIdentity)dense_13/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityÿ
NoOpNoOp!^dense_13/StatefulPartitionedCall#^dropout_26/StatefulPartitionedCall#^dropout_27/StatefulPartitionedCall ^lstm_26/StatefulPartitionedCall ^lstm_27/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ]: : : : : : : : 2D
 dense_13/StatefulPartitionedCall dense_13/StatefulPartitionedCall2H
"dropout_26/StatefulPartitionedCall"dropout_26/StatefulPartitionedCall2H
"dropout_27/StatefulPartitionedCall"dropout_27/StatefulPartitionedCall2B
lstm_26/StatefulPartitionedCalllstm_26/StatefulPartitionedCall2B
lstm_27/StatefulPartitionedCalllstm_27/StatefulPartitionedCall:Z V
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
'
_user_specified_namelstm_26_input
\

E__inference_lstm_27_layer_call_and_return_conditional_losses_39974133

inputs?
+lstm_cell_27_matmul_readvariableop_resource:
ÚA
-lstm_cell_27_matmul_1_readvariableop_resource:
æ;
,lstm_cell_27_biasadd_readvariableop_resource:	
identity¢#lstm_cell_27/BiasAdd/ReadVariableOp¢"lstm_cell_27/MatMul/ReadVariableOp¢$lstm_cell_27/MatMul_1/ReadVariableOp¢whileD
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
B :æ2
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
B :æ2
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
:ÿÿÿÿÿÿÿÿÿæ2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :æ2
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
B :æ2
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
:ÿÿÿÿÿÿÿÿÿæ2	
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
:ÿÿÿÿÿÿÿÿÿÚ2
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
valueB"ÿÿÿÿZ  27
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
:ÿÿÿÿÿÿÿÿÿÚ*
shrink_axis_mask2
strided_slice_2¶
"lstm_cell_27/MatMul/ReadVariableOpReadVariableOp+lstm_cell_27_matmul_readvariableop_resource* 
_output_shapes
:
Ú*
dtype02$
"lstm_cell_27/MatMul/ReadVariableOp­
lstm_cell_27/MatMulMatMulstrided_slice_2:output:0*lstm_cell_27/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_27/MatMul¼
$lstm_cell_27/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_27_matmul_1_readvariableop_resource* 
_output_shapes
:
æ*
dtype02&
$lstm_cell_27/MatMul_1/ReadVariableOp©
lstm_cell_27/MatMul_1MatMulzeros:output:0,lstm_cell_27/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_27/MatMul_1 
lstm_cell_27/addAddV2lstm_cell_27/MatMul:product:0lstm_cell_27/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_27/add´
#lstm_cell_27/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_27_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02%
#lstm_cell_27/BiasAdd/ReadVariableOp­
lstm_cell_27/BiasAddBiasAddlstm_cell_27/add:z:0+lstm_cell_27/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_27/BiasAdd~
lstm_cell_27/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_27/split/split_dim÷
lstm_cell_27/splitSplit%lstm_cell_27/split/split_dim:output:0lstm_cell_27/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿæ:ÿÿÿÿÿÿÿÿÿæ:ÿÿÿÿÿÿÿÿÿæ:ÿÿÿÿÿÿÿÿÿæ*
	num_split2
lstm_cell_27/split
lstm_cell_27/SigmoidSigmoidlstm_cell_27/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2
lstm_cell_27/Sigmoid
lstm_cell_27/Sigmoid_1Sigmoidlstm_cell_27/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2
lstm_cell_27/Sigmoid_1
lstm_cell_27/mulMullstm_cell_27/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2
lstm_cell_27/mul~
lstm_cell_27/ReluRelulstm_cell_27/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2
lstm_cell_27/Relu
lstm_cell_27/mul_1Mullstm_cell_27/Sigmoid:y:0lstm_cell_27/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2
lstm_cell_27/mul_1
lstm_cell_27/add_1AddV2lstm_cell_27/mul:z:0lstm_cell_27/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2
lstm_cell_27/add_1
lstm_cell_27/Sigmoid_2Sigmoidlstm_cell_27/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2
lstm_cell_27/Sigmoid_2}
lstm_cell_27/Relu_1Relulstm_cell_27/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2
lstm_cell_27/Relu_1¡
lstm_cell_27/mul_2Mullstm_cell_27/Sigmoid_2:y:0!lstm_cell_27/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2
lstm_cell_27/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿf  2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_27_matmul_readvariableop_resource-lstm_cell_27_matmul_1_readvariableop_resource,lstm_cell_27_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿæ:ÿÿÿÿÿÿÿÿÿæ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_39974049*
condR
while_cond_39974048*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿæ:ÿÿÿÿÿÿÿÿÿæ: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿf  22
0TensorArrayV2Stack/TensorListStack/element_shapeé
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ*
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
:ÿÿÿÿÿÿÿÿÿæ*
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
:ÿÿÿÿÿÿÿÿÿæ2
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
:ÿÿÿÿÿÿÿÿÿæ2

IdentityÈ
NoOpNoOp$^lstm_cell_27/BiasAdd/ReadVariableOp#^lstm_cell_27/MatMul/ReadVariableOp%^lstm_cell_27/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:ÿÿÿÿÿÿÿÿÿÚ: : : 2J
#lstm_cell_27/BiasAdd/ReadVariableOp#lstm_cell_27/BiasAdd/ReadVariableOp2H
"lstm_cell_27/MatMul/ReadVariableOp"lstm_cell_27/MatMul/ReadVariableOp2L
$lstm_cell_27/MatMul_1/ReadVariableOp$lstm_cell_27/MatMul_1/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ
 
_user_specified_nameinputs
éJ
Ö

lstm_27_while_body_39975090,
(lstm_27_while_lstm_27_while_loop_counter2
.lstm_27_while_lstm_27_while_maximum_iterations
lstm_27_while_placeholder
lstm_27_while_placeholder_1
lstm_27_while_placeholder_2
lstm_27_while_placeholder_3+
'lstm_27_while_lstm_27_strided_slice_1_0g
clstm_27_while_tensorarrayv2read_tensorlistgetitem_lstm_27_tensorarrayunstack_tensorlistfromtensor_0O
;lstm_27_while_lstm_cell_27_matmul_readvariableop_resource_0:
ÚQ
=lstm_27_while_lstm_cell_27_matmul_1_readvariableop_resource_0:
æK
<lstm_27_while_lstm_cell_27_biasadd_readvariableop_resource_0:	
lstm_27_while_identity
lstm_27_while_identity_1
lstm_27_while_identity_2
lstm_27_while_identity_3
lstm_27_while_identity_4
lstm_27_while_identity_5)
%lstm_27_while_lstm_27_strided_slice_1e
alstm_27_while_tensorarrayv2read_tensorlistgetitem_lstm_27_tensorarrayunstack_tensorlistfromtensorM
9lstm_27_while_lstm_cell_27_matmul_readvariableop_resource:
ÚO
;lstm_27_while_lstm_cell_27_matmul_1_readvariableop_resource:
æI
:lstm_27_while_lstm_cell_27_biasadd_readvariableop_resource:	¢1lstm_27/while/lstm_cell_27/BiasAdd/ReadVariableOp¢0lstm_27/while/lstm_cell_27/MatMul/ReadVariableOp¢2lstm_27/while/lstm_cell_27/MatMul_1/ReadVariableOpÓ
?lstm_27/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿZ  2A
?lstm_27/while/TensorArrayV2Read/TensorListGetItem/element_shape
1lstm_27/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemclstm_27_while_tensorarrayv2read_tensorlistgetitem_lstm_27_tensorarrayunstack_tensorlistfromtensor_0lstm_27_while_placeholderHlstm_27/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ*
element_dtype023
1lstm_27/while/TensorArrayV2Read/TensorListGetItemâ
0lstm_27/while/lstm_cell_27/MatMul/ReadVariableOpReadVariableOp;lstm_27_while_lstm_cell_27_matmul_readvariableop_resource_0* 
_output_shapes
:
Ú*
dtype022
0lstm_27/while/lstm_cell_27/MatMul/ReadVariableOp÷
!lstm_27/while/lstm_cell_27/MatMulMatMul8lstm_27/while/TensorArrayV2Read/TensorListGetItem:item:08lstm_27/while/lstm_cell_27/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!lstm_27/while/lstm_cell_27/MatMulè
2lstm_27/while/lstm_cell_27/MatMul_1/ReadVariableOpReadVariableOp=lstm_27_while_lstm_cell_27_matmul_1_readvariableop_resource_0* 
_output_shapes
:
æ*
dtype024
2lstm_27/while/lstm_cell_27/MatMul_1/ReadVariableOpà
#lstm_27/while/lstm_cell_27/MatMul_1MatMullstm_27_while_placeholder_2:lstm_27/while/lstm_cell_27/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2%
#lstm_27/while/lstm_cell_27/MatMul_1Ø
lstm_27/while/lstm_cell_27/addAddV2+lstm_27/while/lstm_cell_27/MatMul:product:0-lstm_27/while/lstm_cell_27/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
lstm_27/while/lstm_cell_27/addà
1lstm_27/while/lstm_cell_27/BiasAdd/ReadVariableOpReadVariableOp<lstm_27_while_lstm_cell_27_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype023
1lstm_27/while/lstm_cell_27/BiasAdd/ReadVariableOpå
"lstm_27/while/lstm_cell_27/BiasAddBiasAdd"lstm_27/while/lstm_cell_27/add:z:09lstm_27/while/lstm_cell_27/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2$
"lstm_27/while/lstm_cell_27/BiasAdd
*lstm_27/while/lstm_cell_27/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2,
*lstm_27/while/lstm_cell_27/split/split_dim¯
 lstm_27/while/lstm_cell_27/splitSplit3lstm_27/while/lstm_cell_27/split/split_dim:output:0+lstm_27/while/lstm_cell_27/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿæ:ÿÿÿÿÿÿÿÿÿæ:ÿÿÿÿÿÿÿÿÿæ:ÿÿÿÿÿÿÿÿÿæ*
	num_split2"
 lstm_27/while/lstm_cell_27/split±
"lstm_27/while/lstm_cell_27/SigmoidSigmoid)lstm_27/while/lstm_cell_27/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2$
"lstm_27/while/lstm_cell_27/Sigmoidµ
$lstm_27/while/lstm_cell_27/Sigmoid_1Sigmoid)lstm_27/while/lstm_cell_27/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2&
$lstm_27/while/lstm_cell_27/Sigmoid_1Á
lstm_27/while/lstm_cell_27/mulMul(lstm_27/while/lstm_cell_27/Sigmoid_1:y:0lstm_27_while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2 
lstm_27/while/lstm_cell_27/mul¨
lstm_27/while/lstm_cell_27/ReluRelu)lstm_27/while/lstm_cell_27/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2!
lstm_27/while/lstm_cell_27/ReluÕ
 lstm_27/while/lstm_cell_27/mul_1Mul&lstm_27/while/lstm_cell_27/Sigmoid:y:0-lstm_27/while/lstm_cell_27/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2"
 lstm_27/while/lstm_cell_27/mul_1Ê
 lstm_27/while/lstm_cell_27/add_1AddV2"lstm_27/while/lstm_cell_27/mul:z:0$lstm_27/while/lstm_cell_27/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2"
 lstm_27/while/lstm_cell_27/add_1µ
$lstm_27/while/lstm_cell_27/Sigmoid_2Sigmoid)lstm_27/while/lstm_cell_27/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2&
$lstm_27/while/lstm_cell_27/Sigmoid_2§
!lstm_27/while/lstm_cell_27/Relu_1Relu$lstm_27/while/lstm_cell_27/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2#
!lstm_27/while/lstm_cell_27/Relu_1Ù
 lstm_27/while/lstm_cell_27/mul_2Mul(lstm_27/while/lstm_cell_27/Sigmoid_2:y:0/lstm_27/while/lstm_cell_27/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2"
 lstm_27/while/lstm_cell_27/mul_2
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
lstm_27/while/add/y
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
lstm_27/while/add_1/y
lstm_27/while/add_1AddV2(lstm_27_while_lstm_27_while_loop_counterlstm_27/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lstm_27/while/add_1
lstm_27/while/IdentityIdentitylstm_27/while/add_1:z:0^lstm_27/while/NoOp*
T0*
_output_shapes
: 2
lstm_27/while/Identity¦
lstm_27/while/Identity_1Identity.lstm_27_while_lstm_27_while_maximum_iterations^lstm_27/while/NoOp*
T0*
_output_shapes
: 2
lstm_27/while/Identity_1
lstm_27/while/Identity_2Identitylstm_27/while/add:z:0^lstm_27/while/NoOp*
T0*
_output_shapes
: 2
lstm_27/while/Identity_2º
lstm_27/while/Identity_3IdentityBlstm_27/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_27/while/NoOp*
T0*
_output_shapes
: 2
lstm_27/while/Identity_3®
lstm_27/while/Identity_4Identity$lstm_27/while/lstm_cell_27/mul_2:z:0^lstm_27/while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2
lstm_27/while/Identity_4®
lstm_27/while/Identity_5Identity$lstm_27/while/lstm_cell_27/add_1:z:0^lstm_27/while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2
lstm_27/while/Identity_5
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
9lstm_27_while_lstm_cell_27_matmul_readvariableop_resource;lstm_27_while_lstm_cell_27_matmul_readvariableop_resource_0"È
alstm_27_while_tensorarrayv2read_tensorlistgetitem_lstm_27_tensorarrayunstack_tensorlistfromtensorclstm_27_while_tensorarrayv2read_tensorlistgetitem_lstm_27_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿæ:ÿÿÿÿÿÿÿÿÿæ: : : : : 2f
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
:ÿÿÿÿÿÿÿÿÿæ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ:

_output_shapes
: :

_output_shapes
: 

f
H__inference_dropout_26_layer_call_and_return_conditional_losses_39975872

inputs

identity_1_
IdentityIdentityinputs*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2

Identityn

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿÚ:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ
 
_user_specified_nameinputs
ÐF

E__inference_lstm_27_layer_call_and_return_conditional_losses_39973280

inputs)
lstm_cell_27_39973198:
Ú)
lstm_cell_27_39973200:
æ$
lstm_cell_27_39973202:	
identity¢$lstm_cell_27/StatefulPartitionedCall¢whileD
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
B :æ2
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
B :æ2
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
:ÿÿÿÿÿÿÿÿÿæ2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :æ2
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
B :æ2
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
:ÿÿÿÿÿÿÿÿÿæ2	
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
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÚ2
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
valueB"ÿÿÿÿZ  27
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
:ÿÿÿÿÿÿÿÿÿÚ*
shrink_axis_mask2
strided_slice_2¨
$lstm_cell_27/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_27_39973198lstm_cell_27_39973200lstm_cell_27_39973202*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿæ:ÿÿÿÿÿÿÿÿÿæ:ÿÿÿÿÿÿÿÿÿæ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_lstm_cell_27_layer_call_and_return_conditional_losses_399731332&
$lstm_cell_27/StatefulPartitionedCall
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿf  2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_27_39973198lstm_cell_27_39973200lstm_cell_27_39973202*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿæ:ÿÿÿÿÿÿÿÿÿæ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_39973211*
condR
while_cond_39973210*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿæ:ÿÿÿÿÿÿÿÿÿæ: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿf  22
0TensorArrayV2Stack/TensorListStack/element_shapeò
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿæ*
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
:ÿÿÿÿÿÿÿÿÿæ*
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
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿæ2
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
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿæ2

Identity}
NoOpNoOp%^lstm_cell_27/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÚ: : : 2L
$lstm_cell_27/StatefulPartitionedCall$lstm_cell_27/StatefulPartitionedCall2
whilewhile:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÚ
 
_user_specified_nameinputs
Ôù

K__inference_sequential_13_layer_call_and_return_conditional_losses_39974868

inputsF
3lstm_26_lstm_cell_26_matmul_readvariableop_resource:	]è
I
5lstm_26_lstm_cell_26_matmul_1_readvariableop_resource:
Úè
C
4lstm_26_lstm_cell_26_biasadd_readvariableop_resource:	è
G
3lstm_27_lstm_cell_27_matmul_readvariableop_resource:
ÚI
5lstm_27_lstm_cell_27_matmul_1_readvariableop_resource:
æC
4lstm_27_lstm_cell_27_biasadd_readvariableop_resource:	=
*dense_13_tensordot_readvariableop_resource:	æ6
(dense_13_biasadd_readvariableop_resource:
identity¢dense_13/BiasAdd/ReadVariableOp¢!dense_13/Tensordot/ReadVariableOp¢+lstm_26/lstm_cell_26/BiasAdd/ReadVariableOp¢*lstm_26/lstm_cell_26/MatMul/ReadVariableOp¢,lstm_26/lstm_cell_26/MatMul_1/ReadVariableOp¢lstm_26/while¢+lstm_27/lstm_cell_27/BiasAdd/ReadVariableOp¢*lstm_27/lstm_cell_27/MatMul/ReadVariableOp¢,lstm_27/lstm_cell_27/MatMul_1/ReadVariableOp¢lstm_27/whileT
lstm_26/ShapeShapeinputs*
T0*
_output_shapes
:2
lstm_26/Shape
lstm_26/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_26/strided_slice/stack
lstm_26/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_26/strided_slice/stack_1
lstm_26/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_26/strided_slice/stack_2
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
B :Ú2
lstm_26/zeros/mul/y
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
B :è2
lstm_26/zeros/Less/y
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
B :Ú2
lstm_26/zeros/packed/1£
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
lstm_26/zeros/Const
lstm_26/zerosFilllstm_26/zeros/packed:output:0lstm_26/zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
lstm_26/zerosq
lstm_26/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :Ú2
lstm_26/zeros_1/mul/y
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
B :è2
lstm_26/zeros_1/Less/y
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
B :Ú2
lstm_26/zeros_1/packed/1©
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
lstm_26/zeros_1/Const
lstm_26/zeros_1Filllstm_26/zeros_1/packed:output:0lstm_26/zeros_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
lstm_26/zeros_1
lstm_26/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_26/transpose/perm
lstm_26/transpose	Transposeinputslstm_26/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]2
lstm_26/transposeg
lstm_26/Shape_1Shapelstm_26/transpose:y:0*
T0*
_output_shapes
:2
lstm_26/Shape_1
lstm_26/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_26/strided_slice_1/stack
lstm_26/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_26/strided_slice_1/stack_1
lstm_26/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_26/strided_slice_1/stack_2
lstm_26/strided_slice_1StridedSlicelstm_26/Shape_1:output:0&lstm_26/strided_slice_1/stack:output:0(lstm_26/strided_slice_1/stack_1:output:0(lstm_26/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_26/strided_slice_1
#lstm_26/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2%
#lstm_26/TensorArrayV2/element_shapeÒ
lstm_26/TensorArrayV2TensorListReserve,lstm_26/TensorArrayV2/element_shape:output:0 lstm_26/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_26/TensorArrayV2Ï
=lstm_26/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ]   2?
=lstm_26/TensorArrayUnstack/TensorListFromTensor/element_shape
/lstm_26/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_26/transpose:y:0Flstm_26/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type021
/lstm_26/TensorArrayUnstack/TensorListFromTensor
lstm_26/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_26/strided_slice_2/stack
lstm_26/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_26/strided_slice_2/stack_1
lstm_26/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_26/strided_slice_2/stack_2¬
lstm_26/strided_slice_2StridedSlicelstm_26/transpose:y:0&lstm_26/strided_slice_2/stack:output:0(lstm_26/strided_slice_2/stack_1:output:0(lstm_26/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]*
shrink_axis_mask2
lstm_26/strided_slice_2Í
*lstm_26/lstm_cell_26/MatMul/ReadVariableOpReadVariableOp3lstm_26_lstm_cell_26_matmul_readvariableop_resource*
_output_shapes
:	]è
*
dtype02,
*lstm_26/lstm_cell_26/MatMul/ReadVariableOpÍ
lstm_26/lstm_cell_26/MatMulMatMul lstm_26/strided_slice_2:output:02lstm_26/lstm_cell_26/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè
2
lstm_26/lstm_cell_26/MatMulÔ
,lstm_26/lstm_cell_26/MatMul_1/ReadVariableOpReadVariableOp5lstm_26_lstm_cell_26_matmul_1_readvariableop_resource* 
_output_shapes
:
Úè
*
dtype02.
,lstm_26/lstm_cell_26/MatMul_1/ReadVariableOpÉ
lstm_26/lstm_cell_26/MatMul_1MatMullstm_26/zeros:output:04lstm_26/lstm_cell_26/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè
2
lstm_26/lstm_cell_26/MatMul_1À
lstm_26/lstm_cell_26/addAddV2%lstm_26/lstm_cell_26/MatMul:product:0'lstm_26/lstm_cell_26/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè
2
lstm_26/lstm_cell_26/addÌ
+lstm_26/lstm_cell_26/BiasAdd/ReadVariableOpReadVariableOp4lstm_26_lstm_cell_26_biasadd_readvariableop_resource*
_output_shapes	
:è
*
dtype02-
+lstm_26/lstm_cell_26/BiasAdd/ReadVariableOpÍ
lstm_26/lstm_cell_26/BiasAddBiasAddlstm_26/lstm_cell_26/add:z:03lstm_26/lstm_cell_26/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè
2
lstm_26/lstm_cell_26/BiasAdd
$lstm_26/lstm_cell_26/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2&
$lstm_26/lstm_cell_26/split/split_dim
lstm_26/lstm_cell_26/splitSplit-lstm_26/lstm_cell_26/split/split_dim:output:0%lstm_26/lstm_cell_26/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿÚ:ÿÿÿÿÿÿÿÿÿÚ:ÿÿÿÿÿÿÿÿÿÚ:ÿÿÿÿÿÿÿÿÿÚ*
	num_split2
lstm_26/lstm_cell_26/split
lstm_26/lstm_cell_26/SigmoidSigmoid#lstm_26/lstm_cell_26/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
lstm_26/lstm_cell_26/Sigmoid£
lstm_26/lstm_cell_26/Sigmoid_1Sigmoid#lstm_26/lstm_cell_26/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2 
lstm_26/lstm_cell_26/Sigmoid_1¬
lstm_26/lstm_cell_26/mulMul"lstm_26/lstm_cell_26/Sigmoid_1:y:0lstm_26/zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
lstm_26/lstm_cell_26/mul
lstm_26/lstm_cell_26/ReluRelu#lstm_26/lstm_cell_26/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
lstm_26/lstm_cell_26/Relu½
lstm_26/lstm_cell_26/mul_1Mul lstm_26/lstm_cell_26/Sigmoid:y:0'lstm_26/lstm_cell_26/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
lstm_26/lstm_cell_26/mul_1²
lstm_26/lstm_cell_26/add_1AddV2lstm_26/lstm_cell_26/mul:z:0lstm_26/lstm_cell_26/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
lstm_26/lstm_cell_26/add_1£
lstm_26/lstm_cell_26/Sigmoid_2Sigmoid#lstm_26/lstm_cell_26/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2 
lstm_26/lstm_cell_26/Sigmoid_2
lstm_26/lstm_cell_26/Relu_1Relulstm_26/lstm_cell_26/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
lstm_26/lstm_cell_26/Relu_1Á
lstm_26/lstm_cell_26/mul_2Mul"lstm_26/lstm_cell_26/Sigmoid_2:y:0)lstm_26/lstm_cell_26/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
lstm_26/lstm_cell_26/mul_2
%lstm_26/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿZ  2'
%lstm_26/TensorArrayV2_1/element_shapeØ
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
lstm_26/time
 lstm_26/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2"
 lstm_26/while/maximum_iterationsz
lstm_26/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_26/while/loop_counter
lstm_26/whileWhile#lstm_26/while/loop_counter:output:0)lstm_26/while/maximum_iterations:output:0lstm_26/time:output:0 lstm_26/TensorArrayV2_1:handle:0lstm_26/zeros:output:0lstm_26/zeros_1:output:0 lstm_26/strided_slice_1:output:0?lstm_26/TensorArrayUnstack/TensorListFromTensor:output_handle:03lstm_26_lstm_cell_26_matmul_readvariableop_resource5lstm_26_lstm_cell_26_matmul_1_readvariableop_resource4lstm_26_lstm_cell_26_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿÚ:ÿÿÿÿÿÿÿÿÿÚ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *'
bodyR
lstm_26_while_body_39974608*'
condR
lstm_26_while_cond_39974607*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿÚ:ÿÿÿÿÿÿÿÿÿÚ: : : : : *
parallel_iterations 2
lstm_26/whileÅ
8lstm_26/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿZ  2:
8lstm_26/TensorArrayV2Stack/TensorListStack/element_shape
*lstm_26/TensorArrayV2Stack/TensorListStackTensorListStacklstm_26/while:output:3Alstm_26/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ*
element_dtype02,
*lstm_26/TensorArrayV2Stack/TensorListStack
lstm_26/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
lstm_26/strided_slice_3/stack
lstm_26/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2!
lstm_26/strided_slice_3/stack_1
lstm_26/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_26/strided_slice_3/stack_2Ë
lstm_26/strided_slice_3StridedSlice3lstm_26/TensorArrayV2Stack/TensorListStack:tensor:0&lstm_26/strided_slice_3/stack:output:0(lstm_26/strided_slice_3/stack_1:output:0(lstm_26/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ*
shrink_axis_mask2
lstm_26/strided_slice_3
lstm_26/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_26/transpose_1/permÆ
lstm_26/transpose_1	Transpose3lstm_26/TensorArrayV2Stack/TensorListStack:tensor:0!lstm_26/transpose_1/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
lstm_26/transpose_1v
lstm_26/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_26/runtime
dropout_26/IdentityIdentitylstm_26/transpose_1:y:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
dropout_26/Identityj
lstm_27/ShapeShapedropout_26/Identity:output:0*
T0*
_output_shapes
:2
lstm_27/Shape
lstm_27/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_27/strided_slice/stack
lstm_27/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_27/strided_slice/stack_1
lstm_27/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_27/strided_slice/stack_2
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
B :æ2
lstm_27/zeros/mul/y
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
B :è2
lstm_27/zeros/Less/y
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
B :æ2
lstm_27/zeros/packed/1£
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
lstm_27/zeros/Const
lstm_27/zerosFilllstm_27/zeros/packed:output:0lstm_27/zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2
lstm_27/zerosq
lstm_27/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :æ2
lstm_27/zeros_1/mul/y
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
B :è2
lstm_27/zeros_1/Less/y
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
B :æ2
lstm_27/zeros_1/packed/1©
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
lstm_27/zeros_1/Const
lstm_27/zeros_1Filllstm_27/zeros_1/packed:output:0lstm_27/zeros_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2
lstm_27/zeros_1
lstm_27/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_27/transpose/perm©
lstm_27/transpose	Transposedropout_26/Identity:output:0lstm_27/transpose/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
lstm_27/transposeg
lstm_27/Shape_1Shapelstm_27/transpose:y:0*
T0*
_output_shapes
:2
lstm_27/Shape_1
lstm_27/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_27/strided_slice_1/stack
lstm_27/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_27/strided_slice_1/stack_1
lstm_27/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_27/strided_slice_1/stack_2
lstm_27/strided_slice_1StridedSlicelstm_27/Shape_1:output:0&lstm_27/strided_slice_1/stack:output:0(lstm_27/strided_slice_1/stack_1:output:0(lstm_27/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_27/strided_slice_1
#lstm_27/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2%
#lstm_27/TensorArrayV2/element_shapeÒ
lstm_27/TensorArrayV2TensorListReserve,lstm_27/TensorArrayV2/element_shape:output:0 lstm_27/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_27/TensorArrayV2Ï
=lstm_27/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿZ  2?
=lstm_27/TensorArrayUnstack/TensorListFromTensor/element_shape
/lstm_27/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_27/transpose:y:0Flstm_27/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type021
/lstm_27/TensorArrayUnstack/TensorListFromTensor
lstm_27/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_27/strided_slice_2/stack
lstm_27/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_27/strided_slice_2/stack_1
lstm_27/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_27/strided_slice_2/stack_2­
lstm_27/strided_slice_2StridedSlicelstm_27/transpose:y:0&lstm_27/strided_slice_2/stack:output:0(lstm_27/strided_slice_2/stack_1:output:0(lstm_27/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ*
shrink_axis_mask2
lstm_27/strided_slice_2Î
*lstm_27/lstm_cell_27/MatMul/ReadVariableOpReadVariableOp3lstm_27_lstm_cell_27_matmul_readvariableop_resource* 
_output_shapes
:
Ú*
dtype02,
*lstm_27/lstm_cell_27/MatMul/ReadVariableOpÍ
lstm_27/lstm_cell_27/MatMulMatMul lstm_27/strided_slice_2:output:02lstm_27/lstm_cell_27/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_27/lstm_cell_27/MatMulÔ
,lstm_27/lstm_cell_27/MatMul_1/ReadVariableOpReadVariableOp5lstm_27_lstm_cell_27_matmul_1_readvariableop_resource* 
_output_shapes
:
æ*
dtype02.
,lstm_27/lstm_cell_27/MatMul_1/ReadVariableOpÉ
lstm_27/lstm_cell_27/MatMul_1MatMullstm_27/zeros:output:04lstm_27/lstm_cell_27/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_27/lstm_cell_27/MatMul_1À
lstm_27/lstm_cell_27/addAddV2%lstm_27/lstm_cell_27/MatMul:product:0'lstm_27/lstm_cell_27/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_27/lstm_cell_27/addÌ
+lstm_27/lstm_cell_27/BiasAdd/ReadVariableOpReadVariableOp4lstm_27_lstm_cell_27_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02-
+lstm_27/lstm_cell_27/BiasAdd/ReadVariableOpÍ
lstm_27/lstm_cell_27/BiasAddBiasAddlstm_27/lstm_cell_27/add:z:03lstm_27/lstm_cell_27/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_27/lstm_cell_27/BiasAdd
$lstm_27/lstm_cell_27/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2&
$lstm_27/lstm_cell_27/split/split_dim
lstm_27/lstm_cell_27/splitSplit-lstm_27/lstm_cell_27/split/split_dim:output:0%lstm_27/lstm_cell_27/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿæ:ÿÿÿÿÿÿÿÿÿæ:ÿÿÿÿÿÿÿÿÿæ:ÿÿÿÿÿÿÿÿÿæ*
	num_split2
lstm_27/lstm_cell_27/split
lstm_27/lstm_cell_27/SigmoidSigmoid#lstm_27/lstm_cell_27/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2
lstm_27/lstm_cell_27/Sigmoid£
lstm_27/lstm_cell_27/Sigmoid_1Sigmoid#lstm_27/lstm_cell_27/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2 
lstm_27/lstm_cell_27/Sigmoid_1¬
lstm_27/lstm_cell_27/mulMul"lstm_27/lstm_cell_27/Sigmoid_1:y:0lstm_27/zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2
lstm_27/lstm_cell_27/mul
lstm_27/lstm_cell_27/ReluRelu#lstm_27/lstm_cell_27/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2
lstm_27/lstm_cell_27/Relu½
lstm_27/lstm_cell_27/mul_1Mul lstm_27/lstm_cell_27/Sigmoid:y:0'lstm_27/lstm_cell_27/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2
lstm_27/lstm_cell_27/mul_1²
lstm_27/lstm_cell_27/add_1AddV2lstm_27/lstm_cell_27/mul:z:0lstm_27/lstm_cell_27/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2
lstm_27/lstm_cell_27/add_1£
lstm_27/lstm_cell_27/Sigmoid_2Sigmoid#lstm_27/lstm_cell_27/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2 
lstm_27/lstm_cell_27/Sigmoid_2
lstm_27/lstm_cell_27/Relu_1Relulstm_27/lstm_cell_27/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2
lstm_27/lstm_cell_27/Relu_1Á
lstm_27/lstm_cell_27/mul_2Mul"lstm_27/lstm_cell_27/Sigmoid_2:y:0)lstm_27/lstm_cell_27/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2
lstm_27/lstm_cell_27/mul_2
%lstm_27/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿf  2'
%lstm_27/TensorArrayV2_1/element_shapeØ
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
lstm_27/time
 lstm_27/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2"
 lstm_27/while/maximum_iterationsz
lstm_27/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_27/while/loop_counter
lstm_27/whileWhile#lstm_27/while/loop_counter:output:0)lstm_27/while/maximum_iterations:output:0lstm_27/time:output:0 lstm_27/TensorArrayV2_1:handle:0lstm_27/zeros:output:0lstm_27/zeros_1:output:0 lstm_27/strided_slice_1:output:0?lstm_27/TensorArrayUnstack/TensorListFromTensor:output_handle:03lstm_27_lstm_cell_27_matmul_readvariableop_resource5lstm_27_lstm_cell_27_matmul_1_readvariableop_resource4lstm_27_lstm_cell_27_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿæ:ÿÿÿÿÿÿÿÿÿæ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *'
bodyR
lstm_27_while_body_39974756*'
condR
lstm_27_while_cond_39974755*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿæ:ÿÿÿÿÿÿÿÿÿæ: : : : : *
parallel_iterations 2
lstm_27/whileÅ
8lstm_27/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿf  2:
8lstm_27/TensorArrayV2Stack/TensorListStack/element_shape
*lstm_27/TensorArrayV2Stack/TensorListStackTensorListStacklstm_27/while:output:3Alstm_27/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ*
element_dtype02,
*lstm_27/TensorArrayV2Stack/TensorListStack
lstm_27/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
lstm_27/strided_slice_3/stack
lstm_27/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2!
lstm_27/strided_slice_3/stack_1
lstm_27/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_27/strided_slice_3/stack_2Ë
lstm_27/strided_slice_3StridedSlice3lstm_27/TensorArrayV2Stack/TensorListStack:tensor:0&lstm_27/strided_slice_3/stack:output:0(lstm_27/strided_slice_3/stack_1:output:0(lstm_27/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ*
shrink_axis_mask2
lstm_27/strided_slice_3
lstm_27/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_27/transpose_1/permÆ
lstm_27/transpose_1	Transpose3lstm_27/TensorArrayV2Stack/TensorListStack:tensor:0!lstm_27/transpose_1/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2
lstm_27/transpose_1v
lstm_27/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_27/runtime
dropout_27/IdentityIdentitylstm_27/transpose_1:y:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2
dropout_27/Identity²
!dense_13/Tensordot/ReadVariableOpReadVariableOp*dense_13_tensordot_readvariableop_resource*
_output_shapes
:	æ*
dtype02#
!dense_13/Tensordot/ReadVariableOp|
dense_13/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_13/Tensordot/axes
dense_13/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_13/Tensordot/free
dense_13/Tensordot/ShapeShapedropout_27/Identity:output:0*
T0*
_output_shapes
:2
dense_13/Tensordot/Shape
 dense_13/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_13/Tensordot/GatherV2/axisþ
dense_13/Tensordot/GatherV2GatherV2!dense_13/Tensordot/Shape:output:0 dense_13/Tensordot/free:output:0)dense_13/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_13/Tensordot/GatherV2
"dense_13/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"dense_13/Tensordot/GatherV2_1/axis
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
dense_13/Tensordot/Const¤
dense_13/Tensordot/ProdProd$dense_13/Tensordot/GatherV2:output:0!dense_13/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_13/Tensordot/Prod
dense_13/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_13/Tensordot/Const_1¬
dense_13/Tensordot/Prod_1Prod&dense_13/Tensordot/GatherV2_1:output:0#dense_13/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_13/Tensordot/Prod_1
dense_13/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2 
dense_13/Tensordot/concat/axisÝ
dense_13/Tensordot/concatConcatV2 dense_13/Tensordot/free:output:0 dense_13/Tensordot/axes:output:0'dense_13/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_13/Tensordot/concat°
dense_13/Tensordot/stackPack dense_13/Tensordot/Prod:output:0"dense_13/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_13/Tensordot/stackÂ
dense_13/Tensordot/transpose	Transposedropout_27/Identity:output:0"dense_13/Tensordot/concat:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2
dense_13/Tensordot/transposeÃ
dense_13/Tensordot/ReshapeReshape dense_13/Tensordot/transpose:y:0!dense_13/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
dense_13/Tensordot/ReshapeÂ
dense_13/Tensordot/MatMulMatMul#dense_13/Tensordot/Reshape:output:0)dense_13/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_13/Tensordot/MatMul
dense_13/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
dense_13/Tensordot/Const_2
 dense_13/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_13/Tensordot/concat_1/axisê
dense_13/Tensordot/concat_1ConcatV2$dense_13/Tensordot/GatherV2:output:0#dense_13/Tensordot/Const_2:output:0)dense_13/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_13/Tensordot/concat_1´
dense_13/TensordotReshape#dense_13/Tensordot/MatMul:product:0$dense_13/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_13/Tensordot§
dense_13/BiasAdd/ReadVariableOpReadVariableOp(dense_13_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_13/BiasAdd/ReadVariableOp«
dense_13/BiasAddBiasAdddense_13/Tensordot:output:0'dense_13/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_13/BiasAdd
dense_13/SoftmaxSoftmaxdense_13/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_13/Softmaxy
IdentityIdentitydense_13/Softmax:softmax:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

IdentityÈ
NoOpNoOp ^dense_13/BiasAdd/ReadVariableOp"^dense_13/Tensordot/ReadVariableOp,^lstm_26/lstm_cell_26/BiasAdd/ReadVariableOp+^lstm_26/lstm_cell_26/MatMul/ReadVariableOp-^lstm_26/lstm_cell_26/MatMul_1/ReadVariableOp^lstm_26/while,^lstm_27/lstm_cell_27/BiasAdd/ReadVariableOp+^lstm_27/lstm_cell_27/MatMul/ReadVariableOp-^lstm_27/lstm_cell_27/MatMul_1/ReadVariableOp^lstm_27/while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ]: : : : : : : : 2B
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
:ÿÿÿÿÿÿÿÿÿ]
 
_user_specified_nameinputs
ã
Í
while_cond_39973210
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_39973210___redundant_placeholder06
2while_while_cond_39973210___redundant_placeholder16
2while_while_cond_39973210___redundant_placeholder26
2while_while_cond_39973210___redundant_placeholder3
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
B: : : : :ÿÿÿÿÿÿÿÿÿæ:ÿÿÿÿÿÿÿÿÿæ: ::::: 

_output_shapes
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
:ÿÿÿÿÿÿÿÿÿæ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ:

_output_shapes
: :

_output_shapes
:
Ç
ù
/__inference_lstm_cell_26_layer_call_fn_39976616

inputs
states_0
states_1
unknown:	]è

	unknown_0:
Úè

	unknown_1:	è

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
<:ÿÿÿÿÿÿÿÿÿÚ:ÿÿÿÿÿÿÿÿÿÚ:ÿÿÿÿÿÿÿÿÿÚ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_lstm_cell_26_layer_call_and_return_conditional_losses_399723572
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2

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
A:ÿÿÿÿÿÿÿÿÿ]:ÿÿÿÿÿÿÿÿÿÚ:ÿÿÿÿÿÿÿÿÿÚ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
 
_user_specified_nameinputs:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ
"
_user_specified_name
states/0:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ
"
_user_specified_name
states/1
ò

K__inference_sequential_13_layer_call_and_return_conditional_losses_39975209

inputsF
3lstm_26_lstm_cell_26_matmul_readvariableop_resource:	]è
I
5lstm_26_lstm_cell_26_matmul_1_readvariableop_resource:
Úè
C
4lstm_26_lstm_cell_26_biasadd_readvariableop_resource:	è
G
3lstm_27_lstm_cell_27_matmul_readvariableop_resource:
ÚI
5lstm_27_lstm_cell_27_matmul_1_readvariableop_resource:
æC
4lstm_27_lstm_cell_27_biasadd_readvariableop_resource:	=
*dense_13_tensordot_readvariableop_resource:	æ6
(dense_13_biasadd_readvariableop_resource:
identity¢dense_13/BiasAdd/ReadVariableOp¢!dense_13/Tensordot/ReadVariableOp¢+lstm_26/lstm_cell_26/BiasAdd/ReadVariableOp¢*lstm_26/lstm_cell_26/MatMul/ReadVariableOp¢,lstm_26/lstm_cell_26/MatMul_1/ReadVariableOp¢lstm_26/while¢+lstm_27/lstm_cell_27/BiasAdd/ReadVariableOp¢*lstm_27/lstm_cell_27/MatMul/ReadVariableOp¢,lstm_27/lstm_cell_27/MatMul_1/ReadVariableOp¢lstm_27/whileT
lstm_26/ShapeShapeinputs*
T0*
_output_shapes
:2
lstm_26/Shape
lstm_26/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_26/strided_slice/stack
lstm_26/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_26/strided_slice/stack_1
lstm_26/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_26/strided_slice/stack_2
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
B :Ú2
lstm_26/zeros/mul/y
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
B :è2
lstm_26/zeros/Less/y
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
B :Ú2
lstm_26/zeros/packed/1£
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
lstm_26/zeros/Const
lstm_26/zerosFilllstm_26/zeros/packed:output:0lstm_26/zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
lstm_26/zerosq
lstm_26/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :Ú2
lstm_26/zeros_1/mul/y
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
B :è2
lstm_26/zeros_1/Less/y
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
B :Ú2
lstm_26/zeros_1/packed/1©
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
lstm_26/zeros_1/Const
lstm_26/zeros_1Filllstm_26/zeros_1/packed:output:0lstm_26/zeros_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
lstm_26/zeros_1
lstm_26/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_26/transpose/perm
lstm_26/transpose	Transposeinputslstm_26/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]2
lstm_26/transposeg
lstm_26/Shape_1Shapelstm_26/transpose:y:0*
T0*
_output_shapes
:2
lstm_26/Shape_1
lstm_26/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_26/strided_slice_1/stack
lstm_26/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_26/strided_slice_1/stack_1
lstm_26/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_26/strided_slice_1/stack_2
lstm_26/strided_slice_1StridedSlicelstm_26/Shape_1:output:0&lstm_26/strided_slice_1/stack:output:0(lstm_26/strided_slice_1/stack_1:output:0(lstm_26/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_26/strided_slice_1
#lstm_26/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2%
#lstm_26/TensorArrayV2/element_shapeÒ
lstm_26/TensorArrayV2TensorListReserve,lstm_26/TensorArrayV2/element_shape:output:0 lstm_26/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_26/TensorArrayV2Ï
=lstm_26/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ]   2?
=lstm_26/TensorArrayUnstack/TensorListFromTensor/element_shape
/lstm_26/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_26/transpose:y:0Flstm_26/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type021
/lstm_26/TensorArrayUnstack/TensorListFromTensor
lstm_26/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_26/strided_slice_2/stack
lstm_26/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_26/strided_slice_2/stack_1
lstm_26/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_26/strided_slice_2/stack_2¬
lstm_26/strided_slice_2StridedSlicelstm_26/transpose:y:0&lstm_26/strided_slice_2/stack:output:0(lstm_26/strided_slice_2/stack_1:output:0(lstm_26/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]*
shrink_axis_mask2
lstm_26/strided_slice_2Í
*lstm_26/lstm_cell_26/MatMul/ReadVariableOpReadVariableOp3lstm_26_lstm_cell_26_matmul_readvariableop_resource*
_output_shapes
:	]è
*
dtype02,
*lstm_26/lstm_cell_26/MatMul/ReadVariableOpÍ
lstm_26/lstm_cell_26/MatMulMatMul lstm_26/strided_slice_2:output:02lstm_26/lstm_cell_26/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè
2
lstm_26/lstm_cell_26/MatMulÔ
,lstm_26/lstm_cell_26/MatMul_1/ReadVariableOpReadVariableOp5lstm_26_lstm_cell_26_matmul_1_readvariableop_resource* 
_output_shapes
:
Úè
*
dtype02.
,lstm_26/lstm_cell_26/MatMul_1/ReadVariableOpÉ
lstm_26/lstm_cell_26/MatMul_1MatMullstm_26/zeros:output:04lstm_26/lstm_cell_26/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè
2
lstm_26/lstm_cell_26/MatMul_1À
lstm_26/lstm_cell_26/addAddV2%lstm_26/lstm_cell_26/MatMul:product:0'lstm_26/lstm_cell_26/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè
2
lstm_26/lstm_cell_26/addÌ
+lstm_26/lstm_cell_26/BiasAdd/ReadVariableOpReadVariableOp4lstm_26_lstm_cell_26_biasadd_readvariableop_resource*
_output_shapes	
:è
*
dtype02-
+lstm_26/lstm_cell_26/BiasAdd/ReadVariableOpÍ
lstm_26/lstm_cell_26/BiasAddBiasAddlstm_26/lstm_cell_26/add:z:03lstm_26/lstm_cell_26/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè
2
lstm_26/lstm_cell_26/BiasAdd
$lstm_26/lstm_cell_26/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2&
$lstm_26/lstm_cell_26/split/split_dim
lstm_26/lstm_cell_26/splitSplit-lstm_26/lstm_cell_26/split/split_dim:output:0%lstm_26/lstm_cell_26/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿÚ:ÿÿÿÿÿÿÿÿÿÚ:ÿÿÿÿÿÿÿÿÿÚ:ÿÿÿÿÿÿÿÿÿÚ*
	num_split2
lstm_26/lstm_cell_26/split
lstm_26/lstm_cell_26/SigmoidSigmoid#lstm_26/lstm_cell_26/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
lstm_26/lstm_cell_26/Sigmoid£
lstm_26/lstm_cell_26/Sigmoid_1Sigmoid#lstm_26/lstm_cell_26/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2 
lstm_26/lstm_cell_26/Sigmoid_1¬
lstm_26/lstm_cell_26/mulMul"lstm_26/lstm_cell_26/Sigmoid_1:y:0lstm_26/zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
lstm_26/lstm_cell_26/mul
lstm_26/lstm_cell_26/ReluRelu#lstm_26/lstm_cell_26/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
lstm_26/lstm_cell_26/Relu½
lstm_26/lstm_cell_26/mul_1Mul lstm_26/lstm_cell_26/Sigmoid:y:0'lstm_26/lstm_cell_26/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
lstm_26/lstm_cell_26/mul_1²
lstm_26/lstm_cell_26/add_1AddV2lstm_26/lstm_cell_26/mul:z:0lstm_26/lstm_cell_26/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
lstm_26/lstm_cell_26/add_1£
lstm_26/lstm_cell_26/Sigmoid_2Sigmoid#lstm_26/lstm_cell_26/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2 
lstm_26/lstm_cell_26/Sigmoid_2
lstm_26/lstm_cell_26/Relu_1Relulstm_26/lstm_cell_26/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
lstm_26/lstm_cell_26/Relu_1Á
lstm_26/lstm_cell_26/mul_2Mul"lstm_26/lstm_cell_26/Sigmoid_2:y:0)lstm_26/lstm_cell_26/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
lstm_26/lstm_cell_26/mul_2
%lstm_26/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿZ  2'
%lstm_26/TensorArrayV2_1/element_shapeØ
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
lstm_26/time
 lstm_26/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2"
 lstm_26/while/maximum_iterationsz
lstm_26/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_26/while/loop_counter
lstm_26/whileWhile#lstm_26/while/loop_counter:output:0)lstm_26/while/maximum_iterations:output:0lstm_26/time:output:0 lstm_26/TensorArrayV2_1:handle:0lstm_26/zeros:output:0lstm_26/zeros_1:output:0 lstm_26/strided_slice_1:output:0?lstm_26/TensorArrayUnstack/TensorListFromTensor:output_handle:03lstm_26_lstm_cell_26_matmul_readvariableop_resource5lstm_26_lstm_cell_26_matmul_1_readvariableop_resource4lstm_26_lstm_cell_26_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿÚ:ÿÿÿÿÿÿÿÿÿÚ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *'
bodyR
lstm_26_while_body_39974935*'
condR
lstm_26_while_cond_39974934*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿÚ:ÿÿÿÿÿÿÿÿÿÚ: : : : : *
parallel_iterations 2
lstm_26/whileÅ
8lstm_26/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿZ  2:
8lstm_26/TensorArrayV2Stack/TensorListStack/element_shape
*lstm_26/TensorArrayV2Stack/TensorListStackTensorListStacklstm_26/while:output:3Alstm_26/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ*
element_dtype02,
*lstm_26/TensorArrayV2Stack/TensorListStack
lstm_26/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
lstm_26/strided_slice_3/stack
lstm_26/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2!
lstm_26/strided_slice_3/stack_1
lstm_26/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_26/strided_slice_3/stack_2Ë
lstm_26/strided_slice_3StridedSlice3lstm_26/TensorArrayV2Stack/TensorListStack:tensor:0&lstm_26/strided_slice_3/stack:output:0(lstm_26/strided_slice_3/stack_1:output:0(lstm_26/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ*
shrink_axis_mask2
lstm_26/strided_slice_3
lstm_26/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_26/transpose_1/permÆ
lstm_26/transpose_1	Transpose3lstm_26/TensorArrayV2Stack/TensorListStack:tensor:0!lstm_26/transpose_1/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
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
 *«ªª?2
dropout_26/dropout/Constª
dropout_26/dropout/MulMullstm_26/transpose_1:y:0!dropout_26/dropout/Const:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
dropout_26/dropout/Mul{
dropout_26/dropout/ShapeShapelstm_26/transpose_1:y:0*
T0*
_output_shapes
:2
dropout_26/dropout/ShapeÚ
/dropout_26/dropout/random_uniform/RandomUniformRandomUniform!dropout_26/dropout/Shape:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ*
dtype021
/dropout_26/dropout/random_uniform/RandomUniform
!dropout_26/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  >2#
!dropout_26/dropout/GreaterEqual/yï
dropout_26/dropout/GreaterEqualGreaterEqual8dropout_26/dropout/random_uniform/RandomUniform:output:0*dropout_26/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2!
dropout_26/dropout/GreaterEqual¥
dropout_26/dropout/CastCast#dropout_26/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
dropout_26/dropout/Cast«
dropout_26/dropout/Mul_1Muldropout_26/dropout/Mul:z:0dropout_26/dropout/Cast:y:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
dropout_26/dropout/Mul_1j
lstm_27/ShapeShapedropout_26/dropout/Mul_1:z:0*
T0*
_output_shapes
:2
lstm_27/Shape
lstm_27/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_27/strided_slice/stack
lstm_27/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_27/strided_slice/stack_1
lstm_27/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_27/strided_slice/stack_2
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
B :æ2
lstm_27/zeros/mul/y
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
B :è2
lstm_27/zeros/Less/y
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
B :æ2
lstm_27/zeros/packed/1£
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
lstm_27/zeros/Const
lstm_27/zerosFilllstm_27/zeros/packed:output:0lstm_27/zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2
lstm_27/zerosq
lstm_27/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :æ2
lstm_27/zeros_1/mul/y
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
B :è2
lstm_27/zeros_1/Less/y
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
B :æ2
lstm_27/zeros_1/packed/1©
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
lstm_27/zeros_1/Const
lstm_27/zeros_1Filllstm_27/zeros_1/packed:output:0lstm_27/zeros_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2
lstm_27/zeros_1
lstm_27/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_27/transpose/perm©
lstm_27/transpose	Transposedropout_26/dropout/Mul_1:z:0lstm_27/transpose/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
lstm_27/transposeg
lstm_27/Shape_1Shapelstm_27/transpose:y:0*
T0*
_output_shapes
:2
lstm_27/Shape_1
lstm_27/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_27/strided_slice_1/stack
lstm_27/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_27/strided_slice_1/stack_1
lstm_27/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_27/strided_slice_1/stack_2
lstm_27/strided_slice_1StridedSlicelstm_27/Shape_1:output:0&lstm_27/strided_slice_1/stack:output:0(lstm_27/strided_slice_1/stack_1:output:0(lstm_27/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_27/strided_slice_1
#lstm_27/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2%
#lstm_27/TensorArrayV2/element_shapeÒ
lstm_27/TensorArrayV2TensorListReserve,lstm_27/TensorArrayV2/element_shape:output:0 lstm_27/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_27/TensorArrayV2Ï
=lstm_27/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿZ  2?
=lstm_27/TensorArrayUnstack/TensorListFromTensor/element_shape
/lstm_27/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_27/transpose:y:0Flstm_27/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type021
/lstm_27/TensorArrayUnstack/TensorListFromTensor
lstm_27/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_27/strided_slice_2/stack
lstm_27/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_27/strided_slice_2/stack_1
lstm_27/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_27/strided_slice_2/stack_2­
lstm_27/strided_slice_2StridedSlicelstm_27/transpose:y:0&lstm_27/strided_slice_2/stack:output:0(lstm_27/strided_slice_2/stack_1:output:0(lstm_27/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ*
shrink_axis_mask2
lstm_27/strided_slice_2Î
*lstm_27/lstm_cell_27/MatMul/ReadVariableOpReadVariableOp3lstm_27_lstm_cell_27_matmul_readvariableop_resource* 
_output_shapes
:
Ú*
dtype02,
*lstm_27/lstm_cell_27/MatMul/ReadVariableOpÍ
lstm_27/lstm_cell_27/MatMulMatMul lstm_27/strided_slice_2:output:02lstm_27/lstm_cell_27/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_27/lstm_cell_27/MatMulÔ
,lstm_27/lstm_cell_27/MatMul_1/ReadVariableOpReadVariableOp5lstm_27_lstm_cell_27_matmul_1_readvariableop_resource* 
_output_shapes
:
æ*
dtype02.
,lstm_27/lstm_cell_27/MatMul_1/ReadVariableOpÉ
lstm_27/lstm_cell_27/MatMul_1MatMullstm_27/zeros:output:04lstm_27/lstm_cell_27/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_27/lstm_cell_27/MatMul_1À
lstm_27/lstm_cell_27/addAddV2%lstm_27/lstm_cell_27/MatMul:product:0'lstm_27/lstm_cell_27/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_27/lstm_cell_27/addÌ
+lstm_27/lstm_cell_27/BiasAdd/ReadVariableOpReadVariableOp4lstm_27_lstm_cell_27_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02-
+lstm_27/lstm_cell_27/BiasAdd/ReadVariableOpÍ
lstm_27/lstm_cell_27/BiasAddBiasAddlstm_27/lstm_cell_27/add:z:03lstm_27/lstm_cell_27/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_27/lstm_cell_27/BiasAdd
$lstm_27/lstm_cell_27/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2&
$lstm_27/lstm_cell_27/split/split_dim
lstm_27/lstm_cell_27/splitSplit-lstm_27/lstm_cell_27/split/split_dim:output:0%lstm_27/lstm_cell_27/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿæ:ÿÿÿÿÿÿÿÿÿæ:ÿÿÿÿÿÿÿÿÿæ:ÿÿÿÿÿÿÿÿÿæ*
	num_split2
lstm_27/lstm_cell_27/split
lstm_27/lstm_cell_27/SigmoidSigmoid#lstm_27/lstm_cell_27/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2
lstm_27/lstm_cell_27/Sigmoid£
lstm_27/lstm_cell_27/Sigmoid_1Sigmoid#lstm_27/lstm_cell_27/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2 
lstm_27/lstm_cell_27/Sigmoid_1¬
lstm_27/lstm_cell_27/mulMul"lstm_27/lstm_cell_27/Sigmoid_1:y:0lstm_27/zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2
lstm_27/lstm_cell_27/mul
lstm_27/lstm_cell_27/ReluRelu#lstm_27/lstm_cell_27/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2
lstm_27/lstm_cell_27/Relu½
lstm_27/lstm_cell_27/mul_1Mul lstm_27/lstm_cell_27/Sigmoid:y:0'lstm_27/lstm_cell_27/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2
lstm_27/lstm_cell_27/mul_1²
lstm_27/lstm_cell_27/add_1AddV2lstm_27/lstm_cell_27/mul:z:0lstm_27/lstm_cell_27/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2
lstm_27/lstm_cell_27/add_1£
lstm_27/lstm_cell_27/Sigmoid_2Sigmoid#lstm_27/lstm_cell_27/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2 
lstm_27/lstm_cell_27/Sigmoid_2
lstm_27/lstm_cell_27/Relu_1Relulstm_27/lstm_cell_27/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2
lstm_27/lstm_cell_27/Relu_1Á
lstm_27/lstm_cell_27/mul_2Mul"lstm_27/lstm_cell_27/Sigmoid_2:y:0)lstm_27/lstm_cell_27/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2
lstm_27/lstm_cell_27/mul_2
%lstm_27/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿf  2'
%lstm_27/TensorArrayV2_1/element_shapeØ
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
lstm_27/time
 lstm_27/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2"
 lstm_27/while/maximum_iterationsz
lstm_27/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_27/while/loop_counter
lstm_27/whileWhile#lstm_27/while/loop_counter:output:0)lstm_27/while/maximum_iterations:output:0lstm_27/time:output:0 lstm_27/TensorArrayV2_1:handle:0lstm_27/zeros:output:0lstm_27/zeros_1:output:0 lstm_27/strided_slice_1:output:0?lstm_27/TensorArrayUnstack/TensorListFromTensor:output_handle:03lstm_27_lstm_cell_27_matmul_readvariableop_resource5lstm_27_lstm_cell_27_matmul_1_readvariableop_resource4lstm_27_lstm_cell_27_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿæ:ÿÿÿÿÿÿÿÿÿæ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *'
bodyR
lstm_27_while_body_39975090*'
condR
lstm_27_while_cond_39975089*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿæ:ÿÿÿÿÿÿÿÿÿæ: : : : : *
parallel_iterations 2
lstm_27/whileÅ
8lstm_27/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿf  2:
8lstm_27/TensorArrayV2Stack/TensorListStack/element_shape
*lstm_27/TensorArrayV2Stack/TensorListStackTensorListStacklstm_27/while:output:3Alstm_27/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ*
element_dtype02,
*lstm_27/TensorArrayV2Stack/TensorListStack
lstm_27/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
lstm_27/strided_slice_3/stack
lstm_27/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2!
lstm_27/strided_slice_3/stack_1
lstm_27/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_27/strided_slice_3/stack_2Ë
lstm_27/strided_slice_3StridedSlice3lstm_27/TensorArrayV2Stack/TensorListStack:tensor:0&lstm_27/strided_slice_3/stack:output:0(lstm_27/strided_slice_3/stack_1:output:0(lstm_27/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ*
shrink_axis_mask2
lstm_27/strided_slice_3
lstm_27/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_27/transpose_1/permÆ
lstm_27/transpose_1	Transpose3lstm_27/TensorArrayV2Stack/TensorListStack:tensor:0!lstm_27/transpose_1/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2
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
 *   ?2
dropout_27/dropout/Constª
dropout_27/dropout/MulMullstm_27/transpose_1:y:0!dropout_27/dropout/Const:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2
dropout_27/dropout/Mul{
dropout_27/dropout/ShapeShapelstm_27/transpose_1:y:0*
T0*
_output_shapes
:2
dropout_27/dropout/ShapeÚ
/dropout_27/dropout/random_uniform/RandomUniformRandomUniform!dropout_27/dropout/Shape:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ*
dtype021
/dropout_27/dropout/random_uniform/RandomUniform
!dropout_27/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>2#
!dropout_27/dropout/GreaterEqual/yï
dropout_27/dropout/GreaterEqualGreaterEqual8dropout_27/dropout/random_uniform/RandomUniform:output:0*dropout_27/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2!
dropout_27/dropout/GreaterEqual¥
dropout_27/dropout/CastCast#dropout_27/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2
dropout_27/dropout/Cast«
dropout_27/dropout/Mul_1Muldropout_27/dropout/Mul:z:0dropout_27/dropout/Cast:y:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2
dropout_27/dropout/Mul_1²
!dense_13/Tensordot/ReadVariableOpReadVariableOp*dense_13_tensordot_readvariableop_resource*
_output_shapes
:	æ*
dtype02#
!dense_13/Tensordot/ReadVariableOp|
dense_13/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_13/Tensordot/axes
dense_13/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_13/Tensordot/free
dense_13/Tensordot/ShapeShapedropout_27/dropout/Mul_1:z:0*
T0*
_output_shapes
:2
dense_13/Tensordot/Shape
 dense_13/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_13/Tensordot/GatherV2/axisþ
dense_13/Tensordot/GatherV2GatherV2!dense_13/Tensordot/Shape:output:0 dense_13/Tensordot/free:output:0)dense_13/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_13/Tensordot/GatherV2
"dense_13/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"dense_13/Tensordot/GatherV2_1/axis
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
dense_13/Tensordot/Const¤
dense_13/Tensordot/ProdProd$dense_13/Tensordot/GatherV2:output:0!dense_13/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_13/Tensordot/Prod
dense_13/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_13/Tensordot/Const_1¬
dense_13/Tensordot/Prod_1Prod&dense_13/Tensordot/GatherV2_1:output:0#dense_13/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_13/Tensordot/Prod_1
dense_13/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2 
dense_13/Tensordot/concat/axisÝ
dense_13/Tensordot/concatConcatV2 dense_13/Tensordot/free:output:0 dense_13/Tensordot/axes:output:0'dense_13/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_13/Tensordot/concat°
dense_13/Tensordot/stackPack dense_13/Tensordot/Prod:output:0"dense_13/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_13/Tensordot/stackÂ
dense_13/Tensordot/transpose	Transposedropout_27/dropout/Mul_1:z:0"dense_13/Tensordot/concat:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2
dense_13/Tensordot/transposeÃ
dense_13/Tensordot/ReshapeReshape dense_13/Tensordot/transpose:y:0!dense_13/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
dense_13/Tensordot/ReshapeÂ
dense_13/Tensordot/MatMulMatMul#dense_13/Tensordot/Reshape:output:0)dense_13/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_13/Tensordot/MatMul
dense_13/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
dense_13/Tensordot/Const_2
 dense_13/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_13/Tensordot/concat_1/axisê
dense_13/Tensordot/concat_1ConcatV2$dense_13/Tensordot/GatherV2:output:0#dense_13/Tensordot/Const_2:output:0)dense_13/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_13/Tensordot/concat_1´
dense_13/TensordotReshape#dense_13/Tensordot/MatMul:product:0$dense_13/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_13/Tensordot§
dense_13/BiasAdd/ReadVariableOpReadVariableOp(dense_13_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_13/BiasAdd/ReadVariableOp«
dense_13/BiasAddBiasAdddense_13/Tensordot:output:0'dense_13/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_13/BiasAdd
dense_13/SoftmaxSoftmaxdense_13/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_13/Softmaxy
IdentityIdentitydense_13/Softmax:softmax:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

IdentityÈ
NoOpNoOp ^dense_13/BiasAdd/ReadVariableOp"^dense_13/Tensordot/ReadVariableOp,^lstm_26/lstm_cell_26/BiasAdd/ReadVariableOp+^lstm_26/lstm_cell_26/MatMul/ReadVariableOp-^lstm_26/lstm_cell_26/MatMul_1/ReadVariableOp^lstm_26/while,^lstm_27/lstm_cell_27/BiasAdd/ReadVariableOp+^lstm_27/lstm_cell_27/MatMul/ReadVariableOp-^lstm_27/lstm_cell_27/MatMul_1/ReadVariableOp^lstm_27/while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ]: : : : : : : : 2B
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
:ÿÿÿÿÿÿÿÿÿ]
 
_user_specified_nameinputs
ã
Í
while_cond_39975772
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_39975772___redundant_placeholder06
2while_while_cond_39975772___redundant_placeholder16
2while_while_cond_39975772___redundant_placeholder26
2while_while_cond_39975772___redundant_placeholder3
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
B: : : : :ÿÿÿÿÿÿÿÿÿÚ:ÿÿÿÿÿÿÿÿÿÚ: ::::: 

_output_shapes
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
:ÿÿÿÿÿÿÿÿÿÚ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ:

_output_shapes
: :

_output_shapes
:
Ô

í
lstm_27_while_cond_39975089,
(lstm_27_while_lstm_27_while_loop_counter2
.lstm_27_while_lstm_27_while_maximum_iterations
lstm_27_while_placeholder
lstm_27_while_placeholder_1
lstm_27_while_placeholder_2
lstm_27_while_placeholder_3.
*lstm_27_while_less_lstm_27_strided_slice_1F
Blstm_27_while_lstm_27_while_cond_39975089___redundant_placeholder0F
Blstm_27_while_lstm_27_while_cond_39975089___redundant_placeholder1F
Blstm_27_while_lstm_27_while_cond_39975089___redundant_placeholder2F
Blstm_27_while_lstm_27_while_cond_39975089___redundant_placeholder3
lstm_27_while_identity

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
B: : : : :ÿÿÿÿÿÿÿÿÿæ:ÿÿÿÿÿÿÿÿÿæ: ::::: 

_output_shapes
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
:ÿÿÿÿÿÿÿÿÿæ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ:

_output_shapes
: :

_output_shapes
:


J__inference_lstm_cell_27_layer_call_and_return_conditional_losses_39976795

inputs
states_0
states_12
matmul_readvariableop_resource:
Ú4
 matmul_1_readvariableop_resource:
æ.
biasadd_readvariableop_resource:	
identity

identity_1

identity_2¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
Ú*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
æ*
dtype02
MatMul_1/ReadVariableOp|
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
add
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
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
P:ÿÿÿÿÿÿÿÿÿæ:ÿÿÿÿÿÿÿÿÿæ:ÿÿÿÿÿÿÿÿÿæ:ÿÿÿÿÿÿÿÿÿæ*
	num_split2
split`
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2	
Sigmoidd
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2
	Sigmoid_1]
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2
mulW
ReluRelusplit:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2
Relui
mul_1MulSigmoid:y:0Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2
mul_1^
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2
add_1d
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2
	Sigmoid_2V
Relu_1Relu	add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2
Relu_1m
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2
mul_2e
IdentityIdentity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2

Identityi

Identity_1Identity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2

Identity_1i

Identity_2Identity	add_1:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2

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
B:ÿÿÿÿÿÿÿÿÿÚ:ÿÿÿÿÿÿÿÿÿæ:ÿÿÿÿÿÿÿÿÿæ: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ
 
_user_specified_nameinputs:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ
"
_user_specified_name
states/0:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ
"
_user_specified_name
states/1
½
ø
K__inference_sequential_13_layer_call_and_return_conditional_losses_39974451
lstm_26_input#
lstm_26_39974429:	]è
$
lstm_26_39974431:
Úè

lstm_26_39974433:	è
$
lstm_27_39974437:
Ú$
lstm_27_39974439:
æ
lstm_27_39974441:	$
dense_13_39974445:	æ
dense_13_39974447:
identity¢ dense_13/StatefulPartitionedCall¢lstm_26/StatefulPartitionedCall¢lstm_27/StatefulPartitionedCallµ
lstm_26/StatefulPartitionedCallStatefulPartitionedCalllstm_26_inputlstm_26_39974429lstm_26_39974431lstm_26_39974433*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_lstm_26_layer_call_and_return_conditional_losses_399736992!
lstm_26/StatefulPartitionedCall
dropout_26/PartitionedCallPartitionedCall(lstm_26/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_dropout_26_layer_call_and_return_conditional_losses_399737122
dropout_26/PartitionedCallË
lstm_27/StatefulPartitionedCallStatefulPartitionedCall#dropout_26/PartitionedCall:output:0lstm_27_39974437lstm_27_39974439lstm_27_39974441*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_lstm_27_layer_call_and_return_conditional_losses_399738642!
lstm_27/StatefulPartitionedCall
dropout_27/PartitionedCallPartitionedCall(lstm_27/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_dropout_27_layer_call_and_return_conditional_losses_399738772
dropout_27/PartitionedCall»
 dense_13/StatefulPartitionedCallStatefulPartitionedCall#dropout_27/PartitionedCall:output:0dense_13_39974445dense_13_39974447*
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
F__inference_dense_13_layer_call_and_return_conditional_losses_399739102"
 dense_13/StatefulPartitionedCall
IdentityIdentity)dense_13/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityµ
NoOpNoOp!^dense_13/StatefulPartitionedCall ^lstm_26/StatefulPartitionedCall ^lstm_27/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ]: : : : : : : : 2D
 dense_13/StatefulPartitionedCall dense_13/StatefulPartitionedCall2B
lstm_26/StatefulPartitionedCalllstm_26/StatefulPartitionedCall2B
lstm_27/StatefulPartitionedCalllstm_27/StatefulPartitionedCall:Z V
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
'
_user_specified_namelstm_26_input
×
g
H__inference_dropout_26_layer_call_and_return_conditional_losses_39975884

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
:ÿÿÿÿÿÿÿÿÿÚ2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape¹
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ*
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
:ÿÿÿÿÿÿÿÿÿÚ2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
dropout/Mul_1j
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿÚ:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ
 
_user_specified_nameinputs
´?
Ö
while_body_39975995
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
3while_lstm_cell_27_matmul_readvariableop_resource_0:
ÚI
5while_lstm_cell_27_matmul_1_readvariableop_resource_0:
æC
4while_lstm_cell_27_biasadd_readvariableop_resource_0:	
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
1while_lstm_cell_27_matmul_readvariableop_resource:
ÚG
3while_lstm_cell_27_matmul_1_readvariableop_resource:
æA
2while_lstm_cell_27_biasadd_readvariableop_resource:	¢)while/lstm_cell_27/BiasAdd/ReadVariableOp¢(while/lstm_cell_27/MatMul/ReadVariableOp¢*while/lstm_cell_27/MatMul_1/ReadVariableOpÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿZ  29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÔ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemÊ
(while/lstm_cell_27/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_27_matmul_readvariableop_resource_0* 
_output_shapes
:
Ú*
dtype02*
(while/lstm_cell_27/MatMul/ReadVariableOp×
while/lstm_cell_27/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_27/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_27/MatMulÐ
*while/lstm_cell_27/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_27_matmul_1_readvariableop_resource_0* 
_output_shapes
:
æ*
dtype02,
*while/lstm_cell_27/MatMul_1/ReadVariableOpÀ
while/lstm_cell_27/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_27/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_27/MatMul_1¸
while/lstm_cell_27/addAddV2#while/lstm_cell_27/MatMul:product:0%while/lstm_cell_27/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_27/addÈ
)while/lstm_cell_27/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_27_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02+
)while/lstm_cell_27/BiasAdd/ReadVariableOpÅ
while/lstm_cell_27/BiasAddBiasAddwhile/lstm_cell_27/add:z:01while/lstm_cell_27/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_27/BiasAdd
"while/lstm_cell_27/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_27/split/split_dim
while/lstm_cell_27/splitSplit+while/lstm_cell_27/split/split_dim:output:0#while/lstm_cell_27/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿæ:ÿÿÿÿÿÿÿÿÿæ:ÿÿÿÿÿÿÿÿÿæ:ÿÿÿÿÿÿÿÿÿæ*
	num_split2
while/lstm_cell_27/split
while/lstm_cell_27/SigmoidSigmoid!while/lstm_cell_27/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2
while/lstm_cell_27/Sigmoid
while/lstm_cell_27/Sigmoid_1Sigmoid!while/lstm_cell_27/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2
while/lstm_cell_27/Sigmoid_1¡
while/lstm_cell_27/mulMul while/lstm_cell_27/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2
while/lstm_cell_27/mul
while/lstm_cell_27/ReluRelu!while/lstm_cell_27/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2
while/lstm_cell_27/Reluµ
while/lstm_cell_27/mul_1Mulwhile/lstm_cell_27/Sigmoid:y:0%while/lstm_cell_27/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2
while/lstm_cell_27/mul_1ª
while/lstm_cell_27/add_1AddV2while/lstm_cell_27/mul:z:0while/lstm_cell_27/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2
while/lstm_cell_27/add_1
while/lstm_cell_27/Sigmoid_2Sigmoid!while/lstm_cell_27/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2
while/lstm_cell_27/Sigmoid_2
while/lstm_cell_27/Relu_1Reluwhile/lstm_cell_27/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2
while/lstm_cell_27/Relu_1¹
while/lstm_cell_27/mul_2Mul while/lstm_cell_27/Sigmoid_2:y:0'while/lstm_cell_27/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2
while/lstm_cell_27/mul_2à
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
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_27/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_27/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2
while/Identity_5Þ

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
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿæ:ÿÿÿÿÿÿÿÿÿæ: : : : : 2V
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
:ÿÿÿÿÿÿÿÿÿæ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ:

_output_shapes
: :

_output_shapes
: 
&
ó
while_body_39972581
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_00
while_lstm_cell_26_39972605_0:	]è
1
while_lstm_cell_26_39972607_0:
Úè
,
while_lstm_cell_26_39972609_0:	è

while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor.
while_lstm_cell_26_39972605:	]è
/
while_lstm_cell_26_39972607:
Úè
*
while_lstm_cell_26_39972609:	è
¢*while/lstm_cell_26/StatefulPartitionedCallÃ
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
*while/lstm_cell_26/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_26_39972605_0while_lstm_cell_26_39972607_0while_lstm_cell_26_39972609_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿÚ:ÿÿÿÿÿÿÿÿÿÚ:ÿÿÿÿÿÿÿÿÿÚ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_lstm_cell_26_layer_call_and_return_conditional_losses_399725032,
*while/lstm_cell_26/StatefulPartitionedCall÷
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
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3¥
while/Identity_4Identity3while/lstm_cell_26/StatefulPartitionedCall:output:1^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
while/Identity_4¥
while/Identity_5Identity3while/lstm_cell_26/StatefulPartitionedCall:output:2^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
while/Identity_5

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
while_lstm_cell_26_39972605while_lstm_cell_26_39972605_0"<
while_lstm_cell_26_39972607while_lstm_cell_26_39972607_0"<
while_lstm_cell_26_39972609while_lstm_cell_26_39972609_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿÚ:ÿÿÿÿÿÿÿÿÿÚ: : : : : 2X
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
:ÿÿÿÿÿÿÿÿÿÚ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ:

_output_shapes
: :

_output_shapes
: 


J__inference_lstm_cell_26_layer_call_and_return_conditional_losses_39972503

inputs

states
states_11
matmul_readvariableop_resource:	]è
4
 matmul_1_readvariableop_resource:
Úè
.
biasadd_readvariableop_resource:	è

identity

identity_1

identity_2¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	]è
*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè
2
MatMul
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
Úè
*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè
2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè
2
add
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:è
*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè
2	
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
P:ÿÿÿÿÿÿÿÿÿÚ:ÿÿÿÿÿÿÿÿÿÚ:ÿÿÿÿÿÿÿÿÿÚ:ÿÿÿÿÿÿÿÿÿÚ*
	num_split2
split`
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2	
Sigmoidd
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
	Sigmoid_1]
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
mulW
ReluRelusplit:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
Relui
mul_1MulSigmoid:y:0Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
mul_1^
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
add_1d
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
	Sigmoid_2V
Relu_1Relu	add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
Relu_1m
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
mul_2e
IdentityIdentity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2

Identityi

Identity_1Identity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2

Identity_1i

Identity_2Identity	add_1:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2

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
A:ÿÿÿÿÿÿÿÿÿ]:ÿÿÿÿÿÿÿÿÿÚ:ÿÿÿÿÿÿÿÿÿÚ: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
 
_user_specified_nameinputs:PL
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ
 
_user_specified_namestates:PL
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ
 
_user_specified_namestates
®

Ô
0__inference_sequential_13_layer_call_fn_39973936
lstm_26_input
unknown:	]è

	unknown_0:
Úè

	unknown_1:	è

	unknown_2:
Ú
	unknown_3:
æ
	unknown_4:	
	unknown_5:	æ
	unknown_6:
identity¢StatefulPartitionedCallÔ
StatefulPartitionedCallStatefulPartitionedCalllstm_26_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
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
K__inference_sequential_13_layer_call_and_return_conditional_losses_399739172
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
_user_specified_namelstm_26_input
ÐF

E__inference_lstm_27_layer_call_and_return_conditional_losses_39973070

inputs)
lstm_cell_27_39972988:
Ú)
lstm_cell_27_39972990:
æ$
lstm_cell_27_39972992:	
identity¢$lstm_cell_27/StatefulPartitionedCall¢whileD
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
B :æ2
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
B :æ2
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
:ÿÿÿÿÿÿÿÿÿæ2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :æ2
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
B :æ2
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
:ÿÿÿÿÿÿÿÿÿæ2	
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
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÚ2
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
valueB"ÿÿÿÿZ  27
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
:ÿÿÿÿÿÿÿÿÿÚ*
shrink_axis_mask2
strided_slice_2¨
$lstm_cell_27/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_27_39972988lstm_cell_27_39972990lstm_cell_27_39972992*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿæ:ÿÿÿÿÿÿÿÿÿæ:ÿÿÿÿÿÿÿÿÿæ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_lstm_cell_27_layer_call_and_return_conditional_losses_399729872&
$lstm_cell_27/StatefulPartitionedCall
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿf  2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_27_39972988lstm_cell_27_39972990lstm_cell_27_39972992*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿæ:ÿÿÿÿÿÿÿÿÿæ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_39973001*
condR
while_cond_39973000*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿæ:ÿÿÿÿÿÿÿÿÿæ: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿf  22
0TensorArrayV2Stack/TensorListStack/element_shapeò
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿæ*
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
:ÿÿÿÿÿÿÿÿÿæ*
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
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿæ2
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
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿæ2

Identity}
NoOpNoOp%^lstm_cell_27/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÚ: : : 2L
$lstm_cell_27/StatefulPartitionedCall$lstm_cell_27/StatefulPartitionedCall2
whilewhile:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÚ
 
_user_specified_nameinputs
´?
Ö
while_body_39976448
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
3while_lstm_cell_27_matmul_readvariableop_resource_0:
ÚI
5while_lstm_cell_27_matmul_1_readvariableop_resource_0:
æC
4while_lstm_cell_27_biasadd_readvariableop_resource_0:	
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
1while_lstm_cell_27_matmul_readvariableop_resource:
ÚG
3while_lstm_cell_27_matmul_1_readvariableop_resource:
æA
2while_lstm_cell_27_biasadd_readvariableop_resource:	¢)while/lstm_cell_27/BiasAdd/ReadVariableOp¢(while/lstm_cell_27/MatMul/ReadVariableOp¢*while/lstm_cell_27/MatMul_1/ReadVariableOpÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿZ  29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÔ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemÊ
(while/lstm_cell_27/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_27_matmul_readvariableop_resource_0* 
_output_shapes
:
Ú*
dtype02*
(while/lstm_cell_27/MatMul/ReadVariableOp×
while/lstm_cell_27/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_27/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_27/MatMulÐ
*while/lstm_cell_27/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_27_matmul_1_readvariableop_resource_0* 
_output_shapes
:
æ*
dtype02,
*while/lstm_cell_27/MatMul_1/ReadVariableOpÀ
while/lstm_cell_27/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_27/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_27/MatMul_1¸
while/lstm_cell_27/addAddV2#while/lstm_cell_27/MatMul:product:0%while/lstm_cell_27/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_27/addÈ
)while/lstm_cell_27/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_27_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02+
)while/lstm_cell_27/BiasAdd/ReadVariableOpÅ
while/lstm_cell_27/BiasAddBiasAddwhile/lstm_cell_27/add:z:01while/lstm_cell_27/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_27/BiasAdd
"while/lstm_cell_27/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_27/split/split_dim
while/lstm_cell_27/splitSplit+while/lstm_cell_27/split/split_dim:output:0#while/lstm_cell_27/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿæ:ÿÿÿÿÿÿÿÿÿæ:ÿÿÿÿÿÿÿÿÿæ:ÿÿÿÿÿÿÿÿÿæ*
	num_split2
while/lstm_cell_27/split
while/lstm_cell_27/SigmoidSigmoid!while/lstm_cell_27/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2
while/lstm_cell_27/Sigmoid
while/lstm_cell_27/Sigmoid_1Sigmoid!while/lstm_cell_27/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2
while/lstm_cell_27/Sigmoid_1¡
while/lstm_cell_27/mulMul while/lstm_cell_27/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2
while/lstm_cell_27/mul
while/lstm_cell_27/ReluRelu!while/lstm_cell_27/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2
while/lstm_cell_27/Reluµ
while/lstm_cell_27/mul_1Mulwhile/lstm_cell_27/Sigmoid:y:0%while/lstm_cell_27/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2
while/lstm_cell_27/mul_1ª
while/lstm_cell_27/add_1AddV2while/lstm_cell_27/mul:z:0while/lstm_cell_27/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2
while/lstm_cell_27/add_1
while/lstm_cell_27/Sigmoid_2Sigmoid!while/lstm_cell_27/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2
while/lstm_cell_27/Sigmoid_2
while/lstm_cell_27/Relu_1Reluwhile/lstm_cell_27/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2
while/lstm_cell_27/Relu_1¹
while/lstm_cell_27/mul_2Mul while/lstm_cell_27/Sigmoid_2:y:0'while/lstm_cell_27/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2
while/lstm_cell_27/mul_2à
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
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_27/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_27/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2
while/Identity_5Þ

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
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿæ:ÿÿÿÿÿÿÿÿÿæ: : : : : 2V
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
:ÿÿÿÿÿÿÿÿÿæ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ:

_output_shapes
: :

_output_shapes
: 
¹
¹
*__inference_lstm_27_layer_call_fn_39975928

inputs
unknown:
Ú
	unknown_0:
æ
	unknown_1:	
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_lstm_27_layer_call_and_return_conditional_losses_399741332
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:ÿÿÿÿÿÿÿÿÿÚ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ
 
_user_specified_nameinputs
×
g
H__inference_dropout_26_layer_call_and_return_conditional_losses_39974162

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
:ÿÿÿÿÿÿÿÿÿÚ2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape¹
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ*
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
:ÿÿÿÿÿÿÿÿÿÚ2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
dropout/Mul_1j
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿÚ:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ
 
_user_specified_nameinputs
\

E__inference_lstm_27_layer_call_and_return_conditional_losses_39973864

inputs?
+lstm_cell_27_matmul_readvariableop_resource:
ÚA
-lstm_cell_27_matmul_1_readvariableop_resource:
æ;
,lstm_cell_27_biasadd_readvariableop_resource:	
identity¢#lstm_cell_27/BiasAdd/ReadVariableOp¢"lstm_cell_27/MatMul/ReadVariableOp¢$lstm_cell_27/MatMul_1/ReadVariableOp¢whileD
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
B :æ2
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
B :æ2
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
:ÿÿÿÿÿÿÿÿÿæ2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :æ2
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
B :æ2
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
:ÿÿÿÿÿÿÿÿÿæ2	
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
:ÿÿÿÿÿÿÿÿÿÚ2
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
valueB"ÿÿÿÿZ  27
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
:ÿÿÿÿÿÿÿÿÿÚ*
shrink_axis_mask2
strided_slice_2¶
"lstm_cell_27/MatMul/ReadVariableOpReadVariableOp+lstm_cell_27_matmul_readvariableop_resource* 
_output_shapes
:
Ú*
dtype02$
"lstm_cell_27/MatMul/ReadVariableOp­
lstm_cell_27/MatMulMatMulstrided_slice_2:output:0*lstm_cell_27/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_27/MatMul¼
$lstm_cell_27/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_27_matmul_1_readvariableop_resource* 
_output_shapes
:
æ*
dtype02&
$lstm_cell_27/MatMul_1/ReadVariableOp©
lstm_cell_27/MatMul_1MatMulzeros:output:0,lstm_cell_27/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_27/MatMul_1 
lstm_cell_27/addAddV2lstm_cell_27/MatMul:product:0lstm_cell_27/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_27/add´
#lstm_cell_27/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_27_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02%
#lstm_cell_27/BiasAdd/ReadVariableOp­
lstm_cell_27/BiasAddBiasAddlstm_cell_27/add:z:0+lstm_cell_27/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_27/BiasAdd~
lstm_cell_27/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_27/split/split_dim÷
lstm_cell_27/splitSplit%lstm_cell_27/split/split_dim:output:0lstm_cell_27/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿæ:ÿÿÿÿÿÿÿÿÿæ:ÿÿÿÿÿÿÿÿÿæ:ÿÿÿÿÿÿÿÿÿæ*
	num_split2
lstm_cell_27/split
lstm_cell_27/SigmoidSigmoidlstm_cell_27/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2
lstm_cell_27/Sigmoid
lstm_cell_27/Sigmoid_1Sigmoidlstm_cell_27/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2
lstm_cell_27/Sigmoid_1
lstm_cell_27/mulMullstm_cell_27/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2
lstm_cell_27/mul~
lstm_cell_27/ReluRelulstm_cell_27/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2
lstm_cell_27/Relu
lstm_cell_27/mul_1Mullstm_cell_27/Sigmoid:y:0lstm_cell_27/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2
lstm_cell_27/mul_1
lstm_cell_27/add_1AddV2lstm_cell_27/mul:z:0lstm_cell_27/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2
lstm_cell_27/add_1
lstm_cell_27/Sigmoid_2Sigmoidlstm_cell_27/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2
lstm_cell_27/Sigmoid_2}
lstm_cell_27/Relu_1Relulstm_cell_27/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2
lstm_cell_27/Relu_1¡
lstm_cell_27/mul_2Mullstm_cell_27/Sigmoid_2:y:0!lstm_cell_27/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2
lstm_cell_27/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿf  2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_27_matmul_readvariableop_resource-lstm_cell_27_matmul_1_readvariableop_resource,lstm_cell_27_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿæ:ÿÿÿÿÿÿÿÿÿæ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_39973780*
condR
while_cond_39973779*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿæ:ÿÿÿÿÿÿÿÿÿæ: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿf  22
0TensorArrayV2Stack/TensorListStack/element_shapeé
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ*
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
:ÿÿÿÿÿÿÿÿÿæ*
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
:ÿÿÿÿÿÿÿÿÿæ2
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
:ÿÿÿÿÿÿÿÿÿæ2

IdentityÈ
NoOpNoOp$^lstm_cell_27/BiasAdd/ReadVariableOp#^lstm_cell_27/MatMul/ReadVariableOp%^lstm_cell_27/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:ÿÿÿÿÿÿÿÿÿÚ: : : 2J
#lstm_cell_27/BiasAdd/ReadVariableOp#lstm_cell_27/BiasAdd/ReadVariableOp2H
"lstm_cell_27/MatMul/ReadVariableOp"lstm_cell_27/MatMul/ReadVariableOp2L
$lstm_cell_27/MatMul_1/ReadVariableOp$lstm_cell_27/MatMul_1/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ
 
_user_specified_nameinputs
&
õ
while_body_39973001
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_01
while_lstm_cell_27_39973025_0:
Ú1
while_lstm_cell_27_39973027_0:
æ,
while_lstm_cell_27_39973029_0:	
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor/
while_lstm_cell_27_39973025:
Ú/
while_lstm_cell_27_39973027:
æ*
while_lstm_cell_27_39973029:	¢*while/lstm_cell_27/StatefulPartitionedCallÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿZ  29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÔ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemì
*while/lstm_cell_27/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_27_39973025_0while_lstm_cell_27_39973027_0while_lstm_cell_27_39973029_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿæ:ÿÿÿÿÿÿÿÿÿæ:ÿÿÿÿÿÿÿÿÿæ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_lstm_cell_27_layer_call_and_return_conditional_losses_399729872,
*while/lstm_cell_27/StatefulPartitionedCall÷
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
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3¥
while/Identity_4Identity3while/lstm_cell_27/StatefulPartitionedCall:output:1^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2
while/Identity_4¥
while/Identity_5Identity3while/lstm_cell_27/StatefulPartitionedCall:output:2^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2
while/Identity_5

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
while_lstm_cell_27_39973025while_lstm_cell_27_39973025_0"<
while_lstm_cell_27_39973027while_lstm_cell_27_39973027_0"<
while_lstm_cell_27_39973029while_lstm_cell_27_39973029_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿæ:ÿÿÿÿÿÿÿÿÿæ: : : : : 2X
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
:ÿÿÿÿÿÿÿÿÿæ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ:

_output_shapes
: :

_output_shapes
: 
à
º
*__inference_lstm_26_layer_call_fn_39975220
inputs_0
unknown:	]è

	unknown_0:
Úè

	unknown_1:	è

identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÚ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_lstm_26_layer_call_and_return_conditional_losses_399724402
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÚ2

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


J__inference_lstm_cell_27_layer_call_and_return_conditional_losses_39973133

inputs

states
states_12
matmul_readvariableop_resource:
Ú4
 matmul_1_readvariableop_resource:
æ.
biasadd_readvariableop_resource:	
identity

identity_1

identity_2¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
Ú*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
æ*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
add
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
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
P:ÿÿÿÿÿÿÿÿÿæ:ÿÿÿÿÿÿÿÿÿæ:ÿÿÿÿÿÿÿÿÿæ:ÿÿÿÿÿÿÿÿÿæ*
	num_split2
split`
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2	
Sigmoidd
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2
	Sigmoid_1]
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2
mulW
ReluRelusplit:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2
Relui
mul_1MulSigmoid:y:0Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2
mul_1^
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2
add_1d
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2
	Sigmoid_2V
Relu_1Relu	add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2
Relu_1m
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2
mul_2e
IdentityIdentity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2

Identityi

Identity_1Identity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2

Identity_1i

Identity_2Identity	add_1:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2

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
B:ÿÿÿÿÿÿÿÿÿÚ:ÿÿÿÿÿÿÿÿÿæ:ÿÿÿÿÿÿÿÿÿæ: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ
 
_user_specified_nameinputs:PL
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ
 
_user_specified_namestates:PL
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ
 
_user_specified_namestates
ù

)sequential_13_lstm_26_while_cond_39972021H
Dsequential_13_lstm_26_while_sequential_13_lstm_26_while_loop_counterN
Jsequential_13_lstm_26_while_sequential_13_lstm_26_while_maximum_iterations+
'sequential_13_lstm_26_while_placeholder-
)sequential_13_lstm_26_while_placeholder_1-
)sequential_13_lstm_26_while_placeholder_2-
)sequential_13_lstm_26_while_placeholder_3J
Fsequential_13_lstm_26_while_less_sequential_13_lstm_26_strided_slice_1b
^sequential_13_lstm_26_while_sequential_13_lstm_26_while_cond_39972021___redundant_placeholder0b
^sequential_13_lstm_26_while_sequential_13_lstm_26_while_cond_39972021___redundant_placeholder1b
^sequential_13_lstm_26_while_sequential_13_lstm_26_while_cond_39972021___redundant_placeholder2b
^sequential_13_lstm_26_while_sequential_13_lstm_26_while_cond_39972021___redundant_placeholder3(
$sequential_13_lstm_26_while_identity
Þ
 sequential_13/lstm_26/while/LessLess'sequential_13_lstm_26_while_placeholderFsequential_13_lstm_26_while_less_sequential_13_lstm_26_strided_slice_1*
T0*
_output_shapes
: 2"
 sequential_13/lstm_26/while/Less
$sequential_13/lstm_26/while/IdentityIdentity$sequential_13/lstm_26/while/Less:z:0*
T0
*
_output_shapes
: 2&
$sequential_13/lstm_26/while/Identity"U
$sequential_13_lstm_26_while_identity-sequential_13/lstm_26/while/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :ÿÿÿÿÿÿÿÿÿÚ:ÿÿÿÿÿÿÿÿÿÚ: ::::: 

_output_shapes
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
:ÿÿÿÿÿÿÿÿÿÚ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ:

_output_shapes
: :

_output_shapes
:
¹
¹
*__inference_lstm_27_layer_call_fn_39975917

inputs
unknown:
Ú
	unknown_0:
æ
	unknown_1:	
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_lstm_27_layer_call_and_return_conditional_losses_399738642
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:ÿÿÿÿÿÿÿÿÿÚ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ
 
_user_specified_nameinputs
ã
Í
while_cond_39975621
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_39975621___redundant_placeholder06
2while_while_cond_39975621___redundant_placeholder16
2while_while_cond_39975621___redundant_placeholder26
2while_while_cond_39975621___redundant_placeholder3
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
B: : : : :ÿÿÿÿÿÿÿÿÿÚ:ÿÿÿÿÿÿÿÿÿÚ: ::::: 

_output_shapes
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
:ÿÿÿÿÿÿÿÿÿÚ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ:

_output_shapes
: :

_output_shapes
:
¶
¸
*__inference_lstm_26_layer_call_fn_39975253

inputs
unknown:	]è

	unknown_0:
Úè

	unknown_1:	è

identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_lstm_26_layer_call_and_return_conditional_losses_399743292
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2

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
¨
ñ
K__inference_sequential_13_layer_call_and_return_conditional_losses_39973917

inputs#
lstm_26_39973700:	]è
$
lstm_26_39973702:
Úè

lstm_26_39973704:	è
$
lstm_27_39973865:
Ú$
lstm_27_39973867:
æ
lstm_27_39973869:	$
dense_13_39973911:	æ
dense_13_39973913:
identity¢ dense_13/StatefulPartitionedCall¢lstm_26/StatefulPartitionedCall¢lstm_27/StatefulPartitionedCall®
lstm_26/StatefulPartitionedCallStatefulPartitionedCallinputslstm_26_39973700lstm_26_39973702lstm_26_39973704*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_lstm_26_layer_call_and_return_conditional_losses_399736992!
lstm_26/StatefulPartitionedCall
dropout_26/PartitionedCallPartitionedCall(lstm_26/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_dropout_26_layer_call_and_return_conditional_losses_399737122
dropout_26/PartitionedCallË
lstm_27/StatefulPartitionedCallStatefulPartitionedCall#dropout_26/PartitionedCall:output:0lstm_27_39973865lstm_27_39973867lstm_27_39973869*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_lstm_27_layer_call_and_return_conditional_losses_399738642!
lstm_27/StatefulPartitionedCall
dropout_27/PartitionedCallPartitionedCall(lstm_27/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_dropout_27_layer_call_and_return_conditional_losses_399738772
dropout_27/PartitionedCall»
 dense_13/StatefulPartitionedCallStatefulPartitionedCall#dropout_27/PartitionedCall:output:0dense_13_39973911dense_13_39973913*
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
F__inference_dense_13_layer_call_and_return_conditional_losses_399739102"
 dense_13/StatefulPartitionedCall
IdentityIdentity)dense_13/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityµ
NoOpNoOp!^dense_13/StatefulPartitionedCall ^lstm_26/StatefulPartitionedCall ^lstm_27/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ]: : : : : : : : 2D
 dense_13/StatefulPartitionedCall dense_13/StatefulPartitionedCall2B
lstm_26/StatefulPartitionedCalllstm_26/StatefulPartitionedCall2B
lstm_27/StatefulPartitionedCalllstm_27/StatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
 
_user_specified_nameinputs


Í
0__inference_sequential_13_layer_call_fn_39974541

inputs
unknown:	]è

	unknown_0:
Úè

	unknown_1:	è

	unknown_2:
Ú
	unknown_3:
æ
	unknown_4:	
	unknown_5:	æ
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
K__inference_sequential_13_layer_call_and_return_conditional_losses_399743862
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
ù

)sequential_13_lstm_27_while_cond_39972169H
Dsequential_13_lstm_27_while_sequential_13_lstm_27_while_loop_counterN
Jsequential_13_lstm_27_while_sequential_13_lstm_27_while_maximum_iterations+
'sequential_13_lstm_27_while_placeholder-
)sequential_13_lstm_27_while_placeholder_1-
)sequential_13_lstm_27_while_placeholder_2-
)sequential_13_lstm_27_while_placeholder_3J
Fsequential_13_lstm_27_while_less_sequential_13_lstm_27_strided_slice_1b
^sequential_13_lstm_27_while_sequential_13_lstm_27_while_cond_39972169___redundant_placeholder0b
^sequential_13_lstm_27_while_sequential_13_lstm_27_while_cond_39972169___redundant_placeholder1b
^sequential_13_lstm_27_while_sequential_13_lstm_27_while_cond_39972169___redundant_placeholder2b
^sequential_13_lstm_27_while_sequential_13_lstm_27_while_cond_39972169___redundant_placeholder3(
$sequential_13_lstm_27_while_identity
Þ
 sequential_13/lstm_27/while/LessLess'sequential_13_lstm_27_while_placeholderFsequential_13_lstm_27_while_less_sequential_13_lstm_27_strided_slice_1*
T0*
_output_shapes
: 2"
 sequential_13/lstm_27/while/Less
$sequential_13/lstm_27/while/IdentityIdentity$sequential_13/lstm_27/while/Less:z:0*
T0
*
_output_shapes
: 2&
$sequential_13/lstm_27/while/Identity"U
$sequential_13_lstm_27_while_identity-sequential_13/lstm_27/while/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :ÿÿÿÿÿÿÿÿÿæ:ÿÿÿÿÿÿÿÿÿæ: ::::: 

_output_shapes
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
:ÿÿÿÿÿÿÿÿÿæ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ:

_output_shapes
: :

_output_shapes
:
à
º
*__inference_lstm_26_layer_call_fn_39975231
inputs_0
unknown:	]è

	unknown_0:
Úè

	unknown_1:	è

identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÚ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_lstm_26_layer_call_and_return_conditional_losses_399726502
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÚ2

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
Ã\
 
E__inference_lstm_26_layer_call_and_return_conditional_losses_39975555
inputs_0>
+lstm_cell_26_matmul_readvariableop_resource:	]è
A
-lstm_cell_26_matmul_1_readvariableop_resource:
Úè
;
,lstm_cell_26_biasadd_readvariableop_resource:	è

identity¢#lstm_cell_26/BiasAdd/ReadVariableOp¢"lstm_cell_26/MatMul/ReadVariableOp¢$lstm_cell_26/MatMul_1/ReadVariableOp¢whileF
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
B :Ú2
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
B :Ú2
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
:ÿÿÿÿÿÿÿÿÿÚ2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :Ú2
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
B :Ú2
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
:ÿÿÿÿÿÿÿÿÿÚ2	
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
"lstm_cell_26/MatMul/ReadVariableOpReadVariableOp+lstm_cell_26_matmul_readvariableop_resource*
_output_shapes
:	]è
*
dtype02$
"lstm_cell_26/MatMul/ReadVariableOp­
lstm_cell_26/MatMulMatMulstrided_slice_2:output:0*lstm_cell_26/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè
2
lstm_cell_26/MatMul¼
$lstm_cell_26/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_26_matmul_1_readvariableop_resource* 
_output_shapes
:
Úè
*
dtype02&
$lstm_cell_26/MatMul_1/ReadVariableOp©
lstm_cell_26/MatMul_1MatMulzeros:output:0,lstm_cell_26/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè
2
lstm_cell_26/MatMul_1 
lstm_cell_26/addAddV2lstm_cell_26/MatMul:product:0lstm_cell_26/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè
2
lstm_cell_26/add´
#lstm_cell_26/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_26_biasadd_readvariableop_resource*
_output_shapes	
:è
*
dtype02%
#lstm_cell_26/BiasAdd/ReadVariableOp­
lstm_cell_26/BiasAddBiasAddlstm_cell_26/add:z:0+lstm_cell_26/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè
2
lstm_cell_26/BiasAdd~
lstm_cell_26/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_26/split/split_dim÷
lstm_cell_26/splitSplit%lstm_cell_26/split/split_dim:output:0lstm_cell_26/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿÚ:ÿÿÿÿÿÿÿÿÿÚ:ÿÿÿÿÿÿÿÿÿÚ:ÿÿÿÿÿÿÿÿÿÚ*
	num_split2
lstm_cell_26/split
lstm_cell_26/SigmoidSigmoidlstm_cell_26/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
lstm_cell_26/Sigmoid
lstm_cell_26/Sigmoid_1Sigmoidlstm_cell_26/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
lstm_cell_26/Sigmoid_1
lstm_cell_26/mulMullstm_cell_26/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
lstm_cell_26/mul~
lstm_cell_26/ReluRelulstm_cell_26/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
lstm_cell_26/Relu
lstm_cell_26/mul_1Mullstm_cell_26/Sigmoid:y:0lstm_cell_26/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
lstm_cell_26/mul_1
lstm_cell_26/add_1AddV2lstm_cell_26/mul:z:0lstm_cell_26/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
lstm_cell_26/add_1
lstm_cell_26/Sigmoid_2Sigmoidlstm_cell_26/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
lstm_cell_26/Sigmoid_2}
lstm_cell_26/Relu_1Relulstm_cell_26/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
lstm_cell_26/Relu_1¡
lstm_cell_26/mul_2Mullstm_cell_26/Sigmoid_2:y:0!lstm_cell_26/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
lstm_cell_26/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿZ  2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_26_matmul_readvariableop_resource-lstm_cell_26_matmul_1_readvariableop_resource,lstm_cell_26_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿÚ:ÿÿÿÿÿÿÿÿÿÚ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_39975471*
condR
while_cond_39975470*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿÚ:ÿÿÿÿÿÿÿÿÿÚ: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿZ  22
0TensorArrayV2Stack/TensorListStack/element_shapeò
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÚ*
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
:ÿÿÿÿÿÿÿÿÿÚ*
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
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÚ2
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
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÚ2

IdentityÈ
NoOpNoOp$^lstm_cell_26/BiasAdd/ReadVariableOp#^lstm_cell_26/MatMul/ReadVariableOp%^lstm_cell_26/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ]: : : 2J
#lstm_cell_26/BiasAdd/ReadVariableOp#lstm_cell_26/BiasAdd/ReadVariableOp2H
"lstm_cell_26/MatMul/ReadVariableOp"lstm_cell_26/MatMul/ReadVariableOp2L
$lstm_cell_26/MatMul_1/ReadVariableOp$lstm_cell_26/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ]
"
_user_specified_name
inputs/0
¶
¸
*__inference_lstm_26_layer_call_fn_39975242

inputs
unknown:	]è

	unknown_0:
Úè

	unknown_1:	è

identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_lstm_26_layer_call_and_return_conditional_losses_399736992
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2

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
è^

)sequential_13_lstm_27_while_body_39972170H
Dsequential_13_lstm_27_while_sequential_13_lstm_27_while_loop_counterN
Jsequential_13_lstm_27_while_sequential_13_lstm_27_while_maximum_iterations+
'sequential_13_lstm_27_while_placeholder-
)sequential_13_lstm_27_while_placeholder_1-
)sequential_13_lstm_27_while_placeholder_2-
)sequential_13_lstm_27_while_placeholder_3G
Csequential_13_lstm_27_while_sequential_13_lstm_27_strided_slice_1_0
sequential_13_lstm_27_while_tensorarrayv2read_tensorlistgetitem_sequential_13_lstm_27_tensorarrayunstack_tensorlistfromtensor_0]
Isequential_13_lstm_27_while_lstm_cell_27_matmul_readvariableop_resource_0:
Ú_
Ksequential_13_lstm_27_while_lstm_cell_27_matmul_1_readvariableop_resource_0:
æY
Jsequential_13_lstm_27_while_lstm_cell_27_biasadd_readvariableop_resource_0:	(
$sequential_13_lstm_27_while_identity*
&sequential_13_lstm_27_while_identity_1*
&sequential_13_lstm_27_while_identity_2*
&sequential_13_lstm_27_while_identity_3*
&sequential_13_lstm_27_while_identity_4*
&sequential_13_lstm_27_while_identity_5E
Asequential_13_lstm_27_while_sequential_13_lstm_27_strided_slice_1
}sequential_13_lstm_27_while_tensorarrayv2read_tensorlistgetitem_sequential_13_lstm_27_tensorarrayunstack_tensorlistfromtensor[
Gsequential_13_lstm_27_while_lstm_cell_27_matmul_readvariableop_resource:
Ú]
Isequential_13_lstm_27_while_lstm_cell_27_matmul_1_readvariableop_resource:
æW
Hsequential_13_lstm_27_while_lstm_cell_27_biasadd_readvariableop_resource:	¢?sequential_13/lstm_27/while/lstm_cell_27/BiasAdd/ReadVariableOp¢>sequential_13/lstm_27/while/lstm_cell_27/MatMul/ReadVariableOp¢@sequential_13/lstm_27/while/lstm_cell_27/MatMul_1/ReadVariableOpï
Msequential_13/lstm_27/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿZ  2O
Msequential_13/lstm_27/while/TensorArrayV2Read/TensorListGetItem/element_shapeØ
?sequential_13/lstm_27/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemsequential_13_lstm_27_while_tensorarrayv2read_tensorlistgetitem_sequential_13_lstm_27_tensorarrayunstack_tensorlistfromtensor_0'sequential_13_lstm_27_while_placeholderVsequential_13/lstm_27/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ*
element_dtype02A
?sequential_13/lstm_27/while/TensorArrayV2Read/TensorListGetItem
>sequential_13/lstm_27/while/lstm_cell_27/MatMul/ReadVariableOpReadVariableOpIsequential_13_lstm_27_while_lstm_cell_27_matmul_readvariableop_resource_0* 
_output_shapes
:
Ú*
dtype02@
>sequential_13/lstm_27/while/lstm_cell_27/MatMul/ReadVariableOp¯
/sequential_13/lstm_27/while/lstm_cell_27/MatMulMatMulFsequential_13/lstm_27/while/TensorArrayV2Read/TensorListGetItem:item:0Fsequential_13/lstm_27/while/lstm_cell_27/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ21
/sequential_13/lstm_27/while/lstm_cell_27/MatMul
@sequential_13/lstm_27/while/lstm_cell_27/MatMul_1/ReadVariableOpReadVariableOpKsequential_13_lstm_27_while_lstm_cell_27_matmul_1_readvariableop_resource_0* 
_output_shapes
:
æ*
dtype02B
@sequential_13/lstm_27/while/lstm_cell_27/MatMul_1/ReadVariableOp
1sequential_13/lstm_27/while/lstm_cell_27/MatMul_1MatMul)sequential_13_lstm_27_while_placeholder_2Hsequential_13/lstm_27/while/lstm_cell_27/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ23
1sequential_13/lstm_27/while/lstm_cell_27/MatMul_1
,sequential_13/lstm_27/while/lstm_cell_27/addAddV29sequential_13/lstm_27/while/lstm_cell_27/MatMul:product:0;sequential_13/lstm_27/while/lstm_cell_27/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2.
,sequential_13/lstm_27/while/lstm_cell_27/add
?sequential_13/lstm_27/while/lstm_cell_27/BiasAdd/ReadVariableOpReadVariableOpJsequential_13_lstm_27_while_lstm_cell_27_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02A
?sequential_13/lstm_27/while/lstm_cell_27/BiasAdd/ReadVariableOp
0sequential_13/lstm_27/while/lstm_cell_27/BiasAddBiasAdd0sequential_13/lstm_27/while/lstm_cell_27/add:z:0Gsequential_13/lstm_27/while/lstm_cell_27/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
0sequential_13/lstm_27/while/lstm_cell_27/BiasAdd¶
8sequential_13/lstm_27/while/lstm_cell_27/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2:
8sequential_13/lstm_27/while/lstm_cell_27/split/split_dimç
.sequential_13/lstm_27/while/lstm_cell_27/splitSplitAsequential_13/lstm_27/while/lstm_cell_27/split/split_dim:output:09sequential_13/lstm_27/while/lstm_cell_27/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿæ:ÿÿÿÿÿÿÿÿÿæ:ÿÿÿÿÿÿÿÿÿæ:ÿÿÿÿÿÿÿÿÿæ*
	num_split20
.sequential_13/lstm_27/while/lstm_cell_27/splitÛ
0sequential_13/lstm_27/while/lstm_cell_27/SigmoidSigmoid7sequential_13/lstm_27/while/lstm_cell_27/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ22
0sequential_13/lstm_27/while/lstm_cell_27/Sigmoidß
2sequential_13/lstm_27/while/lstm_cell_27/Sigmoid_1Sigmoid7sequential_13/lstm_27/while/lstm_cell_27/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ24
2sequential_13/lstm_27/while/lstm_cell_27/Sigmoid_1ù
,sequential_13/lstm_27/while/lstm_cell_27/mulMul6sequential_13/lstm_27/while/lstm_cell_27/Sigmoid_1:y:0)sequential_13_lstm_27_while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2.
,sequential_13/lstm_27/while/lstm_cell_27/mulÒ
-sequential_13/lstm_27/while/lstm_cell_27/ReluRelu7sequential_13/lstm_27/while/lstm_cell_27/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2/
-sequential_13/lstm_27/while/lstm_cell_27/Relu
.sequential_13/lstm_27/while/lstm_cell_27/mul_1Mul4sequential_13/lstm_27/while/lstm_cell_27/Sigmoid:y:0;sequential_13/lstm_27/while/lstm_cell_27/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ20
.sequential_13/lstm_27/while/lstm_cell_27/mul_1
.sequential_13/lstm_27/while/lstm_cell_27/add_1AddV20sequential_13/lstm_27/while/lstm_cell_27/mul:z:02sequential_13/lstm_27/while/lstm_cell_27/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ20
.sequential_13/lstm_27/while/lstm_cell_27/add_1ß
2sequential_13/lstm_27/while/lstm_cell_27/Sigmoid_2Sigmoid7sequential_13/lstm_27/while/lstm_cell_27/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ24
2sequential_13/lstm_27/while/lstm_cell_27/Sigmoid_2Ñ
/sequential_13/lstm_27/while/lstm_cell_27/Relu_1Relu2sequential_13/lstm_27/while/lstm_cell_27/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ21
/sequential_13/lstm_27/while/lstm_cell_27/Relu_1
.sequential_13/lstm_27/while/lstm_cell_27/mul_2Mul6sequential_13/lstm_27/while/lstm_cell_27/Sigmoid_2:y:0=sequential_13/lstm_27/while/lstm_cell_27/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ20
.sequential_13/lstm_27/while/lstm_cell_27/mul_2Î
@sequential_13/lstm_27/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem)sequential_13_lstm_27_while_placeholder_1'sequential_13_lstm_27_while_placeholder2sequential_13/lstm_27/while/lstm_cell_27/mul_2:z:0*
_output_shapes
: *
element_dtype02B
@sequential_13/lstm_27/while/TensorArrayV2Write/TensorListSetItem
!sequential_13/lstm_27/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2#
!sequential_13/lstm_27/while/add/yÁ
sequential_13/lstm_27/while/addAddV2'sequential_13_lstm_27_while_placeholder*sequential_13/lstm_27/while/add/y:output:0*
T0*
_output_shapes
: 2!
sequential_13/lstm_27/while/add
#sequential_13/lstm_27/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2%
#sequential_13/lstm_27/while/add_1/yä
!sequential_13/lstm_27/while/add_1AddV2Dsequential_13_lstm_27_while_sequential_13_lstm_27_while_loop_counter,sequential_13/lstm_27/while/add_1/y:output:0*
T0*
_output_shapes
: 2#
!sequential_13/lstm_27/while/add_1Ã
$sequential_13/lstm_27/while/IdentityIdentity%sequential_13/lstm_27/while/add_1:z:0!^sequential_13/lstm_27/while/NoOp*
T0*
_output_shapes
: 2&
$sequential_13/lstm_27/while/Identityì
&sequential_13/lstm_27/while/Identity_1IdentityJsequential_13_lstm_27_while_sequential_13_lstm_27_while_maximum_iterations!^sequential_13/lstm_27/while/NoOp*
T0*
_output_shapes
: 2(
&sequential_13/lstm_27/while/Identity_1Å
&sequential_13/lstm_27/while/Identity_2Identity#sequential_13/lstm_27/while/add:z:0!^sequential_13/lstm_27/while/NoOp*
T0*
_output_shapes
: 2(
&sequential_13/lstm_27/while/Identity_2ò
&sequential_13/lstm_27/while/Identity_3IdentityPsequential_13/lstm_27/while/TensorArrayV2Write/TensorListSetItem:output_handle:0!^sequential_13/lstm_27/while/NoOp*
T0*
_output_shapes
: 2(
&sequential_13/lstm_27/while/Identity_3æ
&sequential_13/lstm_27/while/Identity_4Identity2sequential_13/lstm_27/while/lstm_cell_27/mul_2:z:0!^sequential_13/lstm_27/while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2(
&sequential_13/lstm_27/while/Identity_4æ
&sequential_13/lstm_27/while/Identity_5Identity2sequential_13/lstm_27/while/lstm_cell_27/add_1:z:0!^sequential_13/lstm_27/while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2(
&sequential_13/lstm_27/while/Identity_5Ì
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
&sequential_13_lstm_27_while_identity_5/sequential_13/lstm_27/while/Identity_5:output:0"
Hsequential_13_lstm_27_while_lstm_cell_27_biasadd_readvariableop_resourceJsequential_13_lstm_27_while_lstm_cell_27_biasadd_readvariableop_resource_0"
Isequential_13_lstm_27_while_lstm_cell_27_matmul_1_readvariableop_resourceKsequential_13_lstm_27_while_lstm_cell_27_matmul_1_readvariableop_resource_0"
Gsequential_13_lstm_27_while_lstm_cell_27_matmul_readvariableop_resourceIsequential_13_lstm_27_while_lstm_cell_27_matmul_readvariableop_resource_0"
Asequential_13_lstm_27_while_sequential_13_lstm_27_strided_slice_1Csequential_13_lstm_27_while_sequential_13_lstm_27_strided_slice_1_0"
}sequential_13_lstm_27_while_tensorarrayv2read_tensorlistgetitem_sequential_13_lstm_27_tensorarrayunstack_tensorlistfromtensorsequential_13_lstm_27_while_tensorarrayv2read_tensorlistgetitem_sequential_13_lstm_27_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿæ:ÿÿÿÿÿÿÿÿÿæ: : : : : 2
?sequential_13/lstm_27/while/lstm_cell_27/BiasAdd/ReadVariableOp?sequential_13/lstm_27/while/lstm_cell_27/BiasAdd/ReadVariableOp2
>sequential_13/lstm_27/while/lstm_cell_27/MatMul/ReadVariableOp>sequential_13/lstm_27/while/lstm_cell_27/MatMul/ReadVariableOp2
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
:ÿÿÿÿÿÿÿÿÿæ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ:

_output_shapes
: :

_output_shapes
: 
åJ
Ô

lstm_26_while_body_39974935,
(lstm_26_while_lstm_26_while_loop_counter2
.lstm_26_while_lstm_26_while_maximum_iterations
lstm_26_while_placeholder
lstm_26_while_placeholder_1
lstm_26_while_placeholder_2
lstm_26_while_placeholder_3+
'lstm_26_while_lstm_26_strided_slice_1_0g
clstm_26_while_tensorarrayv2read_tensorlistgetitem_lstm_26_tensorarrayunstack_tensorlistfromtensor_0N
;lstm_26_while_lstm_cell_26_matmul_readvariableop_resource_0:	]è
Q
=lstm_26_while_lstm_cell_26_matmul_1_readvariableop_resource_0:
Úè
K
<lstm_26_while_lstm_cell_26_biasadd_readvariableop_resource_0:	è

lstm_26_while_identity
lstm_26_while_identity_1
lstm_26_while_identity_2
lstm_26_while_identity_3
lstm_26_while_identity_4
lstm_26_while_identity_5)
%lstm_26_while_lstm_26_strided_slice_1e
alstm_26_while_tensorarrayv2read_tensorlistgetitem_lstm_26_tensorarrayunstack_tensorlistfromtensorL
9lstm_26_while_lstm_cell_26_matmul_readvariableop_resource:	]è
O
;lstm_26_while_lstm_cell_26_matmul_1_readvariableop_resource:
Úè
I
:lstm_26_while_lstm_cell_26_biasadd_readvariableop_resource:	è
¢1lstm_26/while/lstm_cell_26/BiasAdd/ReadVariableOp¢0lstm_26/while/lstm_cell_26/MatMul/ReadVariableOp¢2lstm_26/while/lstm_cell_26/MatMul_1/ReadVariableOpÓ
?lstm_26/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ]   2A
?lstm_26/while/TensorArrayV2Read/TensorListGetItem/element_shape
1lstm_26/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemclstm_26_while_tensorarrayv2read_tensorlistgetitem_lstm_26_tensorarrayunstack_tensorlistfromtensor_0lstm_26_while_placeholderHlstm_26/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]*
element_dtype023
1lstm_26/while/TensorArrayV2Read/TensorListGetItemá
0lstm_26/while/lstm_cell_26/MatMul/ReadVariableOpReadVariableOp;lstm_26_while_lstm_cell_26_matmul_readvariableop_resource_0*
_output_shapes
:	]è
*
dtype022
0lstm_26/while/lstm_cell_26/MatMul/ReadVariableOp÷
!lstm_26/while/lstm_cell_26/MatMulMatMul8lstm_26/while/TensorArrayV2Read/TensorListGetItem:item:08lstm_26/while/lstm_cell_26/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè
2#
!lstm_26/while/lstm_cell_26/MatMulè
2lstm_26/while/lstm_cell_26/MatMul_1/ReadVariableOpReadVariableOp=lstm_26_while_lstm_cell_26_matmul_1_readvariableop_resource_0* 
_output_shapes
:
Úè
*
dtype024
2lstm_26/while/lstm_cell_26/MatMul_1/ReadVariableOpà
#lstm_26/while/lstm_cell_26/MatMul_1MatMullstm_26_while_placeholder_2:lstm_26/while/lstm_cell_26/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè
2%
#lstm_26/while/lstm_cell_26/MatMul_1Ø
lstm_26/while/lstm_cell_26/addAddV2+lstm_26/while/lstm_cell_26/MatMul:product:0-lstm_26/while/lstm_cell_26/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè
2 
lstm_26/while/lstm_cell_26/addà
1lstm_26/while/lstm_cell_26/BiasAdd/ReadVariableOpReadVariableOp<lstm_26_while_lstm_cell_26_biasadd_readvariableop_resource_0*
_output_shapes	
:è
*
dtype023
1lstm_26/while/lstm_cell_26/BiasAdd/ReadVariableOpå
"lstm_26/while/lstm_cell_26/BiasAddBiasAdd"lstm_26/while/lstm_cell_26/add:z:09lstm_26/while/lstm_cell_26/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè
2$
"lstm_26/while/lstm_cell_26/BiasAdd
*lstm_26/while/lstm_cell_26/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2,
*lstm_26/while/lstm_cell_26/split/split_dim¯
 lstm_26/while/lstm_cell_26/splitSplit3lstm_26/while/lstm_cell_26/split/split_dim:output:0+lstm_26/while/lstm_cell_26/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿÚ:ÿÿÿÿÿÿÿÿÿÚ:ÿÿÿÿÿÿÿÿÿÚ:ÿÿÿÿÿÿÿÿÿÚ*
	num_split2"
 lstm_26/while/lstm_cell_26/split±
"lstm_26/while/lstm_cell_26/SigmoidSigmoid)lstm_26/while/lstm_cell_26/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2$
"lstm_26/while/lstm_cell_26/Sigmoidµ
$lstm_26/while/lstm_cell_26/Sigmoid_1Sigmoid)lstm_26/while/lstm_cell_26/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2&
$lstm_26/while/lstm_cell_26/Sigmoid_1Á
lstm_26/while/lstm_cell_26/mulMul(lstm_26/while/lstm_cell_26/Sigmoid_1:y:0lstm_26_while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2 
lstm_26/while/lstm_cell_26/mul¨
lstm_26/while/lstm_cell_26/ReluRelu)lstm_26/while/lstm_cell_26/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2!
lstm_26/while/lstm_cell_26/ReluÕ
 lstm_26/while/lstm_cell_26/mul_1Mul&lstm_26/while/lstm_cell_26/Sigmoid:y:0-lstm_26/while/lstm_cell_26/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2"
 lstm_26/while/lstm_cell_26/mul_1Ê
 lstm_26/while/lstm_cell_26/add_1AddV2"lstm_26/while/lstm_cell_26/mul:z:0$lstm_26/while/lstm_cell_26/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2"
 lstm_26/while/lstm_cell_26/add_1µ
$lstm_26/while/lstm_cell_26/Sigmoid_2Sigmoid)lstm_26/while/lstm_cell_26/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2&
$lstm_26/while/lstm_cell_26/Sigmoid_2§
!lstm_26/while/lstm_cell_26/Relu_1Relu$lstm_26/while/lstm_cell_26/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2#
!lstm_26/while/lstm_cell_26/Relu_1Ù
 lstm_26/while/lstm_cell_26/mul_2Mul(lstm_26/while/lstm_cell_26/Sigmoid_2:y:0/lstm_26/while/lstm_cell_26/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2"
 lstm_26/while/lstm_cell_26/mul_2
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
lstm_26/while/add/y
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
lstm_26/while/add_1/y
lstm_26/while/add_1AddV2(lstm_26_while_lstm_26_while_loop_counterlstm_26/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lstm_26/while/add_1
lstm_26/while/IdentityIdentitylstm_26/while/add_1:z:0^lstm_26/while/NoOp*
T0*
_output_shapes
: 2
lstm_26/while/Identity¦
lstm_26/while/Identity_1Identity.lstm_26_while_lstm_26_while_maximum_iterations^lstm_26/while/NoOp*
T0*
_output_shapes
: 2
lstm_26/while/Identity_1
lstm_26/while/Identity_2Identitylstm_26/while/add:z:0^lstm_26/while/NoOp*
T0*
_output_shapes
: 2
lstm_26/while/Identity_2º
lstm_26/while/Identity_3IdentityBlstm_26/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_26/while/NoOp*
T0*
_output_shapes
: 2
lstm_26/while/Identity_3®
lstm_26/while/Identity_4Identity$lstm_26/while/lstm_cell_26/mul_2:z:0^lstm_26/while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
lstm_26/while/Identity_4®
lstm_26/while/Identity_5Identity$lstm_26/while/lstm_cell_26/add_1:z:0^lstm_26/while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
lstm_26/while/Identity_5
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
9lstm_26_while_lstm_cell_26_matmul_readvariableop_resource;lstm_26_while_lstm_cell_26_matmul_readvariableop_resource_0"È
alstm_26_while_tensorarrayv2read_tensorlistgetitem_lstm_26_tensorarrayunstack_tensorlistfromtensorclstm_26_while_tensorarrayv2read_tensorlistgetitem_lstm_26_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿÚ:ÿÿÿÿÿÿÿÿÿÚ: : : : : 2f
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
:ÿÿÿÿÿÿÿÿÿÚ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ:

_output_shapes
: :

_output_shapes
: 
ËF

E__inference_lstm_26_layer_call_and_return_conditional_losses_39972650

inputs(
lstm_cell_26_39972568:	]è
)
lstm_cell_26_39972570:
Úè
$
lstm_cell_26_39972572:	è

identity¢$lstm_cell_26/StatefulPartitionedCall¢whileD
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
B :Ú2
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
B :Ú2
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
:ÿÿÿÿÿÿÿÿÿÚ2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :Ú2
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
B :Ú2
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
:ÿÿÿÿÿÿÿÿÿÚ2	
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
$lstm_cell_26/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_26_39972568lstm_cell_26_39972570lstm_cell_26_39972572*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿÚ:ÿÿÿÿÿÿÿÿÿÚ:ÿÿÿÿÿÿÿÿÿÚ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_lstm_cell_26_layer_call_and_return_conditional_losses_399725032&
$lstm_cell_26/StatefulPartitionedCall
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿZ  2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_26_39972568lstm_cell_26_39972570lstm_cell_26_39972572*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿÚ:ÿÿÿÿÿÿÿÿÿÚ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_39972581*
condR
while_cond_39972580*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿÚ:ÿÿÿÿÿÿÿÿÿÚ: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿZ  22
0TensorArrayV2Stack/TensorListStack/element_shapeò
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÚ*
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
:ÿÿÿÿÿÿÿÿÿÚ*
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
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÚ2
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
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÚ2

Identity}
NoOpNoOp%^lstm_cell_26/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ]: : : 2L
$lstm_cell_26/StatefulPartitionedCall$lstm_cell_26/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ]
 
_user_specified_nameinputs
ã
Í
while_cond_39976296
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_39976296___redundant_placeholder06
2while_while_cond_39976296___redundant_placeholder16
2while_while_cond_39976296___redundant_placeholder26
2while_while_cond_39976296___redundant_placeholder3
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
B: : : : :ÿÿÿÿÿÿÿÿÿæ:ÿÿÿÿÿÿÿÿÿæ: ::::: 

_output_shapes
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
:ÿÿÿÿÿÿÿÿÿæ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ:

_output_shapes
: :

_output_shapes
:
°?
Ô
while_body_39975773
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_26_matmul_readvariableop_resource_0:	]è
I
5while_lstm_cell_26_matmul_1_readvariableop_resource_0:
Úè
C
4while_lstm_cell_26_biasadd_readvariableop_resource_0:	è

while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_26_matmul_readvariableop_resource:	]è
G
3while_lstm_cell_26_matmul_1_readvariableop_resource:
Úè
A
2while_lstm_cell_26_biasadd_readvariableop_resource:	è
¢)while/lstm_cell_26/BiasAdd/ReadVariableOp¢(while/lstm_cell_26/MatMul/ReadVariableOp¢*while/lstm_cell_26/MatMul_1/ReadVariableOpÃ
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
(while/lstm_cell_26/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_26_matmul_readvariableop_resource_0*
_output_shapes
:	]è
*
dtype02*
(while/lstm_cell_26/MatMul/ReadVariableOp×
while/lstm_cell_26/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_26/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè
2
while/lstm_cell_26/MatMulÐ
*while/lstm_cell_26/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_26_matmul_1_readvariableop_resource_0* 
_output_shapes
:
Úè
*
dtype02,
*while/lstm_cell_26/MatMul_1/ReadVariableOpÀ
while/lstm_cell_26/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_26/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè
2
while/lstm_cell_26/MatMul_1¸
while/lstm_cell_26/addAddV2#while/lstm_cell_26/MatMul:product:0%while/lstm_cell_26/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè
2
while/lstm_cell_26/addÈ
)while/lstm_cell_26/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_26_biasadd_readvariableop_resource_0*
_output_shapes	
:è
*
dtype02+
)while/lstm_cell_26/BiasAdd/ReadVariableOpÅ
while/lstm_cell_26/BiasAddBiasAddwhile/lstm_cell_26/add:z:01while/lstm_cell_26/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè
2
while/lstm_cell_26/BiasAdd
"while/lstm_cell_26/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_26/split/split_dim
while/lstm_cell_26/splitSplit+while/lstm_cell_26/split/split_dim:output:0#while/lstm_cell_26/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿÚ:ÿÿÿÿÿÿÿÿÿÚ:ÿÿÿÿÿÿÿÿÿÚ:ÿÿÿÿÿÿÿÿÿÚ*
	num_split2
while/lstm_cell_26/split
while/lstm_cell_26/SigmoidSigmoid!while/lstm_cell_26/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
while/lstm_cell_26/Sigmoid
while/lstm_cell_26/Sigmoid_1Sigmoid!while/lstm_cell_26/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
while/lstm_cell_26/Sigmoid_1¡
while/lstm_cell_26/mulMul while/lstm_cell_26/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
while/lstm_cell_26/mul
while/lstm_cell_26/ReluRelu!while/lstm_cell_26/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
while/lstm_cell_26/Reluµ
while/lstm_cell_26/mul_1Mulwhile/lstm_cell_26/Sigmoid:y:0%while/lstm_cell_26/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
while/lstm_cell_26/mul_1ª
while/lstm_cell_26/add_1AddV2while/lstm_cell_26/mul:z:0while/lstm_cell_26/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
while/lstm_cell_26/add_1
while/lstm_cell_26/Sigmoid_2Sigmoid!while/lstm_cell_26/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
while/lstm_cell_26/Sigmoid_2
while/lstm_cell_26/Relu_1Reluwhile/lstm_cell_26/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
while/lstm_cell_26/Relu_1¹
while/lstm_cell_26/mul_2Mul while/lstm_cell_26/Sigmoid_2:y:0'while/lstm_cell_26/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
while/lstm_cell_26/mul_2à
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
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_26/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_26/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
while/Identity_5Þ

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
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿÚ:ÿÿÿÿÿÿÿÿÿÚ: : : : : 2V
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
:ÿÿÿÿÿÿÿÿÿÚ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ:

_output_shapes
: :

_output_shapes
: 
Ç
ù
/__inference_lstm_cell_26_layer_call_fn_39976633

inputs
states_0
states_1
unknown:	]è

	unknown_0:
Úè

	unknown_1:	è

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
<:ÿÿÿÿÿÿÿÿÿÚ:ÿÿÿÿÿÿÿÿÿÚ:ÿÿÿÿÿÿÿÿÿÚ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_lstm_cell_26_layer_call_and_return_conditional_losses_399725032
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2

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
A:ÿÿÿÿÿÿÿÿÿ]:ÿÿÿÿÿÿÿÿÿÚ:ÿÿÿÿÿÿÿÿÿÚ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
 
_user_specified_nameinputs:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ
"
_user_specified_name
states/0:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ
"
_user_specified_name
states/1


+__inference_dense_13_layer_call_fn_39976568

inputs
unknown:	æ
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
F__inference_dense_13_layer_call_and_return_conditional_losses_399739102
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
:ÿÿÿÿÿÿÿÿÿæ: : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ
 
_user_specified_nameinputs
ã
»
*__inference_lstm_27_layer_call_fn_39975895
inputs_0
unknown:
Ú
	unknown_0:
æ
	unknown_1:	
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿæ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_lstm_27_layer_call_and_return_conditional_losses_399730702
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿæ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÚ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÚ
"
_user_specified_name
inputs/0
´?
Ö
while_body_39976146
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
3while_lstm_cell_27_matmul_readvariableop_resource_0:
ÚI
5while_lstm_cell_27_matmul_1_readvariableop_resource_0:
æC
4while_lstm_cell_27_biasadd_readvariableop_resource_0:	
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
1while_lstm_cell_27_matmul_readvariableop_resource:
ÚG
3while_lstm_cell_27_matmul_1_readvariableop_resource:
æA
2while_lstm_cell_27_biasadd_readvariableop_resource:	¢)while/lstm_cell_27/BiasAdd/ReadVariableOp¢(while/lstm_cell_27/MatMul/ReadVariableOp¢*while/lstm_cell_27/MatMul_1/ReadVariableOpÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿZ  29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÔ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemÊ
(while/lstm_cell_27/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_27_matmul_readvariableop_resource_0* 
_output_shapes
:
Ú*
dtype02*
(while/lstm_cell_27/MatMul/ReadVariableOp×
while/lstm_cell_27/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_27/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_27/MatMulÐ
*while/lstm_cell_27/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_27_matmul_1_readvariableop_resource_0* 
_output_shapes
:
æ*
dtype02,
*while/lstm_cell_27/MatMul_1/ReadVariableOpÀ
while/lstm_cell_27/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_27/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_27/MatMul_1¸
while/lstm_cell_27/addAddV2#while/lstm_cell_27/MatMul:product:0%while/lstm_cell_27/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_27/addÈ
)while/lstm_cell_27/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_27_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02+
)while/lstm_cell_27/BiasAdd/ReadVariableOpÅ
while/lstm_cell_27/BiasAddBiasAddwhile/lstm_cell_27/add:z:01while/lstm_cell_27/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_27/BiasAdd
"while/lstm_cell_27/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_27/split/split_dim
while/lstm_cell_27/splitSplit+while/lstm_cell_27/split/split_dim:output:0#while/lstm_cell_27/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿæ:ÿÿÿÿÿÿÿÿÿæ:ÿÿÿÿÿÿÿÿÿæ:ÿÿÿÿÿÿÿÿÿæ*
	num_split2
while/lstm_cell_27/split
while/lstm_cell_27/SigmoidSigmoid!while/lstm_cell_27/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2
while/lstm_cell_27/Sigmoid
while/lstm_cell_27/Sigmoid_1Sigmoid!while/lstm_cell_27/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2
while/lstm_cell_27/Sigmoid_1¡
while/lstm_cell_27/mulMul while/lstm_cell_27/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2
while/lstm_cell_27/mul
while/lstm_cell_27/ReluRelu!while/lstm_cell_27/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2
while/lstm_cell_27/Reluµ
while/lstm_cell_27/mul_1Mulwhile/lstm_cell_27/Sigmoid:y:0%while/lstm_cell_27/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2
while/lstm_cell_27/mul_1ª
while/lstm_cell_27/add_1AddV2while/lstm_cell_27/mul:z:0while/lstm_cell_27/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2
while/lstm_cell_27/add_1
while/lstm_cell_27/Sigmoid_2Sigmoid!while/lstm_cell_27/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2
while/lstm_cell_27/Sigmoid_2
while/lstm_cell_27/Relu_1Reluwhile/lstm_cell_27/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2
while/lstm_cell_27/Relu_1¹
while/lstm_cell_27/mul_2Mul while/lstm_cell_27/Sigmoid_2:y:0'while/lstm_cell_27/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2
while/lstm_cell_27/mul_2à
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
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_27/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_27/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2
while/Identity_5Þ

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
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿæ:ÿÿÿÿÿÿÿÿÿæ: : : : : 2V
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
:ÿÿÿÿÿÿÿÿÿæ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ:

_output_shapes
: :

_output_shapes
: 
ã
Í
while_cond_39976447
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_39976447___redundant_placeholder06
2while_while_cond_39976447___redundant_placeholder16
2while_while_cond_39976447___redundant_placeholder26
2while_while_cond_39976447___redundant_placeholder3
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
B: : : : :ÿÿÿÿÿÿÿÿÿæ:ÿÿÿÿÿÿÿÿÿæ: ::::: 

_output_shapes
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
:ÿÿÿÿÿÿÿÿÿæ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ:

_output_shapes
: :

_output_shapes
:
ËF

E__inference_lstm_26_layer_call_and_return_conditional_losses_39972440

inputs(
lstm_cell_26_39972358:	]è
)
lstm_cell_26_39972360:
Úè
$
lstm_cell_26_39972362:	è

identity¢$lstm_cell_26/StatefulPartitionedCall¢whileD
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
B :Ú2
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
B :Ú2
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
:ÿÿÿÿÿÿÿÿÿÚ2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :Ú2
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
B :Ú2
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
:ÿÿÿÿÿÿÿÿÿÚ2	
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
$lstm_cell_26/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_26_39972358lstm_cell_26_39972360lstm_cell_26_39972362*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿÚ:ÿÿÿÿÿÿÿÿÿÚ:ÿÿÿÿÿÿÿÿÿÚ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_lstm_cell_26_layer_call_and_return_conditional_losses_399723572&
$lstm_cell_26/StatefulPartitionedCall
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿZ  2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_26_39972358lstm_cell_26_39972360lstm_cell_26_39972362*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿÚ:ÿÿÿÿÿÿÿÿÿÚ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_39972371*
condR
while_cond_39972370*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿÚ:ÿÿÿÿÿÿÿÿÿÚ: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿZ  22
0TensorArrayV2Stack/TensorListStack/element_shapeò
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÚ*
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
:ÿÿÿÿÿÿÿÿÿÚ*
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
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÚ2
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
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÚ2

Identity}
NoOpNoOp%^lstm_cell_26/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ]: : : 2L
$lstm_cell_26/StatefulPartitionedCall$lstm_cell_26/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ]
 
_user_specified_nameinputs
´?
Ö
while_body_39973780
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
3while_lstm_cell_27_matmul_readvariableop_resource_0:
ÚI
5while_lstm_cell_27_matmul_1_readvariableop_resource_0:
æC
4while_lstm_cell_27_biasadd_readvariableop_resource_0:	
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
1while_lstm_cell_27_matmul_readvariableop_resource:
ÚG
3while_lstm_cell_27_matmul_1_readvariableop_resource:
æA
2while_lstm_cell_27_biasadd_readvariableop_resource:	¢)while/lstm_cell_27/BiasAdd/ReadVariableOp¢(while/lstm_cell_27/MatMul/ReadVariableOp¢*while/lstm_cell_27/MatMul_1/ReadVariableOpÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿZ  29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÔ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemÊ
(while/lstm_cell_27/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_27_matmul_readvariableop_resource_0* 
_output_shapes
:
Ú*
dtype02*
(while/lstm_cell_27/MatMul/ReadVariableOp×
while/lstm_cell_27/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_27/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_27/MatMulÐ
*while/lstm_cell_27/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_27_matmul_1_readvariableop_resource_0* 
_output_shapes
:
æ*
dtype02,
*while/lstm_cell_27/MatMul_1/ReadVariableOpÀ
while/lstm_cell_27/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_27/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_27/MatMul_1¸
while/lstm_cell_27/addAddV2#while/lstm_cell_27/MatMul:product:0%while/lstm_cell_27/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_27/addÈ
)while/lstm_cell_27/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_27_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02+
)while/lstm_cell_27/BiasAdd/ReadVariableOpÅ
while/lstm_cell_27/BiasAddBiasAddwhile/lstm_cell_27/add:z:01while/lstm_cell_27/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_27/BiasAdd
"while/lstm_cell_27/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_27/split/split_dim
while/lstm_cell_27/splitSplit+while/lstm_cell_27/split/split_dim:output:0#while/lstm_cell_27/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿæ:ÿÿÿÿÿÿÿÿÿæ:ÿÿÿÿÿÿÿÿÿæ:ÿÿÿÿÿÿÿÿÿæ*
	num_split2
while/lstm_cell_27/split
while/lstm_cell_27/SigmoidSigmoid!while/lstm_cell_27/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2
while/lstm_cell_27/Sigmoid
while/lstm_cell_27/Sigmoid_1Sigmoid!while/lstm_cell_27/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2
while/lstm_cell_27/Sigmoid_1¡
while/lstm_cell_27/mulMul while/lstm_cell_27/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2
while/lstm_cell_27/mul
while/lstm_cell_27/ReluRelu!while/lstm_cell_27/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2
while/lstm_cell_27/Reluµ
while/lstm_cell_27/mul_1Mulwhile/lstm_cell_27/Sigmoid:y:0%while/lstm_cell_27/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2
while/lstm_cell_27/mul_1ª
while/lstm_cell_27/add_1AddV2while/lstm_cell_27/mul:z:0while/lstm_cell_27/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2
while/lstm_cell_27/add_1
while/lstm_cell_27/Sigmoid_2Sigmoid!while/lstm_cell_27/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2
while/lstm_cell_27/Sigmoid_2
while/lstm_cell_27/Relu_1Reluwhile/lstm_cell_27/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2
while/lstm_cell_27/Relu_1¹
while/lstm_cell_27/mul_2Mul while/lstm_cell_27/Sigmoid_2:y:0'while/lstm_cell_27/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2
while/lstm_cell_27/mul_2à
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
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_27/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_27/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2
while/Identity_5Þ

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
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿæ:ÿÿÿÿÿÿÿÿÿæ: : : : : 2V
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
:ÿÿÿÿÿÿÿÿÿæ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ:

_output_shapes
: :

_output_shapes
: 
\

E__inference_lstm_26_layer_call_and_return_conditional_losses_39975706

inputs>
+lstm_cell_26_matmul_readvariableop_resource:	]è
A
-lstm_cell_26_matmul_1_readvariableop_resource:
Úè
;
,lstm_cell_26_biasadd_readvariableop_resource:	è

identity¢#lstm_cell_26/BiasAdd/ReadVariableOp¢"lstm_cell_26/MatMul/ReadVariableOp¢$lstm_cell_26/MatMul_1/ReadVariableOp¢whileD
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
B :Ú2
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
B :Ú2
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
:ÿÿÿÿÿÿÿÿÿÚ2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :Ú2
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
B :Ú2
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
:ÿÿÿÿÿÿÿÿÿÚ2	
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
"lstm_cell_26/MatMul/ReadVariableOpReadVariableOp+lstm_cell_26_matmul_readvariableop_resource*
_output_shapes
:	]è
*
dtype02$
"lstm_cell_26/MatMul/ReadVariableOp­
lstm_cell_26/MatMulMatMulstrided_slice_2:output:0*lstm_cell_26/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè
2
lstm_cell_26/MatMul¼
$lstm_cell_26/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_26_matmul_1_readvariableop_resource* 
_output_shapes
:
Úè
*
dtype02&
$lstm_cell_26/MatMul_1/ReadVariableOp©
lstm_cell_26/MatMul_1MatMulzeros:output:0,lstm_cell_26/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè
2
lstm_cell_26/MatMul_1 
lstm_cell_26/addAddV2lstm_cell_26/MatMul:product:0lstm_cell_26/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè
2
lstm_cell_26/add´
#lstm_cell_26/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_26_biasadd_readvariableop_resource*
_output_shapes	
:è
*
dtype02%
#lstm_cell_26/BiasAdd/ReadVariableOp­
lstm_cell_26/BiasAddBiasAddlstm_cell_26/add:z:0+lstm_cell_26/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè
2
lstm_cell_26/BiasAdd~
lstm_cell_26/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_26/split/split_dim÷
lstm_cell_26/splitSplit%lstm_cell_26/split/split_dim:output:0lstm_cell_26/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿÚ:ÿÿÿÿÿÿÿÿÿÚ:ÿÿÿÿÿÿÿÿÿÚ:ÿÿÿÿÿÿÿÿÿÚ*
	num_split2
lstm_cell_26/split
lstm_cell_26/SigmoidSigmoidlstm_cell_26/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
lstm_cell_26/Sigmoid
lstm_cell_26/Sigmoid_1Sigmoidlstm_cell_26/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
lstm_cell_26/Sigmoid_1
lstm_cell_26/mulMullstm_cell_26/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
lstm_cell_26/mul~
lstm_cell_26/ReluRelulstm_cell_26/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
lstm_cell_26/Relu
lstm_cell_26/mul_1Mullstm_cell_26/Sigmoid:y:0lstm_cell_26/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
lstm_cell_26/mul_1
lstm_cell_26/add_1AddV2lstm_cell_26/mul:z:0lstm_cell_26/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
lstm_cell_26/add_1
lstm_cell_26/Sigmoid_2Sigmoidlstm_cell_26/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
lstm_cell_26/Sigmoid_2}
lstm_cell_26/Relu_1Relulstm_cell_26/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
lstm_cell_26/Relu_1¡
lstm_cell_26/mul_2Mullstm_cell_26/Sigmoid_2:y:0!lstm_cell_26/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
lstm_cell_26/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿZ  2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_26_matmul_readvariableop_resource-lstm_cell_26_matmul_1_readvariableop_resource,lstm_cell_26_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿÚ:ÿÿÿÿÿÿÿÿÿÚ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_39975622*
condR
while_cond_39975621*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿÚ:ÿÿÿÿÿÿÿÿÿÚ: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿZ  22
0TensorArrayV2Stack/TensorListStack/element_shapeé
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ*
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
:ÿÿÿÿÿÿÿÿÿÚ*
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
:ÿÿÿÿÿÿÿÿÿÚ2
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
:ÿÿÿÿÿÿÿÿÿÚ2

IdentityÈ
NoOpNoOp$^lstm_cell_26/BiasAdd/ReadVariableOp#^lstm_cell_26/MatMul/ReadVariableOp%^lstm_cell_26/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ]: : : 2J
#lstm_cell_26/BiasAdd/ReadVariableOp#lstm_cell_26/BiasAdd/ReadVariableOp2H
"lstm_cell_26/MatMul/ReadVariableOp"lstm_cell_26/MatMul/ReadVariableOp2L
$lstm_cell_26/MatMul_1/ReadVariableOp$lstm_cell_26/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
 
_user_specified_nameinputs


J__inference_lstm_cell_27_layer_call_and_return_conditional_losses_39972987

inputs

states
states_12
matmul_readvariableop_resource:
Ú4
 matmul_1_readvariableop_resource:
æ.
biasadd_readvariableop_resource:	
identity

identity_1

identity_2¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
Ú*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
æ*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
add
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
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
P:ÿÿÿÿÿÿÿÿÿæ:ÿÿÿÿÿÿÿÿÿæ:ÿÿÿÿÿÿÿÿÿæ:ÿÿÿÿÿÿÿÿÿæ*
	num_split2
split`
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2	
Sigmoidd
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2
	Sigmoid_1]
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2
mulW
ReluRelusplit:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2
Relui
mul_1MulSigmoid:y:0Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2
mul_1^
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2
add_1d
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2
	Sigmoid_2V
Relu_1Relu	add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2
Relu_1m
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2
mul_2e
IdentityIdentity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2

Identityi

Identity_1Identity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2

Identity_1i

Identity_2Identity	add_1:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2

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
B:ÿÿÿÿÿÿÿÿÿÚ:ÿÿÿÿÿÿÿÿÿæ:ÿÿÿÿÿÿÿÿÿæ: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ
 
_user_specified_nameinputs:PL
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ
 
_user_specified_namestates:PL
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ
 
_user_specified_namestates
&
õ
while_body_39973211
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_01
while_lstm_cell_27_39973235_0:
Ú1
while_lstm_cell_27_39973237_0:
æ,
while_lstm_cell_27_39973239_0:	
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor/
while_lstm_cell_27_39973235:
Ú/
while_lstm_cell_27_39973237:
æ*
while_lstm_cell_27_39973239:	¢*while/lstm_cell_27/StatefulPartitionedCallÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿZ  29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÔ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemì
*while/lstm_cell_27/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_27_39973235_0while_lstm_cell_27_39973237_0while_lstm_cell_27_39973239_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿæ:ÿÿÿÿÿÿÿÿÿæ:ÿÿÿÿÿÿÿÿÿæ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_lstm_cell_27_layer_call_and_return_conditional_losses_399731332,
*while/lstm_cell_27/StatefulPartitionedCall÷
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
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3¥
while/Identity_4Identity3while/lstm_cell_27/StatefulPartitionedCall:output:1^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2
while/Identity_4¥
while/Identity_5Identity3while/lstm_cell_27/StatefulPartitionedCall:output:2^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2
while/Identity_5

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
while_lstm_cell_27_39973235while_lstm_cell_27_39973235_0"<
while_lstm_cell_27_39973237while_lstm_cell_27_39973237_0"<
while_lstm_cell_27_39973239while_lstm_cell_27_39973239_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿæ:ÿÿÿÿÿÿÿÿÿæ: : : : : 2X
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
:ÿÿÿÿÿÿÿÿÿæ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ:

_output_shapes
: :

_output_shapes
: 
Ê
ú
/__inference_lstm_cell_27_layer_call_fn_39976731

inputs
states_0
states_1
unknown:
Ú
	unknown_0:
æ
	unknown_1:	
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
<:ÿÿÿÿÿÿÿÿÿæ:ÿÿÿÿÿÿÿÿÿæ:ÿÿÿÿÿÿÿÿÿæ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_lstm_cell_27_layer_call_and_return_conditional_losses_399731332
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2

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
B:ÿÿÿÿÿÿÿÿÿÚ:ÿÿÿÿÿÿÿÿÿæ:ÿÿÿÿÿÿÿÿÿæ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ
 
_user_specified_nameinputs:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ
"
_user_specified_name
states/0:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ
"
_user_specified_name
states/1
°?
Ô
while_body_39973615
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_26_matmul_readvariableop_resource_0:	]è
I
5while_lstm_cell_26_matmul_1_readvariableop_resource_0:
Úè
C
4while_lstm_cell_26_biasadd_readvariableop_resource_0:	è

while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_26_matmul_readvariableop_resource:	]è
G
3while_lstm_cell_26_matmul_1_readvariableop_resource:
Úè
A
2while_lstm_cell_26_biasadd_readvariableop_resource:	è
¢)while/lstm_cell_26/BiasAdd/ReadVariableOp¢(while/lstm_cell_26/MatMul/ReadVariableOp¢*while/lstm_cell_26/MatMul_1/ReadVariableOpÃ
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
(while/lstm_cell_26/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_26_matmul_readvariableop_resource_0*
_output_shapes
:	]è
*
dtype02*
(while/lstm_cell_26/MatMul/ReadVariableOp×
while/lstm_cell_26/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_26/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè
2
while/lstm_cell_26/MatMulÐ
*while/lstm_cell_26/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_26_matmul_1_readvariableop_resource_0* 
_output_shapes
:
Úè
*
dtype02,
*while/lstm_cell_26/MatMul_1/ReadVariableOpÀ
while/lstm_cell_26/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_26/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè
2
while/lstm_cell_26/MatMul_1¸
while/lstm_cell_26/addAddV2#while/lstm_cell_26/MatMul:product:0%while/lstm_cell_26/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè
2
while/lstm_cell_26/addÈ
)while/lstm_cell_26/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_26_biasadd_readvariableop_resource_0*
_output_shapes	
:è
*
dtype02+
)while/lstm_cell_26/BiasAdd/ReadVariableOpÅ
while/lstm_cell_26/BiasAddBiasAddwhile/lstm_cell_26/add:z:01while/lstm_cell_26/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè
2
while/lstm_cell_26/BiasAdd
"while/lstm_cell_26/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_26/split/split_dim
while/lstm_cell_26/splitSplit+while/lstm_cell_26/split/split_dim:output:0#while/lstm_cell_26/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿÚ:ÿÿÿÿÿÿÿÿÿÚ:ÿÿÿÿÿÿÿÿÿÚ:ÿÿÿÿÿÿÿÿÿÚ*
	num_split2
while/lstm_cell_26/split
while/lstm_cell_26/SigmoidSigmoid!while/lstm_cell_26/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
while/lstm_cell_26/Sigmoid
while/lstm_cell_26/Sigmoid_1Sigmoid!while/lstm_cell_26/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
while/lstm_cell_26/Sigmoid_1¡
while/lstm_cell_26/mulMul while/lstm_cell_26/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
while/lstm_cell_26/mul
while/lstm_cell_26/ReluRelu!while/lstm_cell_26/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
while/lstm_cell_26/Reluµ
while/lstm_cell_26/mul_1Mulwhile/lstm_cell_26/Sigmoid:y:0%while/lstm_cell_26/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
while/lstm_cell_26/mul_1ª
while/lstm_cell_26/add_1AddV2while/lstm_cell_26/mul:z:0while/lstm_cell_26/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
while/lstm_cell_26/add_1
while/lstm_cell_26/Sigmoid_2Sigmoid!while/lstm_cell_26/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
while/lstm_cell_26/Sigmoid_2
while/lstm_cell_26/Relu_1Reluwhile/lstm_cell_26/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
while/lstm_cell_26/Relu_1¹
while/lstm_cell_26/mul_2Mul while/lstm_cell_26/Sigmoid_2:y:0'while/lstm_cell_26/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
while/lstm_cell_26/mul_2à
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
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_26/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_26/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
while/Identity_5Þ

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
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿÚ:ÿÿÿÿÿÿÿÿÿÚ: : : : : 2V
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
:ÿÿÿÿÿÿÿÿÿÚ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ:

_output_shapes
: :

_output_shapes
: 
Õ!
þ
F__inference_dense_13_layer_call_and_return_conditional_losses_39976599

inputs4
!tensordot_readvariableop_resource:	æ-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢Tensordot/ReadVariableOp
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes
:	æ*
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
:ÿÿÿÿÿÿÿÿÿæ2
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
:ÿÿÿÿÿÿÿÿÿæ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ
 
_user_specified_nameinputs
ø%
à
!__inference__traced_save_39976854
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
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_dense_13_kernel_read_readvariableop(savev2_dense_13_bias_read_readvariableop6savev2_lstm_26_lstm_cell_26_kernel_read_readvariableop@savev2_lstm_26_lstm_cell_26_recurrent_kernel_read_readvariableop4savev2_lstm_26_lstm_cell_26_bias_read_readvariableop6savev2_lstm_27_lstm_cell_27_kernel_read_readvariableop@savev2_lstm_27_lstm_cell_27_recurrent_kernel_read_readvariableop4savev2_lstm_27_lstm_cell_27_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableopsavev2_const"/device:CPU:0*
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
Z: :	æ::	]è
:
Úè
:è
:
Ú:
æ:: : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	æ: 

_output_shapes
::%!

_output_shapes
:	]è
:&"
 
_output_shapes
:
Úè
:!

_output_shapes	
:è
:&"
 
_output_shapes
:
Ú:&"
 
_output_shapes
:
æ:!

_output_shapes	
::	
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
ä^

)sequential_13_lstm_26_while_body_39972022H
Dsequential_13_lstm_26_while_sequential_13_lstm_26_while_loop_counterN
Jsequential_13_lstm_26_while_sequential_13_lstm_26_while_maximum_iterations+
'sequential_13_lstm_26_while_placeholder-
)sequential_13_lstm_26_while_placeholder_1-
)sequential_13_lstm_26_while_placeholder_2-
)sequential_13_lstm_26_while_placeholder_3G
Csequential_13_lstm_26_while_sequential_13_lstm_26_strided_slice_1_0
sequential_13_lstm_26_while_tensorarrayv2read_tensorlistgetitem_sequential_13_lstm_26_tensorarrayunstack_tensorlistfromtensor_0\
Isequential_13_lstm_26_while_lstm_cell_26_matmul_readvariableop_resource_0:	]è
_
Ksequential_13_lstm_26_while_lstm_cell_26_matmul_1_readvariableop_resource_0:
Úè
Y
Jsequential_13_lstm_26_while_lstm_cell_26_biasadd_readvariableop_resource_0:	è
(
$sequential_13_lstm_26_while_identity*
&sequential_13_lstm_26_while_identity_1*
&sequential_13_lstm_26_while_identity_2*
&sequential_13_lstm_26_while_identity_3*
&sequential_13_lstm_26_while_identity_4*
&sequential_13_lstm_26_while_identity_5E
Asequential_13_lstm_26_while_sequential_13_lstm_26_strided_slice_1
}sequential_13_lstm_26_while_tensorarrayv2read_tensorlistgetitem_sequential_13_lstm_26_tensorarrayunstack_tensorlistfromtensorZ
Gsequential_13_lstm_26_while_lstm_cell_26_matmul_readvariableop_resource:	]è
]
Isequential_13_lstm_26_while_lstm_cell_26_matmul_1_readvariableop_resource:
Úè
W
Hsequential_13_lstm_26_while_lstm_cell_26_biasadd_readvariableop_resource:	è
¢?sequential_13/lstm_26/while/lstm_cell_26/BiasAdd/ReadVariableOp¢>sequential_13/lstm_26/while/lstm_cell_26/MatMul/ReadVariableOp¢@sequential_13/lstm_26/while/lstm_cell_26/MatMul_1/ReadVariableOpï
Msequential_13/lstm_26/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ]   2O
Msequential_13/lstm_26/while/TensorArrayV2Read/TensorListGetItem/element_shape×
?sequential_13/lstm_26/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemsequential_13_lstm_26_while_tensorarrayv2read_tensorlistgetitem_sequential_13_lstm_26_tensorarrayunstack_tensorlistfromtensor_0'sequential_13_lstm_26_while_placeholderVsequential_13/lstm_26/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]*
element_dtype02A
?sequential_13/lstm_26/while/TensorArrayV2Read/TensorListGetItem
>sequential_13/lstm_26/while/lstm_cell_26/MatMul/ReadVariableOpReadVariableOpIsequential_13_lstm_26_while_lstm_cell_26_matmul_readvariableop_resource_0*
_output_shapes
:	]è
*
dtype02@
>sequential_13/lstm_26/while/lstm_cell_26/MatMul/ReadVariableOp¯
/sequential_13/lstm_26/while/lstm_cell_26/MatMulMatMulFsequential_13/lstm_26/while/TensorArrayV2Read/TensorListGetItem:item:0Fsequential_13/lstm_26/while/lstm_cell_26/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè
21
/sequential_13/lstm_26/while/lstm_cell_26/MatMul
@sequential_13/lstm_26/while/lstm_cell_26/MatMul_1/ReadVariableOpReadVariableOpKsequential_13_lstm_26_while_lstm_cell_26_matmul_1_readvariableop_resource_0* 
_output_shapes
:
Úè
*
dtype02B
@sequential_13/lstm_26/while/lstm_cell_26/MatMul_1/ReadVariableOp
1sequential_13/lstm_26/while/lstm_cell_26/MatMul_1MatMul)sequential_13_lstm_26_while_placeholder_2Hsequential_13/lstm_26/while/lstm_cell_26/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè
23
1sequential_13/lstm_26/while/lstm_cell_26/MatMul_1
,sequential_13/lstm_26/while/lstm_cell_26/addAddV29sequential_13/lstm_26/while/lstm_cell_26/MatMul:product:0;sequential_13/lstm_26/while/lstm_cell_26/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè
2.
,sequential_13/lstm_26/while/lstm_cell_26/add
?sequential_13/lstm_26/while/lstm_cell_26/BiasAdd/ReadVariableOpReadVariableOpJsequential_13_lstm_26_while_lstm_cell_26_biasadd_readvariableop_resource_0*
_output_shapes	
:è
*
dtype02A
?sequential_13/lstm_26/while/lstm_cell_26/BiasAdd/ReadVariableOp
0sequential_13/lstm_26/while/lstm_cell_26/BiasAddBiasAdd0sequential_13/lstm_26/while/lstm_cell_26/add:z:0Gsequential_13/lstm_26/while/lstm_cell_26/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè
22
0sequential_13/lstm_26/while/lstm_cell_26/BiasAdd¶
8sequential_13/lstm_26/while/lstm_cell_26/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2:
8sequential_13/lstm_26/while/lstm_cell_26/split/split_dimç
.sequential_13/lstm_26/while/lstm_cell_26/splitSplitAsequential_13/lstm_26/while/lstm_cell_26/split/split_dim:output:09sequential_13/lstm_26/while/lstm_cell_26/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿÚ:ÿÿÿÿÿÿÿÿÿÚ:ÿÿÿÿÿÿÿÿÿÚ:ÿÿÿÿÿÿÿÿÿÚ*
	num_split20
.sequential_13/lstm_26/while/lstm_cell_26/splitÛ
0sequential_13/lstm_26/while/lstm_cell_26/SigmoidSigmoid7sequential_13/lstm_26/while/lstm_cell_26/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ22
0sequential_13/lstm_26/while/lstm_cell_26/Sigmoidß
2sequential_13/lstm_26/while/lstm_cell_26/Sigmoid_1Sigmoid7sequential_13/lstm_26/while/lstm_cell_26/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ24
2sequential_13/lstm_26/while/lstm_cell_26/Sigmoid_1ù
,sequential_13/lstm_26/while/lstm_cell_26/mulMul6sequential_13/lstm_26/while/lstm_cell_26/Sigmoid_1:y:0)sequential_13_lstm_26_while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2.
,sequential_13/lstm_26/while/lstm_cell_26/mulÒ
-sequential_13/lstm_26/while/lstm_cell_26/ReluRelu7sequential_13/lstm_26/while/lstm_cell_26/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2/
-sequential_13/lstm_26/while/lstm_cell_26/Relu
.sequential_13/lstm_26/while/lstm_cell_26/mul_1Mul4sequential_13/lstm_26/while/lstm_cell_26/Sigmoid:y:0;sequential_13/lstm_26/while/lstm_cell_26/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ20
.sequential_13/lstm_26/while/lstm_cell_26/mul_1
.sequential_13/lstm_26/while/lstm_cell_26/add_1AddV20sequential_13/lstm_26/while/lstm_cell_26/mul:z:02sequential_13/lstm_26/while/lstm_cell_26/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ20
.sequential_13/lstm_26/while/lstm_cell_26/add_1ß
2sequential_13/lstm_26/while/lstm_cell_26/Sigmoid_2Sigmoid7sequential_13/lstm_26/while/lstm_cell_26/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ24
2sequential_13/lstm_26/while/lstm_cell_26/Sigmoid_2Ñ
/sequential_13/lstm_26/while/lstm_cell_26/Relu_1Relu2sequential_13/lstm_26/while/lstm_cell_26/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ21
/sequential_13/lstm_26/while/lstm_cell_26/Relu_1
.sequential_13/lstm_26/while/lstm_cell_26/mul_2Mul6sequential_13/lstm_26/while/lstm_cell_26/Sigmoid_2:y:0=sequential_13/lstm_26/while/lstm_cell_26/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ20
.sequential_13/lstm_26/while/lstm_cell_26/mul_2Î
@sequential_13/lstm_26/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem)sequential_13_lstm_26_while_placeholder_1'sequential_13_lstm_26_while_placeholder2sequential_13/lstm_26/while/lstm_cell_26/mul_2:z:0*
_output_shapes
: *
element_dtype02B
@sequential_13/lstm_26/while/TensorArrayV2Write/TensorListSetItem
!sequential_13/lstm_26/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2#
!sequential_13/lstm_26/while/add/yÁ
sequential_13/lstm_26/while/addAddV2'sequential_13_lstm_26_while_placeholder*sequential_13/lstm_26/while/add/y:output:0*
T0*
_output_shapes
: 2!
sequential_13/lstm_26/while/add
#sequential_13/lstm_26/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2%
#sequential_13/lstm_26/while/add_1/yä
!sequential_13/lstm_26/while/add_1AddV2Dsequential_13_lstm_26_while_sequential_13_lstm_26_while_loop_counter,sequential_13/lstm_26/while/add_1/y:output:0*
T0*
_output_shapes
: 2#
!sequential_13/lstm_26/while/add_1Ã
$sequential_13/lstm_26/while/IdentityIdentity%sequential_13/lstm_26/while/add_1:z:0!^sequential_13/lstm_26/while/NoOp*
T0*
_output_shapes
: 2&
$sequential_13/lstm_26/while/Identityì
&sequential_13/lstm_26/while/Identity_1IdentityJsequential_13_lstm_26_while_sequential_13_lstm_26_while_maximum_iterations!^sequential_13/lstm_26/while/NoOp*
T0*
_output_shapes
: 2(
&sequential_13/lstm_26/while/Identity_1Å
&sequential_13/lstm_26/while/Identity_2Identity#sequential_13/lstm_26/while/add:z:0!^sequential_13/lstm_26/while/NoOp*
T0*
_output_shapes
: 2(
&sequential_13/lstm_26/while/Identity_2ò
&sequential_13/lstm_26/while/Identity_3IdentityPsequential_13/lstm_26/while/TensorArrayV2Write/TensorListSetItem:output_handle:0!^sequential_13/lstm_26/while/NoOp*
T0*
_output_shapes
: 2(
&sequential_13/lstm_26/while/Identity_3æ
&sequential_13/lstm_26/while/Identity_4Identity2sequential_13/lstm_26/while/lstm_cell_26/mul_2:z:0!^sequential_13/lstm_26/while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2(
&sequential_13/lstm_26/while/Identity_4æ
&sequential_13/lstm_26/while/Identity_5Identity2sequential_13/lstm_26/while/lstm_cell_26/add_1:z:0!^sequential_13/lstm_26/while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2(
&sequential_13/lstm_26/while/Identity_5Ì
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
&sequential_13_lstm_26_while_identity_5/sequential_13/lstm_26/while/Identity_5:output:0"
Hsequential_13_lstm_26_while_lstm_cell_26_biasadd_readvariableop_resourceJsequential_13_lstm_26_while_lstm_cell_26_biasadd_readvariableop_resource_0"
Isequential_13_lstm_26_while_lstm_cell_26_matmul_1_readvariableop_resourceKsequential_13_lstm_26_while_lstm_cell_26_matmul_1_readvariableop_resource_0"
Gsequential_13_lstm_26_while_lstm_cell_26_matmul_readvariableop_resourceIsequential_13_lstm_26_while_lstm_cell_26_matmul_readvariableop_resource_0"
Asequential_13_lstm_26_while_sequential_13_lstm_26_strided_slice_1Csequential_13_lstm_26_while_sequential_13_lstm_26_strided_slice_1_0"
}sequential_13_lstm_26_while_tensorarrayv2read_tensorlistgetitem_sequential_13_lstm_26_tensorarrayunstack_tensorlistfromtensorsequential_13_lstm_26_while_tensorarrayv2read_tensorlistgetitem_sequential_13_lstm_26_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿÚ:ÿÿÿÿÿÿÿÿÿÚ: : : : : 2
?sequential_13/lstm_26/while/lstm_cell_26/BiasAdd/ReadVariableOp?sequential_13/lstm_26/while/lstm_cell_26/BiasAdd/ReadVariableOp2
>sequential_13/lstm_26/while/lstm_cell_26/MatMul/ReadVariableOp>sequential_13/lstm_26/while/lstm_cell_26/MatMul/ReadVariableOp2
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
:ÿÿÿÿÿÿÿÿÿÚ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ:

_output_shapes
: :

_output_shapes
: 
ã
Í
while_cond_39972370
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_39972370___redundant_placeholder06
2while_while_cond_39972370___redundant_placeholder16
2while_while_cond_39972370___redundant_placeholder26
2while_while_cond_39972370___redundant_placeholder3
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
B: : : : :ÿÿÿÿÿÿÿÿÿÚ:ÿÿÿÿÿÿÿÿÿÚ: ::::: 

_output_shapes
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
:ÿÿÿÿÿÿÿÿÿÚ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ:

_output_shapes
: :

_output_shapes
:
ã
Í
while_cond_39972580
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_39972580___redundant_placeholder06
2while_while_cond_39972580___redundant_placeholder16
2while_while_cond_39972580___redundant_placeholder26
2while_while_cond_39972580___redundant_placeholder3
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
B: : : : :ÿÿÿÿÿÿÿÿÿÚ:ÿÿÿÿÿÿÿÿÿÚ: ::::: 

_output_shapes
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
:ÿÿÿÿÿÿÿÿÿÚ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ:

_output_shapes
: :

_output_shapes
:
ã
Í
while_cond_39973779
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_39973779___redundant_placeholder06
2while_while_cond_39973779___redundant_placeholder16
2while_while_cond_39973779___redundant_placeholder26
2while_while_cond_39973779___redundant_placeholder3
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
B: : : : :ÿÿÿÿÿÿÿÿÿæ:ÿÿÿÿÿÿÿÿÿæ: ::::: 

_output_shapes
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
:ÿÿÿÿÿÿÿÿÿæ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ:

_output_shapes
: :

_output_shapes
:
Ô

í
lstm_26_while_cond_39974934,
(lstm_26_while_lstm_26_while_loop_counter2
.lstm_26_while_lstm_26_while_maximum_iterations
lstm_26_while_placeholder
lstm_26_while_placeholder_1
lstm_26_while_placeholder_2
lstm_26_while_placeholder_3.
*lstm_26_while_less_lstm_26_strided_slice_1F
Blstm_26_while_lstm_26_while_cond_39974934___redundant_placeholder0F
Blstm_26_while_lstm_26_while_cond_39974934___redundant_placeholder1F
Blstm_26_while_lstm_26_while_cond_39974934___redundant_placeholder2F
Blstm_26_while_lstm_26_while_cond_39974934___redundant_placeholder3
lstm_26_while_identity

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
B: : : : :ÿÿÿÿÿÿÿÿÿÚ:ÿÿÿÿÿÿÿÿÿÚ: ::::: 

_output_shapes
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
:ÿÿÿÿÿÿÿÿÿÚ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ:

_output_shapes
: :

_output_shapes
:
åJ
Ô

lstm_26_while_body_39974608,
(lstm_26_while_lstm_26_while_loop_counter2
.lstm_26_while_lstm_26_while_maximum_iterations
lstm_26_while_placeholder
lstm_26_while_placeholder_1
lstm_26_while_placeholder_2
lstm_26_while_placeholder_3+
'lstm_26_while_lstm_26_strided_slice_1_0g
clstm_26_while_tensorarrayv2read_tensorlistgetitem_lstm_26_tensorarrayunstack_tensorlistfromtensor_0N
;lstm_26_while_lstm_cell_26_matmul_readvariableop_resource_0:	]è
Q
=lstm_26_while_lstm_cell_26_matmul_1_readvariableop_resource_0:
Úè
K
<lstm_26_while_lstm_cell_26_biasadd_readvariableop_resource_0:	è

lstm_26_while_identity
lstm_26_while_identity_1
lstm_26_while_identity_2
lstm_26_while_identity_3
lstm_26_while_identity_4
lstm_26_while_identity_5)
%lstm_26_while_lstm_26_strided_slice_1e
alstm_26_while_tensorarrayv2read_tensorlistgetitem_lstm_26_tensorarrayunstack_tensorlistfromtensorL
9lstm_26_while_lstm_cell_26_matmul_readvariableop_resource:	]è
O
;lstm_26_while_lstm_cell_26_matmul_1_readvariableop_resource:
Úè
I
:lstm_26_while_lstm_cell_26_biasadd_readvariableop_resource:	è
¢1lstm_26/while/lstm_cell_26/BiasAdd/ReadVariableOp¢0lstm_26/while/lstm_cell_26/MatMul/ReadVariableOp¢2lstm_26/while/lstm_cell_26/MatMul_1/ReadVariableOpÓ
?lstm_26/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ]   2A
?lstm_26/while/TensorArrayV2Read/TensorListGetItem/element_shape
1lstm_26/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemclstm_26_while_tensorarrayv2read_tensorlistgetitem_lstm_26_tensorarrayunstack_tensorlistfromtensor_0lstm_26_while_placeholderHlstm_26/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]*
element_dtype023
1lstm_26/while/TensorArrayV2Read/TensorListGetItemá
0lstm_26/while/lstm_cell_26/MatMul/ReadVariableOpReadVariableOp;lstm_26_while_lstm_cell_26_matmul_readvariableop_resource_0*
_output_shapes
:	]è
*
dtype022
0lstm_26/while/lstm_cell_26/MatMul/ReadVariableOp÷
!lstm_26/while/lstm_cell_26/MatMulMatMul8lstm_26/while/TensorArrayV2Read/TensorListGetItem:item:08lstm_26/while/lstm_cell_26/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè
2#
!lstm_26/while/lstm_cell_26/MatMulè
2lstm_26/while/lstm_cell_26/MatMul_1/ReadVariableOpReadVariableOp=lstm_26_while_lstm_cell_26_matmul_1_readvariableop_resource_0* 
_output_shapes
:
Úè
*
dtype024
2lstm_26/while/lstm_cell_26/MatMul_1/ReadVariableOpà
#lstm_26/while/lstm_cell_26/MatMul_1MatMullstm_26_while_placeholder_2:lstm_26/while/lstm_cell_26/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè
2%
#lstm_26/while/lstm_cell_26/MatMul_1Ø
lstm_26/while/lstm_cell_26/addAddV2+lstm_26/while/lstm_cell_26/MatMul:product:0-lstm_26/while/lstm_cell_26/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè
2 
lstm_26/while/lstm_cell_26/addà
1lstm_26/while/lstm_cell_26/BiasAdd/ReadVariableOpReadVariableOp<lstm_26_while_lstm_cell_26_biasadd_readvariableop_resource_0*
_output_shapes	
:è
*
dtype023
1lstm_26/while/lstm_cell_26/BiasAdd/ReadVariableOpå
"lstm_26/while/lstm_cell_26/BiasAddBiasAdd"lstm_26/while/lstm_cell_26/add:z:09lstm_26/while/lstm_cell_26/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè
2$
"lstm_26/while/lstm_cell_26/BiasAdd
*lstm_26/while/lstm_cell_26/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2,
*lstm_26/while/lstm_cell_26/split/split_dim¯
 lstm_26/while/lstm_cell_26/splitSplit3lstm_26/while/lstm_cell_26/split/split_dim:output:0+lstm_26/while/lstm_cell_26/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿÚ:ÿÿÿÿÿÿÿÿÿÚ:ÿÿÿÿÿÿÿÿÿÚ:ÿÿÿÿÿÿÿÿÿÚ*
	num_split2"
 lstm_26/while/lstm_cell_26/split±
"lstm_26/while/lstm_cell_26/SigmoidSigmoid)lstm_26/while/lstm_cell_26/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2$
"lstm_26/while/lstm_cell_26/Sigmoidµ
$lstm_26/while/lstm_cell_26/Sigmoid_1Sigmoid)lstm_26/while/lstm_cell_26/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2&
$lstm_26/while/lstm_cell_26/Sigmoid_1Á
lstm_26/while/lstm_cell_26/mulMul(lstm_26/while/lstm_cell_26/Sigmoid_1:y:0lstm_26_while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2 
lstm_26/while/lstm_cell_26/mul¨
lstm_26/while/lstm_cell_26/ReluRelu)lstm_26/while/lstm_cell_26/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2!
lstm_26/while/lstm_cell_26/ReluÕ
 lstm_26/while/lstm_cell_26/mul_1Mul&lstm_26/while/lstm_cell_26/Sigmoid:y:0-lstm_26/while/lstm_cell_26/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2"
 lstm_26/while/lstm_cell_26/mul_1Ê
 lstm_26/while/lstm_cell_26/add_1AddV2"lstm_26/while/lstm_cell_26/mul:z:0$lstm_26/while/lstm_cell_26/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2"
 lstm_26/while/lstm_cell_26/add_1µ
$lstm_26/while/lstm_cell_26/Sigmoid_2Sigmoid)lstm_26/while/lstm_cell_26/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2&
$lstm_26/while/lstm_cell_26/Sigmoid_2§
!lstm_26/while/lstm_cell_26/Relu_1Relu$lstm_26/while/lstm_cell_26/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2#
!lstm_26/while/lstm_cell_26/Relu_1Ù
 lstm_26/while/lstm_cell_26/mul_2Mul(lstm_26/while/lstm_cell_26/Sigmoid_2:y:0/lstm_26/while/lstm_cell_26/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2"
 lstm_26/while/lstm_cell_26/mul_2
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
lstm_26/while/add/y
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
lstm_26/while/add_1/y
lstm_26/while/add_1AddV2(lstm_26_while_lstm_26_while_loop_counterlstm_26/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lstm_26/while/add_1
lstm_26/while/IdentityIdentitylstm_26/while/add_1:z:0^lstm_26/while/NoOp*
T0*
_output_shapes
: 2
lstm_26/while/Identity¦
lstm_26/while/Identity_1Identity.lstm_26_while_lstm_26_while_maximum_iterations^lstm_26/while/NoOp*
T0*
_output_shapes
: 2
lstm_26/while/Identity_1
lstm_26/while/Identity_2Identitylstm_26/while/add:z:0^lstm_26/while/NoOp*
T0*
_output_shapes
: 2
lstm_26/while/Identity_2º
lstm_26/while/Identity_3IdentityBlstm_26/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_26/while/NoOp*
T0*
_output_shapes
: 2
lstm_26/while/Identity_3®
lstm_26/while/Identity_4Identity$lstm_26/while/lstm_cell_26/mul_2:z:0^lstm_26/while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
lstm_26/while/Identity_4®
lstm_26/while/Identity_5Identity$lstm_26/while/lstm_cell_26/add_1:z:0^lstm_26/while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
lstm_26/while/Identity_5
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
9lstm_26_while_lstm_cell_26_matmul_readvariableop_resource;lstm_26_while_lstm_cell_26_matmul_readvariableop_resource_0"È
alstm_26_while_tensorarrayv2read_tensorlistgetitem_lstm_26_tensorarrayunstack_tensorlistfromtensorclstm_26_while_tensorarrayv2read_tensorlistgetitem_lstm_26_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿÚ:ÿÿÿÿÿÿÿÿÿÚ: : : : : 2f
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
:ÿÿÿÿÿÿÿÿÿÚ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ:

_output_shapes
: :

_output_shapes
: 

f
H__inference_dropout_27_layer_call_and_return_conditional_losses_39976547

inputs

identity_1_
IdentityIdentityinputs*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2

Identityn

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿæ:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ
 
_user_specified_nameinputs
Õ!
þ
F__inference_dense_13_layer_call_and_return_conditional_losses_39973910

inputs4
!tensordot_readvariableop_resource:	æ-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢Tensordot/ReadVariableOp
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes
:	æ*
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
:ÿÿÿÿÿÿÿÿÿæ2
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
:ÿÿÿÿÿÿÿÿÿæ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ
 
_user_specified_nameinputs
µ
»
K__inference_sequential_13_layer_call_and_return_conditional_losses_39974386

inputs#
lstm_26_39974364:	]è
$
lstm_26_39974366:
Úè

lstm_26_39974368:	è
$
lstm_27_39974372:
Ú$
lstm_27_39974374:
æ
lstm_27_39974376:	$
dense_13_39974380:	æ
dense_13_39974382:
identity¢ dense_13/StatefulPartitionedCall¢"dropout_26/StatefulPartitionedCall¢"dropout_27/StatefulPartitionedCall¢lstm_26/StatefulPartitionedCall¢lstm_27/StatefulPartitionedCall®
lstm_26/StatefulPartitionedCallStatefulPartitionedCallinputslstm_26_39974364lstm_26_39974366lstm_26_39974368*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_lstm_26_layer_call_and_return_conditional_losses_399743292!
lstm_26/StatefulPartitionedCall
"dropout_26/StatefulPartitionedCallStatefulPartitionedCall(lstm_26/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_dropout_26_layer_call_and_return_conditional_losses_399741622$
"dropout_26/StatefulPartitionedCallÓ
lstm_27/StatefulPartitionedCallStatefulPartitionedCall+dropout_26/StatefulPartitionedCall:output:0lstm_27_39974372lstm_27_39974374lstm_27_39974376*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_lstm_27_layer_call_and_return_conditional_losses_399741332!
lstm_27/StatefulPartitionedCallÀ
"dropout_27/StatefulPartitionedCallStatefulPartitionedCall(lstm_27/StatefulPartitionedCall:output:0#^dropout_26/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_dropout_27_layer_call_and_return_conditional_losses_399739662$
"dropout_27/StatefulPartitionedCallÃ
 dense_13/StatefulPartitionedCallStatefulPartitionedCall+dropout_27/StatefulPartitionedCall:output:0dense_13_39974380dense_13_39974382*
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
F__inference_dense_13_layer_call_and_return_conditional_losses_399739102"
 dense_13/StatefulPartitionedCall
IdentityIdentity)dense_13/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityÿ
NoOpNoOp!^dense_13/StatefulPartitionedCall#^dropout_26/StatefulPartitionedCall#^dropout_27/StatefulPartitionedCall ^lstm_26/StatefulPartitionedCall ^lstm_27/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ]: : : : : : : : 2D
 dense_13/StatefulPartitionedCall dense_13/StatefulPartitionedCall2H
"dropout_26/StatefulPartitionedCall"dropout_26/StatefulPartitionedCall2H
"dropout_27/StatefulPartitionedCall"dropout_27/StatefulPartitionedCall2B
lstm_26/StatefulPartitionedCalllstm_26/StatefulPartitionedCall2B
lstm_27/StatefulPartitionedCalllstm_27/StatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
 
_user_specified_nameinputs


Í
0__inference_sequential_13_layer_call_fn_39974520

inputs
unknown:	]è

	unknown_0:
Úè

	unknown_1:	è

	unknown_2:
Ú
	unknown_3:
æ
	unknown_4:	
	unknown_5:	æ
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
K__inference_sequential_13_layer_call_and_return_conditional_losses_399739172
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
Ô

í
lstm_27_while_cond_39974755,
(lstm_27_while_lstm_27_while_loop_counter2
.lstm_27_while_lstm_27_while_maximum_iterations
lstm_27_while_placeholder
lstm_27_while_placeholder_1
lstm_27_while_placeholder_2
lstm_27_while_placeholder_3.
*lstm_27_while_less_lstm_27_strided_slice_1F
Blstm_27_while_lstm_27_while_cond_39974755___redundant_placeholder0F
Blstm_27_while_lstm_27_while_cond_39974755___redundant_placeholder1F
Blstm_27_while_lstm_27_while_cond_39974755___redundant_placeholder2F
Blstm_27_while_lstm_27_while_cond_39974755___redundant_placeholder3
lstm_27_while_identity

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
B: : : : :ÿÿÿÿÿÿÿÿÿæ:ÿÿÿÿÿÿÿÿÿæ: ::::: 

_output_shapes
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
:ÿÿÿÿÿÿÿÿÿæ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ:

_output_shapes
: :

_output_shapes
:
ã
Í
while_cond_39975319
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_39975319___redundant_placeholder06
2while_while_cond_39975319___redundant_placeholder16
2while_while_cond_39975319___redundant_placeholder26
2while_while_cond_39975319___redundant_placeholder3
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
B: : : : :ÿÿÿÿÿÿÿÿÿÚ:ÿÿÿÿÿÿÿÿÿÚ: ::::: 

_output_shapes
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
:ÿÿÿÿÿÿÿÿÿÚ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ:
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
lstm_26_input:
serving_default_lstm_26_input:0ÿÿÿÿÿÿÿÿÿ]@
dense_134
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
": 	æ2dense_13/kernel
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
.:,	]è
2lstm_26/lstm_cell_26/kernel
9:7
Úè
2%lstm_26/lstm_cell_26/recurrent_kernel
(:&è
2lstm_26/lstm_cell_26/bias
/:-
Ú2lstm_27/lstm_cell_27/kernel
9:7
æ2%lstm_27/lstm_cell_27/recurrent_kernel
(:&2lstm_27/lstm_cell_27/bias
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
0__inference_sequential_13_layer_call_fn_39973936
0__inference_sequential_13_layer_call_fn_39974520
0__inference_sequential_13_layer_call_fn_39974541
0__inference_sequential_13_layer_call_fn_39974426À
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
K__inference_sequential_13_layer_call_and_return_conditional_losses_39974868
K__inference_sequential_13_layer_call_and_return_conditional_losses_39975209
K__inference_sequential_13_layer_call_and_return_conditional_losses_39974451
K__inference_sequential_13_layer_call_and_return_conditional_losses_39974476À
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
#__inference__wrapped_model_39972282lstm_26_input"
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
*__inference_lstm_26_layer_call_fn_39975220
*__inference_lstm_26_layer_call_fn_39975231
*__inference_lstm_26_layer_call_fn_39975242
*__inference_lstm_26_layer_call_fn_39975253Õ
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
E__inference_lstm_26_layer_call_and_return_conditional_losses_39975404
E__inference_lstm_26_layer_call_and_return_conditional_losses_39975555
E__inference_lstm_26_layer_call_and_return_conditional_losses_39975706
E__inference_lstm_26_layer_call_and_return_conditional_losses_39975857Õ
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
-__inference_dropout_26_layer_call_fn_39975862
-__inference_dropout_26_layer_call_fn_39975867´
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
H__inference_dropout_26_layer_call_and_return_conditional_losses_39975872
H__inference_dropout_26_layer_call_and_return_conditional_losses_39975884´
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
*__inference_lstm_27_layer_call_fn_39975895
*__inference_lstm_27_layer_call_fn_39975906
*__inference_lstm_27_layer_call_fn_39975917
*__inference_lstm_27_layer_call_fn_39975928Õ
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
E__inference_lstm_27_layer_call_and_return_conditional_losses_39976079
E__inference_lstm_27_layer_call_and_return_conditional_losses_39976230
E__inference_lstm_27_layer_call_and_return_conditional_losses_39976381
E__inference_lstm_27_layer_call_and_return_conditional_losses_39976532Õ
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
-__inference_dropout_27_layer_call_fn_39976537
-__inference_dropout_27_layer_call_fn_39976542´
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
H__inference_dropout_27_layer_call_and_return_conditional_losses_39976547
H__inference_dropout_27_layer_call_and_return_conditional_losses_39976559´
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
+__inference_dense_13_layer_call_fn_39976568¢
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
F__inference_dense_13_layer_call_and_return_conditional_losses_39976599¢
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
&__inference_signature_wrapper_39974499lstm_26_input"
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
/__inference_lstm_cell_26_layer_call_fn_39976616
/__inference_lstm_cell_26_layer_call_fn_39976633¾
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
J__inference_lstm_cell_26_layer_call_and_return_conditional_losses_39976665
J__inference_lstm_cell_26_layer_call_and_return_conditional_losses_39976697¾
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
/__inference_lstm_cell_27_layer_call_fn_39976714
/__inference_lstm_cell_27_layer_call_fn_39976731¾
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
J__inference_lstm_cell_27_layer_call_and_return_conditional_losses_39976763
J__inference_lstm_cell_27_layer_call_and_return_conditional_losses_39976795¾
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
#__inference__wrapped_model_39972282&'()*+ !:¢7
0¢-
+(
lstm_26_inputÿÿÿÿÿÿÿÿÿ]
ª "7ª4
2
dense_13&#
dense_13ÿÿÿÿÿÿÿÿÿ¯
F__inference_dense_13_layer_call_and_return_conditional_losses_39976599e !4¢1
*¢'
%"
inputsÿÿÿÿÿÿÿÿÿæ
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ
 
+__inference_dense_13_layer_call_fn_39976568X !4¢1
*¢'
%"
inputsÿÿÿÿÿÿÿÿÿæ
ª "ÿÿÿÿÿÿÿÿÿ²
H__inference_dropout_26_layer_call_and_return_conditional_losses_39975872f8¢5
.¢+
%"
inputsÿÿÿÿÿÿÿÿÿÚ
p 
ª "*¢'
 
0ÿÿÿÿÿÿÿÿÿÚ
 ²
H__inference_dropout_26_layer_call_and_return_conditional_losses_39975884f8¢5
.¢+
%"
inputsÿÿÿÿÿÿÿÿÿÚ
p
ª "*¢'
 
0ÿÿÿÿÿÿÿÿÿÚ
 
-__inference_dropout_26_layer_call_fn_39975862Y8¢5
.¢+
%"
inputsÿÿÿÿÿÿÿÿÿÚ
p 
ª "ÿÿÿÿÿÿÿÿÿÚ
-__inference_dropout_26_layer_call_fn_39975867Y8¢5
.¢+
%"
inputsÿÿÿÿÿÿÿÿÿÚ
p
ª "ÿÿÿÿÿÿÿÿÿÚ²
H__inference_dropout_27_layer_call_and_return_conditional_losses_39976547f8¢5
.¢+
%"
inputsÿÿÿÿÿÿÿÿÿæ
p 
ª "*¢'
 
0ÿÿÿÿÿÿÿÿÿæ
 ²
H__inference_dropout_27_layer_call_and_return_conditional_losses_39976559f8¢5
.¢+
%"
inputsÿÿÿÿÿÿÿÿÿæ
p
ª "*¢'
 
0ÿÿÿÿÿÿÿÿÿæ
 
-__inference_dropout_27_layer_call_fn_39976537Y8¢5
.¢+
%"
inputsÿÿÿÿÿÿÿÿÿæ
p 
ª "ÿÿÿÿÿÿÿÿÿæ
-__inference_dropout_27_layer_call_fn_39976542Y8¢5
.¢+
%"
inputsÿÿÿÿÿÿÿÿÿæ
p
ª "ÿÿÿÿÿÿÿÿÿæÕ
E__inference_lstm_26_layer_call_and_return_conditional_losses_39975404&'(O¢L
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
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÚ
 Õ
E__inference_lstm_26_layer_call_and_return_conditional_losses_39975555&'(O¢L
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
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÚ
 »
E__inference_lstm_26_layer_call_and_return_conditional_losses_39975706r&'(?¢<
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
0ÿÿÿÿÿÿÿÿÿÚ
 »
E__inference_lstm_26_layer_call_and_return_conditional_losses_39975857r&'(?¢<
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
0ÿÿÿÿÿÿÿÿÿÚ
 ¬
*__inference_lstm_26_layer_call_fn_39975220~&'(O¢L
E¢B
41
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ]

 
p 

 
ª "&#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÚ¬
*__inference_lstm_26_layer_call_fn_39975231~&'(O¢L
E¢B
41
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ]

 
p

 
ª "&#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÚ
*__inference_lstm_26_layer_call_fn_39975242e&'(?¢<
5¢2
$!
inputsÿÿÿÿÿÿÿÿÿ]

 
p 

 
ª "ÿÿÿÿÿÿÿÿÿÚ
*__inference_lstm_26_layer_call_fn_39975253e&'(?¢<
5¢2
$!
inputsÿÿÿÿÿÿÿÿÿ]

 
p

 
ª "ÿÿÿÿÿÿÿÿÿÚÖ
E__inference_lstm_27_layer_call_and_return_conditional_losses_39976079)*+P¢M
F¢C
52
0-
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÚ

 
p 

 
ª "3¢0
)&
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿæ
 Ö
E__inference_lstm_27_layer_call_and_return_conditional_losses_39976230)*+P¢M
F¢C
52
0-
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÚ

 
p

 
ª "3¢0
)&
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿæ
 ¼
E__inference_lstm_27_layer_call_and_return_conditional_losses_39976381s)*+@¢=
6¢3
%"
inputsÿÿÿÿÿÿÿÿÿÚ

 
p 

 
ª "*¢'
 
0ÿÿÿÿÿÿÿÿÿæ
 ¼
E__inference_lstm_27_layer_call_and_return_conditional_losses_39976532s)*+@¢=
6¢3
%"
inputsÿÿÿÿÿÿÿÿÿÚ

 
p

 
ª "*¢'
 
0ÿÿÿÿÿÿÿÿÿæ
 ­
*__inference_lstm_27_layer_call_fn_39975895)*+P¢M
F¢C
52
0-
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÚ

 
p 

 
ª "&#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿæ­
*__inference_lstm_27_layer_call_fn_39975906)*+P¢M
F¢C
52
0-
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÚ

 
p

 
ª "&#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿæ
*__inference_lstm_27_layer_call_fn_39975917f)*+@¢=
6¢3
%"
inputsÿÿÿÿÿÿÿÿÿÚ

 
p 

 
ª "ÿÿÿÿÿÿÿÿÿæ
*__inference_lstm_27_layer_call_fn_39975928f)*+@¢=
6¢3
%"
inputsÿÿÿÿÿÿÿÿÿÚ

 
p

 
ª "ÿÿÿÿÿÿÿÿÿæÑ
J__inference_lstm_cell_26_layer_call_and_return_conditional_losses_39976665&'(¢
x¢u
 
inputsÿÿÿÿÿÿÿÿÿ]
M¢J
# 
states/0ÿÿÿÿÿÿÿÿÿÚ
# 
states/1ÿÿÿÿÿÿÿÿÿÚ
p 
ª "v¢s
l¢i

0/0ÿÿÿÿÿÿÿÿÿÚ
GD
 
0/1/0ÿÿÿÿÿÿÿÿÿÚ
 
0/1/1ÿÿÿÿÿÿÿÿÿÚ
 Ñ
J__inference_lstm_cell_26_layer_call_and_return_conditional_losses_39976697&'(¢
x¢u
 
inputsÿÿÿÿÿÿÿÿÿ]
M¢J
# 
states/0ÿÿÿÿÿÿÿÿÿÚ
# 
states/1ÿÿÿÿÿÿÿÿÿÚ
p
ª "v¢s
l¢i

0/0ÿÿÿÿÿÿÿÿÿÚ
GD
 
0/1/0ÿÿÿÿÿÿÿÿÿÚ
 
0/1/1ÿÿÿÿÿÿÿÿÿÚ
 ¦
/__inference_lstm_cell_26_layer_call_fn_39976616ò&'(¢
x¢u
 
inputsÿÿÿÿÿÿÿÿÿ]
M¢J
# 
states/0ÿÿÿÿÿÿÿÿÿÚ
# 
states/1ÿÿÿÿÿÿÿÿÿÚ
p 
ª "f¢c

0ÿÿÿÿÿÿÿÿÿÚ
C@

1/0ÿÿÿÿÿÿÿÿÿÚ

1/1ÿÿÿÿÿÿÿÿÿÚ¦
/__inference_lstm_cell_26_layer_call_fn_39976633ò&'(¢
x¢u
 
inputsÿÿÿÿÿÿÿÿÿ]
M¢J
# 
states/0ÿÿÿÿÿÿÿÿÿÚ
# 
states/1ÿÿÿÿÿÿÿÿÿÚ
p
ª "f¢c

0ÿÿÿÿÿÿÿÿÿÚ
C@

1/0ÿÿÿÿÿÿÿÿÿÚ

1/1ÿÿÿÿÿÿÿÿÿÚÓ
J__inference_lstm_cell_27_layer_call_and_return_conditional_losses_39976763)*+¢
y¢v
!
inputsÿÿÿÿÿÿÿÿÿÚ
M¢J
# 
states/0ÿÿÿÿÿÿÿÿÿæ
# 
states/1ÿÿÿÿÿÿÿÿÿæ
p 
ª "v¢s
l¢i

0/0ÿÿÿÿÿÿÿÿÿæ
GD
 
0/1/0ÿÿÿÿÿÿÿÿÿæ
 
0/1/1ÿÿÿÿÿÿÿÿÿæ
 Ó
J__inference_lstm_cell_27_layer_call_and_return_conditional_losses_39976795)*+¢
y¢v
!
inputsÿÿÿÿÿÿÿÿÿÚ
M¢J
# 
states/0ÿÿÿÿÿÿÿÿÿæ
# 
states/1ÿÿÿÿÿÿÿÿÿæ
p
ª "v¢s
l¢i

0/0ÿÿÿÿÿÿÿÿÿæ
GD
 
0/1/0ÿÿÿÿÿÿÿÿÿæ
 
0/1/1ÿÿÿÿÿÿÿÿÿæ
 ¨
/__inference_lstm_cell_27_layer_call_fn_39976714ô)*+¢
y¢v
!
inputsÿÿÿÿÿÿÿÿÿÚ
M¢J
# 
states/0ÿÿÿÿÿÿÿÿÿæ
# 
states/1ÿÿÿÿÿÿÿÿÿæ
p 
ª "f¢c

0ÿÿÿÿÿÿÿÿÿæ
C@

1/0ÿÿÿÿÿÿÿÿÿæ

1/1ÿÿÿÿÿÿÿÿÿæ¨
/__inference_lstm_cell_27_layer_call_fn_39976731ô)*+¢
y¢v
!
inputsÿÿÿÿÿÿÿÿÿÚ
M¢J
# 
states/0ÿÿÿÿÿÿÿÿÿæ
# 
states/1ÿÿÿÿÿÿÿÿÿæ
p
ª "f¢c

0ÿÿÿÿÿÿÿÿÿæ
C@

1/0ÿÿÿÿÿÿÿÿÿæ

1/1ÿÿÿÿÿÿÿÿÿæÈ
K__inference_sequential_13_layer_call_and_return_conditional_losses_39974451y&'()*+ !B¢?
8¢5
+(
lstm_26_inputÿÿÿÿÿÿÿÿÿ]
p 

 
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ
 È
K__inference_sequential_13_layer_call_and_return_conditional_losses_39974476y&'()*+ !B¢?
8¢5
+(
lstm_26_inputÿÿÿÿÿÿÿÿÿ]
p

 
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ
 Á
K__inference_sequential_13_layer_call_and_return_conditional_losses_39974868r&'()*+ !;¢8
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
K__inference_sequential_13_layer_call_and_return_conditional_losses_39975209r&'()*+ !;¢8
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
0__inference_sequential_13_layer_call_fn_39973936l&'()*+ !B¢?
8¢5
+(
lstm_26_inputÿÿÿÿÿÿÿÿÿ]
p 

 
ª "ÿÿÿÿÿÿÿÿÿ 
0__inference_sequential_13_layer_call_fn_39974426l&'()*+ !B¢?
8¢5
+(
lstm_26_inputÿÿÿÿÿÿÿÿÿ]
p

 
ª "ÿÿÿÿÿÿÿÿÿ
0__inference_sequential_13_layer_call_fn_39974520e&'()*+ !;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿ]
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
0__inference_sequential_13_layer_call_fn_39974541e&'()*+ !;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿ]
p

 
ª "ÿÿÿÿÿÿÿÿÿ»
&__inference_signature_wrapper_39974499&'()*+ !K¢H
¢ 
Aª>
<
lstm_26_input+(
lstm_26_inputÿÿÿÿÿÿÿÿÿ]"7ª4
2
dense_13&#
dense_13ÿÿÿÿÿÿÿÿÿ