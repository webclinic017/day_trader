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
dense_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:S*
shared_namedense_5/kernel
q
"dense_5/kernel/Read/ReadVariableOpReadVariableOpdense_5/kernel*
_output_shapes

:S*
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

lstm_10/lstm_cell_10/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	]è	*,
shared_namelstm_10/lstm_cell_10/kernel

/lstm_10/lstm_cell_10/kernel/Read/ReadVariableOpReadVariableOplstm_10/lstm_cell_10/kernel*
_output_shapes
:	]è	*
dtype0
¨
%lstm_10/lstm_cell_10/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ºè	*6
shared_name'%lstm_10/lstm_cell_10/recurrent_kernel
¡
9lstm_10/lstm_cell_10/recurrent_kernel/Read/ReadVariableOpReadVariableOp%lstm_10/lstm_cell_10/recurrent_kernel* 
_output_shapes
:
ºè	*
dtype0

lstm_10/lstm_cell_10/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:è	**
shared_namelstm_10/lstm_cell_10/bias

-lstm_10/lstm_cell_10/bias/Read/ReadVariableOpReadVariableOplstm_10/lstm_cell_10/bias*
_output_shapes	
:è	*
dtype0

lstm_11/lstm_cell_11/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ºÌ*,
shared_namelstm_11/lstm_cell_11/kernel

/lstm_11/lstm_cell_11/kernel/Read/ReadVariableOpReadVariableOplstm_11/lstm_cell_11/kernel* 
_output_shapes
:
ºÌ*
dtype0
§
%lstm_11/lstm_cell_11/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	SÌ*6
shared_name'%lstm_11/lstm_cell_11/recurrent_kernel
 
9lstm_11/lstm_cell_11/recurrent_kernel/Read/ReadVariableOpReadVariableOp%lstm_11/lstm_cell_11/recurrent_kernel*
_output_shapes
:	SÌ*
dtype0

lstm_11/lstm_cell_11/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:Ì**
shared_namelstm_11/lstm_cell_11/bias

-lstm_11/lstm_cell_11/bias/Read/ReadVariableOpReadVariableOplstm_11/lstm_cell_11/bias*
_output_shapes	
:Ì*
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
serving_default_lstm_10_inputPlaceholder*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]*
dtype0* 
shape:ÿÿÿÿÿÿÿÿÿ]
ª
StatefulPartitionedCallStatefulPartitionedCallserving_default_lstm_10_inputlstm_10/lstm_cell_10/kernel%lstm_10/lstm_cell_10/recurrent_kernellstm_10/lstm_cell_10/biaslstm_11/lstm_cell_11/kernel%lstm_11/lstm_cell_11/recurrent_kernellstm_11/lstm_cell_11/biasdense_5/kerneldense_5/bias*
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
&__inference_signature_wrapper_19556331
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 

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
GPU 2J 8 **
f%R#
!__inference__traced_save_19558686
 
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
GPU 2J 8 *-
f(R&
$__inference__traced_restore_19558732Èì#
¬]
õ
(sequential_5_lstm_10_while_body_19553854F
Bsequential_5_lstm_10_while_sequential_5_lstm_10_while_loop_counterL
Hsequential_5_lstm_10_while_sequential_5_lstm_10_while_maximum_iterations*
&sequential_5_lstm_10_while_placeholder,
(sequential_5_lstm_10_while_placeholder_1,
(sequential_5_lstm_10_while_placeholder_2,
(sequential_5_lstm_10_while_placeholder_3E
Asequential_5_lstm_10_while_sequential_5_lstm_10_strided_slice_1_0
}sequential_5_lstm_10_while_tensorarrayv2read_tensorlistgetitem_sequential_5_lstm_10_tensorarrayunstack_tensorlistfromtensor_0[
Hsequential_5_lstm_10_while_lstm_cell_10_matmul_readvariableop_resource_0:	]è	^
Jsequential_5_lstm_10_while_lstm_cell_10_matmul_1_readvariableop_resource_0:
ºè	X
Isequential_5_lstm_10_while_lstm_cell_10_biasadd_readvariableop_resource_0:	è	'
#sequential_5_lstm_10_while_identity)
%sequential_5_lstm_10_while_identity_1)
%sequential_5_lstm_10_while_identity_2)
%sequential_5_lstm_10_while_identity_3)
%sequential_5_lstm_10_while_identity_4)
%sequential_5_lstm_10_while_identity_5C
?sequential_5_lstm_10_while_sequential_5_lstm_10_strided_slice_1
{sequential_5_lstm_10_while_tensorarrayv2read_tensorlistgetitem_sequential_5_lstm_10_tensorarrayunstack_tensorlistfromtensorY
Fsequential_5_lstm_10_while_lstm_cell_10_matmul_readvariableop_resource:	]è	\
Hsequential_5_lstm_10_while_lstm_cell_10_matmul_1_readvariableop_resource:
ºè	V
Gsequential_5_lstm_10_while_lstm_cell_10_biasadd_readvariableop_resource:	è	¢>sequential_5/lstm_10/while/lstm_cell_10/BiasAdd/ReadVariableOp¢=sequential_5/lstm_10/while/lstm_cell_10/MatMul/ReadVariableOp¢?sequential_5/lstm_10/while/lstm_cell_10/MatMul_1/ReadVariableOpí
Lsequential_5/lstm_10/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ]   2N
Lsequential_5/lstm_10/while/TensorArrayV2Read/TensorListGetItem/element_shapeÑ
>sequential_5/lstm_10/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem}sequential_5_lstm_10_while_tensorarrayv2read_tensorlistgetitem_sequential_5_lstm_10_tensorarrayunstack_tensorlistfromtensor_0&sequential_5_lstm_10_while_placeholderUsequential_5/lstm_10/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]*
element_dtype02@
>sequential_5/lstm_10/while/TensorArrayV2Read/TensorListGetItem
=sequential_5/lstm_10/while/lstm_cell_10/MatMul/ReadVariableOpReadVariableOpHsequential_5_lstm_10_while_lstm_cell_10_matmul_readvariableop_resource_0*
_output_shapes
:	]è	*
dtype02?
=sequential_5/lstm_10/while/lstm_cell_10/MatMul/ReadVariableOp«
.sequential_5/lstm_10/while/lstm_cell_10/MatMulMatMulEsequential_5/lstm_10/while/TensorArrayV2Read/TensorListGetItem:item:0Esequential_5/lstm_10/while/lstm_cell_10/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè	20
.sequential_5/lstm_10/while/lstm_cell_10/MatMul
?sequential_5/lstm_10/while/lstm_cell_10/MatMul_1/ReadVariableOpReadVariableOpJsequential_5_lstm_10_while_lstm_cell_10_matmul_1_readvariableop_resource_0* 
_output_shapes
:
ºè	*
dtype02A
?sequential_5/lstm_10/while/lstm_cell_10/MatMul_1/ReadVariableOp
0sequential_5/lstm_10/while/lstm_cell_10/MatMul_1MatMul(sequential_5_lstm_10_while_placeholder_2Gsequential_5/lstm_10/while/lstm_cell_10/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè	22
0sequential_5/lstm_10/while/lstm_cell_10/MatMul_1
+sequential_5/lstm_10/while/lstm_cell_10/addAddV28sequential_5/lstm_10/while/lstm_cell_10/MatMul:product:0:sequential_5/lstm_10/while/lstm_cell_10/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè	2-
+sequential_5/lstm_10/while/lstm_cell_10/add
>sequential_5/lstm_10/while/lstm_cell_10/BiasAdd/ReadVariableOpReadVariableOpIsequential_5_lstm_10_while_lstm_cell_10_biasadd_readvariableop_resource_0*
_output_shapes	
:è	*
dtype02@
>sequential_5/lstm_10/while/lstm_cell_10/BiasAdd/ReadVariableOp
/sequential_5/lstm_10/while/lstm_cell_10/BiasAddBiasAdd/sequential_5/lstm_10/while/lstm_cell_10/add:z:0Fsequential_5/lstm_10/while/lstm_cell_10/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè	21
/sequential_5/lstm_10/while/lstm_cell_10/BiasAdd´
7sequential_5/lstm_10/while/lstm_cell_10/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :29
7sequential_5/lstm_10/while/lstm_cell_10/split/split_dimã
-sequential_5/lstm_10/while/lstm_cell_10/splitSplit@sequential_5/lstm_10/while/lstm_cell_10/split/split_dim:output:08sequential_5/lstm_10/while/lstm_cell_10/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿº:ÿÿÿÿÿÿÿÿÿº:ÿÿÿÿÿÿÿÿÿº:ÿÿÿÿÿÿÿÿÿº*
	num_split2/
-sequential_5/lstm_10/while/lstm_cell_10/splitØ
/sequential_5/lstm_10/while/lstm_cell_10/SigmoidSigmoid6sequential_5/lstm_10/while/lstm_cell_10/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº21
/sequential_5/lstm_10/while/lstm_cell_10/SigmoidÜ
1sequential_5/lstm_10/while/lstm_cell_10/Sigmoid_1Sigmoid6sequential_5/lstm_10/while/lstm_cell_10/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº23
1sequential_5/lstm_10/while/lstm_cell_10/Sigmoid_1õ
+sequential_5/lstm_10/while/lstm_cell_10/mulMul5sequential_5/lstm_10/while/lstm_cell_10/Sigmoid_1:y:0(sequential_5_lstm_10_while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2-
+sequential_5/lstm_10/while/lstm_cell_10/mulÏ
,sequential_5/lstm_10/while/lstm_cell_10/ReluRelu6sequential_5/lstm_10/while/lstm_cell_10/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2.
,sequential_5/lstm_10/while/lstm_cell_10/Relu
-sequential_5/lstm_10/while/lstm_cell_10/mul_1Mul3sequential_5/lstm_10/while/lstm_cell_10/Sigmoid:y:0:sequential_5/lstm_10/while/lstm_cell_10/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2/
-sequential_5/lstm_10/while/lstm_cell_10/mul_1þ
-sequential_5/lstm_10/while/lstm_cell_10/add_1AddV2/sequential_5/lstm_10/while/lstm_cell_10/mul:z:01sequential_5/lstm_10/while/lstm_cell_10/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2/
-sequential_5/lstm_10/while/lstm_cell_10/add_1Ü
1sequential_5/lstm_10/while/lstm_cell_10/Sigmoid_2Sigmoid6sequential_5/lstm_10/while/lstm_cell_10/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº23
1sequential_5/lstm_10/while/lstm_cell_10/Sigmoid_2Î
.sequential_5/lstm_10/while/lstm_cell_10/Relu_1Relu1sequential_5/lstm_10/while/lstm_cell_10/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº20
.sequential_5/lstm_10/while/lstm_cell_10/Relu_1
-sequential_5/lstm_10/while/lstm_cell_10/mul_2Mul5sequential_5/lstm_10/while/lstm_cell_10/Sigmoid_2:y:0<sequential_5/lstm_10/while/lstm_cell_10/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2/
-sequential_5/lstm_10/while/lstm_cell_10/mul_2É
?sequential_5/lstm_10/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem(sequential_5_lstm_10_while_placeholder_1&sequential_5_lstm_10_while_placeholder1sequential_5/lstm_10/while/lstm_cell_10/mul_2:z:0*
_output_shapes
: *
element_dtype02A
?sequential_5/lstm_10/while/TensorArrayV2Write/TensorListSetItem
 sequential_5/lstm_10/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2"
 sequential_5/lstm_10/while/add/y½
sequential_5/lstm_10/while/addAddV2&sequential_5_lstm_10_while_placeholder)sequential_5/lstm_10/while/add/y:output:0*
T0*
_output_shapes
: 2 
sequential_5/lstm_10/while/add
"sequential_5/lstm_10/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2$
"sequential_5/lstm_10/while/add_1/yß
 sequential_5/lstm_10/while/add_1AddV2Bsequential_5_lstm_10_while_sequential_5_lstm_10_while_loop_counter+sequential_5/lstm_10/while/add_1/y:output:0*
T0*
_output_shapes
: 2"
 sequential_5/lstm_10/while/add_1¿
#sequential_5/lstm_10/while/IdentityIdentity$sequential_5/lstm_10/while/add_1:z:0 ^sequential_5/lstm_10/while/NoOp*
T0*
_output_shapes
: 2%
#sequential_5/lstm_10/while/Identityç
%sequential_5/lstm_10/while/Identity_1IdentityHsequential_5_lstm_10_while_sequential_5_lstm_10_while_maximum_iterations ^sequential_5/lstm_10/while/NoOp*
T0*
_output_shapes
: 2'
%sequential_5/lstm_10/while/Identity_1Á
%sequential_5/lstm_10/while/Identity_2Identity"sequential_5/lstm_10/while/add:z:0 ^sequential_5/lstm_10/while/NoOp*
T0*
_output_shapes
: 2'
%sequential_5/lstm_10/while/Identity_2î
%sequential_5/lstm_10/while/Identity_3IdentityOsequential_5/lstm_10/while/TensorArrayV2Write/TensorListSetItem:output_handle:0 ^sequential_5/lstm_10/while/NoOp*
T0*
_output_shapes
: 2'
%sequential_5/lstm_10/while/Identity_3â
%sequential_5/lstm_10/while/Identity_4Identity1sequential_5/lstm_10/while/lstm_cell_10/mul_2:z:0 ^sequential_5/lstm_10/while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2'
%sequential_5/lstm_10/while/Identity_4â
%sequential_5/lstm_10/while/Identity_5Identity1sequential_5/lstm_10/while/lstm_cell_10/add_1:z:0 ^sequential_5/lstm_10/while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2'
%sequential_5/lstm_10/while/Identity_5Ç
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
%sequential_5_lstm_10_while_identity_5.sequential_5/lstm_10/while/Identity_5:output:0"
Gsequential_5_lstm_10_while_lstm_cell_10_biasadd_readvariableop_resourceIsequential_5_lstm_10_while_lstm_cell_10_biasadd_readvariableop_resource_0"
Hsequential_5_lstm_10_while_lstm_cell_10_matmul_1_readvariableop_resourceJsequential_5_lstm_10_while_lstm_cell_10_matmul_1_readvariableop_resource_0"
Fsequential_5_lstm_10_while_lstm_cell_10_matmul_readvariableop_resourceHsequential_5_lstm_10_while_lstm_cell_10_matmul_readvariableop_resource_0"
?sequential_5_lstm_10_while_sequential_5_lstm_10_strided_slice_1Asequential_5_lstm_10_while_sequential_5_lstm_10_strided_slice_1_0"ü
{sequential_5_lstm_10_while_tensorarrayv2read_tensorlistgetitem_sequential_5_lstm_10_tensorarrayunstack_tensorlistfromtensor}sequential_5_lstm_10_while_tensorarrayv2read_tensorlistgetitem_sequential_5_lstm_10_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿº:ÿÿÿÿÿÿÿÿÿº: : : : : 2
>sequential_5/lstm_10/while/lstm_cell_10/BiasAdd/ReadVariableOp>sequential_5/lstm_10/while/lstm_cell_10/BiasAdd/ReadVariableOp2~
=sequential_5/lstm_10/while/lstm_cell_10/MatMul/ReadVariableOp=sequential_5/lstm_10/while/lstm_cell_10/MatMul/ReadVariableOp2
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
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº:

_output_shapes
: :

_output_shapes
: 
°?
Ô
while_body_19556077
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_10_matmul_readvariableop_resource_0:	]è	I
5while_lstm_cell_10_matmul_1_readvariableop_resource_0:
ºè	C
4while_lstm_cell_10_biasadd_readvariableop_resource_0:	è	
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_10_matmul_readvariableop_resource:	]è	G
3while_lstm_cell_10_matmul_1_readvariableop_resource:
ºè	A
2while_lstm_cell_10_biasadd_readvariableop_resource:	è	¢)while/lstm_cell_10/BiasAdd/ReadVariableOp¢(while/lstm_cell_10/MatMul/ReadVariableOp¢*while/lstm_cell_10/MatMul_1/ReadVariableOpÃ
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
(while/lstm_cell_10/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_10_matmul_readvariableop_resource_0*
_output_shapes
:	]è	*
dtype02*
(while/lstm_cell_10/MatMul/ReadVariableOp×
while/lstm_cell_10/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_10/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè	2
while/lstm_cell_10/MatMulÐ
*while/lstm_cell_10/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_10_matmul_1_readvariableop_resource_0* 
_output_shapes
:
ºè	*
dtype02,
*while/lstm_cell_10/MatMul_1/ReadVariableOpÀ
while/lstm_cell_10/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_10/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè	2
while/lstm_cell_10/MatMul_1¸
while/lstm_cell_10/addAddV2#while/lstm_cell_10/MatMul:product:0%while/lstm_cell_10/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè	2
while/lstm_cell_10/addÈ
)while/lstm_cell_10/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_10_biasadd_readvariableop_resource_0*
_output_shapes	
:è	*
dtype02+
)while/lstm_cell_10/BiasAdd/ReadVariableOpÅ
while/lstm_cell_10/BiasAddBiasAddwhile/lstm_cell_10/add:z:01while/lstm_cell_10/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè	2
while/lstm_cell_10/BiasAdd
"while/lstm_cell_10/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_10/split/split_dim
while/lstm_cell_10/splitSplit+while/lstm_cell_10/split/split_dim:output:0#while/lstm_cell_10/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿº:ÿÿÿÿÿÿÿÿÿº:ÿÿÿÿÿÿÿÿÿº:ÿÿÿÿÿÿÿÿÿº*
	num_split2
while/lstm_cell_10/split
while/lstm_cell_10/SigmoidSigmoid!while/lstm_cell_10/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2
while/lstm_cell_10/Sigmoid
while/lstm_cell_10/Sigmoid_1Sigmoid!while/lstm_cell_10/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2
while/lstm_cell_10/Sigmoid_1¡
while/lstm_cell_10/mulMul while/lstm_cell_10/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2
while/lstm_cell_10/mul
while/lstm_cell_10/ReluRelu!while/lstm_cell_10/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2
while/lstm_cell_10/Reluµ
while/lstm_cell_10/mul_1Mulwhile/lstm_cell_10/Sigmoid:y:0%while/lstm_cell_10/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2
while/lstm_cell_10/mul_1ª
while/lstm_cell_10/add_1AddV2while/lstm_cell_10/mul:z:0while/lstm_cell_10/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2
while/lstm_cell_10/add_1
while/lstm_cell_10/Sigmoid_2Sigmoid!while/lstm_cell_10/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2
while/lstm_cell_10/Sigmoid_2
while/lstm_cell_10/Relu_1Reluwhile/lstm_cell_10/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2
while/lstm_cell_10/Relu_1¹
while/lstm_cell_10/mul_2Mul while/lstm_cell_10/Sigmoid_2:y:0'while/lstm_cell_10/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2
while/lstm_cell_10/mul_2à
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
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_10/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_10/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2
while/Identity_5Þ

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
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿº:ÿÿÿÿÿÿÿÿÿº: : : : : 2V
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
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº:

_output_shapes
: :

_output_shapes
: 
×
g
H__inference_dropout_10_layer_call_and_return_conditional_losses_19555994

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
:ÿÿÿÿÿÿÿÿÿº2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape¹
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿº*
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
:ÿÿÿÿÿÿÿÿÿº2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2
dropout/Mul_1j
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿº:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿº
 
_user_specified_nameinputs
Ç
ù
/__inference_lstm_cell_10_layer_call_fn_19558529

inputs
states_0
states_1
unknown:	]è	
	unknown_0:
ºè	
	unknown_1:	è	
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
<:ÿÿÿÿÿÿÿÿÿº:ÿÿÿÿÿÿÿÿÿº:ÿÿÿÿÿÿÿÿÿº*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_lstm_cell_10_layer_call_and_return_conditional_losses_195543352
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2

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
A:ÿÿÿÿÿÿÿÿÿ]:ÿÿÿÿÿÿÿÿÿº:ÿÿÿÿÿÿÿÿÿº: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
 
_user_specified_nameinputs:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº
"
_user_specified_name
states/0:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº
"
_user_specified_name
states/1
¶
¸
*__inference_lstm_10_layer_call_fn_19557689

inputs
unknown:	]è	
	unknown_0:
ºè	
	unknown_1:	è	
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿº*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_lstm_10_layer_call_and_return_conditional_losses_195561612
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2

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
ß
Í
while_cond_19555042
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_19555042___redundant_placeholder06
2while_while_cond_19555042___redundant_placeholder16
2while_while_cond_19555042___redundant_placeholder26
2while_while_cond_19555042___redundant_placeholder3
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
@: : : : :ÿÿÿÿÿÿÿÿÿS:ÿÿÿÿÿÿÿÿÿS: ::::: 

_output_shapes
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
:ÿÿÿÿÿÿÿÿÿS:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS:

_output_shapes
: :

_output_shapes
:
ß
Í
while_cond_19557782
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_19557782___redundant_placeholder06
2while_while_cond_19557782___redundant_placeholder16
2while_while_cond_19557782___redundant_placeholder26
2while_while_cond_19557782___redundant_placeholder3
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
@: : : : :ÿÿÿÿÿÿÿÿÿS:ÿÿÿÿÿÿÿÿÿS: ::::: 

_output_shapes
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
:ÿÿÿÿÿÿÿÿÿS:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS:

_output_shapes
: :

_output_shapes
:
à
º
*__inference_lstm_11_layer_call_fn_19558342
inputs_0
unknown:
ºÌ
	unknown_0:	SÌ
	unknown_1:	Ì
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿS*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_lstm_11_layer_call_and_return_conditional_losses_195551122
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿS2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿº: : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿº
"
_user_specified_name
inputs/0
ß
Í
while_cond_19555611
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_19555611___redundant_placeholder06
2while_while_cond_19555611___redundant_placeholder16
2while_while_cond_19555611___redundant_placeholder26
2while_while_cond_19555611___redundant_placeholder3
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
@: : : : :ÿÿÿÿÿÿÿÿÿS:ÿÿÿÿÿÿÿÿÿS: ::::: 

_output_shapes
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
:ÿÿÿÿÿÿÿÿÿS:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS:

_output_shapes
: :

_output_shapes
:
?
Ô
while_body_19555612
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
3while_lstm_cell_11_matmul_readvariableop_resource_0:
ºÌH
5while_lstm_cell_11_matmul_1_readvariableop_resource_0:	SÌC
4while_lstm_cell_11_biasadd_readvariableop_resource_0:	Ì
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
1while_lstm_cell_11_matmul_readvariableop_resource:
ºÌF
3while_lstm_cell_11_matmul_1_readvariableop_resource:	SÌA
2while_lstm_cell_11_biasadd_readvariableop_resource:	Ì¢)while/lstm_cell_11/BiasAdd/ReadVariableOp¢(while/lstm_cell_11/MatMul/ReadVariableOp¢*while/lstm_cell_11/MatMul_1/ReadVariableOpÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ:  29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÔ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemÊ
(while/lstm_cell_11/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_11_matmul_readvariableop_resource_0* 
_output_shapes
:
ºÌ*
dtype02*
(while/lstm_cell_11/MatMul/ReadVariableOp×
while/lstm_cell_11/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_11/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÌ2
while/lstm_cell_11/MatMulÏ
*while/lstm_cell_11/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_11_matmul_1_readvariableop_resource_0*
_output_shapes
:	SÌ*
dtype02,
*while/lstm_cell_11/MatMul_1/ReadVariableOpÀ
while/lstm_cell_11/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_11/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÌ2
while/lstm_cell_11/MatMul_1¸
while/lstm_cell_11/addAddV2#while/lstm_cell_11/MatMul:product:0%while/lstm_cell_11/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÌ2
while/lstm_cell_11/addÈ
)while/lstm_cell_11/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_11_biasadd_readvariableop_resource_0*
_output_shapes	
:Ì*
dtype02+
)while/lstm_cell_11/BiasAdd/ReadVariableOpÅ
while/lstm_cell_11/BiasAddBiasAddwhile/lstm_cell_11/add:z:01while/lstm_cell_11/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÌ2
while/lstm_cell_11/BiasAdd
"while/lstm_cell_11/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_11/split/split_dim
while/lstm_cell_11/splitSplit+while/lstm_cell_11/split/split_dim:output:0#while/lstm_cell_11/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿS:ÿÿÿÿÿÿÿÿÿS:ÿÿÿÿÿÿÿÿÿS:ÿÿÿÿÿÿÿÿÿS*
	num_split2
while/lstm_cell_11/split
while/lstm_cell_11/SigmoidSigmoid!while/lstm_cell_11/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2
while/lstm_cell_11/Sigmoid
while/lstm_cell_11/Sigmoid_1Sigmoid!while/lstm_cell_11/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2
while/lstm_cell_11/Sigmoid_1 
while/lstm_cell_11/mulMul while/lstm_cell_11/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2
while/lstm_cell_11/mul
while/lstm_cell_11/ReluRelu!while/lstm_cell_11/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2
while/lstm_cell_11/Relu´
while/lstm_cell_11/mul_1Mulwhile/lstm_cell_11/Sigmoid:y:0%while/lstm_cell_11/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2
while/lstm_cell_11/mul_1©
while/lstm_cell_11/add_1AddV2while/lstm_cell_11/mul:z:0while/lstm_cell_11/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2
while/lstm_cell_11/add_1
while/lstm_cell_11/Sigmoid_2Sigmoid!while/lstm_cell_11/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2
while/lstm_cell_11/Sigmoid_2
while/lstm_cell_11/Relu_1Reluwhile/lstm_cell_11/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2
while/lstm_cell_11/Relu_1¸
while/lstm_cell_11/mul_2Mul while/lstm_cell_11/Sigmoid_2:y:0'while/lstm_cell_11/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2
while/lstm_cell_11/mul_2à
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
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_11/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_11/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2
while/Identity_5Þ

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
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿS:ÿÿÿÿÿÿÿÿÿS: : : : : 2V
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
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS:

_output_shapes
: :

_output_shapes
: 
Ï!
ü
E__inference_dense_5_layer_call_and_return_conditional_losses_19555742

inputs3
!tensordot_readvariableop_resource:S-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢Tensordot/ReadVariableOp
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:S*
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
:ÿÿÿÿÿÿÿÿÿS2
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
:ÿÿÿÿÿÿÿÿÿS: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿS
 
_user_specified_nameinputs
°?
Ô
while_body_19557108
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_10_matmul_readvariableop_resource_0:	]è	I
5while_lstm_cell_10_matmul_1_readvariableop_resource_0:
ºè	C
4while_lstm_cell_10_biasadd_readvariableop_resource_0:	è	
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_10_matmul_readvariableop_resource:	]è	G
3while_lstm_cell_10_matmul_1_readvariableop_resource:
ºè	A
2while_lstm_cell_10_biasadd_readvariableop_resource:	è	¢)while/lstm_cell_10/BiasAdd/ReadVariableOp¢(while/lstm_cell_10/MatMul/ReadVariableOp¢*while/lstm_cell_10/MatMul_1/ReadVariableOpÃ
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
(while/lstm_cell_10/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_10_matmul_readvariableop_resource_0*
_output_shapes
:	]è	*
dtype02*
(while/lstm_cell_10/MatMul/ReadVariableOp×
while/lstm_cell_10/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_10/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè	2
while/lstm_cell_10/MatMulÐ
*while/lstm_cell_10/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_10_matmul_1_readvariableop_resource_0* 
_output_shapes
:
ºè	*
dtype02,
*while/lstm_cell_10/MatMul_1/ReadVariableOpÀ
while/lstm_cell_10/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_10/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè	2
while/lstm_cell_10/MatMul_1¸
while/lstm_cell_10/addAddV2#while/lstm_cell_10/MatMul:product:0%while/lstm_cell_10/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè	2
while/lstm_cell_10/addÈ
)while/lstm_cell_10/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_10_biasadd_readvariableop_resource_0*
_output_shapes	
:è	*
dtype02+
)while/lstm_cell_10/BiasAdd/ReadVariableOpÅ
while/lstm_cell_10/BiasAddBiasAddwhile/lstm_cell_10/add:z:01while/lstm_cell_10/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè	2
while/lstm_cell_10/BiasAdd
"while/lstm_cell_10/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_10/split/split_dim
while/lstm_cell_10/splitSplit+while/lstm_cell_10/split/split_dim:output:0#while/lstm_cell_10/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿº:ÿÿÿÿÿÿÿÿÿº:ÿÿÿÿÿÿÿÿÿº:ÿÿÿÿÿÿÿÿÿº*
	num_split2
while/lstm_cell_10/split
while/lstm_cell_10/SigmoidSigmoid!while/lstm_cell_10/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2
while/lstm_cell_10/Sigmoid
while/lstm_cell_10/Sigmoid_1Sigmoid!while/lstm_cell_10/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2
while/lstm_cell_10/Sigmoid_1¡
while/lstm_cell_10/mulMul while/lstm_cell_10/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2
while/lstm_cell_10/mul
while/lstm_cell_10/ReluRelu!while/lstm_cell_10/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2
while/lstm_cell_10/Reluµ
while/lstm_cell_10/mul_1Mulwhile/lstm_cell_10/Sigmoid:y:0%while/lstm_cell_10/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2
while/lstm_cell_10/mul_1ª
while/lstm_cell_10/add_1AddV2while/lstm_cell_10/mul:z:0while/lstm_cell_10/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2
while/lstm_cell_10/add_1
while/lstm_cell_10/Sigmoid_2Sigmoid!while/lstm_cell_10/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2
while/lstm_cell_10/Sigmoid_2
while/lstm_cell_10/Relu_1Reluwhile/lstm_cell_10/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2
while/lstm_cell_10/Relu_1¹
while/lstm_cell_10/mul_2Mul while/lstm_cell_10/Sigmoid_2:y:0'while/lstm_cell_10/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2
while/lstm_cell_10/mul_2à
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
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_10/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_10/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2
while/Identity_5Þ

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
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿº:ÿÿÿÿÿÿÿÿÿº: : : : : 2V
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
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº:

_output_shapes
: :

_output_shapes
: 
Ã\
 
E__inference_lstm_10_layer_call_and_return_conditional_losses_19557192
inputs_0>
+lstm_cell_10_matmul_readvariableop_resource:	]è	A
-lstm_cell_10_matmul_1_readvariableop_resource:
ºè	;
,lstm_cell_10_biasadd_readvariableop_resource:	è	
identity¢#lstm_cell_10/BiasAdd/ReadVariableOp¢"lstm_cell_10/MatMul/ReadVariableOp¢$lstm_cell_10/MatMul_1/ReadVariableOp¢whileF
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
B :º2
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
B :º2
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
:ÿÿÿÿÿÿÿÿÿº2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :º2
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
B :º2
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
:ÿÿÿÿÿÿÿÿÿº2	
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
"lstm_cell_10/MatMul/ReadVariableOpReadVariableOp+lstm_cell_10_matmul_readvariableop_resource*
_output_shapes
:	]è	*
dtype02$
"lstm_cell_10/MatMul/ReadVariableOp­
lstm_cell_10/MatMulMatMulstrided_slice_2:output:0*lstm_cell_10/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè	2
lstm_cell_10/MatMul¼
$lstm_cell_10/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_10_matmul_1_readvariableop_resource* 
_output_shapes
:
ºè	*
dtype02&
$lstm_cell_10/MatMul_1/ReadVariableOp©
lstm_cell_10/MatMul_1MatMulzeros:output:0,lstm_cell_10/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè	2
lstm_cell_10/MatMul_1 
lstm_cell_10/addAddV2lstm_cell_10/MatMul:product:0lstm_cell_10/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè	2
lstm_cell_10/add´
#lstm_cell_10/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_10_biasadd_readvariableop_resource*
_output_shapes	
:è	*
dtype02%
#lstm_cell_10/BiasAdd/ReadVariableOp­
lstm_cell_10/BiasAddBiasAddlstm_cell_10/add:z:0+lstm_cell_10/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè	2
lstm_cell_10/BiasAdd~
lstm_cell_10/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_10/split/split_dim÷
lstm_cell_10/splitSplit%lstm_cell_10/split/split_dim:output:0lstm_cell_10/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿº:ÿÿÿÿÿÿÿÿÿº:ÿÿÿÿÿÿÿÿÿº:ÿÿÿÿÿÿÿÿÿº*
	num_split2
lstm_cell_10/split
lstm_cell_10/SigmoidSigmoidlstm_cell_10/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2
lstm_cell_10/Sigmoid
lstm_cell_10/Sigmoid_1Sigmoidlstm_cell_10/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2
lstm_cell_10/Sigmoid_1
lstm_cell_10/mulMullstm_cell_10/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2
lstm_cell_10/mul~
lstm_cell_10/ReluRelulstm_cell_10/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2
lstm_cell_10/Relu
lstm_cell_10/mul_1Mullstm_cell_10/Sigmoid:y:0lstm_cell_10/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2
lstm_cell_10/mul_1
lstm_cell_10/add_1AddV2lstm_cell_10/mul:z:0lstm_cell_10/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2
lstm_cell_10/add_1
lstm_cell_10/Sigmoid_2Sigmoidlstm_cell_10/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2
lstm_cell_10/Sigmoid_2}
lstm_cell_10/Relu_1Relulstm_cell_10/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2
lstm_cell_10/Relu_1¡
lstm_cell_10/mul_2Mullstm_cell_10/Sigmoid_2:y:0!lstm_cell_10/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2
lstm_cell_10/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ:  2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_10_matmul_readvariableop_resource-lstm_cell_10_matmul_1_readvariableop_resource,lstm_cell_10_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿº:ÿÿÿÿÿÿÿÿÿº: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_19557108*
condR
while_cond_19557107*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿº:ÿÿÿÿÿÿÿÿÿº: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ:  22
0TensorArrayV2Stack/TensorListStack/element_shapeò
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿº*
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
:ÿÿÿÿÿÿÿÿÿº*
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
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿº2
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
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿº2

IdentityÈ
NoOpNoOp$^lstm_cell_10/BiasAdd/ReadVariableOp#^lstm_cell_10/MatMul/ReadVariableOp%^lstm_cell_10/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ]: : : 2J
#lstm_cell_10/BiasAdd/ReadVariableOp#lstm_cell_10/BiasAdd/ReadVariableOp2H
"lstm_cell_10/MatMul/ReadVariableOp"lstm_cell_10/MatMul/ReadVariableOp2L
$lstm_cell_10/MatMul_1/ReadVariableOp$lstm_cell_10/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ]
"
_user_specified_name
inputs/0
à
º
*__inference_lstm_10_layer_call_fn_19557667
inputs_0
unknown:	]è	
	unknown_0:
ºè	
	unknown_1:	è	
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿº*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_lstm_10_layer_call_and_return_conditional_losses_195544822
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿº2

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
µ
¸
*__inference_lstm_11_layer_call_fn_19558364

inputs
unknown:
ºÌ
	unknown_0:	SÌ
	unknown_1:	Ì
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿS*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_lstm_11_layer_call_and_return_conditional_losses_195559652
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:ÿÿÿÿÿÿÿÿÿº: : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿº
 
_user_specified_nameinputs
Ï
g
H__inference_dropout_11_layer_call_and_return_conditional_losses_19558381

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
:ÿÿÿÿÿÿÿÿÿS2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape¸
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿS*
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
:ÿÿÿÿÿÿÿÿÿS2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2
dropout/Cast~
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2
dropout/Mul_1i
IdentityIdentitydropout/Mul_1:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿS:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿS
 
_user_specified_nameinputs


J__inference_lstm_cell_10_layer_call_and_return_conditional_losses_19554189

inputs

states
states_11
matmul_readvariableop_resource:	]è	4
 matmul_1_readvariableop_resource:
ºè	.
biasadd_readvariableop_resource:	è	
identity

identity_1

identity_2¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	]è	*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè	2
MatMul
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
ºè	*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè	2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè	2
add
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:è	*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè	2	
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
P:ÿÿÿÿÿÿÿÿÿº:ÿÿÿÿÿÿÿÿÿº:ÿÿÿÿÿÿÿÿÿº:ÿÿÿÿÿÿÿÿÿº*
	num_split2
split`
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2	
Sigmoidd
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2
	Sigmoid_1]
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2
mulW
ReluRelusplit:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2
Relui
mul_1MulSigmoid:y:0Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2
mul_1^
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2
add_1d
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2
	Sigmoid_2V
Relu_1Relu	add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2
Relu_1m
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2
mul_2e
IdentityIdentity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2

Identityi

Identity_1Identity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2

Identity_1i

Identity_2Identity	add_1:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2

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
A:ÿÿÿÿÿÿÿÿÿ]:ÿÿÿÿÿÿÿÿÿº:ÿÿÿÿÿÿÿÿÿº: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
 
_user_specified_nameinputs:PL
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº
 
_user_specified_namestates:PL
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº
 
_user_specified_namestates


*__inference_dense_5_layer_call_fn_19558431

inputs
unknown:S
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
E__inference_dense_5_layer_call_and_return_conditional_losses_195557422
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
:ÿÿÿÿÿÿÿÿÿS: : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿS
 
_user_specified_nameinputs
ð%
Þ
!__inference__traced_save_19558686
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
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0)savev2_dense_5_kernel_read_readvariableop'savev2_dense_5_bias_read_readvariableop6savev2_lstm_10_lstm_cell_10_kernel_read_readvariableop@savev2_lstm_10_lstm_cell_10_recurrent_kernel_read_readvariableop4savev2_lstm_10_lstm_cell_10_bias_read_readvariableop6savev2_lstm_11_lstm_cell_11_kernel_read_readvariableop@savev2_lstm_11_lstm_cell_11_recurrent_kernel_read_readvariableop4savev2_lstm_11_lstm_cell_11_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableopsavev2_const"/device:CPU:0*
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
X: :S::	]è	:
ºè	:è	:
ºÌ:	SÌ:Ì: : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:S: 

_output_shapes
::%!

_output_shapes
:	]è	:&"
 
_output_shapes
:
ºè	:!

_output_shapes	
:è	:&"
 
_output_shapes
:
ºÌ:%!

_output_shapes
:	SÌ:!

_output_shapes	
:Ì:	
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
Öø

J__inference_sequential_5_layer_call_and_return_conditional_losses_19556658

inputsF
3lstm_10_lstm_cell_10_matmul_readvariableop_resource:	]è	I
5lstm_10_lstm_cell_10_matmul_1_readvariableop_resource:
ºè	C
4lstm_10_lstm_cell_10_biasadd_readvariableop_resource:	è	G
3lstm_11_lstm_cell_11_matmul_readvariableop_resource:
ºÌH
5lstm_11_lstm_cell_11_matmul_1_readvariableop_resource:	SÌC
4lstm_11_lstm_cell_11_biasadd_readvariableop_resource:	Ì;
)dense_5_tensordot_readvariableop_resource:S5
'dense_5_biasadd_readvariableop_resource:
identity¢dense_5/BiasAdd/ReadVariableOp¢ dense_5/Tensordot/ReadVariableOp¢+lstm_10/lstm_cell_10/BiasAdd/ReadVariableOp¢*lstm_10/lstm_cell_10/MatMul/ReadVariableOp¢,lstm_10/lstm_cell_10/MatMul_1/ReadVariableOp¢lstm_10/while¢+lstm_11/lstm_cell_11/BiasAdd/ReadVariableOp¢*lstm_11/lstm_cell_11/MatMul/ReadVariableOp¢,lstm_11/lstm_cell_11/MatMul_1/ReadVariableOp¢lstm_11/whileT
lstm_10/ShapeShapeinputs*
T0*
_output_shapes
:2
lstm_10/Shape
lstm_10/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_10/strided_slice/stack
lstm_10/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_10/strided_slice/stack_1
lstm_10/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_10/strided_slice/stack_2
lstm_10/strided_sliceStridedSlicelstm_10/Shape:output:0$lstm_10/strided_slice/stack:output:0&lstm_10/strided_slice/stack_1:output:0&lstm_10/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_10/strided_slicem
lstm_10/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :º2
lstm_10/zeros/mul/y
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
B :è2
lstm_10/zeros/Less/y
lstm_10/zeros/LessLesslstm_10/zeros/mul:z:0lstm_10/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_10/zeros/Lesss
lstm_10/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :º2
lstm_10/zeros/packed/1£
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
lstm_10/zeros/Const
lstm_10/zerosFilllstm_10/zeros/packed:output:0lstm_10/zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2
lstm_10/zerosq
lstm_10/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :º2
lstm_10/zeros_1/mul/y
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
B :è2
lstm_10/zeros_1/Less/y
lstm_10/zeros_1/LessLesslstm_10/zeros_1/mul:z:0lstm_10/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_10/zeros_1/Lessw
lstm_10/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :º2
lstm_10/zeros_1/packed/1©
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
lstm_10/zeros_1/Const
lstm_10/zeros_1Filllstm_10/zeros_1/packed:output:0lstm_10/zeros_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2
lstm_10/zeros_1
lstm_10/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_10/transpose/perm
lstm_10/transpose	Transposeinputslstm_10/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]2
lstm_10/transposeg
lstm_10/Shape_1Shapelstm_10/transpose:y:0*
T0*
_output_shapes
:2
lstm_10/Shape_1
lstm_10/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_10/strided_slice_1/stack
lstm_10/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_10/strided_slice_1/stack_1
lstm_10/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_10/strided_slice_1/stack_2
lstm_10/strided_slice_1StridedSlicelstm_10/Shape_1:output:0&lstm_10/strided_slice_1/stack:output:0(lstm_10/strided_slice_1/stack_1:output:0(lstm_10/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_10/strided_slice_1
#lstm_10/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2%
#lstm_10/TensorArrayV2/element_shapeÒ
lstm_10/TensorArrayV2TensorListReserve,lstm_10/TensorArrayV2/element_shape:output:0 lstm_10/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_10/TensorArrayV2Ï
=lstm_10/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ]   2?
=lstm_10/TensorArrayUnstack/TensorListFromTensor/element_shape
/lstm_10/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_10/transpose:y:0Flstm_10/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type021
/lstm_10/TensorArrayUnstack/TensorListFromTensor
lstm_10/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_10/strided_slice_2/stack
lstm_10/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_10/strided_slice_2/stack_1
lstm_10/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_10/strided_slice_2/stack_2¬
lstm_10/strided_slice_2StridedSlicelstm_10/transpose:y:0&lstm_10/strided_slice_2/stack:output:0(lstm_10/strided_slice_2/stack_1:output:0(lstm_10/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]*
shrink_axis_mask2
lstm_10/strided_slice_2Í
*lstm_10/lstm_cell_10/MatMul/ReadVariableOpReadVariableOp3lstm_10_lstm_cell_10_matmul_readvariableop_resource*
_output_shapes
:	]è	*
dtype02,
*lstm_10/lstm_cell_10/MatMul/ReadVariableOpÍ
lstm_10/lstm_cell_10/MatMulMatMul lstm_10/strided_slice_2:output:02lstm_10/lstm_cell_10/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè	2
lstm_10/lstm_cell_10/MatMulÔ
,lstm_10/lstm_cell_10/MatMul_1/ReadVariableOpReadVariableOp5lstm_10_lstm_cell_10_matmul_1_readvariableop_resource* 
_output_shapes
:
ºè	*
dtype02.
,lstm_10/lstm_cell_10/MatMul_1/ReadVariableOpÉ
lstm_10/lstm_cell_10/MatMul_1MatMullstm_10/zeros:output:04lstm_10/lstm_cell_10/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè	2
lstm_10/lstm_cell_10/MatMul_1À
lstm_10/lstm_cell_10/addAddV2%lstm_10/lstm_cell_10/MatMul:product:0'lstm_10/lstm_cell_10/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè	2
lstm_10/lstm_cell_10/addÌ
+lstm_10/lstm_cell_10/BiasAdd/ReadVariableOpReadVariableOp4lstm_10_lstm_cell_10_biasadd_readvariableop_resource*
_output_shapes	
:è	*
dtype02-
+lstm_10/lstm_cell_10/BiasAdd/ReadVariableOpÍ
lstm_10/lstm_cell_10/BiasAddBiasAddlstm_10/lstm_cell_10/add:z:03lstm_10/lstm_cell_10/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè	2
lstm_10/lstm_cell_10/BiasAdd
$lstm_10/lstm_cell_10/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2&
$lstm_10/lstm_cell_10/split/split_dim
lstm_10/lstm_cell_10/splitSplit-lstm_10/lstm_cell_10/split/split_dim:output:0%lstm_10/lstm_cell_10/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿº:ÿÿÿÿÿÿÿÿÿº:ÿÿÿÿÿÿÿÿÿº:ÿÿÿÿÿÿÿÿÿº*
	num_split2
lstm_10/lstm_cell_10/split
lstm_10/lstm_cell_10/SigmoidSigmoid#lstm_10/lstm_cell_10/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2
lstm_10/lstm_cell_10/Sigmoid£
lstm_10/lstm_cell_10/Sigmoid_1Sigmoid#lstm_10/lstm_cell_10/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2 
lstm_10/lstm_cell_10/Sigmoid_1¬
lstm_10/lstm_cell_10/mulMul"lstm_10/lstm_cell_10/Sigmoid_1:y:0lstm_10/zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2
lstm_10/lstm_cell_10/mul
lstm_10/lstm_cell_10/ReluRelu#lstm_10/lstm_cell_10/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2
lstm_10/lstm_cell_10/Relu½
lstm_10/lstm_cell_10/mul_1Mul lstm_10/lstm_cell_10/Sigmoid:y:0'lstm_10/lstm_cell_10/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2
lstm_10/lstm_cell_10/mul_1²
lstm_10/lstm_cell_10/add_1AddV2lstm_10/lstm_cell_10/mul:z:0lstm_10/lstm_cell_10/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2
lstm_10/lstm_cell_10/add_1£
lstm_10/lstm_cell_10/Sigmoid_2Sigmoid#lstm_10/lstm_cell_10/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2 
lstm_10/lstm_cell_10/Sigmoid_2
lstm_10/lstm_cell_10/Relu_1Relulstm_10/lstm_cell_10/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2
lstm_10/lstm_cell_10/Relu_1Á
lstm_10/lstm_cell_10/mul_2Mul"lstm_10/lstm_cell_10/Sigmoid_2:y:0)lstm_10/lstm_cell_10/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2
lstm_10/lstm_cell_10/mul_2
%lstm_10/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ:  2'
%lstm_10/TensorArrayV2_1/element_shapeØ
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
lstm_10/time
 lstm_10/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2"
 lstm_10/while/maximum_iterationsz
lstm_10/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_10/while/loop_counter
lstm_10/whileWhile#lstm_10/while/loop_counter:output:0)lstm_10/while/maximum_iterations:output:0lstm_10/time:output:0 lstm_10/TensorArrayV2_1:handle:0lstm_10/zeros:output:0lstm_10/zeros_1:output:0 lstm_10/strided_slice_1:output:0?lstm_10/TensorArrayUnstack/TensorListFromTensor:output_handle:03lstm_10_lstm_cell_10_matmul_readvariableop_resource5lstm_10_lstm_cell_10_matmul_1_readvariableop_resource4lstm_10_lstm_cell_10_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿº:ÿÿÿÿÿÿÿÿÿº: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *'
bodyR
lstm_10_while_body_19556398*'
condR
lstm_10_while_cond_19556397*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿº:ÿÿÿÿÿÿÿÿÿº: : : : : *
parallel_iterations 2
lstm_10/whileÅ
8lstm_10/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ:  2:
8lstm_10/TensorArrayV2Stack/TensorListStack/element_shape
*lstm_10/TensorArrayV2Stack/TensorListStackTensorListStacklstm_10/while:output:3Alstm_10/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿº*
element_dtype02,
*lstm_10/TensorArrayV2Stack/TensorListStack
lstm_10/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
lstm_10/strided_slice_3/stack
lstm_10/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2!
lstm_10/strided_slice_3/stack_1
lstm_10/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_10/strided_slice_3/stack_2Ë
lstm_10/strided_slice_3StridedSlice3lstm_10/TensorArrayV2Stack/TensorListStack:tensor:0&lstm_10/strided_slice_3/stack:output:0(lstm_10/strided_slice_3/stack_1:output:0(lstm_10/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº*
shrink_axis_mask2
lstm_10/strided_slice_3
lstm_10/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_10/transpose_1/permÆ
lstm_10/transpose_1	Transpose3lstm_10/TensorArrayV2Stack/TensorListStack:tensor:0!lstm_10/transpose_1/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2
lstm_10/transpose_1v
lstm_10/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_10/runtime
dropout_10/IdentityIdentitylstm_10/transpose_1:y:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2
dropout_10/Identityj
lstm_11/ShapeShapedropout_10/Identity:output:0*
T0*
_output_shapes
:2
lstm_11/Shape
lstm_11/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_11/strided_slice/stack
lstm_11/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_11/strided_slice/stack_1
lstm_11/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_11/strided_slice/stack_2
lstm_11/strided_sliceStridedSlicelstm_11/Shape:output:0$lstm_11/strided_slice/stack:output:0&lstm_11/strided_slice/stack_1:output:0&lstm_11/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_11/strided_slicel
lstm_11/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :S2
lstm_11/zeros/mul/y
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
B :è2
lstm_11/zeros/Less/y
lstm_11/zeros/LessLesslstm_11/zeros/mul:z:0lstm_11/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_11/zeros/Lessr
lstm_11/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :S2
lstm_11/zeros/packed/1£
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
lstm_11/zeros/Const
lstm_11/zerosFilllstm_11/zeros/packed:output:0lstm_11/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2
lstm_11/zerosp
lstm_11/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :S2
lstm_11/zeros_1/mul/y
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
B :è2
lstm_11/zeros_1/Less/y
lstm_11/zeros_1/LessLesslstm_11/zeros_1/mul:z:0lstm_11/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_11/zeros_1/Lessv
lstm_11/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :S2
lstm_11/zeros_1/packed/1©
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
lstm_11/zeros_1/Const
lstm_11/zeros_1Filllstm_11/zeros_1/packed:output:0lstm_11/zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2
lstm_11/zeros_1
lstm_11/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_11/transpose/perm©
lstm_11/transpose	Transposedropout_10/Identity:output:0lstm_11/transpose/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2
lstm_11/transposeg
lstm_11/Shape_1Shapelstm_11/transpose:y:0*
T0*
_output_shapes
:2
lstm_11/Shape_1
lstm_11/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_11/strided_slice_1/stack
lstm_11/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_11/strided_slice_1/stack_1
lstm_11/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_11/strided_slice_1/stack_2
lstm_11/strided_slice_1StridedSlicelstm_11/Shape_1:output:0&lstm_11/strided_slice_1/stack:output:0(lstm_11/strided_slice_1/stack_1:output:0(lstm_11/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_11/strided_slice_1
#lstm_11/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2%
#lstm_11/TensorArrayV2/element_shapeÒ
lstm_11/TensorArrayV2TensorListReserve,lstm_11/TensorArrayV2/element_shape:output:0 lstm_11/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_11/TensorArrayV2Ï
=lstm_11/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ:  2?
=lstm_11/TensorArrayUnstack/TensorListFromTensor/element_shape
/lstm_11/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_11/transpose:y:0Flstm_11/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type021
/lstm_11/TensorArrayUnstack/TensorListFromTensor
lstm_11/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_11/strided_slice_2/stack
lstm_11/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_11/strided_slice_2/stack_1
lstm_11/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_11/strided_slice_2/stack_2­
lstm_11/strided_slice_2StridedSlicelstm_11/transpose:y:0&lstm_11/strided_slice_2/stack:output:0(lstm_11/strided_slice_2/stack_1:output:0(lstm_11/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº*
shrink_axis_mask2
lstm_11/strided_slice_2Î
*lstm_11/lstm_cell_11/MatMul/ReadVariableOpReadVariableOp3lstm_11_lstm_cell_11_matmul_readvariableop_resource* 
_output_shapes
:
ºÌ*
dtype02,
*lstm_11/lstm_cell_11/MatMul/ReadVariableOpÍ
lstm_11/lstm_cell_11/MatMulMatMul lstm_11/strided_slice_2:output:02lstm_11/lstm_cell_11/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÌ2
lstm_11/lstm_cell_11/MatMulÓ
,lstm_11/lstm_cell_11/MatMul_1/ReadVariableOpReadVariableOp5lstm_11_lstm_cell_11_matmul_1_readvariableop_resource*
_output_shapes
:	SÌ*
dtype02.
,lstm_11/lstm_cell_11/MatMul_1/ReadVariableOpÉ
lstm_11/lstm_cell_11/MatMul_1MatMullstm_11/zeros:output:04lstm_11/lstm_cell_11/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÌ2
lstm_11/lstm_cell_11/MatMul_1À
lstm_11/lstm_cell_11/addAddV2%lstm_11/lstm_cell_11/MatMul:product:0'lstm_11/lstm_cell_11/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÌ2
lstm_11/lstm_cell_11/addÌ
+lstm_11/lstm_cell_11/BiasAdd/ReadVariableOpReadVariableOp4lstm_11_lstm_cell_11_biasadd_readvariableop_resource*
_output_shapes	
:Ì*
dtype02-
+lstm_11/lstm_cell_11/BiasAdd/ReadVariableOpÍ
lstm_11/lstm_cell_11/BiasAddBiasAddlstm_11/lstm_cell_11/add:z:03lstm_11/lstm_cell_11/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÌ2
lstm_11/lstm_cell_11/BiasAdd
$lstm_11/lstm_cell_11/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2&
$lstm_11/lstm_cell_11/split/split_dim
lstm_11/lstm_cell_11/splitSplit-lstm_11/lstm_cell_11/split/split_dim:output:0%lstm_11/lstm_cell_11/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿS:ÿÿÿÿÿÿÿÿÿS:ÿÿÿÿÿÿÿÿÿS:ÿÿÿÿÿÿÿÿÿS*
	num_split2
lstm_11/lstm_cell_11/split
lstm_11/lstm_cell_11/SigmoidSigmoid#lstm_11/lstm_cell_11/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2
lstm_11/lstm_cell_11/Sigmoid¢
lstm_11/lstm_cell_11/Sigmoid_1Sigmoid#lstm_11/lstm_cell_11/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2 
lstm_11/lstm_cell_11/Sigmoid_1«
lstm_11/lstm_cell_11/mulMul"lstm_11/lstm_cell_11/Sigmoid_1:y:0lstm_11/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2
lstm_11/lstm_cell_11/mul
lstm_11/lstm_cell_11/ReluRelu#lstm_11/lstm_cell_11/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2
lstm_11/lstm_cell_11/Relu¼
lstm_11/lstm_cell_11/mul_1Mul lstm_11/lstm_cell_11/Sigmoid:y:0'lstm_11/lstm_cell_11/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2
lstm_11/lstm_cell_11/mul_1±
lstm_11/lstm_cell_11/add_1AddV2lstm_11/lstm_cell_11/mul:z:0lstm_11/lstm_cell_11/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2
lstm_11/lstm_cell_11/add_1¢
lstm_11/lstm_cell_11/Sigmoid_2Sigmoid#lstm_11/lstm_cell_11/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2 
lstm_11/lstm_cell_11/Sigmoid_2
lstm_11/lstm_cell_11/Relu_1Relulstm_11/lstm_cell_11/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2
lstm_11/lstm_cell_11/Relu_1À
lstm_11/lstm_cell_11/mul_2Mul"lstm_11/lstm_cell_11/Sigmoid_2:y:0)lstm_11/lstm_cell_11/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2
lstm_11/lstm_cell_11/mul_2
%lstm_11/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿS   2'
%lstm_11/TensorArrayV2_1/element_shapeØ
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
lstm_11/time
 lstm_11/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2"
 lstm_11/while/maximum_iterationsz
lstm_11/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_11/while/loop_counter
lstm_11/whileWhile#lstm_11/while/loop_counter:output:0)lstm_11/while/maximum_iterations:output:0lstm_11/time:output:0 lstm_11/TensorArrayV2_1:handle:0lstm_11/zeros:output:0lstm_11/zeros_1:output:0 lstm_11/strided_slice_1:output:0?lstm_11/TensorArrayUnstack/TensorListFromTensor:output_handle:03lstm_11_lstm_cell_11_matmul_readvariableop_resource5lstm_11_lstm_cell_11_matmul_1_readvariableop_resource4lstm_11_lstm_cell_11_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿS:ÿÿÿÿÿÿÿÿÿS: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *'
bodyR
lstm_11_while_body_19556546*'
condR
lstm_11_while_cond_19556545*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿS:ÿÿÿÿÿÿÿÿÿS: : : : : *
parallel_iterations 2
lstm_11/whileÅ
8lstm_11/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿS   2:
8lstm_11/TensorArrayV2Stack/TensorListStack/element_shape
*lstm_11/TensorArrayV2Stack/TensorListStackTensorListStacklstm_11/while:output:3Alstm_11/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿS*
element_dtype02,
*lstm_11/TensorArrayV2Stack/TensorListStack
lstm_11/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
lstm_11/strided_slice_3/stack
lstm_11/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2!
lstm_11/strided_slice_3/stack_1
lstm_11/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_11/strided_slice_3/stack_2Ê
lstm_11/strided_slice_3StridedSlice3lstm_11/TensorArrayV2Stack/TensorListStack:tensor:0&lstm_11/strided_slice_3/stack:output:0(lstm_11/strided_slice_3/stack_1:output:0(lstm_11/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS*
shrink_axis_mask2
lstm_11/strided_slice_3
lstm_11/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_11/transpose_1/permÅ
lstm_11/transpose_1	Transpose3lstm_11/TensorArrayV2Stack/TensorListStack:tensor:0!lstm_11/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2
lstm_11/transpose_1v
lstm_11/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_11/runtime
dropout_11/IdentityIdentitylstm_11/transpose_1:y:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2
dropout_11/Identity®
 dense_5/Tensordot/ReadVariableOpReadVariableOp)dense_5_tensordot_readvariableop_resource*
_output_shapes

:S*
dtype02"
 dense_5/Tensordot/ReadVariableOpz
dense_5/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_5/Tensordot/axes
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
dense_5/Tensordot/Shape
dense_5/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_5/Tensordot/GatherV2/axisù
dense_5/Tensordot/GatherV2GatherV2 dense_5/Tensordot/Shape:output:0dense_5/Tensordot/free:output:0(dense_5/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_5/Tensordot/GatherV2
!dense_5/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_5/Tensordot/GatherV2_1/axisÿ
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
dense_5/Tensordot/Const 
dense_5/Tensordot/ProdProd#dense_5/Tensordot/GatherV2:output:0 dense_5/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_5/Tensordot/Prod
dense_5/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_5/Tensordot/Const_1¨
dense_5/Tensordot/Prod_1Prod%dense_5/Tensordot/GatherV2_1:output:0"dense_5/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_5/Tensordot/Prod_1
dense_5/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense_5/Tensordot/concat/axisØ
dense_5/Tensordot/concatConcatV2dense_5/Tensordot/free:output:0dense_5/Tensordot/axes:output:0&dense_5/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_5/Tensordot/concat¬
dense_5/Tensordot/stackPackdense_5/Tensordot/Prod:output:0!dense_5/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_5/Tensordot/stack¾
dense_5/Tensordot/transpose	Transposedropout_11/Identity:output:0!dense_5/Tensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2
dense_5/Tensordot/transpose¿
dense_5/Tensordot/ReshapeReshapedense_5/Tensordot/transpose:y:0 dense_5/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
dense_5/Tensordot/Reshape¾
dense_5/Tensordot/MatMulMatMul"dense_5/Tensordot/Reshape:output:0(dense_5/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_5/Tensordot/MatMul
dense_5/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
dense_5/Tensordot/Const_2
dense_5/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_5/Tensordot/concat_1/axiså
dense_5/Tensordot/concat_1ConcatV2#dense_5/Tensordot/GatherV2:output:0"dense_5/Tensordot/Const_2:output:0(dense_5/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_5/Tensordot/concat_1°
dense_5/TensordotReshape"dense_5/Tensordot/MatMul:product:0#dense_5/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_5/Tensordot¤
dense_5/BiasAdd/ReadVariableOpReadVariableOp'dense_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_5/BiasAdd/ReadVariableOp§
dense_5/BiasAddBiasAdddense_5/Tensordot:output:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_5/BiasAdd}
dense_5/SoftmaxSoftmaxdense_5/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_5/Softmaxx
IdentityIdentitydense_5/Softmax:softmax:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

IdentityÆ
NoOpNoOp^dense_5/BiasAdd/ReadVariableOp!^dense_5/Tensordot/ReadVariableOp,^lstm_10/lstm_cell_10/BiasAdd/ReadVariableOp+^lstm_10/lstm_cell_10/MatMul/ReadVariableOp-^lstm_10/lstm_cell_10/MatMul_1/ReadVariableOp^lstm_10/while,^lstm_11/lstm_cell_11/BiasAdd/ReadVariableOp+^lstm_11/lstm_cell_11/MatMul/ReadVariableOp-^lstm_11/lstm_cell_11/MatMul_1/ReadVariableOp^lstm_11/while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ]: : : : : : : : 2@
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
:ÿÿÿÿÿÿÿÿÿ]
 
_user_specified_nameinputs
à
º
*__inference_lstm_11_layer_call_fn_19558331
inputs_0
unknown:
ºÌ
	unknown_0:	SÌ
	unknown_1:	Ì
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿS*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_lstm_11_layer_call_and_return_conditional_losses_195549022
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿS2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿº: : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿº
"
_user_specified_name
inputs/0
à
º
*__inference_lstm_10_layer_call_fn_19557656
inputs_0
unknown:	]è	
	unknown_0:
ºè	
	unknown_1:	è	
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿº*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_lstm_10_layer_call_and_return_conditional_losses_195542722
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿº2

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
?
Ô
while_body_19555881
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
3while_lstm_cell_11_matmul_readvariableop_resource_0:
ºÌH
5while_lstm_cell_11_matmul_1_readvariableop_resource_0:	SÌC
4while_lstm_cell_11_biasadd_readvariableop_resource_0:	Ì
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
1while_lstm_cell_11_matmul_readvariableop_resource:
ºÌF
3while_lstm_cell_11_matmul_1_readvariableop_resource:	SÌA
2while_lstm_cell_11_biasadd_readvariableop_resource:	Ì¢)while/lstm_cell_11/BiasAdd/ReadVariableOp¢(while/lstm_cell_11/MatMul/ReadVariableOp¢*while/lstm_cell_11/MatMul_1/ReadVariableOpÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ:  29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÔ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemÊ
(while/lstm_cell_11/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_11_matmul_readvariableop_resource_0* 
_output_shapes
:
ºÌ*
dtype02*
(while/lstm_cell_11/MatMul/ReadVariableOp×
while/lstm_cell_11/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_11/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÌ2
while/lstm_cell_11/MatMulÏ
*while/lstm_cell_11/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_11_matmul_1_readvariableop_resource_0*
_output_shapes
:	SÌ*
dtype02,
*while/lstm_cell_11/MatMul_1/ReadVariableOpÀ
while/lstm_cell_11/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_11/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÌ2
while/lstm_cell_11/MatMul_1¸
while/lstm_cell_11/addAddV2#while/lstm_cell_11/MatMul:product:0%while/lstm_cell_11/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÌ2
while/lstm_cell_11/addÈ
)while/lstm_cell_11/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_11_biasadd_readvariableop_resource_0*
_output_shapes	
:Ì*
dtype02+
)while/lstm_cell_11/BiasAdd/ReadVariableOpÅ
while/lstm_cell_11/BiasAddBiasAddwhile/lstm_cell_11/add:z:01while/lstm_cell_11/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÌ2
while/lstm_cell_11/BiasAdd
"while/lstm_cell_11/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_11/split/split_dim
while/lstm_cell_11/splitSplit+while/lstm_cell_11/split/split_dim:output:0#while/lstm_cell_11/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿS:ÿÿÿÿÿÿÿÿÿS:ÿÿÿÿÿÿÿÿÿS:ÿÿÿÿÿÿÿÿÿS*
	num_split2
while/lstm_cell_11/split
while/lstm_cell_11/SigmoidSigmoid!while/lstm_cell_11/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2
while/lstm_cell_11/Sigmoid
while/lstm_cell_11/Sigmoid_1Sigmoid!while/lstm_cell_11/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2
while/lstm_cell_11/Sigmoid_1 
while/lstm_cell_11/mulMul while/lstm_cell_11/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2
while/lstm_cell_11/mul
while/lstm_cell_11/ReluRelu!while/lstm_cell_11/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2
while/lstm_cell_11/Relu´
while/lstm_cell_11/mul_1Mulwhile/lstm_cell_11/Sigmoid:y:0%while/lstm_cell_11/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2
while/lstm_cell_11/mul_1©
while/lstm_cell_11/add_1AddV2while/lstm_cell_11/mul:z:0while/lstm_cell_11/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2
while/lstm_cell_11/add_1
while/lstm_cell_11/Sigmoid_2Sigmoid!while/lstm_cell_11/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2
while/lstm_cell_11/Sigmoid_2
while/lstm_cell_11/Relu_1Reluwhile/lstm_cell_11/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2
while/lstm_cell_11/Relu_1¸
while/lstm_cell_11/mul_2Mul while/lstm_cell_11/Sigmoid_2:y:0'while/lstm_cell_11/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2
while/lstm_cell_11/mul_2à
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
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_11/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_11/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2
while/Identity_5Þ

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
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿS:ÿÿÿÿÿÿÿÿÿS: : : : : 2V
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
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS:

_output_shapes
: :

_output_shapes
: 
ËF

E__inference_lstm_10_layer_call_and_return_conditional_losses_19554272

inputs(
lstm_cell_10_19554190:	]è	)
lstm_cell_10_19554192:
ºè	$
lstm_cell_10_19554194:	è	
identity¢$lstm_cell_10/StatefulPartitionedCall¢whileD
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
B :º2
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
B :º2
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
:ÿÿÿÿÿÿÿÿÿº2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :º2
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
B :º2
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
:ÿÿÿÿÿÿÿÿÿº2	
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
$lstm_cell_10/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_10_19554190lstm_cell_10_19554192lstm_cell_10_19554194*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿº:ÿÿÿÿÿÿÿÿÿº:ÿÿÿÿÿÿÿÿÿº*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_lstm_cell_10_layer_call_and_return_conditional_losses_195541892&
$lstm_cell_10/StatefulPartitionedCall
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ:  2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_10_19554190lstm_cell_10_19554192lstm_cell_10_19554194*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿº:ÿÿÿÿÿÿÿÿÿº: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_19554203*
condR
while_cond_19554202*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿº:ÿÿÿÿÿÿÿÿÿº: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ:  22
0TensorArrayV2Stack/TensorListStack/element_shapeò
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿº*
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
:ÿÿÿÿÿÿÿÿÿº*
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
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿº2
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
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿº2

Identity}
NoOpNoOp%^lstm_cell_10/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ]: : : 2L
$lstm_cell_10/StatefulPartitionedCall$lstm_cell_10/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ]
 
_user_specified_nameinputs
½
ù
/__inference_lstm_cell_11_layer_call_fn_19558610

inputs
states_0
states_1
unknown:
ºÌ
	unknown_0:	SÌ
	unknown_1:	Ì
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
9:ÿÿÿÿÿÿÿÿÿS:ÿÿÿÿÿÿÿÿÿS:ÿÿÿÿÿÿÿÿÿS*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_lstm_cell_11_layer_call_and_return_conditional_losses_195548192
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2

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
@:ÿÿÿÿÿÿÿÿÿº:ÿÿÿÿÿÿÿÿÿS:ÿÿÿÿÿÿÿÿÿS: : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº
 
_user_specified_nameinputs:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS
"
_user_specified_name
states/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS
"
_user_specified_name
states/1
Ã\
 
E__inference_lstm_10_layer_call_and_return_conditional_losses_19557343
inputs_0>
+lstm_cell_10_matmul_readvariableop_resource:	]è	A
-lstm_cell_10_matmul_1_readvariableop_resource:
ºè	;
,lstm_cell_10_biasadd_readvariableop_resource:	è	
identity¢#lstm_cell_10/BiasAdd/ReadVariableOp¢"lstm_cell_10/MatMul/ReadVariableOp¢$lstm_cell_10/MatMul_1/ReadVariableOp¢whileF
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
B :º2
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
B :º2
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
:ÿÿÿÿÿÿÿÿÿº2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :º2
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
B :º2
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
:ÿÿÿÿÿÿÿÿÿº2	
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
"lstm_cell_10/MatMul/ReadVariableOpReadVariableOp+lstm_cell_10_matmul_readvariableop_resource*
_output_shapes
:	]è	*
dtype02$
"lstm_cell_10/MatMul/ReadVariableOp­
lstm_cell_10/MatMulMatMulstrided_slice_2:output:0*lstm_cell_10/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè	2
lstm_cell_10/MatMul¼
$lstm_cell_10/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_10_matmul_1_readvariableop_resource* 
_output_shapes
:
ºè	*
dtype02&
$lstm_cell_10/MatMul_1/ReadVariableOp©
lstm_cell_10/MatMul_1MatMulzeros:output:0,lstm_cell_10/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè	2
lstm_cell_10/MatMul_1 
lstm_cell_10/addAddV2lstm_cell_10/MatMul:product:0lstm_cell_10/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè	2
lstm_cell_10/add´
#lstm_cell_10/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_10_biasadd_readvariableop_resource*
_output_shapes	
:è	*
dtype02%
#lstm_cell_10/BiasAdd/ReadVariableOp­
lstm_cell_10/BiasAddBiasAddlstm_cell_10/add:z:0+lstm_cell_10/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè	2
lstm_cell_10/BiasAdd~
lstm_cell_10/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_10/split/split_dim÷
lstm_cell_10/splitSplit%lstm_cell_10/split/split_dim:output:0lstm_cell_10/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿº:ÿÿÿÿÿÿÿÿÿº:ÿÿÿÿÿÿÿÿÿº:ÿÿÿÿÿÿÿÿÿº*
	num_split2
lstm_cell_10/split
lstm_cell_10/SigmoidSigmoidlstm_cell_10/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2
lstm_cell_10/Sigmoid
lstm_cell_10/Sigmoid_1Sigmoidlstm_cell_10/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2
lstm_cell_10/Sigmoid_1
lstm_cell_10/mulMullstm_cell_10/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2
lstm_cell_10/mul~
lstm_cell_10/ReluRelulstm_cell_10/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2
lstm_cell_10/Relu
lstm_cell_10/mul_1Mullstm_cell_10/Sigmoid:y:0lstm_cell_10/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2
lstm_cell_10/mul_1
lstm_cell_10/add_1AddV2lstm_cell_10/mul:z:0lstm_cell_10/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2
lstm_cell_10/add_1
lstm_cell_10/Sigmoid_2Sigmoidlstm_cell_10/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2
lstm_cell_10/Sigmoid_2}
lstm_cell_10/Relu_1Relulstm_cell_10/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2
lstm_cell_10/Relu_1¡
lstm_cell_10/mul_2Mullstm_cell_10/Sigmoid_2:y:0!lstm_cell_10/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2
lstm_cell_10/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ:  2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_10_matmul_readvariableop_resource-lstm_cell_10_matmul_1_readvariableop_resource,lstm_cell_10_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿº:ÿÿÿÿÿÿÿÿÿº: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_19557259*
condR
while_cond_19557258*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿº:ÿÿÿÿÿÿÿÿÿº: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ:  22
0TensorArrayV2Stack/TensorListStack/element_shapeò
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿº*
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
:ÿÿÿÿÿÿÿÿÿº*
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
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿº2
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
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿº2

IdentityÈ
NoOpNoOp$^lstm_cell_10/BiasAdd/ReadVariableOp#^lstm_cell_10/MatMul/ReadVariableOp%^lstm_cell_10/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ]: : : 2J
#lstm_cell_10/BiasAdd/ReadVariableOp#lstm_cell_10/BiasAdd/ReadVariableOp2H
"lstm_cell_10/MatMul/ReadVariableOp"lstm_cell_10/MatMul/ReadVariableOp2L
$lstm_cell_10/MatMul_1/ReadVariableOp$lstm_cell_10/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ]
"
_user_specified_name
inputs/0
Ï
g
H__inference_dropout_11_layer_call_and_return_conditional_losses_19555798

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
:ÿÿÿÿÿÿÿÿÿS2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape¸
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿS*
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
:ÿÿÿÿÿÿÿÿÿS2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2
dropout/Cast~
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2
dropout/Mul_1i
IdentityIdentitydropout/Mul_1:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿS:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿS
 
_user_specified_nameinputs
ù

J__inference_lstm_cell_11_layer_call_and_return_conditional_losses_19554819

inputs

states
states_12
matmul_readvariableop_resource:
ºÌ3
 matmul_1_readvariableop_resource:	SÌ.
biasadd_readvariableop_resource:	Ì
identity

identity_1

identity_2¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
ºÌ*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÌ2
MatMul
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	SÌ*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÌ2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÌ2
add
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:Ì*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÌ2	
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
L:ÿÿÿÿÿÿÿÿÿS:ÿÿÿÿÿÿÿÿÿS:ÿÿÿÿÿÿÿÿÿS:ÿÿÿÿÿÿÿÿÿS*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2
	Sigmoid_1\
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2
mulV
ReluRelusplit:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2
	Sigmoid_2U
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2
mul_2d
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2

Identityh

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2

Identity_1h

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2

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
@:ÿÿÿÿÿÿÿÿÿº:ÿÿÿÿÿÿÿÿÿS:ÿÿÿÿÿÿÿÿÿS: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS
 
_user_specified_namestates:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS
 
_user_specified_namestates
åJ
Ô

lstm_10_while_body_19556398,
(lstm_10_while_lstm_10_while_loop_counter2
.lstm_10_while_lstm_10_while_maximum_iterations
lstm_10_while_placeholder
lstm_10_while_placeholder_1
lstm_10_while_placeholder_2
lstm_10_while_placeholder_3+
'lstm_10_while_lstm_10_strided_slice_1_0g
clstm_10_while_tensorarrayv2read_tensorlistgetitem_lstm_10_tensorarrayunstack_tensorlistfromtensor_0N
;lstm_10_while_lstm_cell_10_matmul_readvariableop_resource_0:	]è	Q
=lstm_10_while_lstm_cell_10_matmul_1_readvariableop_resource_0:
ºè	K
<lstm_10_while_lstm_cell_10_biasadd_readvariableop_resource_0:	è	
lstm_10_while_identity
lstm_10_while_identity_1
lstm_10_while_identity_2
lstm_10_while_identity_3
lstm_10_while_identity_4
lstm_10_while_identity_5)
%lstm_10_while_lstm_10_strided_slice_1e
alstm_10_while_tensorarrayv2read_tensorlistgetitem_lstm_10_tensorarrayunstack_tensorlistfromtensorL
9lstm_10_while_lstm_cell_10_matmul_readvariableop_resource:	]è	O
;lstm_10_while_lstm_cell_10_matmul_1_readvariableop_resource:
ºè	I
:lstm_10_while_lstm_cell_10_biasadd_readvariableop_resource:	è	¢1lstm_10/while/lstm_cell_10/BiasAdd/ReadVariableOp¢0lstm_10/while/lstm_cell_10/MatMul/ReadVariableOp¢2lstm_10/while/lstm_cell_10/MatMul_1/ReadVariableOpÓ
?lstm_10/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ]   2A
?lstm_10/while/TensorArrayV2Read/TensorListGetItem/element_shape
1lstm_10/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemclstm_10_while_tensorarrayv2read_tensorlistgetitem_lstm_10_tensorarrayunstack_tensorlistfromtensor_0lstm_10_while_placeholderHlstm_10/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]*
element_dtype023
1lstm_10/while/TensorArrayV2Read/TensorListGetItemá
0lstm_10/while/lstm_cell_10/MatMul/ReadVariableOpReadVariableOp;lstm_10_while_lstm_cell_10_matmul_readvariableop_resource_0*
_output_shapes
:	]è	*
dtype022
0lstm_10/while/lstm_cell_10/MatMul/ReadVariableOp÷
!lstm_10/while/lstm_cell_10/MatMulMatMul8lstm_10/while/TensorArrayV2Read/TensorListGetItem:item:08lstm_10/while/lstm_cell_10/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè	2#
!lstm_10/while/lstm_cell_10/MatMulè
2lstm_10/while/lstm_cell_10/MatMul_1/ReadVariableOpReadVariableOp=lstm_10_while_lstm_cell_10_matmul_1_readvariableop_resource_0* 
_output_shapes
:
ºè	*
dtype024
2lstm_10/while/lstm_cell_10/MatMul_1/ReadVariableOpà
#lstm_10/while/lstm_cell_10/MatMul_1MatMullstm_10_while_placeholder_2:lstm_10/while/lstm_cell_10/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè	2%
#lstm_10/while/lstm_cell_10/MatMul_1Ø
lstm_10/while/lstm_cell_10/addAddV2+lstm_10/while/lstm_cell_10/MatMul:product:0-lstm_10/while/lstm_cell_10/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè	2 
lstm_10/while/lstm_cell_10/addà
1lstm_10/while/lstm_cell_10/BiasAdd/ReadVariableOpReadVariableOp<lstm_10_while_lstm_cell_10_biasadd_readvariableop_resource_0*
_output_shapes	
:è	*
dtype023
1lstm_10/while/lstm_cell_10/BiasAdd/ReadVariableOpå
"lstm_10/while/lstm_cell_10/BiasAddBiasAdd"lstm_10/while/lstm_cell_10/add:z:09lstm_10/while/lstm_cell_10/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè	2$
"lstm_10/while/lstm_cell_10/BiasAdd
*lstm_10/while/lstm_cell_10/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2,
*lstm_10/while/lstm_cell_10/split/split_dim¯
 lstm_10/while/lstm_cell_10/splitSplit3lstm_10/while/lstm_cell_10/split/split_dim:output:0+lstm_10/while/lstm_cell_10/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿº:ÿÿÿÿÿÿÿÿÿº:ÿÿÿÿÿÿÿÿÿº:ÿÿÿÿÿÿÿÿÿº*
	num_split2"
 lstm_10/while/lstm_cell_10/split±
"lstm_10/while/lstm_cell_10/SigmoidSigmoid)lstm_10/while/lstm_cell_10/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2$
"lstm_10/while/lstm_cell_10/Sigmoidµ
$lstm_10/while/lstm_cell_10/Sigmoid_1Sigmoid)lstm_10/while/lstm_cell_10/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2&
$lstm_10/while/lstm_cell_10/Sigmoid_1Á
lstm_10/while/lstm_cell_10/mulMul(lstm_10/while/lstm_cell_10/Sigmoid_1:y:0lstm_10_while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2 
lstm_10/while/lstm_cell_10/mul¨
lstm_10/while/lstm_cell_10/ReluRelu)lstm_10/while/lstm_cell_10/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2!
lstm_10/while/lstm_cell_10/ReluÕ
 lstm_10/while/lstm_cell_10/mul_1Mul&lstm_10/while/lstm_cell_10/Sigmoid:y:0-lstm_10/while/lstm_cell_10/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2"
 lstm_10/while/lstm_cell_10/mul_1Ê
 lstm_10/while/lstm_cell_10/add_1AddV2"lstm_10/while/lstm_cell_10/mul:z:0$lstm_10/while/lstm_cell_10/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2"
 lstm_10/while/lstm_cell_10/add_1µ
$lstm_10/while/lstm_cell_10/Sigmoid_2Sigmoid)lstm_10/while/lstm_cell_10/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2&
$lstm_10/while/lstm_cell_10/Sigmoid_2§
!lstm_10/while/lstm_cell_10/Relu_1Relu$lstm_10/while/lstm_cell_10/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2#
!lstm_10/while/lstm_cell_10/Relu_1Ù
 lstm_10/while/lstm_cell_10/mul_2Mul(lstm_10/while/lstm_cell_10/Sigmoid_2:y:0/lstm_10/while/lstm_cell_10/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2"
 lstm_10/while/lstm_cell_10/mul_2
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
lstm_10/while/add/y
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
lstm_10/while/add_1/y
lstm_10/while/add_1AddV2(lstm_10_while_lstm_10_while_loop_counterlstm_10/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lstm_10/while/add_1
lstm_10/while/IdentityIdentitylstm_10/while/add_1:z:0^lstm_10/while/NoOp*
T0*
_output_shapes
: 2
lstm_10/while/Identity¦
lstm_10/while/Identity_1Identity.lstm_10_while_lstm_10_while_maximum_iterations^lstm_10/while/NoOp*
T0*
_output_shapes
: 2
lstm_10/while/Identity_1
lstm_10/while/Identity_2Identitylstm_10/while/add:z:0^lstm_10/while/NoOp*
T0*
_output_shapes
: 2
lstm_10/while/Identity_2º
lstm_10/while/Identity_3IdentityBlstm_10/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_10/while/NoOp*
T0*
_output_shapes
: 2
lstm_10/while/Identity_3®
lstm_10/while/Identity_4Identity$lstm_10/while/lstm_cell_10/mul_2:z:0^lstm_10/while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2
lstm_10/while/Identity_4®
lstm_10/while/Identity_5Identity$lstm_10/while/lstm_cell_10/add_1:z:0^lstm_10/while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2
lstm_10/while/Identity_5
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
9lstm_10_while_lstm_cell_10_matmul_readvariableop_resource;lstm_10_while_lstm_cell_10_matmul_readvariableop_resource_0"È
alstm_10_while_tensorarrayv2read_tensorlistgetitem_lstm_10_tensorarrayunstack_tensorlistfromtensorclstm_10_while_tensorarrayv2read_tensorlistgetitem_lstm_10_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿº:ÿÿÿÿÿÿÿÿÿº: : : : : 2f
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
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº:

_output_shapes
: :

_output_shapes
: 
ã
Í
while_cond_19557560
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_19557560___redundant_placeholder06
2while_while_cond_19557560___redundant_placeholder16
2while_while_cond_19557560___redundant_placeholder26
2while_while_cond_19557560___redundant_placeholder3
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
B: : : : :ÿÿÿÿÿÿÿÿÿº:ÿÿÿÿÿÿÿÿÿº: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº:

_output_shapes
: :

_output_shapes
:


Ê
/__inference_sequential_5_layer_call_fn_19557041

inputs
unknown:	]è	
	unknown_0:
ºè	
	unknown_1:	è	
	unknown_2:
ºÌ
	unknown_3:	SÌ
	unknown_4:	Ì
	unknown_5:S
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
J__inference_sequential_5_layer_call_and_return_conditional_losses_195562182
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
E__inference_lstm_11_layer_call_and_return_conditional_losses_19555965

inputs?
+lstm_cell_11_matmul_readvariableop_resource:
ºÌ@
-lstm_cell_11_matmul_1_readvariableop_resource:	SÌ;
,lstm_cell_11_biasadd_readvariableop_resource:	Ì
identity¢#lstm_cell_11/BiasAdd/ReadVariableOp¢"lstm_cell_11/MatMul/ReadVariableOp¢$lstm_cell_11/MatMul_1/ReadVariableOp¢whileD
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
value	B :S2
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
value	B :S2
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
:ÿÿÿÿÿÿÿÿÿS2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :S2
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
value	B :S2
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
:ÿÿÿÿÿÿÿÿÿS2	
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
:ÿÿÿÿÿÿÿÿÿº2
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
valueB"ÿÿÿÿ:  27
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
:ÿÿÿÿÿÿÿÿÿº*
shrink_axis_mask2
strided_slice_2¶
"lstm_cell_11/MatMul/ReadVariableOpReadVariableOp+lstm_cell_11_matmul_readvariableop_resource* 
_output_shapes
:
ºÌ*
dtype02$
"lstm_cell_11/MatMul/ReadVariableOp­
lstm_cell_11/MatMulMatMulstrided_slice_2:output:0*lstm_cell_11/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÌ2
lstm_cell_11/MatMul»
$lstm_cell_11/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_11_matmul_1_readvariableop_resource*
_output_shapes
:	SÌ*
dtype02&
$lstm_cell_11/MatMul_1/ReadVariableOp©
lstm_cell_11/MatMul_1MatMulzeros:output:0,lstm_cell_11/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÌ2
lstm_cell_11/MatMul_1 
lstm_cell_11/addAddV2lstm_cell_11/MatMul:product:0lstm_cell_11/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÌ2
lstm_cell_11/add´
#lstm_cell_11/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_11_biasadd_readvariableop_resource*
_output_shapes	
:Ì*
dtype02%
#lstm_cell_11/BiasAdd/ReadVariableOp­
lstm_cell_11/BiasAddBiasAddlstm_cell_11/add:z:0+lstm_cell_11/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÌ2
lstm_cell_11/BiasAdd~
lstm_cell_11/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_11/split/split_dimó
lstm_cell_11/splitSplit%lstm_cell_11/split/split_dim:output:0lstm_cell_11/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿS:ÿÿÿÿÿÿÿÿÿS:ÿÿÿÿÿÿÿÿÿS:ÿÿÿÿÿÿÿÿÿS*
	num_split2
lstm_cell_11/split
lstm_cell_11/SigmoidSigmoidlstm_cell_11/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2
lstm_cell_11/Sigmoid
lstm_cell_11/Sigmoid_1Sigmoidlstm_cell_11/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2
lstm_cell_11/Sigmoid_1
lstm_cell_11/mulMullstm_cell_11/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2
lstm_cell_11/mul}
lstm_cell_11/ReluRelulstm_cell_11/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2
lstm_cell_11/Relu
lstm_cell_11/mul_1Mullstm_cell_11/Sigmoid:y:0lstm_cell_11/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2
lstm_cell_11/mul_1
lstm_cell_11/add_1AddV2lstm_cell_11/mul:z:0lstm_cell_11/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2
lstm_cell_11/add_1
lstm_cell_11/Sigmoid_2Sigmoidlstm_cell_11/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2
lstm_cell_11/Sigmoid_2|
lstm_cell_11/Relu_1Relulstm_cell_11/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2
lstm_cell_11/Relu_1 
lstm_cell_11/mul_2Mullstm_cell_11/Sigmoid_2:y:0!lstm_cell_11/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2
lstm_cell_11/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿS   2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_11_matmul_readvariableop_resource-lstm_cell_11_matmul_1_readvariableop_resource,lstm_cell_11_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿS:ÿÿÿÿÿÿÿÿÿS: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_19555881*
condR
while_cond_19555880*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿS:ÿÿÿÿÿÿÿÿÿS: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿS   22
0TensorArrayV2Stack/TensorListStack/element_shapeè
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿS*
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
:ÿÿÿÿÿÿÿÿÿS*
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
:ÿÿÿÿÿÿÿÿÿS2
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
:ÿÿÿÿÿÿÿÿÿS2

IdentityÈ
NoOpNoOp$^lstm_cell_11/BiasAdd/ReadVariableOp#^lstm_cell_11/MatMul/ReadVariableOp%^lstm_cell_11/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:ÿÿÿÿÿÿÿÿÿº: : : 2J
#lstm_cell_11/BiasAdd/ReadVariableOp#lstm_cell_11/BiasAdd/ReadVariableOp2H
"lstm_cell_11/MatMul/ReadVariableOp"lstm_cell_11/MatMul/ReadVariableOp2L
$lstm_cell_11/MatMul_1/ReadVariableOp$lstm_cell_11/MatMul_1/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿº
 
_user_specified_nameinputs
°?
Ô
while_body_19557561
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_10_matmul_readvariableop_resource_0:	]è	I
5while_lstm_cell_10_matmul_1_readvariableop_resource_0:
ºè	C
4while_lstm_cell_10_biasadd_readvariableop_resource_0:	è	
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_10_matmul_readvariableop_resource:	]è	G
3while_lstm_cell_10_matmul_1_readvariableop_resource:
ºè	A
2while_lstm_cell_10_biasadd_readvariableop_resource:	è	¢)while/lstm_cell_10/BiasAdd/ReadVariableOp¢(while/lstm_cell_10/MatMul/ReadVariableOp¢*while/lstm_cell_10/MatMul_1/ReadVariableOpÃ
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
(while/lstm_cell_10/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_10_matmul_readvariableop_resource_0*
_output_shapes
:	]è	*
dtype02*
(while/lstm_cell_10/MatMul/ReadVariableOp×
while/lstm_cell_10/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_10/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè	2
while/lstm_cell_10/MatMulÐ
*while/lstm_cell_10/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_10_matmul_1_readvariableop_resource_0* 
_output_shapes
:
ºè	*
dtype02,
*while/lstm_cell_10/MatMul_1/ReadVariableOpÀ
while/lstm_cell_10/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_10/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè	2
while/lstm_cell_10/MatMul_1¸
while/lstm_cell_10/addAddV2#while/lstm_cell_10/MatMul:product:0%while/lstm_cell_10/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè	2
while/lstm_cell_10/addÈ
)while/lstm_cell_10/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_10_biasadd_readvariableop_resource_0*
_output_shapes	
:è	*
dtype02+
)while/lstm_cell_10/BiasAdd/ReadVariableOpÅ
while/lstm_cell_10/BiasAddBiasAddwhile/lstm_cell_10/add:z:01while/lstm_cell_10/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè	2
while/lstm_cell_10/BiasAdd
"while/lstm_cell_10/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_10/split/split_dim
while/lstm_cell_10/splitSplit+while/lstm_cell_10/split/split_dim:output:0#while/lstm_cell_10/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿº:ÿÿÿÿÿÿÿÿÿº:ÿÿÿÿÿÿÿÿÿº:ÿÿÿÿÿÿÿÿÿº*
	num_split2
while/lstm_cell_10/split
while/lstm_cell_10/SigmoidSigmoid!while/lstm_cell_10/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2
while/lstm_cell_10/Sigmoid
while/lstm_cell_10/Sigmoid_1Sigmoid!while/lstm_cell_10/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2
while/lstm_cell_10/Sigmoid_1¡
while/lstm_cell_10/mulMul while/lstm_cell_10/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2
while/lstm_cell_10/mul
while/lstm_cell_10/ReluRelu!while/lstm_cell_10/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2
while/lstm_cell_10/Reluµ
while/lstm_cell_10/mul_1Mulwhile/lstm_cell_10/Sigmoid:y:0%while/lstm_cell_10/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2
while/lstm_cell_10/mul_1ª
while/lstm_cell_10/add_1AddV2while/lstm_cell_10/mul:z:0while/lstm_cell_10/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2
while/lstm_cell_10/add_1
while/lstm_cell_10/Sigmoid_2Sigmoid!while/lstm_cell_10/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2
while/lstm_cell_10/Sigmoid_2
while/lstm_cell_10/Relu_1Reluwhile/lstm_cell_10/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2
while/lstm_cell_10/Relu_1¹
while/lstm_cell_10/mul_2Mul while/lstm_cell_10/Sigmoid_2:y:0'while/lstm_cell_10/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2
while/lstm_cell_10/mul_2à
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
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_10/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_10/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2
while/Identity_5Þ

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
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿº:ÿÿÿÿÿÿÿÿÿº: : : : : 2V
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
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº:

_output_shapes
: :

_output_shapes
: 
ß
Í
while_cond_19554832
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_19554832___redundant_placeholder06
2while_while_cond_19554832___redundant_placeholder16
2while_while_cond_19554832___redundant_placeholder26
2while_while_cond_19554832___redundant_placeholder3
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
@: : : : :ÿÿÿÿÿÿÿÿÿS:ÿÿÿÿÿÿÿÿÿS: ::::: 

_output_shapes
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
:ÿÿÿÿÿÿÿÿÿS:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS:

_output_shapes
: :

_output_shapes
:
Ô

í
lstm_10_while_cond_19556397,
(lstm_10_while_lstm_10_while_loop_counter2
.lstm_10_while_lstm_10_while_maximum_iterations
lstm_10_while_placeholder
lstm_10_while_placeholder_1
lstm_10_while_placeholder_2
lstm_10_while_placeholder_3.
*lstm_10_while_less_lstm_10_strided_slice_1F
Blstm_10_while_lstm_10_while_cond_19556397___redundant_placeholder0F
Blstm_10_while_lstm_10_while_cond_19556397___redundant_placeholder1F
Blstm_10_while_lstm_10_while_cond_19556397___redundant_placeholder2F
Blstm_10_while_lstm_10_while_cond_19556397___redundant_placeholder3
lstm_10_while_identity

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
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :ÿÿÿÿÿÿÿÿÿº:ÿÿÿÿÿÿÿÿÿº: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº:

_output_shapes
: :

_output_shapes
:


J__inference_lstm_cell_10_layer_call_and_return_conditional_losses_19558463

inputs
states_0
states_11
matmul_readvariableop_resource:	]è	4
 matmul_1_readvariableop_resource:
ºè	.
biasadd_readvariableop_resource:	è	
identity

identity_1

identity_2¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	]è	*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè	2
MatMul
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
ºè	*
dtype02
MatMul_1/ReadVariableOp|
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè	2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè	2
add
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:è	*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè	2	
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
P:ÿÿÿÿÿÿÿÿÿº:ÿÿÿÿÿÿÿÿÿº:ÿÿÿÿÿÿÿÿÿº:ÿÿÿÿÿÿÿÿÿº*
	num_split2
split`
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2	
Sigmoidd
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2
	Sigmoid_1]
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2
mulW
ReluRelusplit:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2
Relui
mul_1MulSigmoid:y:0Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2
mul_1^
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2
add_1d
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2
	Sigmoid_2V
Relu_1Relu	add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2
Relu_1m
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2
mul_2e
IdentityIdentity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2

Identityi

Identity_1Identity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2

Identity_1i

Identity_2Identity	add_1:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2

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
A:ÿÿÿÿÿÿÿÿÿ]:ÿÿÿÿÿÿÿÿÿº:ÿÿÿÿÿÿÿÿÿº: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
 
_user_specified_nameinputs:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº
"
_user_specified_name
states/0:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº
"
_user_specified_name
states/1
ÓJ
Ô

lstm_11_while_body_19556546,
(lstm_11_while_lstm_11_while_loop_counter2
.lstm_11_while_lstm_11_while_maximum_iterations
lstm_11_while_placeholder
lstm_11_while_placeholder_1
lstm_11_while_placeholder_2
lstm_11_while_placeholder_3+
'lstm_11_while_lstm_11_strided_slice_1_0g
clstm_11_while_tensorarrayv2read_tensorlistgetitem_lstm_11_tensorarrayunstack_tensorlistfromtensor_0O
;lstm_11_while_lstm_cell_11_matmul_readvariableop_resource_0:
ºÌP
=lstm_11_while_lstm_cell_11_matmul_1_readvariableop_resource_0:	SÌK
<lstm_11_while_lstm_cell_11_biasadd_readvariableop_resource_0:	Ì
lstm_11_while_identity
lstm_11_while_identity_1
lstm_11_while_identity_2
lstm_11_while_identity_3
lstm_11_while_identity_4
lstm_11_while_identity_5)
%lstm_11_while_lstm_11_strided_slice_1e
alstm_11_while_tensorarrayv2read_tensorlistgetitem_lstm_11_tensorarrayunstack_tensorlistfromtensorM
9lstm_11_while_lstm_cell_11_matmul_readvariableop_resource:
ºÌN
;lstm_11_while_lstm_cell_11_matmul_1_readvariableop_resource:	SÌI
:lstm_11_while_lstm_cell_11_biasadd_readvariableop_resource:	Ì¢1lstm_11/while/lstm_cell_11/BiasAdd/ReadVariableOp¢0lstm_11/while/lstm_cell_11/MatMul/ReadVariableOp¢2lstm_11/while/lstm_cell_11/MatMul_1/ReadVariableOpÓ
?lstm_11/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ:  2A
?lstm_11/while/TensorArrayV2Read/TensorListGetItem/element_shape
1lstm_11/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemclstm_11_while_tensorarrayv2read_tensorlistgetitem_lstm_11_tensorarrayunstack_tensorlistfromtensor_0lstm_11_while_placeholderHlstm_11/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº*
element_dtype023
1lstm_11/while/TensorArrayV2Read/TensorListGetItemâ
0lstm_11/while/lstm_cell_11/MatMul/ReadVariableOpReadVariableOp;lstm_11_while_lstm_cell_11_matmul_readvariableop_resource_0* 
_output_shapes
:
ºÌ*
dtype022
0lstm_11/while/lstm_cell_11/MatMul/ReadVariableOp÷
!lstm_11/while/lstm_cell_11/MatMulMatMul8lstm_11/while/TensorArrayV2Read/TensorListGetItem:item:08lstm_11/while/lstm_cell_11/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÌ2#
!lstm_11/while/lstm_cell_11/MatMulç
2lstm_11/while/lstm_cell_11/MatMul_1/ReadVariableOpReadVariableOp=lstm_11_while_lstm_cell_11_matmul_1_readvariableop_resource_0*
_output_shapes
:	SÌ*
dtype024
2lstm_11/while/lstm_cell_11/MatMul_1/ReadVariableOpà
#lstm_11/while/lstm_cell_11/MatMul_1MatMullstm_11_while_placeholder_2:lstm_11/while/lstm_cell_11/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÌ2%
#lstm_11/while/lstm_cell_11/MatMul_1Ø
lstm_11/while/lstm_cell_11/addAddV2+lstm_11/while/lstm_cell_11/MatMul:product:0-lstm_11/while/lstm_cell_11/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÌ2 
lstm_11/while/lstm_cell_11/addà
1lstm_11/while/lstm_cell_11/BiasAdd/ReadVariableOpReadVariableOp<lstm_11_while_lstm_cell_11_biasadd_readvariableop_resource_0*
_output_shapes	
:Ì*
dtype023
1lstm_11/while/lstm_cell_11/BiasAdd/ReadVariableOpå
"lstm_11/while/lstm_cell_11/BiasAddBiasAdd"lstm_11/while/lstm_cell_11/add:z:09lstm_11/while/lstm_cell_11/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÌ2$
"lstm_11/while/lstm_cell_11/BiasAdd
*lstm_11/while/lstm_cell_11/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2,
*lstm_11/while/lstm_cell_11/split/split_dim«
 lstm_11/while/lstm_cell_11/splitSplit3lstm_11/while/lstm_cell_11/split/split_dim:output:0+lstm_11/while/lstm_cell_11/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿS:ÿÿÿÿÿÿÿÿÿS:ÿÿÿÿÿÿÿÿÿS:ÿÿÿÿÿÿÿÿÿS*
	num_split2"
 lstm_11/while/lstm_cell_11/split°
"lstm_11/while/lstm_cell_11/SigmoidSigmoid)lstm_11/while/lstm_cell_11/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2$
"lstm_11/while/lstm_cell_11/Sigmoid´
$lstm_11/while/lstm_cell_11/Sigmoid_1Sigmoid)lstm_11/while/lstm_cell_11/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2&
$lstm_11/while/lstm_cell_11/Sigmoid_1À
lstm_11/while/lstm_cell_11/mulMul(lstm_11/while/lstm_cell_11/Sigmoid_1:y:0lstm_11_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2 
lstm_11/while/lstm_cell_11/mul§
lstm_11/while/lstm_cell_11/ReluRelu)lstm_11/while/lstm_cell_11/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2!
lstm_11/while/lstm_cell_11/ReluÔ
 lstm_11/while/lstm_cell_11/mul_1Mul&lstm_11/while/lstm_cell_11/Sigmoid:y:0-lstm_11/while/lstm_cell_11/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2"
 lstm_11/while/lstm_cell_11/mul_1É
 lstm_11/while/lstm_cell_11/add_1AddV2"lstm_11/while/lstm_cell_11/mul:z:0$lstm_11/while/lstm_cell_11/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2"
 lstm_11/while/lstm_cell_11/add_1´
$lstm_11/while/lstm_cell_11/Sigmoid_2Sigmoid)lstm_11/while/lstm_cell_11/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2&
$lstm_11/while/lstm_cell_11/Sigmoid_2¦
!lstm_11/while/lstm_cell_11/Relu_1Relu$lstm_11/while/lstm_cell_11/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2#
!lstm_11/while/lstm_cell_11/Relu_1Ø
 lstm_11/while/lstm_cell_11/mul_2Mul(lstm_11/while/lstm_cell_11/Sigmoid_2:y:0/lstm_11/while/lstm_cell_11/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2"
 lstm_11/while/lstm_cell_11/mul_2
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
lstm_11/while/add/y
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
lstm_11/while/add_1/y
lstm_11/while/add_1AddV2(lstm_11_while_lstm_11_while_loop_counterlstm_11/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lstm_11/while/add_1
lstm_11/while/IdentityIdentitylstm_11/while/add_1:z:0^lstm_11/while/NoOp*
T0*
_output_shapes
: 2
lstm_11/while/Identity¦
lstm_11/while/Identity_1Identity.lstm_11_while_lstm_11_while_maximum_iterations^lstm_11/while/NoOp*
T0*
_output_shapes
: 2
lstm_11/while/Identity_1
lstm_11/while/Identity_2Identitylstm_11/while/add:z:0^lstm_11/while/NoOp*
T0*
_output_shapes
: 2
lstm_11/while/Identity_2º
lstm_11/while/Identity_3IdentityBlstm_11/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_11/while/NoOp*
T0*
_output_shapes
: 2
lstm_11/while/Identity_3­
lstm_11/while/Identity_4Identity$lstm_11/while/lstm_cell_11/mul_2:z:0^lstm_11/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2
lstm_11/while/Identity_4­
lstm_11/while/Identity_5Identity$lstm_11/while/lstm_cell_11/add_1:z:0^lstm_11/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2
lstm_11/while/Identity_5
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
9lstm_11_while_lstm_cell_11_matmul_readvariableop_resource;lstm_11_while_lstm_cell_11_matmul_readvariableop_resource_0"È
alstm_11_while_tensorarrayv2read_tensorlistgetitem_lstm_11_tensorarrayunstack_tensorlistfromtensorclstm_11_while_tensorarrayv2read_tensorlistgetitem_lstm_11_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿS:ÿÿÿÿÿÿÿÿÿS: : : : : 2f
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
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS:

_output_shapes
: :

_output_shapes
: 
ã
Í
while_cond_19557409
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_19557409___redundant_placeholder06
2while_while_cond_19557409___redundant_placeholder16
2while_while_cond_19557409___redundant_placeholder26
2while_while_cond_19557409___redundant_placeholder3
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
B: : : : :ÿÿÿÿÿÿÿÿÿº:ÿÿÿÿÿÿÿÿÿº: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº:

_output_shapes
: :

_output_shapes
:
×
g
H__inference_dropout_10_layer_call_and_return_conditional_losses_19557706

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
:ÿÿÿÿÿÿÿÿÿº2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape¹
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿº*
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
:ÿÿÿÿÿÿÿÿÿº2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2
dropout/Mul_1j
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿº:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿº
 
_user_specified_nameinputs
ª

Ñ
/__inference_sequential_5_layer_call_fn_19555768
lstm_10_input
unknown:	]è	
	unknown_0:
ºè	
	unknown_1:	è	
	unknown_2:
ºÌ
	unknown_3:	SÌ
	unknown_4:	Ì
	unknown_5:S
	unknown_6:
identity¢StatefulPartitionedCallÓ
StatefulPartitionedCallStatefulPartitionedCalllstm_10_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
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
J__inference_sequential_5_layer_call_and_return_conditional_losses_195557492
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
_user_specified_namelstm_10_input
í[

E__inference_lstm_11_layer_call_and_return_conditional_losses_19558169

inputs?
+lstm_cell_11_matmul_readvariableop_resource:
ºÌ@
-lstm_cell_11_matmul_1_readvariableop_resource:	SÌ;
,lstm_cell_11_biasadd_readvariableop_resource:	Ì
identity¢#lstm_cell_11/BiasAdd/ReadVariableOp¢"lstm_cell_11/MatMul/ReadVariableOp¢$lstm_cell_11/MatMul_1/ReadVariableOp¢whileD
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
value	B :S2
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
value	B :S2
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
:ÿÿÿÿÿÿÿÿÿS2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :S2
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
value	B :S2
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
:ÿÿÿÿÿÿÿÿÿS2	
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
:ÿÿÿÿÿÿÿÿÿº2
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
valueB"ÿÿÿÿ:  27
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
:ÿÿÿÿÿÿÿÿÿº*
shrink_axis_mask2
strided_slice_2¶
"lstm_cell_11/MatMul/ReadVariableOpReadVariableOp+lstm_cell_11_matmul_readvariableop_resource* 
_output_shapes
:
ºÌ*
dtype02$
"lstm_cell_11/MatMul/ReadVariableOp­
lstm_cell_11/MatMulMatMulstrided_slice_2:output:0*lstm_cell_11/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÌ2
lstm_cell_11/MatMul»
$lstm_cell_11/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_11_matmul_1_readvariableop_resource*
_output_shapes
:	SÌ*
dtype02&
$lstm_cell_11/MatMul_1/ReadVariableOp©
lstm_cell_11/MatMul_1MatMulzeros:output:0,lstm_cell_11/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÌ2
lstm_cell_11/MatMul_1 
lstm_cell_11/addAddV2lstm_cell_11/MatMul:product:0lstm_cell_11/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÌ2
lstm_cell_11/add´
#lstm_cell_11/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_11_biasadd_readvariableop_resource*
_output_shapes	
:Ì*
dtype02%
#lstm_cell_11/BiasAdd/ReadVariableOp­
lstm_cell_11/BiasAddBiasAddlstm_cell_11/add:z:0+lstm_cell_11/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÌ2
lstm_cell_11/BiasAdd~
lstm_cell_11/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_11/split/split_dimó
lstm_cell_11/splitSplit%lstm_cell_11/split/split_dim:output:0lstm_cell_11/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿS:ÿÿÿÿÿÿÿÿÿS:ÿÿÿÿÿÿÿÿÿS:ÿÿÿÿÿÿÿÿÿS*
	num_split2
lstm_cell_11/split
lstm_cell_11/SigmoidSigmoidlstm_cell_11/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2
lstm_cell_11/Sigmoid
lstm_cell_11/Sigmoid_1Sigmoidlstm_cell_11/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2
lstm_cell_11/Sigmoid_1
lstm_cell_11/mulMullstm_cell_11/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2
lstm_cell_11/mul}
lstm_cell_11/ReluRelulstm_cell_11/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2
lstm_cell_11/Relu
lstm_cell_11/mul_1Mullstm_cell_11/Sigmoid:y:0lstm_cell_11/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2
lstm_cell_11/mul_1
lstm_cell_11/add_1AddV2lstm_cell_11/mul:z:0lstm_cell_11/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2
lstm_cell_11/add_1
lstm_cell_11/Sigmoid_2Sigmoidlstm_cell_11/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2
lstm_cell_11/Sigmoid_2|
lstm_cell_11/Relu_1Relulstm_cell_11/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2
lstm_cell_11/Relu_1 
lstm_cell_11/mul_2Mullstm_cell_11/Sigmoid_2:y:0!lstm_cell_11/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2
lstm_cell_11/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿS   2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_11_matmul_readvariableop_resource-lstm_cell_11_matmul_1_readvariableop_resource,lstm_cell_11_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿS:ÿÿÿÿÿÿÿÿÿS: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_19558085*
condR
while_cond_19558084*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿS:ÿÿÿÿÿÿÿÿÿS: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿS   22
0TensorArrayV2Stack/TensorListStack/element_shapeè
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿS*
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
:ÿÿÿÿÿÿÿÿÿS*
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
:ÿÿÿÿÿÿÿÿÿS2
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
:ÿÿÿÿÿÿÿÿÿS2

IdentityÈ
NoOpNoOp$^lstm_cell_11/BiasAdd/ReadVariableOp#^lstm_cell_11/MatMul/ReadVariableOp%^lstm_cell_11/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:ÿÿÿÿÿÿÿÿÿº: : : 2J
#lstm_cell_11/BiasAdd/ReadVariableOp#lstm_cell_11/BiasAdd/ReadVariableOp2H
"lstm_cell_11/MatMul/ReadVariableOp"lstm_cell_11/MatMul/ReadVariableOp2L
$lstm_cell_11/MatMul_1/ReadVariableOp$lstm_cell_11/MatMul_1/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿº
 
_user_specified_nameinputs


J__inference_lstm_cell_10_layer_call_and_return_conditional_losses_19558495

inputs
states_0
states_11
matmul_readvariableop_resource:	]è	4
 matmul_1_readvariableop_resource:
ºè	.
biasadd_readvariableop_resource:	è	
identity

identity_1

identity_2¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	]è	*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè	2
MatMul
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
ºè	*
dtype02
MatMul_1/ReadVariableOp|
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè	2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè	2
add
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:è	*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè	2	
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
P:ÿÿÿÿÿÿÿÿÿº:ÿÿÿÿÿÿÿÿÿº:ÿÿÿÿÿÿÿÿÿº:ÿÿÿÿÿÿÿÿÿº*
	num_split2
split`
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2	
Sigmoidd
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2
	Sigmoid_1]
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2
mulW
ReluRelusplit:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2
Relui
mul_1MulSigmoid:y:0Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2
mul_1^
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2
add_1d
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2
	Sigmoid_2V
Relu_1Relu	add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2
Relu_1m
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2
mul_2e
IdentityIdentity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2

Identityi

Identity_1Identity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2

Identity_1i

Identity_2Identity	add_1:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2

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
A:ÿÿÿÿÿÿÿÿÿ]:ÿÿÿÿÿÿÿÿÿº:ÿÿÿÿÿÿÿÿÿº: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
 
_user_specified_nameinputs:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº
"
_user_specified_name
states/0:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº
"
_user_specified_name
states/1
°?
Ô
while_body_19555447
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_10_matmul_readvariableop_resource_0:	]è	I
5while_lstm_cell_10_matmul_1_readvariableop_resource_0:
ºè	C
4while_lstm_cell_10_biasadd_readvariableop_resource_0:	è	
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_10_matmul_readvariableop_resource:	]è	G
3while_lstm_cell_10_matmul_1_readvariableop_resource:
ºè	A
2while_lstm_cell_10_biasadd_readvariableop_resource:	è	¢)while/lstm_cell_10/BiasAdd/ReadVariableOp¢(while/lstm_cell_10/MatMul/ReadVariableOp¢*while/lstm_cell_10/MatMul_1/ReadVariableOpÃ
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
(while/lstm_cell_10/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_10_matmul_readvariableop_resource_0*
_output_shapes
:	]è	*
dtype02*
(while/lstm_cell_10/MatMul/ReadVariableOp×
while/lstm_cell_10/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_10/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè	2
while/lstm_cell_10/MatMulÐ
*while/lstm_cell_10/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_10_matmul_1_readvariableop_resource_0* 
_output_shapes
:
ºè	*
dtype02,
*while/lstm_cell_10/MatMul_1/ReadVariableOpÀ
while/lstm_cell_10/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_10/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè	2
while/lstm_cell_10/MatMul_1¸
while/lstm_cell_10/addAddV2#while/lstm_cell_10/MatMul:product:0%while/lstm_cell_10/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè	2
while/lstm_cell_10/addÈ
)while/lstm_cell_10/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_10_biasadd_readvariableop_resource_0*
_output_shapes	
:è	*
dtype02+
)while/lstm_cell_10/BiasAdd/ReadVariableOpÅ
while/lstm_cell_10/BiasAddBiasAddwhile/lstm_cell_10/add:z:01while/lstm_cell_10/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè	2
while/lstm_cell_10/BiasAdd
"while/lstm_cell_10/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_10/split/split_dim
while/lstm_cell_10/splitSplit+while/lstm_cell_10/split/split_dim:output:0#while/lstm_cell_10/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿº:ÿÿÿÿÿÿÿÿÿº:ÿÿÿÿÿÿÿÿÿº:ÿÿÿÿÿÿÿÿÿº*
	num_split2
while/lstm_cell_10/split
while/lstm_cell_10/SigmoidSigmoid!while/lstm_cell_10/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2
while/lstm_cell_10/Sigmoid
while/lstm_cell_10/Sigmoid_1Sigmoid!while/lstm_cell_10/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2
while/lstm_cell_10/Sigmoid_1¡
while/lstm_cell_10/mulMul while/lstm_cell_10/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2
while/lstm_cell_10/mul
while/lstm_cell_10/ReluRelu!while/lstm_cell_10/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2
while/lstm_cell_10/Reluµ
while/lstm_cell_10/mul_1Mulwhile/lstm_cell_10/Sigmoid:y:0%while/lstm_cell_10/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2
while/lstm_cell_10/mul_1ª
while/lstm_cell_10/add_1AddV2while/lstm_cell_10/mul:z:0while/lstm_cell_10/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2
while/lstm_cell_10/add_1
while/lstm_cell_10/Sigmoid_2Sigmoid!while/lstm_cell_10/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2
while/lstm_cell_10/Sigmoid_2
while/lstm_cell_10/Relu_1Reluwhile/lstm_cell_10/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2
while/lstm_cell_10/Relu_1¹
while/lstm_cell_10/mul_2Mul while/lstm_cell_10/Sigmoid_2:y:0'while/lstm_cell_10/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2
while/lstm_cell_10/mul_2à
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
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_10/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_10/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2
while/Identity_5Þ

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
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿº:ÿÿÿÿÿÿÿÿÿº: : : : : 2V
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
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº:

_output_shapes
: :

_output_shapes
: 
°?
Ô
while_body_19557410
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_10_matmul_readvariableop_resource_0:	]è	I
5while_lstm_cell_10_matmul_1_readvariableop_resource_0:
ºè	C
4while_lstm_cell_10_biasadd_readvariableop_resource_0:	è	
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_10_matmul_readvariableop_resource:	]è	G
3while_lstm_cell_10_matmul_1_readvariableop_resource:
ºè	A
2while_lstm_cell_10_biasadd_readvariableop_resource:	è	¢)while/lstm_cell_10/BiasAdd/ReadVariableOp¢(while/lstm_cell_10/MatMul/ReadVariableOp¢*while/lstm_cell_10/MatMul_1/ReadVariableOpÃ
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
(while/lstm_cell_10/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_10_matmul_readvariableop_resource_0*
_output_shapes
:	]è	*
dtype02*
(while/lstm_cell_10/MatMul/ReadVariableOp×
while/lstm_cell_10/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_10/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè	2
while/lstm_cell_10/MatMulÐ
*while/lstm_cell_10/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_10_matmul_1_readvariableop_resource_0* 
_output_shapes
:
ºè	*
dtype02,
*while/lstm_cell_10/MatMul_1/ReadVariableOpÀ
while/lstm_cell_10/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_10/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè	2
while/lstm_cell_10/MatMul_1¸
while/lstm_cell_10/addAddV2#while/lstm_cell_10/MatMul:product:0%while/lstm_cell_10/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè	2
while/lstm_cell_10/addÈ
)while/lstm_cell_10/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_10_biasadd_readvariableop_resource_0*
_output_shapes	
:è	*
dtype02+
)while/lstm_cell_10/BiasAdd/ReadVariableOpÅ
while/lstm_cell_10/BiasAddBiasAddwhile/lstm_cell_10/add:z:01while/lstm_cell_10/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè	2
while/lstm_cell_10/BiasAdd
"while/lstm_cell_10/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_10/split/split_dim
while/lstm_cell_10/splitSplit+while/lstm_cell_10/split/split_dim:output:0#while/lstm_cell_10/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿº:ÿÿÿÿÿÿÿÿÿº:ÿÿÿÿÿÿÿÿÿº:ÿÿÿÿÿÿÿÿÿº*
	num_split2
while/lstm_cell_10/split
while/lstm_cell_10/SigmoidSigmoid!while/lstm_cell_10/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2
while/lstm_cell_10/Sigmoid
while/lstm_cell_10/Sigmoid_1Sigmoid!while/lstm_cell_10/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2
while/lstm_cell_10/Sigmoid_1¡
while/lstm_cell_10/mulMul while/lstm_cell_10/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2
while/lstm_cell_10/mul
while/lstm_cell_10/ReluRelu!while/lstm_cell_10/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2
while/lstm_cell_10/Reluµ
while/lstm_cell_10/mul_1Mulwhile/lstm_cell_10/Sigmoid:y:0%while/lstm_cell_10/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2
while/lstm_cell_10/mul_1ª
while/lstm_cell_10/add_1AddV2while/lstm_cell_10/mul:z:0while/lstm_cell_10/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2
while/lstm_cell_10/add_1
while/lstm_cell_10/Sigmoid_2Sigmoid!while/lstm_cell_10/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2
while/lstm_cell_10/Sigmoid_2
while/lstm_cell_10/Relu_1Reluwhile/lstm_cell_10/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2
while/lstm_cell_10/Relu_1¹
while/lstm_cell_10/mul_2Mul while/lstm_cell_10/Sigmoid_2:y:0'while/lstm_cell_10/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2
while/lstm_cell_10/mul_2à
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
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_10/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_10/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2
while/Identity_5Þ

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
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿº:ÿÿÿÿÿÿÿÿÿº: : : : : 2V
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
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº:

_output_shapes
: :

_output_shapes
: 
ã
Í
while_cond_19556076
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_19556076___redundant_placeholder06
2while_while_cond_19556076___redundant_placeholder16
2while_while_cond_19556076___redundant_placeholder26
2while_while_cond_19556076___redundant_placeholder3
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
B: : : : :ÿÿÿÿÿÿÿÿÿº:ÿÿÿÿÿÿÿÿÿº: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº:

_output_shapes
: :

_output_shapes
:
ß
Í
while_cond_19558084
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_19558084___redundant_placeholder06
2while_while_cond_19558084___redundant_placeholder16
2while_while_cond_19558084___redundant_placeholder26
2while_while_cond_19558084___redundant_placeholder3
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
@: : : : :ÿÿÿÿÿÿÿÿÿS:ÿÿÿÿÿÿÿÿÿS: ::::: 

_output_shapes
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
:ÿÿÿÿÿÿÿÿÿS:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS:

_output_shapes
: :

_output_shapes
:
ÓJ
Ô

lstm_11_while_body_19556880,
(lstm_11_while_lstm_11_while_loop_counter2
.lstm_11_while_lstm_11_while_maximum_iterations
lstm_11_while_placeholder
lstm_11_while_placeholder_1
lstm_11_while_placeholder_2
lstm_11_while_placeholder_3+
'lstm_11_while_lstm_11_strided_slice_1_0g
clstm_11_while_tensorarrayv2read_tensorlistgetitem_lstm_11_tensorarrayunstack_tensorlistfromtensor_0O
;lstm_11_while_lstm_cell_11_matmul_readvariableop_resource_0:
ºÌP
=lstm_11_while_lstm_cell_11_matmul_1_readvariableop_resource_0:	SÌK
<lstm_11_while_lstm_cell_11_biasadd_readvariableop_resource_0:	Ì
lstm_11_while_identity
lstm_11_while_identity_1
lstm_11_while_identity_2
lstm_11_while_identity_3
lstm_11_while_identity_4
lstm_11_while_identity_5)
%lstm_11_while_lstm_11_strided_slice_1e
alstm_11_while_tensorarrayv2read_tensorlistgetitem_lstm_11_tensorarrayunstack_tensorlistfromtensorM
9lstm_11_while_lstm_cell_11_matmul_readvariableop_resource:
ºÌN
;lstm_11_while_lstm_cell_11_matmul_1_readvariableop_resource:	SÌI
:lstm_11_while_lstm_cell_11_biasadd_readvariableop_resource:	Ì¢1lstm_11/while/lstm_cell_11/BiasAdd/ReadVariableOp¢0lstm_11/while/lstm_cell_11/MatMul/ReadVariableOp¢2lstm_11/while/lstm_cell_11/MatMul_1/ReadVariableOpÓ
?lstm_11/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ:  2A
?lstm_11/while/TensorArrayV2Read/TensorListGetItem/element_shape
1lstm_11/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemclstm_11_while_tensorarrayv2read_tensorlistgetitem_lstm_11_tensorarrayunstack_tensorlistfromtensor_0lstm_11_while_placeholderHlstm_11/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº*
element_dtype023
1lstm_11/while/TensorArrayV2Read/TensorListGetItemâ
0lstm_11/while/lstm_cell_11/MatMul/ReadVariableOpReadVariableOp;lstm_11_while_lstm_cell_11_matmul_readvariableop_resource_0* 
_output_shapes
:
ºÌ*
dtype022
0lstm_11/while/lstm_cell_11/MatMul/ReadVariableOp÷
!lstm_11/while/lstm_cell_11/MatMulMatMul8lstm_11/while/TensorArrayV2Read/TensorListGetItem:item:08lstm_11/while/lstm_cell_11/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÌ2#
!lstm_11/while/lstm_cell_11/MatMulç
2lstm_11/while/lstm_cell_11/MatMul_1/ReadVariableOpReadVariableOp=lstm_11_while_lstm_cell_11_matmul_1_readvariableop_resource_0*
_output_shapes
:	SÌ*
dtype024
2lstm_11/while/lstm_cell_11/MatMul_1/ReadVariableOpà
#lstm_11/while/lstm_cell_11/MatMul_1MatMullstm_11_while_placeholder_2:lstm_11/while/lstm_cell_11/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÌ2%
#lstm_11/while/lstm_cell_11/MatMul_1Ø
lstm_11/while/lstm_cell_11/addAddV2+lstm_11/while/lstm_cell_11/MatMul:product:0-lstm_11/while/lstm_cell_11/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÌ2 
lstm_11/while/lstm_cell_11/addà
1lstm_11/while/lstm_cell_11/BiasAdd/ReadVariableOpReadVariableOp<lstm_11_while_lstm_cell_11_biasadd_readvariableop_resource_0*
_output_shapes	
:Ì*
dtype023
1lstm_11/while/lstm_cell_11/BiasAdd/ReadVariableOpå
"lstm_11/while/lstm_cell_11/BiasAddBiasAdd"lstm_11/while/lstm_cell_11/add:z:09lstm_11/while/lstm_cell_11/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÌ2$
"lstm_11/while/lstm_cell_11/BiasAdd
*lstm_11/while/lstm_cell_11/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2,
*lstm_11/while/lstm_cell_11/split/split_dim«
 lstm_11/while/lstm_cell_11/splitSplit3lstm_11/while/lstm_cell_11/split/split_dim:output:0+lstm_11/while/lstm_cell_11/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿS:ÿÿÿÿÿÿÿÿÿS:ÿÿÿÿÿÿÿÿÿS:ÿÿÿÿÿÿÿÿÿS*
	num_split2"
 lstm_11/while/lstm_cell_11/split°
"lstm_11/while/lstm_cell_11/SigmoidSigmoid)lstm_11/while/lstm_cell_11/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2$
"lstm_11/while/lstm_cell_11/Sigmoid´
$lstm_11/while/lstm_cell_11/Sigmoid_1Sigmoid)lstm_11/while/lstm_cell_11/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2&
$lstm_11/while/lstm_cell_11/Sigmoid_1À
lstm_11/while/lstm_cell_11/mulMul(lstm_11/while/lstm_cell_11/Sigmoid_1:y:0lstm_11_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2 
lstm_11/while/lstm_cell_11/mul§
lstm_11/while/lstm_cell_11/ReluRelu)lstm_11/while/lstm_cell_11/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2!
lstm_11/while/lstm_cell_11/ReluÔ
 lstm_11/while/lstm_cell_11/mul_1Mul&lstm_11/while/lstm_cell_11/Sigmoid:y:0-lstm_11/while/lstm_cell_11/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2"
 lstm_11/while/lstm_cell_11/mul_1É
 lstm_11/while/lstm_cell_11/add_1AddV2"lstm_11/while/lstm_cell_11/mul:z:0$lstm_11/while/lstm_cell_11/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2"
 lstm_11/while/lstm_cell_11/add_1´
$lstm_11/while/lstm_cell_11/Sigmoid_2Sigmoid)lstm_11/while/lstm_cell_11/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2&
$lstm_11/while/lstm_cell_11/Sigmoid_2¦
!lstm_11/while/lstm_cell_11/Relu_1Relu$lstm_11/while/lstm_cell_11/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2#
!lstm_11/while/lstm_cell_11/Relu_1Ø
 lstm_11/while/lstm_cell_11/mul_2Mul(lstm_11/while/lstm_cell_11/Sigmoid_2:y:0/lstm_11/while/lstm_cell_11/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2"
 lstm_11/while/lstm_cell_11/mul_2
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
lstm_11/while/add/y
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
lstm_11/while/add_1/y
lstm_11/while/add_1AddV2(lstm_11_while_lstm_11_while_loop_counterlstm_11/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lstm_11/while/add_1
lstm_11/while/IdentityIdentitylstm_11/while/add_1:z:0^lstm_11/while/NoOp*
T0*
_output_shapes
: 2
lstm_11/while/Identity¦
lstm_11/while/Identity_1Identity.lstm_11_while_lstm_11_while_maximum_iterations^lstm_11/while/NoOp*
T0*
_output_shapes
: 2
lstm_11/while/Identity_1
lstm_11/while/Identity_2Identitylstm_11/while/add:z:0^lstm_11/while/NoOp*
T0*
_output_shapes
: 2
lstm_11/while/Identity_2º
lstm_11/while/Identity_3IdentityBlstm_11/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_11/while/NoOp*
T0*
_output_shapes
: 2
lstm_11/while/Identity_3­
lstm_11/while/Identity_4Identity$lstm_11/while/lstm_cell_11/mul_2:z:0^lstm_11/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2
lstm_11/while/Identity_4­
lstm_11/while/Identity_5Identity$lstm_11/while/lstm_cell_11/add_1:z:0^lstm_11/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2
lstm_11/while/Identity_5
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
9lstm_11_while_lstm_cell_11_matmul_readvariableop_resource;lstm_11_while_lstm_cell_11_matmul_readvariableop_resource_0"È
alstm_11_while_tensorarrayv2read_tensorlistgetitem_lstm_11_tensorarrayunstack_tensorlistfromtensorclstm_11_while_tensorarrayv2read_tensorlistgetitem_lstm_11_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿS:ÿÿÿÿÿÿÿÿÿS: : : : : 2f
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
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS:

_output_shapes
: :

_output_shapes
: 

f
H__inference_dropout_10_layer_call_and_return_conditional_losses_19555544

inputs

identity_1_
IdentityIdentityinputs*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2

Identityn

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿº:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿº
 
_user_specified_nameinputs
\

E__inference_lstm_10_layer_call_and_return_conditional_losses_19556161

inputs>
+lstm_cell_10_matmul_readvariableop_resource:	]è	A
-lstm_cell_10_matmul_1_readvariableop_resource:
ºè	;
,lstm_cell_10_biasadd_readvariableop_resource:	è	
identity¢#lstm_cell_10/BiasAdd/ReadVariableOp¢"lstm_cell_10/MatMul/ReadVariableOp¢$lstm_cell_10/MatMul_1/ReadVariableOp¢whileD
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
B :º2
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
B :º2
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
:ÿÿÿÿÿÿÿÿÿº2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :º2
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
B :º2
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
:ÿÿÿÿÿÿÿÿÿº2	
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
"lstm_cell_10/MatMul/ReadVariableOpReadVariableOp+lstm_cell_10_matmul_readvariableop_resource*
_output_shapes
:	]è	*
dtype02$
"lstm_cell_10/MatMul/ReadVariableOp­
lstm_cell_10/MatMulMatMulstrided_slice_2:output:0*lstm_cell_10/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè	2
lstm_cell_10/MatMul¼
$lstm_cell_10/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_10_matmul_1_readvariableop_resource* 
_output_shapes
:
ºè	*
dtype02&
$lstm_cell_10/MatMul_1/ReadVariableOp©
lstm_cell_10/MatMul_1MatMulzeros:output:0,lstm_cell_10/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè	2
lstm_cell_10/MatMul_1 
lstm_cell_10/addAddV2lstm_cell_10/MatMul:product:0lstm_cell_10/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè	2
lstm_cell_10/add´
#lstm_cell_10/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_10_biasadd_readvariableop_resource*
_output_shapes	
:è	*
dtype02%
#lstm_cell_10/BiasAdd/ReadVariableOp­
lstm_cell_10/BiasAddBiasAddlstm_cell_10/add:z:0+lstm_cell_10/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè	2
lstm_cell_10/BiasAdd~
lstm_cell_10/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_10/split/split_dim÷
lstm_cell_10/splitSplit%lstm_cell_10/split/split_dim:output:0lstm_cell_10/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿº:ÿÿÿÿÿÿÿÿÿº:ÿÿÿÿÿÿÿÿÿº:ÿÿÿÿÿÿÿÿÿº*
	num_split2
lstm_cell_10/split
lstm_cell_10/SigmoidSigmoidlstm_cell_10/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2
lstm_cell_10/Sigmoid
lstm_cell_10/Sigmoid_1Sigmoidlstm_cell_10/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2
lstm_cell_10/Sigmoid_1
lstm_cell_10/mulMullstm_cell_10/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2
lstm_cell_10/mul~
lstm_cell_10/ReluRelulstm_cell_10/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2
lstm_cell_10/Relu
lstm_cell_10/mul_1Mullstm_cell_10/Sigmoid:y:0lstm_cell_10/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2
lstm_cell_10/mul_1
lstm_cell_10/add_1AddV2lstm_cell_10/mul:z:0lstm_cell_10/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2
lstm_cell_10/add_1
lstm_cell_10/Sigmoid_2Sigmoidlstm_cell_10/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2
lstm_cell_10/Sigmoid_2}
lstm_cell_10/Relu_1Relulstm_cell_10/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2
lstm_cell_10/Relu_1¡
lstm_cell_10/mul_2Mullstm_cell_10/Sigmoid_2:y:0!lstm_cell_10/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2
lstm_cell_10/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ:  2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_10_matmul_readvariableop_resource-lstm_cell_10_matmul_1_readvariableop_resource,lstm_cell_10_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿº:ÿÿÿÿÿÿÿÿÿº: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_19556077*
condR
while_cond_19556076*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿº:ÿÿÿÿÿÿÿÿÿº: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ:  22
0TensorArrayV2Stack/TensorListStack/element_shapeé
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿº*
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
:ÿÿÿÿÿÿÿÿÿº*
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
:ÿÿÿÿÿÿÿÿÿº2
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
:ÿÿÿÿÿÿÿÿÿº2

IdentityÈ
NoOpNoOp$^lstm_cell_10/BiasAdd/ReadVariableOp#^lstm_cell_10/MatMul/ReadVariableOp%^lstm_cell_10/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ]: : : 2J
#lstm_cell_10/BiasAdd/ReadVariableOp#lstm_cell_10/BiasAdd/ReadVariableOp2H
"lstm_cell_10/MatMul/ReadVariableOp"lstm_cell_10/MatMul/ReadVariableOp2L
$lstm_cell_10/MatMul_1/ReadVariableOp$lstm_cell_10/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
 
_user_specified_nameinputs
Ð

í
lstm_11_while_cond_19556545,
(lstm_11_while_lstm_11_while_loop_counter2
.lstm_11_while_lstm_11_while_maximum_iterations
lstm_11_while_placeholder
lstm_11_while_placeholder_1
lstm_11_while_placeholder_2
lstm_11_while_placeholder_3.
*lstm_11_while_less_lstm_11_strided_slice_1F
Blstm_11_while_lstm_11_while_cond_19556545___redundant_placeholder0F
Blstm_11_while_lstm_11_while_cond_19556545___redundant_placeholder1F
Blstm_11_while_lstm_11_while_cond_19556545___redundant_placeholder2F
Blstm_11_while_lstm_11_while_cond_19556545___redundant_placeholder3
lstm_11_while_identity

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
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿS:ÿÿÿÿÿÿÿÿÿS: ::::: 

_output_shapes
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
:ÿÿÿÿÿÿÿÿÿS:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS:

_output_shapes
: :

_output_shapes
:
ð

J__inference_sequential_5_layer_call_and_return_conditional_losses_19556999

inputsF
3lstm_10_lstm_cell_10_matmul_readvariableop_resource:	]è	I
5lstm_10_lstm_cell_10_matmul_1_readvariableop_resource:
ºè	C
4lstm_10_lstm_cell_10_biasadd_readvariableop_resource:	è	G
3lstm_11_lstm_cell_11_matmul_readvariableop_resource:
ºÌH
5lstm_11_lstm_cell_11_matmul_1_readvariableop_resource:	SÌC
4lstm_11_lstm_cell_11_biasadd_readvariableop_resource:	Ì;
)dense_5_tensordot_readvariableop_resource:S5
'dense_5_biasadd_readvariableop_resource:
identity¢dense_5/BiasAdd/ReadVariableOp¢ dense_5/Tensordot/ReadVariableOp¢+lstm_10/lstm_cell_10/BiasAdd/ReadVariableOp¢*lstm_10/lstm_cell_10/MatMul/ReadVariableOp¢,lstm_10/lstm_cell_10/MatMul_1/ReadVariableOp¢lstm_10/while¢+lstm_11/lstm_cell_11/BiasAdd/ReadVariableOp¢*lstm_11/lstm_cell_11/MatMul/ReadVariableOp¢,lstm_11/lstm_cell_11/MatMul_1/ReadVariableOp¢lstm_11/whileT
lstm_10/ShapeShapeinputs*
T0*
_output_shapes
:2
lstm_10/Shape
lstm_10/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_10/strided_slice/stack
lstm_10/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_10/strided_slice/stack_1
lstm_10/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_10/strided_slice/stack_2
lstm_10/strided_sliceStridedSlicelstm_10/Shape:output:0$lstm_10/strided_slice/stack:output:0&lstm_10/strided_slice/stack_1:output:0&lstm_10/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_10/strided_slicem
lstm_10/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :º2
lstm_10/zeros/mul/y
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
B :è2
lstm_10/zeros/Less/y
lstm_10/zeros/LessLesslstm_10/zeros/mul:z:0lstm_10/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_10/zeros/Lesss
lstm_10/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :º2
lstm_10/zeros/packed/1£
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
lstm_10/zeros/Const
lstm_10/zerosFilllstm_10/zeros/packed:output:0lstm_10/zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2
lstm_10/zerosq
lstm_10/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :º2
lstm_10/zeros_1/mul/y
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
B :è2
lstm_10/zeros_1/Less/y
lstm_10/zeros_1/LessLesslstm_10/zeros_1/mul:z:0lstm_10/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_10/zeros_1/Lessw
lstm_10/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :º2
lstm_10/zeros_1/packed/1©
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
lstm_10/zeros_1/Const
lstm_10/zeros_1Filllstm_10/zeros_1/packed:output:0lstm_10/zeros_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2
lstm_10/zeros_1
lstm_10/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_10/transpose/perm
lstm_10/transpose	Transposeinputslstm_10/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]2
lstm_10/transposeg
lstm_10/Shape_1Shapelstm_10/transpose:y:0*
T0*
_output_shapes
:2
lstm_10/Shape_1
lstm_10/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_10/strided_slice_1/stack
lstm_10/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_10/strided_slice_1/stack_1
lstm_10/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_10/strided_slice_1/stack_2
lstm_10/strided_slice_1StridedSlicelstm_10/Shape_1:output:0&lstm_10/strided_slice_1/stack:output:0(lstm_10/strided_slice_1/stack_1:output:0(lstm_10/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_10/strided_slice_1
#lstm_10/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2%
#lstm_10/TensorArrayV2/element_shapeÒ
lstm_10/TensorArrayV2TensorListReserve,lstm_10/TensorArrayV2/element_shape:output:0 lstm_10/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_10/TensorArrayV2Ï
=lstm_10/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ]   2?
=lstm_10/TensorArrayUnstack/TensorListFromTensor/element_shape
/lstm_10/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_10/transpose:y:0Flstm_10/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type021
/lstm_10/TensorArrayUnstack/TensorListFromTensor
lstm_10/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_10/strided_slice_2/stack
lstm_10/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_10/strided_slice_2/stack_1
lstm_10/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_10/strided_slice_2/stack_2¬
lstm_10/strided_slice_2StridedSlicelstm_10/transpose:y:0&lstm_10/strided_slice_2/stack:output:0(lstm_10/strided_slice_2/stack_1:output:0(lstm_10/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]*
shrink_axis_mask2
lstm_10/strided_slice_2Í
*lstm_10/lstm_cell_10/MatMul/ReadVariableOpReadVariableOp3lstm_10_lstm_cell_10_matmul_readvariableop_resource*
_output_shapes
:	]è	*
dtype02,
*lstm_10/lstm_cell_10/MatMul/ReadVariableOpÍ
lstm_10/lstm_cell_10/MatMulMatMul lstm_10/strided_slice_2:output:02lstm_10/lstm_cell_10/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè	2
lstm_10/lstm_cell_10/MatMulÔ
,lstm_10/lstm_cell_10/MatMul_1/ReadVariableOpReadVariableOp5lstm_10_lstm_cell_10_matmul_1_readvariableop_resource* 
_output_shapes
:
ºè	*
dtype02.
,lstm_10/lstm_cell_10/MatMul_1/ReadVariableOpÉ
lstm_10/lstm_cell_10/MatMul_1MatMullstm_10/zeros:output:04lstm_10/lstm_cell_10/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè	2
lstm_10/lstm_cell_10/MatMul_1À
lstm_10/lstm_cell_10/addAddV2%lstm_10/lstm_cell_10/MatMul:product:0'lstm_10/lstm_cell_10/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè	2
lstm_10/lstm_cell_10/addÌ
+lstm_10/lstm_cell_10/BiasAdd/ReadVariableOpReadVariableOp4lstm_10_lstm_cell_10_biasadd_readvariableop_resource*
_output_shapes	
:è	*
dtype02-
+lstm_10/lstm_cell_10/BiasAdd/ReadVariableOpÍ
lstm_10/lstm_cell_10/BiasAddBiasAddlstm_10/lstm_cell_10/add:z:03lstm_10/lstm_cell_10/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè	2
lstm_10/lstm_cell_10/BiasAdd
$lstm_10/lstm_cell_10/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2&
$lstm_10/lstm_cell_10/split/split_dim
lstm_10/lstm_cell_10/splitSplit-lstm_10/lstm_cell_10/split/split_dim:output:0%lstm_10/lstm_cell_10/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿº:ÿÿÿÿÿÿÿÿÿº:ÿÿÿÿÿÿÿÿÿº:ÿÿÿÿÿÿÿÿÿº*
	num_split2
lstm_10/lstm_cell_10/split
lstm_10/lstm_cell_10/SigmoidSigmoid#lstm_10/lstm_cell_10/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2
lstm_10/lstm_cell_10/Sigmoid£
lstm_10/lstm_cell_10/Sigmoid_1Sigmoid#lstm_10/lstm_cell_10/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2 
lstm_10/lstm_cell_10/Sigmoid_1¬
lstm_10/lstm_cell_10/mulMul"lstm_10/lstm_cell_10/Sigmoid_1:y:0lstm_10/zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2
lstm_10/lstm_cell_10/mul
lstm_10/lstm_cell_10/ReluRelu#lstm_10/lstm_cell_10/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2
lstm_10/lstm_cell_10/Relu½
lstm_10/lstm_cell_10/mul_1Mul lstm_10/lstm_cell_10/Sigmoid:y:0'lstm_10/lstm_cell_10/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2
lstm_10/lstm_cell_10/mul_1²
lstm_10/lstm_cell_10/add_1AddV2lstm_10/lstm_cell_10/mul:z:0lstm_10/lstm_cell_10/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2
lstm_10/lstm_cell_10/add_1£
lstm_10/lstm_cell_10/Sigmoid_2Sigmoid#lstm_10/lstm_cell_10/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2 
lstm_10/lstm_cell_10/Sigmoid_2
lstm_10/lstm_cell_10/Relu_1Relulstm_10/lstm_cell_10/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2
lstm_10/lstm_cell_10/Relu_1Á
lstm_10/lstm_cell_10/mul_2Mul"lstm_10/lstm_cell_10/Sigmoid_2:y:0)lstm_10/lstm_cell_10/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2
lstm_10/lstm_cell_10/mul_2
%lstm_10/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ:  2'
%lstm_10/TensorArrayV2_1/element_shapeØ
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
lstm_10/time
 lstm_10/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2"
 lstm_10/while/maximum_iterationsz
lstm_10/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_10/while/loop_counter
lstm_10/whileWhile#lstm_10/while/loop_counter:output:0)lstm_10/while/maximum_iterations:output:0lstm_10/time:output:0 lstm_10/TensorArrayV2_1:handle:0lstm_10/zeros:output:0lstm_10/zeros_1:output:0 lstm_10/strided_slice_1:output:0?lstm_10/TensorArrayUnstack/TensorListFromTensor:output_handle:03lstm_10_lstm_cell_10_matmul_readvariableop_resource5lstm_10_lstm_cell_10_matmul_1_readvariableop_resource4lstm_10_lstm_cell_10_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿº:ÿÿÿÿÿÿÿÿÿº: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *'
bodyR
lstm_10_while_body_19556725*'
condR
lstm_10_while_cond_19556724*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿº:ÿÿÿÿÿÿÿÿÿº: : : : : *
parallel_iterations 2
lstm_10/whileÅ
8lstm_10/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ:  2:
8lstm_10/TensorArrayV2Stack/TensorListStack/element_shape
*lstm_10/TensorArrayV2Stack/TensorListStackTensorListStacklstm_10/while:output:3Alstm_10/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿº*
element_dtype02,
*lstm_10/TensorArrayV2Stack/TensorListStack
lstm_10/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
lstm_10/strided_slice_3/stack
lstm_10/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2!
lstm_10/strided_slice_3/stack_1
lstm_10/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_10/strided_slice_3/stack_2Ë
lstm_10/strided_slice_3StridedSlice3lstm_10/TensorArrayV2Stack/TensorListStack:tensor:0&lstm_10/strided_slice_3/stack:output:0(lstm_10/strided_slice_3/stack_1:output:0(lstm_10/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº*
shrink_axis_mask2
lstm_10/strided_slice_3
lstm_10/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_10/transpose_1/permÆ
lstm_10/transpose_1	Transpose3lstm_10/TensorArrayV2Stack/TensorListStack:tensor:0!lstm_10/transpose_1/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2
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
 *«ªª?2
dropout_10/dropout/Constª
dropout_10/dropout/MulMullstm_10/transpose_1:y:0!dropout_10/dropout/Const:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2
dropout_10/dropout/Mul{
dropout_10/dropout/ShapeShapelstm_10/transpose_1:y:0*
T0*
_output_shapes
:2
dropout_10/dropout/ShapeÚ
/dropout_10/dropout/random_uniform/RandomUniformRandomUniform!dropout_10/dropout/Shape:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿº*
dtype021
/dropout_10/dropout/random_uniform/RandomUniform
!dropout_10/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  >2#
!dropout_10/dropout/GreaterEqual/yï
dropout_10/dropout/GreaterEqualGreaterEqual8dropout_10/dropout/random_uniform/RandomUniform:output:0*dropout_10/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2!
dropout_10/dropout/GreaterEqual¥
dropout_10/dropout/CastCast#dropout_10/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2
dropout_10/dropout/Cast«
dropout_10/dropout/Mul_1Muldropout_10/dropout/Mul:z:0dropout_10/dropout/Cast:y:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2
dropout_10/dropout/Mul_1j
lstm_11/ShapeShapedropout_10/dropout/Mul_1:z:0*
T0*
_output_shapes
:2
lstm_11/Shape
lstm_11/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_11/strided_slice/stack
lstm_11/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_11/strided_slice/stack_1
lstm_11/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_11/strided_slice/stack_2
lstm_11/strided_sliceStridedSlicelstm_11/Shape:output:0$lstm_11/strided_slice/stack:output:0&lstm_11/strided_slice/stack_1:output:0&lstm_11/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_11/strided_slicel
lstm_11/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :S2
lstm_11/zeros/mul/y
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
B :è2
lstm_11/zeros/Less/y
lstm_11/zeros/LessLesslstm_11/zeros/mul:z:0lstm_11/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_11/zeros/Lessr
lstm_11/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :S2
lstm_11/zeros/packed/1£
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
lstm_11/zeros/Const
lstm_11/zerosFilllstm_11/zeros/packed:output:0lstm_11/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2
lstm_11/zerosp
lstm_11/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :S2
lstm_11/zeros_1/mul/y
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
B :è2
lstm_11/zeros_1/Less/y
lstm_11/zeros_1/LessLesslstm_11/zeros_1/mul:z:0lstm_11/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_11/zeros_1/Lessv
lstm_11/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :S2
lstm_11/zeros_1/packed/1©
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
lstm_11/zeros_1/Const
lstm_11/zeros_1Filllstm_11/zeros_1/packed:output:0lstm_11/zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2
lstm_11/zeros_1
lstm_11/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_11/transpose/perm©
lstm_11/transpose	Transposedropout_10/dropout/Mul_1:z:0lstm_11/transpose/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2
lstm_11/transposeg
lstm_11/Shape_1Shapelstm_11/transpose:y:0*
T0*
_output_shapes
:2
lstm_11/Shape_1
lstm_11/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_11/strided_slice_1/stack
lstm_11/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_11/strided_slice_1/stack_1
lstm_11/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_11/strided_slice_1/stack_2
lstm_11/strided_slice_1StridedSlicelstm_11/Shape_1:output:0&lstm_11/strided_slice_1/stack:output:0(lstm_11/strided_slice_1/stack_1:output:0(lstm_11/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_11/strided_slice_1
#lstm_11/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2%
#lstm_11/TensorArrayV2/element_shapeÒ
lstm_11/TensorArrayV2TensorListReserve,lstm_11/TensorArrayV2/element_shape:output:0 lstm_11/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_11/TensorArrayV2Ï
=lstm_11/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ:  2?
=lstm_11/TensorArrayUnstack/TensorListFromTensor/element_shape
/lstm_11/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_11/transpose:y:0Flstm_11/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type021
/lstm_11/TensorArrayUnstack/TensorListFromTensor
lstm_11/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_11/strided_slice_2/stack
lstm_11/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_11/strided_slice_2/stack_1
lstm_11/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_11/strided_slice_2/stack_2­
lstm_11/strided_slice_2StridedSlicelstm_11/transpose:y:0&lstm_11/strided_slice_2/stack:output:0(lstm_11/strided_slice_2/stack_1:output:0(lstm_11/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº*
shrink_axis_mask2
lstm_11/strided_slice_2Î
*lstm_11/lstm_cell_11/MatMul/ReadVariableOpReadVariableOp3lstm_11_lstm_cell_11_matmul_readvariableop_resource* 
_output_shapes
:
ºÌ*
dtype02,
*lstm_11/lstm_cell_11/MatMul/ReadVariableOpÍ
lstm_11/lstm_cell_11/MatMulMatMul lstm_11/strided_slice_2:output:02lstm_11/lstm_cell_11/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÌ2
lstm_11/lstm_cell_11/MatMulÓ
,lstm_11/lstm_cell_11/MatMul_1/ReadVariableOpReadVariableOp5lstm_11_lstm_cell_11_matmul_1_readvariableop_resource*
_output_shapes
:	SÌ*
dtype02.
,lstm_11/lstm_cell_11/MatMul_1/ReadVariableOpÉ
lstm_11/lstm_cell_11/MatMul_1MatMullstm_11/zeros:output:04lstm_11/lstm_cell_11/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÌ2
lstm_11/lstm_cell_11/MatMul_1À
lstm_11/lstm_cell_11/addAddV2%lstm_11/lstm_cell_11/MatMul:product:0'lstm_11/lstm_cell_11/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÌ2
lstm_11/lstm_cell_11/addÌ
+lstm_11/lstm_cell_11/BiasAdd/ReadVariableOpReadVariableOp4lstm_11_lstm_cell_11_biasadd_readvariableop_resource*
_output_shapes	
:Ì*
dtype02-
+lstm_11/lstm_cell_11/BiasAdd/ReadVariableOpÍ
lstm_11/lstm_cell_11/BiasAddBiasAddlstm_11/lstm_cell_11/add:z:03lstm_11/lstm_cell_11/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÌ2
lstm_11/lstm_cell_11/BiasAdd
$lstm_11/lstm_cell_11/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2&
$lstm_11/lstm_cell_11/split/split_dim
lstm_11/lstm_cell_11/splitSplit-lstm_11/lstm_cell_11/split/split_dim:output:0%lstm_11/lstm_cell_11/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿS:ÿÿÿÿÿÿÿÿÿS:ÿÿÿÿÿÿÿÿÿS:ÿÿÿÿÿÿÿÿÿS*
	num_split2
lstm_11/lstm_cell_11/split
lstm_11/lstm_cell_11/SigmoidSigmoid#lstm_11/lstm_cell_11/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2
lstm_11/lstm_cell_11/Sigmoid¢
lstm_11/lstm_cell_11/Sigmoid_1Sigmoid#lstm_11/lstm_cell_11/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2 
lstm_11/lstm_cell_11/Sigmoid_1«
lstm_11/lstm_cell_11/mulMul"lstm_11/lstm_cell_11/Sigmoid_1:y:0lstm_11/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2
lstm_11/lstm_cell_11/mul
lstm_11/lstm_cell_11/ReluRelu#lstm_11/lstm_cell_11/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2
lstm_11/lstm_cell_11/Relu¼
lstm_11/lstm_cell_11/mul_1Mul lstm_11/lstm_cell_11/Sigmoid:y:0'lstm_11/lstm_cell_11/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2
lstm_11/lstm_cell_11/mul_1±
lstm_11/lstm_cell_11/add_1AddV2lstm_11/lstm_cell_11/mul:z:0lstm_11/lstm_cell_11/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2
lstm_11/lstm_cell_11/add_1¢
lstm_11/lstm_cell_11/Sigmoid_2Sigmoid#lstm_11/lstm_cell_11/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2 
lstm_11/lstm_cell_11/Sigmoid_2
lstm_11/lstm_cell_11/Relu_1Relulstm_11/lstm_cell_11/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2
lstm_11/lstm_cell_11/Relu_1À
lstm_11/lstm_cell_11/mul_2Mul"lstm_11/lstm_cell_11/Sigmoid_2:y:0)lstm_11/lstm_cell_11/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2
lstm_11/lstm_cell_11/mul_2
%lstm_11/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿS   2'
%lstm_11/TensorArrayV2_1/element_shapeØ
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
lstm_11/time
 lstm_11/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2"
 lstm_11/while/maximum_iterationsz
lstm_11/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_11/while/loop_counter
lstm_11/whileWhile#lstm_11/while/loop_counter:output:0)lstm_11/while/maximum_iterations:output:0lstm_11/time:output:0 lstm_11/TensorArrayV2_1:handle:0lstm_11/zeros:output:0lstm_11/zeros_1:output:0 lstm_11/strided_slice_1:output:0?lstm_11/TensorArrayUnstack/TensorListFromTensor:output_handle:03lstm_11_lstm_cell_11_matmul_readvariableop_resource5lstm_11_lstm_cell_11_matmul_1_readvariableop_resource4lstm_11_lstm_cell_11_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿS:ÿÿÿÿÿÿÿÿÿS: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *'
bodyR
lstm_11_while_body_19556880*'
condR
lstm_11_while_cond_19556879*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿS:ÿÿÿÿÿÿÿÿÿS: : : : : *
parallel_iterations 2
lstm_11/whileÅ
8lstm_11/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿS   2:
8lstm_11/TensorArrayV2Stack/TensorListStack/element_shape
*lstm_11/TensorArrayV2Stack/TensorListStackTensorListStacklstm_11/while:output:3Alstm_11/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿS*
element_dtype02,
*lstm_11/TensorArrayV2Stack/TensorListStack
lstm_11/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
lstm_11/strided_slice_3/stack
lstm_11/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2!
lstm_11/strided_slice_3/stack_1
lstm_11/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_11/strided_slice_3/stack_2Ê
lstm_11/strided_slice_3StridedSlice3lstm_11/TensorArrayV2Stack/TensorListStack:tensor:0&lstm_11/strided_slice_3/stack:output:0(lstm_11/strided_slice_3/stack_1:output:0(lstm_11/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS*
shrink_axis_mask2
lstm_11/strided_slice_3
lstm_11/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_11/transpose_1/permÅ
lstm_11/transpose_1	Transpose3lstm_11/TensorArrayV2Stack/TensorListStack:tensor:0!lstm_11/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2
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
 *   ?2
dropout_11/dropout/Const©
dropout_11/dropout/MulMullstm_11/transpose_1:y:0!dropout_11/dropout/Const:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2
dropout_11/dropout/Mul{
dropout_11/dropout/ShapeShapelstm_11/transpose_1:y:0*
T0*
_output_shapes
:2
dropout_11/dropout/ShapeÙ
/dropout_11/dropout/random_uniform/RandomUniformRandomUniform!dropout_11/dropout/Shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿS*
dtype021
/dropout_11/dropout/random_uniform/RandomUniform
!dropout_11/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>2#
!dropout_11/dropout/GreaterEqual/yî
dropout_11/dropout/GreaterEqualGreaterEqual8dropout_11/dropout/random_uniform/RandomUniform:output:0*dropout_11/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2!
dropout_11/dropout/GreaterEqual¤
dropout_11/dropout/CastCast#dropout_11/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2
dropout_11/dropout/Castª
dropout_11/dropout/Mul_1Muldropout_11/dropout/Mul:z:0dropout_11/dropout/Cast:y:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2
dropout_11/dropout/Mul_1®
 dense_5/Tensordot/ReadVariableOpReadVariableOp)dense_5_tensordot_readvariableop_resource*
_output_shapes

:S*
dtype02"
 dense_5/Tensordot/ReadVariableOpz
dense_5/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_5/Tensordot/axes
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
dense_5/Tensordot/Shape
dense_5/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_5/Tensordot/GatherV2/axisù
dense_5/Tensordot/GatherV2GatherV2 dense_5/Tensordot/Shape:output:0dense_5/Tensordot/free:output:0(dense_5/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_5/Tensordot/GatherV2
!dense_5/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_5/Tensordot/GatherV2_1/axisÿ
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
dense_5/Tensordot/Const 
dense_5/Tensordot/ProdProd#dense_5/Tensordot/GatherV2:output:0 dense_5/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_5/Tensordot/Prod
dense_5/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_5/Tensordot/Const_1¨
dense_5/Tensordot/Prod_1Prod%dense_5/Tensordot/GatherV2_1:output:0"dense_5/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_5/Tensordot/Prod_1
dense_5/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense_5/Tensordot/concat/axisØ
dense_5/Tensordot/concatConcatV2dense_5/Tensordot/free:output:0dense_5/Tensordot/axes:output:0&dense_5/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_5/Tensordot/concat¬
dense_5/Tensordot/stackPackdense_5/Tensordot/Prod:output:0!dense_5/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_5/Tensordot/stack¾
dense_5/Tensordot/transpose	Transposedropout_11/dropout/Mul_1:z:0!dense_5/Tensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2
dense_5/Tensordot/transpose¿
dense_5/Tensordot/ReshapeReshapedense_5/Tensordot/transpose:y:0 dense_5/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
dense_5/Tensordot/Reshape¾
dense_5/Tensordot/MatMulMatMul"dense_5/Tensordot/Reshape:output:0(dense_5/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_5/Tensordot/MatMul
dense_5/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
dense_5/Tensordot/Const_2
dense_5/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_5/Tensordot/concat_1/axiså
dense_5/Tensordot/concat_1ConcatV2#dense_5/Tensordot/GatherV2:output:0"dense_5/Tensordot/Const_2:output:0(dense_5/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_5/Tensordot/concat_1°
dense_5/TensordotReshape"dense_5/Tensordot/MatMul:product:0#dense_5/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_5/Tensordot¤
dense_5/BiasAdd/ReadVariableOpReadVariableOp'dense_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_5/BiasAdd/ReadVariableOp§
dense_5/BiasAddBiasAdddense_5/Tensordot:output:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_5/BiasAdd}
dense_5/SoftmaxSoftmaxdense_5/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_5/Softmaxx
IdentityIdentitydense_5/Softmax:softmax:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

IdentityÆ
NoOpNoOp^dense_5/BiasAdd/ReadVariableOp!^dense_5/Tensordot/ReadVariableOp,^lstm_10/lstm_cell_10/BiasAdd/ReadVariableOp+^lstm_10/lstm_cell_10/MatMul/ReadVariableOp-^lstm_10/lstm_cell_10/MatMul_1/ReadVariableOp^lstm_10/while,^lstm_11/lstm_cell_11/BiasAdd/ReadVariableOp+^lstm_11/lstm_cell_11/MatMul/ReadVariableOp-^lstm_11/lstm_cell_11/MatMul_1/ReadVariableOp^lstm_11/while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ]: : : : : : : : 2@
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
:ÿÿÿÿÿÿÿÿÿ]
 
_user_specified_nameinputs

ë
J__inference_sequential_5_layer_call_and_return_conditional_losses_19555749

inputs#
lstm_10_19555532:	]è	$
lstm_10_19555534:
ºè	
lstm_10_19555536:	è	$
lstm_11_19555697:
ºÌ#
lstm_11_19555699:	SÌ
lstm_11_19555701:	Ì"
dense_5_19555743:S
dense_5_19555745:
identity¢dense_5/StatefulPartitionedCall¢lstm_10/StatefulPartitionedCall¢lstm_11/StatefulPartitionedCall®
lstm_10/StatefulPartitionedCallStatefulPartitionedCallinputslstm_10_19555532lstm_10_19555534lstm_10_19555536*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿº*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_lstm_10_layer_call_and_return_conditional_losses_195555312!
lstm_10/StatefulPartitionedCall
dropout_10/PartitionedCallPartitionedCall(lstm_10/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿº* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_dropout_10_layer_call_and_return_conditional_losses_195555442
dropout_10/PartitionedCallÊ
lstm_11/StatefulPartitionedCallStatefulPartitionedCall#dropout_10/PartitionedCall:output:0lstm_11_19555697lstm_11_19555699lstm_11_19555701*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿS*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_lstm_11_layer_call_and_return_conditional_losses_195556962!
lstm_11/StatefulPartitionedCall
dropout_11/PartitionedCallPartitionedCall(lstm_11/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿS* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_dropout_11_layer_call_and_return_conditional_losses_195557092
dropout_11/PartitionedCall¶
dense_5/StatefulPartitionedCallStatefulPartitionedCall#dropout_11/PartitionedCall:output:0dense_5_19555743dense_5_19555745*
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
E__inference_dense_5_layer_call_and_return_conditional_losses_195557422!
dense_5/StatefulPartitionedCall
IdentityIdentity(dense_5/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity´
NoOpNoOp ^dense_5/StatefulPartitionedCall ^lstm_10/StatefulPartitionedCall ^lstm_11/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ]: : : : : : : : 2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2B
lstm_10/StatefulPartitionedCalllstm_10/StatefulPartitionedCall2B
lstm_11/StatefulPartitionedCalllstm_11/StatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
 
_user_specified_nameinputs
?
Ô
while_body_19558085
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
3while_lstm_cell_11_matmul_readvariableop_resource_0:
ºÌH
5while_lstm_cell_11_matmul_1_readvariableop_resource_0:	SÌC
4while_lstm_cell_11_biasadd_readvariableop_resource_0:	Ì
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
1while_lstm_cell_11_matmul_readvariableop_resource:
ºÌF
3while_lstm_cell_11_matmul_1_readvariableop_resource:	SÌA
2while_lstm_cell_11_biasadd_readvariableop_resource:	Ì¢)while/lstm_cell_11/BiasAdd/ReadVariableOp¢(while/lstm_cell_11/MatMul/ReadVariableOp¢*while/lstm_cell_11/MatMul_1/ReadVariableOpÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ:  29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÔ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemÊ
(while/lstm_cell_11/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_11_matmul_readvariableop_resource_0* 
_output_shapes
:
ºÌ*
dtype02*
(while/lstm_cell_11/MatMul/ReadVariableOp×
while/lstm_cell_11/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_11/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÌ2
while/lstm_cell_11/MatMulÏ
*while/lstm_cell_11/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_11_matmul_1_readvariableop_resource_0*
_output_shapes
:	SÌ*
dtype02,
*while/lstm_cell_11/MatMul_1/ReadVariableOpÀ
while/lstm_cell_11/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_11/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÌ2
while/lstm_cell_11/MatMul_1¸
while/lstm_cell_11/addAddV2#while/lstm_cell_11/MatMul:product:0%while/lstm_cell_11/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÌ2
while/lstm_cell_11/addÈ
)while/lstm_cell_11/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_11_biasadd_readvariableop_resource_0*
_output_shapes	
:Ì*
dtype02+
)while/lstm_cell_11/BiasAdd/ReadVariableOpÅ
while/lstm_cell_11/BiasAddBiasAddwhile/lstm_cell_11/add:z:01while/lstm_cell_11/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÌ2
while/lstm_cell_11/BiasAdd
"while/lstm_cell_11/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_11/split/split_dim
while/lstm_cell_11/splitSplit+while/lstm_cell_11/split/split_dim:output:0#while/lstm_cell_11/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿS:ÿÿÿÿÿÿÿÿÿS:ÿÿÿÿÿÿÿÿÿS:ÿÿÿÿÿÿÿÿÿS*
	num_split2
while/lstm_cell_11/split
while/lstm_cell_11/SigmoidSigmoid!while/lstm_cell_11/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2
while/lstm_cell_11/Sigmoid
while/lstm_cell_11/Sigmoid_1Sigmoid!while/lstm_cell_11/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2
while/lstm_cell_11/Sigmoid_1 
while/lstm_cell_11/mulMul while/lstm_cell_11/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2
while/lstm_cell_11/mul
while/lstm_cell_11/ReluRelu!while/lstm_cell_11/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2
while/lstm_cell_11/Relu´
while/lstm_cell_11/mul_1Mulwhile/lstm_cell_11/Sigmoid:y:0%while/lstm_cell_11/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2
while/lstm_cell_11/mul_1©
while/lstm_cell_11/add_1AddV2while/lstm_cell_11/mul:z:0while/lstm_cell_11/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2
while/lstm_cell_11/add_1
while/lstm_cell_11/Sigmoid_2Sigmoid!while/lstm_cell_11/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2
while/lstm_cell_11/Sigmoid_2
while/lstm_cell_11/Relu_1Reluwhile/lstm_cell_11/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2
while/lstm_cell_11/Relu_1¸
while/lstm_cell_11/mul_2Mul while/lstm_cell_11/Sigmoid_2:y:0'while/lstm_cell_11/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2
while/lstm_cell_11/mul_2à
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
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_11/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_11/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2
while/Identity_5Þ

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
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿS:ÿÿÿÿÿÿÿÿÿS: : : : : 2V
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
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS:

_output_shapes
: :

_output_shapes
: 
Û
ñ
(sequential_5_lstm_10_while_cond_19553853F
Bsequential_5_lstm_10_while_sequential_5_lstm_10_while_loop_counterL
Hsequential_5_lstm_10_while_sequential_5_lstm_10_while_maximum_iterations*
&sequential_5_lstm_10_while_placeholder,
(sequential_5_lstm_10_while_placeholder_1,
(sequential_5_lstm_10_while_placeholder_2,
(sequential_5_lstm_10_while_placeholder_3H
Dsequential_5_lstm_10_while_less_sequential_5_lstm_10_strided_slice_1`
\sequential_5_lstm_10_while_sequential_5_lstm_10_while_cond_19553853___redundant_placeholder0`
\sequential_5_lstm_10_while_sequential_5_lstm_10_while_cond_19553853___redundant_placeholder1`
\sequential_5_lstm_10_while_sequential_5_lstm_10_while_cond_19553853___redundant_placeholder2`
\sequential_5_lstm_10_while_sequential_5_lstm_10_while_cond_19553853___redundant_placeholder3'
#sequential_5_lstm_10_while_identity
Ù
sequential_5/lstm_10/while/LessLess&sequential_5_lstm_10_while_placeholderDsequential_5_lstm_10_while_less_sequential_5_lstm_10_strided_slice_1*
T0*
_output_shapes
: 2!
sequential_5/lstm_10/while/Less
#sequential_5/lstm_10/while/IdentityIdentity#sequential_5/lstm_10/while/Less:z:0*
T0
*
_output_shapes
: 2%
#sequential_5/lstm_10/while/Identity"S
#sequential_5_lstm_10_while_identity,sequential_5/lstm_10/while/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :ÿÿÿÿÿÿÿÿÿº:ÿÿÿÿÿÿÿÿÿº: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº:

_output_shapes
: :

_output_shapes
:
åJ
Ô

lstm_10_while_body_19556725,
(lstm_10_while_lstm_10_while_loop_counter2
.lstm_10_while_lstm_10_while_maximum_iterations
lstm_10_while_placeholder
lstm_10_while_placeholder_1
lstm_10_while_placeholder_2
lstm_10_while_placeholder_3+
'lstm_10_while_lstm_10_strided_slice_1_0g
clstm_10_while_tensorarrayv2read_tensorlistgetitem_lstm_10_tensorarrayunstack_tensorlistfromtensor_0N
;lstm_10_while_lstm_cell_10_matmul_readvariableop_resource_0:	]è	Q
=lstm_10_while_lstm_cell_10_matmul_1_readvariableop_resource_0:
ºè	K
<lstm_10_while_lstm_cell_10_biasadd_readvariableop_resource_0:	è	
lstm_10_while_identity
lstm_10_while_identity_1
lstm_10_while_identity_2
lstm_10_while_identity_3
lstm_10_while_identity_4
lstm_10_while_identity_5)
%lstm_10_while_lstm_10_strided_slice_1e
alstm_10_while_tensorarrayv2read_tensorlistgetitem_lstm_10_tensorarrayunstack_tensorlistfromtensorL
9lstm_10_while_lstm_cell_10_matmul_readvariableop_resource:	]è	O
;lstm_10_while_lstm_cell_10_matmul_1_readvariableop_resource:
ºè	I
:lstm_10_while_lstm_cell_10_biasadd_readvariableop_resource:	è	¢1lstm_10/while/lstm_cell_10/BiasAdd/ReadVariableOp¢0lstm_10/while/lstm_cell_10/MatMul/ReadVariableOp¢2lstm_10/while/lstm_cell_10/MatMul_1/ReadVariableOpÓ
?lstm_10/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ]   2A
?lstm_10/while/TensorArrayV2Read/TensorListGetItem/element_shape
1lstm_10/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemclstm_10_while_tensorarrayv2read_tensorlistgetitem_lstm_10_tensorarrayunstack_tensorlistfromtensor_0lstm_10_while_placeholderHlstm_10/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]*
element_dtype023
1lstm_10/while/TensorArrayV2Read/TensorListGetItemá
0lstm_10/while/lstm_cell_10/MatMul/ReadVariableOpReadVariableOp;lstm_10_while_lstm_cell_10_matmul_readvariableop_resource_0*
_output_shapes
:	]è	*
dtype022
0lstm_10/while/lstm_cell_10/MatMul/ReadVariableOp÷
!lstm_10/while/lstm_cell_10/MatMulMatMul8lstm_10/while/TensorArrayV2Read/TensorListGetItem:item:08lstm_10/while/lstm_cell_10/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè	2#
!lstm_10/while/lstm_cell_10/MatMulè
2lstm_10/while/lstm_cell_10/MatMul_1/ReadVariableOpReadVariableOp=lstm_10_while_lstm_cell_10_matmul_1_readvariableop_resource_0* 
_output_shapes
:
ºè	*
dtype024
2lstm_10/while/lstm_cell_10/MatMul_1/ReadVariableOpà
#lstm_10/while/lstm_cell_10/MatMul_1MatMullstm_10_while_placeholder_2:lstm_10/while/lstm_cell_10/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè	2%
#lstm_10/while/lstm_cell_10/MatMul_1Ø
lstm_10/while/lstm_cell_10/addAddV2+lstm_10/while/lstm_cell_10/MatMul:product:0-lstm_10/while/lstm_cell_10/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè	2 
lstm_10/while/lstm_cell_10/addà
1lstm_10/while/lstm_cell_10/BiasAdd/ReadVariableOpReadVariableOp<lstm_10_while_lstm_cell_10_biasadd_readvariableop_resource_0*
_output_shapes	
:è	*
dtype023
1lstm_10/while/lstm_cell_10/BiasAdd/ReadVariableOpå
"lstm_10/while/lstm_cell_10/BiasAddBiasAdd"lstm_10/while/lstm_cell_10/add:z:09lstm_10/while/lstm_cell_10/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè	2$
"lstm_10/while/lstm_cell_10/BiasAdd
*lstm_10/while/lstm_cell_10/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2,
*lstm_10/while/lstm_cell_10/split/split_dim¯
 lstm_10/while/lstm_cell_10/splitSplit3lstm_10/while/lstm_cell_10/split/split_dim:output:0+lstm_10/while/lstm_cell_10/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿº:ÿÿÿÿÿÿÿÿÿº:ÿÿÿÿÿÿÿÿÿº:ÿÿÿÿÿÿÿÿÿº*
	num_split2"
 lstm_10/while/lstm_cell_10/split±
"lstm_10/while/lstm_cell_10/SigmoidSigmoid)lstm_10/while/lstm_cell_10/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2$
"lstm_10/while/lstm_cell_10/Sigmoidµ
$lstm_10/while/lstm_cell_10/Sigmoid_1Sigmoid)lstm_10/while/lstm_cell_10/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2&
$lstm_10/while/lstm_cell_10/Sigmoid_1Á
lstm_10/while/lstm_cell_10/mulMul(lstm_10/while/lstm_cell_10/Sigmoid_1:y:0lstm_10_while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2 
lstm_10/while/lstm_cell_10/mul¨
lstm_10/while/lstm_cell_10/ReluRelu)lstm_10/while/lstm_cell_10/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2!
lstm_10/while/lstm_cell_10/ReluÕ
 lstm_10/while/lstm_cell_10/mul_1Mul&lstm_10/while/lstm_cell_10/Sigmoid:y:0-lstm_10/while/lstm_cell_10/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2"
 lstm_10/while/lstm_cell_10/mul_1Ê
 lstm_10/while/lstm_cell_10/add_1AddV2"lstm_10/while/lstm_cell_10/mul:z:0$lstm_10/while/lstm_cell_10/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2"
 lstm_10/while/lstm_cell_10/add_1µ
$lstm_10/while/lstm_cell_10/Sigmoid_2Sigmoid)lstm_10/while/lstm_cell_10/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2&
$lstm_10/while/lstm_cell_10/Sigmoid_2§
!lstm_10/while/lstm_cell_10/Relu_1Relu$lstm_10/while/lstm_cell_10/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2#
!lstm_10/while/lstm_cell_10/Relu_1Ù
 lstm_10/while/lstm_cell_10/mul_2Mul(lstm_10/while/lstm_cell_10/Sigmoid_2:y:0/lstm_10/while/lstm_cell_10/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2"
 lstm_10/while/lstm_cell_10/mul_2
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
lstm_10/while/add/y
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
lstm_10/while/add_1/y
lstm_10/while/add_1AddV2(lstm_10_while_lstm_10_while_loop_counterlstm_10/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lstm_10/while/add_1
lstm_10/while/IdentityIdentitylstm_10/while/add_1:z:0^lstm_10/while/NoOp*
T0*
_output_shapes
: 2
lstm_10/while/Identity¦
lstm_10/while/Identity_1Identity.lstm_10_while_lstm_10_while_maximum_iterations^lstm_10/while/NoOp*
T0*
_output_shapes
: 2
lstm_10/while/Identity_1
lstm_10/while/Identity_2Identitylstm_10/while/add:z:0^lstm_10/while/NoOp*
T0*
_output_shapes
: 2
lstm_10/while/Identity_2º
lstm_10/while/Identity_3IdentityBlstm_10/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_10/while/NoOp*
T0*
_output_shapes
: 2
lstm_10/while/Identity_3®
lstm_10/while/Identity_4Identity$lstm_10/while/lstm_cell_10/mul_2:z:0^lstm_10/while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2
lstm_10/while/Identity_4®
lstm_10/while/Identity_5Identity$lstm_10/while/lstm_cell_10/add_1:z:0^lstm_10/while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2
lstm_10/while/Identity_5
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
9lstm_10_while_lstm_cell_10_matmul_readvariableop_resource;lstm_10_while_lstm_cell_10_matmul_readvariableop_resource_0"È
alstm_10_while_tensorarrayv2read_tensorlistgetitem_lstm_10_tensorarrayunstack_tensorlistfromtensorclstm_10_while_tensorarrayv2read_tensorlistgetitem_lstm_10_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿº:ÿÿÿÿÿÿÿÿÿº: : : : : 2f
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
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº:

_output_shapes
: :

_output_shapes
: 
ý%
ó
while_body_19554833
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_01
while_lstm_cell_11_19554857_0:
ºÌ0
while_lstm_cell_11_19554859_0:	SÌ,
while_lstm_cell_11_19554861_0:	Ì
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor/
while_lstm_cell_11_19554857:
ºÌ.
while_lstm_cell_11_19554859:	SÌ*
while_lstm_cell_11_19554861:	Ì¢*while/lstm_cell_11/StatefulPartitionedCallÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ:  29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÔ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemé
*while/lstm_cell_11/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_11_19554857_0while_lstm_cell_11_19554859_0while_lstm_cell_11_19554861_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿS:ÿÿÿÿÿÿÿÿÿS:ÿÿÿÿÿÿÿÿÿS*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_lstm_cell_11_layer_call_and_return_conditional_losses_195548192,
*while/lstm_cell_11/StatefulPartitionedCall÷
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
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3¤
while/Identity_4Identity3while/lstm_cell_11/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2
while/Identity_4¤
while/Identity_5Identity3while/lstm_cell_11/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2
while/Identity_5

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
while_lstm_cell_11_19554857while_lstm_cell_11_19554857_0"<
while_lstm_cell_11_19554859while_lstm_cell_11_19554859_0"<
while_lstm_cell_11_19554861while_lstm_cell_11_19554861_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿS:ÿÿÿÿÿÿÿÿÿS: : : : : 2X
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
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS:

_output_shapes
: :

_output_shapes
: 
¬\
 
E__inference_lstm_11_layer_call_and_return_conditional_losses_19557867
inputs_0?
+lstm_cell_11_matmul_readvariableop_resource:
ºÌ@
-lstm_cell_11_matmul_1_readvariableop_resource:	SÌ;
,lstm_cell_11_biasadd_readvariableop_resource:	Ì
identity¢#lstm_cell_11/BiasAdd/ReadVariableOp¢"lstm_cell_11/MatMul/ReadVariableOp¢$lstm_cell_11/MatMul_1/ReadVariableOp¢whileF
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
value	B :S2
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
value	B :S2
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
:ÿÿÿÿÿÿÿÿÿS2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :S2
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
value	B :S2
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
:ÿÿÿÿÿÿÿÿÿS2	
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
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿº2
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
valueB"ÿÿÿÿ:  27
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
:ÿÿÿÿÿÿÿÿÿº*
shrink_axis_mask2
strided_slice_2¶
"lstm_cell_11/MatMul/ReadVariableOpReadVariableOp+lstm_cell_11_matmul_readvariableop_resource* 
_output_shapes
:
ºÌ*
dtype02$
"lstm_cell_11/MatMul/ReadVariableOp­
lstm_cell_11/MatMulMatMulstrided_slice_2:output:0*lstm_cell_11/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÌ2
lstm_cell_11/MatMul»
$lstm_cell_11/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_11_matmul_1_readvariableop_resource*
_output_shapes
:	SÌ*
dtype02&
$lstm_cell_11/MatMul_1/ReadVariableOp©
lstm_cell_11/MatMul_1MatMulzeros:output:0,lstm_cell_11/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÌ2
lstm_cell_11/MatMul_1 
lstm_cell_11/addAddV2lstm_cell_11/MatMul:product:0lstm_cell_11/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÌ2
lstm_cell_11/add´
#lstm_cell_11/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_11_biasadd_readvariableop_resource*
_output_shapes	
:Ì*
dtype02%
#lstm_cell_11/BiasAdd/ReadVariableOp­
lstm_cell_11/BiasAddBiasAddlstm_cell_11/add:z:0+lstm_cell_11/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÌ2
lstm_cell_11/BiasAdd~
lstm_cell_11/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_11/split/split_dimó
lstm_cell_11/splitSplit%lstm_cell_11/split/split_dim:output:0lstm_cell_11/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿS:ÿÿÿÿÿÿÿÿÿS:ÿÿÿÿÿÿÿÿÿS:ÿÿÿÿÿÿÿÿÿS*
	num_split2
lstm_cell_11/split
lstm_cell_11/SigmoidSigmoidlstm_cell_11/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2
lstm_cell_11/Sigmoid
lstm_cell_11/Sigmoid_1Sigmoidlstm_cell_11/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2
lstm_cell_11/Sigmoid_1
lstm_cell_11/mulMullstm_cell_11/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2
lstm_cell_11/mul}
lstm_cell_11/ReluRelulstm_cell_11/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2
lstm_cell_11/Relu
lstm_cell_11/mul_1Mullstm_cell_11/Sigmoid:y:0lstm_cell_11/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2
lstm_cell_11/mul_1
lstm_cell_11/add_1AddV2lstm_cell_11/mul:z:0lstm_cell_11/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2
lstm_cell_11/add_1
lstm_cell_11/Sigmoid_2Sigmoidlstm_cell_11/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2
lstm_cell_11/Sigmoid_2|
lstm_cell_11/Relu_1Relulstm_cell_11/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2
lstm_cell_11/Relu_1 
lstm_cell_11/mul_2Mullstm_cell_11/Sigmoid_2:y:0!lstm_cell_11/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2
lstm_cell_11/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿS   2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_11_matmul_readvariableop_resource-lstm_cell_11_matmul_1_readvariableop_resource,lstm_cell_11_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿS:ÿÿÿÿÿÿÿÿÿS: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_19557783*
condR
while_cond_19557782*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿS:ÿÿÿÿÿÿÿÿÿS: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿS   22
0TensorArrayV2Stack/TensorListStack/element_shapeñ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿS*
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
:ÿÿÿÿÿÿÿÿÿS*
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
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿS2
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
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿS2

IdentityÈ
NoOpNoOp$^lstm_cell_11/BiasAdd/ReadVariableOp#^lstm_cell_11/MatMul/ReadVariableOp%^lstm_cell_11/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿº: : : 2J
#lstm_cell_11/BiasAdd/ReadVariableOp#lstm_cell_11/BiasAdd/ReadVariableOp2H
"lstm_cell_11/MatMul/ReadVariableOp"lstm_cell_11/MatMul/ReadVariableOp2L
$lstm_cell_11/MatMul_1/ReadVariableOp$lstm_cell_11/MatMul_1/ReadVariableOp2
whilewhile:_ [
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿº
"
_user_specified_name
inputs/0
Ï!
ü
E__inference_dense_5_layer_call_and_return_conditional_losses_19558422

inputs3
!tensordot_readvariableop_resource:S-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢Tensordot/ReadVariableOp
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:S*
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
:ÿÿÿÿÿÿÿÿÿS2
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
:ÿÿÿÿÿÿÿÿÿS: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿS
 
_user_specified_nameinputs
×
ñ
(sequential_5_lstm_11_while_cond_19554001F
Bsequential_5_lstm_11_while_sequential_5_lstm_11_while_loop_counterL
Hsequential_5_lstm_11_while_sequential_5_lstm_11_while_maximum_iterations*
&sequential_5_lstm_11_while_placeholder,
(sequential_5_lstm_11_while_placeholder_1,
(sequential_5_lstm_11_while_placeholder_2,
(sequential_5_lstm_11_while_placeholder_3H
Dsequential_5_lstm_11_while_less_sequential_5_lstm_11_strided_slice_1`
\sequential_5_lstm_11_while_sequential_5_lstm_11_while_cond_19554001___redundant_placeholder0`
\sequential_5_lstm_11_while_sequential_5_lstm_11_while_cond_19554001___redundant_placeholder1`
\sequential_5_lstm_11_while_sequential_5_lstm_11_while_cond_19554001___redundant_placeholder2`
\sequential_5_lstm_11_while_sequential_5_lstm_11_while_cond_19554001___redundant_placeholder3'
#sequential_5_lstm_11_while_identity
Ù
sequential_5/lstm_11/while/LessLess&sequential_5_lstm_11_while_placeholderDsequential_5_lstm_11_while_less_sequential_5_lstm_11_strided_slice_1*
T0*
_output_shapes
: 2!
sequential_5/lstm_11/while/Less
#sequential_5/lstm_11/while/IdentityIdentity#sequential_5/lstm_11/while/Less:z:0*
T0
*
_output_shapes
: 2%
#sequential_5/lstm_11/while/Identity"S
#sequential_5_lstm_11_while_identity,sequential_5/lstm_11/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿS:ÿÿÿÿÿÿÿÿÿS: ::::: 

_output_shapes
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
:ÿÿÿÿÿÿÿÿÿS:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS:

_output_shapes
: :

_output_shapes
:
ß
Í
while_cond_19557933
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_19557933___redundant_placeholder06
2while_while_cond_19557933___redundant_placeholder16
2while_while_cond_19557933___redundant_placeholder26
2while_while_cond_19557933___redundant_placeholder3
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
@: : : : :ÿÿÿÿÿÿÿÿÿS:ÿÿÿÿÿÿÿÿÿS: ::::: 

_output_shapes
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
:ÿÿÿÿÿÿÿÿÿS:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS:

_output_shapes
: :

_output_shapes
:

f
H__inference_dropout_10_layer_call_and_return_conditional_losses_19557694

inputs

identity_1_
IdentityIdentityinputs*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2

Identityn

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿº:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿº
 
_user_specified_nameinputs
ß
Í
while_cond_19558235
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_19558235___redundant_placeholder06
2while_while_cond_19558235___redundant_placeholder16
2while_while_cond_19558235___redundant_placeholder26
2while_while_cond_19558235___redundant_placeholder3
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
@: : : : :ÿÿÿÿÿÿÿÿÿS:ÿÿÿÿÿÿÿÿÿS: ::::: 

_output_shapes
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
:ÿÿÿÿÿÿÿÿÿS:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS:

_output_shapes
: :

_output_shapes
:
½
ù
/__inference_lstm_cell_11_layer_call_fn_19558627

inputs
states_0
states_1
unknown:
ºÌ
	unknown_0:	SÌ
	unknown_1:	Ì
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
9:ÿÿÿÿÿÿÿÿÿS:ÿÿÿÿÿÿÿÿÿS:ÿÿÿÿÿÿÿÿÿS*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_lstm_cell_11_layer_call_and_return_conditional_losses_195549652
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2

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
@:ÿÿÿÿÿÿÿÿÿº:ÿÿÿÿÿÿÿÿÿS:ÿÿÿÿÿÿÿÿÿS: : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº
 
_user_specified_nameinputs:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS
"
_user_specified_name
states/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS
"
_user_specified_name
states/1
¤
µ
J__inference_sequential_5_layer_call_and_return_conditional_losses_19556218

inputs#
lstm_10_19556196:	]è	$
lstm_10_19556198:
ºè	
lstm_10_19556200:	è	$
lstm_11_19556204:
ºÌ#
lstm_11_19556206:	SÌ
lstm_11_19556208:	Ì"
dense_5_19556212:S
dense_5_19556214:
identity¢dense_5/StatefulPartitionedCall¢"dropout_10/StatefulPartitionedCall¢"dropout_11/StatefulPartitionedCall¢lstm_10/StatefulPartitionedCall¢lstm_11/StatefulPartitionedCall®
lstm_10/StatefulPartitionedCallStatefulPartitionedCallinputslstm_10_19556196lstm_10_19556198lstm_10_19556200*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿº*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_lstm_10_layer_call_and_return_conditional_losses_195561612!
lstm_10/StatefulPartitionedCall
"dropout_10/StatefulPartitionedCallStatefulPartitionedCall(lstm_10/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿº* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_dropout_10_layer_call_and_return_conditional_losses_195559942$
"dropout_10/StatefulPartitionedCallÒ
lstm_11/StatefulPartitionedCallStatefulPartitionedCall+dropout_10/StatefulPartitionedCall:output:0lstm_11_19556204lstm_11_19556206lstm_11_19556208*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿS*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_lstm_11_layer_call_and_return_conditional_losses_195559652!
lstm_11/StatefulPartitionedCall¿
"dropout_11/StatefulPartitionedCallStatefulPartitionedCall(lstm_11/StatefulPartitionedCall:output:0#^dropout_10/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿS* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_dropout_11_layer_call_and_return_conditional_losses_195557982$
"dropout_11/StatefulPartitionedCall¾
dense_5/StatefulPartitionedCallStatefulPartitionedCall+dropout_11/StatefulPartitionedCall:output:0dense_5_19556212dense_5_19556214*
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
E__inference_dense_5_layer_call_and_return_conditional_losses_195557422!
dense_5/StatefulPartitionedCall
IdentityIdentity(dense_5/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityþ
NoOpNoOp ^dense_5/StatefulPartitionedCall#^dropout_10/StatefulPartitionedCall#^dropout_11/StatefulPartitionedCall ^lstm_10/StatefulPartitionedCall ^lstm_11/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ]: : : : : : : : 2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2H
"dropout_10/StatefulPartitionedCall"dropout_10/StatefulPartitionedCall2H
"dropout_11/StatefulPartitionedCall"dropout_11/StatefulPartitionedCall2B
lstm_10/StatefulPartitionedCalllstm_10/StatefulPartitionedCall2B
lstm_11/StatefulPartitionedCalllstm_11/StatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
 
_user_specified_nameinputs
\

E__inference_lstm_10_layer_call_and_return_conditional_losses_19555531

inputs>
+lstm_cell_10_matmul_readvariableop_resource:	]è	A
-lstm_cell_10_matmul_1_readvariableop_resource:
ºè	;
,lstm_cell_10_biasadd_readvariableop_resource:	è	
identity¢#lstm_cell_10/BiasAdd/ReadVariableOp¢"lstm_cell_10/MatMul/ReadVariableOp¢$lstm_cell_10/MatMul_1/ReadVariableOp¢whileD
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
B :º2
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
B :º2
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
:ÿÿÿÿÿÿÿÿÿº2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :º2
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
B :º2
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
:ÿÿÿÿÿÿÿÿÿº2	
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
"lstm_cell_10/MatMul/ReadVariableOpReadVariableOp+lstm_cell_10_matmul_readvariableop_resource*
_output_shapes
:	]è	*
dtype02$
"lstm_cell_10/MatMul/ReadVariableOp­
lstm_cell_10/MatMulMatMulstrided_slice_2:output:0*lstm_cell_10/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè	2
lstm_cell_10/MatMul¼
$lstm_cell_10/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_10_matmul_1_readvariableop_resource* 
_output_shapes
:
ºè	*
dtype02&
$lstm_cell_10/MatMul_1/ReadVariableOp©
lstm_cell_10/MatMul_1MatMulzeros:output:0,lstm_cell_10/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè	2
lstm_cell_10/MatMul_1 
lstm_cell_10/addAddV2lstm_cell_10/MatMul:product:0lstm_cell_10/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè	2
lstm_cell_10/add´
#lstm_cell_10/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_10_biasadd_readvariableop_resource*
_output_shapes	
:è	*
dtype02%
#lstm_cell_10/BiasAdd/ReadVariableOp­
lstm_cell_10/BiasAddBiasAddlstm_cell_10/add:z:0+lstm_cell_10/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè	2
lstm_cell_10/BiasAdd~
lstm_cell_10/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_10/split/split_dim÷
lstm_cell_10/splitSplit%lstm_cell_10/split/split_dim:output:0lstm_cell_10/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿº:ÿÿÿÿÿÿÿÿÿº:ÿÿÿÿÿÿÿÿÿº:ÿÿÿÿÿÿÿÿÿº*
	num_split2
lstm_cell_10/split
lstm_cell_10/SigmoidSigmoidlstm_cell_10/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2
lstm_cell_10/Sigmoid
lstm_cell_10/Sigmoid_1Sigmoidlstm_cell_10/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2
lstm_cell_10/Sigmoid_1
lstm_cell_10/mulMullstm_cell_10/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2
lstm_cell_10/mul~
lstm_cell_10/ReluRelulstm_cell_10/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2
lstm_cell_10/Relu
lstm_cell_10/mul_1Mullstm_cell_10/Sigmoid:y:0lstm_cell_10/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2
lstm_cell_10/mul_1
lstm_cell_10/add_1AddV2lstm_cell_10/mul:z:0lstm_cell_10/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2
lstm_cell_10/add_1
lstm_cell_10/Sigmoid_2Sigmoidlstm_cell_10/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2
lstm_cell_10/Sigmoid_2}
lstm_cell_10/Relu_1Relulstm_cell_10/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2
lstm_cell_10/Relu_1¡
lstm_cell_10/mul_2Mullstm_cell_10/Sigmoid_2:y:0!lstm_cell_10/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2
lstm_cell_10/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ:  2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_10_matmul_readvariableop_resource-lstm_cell_10_matmul_1_readvariableop_resource,lstm_cell_10_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿº:ÿÿÿÿÿÿÿÿÿº: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_19555447*
condR
while_cond_19555446*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿº:ÿÿÿÿÿÿÿÿÿº: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ:  22
0TensorArrayV2Stack/TensorListStack/element_shapeé
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿº*
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
:ÿÿÿÿÿÿÿÿÿº*
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
:ÿÿÿÿÿÿÿÿÿº2
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
:ÿÿÿÿÿÿÿÿÿº2

IdentityÈ
NoOpNoOp$^lstm_cell_10/BiasAdd/ReadVariableOp#^lstm_cell_10/MatMul/ReadVariableOp%^lstm_cell_10/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ]: : : 2J
#lstm_cell_10/BiasAdd/ReadVariableOp#lstm_cell_10/BiasAdd/ReadVariableOp2H
"lstm_cell_10/MatMul/ReadVariableOp"lstm_cell_10/MatMul/ReadVariableOp2L
$lstm_cell_10/MatMul_1/ReadVariableOp$lstm_cell_10/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
 
_user_specified_nameinputs


J__inference_lstm_cell_10_layer_call_and_return_conditional_losses_19554335

inputs

states
states_11
matmul_readvariableop_resource:	]è	4
 matmul_1_readvariableop_resource:
ºè	.
biasadd_readvariableop_resource:	è	
identity

identity_1

identity_2¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	]è	*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè	2
MatMul
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
ºè	*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè	2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè	2
add
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:è	*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè	2	
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
P:ÿÿÿÿÿÿÿÿÿº:ÿÿÿÿÿÿÿÿÿº:ÿÿÿÿÿÿÿÿÿº:ÿÿÿÿÿÿÿÿÿº*
	num_split2
split`
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2	
Sigmoidd
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2
	Sigmoid_1]
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2
mulW
ReluRelusplit:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2
Relui
mul_1MulSigmoid:y:0Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2
mul_1^
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2
add_1d
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2
	Sigmoid_2V
Relu_1Relu	add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2
Relu_1m
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2
mul_2e
IdentityIdentity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2

Identityi

Identity_1Identity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2

Identity_1i

Identity_2Identity	add_1:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2

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
A:ÿÿÿÿÿÿÿÿÿ]:ÿÿÿÿÿÿÿÿÿº:ÿÿÿÿÿÿÿÿÿº: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
 
_user_specified_nameinputs:PL
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº
 
_user_specified_namestates:PL
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº
 
_user_specified_namestates
ú	
È
&__inference_signature_wrapper_19556331
lstm_10_input
unknown:	]è	
	unknown_0:
ºè	
	unknown_1:	è	
	unknown_2:
ºÌ
	unknown_3:	SÌ
	unknown_4:	Ì
	unknown_5:S
	unknown_6:
identity¢StatefulPartitionedCall¬
StatefulPartitionedCallStatefulPartitionedCalllstm_10_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
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
#__inference__wrapped_model_195541142
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
_user_specified_namelstm_10_input
Ø
I
-__inference_dropout_10_layer_call_fn_19557711

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
:ÿÿÿÿÿÿÿÿÿº* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_dropout_10_layer_call_and_return_conditional_losses_195555442
PartitionedCallq
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿº:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿº
 
_user_specified_nameinputs
ã
Í
while_cond_19554202
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_19554202___redundant_placeholder06
2while_while_cond_19554202___redundant_placeholder16
2while_while_cond_19554202___redundant_placeholder26
2while_while_cond_19554202___redundant_placeholder3
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
B: : : : :ÿÿÿÿÿÿÿÿÿº:ÿÿÿÿÿÿÿÿÿº: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº:

_output_shapes
: :

_output_shapes
:
¬
ò
J__inference_sequential_5_layer_call_and_return_conditional_losses_19556283
lstm_10_input#
lstm_10_19556261:	]è	$
lstm_10_19556263:
ºè	
lstm_10_19556265:	è	$
lstm_11_19556269:
ºÌ#
lstm_11_19556271:	SÌ
lstm_11_19556273:	Ì"
dense_5_19556277:S
dense_5_19556279:
identity¢dense_5/StatefulPartitionedCall¢lstm_10/StatefulPartitionedCall¢lstm_11/StatefulPartitionedCallµ
lstm_10/StatefulPartitionedCallStatefulPartitionedCalllstm_10_inputlstm_10_19556261lstm_10_19556263lstm_10_19556265*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿº*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_lstm_10_layer_call_and_return_conditional_losses_195555312!
lstm_10/StatefulPartitionedCall
dropout_10/PartitionedCallPartitionedCall(lstm_10/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿº* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_dropout_10_layer_call_and_return_conditional_losses_195555442
dropout_10/PartitionedCallÊ
lstm_11/StatefulPartitionedCallStatefulPartitionedCall#dropout_10/PartitionedCall:output:0lstm_11_19556269lstm_11_19556271lstm_11_19556273*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿS*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_lstm_11_layer_call_and_return_conditional_losses_195556962!
lstm_11/StatefulPartitionedCall
dropout_11/PartitionedCallPartitionedCall(lstm_11/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿS* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_dropout_11_layer_call_and_return_conditional_losses_195557092
dropout_11/PartitionedCall¶
dense_5/StatefulPartitionedCallStatefulPartitionedCall#dropout_11/PartitionedCall:output:0dense_5_19556277dense_5_19556279*
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
E__inference_dense_5_layer_call_and_return_conditional_losses_195557422!
dense_5/StatefulPartitionedCall
IdentityIdentity(dense_5/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity´
NoOpNoOp ^dense_5/StatefulPartitionedCall ^lstm_10/StatefulPartitionedCall ^lstm_11/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ]: : : : : : : : 2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2B
lstm_10/StatefulPartitionedCalllstm_10/StatefulPartitionedCall2B
lstm_11/StatefulPartitionedCalllstm_11/StatefulPartitionedCall:Z V
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
'
_user_specified_namelstm_10_input
ý%
ó
while_body_19555043
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_01
while_lstm_cell_11_19555067_0:
ºÌ0
while_lstm_cell_11_19555069_0:	SÌ,
while_lstm_cell_11_19555071_0:	Ì
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor/
while_lstm_cell_11_19555067:
ºÌ.
while_lstm_cell_11_19555069:	SÌ*
while_lstm_cell_11_19555071:	Ì¢*while/lstm_cell_11/StatefulPartitionedCallÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ:  29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÔ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemé
*while/lstm_cell_11/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_11_19555067_0while_lstm_cell_11_19555069_0while_lstm_cell_11_19555071_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿS:ÿÿÿÿÿÿÿÿÿS:ÿÿÿÿÿÿÿÿÿS*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_lstm_cell_11_layer_call_and_return_conditional_losses_195549652,
*while/lstm_cell_11/StatefulPartitionedCall÷
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
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3¤
while/Identity_4Identity3while/lstm_cell_11/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2
while/Identity_4¤
while/Identity_5Identity3while/lstm_cell_11/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2
while/Identity_5

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
while_lstm_cell_11_19555067while_lstm_cell_11_19555067_0"<
while_lstm_cell_11_19555069while_lstm_cell_11_19555069_0"<
while_lstm_cell_11_19555071while_lstm_cell_11_19555071_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿS:ÿÿÿÿÿÿÿÿÿS: : : : : 2X
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
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS:

_output_shapes
: :

_output_shapes
: 
\

E__inference_lstm_10_layer_call_and_return_conditional_losses_19557494

inputs>
+lstm_cell_10_matmul_readvariableop_resource:	]è	A
-lstm_cell_10_matmul_1_readvariableop_resource:
ºè	;
,lstm_cell_10_biasadd_readvariableop_resource:	è	
identity¢#lstm_cell_10/BiasAdd/ReadVariableOp¢"lstm_cell_10/MatMul/ReadVariableOp¢$lstm_cell_10/MatMul_1/ReadVariableOp¢whileD
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
B :º2
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
B :º2
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
:ÿÿÿÿÿÿÿÿÿº2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :º2
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
B :º2
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
:ÿÿÿÿÿÿÿÿÿº2	
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
"lstm_cell_10/MatMul/ReadVariableOpReadVariableOp+lstm_cell_10_matmul_readvariableop_resource*
_output_shapes
:	]è	*
dtype02$
"lstm_cell_10/MatMul/ReadVariableOp­
lstm_cell_10/MatMulMatMulstrided_slice_2:output:0*lstm_cell_10/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè	2
lstm_cell_10/MatMul¼
$lstm_cell_10/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_10_matmul_1_readvariableop_resource* 
_output_shapes
:
ºè	*
dtype02&
$lstm_cell_10/MatMul_1/ReadVariableOp©
lstm_cell_10/MatMul_1MatMulzeros:output:0,lstm_cell_10/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè	2
lstm_cell_10/MatMul_1 
lstm_cell_10/addAddV2lstm_cell_10/MatMul:product:0lstm_cell_10/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè	2
lstm_cell_10/add´
#lstm_cell_10/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_10_biasadd_readvariableop_resource*
_output_shapes	
:è	*
dtype02%
#lstm_cell_10/BiasAdd/ReadVariableOp­
lstm_cell_10/BiasAddBiasAddlstm_cell_10/add:z:0+lstm_cell_10/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè	2
lstm_cell_10/BiasAdd~
lstm_cell_10/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_10/split/split_dim÷
lstm_cell_10/splitSplit%lstm_cell_10/split/split_dim:output:0lstm_cell_10/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿº:ÿÿÿÿÿÿÿÿÿº:ÿÿÿÿÿÿÿÿÿº:ÿÿÿÿÿÿÿÿÿº*
	num_split2
lstm_cell_10/split
lstm_cell_10/SigmoidSigmoidlstm_cell_10/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2
lstm_cell_10/Sigmoid
lstm_cell_10/Sigmoid_1Sigmoidlstm_cell_10/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2
lstm_cell_10/Sigmoid_1
lstm_cell_10/mulMullstm_cell_10/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2
lstm_cell_10/mul~
lstm_cell_10/ReluRelulstm_cell_10/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2
lstm_cell_10/Relu
lstm_cell_10/mul_1Mullstm_cell_10/Sigmoid:y:0lstm_cell_10/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2
lstm_cell_10/mul_1
lstm_cell_10/add_1AddV2lstm_cell_10/mul:z:0lstm_cell_10/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2
lstm_cell_10/add_1
lstm_cell_10/Sigmoid_2Sigmoidlstm_cell_10/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2
lstm_cell_10/Sigmoid_2}
lstm_cell_10/Relu_1Relulstm_cell_10/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2
lstm_cell_10/Relu_1¡
lstm_cell_10/mul_2Mullstm_cell_10/Sigmoid_2:y:0!lstm_cell_10/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2
lstm_cell_10/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ:  2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_10_matmul_readvariableop_resource-lstm_cell_10_matmul_1_readvariableop_resource,lstm_cell_10_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿº:ÿÿÿÿÿÿÿÿÿº: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_19557410*
condR
while_cond_19557409*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿº:ÿÿÿÿÿÿÿÿÿº: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ:  22
0TensorArrayV2Stack/TensorListStack/element_shapeé
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿº*
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
:ÿÿÿÿÿÿÿÿÿº*
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
:ÿÿÿÿÿÿÿÿÿº2
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
:ÿÿÿÿÿÿÿÿÿº2

IdentityÈ
NoOpNoOp$^lstm_cell_10/BiasAdd/ReadVariableOp#^lstm_cell_10/MatMul/ReadVariableOp%^lstm_cell_10/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ]: : : 2J
#lstm_cell_10/BiasAdd/ReadVariableOp#lstm_cell_10/BiasAdd/ReadVariableOp2H
"lstm_cell_10/MatMul/ReadVariableOp"lstm_cell_10/MatMul/ReadVariableOp2L
$lstm_cell_10/MatMul_1/ReadVariableOp$lstm_cell_10/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
 
_user_specified_nameinputs
ª

Ñ
/__inference_sequential_5_layer_call_fn_19556258
lstm_10_input
unknown:	]è	
	unknown_0:
ºè	
	unknown_1:	è	
	unknown_2:
ºÌ
	unknown_3:	SÌ
	unknown_4:	Ì
	unknown_5:S
	unknown_6:
identity¢StatefulPartitionedCallÓ
StatefulPartitionedCallStatefulPartitionedCalllstm_10_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
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
J__inference_sequential_5_layer_call_and_return_conditional_losses_195562182
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
_user_specified_namelstm_10_input
¬\
 
E__inference_lstm_11_layer_call_and_return_conditional_losses_19558018
inputs_0?
+lstm_cell_11_matmul_readvariableop_resource:
ºÌ@
-lstm_cell_11_matmul_1_readvariableop_resource:	SÌ;
,lstm_cell_11_biasadd_readvariableop_resource:	Ì
identity¢#lstm_cell_11/BiasAdd/ReadVariableOp¢"lstm_cell_11/MatMul/ReadVariableOp¢$lstm_cell_11/MatMul_1/ReadVariableOp¢whileF
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
value	B :S2
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
value	B :S2
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
:ÿÿÿÿÿÿÿÿÿS2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :S2
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
value	B :S2
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
:ÿÿÿÿÿÿÿÿÿS2	
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
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿº2
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
valueB"ÿÿÿÿ:  27
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
:ÿÿÿÿÿÿÿÿÿº*
shrink_axis_mask2
strided_slice_2¶
"lstm_cell_11/MatMul/ReadVariableOpReadVariableOp+lstm_cell_11_matmul_readvariableop_resource* 
_output_shapes
:
ºÌ*
dtype02$
"lstm_cell_11/MatMul/ReadVariableOp­
lstm_cell_11/MatMulMatMulstrided_slice_2:output:0*lstm_cell_11/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÌ2
lstm_cell_11/MatMul»
$lstm_cell_11/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_11_matmul_1_readvariableop_resource*
_output_shapes
:	SÌ*
dtype02&
$lstm_cell_11/MatMul_1/ReadVariableOp©
lstm_cell_11/MatMul_1MatMulzeros:output:0,lstm_cell_11/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÌ2
lstm_cell_11/MatMul_1 
lstm_cell_11/addAddV2lstm_cell_11/MatMul:product:0lstm_cell_11/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÌ2
lstm_cell_11/add´
#lstm_cell_11/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_11_biasadd_readvariableop_resource*
_output_shapes	
:Ì*
dtype02%
#lstm_cell_11/BiasAdd/ReadVariableOp­
lstm_cell_11/BiasAddBiasAddlstm_cell_11/add:z:0+lstm_cell_11/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÌ2
lstm_cell_11/BiasAdd~
lstm_cell_11/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_11/split/split_dimó
lstm_cell_11/splitSplit%lstm_cell_11/split/split_dim:output:0lstm_cell_11/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿS:ÿÿÿÿÿÿÿÿÿS:ÿÿÿÿÿÿÿÿÿS:ÿÿÿÿÿÿÿÿÿS*
	num_split2
lstm_cell_11/split
lstm_cell_11/SigmoidSigmoidlstm_cell_11/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2
lstm_cell_11/Sigmoid
lstm_cell_11/Sigmoid_1Sigmoidlstm_cell_11/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2
lstm_cell_11/Sigmoid_1
lstm_cell_11/mulMullstm_cell_11/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2
lstm_cell_11/mul}
lstm_cell_11/ReluRelulstm_cell_11/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2
lstm_cell_11/Relu
lstm_cell_11/mul_1Mullstm_cell_11/Sigmoid:y:0lstm_cell_11/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2
lstm_cell_11/mul_1
lstm_cell_11/add_1AddV2lstm_cell_11/mul:z:0lstm_cell_11/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2
lstm_cell_11/add_1
lstm_cell_11/Sigmoid_2Sigmoidlstm_cell_11/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2
lstm_cell_11/Sigmoid_2|
lstm_cell_11/Relu_1Relulstm_cell_11/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2
lstm_cell_11/Relu_1 
lstm_cell_11/mul_2Mullstm_cell_11/Sigmoid_2:y:0!lstm_cell_11/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2
lstm_cell_11/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿS   2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_11_matmul_readvariableop_resource-lstm_cell_11_matmul_1_readvariableop_resource,lstm_cell_11_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿS:ÿÿÿÿÿÿÿÿÿS: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_19557934*
condR
while_cond_19557933*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿS:ÿÿÿÿÿÿÿÿÿS: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿS   22
0TensorArrayV2Stack/TensorListStack/element_shapeñ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿS*
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
:ÿÿÿÿÿÿÿÿÿS*
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
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿS2
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
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿS2

IdentityÈ
NoOpNoOp$^lstm_cell_11/BiasAdd/ReadVariableOp#^lstm_cell_11/MatMul/ReadVariableOp%^lstm_cell_11/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿº: : : 2J
#lstm_cell_11/BiasAdd/ReadVariableOp#lstm_cell_11/BiasAdd/ReadVariableOp2H
"lstm_cell_11/MatMul/ReadVariableOp"lstm_cell_11/MatMul/ReadVariableOp2L
$lstm_cell_11/MatMul_1/ReadVariableOp$lstm_cell_11/MatMul_1/ReadVariableOp2
whilewhile:_ [
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿº
"
_user_specified_name
inputs/0
ù

J__inference_lstm_cell_11_layer_call_and_return_conditional_losses_19554965

inputs

states
states_12
matmul_readvariableop_resource:
ºÌ3
 matmul_1_readvariableop_resource:	SÌ.
biasadd_readvariableop_resource:	Ì
identity

identity_1

identity_2¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
ºÌ*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÌ2
MatMul
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	SÌ*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÌ2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÌ2
add
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:Ì*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÌ2	
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
L:ÿÿÿÿÿÿÿÿÿS:ÿÿÿÿÿÿÿÿÿS:ÿÿÿÿÿÿÿÿÿS:ÿÿÿÿÿÿÿÿÿS*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2
	Sigmoid_1\
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2
mulV
ReluRelusplit:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2
	Sigmoid_2U
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2
mul_2d
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2

Identityh

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2

Identity_1h

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2

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
@:ÿÿÿÿÿÿÿÿÿº:ÿÿÿÿÿÿÿÿÿS:ÿÿÿÿÿÿÿÿÿS: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS
 
_user_specified_namestates:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS
 
_user_specified_namestates
&
ó
while_body_19554413
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_00
while_lstm_cell_10_19554437_0:	]è	1
while_lstm_cell_10_19554439_0:
ºè	,
while_lstm_cell_10_19554441_0:	è	
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor.
while_lstm_cell_10_19554437:	]è	/
while_lstm_cell_10_19554439:
ºè	*
while_lstm_cell_10_19554441:	è	¢*while/lstm_cell_10/StatefulPartitionedCallÃ
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
*while/lstm_cell_10/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_10_19554437_0while_lstm_cell_10_19554439_0while_lstm_cell_10_19554441_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿº:ÿÿÿÿÿÿÿÿÿº:ÿÿÿÿÿÿÿÿÿº*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_lstm_cell_10_layer_call_and_return_conditional_losses_195543352,
*while/lstm_cell_10/StatefulPartitionedCall÷
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
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3¥
while/Identity_4Identity3while/lstm_cell_10/StatefulPartitionedCall:output:1^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2
while/Identity_4¥
while/Identity_5Identity3while/lstm_cell_10/StatefulPartitionedCall:output:2^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2
while/Identity_5

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
while_lstm_cell_10_19554437while_lstm_cell_10_19554437_0"<
while_lstm_cell_10_19554439while_lstm_cell_10_19554439_0"<
while_lstm_cell_10_19554441while_lstm_cell_10_19554441_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿº:ÿÿÿÿÿÿÿÿÿº: : : : : 2X
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
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº:

_output_shapes
: :

_output_shapes
: 
ã
Í
while_cond_19555446
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_19555446___redundant_placeholder06
2while_while_cond_19555446___redundant_placeholder16
2while_while_cond_19555446___redundant_placeholder26
2while_while_cond_19555446___redundant_placeholder3
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
B: : : : :ÿÿÿÿÿÿÿÿÿº:ÿÿÿÿÿÿÿÿÿº: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº:

_output_shapes
: :

_output_shapes
:
Ô

í
lstm_10_while_cond_19556724,
(lstm_10_while_lstm_10_while_loop_counter2
.lstm_10_while_lstm_10_while_maximum_iterations
lstm_10_while_placeholder
lstm_10_while_placeholder_1
lstm_10_while_placeholder_2
lstm_10_while_placeholder_3.
*lstm_10_while_less_lstm_10_strided_slice_1F
Blstm_10_while_lstm_10_while_cond_19556724___redundant_placeholder0F
Blstm_10_while_lstm_10_while_cond_19556724___redundant_placeholder1F
Blstm_10_while_lstm_10_while_cond_19556724___redundant_placeholder2F
Blstm_10_while_lstm_10_while_cond_19556724___redundant_placeholder3
lstm_10_while_identity

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
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :ÿÿÿÿÿÿÿÿÿº:ÿÿÿÿÿÿÿÿÿº: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº:

_output_shapes
: :

_output_shapes
:
¹
¼
J__inference_sequential_5_layer_call_and_return_conditional_losses_19556308
lstm_10_input#
lstm_10_19556286:	]è	$
lstm_10_19556288:
ºè	
lstm_10_19556290:	è	$
lstm_11_19556294:
ºÌ#
lstm_11_19556296:	SÌ
lstm_11_19556298:	Ì"
dense_5_19556302:S
dense_5_19556304:
identity¢dense_5/StatefulPartitionedCall¢"dropout_10/StatefulPartitionedCall¢"dropout_11/StatefulPartitionedCall¢lstm_10/StatefulPartitionedCall¢lstm_11/StatefulPartitionedCallµ
lstm_10/StatefulPartitionedCallStatefulPartitionedCalllstm_10_inputlstm_10_19556286lstm_10_19556288lstm_10_19556290*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿº*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_lstm_10_layer_call_and_return_conditional_losses_195561612!
lstm_10/StatefulPartitionedCall
"dropout_10/StatefulPartitionedCallStatefulPartitionedCall(lstm_10/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿº* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_dropout_10_layer_call_and_return_conditional_losses_195559942$
"dropout_10/StatefulPartitionedCallÒ
lstm_11/StatefulPartitionedCallStatefulPartitionedCall+dropout_10/StatefulPartitionedCall:output:0lstm_11_19556294lstm_11_19556296lstm_11_19556298*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿS*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_lstm_11_layer_call_and_return_conditional_losses_195559652!
lstm_11/StatefulPartitionedCall¿
"dropout_11/StatefulPartitionedCallStatefulPartitionedCall(lstm_11/StatefulPartitionedCall:output:0#^dropout_10/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿS* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_dropout_11_layer_call_and_return_conditional_losses_195557982$
"dropout_11/StatefulPartitionedCall¾
dense_5/StatefulPartitionedCallStatefulPartitionedCall+dropout_11/StatefulPartitionedCall:output:0dense_5_19556302dense_5_19556304*
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
E__inference_dense_5_layer_call_and_return_conditional_losses_195557422!
dense_5/StatefulPartitionedCall
IdentityIdentity(dense_5/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityþ
NoOpNoOp ^dense_5/StatefulPartitionedCall#^dropout_10/StatefulPartitionedCall#^dropout_11/StatefulPartitionedCall ^lstm_10/StatefulPartitionedCall ^lstm_11/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ]: : : : : : : : 2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2H
"dropout_10/StatefulPartitionedCall"dropout_10/StatefulPartitionedCall2H
"dropout_11/StatefulPartitionedCall"dropout_11/StatefulPartitionedCall2B
lstm_10/StatefulPartitionedCalllstm_10/StatefulPartitionedCall2B
lstm_11/StatefulPartitionedCalllstm_11/StatefulPartitionedCall:Z V
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
'
_user_specified_namelstm_10_input
»
f
-__inference_dropout_10_layer_call_fn_19557716

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
:ÿÿÿÿÿÿÿÿÿº* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_dropout_10_layer_call_and_return_conditional_losses_195559942
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿº22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿº
 
_user_specified_nameinputs
¾F

E__inference_lstm_11_layer_call_and_return_conditional_losses_19554902

inputs)
lstm_cell_11_19554820:
ºÌ(
lstm_cell_11_19554822:	SÌ$
lstm_cell_11_19554824:	Ì
identity¢$lstm_cell_11/StatefulPartitionedCall¢whileD
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
value	B :S2
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
value	B :S2
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
:ÿÿÿÿÿÿÿÿÿS2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :S2
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
value	B :S2
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
:ÿÿÿÿÿÿÿÿÿS2	
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
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿº2
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
valueB"ÿÿÿÿ:  27
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
:ÿÿÿÿÿÿÿÿÿº*
shrink_axis_mask2
strided_slice_2¥
$lstm_cell_11/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_11_19554820lstm_cell_11_19554822lstm_cell_11_19554824*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿS:ÿÿÿÿÿÿÿÿÿS:ÿÿÿÿÿÿÿÿÿS*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_lstm_cell_11_layer_call_and_return_conditional_losses_195548192&
$lstm_cell_11/StatefulPartitionedCall
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿS   2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_11_19554820lstm_cell_11_19554822lstm_cell_11_19554824*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿS:ÿÿÿÿÿÿÿÿÿS: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_19554833*
condR
while_cond_19554832*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿS:ÿÿÿÿÿÿÿÿÿS: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿS   22
0TensorArrayV2Stack/TensorListStack/element_shapeñ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿS*
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
:ÿÿÿÿÿÿÿÿÿS*
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
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿS2
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
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿS2

Identity}
NoOpNoOp%^lstm_cell_11/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿº: : : 2L
$lstm_cell_11/StatefulPartitionedCall$lstm_cell_11/StatefulPartitionedCall2
whilewhile:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿº
 
_user_specified_nameinputs
Ô
I
-__inference_dropout_11_layer_call_fn_19558386

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
:ÿÿÿÿÿÿÿÿÿS* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_dropout_11_layer_call_and_return_conditional_losses_195557092
PartitionedCallp
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿS:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿS
 
_user_specified_nameinputs
?
Ô
while_body_19558236
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
3while_lstm_cell_11_matmul_readvariableop_resource_0:
ºÌH
5while_lstm_cell_11_matmul_1_readvariableop_resource_0:	SÌC
4while_lstm_cell_11_biasadd_readvariableop_resource_0:	Ì
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
1while_lstm_cell_11_matmul_readvariableop_resource:
ºÌF
3while_lstm_cell_11_matmul_1_readvariableop_resource:	SÌA
2while_lstm_cell_11_biasadd_readvariableop_resource:	Ì¢)while/lstm_cell_11/BiasAdd/ReadVariableOp¢(while/lstm_cell_11/MatMul/ReadVariableOp¢*while/lstm_cell_11/MatMul_1/ReadVariableOpÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ:  29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÔ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemÊ
(while/lstm_cell_11/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_11_matmul_readvariableop_resource_0* 
_output_shapes
:
ºÌ*
dtype02*
(while/lstm_cell_11/MatMul/ReadVariableOp×
while/lstm_cell_11/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_11/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÌ2
while/lstm_cell_11/MatMulÏ
*while/lstm_cell_11/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_11_matmul_1_readvariableop_resource_0*
_output_shapes
:	SÌ*
dtype02,
*while/lstm_cell_11/MatMul_1/ReadVariableOpÀ
while/lstm_cell_11/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_11/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÌ2
while/lstm_cell_11/MatMul_1¸
while/lstm_cell_11/addAddV2#while/lstm_cell_11/MatMul:product:0%while/lstm_cell_11/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÌ2
while/lstm_cell_11/addÈ
)while/lstm_cell_11/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_11_biasadd_readvariableop_resource_0*
_output_shapes	
:Ì*
dtype02+
)while/lstm_cell_11/BiasAdd/ReadVariableOpÅ
while/lstm_cell_11/BiasAddBiasAddwhile/lstm_cell_11/add:z:01while/lstm_cell_11/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÌ2
while/lstm_cell_11/BiasAdd
"while/lstm_cell_11/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_11/split/split_dim
while/lstm_cell_11/splitSplit+while/lstm_cell_11/split/split_dim:output:0#while/lstm_cell_11/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿS:ÿÿÿÿÿÿÿÿÿS:ÿÿÿÿÿÿÿÿÿS:ÿÿÿÿÿÿÿÿÿS*
	num_split2
while/lstm_cell_11/split
while/lstm_cell_11/SigmoidSigmoid!while/lstm_cell_11/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2
while/lstm_cell_11/Sigmoid
while/lstm_cell_11/Sigmoid_1Sigmoid!while/lstm_cell_11/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2
while/lstm_cell_11/Sigmoid_1 
while/lstm_cell_11/mulMul while/lstm_cell_11/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2
while/lstm_cell_11/mul
while/lstm_cell_11/ReluRelu!while/lstm_cell_11/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2
while/lstm_cell_11/Relu´
while/lstm_cell_11/mul_1Mulwhile/lstm_cell_11/Sigmoid:y:0%while/lstm_cell_11/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2
while/lstm_cell_11/mul_1©
while/lstm_cell_11/add_1AddV2while/lstm_cell_11/mul:z:0while/lstm_cell_11/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2
while/lstm_cell_11/add_1
while/lstm_cell_11/Sigmoid_2Sigmoid!while/lstm_cell_11/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2
while/lstm_cell_11/Sigmoid_2
while/lstm_cell_11/Relu_1Reluwhile/lstm_cell_11/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2
while/lstm_cell_11/Relu_1¸
while/lstm_cell_11/mul_2Mul while/lstm_cell_11/Sigmoid_2:y:0'while/lstm_cell_11/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2
while/lstm_cell_11/mul_2à
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
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_11/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_11/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2
while/Identity_5Þ

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
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿS:ÿÿÿÿÿÿÿÿÿS: : : : : 2V
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
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS:

_output_shapes
: :

_output_shapes
: 
°?
Ô
while_body_19557259
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_10_matmul_readvariableop_resource_0:	]è	I
5while_lstm_cell_10_matmul_1_readvariableop_resource_0:
ºè	C
4while_lstm_cell_10_biasadd_readvariableop_resource_0:	è	
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_10_matmul_readvariableop_resource:	]è	G
3while_lstm_cell_10_matmul_1_readvariableop_resource:
ºè	A
2while_lstm_cell_10_biasadd_readvariableop_resource:	è	¢)while/lstm_cell_10/BiasAdd/ReadVariableOp¢(while/lstm_cell_10/MatMul/ReadVariableOp¢*while/lstm_cell_10/MatMul_1/ReadVariableOpÃ
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
(while/lstm_cell_10/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_10_matmul_readvariableop_resource_0*
_output_shapes
:	]è	*
dtype02*
(while/lstm_cell_10/MatMul/ReadVariableOp×
while/lstm_cell_10/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_10/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè	2
while/lstm_cell_10/MatMulÐ
*while/lstm_cell_10/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_10_matmul_1_readvariableop_resource_0* 
_output_shapes
:
ºè	*
dtype02,
*while/lstm_cell_10/MatMul_1/ReadVariableOpÀ
while/lstm_cell_10/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_10/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè	2
while/lstm_cell_10/MatMul_1¸
while/lstm_cell_10/addAddV2#while/lstm_cell_10/MatMul:product:0%while/lstm_cell_10/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè	2
while/lstm_cell_10/addÈ
)while/lstm_cell_10/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_10_biasadd_readvariableop_resource_0*
_output_shapes	
:è	*
dtype02+
)while/lstm_cell_10/BiasAdd/ReadVariableOpÅ
while/lstm_cell_10/BiasAddBiasAddwhile/lstm_cell_10/add:z:01while/lstm_cell_10/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè	2
while/lstm_cell_10/BiasAdd
"while/lstm_cell_10/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_10/split/split_dim
while/lstm_cell_10/splitSplit+while/lstm_cell_10/split/split_dim:output:0#while/lstm_cell_10/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿº:ÿÿÿÿÿÿÿÿÿº:ÿÿÿÿÿÿÿÿÿº:ÿÿÿÿÿÿÿÿÿº*
	num_split2
while/lstm_cell_10/split
while/lstm_cell_10/SigmoidSigmoid!while/lstm_cell_10/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2
while/lstm_cell_10/Sigmoid
while/lstm_cell_10/Sigmoid_1Sigmoid!while/lstm_cell_10/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2
while/lstm_cell_10/Sigmoid_1¡
while/lstm_cell_10/mulMul while/lstm_cell_10/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2
while/lstm_cell_10/mul
while/lstm_cell_10/ReluRelu!while/lstm_cell_10/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2
while/lstm_cell_10/Reluµ
while/lstm_cell_10/mul_1Mulwhile/lstm_cell_10/Sigmoid:y:0%while/lstm_cell_10/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2
while/lstm_cell_10/mul_1ª
while/lstm_cell_10/add_1AddV2while/lstm_cell_10/mul:z:0while/lstm_cell_10/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2
while/lstm_cell_10/add_1
while/lstm_cell_10/Sigmoid_2Sigmoid!while/lstm_cell_10/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2
while/lstm_cell_10/Sigmoid_2
while/lstm_cell_10/Relu_1Reluwhile/lstm_cell_10/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2
while/lstm_cell_10/Relu_1¹
while/lstm_cell_10/mul_2Mul while/lstm_cell_10/Sigmoid_2:y:0'while/lstm_cell_10/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2
while/lstm_cell_10/mul_2à
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
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_10/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_10/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2
while/Identity_5Þ

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
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿº:ÿÿÿÿÿÿÿÿÿº: : : : : 2V
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
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº:

_output_shapes
: :

_output_shapes
: 
&
ó
while_body_19554203
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_00
while_lstm_cell_10_19554227_0:	]è	1
while_lstm_cell_10_19554229_0:
ºè	,
while_lstm_cell_10_19554231_0:	è	
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor.
while_lstm_cell_10_19554227:	]è	/
while_lstm_cell_10_19554229:
ºè	*
while_lstm_cell_10_19554231:	è	¢*while/lstm_cell_10/StatefulPartitionedCallÃ
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
*while/lstm_cell_10/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_10_19554227_0while_lstm_cell_10_19554229_0while_lstm_cell_10_19554231_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿº:ÿÿÿÿÿÿÿÿÿº:ÿÿÿÿÿÿÿÿÿº*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_lstm_cell_10_layer_call_and_return_conditional_losses_195541892,
*while/lstm_cell_10/StatefulPartitionedCall÷
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
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3¥
while/Identity_4Identity3while/lstm_cell_10/StatefulPartitionedCall:output:1^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2
while/Identity_4¥
while/Identity_5Identity3while/lstm_cell_10/StatefulPartitionedCall:output:2^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2
while/Identity_5

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
while_lstm_cell_10_19554227while_lstm_cell_10_19554227_0"<
while_lstm_cell_10_19554229while_lstm_cell_10_19554229_0"<
while_lstm_cell_10_19554231while_lstm_cell_10_19554231_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿº:ÿÿÿÿÿÿÿÿÿº: : : : : 2X
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
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº:

_output_shapes
: :

_output_shapes
: 

f
H__inference_dropout_11_layer_call_and_return_conditional_losses_19558369

inputs

identity_1^
IdentityIdentityinputs*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2

Identitym

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿS:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿS
 
_user_specified_nameinputs
ËF

E__inference_lstm_10_layer_call_and_return_conditional_losses_19554482

inputs(
lstm_cell_10_19554400:	]è	)
lstm_cell_10_19554402:
ºè	$
lstm_cell_10_19554404:	è	
identity¢$lstm_cell_10/StatefulPartitionedCall¢whileD
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
B :º2
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
B :º2
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
:ÿÿÿÿÿÿÿÿÿº2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :º2
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
B :º2
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
:ÿÿÿÿÿÿÿÿÿº2	
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
$lstm_cell_10/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_10_19554400lstm_cell_10_19554402lstm_cell_10_19554404*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿº:ÿÿÿÿÿÿÿÿÿº:ÿÿÿÿÿÿÿÿÿº*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_lstm_cell_10_layer_call_and_return_conditional_losses_195543352&
$lstm_cell_10/StatefulPartitionedCall
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ:  2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_10_19554400lstm_cell_10_19554402lstm_cell_10_19554404*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿº:ÿÿÿÿÿÿÿÿÿº: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_19554413*
condR
while_cond_19554412*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿº:ÿÿÿÿÿÿÿÿÿº: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ:  22
0TensorArrayV2Stack/TensorListStack/element_shapeò
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿº*
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
:ÿÿÿÿÿÿÿÿÿº*
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
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿº2
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
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿº2

Identity}
NoOpNoOp%^lstm_cell_10/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ]: : : 2L
$lstm_cell_10/StatefulPartitionedCall$lstm_cell_10/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ]
 
_user_specified_nameinputs
?
Ô
while_body_19557934
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
3while_lstm_cell_11_matmul_readvariableop_resource_0:
ºÌH
5while_lstm_cell_11_matmul_1_readvariableop_resource_0:	SÌC
4while_lstm_cell_11_biasadd_readvariableop_resource_0:	Ì
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
1while_lstm_cell_11_matmul_readvariableop_resource:
ºÌF
3while_lstm_cell_11_matmul_1_readvariableop_resource:	SÌA
2while_lstm_cell_11_biasadd_readvariableop_resource:	Ì¢)while/lstm_cell_11/BiasAdd/ReadVariableOp¢(while/lstm_cell_11/MatMul/ReadVariableOp¢*while/lstm_cell_11/MatMul_1/ReadVariableOpÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ:  29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÔ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemÊ
(while/lstm_cell_11/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_11_matmul_readvariableop_resource_0* 
_output_shapes
:
ºÌ*
dtype02*
(while/lstm_cell_11/MatMul/ReadVariableOp×
while/lstm_cell_11/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_11/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÌ2
while/lstm_cell_11/MatMulÏ
*while/lstm_cell_11/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_11_matmul_1_readvariableop_resource_0*
_output_shapes
:	SÌ*
dtype02,
*while/lstm_cell_11/MatMul_1/ReadVariableOpÀ
while/lstm_cell_11/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_11/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÌ2
while/lstm_cell_11/MatMul_1¸
while/lstm_cell_11/addAddV2#while/lstm_cell_11/MatMul:product:0%while/lstm_cell_11/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÌ2
while/lstm_cell_11/addÈ
)while/lstm_cell_11/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_11_biasadd_readvariableop_resource_0*
_output_shapes	
:Ì*
dtype02+
)while/lstm_cell_11/BiasAdd/ReadVariableOpÅ
while/lstm_cell_11/BiasAddBiasAddwhile/lstm_cell_11/add:z:01while/lstm_cell_11/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÌ2
while/lstm_cell_11/BiasAdd
"while/lstm_cell_11/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_11/split/split_dim
while/lstm_cell_11/splitSplit+while/lstm_cell_11/split/split_dim:output:0#while/lstm_cell_11/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿS:ÿÿÿÿÿÿÿÿÿS:ÿÿÿÿÿÿÿÿÿS:ÿÿÿÿÿÿÿÿÿS*
	num_split2
while/lstm_cell_11/split
while/lstm_cell_11/SigmoidSigmoid!while/lstm_cell_11/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2
while/lstm_cell_11/Sigmoid
while/lstm_cell_11/Sigmoid_1Sigmoid!while/lstm_cell_11/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2
while/lstm_cell_11/Sigmoid_1 
while/lstm_cell_11/mulMul while/lstm_cell_11/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2
while/lstm_cell_11/mul
while/lstm_cell_11/ReluRelu!while/lstm_cell_11/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2
while/lstm_cell_11/Relu´
while/lstm_cell_11/mul_1Mulwhile/lstm_cell_11/Sigmoid:y:0%while/lstm_cell_11/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2
while/lstm_cell_11/mul_1©
while/lstm_cell_11/add_1AddV2while/lstm_cell_11/mul:z:0while/lstm_cell_11/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2
while/lstm_cell_11/add_1
while/lstm_cell_11/Sigmoid_2Sigmoid!while/lstm_cell_11/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2
while/lstm_cell_11/Sigmoid_2
while/lstm_cell_11/Relu_1Reluwhile/lstm_cell_11/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2
while/lstm_cell_11/Relu_1¸
while/lstm_cell_11/mul_2Mul while/lstm_cell_11/Sigmoid_2:y:0'while/lstm_cell_11/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2
while/lstm_cell_11/mul_2à
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
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_11/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_11/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2
while/Identity_5Þ

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
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿS:ÿÿÿÿÿÿÿÿÿS: : : : : 2V
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
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS:

_output_shapes
: :

_output_shapes
: 


Ê
/__inference_sequential_5_layer_call_fn_19557020

inputs
unknown:	]è	
	unknown_0:
ºè	
	unknown_1:	è	
	unknown_2:
ºÌ
	unknown_3:	SÌ
	unknown_4:	Ì
	unknown_5:S
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
J__inference_sequential_5_layer_call_and_return_conditional_losses_195557492
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
ß
Í
while_cond_19555880
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_19555880___redundant_placeholder06
2while_while_cond_19555880___redundant_placeholder16
2while_while_cond_19555880___redundant_placeholder26
2while_while_cond_19555880___redundant_placeholder3
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
@: : : : :ÿÿÿÿÿÿÿÿÿS:ÿÿÿÿÿÿÿÿÿS: ::::: 

_output_shapes
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
:ÿÿÿÿÿÿÿÿÿS:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS:

_output_shapes
: :

_output_shapes
:
í[

E__inference_lstm_11_layer_call_and_return_conditional_losses_19558320

inputs?
+lstm_cell_11_matmul_readvariableop_resource:
ºÌ@
-lstm_cell_11_matmul_1_readvariableop_resource:	SÌ;
,lstm_cell_11_biasadd_readvariableop_resource:	Ì
identity¢#lstm_cell_11/BiasAdd/ReadVariableOp¢"lstm_cell_11/MatMul/ReadVariableOp¢$lstm_cell_11/MatMul_1/ReadVariableOp¢whileD
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
value	B :S2
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
value	B :S2
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
:ÿÿÿÿÿÿÿÿÿS2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :S2
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
value	B :S2
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
:ÿÿÿÿÿÿÿÿÿS2	
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
:ÿÿÿÿÿÿÿÿÿº2
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
valueB"ÿÿÿÿ:  27
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
:ÿÿÿÿÿÿÿÿÿº*
shrink_axis_mask2
strided_slice_2¶
"lstm_cell_11/MatMul/ReadVariableOpReadVariableOp+lstm_cell_11_matmul_readvariableop_resource* 
_output_shapes
:
ºÌ*
dtype02$
"lstm_cell_11/MatMul/ReadVariableOp­
lstm_cell_11/MatMulMatMulstrided_slice_2:output:0*lstm_cell_11/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÌ2
lstm_cell_11/MatMul»
$lstm_cell_11/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_11_matmul_1_readvariableop_resource*
_output_shapes
:	SÌ*
dtype02&
$lstm_cell_11/MatMul_1/ReadVariableOp©
lstm_cell_11/MatMul_1MatMulzeros:output:0,lstm_cell_11/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÌ2
lstm_cell_11/MatMul_1 
lstm_cell_11/addAddV2lstm_cell_11/MatMul:product:0lstm_cell_11/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÌ2
lstm_cell_11/add´
#lstm_cell_11/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_11_biasadd_readvariableop_resource*
_output_shapes	
:Ì*
dtype02%
#lstm_cell_11/BiasAdd/ReadVariableOp­
lstm_cell_11/BiasAddBiasAddlstm_cell_11/add:z:0+lstm_cell_11/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÌ2
lstm_cell_11/BiasAdd~
lstm_cell_11/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_11/split/split_dimó
lstm_cell_11/splitSplit%lstm_cell_11/split/split_dim:output:0lstm_cell_11/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿS:ÿÿÿÿÿÿÿÿÿS:ÿÿÿÿÿÿÿÿÿS:ÿÿÿÿÿÿÿÿÿS*
	num_split2
lstm_cell_11/split
lstm_cell_11/SigmoidSigmoidlstm_cell_11/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2
lstm_cell_11/Sigmoid
lstm_cell_11/Sigmoid_1Sigmoidlstm_cell_11/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2
lstm_cell_11/Sigmoid_1
lstm_cell_11/mulMullstm_cell_11/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2
lstm_cell_11/mul}
lstm_cell_11/ReluRelulstm_cell_11/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2
lstm_cell_11/Relu
lstm_cell_11/mul_1Mullstm_cell_11/Sigmoid:y:0lstm_cell_11/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2
lstm_cell_11/mul_1
lstm_cell_11/add_1AddV2lstm_cell_11/mul:z:0lstm_cell_11/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2
lstm_cell_11/add_1
lstm_cell_11/Sigmoid_2Sigmoidlstm_cell_11/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2
lstm_cell_11/Sigmoid_2|
lstm_cell_11/Relu_1Relulstm_cell_11/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2
lstm_cell_11/Relu_1 
lstm_cell_11/mul_2Mullstm_cell_11/Sigmoid_2:y:0!lstm_cell_11/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2
lstm_cell_11/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿS   2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_11_matmul_readvariableop_resource-lstm_cell_11_matmul_1_readvariableop_resource,lstm_cell_11_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿS:ÿÿÿÿÿÿÿÿÿS: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_19558236*
condR
while_cond_19558235*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿS:ÿÿÿÿÿÿÿÿÿS: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿS   22
0TensorArrayV2Stack/TensorListStack/element_shapeè
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿS*
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
:ÿÿÿÿÿÿÿÿÿS*
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
:ÿÿÿÿÿÿÿÿÿS2
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
:ÿÿÿÿÿÿÿÿÿS2

IdentityÈ
NoOpNoOp$^lstm_cell_11/BiasAdd/ReadVariableOp#^lstm_cell_11/MatMul/ReadVariableOp%^lstm_cell_11/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:ÿÿÿÿÿÿÿÿÿº: : : 2J
#lstm_cell_11/BiasAdd/ReadVariableOp#lstm_cell_11/BiasAdd/ReadVariableOp2H
"lstm_cell_11/MatMul/ReadVariableOp"lstm_cell_11/MatMul/ReadVariableOp2L
$lstm_cell_11/MatMul_1/ReadVariableOp$lstm_cell_11/MatMul_1/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿº
 
_user_specified_nameinputs
ã
Í
while_cond_19557107
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_19557107___redundant_placeholder06
2while_while_cond_19557107___redundant_placeholder16
2while_while_cond_19557107___redundant_placeholder26
2while_while_cond_19557107___redundant_placeholder3
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
B: : : : :ÿÿÿÿÿÿÿÿÿº:ÿÿÿÿÿÿÿÿÿº: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº:

_output_shapes
: :

_output_shapes
:
¶
f
-__inference_dropout_11_layer_call_fn_19558391

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
:ÿÿÿÿÿÿÿÿÿS* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_dropout_11_layer_call_and_return_conditional_losses_195557982
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿS22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿS
 
_user_specified_nameinputs
¶
¸
*__inference_lstm_10_layer_call_fn_19557678

inputs
unknown:	]è	
	unknown_0:
ºè	
	unknown_1:	è	
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿº*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_lstm_10_layer_call_and_return_conditional_losses_195555312
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2

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
while_cond_19557258
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_19557258___redundant_placeholder06
2while_while_cond_19557258___redundant_placeholder16
2while_while_cond_19557258___redundant_placeholder26
2while_while_cond_19557258___redundant_placeholder3
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
B: : : : :ÿÿÿÿÿÿÿÿÿº:ÿÿÿÿÿÿÿÿÿº: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº:

_output_shapes
: :

_output_shapes
:

f
H__inference_dropout_11_layer_call_and_return_conditional_losses_19555709

inputs

identity_1^
IdentityIdentityinputs*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2

Identitym

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿS:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿS
 
_user_specified_nameinputs
?
Ô
while_body_19557783
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
3while_lstm_cell_11_matmul_readvariableop_resource_0:
ºÌH
5while_lstm_cell_11_matmul_1_readvariableop_resource_0:	SÌC
4while_lstm_cell_11_biasadd_readvariableop_resource_0:	Ì
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
1while_lstm_cell_11_matmul_readvariableop_resource:
ºÌF
3while_lstm_cell_11_matmul_1_readvariableop_resource:	SÌA
2while_lstm_cell_11_biasadd_readvariableop_resource:	Ì¢)while/lstm_cell_11/BiasAdd/ReadVariableOp¢(while/lstm_cell_11/MatMul/ReadVariableOp¢*while/lstm_cell_11/MatMul_1/ReadVariableOpÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ:  29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÔ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemÊ
(while/lstm_cell_11/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_11_matmul_readvariableop_resource_0* 
_output_shapes
:
ºÌ*
dtype02*
(while/lstm_cell_11/MatMul/ReadVariableOp×
while/lstm_cell_11/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_11/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÌ2
while/lstm_cell_11/MatMulÏ
*while/lstm_cell_11/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_11_matmul_1_readvariableop_resource_0*
_output_shapes
:	SÌ*
dtype02,
*while/lstm_cell_11/MatMul_1/ReadVariableOpÀ
while/lstm_cell_11/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_11/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÌ2
while/lstm_cell_11/MatMul_1¸
while/lstm_cell_11/addAddV2#while/lstm_cell_11/MatMul:product:0%while/lstm_cell_11/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÌ2
while/lstm_cell_11/addÈ
)while/lstm_cell_11/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_11_biasadd_readvariableop_resource_0*
_output_shapes	
:Ì*
dtype02+
)while/lstm_cell_11/BiasAdd/ReadVariableOpÅ
while/lstm_cell_11/BiasAddBiasAddwhile/lstm_cell_11/add:z:01while/lstm_cell_11/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÌ2
while/lstm_cell_11/BiasAdd
"while/lstm_cell_11/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_11/split/split_dim
while/lstm_cell_11/splitSplit+while/lstm_cell_11/split/split_dim:output:0#while/lstm_cell_11/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿS:ÿÿÿÿÿÿÿÿÿS:ÿÿÿÿÿÿÿÿÿS:ÿÿÿÿÿÿÿÿÿS*
	num_split2
while/lstm_cell_11/split
while/lstm_cell_11/SigmoidSigmoid!while/lstm_cell_11/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2
while/lstm_cell_11/Sigmoid
while/lstm_cell_11/Sigmoid_1Sigmoid!while/lstm_cell_11/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2
while/lstm_cell_11/Sigmoid_1 
while/lstm_cell_11/mulMul while/lstm_cell_11/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2
while/lstm_cell_11/mul
while/lstm_cell_11/ReluRelu!while/lstm_cell_11/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2
while/lstm_cell_11/Relu´
while/lstm_cell_11/mul_1Mulwhile/lstm_cell_11/Sigmoid:y:0%while/lstm_cell_11/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2
while/lstm_cell_11/mul_1©
while/lstm_cell_11/add_1AddV2while/lstm_cell_11/mul:z:0while/lstm_cell_11/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2
while/lstm_cell_11/add_1
while/lstm_cell_11/Sigmoid_2Sigmoid!while/lstm_cell_11/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2
while/lstm_cell_11/Sigmoid_2
while/lstm_cell_11/Relu_1Reluwhile/lstm_cell_11/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2
while/lstm_cell_11/Relu_1¸
while/lstm_cell_11/mul_2Mul while/lstm_cell_11/Sigmoid_2:y:0'while/lstm_cell_11/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2
while/lstm_cell_11/mul_2à
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
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_11/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_11/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2
while/Identity_5Þ

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
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿS:ÿÿÿÿÿÿÿÿÿS: : : : : 2V
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
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS:

_output_shapes
: :

_output_shapes
: 
]
õ
(sequential_5_lstm_11_while_body_19554002F
Bsequential_5_lstm_11_while_sequential_5_lstm_11_while_loop_counterL
Hsequential_5_lstm_11_while_sequential_5_lstm_11_while_maximum_iterations*
&sequential_5_lstm_11_while_placeholder,
(sequential_5_lstm_11_while_placeholder_1,
(sequential_5_lstm_11_while_placeholder_2,
(sequential_5_lstm_11_while_placeholder_3E
Asequential_5_lstm_11_while_sequential_5_lstm_11_strided_slice_1_0
}sequential_5_lstm_11_while_tensorarrayv2read_tensorlistgetitem_sequential_5_lstm_11_tensorarrayunstack_tensorlistfromtensor_0\
Hsequential_5_lstm_11_while_lstm_cell_11_matmul_readvariableop_resource_0:
ºÌ]
Jsequential_5_lstm_11_while_lstm_cell_11_matmul_1_readvariableop_resource_0:	SÌX
Isequential_5_lstm_11_while_lstm_cell_11_biasadd_readvariableop_resource_0:	Ì'
#sequential_5_lstm_11_while_identity)
%sequential_5_lstm_11_while_identity_1)
%sequential_5_lstm_11_while_identity_2)
%sequential_5_lstm_11_while_identity_3)
%sequential_5_lstm_11_while_identity_4)
%sequential_5_lstm_11_while_identity_5C
?sequential_5_lstm_11_while_sequential_5_lstm_11_strided_slice_1
{sequential_5_lstm_11_while_tensorarrayv2read_tensorlistgetitem_sequential_5_lstm_11_tensorarrayunstack_tensorlistfromtensorZ
Fsequential_5_lstm_11_while_lstm_cell_11_matmul_readvariableop_resource:
ºÌ[
Hsequential_5_lstm_11_while_lstm_cell_11_matmul_1_readvariableop_resource:	SÌV
Gsequential_5_lstm_11_while_lstm_cell_11_biasadd_readvariableop_resource:	Ì¢>sequential_5/lstm_11/while/lstm_cell_11/BiasAdd/ReadVariableOp¢=sequential_5/lstm_11/while/lstm_cell_11/MatMul/ReadVariableOp¢?sequential_5/lstm_11/while/lstm_cell_11/MatMul_1/ReadVariableOpí
Lsequential_5/lstm_11/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ:  2N
Lsequential_5/lstm_11/while/TensorArrayV2Read/TensorListGetItem/element_shapeÒ
>sequential_5/lstm_11/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem}sequential_5_lstm_11_while_tensorarrayv2read_tensorlistgetitem_sequential_5_lstm_11_tensorarrayunstack_tensorlistfromtensor_0&sequential_5_lstm_11_while_placeholderUsequential_5/lstm_11/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº*
element_dtype02@
>sequential_5/lstm_11/while/TensorArrayV2Read/TensorListGetItem
=sequential_5/lstm_11/while/lstm_cell_11/MatMul/ReadVariableOpReadVariableOpHsequential_5_lstm_11_while_lstm_cell_11_matmul_readvariableop_resource_0* 
_output_shapes
:
ºÌ*
dtype02?
=sequential_5/lstm_11/while/lstm_cell_11/MatMul/ReadVariableOp«
.sequential_5/lstm_11/while/lstm_cell_11/MatMulMatMulEsequential_5/lstm_11/while/TensorArrayV2Read/TensorListGetItem:item:0Esequential_5/lstm_11/while/lstm_cell_11/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÌ20
.sequential_5/lstm_11/while/lstm_cell_11/MatMul
?sequential_5/lstm_11/while/lstm_cell_11/MatMul_1/ReadVariableOpReadVariableOpJsequential_5_lstm_11_while_lstm_cell_11_matmul_1_readvariableop_resource_0*
_output_shapes
:	SÌ*
dtype02A
?sequential_5/lstm_11/while/lstm_cell_11/MatMul_1/ReadVariableOp
0sequential_5/lstm_11/while/lstm_cell_11/MatMul_1MatMul(sequential_5_lstm_11_while_placeholder_2Gsequential_5/lstm_11/while/lstm_cell_11/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÌ22
0sequential_5/lstm_11/while/lstm_cell_11/MatMul_1
+sequential_5/lstm_11/while/lstm_cell_11/addAddV28sequential_5/lstm_11/while/lstm_cell_11/MatMul:product:0:sequential_5/lstm_11/while/lstm_cell_11/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÌ2-
+sequential_5/lstm_11/while/lstm_cell_11/add
>sequential_5/lstm_11/while/lstm_cell_11/BiasAdd/ReadVariableOpReadVariableOpIsequential_5_lstm_11_while_lstm_cell_11_biasadd_readvariableop_resource_0*
_output_shapes	
:Ì*
dtype02@
>sequential_5/lstm_11/while/lstm_cell_11/BiasAdd/ReadVariableOp
/sequential_5/lstm_11/while/lstm_cell_11/BiasAddBiasAdd/sequential_5/lstm_11/while/lstm_cell_11/add:z:0Fsequential_5/lstm_11/while/lstm_cell_11/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÌ21
/sequential_5/lstm_11/while/lstm_cell_11/BiasAdd´
7sequential_5/lstm_11/while/lstm_cell_11/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :29
7sequential_5/lstm_11/while/lstm_cell_11/split/split_dimß
-sequential_5/lstm_11/while/lstm_cell_11/splitSplit@sequential_5/lstm_11/while/lstm_cell_11/split/split_dim:output:08sequential_5/lstm_11/while/lstm_cell_11/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿS:ÿÿÿÿÿÿÿÿÿS:ÿÿÿÿÿÿÿÿÿS:ÿÿÿÿÿÿÿÿÿS*
	num_split2/
-sequential_5/lstm_11/while/lstm_cell_11/split×
/sequential_5/lstm_11/while/lstm_cell_11/SigmoidSigmoid6sequential_5/lstm_11/while/lstm_cell_11/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS21
/sequential_5/lstm_11/while/lstm_cell_11/SigmoidÛ
1sequential_5/lstm_11/while/lstm_cell_11/Sigmoid_1Sigmoid6sequential_5/lstm_11/while/lstm_cell_11/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS23
1sequential_5/lstm_11/while/lstm_cell_11/Sigmoid_1ô
+sequential_5/lstm_11/while/lstm_cell_11/mulMul5sequential_5/lstm_11/while/lstm_cell_11/Sigmoid_1:y:0(sequential_5_lstm_11_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2-
+sequential_5/lstm_11/while/lstm_cell_11/mulÎ
,sequential_5/lstm_11/while/lstm_cell_11/ReluRelu6sequential_5/lstm_11/while/lstm_cell_11/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2.
,sequential_5/lstm_11/while/lstm_cell_11/Relu
-sequential_5/lstm_11/while/lstm_cell_11/mul_1Mul3sequential_5/lstm_11/while/lstm_cell_11/Sigmoid:y:0:sequential_5/lstm_11/while/lstm_cell_11/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2/
-sequential_5/lstm_11/while/lstm_cell_11/mul_1ý
-sequential_5/lstm_11/while/lstm_cell_11/add_1AddV2/sequential_5/lstm_11/while/lstm_cell_11/mul:z:01sequential_5/lstm_11/while/lstm_cell_11/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2/
-sequential_5/lstm_11/while/lstm_cell_11/add_1Û
1sequential_5/lstm_11/while/lstm_cell_11/Sigmoid_2Sigmoid6sequential_5/lstm_11/while/lstm_cell_11/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS23
1sequential_5/lstm_11/while/lstm_cell_11/Sigmoid_2Í
.sequential_5/lstm_11/while/lstm_cell_11/Relu_1Relu1sequential_5/lstm_11/while/lstm_cell_11/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS20
.sequential_5/lstm_11/while/lstm_cell_11/Relu_1
-sequential_5/lstm_11/while/lstm_cell_11/mul_2Mul5sequential_5/lstm_11/while/lstm_cell_11/Sigmoid_2:y:0<sequential_5/lstm_11/while/lstm_cell_11/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2/
-sequential_5/lstm_11/while/lstm_cell_11/mul_2É
?sequential_5/lstm_11/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem(sequential_5_lstm_11_while_placeholder_1&sequential_5_lstm_11_while_placeholder1sequential_5/lstm_11/while/lstm_cell_11/mul_2:z:0*
_output_shapes
: *
element_dtype02A
?sequential_5/lstm_11/while/TensorArrayV2Write/TensorListSetItem
 sequential_5/lstm_11/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2"
 sequential_5/lstm_11/while/add/y½
sequential_5/lstm_11/while/addAddV2&sequential_5_lstm_11_while_placeholder)sequential_5/lstm_11/while/add/y:output:0*
T0*
_output_shapes
: 2 
sequential_5/lstm_11/while/add
"sequential_5/lstm_11/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2$
"sequential_5/lstm_11/while/add_1/yß
 sequential_5/lstm_11/while/add_1AddV2Bsequential_5_lstm_11_while_sequential_5_lstm_11_while_loop_counter+sequential_5/lstm_11/while/add_1/y:output:0*
T0*
_output_shapes
: 2"
 sequential_5/lstm_11/while/add_1¿
#sequential_5/lstm_11/while/IdentityIdentity$sequential_5/lstm_11/while/add_1:z:0 ^sequential_5/lstm_11/while/NoOp*
T0*
_output_shapes
: 2%
#sequential_5/lstm_11/while/Identityç
%sequential_5/lstm_11/while/Identity_1IdentityHsequential_5_lstm_11_while_sequential_5_lstm_11_while_maximum_iterations ^sequential_5/lstm_11/while/NoOp*
T0*
_output_shapes
: 2'
%sequential_5/lstm_11/while/Identity_1Á
%sequential_5/lstm_11/while/Identity_2Identity"sequential_5/lstm_11/while/add:z:0 ^sequential_5/lstm_11/while/NoOp*
T0*
_output_shapes
: 2'
%sequential_5/lstm_11/while/Identity_2î
%sequential_5/lstm_11/while/Identity_3IdentityOsequential_5/lstm_11/while/TensorArrayV2Write/TensorListSetItem:output_handle:0 ^sequential_5/lstm_11/while/NoOp*
T0*
_output_shapes
: 2'
%sequential_5/lstm_11/while/Identity_3á
%sequential_5/lstm_11/while/Identity_4Identity1sequential_5/lstm_11/while/lstm_cell_11/mul_2:z:0 ^sequential_5/lstm_11/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2'
%sequential_5/lstm_11/while/Identity_4á
%sequential_5/lstm_11/while/Identity_5Identity1sequential_5/lstm_11/while/lstm_cell_11/add_1:z:0 ^sequential_5/lstm_11/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2'
%sequential_5/lstm_11/while/Identity_5Ç
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
%sequential_5_lstm_11_while_identity_5.sequential_5/lstm_11/while/Identity_5:output:0"
Gsequential_5_lstm_11_while_lstm_cell_11_biasadd_readvariableop_resourceIsequential_5_lstm_11_while_lstm_cell_11_biasadd_readvariableop_resource_0"
Hsequential_5_lstm_11_while_lstm_cell_11_matmul_1_readvariableop_resourceJsequential_5_lstm_11_while_lstm_cell_11_matmul_1_readvariableop_resource_0"
Fsequential_5_lstm_11_while_lstm_cell_11_matmul_readvariableop_resourceHsequential_5_lstm_11_while_lstm_cell_11_matmul_readvariableop_resource_0"
?sequential_5_lstm_11_while_sequential_5_lstm_11_strided_slice_1Asequential_5_lstm_11_while_sequential_5_lstm_11_strided_slice_1_0"ü
{sequential_5_lstm_11_while_tensorarrayv2read_tensorlistgetitem_sequential_5_lstm_11_tensorarrayunstack_tensorlistfromtensor}sequential_5_lstm_11_while_tensorarrayv2read_tensorlistgetitem_sequential_5_lstm_11_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿS:ÿÿÿÿÿÿÿÿÿS: : : : : 2
>sequential_5/lstm_11/while/lstm_cell_11/BiasAdd/ReadVariableOp>sequential_5/lstm_11/while/lstm_cell_11/BiasAdd/ReadVariableOp2~
=sequential_5/lstm_11/while/lstm_cell_11/MatMul/ReadVariableOp=sequential_5/lstm_11/while/lstm_cell_11/MatMul/ReadVariableOp2
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
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS:

_output_shapes
: :

_output_shapes
: 
µ
¸
*__inference_lstm_11_layer_call_fn_19558353

inputs
unknown:
ºÌ
	unknown_0:	SÌ
	unknown_1:	Ì
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿS*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_lstm_11_layer_call_and_return_conditional_losses_195556962
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:ÿÿÿÿÿÿÿÿÿº: : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿº
 
_user_specified_nameinputs


J__inference_lstm_cell_11_layer_call_and_return_conditional_losses_19558561

inputs
states_0
states_12
matmul_readvariableop_resource:
ºÌ3
 matmul_1_readvariableop_resource:	SÌ.
biasadd_readvariableop_resource:	Ì
identity

identity_1

identity_2¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
ºÌ*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÌ2
MatMul
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	SÌ*
dtype02
MatMul_1/ReadVariableOp|
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÌ2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÌ2
add
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:Ì*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÌ2	
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
L:ÿÿÿÿÿÿÿÿÿS:ÿÿÿÿÿÿÿÿÿS:ÿÿÿÿÿÿÿÿÿS:ÿÿÿÿÿÿÿÿÿS*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2
	Sigmoid_1\
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2
mulV
ReluRelusplit:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2
	Sigmoid_2U
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2
mul_2d
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2

Identityh

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2

Identity_1h

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2

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
@:ÿÿÿÿÿÿÿÿÿº:ÿÿÿÿÿÿÿÿÿS:ÿÿÿÿÿÿÿÿÿS: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº
 
_user_specified_nameinputs:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS
"
_user_specified_name
states/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS
"
_user_specified_name
states/1
È7
Û
$__inference__traced_restore_19558732
file_prefix1
assignvariableop_dense_5_kernel:S-
assignvariableop_1_dense_5_bias:A
.assignvariableop_2_lstm_10_lstm_cell_10_kernel:	]è	L
8assignvariableop_3_lstm_10_lstm_cell_10_recurrent_kernel:
ºè	;
,assignvariableop_4_lstm_10_lstm_cell_10_bias:	è	B
.assignvariableop_5_lstm_11_lstm_cell_11_kernel:
ºÌK
8assignvariableop_6_lstm_11_lstm_cell_11_recurrent_kernel:	SÌ;
,assignvariableop_7_lstm_11_lstm_cell_11_bias:	Ì"
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
AssignVariableOpAssignVariableOpassignvariableop_dense_5_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1¤
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_5_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2³
AssignVariableOp_2AssignVariableOp.assignvariableop_2_lstm_10_lstm_cell_10_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3½
AssignVariableOp_3AssignVariableOp8assignvariableop_3_lstm_10_lstm_cell_10_recurrent_kernelIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4±
AssignVariableOp_4AssignVariableOp,assignvariableop_4_lstm_10_lstm_cell_10_biasIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5³
AssignVariableOp_5AssignVariableOp.assignvariableop_5_lstm_11_lstm_cell_11_kernelIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6½
AssignVariableOp_6AssignVariableOp8assignvariableop_6_lstm_11_lstm_cell_11_recurrent_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7±
AssignVariableOp_7AssignVariableOp,assignvariableop_7_lstm_11_lstm_cell_11_biasIdentity_7:output:0"/device:CPU:0*
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
è³
Ï	
#__inference__wrapped_model_19554114
lstm_10_inputS
@sequential_5_lstm_10_lstm_cell_10_matmul_readvariableop_resource:	]è	V
Bsequential_5_lstm_10_lstm_cell_10_matmul_1_readvariableop_resource:
ºè	P
Asequential_5_lstm_10_lstm_cell_10_biasadd_readvariableop_resource:	è	T
@sequential_5_lstm_11_lstm_cell_11_matmul_readvariableop_resource:
ºÌU
Bsequential_5_lstm_11_lstm_cell_11_matmul_1_readvariableop_resource:	SÌP
Asequential_5_lstm_11_lstm_cell_11_biasadd_readvariableop_resource:	ÌH
6sequential_5_dense_5_tensordot_readvariableop_resource:SB
4sequential_5_dense_5_biasadd_readvariableop_resource:
identity¢+sequential_5/dense_5/BiasAdd/ReadVariableOp¢-sequential_5/dense_5/Tensordot/ReadVariableOp¢8sequential_5/lstm_10/lstm_cell_10/BiasAdd/ReadVariableOp¢7sequential_5/lstm_10/lstm_cell_10/MatMul/ReadVariableOp¢9sequential_5/lstm_10/lstm_cell_10/MatMul_1/ReadVariableOp¢sequential_5/lstm_10/while¢8sequential_5/lstm_11/lstm_cell_11/BiasAdd/ReadVariableOp¢7sequential_5/lstm_11/lstm_cell_11/MatMul/ReadVariableOp¢9sequential_5/lstm_11/lstm_cell_11/MatMul_1/ReadVariableOp¢sequential_5/lstm_11/whileu
sequential_5/lstm_10/ShapeShapelstm_10_input*
T0*
_output_shapes
:2
sequential_5/lstm_10/Shape
(sequential_5/lstm_10/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(sequential_5/lstm_10/strided_slice/stack¢
*sequential_5/lstm_10/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*sequential_5/lstm_10/strided_slice/stack_1¢
*sequential_5/lstm_10/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*sequential_5/lstm_10/strided_slice/stack_2à
"sequential_5/lstm_10/strided_sliceStridedSlice#sequential_5/lstm_10/Shape:output:01sequential_5/lstm_10/strided_slice/stack:output:03sequential_5/lstm_10/strided_slice/stack_1:output:03sequential_5/lstm_10/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"sequential_5/lstm_10/strided_slice
 sequential_5/lstm_10/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :º2"
 sequential_5/lstm_10/zeros/mul/yÀ
sequential_5/lstm_10/zeros/mulMul+sequential_5/lstm_10/strided_slice:output:0)sequential_5/lstm_10/zeros/mul/y:output:0*
T0*
_output_shapes
: 2 
sequential_5/lstm_10/zeros/mul
!sequential_5/lstm_10/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2#
!sequential_5/lstm_10/zeros/Less/y»
sequential_5/lstm_10/zeros/LessLess"sequential_5/lstm_10/zeros/mul:z:0*sequential_5/lstm_10/zeros/Less/y:output:0*
T0*
_output_shapes
: 2!
sequential_5/lstm_10/zeros/Less
#sequential_5/lstm_10/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :º2%
#sequential_5/lstm_10/zeros/packed/1×
!sequential_5/lstm_10/zeros/packedPack+sequential_5/lstm_10/strided_slice:output:0,sequential_5/lstm_10/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2#
!sequential_5/lstm_10/zeros/packed
 sequential_5/lstm_10/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 sequential_5/lstm_10/zeros/ConstÊ
sequential_5/lstm_10/zerosFill*sequential_5/lstm_10/zeros/packed:output:0)sequential_5/lstm_10/zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2
sequential_5/lstm_10/zeros
"sequential_5/lstm_10/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :º2$
"sequential_5/lstm_10/zeros_1/mul/yÆ
 sequential_5/lstm_10/zeros_1/mulMul+sequential_5/lstm_10/strided_slice:output:0+sequential_5/lstm_10/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2"
 sequential_5/lstm_10/zeros_1/mul
#sequential_5/lstm_10/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2%
#sequential_5/lstm_10/zeros_1/Less/yÃ
!sequential_5/lstm_10/zeros_1/LessLess$sequential_5/lstm_10/zeros_1/mul:z:0,sequential_5/lstm_10/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2#
!sequential_5/lstm_10/zeros_1/Less
%sequential_5/lstm_10/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :º2'
%sequential_5/lstm_10/zeros_1/packed/1Ý
#sequential_5/lstm_10/zeros_1/packedPack+sequential_5/lstm_10/strided_slice:output:0.sequential_5/lstm_10/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2%
#sequential_5/lstm_10/zeros_1/packed
"sequential_5/lstm_10/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"sequential_5/lstm_10/zeros_1/ConstÒ
sequential_5/lstm_10/zeros_1Fill,sequential_5/lstm_10/zeros_1/packed:output:0+sequential_5/lstm_10/zeros_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2
sequential_5/lstm_10/zeros_1
#sequential_5/lstm_10/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2%
#sequential_5/lstm_10/transpose/permÀ
sequential_5/lstm_10/transpose	Transposelstm_10_input,sequential_5/lstm_10/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]2 
sequential_5/lstm_10/transpose
sequential_5/lstm_10/Shape_1Shape"sequential_5/lstm_10/transpose:y:0*
T0*
_output_shapes
:2
sequential_5/lstm_10/Shape_1¢
*sequential_5/lstm_10/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2,
*sequential_5/lstm_10/strided_slice_1/stack¦
,sequential_5/lstm_10/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_5/lstm_10/strided_slice_1/stack_1¦
,sequential_5/lstm_10/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_5/lstm_10/strided_slice_1/stack_2ì
$sequential_5/lstm_10/strided_slice_1StridedSlice%sequential_5/lstm_10/Shape_1:output:03sequential_5/lstm_10/strided_slice_1/stack:output:05sequential_5/lstm_10/strided_slice_1/stack_1:output:05sequential_5/lstm_10/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2&
$sequential_5/lstm_10/strided_slice_1¯
0sequential_5/lstm_10/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ22
0sequential_5/lstm_10/TensorArrayV2/element_shape
"sequential_5/lstm_10/TensorArrayV2TensorListReserve9sequential_5/lstm_10/TensorArrayV2/element_shape:output:0-sequential_5/lstm_10/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02$
"sequential_5/lstm_10/TensorArrayV2é
Jsequential_5/lstm_10/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ]   2L
Jsequential_5/lstm_10/TensorArrayUnstack/TensorListFromTensor/element_shapeÌ
<sequential_5/lstm_10/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor"sequential_5/lstm_10/transpose:y:0Ssequential_5/lstm_10/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02>
<sequential_5/lstm_10/TensorArrayUnstack/TensorListFromTensor¢
*sequential_5/lstm_10/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2,
*sequential_5/lstm_10/strided_slice_2/stack¦
,sequential_5/lstm_10/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_5/lstm_10/strided_slice_2/stack_1¦
,sequential_5/lstm_10/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_5/lstm_10/strided_slice_2/stack_2ú
$sequential_5/lstm_10/strided_slice_2StridedSlice"sequential_5/lstm_10/transpose:y:03sequential_5/lstm_10/strided_slice_2/stack:output:05sequential_5/lstm_10/strided_slice_2/stack_1:output:05sequential_5/lstm_10/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]*
shrink_axis_mask2&
$sequential_5/lstm_10/strided_slice_2ô
7sequential_5/lstm_10/lstm_cell_10/MatMul/ReadVariableOpReadVariableOp@sequential_5_lstm_10_lstm_cell_10_matmul_readvariableop_resource*
_output_shapes
:	]è	*
dtype029
7sequential_5/lstm_10/lstm_cell_10/MatMul/ReadVariableOp
(sequential_5/lstm_10/lstm_cell_10/MatMulMatMul-sequential_5/lstm_10/strided_slice_2:output:0?sequential_5/lstm_10/lstm_cell_10/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè	2*
(sequential_5/lstm_10/lstm_cell_10/MatMulû
9sequential_5/lstm_10/lstm_cell_10/MatMul_1/ReadVariableOpReadVariableOpBsequential_5_lstm_10_lstm_cell_10_matmul_1_readvariableop_resource* 
_output_shapes
:
ºè	*
dtype02;
9sequential_5/lstm_10/lstm_cell_10/MatMul_1/ReadVariableOpý
*sequential_5/lstm_10/lstm_cell_10/MatMul_1MatMul#sequential_5/lstm_10/zeros:output:0Asequential_5/lstm_10/lstm_cell_10/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè	2,
*sequential_5/lstm_10/lstm_cell_10/MatMul_1ô
%sequential_5/lstm_10/lstm_cell_10/addAddV22sequential_5/lstm_10/lstm_cell_10/MatMul:product:04sequential_5/lstm_10/lstm_cell_10/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè	2'
%sequential_5/lstm_10/lstm_cell_10/addó
8sequential_5/lstm_10/lstm_cell_10/BiasAdd/ReadVariableOpReadVariableOpAsequential_5_lstm_10_lstm_cell_10_biasadd_readvariableop_resource*
_output_shapes	
:è	*
dtype02:
8sequential_5/lstm_10/lstm_cell_10/BiasAdd/ReadVariableOp
)sequential_5/lstm_10/lstm_cell_10/BiasAddBiasAdd)sequential_5/lstm_10/lstm_cell_10/add:z:0@sequential_5/lstm_10/lstm_cell_10/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè	2+
)sequential_5/lstm_10/lstm_cell_10/BiasAdd¨
1sequential_5/lstm_10/lstm_cell_10/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :23
1sequential_5/lstm_10/lstm_cell_10/split/split_dimË
'sequential_5/lstm_10/lstm_cell_10/splitSplit:sequential_5/lstm_10/lstm_cell_10/split/split_dim:output:02sequential_5/lstm_10/lstm_cell_10/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿº:ÿÿÿÿÿÿÿÿÿº:ÿÿÿÿÿÿÿÿÿº:ÿÿÿÿÿÿÿÿÿº*
	num_split2)
'sequential_5/lstm_10/lstm_cell_10/splitÆ
)sequential_5/lstm_10/lstm_cell_10/SigmoidSigmoid0sequential_5/lstm_10/lstm_cell_10/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2+
)sequential_5/lstm_10/lstm_cell_10/SigmoidÊ
+sequential_5/lstm_10/lstm_cell_10/Sigmoid_1Sigmoid0sequential_5/lstm_10/lstm_cell_10/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2-
+sequential_5/lstm_10/lstm_cell_10/Sigmoid_1à
%sequential_5/lstm_10/lstm_cell_10/mulMul/sequential_5/lstm_10/lstm_cell_10/Sigmoid_1:y:0%sequential_5/lstm_10/zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2'
%sequential_5/lstm_10/lstm_cell_10/mul½
&sequential_5/lstm_10/lstm_cell_10/ReluRelu0sequential_5/lstm_10/lstm_cell_10/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2(
&sequential_5/lstm_10/lstm_cell_10/Reluñ
'sequential_5/lstm_10/lstm_cell_10/mul_1Mul-sequential_5/lstm_10/lstm_cell_10/Sigmoid:y:04sequential_5/lstm_10/lstm_cell_10/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2)
'sequential_5/lstm_10/lstm_cell_10/mul_1æ
'sequential_5/lstm_10/lstm_cell_10/add_1AddV2)sequential_5/lstm_10/lstm_cell_10/mul:z:0+sequential_5/lstm_10/lstm_cell_10/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2)
'sequential_5/lstm_10/lstm_cell_10/add_1Ê
+sequential_5/lstm_10/lstm_cell_10/Sigmoid_2Sigmoid0sequential_5/lstm_10/lstm_cell_10/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2-
+sequential_5/lstm_10/lstm_cell_10/Sigmoid_2¼
(sequential_5/lstm_10/lstm_cell_10/Relu_1Relu+sequential_5/lstm_10/lstm_cell_10/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2*
(sequential_5/lstm_10/lstm_cell_10/Relu_1õ
'sequential_5/lstm_10/lstm_cell_10/mul_2Mul/sequential_5/lstm_10/lstm_cell_10/Sigmoid_2:y:06sequential_5/lstm_10/lstm_cell_10/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2)
'sequential_5/lstm_10/lstm_cell_10/mul_2¹
2sequential_5/lstm_10/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ:  24
2sequential_5/lstm_10/TensorArrayV2_1/element_shape
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
sequential_5/lstm_10/time©
-sequential_5/lstm_10/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2/
-sequential_5/lstm_10/while/maximum_iterations
'sequential_5/lstm_10/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2)
'sequential_5/lstm_10/while/loop_counterÎ
sequential_5/lstm_10/whileWhile0sequential_5/lstm_10/while/loop_counter:output:06sequential_5/lstm_10/while/maximum_iterations:output:0"sequential_5/lstm_10/time:output:0-sequential_5/lstm_10/TensorArrayV2_1:handle:0#sequential_5/lstm_10/zeros:output:0%sequential_5/lstm_10/zeros_1:output:0-sequential_5/lstm_10/strided_slice_1:output:0Lsequential_5/lstm_10/TensorArrayUnstack/TensorListFromTensor:output_handle:0@sequential_5_lstm_10_lstm_cell_10_matmul_readvariableop_resourceBsequential_5_lstm_10_lstm_cell_10_matmul_1_readvariableop_resourceAsequential_5_lstm_10_lstm_cell_10_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿº:ÿÿÿÿÿÿÿÿÿº: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *4
body,R*
(sequential_5_lstm_10_while_body_19553854*4
cond,R*
(sequential_5_lstm_10_while_cond_19553853*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿº:ÿÿÿÿÿÿÿÿÿº: : : : : *
parallel_iterations 2
sequential_5/lstm_10/whileß
Esequential_5/lstm_10/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ:  2G
Esequential_5/lstm_10/TensorArrayV2Stack/TensorListStack/element_shape½
7sequential_5/lstm_10/TensorArrayV2Stack/TensorListStackTensorListStack#sequential_5/lstm_10/while:output:3Nsequential_5/lstm_10/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿº*
element_dtype029
7sequential_5/lstm_10/TensorArrayV2Stack/TensorListStack«
*sequential_5/lstm_10/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2,
*sequential_5/lstm_10/strided_slice_3/stack¦
,sequential_5/lstm_10/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2.
,sequential_5/lstm_10/strided_slice_3/stack_1¦
,sequential_5/lstm_10/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_5/lstm_10/strided_slice_3/stack_2
$sequential_5/lstm_10/strided_slice_3StridedSlice@sequential_5/lstm_10/TensorArrayV2Stack/TensorListStack:tensor:03sequential_5/lstm_10/strided_slice_3/stack:output:05sequential_5/lstm_10/strided_slice_3/stack_1:output:05sequential_5/lstm_10/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº*
shrink_axis_mask2&
$sequential_5/lstm_10/strided_slice_3£
%sequential_5/lstm_10/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2'
%sequential_5/lstm_10/transpose_1/permú
 sequential_5/lstm_10/transpose_1	Transpose@sequential_5/lstm_10/TensorArrayV2Stack/TensorListStack:tensor:0.sequential_5/lstm_10/transpose_1/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2"
 sequential_5/lstm_10/transpose_1
sequential_5/lstm_10/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
sequential_5/lstm_10/runtime­
 sequential_5/dropout_10/IdentityIdentity$sequential_5/lstm_10/transpose_1:y:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2"
 sequential_5/dropout_10/Identity
sequential_5/lstm_11/ShapeShape)sequential_5/dropout_10/Identity:output:0*
T0*
_output_shapes
:2
sequential_5/lstm_11/Shape
(sequential_5/lstm_11/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(sequential_5/lstm_11/strided_slice/stack¢
*sequential_5/lstm_11/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*sequential_5/lstm_11/strided_slice/stack_1¢
*sequential_5/lstm_11/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*sequential_5/lstm_11/strided_slice/stack_2à
"sequential_5/lstm_11/strided_sliceStridedSlice#sequential_5/lstm_11/Shape:output:01sequential_5/lstm_11/strided_slice/stack:output:03sequential_5/lstm_11/strided_slice/stack_1:output:03sequential_5/lstm_11/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"sequential_5/lstm_11/strided_slice
 sequential_5/lstm_11/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :S2"
 sequential_5/lstm_11/zeros/mul/yÀ
sequential_5/lstm_11/zeros/mulMul+sequential_5/lstm_11/strided_slice:output:0)sequential_5/lstm_11/zeros/mul/y:output:0*
T0*
_output_shapes
: 2 
sequential_5/lstm_11/zeros/mul
!sequential_5/lstm_11/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2#
!sequential_5/lstm_11/zeros/Less/y»
sequential_5/lstm_11/zeros/LessLess"sequential_5/lstm_11/zeros/mul:z:0*sequential_5/lstm_11/zeros/Less/y:output:0*
T0*
_output_shapes
: 2!
sequential_5/lstm_11/zeros/Less
#sequential_5/lstm_11/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :S2%
#sequential_5/lstm_11/zeros/packed/1×
!sequential_5/lstm_11/zeros/packedPack+sequential_5/lstm_11/strided_slice:output:0,sequential_5/lstm_11/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2#
!sequential_5/lstm_11/zeros/packed
 sequential_5/lstm_11/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 sequential_5/lstm_11/zeros/ConstÉ
sequential_5/lstm_11/zerosFill*sequential_5/lstm_11/zeros/packed:output:0)sequential_5/lstm_11/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2
sequential_5/lstm_11/zeros
"sequential_5/lstm_11/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :S2$
"sequential_5/lstm_11/zeros_1/mul/yÆ
 sequential_5/lstm_11/zeros_1/mulMul+sequential_5/lstm_11/strided_slice:output:0+sequential_5/lstm_11/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2"
 sequential_5/lstm_11/zeros_1/mul
#sequential_5/lstm_11/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2%
#sequential_5/lstm_11/zeros_1/Less/yÃ
!sequential_5/lstm_11/zeros_1/LessLess$sequential_5/lstm_11/zeros_1/mul:z:0,sequential_5/lstm_11/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2#
!sequential_5/lstm_11/zeros_1/Less
%sequential_5/lstm_11/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :S2'
%sequential_5/lstm_11/zeros_1/packed/1Ý
#sequential_5/lstm_11/zeros_1/packedPack+sequential_5/lstm_11/strided_slice:output:0.sequential_5/lstm_11/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2%
#sequential_5/lstm_11/zeros_1/packed
"sequential_5/lstm_11/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"sequential_5/lstm_11/zeros_1/ConstÑ
sequential_5/lstm_11/zeros_1Fill,sequential_5/lstm_11/zeros_1/packed:output:0+sequential_5/lstm_11/zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2
sequential_5/lstm_11/zeros_1
#sequential_5/lstm_11/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2%
#sequential_5/lstm_11/transpose/permÝ
sequential_5/lstm_11/transpose	Transpose)sequential_5/dropout_10/Identity:output:0,sequential_5/lstm_11/transpose/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2 
sequential_5/lstm_11/transpose
sequential_5/lstm_11/Shape_1Shape"sequential_5/lstm_11/transpose:y:0*
T0*
_output_shapes
:2
sequential_5/lstm_11/Shape_1¢
*sequential_5/lstm_11/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2,
*sequential_5/lstm_11/strided_slice_1/stack¦
,sequential_5/lstm_11/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_5/lstm_11/strided_slice_1/stack_1¦
,sequential_5/lstm_11/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_5/lstm_11/strided_slice_1/stack_2ì
$sequential_5/lstm_11/strided_slice_1StridedSlice%sequential_5/lstm_11/Shape_1:output:03sequential_5/lstm_11/strided_slice_1/stack:output:05sequential_5/lstm_11/strided_slice_1/stack_1:output:05sequential_5/lstm_11/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2&
$sequential_5/lstm_11/strided_slice_1¯
0sequential_5/lstm_11/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ22
0sequential_5/lstm_11/TensorArrayV2/element_shape
"sequential_5/lstm_11/TensorArrayV2TensorListReserve9sequential_5/lstm_11/TensorArrayV2/element_shape:output:0-sequential_5/lstm_11/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02$
"sequential_5/lstm_11/TensorArrayV2é
Jsequential_5/lstm_11/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ:  2L
Jsequential_5/lstm_11/TensorArrayUnstack/TensorListFromTensor/element_shapeÌ
<sequential_5/lstm_11/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor"sequential_5/lstm_11/transpose:y:0Ssequential_5/lstm_11/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02>
<sequential_5/lstm_11/TensorArrayUnstack/TensorListFromTensor¢
*sequential_5/lstm_11/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2,
*sequential_5/lstm_11/strided_slice_2/stack¦
,sequential_5/lstm_11/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_5/lstm_11/strided_slice_2/stack_1¦
,sequential_5/lstm_11/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_5/lstm_11/strided_slice_2/stack_2û
$sequential_5/lstm_11/strided_slice_2StridedSlice"sequential_5/lstm_11/transpose:y:03sequential_5/lstm_11/strided_slice_2/stack:output:05sequential_5/lstm_11/strided_slice_2/stack_1:output:05sequential_5/lstm_11/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº*
shrink_axis_mask2&
$sequential_5/lstm_11/strided_slice_2õ
7sequential_5/lstm_11/lstm_cell_11/MatMul/ReadVariableOpReadVariableOp@sequential_5_lstm_11_lstm_cell_11_matmul_readvariableop_resource* 
_output_shapes
:
ºÌ*
dtype029
7sequential_5/lstm_11/lstm_cell_11/MatMul/ReadVariableOp
(sequential_5/lstm_11/lstm_cell_11/MatMulMatMul-sequential_5/lstm_11/strided_slice_2:output:0?sequential_5/lstm_11/lstm_cell_11/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÌ2*
(sequential_5/lstm_11/lstm_cell_11/MatMulú
9sequential_5/lstm_11/lstm_cell_11/MatMul_1/ReadVariableOpReadVariableOpBsequential_5_lstm_11_lstm_cell_11_matmul_1_readvariableop_resource*
_output_shapes
:	SÌ*
dtype02;
9sequential_5/lstm_11/lstm_cell_11/MatMul_1/ReadVariableOpý
*sequential_5/lstm_11/lstm_cell_11/MatMul_1MatMul#sequential_5/lstm_11/zeros:output:0Asequential_5/lstm_11/lstm_cell_11/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÌ2,
*sequential_5/lstm_11/lstm_cell_11/MatMul_1ô
%sequential_5/lstm_11/lstm_cell_11/addAddV22sequential_5/lstm_11/lstm_cell_11/MatMul:product:04sequential_5/lstm_11/lstm_cell_11/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÌ2'
%sequential_5/lstm_11/lstm_cell_11/addó
8sequential_5/lstm_11/lstm_cell_11/BiasAdd/ReadVariableOpReadVariableOpAsequential_5_lstm_11_lstm_cell_11_biasadd_readvariableop_resource*
_output_shapes	
:Ì*
dtype02:
8sequential_5/lstm_11/lstm_cell_11/BiasAdd/ReadVariableOp
)sequential_5/lstm_11/lstm_cell_11/BiasAddBiasAdd)sequential_5/lstm_11/lstm_cell_11/add:z:0@sequential_5/lstm_11/lstm_cell_11/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÌ2+
)sequential_5/lstm_11/lstm_cell_11/BiasAdd¨
1sequential_5/lstm_11/lstm_cell_11/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :23
1sequential_5/lstm_11/lstm_cell_11/split/split_dimÇ
'sequential_5/lstm_11/lstm_cell_11/splitSplit:sequential_5/lstm_11/lstm_cell_11/split/split_dim:output:02sequential_5/lstm_11/lstm_cell_11/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿS:ÿÿÿÿÿÿÿÿÿS:ÿÿÿÿÿÿÿÿÿS:ÿÿÿÿÿÿÿÿÿS*
	num_split2)
'sequential_5/lstm_11/lstm_cell_11/splitÅ
)sequential_5/lstm_11/lstm_cell_11/SigmoidSigmoid0sequential_5/lstm_11/lstm_cell_11/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2+
)sequential_5/lstm_11/lstm_cell_11/SigmoidÉ
+sequential_5/lstm_11/lstm_cell_11/Sigmoid_1Sigmoid0sequential_5/lstm_11/lstm_cell_11/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2-
+sequential_5/lstm_11/lstm_cell_11/Sigmoid_1ß
%sequential_5/lstm_11/lstm_cell_11/mulMul/sequential_5/lstm_11/lstm_cell_11/Sigmoid_1:y:0%sequential_5/lstm_11/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2'
%sequential_5/lstm_11/lstm_cell_11/mul¼
&sequential_5/lstm_11/lstm_cell_11/ReluRelu0sequential_5/lstm_11/lstm_cell_11/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2(
&sequential_5/lstm_11/lstm_cell_11/Reluð
'sequential_5/lstm_11/lstm_cell_11/mul_1Mul-sequential_5/lstm_11/lstm_cell_11/Sigmoid:y:04sequential_5/lstm_11/lstm_cell_11/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2)
'sequential_5/lstm_11/lstm_cell_11/mul_1å
'sequential_5/lstm_11/lstm_cell_11/add_1AddV2)sequential_5/lstm_11/lstm_cell_11/mul:z:0+sequential_5/lstm_11/lstm_cell_11/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2)
'sequential_5/lstm_11/lstm_cell_11/add_1É
+sequential_5/lstm_11/lstm_cell_11/Sigmoid_2Sigmoid0sequential_5/lstm_11/lstm_cell_11/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2-
+sequential_5/lstm_11/lstm_cell_11/Sigmoid_2»
(sequential_5/lstm_11/lstm_cell_11/Relu_1Relu+sequential_5/lstm_11/lstm_cell_11/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2*
(sequential_5/lstm_11/lstm_cell_11/Relu_1ô
'sequential_5/lstm_11/lstm_cell_11/mul_2Mul/sequential_5/lstm_11/lstm_cell_11/Sigmoid_2:y:06sequential_5/lstm_11/lstm_cell_11/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2)
'sequential_5/lstm_11/lstm_cell_11/mul_2¹
2sequential_5/lstm_11/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿS   24
2sequential_5/lstm_11/TensorArrayV2_1/element_shape
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
sequential_5/lstm_11/time©
-sequential_5/lstm_11/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2/
-sequential_5/lstm_11/while/maximum_iterations
'sequential_5/lstm_11/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2)
'sequential_5/lstm_11/while/loop_counterÊ
sequential_5/lstm_11/whileWhile0sequential_5/lstm_11/while/loop_counter:output:06sequential_5/lstm_11/while/maximum_iterations:output:0"sequential_5/lstm_11/time:output:0-sequential_5/lstm_11/TensorArrayV2_1:handle:0#sequential_5/lstm_11/zeros:output:0%sequential_5/lstm_11/zeros_1:output:0-sequential_5/lstm_11/strided_slice_1:output:0Lsequential_5/lstm_11/TensorArrayUnstack/TensorListFromTensor:output_handle:0@sequential_5_lstm_11_lstm_cell_11_matmul_readvariableop_resourceBsequential_5_lstm_11_lstm_cell_11_matmul_1_readvariableop_resourceAsequential_5_lstm_11_lstm_cell_11_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿS:ÿÿÿÿÿÿÿÿÿS: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *4
body,R*
(sequential_5_lstm_11_while_body_19554002*4
cond,R*
(sequential_5_lstm_11_while_cond_19554001*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿS:ÿÿÿÿÿÿÿÿÿS: : : : : *
parallel_iterations 2
sequential_5/lstm_11/whileß
Esequential_5/lstm_11/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿS   2G
Esequential_5/lstm_11/TensorArrayV2Stack/TensorListStack/element_shape¼
7sequential_5/lstm_11/TensorArrayV2Stack/TensorListStackTensorListStack#sequential_5/lstm_11/while:output:3Nsequential_5/lstm_11/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿS*
element_dtype029
7sequential_5/lstm_11/TensorArrayV2Stack/TensorListStack«
*sequential_5/lstm_11/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2,
*sequential_5/lstm_11/strided_slice_3/stack¦
,sequential_5/lstm_11/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2.
,sequential_5/lstm_11/strided_slice_3/stack_1¦
,sequential_5/lstm_11/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_5/lstm_11/strided_slice_3/stack_2
$sequential_5/lstm_11/strided_slice_3StridedSlice@sequential_5/lstm_11/TensorArrayV2Stack/TensorListStack:tensor:03sequential_5/lstm_11/strided_slice_3/stack:output:05sequential_5/lstm_11/strided_slice_3/stack_1:output:05sequential_5/lstm_11/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS*
shrink_axis_mask2&
$sequential_5/lstm_11/strided_slice_3£
%sequential_5/lstm_11/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2'
%sequential_5/lstm_11/transpose_1/permù
 sequential_5/lstm_11/transpose_1	Transpose@sequential_5/lstm_11/TensorArrayV2Stack/TensorListStack:tensor:0.sequential_5/lstm_11/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2"
 sequential_5/lstm_11/transpose_1
sequential_5/lstm_11/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
sequential_5/lstm_11/runtime¬
 sequential_5/dropout_11/IdentityIdentity$sequential_5/lstm_11/transpose_1:y:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2"
 sequential_5/dropout_11/IdentityÕ
-sequential_5/dense_5/Tensordot/ReadVariableOpReadVariableOp6sequential_5_dense_5_tensordot_readvariableop_resource*
_output_shapes

:S*
dtype02/
-sequential_5/dense_5/Tensordot/ReadVariableOp
#sequential_5/dense_5/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2%
#sequential_5/dense_5/Tensordot/axes
#sequential_5/dense_5/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2%
#sequential_5/dense_5/Tensordot/free¥
$sequential_5/dense_5/Tensordot/ShapeShape)sequential_5/dropout_11/Identity:output:0*
T0*
_output_shapes
:2&
$sequential_5/dense_5/Tensordot/Shape
,sequential_5/dense_5/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,sequential_5/dense_5/Tensordot/GatherV2/axisº
'sequential_5/dense_5/Tensordot/GatherV2GatherV2-sequential_5/dense_5/Tensordot/Shape:output:0,sequential_5/dense_5/Tensordot/free:output:05sequential_5/dense_5/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2)
'sequential_5/dense_5/Tensordot/GatherV2¢
.sequential_5/dense_5/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.sequential_5/dense_5/Tensordot/GatherV2_1/axisÀ
)sequential_5/dense_5/Tensordot/GatherV2_1GatherV2-sequential_5/dense_5/Tensordot/Shape:output:0,sequential_5/dense_5/Tensordot/axes:output:07sequential_5/dense_5/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2+
)sequential_5/dense_5/Tensordot/GatherV2_1
$sequential_5/dense_5/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2&
$sequential_5/dense_5/Tensordot/ConstÔ
#sequential_5/dense_5/Tensordot/ProdProd0sequential_5/dense_5/Tensordot/GatherV2:output:0-sequential_5/dense_5/Tensordot/Const:output:0*
T0*
_output_shapes
: 2%
#sequential_5/dense_5/Tensordot/Prod
&sequential_5/dense_5/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2(
&sequential_5/dense_5/Tensordot/Const_1Ü
%sequential_5/dense_5/Tensordot/Prod_1Prod2sequential_5/dense_5/Tensordot/GatherV2_1:output:0/sequential_5/dense_5/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2'
%sequential_5/dense_5/Tensordot/Prod_1
*sequential_5/dense_5/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2,
*sequential_5/dense_5/Tensordot/concat/axis
%sequential_5/dense_5/Tensordot/concatConcatV2,sequential_5/dense_5/Tensordot/free:output:0,sequential_5/dense_5/Tensordot/axes:output:03sequential_5/dense_5/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2'
%sequential_5/dense_5/Tensordot/concatà
$sequential_5/dense_5/Tensordot/stackPack,sequential_5/dense_5/Tensordot/Prod:output:0.sequential_5/dense_5/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2&
$sequential_5/dense_5/Tensordot/stackò
(sequential_5/dense_5/Tensordot/transpose	Transpose)sequential_5/dropout_11/Identity:output:0.sequential_5/dense_5/Tensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2*
(sequential_5/dense_5/Tensordot/transposeó
&sequential_5/dense_5/Tensordot/ReshapeReshape,sequential_5/dense_5/Tensordot/transpose:y:0-sequential_5/dense_5/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2(
&sequential_5/dense_5/Tensordot/Reshapeò
%sequential_5/dense_5/Tensordot/MatMulMatMul/sequential_5/dense_5/Tensordot/Reshape:output:05sequential_5/dense_5/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%sequential_5/dense_5/Tensordot/MatMul
&sequential_5/dense_5/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&sequential_5/dense_5/Tensordot/Const_2
,sequential_5/dense_5/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,sequential_5/dense_5/Tensordot/concat_1/axis¦
'sequential_5/dense_5/Tensordot/concat_1ConcatV20sequential_5/dense_5/Tensordot/GatherV2:output:0/sequential_5/dense_5/Tensordot/Const_2:output:05sequential_5/dense_5/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2)
'sequential_5/dense_5/Tensordot/concat_1ä
sequential_5/dense_5/TensordotReshape/sequential_5/dense_5/Tensordot/MatMul:product:00sequential_5/dense_5/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
sequential_5/dense_5/TensordotË
+sequential_5/dense_5/BiasAdd/ReadVariableOpReadVariableOp4sequential_5_dense_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02-
+sequential_5/dense_5/BiasAdd/ReadVariableOpÛ
sequential_5/dense_5/BiasAddBiasAdd'sequential_5/dense_5/Tensordot:output:03sequential_5/dense_5/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential_5/dense_5/BiasAdd¤
sequential_5/dense_5/SoftmaxSoftmax%sequential_5/dense_5/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential_5/dense_5/Softmax
IdentityIdentity&sequential_5/dense_5/Softmax:softmax:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

IdentityÈ
NoOpNoOp,^sequential_5/dense_5/BiasAdd/ReadVariableOp.^sequential_5/dense_5/Tensordot/ReadVariableOp9^sequential_5/lstm_10/lstm_cell_10/BiasAdd/ReadVariableOp8^sequential_5/lstm_10/lstm_cell_10/MatMul/ReadVariableOp:^sequential_5/lstm_10/lstm_cell_10/MatMul_1/ReadVariableOp^sequential_5/lstm_10/while9^sequential_5/lstm_11/lstm_cell_11/BiasAdd/ReadVariableOp8^sequential_5/lstm_11/lstm_cell_11/MatMul/ReadVariableOp:^sequential_5/lstm_11/lstm_cell_11/MatMul_1/ReadVariableOp^sequential_5/lstm_11/while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ]: : : : : : : : 2Z
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
:ÿÿÿÿÿÿÿÿÿ]
'
_user_specified_namelstm_10_input
ã
Í
while_cond_19554412
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_19554412___redundant_placeholder06
2while_while_cond_19554412___redundant_placeholder16
2while_while_cond_19554412___redundant_placeholder26
2while_while_cond_19554412___redundant_placeholder3
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
B: : : : :ÿÿÿÿÿÿÿÿÿº:ÿÿÿÿÿÿÿÿÿº: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº:

_output_shapes
: :

_output_shapes
:


J__inference_lstm_cell_11_layer_call_and_return_conditional_losses_19558593

inputs
states_0
states_12
matmul_readvariableop_resource:
ºÌ3
 matmul_1_readvariableop_resource:	SÌ.
biasadd_readvariableop_resource:	Ì
identity

identity_1

identity_2¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
ºÌ*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÌ2
MatMul
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	SÌ*
dtype02
MatMul_1/ReadVariableOp|
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÌ2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÌ2
add
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:Ì*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÌ2	
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
L:ÿÿÿÿÿÿÿÿÿS:ÿÿÿÿÿÿÿÿÿS:ÿÿÿÿÿÿÿÿÿS:ÿÿÿÿÿÿÿÿÿS*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2
	Sigmoid_1\
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2
mulV
ReluRelusplit:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2
	Sigmoid_2U
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2
mul_2d
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2

Identityh

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2

Identity_1h

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2

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
@:ÿÿÿÿÿÿÿÿÿº:ÿÿÿÿÿÿÿÿÿS:ÿÿÿÿÿÿÿÿÿS: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº
 
_user_specified_nameinputs:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS
"
_user_specified_name
states/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS
"
_user_specified_name
states/1
\

E__inference_lstm_10_layer_call_and_return_conditional_losses_19557645

inputs>
+lstm_cell_10_matmul_readvariableop_resource:	]è	A
-lstm_cell_10_matmul_1_readvariableop_resource:
ºè	;
,lstm_cell_10_biasadd_readvariableop_resource:	è	
identity¢#lstm_cell_10/BiasAdd/ReadVariableOp¢"lstm_cell_10/MatMul/ReadVariableOp¢$lstm_cell_10/MatMul_1/ReadVariableOp¢whileD
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
B :º2
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
B :º2
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
:ÿÿÿÿÿÿÿÿÿº2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :º2
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
B :º2
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
:ÿÿÿÿÿÿÿÿÿº2	
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
"lstm_cell_10/MatMul/ReadVariableOpReadVariableOp+lstm_cell_10_matmul_readvariableop_resource*
_output_shapes
:	]è	*
dtype02$
"lstm_cell_10/MatMul/ReadVariableOp­
lstm_cell_10/MatMulMatMulstrided_slice_2:output:0*lstm_cell_10/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè	2
lstm_cell_10/MatMul¼
$lstm_cell_10/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_10_matmul_1_readvariableop_resource* 
_output_shapes
:
ºè	*
dtype02&
$lstm_cell_10/MatMul_1/ReadVariableOp©
lstm_cell_10/MatMul_1MatMulzeros:output:0,lstm_cell_10/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè	2
lstm_cell_10/MatMul_1 
lstm_cell_10/addAddV2lstm_cell_10/MatMul:product:0lstm_cell_10/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè	2
lstm_cell_10/add´
#lstm_cell_10/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_10_biasadd_readvariableop_resource*
_output_shapes	
:è	*
dtype02%
#lstm_cell_10/BiasAdd/ReadVariableOp­
lstm_cell_10/BiasAddBiasAddlstm_cell_10/add:z:0+lstm_cell_10/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè	2
lstm_cell_10/BiasAdd~
lstm_cell_10/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_10/split/split_dim÷
lstm_cell_10/splitSplit%lstm_cell_10/split/split_dim:output:0lstm_cell_10/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿº:ÿÿÿÿÿÿÿÿÿº:ÿÿÿÿÿÿÿÿÿº:ÿÿÿÿÿÿÿÿÿº*
	num_split2
lstm_cell_10/split
lstm_cell_10/SigmoidSigmoidlstm_cell_10/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2
lstm_cell_10/Sigmoid
lstm_cell_10/Sigmoid_1Sigmoidlstm_cell_10/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2
lstm_cell_10/Sigmoid_1
lstm_cell_10/mulMullstm_cell_10/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2
lstm_cell_10/mul~
lstm_cell_10/ReluRelulstm_cell_10/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2
lstm_cell_10/Relu
lstm_cell_10/mul_1Mullstm_cell_10/Sigmoid:y:0lstm_cell_10/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2
lstm_cell_10/mul_1
lstm_cell_10/add_1AddV2lstm_cell_10/mul:z:0lstm_cell_10/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2
lstm_cell_10/add_1
lstm_cell_10/Sigmoid_2Sigmoidlstm_cell_10/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2
lstm_cell_10/Sigmoid_2}
lstm_cell_10/Relu_1Relulstm_cell_10/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2
lstm_cell_10/Relu_1¡
lstm_cell_10/mul_2Mullstm_cell_10/Sigmoid_2:y:0!lstm_cell_10/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2
lstm_cell_10/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ:  2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_10_matmul_readvariableop_resource-lstm_cell_10_matmul_1_readvariableop_resource,lstm_cell_10_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿº:ÿÿÿÿÿÿÿÿÿº: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_19557561*
condR
while_cond_19557560*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿº:ÿÿÿÿÿÿÿÿÿº: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ:  22
0TensorArrayV2Stack/TensorListStack/element_shapeé
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿº*
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
:ÿÿÿÿÿÿÿÿÿº*
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
:ÿÿÿÿÿÿÿÿÿº2
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
:ÿÿÿÿÿÿÿÿÿº2

IdentityÈ
NoOpNoOp$^lstm_cell_10/BiasAdd/ReadVariableOp#^lstm_cell_10/MatMul/ReadVariableOp%^lstm_cell_10/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ]: : : 2J
#lstm_cell_10/BiasAdd/ReadVariableOp#lstm_cell_10/BiasAdd/ReadVariableOp2H
"lstm_cell_10/MatMul/ReadVariableOp"lstm_cell_10/MatMul/ReadVariableOp2L
$lstm_cell_10/MatMul_1/ReadVariableOp$lstm_cell_10/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
 
_user_specified_nameinputs
Ð

í
lstm_11_while_cond_19556879,
(lstm_11_while_lstm_11_while_loop_counter2
.lstm_11_while_lstm_11_while_maximum_iterations
lstm_11_while_placeholder
lstm_11_while_placeholder_1
lstm_11_while_placeholder_2
lstm_11_while_placeholder_3.
*lstm_11_while_less_lstm_11_strided_slice_1F
Blstm_11_while_lstm_11_while_cond_19556879___redundant_placeholder0F
Blstm_11_while_lstm_11_while_cond_19556879___redundant_placeholder1F
Blstm_11_while_lstm_11_while_cond_19556879___redundant_placeholder2F
Blstm_11_while_lstm_11_while_cond_19556879___redundant_placeholder3
lstm_11_while_identity

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
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿS:ÿÿÿÿÿÿÿÿÿS: ::::: 

_output_shapes
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
:ÿÿÿÿÿÿÿÿÿS:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS:

_output_shapes
: :

_output_shapes
:
í[

E__inference_lstm_11_layer_call_and_return_conditional_losses_19555696

inputs?
+lstm_cell_11_matmul_readvariableop_resource:
ºÌ@
-lstm_cell_11_matmul_1_readvariableop_resource:	SÌ;
,lstm_cell_11_biasadd_readvariableop_resource:	Ì
identity¢#lstm_cell_11/BiasAdd/ReadVariableOp¢"lstm_cell_11/MatMul/ReadVariableOp¢$lstm_cell_11/MatMul_1/ReadVariableOp¢whileD
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
value	B :S2
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
value	B :S2
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
:ÿÿÿÿÿÿÿÿÿS2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :S2
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
value	B :S2
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
:ÿÿÿÿÿÿÿÿÿS2	
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
:ÿÿÿÿÿÿÿÿÿº2
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
valueB"ÿÿÿÿ:  27
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
:ÿÿÿÿÿÿÿÿÿº*
shrink_axis_mask2
strided_slice_2¶
"lstm_cell_11/MatMul/ReadVariableOpReadVariableOp+lstm_cell_11_matmul_readvariableop_resource* 
_output_shapes
:
ºÌ*
dtype02$
"lstm_cell_11/MatMul/ReadVariableOp­
lstm_cell_11/MatMulMatMulstrided_slice_2:output:0*lstm_cell_11/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÌ2
lstm_cell_11/MatMul»
$lstm_cell_11/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_11_matmul_1_readvariableop_resource*
_output_shapes
:	SÌ*
dtype02&
$lstm_cell_11/MatMul_1/ReadVariableOp©
lstm_cell_11/MatMul_1MatMulzeros:output:0,lstm_cell_11/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÌ2
lstm_cell_11/MatMul_1 
lstm_cell_11/addAddV2lstm_cell_11/MatMul:product:0lstm_cell_11/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÌ2
lstm_cell_11/add´
#lstm_cell_11/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_11_biasadd_readvariableop_resource*
_output_shapes	
:Ì*
dtype02%
#lstm_cell_11/BiasAdd/ReadVariableOp­
lstm_cell_11/BiasAddBiasAddlstm_cell_11/add:z:0+lstm_cell_11/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÌ2
lstm_cell_11/BiasAdd~
lstm_cell_11/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_11/split/split_dimó
lstm_cell_11/splitSplit%lstm_cell_11/split/split_dim:output:0lstm_cell_11/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿS:ÿÿÿÿÿÿÿÿÿS:ÿÿÿÿÿÿÿÿÿS:ÿÿÿÿÿÿÿÿÿS*
	num_split2
lstm_cell_11/split
lstm_cell_11/SigmoidSigmoidlstm_cell_11/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2
lstm_cell_11/Sigmoid
lstm_cell_11/Sigmoid_1Sigmoidlstm_cell_11/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2
lstm_cell_11/Sigmoid_1
lstm_cell_11/mulMullstm_cell_11/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2
lstm_cell_11/mul}
lstm_cell_11/ReluRelulstm_cell_11/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2
lstm_cell_11/Relu
lstm_cell_11/mul_1Mullstm_cell_11/Sigmoid:y:0lstm_cell_11/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2
lstm_cell_11/mul_1
lstm_cell_11/add_1AddV2lstm_cell_11/mul:z:0lstm_cell_11/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2
lstm_cell_11/add_1
lstm_cell_11/Sigmoid_2Sigmoidlstm_cell_11/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2
lstm_cell_11/Sigmoid_2|
lstm_cell_11/Relu_1Relulstm_cell_11/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2
lstm_cell_11/Relu_1 
lstm_cell_11/mul_2Mullstm_cell_11/Sigmoid_2:y:0!lstm_cell_11/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS2
lstm_cell_11/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿS   2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_11_matmul_readvariableop_resource-lstm_cell_11_matmul_1_readvariableop_resource,lstm_cell_11_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿS:ÿÿÿÿÿÿÿÿÿS: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_19555612*
condR
while_cond_19555611*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿS:ÿÿÿÿÿÿÿÿÿS: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿS   22
0TensorArrayV2Stack/TensorListStack/element_shapeè
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿS*
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
:ÿÿÿÿÿÿÿÿÿS*
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
:ÿÿÿÿÿÿÿÿÿS2
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
:ÿÿÿÿÿÿÿÿÿS2

IdentityÈ
NoOpNoOp$^lstm_cell_11/BiasAdd/ReadVariableOp#^lstm_cell_11/MatMul/ReadVariableOp%^lstm_cell_11/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:ÿÿÿÿÿÿÿÿÿº: : : 2J
#lstm_cell_11/BiasAdd/ReadVariableOp#lstm_cell_11/BiasAdd/ReadVariableOp2H
"lstm_cell_11/MatMul/ReadVariableOp"lstm_cell_11/MatMul/ReadVariableOp2L
$lstm_cell_11/MatMul_1/ReadVariableOp$lstm_cell_11/MatMul_1/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿº
 
_user_specified_nameinputs
Ç
ù
/__inference_lstm_cell_10_layer_call_fn_19558512

inputs
states_0
states_1
unknown:	]è	
	unknown_0:
ºè	
	unknown_1:	è	
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
<:ÿÿÿÿÿÿÿÿÿº:ÿÿÿÿÿÿÿÿÿº:ÿÿÿÿÿÿÿÿÿº*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_lstm_cell_10_layer_call_and_return_conditional_losses_195541892
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº2

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
A:ÿÿÿÿÿÿÿÿÿ]:ÿÿÿÿÿÿÿÿÿº:ÿÿÿÿÿÿÿÿÿº: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
 
_user_specified_nameinputs:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº
"
_user_specified_name
states/0:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº
"
_user_specified_name
states/1
¾F

E__inference_lstm_11_layer_call_and_return_conditional_losses_19555112

inputs)
lstm_cell_11_19555030:
ºÌ(
lstm_cell_11_19555032:	SÌ$
lstm_cell_11_19555034:	Ì
identity¢$lstm_cell_11/StatefulPartitionedCall¢whileD
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
value	B :S2
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
value	B :S2
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
:ÿÿÿÿÿÿÿÿÿS2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :S2
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
value	B :S2
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
:ÿÿÿÿÿÿÿÿÿS2	
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
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿº2
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
valueB"ÿÿÿÿ:  27
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
:ÿÿÿÿÿÿÿÿÿº*
shrink_axis_mask2
strided_slice_2¥
$lstm_cell_11/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_11_19555030lstm_cell_11_19555032lstm_cell_11_19555034*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿS:ÿÿÿÿÿÿÿÿÿS:ÿÿÿÿÿÿÿÿÿS*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_lstm_cell_11_layer_call_and_return_conditional_losses_195549652&
$lstm_cell_11/StatefulPartitionedCall
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿS   2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_11_19555030lstm_cell_11_19555032lstm_cell_11_19555034*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿS:ÿÿÿÿÿÿÿÿÿS: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_19555043*
condR
while_cond_19555042*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿS:ÿÿÿÿÿÿÿÿÿS: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿS   22
0TensorArrayV2Stack/TensorListStack/element_shapeñ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿS*
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
:ÿÿÿÿÿÿÿÿÿS*
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
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿS2
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
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿS2

Identity}
NoOpNoOp%^lstm_cell_11/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿº: : : 2L
$lstm_cell_11/StatefulPartitionedCall$lstm_cell_11/StatefulPartitionedCall2
whilewhile:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿº
 
_user_specified_nameinputs"¨L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*¾
serving_defaultª
K
lstm_10_input:
serving_default_lstm_10_input:0ÿÿÿÿÿÿÿÿÿ]?
dense_54
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
 :S2dense_5/kernel
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
.:,	]è	2lstm_10/lstm_cell_10/kernel
9:7
ºè	2%lstm_10/lstm_cell_10/recurrent_kernel
(:&è	2lstm_10/lstm_cell_10/bias
/:-
ºÌ2lstm_11/lstm_cell_11/kernel
8:6	SÌ2%lstm_11/lstm_cell_11/recurrent_kernel
(:&Ì2lstm_11/lstm_cell_11/bias
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
#__inference__wrapped_model_19554114lstm_10_input"
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
J__inference_sequential_5_layer_call_and_return_conditional_losses_19556658
J__inference_sequential_5_layer_call_and_return_conditional_losses_19556999
J__inference_sequential_5_layer_call_and_return_conditional_losses_19556283
J__inference_sequential_5_layer_call_and_return_conditional_losses_19556308À
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
/__inference_sequential_5_layer_call_fn_19555768
/__inference_sequential_5_layer_call_fn_19557020
/__inference_sequential_5_layer_call_fn_19557041
/__inference_sequential_5_layer_call_fn_19556258À
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
E__inference_lstm_10_layer_call_and_return_conditional_losses_19557192
E__inference_lstm_10_layer_call_and_return_conditional_losses_19557343
E__inference_lstm_10_layer_call_and_return_conditional_losses_19557494
E__inference_lstm_10_layer_call_and_return_conditional_losses_19557645Õ
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
*__inference_lstm_10_layer_call_fn_19557656
*__inference_lstm_10_layer_call_fn_19557667
*__inference_lstm_10_layer_call_fn_19557678
*__inference_lstm_10_layer_call_fn_19557689Õ
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
H__inference_dropout_10_layer_call_and_return_conditional_losses_19557694
H__inference_dropout_10_layer_call_and_return_conditional_losses_19557706´
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
-__inference_dropout_10_layer_call_fn_19557711
-__inference_dropout_10_layer_call_fn_19557716´
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
E__inference_lstm_11_layer_call_and_return_conditional_losses_19557867
E__inference_lstm_11_layer_call_and_return_conditional_losses_19558018
E__inference_lstm_11_layer_call_and_return_conditional_losses_19558169
E__inference_lstm_11_layer_call_and_return_conditional_losses_19558320Õ
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
*__inference_lstm_11_layer_call_fn_19558331
*__inference_lstm_11_layer_call_fn_19558342
*__inference_lstm_11_layer_call_fn_19558353
*__inference_lstm_11_layer_call_fn_19558364Õ
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
H__inference_dropout_11_layer_call_and_return_conditional_losses_19558369
H__inference_dropout_11_layer_call_and_return_conditional_losses_19558381´
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
-__inference_dropout_11_layer_call_fn_19558386
-__inference_dropout_11_layer_call_fn_19558391´
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
E__inference_dense_5_layer_call_and_return_conditional_losses_19558422¢
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
*__inference_dense_5_layer_call_fn_19558431¢
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
&__inference_signature_wrapper_19556331lstm_10_input"
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
J__inference_lstm_cell_10_layer_call_and_return_conditional_losses_19558463
J__inference_lstm_cell_10_layer_call_and_return_conditional_losses_19558495¾
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
/__inference_lstm_cell_10_layer_call_fn_19558512
/__inference_lstm_cell_10_layer_call_fn_19558529¾
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
J__inference_lstm_cell_11_layer_call_and_return_conditional_losses_19558561
J__inference_lstm_cell_11_layer_call_and_return_conditional_losses_19558593¾
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
/__inference_lstm_cell_11_layer_call_fn_19558610
/__inference_lstm_cell_11_layer_call_fn_19558627¾
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
#__inference__wrapped_model_19554114}&'()*+ !:¢7
0¢-
+(
lstm_10_inputÿÿÿÿÿÿÿÿÿ]
ª "5ª2
0
dense_5%"
dense_5ÿÿÿÿÿÿÿÿÿ­
E__inference_dense_5_layer_call_and_return_conditional_losses_19558422d !3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿS
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ
 
*__inference_dense_5_layer_call_fn_19558431W !3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿS
ª "ÿÿÿÿÿÿÿÿÿ²
H__inference_dropout_10_layer_call_and_return_conditional_losses_19557694f8¢5
.¢+
%"
inputsÿÿÿÿÿÿÿÿÿº
p 
ª "*¢'
 
0ÿÿÿÿÿÿÿÿÿº
 ²
H__inference_dropout_10_layer_call_and_return_conditional_losses_19557706f8¢5
.¢+
%"
inputsÿÿÿÿÿÿÿÿÿº
p
ª "*¢'
 
0ÿÿÿÿÿÿÿÿÿº
 
-__inference_dropout_10_layer_call_fn_19557711Y8¢5
.¢+
%"
inputsÿÿÿÿÿÿÿÿÿº
p 
ª "ÿÿÿÿÿÿÿÿÿº
-__inference_dropout_10_layer_call_fn_19557716Y8¢5
.¢+
%"
inputsÿÿÿÿÿÿÿÿÿº
p
ª "ÿÿÿÿÿÿÿÿÿº°
H__inference_dropout_11_layer_call_and_return_conditional_losses_19558369d7¢4
-¢*
$!
inputsÿÿÿÿÿÿÿÿÿS
p 
ª ")¢&

0ÿÿÿÿÿÿÿÿÿS
 °
H__inference_dropout_11_layer_call_and_return_conditional_losses_19558381d7¢4
-¢*
$!
inputsÿÿÿÿÿÿÿÿÿS
p
ª ")¢&

0ÿÿÿÿÿÿÿÿÿS
 
-__inference_dropout_11_layer_call_fn_19558386W7¢4
-¢*
$!
inputsÿÿÿÿÿÿÿÿÿS
p 
ª "ÿÿÿÿÿÿÿÿÿS
-__inference_dropout_11_layer_call_fn_19558391W7¢4
-¢*
$!
inputsÿÿÿÿÿÿÿÿÿS
p
ª "ÿÿÿÿÿÿÿÿÿSÕ
E__inference_lstm_10_layer_call_and_return_conditional_losses_19557192&'(O¢L
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
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿº
 Õ
E__inference_lstm_10_layer_call_and_return_conditional_losses_19557343&'(O¢L
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
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿº
 »
E__inference_lstm_10_layer_call_and_return_conditional_losses_19557494r&'(?¢<
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
0ÿÿÿÿÿÿÿÿÿº
 »
E__inference_lstm_10_layer_call_and_return_conditional_losses_19557645r&'(?¢<
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
0ÿÿÿÿÿÿÿÿÿº
 ¬
*__inference_lstm_10_layer_call_fn_19557656~&'(O¢L
E¢B
41
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ]

 
p 

 
ª "&#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿº¬
*__inference_lstm_10_layer_call_fn_19557667~&'(O¢L
E¢B
41
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ]

 
p

 
ª "&#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿº
*__inference_lstm_10_layer_call_fn_19557678e&'(?¢<
5¢2
$!
inputsÿÿÿÿÿÿÿÿÿ]

 
p 

 
ª "ÿÿÿÿÿÿÿÿÿº
*__inference_lstm_10_layer_call_fn_19557689e&'(?¢<
5¢2
$!
inputsÿÿÿÿÿÿÿÿÿ]

 
p

 
ª "ÿÿÿÿÿÿÿÿÿºÕ
E__inference_lstm_11_layer_call_and_return_conditional_losses_19557867)*+P¢M
F¢C
52
0-
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿº

 
p 

 
ª "2¢/
(%
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿS
 Õ
E__inference_lstm_11_layer_call_and_return_conditional_losses_19558018)*+P¢M
F¢C
52
0-
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿº

 
p

 
ª "2¢/
(%
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿS
 »
E__inference_lstm_11_layer_call_and_return_conditional_losses_19558169r)*+@¢=
6¢3
%"
inputsÿÿÿÿÿÿÿÿÿº

 
p 

 
ª ")¢&

0ÿÿÿÿÿÿÿÿÿS
 »
E__inference_lstm_11_layer_call_and_return_conditional_losses_19558320r)*+@¢=
6¢3
%"
inputsÿÿÿÿÿÿÿÿÿº

 
p

 
ª ")¢&

0ÿÿÿÿÿÿÿÿÿS
 ¬
*__inference_lstm_11_layer_call_fn_19558331~)*+P¢M
F¢C
52
0-
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿº

 
p 

 
ª "%"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿS¬
*__inference_lstm_11_layer_call_fn_19558342~)*+P¢M
F¢C
52
0-
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿº

 
p

 
ª "%"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿS
*__inference_lstm_11_layer_call_fn_19558353e)*+@¢=
6¢3
%"
inputsÿÿÿÿÿÿÿÿÿº

 
p 

 
ª "ÿÿÿÿÿÿÿÿÿS
*__inference_lstm_11_layer_call_fn_19558364e)*+@¢=
6¢3
%"
inputsÿÿÿÿÿÿÿÿÿº

 
p

 
ª "ÿÿÿÿÿÿÿÿÿSÑ
J__inference_lstm_cell_10_layer_call_and_return_conditional_losses_19558463&'(¢
x¢u
 
inputsÿÿÿÿÿÿÿÿÿ]
M¢J
# 
states/0ÿÿÿÿÿÿÿÿÿº
# 
states/1ÿÿÿÿÿÿÿÿÿº
p 
ª "v¢s
l¢i

0/0ÿÿÿÿÿÿÿÿÿº
GD
 
0/1/0ÿÿÿÿÿÿÿÿÿº
 
0/1/1ÿÿÿÿÿÿÿÿÿº
 Ñ
J__inference_lstm_cell_10_layer_call_and_return_conditional_losses_19558495&'(¢
x¢u
 
inputsÿÿÿÿÿÿÿÿÿ]
M¢J
# 
states/0ÿÿÿÿÿÿÿÿÿº
# 
states/1ÿÿÿÿÿÿÿÿÿº
p
ª "v¢s
l¢i

0/0ÿÿÿÿÿÿÿÿÿº
GD
 
0/1/0ÿÿÿÿÿÿÿÿÿº
 
0/1/1ÿÿÿÿÿÿÿÿÿº
 ¦
/__inference_lstm_cell_10_layer_call_fn_19558512ò&'(¢
x¢u
 
inputsÿÿÿÿÿÿÿÿÿ]
M¢J
# 
states/0ÿÿÿÿÿÿÿÿÿº
# 
states/1ÿÿÿÿÿÿÿÿÿº
p 
ª "f¢c

0ÿÿÿÿÿÿÿÿÿº
C@

1/0ÿÿÿÿÿÿÿÿÿº

1/1ÿÿÿÿÿÿÿÿÿº¦
/__inference_lstm_cell_10_layer_call_fn_19558529ò&'(¢
x¢u
 
inputsÿÿÿÿÿÿÿÿÿ]
M¢J
# 
states/0ÿÿÿÿÿÿÿÿÿº
# 
states/1ÿÿÿÿÿÿÿÿÿº
p
ª "f¢c

0ÿÿÿÿÿÿÿÿÿº
C@

1/0ÿÿÿÿÿÿÿÿÿº

1/1ÿÿÿÿÿÿÿÿÿºÍ
J__inference_lstm_cell_11_layer_call_and_return_conditional_losses_19558561þ)*+¢~
w¢t
!
inputsÿÿÿÿÿÿÿÿÿº
K¢H
"
states/0ÿÿÿÿÿÿÿÿÿS
"
states/1ÿÿÿÿÿÿÿÿÿS
p 
ª "s¢p
i¢f

0/0ÿÿÿÿÿÿÿÿÿS
EB

0/1/0ÿÿÿÿÿÿÿÿÿS

0/1/1ÿÿÿÿÿÿÿÿÿS
 Í
J__inference_lstm_cell_11_layer_call_and_return_conditional_losses_19558593þ)*+¢~
w¢t
!
inputsÿÿÿÿÿÿÿÿÿº
K¢H
"
states/0ÿÿÿÿÿÿÿÿÿS
"
states/1ÿÿÿÿÿÿÿÿÿS
p
ª "s¢p
i¢f

0/0ÿÿÿÿÿÿÿÿÿS
EB

0/1/0ÿÿÿÿÿÿÿÿÿS

0/1/1ÿÿÿÿÿÿÿÿÿS
 ¢
/__inference_lstm_cell_11_layer_call_fn_19558610î)*+¢~
w¢t
!
inputsÿÿÿÿÿÿÿÿÿº
K¢H
"
states/0ÿÿÿÿÿÿÿÿÿS
"
states/1ÿÿÿÿÿÿÿÿÿS
p 
ª "c¢`

0ÿÿÿÿÿÿÿÿÿS
A>

1/0ÿÿÿÿÿÿÿÿÿS

1/1ÿÿÿÿÿÿÿÿÿS¢
/__inference_lstm_cell_11_layer_call_fn_19558627î)*+¢~
w¢t
!
inputsÿÿÿÿÿÿÿÿÿº
K¢H
"
states/0ÿÿÿÿÿÿÿÿÿS
"
states/1ÿÿÿÿÿÿÿÿÿS
p
ª "c¢`

0ÿÿÿÿÿÿÿÿÿS
A>

1/0ÿÿÿÿÿÿÿÿÿS

1/1ÿÿÿÿÿÿÿÿÿSÇ
J__inference_sequential_5_layer_call_and_return_conditional_losses_19556283y&'()*+ !B¢?
8¢5
+(
lstm_10_inputÿÿÿÿÿÿÿÿÿ]
p 

 
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ
 Ç
J__inference_sequential_5_layer_call_and_return_conditional_losses_19556308y&'()*+ !B¢?
8¢5
+(
lstm_10_inputÿÿÿÿÿÿÿÿÿ]
p

 
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ
 À
J__inference_sequential_5_layer_call_and_return_conditional_losses_19556658r&'()*+ !;¢8
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
J__inference_sequential_5_layer_call_and_return_conditional_losses_19556999r&'()*+ !;¢8
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
/__inference_sequential_5_layer_call_fn_19555768l&'()*+ !B¢?
8¢5
+(
lstm_10_inputÿÿÿÿÿÿÿÿÿ]
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
/__inference_sequential_5_layer_call_fn_19556258l&'()*+ !B¢?
8¢5
+(
lstm_10_inputÿÿÿÿÿÿÿÿÿ]
p

 
ª "ÿÿÿÿÿÿÿÿÿ
/__inference_sequential_5_layer_call_fn_19557020e&'()*+ !;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿ]
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
/__inference_sequential_5_layer_call_fn_19557041e&'()*+ !;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿ]
p

 
ª "ÿÿÿÿÿÿÿÿÿ¹
&__inference_signature_wrapper_19556331&'()*+ !K¢H
¢ 
Aª>
<
lstm_10_input+(
lstm_10_inputÿÿÿÿÿÿÿÿÿ]"5ª2
0
dense_5%"
dense_5ÿÿÿÿÿÿÿÿÿ