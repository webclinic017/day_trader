Ì&
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
"serve*2.6.02v2.6.0-rc2-32-g919f693420e8­½$
y
dense_9/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	×*
shared_namedense_9/kernel
r
"dense_9/kernel/Read/ReadVariableOpReadVariableOpdense_9/kernel*
_output_shapes
:	×*
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

lstm_18/lstm_cell_18/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	]Ô*,
shared_namelstm_18/lstm_cell_18/kernel

/lstm_18/lstm_cell_18/kernel/Read/ReadVariableOpReadVariableOplstm_18/lstm_cell_18/kernel*
_output_shapes
:	]Ô*
dtype0
¨
%lstm_18/lstm_cell_18/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
Ô*6
shared_name'%lstm_18/lstm_cell_18/recurrent_kernel
¡
9lstm_18/lstm_cell_18/recurrent_kernel/Read/ReadVariableOpReadVariableOp%lstm_18/lstm_cell_18/recurrent_kernel* 
_output_shapes
:
Ô*
dtype0

lstm_18/lstm_cell_18/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:Ô**
shared_namelstm_18/lstm_cell_18/bias

-lstm_18/lstm_cell_18/bias/Read/ReadVariableOpReadVariableOplstm_18/lstm_cell_18/bias*
_output_shapes	
:Ô*
dtype0

lstm_19/lstm_cell_19/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
Ü*,
shared_namelstm_19/lstm_cell_19/kernel

/lstm_19/lstm_cell_19/kernel/Read/ReadVariableOpReadVariableOplstm_19/lstm_cell_19/kernel* 
_output_shapes
:
Ü*
dtype0
¨
%lstm_19/lstm_cell_19/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
×Ü*6
shared_name'%lstm_19/lstm_cell_19/recurrent_kernel
¡
9lstm_19/lstm_cell_19/recurrent_kernel/Read/ReadVariableOpReadVariableOp%lstm_19/lstm_cell_19/recurrent_kernel* 
_output_shapes
:
×Ü*
dtype0

lstm_19/lstm_cell_19/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:Ü**
shared_namelstm_19/lstm_cell_19/bias

-lstm_19/lstm_cell_19/bias/Read/ReadVariableOpReadVariableOplstm_19/lstm_cell_19/bias*
_output_shapes	
:Ü*
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
,metrics
trainable_variables
regularization_losses
-layer_regularization_losses
.non_trainable_variables

/layers
		variables
0layer_metrics
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
­
<metrics
trainable_variables
=layer_regularization_losses
>non_trainable_variables
regularization_losses

?layers
	variables
@layer_metrics
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
­
Lmetrics
trainable_variables
Mlayer_regularization_losses
Nnon_trainable_variables
regularization_losses

Olayers
	variables
Player_metrics
ZX
VARIABLE_VALUEdense_9/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_9/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

 0
!1
 

 0
!1
­
Qmetrics
"trainable_variables
Rlayer_regularization_losses
Snon_trainable_variables
#regularization_losses

Tlayers
$	variables
Ulayer_metrics
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
­
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
­
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

serving_default_lstm_18_inputPlaceholder*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]*
dtype0* 
shape:ÿÿÿÿÿÿÿÿÿ]
ª
StatefulPartitionedCallStatefulPartitionedCallserving_default_lstm_18_inputlstm_18/lstm_cell_18/kernel%lstm_18/lstm_cell_18/recurrent_kernellstm_18/lstm_cell_18/biaslstm_19/lstm_cell_19/kernel%lstm_19/lstm_cell_19/recurrent_kernellstm_19/lstm_cell_19/biasdense_9/kerneldense_9/bias*
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
&__inference_signature_wrapper_32580911
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 

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
GPU 2J 8 **
f%R#
!__inference__traced_save_32583266
 
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
GPU 2J 8 *-
f(R&
$__inference__traced_restore_32583312Ôò#
ã
Í
while_cond_32580026
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_32580026___redundant_placeholder06
2while_while_cond_32580026___redundant_placeholder16
2while_while_cond_32580026___redundant_placeholder26
2while_while_cond_32580026___redundant_placeholder3
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
B: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
ã
»
*__inference_lstm_19_layer_call_fn_32582911
inputs_0
unknown:
Ü
	unknown_0:
×Ü
	unknown_1:	Ü
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ×*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_lstm_19_layer_call_and_return_conditional_losses_325794822
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ×2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0

f
H__inference_dropout_18_layer_call_and_return_conditional_losses_32582274

inputs

identity_1_
IdentityIdentityinputs*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityn

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
×
g
H__inference_dropout_19_layer_call_and_return_conditional_losses_32580378

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
:ÿÿÿÿÿÿÿÿÿ×2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape¹
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×*
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
:ÿÿÿÿÿÿÿÿÿ×2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2
dropout/Mul_1j
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ×:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×
 
_user_specified_nameinputs
ã
Í
while_cond_32582664
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_32582664___redundant_placeholder06
2while_while_cond_32582664___redundant_placeholder16
2while_while_cond_32582664___redundant_placeholder26
2while_while_cond_32582664___redundant_placeholder3
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
B: : : : :ÿÿÿÿÿÿÿÿÿ×:ÿÿÿÿÿÿÿÿÿ×: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×:

_output_shapes
: :

_output_shapes
:
\

E__inference_lstm_19_layer_call_and_return_conditional_losses_32582749

inputs?
+lstm_cell_19_matmul_readvariableop_resource:
ÜA
-lstm_cell_19_matmul_1_readvariableop_resource:
×Ü;
,lstm_cell_19_biasadd_readvariableop_resource:	Ü
identity¢#lstm_cell_19/BiasAdd/ReadVariableOp¢"lstm_cell_19/MatMul/ReadVariableOp¢$lstm_cell_19/MatMul_1/ReadVariableOp¢whileD
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
B :×2
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
B :×2
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
:ÿÿÿÿÿÿÿÿÿ×2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :×2
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
B :×2
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
:ÿÿÿÿÿÿÿÿÿ×2	
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
:ÿÿÿÿÿÿÿÿÿ2
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
valueB"ÿÿÿÿ  27
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
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
strided_slice_2¶
"lstm_cell_19/MatMul/ReadVariableOpReadVariableOp+lstm_cell_19_matmul_readvariableop_resource* 
_output_shapes
:
Ü*
dtype02$
"lstm_cell_19/MatMul/ReadVariableOp­
lstm_cell_19/MatMulMatMulstrided_slice_2:output:0*lstm_cell_19/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2
lstm_cell_19/MatMul¼
$lstm_cell_19/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_19_matmul_1_readvariableop_resource* 
_output_shapes
:
×Ü*
dtype02&
$lstm_cell_19/MatMul_1/ReadVariableOp©
lstm_cell_19/MatMul_1MatMulzeros:output:0,lstm_cell_19/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2
lstm_cell_19/MatMul_1 
lstm_cell_19/addAddV2lstm_cell_19/MatMul:product:0lstm_cell_19/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2
lstm_cell_19/add´
#lstm_cell_19/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_19_biasadd_readvariableop_resource*
_output_shapes	
:Ü*
dtype02%
#lstm_cell_19/BiasAdd/ReadVariableOp­
lstm_cell_19/BiasAddBiasAddlstm_cell_19/add:z:0+lstm_cell_19/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2
lstm_cell_19/BiasAdd~
lstm_cell_19/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_19/split/split_dim÷
lstm_cell_19/splitSplit%lstm_cell_19/split/split_dim:output:0lstm_cell_19/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ×:ÿÿÿÿÿÿÿÿÿ×:ÿÿÿÿÿÿÿÿÿ×:ÿÿÿÿÿÿÿÿÿ×*
	num_split2
lstm_cell_19/split
lstm_cell_19/SigmoidSigmoidlstm_cell_19/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2
lstm_cell_19/Sigmoid
lstm_cell_19/Sigmoid_1Sigmoidlstm_cell_19/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2
lstm_cell_19/Sigmoid_1
lstm_cell_19/mulMullstm_cell_19/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2
lstm_cell_19/mul~
lstm_cell_19/ReluRelulstm_cell_19/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2
lstm_cell_19/Relu
lstm_cell_19/mul_1Mullstm_cell_19/Sigmoid:y:0lstm_cell_19/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2
lstm_cell_19/mul_1
lstm_cell_19/add_1AddV2lstm_cell_19/mul:z:0lstm_cell_19/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2
lstm_cell_19/add_1
lstm_cell_19/Sigmoid_2Sigmoidlstm_cell_19/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2
lstm_cell_19/Sigmoid_2}
lstm_cell_19/Relu_1Relulstm_cell_19/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2
lstm_cell_19/Relu_1¡
lstm_cell_19/mul_2Mullstm_cell_19/Sigmoid_2:y:0!lstm_cell_19/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2
lstm_cell_19/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ×   2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_19_matmul_readvariableop_resource-lstm_cell_19_matmul_1_readvariableop_resource,lstm_cell_19_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ×:ÿÿÿÿÿÿÿÿÿ×: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_32582665*
condR
while_cond_32582664*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ×:ÿÿÿÿÿÿÿÿÿ×: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ×   22
0TensorArrayV2Stack/TensorListStack/element_shapeé
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×*
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
:ÿÿÿÿÿÿÿÿÿ×*
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
:ÿÿÿÿÿÿÿÿÿ×2
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
:ÿÿÿÿÿÿÿÿÿ×2

IdentityÈ
NoOpNoOp$^lstm_cell_19/BiasAdd/ReadVariableOp#^lstm_cell_19/MatMul/ReadVariableOp%^lstm_cell_19/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:ÿÿÿÿÿÿÿÿÿ: : : 2J
#lstm_cell_19/BiasAdd/ReadVariableOp#lstm_cell_19/BiasAdd/ReadVariableOp2H
"lstm_cell_19/MatMul/ReadVariableOp"lstm_cell_19/MatMul/ReadVariableOp2L
$lstm_cell_19/MatMul_1/ReadVariableOp$lstm_cell_19/MatMul_1/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ã
Í
while_cond_32581989
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_32581989___redundant_placeholder06
2while_while_cond_32581989___redundant_placeholder16
2while_while_cond_32581989___redundant_placeholder26
2while_while_cond_32581989___redundant_placeholder3
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
B: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:


Ì
/__inference_sequential_9_layer_call_fn_32581621

inputs
unknown:	]Ô
	unknown_0:
Ô
	unknown_1:	Ô
	unknown_2:
Ü
	unknown_3:
×Ü
	unknown_4:	Ü
	unknown_5:	×
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
J__inference_sequential_9_layer_call_and_return_conditional_losses_325807982
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


J__inference_lstm_cell_19_layer_call_and_return_conditional_losses_32583173

inputs
states_0
states_12
matmul_readvariableop_resource:
Ü4
 matmul_1_readvariableop_resource:
×Ü.
biasadd_readvariableop_resource:	Ü
identity

identity_1

identity_2¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
Ü*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2
MatMul
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
×Ü*
dtype02
MatMul_1/ReadVariableOp|
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2
add
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:Ü*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2	
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
P:ÿÿÿÿÿÿÿÿÿ×:ÿÿÿÿÿÿÿÿÿ×:ÿÿÿÿÿÿÿÿÿ×:ÿÿÿÿÿÿÿÿÿ×*
	num_split2
split`
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2	
Sigmoidd
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2
	Sigmoid_1]
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2
mulW
ReluRelusplit:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2
Relui
mul_1MulSigmoid:y:0Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2
mul_1^
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2
add_1d
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2
	Sigmoid_2V
Relu_1Relu	add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2
Relu_1m
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2
mul_2e
IdentityIdentity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2

Identityi

Identity_1Identity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2

Identity_1i

Identity_2Identity	add_1:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2

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
B:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ×:ÿÿÿÿÿÿÿÿÿ×: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×
"
_user_specified_name
states/0:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×
"
_user_specified_name
states/1


J__inference_lstm_cell_18_layer_call_and_return_conditional_losses_32583043

inputs
states_0
states_11
matmul_readvariableop_resource:	]Ô4
 matmul_1_readvariableop_resource:
Ô.
biasadd_readvariableop_resource:	Ô
identity

identity_1

identity_2¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	]Ô*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÔ2
MatMul
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
Ô*
dtype02
MatMul_1/ReadVariableOp|
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÔ2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÔ2
add
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:Ô*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÔ2	
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
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2
split`
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
Sigmoidd
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	Sigmoid_1]
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mulW
ReluRelusplit:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relui
mul_1MulSigmoid:y:0Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mul_1^
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
add_1d
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	Sigmoid_2V
Relu_1Relu	add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu_1m
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mul_2e
IdentityIdentity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityi

Identity_1Identity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1i

Identity_2Identity	add_1:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

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
A:ÿÿÿÿÿÿÿÿÿ]:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
 
_user_specified_nameinputs:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
states/0:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
states/1
½
¾
J__inference_sequential_9_layer_call_and_return_conditional_losses_32580888
lstm_18_input#
lstm_18_32580866:	]Ô$
lstm_18_32580868:
Ô
lstm_18_32580870:	Ô$
lstm_19_32580874:
Ü$
lstm_19_32580876:
×Ü
lstm_19_32580878:	Ü#
dense_9_32580882:	×
dense_9_32580884:
identity¢dense_9/StatefulPartitionedCall¢"dropout_18/StatefulPartitionedCall¢"dropout_19/StatefulPartitionedCall¢lstm_18/StatefulPartitionedCall¢lstm_19/StatefulPartitionedCallµ
lstm_18/StatefulPartitionedCallStatefulPartitionedCalllstm_18_inputlstm_18_32580866lstm_18_32580868lstm_18_32580870*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_lstm_18_layer_call_and_return_conditional_losses_325807412!
lstm_18/StatefulPartitionedCall
"dropout_18/StatefulPartitionedCallStatefulPartitionedCall(lstm_18/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_dropout_18_layer_call_and_return_conditional_losses_325805742$
"dropout_18/StatefulPartitionedCallÓ
lstm_19/StatefulPartitionedCallStatefulPartitionedCall+dropout_18/StatefulPartitionedCall:output:0lstm_19_32580874lstm_19_32580876lstm_19_32580878*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_lstm_19_layer_call_and_return_conditional_losses_325805452!
lstm_19/StatefulPartitionedCallÀ
"dropout_19/StatefulPartitionedCallStatefulPartitionedCall(lstm_19/StatefulPartitionedCall:output:0#^dropout_18/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_dropout_19_layer_call_and_return_conditional_losses_325803782$
"dropout_19/StatefulPartitionedCall¾
dense_9/StatefulPartitionedCallStatefulPartitionedCall+dropout_19/StatefulPartitionedCall:output:0dense_9_32580882dense_9_32580884*
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
E__inference_dense_9_layer_call_and_return_conditional_losses_325803222!
dense_9/StatefulPartitionedCall
IdentityIdentity(dense_9/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityþ
NoOpNoOp ^dense_9/StatefulPartitionedCall#^dropout_18/StatefulPartitionedCall#^dropout_19/StatefulPartitionedCall ^lstm_18/StatefulPartitionedCall ^lstm_19/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ]: : : : : : : : 2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall2H
"dropout_18/StatefulPartitionedCall"dropout_18/StatefulPartitionedCall2H
"dropout_19/StatefulPartitionedCall"dropout_19/StatefulPartitionedCall2B
lstm_18/StatefulPartitionedCalllstm_18/StatefulPartitionedCall2B
lstm_19/StatefulPartitionedCalllstm_19/StatefulPartitionedCall:Z V
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
'
_user_specified_namelstm_18_input
Û
ñ
(sequential_9_lstm_18_while_cond_32578433F
Bsequential_9_lstm_18_while_sequential_9_lstm_18_while_loop_counterL
Hsequential_9_lstm_18_while_sequential_9_lstm_18_while_maximum_iterations*
&sequential_9_lstm_18_while_placeholder,
(sequential_9_lstm_18_while_placeholder_1,
(sequential_9_lstm_18_while_placeholder_2,
(sequential_9_lstm_18_while_placeholder_3H
Dsequential_9_lstm_18_while_less_sequential_9_lstm_18_strided_slice_1`
\sequential_9_lstm_18_while_sequential_9_lstm_18_while_cond_32578433___redundant_placeholder0`
\sequential_9_lstm_18_while_sequential_9_lstm_18_while_cond_32578433___redundant_placeholder1`
\sequential_9_lstm_18_while_sequential_9_lstm_18_while_cond_32578433___redundant_placeholder2`
\sequential_9_lstm_18_while_sequential_9_lstm_18_while_cond_32578433___redundant_placeholder3'
#sequential_9_lstm_18_while_identity
Ù
sequential_9/lstm_18/while/LessLess&sequential_9_lstm_18_while_placeholderDsequential_9_lstm_18_while_less_sequential_9_lstm_18_strided_slice_1*
T0*
_output_shapes
: 2!
sequential_9/lstm_18/while/Less
#sequential_9/lstm_18/while/IdentityIdentity#sequential_9/lstm_18/while/Less:z:0*
T0
*
_output_shapes
: 2%
#sequential_9/lstm_18/while/Identity"S
#sequential_9_lstm_18_while_identity,sequential_9/lstm_18/while/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
ã
Í
while_cond_32582513
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_32582513___redundant_placeholder06
2while_while_cond_32582513___redundant_placeholder16
2while_while_cond_32582513___redundant_placeholder26
2while_while_cond_32582513___redundant_placeholder3
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
B: : : : :ÿÿÿÿÿÿÿÿÿ×:ÿÿÿÿÿÿÿÿÿ×: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×:

_output_shapes
: :

_output_shapes
:


Ì
/__inference_sequential_9_layer_call_fn_32581600

inputs
unknown:	]Ô
	unknown_0:
Ô
	unknown_1:	Ô
	unknown_2:
Ü
	unknown_3:
×Ü
	unknown_4:	Ü
	unknown_5:	×
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
J__inference_sequential_9_layer_call_and_return_conditional_losses_325803292
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
éJ
Ö

lstm_19_while_body_32581460,
(lstm_19_while_lstm_19_while_loop_counter2
.lstm_19_while_lstm_19_while_maximum_iterations
lstm_19_while_placeholder
lstm_19_while_placeholder_1
lstm_19_while_placeholder_2
lstm_19_while_placeholder_3+
'lstm_19_while_lstm_19_strided_slice_1_0g
clstm_19_while_tensorarrayv2read_tensorlistgetitem_lstm_19_tensorarrayunstack_tensorlistfromtensor_0O
;lstm_19_while_lstm_cell_19_matmul_readvariableop_resource_0:
ÜQ
=lstm_19_while_lstm_cell_19_matmul_1_readvariableop_resource_0:
×ÜK
<lstm_19_while_lstm_cell_19_biasadd_readvariableop_resource_0:	Ü
lstm_19_while_identity
lstm_19_while_identity_1
lstm_19_while_identity_2
lstm_19_while_identity_3
lstm_19_while_identity_4
lstm_19_while_identity_5)
%lstm_19_while_lstm_19_strided_slice_1e
alstm_19_while_tensorarrayv2read_tensorlistgetitem_lstm_19_tensorarrayunstack_tensorlistfromtensorM
9lstm_19_while_lstm_cell_19_matmul_readvariableop_resource:
ÜO
;lstm_19_while_lstm_cell_19_matmul_1_readvariableop_resource:
×ÜI
:lstm_19_while_lstm_cell_19_biasadd_readvariableop_resource:	Ü¢1lstm_19/while/lstm_cell_19/BiasAdd/ReadVariableOp¢0lstm_19/while/lstm_cell_19/MatMul/ReadVariableOp¢2lstm_19/while/lstm_cell_19/MatMul_1/ReadVariableOpÓ
?lstm_19/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ  2A
?lstm_19/while/TensorArrayV2Read/TensorListGetItem/element_shape
1lstm_19/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemclstm_19_while_tensorarrayv2read_tensorlistgetitem_lstm_19_tensorarrayunstack_tensorlistfromtensor_0lstm_19_while_placeholderHlstm_19/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype023
1lstm_19/while/TensorArrayV2Read/TensorListGetItemâ
0lstm_19/while/lstm_cell_19/MatMul/ReadVariableOpReadVariableOp;lstm_19_while_lstm_cell_19_matmul_readvariableop_resource_0* 
_output_shapes
:
Ü*
dtype022
0lstm_19/while/lstm_cell_19/MatMul/ReadVariableOp÷
!lstm_19/while/lstm_cell_19/MatMulMatMul8lstm_19/while/TensorArrayV2Read/TensorListGetItem:item:08lstm_19/while/lstm_cell_19/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2#
!lstm_19/while/lstm_cell_19/MatMulè
2lstm_19/while/lstm_cell_19/MatMul_1/ReadVariableOpReadVariableOp=lstm_19_while_lstm_cell_19_matmul_1_readvariableop_resource_0* 
_output_shapes
:
×Ü*
dtype024
2lstm_19/while/lstm_cell_19/MatMul_1/ReadVariableOpà
#lstm_19/while/lstm_cell_19/MatMul_1MatMullstm_19_while_placeholder_2:lstm_19/while/lstm_cell_19/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2%
#lstm_19/while/lstm_cell_19/MatMul_1Ø
lstm_19/while/lstm_cell_19/addAddV2+lstm_19/while/lstm_cell_19/MatMul:product:0-lstm_19/while/lstm_cell_19/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2 
lstm_19/while/lstm_cell_19/addà
1lstm_19/while/lstm_cell_19/BiasAdd/ReadVariableOpReadVariableOp<lstm_19_while_lstm_cell_19_biasadd_readvariableop_resource_0*
_output_shapes	
:Ü*
dtype023
1lstm_19/while/lstm_cell_19/BiasAdd/ReadVariableOpå
"lstm_19/while/lstm_cell_19/BiasAddBiasAdd"lstm_19/while/lstm_cell_19/add:z:09lstm_19/while/lstm_cell_19/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2$
"lstm_19/while/lstm_cell_19/BiasAdd
*lstm_19/while/lstm_cell_19/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2,
*lstm_19/while/lstm_cell_19/split/split_dim¯
 lstm_19/while/lstm_cell_19/splitSplit3lstm_19/while/lstm_cell_19/split/split_dim:output:0+lstm_19/while/lstm_cell_19/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ×:ÿÿÿÿÿÿÿÿÿ×:ÿÿÿÿÿÿÿÿÿ×:ÿÿÿÿÿÿÿÿÿ×*
	num_split2"
 lstm_19/while/lstm_cell_19/split±
"lstm_19/while/lstm_cell_19/SigmoidSigmoid)lstm_19/while/lstm_cell_19/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2$
"lstm_19/while/lstm_cell_19/Sigmoidµ
$lstm_19/while/lstm_cell_19/Sigmoid_1Sigmoid)lstm_19/while/lstm_cell_19/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2&
$lstm_19/while/lstm_cell_19/Sigmoid_1Á
lstm_19/while/lstm_cell_19/mulMul(lstm_19/while/lstm_cell_19/Sigmoid_1:y:0lstm_19_while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2 
lstm_19/while/lstm_cell_19/mul¨
lstm_19/while/lstm_cell_19/ReluRelu)lstm_19/while/lstm_cell_19/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2!
lstm_19/while/lstm_cell_19/ReluÕ
 lstm_19/while/lstm_cell_19/mul_1Mul&lstm_19/while/lstm_cell_19/Sigmoid:y:0-lstm_19/while/lstm_cell_19/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2"
 lstm_19/while/lstm_cell_19/mul_1Ê
 lstm_19/while/lstm_cell_19/add_1AddV2"lstm_19/while/lstm_cell_19/mul:z:0$lstm_19/while/lstm_cell_19/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2"
 lstm_19/while/lstm_cell_19/add_1µ
$lstm_19/while/lstm_cell_19/Sigmoid_2Sigmoid)lstm_19/while/lstm_cell_19/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2&
$lstm_19/while/lstm_cell_19/Sigmoid_2§
!lstm_19/while/lstm_cell_19/Relu_1Relu$lstm_19/while/lstm_cell_19/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2#
!lstm_19/while/lstm_cell_19/Relu_1Ù
 lstm_19/while/lstm_cell_19/mul_2Mul(lstm_19/while/lstm_cell_19/Sigmoid_2:y:0/lstm_19/while/lstm_cell_19/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2"
 lstm_19/while/lstm_cell_19/mul_2
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
lstm_19/while/add/y
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
lstm_19/while/add_1/y
lstm_19/while/add_1AddV2(lstm_19_while_lstm_19_while_loop_counterlstm_19/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lstm_19/while/add_1
lstm_19/while/IdentityIdentitylstm_19/while/add_1:z:0^lstm_19/while/NoOp*
T0*
_output_shapes
: 2
lstm_19/while/Identity¦
lstm_19/while/Identity_1Identity.lstm_19_while_lstm_19_while_maximum_iterations^lstm_19/while/NoOp*
T0*
_output_shapes
: 2
lstm_19/while/Identity_1
lstm_19/while/Identity_2Identitylstm_19/while/add:z:0^lstm_19/while/NoOp*
T0*
_output_shapes
: 2
lstm_19/while/Identity_2º
lstm_19/while/Identity_3IdentityBlstm_19/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_19/while/NoOp*
T0*
_output_shapes
: 2
lstm_19/while/Identity_3®
lstm_19/while/Identity_4Identity$lstm_19/while/lstm_cell_19/mul_2:z:0^lstm_19/while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2
lstm_19/while/Identity_4®
lstm_19/while/Identity_5Identity$lstm_19/while/lstm_cell_19/add_1:z:0^lstm_19/while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2
lstm_19/while/Identity_5
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
9lstm_19_while_lstm_cell_19_matmul_readvariableop_resource;lstm_19_while_lstm_cell_19_matmul_readvariableop_resource_0"È
alstm_19_while_tensorarrayv2read_tensorlistgetitem_lstm_19_tensorarrayunstack_tensorlistfromtensorclstm_19_while_tensorarrayv2read_tensorlistgetitem_lstm_19_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ×:ÿÿÿÿÿÿÿÿÿ×: : : : : 2f
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
:ÿÿÿÿÿÿÿÿÿ×:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×:

_output_shapes
: :

_output_shapes
: 

f
H__inference_dropout_19_layer_call_and_return_conditional_losses_32582949

inputs

identity_1_
IdentityIdentityinputs*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2

Identityn

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ×:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×
 
_user_specified_nameinputs
ã
Í
while_cond_32582140
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_32582140___redundant_placeholder06
2while_while_cond_32582140___redundant_placeholder16
2while_while_cond_32582140___redundant_placeholder26
2while_while_cond_32582140___redundant_placeholder3
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
B: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
´?
Ö
while_body_32582816
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
3while_lstm_cell_19_matmul_readvariableop_resource_0:
ÜI
5while_lstm_cell_19_matmul_1_readvariableop_resource_0:
×ÜC
4while_lstm_cell_19_biasadd_readvariableop_resource_0:	Ü
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
1while_lstm_cell_19_matmul_readvariableop_resource:
ÜG
3while_lstm_cell_19_matmul_1_readvariableop_resource:
×ÜA
2while_lstm_cell_19_biasadd_readvariableop_resource:	Ü¢)while/lstm_cell_19/BiasAdd/ReadVariableOp¢(while/lstm_cell_19/MatMul/ReadVariableOp¢*while/lstm_cell_19/MatMul_1/ReadVariableOpÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ  29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÔ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemÊ
(while/lstm_cell_19/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_19_matmul_readvariableop_resource_0* 
_output_shapes
:
Ü*
dtype02*
(while/lstm_cell_19/MatMul/ReadVariableOp×
while/lstm_cell_19/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_19/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2
while/lstm_cell_19/MatMulÐ
*while/lstm_cell_19/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_19_matmul_1_readvariableop_resource_0* 
_output_shapes
:
×Ü*
dtype02,
*while/lstm_cell_19/MatMul_1/ReadVariableOpÀ
while/lstm_cell_19/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_19/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2
while/lstm_cell_19/MatMul_1¸
while/lstm_cell_19/addAddV2#while/lstm_cell_19/MatMul:product:0%while/lstm_cell_19/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2
while/lstm_cell_19/addÈ
)while/lstm_cell_19/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_19_biasadd_readvariableop_resource_0*
_output_shapes	
:Ü*
dtype02+
)while/lstm_cell_19/BiasAdd/ReadVariableOpÅ
while/lstm_cell_19/BiasAddBiasAddwhile/lstm_cell_19/add:z:01while/lstm_cell_19/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2
while/lstm_cell_19/BiasAdd
"while/lstm_cell_19/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_19/split/split_dim
while/lstm_cell_19/splitSplit+while/lstm_cell_19/split/split_dim:output:0#while/lstm_cell_19/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ×:ÿÿÿÿÿÿÿÿÿ×:ÿÿÿÿÿÿÿÿÿ×:ÿÿÿÿÿÿÿÿÿ×*
	num_split2
while/lstm_cell_19/split
while/lstm_cell_19/SigmoidSigmoid!while/lstm_cell_19/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2
while/lstm_cell_19/Sigmoid
while/lstm_cell_19/Sigmoid_1Sigmoid!while/lstm_cell_19/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2
while/lstm_cell_19/Sigmoid_1¡
while/lstm_cell_19/mulMul while/lstm_cell_19/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2
while/lstm_cell_19/mul
while/lstm_cell_19/ReluRelu!while/lstm_cell_19/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2
while/lstm_cell_19/Reluµ
while/lstm_cell_19/mul_1Mulwhile/lstm_cell_19/Sigmoid:y:0%while/lstm_cell_19/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2
while/lstm_cell_19/mul_1ª
while/lstm_cell_19/add_1AddV2while/lstm_cell_19/mul:z:0while/lstm_cell_19/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2
while/lstm_cell_19/add_1
while/lstm_cell_19/Sigmoid_2Sigmoid!while/lstm_cell_19/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2
while/lstm_cell_19/Sigmoid_2
while/lstm_cell_19/Relu_1Reluwhile/lstm_cell_19/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2
while/lstm_cell_19/Relu_1¹
while/lstm_cell_19/mul_2Mul while/lstm_cell_19/Sigmoid_2:y:0'while/lstm_cell_19/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2
while/lstm_cell_19/mul_2à
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
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_19/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_19/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2
while/Identity_5Þ

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
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ×:ÿÿÿÿÿÿÿÿÿ×: : : : : 2V
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
:ÿÿÿÿÿÿÿÿÿ×:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×:

_output_shapes
: :

_output_shapes
: 
Ã\
 
E__inference_lstm_18_layer_call_and_return_conditional_losses_32581772
inputs_0>
+lstm_cell_18_matmul_readvariableop_resource:	]ÔA
-lstm_cell_18_matmul_1_readvariableop_resource:
Ô;
,lstm_cell_18_biasadd_readvariableop_resource:	Ô
identity¢#lstm_cell_18/BiasAdd/ReadVariableOp¢"lstm_cell_18/MatMul/ReadVariableOp¢$lstm_cell_18/MatMul_1/ReadVariableOp¢whileF
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
B :2
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
B :2
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
:ÿÿÿÿÿÿÿÿÿ2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :2
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
B :2
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
:ÿÿÿÿÿÿÿÿÿ2	
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
"lstm_cell_18/MatMul/ReadVariableOpReadVariableOp+lstm_cell_18_matmul_readvariableop_resource*
_output_shapes
:	]Ô*
dtype02$
"lstm_cell_18/MatMul/ReadVariableOp­
lstm_cell_18/MatMulMatMulstrided_slice_2:output:0*lstm_cell_18/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÔ2
lstm_cell_18/MatMul¼
$lstm_cell_18/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_18_matmul_1_readvariableop_resource* 
_output_shapes
:
Ô*
dtype02&
$lstm_cell_18/MatMul_1/ReadVariableOp©
lstm_cell_18/MatMul_1MatMulzeros:output:0,lstm_cell_18/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÔ2
lstm_cell_18/MatMul_1 
lstm_cell_18/addAddV2lstm_cell_18/MatMul:product:0lstm_cell_18/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÔ2
lstm_cell_18/add´
#lstm_cell_18/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_18_biasadd_readvariableop_resource*
_output_shapes	
:Ô*
dtype02%
#lstm_cell_18/BiasAdd/ReadVariableOp­
lstm_cell_18/BiasAddBiasAddlstm_cell_18/add:z:0+lstm_cell_18/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÔ2
lstm_cell_18/BiasAdd~
lstm_cell_18/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_18/split/split_dim÷
lstm_cell_18/splitSplit%lstm_cell_18/split/split_dim:output:0lstm_cell_18/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2
lstm_cell_18/split
lstm_cell_18/SigmoidSigmoidlstm_cell_18/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_18/Sigmoid
lstm_cell_18/Sigmoid_1Sigmoidlstm_cell_18/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_18/Sigmoid_1
lstm_cell_18/mulMullstm_cell_18/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_18/mul~
lstm_cell_18/ReluRelulstm_cell_18/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_18/Relu
lstm_cell_18/mul_1Mullstm_cell_18/Sigmoid:y:0lstm_cell_18/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_18/mul_1
lstm_cell_18/add_1AddV2lstm_cell_18/mul:z:0lstm_cell_18/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_18/add_1
lstm_cell_18/Sigmoid_2Sigmoidlstm_cell_18/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_18/Sigmoid_2}
lstm_cell_18/Relu_1Relulstm_cell_18/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_18/Relu_1¡
lstm_cell_18/mul_2Mullstm_cell_18/Sigmoid_2:y:0!lstm_cell_18/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_18/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ  2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_18_matmul_readvariableop_resource-lstm_cell_18_matmul_1_readvariableop_resource,lstm_cell_18_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_32581688*
condR
while_cond_32581687*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ  22
0TensorArrayV2Stack/TensorListStack/element_shapeò
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
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
:ÿÿÿÿÿÿÿÿÿ*
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
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
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
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

IdentityÈ
NoOpNoOp$^lstm_cell_18/BiasAdd/ReadVariableOp#^lstm_cell_18/MatMul/ReadVariableOp%^lstm_cell_18/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ]: : : 2J
#lstm_cell_18/BiasAdd/ReadVariableOp#lstm_cell_18/BiasAdd/ReadVariableOp2H
"lstm_cell_18/MatMul/ReadVariableOp"lstm_cell_18/MatMul/ReadVariableOp2L
$lstm_cell_18/MatMul_1/ReadVariableOp$lstm_cell_18/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ]
"
_user_specified_name
inputs/0
ËF

E__inference_lstm_18_layer_call_and_return_conditional_losses_32578852

inputs(
lstm_cell_18_32578770:	]Ô)
lstm_cell_18_32578772:
Ô$
lstm_cell_18_32578774:	Ô
identity¢$lstm_cell_18/StatefulPartitionedCall¢whileD
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
B :2
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
B :2
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
:ÿÿÿÿÿÿÿÿÿ2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :2
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
B :2
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
:ÿÿÿÿÿÿÿÿÿ2	
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
$lstm_cell_18/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_18_32578770lstm_cell_18_32578772lstm_cell_18_32578774*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_lstm_cell_18_layer_call_and_return_conditional_losses_325787692&
$lstm_cell_18/StatefulPartitionedCall
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ  2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_18_32578770lstm_cell_18_32578772lstm_cell_18_32578774*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_32578783*
condR
while_cond_32578782*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ  22
0TensorArrayV2Stack/TensorListStack/element_shapeò
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
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
:ÿÿÿÿÿÿÿÿÿ*
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
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
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
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity}
NoOpNoOp%^lstm_cell_18/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ]: : : 2L
$lstm_cell_18/StatefulPartitionedCall$lstm_cell_18/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ]
 
_user_specified_nameinputs
ËF

E__inference_lstm_18_layer_call_and_return_conditional_losses_32579062

inputs(
lstm_cell_18_32578980:	]Ô)
lstm_cell_18_32578982:
Ô$
lstm_cell_18_32578984:	Ô
identity¢$lstm_cell_18/StatefulPartitionedCall¢whileD
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
B :2
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
B :2
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
:ÿÿÿÿÿÿÿÿÿ2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :2
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
B :2
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
:ÿÿÿÿÿÿÿÿÿ2	
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
$lstm_cell_18/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_18_32578980lstm_cell_18_32578982lstm_cell_18_32578984*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_lstm_cell_18_layer_call_and_return_conditional_losses_325789152&
$lstm_cell_18/StatefulPartitionedCall
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ  2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_18_32578980lstm_cell_18_32578982lstm_cell_18_32578984*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_32578993*
condR
while_cond_32578992*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ  22
0TensorArrayV2Stack/TensorListStack/element_shapeò
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
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
:ÿÿÿÿÿÿÿÿÿ*
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
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
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
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity}
NoOpNoOp%^lstm_cell_18/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ]: : : 2L
$lstm_cell_18/StatefulPartitionedCall$lstm_cell_18/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ]
 
_user_specified_nameinputs

í
J__inference_sequential_9_layer_call_and_return_conditional_losses_32580329

inputs#
lstm_18_32580112:	]Ô$
lstm_18_32580114:
Ô
lstm_18_32580116:	Ô$
lstm_19_32580277:
Ü$
lstm_19_32580279:
×Ü
lstm_19_32580281:	Ü#
dense_9_32580323:	×
dense_9_32580325:
identity¢dense_9/StatefulPartitionedCall¢lstm_18/StatefulPartitionedCall¢lstm_19/StatefulPartitionedCall®
lstm_18/StatefulPartitionedCallStatefulPartitionedCallinputslstm_18_32580112lstm_18_32580114lstm_18_32580116*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_lstm_18_layer_call_and_return_conditional_losses_325801112!
lstm_18/StatefulPartitionedCall
dropout_18/PartitionedCallPartitionedCall(lstm_18/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_dropout_18_layer_call_and_return_conditional_losses_325801242
dropout_18/PartitionedCallË
lstm_19/StatefulPartitionedCallStatefulPartitionedCall#dropout_18/PartitionedCall:output:0lstm_19_32580277lstm_19_32580279lstm_19_32580281*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_lstm_19_layer_call_and_return_conditional_losses_325802762!
lstm_19/StatefulPartitionedCall
dropout_19/PartitionedCallPartitionedCall(lstm_19/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_dropout_19_layer_call_and_return_conditional_losses_325802892
dropout_19/PartitionedCall¶
dense_9/StatefulPartitionedCallStatefulPartitionedCall#dropout_19/PartitionedCall:output:0dense_9_32580323dense_9_32580325*
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
E__inference_dense_9_layer_call_and_return_conditional_losses_325803222!
dense_9/StatefulPartitionedCall
IdentityIdentity(dense_9/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity´
NoOpNoOp ^dense_9/StatefulPartitionedCall ^lstm_18/StatefulPartitionedCall ^lstm_19/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ]: : : : : : : : 2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall2B
lstm_18/StatefulPartitionedCalllstm_18/StatefulPartitionedCall2B
lstm_19/StatefulPartitionedCalllstm_19/StatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
 
_user_specified_nameinputs
ã
Í
while_cond_32579412
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_32579412___redundant_placeholder06
2while_while_cond_32579412___redundant_placeholder16
2while_while_cond_32579412___redundant_placeholder26
2while_while_cond_32579412___redundant_placeholder3
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
B: : : : :ÿÿÿÿÿÿÿÿÿ×:ÿÿÿÿÿÿÿÿÿ×: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×:

_output_shapes
: :

_output_shapes
:


J__inference_sequential_9_layer_call_and_return_conditional_losses_32581579

inputsF
3lstm_18_lstm_cell_18_matmul_readvariableop_resource:	]ÔI
5lstm_18_lstm_cell_18_matmul_1_readvariableop_resource:
ÔC
4lstm_18_lstm_cell_18_biasadd_readvariableop_resource:	ÔG
3lstm_19_lstm_cell_19_matmul_readvariableop_resource:
ÜI
5lstm_19_lstm_cell_19_matmul_1_readvariableop_resource:
×ÜC
4lstm_19_lstm_cell_19_biasadd_readvariableop_resource:	Ü<
)dense_9_tensordot_readvariableop_resource:	×5
'dense_9_biasadd_readvariableop_resource:
identity¢dense_9/BiasAdd/ReadVariableOp¢ dense_9/Tensordot/ReadVariableOp¢+lstm_18/lstm_cell_18/BiasAdd/ReadVariableOp¢*lstm_18/lstm_cell_18/MatMul/ReadVariableOp¢,lstm_18/lstm_cell_18/MatMul_1/ReadVariableOp¢lstm_18/while¢+lstm_19/lstm_cell_19/BiasAdd/ReadVariableOp¢*lstm_19/lstm_cell_19/MatMul/ReadVariableOp¢,lstm_19/lstm_cell_19/MatMul_1/ReadVariableOp¢lstm_19/whileT
lstm_18/ShapeShapeinputs*
T0*
_output_shapes
:2
lstm_18/Shape
lstm_18/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_18/strided_slice/stack
lstm_18/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_18/strided_slice/stack_1
lstm_18/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_18/strided_slice/stack_2
lstm_18/strided_sliceStridedSlicelstm_18/Shape:output:0$lstm_18/strided_slice/stack:output:0&lstm_18/strided_slice/stack_1:output:0&lstm_18/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_18/strided_slicem
lstm_18/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :2
lstm_18/zeros/mul/y
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
B :è2
lstm_18/zeros/Less/y
lstm_18/zeros/LessLesslstm_18/zeros/mul:z:0lstm_18/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_18/zeros/Lesss
lstm_18/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :2
lstm_18/zeros/packed/1£
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
lstm_18/zeros/Const
lstm_18/zerosFilllstm_18/zeros/packed:output:0lstm_18/zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_18/zerosq
lstm_18/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :2
lstm_18/zeros_1/mul/y
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
B :è2
lstm_18/zeros_1/Less/y
lstm_18/zeros_1/LessLesslstm_18/zeros_1/mul:z:0lstm_18/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_18/zeros_1/Lessw
lstm_18/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :2
lstm_18/zeros_1/packed/1©
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
lstm_18/zeros_1/Const
lstm_18/zeros_1Filllstm_18/zeros_1/packed:output:0lstm_18/zeros_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_18/zeros_1
lstm_18/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_18/transpose/perm
lstm_18/transpose	Transposeinputslstm_18/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]2
lstm_18/transposeg
lstm_18/Shape_1Shapelstm_18/transpose:y:0*
T0*
_output_shapes
:2
lstm_18/Shape_1
lstm_18/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_18/strided_slice_1/stack
lstm_18/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_18/strided_slice_1/stack_1
lstm_18/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_18/strided_slice_1/stack_2
lstm_18/strided_slice_1StridedSlicelstm_18/Shape_1:output:0&lstm_18/strided_slice_1/stack:output:0(lstm_18/strided_slice_1/stack_1:output:0(lstm_18/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_18/strided_slice_1
#lstm_18/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2%
#lstm_18/TensorArrayV2/element_shapeÒ
lstm_18/TensorArrayV2TensorListReserve,lstm_18/TensorArrayV2/element_shape:output:0 lstm_18/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_18/TensorArrayV2Ï
=lstm_18/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ]   2?
=lstm_18/TensorArrayUnstack/TensorListFromTensor/element_shape
/lstm_18/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_18/transpose:y:0Flstm_18/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type021
/lstm_18/TensorArrayUnstack/TensorListFromTensor
lstm_18/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_18/strided_slice_2/stack
lstm_18/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_18/strided_slice_2/stack_1
lstm_18/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_18/strided_slice_2/stack_2¬
lstm_18/strided_slice_2StridedSlicelstm_18/transpose:y:0&lstm_18/strided_slice_2/stack:output:0(lstm_18/strided_slice_2/stack_1:output:0(lstm_18/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]*
shrink_axis_mask2
lstm_18/strided_slice_2Í
*lstm_18/lstm_cell_18/MatMul/ReadVariableOpReadVariableOp3lstm_18_lstm_cell_18_matmul_readvariableop_resource*
_output_shapes
:	]Ô*
dtype02,
*lstm_18/lstm_cell_18/MatMul/ReadVariableOpÍ
lstm_18/lstm_cell_18/MatMulMatMul lstm_18/strided_slice_2:output:02lstm_18/lstm_cell_18/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÔ2
lstm_18/lstm_cell_18/MatMulÔ
,lstm_18/lstm_cell_18/MatMul_1/ReadVariableOpReadVariableOp5lstm_18_lstm_cell_18_matmul_1_readvariableop_resource* 
_output_shapes
:
Ô*
dtype02.
,lstm_18/lstm_cell_18/MatMul_1/ReadVariableOpÉ
lstm_18/lstm_cell_18/MatMul_1MatMullstm_18/zeros:output:04lstm_18/lstm_cell_18/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÔ2
lstm_18/lstm_cell_18/MatMul_1À
lstm_18/lstm_cell_18/addAddV2%lstm_18/lstm_cell_18/MatMul:product:0'lstm_18/lstm_cell_18/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÔ2
lstm_18/lstm_cell_18/addÌ
+lstm_18/lstm_cell_18/BiasAdd/ReadVariableOpReadVariableOp4lstm_18_lstm_cell_18_biasadd_readvariableop_resource*
_output_shapes	
:Ô*
dtype02-
+lstm_18/lstm_cell_18/BiasAdd/ReadVariableOpÍ
lstm_18/lstm_cell_18/BiasAddBiasAddlstm_18/lstm_cell_18/add:z:03lstm_18/lstm_cell_18/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÔ2
lstm_18/lstm_cell_18/BiasAdd
$lstm_18/lstm_cell_18/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2&
$lstm_18/lstm_cell_18/split/split_dim
lstm_18/lstm_cell_18/splitSplit-lstm_18/lstm_cell_18/split/split_dim:output:0%lstm_18/lstm_cell_18/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2
lstm_18/lstm_cell_18/split
lstm_18/lstm_cell_18/SigmoidSigmoid#lstm_18/lstm_cell_18/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_18/lstm_cell_18/Sigmoid£
lstm_18/lstm_cell_18/Sigmoid_1Sigmoid#lstm_18/lstm_cell_18/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
lstm_18/lstm_cell_18/Sigmoid_1¬
lstm_18/lstm_cell_18/mulMul"lstm_18/lstm_cell_18/Sigmoid_1:y:0lstm_18/zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_18/lstm_cell_18/mul
lstm_18/lstm_cell_18/ReluRelu#lstm_18/lstm_cell_18/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_18/lstm_cell_18/Relu½
lstm_18/lstm_cell_18/mul_1Mul lstm_18/lstm_cell_18/Sigmoid:y:0'lstm_18/lstm_cell_18/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_18/lstm_cell_18/mul_1²
lstm_18/lstm_cell_18/add_1AddV2lstm_18/lstm_cell_18/mul:z:0lstm_18/lstm_cell_18/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_18/lstm_cell_18/add_1£
lstm_18/lstm_cell_18/Sigmoid_2Sigmoid#lstm_18/lstm_cell_18/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
lstm_18/lstm_cell_18/Sigmoid_2
lstm_18/lstm_cell_18/Relu_1Relulstm_18/lstm_cell_18/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_18/lstm_cell_18/Relu_1Á
lstm_18/lstm_cell_18/mul_2Mul"lstm_18/lstm_cell_18/Sigmoid_2:y:0)lstm_18/lstm_cell_18/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_18/lstm_cell_18/mul_2
%lstm_18/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ  2'
%lstm_18/TensorArrayV2_1/element_shapeØ
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
lstm_18/time
 lstm_18/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2"
 lstm_18/while/maximum_iterationsz
lstm_18/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_18/while/loop_counter
lstm_18/whileWhile#lstm_18/while/loop_counter:output:0)lstm_18/while/maximum_iterations:output:0lstm_18/time:output:0 lstm_18/TensorArrayV2_1:handle:0lstm_18/zeros:output:0lstm_18/zeros_1:output:0 lstm_18/strided_slice_1:output:0?lstm_18/TensorArrayUnstack/TensorListFromTensor:output_handle:03lstm_18_lstm_cell_18_matmul_readvariableop_resource5lstm_18_lstm_cell_18_matmul_1_readvariableop_resource4lstm_18_lstm_cell_18_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *'
bodyR
lstm_18_while_body_32581305*'
condR
lstm_18_while_cond_32581304*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 2
lstm_18/whileÅ
8lstm_18/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ  2:
8lstm_18/TensorArrayV2Stack/TensorListStack/element_shape
*lstm_18/TensorArrayV2Stack/TensorListStackTensorListStacklstm_18/while:output:3Alstm_18/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype02,
*lstm_18/TensorArrayV2Stack/TensorListStack
lstm_18/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
lstm_18/strided_slice_3/stack
lstm_18/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2!
lstm_18/strided_slice_3/stack_1
lstm_18/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_18/strided_slice_3/stack_2Ë
lstm_18/strided_slice_3StridedSlice3lstm_18/TensorArrayV2Stack/TensorListStack:tensor:0&lstm_18/strided_slice_3/stack:output:0(lstm_18/strided_slice_3/stack_1:output:0(lstm_18/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
lstm_18/strided_slice_3
lstm_18/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_18/transpose_1/permÆ
lstm_18/transpose_1	Transpose3lstm_18/TensorArrayV2Stack/TensorListStack:tensor:0!lstm_18/transpose_1/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
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
 *«ªª?2
dropout_18/dropout/Constª
dropout_18/dropout/MulMullstm_18/transpose_1:y:0!dropout_18/dropout/Const:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout_18/dropout/Mul{
dropout_18/dropout/ShapeShapelstm_18/transpose_1:y:0*
T0*
_output_shapes
:2
dropout_18/dropout/ShapeÚ
/dropout_18/dropout/random_uniform/RandomUniformRandomUniform!dropout_18/dropout/Shape:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype021
/dropout_18/dropout/random_uniform/RandomUniform
!dropout_18/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  >2#
!dropout_18/dropout/GreaterEqual/yï
dropout_18/dropout/GreaterEqualGreaterEqual8dropout_18/dropout/random_uniform/RandomUniform:output:0*dropout_18/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
dropout_18/dropout/GreaterEqual¥
dropout_18/dropout/CastCast#dropout_18/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout_18/dropout/Cast«
dropout_18/dropout/Mul_1Muldropout_18/dropout/Mul:z:0dropout_18/dropout/Cast:y:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout_18/dropout/Mul_1j
lstm_19/ShapeShapedropout_18/dropout/Mul_1:z:0*
T0*
_output_shapes
:2
lstm_19/Shape
lstm_19/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_19/strided_slice/stack
lstm_19/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_19/strided_slice/stack_1
lstm_19/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_19/strided_slice/stack_2
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
B :×2
lstm_19/zeros/mul/y
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
B :è2
lstm_19/zeros/Less/y
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
B :×2
lstm_19/zeros/packed/1£
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
lstm_19/zeros/Const
lstm_19/zerosFilllstm_19/zeros/packed:output:0lstm_19/zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2
lstm_19/zerosq
lstm_19/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :×2
lstm_19/zeros_1/mul/y
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
B :è2
lstm_19/zeros_1/Less/y
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
B :×2
lstm_19/zeros_1/packed/1©
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
lstm_19/zeros_1/Const
lstm_19/zeros_1Filllstm_19/zeros_1/packed:output:0lstm_19/zeros_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2
lstm_19/zeros_1
lstm_19/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_19/transpose/perm©
lstm_19/transpose	Transposedropout_18/dropout/Mul_1:z:0lstm_19/transpose/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_19/transposeg
lstm_19/Shape_1Shapelstm_19/transpose:y:0*
T0*
_output_shapes
:2
lstm_19/Shape_1
lstm_19/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_19/strided_slice_1/stack
lstm_19/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_19/strided_slice_1/stack_1
lstm_19/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_19/strided_slice_1/stack_2
lstm_19/strided_slice_1StridedSlicelstm_19/Shape_1:output:0&lstm_19/strided_slice_1/stack:output:0(lstm_19/strided_slice_1/stack_1:output:0(lstm_19/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_19/strided_slice_1
#lstm_19/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2%
#lstm_19/TensorArrayV2/element_shapeÒ
lstm_19/TensorArrayV2TensorListReserve,lstm_19/TensorArrayV2/element_shape:output:0 lstm_19/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_19/TensorArrayV2Ï
=lstm_19/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ  2?
=lstm_19/TensorArrayUnstack/TensorListFromTensor/element_shape
/lstm_19/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_19/transpose:y:0Flstm_19/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type021
/lstm_19/TensorArrayUnstack/TensorListFromTensor
lstm_19/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_19/strided_slice_2/stack
lstm_19/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_19/strided_slice_2/stack_1
lstm_19/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_19/strided_slice_2/stack_2­
lstm_19/strided_slice_2StridedSlicelstm_19/transpose:y:0&lstm_19/strided_slice_2/stack:output:0(lstm_19/strided_slice_2/stack_1:output:0(lstm_19/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
lstm_19/strided_slice_2Î
*lstm_19/lstm_cell_19/MatMul/ReadVariableOpReadVariableOp3lstm_19_lstm_cell_19_matmul_readvariableop_resource* 
_output_shapes
:
Ü*
dtype02,
*lstm_19/lstm_cell_19/MatMul/ReadVariableOpÍ
lstm_19/lstm_cell_19/MatMulMatMul lstm_19/strided_slice_2:output:02lstm_19/lstm_cell_19/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2
lstm_19/lstm_cell_19/MatMulÔ
,lstm_19/lstm_cell_19/MatMul_1/ReadVariableOpReadVariableOp5lstm_19_lstm_cell_19_matmul_1_readvariableop_resource* 
_output_shapes
:
×Ü*
dtype02.
,lstm_19/lstm_cell_19/MatMul_1/ReadVariableOpÉ
lstm_19/lstm_cell_19/MatMul_1MatMullstm_19/zeros:output:04lstm_19/lstm_cell_19/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2
lstm_19/lstm_cell_19/MatMul_1À
lstm_19/lstm_cell_19/addAddV2%lstm_19/lstm_cell_19/MatMul:product:0'lstm_19/lstm_cell_19/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2
lstm_19/lstm_cell_19/addÌ
+lstm_19/lstm_cell_19/BiasAdd/ReadVariableOpReadVariableOp4lstm_19_lstm_cell_19_biasadd_readvariableop_resource*
_output_shapes	
:Ü*
dtype02-
+lstm_19/lstm_cell_19/BiasAdd/ReadVariableOpÍ
lstm_19/lstm_cell_19/BiasAddBiasAddlstm_19/lstm_cell_19/add:z:03lstm_19/lstm_cell_19/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2
lstm_19/lstm_cell_19/BiasAdd
$lstm_19/lstm_cell_19/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2&
$lstm_19/lstm_cell_19/split/split_dim
lstm_19/lstm_cell_19/splitSplit-lstm_19/lstm_cell_19/split/split_dim:output:0%lstm_19/lstm_cell_19/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ×:ÿÿÿÿÿÿÿÿÿ×:ÿÿÿÿÿÿÿÿÿ×:ÿÿÿÿÿÿÿÿÿ×*
	num_split2
lstm_19/lstm_cell_19/split
lstm_19/lstm_cell_19/SigmoidSigmoid#lstm_19/lstm_cell_19/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2
lstm_19/lstm_cell_19/Sigmoid£
lstm_19/lstm_cell_19/Sigmoid_1Sigmoid#lstm_19/lstm_cell_19/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2 
lstm_19/lstm_cell_19/Sigmoid_1¬
lstm_19/lstm_cell_19/mulMul"lstm_19/lstm_cell_19/Sigmoid_1:y:0lstm_19/zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2
lstm_19/lstm_cell_19/mul
lstm_19/lstm_cell_19/ReluRelu#lstm_19/lstm_cell_19/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2
lstm_19/lstm_cell_19/Relu½
lstm_19/lstm_cell_19/mul_1Mul lstm_19/lstm_cell_19/Sigmoid:y:0'lstm_19/lstm_cell_19/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2
lstm_19/lstm_cell_19/mul_1²
lstm_19/lstm_cell_19/add_1AddV2lstm_19/lstm_cell_19/mul:z:0lstm_19/lstm_cell_19/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2
lstm_19/lstm_cell_19/add_1£
lstm_19/lstm_cell_19/Sigmoid_2Sigmoid#lstm_19/lstm_cell_19/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2 
lstm_19/lstm_cell_19/Sigmoid_2
lstm_19/lstm_cell_19/Relu_1Relulstm_19/lstm_cell_19/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2
lstm_19/lstm_cell_19/Relu_1Á
lstm_19/lstm_cell_19/mul_2Mul"lstm_19/lstm_cell_19/Sigmoid_2:y:0)lstm_19/lstm_cell_19/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2
lstm_19/lstm_cell_19/mul_2
%lstm_19/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ×   2'
%lstm_19/TensorArrayV2_1/element_shapeØ
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
lstm_19/time
 lstm_19/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2"
 lstm_19/while/maximum_iterationsz
lstm_19/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_19/while/loop_counter
lstm_19/whileWhile#lstm_19/while/loop_counter:output:0)lstm_19/while/maximum_iterations:output:0lstm_19/time:output:0 lstm_19/TensorArrayV2_1:handle:0lstm_19/zeros:output:0lstm_19/zeros_1:output:0 lstm_19/strided_slice_1:output:0?lstm_19/TensorArrayUnstack/TensorListFromTensor:output_handle:03lstm_19_lstm_cell_19_matmul_readvariableop_resource5lstm_19_lstm_cell_19_matmul_1_readvariableop_resource4lstm_19_lstm_cell_19_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ×:ÿÿÿÿÿÿÿÿÿ×: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *'
bodyR
lstm_19_while_body_32581460*'
condR
lstm_19_while_cond_32581459*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ×:ÿÿÿÿÿÿÿÿÿ×: : : : : *
parallel_iterations 2
lstm_19/whileÅ
8lstm_19/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ×   2:
8lstm_19/TensorArrayV2Stack/TensorListStack/element_shape
*lstm_19/TensorArrayV2Stack/TensorListStackTensorListStacklstm_19/while:output:3Alstm_19/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×*
element_dtype02,
*lstm_19/TensorArrayV2Stack/TensorListStack
lstm_19/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
lstm_19/strided_slice_3/stack
lstm_19/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2!
lstm_19/strided_slice_3/stack_1
lstm_19/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_19/strided_slice_3/stack_2Ë
lstm_19/strided_slice_3StridedSlice3lstm_19/TensorArrayV2Stack/TensorListStack:tensor:0&lstm_19/strided_slice_3/stack:output:0(lstm_19/strided_slice_3/stack_1:output:0(lstm_19/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×*
shrink_axis_mask2
lstm_19/strided_slice_3
lstm_19/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_19/transpose_1/permÆ
lstm_19/transpose_1	Transpose3lstm_19/TensorArrayV2Stack/TensorListStack:tensor:0!lstm_19/transpose_1/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2
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
 *   ?2
dropout_19/dropout/Constª
dropout_19/dropout/MulMullstm_19/transpose_1:y:0!dropout_19/dropout/Const:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2
dropout_19/dropout/Mul{
dropout_19/dropout/ShapeShapelstm_19/transpose_1:y:0*
T0*
_output_shapes
:2
dropout_19/dropout/ShapeÚ
/dropout_19/dropout/random_uniform/RandomUniformRandomUniform!dropout_19/dropout/Shape:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×*
dtype021
/dropout_19/dropout/random_uniform/RandomUniform
!dropout_19/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>2#
!dropout_19/dropout/GreaterEqual/yï
dropout_19/dropout/GreaterEqualGreaterEqual8dropout_19/dropout/random_uniform/RandomUniform:output:0*dropout_19/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2!
dropout_19/dropout/GreaterEqual¥
dropout_19/dropout/CastCast#dropout_19/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2
dropout_19/dropout/Cast«
dropout_19/dropout/Mul_1Muldropout_19/dropout/Mul:z:0dropout_19/dropout/Cast:y:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2
dropout_19/dropout/Mul_1¯
 dense_9/Tensordot/ReadVariableOpReadVariableOp)dense_9_tensordot_readvariableop_resource*
_output_shapes
:	×*
dtype02"
 dense_9/Tensordot/ReadVariableOpz
dense_9/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_9/Tensordot/axes
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
dense_9/Tensordot/Shape
dense_9/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_9/Tensordot/GatherV2/axisù
dense_9/Tensordot/GatherV2GatherV2 dense_9/Tensordot/Shape:output:0dense_9/Tensordot/free:output:0(dense_9/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_9/Tensordot/GatherV2
!dense_9/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_9/Tensordot/GatherV2_1/axisÿ
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
dense_9/Tensordot/Const 
dense_9/Tensordot/ProdProd#dense_9/Tensordot/GatherV2:output:0 dense_9/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_9/Tensordot/Prod
dense_9/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_9/Tensordot/Const_1¨
dense_9/Tensordot/Prod_1Prod%dense_9/Tensordot/GatherV2_1:output:0"dense_9/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_9/Tensordot/Prod_1
dense_9/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense_9/Tensordot/concat/axisØ
dense_9/Tensordot/concatConcatV2dense_9/Tensordot/free:output:0dense_9/Tensordot/axes:output:0&dense_9/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_9/Tensordot/concat¬
dense_9/Tensordot/stackPackdense_9/Tensordot/Prod:output:0!dense_9/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_9/Tensordot/stack¿
dense_9/Tensordot/transpose	Transposedropout_19/dropout/Mul_1:z:0!dense_9/Tensordot/concat:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2
dense_9/Tensordot/transpose¿
dense_9/Tensordot/ReshapeReshapedense_9/Tensordot/transpose:y:0 dense_9/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
dense_9/Tensordot/Reshape¾
dense_9/Tensordot/MatMulMatMul"dense_9/Tensordot/Reshape:output:0(dense_9/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_9/Tensordot/MatMul
dense_9/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
dense_9/Tensordot/Const_2
dense_9/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_9/Tensordot/concat_1/axiså
dense_9/Tensordot/concat_1ConcatV2#dense_9/Tensordot/GatherV2:output:0"dense_9/Tensordot/Const_2:output:0(dense_9/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_9/Tensordot/concat_1°
dense_9/TensordotReshape"dense_9/Tensordot/MatMul:product:0#dense_9/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_9/Tensordot¤
dense_9/BiasAdd/ReadVariableOpReadVariableOp'dense_9_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_9/BiasAdd/ReadVariableOp§
dense_9/BiasAddBiasAdddense_9/Tensordot:output:0&dense_9/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_9/BiasAdd}
dense_9/SoftmaxSoftmaxdense_9/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_9/Softmaxx
IdentityIdentitydense_9/Softmax:softmax:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

IdentityÆ
NoOpNoOp^dense_9/BiasAdd/ReadVariableOp!^dense_9/Tensordot/ReadVariableOp,^lstm_18/lstm_cell_18/BiasAdd/ReadVariableOp+^lstm_18/lstm_cell_18/MatMul/ReadVariableOp-^lstm_18/lstm_cell_18/MatMul_1/ReadVariableOp^lstm_18/while,^lstm_19/lstm_cell_19/BiasAdd/ReadVariableOp+^lstm_19/lstm_cell_19/MatMul/ReadVariableOp-^lstm_19/lstm_cell_19/MatMul_1/ReadVariableOp^lstm_19/while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ]: : : : : : : : 2@
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
:ÿÿÿÿÿÿÿÿÿ]
 
_user_specified_nameinputs
éJ
Ö

lstm_19_while_body_32581126,
(lstm_19_while_lstm_19_while_loop_counter2
.lstm_19_while_lstm_19_while_maximum_iterations
lstm_19_while_placeholder
lstm_19_while_placeholder_1
lstm_19_while_placeholder_2
lstm_19_while_placeholder_3+
'lstm_19_while_lstm_19_strided_slice_1_0g
clstm_19_while_tensorarrayv2read_tensorlistgetitem_lstm_19_tensorarrayunstack_tensorlistfromtensor_0O
;lstm_19_while_lstm_cell_19_matmul_readvariableop_resource_0:
ÜQ
=lstm_19_while_lstm_cell_19_matmul_1_readvariableop_resource_0:
×ÜK
<lstm_19_while_lstm_cell_19_biasadd_readvariableop_resource_0:	Ü
lstm_19_while_identity
lstm_19_while_identity_1
lstm_19_while_identity_2
lstm_19_while_identity_3
lstm_19_while_identity_4
lstm_19_while_identity_5)
%lstm_19_while_lstm_19_strided_slice_1e
alstm_19_while_tensorarrayv2read_tensorlistgetitem_lstm_19_tensorarrayunstack_tensorlistfromtensorM
9lstm_19_while_lstm_cell_19_matmul_readvariableop_resource:
ÜO
;lstm_19_while_lstm_cell_19_matmul_1_readvariableop_resource:
×ÜI
:lstm_19_while_lstm_cell_19_biasadd_readvariableop_resource:	Ü¢1lstm_19/while/lstm_cell_19/BiasAdd/ReadVariableOp¢0lstm_19/while/lstm_cell_19/MatMul/ReadVariableOp¢2lstm_19/while/lstm_cell_19/MatMul_1/ReadVariableOpÓ
?lstm_19/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ  2A
?lstm_19/while/TensorArrayV2Read/TensorListGetItem/element_shape
1lstm_19/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemclstm_19_while_tensorarrayv2read_tensorlistgetitem_lstm_19_tensorarrayunstack_tensorlistfromtensor_0lstm_19_while_placeholderHlstm_19/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype023
1lstm_19/while/TensorArrayV2Read/TensorListGetItemâ
0lstm_19/while/lstm_cell_19/MatMul/ReadVariableOpReadVariableOp;lstm_19_while_lstm_cell_19_matmul_readvariableop_resource_0* 
_output_shapes
:
Ü*
dtype022
0lstm_19/while/lstm_cell_19/MatMul/ReadVariableOp÷
!lstm_19/while/lstm_cell_19/MatMulMatMul8lstm_19/while/TensorArrayV2Read/TensorListGetItem:item:08lstm_19/while/lstm_cell_19/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2#
!lstm_19/while/lstm_cell_19/MatMulè
2lstm_19/while/lstm_cell_19/MatMul_1/ReadVariableOpReadVariableOp=lstm_19_while_lstm_cell_19_matmul_1_readvariableop_resource_0* 
_output_shapes
:
×Ü*
dtype024
2lstm_19/while/lstm_cell_19/MatMul_1/ReadVariableOpà
#lstm_19/while/lstm_cell_19/MatMul_1MatMullstm_19_while_placeholder_2:lstm_19/while/lstm_cell_19/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2%
#lstm_19/while/lstm_cell_19/MatMul_1Ø
lstm_19/while/lstm_cell_19/addAddV2+lstm_19/while/lstm_cell_19/MatMul:product:0-lstm_19/while/lstm_cell_19/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2 
lstm_19/while/lstm_cell_19/addà
1lstm_19/while/lstm_cell_19/BiasAdd/ReadVariableOpReadVariableOp<lstm_19_while_lstm_cell_19_biasadd_readvariableop_resource_0*
_output_shapes	
:Ü*
dtype023
1lstm_19/while/lstm_cell_19/BiasAdd/ReadVariableOpå
"lstm_19/while/lstm_cell_19/BiasAddBiasAdd"lstm_19/while/lstm_cell_19/add:z:09lstm_19/while/lstm_cell_19/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2$
"lstm_19/while/lstm_cell_19/BiasAdd
*lstm_19/while/lstm_cell_19/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2,
*lstm_19/while/lstm_cell_19/split/split_dim¯
 lstm_19/while/lstm_cell_19/splitSplit3lstm_19/while/lstm_cell_19/split/split_dim:output:0+lstm_19/while/lstm_cell_19/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ×:ÿÿÿÿÿÿÿÿÿ×:ÿÿÿÿÿÿÿÿÿ×:ÿÿÿÿÿÿÿÿÿ×*
	num_split2"
 lstm_19/while/lstm_cell_19/split±
"lstm_19/while/lstm_cell_19/SigmoidSigmoid)lstm_19/while/lstm_cell_19/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2$
"lstm_19/while/lstm_cell_19/Sigmoidµ
$lstm_19/while/lstm_cell_19/Sigmoid_1Sigmoid)lstm_19/while/lstm_cell_19/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2&
$lstm_19/while/lstm_cell_19/Sigmoid_1Á
lstm_19/while/lstm_cell_19/mulMul(lstm_19/while/lstm_cell_19/Sigmoid_1:y:0lstm_19_while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2 
lstm_19/while/lstm_cell_19/mul¨
lstm_19/while/lstm_cell_19/ReluRelu)lstm_19/while/lstm_cell_19/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2!
lstm_19/while/lstm_cell_19/ReluÕ
 lstm_19/while/lstm_cell_19/mul_1Mul&lstm_19/while/lstm_cell_19/Sigmoid:y:0-lstm_19/while/lstm_cell_19/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2"
 lstm_19/while/lstm_cell_19/mul_1Ê
 lstm_19/while/lstm_cell_19/add_1AddV2"lstm_19/while/lstm_cell_19/mul:z:0$lstm_19/while/lstm_cell_19/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2"
 lstm_19/while/lstm_cell_19/add_1µ
$lstm_19/while/lstm_cell_19/Sigmoid_2Sigmoid)lstm_19/while/lstm_cell_19/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2&
$lstm_19/while/lstm_cell_19/Sigmoid_2§
!lstm_19/while/lstm_cell_19/Relu_1Relu$lstm_19/while/lstm_cell_19/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2#
!lstm_19/while/lstm_cell_19/Relu_1Ù
 lstm_19/while/lstm_cell_19/mul_2Mul(lstm_19/while/lstm_cell_19/Sigmoid_2:y:0/lstm_19/while/lstm_cell_19/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2"
 lstm_19/while/lstm_cell_19/mul_2
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
lstm_19/while/add/y
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
lstm_19/while/add_1/y
lstm_19/while/add_1AddV2(lstm_19_while_lstm_19_while_loop_counterlstm_19/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lstm_19/while/add_1
lstm_19/while/IdentityIdentitylstm_19/while/add_1:z:0^lstm_19/while/NoOp*
T0*
_output_shapes
: 2
lstm_19/while/Identity¦
lstm_19/while/Identity_1Identity.lstm_19_while_lstm_19_while_maximum_iterations^lstm_19/while/NoOp*
T0*
_output_shapes
: 2
lstm_19/while/Identity_1
lstm_19/while/Identity_2Identitylstm_19/while/add:z:0^lstm_19/while/NoOp*
T0*
_output_shapes
: 2
lstm_19/while/Identity_2º
lstm_19/while/Identity_3IdentityBlstm_19/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_19/while/NoOp*
T0*
_output_shapes
: 2
lstm_19/while/Identity_3®
lstm_19/while/Identity_4Identity$lstm_19/while/lstm_cell_19/mul_2:z:0^lstm_19/while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2
lstm_19/while/Identity_4®
lstm_19/while/Identity_5Identity$lstm_19/while/lstm_cell_19/add_1:z:0^lstm_19/while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2
lstm_19/while/Identity_5
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
9lstm_19_while_lstm_cell_19_matmul_readvariableop_resource;lstm_19_while_lstm_cell_19_matmul_readvariableop_resource_0"È
alstm_19_while_tensorarrayv2read_tensorlistgetitem_lstm_19_tensorarrayunstack_tensorlistfromtensorclstm_19_while_tensorarrayv2read_tensorlistgetitem_lstm_19_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ×:ÿÿÿÿÿÿÿÿÿ×: : : : : 2f
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
:ÿÿÿÿÿÿÿÿÿ×:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×:

_output_shapes
: :

_output_shapes
: 
Ô!
ý
E__inference_dense_9_layer_call_and_return_conditional_losses_32580322

inputs4
!tensordot_readvariableop_resource:	×-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢Tensordot/ReadVariableOp
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes
:	×*
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
:ÿÿÿÿÿÿÿÿÿ×2
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
:ÿÿÿÿÿÿÿÿÿ×: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×
 
_user_specified_nameinputs


J__inference_lstm_cell_18_layer_call_and_return_conditional_losses_32578915

inputs

states
states_11
matmul_readvariableop_resource:	]Ô4
 matmul_1_readvariableop_resource:
Ô.
biasadd_readvariableop_resource:	Ô
identity

identity_1

identity_2¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	]Ô*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÔ2
MatMul
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
Ô*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÔ2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÔ2
add
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:Ô*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÔ2	
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
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2
split`
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
Sigmoidd
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	Sigmoid_1]
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mulW
ReluRelusplit:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relui
mul_1MulSigmoid:y:0Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mul_1^
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
add_1d
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	Sigmoid_2V
Relu_1Relu	add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu_1m
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mul_2e
IdentityIdentity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityi

Identity_1Identity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1i

Identity_2Identity	add_1:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

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
A:ÿÿÿÿÿÿÿÿÿ]:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
 
_user_specified_nameinputs:PL
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_namestates:PL
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_namestates
ÐF

E__inference_lstm_19_layer_call_and_return_conditional_losses_32579482

inputs)
lstm_cell_19_32579400:
Ü)
lstm_cell_19_32579402:
×Ü$
lstm_cell_19_32579404:	Ü
identity¢$lstm_cell_19/StatefulPartitionedCall¢whileD
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
B :×2
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
B :×2
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
:ÿÿÿÿÿÿÿÿÿ×2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :×2
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
B :×2
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
:ÿÿÿÿÿÿÿÿÿ×2	
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
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
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
valueB"ÿÿÿÿ  27
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
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
strided_slice_2¨
$lstm_cell_19/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_19_32579400lstm_cell_19_32579402lstm_cell_19_32579404*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ×:ÿÿÿÿÿÿÿÿÿ×:ÿÿÿÿÿÿÿÿÿ×*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_lstm_cell_19_layer_call_and_return_conditional_losses_325793992&
$lstm_cell_19/StatefulPartitionedCall
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ×   2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_19_32579400lstm_cell_19_32579402lstm_cell_19_32579404*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ×:ÿÿÿÿÿÿÿÿÿ×: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_32579413*
condR
while_cond_32579412*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ×:ÿÿÿÿÿÿÿÿÿ×: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ×   22
0TensorArrayV2Stack/TensorListStack/element_shapeò
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ×*
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
:ÿÿÿÿÿÿÿÿÿ×*
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
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ×2
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
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ×2

Identity}
NoOpNoOp%^lstm_cell_19/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 2L
$lstm_cell_19/StatefulPartitionedCall$lstm_cell_19/StatefulPartitionedCall2
whilewhile:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ã
Í
while_cond_32581838
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_32581838___redundant_placeholder06
2while_while_cond_32581838___redundant_placeholder16
2while_while_cond_32581838___redundant_placeholder26
2while_while_cond_32581838___redundant_placeholder3
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
B: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:


*__inference_dense_9_layer_call_fn_32583011

inputs
unknown:	×
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
E__inference_dense_9_layer_call_and_return_conditional_losses_325803222
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
:ÿÿÿÿÿÿÿÿÿ×: : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×
 
_user_specified_nameinputs
´?
Ö
while_body_32580461
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
3while_lstm_cell_19_matmul_readvariableop_resource_0:
ÜI
5while_lstm_cell_19_matmul_1_readvariableop_resource_0:
×ÜC
4while_lstm_cell_19_biasadd_readvariableop_resource_0:	Ü
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
1while_lstm_cell_19_matmul_readvariableop_resource:
ÜG
3while_lstm_cell_19_matmul_1_readvariableop_resource:
×ÜA
2while_lstm_cell_19_biasadd_readvariableop_resource:	Ü¢)while/lstm_cell_19/BiasAdd/ReadVariableOp¢(while/lstm_cell_19/MatMul/ReadVariableOp¢*while/lstm_cell_19/MatMul_1/ReadVariableOpÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ  29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÔ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemÊ
(while/lstm_cell_19/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_19_matmul_readvariableop_resource_0* 
_output_shapes
:
Ü*
dtype02*
(while/lstm_cell_19/MatMul/ReadVariableOp×
while/lstm_cell_19/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_19/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2
while/lstm_cell_19/MatMulÐ
*while/lstm_cell_19/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_19_matmul_1_readvariableop_resource_0* 
_output_shapes
:
×Ü*
dtype02,
*while/lstm_cell_19/MatMul_1/ReadVariableOpÀ
while/lstm_cell_19/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_19/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2
while/lstm_cell_19/MatMul_1¸
while/lstm_cell_19/addAddV2#while/lstm_cell_19/MatMul:product:0%while/lstm_cell_19/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2
while/lstm_cell_19/addÈ
)while/lstm_cell_19/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_19_biasadd_readvariableop_resource_0*
_output_shapes	
:Ü*
dtype02+
)while/lstm_cell_19/BiasAdd/ReadVariableOpÅ
while/lstm_cell_19/BiasAddBiasAddwhile/lstm_cell_19/add:z:01while/lstm_cell_19/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2
while/lstm_cell_19/BiasAdd
"while/lstm_cell_19/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_19/split/split_dim
while/lstm_cell_19/splitSplit+while/lstm_cell_19/split/split_dim:output:0#while/lstm_cell_19/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ×:ÿÿÿÿÿÿÿÿÿ×:ÿÿÿÿÿÿÿÿÿ×:ÿÿÿÿÿÿÿÿÿ×*
	num_split2
while/lstm_cell_19/split
while/lstm_cell_19/SigmoidSigmoid!while/lstm_cell_19/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2
while/lstm_cell_19/Sigmoid
while/lstm_cell_19/Sigmoid_1Sigmoid!while/lstm_cell_19/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2
while/lstm_cell_19/Sigmoid_1¡
while/lstm_cell_19/mulMul while/lstm_cell_19/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2
while/lstm_cell_19/mul
while/lstm_cell_19/ReluRelu!while/lstm_cell_19/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2
while/lstm_cell_19/Reluµ
while/lstm_cell_19/mul_1Mulwhile/lstm_cell_19/Sigmoid:y:0%while/lstm_cell_19/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2
while/lstm_cell_19/mul_1ª
while/lstm_cell_19/add_1AddV2while/lstm_cell_19/mul:z:0while/lstm_cell_19/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2
while/lstm_cell_19/add_1
while/lstm_cell_19/Sigmoid_2Sigmoid!while/lstm_cell_19/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2
while/lstm_cell_19/Sigmoid_2
while/lstm_cell_19/Relu_1Reluwhile/lstm_cell_19/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2
while/lstm_cell_19/Relu_1¹
while/lstm_cell_19/mul_2Mul while/lstm_cell_19/Sigmoid_2:y:0'while/lstm_cell_19/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2
while/lstm_cell_19/mul_2à
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
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_19/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_19/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2
while/Identity_5Þ

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
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ×:ÿÿÿÿÿÿÿÿÿ×: : : : : 2V
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
:ÿÿÿÿÿÿÿÿÿ×:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×:

_output_shapes
: :

_output_shapes
: 
Û
ñ
(sequential_9_lstm_19_while_cond_32578581F
Bsequential_9_lstm_19_while_sequential_9_lstm_19_while_loop_counterL
Hsequential_9_lstm_19_while_sequential_9_lstm_19_while_maximum_iterations*
&sequential_9_lstm_19_while_placeholder,
(sequential_9_lstm_19_while_placeholder_1,
(sequential_9_lstm_19_while_placeholder_2,
(sequential_9_lstm_19_while_placeholder_3H
Dsequential_9_lstm_19_while_less_sequential_9_lstm_19_strided_slice_1`
\sequential_9_lstm_19_while_sequential_9_lstm_19_while_cond_32578581___redundant_placeholder0`
\sequential_9_lstm_19_while_sequential_9_lstm_19_while_cond_32578581___redundant_placeholder1`
\sequential_9_lstm_19_while_sequential_9_lstm_19_while_cond_32578581___redundant_placeholder2`
\sequential_9_lstm_19_while_sequential_9_lstm_19_while_cond_32578581___redundant_placeholder3'
#sequential_9_lstm_19_while_identity
Ù
sequential_9/lstm_19/while/LessLess&sequential_9_lstm_19_while_placeholderDsequential_9_lstm_19_while_less_sequential_9_lstm_19_strided_slice_1*
T0*
_output_shapes
: 2!
sequential_9/lstm_19/while/Less
#sequential_9/lstm_19/while/IdentityIdentity#sequential_9/lstm_19/while/Less:z:0*
T0
*
_output_shapes
: 2%
#sequential_9/lstm_19/while/Identity"S
#sequential_9_lstm_19_while_identity,sequential_9/lstm_19/while/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :ÿÿÿÿÿÿÿÿÿ×:ÿÿÿÿÿÿÿÿÿ×: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×:

_output_shapes
: :

_output_shapes
:
Ô

í
lstm_18_while_cond_32580977,
(lstm_18_while_lstm_18_while_loop_counter2
.lstm_18_while_lstm_18_while_maximum_iterations
lstm_18_while_placeholder
lstm_18_while_placeholder_1
lstm_18_while_placeholder_2
lstm_18_while_placeholder_3.
*lstm_18_while_less_lstm_18_strided_slice_1F
Blstm_18_while_lstm_18_while_cond_32580977___redundant_placeholder0F
Blstm_18_while_lstm_18_while_cond_32580977___redundant_placeholder1F
Blstm_18_while_lstm_18_while_cond_32580977___redundant_placeholder2F
Blstm_18_while_lstm_18_while_cond_32580977___redundant_placeholder3
lstm_18_while_identity

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
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
°?
Ô
while_body_32581688
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_18_matmul_readvariableop_resource_0:	]ÔI
5while_lstm_cell_18_matmul_1_readvariableop_resource_0:
ÔC
4while_lstm_cell_18_biasadd_readvariableop_resource_0:	Ô
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_18_matmul_readvariableop_resource:	]ÔG
3while_lstm_cell_18_matmul_1_readvariableop_resource:
ÔA
2while_lstm_cell_18_biasadd_readvariableop_resource:	Ô¢)while/lstm_cell_18/BiasAdd/ReadVariableOp¢(while/lstm_cell_18/MatMul/ReadVariableOp¢*while/lstm_cell_18/MatMul_1/ReadVariableOpÃ
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
(while/lstm_cell_18/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_18_matmul_readvariableop_resource_0*
_output_shapes
:	]Ô*
dtype02*
(while/lstm_cell_18/MatMul/ReadVariableOp×
while/lstm_cell_18/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_18/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÔ2
while/lstm_cell_18/MatMulÐ
*while/lstm_cell_18/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_18_matmul_1_readvariableop_resource_0* 
_output_shapes
:
Ô*
dtype02,
*while/lstm_cell_18/MatMul_1/ReadVariableOpÀ
while/lstm_cell_18/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_18/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÔ2
while/lstm_cell_18/MatMul_1¸
while/lstm_cell_18/addAddV2#while/lstm_cell_18/MatMul:product:0%while/lstm_cell_18/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÔ2
while/lstm_cell_18/addÈ
)while/lstm_cell_18/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_18_biasadd_readvariableop_resource_0*
_output_shapes	
:Ô*
dtype02+
)while/lstm_cell_18/BiasAdd/ReadVariableOpÅ
while/lstm_cell_18/BiasAddBiasAddwhile/lstm_cell_18/add:z:01while/lstm_cell_18/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÔ2
while/lstm_cell_18/BiasAdd
"while/lstm_cell_18/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_18/split/split_dim
while/lstm_cell_18/splitSplit+while/lstm_cell_18/split/split_dim:output:0#while/lstm_cell_18/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2
while/lstm_cell_18/split
while/lstm_cell_18/SigmoidSigmoid!while/lstm_cell_18/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_18/Sigmoid
while/lstm_cell_18/Sigmoid_1Sigmoid!while/lstm_cell_18/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_18/Sigmoid_1¡
while/lstm_cell_18/mulMul while/lstm_cell_18/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_18/mul
while/lstm_cell_18/ReluRelu!while/lstm_cell_18/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_18/Reluµ
while/lstm_cell_18/mul_1Mulwhile/lstm_cell_18/Sigmoid:y:0%while/lstm_cell_18/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_18/mul_1ª
while/lstm_cell_18/add_1AddV2while/lstm_cell_18/mul:z:0while/lstm_cell_18/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_18/add_1
while/lstm_cell_18/Sigmoid_2Sigmoid!while/lstm_cell_18/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_18/Sigmoid_2
while/lstm_cell_18/Relu_1Reluwhile/lstm_cell_18/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_18/Relu_1¹
while/lstm_cell_18/mul_2Mul while/lstm_cell_18/Sigmoid_2:y:0'while/lstm_cell_18/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_18/mul_2à
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
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_18/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_18/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/Identity_5Þ

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
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2V
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
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
Ø
I
-__inference_dropout_18_layer_call_fn_32582291

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
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_dropout_18_layer_call_and_return_conditional_losses_325801242
PartitionedCallq
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


J__inference_lstm_cell_19_layer_call_and_return_conditional_losses_32583141

inputs
states_0
states_12
matmul_readvariableop_resource:
Ü4
 matmul_1_readvariableop_resource:
×Ü.
biasadd_readvariableop_resource:	Ü
identity

identity_1

identity_2¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
Ü*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2
MatMul
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
×Ü*
dtype02
MatMul_1/ReadVariableOp|
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2
add
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:Ü*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2	
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
P:ÿÿÿÿÿÿÿÿÿ×:ÿÿÿÿÿÿÿÿÿ×:ÿÿÿÿÿÿÿÿÿ×:ÿÿÿÿÿÿÿÿÿ×*
	num_split2
split`
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2	
Sigmoidd
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2
	Sigmoid_1]
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2
mulW
ReluRelusplit:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2
Relui
mul_1MulSigmoid:y:0Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2
mul_1^
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2
add_1d
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2
	Sigmoid_2V
Relu_1Relu	add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2
Relu_1m
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2
mul_2e
IdentityIdentity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2

Identityi

Identity_1Identity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2

Identity_1i

Identity_2Identity	add_1:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2

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
B:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ×:ÿÿÿÿÿÿÿÿÿ×: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×
"
_user_specified_name
states/0:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×
"
_user_specified_name
states/1
ã
Í
while_cond_32578992
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_32578992___redundant_placeholder06
2while_while_cond_32578992___redundant_placeholder16
2while_while_cond_32578992___redundant_placeholder26
2while_while_cond_32578992___redundant_placeholder3
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
B: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
´?
Ö
while_body_32582665
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
3while_lstm_cell_19_matmul_readvariableop_resource_0:
ÜI
5while_lstm_cell_19_matmul_1_readvariableop_resource_0:
×ÜC
4while_lstm_cell_19_biasadd_readvariableop_resource_0:	Ü
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
1while_lstm_cell_19_matmul_readvariableop_resource:
ÜG
3while_lstm_cell_19_matmul_1_readvariableop_resource:
×ÜA
2while_lstm_cell_19_biasadd_readvariableop_resource:	Ü¢)while/lstm_cell_19/BiasAdd/ReadVariableOp¢(while/lstm_cell_19/MatMul/ReadVariableOp¢*while/lstm_cell_19/MatMul_1/ReadVariableOpÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ  29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÔ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemÊ
(while/lstm_cell_19/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_19_matmul_readvariableop_resource_0* 
_output_shapes
:
Ü*
dtype02*
(while/lstm_cell_19/MatMul/ReadVariableOp×
while/lstm_cell_19/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_19/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2
while/lstm_cell_19/MatMulÐ
*while/lstm_cell_19/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_19_matmul_1_readvariableop_resource_0* 
_output_shapes
:
×Ü*
dtype02,
*while/lstm_cell_19/MatMul_1/ReadVariableOpÀ
while/lstm_cell_19/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_19/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2
while/lstm_cell_19/MatMul_1¸
while/lstm_cell_19/addAddV2#while/lstm_cell_19/MatMul:product:0%while/lstm_cell_19/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2
while/lstm_cell_19/addÈ
)while/lstm_cell_19/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_19_biasadd_readvariableop_resource_0*
_output_shapes	
:Ü*
dtype02+
)while/lstm_cell_19/BiasAdd/ReadVariableOpÅ
while/lstm_cell_19/BiasAddBiasAddwhile/lstm_cell_19/add:z:01while/lstm_cell_19/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2
while/lstm_cell_19/BiasAdd
"while/lstm_cell_19/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_19/split/split_dim
while/lstm_cell_19/splitSplit+while/lstm_cell_19/split/split_dim:output:0#while/lstm_cell_19/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ×:ÿÿÿÿÿÿÿÿÿ×:ÿÿÿÿÿÿÿÿÿ×:ÿÿÿÿÿÿÿÿÿ×*
	num_split2
while/lstm_cell_19/split
while/lstm_cell_19/SigmoidSigmoid!while/lstm_cell_19/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2
while/lstm_cell_19/Sigmoid
while/lstm_cell_19/Sigmoid_1Sigmoid!while/lstm_cell_19/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2
while/lstm_cell_19/Sigmoid_1¡
while/lstm_cell_19/mulMul while/lstm_cell_19/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2
while/lstm_cell_19/mul
while/lstm_cell_19/ReluRelu!while/lstm_cell_19/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2
while/lstm_cell_19/Reluµ
while/lstm_cell_19/mul_1Mulwhile/lstm_cell_19/Sigmoid:y:0%while/lstm_cell_19/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2
while/lstm_cell_19/mul_1ª
while/lstm_cell_19/add_1AddV2while/lstm_cell_19/mul:z:0while/lstm_cell_19/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2
while/lstm_cell_19/add_1
while/lstm_cell_19/Sigmoid_2Sigmoid!while/lstm_cell_19/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2
while/lstm_cell_19/Sigmoid_2
while/lstm_cell_19/Relu_1Reluwhile/lstm_cell_19/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2
while/lstm_cell_19/Relu_1¹
while/lstm_cell_19/mul_2Mul while/lstm_cell_19/Sigmoid_2:y:0'while/lstm_cell_19/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2
while/lstm_cell_19/mul_2à
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
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_19/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_19/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2
while/Identity_5Þ

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
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ×:ÿÿÿÿÿÿÿÿÿ×: : : : : 2V
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
:ÿÿÿÿÿÿÿÿÿ×:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×:

_output_shapes
: :

_output_shapes
: 
¹
¹
*__inference_lstm_19_layer_call_fn_32582944

inputs
unknown:
Ü
	unknown_0:
×Ü
	unknown_1:	Ü
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_lstm_19_layer_call_and_return_conditional_losses_325805452
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:ÿÿÿÿÿÿÿÿÿ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ô

í
lstm_19_while_cond_32581125,
(lstm_19_while_lstm_19_while_loop_counter2
.lstm_19_while_lstm_19_while_maximum_iterations
lstm_19_while_placeholder
lstm_19_while_placeholder_1
lstm_19_while_placeholder_2
lstm_19_while_placeholder_3.
*lstm_19_while_less_lstm_19_strided_slice_1F
Blstm_19_while_lstm_19_while_cond_32581125___redundant_placeholder0F
Blstm_19_while_lstm_19_while_cond_32581125___redundant_placeholder1F
Blstm_19_while_lstm_19_while_cond_32581125___redundant_placeholder2F
Blstm_19_while_lstm_19_while_cond_32581125___redundant_placeholder3
lstm_19_while_identity

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
B: : : : :ÿÿÿÿÿÿÿÿÿ×:ÿÿÿÿÿÿÿÿÿ×: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×:

_output_shapes
: :

_output_shapes
:
ü	
Ê
&__inference_signature_wrapper_32580911
lstm_18_input
unknown:	]Ô
	unknown_0:
Ô
	unknown_1:	Ô
	unknown_2:
Ü
	unknown_3:
×Ü
	unknown_4:	Ü
	unknown_5:	×
	unknown_6:
identity¢StatefulPartitionedCall¬
StatefulPartitionedCallStatefulPartitionedCalllstm_18_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
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
#__inference__wrapped_model_325786942
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
_user_specified_namelstm_18_input
&
ó
while_body_32578993
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_00
while_lstm_cell_18_32579017_0:	]Ô1
while_lstm_cell_18_32579019_0:
Ô,
while_lstm_cell_18_32579021_0:	Ô
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor.
while_lstm_cell_18_32579017:	]Ô/
while_lstm_cell_18_32579019:
Ô*
while_lstm_cell_18_32579021:	Ô¢*while/lstm_cell_18/StatefulPartitionedCallÃ
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
*while/lstm_cell_18/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_18_32579017_0while_lstm_cell_18_32579019_0while_lstm_cell_18_32579021_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_lstm_cell_18_layer_call_and_return_conditional_losses_325789152,
*while/lstm_cell_18/StatefulPartitionedCall÷
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
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3¥
while/Identity_4Identity3while/lstm_cell_18/StatefulPartitionedCall:output:1^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/Identity_4¥
while/Identity_5Identity3while/lstm_cell_18/StatefulPartitionedCall:output:2^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/Identity_5

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
while_lstm_cell_18_32579017while_lstm_cell_18_32579017_0"<
while_lstm_cell_18_32579019while_lstm_cell_18_32579019_0"<
while_lstm_cell_18_32579021while_lstm_cell_18_32579021_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2X
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
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
ÐF

E__inference_lstm_19_layer_call_and_return_conditional_losses_32579692

inputs)
lstm_cell_19_32579610:
Ü)
lstm_cell_19_32579612:
×Ü$
lstm_cell_19_32579614:	Ü
identity¢$lstm_cell_19/StatefulPartitionedCall¢whileD
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
B :×2
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
B :×2
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
:ÿÿÿÿÿÿÿÿÿ×2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :×2
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
B :×2
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
:ÿÿÿÿÿÿÿÿÿ×2	
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
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
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
valueB"ÿÿÿÿ  27
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
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
strided_slice_2¨
$lstm_cell_19/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_19_32579610lstm_cell_19_32579612lstm_cell_19_32579614*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ×:ÿÿÿÿÿÿÿÿÿ×:ÿÿÿÿÿÿÿÿÿ×*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_lstm_cell_19_layer_call_and_return_conditional_losses_325795452&
$lstm_cell_19/StatefulPartitionedCall
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ×   2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_19_32579610lstm_cell_19_32579612lstm_cell_19_32579614*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ×:ÿÿÿÿÿÿÿÿÿ×: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_32579623*
condR
while_cond_32579622*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ×:ÿÿÿÿÿÿÿÿÿ×: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ×   22
0TensorArrayV2Stack/TensorListStack/element_shapeò
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ×*
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
:ÿÿÿÿÿÿÿÿÿ×*
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
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ×2
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
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ×2

Identity}
NoOpNoOp%^lstm_cell_19/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 2L
$lstm_cell_19/StatefulPartitionedCall$lstm_cell_19/StatefulPartitionedCall2
whilewhile:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
åJ
Ô

lstm_18_while_body_32580978,
(lstm_18_while_lstm_18_while_loop_counter2
.lstm_18_while_lstm_18_while_maximum_iterations
lstm_18_while_placeholder
lstm_18_while_placeholder_1
lstm_18_while_placeholder_2
lstm_18_while_placeholder_3+
'lstm_18_while_lstm_18_strided_slice_1_0g
clstm_18_while_tensorarrayv2read_tensorlistgetitem_lstm_18_tensorarrayunstack_tensorlistfromtensor_0N
;lstm_18_while_lstm_cell_18_matmul_readvariableop_resource_0:	]ÔQ
=lstm_18_while_lstm_cell_18_matmul_1_readvariableop_resource_0:
ÔK
<lstm_18_while_lstm_cell_18_biasadd_readvariableop_resource_0:	Ô
lstm_18_while_identity
lstm_18_while_identity_1
lstm_18_while_identity_2
lstm_18_while_identity_3
lstm_18_while_identity_4
lstm_18_while_identity_5)
%lstm_18_while_lstm_18_strided_slice_1e
alstm_18_while_tensorarrayv2read_tensorlistgetitem_lstm_18_tensorarrayunstack_tensorlistfromtensorL
9lstm_18_while_lstm_cell_18_matmul_readvariableop_resource:	]ÔO
;lstm_18_while_lstm_cell_18_matmul_1_readvariableop_resource:
ÔI
:lstm_18_while_lstm_cell_18_biasadd_readvariableop_resource:	Ô¢1lstm_18/while/lstm_cell_18/BiasAdd/ReadVariableOp¢0lstm_18/while/lstm_cell_18/MatMul/ReadVariableOp¢2lstm_18/while/lstm_cell_18/MatMul_1/ReadVariableOpÓ
?lstm_18/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ]   2A
?lstm_18/while/TensorArrayV2Read/TensorListGetItem/element_shape
1lstm_18/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemclstm_18_while_tensorarrayv2read_tensorlistgetitem_lstm_18_tensorarrayunstack_tensorlistfromtensor_0lstm_18_while_placeholderHlstm_18/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]*
element_dtype023
1lstm_18/while/TensorArrayV2Read/TensorListGetItemá
0lstm_18/while/lstm_cell_18/MatMul/ReadVariableOpReadVariableOp;lstm_18_while_lstm_cell_18_matmul_readvariableop_resource_0*
_output_shapes
:	]Ô*
dtype022
0lstm_18/while/lstm_cell_18/MatMul/ReadVariableOp÷
!lstm_18/while/lstm_cell_18/MatMulMatMul8lstm_18/while/TensorArrayV2Read/TensorListGetItem:item:08lstm_18/while/lstm_cell_18/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÔ2#
!lstm_18/while/lstm_cell_18/MatMulè
2lstm_18/while/lstm_cell_18/MatMul_1/ReadVariableOpReadVariableOp=lstm_18_while_lstm_cell_18_matmul_1_readvariableop_resource_0* 
_output_shapes
:
Ô*
dtype024
2lstm_18/while/lstm_cell_18/MatMul_1/ReadVariableOpà
#lstm_18/while/lstm_cell_18/MatMul_1MatMullstm_18_while_placeholder_2:lstm_18/while/lstm_cell_18/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÔ2%
#lstm_18/while/lstm_cell_18/MatMul_1Ø
lstm_18/while/lstm_cell_18/addAddV2+lstm_18/while/lstm_cell_18/MatMul:product:0-lstm_18/while/lstm_cell_18/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÔ2 
lstm_18/while/lstm_cell_18/addà
1lstm_18/while/lstm_cell_18/BiasAdd/ReadVariableOpReadVariableOp<lstm_18_while_lstm_cell_18_biasadd_readvariableop_resource_0*
_output_shapes	
:Ô*
dtype023
1lstm_18/while/lstm_cell_18/BiasAdd/ReadVariableOpå
"lstm_18/while/lstm_cell_18/BiasAddBiasAdd"lstm_18/while/lstm_cell_18/add:z:09lstm_18/while/lstm_cell_18/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÔ2$
"lstm_18/while/lstm_cell_18/BiasAdd
*lstm_18/while/lstm_cell_18/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2,
*lstm_18/while/lstm_cell_18/split/split_dim¯
 lstm_18/while/lstm_cell_18/splitSplit3lstm_18/while/lstm_cell_18/split/split_dim:output:0+lstm_18/while/lstm_cell_18/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2"
 lstm_18/while/lstm_cell_18/split±
"lstm_18/while/lstm_cell_18/SigmoidSigmoid)lstm_18/while/lstm_cell_18/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2$
"lstm_18/while/lstm_cell_18/Sigmoidµ
$lstm_18/while/lstm_cell_18/Sigmoid_1Sigmoid)lstm_18/while/lstm_cell_18/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$lstm_18/while/lstm_cell_18/Sigmoid_1Á
lstm_18/while/lstm_cell_18/mulMul(lstm_18/while/lstm_cell_18/Sigmoid_1:y:0lstm_18_while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
lstm_18/while/lstm_cell_18/mul¨
lstm_18/while/lstm_cell_18/ReluRelu)lstm_18/while/lstm_cell_18/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
lstm_18/while/lstm_cell_18/ReluÕ
 lstm_18/while/lstm_cell_18/mul_1Mul&lstm_18/while/lstm_cell_18/Sigmoid:y:0-lstm_18/while/lstm_cell_18/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 lstm_18/while/lstm_cell_18/mul_1Ê
 lstm_18/while/lstm_cell_18/add_1AddV2"lstm_18/while/lstm_cell_18/mul:z:0$lstm_18/while/lstm_cell_18/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 lstm_18/while/lstm_cell_18/add_1µ
$lstm_18/while/lstm_cell_18/Sigmoid_2Sigmoid)lstm_18/while/lstm_cell_18/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$lstm_18/while/lstm_cell_18/Sigmoid_2§
!lstm_18/while/lstm_cell_18/Relu_1Relu$lstm_18/while/lstm_cell_18/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!lstm_18/while/lstm_cell_18/Relu_1Ù
 lstm_18/while/lstm_cell_18/mul_2Mul(lstm_18/while/lstm_cell_18/Sigmoid_2:y:0/lstm_18/while/lstm_cell_18/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 lstm_18/while/lstm_cell_18/mul_2
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
lstm_18/while/add/y
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
lstm_18/while/add_1/y
lstm_18/while/add_1AddV2(lstm_18_while_lstm_18_while_loop_counterlstm_18/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lstm_18/while/add_1
lstm_18/while/IdentityIdentitylstm_18/while/add_1:z:0^lstm_18/while/NoOp*
T0*
_output_shapes
: 2
lstm_18/while/Identity¦
lstm_18/while/Identity_1Identity.lstm_18_while_lstm_18_while_maximum_iterations^lstm_18/while/NoOp*
T0*
_output_shapes
: 2
lstm_18/while/Identity_1
lstm_18/while/Identity_2Identitylstm_18/while/add:z:0^lstm_18/while/NoOp*
T0*
_output_shapes
: 2
lstm_18/while/Identity_2º
lstm_18/while/Identity_3IdentityBlstm_18/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_18/while/NoOp*
T0*
_output_shapes
: 2
lstm_18/while/Identity_3®
lstm_18/while/Identity_4Identity$lstm_18/while/lstm_cell_18/mul_2:z:0^lstm_18/while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_18/while/Identity_4®
lstm_18/while/Identity_5Identity$lstm_18/while/lstm_cell_18/add_1:z:0^lstm_18/while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_18/while/Identity_5
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
9lstm_18_while_lstm_cell_18_matmul_readvariableop_resource;lstm_18_while_lstm_cell_18_matmul_readvariableop_resource_0"È
alstm_18_while_tensorarrayv2read_tensorlistgetitem_lstm_18_tensorarrayunstack_tensorlistfromtensorclstm_18_while_tensorarrayv2read_tensorlistgetitem_lstm_18_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2f
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
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
à
º
*__inference_lstm_18_layer_call_fn_32582247
inputs_0
unknown:	]Ô
	unknown_0:
Ô
	unknown_1:	Ô
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_lstm_18_layer_call_and_return_conditional_losses_325790622
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

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
»
f
-__inference_dropout_19_layer_call_fn_32582971

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
:ÿÿÿÿÿÿÿÿÿ×* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_dropout_19_layer_call_and_return_conditional_losses_325803782
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ×22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×
 
_user_specified_nameinputs
ã
Í
while_cond_32580656
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_32580656___redundant_placeholder06
2while_while_cond_32580656___redundant_placeholder16
2while_while_cond_32580656___redundant_placeholder26
2while_while_cond_32580656___redundant_placeholder3
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
B: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
\

E__inference_lstm_18_layer_call_and_return_conditional_losses_32580111

inputs>
+lstm_cell_18_matmul_readvariableop_resource:	]ÔA
-lstm_cell_18_matmul_1_readvariableop_resource:
Ô;
,lstm_cell_18_biasadd_readvariableop_resource:	Ô
identity¢#lstm_cell_18/BiasAdd/ReadVariableOp¢"lstm_cell_18/MatMul/ReadVariableOp¢$lstm_cell_18/MatMul_1/ReadVariableOp¢whileD
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
B :2
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
B :2
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
:ÿÿÿÿÿÿÿÿÿ2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :2
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
B :2
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
:ÿÿÿÿÿÿÿÿÿ2	
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
"lstm_cell_18/MatMul/ReadVariableOpReadVariableOp+lstm_cell_18_matmul_readvariableop_resource*
_output_shapes
:	]Ô*
dtype02$
"lstm_cell_18/MatMul/ReadVariableOp­
lstm_cell_18/MatMulMatMulstrided_slice_2:output:0*lstm_cell_18/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÔ2
lstm_cell_18/MatMul¼
$lstm_cell_18/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_18_matmul_1_readvariableop_resource* 
_output_shapes
:
Ô*
dtype02&
$lstm_cell_18/MatMul_1/ReadVariableOp©
lstm_cell_18/MatMul_1MatMulzeros:output:0,lstm_cell_18/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÔ2
lstm_cell_18/MatMul_1 
lstm_cell_18/addAddV2lstm_cell_18/MatMul:product:0lstm_cell_18/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÔ2
lstm_cell_18/add´
#lstm_cell_18/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_18_biasadd_readvariableop_resource*
_output_shapes	
:Ô*
dtype02%
#lstm_cell_18/BiasAdd/ReadVariableOp­
lstm_cell_18/BiasAddBiasAddlstm_cell_18/add:z:0+lstm_cell_18/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÔ2
lstm_cell_18/BiasAdd~
lstm_cell_18/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_18/split/split_dim÷
lstm_cell_18/splitSplit%lstm_cell_18/split/split_dim:output:0lstm_cell_18/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2
lstm_cell_18/split
lstm_cell_18/SigmoidSigmoidlstm_cell_18/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_18/Sigmoid
lstm_cell_18/Sigmoid_1Sigmoidlstm_cell_18/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_18/Sigmoid_1
lstm_cell_18/mulMullstm_cell_18/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_18/mul~
lstm_cell_18/ReluRelulstm_cell_18/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_18/Relu
lstm_cell_18/mul_1Mullstm_cell_18/Sigmoid:y:0lstm_cell_18/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_18/mul_1
lstm_cell_18/add_1AddV2lstm_cell_18/mul:z:0lstm_cell_18/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_18/add_1
lstm_cell_18/Sigmoid_2Sigmoidlstm_cell_18/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_18/Sigmoid_2}
lstm_cell_18/Relu_1Relulstm_cell_18/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_18/Relu_1¡
lstm_cell_18/mul_2Mullstm_cell_18/Sigmoid_2:y:0!lstm_cell_18/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_18/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ  2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_18_matmul_readvariableop_resource-lstm_cell_18_matmul_1_readvariableop_resource,lstm_cell_18_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_32580027*
condR
while_cond_32580026*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ  22
0TensorArrayV2Stack/TensorListStack/element_shapeé
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
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
:ÿÿÿÿÿÿÿÿÿ*
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
:ÿÿÿÿÿÿÿÿÿ2
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
:ÿÿÿÿÿÿÿÿÿ2

IdentityÈ
NoOpNoOp$^lstm_cell_18/BiasAdd/ReadVariableOp#^lstm_cell_18/MatMul/ReadVariableOp%^lstm_cell_18/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ]: : : 2J
#lstm_cell_18/BiasAdd/ReadVariableOp#lstm_cell_18/BiasAdd/ReadVariableOp2H
"lstm_cell_18/MatMul/ReadVariableOp"lstm_cell_18/MatMul/ReadVariableOp2L
$lstm_cell_18/MatMul_1/ReadVariableOp$lstm_cell_18/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
 
_user_specified_nameinputs
×
g
H__inference_dropout_18_layer_call_and_return_conditional_losses_32582286

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
:ÿÿÿÿÿÿÿÿÿ2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape¹
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
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
:ÿÿÿÿÿÿÿÿÿ2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/Mul_1j
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

f
H__inference_dropout_19_layer_call_and_return_conditional_losses_32580289

inputs

identity_1_
IdentityIdentityinputs*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2

Identityn

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ×:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×
 
_user_specified_nameinputs
Ø
I
-__inference_dropout_19_layer_call_fn_32582966

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
:ÿÿÿÿÿÿÿÿÿ×* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_dropout_19_layer_call_and_return_conditional_losses_325802892
PartitionedCallq
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ×:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×
 
_user_specified_nameinputs
´?
Ö
while_body_32582363
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
3while_lstm_cell_19_matmul_readvariableop_resource_0:
ÜI
5while_lstm_cell_19_matmul_1_readvariableop_resource_0:
×ÜC
4while_lstm_cell_19_biasadd_readvariableop_resource_0:	Ü
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
1while_lstm_cell_19_matmul_readvariableop_resource:
ÜG
3while_lstm_cell_19_matmul_1_readvariableop_resource:
×ÜA
2while_lstm_cell_19_biasadd_readvariableop_resource:	Ü¢)while/lstm_cell_19/BiasAdd/ReadVariableOp¢(while/lstm_cell_19/MatMul/ReadVariableOp¢*while/lstm_cell_19/MatMul_1/ReadVariableOpÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ  29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÔ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemÊ
(while/lstm_cell_19/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_19_matmul_readvariableop_resource_0* 
_output_shapes
:
Ü*
dtype02*
(while/lstm_cell_19/MatMul/ReadVariableOp×
while/lstm_cell_19/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_19/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2
while/lstm_cell_19/MatMulÐ
*while/lstm_cell_19/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_19_matmul_1_readvariableop_resource_0* 
_output_shapes
:
×Ü*
dtype02,
*while/lstm_cell_19/MatMul_1/ReadVariableOpÀ
while/lstm_cell_19/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_19/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2
while/lstm_cell_19/MatMul_1¸
while/lstm_cell_19/addAddV2#while/lstm_cell_19/MatMul:product:0%while/lstm_cell_19/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2
while/lstm_cell_19/addÈ
)while/lstm_cell_19/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_19_biasadd_readvariableop_resource_0*
_output_shapes	
:Ü*
dtype02+
)while/lstm_cell_19/BiasAdd/ReadVariableOpÅ
while/lstm_cell_19/BiasAddBiasAddwhile/lstm_cell_19/add:z:01while/lstm_cell_19/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2
while/lstm_cell_19/BiasAdd
"while/lstm_cell_19/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_19/split/split_dim
while/lstm_cell_19/splitSplit+while/lstm_cell_19/split/split_dim:output:0#while/lstm_cell_19/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ×:ÿÿÿÿÿÿÿÿÿ×:ÿÿÿÿÿÿÿÿÿ×:ÿÿÿÿÿÿÿÿÿ×*
	num_split2
while/lstm_cell_19/split
while/lstm_cell_19/SigmoidSigmoid!while/lstm_cell_19/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2
while/lstm_cell_19/Sigmoid
while/lstm_cell_19/Sigmoid_1Sigmoid!while/lstm_cell_19/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2
while/lstm_cell_19/Sigmoid_1¡
while/lstm_cell_19/mulMul while/lstm_cell_19/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2
while/lstm_cell_19/mul
while/lstm_cell_19/ReluRelu!while/lstm_cell_19/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2
while/lstm_cell_19/Reluµ
while/lstm_cell_19/mul_1Mulwhile/lstm_cell_19/Sigmoid:y:0%while/lstm_cell_19/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2
while/lstm_cell_19/mul_1ª
while/lstm_cell_19/add_1AddV2while/lstm_cell_19/mul:z:0while/lstm_cell_19/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2
while/lstm_cell_19/add_1
while/lstm_cell_19/Sigmoid_2Sigmoid!while/lstm_cell_19/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2
while/lstm_cell_19/Sigmoid_2
while/lstm_cell_19/Relu_1Reluwhile/lstm_cell_19/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2
while/lstm_cell_19/Relu_1¹
while/lstm_cell_19/mul_2Mul while/lstm_cell_19/Sigmoid_2:y:0'while/lstm_cell_19/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2
while/lstm_cell_19/mul_2à
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
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_19/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_19/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2
while/Identity_5Þ

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
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ×:ÿÿÿÿÿÿÿÿÿ×: : : : : 2V
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
:ÿÿÿÿÿÿÿÿÿ×:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×:

_output_shapes
: :

_output_shapes
: 
ã
Í
while_cond_32582815
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_32582815___redundant_placeholder06
2while_while_cond_32582815___redundant_placeholder16
2while_while_cond_32582815___redundant_placeholder26
2while_while_cond_32582815___redundant_placeholder3
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
B: : : : :ÿÿÿÿÿÿÿÿÿ×:ÿÿÿÿÿÿÿÿÿ×: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×:

_output_shapes
: :

_output_shapes
:
°
ô
J__inference_sequential_9_layer_call_and_return_conditional_losses_32580863
lstm_18_input#
lstm_18_32580841:	]Ô$
lstm_18_32580843:
Ô
lstm_18_32580845:	Ô$
lstm_19_32580849:
Ü$
lstm_19_32580851:
×Ü
lstm_19_32580853:	Ü#
dense_9_32580857:	×
dense_9_32580859:
identity¢dense_9/StatefulPartitionedCall¢lstm_18/StatefulPartitionedCall¢lstm_19/StatefulPartitionedCallµ
lstm_18/StatefulPartitionedCallStatefulPartitionedCalllstm_18_inputlstm_18_32580841lstm_18_32580843lstm_18_32580845*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_lstm_18_layer_call_and_return_conditional_losses_325801112!
lstm_18/StatefulPartitionedCall
dropout_18/PartitionedCallPartitionedCall(lstm_18/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_dropout_18_layer_call_and_return_conditional_losses_325801242
dropout_18/PartitionedCallË
lstm_19/StatefulPartitionedCallStatefulPartitionedCall#dropout_18/PartitionedCall:output:0lstm_19_32580849lstm_19_32580851lstm_19_32580853*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_lstm_19_layer_call_and_return_conditional_losses_325802762!
lstm_19/StatefulPartitionedCall
dropout_19/PartitionedCallPartitionedCall(lstm_19/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_dropout_19_layer_call_and_return_conditional_losses_325802892
dropout_19/PartitionedCall¶
dense_9/StatefulPartitionedCallStatefulPartitionedCall#dropout_19/PartitionedCall:output:0dense_9_32580857dense_9_32580859*
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
E__inference_dense_9_layer_call_and_return_conditional_losses_325803222!
dense_9/StatefulPartitionedCall
IdentityIdentity(dense_9/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity´
NoOpNoOp ^dense_9/StatefulPartitionedCall ^lstm_18/StatefulPartitionedCall ^lstm_19/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ]: : : : : : : : 2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall2B
lstm_18/StatefulPartitionedCalllstm_18/StatefulPartitionedCall2B
lstm_19/StatefulPartitionedCalllstm_19/StatefulPartitionedCall:Z V
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
'
_user_specified_namelstm_18_input


J__inference_lstm_cell_18_layer_call_and_return_conditional_losses_32583075

inputs
states_0
states_11
matmul_readvariableop_resource:	]Ô4
 matmul_1_readvariableop_resource:
Ô.
biasadd_readvariableop_resource:	Ô
identity

identity_1

identity_2¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	]Ô*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÔ2
MatMul
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
Ô*
dtype02
MatMul_1/ReadVariableOp|
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÔ2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÔ2
add
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:Ô*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÔ2	
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
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2
split`
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
Sigmoidd
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	Sigmoid_1]
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mulW
ReluRelusplit:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relui
mul_1MulSigmoid:y:0Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mul_1^
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
add_1d
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	Sigmoid_2V
Relu_1Relu	add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu_1m
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mul_2e
IdentityIdentity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityi

Identity_1Identity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1i

Identity_2Identity	add_1:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

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
A:ÿÿÿÿÿÿÿÿÿ]:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
 
_user_specified_nameinputs:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
states/0:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
states/1
°?
Ô
while_body_32581990
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_18_matmul_readvariableop_resource_0:	]ÔI
5while_lstm_cell_18_matmul_1_readvariableop_resource_0:
ÔC
4while_lstm_cell_18_biasadd_readvariableop_resource_0:	Ô
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_18_matmul_readvariableop_resource:	]ÔG
3while_lstm_cell_18_matmul_1_readvariableop_resource:
ÔA
2while_lstm_cell_18_biasadd_readvariableop_resource:	Ô¢)while/lstm_cell_18/BiasAdd/ReadVariableOp¢(while/lstm_cell_18/MatMul/ReadVariableOp¢*while/lstm_cell_18/MatMul_1/ReadVariableOpÃ
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
(while/lstm_cell_18/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_18_matmul_readvariableop_resource_0*
_output_shapes
:	]Ô*
dtype02*
(while/lstm_cell_18/MatMul/ReadVariableOp×
while/lstm_cell_18/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_18/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÔ2
while/lstm_cell_18/MatMulÐ
*while/lstm_cell_18/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_18_matmul_1_readvariableop_resource_0* 
_output_shapes
:
Ô*
dtype02,
*while/lstm_cell_18/MatMul_1/ReadVariableOpÀ
while/lstm_cell_18/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_18/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÔ2
while/lstm_cell_18/MatMul_1¸
while/lstm_cell_18/addAddV2#while/lstm_cell_18/MatMul:product:0%while/lstm_cell_18/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÔ2
while/lstm_cell_18/addÈ
)while/lstm_cell_18/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_18_biasadd_readvariableop_resource_0*
_output_shapes	
:Ô*
dtype02+
)while/lstm_cell_18/BiasAdd/ReadVariableOpÅ
while/lstm_cell_18/BiasAddBiasAddwhile/lstm_cell_18/add:z:01while/lstm_cell_18/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÔ2
while/lstm_cell_18/BiasAdd
"while/lstm_cell_18/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_18/split/split_dim
while/lstm_cell_18/splitSplit+while/lstm_cell_18/split/split_dim:output:0#while/lstm_cell_18/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2
while/lstm_cell_18/split
while/lstm_cell_18/SigmoidSigmoid!while/lstm_cell_18/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_18/Sigmoid
while/lstm_cell_18/Sigmoid_1Sigmoid!while/lstm_cell_18/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_18/Sigmoid_1¡
while/lstm_cell_18/mulMul while/lstm_cell_18/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_18/mul
while/lstm_cell_18/ReluRelu!while/lstm_cell_18/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_18/Reluµ
while/lstm_cell_18/mul_1Mulwhile/lstm_cell_18/Sigmoid:y:0%while/lstm_cell_18/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_18/mul_1ª
while/lstm_cell_18/add_1AddV2while/lstm_cell_18/mul:z:0while/lstm_cell_18/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_18/add_1
while/lstm_cell_18/Sigmoid_2Sigmoid!while/lstm_cell_18/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_18/Sigmoid_2
while/lstm_cell_18/Relu_1Reluwhile/lstm_cell_18/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_18/Relu_1¹
while/lstm_cell_18/mul_2Mul while/lstm_cell_18/Sigmoid_2:y:0'while/lstm_cell_18/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_18/mul_2à
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
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_18/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_18/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/Identity_5Þ

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
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2V
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
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
°?
Ô
while_body_32580657
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_18_matmul_readvariableop_resource_0:	]ÔI
5while_lstm_cell_18_matmul_1_readvariableop_resource_0:
ÔC
4while_lstm_cell_18_biasadd_readvariableop_resource_0:	Ô
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_18_matmul_readvariableop_resource:	]ÔG
3while_lstm_cell_18_matmul_1_readvariableop_resource:
ÔA
2while_lstm_cell_18_biasadd_readvariableop_resource:	Ô¢)while/lstm_cell_18/BiasAdd/ReadVariableOp¢(while/lstm_cell_18/MatMul/ReadVariableOp¢*while/lstm_cell_18/MatMul_1/ReadVariableOpÃ
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
(while/lstm_cell_18/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_18_matmul_readvariableop_resource_0*
_output_shapes
:	]Ô*
dtype02*
(while/lstm_cell_18/MatMul/ReadVariableOp×
while/lstm_cell_18/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_18/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÔ2
while/lstm_cell_18/MatMulÐ
*while/lstm_cell_18/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_18_matmul_1_readvariableop_resource_0* 
_output_shapes
:
Ô*
dtype02,
*while/lstm_cell_18/MatMul_1/ReadVariableOpÀ
while/lstm_cell_18/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_18/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÔ2
while/lstm_cell_18/MatMul_1¸
while/lstm_cell_18/addAddV2#while/lstm_cell_18/MatMul:product:0%while/lstm_cell_18/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÔ2
while/lstm_cell_18/addÈ
)while/lstm_cell_18/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_18_biasadd_readvariableop_resource_0*
_output_shapes	
:Ô*
dtype02+
)while/lstm_cell_18/BiasAdd/ReadVariableOpÅ
while/lstm_cell_18/BiasAddBiasAddwhile/lstm_cell_18/add:z:01while/lstm_cell_18/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÔ2
while/lstm_cell_18/BiasAdd
"while/lstm_cell_18/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_18/split/split_dim
while/lstm_cell_18/splitSplit+while/lstm_cell_18/split/split_dim:output:0#while/lstm_cell_18/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2
while/lstm_cell_18/split
while/lstm_cell_18/SigmoidSigmoid!while/lstm_cell_18/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_18/Sigmoid
while/lstm_cell_18/Sigmoid_1Sigmoid!while/lstm_cell_18/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_18/Sigmoid_1¡
while/lstm_cell_18/mulMul while/lstm_cell_18/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_18/mul
while/lstm_cell_18/ReluRelu!while/lstm_cell_18/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_18/Reluµ
while/lstm_cell_18/mul_1Mulwhile/lstm_cell_18/Sigmoid:y:0%while/lstm_cell_18/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_18/mul_1ª
while/lstm_cell_18/add_1AddV2while/lstm_cell_18/mul:z:0while/lstm_cell_18/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_18/add_1
while/lstm_cell_18/Sigmoid_2Sigmoid!while/lstm_cell_18/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_18/Sigmoid_2
while/lstm_cell_18/Relu_1Reluwhile/lstm_cell_18/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_18/Relu_1¹
while/lstm_cell_18/mul_2Mul while/lstm_cell_18/Sigmoid_2:y:0'while/lstm_cell_18/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_18/mul_2à
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
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_18/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_18/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/Identity_5Þ

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
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2V
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
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
à
º
*__inference_lstm_18_layer_call_fn_32582236
inputs_0
unknown:	]Ô
	unknown_0:
Ô
	unknown_1:	Ô
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_lstm_18_layer_call_and_return_conditional_losses_325788522
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

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
öø

J__inference_sequential_9_layer_call_and_return_conditional_losses_32581238

inputsF
3lstm_18_lstm_cell_18_matmul_readvariableop_resource:	]ÔI
5lstm_18_lstm_cell_18_matmul_1_readvariableop_resource:
ÔC
4lstm_18_lstm_cell_18_biasadd_readvariableop_resource:	ÔG
3lstm_19_lstm_cell_19_matmul_readvariableop_resource:
ÜI
5lstm_19_lstm_cell_19_matmul_1_readvariableop_resource:
×ÜC
4lstm_19_lstm_cell_19_biasadd_readvariableop_resource:	Ü<
)dense_9_tensordot_readvariableop_resource:	×5
'dense_9_biasadd_readvariableop_resource:
identity¢dense_9/BiasAdd/ReadVariableOp¢ dense_9/Tensordot/ReadVariableOp¢+lstm_18/lstm_cell_18/BiasAdd/ReadVariableOp¢*lstm_18/lstm_cell_18/MatMul/ReadVariableOp¢,lstm_18/lstm_cell_18/MatMul_1/ReadVariableOp¢lstm_18/while¢+lstm_19/lstm_cell_19/BiasAdd/ReadVariableOp¢*lstm_19/lstm_cell_19/MatMul/ReadVariableOp¢,lstm_19/lstm_cell_19/MatMul_1/ReadVariableOp¢lstm_19/whileT
lstm_18/ShapeShapeinputs*
T0*
_output_shapes
:2
lstm_18/Shape
lstm_18/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_18/strided_slice/stack
lstm_18/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_18/strided_slice/stack_1
lstm_18/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_18/strided_slice/stack_2
lstm_18/strided_sliceStridedSlicelstm_18/Shape:output:0$lstm_18/strided_slice/stack:output:0&lstm_18/strided_slice/stack_1:output:0&lstm_18/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_18/strided_slicem
lstm_18/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :2
lstm_18/zeros/mul/y
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
B :è2
lstm_18/zeros/Less/y
lstm_18/zeros/LessLesslstm_18/zeros/mul:z:0lstm_18/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_18/zeros/Lesss
lstm_18/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :2
lstm_18/zeros/packed/1£
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
lstm_18/zeros/Const
lstm_18/zerosFilllstm_18/zeros/packed:output:0lstm_18/zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_18/zerosq
lstm_18/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :2
lstm_18/zeros_1/mul/y
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
B :è2
lstm_18/zeros_1/Less/y
lstm_18/zeros_1/LessLesslstm_18/zeros_1/mul:z:0lstm_18/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_18/zeros_1/Lessw
lstm_18/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :2
lstm_18/zeros_1/packed/1©
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
lstm_18/zeros_1/Const
lstm_18/zeros_1Filllstm_18/zeros_1/packed:output:0lstm_18/zeros_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_18/zeros_1
lstm_18/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_18/transpose/perm
lstm_18/transpose	Transposeinputslstm_18/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]2
lstm_18/transposeg
lstm_18/Shape_1Shapelstm_18/transpose:y:0*
T0*
_output_shapes
:2
lstm_18/Shape_1
lstm_18/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_18/strided_slice_1/stack
lstm_18/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_18/strided_slice_1/stack_1
lstm_18/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_18/strided_slice_1/stack_2
lstm_18/strided_slice_1StridedSlicelstm_18/Shape_1:output:0&lstm_18/strided_slice_1/stack:output:0(lstm_18/strided_slice_1/stack_1:output:0(lstm_18/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_18/strided_slice_1
#lstm_18/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2%
#lstm_18/TensorArrayV2/element_shapeÒ
lstm_18/TensorArrayV2TensorListReserve,lstm_18/TensorArrayV2/element_shape:output:0 lstm_18/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_18/TensorArrayV2Ï
=lstm_18/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ]   2?
=lstm_18/TensorArrayUnstack/TensorListFromTensor/element_shape
/lstm_18/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_18/transpose:y:0Flstm_18/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type021
/lstm_18/TensorArrayUnstack/TensorListFromTensor
lstm_18/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_18/strided_slice_2/stack
lstm_18/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_18/strided_slice_2/stack_1
lstm_18/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_18/strided_slice_2/stack_2¬
lstm_18/strided_slice_2StridedSlicelstm_18/transpose:y:0&lstm_18/strided_slice_2/stack:output:0(lstm_18/strided_slice_2/stack_1:output:0(lstm_18/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]*
shrink_axis_mask2
lstm_18/strided_slice_2Í
*lstm_18/lstm_cell_18/MatMul/ReadVariableOpReadVariableOp3lstm_18_lstm_cell_18_matmul_readvariableop_resource*
_output_shapes
:	]Ô*
dtype02,
*lstm_18/lstm_cell_18/MatMul/ReadVariableOpÍ
lstm_18/lstm_cell_18/MatMulMatMul lstm_18/strided_slice_2:output:02lstm_18/lstm_cell_18/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÔ2
lstm_18/lstm_cell_18/MatMulÔ
,lstm_18/lstm_cell_18/MatMul_1/ReadVariableOpReadVariableOp5lstm_18_lstm_cell_18_matmul_1_readvariableop_resource* 
_output_shapes
:
Ô*
dtype02.
,lstm_18/lstm_cell_18/MatMul_1/ReadVariableOpÉ
lstm_18/lstm_cell_18/MatMul_1MatMullstm_18/zeros:output:04lstm_18/lstm_cell_18/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÔ2
lstm_18/lstm_cell_18/MatMul_1À
lstm_18/lstm_cell_18/addAddV2%lstm_18/lstm_cell_18/MatMul:product:0'lstm_18/lstm_cell_18/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÔ2
lstm_18/lstm_cell_18/addÌ
+lstm_18/lstm_cell_18/BiasAdd/ReadVariableOpReadVariableOp4lstm_18_lstm_cell_18_biasadd_readvariableop_resource*
_output_shapes	
:Ô*
dtype02-
+lstm_18/lstm_cell_18/BiasAdd/ReadVariableOpÍ
lstm_18/lstm_cell_18/BiasAddBiasAddlstm_18/lstm_cell_18/add:z:03lstm_18/lstm_cell_18/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÔ2
lstm_18/lstm_cell_18/BiasAdd
$lstm_18/lstm_cell_18/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2&
$lstm_18/lstm_cell_18/split/split_dim
lstm_18/lstm_cell_18/splitSplit-lstm_18/lstm_cell_18/split/split_dim:output:0%lstm_18/lstm_cell_18/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2
lstm_18/lstm_cell_18/split
lstm_18/lstm_cell_18/SigmoidSigmoid#lstm_18/lstm_cell_18/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_18/lstm_cell_18/Sigmoid£
lstm_18/lstm_cell_18/Sigmoid_1Sigmoid#lstm_18/lstm_cell_18/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
lstm_18/lstm_cell_18/Sigmoid_1¬
lstm_18/lstm_cell_18/mulMul"lstm_18/lstm_cell_18/Sigmoid_1:y:0lstm_18/zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_18/lstm_cell_18/mul
lstm_18/lstm_cell_18/ReluRelu#lstm_18/lstm_cell_18/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_18/lstm_cell_18/Relu½
lstm_18/lstm_cell_18/mul_1Mul lstm_18/lstm_cell_18/Sigmoid:y:0'lstm_18/lstm_cell_18/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_18/lstm_cell_18/mul_1²
lstm_18/lstm_cell_18/add_1AddV2lstm_18/lstm_cell_18/mul:z:0lstm_18/lstm_cell_18/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_18/lstm_cell_18/add_1£
lstm_18/lstm_cell_18/Sigmoid_2Sigmoid#lstm_18/lstm_cell_18/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
lstm_18/lstm_cell_18/Sigmoid_2
lstm_18/lstm_cell_18/Relu_1Relulstm_18/lstm_cell_18/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_18/lstm_cell_18/Relu_1Á
lstm_18/lstm_cell_18/mul_2Mul"lstm_18/lstm_cell_18/Sigmoid_2:y:0)lstm_18/lstm_cell_18/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_18/lstm_cell_18/mul_2
%lstm_18/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ  2'
%lstm_18/TensorArrayV2_1/element_shapeØ
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
lstm_18/time
 lstm_18/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2"
 lstm_18/while/maximum_iterationsz
lstm_18/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_18/while/loop_counter
lstm_18/whileWhile#lstm_18/while/loop_counter:output:0)lstm_18/while/maximum_iterations:output:0lstm_18/time:output:0 lstm_18/TensorArrayV2_1:handle:0lstm_18/zeros:output:0lstm_18/zeros_1:output:0 lstm_18/strided_slice_1:output:0?lstm_18/TensorArrayUnstack/TensorListFromTensor:output_handle:03lstm_18_lstm_cell_18_matmul_readvariableop_resource5lstm_18_lstm_cell_18_matmul_1_readvariableop_resource4lstm_18_lstm_cell_18_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *'
bodyR
lstm_18_while_body_32580978*'
condR
lstm_18_while_cond_32580977*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 2
lstm_18/whileÅ
8lstm_18/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ  2:
8lstm_18/TensorArrayV2Stack/TensorListStack/element_shape
*lstm_18/TensorArrayV2Stack/TensorListStackTensorListStacklstm_18/while:output:3Alstm_18/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype02,
*lstm_18/TensorArrayV2Stack/TensorListStack
lstm_18/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
lstm_18/strided_slice_3/stack
lstm_18/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2!
lstm_18/strided_slice_3/stack_1
lstm_18/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_18/strided_slice_3/stack_2Ë
lstm_18/strided_slice_3StridedSlice3lstm_18/TensorArrayV2Stack/TensorListStack:tensor:0&lstm_18/strided_slice_3/stack:output:0(lstm_18/strided_slice_3/stack_1:output:0(lstm_18/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
lstm_18/strided_slice_3
lstm_18/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_18/transpose_1/permÆ
lstm_18/transpose_1	Transpose3lstm_18/TensorArrayV2Stack/TensorListStack:tensor:0!lstm_18/transpose_1/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_18/transpose_1v
lstm_18/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_18/runtime
dropout_18/IdentityIdentitylstm_18/transpose_1:y:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout_18/Identityj
lstm_19/ShapeShapedropout_18/Identity:output:0*
T0*
_output_shapes
:2
lstm_19/Shape
lstm_19/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_19/strided_slice/stack
lstm_19/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_19/strided_slice/stack_1
lstm_19/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_19/strided_slice/stack_2
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
B :×2
lstm_19/zeros/mul/y
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
B :è2
lstm_19/zeros/Less/y
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
B :×2
lstm_19/zeros/packed/1£
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
lstm_19/zeros/Const
lstm_19/zerosFilllstm_19/zeros/packed:output:0lstm_19/zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2
lstm_19/zerosq
lstm_19/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :×2
lstm_19/zeros_1/mul/y
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
B :è2
lstm_19/zeros_1/Less/y
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
B :×2
lstm_19/zeros_1/packed/1©
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
lstm_19/zeros_1/Const
lstm_19/zeros_1Filllstm_19/zeros_1/packed:output:0lstm_19/zeros_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2
lstm_19/zeros_1
lstm_19/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_19/transpose/perm©
lstm_19/transpose	Transposedropout_18/Identity:output:0lstm_19/transpose/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_19/transposeg
lstm_19/Shape_1Shapelstm_19/transpose:y:0*
T0*
_output_shapes
:2
lstm_19/Shape_1
lstm_19/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_19/strided_slice_1/stack
lstm_19/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_19/strided_slice_1/stack_1
lstm_19/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_19/strided_slice_1/stack_2
lstm_19/strided_slice_1StridedSlicelstm_19/Shape_1:output:0&lstm_19/strided_slice_1/stack:output:0(lstm_19/strided_slice_1/stack_1:output:0(lstm_19/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_19/strided_slice_1
#lstm_19/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2%
#lstm_19/TensorArrayV2/element_shapeÒ
lstm_19/TensorArrayV2TensorListReserve,lstm_19/TensorArrayV2/element_shape:output:0 lstm_19/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_19/TensorArrayV2Ï
=lstm_19/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ  2?
=lstm_19/TensorArrayUnstack/TensorListFromTensor/element_shape
/lstm_19/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_19/transpose:y:0Flstm_19/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type021
/lstm_19/TensorArrayUnstack/TensorListFromTensor
lstm_19/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_19/strided_slice_2/stack
lstm_19/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_19/strided_slice_2/stack_1
lstm_19/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_19/strided_slice_2/stack_2­
lstm_19/strided_slice_2StridedSlicelstm_19/transpose:y:0&lstm_19/strided_slice_2/stack:output:0(lstm_19/strided_slice_2/stack_1:output:0(lstm_19/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
lstm_19/strided_slice_2Î
*lstm_19/lstm_cell_19/MatMul/ReadVariableOpReadVariableOp3lstm_19_lstm_cell_19_matmul_readvariableop_resource* 
_output_shapes
:
Ü*
dtype02,
*lstm_19/lstm_cell_19/MatMul/ReadVariableOpÍ
lstm_19/lstm_cell_19/MatMulMatMul lstm_19/strided_slice_2:output:02lstm_19/lstm_cell_19/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2
lstm_19/lstm_cell_19/MatMulÔ
,lstm_19/lstm_cell_19/MatMul_1/ReadVariableOpReadVariableOp5lstm_19_lstm_cell_19_matmul_1_readvariableop_resource* 
_output_shapes
:
×Ü*
dtype02.
,lstm_19/lstm_cell_19/MatMul_1/ReadVariableOpÉ
lstm_19/lstm_cell_19/MatMul_1MatMullstm_19/zeros:output:04lstm_19/lstm_cell_19/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2
lstm_19/lstm_cell_19/MatMul_1À
lstm_19/lstm_cell_19/addAddV2%lstm_19/lstm_cell_19/MatMul:product:0'lstm_19/lstm_cell_19/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2
lstm_19/lstm_cell_19/addÌ
+lstm_19/lstm_cell_19/BiasAdd/ReadVariableOpReadVariableOp4lstm_19_lstm_cell_19_biasadd_readvariableop_resource*
_output_shapes	
:Ü*
dtype02-
+lstm_19/lstm_cell_19/BiasAdd/ReadVariableOpÍ
lstm_19/lstm_cell_19/BiasAddBiasAddlstm_19/lstm_cell_19/add:z:03lstm_19/lstm_cell_19/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2
lstm_19/lstm_cell_19/BiasAdd
$lstm_19/lstm_cell_19/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2&
$lstm_19/lstm_cell_19/split/split_dim
lstm_19/lstm_cell_19/splitSplit-lstm_19/lstm_cell_19/split/split_dim:output:0%lstm_19/lstm_cell_19/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ×:ÿÿÿÿÿÿÿÿÿ×:ÿÿÿÿÿÿÿÿÿ×:ÿÿÿÿÿÿÿÿÿ×*
	num_split2
lstm_19/lstm_cell_19/split
lstm_19/lstm_cell_19/SigmoidSigmoid#lstm_19/lstm_cell_19/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2
lstm_19/lstm_cell_19/Sigmoid£
lstm_19/lstm_cell_19/Sigmoid_1Sigmoid#lstm_19/lstm_cell_19/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2 
lstm_19/lstm_cell_19/Sigmoid_1¬
lstm_19/lstm_cell_19/mulMul"lstm_19/lstm_cell_19/Sigmoid_1:y:0lstm_19/zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2
lstm_19/lstm_cell_19/mul
lstm_19/lstm_cell_19/ReluRelu#lstm_19/lstm_cell_19/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2
lstm_19/lstm_cell_19/Relu½
lstm_19/lstm_cell_19/mul_1Mul lstm_19/lstm_cell_19/Sigmoid:y:0'lstm_19/lstm_cell_19/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2
lstm_19/lstm_cell_19/mul_1²
lstm_19/lstm_cell_19/add_1AddV2lstm_19/lstm_cell_19/mul:z:0lstm_19/lstm_cell_19/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2
lstm_19/lstm_cell_19/add_1£
lstm_19/lstm_cell_19/Sigmoid_2Sigmoid#lstm_19/lstm_cell_19/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2 
lstm_19/lstm_cell_19/Sigmoid_2
lstm_19/lstm_cell_19/Relu_1Relulstm_19/lstm_cell_19/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2
lstm_19/lstm_cell_19/Relu_1Á
lstm_19/lstm_cell_19/mul_2Mul"lstm_19/lstm_cell_19/Sigmoid_2:y:0)lstm_19/lstm_cell_19/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2
lstm_19/lstm_cell_19/mul_2
%lstm_19/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ×   2'
%lstm_19/TensorArrayV2_1/element_shapeØ
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
lstm_19/time
 lstm_19/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2"
 lstm_19/while/maximum_iterationsz
lstm_19/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_19/while/loop_counter
lstm_19/whileWhile#lstm_19/while/loop_counter:output:0)lstm_19/while/maximum_iterations:output:0lstm_19/time:output:0 lstm_19/TensorArrayV2_1:handle:0lstm_19/zeros:output:0lstm_19/zeros_1:output:0 lstm_19/strided_slice_1:output:0?lstm_19/TensorArrayUnstack/TensorListFromTensor:output_handle:03lstm_19_lstm_cell_19_matmul_readvariableop_resource5lstm_19_lstm_cell_19_matmul_1_readvariableop_resource4lstm_19_lstm_cell_19_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ×:ÿÿÿÿÿÿÿÿÿ×: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *'
bodyR
lstm_19_while_body_32581126*'
condR
lstm_19_while_cond_32581125*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ×:ÿÿÿÿÿÿÿÿÿ×: : : : : *
parallel_iterations 2
lstm_19/whileÅ
8lstm_19/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ×   2:
8lstm_19/TensorArrayV2Stack/TensorListStack/element_shape
*lstm_19/TensorArrayV2Stack/TensorListStackTensorListStacklstm_19/while:output:3Alstm_19/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×*
element_dtype02,
*lstm_19/TensorArrayV2Stack/TensorListStack
lstm_19/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
lstm_19/strided_slice_3/stack
lstm_19/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2!
lstm_19/strided_slice_3/stack_1
lstm_19/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_19/strided_slice_3/stack_2Ë
lstm_19/strided_slice_3StridedSlice3lstm_19/TensorArrayV2Stack/TensorListStack:tensor:0&lstm_19/strided_slice_3/stack:output:0(lstm_19/strided_slice_3/stack_1:output:0(lstm_19/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×*
shrink_axis_mask2
lstm_19/strided_slice_3
lstm_19/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_19/transpose_1/permÆ
lstm_19/transpose_1	Transpose3lstm_19/TensorArrayV2Stack/TensorListStack:tensor:0!lstm_19/transpose_1/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2
lstm_19/transpose_1v
lstm_19/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_19/runtime
dropout_19/IdentityIdentitylstm_19/transpose_1:y:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2
dropout_19/Identity¯
 dense_9/Tensordot/ReadVariableOpReadVariableOp)dense_9_tensordot_readvariableop_resource*
_output_shapes
:	×*
dtype02"
 dense_9/Tensordot/ReadVariableOpz
dense_9/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_9/Tensordot/axes
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
dense_9/Tensordot/Shape
dense_9/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_9/Tensordot/GatherV2/axisù
dense_9/Tensordot/GatherV2GatherV2 dense_9/Tensordot/Shape:output:0dense_9/Tensordot/free:output:0(dense_9/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_9/Tensordot/GatherV2
!dense_9/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_9/Tensordot/GatherV2_1/axisÿ
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
dense_9/Tensordot/Const 
dense_9/Tensordot/ProdProd#dense_9/Tensordot/GatherV2:output:0 dense_9/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_9/Tensordot/Prod
dense_9/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_9/Tensordot/Const_1¨
dense_9/Tensordot/Prod_1Prod%dense_9/Tensordot/GatherV2_1:output:0"dense_9/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_9/Tensordot/Prod_1
dense_9/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense_9/Tensordot/concat/axisØ
dense_9/Tensordot/concatConcatV2dense_9/Tensordot/free:output:0dense_9/Tensordot/axes:output:0&dense_9/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_9/Tensordot/concat¬
dense_9/Tensordot/stackPackdense_9/Tensordot/Prod:output:0!dense_9/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_9/Tensordot/stack¿
dense_9/Tensordot/transpose	Transposedropout_19/Identity:output:0!dense_9/Tensordot/concat:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2
dense_9/Tensordot/transpose¿
dense_9/Tensordot/ReshapeReshapedense_9/Tensordot/transpose:y:0 dense_9/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
dense_9/Tensordot/Reshape¾
dense_9/Tensordot/MatMulMatMul"dense_9/Tensordot/Reshape:output:0(dense_9/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_9/Tensordot/MatMul
dense_9/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
dense_9/Tensordot/Const_2
dense_9/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_9/Tensordot/concat_1/axiså
dense_9/Tensordot/concat_1ConcatV2#dense_9/Tensordot/GatherV2:output:0"dense_9/Tensordot/Const_2:output:0(dense_9/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_9/Tensordot/concat_1°
dense_9/TensordotReshape"dense_9/Tensordot/MatMul:product:0#dense_9/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_9/Tensordot¤
dense_9/BiasAdd/ReadVariableOpReadVariableOp'dense_9_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_9/BiasAdd/ReadVariableOp§
dense_9/BiasAddBiasAdddense_9/Tensordot:output:0&dense_9/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_9/BiasAdd}
dense_9/SoftmaxSoftmaxdense_9/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_9/Softmaxx
IdentityIdentitydense_9/Softmax:softmax:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

IdentityÆ
NoOpNoOp^dense_9/BiasAdd/ReadVariableOp!^dense_9/Tensordot/ReadVariableOp,^lstm_18/lstm_cell_18/BiasAdd/ReadVariableOp+^lstm_18/lstm_cell_18/MatMul/ReadVariableOp-^lstm_18/lstm_cell_18/MatMul_1/ReadVariableOp^lstm_18/while,^lstm_19/lstm_cell_19/BiasAdd/ReadVariableOp+^lstm_19/lstm_cell_19/MatMul/ReadVariableOp-^lstm_19/lstm_cell_19/MatMul_1/ReadVariableOp^lstm_19/while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ]: : : : : : : : 2@
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
:ÿÿÿÿÿÿÿÿÿ]
 
_user_specified_nameinputs
&
õ
while_body_32579623
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_01
while_lstm_cell_19_32579647_0:
Ü1
while_lstm_cell_19_32579649_0:
×Ü,
while_lstm_cell_19_32579651_0:	Ü
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor/
while_lstm_cell_19_32579647:
Ü/
while_lstm_cell_19_32579649:
×Ü*
while_lstm_cell_19_32579651:	Ü¢*while/lstm_cell_19/StatefulPartitionedCallÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ  29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÔ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemì
*while/lstm_cell_19/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_19_32579647_0while_lstm_cell_19_32579649_0while_lstm_cell_19_32579651_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ×:ÿÿÿÿÿÿÿÿÿ×:ÿÿÿÿÿÿÿÿÿ×*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_lstm_cell_19_layer_call_and_return_conditional_losses_325795452,
*while/lstm_cell_19/StatefulPartitionedCall÷
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
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3¥
while/Identity_4Identity3while/lstm_cell_19/StatefulPartitionedCall:output:1^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2
while/Identity_4¥
while/Identity_5Identity3while/lstm_cell_19/StatefulPartitionedCall:output:2^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2
while/Identity_5

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
while_lstm_cell_19_32579647while_lstm_cell_19_32579647_0"<
while_lstm_cell_19_32579649while_lstm_cell_19_32579649_0"<
while_lstm_cell_19_32579651while_lstm_cell_19_32579651_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ×:ÿÿÿÿÿÿÿÿÿ×: : : : : 2X
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
:ÿÿÿÿÿÿÿÿÿ×:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×:

_output_shapes
: :

_output_shapes
: 
ã
Í
while_cond_32581687
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_32581687___redundant_placeholder06
2while_while_cond_32581687___redundant_placeholder16
2while_while_cond_32581687___redundant_placeholder26
2while_while_cond_32581687___redundant_placeholder3
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
B: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
Ç
ù
/__inference_lstm_cell_18_layer_call_fn_32583092

inputs
states_0
states_1
unknown:	]Ô
	unknown_0:
Ô
	unknown_1:	Ô
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
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_lstm_cell_18_layer_call_and_return_conditional_losses_325787692
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

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
A:ÿÿÿÿÿÿÿÿÿ]:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
 
_user_specified_nameinputs:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
states/0:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
states/1
\

E__inference_lstm_18_layer_call_and_return_conditional_losses_32580741

inputs>
+lstm_cell_18_matmul_readvariableop_resource:	]ÔA
-lstm_cell_18_matmul_1_readvariableop_resource:
Ô;
,lstm_cell_18_biasadd_readvariableop_resource:	Ô
identity¢#lstm_cell_18/BiasAdd/ReadVariableOp¢"lstm_cell_18/MatMul/ReadVariableOp¢$lstm_cell_18/MatMul_1/ReadVariableOp¢whileD
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
B :2
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
B :2
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
:ÿÿÿÿÿÿÿÿÿ2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :2
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
B :2
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
:ÿÿÿÿÿÿÿÿÿ2	
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
"lstm_cell_18/MatMul/ReadVariableOpReadVariableOp+lstm_cell_18_matmul_readvariableop_resource*
_output_shapes
:	]Ô*
dtype02$
"lstm_cell_18/MatMul/ReadVariableOp­
lstm_cell_18/MatMulMatMulstrided_slice_2:output:0*lstm_cell_18/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÔ2
lstm_cell_18/MatMul¼
$lstm_cell_18/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_18_matmul_1_readvariableop_resource* 
_output_shapes
:
Ô*
dtype02&
$lstm_cell_18/MatMul_1/ReadVariableOp©
lstm_cell_18/MatMul_1MatMulzeros:output:0,lstm_cell_18/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÔ2
lstm_cell_18/MatMul_1 
lstm_cell_18/addAddV2lstm_cell_18/MatMul:product:0lstm_cell_18/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÔ2
lstm_cell_18/add´
#lstm_cell_18/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_18_biasadd_readvariableop_resource*
_output_shapes	
:Ô*
dtype02%
#lstm_cell_18/BiasAdd/ReadVariableOp­
lstm_cell_18/BiasAddBiasAddlstm_cell_18/add:z:0+lstm_cell_18/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÔ2
lstm_cell_18/BiasAdd~
lstm_cell_18/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_18/split/split_dim÷
lstm_cell_18/splitSplit%lstm_cell_18/split/split_dim:output:0lstm_cell_18/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2
lstm_cell_18/split
lstm_cell_18/SigmoidSigmoidlstm_cell_18/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_18/Sigmoid
lstm_cell_18/Sigmoid_1Sigmoidlstm_cell_18/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_18/Sigmoid_1
lstm_cell_18/mulMullstm_cell_18/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_18/mul~
lstm_cell_18/ReluRelulstm_cell_18/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_18/Relu
lstm_cell_18/mul_1Mullstm_cell_18/Sigmoid:y:0lstm_cell_18/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_18/mul_1
lstm_cell_18/add_1AddV2lstm_cell_18/mul:z:0lstm_cell_18/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_18/add_1
lstm_cell_18/Sigmoid_2Sigmoidlstm_cell_18/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_18/Sigmoid_2}
lstm_cell_18/Relu_1Relulstm_cell_18/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_18/Relu_1¡
lstm_cell_18/mul_2Mullstm_cell_18/Sigmoid_2:y:0!lstm_cell_18/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_18/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ  2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_18_matmul_readvariableop_resource-lstm_cell_18_matmul_1_readvariableop_resource,lstm_cell_18_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_32580657*
condR
while_cond_32580656*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ  22
0TensorArrayV2Stack/TensorListStack/element_shapeé
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
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
:ÿÿÿÿÿÿÿÿÿ*
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
:ÿÿÿÿÿÿÿÿÿ2
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
:ÿÿÿÿÿÿÿÿÿ2

IdentityÈ
NoOpNoOp$^lstm_cell_18/BiasAdd/ReadVariableOp#^lstm_cell_18/MatMul/ReadVariableOp%^lstm_cell_18/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ]: : : 2J
#lstm_cell_18/BiasAdd/ReadVariableOp#lstm_cell_18/BiasAdd/ReadVariableOp2H
"lstm_cell_18/MatMul/ReadVariableOp"lstm_cell_18/MatMul/ReadVariableOp2L
$lstm_cell_18/MatMul_1/ReadVariableOp$lstm_cell_18/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
 
_user_specified_nameinputs
Ô!
ý
E__inference_dense_9_layer_call_and_return_conditional_losses_32583002

inputs4
!tensordot_readvariableop_resource:	×-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢Tensordot/ReadVariableOp
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes
:	×*
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
:ÿÿÿÿÿÿÿÿÿ×2
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
:ÿÿÿÿÿÿÿÿÿ×: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×
 
_user_specified_nameinputs
É\
¡
E__inference_lstm_19_layer_call_and_return_conditional_losses_32582447
inputs_0?
+lstm_cell_19_matmul_readvariableop_resource:
ÜA
-lstm_cell_19_matmul_1_readvariableop_resource:
×Ü;
,lstm_cell_19_biasadd_readvariableop_resource:	Ü
identity¢#lstm_cell_19/BiasAdd/ReadVariableOp¢"lstm_cell_19/MatMul/ReadVariableOp¢$lstm_cell_19/MatMul_1/ReadVariableOp¢whileF
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
B :×2
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
B :×2
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
:ÿÿÿÿÿÿÿÿÿ×2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :×2
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
B :×2
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
:ÿÿÿÿÿÿÿÿÿ×2	
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
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
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
valueB"ÿÿÿÿ  27
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
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
strided_slice_2¶
"lstm_cell_19/MatMul/ReadVariableOpReadVariableOp+lstm_cell_19_matmul_readvariableop_resource* 
_output_shapes
:
Ü*
dtype02$
"lstm_cell_19/MatMul/ReadVariableOp­
lstm_cell_19/MatMulMatMulstrided_slice_2:output:0*lstm_cell_19/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2
lstm_cell_19/MatMul¼
$lstm_cell_19/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_19_matmul_1_readvariableop_resource* 
_output_shapes
:
×Ü*
dtype02&
$lstm_cell_19/MatMul_1/ReadVariableOp©
lstm_cell_19/MatMul_1MatMulzeros:output:0,lstm_cell_19/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2
lstm_cell_19/MatMul_1 
lstm_cell_19/addAddV2lstm_cell_19/MatMul:product:0lstm_cell_19/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2
lstm_cell_19/add´
#lstm_cell_19/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_19_biasadd_readvariableop_resource*
_output_shapes	
:Ü*
dtype02%
#lstm_cell_19/BiasAdd/ReadVariableOp­
lstm_cell_19/BiasAddBiasAddlstm_cell_19/add:z:0+lstm_cell_19/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2
lstm_cell_19/BiasAdd~
lstm_cell_19/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_19/split/split_dim÷
lstm_cell_19/splitSplit%lstm_cell_19/split/split_dim:output:0lstm_cell_19/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ×:ÿÿÿÿÿÿÿÿÿ×:ÿÿÿÿÿÿÿÿÿ×:ÿÿÿÿÿÿÿÿÿ×*
	num_split2
lstm_cell_19/split
lstm_cell_19/SigmoidSigmoidlstm_cell_19/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2
lstm_cell_19/Sigmoid
lstm_cell_19/Sigmoid_1Sigmoidlstm_cell_19/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2
lstm_cell_19/Sigmoid_1
lstm_cell_19/mulMullstm_cell_19/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2
lstm_cell_19/mul~
lstm_cell_19/ReluRelulstm_cell_19/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2
lstm_cell_19/Relu
lstm_cell_19/mul_1Mullstm_cell_19/Sigmoid:y:0lstm_cell_19/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2
lstm_cell_19/mul_1
lstm_cell_19/add_1AddV2lstm_cell_19/mul:z:0lstm_cell_19/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2
lstm_cell_19/add_1
lstm_cell_19/Sigmoid_2Sigmoidlstm_cell_19/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2
lstm_cell_19/Sigmoid_2}
lstm_cell_19/Relu_1Relulstm_cell_19/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2
lstm_cell_19/Relu_1¡
lstm_cell_19/mul_2Mullstm_cell_19/Sigmoid_2:y:0!lstm_cell_19/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2
lstm_cell_19/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ×   2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_19_matmul_readvariableop_resource-lstm_cell_19_matmul_1_readvariableop_resource,lstm_cell_19_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ×:ÿÿÿÿÿÿÿÿÿ×: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_32582363*
condR
while_cond_32582362*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ×:ÿÿÿÿÿÿÿÿÿ×: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ×   22
0TensorArrayV2Stack/TensorListStack/element_shapeò
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ×*
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
:ÿÿÿÿÿÿÿÿÿ×*
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
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ×2
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
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ×2

IdentityÈ
NoOpNoOp$^lstm_cell_19/BiasAdd/ReadVariableOp#^lstm_cell_19/MatMul/ReadVariableOp%^lstm_cell_19/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 2J
#lstm_cell_19/BiasAdd/ReadVariableOp#lstm_cell_19/BiasAdd/ReadVariableOp2H
"lstm_cell_19/MatMul/ReadVariableOp"lstm_cell_19/MatMul/ReadVariableOp2L
$lstm_cell_19/MatMul_1/ReadVariableOp$lstm_cell_19/MatMul_1/ReadVariableOp2
whilewhile:_ [
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0
¹
¹
*__inference_lstm_19_layer_call_fn_32582933

inputs
unknown:
Ü
	unknown_0:
×Ü
	unknown_1:	Ü
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_lstm_19_layer_call_and_return_conditional_losses_325802762
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:ÿÿÿÿÿÿÿÿÿ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¬]
õ
(sequential_9_lstm_18_while_body_32578434F
Bsequential_9_lstm_18_while_sequential_9_lstm_18_while_loop_counterL
Hsequential_9_lstm_18_while_sequential_9_lstm_18_while_maximum_iterations*
&sequential_9_lstm_18_while_placeholder,
(sequential_9_lstm_18_while_placeholder_1,
(sequential_9_lstm_18_while_placeholder_2,
(sequential_9_lstm_18_while_placeholder_3E
Asequential_9_lstm_18_while_sequential_9_lstm_18_strided_slice_1_0
}sequential_9_lstm_18_while_tensorarrayv2read_tensorlistgetitem_sequential_9_lstm_18_tensorarrayunstack_tensorlistfromtensor_0[
Hsequential_9_lstm_18_while_lstm_cell_18_matmul_readvariableop_resource_0:	]Ô^
Jsequential_9_lstm_18_while_lstm_cell_18_matmul_1_readvariableop_resource_0:
ÔX
Isequential_9_lstm_18_while_lstm_cell_18_biasadd_readvariableop_resource_0:	Ô'
#sequential_9_lstm_18_while_identity)
%sequential_9_lstm_18_while_identity_1)
%sequential_9_lstm_18_while_identity_2)
%sequential_9_lstm_18_while_identity_3)
%sequential_9_lstm_18_while_identity_4)
%sequential_9_lstm_18_while_identity_5C
?sequential_9_lstm_18_while_sequential_9_lstm_18_strided_slice_1
{sequential_9_lstm_18_while_tensorarrayv2read_tensorlistgetitem_sequential_9_lstm_18_tensorarrayunstack_tensorlistfromtensorY
Fsequential_9_lstm_18_while_lstm_cell_18_matmul_readvariableop_resource:	]Ô\
Hsequential_9_lstm_18_while_lstm_cell_18_matmul_1_readvariableop_resource:
ÔV
Gsequential_9_lstm_18_while_lstm_cell_18_biasadd_readvariableop_resource:	Ô¢>sequential_9/lstm_18/while/lstm_cell_18/BiasAdd/ReadVariableOp¢=sequential_9/lstm_18/while/lstm_cell_18/MatMul/ReadVariableOp¢?sequential_9/lstm_18/while/lstm_cell_18/MatMul_1/ReadVariableOpí
Lsequential_9/lstm_18/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ]   2N
Lsequential_9/lstm_18/while/TensorArrayV2Read/TensorListGetItem/element_shapeÑ
>sequential_9/lstm_18/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem}sequential_9_lstm_18_while_tensorarrayv2read_tensorlistgetitem_sequential_9_lstm_18_tensorarrayunstack_tensorlistfromtensor_0&sequential_9_lstm_18_while_placeholderUsequential_9/lstm_18/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]*
element_dtype02@
>sequential_9/lstm_18/while/TensorArrayV2Read/TensorListGetItem
=sequential_9/lstm_18/while/lstm_cell_18/MatMul/ReadVariableOpReadVariableOpHsequential_9_lstm_18_while_lstm_cell_18_matmul_readvariableop_resource_0*
_output_shapes
:	]Ô*
dtype02?
=sequential_9/lstm_18/while/lstm_cell_18/MatMul/ReadVariableOp«
.sequential_9/lstm_18/while/lstm_cell_18/MatMulMatMulEsequential_9/lstm_18/while/TensorArrayV2Read/TensorListGetItem:item:0Esequential_9/lstm_18/while/lstm_cell_18/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÔ20
.sequential_9/lstm_18/while/lstm_cell_18/MatMul
?sequential_9/lstm_18/while/lstm_cell_18/MatMul_1/ReadVariableOpReadVariableOpJsequential_9_lstm_18_while_lstm_cell_18_matmul_1_readvariableop_resource_0* 
_output_shapes
:
Ô*
dtype02A
?sequential_9/lstm_18/while/lstm_cell_18/MatMul_1/ReadVariableOp
0sequential_9/lstm_18/while/lstm_cell_18/MatMul_1MatMul(sequential_9_lstm_18_while_placeholder_2Gsequential_9/lstm_18/while/lstm_cell_18/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÔ22
0sequential_9/lstm_18/while/lstm_cell_18/MatMul_1
+sequential_9/lstm_18/while/lstm_cell_18/addAddV28sequential_9/lstm_18/while/lstm_cell_18/MatMul:product:0:sequential_9/lstm_18/while/lstm_cell_18/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÔ2-
+sequential_9/lstm_18/while/lstm_cell_18/add
>sequential_9/lstm_18/while/lstm_cell_18/BiasAdd/ReadVariableOpReadVariableOpIsequential_9_lstm_18_while_lstm_cell_18_biasadd_readvariableop_resource_0*
_output_shapes	
:Ô*
dtype02@
>sequential_9/lstm_18/while/lstm_cell_18/BiasAdd/ReadVariableOp
/sequential_9/lstm_18/while/lstm_cell_18/BiasAddBiasAdd/sequential_9/lstm_18/while/lstm_cell_18/add:z:0Fsequential_9/lstm_18/while/lstm_cell_18/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÔ21
/sequential_9/lstm_18/while/lstm_cell_18/BiasAdd´
7sequential_9/lstm_18/while/lstm_cell_18/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :29
7sequential_9/lstm_18/while/lstm_cell_18/split/split_dimã
-sequential_9/lstm_18/while/lstm_cell_18/splitSplit@sequential_9/lstm_18/while/lstm_cell_18/split/split_dim:output:08sequential_9/lstm_18/while/lstm_cell_18/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2/
-sequential_9/lstm_18/while/lstm_cell_18/splitØ
/sequential_9/lstm_18/while/lstm_cell_18/SigmoidSigmoid6sequential_9/lstm_18/while/lstm_cell_18/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ21
/sequential_9/lstm_18/while/lstm_cell_18/SigmoidÜ
1sequential_9/lstm_18/while/lstm_cell_18/Sigmoid_1Sigmoid6sequential_9/lstm_18/while/lstm_cell_18/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ23
1sequential_9/lstm_18/while/lstm_cell_18/Sigmoid_1õ
+sequential_9/lstm_18/while/lstm_cell_18/mulMul5sequential_9/lstm_18/while/lstm_cell_18/Sigmoid_1:y:0(sequential_9_lstm_18_while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2-
+sequential_9/lstm_18/while/lstm_cell_18/mulÏ
,sequential_9/lstm_18/while/lstm_cell_18/ReluRelu6sequential_9/lstm_18/while/lstm_cell_18/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2.
,sequential_9/lstm_18/while/lstm_cell_18/Relu
-sequential_9/lstm_18/while/lstm_cell_18/mul_1Mul3sequential_9/lstm_18/while/lstm_cell_18/Sigmoid:y:0:sequential_9/lstm_18/while/lstm_cell_18/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2/
-sequential_9/lstm_18/while/lstm_cell_18/mul_1þ
-sequential_9/lstm_18/while/lstm_cell_18/add_1AddV2/sequential_9/lstm_18/while/lstm_cell_18/mul:z:01sequential_9/lstm_18/while/lstm_cell_18/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2/
-sequential_9/lstm_18/while/lstm_cell_18/add_1Ü
1sequential_9/lstm_18/while/lstm_cell_18/Sigmoid_2Sigmoid6sequential_9/lstm_18/while/lstm_cell_18/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ23
1sequential_9/lstm_18/while/lstm_cell_18/Sigmoid_2Î
.sequential_9/lstm_18/while/lstm_cell_18/Relu_1Relu1sequential_9/lstm_18/while/lstm_cell_18/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ20
.sequential_9/lstm_18/while/lstm_cell_18/Relu_1
-sequential_9/lstm_18/while/lstm_cell_18/mul_2Mul5sequential_9/lstm_18/while/lstm_cell_18/Sigmoid_2:y:0<sequential_9/lstm_18/while/lstm_cell_18/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2/
-sequential_9/lstm_18/while/lstm_cell_18/mul_2É
?sequential_9/lstm_18/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem(sequential_9_lstm_18_while_placeholder_1&sequential_9_lstm_18_while_placeholder1sequential_9/lstm_18/while/lstm_cell_18/mul_2:z:0*
_output_shapes
: *
element_dtype02A
?sequential_9/lstm_18/while/TensorArrayV2Write/TensorListSetItem
 sequential_9/lstm_18/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2"
 sequential_9/lstm_18/while/add/y½
sequential_9/lstm_18/while/addAddV2&sequential_9_lstm_18_while_placeholder)sequential_9/lstm_18/while/add/y:output:0*
T0*
_output_shapes
: 2 
sequential_9/lstm_18/while/add
"sequential_9/lstm_18/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2$
"sequential_9/lstm_18/while/add_1/yß
 sequential_9/lstm_18/while/add_1AddV2Bsequential_9_lstm_18_while_sequential_9_lstm_18_while_loop_counter+sequential_9/lstm_18/while/add_1/y:output:0*
T0*
_output_shapes
: 2"
 sequential_9/lstm_18/while/add_1¿
#sequential_9/lstm_18/while/IdentityIdentity$sequential_9/lstm_18/while/add_1:z:0 ^sequential_9/lstm_18/while/NoOp*
T0*
_output_shapes
: 2%
#sequential_9/lstm_18/while/Identityç
%sequential_9/lstm_18/while/Identity_1IdentityHsequential_9_lstm_18_while_sequential_9_lstm_18_while_maximum_iterations ^sequential_9/lstm_18/while/NoOp*
T0*
_output_shapes
: 2'
%sequential_9/lstm_18/while/Identity_1Á
%sequential_9/lstm_18/while/Identity_2Identity"sequential_9/lstm_18/while/add:z:0 ^sequential_9/lstm_18/while/NoOp*
T0*
_output_shapes
: 2'
%sequential_9/lstm_18/while/Identity_2î
%sequential_9/lstm_18/while/Identity_3IdentityOsequential_9/lstm_18/while/TensorArrayV2Write/TensorListSetItem:output_handle:0 ^sequential_9/lstm_18/while/NoOp*
T0*
_output_shapes
: 2'
%sequential_9/lstm_18/while/Identity_3â
%sequential_9/lstm_18/while/Identity_4Identity1sequential_9/lstm_18/while/lstm_cell_18/mul_2:z:0 ^sequential_9/lstm_18/while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%sequential_9/lstm_18/while/Identity_4â
%sequential_9/lstm_18/while/Identity_5Identity1sequential_9/lstm_18/while/lstm_cell_18/add_1:z:0 ^sequential_9/lstm_18/while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%sequential_9/lstm_18/while/Identity_5Ç
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
%sequential_9_lstm_18_while_identity_5.sequential_9/lstm_18/while/Identity_5:output:0"
Gsequential_9_lstm_18_while_lstm_cell_18_biasadd_readvariableop_resourceIsequential_9_lstm_18_while_lstm_cell_18_biasadd_readvariableop_resource_0"
Hsequential_9_lstm_18_while_lstm_cell_18_matmul_1_readvariableop_resourceJsequential_9_lstm_18_while_lstm_cell_18_matmul_1_readvariableop_resource_0"
Fsequential_9_lstm_18_while_lstm_cell_18_matmul_readvariableop_resourceHsequential_9_lstm_18_while_lstm_cell_18_matmul_readvariableop_resource_0"
?sequential_9_lstm_18_while_sequential_9_lstm_18_strided_slice_1Asequential_9_lstm_18_while_sequential_9_lstm_18_strided_slice_1_0"ü
{sequential_9_lstm_18_while_tensorarrayv2read_tensorlistgetitem_sequential_9_lstm_18_tensorarrayunstack_tensorlistfromtensor}sequential_9_lstm_18_while_tensorarrayv2read_tensorlistgetitem_sequential_9_lstm_18_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2
>sequential_9/lstm_18/while/lstm_cell_18/BiasAdd/ReadVariableOp>sequential_9/lstm_18/while/lstm_cell_18/BiasAdd/ReadVariableOp2~
=sequential_9/lstm_18/while/lstm_cell_18/MatMul/ReadVariableOp=sequential_9/lstm_18/while/lstm_cell_18/MatMul/ReadVariableOp2
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
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
Ô

í
lstm_18_while_cond_32581304,
(lstm_18_while_lstm_18_while_loop_counter2
.lstm_18_while_lstm_18_while_maximum_iterations
lstm_18_while_placeholder
lstm_18_while_placeholder_1
lstm_18_while_placeholder_2
lstm_18_while_placeholder_3.
*lstm_18_while_less_lstm_18_strided_slice_1F
Blstm_18_while_lstm_18_while_cond_32581304___redundant_placeholder0F
Blstm_18_while_lstm_18_while_cond_32581304___redundant_placeholder1F
Blstm_18_while_lstm_18_while_cond_32581304___redundant_placeholder2F
Blstm_18_while_lstm_18_while_cond_32581304___redundant_placeholder3
lstm_18_while_identity

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
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
¬

Ó
/__inference_sequential_9_layer_call_fn_32580348
lstm_18_input
unknown:	]Ô
	unknown_0:
Ô
	unknown_1:	Ô
	unknown_2:
Ü
	unknown_3:
×Ü
	unknown_4:	Ü
	unknown_5:	×
	unknown_6:
identity¢StatefulPartitionedCallÓ
StatefulPartitionedCallStatefulPartitionedCalllstm_18_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
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
J__inference_sequential_9_layer_call_and_return_conditional_losses_325803292
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
_user_specified_namelstm_18_input
åJ
Ô

lstm_18_while_body_32581305,
(lstm_18_while_lstm_18_while_loop_counter2
.lstm_18_while_lstm_18_while_maximum_iterations
lstm_18_while_placeholder
lstm_18_while_placeholder_1
lstm_18_while_placeholder_2
lstm_18_while_placeholder_3+
'lstm_18_while_lstm_18_strided_slice_1_0g
clstm_18_while_tensorarrayv2read_tensorlistgetitem_lstm_18_tensorarrayunstack_tensorlistfromtensor_0N
;lstm_18_while_lstm_cell_18_matmul_readvariableop_resource_0:	]ÔQ
=lstm_18_while_lstm_cell_18_matmul_1_readvariableop_resource_0:
ÔK
<lstm_18_while_lstm_cell_18_biasadd_readvariableop_resource_0:	Ô
lstm_18_while_identity
lstm_18_while_identity_1
lstm_18_while_identity_2
lstm_18_while_identity_3
lstm_18_while_identity_4
lstm_18_while_identity_5)
%lstm_18_while_lstm_18_strided_slice_1e
alstm_18_while_tensorarrayv2read_tensorlistgetitem_lstm_18_tensorarrayunstack_tensorlistfromtensorL
9lstm_18_while_lstm_cell_18_matmul_readvariableop_resource:	]ÔO
;lstm_18_while_lstm_cell_18_matmul_1_readvariableop_resource:
ÔI
:lstm_18_while_lstm_cell_18_biasadd_readvariableop_resource:	Ô¢1lstm_18/while/lstm_cell_18/BiasAdd/ReadVariableOp¢0lstm_18/while/lstm_cell_18/MatMul/ReadVariableOp¢2lstm_18/while/lstm_cell_18/MatMul_1/ReadVariableOpÓ
?lstm_18/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ]   2A
?lstm_18/while/TensorArrayV2Read/TensorListGetItem/element_shape
1lstm_18/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemclstm_18_while_tensorarrayv2read_tensorlistgetitem_lstm_18_tensorarrayunstack_tensorlistfromtensor_0lstm_18_while_placeholderHlstm_18/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]*
element_dtype023
1lstm_18/while/TensorArrayV2Read/TensorListGetItemá
0lstm_18/while/lstm_cell_18/MatMul/ReadVariableOpReadVariableOp;lstm_18_while_lstm_cell_18_matmul_readvariableop_resource_0*
_output_shapes
:	]Ô*
dtype022
0lstm_18/while/lstm_cell_18/MatMul/ReadVariableOp÷
!lstm_18/while/lstm_cell_18/MatMulMatMul8lstm_18/while/TensorArrayV2Read/TensorListGetItem:item:08lstm_18/while/lstm_cell_18/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÔ2#
!lstm_18/while/lstm_cell_18/MatMulè
2lstm_18/while/lstm_cell_18/MatMul_1/ReadVariableOpReadVariableOp=lstm_18_while_lstm_cell_18_matmul_1_readvariableop_resource_0* 
_output_shapes
:
Ô*
dtype024
2lstm_18/while/lstm_cell_18/MatMul_1/ReadVariableOpà
#lstm_18/while/lstm_cell_18/MatMul_1MatMullstm_18_while_placeholder_2:lstm_18/while/lstm_cell_18/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÔ2%
#lstm_18/while/lstm_cell_18/MatMul_1Ø
lstm_18/while/lstm_cell_18/addAddV2+lstm_18/while/lstm_cell_18/MatMul:product:0-lstm_18/while/lstm_cell_18/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÔ2 
lstm_18/while/lstm_cell_18/addà
1lstm_18/while/lstm_cell_18/BiasAdd/ReadVariableOpReadVariableOp<lstm_18_while_lstm_cell_18_biasadd_readvariableop_resource_0*
_output_shapes	
:Ô*
dtype023
1lstm_18/while/lstm_cell_18/BiasAdd/ReadVariableOpå
"lstm_18/while/lstm_cell_18/BiasAddBiasAdd"lstm_18/while/lstm_cell_18/add:z:09lstm_18/while/lstm_cell_18/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÔ2$
"lstm_18/while/lstm_cell_18/BiasAdd
*lstm_18/while/lstm_cell_18/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2,
*lstm_18/while/lstm_cell_18/split/split_dim¯
 lstm_18/while/lstm_cell_18/splitSplit3lstm_18/while/lstm_cell_18/split/split_dim:output:0+lstm_18/while/lstm_cell_18/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2"
 lstm_18/while/lstm_cell_18/split±
"lstm_18/while/lstm_cell_18/SigmoidSigmoid)lstm_18/while/lstm_cell_18/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2$
"lstm_18/while/lstm_cell_18/Sigmoidµ
$lstm_18/while/lstm_cell_18/Sigmoid_1Sigmoid)lstm_18/while/lstm_cell_18/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$lstm_18/while/lstm_cell_18/Sigmoid_1Á
lstm_18/while/lstm_cell_18/mulMul(lstm_18/while/lstm_cell_18/Sigmoid_1:y:0lstm_18_while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
lstm_18/while/lstm_cell_18/mul¨
lstm_18/while/lstm_cell_18/ReluRelu)lstm_18/while/lstm_cell_18/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
lstm_18/while/lstm_cell_18/ReluÕ
 lstm_18/while/lstm_cell_18/mul_1Mul&lstm_18/while/lstm_cell_18/Sigmoid:y:0-lstm_18/while/lstm_cell_18/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 lstm_18/while/lstm_cell_18/mul_1Ê
 lstm_18/while/lstm_cell_18/add_1AddV2"lstm_18/while/lstm_cell_18/mul:z:0$lstm_18/while/lstm_cell_18/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 lstm_18/while/lstm_cell_18/add_1µ
$lstm_18/while/lstm_cell_18/Sigmoid_2Sigmoid)lstm_18/while/lstm_cell_18/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$lstm_18/while/lstm_cell_18/Sigmoid_2§
!lstm_18/while/lstm_cell_18/Relu_1Relu$lstm_18/while/lstm_cell_18/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!lstm_18/while/lstm_cell_18/Relu_1Ù
 lstm_18/while/lstm_cell_18/mul_2Mul(lstm_18/while/lstm_cell_18/Sigmoid_2:y:0/lstm_18/while/lstm_cell_18/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 lstm_18/while/lstm_cell_18/mul_2
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
lstm_18/while/add/y
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
lstm_18/while/add_1/y
lstm_18/while/add_1AddV2(lstm_18_while_lstm_18_while_loop_counterlstm_18/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lstm_18/while/add_1
lstm_18/while/IdentityIdentitylstm_18/while/add_1:z:0^lstm_18/while/NoOp*
T0*
_output_shapes
: 2
lstm_18/while/Identity¦
lstm_18/while/Identity_1Identity.lstm_18_while_lstm_18_while_maximum_iterations^lstm_18/while/NoOp*
T0*
_output_shapes
: 2
lstm_18/while/Identity_1
lstm_18/while/Identity_2Identitylstm_18/while/add:z:0^lstm_18/while/NoOp*
T0*
_output_shapes
: 2
lstm_18/while/Identity_2º
lstm_18/while/Identity_3IdentityBlstm_18/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_18/while/NoOp*
T0*
_output_shapes
: 2
lstm_18/while/Identity_3®
lstm_18/while/Identity_4Identity$lstm_18/while/lstm_cell_18/mul_2:z:0^lstm_18/while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_18/while/Identity_4®
lstm_18/while/Identity_5Identity$lstm_18/while/lstm_cell_18/add_1:z:0^lstm_18/while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_18/while/Identity_5
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
9lstm_18_while_lstm_cell_18_matmul_readvariableop_resource;lstm_18_while_lstm_cell_18_matmul_readvariableop_resource_0"È
alstm_18_while_tensorarrayv2read_tensorlistgetitem_lstm_18_tensorarrayunstack_tensorlistfromtensorclstm_18_while_tensorarrayv2read_tensorlistgetitem_lstm_18_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2f
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
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
´?
Ö
while_body_32582514
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
3while_lstm_cell_19_matmul_readvariableop_resource_0:
ÜI
5while_lstm_cell_19_matmul_1_readvariableop_resource_0:
×ÜC
4while_lstm_cell_19_biasadd_readvariableop_resource_0:	Ü
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
1while_lstm_cell_19_matmul_readvariableop_resource:
ÜG
3while_lstm_cell_19_matmul_1_readvariableop_resource:
×ÜA
2while_lstm_cell_19_biasadd_readvariableop_resource:	Ü¢)while/lstm_cell_19/BiasAdd/ReadVariableOp¢(while/lstm_cell_19/MatMul/ReadVariableOp¢*while/lstm_cell_19/MatMul_1/ReadVariableOpÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ  29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÔ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemÊ
(while/lstm_cell_19/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_19_matmul_readvariableop_resource_0* 
_output_shapes
:
Ü*
dtype02*
(while/lstm_cell_19/MatMul/ReadVariableOp×
while/lstm_cell_19/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_19/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2
while/lstm_cell_19/MatMulÐ
*while/lstm_cell_19/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_19_matmul_1_readvariableop_resource_0* 
_output_shapes
:
×Ü*
dtype02,
*while/lstm_cell_19/MatMul_1/ReadVariableOpÀ
while/lstm_cell_19/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_19/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2
while/lstm_cell_19/MatMul_1¸
while/lstm_cell_19/addAddV2#while/lstm_cell_19/MatMul:product:0%while/lstm_cell_19/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2
while/lstm_cell_19/addÈ
)while/lstm_cell_19/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_19_biasadd_readvariableop_resource_0*
_output_shapes	
:Ü*
dtype02+
)while/lstm_cell_19/BiasAdd/ReadVariableOpÅ
while/lstm_cell_19/BiasAddBiasAddwhile/lstm_cell_19/add:z:01while/lstm_cell_19/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2
while/lstm_cell_19/BiasAdd
"while/lstm_cell_19/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_19/split/split_dim
while/lstm_cell_19/splitSplit+while/lstm_cell_19/split/split_dim:output:0#while/lstm_cell_19/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ×:ÿÿÿÿÿÿÿÿÿ×:ÿÿÿÿÿÿÿÿÿ×:ÿÿÿÿÿÿÿÿÿ×*
	num_split2
while/lstm_cell_19/split
while/lstm_cell_19/SigmoidSigmoid!while/lstm_cell_19/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2
while/lstm_cell_19/Sigmoid
while/lstm_cell_19/Sigmoid_1Sigmoid!while/lstm_cell_19/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2
while/lstm_cell_19/Sigmoid_1¡
while/lstm_cell_19/mulMul while/lstm_cell_19/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2
while/lstm_cell_19/mul
while/lstm_cell_19/ReluRelu!while/lstm_cell_19/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2
while/lstm_cell_19/Reluµ
while/lstm_cell_19/mul_1Mulwhile/lstm_cell_19/Sigmoid:y:0%while/lstm_cell_19/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2
while/lstm_cell_19/mul_1ª
while/lstm_cell_19/add_1AddV2while/lstm_cell_19/mul:z:0while/lstm_cell_19/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2
while/lstm_cell_19/add_1
while/lstm_cell_19/Sigmoid_2Sigmoid!while/lstm_cell_19/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2
while/lstm_cell_19/Sigmoid_2
while/lstm_cell_19/Relu_1Reluwhile/lstm_cell_19/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2
while/lstm_cell_19/Relu_1¹
while/lstm_cell_19/mul_2Mul while/lstm_cell_19/Sigmoid_2:y:0'while/lstm_cell_19/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2
while/lstm_cell_19/mul_2à
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
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_19/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_19/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2
while/Identity_5Þ

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
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ×:ÿÿÿÿÿÿÿÿÿ×: : : : : 2V
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
:ÿÿÿÿÿÿÿÿÿ×:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×:

_output_shapes
: :

_output_shapes
: 
¬

Ó
/__inference_sequential_9_layer_call_fn_32580838
lstm_18_input
unknown:	]Ô
	unknown_0:
Ô
	unknown_1:	Ô
	unknown_2:
Ü
	unknown_3:
×Ü
	unknown_4:	Ü
	unknown_5:	×
	unknown_6:
identity¢StatefulPartitionedCallÓ
StatefulPartitionedCallStatefulPartitionedCalllstm_18_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
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
J__inference_sequential_9_layer_call_and_return_conditional_losses_325807982
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
_user_specified_namelstm_18_input
\

E__inference_lstm_19_layer_call_and_return_conditional_losses_32582900

inputs?
+lstm_cell_19_matmul_readvariableop_resource:
ÜA
-lstm_cell_19_matmul_1_readvariableop_resource:
×Ü;
,lstm_cell_19_biasadd_readvariableop_resource:	Ü
identity¢#lstm_cell_19/BiasAdd/ReadVariableOp¢"lstm_cell_19/MatMul/ReadVariableOp¢$lstm_cell_19/MatMul_1/ReadVariableOp¢whileD
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
B :×2
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
B :×2
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
:ÿÿÿÿÿÿÿÿÿ×2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :×2
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
B :×2
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
:ÿÿÿÿÿÿÿÿÿ×2	
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
:ÿÿÿÿÿÿÿÿÿ2
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
valueB"ÿÿÿÿ  27
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
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
strided_slice_2¶
"lstm_cell_19/MatMul/ReadVariableOpReadVariableOp+lstm_cell_19_matmul_readvariableop_resource* 
_output_shapes
:
Ü*
dtype02$
"lstm_cell_19/MatMul/ReadVariableOp­
lstm_cell_19/MatMulMatMulstrided_slice_2:output:0*lstm_cell_19/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2
lstm_cell_19/MatMul¼
$lstm_cell_19/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_19_matmul_1_readvariableop_resource* 
_output_shapes
:
×Ü*
dtype02&
$lstm_cell_19/MatMul_1/ReadVariableOp©
lstm_cell_19/MatMul_1MatMulzeros:output:0,lstm_cell_19/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2
lstm_cell_19/MatMul_1 
lstm_cell_19/addAddV2lstm_cell_19/MatMul:product:0lstm_cell_19/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2
lstm_cell_19/add´
#lstm_cell_19/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_19_biasadd_readvariableop_resource*
_output_shapes	
:Ü*
dtype02%
#lstm_cell_19/BiasAdd/ReadVariableOp­
lstm_cell_19/BiasAddBiasAddlstm_cell_19/add:z:0+lstm_cell_19/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2
lstm_cell_19/BiasAdd~
lstm_cell_19/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_19/split/split_dim÷
lstm_cell_19/splitSplit%lstm_cell_19/split/split_dim:output:0lstm_cell_19/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ×:ÿÿÿÿÿÿÿÿÿ×:ÿÿÿÿÿÿÿÿÿ×:ÿÿÿÿÿÿÿÿÿ×*
	num_split2
lstm_cell_19/split
lstm_cell_19/SigmoidSigmoidlstm_cell_19/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2
lstm_cell_19/Sigmoid
lstm_cell_19/Sigmoid_1Sigmoidlstm_cell_19/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2
lstm_cell_19/Sigmoid_1
lstm_cell_19/mulMullstm_cell_19/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2
lstm_cell_19/mul~
lstm_cell_19/ReluRelulstm_cell_19/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2
lstm_cell_19/Relu
lstm_cell_19/mul_1Mullstm_cell_19/Sigmoid:y:0lstm_cell_19/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2
lstm_cell_19/mul_1
lstm_cell_19/add_1AddV2lstm_cell_19/mul:z:0lstm_cell_19/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2
lstm_cell_19/add_1
lstm_cell_19/Sigmoid_2Sigmoidlstm_cell_19/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2
lstm_cell_19/Sigmoid_2}
lstm_cell_19/Relu_1Relulstm_cell_19/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2
lstm_cell_19/Relu_1¡
lstm_cell_19/mul_2Mullstm_cell_19/Sigmoid_2:y:0!lstm_cell_19/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2
lstm_cell_19/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ×   2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_19_matmul_readvariableop_resource-lstm_cell_19_matmul_1_readvariableop_resource,lstm_cell_19_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ×:ÿÿÿÿÿÿÿÿÿ×: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_32582816*
condR
while_cond_32582815*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ×:ÿÿÿÿÿÿÿÿÿ×: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ×   22
0TensorArrayV2Stack/TensorListStack/element_shapeé
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×*
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
:ÿÿÿÿÿÿÿÿÿ×*
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
:ÿÿÿÿÿÿÿÿÿ×2
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
:ÿÿÿÿÿÿÿÿÿ×2

IdentityÈ
NoOpNoOp$^lstm_cell_19/BiasAdd/ReadVariableOp#^lstm_cell_19/MatMul/ReadVariableOp%^lstm_cell_19/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:ÿÿÿÿÿÿÿÿÿ: : : 2J
#lstm_cell_19/BiasAdd/ReadVariableOp#lstm_cell_19/BiasAdd/ReadVariableOp2H
"lstm_cell_19/MatMul/ReadVariableOp"lstm_cell_19/MatMul/ReadVariableOp2L
$lstm_cell_19/MatMul_1/ReadVariableOp$lstm_cell_19/MatMul_1/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


J__inference_lstm_cell_18_layer_call_and_return_conditional_losses_32578769

inputs

states
states_11
matmul_readvariableop_resource:	]Ô4
 matmul_1_readvariableop_resource:
Ô.
biasadd_readvariableop_resource:	Ô
identity

identity_1

identity_2¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	]Ô*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÔ2
MatMul
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
Ô*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÔ2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÔ2
add
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:Ô*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÔ2	
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
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2
split`
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
Sigmoidd
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	Sigmoid_1]
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mulW
ReluRelusplit:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relui
mul_1MulSigmoid:y:0Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mul_1^
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
add_1d
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	Sigmoid_2V
Relu_1Relu	add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu_1m
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mul_2e
IdentityIdentity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityi

Identity_1Identity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1i

Identity_2Identity	add_1:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

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
A:ÿÿÿÿÿÿÿÿÿ]:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
 
_user_specified_nameinputs:PL
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_namestates:PL
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_namestates
&
ó
while_body_32578783
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_00
while_lstm_cell_18_32578807_0:	]Ô1
while_lstm_cell_18_32578809_0:
Ô,
while_lstm_cell_18_32578811_0:	Ô
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor.
while_lstm_cell_18_32578807:	]Ô/
while_lstm_cell_18_32578809:
Ô*
while_lstm_cell_18_32578811:	Ô¢*while/lstm_cell_18/StatefulPartitionedCallÃ
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
*while/lstm_cell_18/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_18_32578807_0while_lstm_cell_18_32578809_0while_lstm_cell_18_32578811_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_lstm_cell_18_layer_call_and_return_conditional_losses_325787692,
*while/lstm_cell_18/StatefulPartitionedCall÷
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
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3¥
while/Identity_4Identity3while/lstm_cell_18/StatefulPartitionedCall:output:1^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/Identity_4¥
while/Identity_5Identity3while/lstm_cell_18/StatefulPartitionedCall:output:2^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/Identity_5

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
while_lstm_cell_18_32578807while_lstm_cell_18_32578807_0"<
while_lstm_cell_18_32578809while_lstm_cell_18_32578809_0"<
while_lstm_cell_18_32578811while_lstm_cell_18_32578811_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2X
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
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
°?
Ô
while_body_32581839
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_18_matmul_readvariableop_resource_0:	]ÔI
5while_lstm_cell_18_matmul_1_readvariableop_resource_0:
ÔC
4while_lstm_cell_18_biasadd_readvariableop_resource_0:	Ô
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_18_matmul_readvariableop_resource:	]ÔG
3while_lstm_cell_18_matmul_1_readvariableop_resource:
ÔA
2while_lstm_cell_18_biasadd_readvariableop_resource:	Ô¢)while/lstm_cell_18/BiasAdd/ReadVariableOp¢(while/lstm_cell_18/MatMul/ReadVariableOp¢*while/lstm_cell_18/MatMul_1/ReadVariableOpÃ
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
(while/lstm_cell_18/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_18_matmul_readvariableop_resource_0*
_output_shapes
:	]Ô*
dtype02*
(while/lstm_cell_18/MatMul/ReadVariableOp×
while/lstm_cell_18/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_18/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÔ2
while/lstm_cell_18/MatMulÐ
*while/lstm_cell_18/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_18_matmul_1_readvariableop_resource_0* 
_output_shapes
:
Ô*
dtype02,
*while/lstm_cell_18/MatMul_1/ReadVariableOpÀ
while/lstm_cell_18/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_18/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÔ2
while/lstm_cell_18/MatMul_1¸
while/lstm_cell_18/addAddV2#while/lstm_cell_18/MatMul:product:0%while/lstm_cell_18/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÔ2
while/lstm_cell_18/addÈ
)while/lstm_cell_18/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_18_biasadd_readvariableop_resource_0*
_output_shapes	
:Ô*
dtype02+
)while/lstm_cell_18/BiasAdd/ReadVariableOpÅ
while/lstm_cell_18/BiasAddBiasAddwhile/lstm_cell_18/add:z:01while/lstm_cell_18/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÔ2
while/lstm_cell_18/BiasAdd
"while/lstm_cell_18/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_18/split/split_dim
while/lstm_cell_18/splitSplit+while/lstm_cell_18/split/split_dim:output:0#while/lstm_cell_18/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2
while/lstm_cell_18/split
while/lstm_cell_18/SigmoidSigmoid!while/lstm_cell_18/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_18/Sigmoid
while/lstm_cell_18/Sigmoid_1Sigmoid!while/lstm_cell_18/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_18/Sigmoid_1¡
while/lstm_cell_18/mulMul while/lstm_cell_18/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_18/mul
while/lstm_cell_18/ReluRelu!while/lstm_cell_18/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_18/Reluµ
while/lstm_cell_18/mul_1Mulwhile/lstm_cell_18/Sigmoid:y:0%while/lstm_cell_18/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_18/mul_1ª
while/lstm_cell_18/add_1AddV2while/lstm_cell_18/mul:z:0while/lstm_cell_18/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_18/add_1
while/lstm_cell_18/Sigmoid_2Sigmoid!while/lstm_cell_18/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_18/Sigmoid_2
while/lstm_cell_18/Relu_1Reluwhile/lstm_cell_18/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_18/Relu_1¹
while/lstm_cell_18/mul_2Mul while/lstm_cell_18/Sigmoid_2:y:0'while/lstm_cell_18/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_18/mul_2à
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
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_18/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_18/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/Identity_5Þ

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
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2V
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
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
Ê
ú
/__inference_lstm_cell_19_layer_call_fn_32583207

inputs
states_0
states_1
unknown:
Ü
	unknown_0:
×Ü
	unknown_1:	Ü
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
<:ÿÿÿÿÿÿÿÿÿ×:ÿÿÿÿÿÿÿÿÿ×:ÿÿÿÿÿÿÿÿÿ×*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_lstm_cell_19_layer_call_and_return_conditional_losses_325795452
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2

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
B:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ×:ÿÿÿÿÿÿÿÿÿ×: : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×
"
_user_specified_name
states/0:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×
"
_user_specified_name
states/1
Ê7
Ý
$__inference__traced_restore_32583312
file_prefix2
assignvariableop_dense_9_kernel:	×-
assignvariableop_1_dense_9_bias:A
.assignvariableop_2_lstm_18_lstm_cell_18_kernel:	]ÔL
8assignvariableop_3_lstm_18_lstm_cell_18_recurrent_kernel:
Ô;
,assignvariableop_4_lstm_18_lstm_cell_18_bias:	ÔB
.assignvariableop_5_lstm_19_lstm_cell_19_kernel:
ÜL
8assignvariableop_6_lstm_19_lstm_cell_19_recurrent_kernel:
×Ü;
,assignvariableop_7_lstm_19_lstm_cell_19_bias:	Ü"
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
AssignVariableOpAssignVariableOpassignvariableop_dense_9_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1¤
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_9_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2³
AssignVariableOp_2AssignVariableOp.assignvariableop_2_lstm_18_lstm_cell_18_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3½
AssignVariableOp_3AssignVariableOp8assignvariableop_3_lstm_18_lstm_cell_18_recurrent_kernelIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4±
AssignVariableOp_4AssignVariableOp,assignvariableop_4_lstm_18_lstm_cell_18_biasIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5³
AssignVariableOp_5AssignVariableOp.assignvariableop_5_lstm_19_lstm_cell_19_kernelIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6½
AssignVariableOp_6AssignVariableOp8assignvariableop_6_lstm_19_lstm_cell_19_recurrent_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7±
AssignVariableOp_7AssignVariableOp,assignvariableop_7_lstm_19_lstm_cell_19_biasIdentity_7:output:0"/device:CPU:0*
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


J__inference_lstm_cell_19_layer_call_and_return_conditional_losses_32579545

inputs

states
states_12
matmul_readvariableop_resource:
Ü4
 matmul_1_readvariableop_resource:
×Ü.
biasadd_readvariableop_resource:	Ü
identity

identity_1

identity_2¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
Ü*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2
MatMul
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
×Ü*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2
add
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:Ü*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2	
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
P:ÿÿÿÿÿÿÿÿÿ×:ÿÿÿÿÿÿÿÿÿ×:ÿÿÿÿÿÿÿÿÿ×:ÿÿÿÿÿÿÿÿÿ×*
	num_split2
split`
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2	
Sigmoidd
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2
	Sigmoid_1]
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2
mulW
ReluRelusplit:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2
Relui
mul_1MulSigmoid:y:0Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2
mul_1^
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2
add_1d
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2
	Sigmoid_2V
Relu_1Relu	add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2
Relu_1m
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2
mul_2e
IdentityIdentity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2

Identityi

Identity_1Identity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2

Identity_1i

Identity_2Identity	add_1:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2

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
B:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ×:ÿÿÿÿÿÿÿÿÿ×: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:PL
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×
 
_user_specified_namestates:PL
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×
 
_user_specified_namestates
´?
Ö
while_body_32580192
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
3while_lstm_cell_19_matmul_readvariableop_resource_0:
ÜI
5while_lstm_cell_19_matmul_1_readvariableop_resource_0:
×ÜC
4while_lstm_cell_19_biasadd_readvariableop_resource_0:	Ü
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
1while_lstm_cell_19_matmul_readvariableop_resource:
ÜG
3while_lstm_cell_19_matmul_1_readvariableop_resource:
×ÜA
2while_lstm_cell_19_biasadd_readvariableop_resource:	Ü¢)while/lstm_cell_19/BiasAdd/ReadVariableOp¢(while/lstm_cell_19/MatMul/ReadVariableOp¢*while/lstm_cell_19/MatMul_1/ReadVariableOpÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ  29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÔ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemÊ
(while/lstm_cell_19/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_19_matmul_readvariableop_resource_0* 
_output_shapes
:
Ü*
dtype02*
(while/lstm_cell_19/MatMul/ReadVariableOp×
while/lstm_cell_19/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_19/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2
while/lstm_cell_19/MatMulÐ
*while/lstm_cell_19/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_19_matmul_1_readvariableop_resource_0* 
_output_shapes
:
×Ü*
dtype02,
*while/lstm_cell_19/MatMul_1/ReadVariableOpÀ
while/lstm_cell_19/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_19/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2
while/lstm_cell_19/MatMul_1¸
while/lstm_cell_19/addAddV2#while/lstm_cell_19/MatMul:product:0%while/lstm_cell_19/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2
while/lstm_cell_19/addÈ
)while/lstm_cell_19/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_19_biasadd_readvariableop_resource_0*
_output_shapes	
:Ü*
dtype02+
)while/lstm_cell_19/BiasAdd/ReadVariableOpÅ
while/lstm_cell_19/BiasAddBiasAddwhile/lstm_cell_19/add:z:01while/lstm_cell_19/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2
while/lstm_cell_19/BiasAdd
"while/lstm_cell_19/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_19/split/split_dim
while/lstm_cell_19/splitSplit+while/lstm_cell_19/split/split_dim:output:0#while/lstm_cell_19/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ×:ÿÿÿÿÿÿÿÿÿ×:ÿÿÿÿÿÿÿÿÿ×:ÿÿÿÿÿÿÿÿÿ×*
	num_split2
while/lstm_cell_19/split
while/lstm_cell_19/SigmoidSigmoid!while/lstm_cell_19/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2
while/lstm_cell_19/Sigmoid
while/lstm_cell_19/Sigmoid_1Sigmoid!while/lstm_cell_19/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2
while/lstm_cell_19/Sigmoid_1¡
while/lstm_cell_19/mulMul while/lstm_cell_19/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2
while/lstm_cell_19/mul
while/lstm_cell_19/ReluRelu!while/lstm_cell_19/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2
while/lstm_cell_19/Reluµ
while/lstm_cell_19/mul_1Mulwhile/lstm_cell_19/Sigmoid:y:0%while/lstm_cell_19/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2
while/lstm_cell_19/mul_1ª
while/lstm_cell_19/add_1AddV2while/lstm_cell_19/mul:z:0while/lstm_cell_19/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2
while/lstm_cell_19/add_1
while/lstm_cell_19/Sigmoid_2Sigmoid!while/lstm_cell_19/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2
while/lstm_cell_19/Sigmoid_2
while/lstm_cell_19/Relu_1Reluwhile/lstm_cell_19/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2
while/lstm_cell_19/Relu_1¹
while/lstm_cell_19/mul_2Mul while/lstm_cell_19/Sigmoid_2:y:0'while/lstm_cell_19/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2
while/lstm_cell_19/mul_2à
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
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_19/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_19/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2
while/Identity_5Þ

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
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ×:ÿÿÿÿÿÿÿÿÿ×: : : : : 2V
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
:ÿÿÿÿÿÿÿÿÿ×:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×:

_output_shapes
: :

_output_shapes
: 
Ê
ú
/__inference_lstm_cell_19_layer_call_fn_32583190

inputs
states_0
states_1
unknown:
Ü
	unknown_0:
×Ü
	unknown_1:	Ü
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
<:ÿÿÿÿÿÿÿÿÿ×:ÿÿÿÿÿÿÿÿÿ×:ÿÿÿÿÿÿÿÿÿ×*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_lstm_cell_19_layer_call_and_return_conditional_losses_325793992
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2

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
B:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ×:ÿÿÿÿÿÿÿÿÿ×: : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×
"
_user_specified_name
states/0:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×
"
_user_specified_name
states/1
Ç
ù
/__inference_lstm_cell_18_layer_call_fn_32583109

inputs
states_0
states_1
unknown:	]Ô
	unknown_0:
Ô
	unknown_1:	Ô
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
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_lstm_cell_18_layer_call_and_return_conditional_losses_325789152
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

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
A:ÿÿÿÿÿÿÿÿÿ]:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
 
_user_specified_nameinputs:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
states/0:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
states/1
ã
»
*__inference_lstm_19_layer_call_fn_32582922
inputs_0
unknown:
Ü
	unknown_0:
×Ü
	unknown_1:	Ü
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ×*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_lstm_19_layer_call_and_return_conditional_losses_325796922
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ×2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0
ã
Í
while_cond_32580191
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_32580191___redundant_placeholder06
2while_while_cond_32580191___redundant_placeholder16
2while_while_cond_32580191___redundant_placeholder26
2while_while_cond_32580191___redundant_placeholder3
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
B: : : : :ÿÿÿÿÿÿÿÿÿ×:ÿÿÿÿÿÿÿÿÿ×: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×:

_output_shapes
: :

_output_shapes
:
¨
·
J__inference_sequential_9_layer_call_and_return_conditional_losses_32580798

inputs#
lstm_18_32580776:	]Ô$
lstm_18_32580778:
Ô
lstm_18_32580780:	Ô$
lstm_19_32580784:
Ü$
lstm_19_32580786:
×Ü
lstm_19_32580788:	Ü#
dense_9_32580792:	×
dense_9_32580794:
identity¢dense_9/StatefulPartitionedCall¢"dropout_18/StatefulPartitionedCall¢"dropout_19/StatefulPartitionedCall¢lstm_18/StatefulPartitionedCall¢lstm_19/StatefulPartitionedCall®
lstm_18/StatefulPartitionedCallStatefulPartitionedCallinputslstm_18_32580776lstm_18_32580778lstm_18_32580780*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_lstm_18_layer_call_and_return_conditional_losses_325807412!
lstm_18/StatefulPartitionedCall
"dropout_18/StatefulPartitionedCallStatefulPartitionedCall(lstm_18/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_dropout_18_layer_call_and_return_conditional_losses_325805742$
"dropout_18/StatefulPartitionedCallÓ
lstm_19/StatefulPartitionedCallStatefulPartitionedCall+dropout_18/StatefulPartitionedCall:output:0lstm_19_32580784lstm_19_32580786lstm_19_32580788*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_lstm_19_layer_call_and_return_conditional_losses_325805452!
lstm_19/StatefulPartitionedCallÀ
"dropout_19/StatefulPartitionedCallStatefulPartitionedCall(lstm_19/StatefulPartitionedCall:output:0#^dropout_18/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_dropout_19_layer_call_and_return_conditional_losses_325803782$
"dropout_19/StatefulPartitionedCall¾
dense_9/StatefulPartitionedCallStatefulPartitionedCall+dropout_19/StatefulPartitionedCall:output:0dense_9_32580792dense_9_32580794*
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
E__inference_dense_9_layer_call_and_return_conditional_losses_325803222!
dense_9/StatefulPartitionedCall
IdentityIdentity(dense_9/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityþ
NoOpNoOp ^dense_9/StatefulPartitionedCall#^dropout_18/StatefulPartitionedCall#^dropout_19/StatefulPartitionedCall ^lstm_18/StatefulPartitionedCall ^lstm_19/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ]: : : : : : : : 2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall2H
"dropout_18/StatefulPartitionedCall"dropout_18/StatefulPartitionedCall2H
"dropout_19/StatefulPartitionedCall"dropout_19/StatefulPartitionedCall2B
lstm_18/StatefulPartitionedCalllstm_18/StatefulPartitionedCall2B
lstm_19/StatefulPartitionedCalllstm_19/StatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
 
_user_specified_nameinputs


J__inference_lstm_cell_19_layer_call_and_return_conditional_losses_32579399

inputs

states
states_12
matmul_readvariableop_resource:
Ü4
 matmul_1_readvariableop_resource:
×Ü.
biasadd_readvariableop_resource:	Ü
identity

identity_1

identity_2¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
Ü*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2
MatMul
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
×Ü*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2
add
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:Ü*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2	
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
P:ÿÿÿÿÿÿÿÿÿ×:ÿÿÿÿÿÿÿÿÿ×:ÿÿÿÿÿÿÿÿÿ×:ÿÿÿÿÿÿÿÿÿ×*
	num_split2
split`
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2	
Sigmoidd
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2
	Sigmoid_1]
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2
mulW
ReluRelusplit:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2
Relui
mul_1MulSigmoid:y:0Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2
mul_1^
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2
add_1d
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2
	Sigmoid_2V
Relu_1Relu	add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2
Relu_1m
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2
mul_2e
IdentityIdentity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2

Identityi

Identity_1Identity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2

Identity_1i

Identity_2Identity	add_1:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2

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
B:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ×:ÿÿÿÿÿÿÿÿÿ×: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:PL
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×
 
_user_specified_namestates:PL
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×
 
_user_specified_namestates
ã
Í
while_cond_32579622
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_32579622___redundant_placeholder06
2while_while_cond_32579622___redundant_placeholder16
2while_while_cond_32579622___redundant_placeholder26
2while_while_cond_32579622___redundant_placeholder3
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
B: : : : :ÿÿÿÿÿÿÿÿÿ×:ÿÿÿÿÿÿÿÿÿ×: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×:

_output_shapes
: :

_output_shapes
:

f
H__inference_dropout_18_layer_call_and_return_conditional_losses_32580124

inputs

identity_1_
IdentityIdentityinputs*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityn

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
\

E__inference_lstm_19_layer_call_and_return_conditional_losses_32580276

inputs?
+lstm_cell_19_matmul_readvariableop_resource:
ÜA
-lstm_cell_19_matmul_1_readvariableop_resource:
×Ü;
,lstm_cell_19_biasadd_readvariableop_resource:	Ü
identity¢#lstm_cell_19/BiasAdd/ReadVariableOp¢"lstm_cell_19/MatMul/ReadVariableOp¢$lstm_cell_19/MatMul_1/ReadVariableOp¢whileD
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
B :×2
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
B :×2
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
:ÿÿÿÿÿÿÿÿÿ×2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :×2
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
B :×2
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
:ÿÿÿÿÿÿÿÿÿ×2	
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
:ÿÿÿÿÿÿÿÿÿ2
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
valueB"ÿÿÿÿ  27
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
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
strided_slice_2¶
"lstm_cell_19/MatMul/ReadVariableOpReadVariableOp+lstm_cell_19_matmul_readvariableop_resource* 
_output_shapes
:
Ü*
dtype02$
"lstm_cell_19/MatMul/ReadVariableOp­
lstm_cell_19/MatMulMatMulstrided_slice_2:output:0*lstm_cell_19/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2
lstm_cell_19/MatMul¼
$lstm_cell_19/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_19_matmul_1_readvariableop_resource* 
_output_shapes
:
×Ü*
dtype02&
$lstm_cell_19/MatMul_1/ReadVariableOp©
lstm_cell_19/MatMul_1MatMulzeros:output:0,lstm_cell_19/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2
lstm_cell_19/MatMul_1 
lstm_cell_19/addAddV2lstm_cell_19/MatMul:product:0lstm_cell_19/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2
lstm_cell_19/add´
#lstm_cell_19/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_19_biasadd_readvariableop_resource*
_output_shapes	
:Ü*
dtype02%
#lstm_cell_19/BiasAdd/ReadVariableOp­
lstm_cell_19/BiasAddBiasAddlstm_cell_19/add:z:0+lstm_cell_19/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2
lstm_cell_19/BiasAdd~
lstm_cell_19/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_19/split/split_dim÷
lstm_cell_19/splitSplit%lstm_cell_19/split/split_dim:output:0lstm_cell_19/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ×:ÿÿÿÿÿÿÿÿÿ×:ÿÿÿÿÿÿÿÿÿ×:ÿÿÿÿÿÿÿÿÿ×*
	num_split2
lstm_cell_19/split
lstm_cell_19/SigmoidSigmoidlstm_cell_19/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2
lstm_cell_19/Sigmoid
lstm_cell_19/Sigmoid_1Sigmoidlstm_cell_19/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2
lstm_cell_19/Sigmoid_1
lstm_cell_19/mulMullstm_cell_19/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2
lstm_cell_19/mul~
lstm_cell_19/ReluRelulstm_cell_19/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2
lstm_cell_19/Relu
lstm_cell_19/mul_1Mullstm_cell_19/Sigmoid:y:0lstm_cell_19/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2
lstm_cell_19/mul_1
lstm_cell_19/add_1AddV2lstm_cell_19/mul:z:0lstm_cell_19/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2
lstm_cell_19/add_1
lstm_cell_19/Sigmoid_2Sigmoidlstm_cell_19/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2
lstm_cell_19/Sigmoid_2}
lstm_cell_19/Relu_1Relulstm_cell_19/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2
lstm_cell_19/Relu_1¡
lstm_cell_19/mul_2Mullstm_cell_19/Sigmoid_2:y:0!lstm_cell_19/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2
lstm_cell_19/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ×   2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_19_matmul_readvariableop_resource-lstm_cell_19_matmul_1_readvariableop_resource,lstm_cell_19_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ×:ÿÿÿÿÿÿÿÿÿ×: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_32580192*
condR
while_cond_32580191*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ×:ÿÿÿÿÿÿÿÿÿ×: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ×   22
0TensorArrayV2Stack/TensorListStack/element_shapeé
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×*
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
:ÿÿÿÿÿÿÿÿÿ×*
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
:ÿÿÿÿÿÿÿÿÿ×2
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
:ÿÿÿÿÿÿÿÿÿ×2

IdentityÈ
NoOpNoOp$^lstm_cell_19/BiasAdd/ReadVariableOp#^lstm_cell_19/MatMul/ReadVariableOp%^lstm_cell_19/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:ÿÿÿÿÿÿÿÿÿ: : : 2J
#lstm_cell_19/BiasAdd/ReadVariableOp#lstm_cell_19/BiasAdd/ReadVariableOp2H
"lstm_cell_19/MatMul/ReadVariableOp"lstm_cell_19/MatMul/ReadVariableOp2L
$lstm_cell_19/MatMul_1/ReadVariableOp$lstm_cell_19/MatMul_1/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
°?
Ô
while_body_32582141
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_18_matmul_readvariableop_resource_0:	]ÔI
5while_lstm_cell_18_matmul_1_readvariableop_resource_0:
ÔC
4while_lstm_cell_18_biasadd_readvariableop_resource_0:	Ô
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_18_matmul_readvariableop_resource:	]ÔG
3while_lstm_cell_18_matmul_1_readvariableop_resource:
ÔA
2while_lstm_cell_18_biasadd_readvariableop_resource:	Ô¢)while/lstm_cell_18/BiasAdd/ReadVariableOp¢(while/lstm_cell_18/MatMul/ReadVariableOp¢*while/lstm_cell_18/MatMul_1/ReadVariableOpÃ
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
(while/lstm_cell_18/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_18_matmul_readvariableop_resource_0*
_output_shapes
:	]Ô*
dtype02*
(while/lstm_cell_18/MatMul/ReadVariableOp×
while/lstm_cell_18/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_18/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÔ2
while/lstm_cell_18/MatMulÐ
*while/lstm_cell_18/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_18_matmul_1_readvariableop_resource_0* 
_output_shapes
:
Ô*
dtype02,
*while/lstm_cell_18/MatMul_1/ReadVariableOpÀ
while/lstm_cell_18/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_18/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÔ2
while/lstm_cell_18/MatMul_1¸
while/lstm_cell_18/addAddV2#while/lstm_cell_18/MatMul:product:0%while/lstm_cell_18/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÔ2
while/lstm_cell_18/addÈ
)while/lstm_cell_18/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_18_biasadd_readvariableop_resource_0*
_output_shapes	
:Ô*
dtype02+
)while/lstm_cell_18/BiasAdd/ReadVariableOpÅ
while/lstm_cell_18/BiasAddBiasAddwhile/lstm_cell_18/add:z:01while/lstm_cell_18/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÔ2
while/lstm_cell_18/BiasAdd
"while/lstm_cell_18/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_18/split/split_dim
while/lstm_cell_18/splitSplit+while/lstm_cell_18/split/split_dim:output:0#while/lstm_cell_18/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2
while/lstm_cell_18/split
while/lstm_cell_18/SigmoidSigmoid!while/lstm_cell_18/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_18/Sigmoid
while/lstm_cell_18/Sigmoid_1Sigmoid!while/lstm_cell_18/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_18/Sigmoid_1¡
while/lstm_cell_18/mulMul while/lstm_cell_18/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_18/mul
while/lstm_cell_18/ReluRelu!while/lstm_cell_18/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_18/Reluµ
while/lstm_cell_18/mul_1Mulwhile/lstm_cell_18/Sigmoid:y:0%while/lstm_cell_18/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_18/mul_1ª
while/lstm_cell_18/add_1AddV2while/lstm_cell_18/mul:z:0while/lstm_cell_18/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_18/add_1
while/lstm_cell_18/Sigmoid_2Sigmoid!while/lstm_cell_18/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_18/Sigmoid_2
while/lstm_cell_18/Relu_1Reluwhile/lstm_cell_18/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_18/Relu_1¹
while/lstm_cell_18/mul_2Mul while/lstm_cell_18/Sigmoid_2:y:0'while/lstm_cell_18/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_18/mul_2à
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
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_18/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_18/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/Identity_5Þ

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
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2V
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
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
¶
¸
*__inference_lstm_18_layer_call_fn_32582258

inputs
unknown:	]Ô
	unknown_0:
Ô
	unknown_1:	Ô
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_lstm_18_layer_call_and_return_conditional_losses_325801112
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

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
\

E__inference_lstm_18_layer_call_and_return_conditional_losses_32582225

inputs>
+lstm_cell_18_matmul_readvariableop_resource:	]ÔA
-lstm_cell_18_matmul_1_readvariableop_resource:
Ô;
,lstm_cell_18_biasadd_readvariableop_resource:	Ô
identity¢#lstm_cell_18/BiasAdd/ReadVariableOp¢"lstm_cell_18/MatMul/ReadVariableOp¢$lstm_cell_18/MatMul_1/ReadVariableOp¢whileD
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
B :2
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
B :2
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
:ÿÿÿÿÿÿÿÿÿ2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :2
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
B :2
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
:ÿÿÿÿÿÿÿÿÿ2	
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
"lstm_cell_18/MatMul/ReadVariableOpReadVariableOp+lstm_cell_18_matmul_readvariableop_resource*
_output_shapes
:	]Ô*
dtype02$
"lstm_cell_18/MatMul/ReadVariableOp­
lstm_cell_18/MatMulMatMulstrided_slice_2:output:0*lstm_cell_18/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÔ2
lstm_cell_18/MatMul¼
$lstm_cell_18/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_18_matmul_1_readvariableop_resource* 
_output_shapes
:
Ô*
dtype02&
$lstm_cell_18/MatMul_1/ReadVariableOp©
lstm_cell_18/MatMul_1MatMulzeros:output:0,lstm_cell_18/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÔ2
lstm_cell_18/MatMul_1 
lstm_cell_18/addAddV2lstm_cell_18/MatMul:product:0lstm_cell_18/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÔ2
lstm_cell_18/add´
#lstm_cell_18/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_18_biasadd_readvariableop_resource*
_output_shapes	
:Ô*
dtype02%
#lstm_cell_18/BiasAdd/ReadVariableOp­
lstm_cell_18/BiasAddBiasAddlstm_cell_18/add:z:0+lstm_cell_18/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÔ2
lstm_cell_18/BiasAdd~
lstm_cell_18/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_18/split/split_dim÷
lstm_cell_18/splitSplit%lstm_cell_18/split/split_dim:output:0lstm_cell_18/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2
lstm_cell_18/split
lstm_cell_18/SigmoidSigmoidlstm_cell_18/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_18/Sigmoid
lstm_cell_18/Sigmoid_1Sigmoidlstm_cell_18/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_18/Sigmoid_1
lstm_cell_18/mulMullstm_cell_18/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_18/mul~
lstm_cell_18/ReluRelulstm_cell_18/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_18/Relu
lstm_cell_18/mul_1Mullstm_cell_18/Sigmoid:y:0lstm_cell_18/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_18/mul_1
lstm_cell_18/add_1AddV2lstm_cell_18/mul:z:0lstm_cell_18/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_18/add_1
lstm_cell_18/Sigmoid_2Sigmoidlstm_cell_18/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_18/Sigmoid_2}
lstm_cell_18/Relu_1Relulstm_cell_18/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_18/Relu_1¡
lstm_cell_18/mul_2Mullstm_cell_18/Sigmoid_2:y:0!lstm_cell_18/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_18/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ  2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_18_matmul_readvariableop_resource-lstm_cell_18_matmul_1_readvariableop_resource,lstm_cell_18_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_32582141*
condR
while_cond_32582140*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ  22
0TensorArrayV2Stack/TensorListStack/element_shapeé
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
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
:ÿÿÿÿÿÿÿÿÿ*
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
:ÿÿÿÿÿÿÿÿÿ2
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
:ÿÿÿÿÿÿÿÿÿ2

IdentityÈ
NoOpNoOp$^lstm_cell_18/BiasAdd/ReadVariableOp#^lstm_cell_18/MatMul/ReadVariableOp%^lstm_cell_18/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ]: : : 2J
#lstm_cell_18/BiasAdd/ReadVariableOp#lstm_cell_18/BiasAdd/ReadVariableOp2H
"lstm_cell_18/MatMul/ReadVariableOp"lstm_cell_18/MatMul/ReadVariableOp2L
$lstm_cell_18/MatMul_1/ReadVariableOp$lstm_cell_18/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
 
_user_specified_nameinputs
ô%
Þ
!__inference__traced_save_32583266
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
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0)savev2_dense_9_kernel_read_readvariableop'savev2_dense_9_bias_read_readvariableop6savev2_lstm_18_lstm_cell_18_kernel_read_readvariableop@savev2_lstm_18_lstm_cell_18_recurrent_kernel_read_readvariableop4savev2_lstm_18_lstm_cell_18_bias_read_readvariableop6savev2_lstm_19_lstm_cell_19_kernel_read_readvariableop@savev2_lstm_19_lstm_cell_19_recurrent_kernel_read_readvariableop4savev2_lstm_19_lstm_cell_19_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableopsavev2_const"/device:CPU:0*
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
Z: :	×::	]Ô:
Ô:Ô:
Ü:
×Ü:Ü: : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	×: 

_output_shapes
::%!

_output_shapes
:	]Ô:&"
 
_output_shapes
:
Ô:!

_output_shapes	
:Ô:&"
 
_output_shapes
:
Ü:&"
 
_output_shapes
:
×Ü:!

_output_shapes	
:Ü:	
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
while_body_32580027
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_18_matmul_readvariableop_resource_0:	]ÔI
5while_lstm_cell_18_matmul_1_readvariableop_resource_0:
ÔC
4while_lstm_cell_18_biasadd_readvariableop_resource_0:	Ô
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_18_matmul_readvariableop_resource:	]ÔG
3while_lstm_cell_18_matmul_1_readvariableop_resource:
ÔA
2while_lstm_cell_18_biasadd_readvariableop_resource:	Ô¢)while/lstm_cell_18/BiasAdd/ReadVariableOp¢(while/lstm_cell_18/MatMul/ReadVariableOp¢*while/lstm_cell_18/MatMul_1/ReadVariableOpÃ
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
(while/lstm_cell_18/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_18_matmul_readvariableop_resource_0*
_output_shapes
:	]Ô*
dtype02*
(while/lstm_cell_18/MatMul/ReadVariableOp×
while/lstm_cell_18/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_18/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÔ2
while/lstm_cell_18/MatMulÐ
*while/lstm_cell_18/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_18_matmul_1_readvariableop_resource_0* 
_output_shapes
:
Ô*
dtype02,
*while/lstm_cell_18/MatMul_1/ReadVariableOpÀ
while/lstm_cell_18/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_18/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÔ2
while/lstm_cell_18/MatMul_1¸
while/lstm_cell_18/addAddV2#while/lstm_cell_18/MatMul:product:0%while/lstm_cell_18/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÔ2
while/lstm_cell_18/addÈ
)while/lstm_cell_18/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_18_biasadd_readvariableop_resource_0*
_output_shapes	
:Ô*
dtype02+
)while/lstm_cell_18/BiasAdd/ReadVariableOpÅ
while/lstm_cell_18/BiasAddBiasAddwhile/lstm_cell_18/add:z:01while/lstm_cell_18/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÔ2
while/lstm_cell_18/BiasAdd
"while/lstm_cell_18/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_18/split/split_dim
while/lstm_cell_18/splitSplit+while/lstm_cell_18/split/split_dim:output:0#while/lstm_cell_18/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2
while/lstm_cell_18/split
while/lstm_cell_18/SigmoidSigmoid!while/lstm_cell_18/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_18/Sigmoid
while/lstm_cell_18/Sigmoid_1Sigmoid!while/lstm_cell_18/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_18/Sigmoid_1¡
while/lstm_cell_18/mulMul while/lstm_cell_18/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_18/mul
while/lstm_cell_18/ReluRelu!while/lstm_cell_18/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_18/Reluµ
while/lstm_cell_18/mul_1Mulwhile/lstm_cell_18/Sigmoid:y:0%while/lstm_cell_18/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_18/mul_1ª
while/lstm_cell_18/add_1AddV2while/lstm_cell_18/mul:z:0while/lstm_cell_18/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_18/add_1
while/lstm_cell_18/Sigmoid_2Sigmoid!while/lstm_cell_18/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_18/Sigmoid_2
while/lstm_cell_18/Relu_1Reluwhile/lstm_cell_18/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_18/Relu_1¹
while/lstm_cell_18/mul_2Mul while/lstm_cell_18/Sigmoid_2:y:0'while/lstm_cell_18/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_18/mul_2à
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
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_18/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_18/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/Identity_5Þ

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
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2V
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
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
×
g
H__inference_dropout_18_layer_call_and_return_conditional_losses_32580574

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
:ÿÿÿÿÿÿÿÿÿ2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape¹
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
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
:ÿÿÿÿÿÿÿÿÿ2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/Mul_1j
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
´
Ñ	
#__inference__wrapped_model_32578694
lstm_18_inputS
@sequential_9_lstm_18_lstm_cell_18_matmul_readvariableop_resource:	]ÔV
Bsequential_9_lstm_18_lstm_cell_18_matmul_1_readvariableop_resource:
ÔP
Asequential_9_lstm_18_lstm_cell_18_biasadd_readvariableop_resource:	ÔT
@sequential_9_lstm_19_lstm_cell_19_matmul_readvariableop_resource:
ÜV
Bsequential_9_lstm_19_lstm_cell_19_matmul_1_readvariableop_resource:
×ÜP
Asequential_9_lstm_19_lstm_cell_19_biasadd_readvariableop_resource:	ÜI
6sequential_9_dense_9_tensordot_readvariableop_resource:	×B
4sequential_9_dense_9_biasadd_readvariableop_resource:
identity¢+sequential_9/dense_9/BiasAdd/ReadVariableOp¢-sequential_9/dense_9/Tensordot/ReadVariableOp¢8sequential_9/lstm_18/lstm_cell_18/BiasAdd/ReadVariableOp¢7sequential_9/lstm_18/lstm_cell_18/MatMul/ReadVariableOp¢9sequential_9/lstm_18/lstm_cell_18/MatMul_1/ReadVariableOp¢sequential_9/lstm_18/while¢8sequential_9/lstm_19/lstm_cell_19/BiasAdd/ReadVariableOp¢7sequential_9/lstm_19/lstm_cell_19/MatMul/ReadVariableOp¢9sequential_9/lstm_19/lstm_cell_19/MatMul_1/ReadVariableOp¢sequential_9/lstm_19/whileu
sequential_9/lstm_18/ShapeShapelstm_18_input*
T0*
_output_shapes
:2
sequential_9/lstm_18/Shape
(sequential_9/lstm_18/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(sequential_9/lstm_18/strided_slice/stack¢
*sequential_9/lstm_18/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*sequential_9/lstm_18/strided_slice/stack_1¢
*sequential_9/lstm_18/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*sequential_9/lstm_18/strided_slice/stack_2à
"sequential_9/lstm_18/strided_sliceStridedSlice#sequential_9/lstm_18/Shape:output:01sequential_9/lstm_18/strided_slice/stack:output:03sequential_9/lstm_18/strided_slice/stack_1:output:03sequential_9/lstm_18/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"sequential_9/lstm_18/strided_slice
 sequential_9/lstm_18/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :2"
 sequential_9/lstm_18/zeros/mul/yÀ
sequential_9/lstm_18/zeros/mulMul+sequential_9/lstm_18/strided_slice:output:0)sequential_9/lstm_18/zeros/mul/y:output:0*
T0*
_output_shapes
: 2 
sequential_9/lstm_18/zeros/mul
!sequential_9/lstm_18/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2#
!sequential_9/lstm_18/zeros/Less/y»
sequential_9/lstm_18/zeros/LessLess"sequential_9/lstm_18/zeros/mul:z:0*sequential_9/lstm_18/zeros/Less/y:output:0*
T0*
_output_shapes
: 2!
sequential_9/lstm_18/zeros/Less
#sequential_9/lstm_18/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :2%
#sequential_9/lstm_18/zeros/packed/1×
!sequential_9/lstm_18/zeros/packedPack+sequential_9/lstm_18/strided_slice:output:0,sequential_9/lstm_18/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2#
!sequential_9/lstm_18/zeros/packed
 sequential_9/lstm_18/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 sequential_9/lstm_18/zeros/ConstÊ
sequential_9/lstm_18/zerosFill*sequential_9/lstm_18/zeros/packed:output:0)sequential_9/lstm_18/zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential_9/lstm_18/zeros
"sequential_9/lstm_18/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :2$
"sequential_9/lstm_18/zeros_1/mul/yÆ
 sequential_9/lstm_18/zeros_1/mulMul+sequential_9/lstm_18/strided_slice:output:0+sequential_9/lstm_18/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2"
 sequential_9/lstm_18/zeros_1/mul
#sequential_9/lstm_18/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2%
#sequential_9/lstm_18/zeros_1/Less/yÃ
!sequential_9/lstm_18/zeros_1/LessLess$sequential_9/lstm_18/zeros_1/mul:z:0,sequential_9/lstm_18/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2#
!sequential_9/lstm_18/zeros_1/Less
%sequential_9/lstm_18/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :2'
%sequential_9/lstm_18/zeros_1/packed/1Ý
#sequential_9/lstm_18/zeros_1/packedPack+sequential_9/lstm_18/strided_slice:output:0.sequential_9/lstm_18/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2%
#sequential_9/lstm_18/zeros_1/packed
"sequential_9/lstm_18/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"sequential_9/lstm_18/zeros_1/ConstÒ
sequential_9/lstm_18/zeros_1Fill,sequential_9/lstm_18/zeros_1/packed:output:0+sequential_9/lstm_18/zeros_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential_9/lstm_18/zeros_1
#sequential_9/lstm_18/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2%
#sequential_9/lstm_18/transpose/permÀ
sequential_9/lstm_18/transpose	Transposelstm_18_input,sequential_9/lstm_18/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]2 
sequential_9/lstm_18/transpose
sequential_9/lstm_18/Shape_1Shape"sequential_9/lstm_18/transpose:y:0*
T0*
_output_shapes
:2
sequential_9/lstm_18/Shape_1¢
*sequential_9/lstm_18/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2,
*sequential_9/lstm_18/strided_slice_1/stack¦
,sequential_9/lstm_18/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_9/lstm_18/strided_slice_1/stack_1¦
,sequential_9/lstm_18/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_9/lstm_18/strided_slice_1/stack_2ì
$sequential_9/lstm_18/strided_slice_1StridedSlice%sequential_9/lstm_18/Shape_1:output:03sequential_9/lstm_18/strided_slice_1/stack:output:05sequential_9/lstm_18/strided_slice_1/stack_1:output:05sequential_9/lstm_18/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2&
$sequential_9/lstm_18/strided_slice_1¯
0sequential_9/lstm_18/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ22
0sequential_9/lstm_18/TensorArrayV2/element_shape
"sequential_9/lstm_18/TensorArrayV2TensorListReserve9sequential_9/lstm_18/TensorArrayV2/element_shape:output:0-sequential_9/lstm_18/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02$
"sequential_9/lstm_18/TensorArrayV2é
Jsequential_9/lstm_18/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ]   2L
Jsequential_9/lstm_18/TensorArrayUnstack/TensorListFromTensor/element_shapeÌ
<sequential_9/lstm_18/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor"sequential_9/lstm_18/transpose:y:0Ssequential_9/lstm_18/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02>
<sequential_9/lstm_18/TensorArrayUnstack/TensorListFromTensor¢
*sequential_9/lstm_18/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2,
*sequential_9/lstm_18/strided_slice_2/stack¦
,sequential_9/lstm_18/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_9/lstm_18/strided_slice_2/stack_1¦
,sequential_9/lstm_18/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_9/lstm_18/strided_slice_2/stack_2ú
$sequential_9/lstm_18/strided_slice_2StridedSlice"sequential_9/lstm_18/transpose:y:03sequential_9/lstm_18/strided_slice_2/stack:output:05sequential_9/lstm_18/strided_slice_2/stack_1:output:05sequential_9/lstm_18/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]*
shrink_axis_mask2&
$sequential_9/lstm_18/strided_slice_2ô
7sequential_9/lstm_18/lstm_cell_18/MatMul/ReadVariableOpReadVariableOp@sequential_9_lstm_18_lstm_cell_18_matmul_readvariableop_resource*
_output_shapes
:	]Ô*
dtype029
7sequential_9/lstm_18/lstm_cell_18/MatMul/ReadVariableOp
(sequential_9/lstm_18/lstm_cell_18/MatMulMatMul-sequential_9/lstm_18/strided_slice_2:output:0?sequential_9/lstm_18/lstm_cell_18/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÔ2*
(sequential_9/lstm_18/lstm_cell_18/MatMulû
9sequential_9/lstm_18/lstm_cell_18/MatMul_1/ReadVariableOpReadVariableOpBsequential_9_lstm_18_lstm_cell_18_matmul_1_readvariableop_resource* 
_output_shapes
:
Ô*
dtype02;
9sequential_9/lstm_18/lstm_cell_18/MatMul_1/ReadVariableOpý
*sequential_9/lstm_18/lstm_cell_18/MatMul_1MatMul#sequential_9/lstm_18/zeros:output:0Asequential_9/lstm_18/lstm_cell_18/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÔ2,
*sequential_9/lstm_18/lstm_cell_18/MatMul_1ô
%sequential_9/lstm_18/lstm_cell_18/addAddV22sequential_9/lstm_18/lstm_cell_18/MatMul:product:04sequential_9/lstm_18/lstm_cell_18/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÔ2'
%sequential_9/lstm_18/lstm_cell_18/addó
8sequential_9/lstm_18/lstm_cell_18/BiasAdd/ReadVariableOpReadVariableOpAsequential_9_lstm_18_lstm_cell_18_biasadd_readvariableop_resource*
_output_shapes	
:Ô*
dtype02:
8sequential_9/lstm_18/lstm_cell_18/BiasAdd/ReadVariableOp
)sequential_9/lstm_18/lstm_cell_18/BiasAddBiasAdd)sequential_9/lstm_18/lstm_cell_18/add:z:0@sequential_9/lstm_18/lstm_cell_18/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÔ2+
)sequential_9/lstm_18/lstm_cell_18/BiasAdd¨
1sequential_9/lstm_18/lstm_cell_18/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :23
1sequential_9/lstm_18/lstm_cell_18/split/split_dimË
'sequential_9/lstm_18/lstm_cell_18/splitSplit:sequential_9/lstm_18/lstm_cell_18/split/split_dim:output:02sequential_9/lstm_18/lstm_cell_18/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2)
'sequential_9/lstm_18/lstm_cell_18/splitÆ
)sequential_9/lstm_18/lstm_cell_18/SigmoidSigmoid0sequential_9/lstm_18/lstm_cell_18/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2+
)sequential_9/lstm_18/lstm_cell_18/SigmoidÊ
+sequential_9/lstm_18/lstm_cell_18/Sigmoid_1Sigmoid0sequential_9/lstm_18/lstm_cell_18/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2-
+sequential_9/lstm_18/lstm_cell_18/Sigmoid_1à
%sequential_9/lstm_18/lstm_cell_18/mulMul/sequential_9/lstm_18/lstm_cell_18/Sigmoid_1:y:0%sequential_9/lstm_18/zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%sequential_9/lstm_18/lstm_cell_18/mul½
&sequential_9/lstm_18/lstm_cell_18/ReluRelu0sequential_9/lstm_18/lstm_cell_18/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&sequential_9/lstm_18/lstm_cell_18/Reluñ
'sequential_9/lstm_18/lstm_cell_18/mul_1Mul-sequential_9/lstm_18/lstm_cell_18/Sigmoid:y:04sequential_9/lstm_18/lstm_cell_18/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'sequential_9/lstm_18/lstm_cell_18/mul_1æ
'sequential_9/lstm_18/lstm_cell_18/add_1AddV2)sequential_9/lstm_18/lstm_cell_18/mul:z:0+sequential_9/lstm_18/lstm_cell_18/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'sequential_9/lstm_18/lstm_cell_18/add_1Ê
+sequential_9/lstm_18/lstm_cell_18/Sigmoid_2Sigmoid0sequential_9/lstm_18/lstm_cell_18/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2-
+sequential_9/lstm_18/lstm_cell_18/Sigmoid_2¼
(sequential_9/lstm_18/lstm_cell_18/Relu_1Relu+sequential_9/lstm_18/lstm_cell_18/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
(sequential_9/lstm_18/lstm_cell_18/Relu_1õ
'sequential_9/lstm_18/lstm_cell_18/mul_2Mul/sequential_9/lstm_18/lstm_cell_18/Sigmoid_2:y:06sequential_9/lstm_18/lstm_cell_18/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'sequential_9/lstm_18/lstm_cell_18/mul_2¹
2sequential_9/lstm_18/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ  24
2sequential_9/lstm_18/TensorArrayV2_1/element_shape
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
sequential_9/lstm_18/time©
-sequential_9/lstm_18/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2/
-sequential_9/lstm_18/while/maximum_iterations
'sequential_9/lstm_18/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2)
'sequential_9/lstm_18/while/loop_counterÎ
sequential_9/lstm_18/whileWhile0sequential_9/lstm_18/while/loop_counter:output:06sequential_9/lstm_18/while/maximum_iterations:output:0"sequential_9/lstm_18/time:output:0-sequential_9/lstm_18/TensorArrayV2_1:handle:0#sequential_9/lstm_18/zeros:output:0%sequential_9/lstm_18/zeros_1:output:0-sequential_9/lstm_18/strided_slice_1:output:0Lsequential_9/lstm_18/TensorArrayUnstack/TensorListFromTensor:output_handle:0@sequential_9_lstm_18_lstm_cell_18_matmul_readvariableop_resourceBsequential_9_lstm_18_lstm_cell_18_matmul_1_readvariableop_resourceAsequential_9_lstm_18_lstm_cell_18_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *4
body,R*
(sequential_9_lstm_18_while_body_32578434*4
cond,R*
(sequential_9_lstm_18_while_cond_32578433*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 2
sequential_9/lstm_18/whileß
Esequential_9/lstm_18/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ  2G
Esequential_9/lstm_18/TensorArrayV2Stack/TensorListStack/element_shape½
7sequential_9/lstm_18/TensorArrayV2Stack/TensorListStackTensorListStack#sequential_9/lstm_18/while:output:3Nsequential_9/lstm_18/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype029
7sequential_9/lstm_18/TensorArrayV2Stack/TensorListStack«
*sequential_9/lstm_18/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2,
*sequential_9/lstm_18/strided_slice_3/stack¦
,sequential_9/lstm_18/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2.
,sequential_9/lstm_18/strided_slice_3/stack_1¦
,sequential_9/lstm_18/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_9/lstm_18/strided_slice_3/stack_2
$sequential_9/lstm_18/strided_slice_3StridedSlice@sequential_9/lstm_18/TensorArrayV2Stack/TensorListStack:tensor:03sequential_9/lstm_18/strided_slice_3/stack:output:05sequential_9/lstm_18/strided_slice_3/stack_1:output:05sequential_9/lstm_18/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2&
$sequential_9/lstm_18/strided_slice_3£
%sequential_9/lstm_18/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2'
%sequential_9/lstm_18/transpose_1/permú
 sequential_9/lstm_18/transpose_1	Transpose@sequential_9/lstm_18/TensorArrayV2Stack/TensorListStack:tensor:0.sequential_9/lstm_18/transpose_1/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 sequential_9/lstm_18/transpose_1
sequential_9/lstm_18/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
sequential_9/lstm_18/runtime­
 sequential_9/dropout_18/IdentityIdentity$sequential_9/lstm_18/transpose_1:y:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 sequential_9/dropout_18/Identity
sequential_9/lstm_19/ShapeShape)sequential_9/dropout_18/Identity:output:0*
T0*
_output_shapes
:2
sequential_9/lstm_19/Shape
(sequential_9/lstm_19/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(sequential_9/lstm_19/strided_slice/stack¢
*sequential_9/lstm_19/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*sequential_9/lstm_19/strided_slice/stack_1¢
*sequential_9/lstm_19/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*sequential_9/lstm_19/strided_slice/stack_2à
"sequential_9/lstm_19/strided_sliceStridedSlice#sequential_9/lstm_19/Shape:output:01sequential_9/lstm_19/strided_slice/stack:output:03sequential_9/lstm_19/strided_slice/stack_1:output:03sequential_9/lstm_19/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"sequential_9/lstm_19/strided_slice
 sequential_9/lstm_19/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :×2"
 sequential_9/lstm_19/zeros/mul/yÀ
sequential_9/lstm_19/zeros/mulMul+sequential_9/lstm_19/strided_slice:output:0)sequential_9/lstm_19/zeros/mul/y:output:0*
T0*
_output_shapes
: 2 
sequential_9/lstm_19/zeros/mul
!sequential_9/lstm_19/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2#
!sequential_9/lstm_19/zeros/Less/y»
sequential_9/lstm_19/zeros/LessLess"sequential_9/lstm_19/zeros/mul:z:0*sequential_9/lstm_19/zeros/Less/y:output:0*
T0*
_output_shapes
: 2!
sequential_9/lstm_19/zeros/Less
#sequential_9/lstm_19/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :×2%
#sequential_9/lstm_19/zeros/packed/1×
!sequential_9/lstm_19/zeros/packedPack+sequential_9/lstm_19/strided_slice:output:0,sequential_9/lstm_19/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2#
!sequential_9/lstm_19/zeros/packed
 sequential_9/lstm_19/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 sequential_9/lstm_19/zeros/ConstÊ
sequential_9/lstm_19/zerosFill*sequential_9/lstm_19/zeros/packed:output:0)sequential_9/lstm_19/zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2
sequential_9/lstm_19/zeros
"sequential_9/lstm_19/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :×2$
"sequential_9/lstm_19/zeros_1/mul/yÆ
 sequential_9/lstm_19/zeros_1/mulMul+sequential_9/lstm_19/strided_slice:output:0+sequential_9/lstm_19/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2"
 sequential_9/lstm_19/zeros_1/mul
#sequential_9/lstm_19/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2%
#sequential_9/lstm_19/zeros_1/Less/yÃ
!sequential_9/lstm_19/zeros_1/LessLess$sequential_9/lstm_19/zeros_1/mul:z:0,sequential_9/lstm_19/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2#
!sequential_9/lstm_19/zeros_1/Less
%sequential_9/lstm_19/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :×2'
%sequential_9/lstm_19/zeros_1/packed/1Ý
#sequential_9/lstm_19/zeros_1/packedPack+sequential_9/lstm_19/strided_slice:output:0.sequential_9/lstm_19/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2%
#sequential_9/lstm_19/zeros_1/packed
"sequential_9/lstm_19/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"sequential_9/lstm_19/zeros_1/ConstÒ
sequential_9/lstm_19/zeros_1Fill,sequential_9/lstm_19/zeros_1/packed:output:0+sequential_9/lstm_19/zeros_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2
sequential_9/lstm_19/zeros_1
#sequential_9/lstm_19/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2%
#sequential_9/lstm_19/transpose/permÝ
sequential_9/lstm_19/transpose	Transpose)sequential_9/dropout_18/Identity:output:0,sequential_9/lstm_19/transpose/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
sequential_9/lstm_19/transpose
sequential_9/lstm_19/Shape_1Shape"sequential_9/lstm_19/transpose:y:0*
T0*
_output_shapes
:2
sequential_9/lstm_19/Shape_1¢
*sequential_9/lstm_19/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2,
*sequential_9/lstm_19/strided_slice_1/stack¦
,sequential_9/lstm_19/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_9/lstm_19/strided_slice_1/stack_1¦
,sequential_9/lstm_19/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_9/lstm_19/strided_slice_1/stack_2ì
$sequential_9/lstm_19/strided_slice_1StridedSlice%sequential_9/lstm_19/Shape_1:output:03sequential_9/lstm_19/strided_slice_1/stack:output:05sequential_9/lstm_19/strided_slice_1/stack_1:output:05sequential_9/lstm_19/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2&
$sequential_9/lstm_19/strided_slice_1¯
0sequential_9/lstm_19/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ22
0sequential_9/lstm_19/TensorArrayV2/element_shape
"sequential_9/lstm_19/TensorArrayV2TensorListReserve9sequential_9/lstm_19/TensorArrayV2/element_shape:output:0-sequential_9/lstm_19/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02$
"sequential_9/lstm_19/TensorArrayV2é
Jsequential_9/lstm_19/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ  2L
Jsequential_9/lstm_19/TensorArrayUnstack/TensorListFromTensor/element_shapeÌ
<sequential_9/lstm_19/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor"sequential_9/lstm_19/transpose:y:0Ssequential_9/lstm_19/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02>
<sequential_9/lstm_19/TensorArrayUnstack/TensorListFromTensor¢
*sequential_9/lstm_19/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2,
*sequential_9/lstm_19/strided_slice_2/stack¦
,sequential_9/lstm_19/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_9/lstm_19/strided_slice_2/stack_1¦
,sequential_9/lstm_19/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_9/lstm_19/strided_slice_2/stack_2û
$sequential_9/lstm_19/strided_slice_2StridedSlice"sequential_9/lstm_19/transpose:y:03sequential_9/lstm_19/strided_slice_2/stack:output:05sequential_9/lstm_19/strided_slice_2/stack_1:output:05sequential_9/lstm_19/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2&
$sequential_9/lstm_19/strided_slice_2õ
7sequential_9/lstm_19/lstm_cell_19/MatMul/ReadVariableOpReadVariableOp@sequential_9_lstm_19_lstm_cell_19_matmul_readvariableop_resource* 
_output_shapes
:
Ü*
dtype029
7sequential_9/lstm_19/lstm_cell_19/MatMul/ReadVariableOp
(sequential_9/lstm_19/lstm_cell_19/MatMulMatMul-sequential_9/lstm_19/strided_slice_2:output:0?sequential_9/lstm_19/lstm_cell_19/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2*
(sequential_9/lstm_19/lstm_cell_19/MatMulû
9sequential_9/lstm_19/lstm_cell_19/MatMul_1/ReadVariableOpReadVariableOpBsequential_9_lstm_19_lstm_cell_19_matmul_1_readvariableop_resource* 
_output_shapes
:
×Ü*
dtype02;
9sequential_9/lstm_19/lstm_cell_19/MatMul_1/ReadVariableOpý
*sequential_9/lstm_19/lstm_cell_19/MatMul_1MatMul#sequential_9/lstm_19/zeros:output:0Asequential_9/lstm_19/lstm_cell_19/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2,
*sequential_9/lstm_19/lstm_cell_19/MatMul_1ô
%sequential_9/lstm_19/lstm_cell_19/addAddV22sequential_9/lstm_19/lstm_cell_19/MatMul:product:04sequential_9/lstm_19/lstm_cell_19/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2'
%sequential_9/lstm_19/lstm_cell_19/addó
8sequential_9/lstm_19/lstm_cell_19/BiasAdd/ReadVariableOpReadVariableOpAsequential_9_lstm_19_lstm_cell_19_biasadd_readvariableop_resource*
_output_shapes	
:Ü*
dtype02:
8sequential_9/lstm_19/lstm_cell_19/BiasAdd/ReadVariableOp
)sequential_9/lstm_19/lstm_cell_19/BiasAddBiasAdd)sequential_9/lstm_19/lstm_cell_19/add:z:0@sequential_9/lstm_19/lstm_cell_19/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2+
)sequential_9/lstm_19/lstm_cell_19/BiasAdd¨
1sequential_9/lstm_19/lstm_cell_19/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :23
1sequential_9/lstm_19/lstm_cell_19/split/split_dimË
'sequential_9/lstm_19/lstm_cell_19/splitSplit:sequential_9/lstm_19/lstm_cell_19/split/split_dim:output:02sequential_9/lstm_19/lstm_cell_19/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ×:ÿÿÿÿÿÿÿÿÿ×:ÿÿÿÿÿÿÿÿÿ×:ÿÿÿÿÿÿÿÿÿ×*
	num_split2)
'sequential_9/lstm_19/lstm_cell_19/splitÆ
)sequential_9/lstm_19/lstm_cell_19/SigmoidSigmoid0sequential_9/lstm_19/lstm_cell_19/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2+
)sequential_9/lstm_19/lstm_cell_19/SigmoidÊ
+sequential_9/lstm_19/lstm_cell_19/Sigmoid_1Sigmoid0sequential_9/lstm_19/lstm_cell_19/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2-
+sequential_9/lstm_19/lstm_cell_19/Sigmoid_1à
%sequential_9/lstm_19/lstm_cell_19/mulMul/sequential_9/lstm_19/lstm_cell_19/Sigmoid_1:y:0%sequential_9/lstm_19/zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2'
%sequential_9/lstm_19/lstm_cell_19/mul½
&sequential_9/lstm_19/lstm_cell_19/ReluRelu0sequential_9/lstm_19/lstm_cell_19/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2(
&sequential_9/lstm_19/lstm_cell_19/Reluñ
'sequential_9/lstm_19/lstm_cell_19/mul_1Mul-sequential_9/lstm_19/lstm_cell_19/Sigmoid:y:04sequential_9/lstm_19/lstm_cell_19/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2)
'sequential_9/lstm_19/lstm_cell_19/mul_1æ
'sequential_9/lstm_19/lstm_cell_19/add_1AddV2)sequential_9/lstm_19/lstm_cell_19/mul:z:0+sequential_9/lstm_19/lstm_cell_19/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2)
'sequential_9/lstm_19/lstm_cell_19/add_1Ê
+sequential_9/lstm_19/lstm_cell_19/Sigmoid_2Sigmoid0sequential_9/lstm_19/lstm_cell_19/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2-
+sequential_9/lstm_19/lstm_cell_19/Sigmoid_2¼
(sequential_9/lstm_19/lstm_cell_19/Relu_1Relu+sequential_9/lstm_19/lstm_cell_19/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2*
(sequential_9/lstm_19/lstm_cell_19/Relu_1õ
'sequential_9/lstm_19/lstm_cell_19/mul_2Mul/sequential_9/lstm_19/lstm_cell_19/Sigmoid_2:y:06sequential_9/lstm_19/lstm_cell_19/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2)
'sequential_9/lstm_19/lstm_cell_19/mul_2¹
2sequential_9/lstm_19/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ×   24
2sequential_9/lstm_19/TensorArrayV2_1/element_shape
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
sequential_9/lstm_19/time©
-sequential_9/lstm_19/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2/
-sequential_9/lstm_19/while/maximum_iterations
'sequential_9/lstm_19/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2)
'sequential_9/lstm_19/while/loop_counterÎ
sequential_9/lstm_19/whileWhile0sequential_9/lstm_19/while/loop_counter:output:06sequential_9/lstm_19/while/maximum_iterations:output:0"sequential_9/lstm_19/time:output:0-sequential_9/lstm_19/TensorArrayV2_1:handle:0#sequential_9/lstm_19/zeros:output:0%sequential_9/lstm_19/zeros_1:output:0-sequential_9/lstm_19/strided_slice_1:output:0Lsequential_9/lstm_19/TensorArrayUnstack/TensorListFromTensor:output_handle:0@sequential_9_lstm_19_lstm_cell_19_matmul_readvariableop_resourceBsequential_9_lstm_19_lstm_cell_19_matmul_1_readvariableop_resourceAsequential_9_lstm_19_lstm_cell_19_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ×:ÿÿÿÿÿÿÿÿÿ×: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *4
body,R*
(sequential_9_lstm_19_while_body_32578582*4
cond,R*
(sequential_9_lstm_19_while_cond_32578581*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ×:ÿÿÿÿÿÿÿÿÿ×: : : : : *
parallel_iterations 2
sequential_9/lstm_19/whileß
Esequential_9/lstm_19/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ×   2G
Esequential_9/lstm_19/TensorArrayV2Stack/TensorListStack/element_shape½
7sequential_9/lstm_19/TensorArrayV2Stack/TensorListStackTensorListStack#sequential_9/lstm_19/while:output:3Nsequential_9/lstm_19/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×*
element_dtype029
7sequential_9/lstm_19/TensorArrayV2Stack/TensorListStack«
*sequential_9/lstm_19/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2,
*sequential_9/lstm_19/strided_slice_3/stack¦
,sequential_9/lstm_19/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2.
,sequential_9/lstm_19/strided_slice_3/stack_1¦
,sequential_9/lstm_19/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_9/lstm_19/strided_slice_3/stack_2
$sequential_9/lstm_19/strided_slice_3StridedSlice@sequential_9/lstm_19/TensorArrayV2Stack/TensorListStack:tensor:03sequential_9/lstm_19/strided_slice_3/stack:output:05sequential_9/lstm_19/strided_slice_3/stack_1:output:05sequential_9/lstm_19/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×*
shrink_axis_mask2&
$sequential_9/lstm_19/strided_slice_3£
%sequential_9/lstm_19/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2'
%sequential_9/lstm_19/transpose_1/permú
 sequential_9/lstm_19/transpose_1	Transpose@sequential_9/lstm_19/TensorArrayV2Stack/TensorListStack:tensor:0.sequential_9/lstm_19/transpose_1/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2"
 sequential_9/lstm_19/transpose_1
sequential_9/lstm_19/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
sequential_9/lstm_19/runtime­
 sequential_9/dropout_19/IdentityIdentity$sequential_9/lstm_19/transpose_1:y:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2"
 sequential_9/dropout_19/IdentityÖ
-sequential_9/dense_9/Tensordot/ReadVariableOpReadVariableOp6sequential_9_dense_9_tensordot_readvariableop_resource*
_output_shapes
:	×*
dtype02/
-sequential_9/dense_9/Tensordot/ReadVariableOp
#sequential_9/dense_9/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2%
#sequential_9/dense_9/Tensordot/axes
#sequential_9/dense_9/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2%
#sequential_9/dense_9/Tensordot/free¥
$sequential_9/dense_9/Tensordot/ShapeShape)sequential_9/dropout_19/Identity:output:0*
T0*
_output_shapes
:2&
$sequential_9/dense_9/Tensordot/Shape
,sequential_9/dense_9/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,sequential_9/dense_9/Tensordot/GatherV2/axisº
'sequential_9/dense_9/Tensordot/GatherV2GatherV2-sequential_9/dense_9/Tensordot/Shape:output:0,sequential_9/dense_9/Tensordot/free:output:05sequential_9/dense_9/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2)
'sequential_9/dense_9/Tensordot/GatherV2¢
.sequential_9/dense_9/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.sequential_9/dense_9/Tensordot/GatherV2_1/axisÀ
)sequential_9/dense_9/Tensordot/GatherV2_1GatherV2-sequential_9/dense_9/Tensordot/Shape:output:0,sequential_9/dense_9/Tensordot/axes:output:07sequential_9/dense_9/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2+
)sequential_9/dense_9/Tensordot/GatherV2_1
$sequential_9/dense_9/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2&
$sequential_9/dense_9/Tensordot/ConstÔ
#sequential_9/dense_9/Tensordot/ProdProd0sequential_9/dense_9/Tensordot/GatherV2:output:0-sequential_9/dense_9/Tensordot/Const:output:0*
T0*
_output_shapes
: 2%
#sequential_9/dense_9/Tensordot/Prod
&sequential_9/dense_9/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2(
&sequential_9/dense_9/Tensordot/Const_1Ü
%sequential_9/dense_9/Tensordot/Prod_1Prod2sequential_9/dense_9/Tensordot/GatherV2_1:output:0/sequential_9/dense_9/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2'
%sequential_9/dense_9/Tensordot/Prod_1
*sequential_9/dense_9/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2,
*sequential_9/dense_9/Tensordot/concat/axis
%sequential_9/dense_9/Tensordot/concatConcatV2,sequential_9/dense_9/Tensordot/free:output:0,sequential_9/dense_9/Tensordot/axes:output:03sequential_9/dense_9/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2'
%sequential_9/dense_9/Tensordot/concatà
$sequential_9/dense_9/Tensordot/stackPack,sequential_9/dense_9/Tensordot/Prod:output:0.sequential_9/dense_9/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2&
$sequential_9/dense_9/Tensordot/stackó
(sequential_9/dense_9/Tensordot/transpose	Transpose)sequential_9/dropout_19/Identity:output:0.sequential_9/dense_9/Tensordot/concat:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2*
(sequential_9/dense_9/Tensordot/transposeó
&sequential_9/dense_9/Tensordot/ReshapeReshape,sequential_9/dense_9/Tensordot/transpose:y:0-sequential_9/dense_9/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2(
&sequential_9/dense_9/Tensordot/Reshapeò
%sequential_9/dense_9/Tensordot/MatMulMatMul/sequential_9/dense_9/Tensordot/Reshape:output:05sequential_9/dense_9/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%sequential_9/dense_9/Tensordot/MatMul
&sequential_9/dense_9/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&sequential_9/dense_9/Tensordot/Const_2
,sequential_9/dense_9/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,sequential_9/dense_9/Tensordot/concat_1/axis¦
'sequential_9/dense_9/Tensordot/concat_1ConcatV20sequential_9/dense_9/Tensordot/GatherV2:output:0/sequential_9/dense_9/Tensordot/Const_2:output:05sequential_9/dense_9/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2)
'sequential_9/dense_9/Tensordot/concat_1ä
sequential_9/dense_9/TensordotReshape/sequential_9/dense_9/Tensordot/MatMul:product:00sequential_9/dense_9/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
sequential_9/dense_9/TensordotË
+sequential_9/dense_9/BiasAdd/ReadVariableOpReadVariableOp4sequential_9_dense_9_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02-
+sequential_9/dense_9/BiasAdd/ReadVariableOpÛ
sequential_9/dense_9/BiasAddBiasAdd'sequential_9/dense_9/Tensordot:output:03sequential_9/dense_9/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential_9/dense_9/BiasAdd¤
sequential_9/dense_9/SoftmaxSoftmax%sequential_9/dense_9/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential_9/dense_9/Softmax
IdentityIdentity&sequential_9/dense_9/Softmax:softmax:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

IdentityÈ
NoOpNoOp,^sequential_9/dense_9/BiasAdd/ReadVariableOp.^sequential_9/dense_9/Tensordot/ReadVariableOp9^sequential_9/lstm_18/lstm_cell_18/BiasAdd/ReadVariableOp8^sequential_9/lstm_18/lstm_cell_18/MatMul/ReadVariableOp:^sequential_9/lstm_18/lstm_cell_18/MatMul_1/ReadVariableOp^sequential_9/lstm_18/while9^sequential_9/lstm_19/lstm_cell_19/BiasAdd/ReadVariableOp8^sequential_9/lstm_19/lstm_cell_19/MatMul/ReadVariableOp:^sequential_9/lstm_19/lstm_cell_19/MatMul_1/ReadVariableOp^sequential_9/lstm_19/while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ]: : : : : : : : 2Z
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
:ÿÿÿÿÿÿÿÿÿ]
'
_user_specified_namelstm_18_input
°]
÷
(sequential_9_lstm_19_while_body_32578582F
Bsequential_9_lstm_19_while_sequential_9_lstm_19_while_loop_counterL
Hsequential_9_lstm_19_while_sequential_9_lstm_19_while_maximum_iterations*
&sequential_9_lstm_19_while_placeholder,
(sequential_9_lstm_19_while_placeholder_1,
(sequential_9_lstm_19_while_placeholder_2,
(sequential_9_lstm_19_while_placeholder_3E
Asequential_9_lstm_19_while_sequential_9_lstm_19_strided_slice_1_0
}sequential_9_lstm_19_while_tensorarrayv2read_tensorlistgetitem_sequential_9_lstm_19_tensorarrayunstack_tensorlistfromtensor_0\
Hsequential_9_lstm_19_while_lstm_cell_19_matmul_readvariableop_resource_0:
Ü^
Jsequential_9_lstm_19_while_lstm_cell_19_matmul_1_readvariableop_resource_0:
×ÜX
Isequential_9_lstm_19_while_lstm_cell_19_biasadd_readvariableop_resource_0:	Ü'
#sequential_9_lstm_19_while_identity)
%sequential_9_lstm_19_while_identity_1)
%sequential_9_lstm_19_while_identity_2)
%sequential_9_lstm_19_while_identity_3)
%sequential_9_lstm_19_while_identity_4)
%sequential_9_lstm_19_while_identity_5C
?sequential_9_lstm_19_while_sequential_9_lstm_19_strided_slice_1
{sequential_9_lstm_19_while_tensorarrayv2read_tensorlistgetitem_sequential_9_lstm_19_tensorarrayunstack_tensorlistfromtensorZ
Fsequential_9_lstm_19_while_lstm_cell_19_matmul_readvariableop_resource:
Ü\
Hsequential_9_lstm_19_while_lstm_cell_19_matmul_1_readvariableop_resource:
×ÜV
Gsequential_9_lstm_19_while_lstm_cell_19_biasadd_readvariableop_resource:	Ü¢>sequential_9/lstm_19/while/lstm_cell_19/BiasAdd/ReadVariableOp¢=sequential_9/lstm_19/while/lstm_cell_19/MatMul/ReadVariableOp¢?sequential_9/lstm_19/while/lstm_cell_19/MatMul_1/ReadVariableOpí
Lsequential_9/lstm_19/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ  2N
Lsequential_9/lstm_19/while/TensorArrayV2Read/TensorListGetItem/element_shapeÒ
>sequential_9/lstm_19/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem}sequential_9_lstm_19_while_tensorarrayv2read_tensorlistgetitem_sequential_9_lstm_19_tensorarrayunstack_tensorlistfromtensor_0&sequential_9_lstm_19_while_placeholderUsequential_9/lstm_19/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype02@
>sequential_9/lstm_19/while/TensorArrayV2Read/TensorListGetItem
=sequential_9/lstm_19/while/lstm_cell_19/MatMul/ReadVariableOpReadVariableOpHsequential_9_lstm_19_while_lstm_cell_19_matmul_readvariableop_resource_0* 
_output_shapes
:
Ü*
dtype02?
=sequential_9/lstm_19/while/lstm_cell_19/MatMul/ReadVariableOp«
.sequential_9/lstm_19/while/lstm_cell_19/MatMulMatMulEsequential_9/lstm_19/while/TensorArrayV2Read/TensorListGetItem:item:0Esequential_9/lstm_19/while/lstm_cell_19/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ20
.sequential_9/lstm_19/while/lstm_cell_19/MatMul
?sequential_9/lstm_19/while/lstm_cell_19/MatMul_1/ReadVariableOpReadVariableOpJsequential_9_lstm_19_while_lstm_cell_19_matmul_1_readvariableop_resource_0* 
_output_shapes
:
×Ü*
dtype02A
?sequential_9/lstm_19/while/lstm_cell_19/MatMul_1/ReadVariableOp
0sequential_9/lstm_19/while/lstm_cell_19/MatMul_1MatMul(sequential_9_lstm_19_while_placeholder_2Gsequential_9/lstm_19/while/lstm_cell_19/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ22
0sequential_9/lstm_19/while/lstm_cell_19/MatMul_1
+sequential_9/lstm_19/while/lstm_cell_19/addAddV28sequential_9/lstm_19/while/lstm_cell_19/MatMul:product:0:sequential_9/lstm_19/while/lstm_cell_19/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2-
+sequential_9/lstm_19/while/lstm_cell_19/add
>sequential_9/lstm_19/while/lstm_cell_19/BiasAdd/ReadVariableOpReadVariableOpIsequential_9_lstm_19_while_lstm_cell_19_biasadd_readvariableop_resource_0*
_output_shapes	
:Ü*
dtype02@
>sequential_9/lstm_19/while/lstm_cell_19/BiasAdd/ReadVariableOp
/sequential_9/lstm_19/while/lstm_cell_19/BiasAddBiasAdd/sequential_9/lstm_19/while/lstm_cell_19/add:z:0Fsequential_9/lstm_19/while/lstm_cell_19/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ21
/sequential_9/lstm_19/while/lstm_cell_19/BiasAdd´
7sequential_9/lstm_19/while/lstm_cell_19/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :29
7sequential_9/lstm_19/while/lstm_cell_19/split/split_dimã
-sequential_9/lstm_19/while/lstm_cell_19/splitSplit@sequential_9/lstm_19/while/lstm_cell_19/split/split_dim:output:08sequential_9/lstm_19/while/lstm_cell_19/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ×:ÿÿÿÿÿÿÿÿÿ×:ÿÿÿÿÿÿÿÿÿ×:ÿÿÿÿÿÿÿÿÿ×*
	num_split2/
-sequential_9/lstm_19/while/lstm_cell_19/splitØ
/sequential_9/lstm_19/while/lstm_cell_19/SigmoidSigmoid6sequential_9/lstm_19/while/lstm_cell_19/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×21
/sequential_9/lstm_19/while/lstm_cell_19/SigmoidÜ
1sequential_9/lstm_19/while/lstm_cell_19/Sigmoid_1Sigmoid6sequential_9/lstm_19/while/lstm_cell_19/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×23
1sequential_9/lstm_19/while/lstm_cell_19/Sigmoid_1õ
+sequential_9/lstm_19/while/lstm_cell_19/mulMul5sequential_9/lstm_19/while/lstm_cell_19/Sigmoid_1:y:0(sequential_9_lstm_19_while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2-
+sequential_9/lstm_19/while/lstm_cell_19/mulÏ
,sequential_9/lstm_19/while/lstm_cell_19/ReluRelu6sequential_9/lstm_19/while/lstm_cell_19/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2.
,sequential_9/lstm_19/while/lstm_cell_19/Relu
-sequential_9/lstm_19/while/lstm_cell_19/mul_1Mul3sequential_9/lstm_19/while/lstm_cell_19/Sigmoid:y:0:sequential_9/lstm_19/while/lstm_cell_19/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2/
-sequential_9/lstm_19/while/lstm_cell_19/mul_1þ
-sequential_9/lstm_19/while/lstm_cell_19/add_1AddV2/sequential_9/lstm_19/while/lstm_cell_19/mul:z:01sequential_9/lstm_19/while/lstm_cell_19/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2/
-sequential_9/lstm_19/while/lstm_cell_19/add_1Ü
1sequential_9/lstm_19/while/lstm_cell_19/Sigmoid_2Sigmoid6sequential_9/lstm_19/while/lstm_cell_19/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×23
1sequential_9/lstm_19/while/lstm_cell_19/Sigmoid_2Î
.sequential_9/lstm_19/while/lstm_cell_19/Relu_1Relu1sequential_9/lstm_19/while/lstm_cell_19/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×20
.sequential_9/lstm_19/while/lstm_cell_19/Relu_1
-sequential_9/lstm_19/while/lstm_cell_19/mul_2Mul5sequential_9/lstm_19/while/lstm_cell_19/Sigmoid_2:y:0<sequential_9/lstm_19/while/lstm_cell_19/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2/
-sequential_9/lstm_19/while/lstm_cell_19/mul_2É
?sequential_9/lstm_19/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem(sequential_9_lstm_19_while_placeholder_1&sequential_9_lstm_19_while_placeholder1sequential_9/lstm_19/while/lstm_cell_19/mul_2:z:0*
_output_shapes
: *
element_dtype02A
?sequential_9/lstm_19/while/TensorArrayV2Write/TensorListSetItem
 sequential_9/lstm_19/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2"
 sequential_9/lstm_19/while/add/y½
sequential_9/lstm_19/while/addAddV2&sequential_9_lstm_19_while_placeholder)sequential_9/lstm_19/while/add/y:output:0*
T0*
_output_shapes
: 2 
sequential_9/lstm_19/while/add
"sequential_9/lstm_19/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2$
"sequential_9/lstm_19/while/add_1/yß
 sequential_9/lstm_19/while/add_1AddV2Bsequential_9_lstm_19_while_sequential_9_lstm_19_while_loop_counter+sequential_9/lstm_19/while/add_1/y:output:0*
T0*
_output_shapes
: 2"
 sequential_9/lstm_19/while/add_1¿
#sequential_9/lstm_19/while/IdentityIdentity$sequential_9/lstm_19/while/add_1:z:0 ^sequential_9/lstm_19/while/NoOp*
T0*
_output_shapes
: 2%
#sequential_9/lstm_19/while/Identityç
%sequential_9/lstm_19/while/Identity_1IdentityHsequential_9_lstm_19_while_sequential_9_lstm_19_while_maximum_iterations ^sequential_9/lstm_19/while/NoOp*
T0*
_output_shapes
: 2'
%sequential_9/lstm_19/while/Identity_1Á
%sequential_9/lstm_19/while/Identity_2Identity"sequential_9/lstm_19/while/add:z:0 ^sequential_9/lstm_19/while/NoOp*
T0*
_output_shapes
: 2'
%sequential_9/lstm_19/while/Identity_2î
%sequential_9/lstm_19/while/Identity_3IdentityOsequential_9/lstm_19/while/TensorArrayV2Write/TensorListSetItem:output_handle:0 ^sequential_9/lstm_19/while/NoOp*
T0*
_output_shapes
: 2'
%sequential_9/lstm_19/while/Identity_3â
%sequential_9/lstm_19/while/Identity_4Identity1sequential_9/lstm_19/while/lstm_cell_19/mul_2:z:0 ^sequential_9/lstm_19/while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2'
%sequential_9/lstm_19/while/Identity_4â
%sequential_9/lstm_19/while/Identity_5Identity1sequential_9/lstm_19/while/lstm_cell_19/add_1:z:0 ^sequential_9/lstm_19/while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2'
%sequential_9/lstm_19/while/Identity_5Ç
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
%sequential_9_lstm_19_while_identity_5.sequential_9/lstm_19/while/Identity_5:output:0"
Gsequential_9_lstm_19_while_lstm_cell_19_biasadd_readvariableop_resourceIsequential_9_lstm_19_while_lstm_cell_19_biasadd_readvariableop_resource_0"
Hsequential_9_lstm_19_while_lstm_cell_19_matmul_1_readvariableop_resourceJsequential_9_lstm_19_while_lstm_cell_19_matmul_1_readvariableop_resource_0"
Fsequential_9_lstm_19_while_lstm_cell_19_matmul_readvariableop_resourceHsequential_9_lstm_19_while_lstm_cell_19_matmul_readvariableop_resource_0"
?sequential_9_lstm_19_while_sequential_9_lstm_19_strided_slice_1Asequential_9_lstm_19_while_sequential_9_lstm_19_strided_slice_1_0"ü
{sequential_9_lstm_19_while_tensorarrayv2read_tensorlistgetitem_sequential_9_lstm_19_tensorarrayunstack_tensorlistfromtensor}sequential_9_lstm_19_while_tensorarrayv2read_tensorlistgetitem_sequential_9_lstm_19_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ×:ÿÿÿÿÿÿÿÿÿ×: : : : : 2
>sequential_9/lstm_19/while/lstm_cell_19/BiasAdd/ReadVariableOp>sequential_9/lstm_19/while/lstm_cell_19/BiasAdd/ReadVariableOp2~
=sequential_9/lstm_19/while/lstm_cell_19/MatMul/ReadVariableOp=sequential_9/lstm_19/while/lstm_cell_19/MatMul/ReadVariableOp2
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
:ÿÿÿÿÿÿÿÿÿ×:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×:

_output_shapes
: :

_output_shapes
: 
»
f
-__inference_dropout_18_layer_call_fn_32582296

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
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_dropout_18_layer_call_and_return_conditional_losses_325805742
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ã
Í
while_cond_32582362
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_32582362___redundant_placeholder06
2while_while_cond_32582362___redundant_placeholder16
2while_while_cond_32582362___redundant_placeholder26
2while_while_cond_32582362___redundant_placeholder3
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
B: : : : :ÿÿÿÿÿÿÿÿÿ×:ÿÿÿÿÿÿÿÿÿ×: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×:

_output_shapes
: :

_output_shapes
:
ã
Í
while_cond_32578782
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_32578782___redundant_placeholder06
2while_while_cond_32578782___redundant_placeholder16
2while_while_cond_32578782___redundant_placeholder26
2while_while_cond_32578782___redundant_placeholder3
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
B: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
É\
¡
E__inference_lstm_19_layer_call_and_return_conditional_losses_32582598
inputs_0?
+lstm_cell_19_matmul_readvariableop_resource:
ÜA
-lstm_cell_19_matmul_1_readvariableop_resource:
×Ü;
,lstm_cell_19_biasadd_readvariableop_resource:	Ü
identity¢#lstm_cell_19/BiasAdd/ReadVariableOp¢"lstm_cell_19/MatMul/ReadVariableOp¢$lstm_cell_19/MatMul_1/ReadVariableOp¢whileF
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
B :×2
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
B :×2
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
:ÿÿÿÿÿÿÿÿÿ×2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :×2
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
B :×2
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
:ÿÿÿÿÿÿÿÿÿ×2	
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
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
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
valueB"ÿÿÿÿ  27
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
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
strided_slice_2¶
"lstm_cell_19/MatMul/ReadVariableOpReadVariableOp+lstm_cell_19_matmul_readvariableop_resource* 
_output_shapes
:
Ü*
dtype02$
"lstm_cell_19/MatMul/ReadVariableOp­
lstm_cell_19/MatMulMatMulstrided_slice_2:output:0*lstm_cell_19/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2
lstm_cell_19/MatMul¼
$lstm_cell_19/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_19_matmul_1_readvariableop_resource* 
_output_shapes
:
×Ü*
dtype02&
$lstm_cell_19/MatMul_1/ReadVariableOp©
lstm_cell_19/MatMul_1MatMulzeros:output:0,lstm_cell_19/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2
lstm_cell_19/MatMul_1 
lstm_cell_19/addAddV2lstm_cell_19/MatMul:product:0lstm_cell_19/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2
lstm_cell_19/add´
#lstm_cell_19/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_19_biasadd_readvariableop_resource*
_output_shapes	
:Ü*
dtype02%
#lstm_cell_19/BiasAdd/ReadVariableOp­
lstm_cell_19/BiasAddBiasAddlstm_cell_19/add:z:0+lstm_cell_19/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2
lstm_cell_19/BiasAdd~
lstm_cell_19/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_19/split/split_dim÷
lstm_cell_19/splitSplit%lstm_cell_19/split/split_dim:output:0lstm_cell_19/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ×:ÿÿÿÿÿÿÿÿÿ×:ÿÿÿÿÿÿÿÿÿ×:ÿÿÿÿÿÿÿÿÿ×*
	num_split2
lstm_cell_19/split
lstm_cell_19/SigmoidSigmoidlstm_cell_19/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2
lstm_cell_19/Sigmoid
lstm_cell_19/Sigmoid_1Sigmoidlstm_cell_19/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2
lstm_cell_19/Sigmoid_1
lstm_cell_19/mulMullstm_cell_19/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2
lstm_cell_19/mul~
lstm_cell_19/ReluRelulstm_cell_19/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2
lstm_cell_19/Relu
lstm_cell_19/mul_1Mullstm_cell_19/Sigmoid:y:0lstm_cell_19/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2
lstm_cell_19/mul_1
lstm_cell_19/add_1AddV2lstm_cell_19/mul:z:0lstm_cell_19/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2
lstm_cell_19/add_1
lstm_cell_19/Sigmoid_2Sigmoidlstm_cell_19/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2
lstm_cell_19/Sigmoid_2}
lstm_cell_19/Relu_1Relulstm_cell_19/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2
lstm_cell_19/Relu_1¡
lstm_cell_19/mul_2Mullstm_cell_19/Sigmoid_2:y:0!lstm_cell_19/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2
lstm_cell_19/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ×   2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_19_matmul_readvariableop_resource-lstm_cell_19_matmul_1_readvariableop_resource,lstm_cell_19_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ×:ÿÿÿÿÿÿÿÿÿ×: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_32582514*
condR
while_cond_32582513*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ×:ÿÿÿÿÿÿÿÿÿ×: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ×   22
0TensorArrayV2Stack/TensorListStack/element_shapeò
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ×*
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
:ÿÿÿÿÿÿÿÿÿ×*
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
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ×2
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
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ×2

IdentityÈ
NoOpNoOp$^lstm_cell_19/BiasAdd/ReadVariableOp#^lstm_cell_19/MatMul/ReadVariableOp%^lstm_cell_19/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 2J
#lstm_cell_19/BiasAdd/ReadVariableOp#lstm_cell_19/BiasAdd/ReadVariableOp2H
"lstm_cell_19/MatMul/ReadVariableOp"lstm_cell_19/MatMul/ReadVariableOp2L
$lstm_cell_19/MatMul_1/ReadVariableOp$lstm_cell_19/MatMul_1/ReadVariableOp2
whilewhile:_ [
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0
¶
¸
*__inference_lstm_18_layer_call_fn_32582269

inputs
unknown:	]Ô
	unknown_0:
Ô
	unknown_1:	Ô
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_lstm_18_layer_call_and_return_conditional_losses_325807412
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

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
&
õ
while_body_32579413
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_01
while_lstm_cell_19_32579437_0:
Ü1
while_lstm_cell_19_32579439_0:
×Ü,
while_lstm_cell_19_32579441_0:	Ü
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor/
while_lstm_cell_19_32579437:
Ü/
while_lstm_cell_19_32579439:
×Ü*
while_lstm_cell_19_32579441:	Ü¢*while/lstm_cell_19/StatefulPartitionedCallÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ  29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÔ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemì
*while/lstm_cell_19/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_19_32579437_0while_lstm_cell_19_32579439_0while_lstm_cell_19_32579441_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ×:ÿÿÿÿÿÿÿÿÿ×:ÿÿÿÿÿÿÿÿÿ×*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_lstm_cell_19_layer_call_and_return_conditional_losses_325793992,
*while/lstm_cell_19/StatefulPartitionedCall÷
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
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3¥
while/Identity_4Identity3while/lstm_cell_19/StatefulPartitionedCall:output:1^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2
while/Identity_4¥
while/Identity_5Identity3while/lstm_cell_19/StatefulPartitionedCall:output:2^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2
while/Identity_5

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
while_lstm_cell_19_32579437while_lstm_cell_19_32579437_0"<
while_lstm_cell_19_32579439while_lstm_cell_19_32579439_0"<
while_lstm_cell_19_32579441while_lstm_cell_19_32579441_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ×:ÿÿÿÿÿÿÿÿÿ×: : : : : 2X
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
:ÿÿÿÿÿÿÿÿÿ×:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×:

_output_shapes
: :

_output_shapes
: 
×
g
H__inference_dropout_19_layer_call_and_return_conditional_losses_32582961

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
:ÿÿÿÿÿÿÿÿÿ×2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape¹
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×*
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
:ÿÿÿÿÿÿÿÿÿ×2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2
dropout/Mul_1j
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ×:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×
 
_user_specified_nameinputs
Ã\
 
E__inference_lstm_18_layer_call_and_return_conditional_losses_32581923
inputs_0>
+lstm_cell_18_matmul_readvariableop_resource:	]ÔA
-lstm_cell_18_matmul_1_readvariableop_resource:
Ô;
,lstm_cell_18_biasadd_readvariableop_resource:	Ô
identity¢#lstm_cell_18/BiasAdd/ReadVariableOp¢"lstm_cell_18/MatMul/ReadVariableOp¢$lstm_cell_18/MatMul_1/ReadVariableOp¢whileF
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
B :2
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
B :2
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
:ÿÿÿÿÿÿÿÿÿ2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :2
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
B :2
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
:ÿÿÿÿÿÿÿÿÿ2	
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
"lstm_cell_18/MatMul/ReadVariableOpReadVariableOp+lstm_cell_18_matmul_readvariableop_resource*
_output_shapes
:	]Ô*
dtype02$
"lstm_cell_18/MatMul/ReadVariableOp­
lstm_cell_18/MatMulMatMulstrided_slice_2:output:0*lstm_cell_18/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÔ2
lstm_cell_18/MatMul¼
$lstm_cell_18/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_18_matmul_1_readvariableop_resource* 
_output_shapes
:
Ô*
dtype02&
$lstm_cell_18/MatMul_1/ReadVariableOp©
lstm_cell_18/MatMul_1MatMulzeros:output:0,lstm_cell_18/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÔ2
lstm_cell_18/MatMul_1 
lstm_cell_18/addAddV2lstm_cell_18/MatMul:product:0lstm_cell_18/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÔ2
lstm_cell_18/add´
#lstm_cell_18/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_18_biasadd_readvariableop_resource*
_output_shapes	
:Ô*
dtype02%
#lstm_cell_18/BiasAdd/ReadVariableOp­
lstm_cell_18/BiasAddBiasAddlstm_cell_18/add:z:0+lstm_cell_18/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÔ2
lstm_cell_18/BiasAdd~
lstm_cell_18/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_18/split/split_dim÷
lstm_cell_18/splitSplit%lstm_cell_18/split/split_dim:output:0lstm_cell_18/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2
lstm_cell_18/split
lstm_cell_18/SigmoidSigmoidlstm_cell_18/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_18/Sigmoid
lstm_cell_18/Sigmoid_1Sigmoidlstm_cell_18/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_18/Sigmoid_1
lstm_cell_18/mulMullstm_cell_18/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_18/mul~
lstm_cell_18/ReluRelulstm_cell_18/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_18/Relu
lstm_cell_18/mul_1Mullstm_cell_18/Sigmoid:y:0lstm_cell_18/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_18/mul_1
lstm_cell_18/add_1AddV2lstm_cell_18/mul:z:0lstm_cell_18/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_18/add_1
lstm_cell_18/Sigmoid_2Sigmoidlstm_cell_18/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_18/Sigmoid_2}
lstm_cell_18/Relu_1Relulstm_cell_18/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_18/Relu_1¡
lstm_cell_18/mul_2Mullstm_cell_18/Sigmoid_2:y:0!lstm_cell_18/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_18/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ  2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_18_matmul_readvariableop_resource-lstm_cell_18_matmul_1_readvariableop_resource,lstm_cell_18_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_32581839*
condR
while_cond_32581838*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ  22
0TensorArrayV2Stack/TensorListStack/element_shapeò
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
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
:ÿÿÿÿÿÿÿÿÿ*
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
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
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
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

IdentityÈ
NoOpNoOp$^lstm_cell_18/BiasAdd/ReadVariableOp#^lstm_cell_18/MatMul/ReadVariableOp%^lstm_cell_18/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ]: : : 2J
#lstm_cell_18/BiasAdd/ReadVariableOp#lstm_cell_18/BiasAdd/ReadVariableOp2H
"lstm_cell_18/MatMul/ReadVariableOp"lstm_cell_18/MatMul/ReadVariableOp2L
$lstm_cell_18/MatMul_1/ReadVariableOp$lstm_cell_18/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ]
"
_user_specified_name
inputs/0
ã
Í
while_cond_32580460
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_32580460___redundant_placeholder06
2while_while_cond_32580460___redundant_placeholder16
2while_while_cond_32580460___redundant_placeholder26
2while_while_cond_32580460___redundant_placeholder3
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
B: : : : :ÿÿÿÿÿÿÿÿÿ×:ÿÿÿÿÿÿÿÿÿ×: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×:

_output_shapes
: :

_output_shapes
:
Ô

í
lstm_19_while_cond_32581459,
(lstm_19_while_lstm_19_while_loop_counter2
.lstm_19_while_lstm_19_while_maximum_iterations
lstm_19_while_placeholder
lstm_19_while_placeholder_1
lstm_19_while_placeholder_2
lstm_19_while_placeholder_3.
*lstm_19_while_less_lstm_19_strided_slice_1F
Blstm_19_while_lstm_19_while_cond_32581459___redundant_placeholder0F
Blstm_19_while_lstm_19_while_cond_32581459___redundant_placeholder1F
Blstm_19_while_lstm_19_while_cond_32581459___redundant_placeholder2F
Blstm_19_while_lstm_19_while_cond_32581459___redundant_placeholder3
lstm_19_while_identity

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
B: : : : :ÿÿÿÿÿÿÿÿÿ×:ÿÿÿÿÿÿÿÿÿ×: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×:

_output_shapes
: :

_output_shapes
:
\

E__inference_lstm_18_layer_call_and_return_conditional_losses_32582074

inputs>
+lstm_cell_18_matmul_readvariableop_resource:	]ÔA
-lstm_cell_18_matmul_1_readvariableop_resource:
Ô;
,lstm_cell_18_biasadd_readvariableop_resource:	Ô
identity¢#lstm_cell_18/BiasAdd/ReadVariableOp¢"lstm_cell_18/MatMul/ReadVariableOp¢$lstm_cell_18/MatMul_1/ReadVariableOp¢whileD
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
B :2
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
B :2
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
:ÿÿÿÿÿÿÿÿÿ2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :2
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
B :2
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
:ÿÿÿÿÿÿÿÿÿ2	
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
"lstm_cell_18/MatMul/ReadVariableOpReadVariableOp+lstm_cell_18_matmul_readvariableop_resource*
_output_shapes
:	]Ô*
dtype02$
"lstm_cell_18/MatMul/ReadVariableOp­
lstm_cell_18/MatMulMatMulstrided_slice_2:output:0*lstm_cell_18/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÔ2
lstm_cell_18/MatMul¼
$lstm_cell_18/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_18_matmul_1_readvariableop_resource* 
_output_shapes
:
Ô*
dtype02&
$lstm_cell_18/MatMul_1/ReadVariableOp©
lstm_cell_18/MatMul_1MatMulzeros:output:0,lstm_cell_18/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÔ2
lstm_cell_18/MatMul_1 
lstm_cell_18/addAddV2lstm_cell_18/MatMul:product:0lstm_cell_18/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÔ2
lstm_cell_18/add´
#lstm_cell_18/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_18_biasadd_readvariableop_resource*
_output_shapes	
:Ô*
dtype02%
#lstm_cell_18/BiasAdd/ReadVariableOp­
lstm_cell_18/BiasAddBiasAddlstm_cell_18/add:z:0+lstm_cell_18/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÔ2
lstm_cell_18/BiasAdd~
lstm_cell_18/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_18/split/split_dim÷
lstm_cell_18/splitSplit%lstm_cell_18/split/split_dim:output:0lstm_cell_18/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2
lstm_cell_18/split
lstm_cell_18/SigmoidSigmoidlstm_cell_18/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_18/Sigmoid
lstm_cell_18/Sigmoid_1Sigmoidlstm_cell_18/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_18/Sigmoid_1
lstm_cell_18/mulMullstm_cell_18/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_18/mul~
lstm_cell_18/ReluRelulstm_cell_18/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_18/Relu
lstm_cell_18/mul_1Mullstm_cell_18/Sigmoid:y:0lstm_cell_18/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_18/mul_1
lstm_cell_18/add_1AddV2lstm_cell_18/mul:z:0lstm_cell_18/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_18/add_1
lstm_cell_18/Sigmoid_2Sigmoidlstm_cell_18/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_18/Sigmoid_2}
lstm_cell_18/Relu_1Relulstm_cell_18/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_18/Relu_1¡
lstm_cell_18/mul_2Mullstm_cell_18/Sigmoid_2:y:0!lstm_cell_18/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_18/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ  2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_18_matmul_readvariableop_resource-lstm_cell_18_matmul_1_readvariableop_resource,lstm_cell_18_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_32581990*
condR
while_cond_32581989*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ  22
0TensorArrayV2Stack/TensorListStack/element_shapeé
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
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
:ÿÿÿÿÿÿÿÿÿ*
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
:ÿÿÿÿÿÿÿÿÿ2
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
:ÿÿÿÿÿÿÿÿÿ2

IdentityÈ
NoOpNoOp$^lstm_cell_18/BiasAdd/ReadVariableOp#^lstm_cell_18/MatMul/ReadVariableOp%^lstm_cell_18/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ]: : : 2J
#lstm_cell_18/BiasAdd/ReadVariableOp#lstm_cell_18/BiasAdd/ReadVariableOp2H
"lstm_cell_18/MatMul/ReadVariableOp"lstm_cell_18/MatMul/ReadVariableOp2L
$lstm_cell_18/MatMul_1/ReadVariableOp$lstm_cell_18/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
 
_user_specified_nameinputs
\

E__inference_lstm_19_layer_call_and_return_conditional_losses_32580545

inputs?
+lstm_cell_19_matmul_readvariableop_resource:
ÜA
-lstm_cell_19_matmul_1_readvariableop_resource:
×Ü;
,lstm_cell_19_biasadd_readvariableop_resource:	Ü
identity¢#lstm_cell_19/BiasAdd/ReadVariableOp¢"lstm_cell_19/MatMul/ReadVariableOp¢$lstm_cell_19/MatMul_1/ReadVariableOp¢whileD
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
B :×2
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
B :×2
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
:ÿÿÿÿÿÿÿÿÿ×2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :×2
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
B :×2
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
:ÿÿÿÿÿÿÿÿÿ×2	
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
:ÿÿÿÿÿÿÿÿÿ2
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
valueB"ÿÿÿÿ  27
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
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
strided_slice_2¶
"lstm_cell_19/MatMul/ReadVariableOpReadVariableOp+lstm_cell_19_matmul_readvariableop_resource* 
_output_shapes
:
Ü*
dtype02$
"lstm_cell_19/MatMul/ReadVariableOp­
lstm_cell_19/MatMulMatMulstrided_slice_2:output:0*lstm_cell_19/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2
lstm_cell_19/MatMul¼
$lstm_cell_19/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_19_matmul_1_readvariableop_resource* 
_output_shapes
:
×Ü*
dtype02&
$lstm_cell_19/MatMul_1/ReadVariableOp©
lstm_cell_19/MatMul_1MatMulzeros:output:0,lstm_cell_19/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2
lstm_cell_19/MatMul_1 
lstm_cell_19/addAddV2lstm_cell_19/MatMul:product:0lstm_cell_19/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2
lstm_cell_19/add´
#lstm_cell_19/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_19_biasadd_readvariableop_resource*
_output_shapes	
:Ü*
dtype02%
#lstm_cell_19/BiasAdd/ReadVariableOp­
lstm_cell_19/BiasAddBiasAddlstm_cell_19/add:z:0+lstm_cell_19/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÜ2
lstm_cell_19/BiasAdd~
lstm_cell_19/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_19/split/split_dim÷
lstm_cell_19/splitSplit%lstm_cell_19/split/split_dim:output:0lstm_cell_19/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ×:ÿÿÿÿÿÿÿÿÿ×:ÿÿÿÿÿÿÿÿÿ×:ÿÿÿÿÿÿÿÿÿ×*
	num_split2
lstm_cell_19/split
lstm_cell_19/SigmoidSigmoidlstm_cell_19/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2
lstm_cell_19/Sigmoid
lstm_cell_19/Sigmoid_1Sigmoidlstm_cell_19/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2
lstm_cell_19/Sigmoid_1
lstm_cell_19/mulMullstm_cell_19/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2
lstm_cell_19/mul~
lstm_cell_19/ReluRelulstm_cell_19/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2
lstm_cell_19/Relu
lstm_cell_19/mul_1Mullstm_cell_19/Sigmoid:y:0lstm_cell_19/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2
lstm_cell_19/mul_1
lstm_cell_19/add_1AddV2lstm_cell_19/mul:z:0lstm_cell_19/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2
lstm_cell_19/add_1
lstm_cell_19/Sigmoid_2Sigmoidlstm_cell_19/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2
lstm_cell_19/Sigmoid_2}
lstm_cell_19/Relu_1Relulstm_cell_19/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2
lstm_cell_19/Relu_1¡
lstm_cell_19/mul_2Mullstm_cell_19/Sigmoid_2:y:0!lstm_cell_19/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×2
lstm_cell_19/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ×   2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_19_matmul_readvariableop_resource-lstm_cell_19_matmul_1_readvariableop_resource,lstm_cell_19_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ×:ÿÿÿÿÿÿÿÿÿ×: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_32580461*
condR
while_cond_32580460*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ×:ÿÿÿÿÿÿÿÿÿ×: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ×   22
0TensorArrayV2Stack/TensorListStack/element_shapeé
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×*
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
:ÿÿÿÿÿÿÿÿÿ×*
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
:ÿÿÿÿÿÿÿÿÿ×2
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
:ÿÿÿÿÿÿÿÿÿ×2

IdentityÈ
NoOpNoOp$^lstm_cell_19/BiasAdd/ReadVariableOp#^lstm_cell_19/MatMul/ReadVariableOp%^lstm_cell_19/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:ÿÿÿÿÿÿÿÿÿ: : : 2J
#lstm_cell_19/BiasAdd/ReadVariableOp#lstm_cell_19/BiasAdd/ReadVariableOp2H
"lstm_cell_19/MatMul/ReadVariableOp"lstm_cell_19/MatMul/ReadVariableOp2L
$lstm_cell_19/MatMul_1/ReadVariableOp$lstm_cell_19/MatMul_1/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
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
lstm_18_input:
serving_default_lstm_18_input:0ÿÿÿÿÿÿÿÿÿ]?
dense_94
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:ô²
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
­
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
­
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
!:	×2dense_9/kernel
:2dense_9/bias
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
.:,	]Ô2lstm_18/lstm_cell_18/kernel
9:7
Ô2%lstm_18/lstm_cell_18/recurrent_kernel
(:&Ô2lstm_18/lstm_cell_18/bias
/:-
Ü2lstm_19/lstm_cell_19/kernel
9:7
×Ü2%lstm_19/lstm_cell_19/recurrent_kernel
(:&Ü2lstm_19/lstm_cell_19/bias
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
­
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
­
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
ÔBÑ
#__inference__wrapped_model_32578694lstm_18_input"
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
J__inference_sequential_9_layer_call_and_return_conditional_losses_32581238
J__inference_sequential_9_layer_call_and_return_conditional_losses_32581579
J__inference_sequential_9_layer_call_and_return_conditional_losses_32580863
J__inference_sequential_9_layer_call_and_return_conditional_losses_32580888À
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
/__inference_sequential_9_layer_call_fn_32580348
/__inference_sequential_9_layer_call_fn_32581600
/__inference_sequential_9_layer_call_fn_32581621
/__inference_sequential_9_layer_call_fn_32580838À
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
E__inference_lstm_18_layer_call_and_return_conditional_losses_32581772
E__inference_lstm_18_layer_call_and_return_conditional_losses_32581923
E__inference_lstm_18_layer_call_and_return_conditional_losses_32582074
E__inference_lstm_18_layer_call_and_return_conditional_losses_32582225Õ
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
*__inference_lstm_18_layer_call_fn_32582236
*__inference_lstm_18_layer_call_fn_32582247
*__inference_lstm_18_layer_call_fn_32582258
*__inference_lstm_18_layer_call_fn_32582269Õ
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
H__inference_dropout_18_layer_call_and_return_conditional_losses_32582274
H__inference_dropout_18_layer_call_and_return_conditional_losses_32582286´
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
-__inference_dropout_18_layer_call_fn_32582291
-__inference_dropout_18_layer_call_fn_32582296´
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
E__inference_lstm_19_layer_call_and_return_conditional_losses_32582447
E__inference_lstm_19_layer_call_and_return_conditional_losses_32582598
E__inference_lstm_19_layer_call_and_return_conditional_losses_32582749
E__inference_lstm_19_layer_call_and_return_conditional_losses_32582900Õ
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
*__inference_lstm_19_layer_call_fn_32582911
*__inference_lstm_19_layer_call_fn_32582922
*__inference_lstm_19_layer_call_fn_32582933
*__inference_lstm_19_layer_call_fn_32582944Õ
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
H__inference_dropout_19_layer_call_and_return_conditional_losses_32582949
H__inference_dropout_19_layer_call_and_return_conditional_losses_32582961´
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
-__inference_dropout_19_layer_call_fn_32582966
-__inference_dropout_19_layer_call_fn_32582971´
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
E__inference_dense_9_layer_call_and_return_conditional_losses_32583002¢
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
*__inference_dense_9_layer_call_fn_32583011¢
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
&__inference_signature_wrapper_32580911lstm_18_input"
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
J__inference_lstm_cell_18_layer_call_and_return_conditional_losses_32583043
J__inference_lstm_cell_18_layer_call_and_return_conditional_losses_32583075¾
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
/__inference_lstm_cell_18_layer_call_fn_32583092
/__inference_lstm_cell_18_layer_call_fn_32583109¾
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
J__inference_lstm_cell_19_layer_call_and_return_conditional_losses_32583141
J__inference_lstm_cell_19_layer_call_and_return_conditional_losses_32583173¾
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
/__inference_lstm_cell_19_layer_call_fn_32583190
/__inference_lstm_cell_19_layer_call_fn_32583207¾
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
#__inference__wrapped_model_32578694}&'()*+ !:¢7
0¢-
+(
lstm_18_inputÿÿÿÿÿÿÿÿÿ]
ª "5ª2
0
dense_9%"
dense_9ÿÿÿÿÿÿÿÿÿ®
E__inference_dense_9_layer_call_and_return_conditional_losses_32583002e !4¢1
*¢'
%"
inputsÿÿÿÿÿÿÿÿÿ×
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ
 
*__inference_dense_9_layer_call_fn_32583011X !4¢1
*¢'
%"
inputsÿÿÿÿÿÿÿÿÿ×
ª "ÿÿÿÿÿÿÿÿÿ²
H__inference_dropout_18_layer_call_and_return_conditional_losses_32582274f8¢5
.¢+
%"
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "*¢'
 
0ÿÿÿÿÿÿÿÿÿ
 ²
H__inference_dropout_18_layer_call_and_return_conditional_losses_32582286f8¢5
.¢+
%"
inputsÿÿÿÿÿÿÿÿÿ
p
ª "*¢'
 
0ÿÿÿÿÿÿÿÿÿ
 
-__inference_dropout_18_layer_call_fn_32582291Y8¢5
.¢+
%"
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "ÿÿÿÿÿÿÿÿÿ
-__inference_dropout_18_layer_call_fn_32582296Y8¢5
.¢+
%"
inputsÿÿÿÿÿÿÿÿÿ
p
ª "ÿÿÿÿÿÿÿÿÿ²
H__inference_dropout_19_layer_call_and_return_conditional_losses_32582949f8¢5
.¢+
%"
inputsÿÿÿÿÿÿÿÿÿ×
p 
ª "*¢'
 
0ÿÿÿÿÿÿÿÿÿ×
 ²
H__inference_dropout_19_layer_call_and_return_conditional_losses_32582961f8¢5
.¢+
%"
inputsÿÿÿÿÿÿÿÿÿ×
p
ª "*¢'
 
0ÿÿÿÿÿÿÿÿÿ×
 
-__inference_dropout_19_layer_call_fn_32582966Y8¢5
.¢+
%"
inputsÿÿÿÿÿÿÿÿÿ×
p 
ª "ÿÿÿÿÿÿÿÿÿ×
-__inference_dropout_19_layer_call_fn_32582971Y8¢5
.¢+
%"
inputsÿÿÿÿÿÿÿÿÿ×
p
ª "ÿÿÿÿÿÿÿÿÿ×Õ
E__inference_lstm_18_layer_call_and_return_conditional_losses_32581772&'(O¢L
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
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Õ
E__inference_lstm_18_layer_call_and_return_conditional_losses_32581923&'(O¢L
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
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 »
E__inference_lstm_18_layer_call_and_return_conditional_losses_32582074r&'(?¢<
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
0ÿÿÿÿÿÿÿÿÿ
 »
E__inference_lstm_18_layer_call_and_return_conditional_losses_32582225r&'(?¢<
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
0ÿÿÿÿÿÿÿÿÿ
 ¬
*__inference_lstm_18_layer_call_fn_32582236~&'(O¢L
E¢B
41
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ]

 
p 

 
ª "&#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬
*__inference_lstm_18_layer_call_fn_32582247~&'(O¢L
E¢B
41
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ]

 
p

 
ª "&#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
*__inference_lstm_18_layer_call_fn_32582258e&'(?¢<
5¢2
$!
inputsÿÿÿÿÿÿÿÿÿ]

 
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
*__inference_lstm_18_layer_call_fn_32582269e&'(?¢<
5¢2
$!
inputsÿÿÿÿÿÿÿÿÿ]

 
p

 
ª "ÿÿÿÿÿÿÿÿÿÖ
E__inference_lstm_19_layer_call_and_return_conditional_losses_32582447)*+P¢M
F¢C
52
0-
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
p 

 
ª "3¢0
)&
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ×
 Ö
E__inference_lstm_19_layer_call_and_return_conditional_losses_32582598)*+P¢M
F¢C
52
0-
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
p

 
ª "3¢0
)&
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ×
 ¼
E__inference_lstm_19_layer_call_and_return_conditional_losses_32582749s)*+@¢=
6¢3
%"
inputsÿÿÿÿÿÿÿÿÿ

 
p 

 
ª "*¢'
 
0ÿÿÿÿÿÿÿÿÿ×
 ¼
E__inference_lstm_19_layer_call_and_return_conditional_losses_32582900s)*+@¢=
6¢3
%"
inputsÿÿÿÿÿÿÿÿÿ

 
p

 
ª "*¢'
 
0ÿÿÿÿÿÿÿÿÿ×
 ­
*__inference_lstm_19_layer_call_fn_32582911)*+P¢M
F¢C
52
0-
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
p 

 
ª "&#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ×­
*__inference_lstm_19_layer_call_fn_32582922)*+P¢M
F¢C
52
0-
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
p

 
ª "&#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ×
*__inference_lstm_19_layer_call_fn_32582933f)*+@¢=
6¢3
%"
inputsÿÿÿÿÿÿÿÿÿ

 
p 

 
ª "ÿÿÿÿÿÿÿÿÿ×
*__inference_lstm_19_layer_call_fn_32582944f)*+@¢=
6¢3
%"
inputsÿÿÿÿÿÿÿÿÿ

 
p

 
ª "ÿÿÿÿÿÿÿÿÿ×Ñ
J__inference_lstm_cell_18_layer_call_and_return_conditional_losses_32583043&'(¢
x¢u
 
inputsÿÿÿÿÿÿÿÿÿ]
M¢J
# 
states/0ÿÿÿÿÿÿÿÿÿ
# 
states/1ÿÿÿÿÿÿÿÿÿ
p 
ª "v¢s
l¢i

0/0ÿÿÿÿÿÿÿÿÿ
GD
 
0/1/0ÿÿÿÿÿÿÿÿÿ
 
0/1/1ÿÿÿÿÿÿÿÿÿ
 Ñ
J__inference_lstm_cell_18_layer_call_and_return_conditional_losses_32583075&'(¢
x¢u
 
inputsÿÿÿÿÿÿÿÿÿ]
M¢J
# 
states/0ÿÿÿÿÿÿÿÿÿ
# 
states/1ÿÿÿÿÿÿÿÿÿ
p
ª "v¢s
l¢i

0/0ÿÿÿÿÿÿÿÿÿ
GD
 
0/1/0ÿÿÿÿÿÿÿÿÿ
 
0/1/1ÿÿÿÿÿÿÿÿÿ
 ¦
/__inference_lstm_cell_18_layer_call_fn_32583092ò&'(¢
x¢u
 
inputsÿÿÿÿÿÿÿÿÿ]
M¢J
# 
states/0ÿÿÿÿÿÿÿÿÿ
# 
states/1ÿÿÿÿÿÿÿÿÿ
p 
ª "f¢c

0ÿÿÿÿÿÿÿÿÿ
C@

1/0ÿÿÿÿÿÿÿÿÿ

1/1ÿÿÿÿÿÿÿÿÿ¦
/__inference_lstm_cell_18_layer_call_fn_32583109ò&'(¢
x¢u
 
inputsÿÿÿÿÿÿÿÿÿ]
M¢J
# 
states/0ÿÿÿÿÿÿÿÿÿ
# 
states/1ÿÿÿÿÿÿÿÿÿ
p
ª "f¢c

0ÿÿÿÿÿÿÿÿÿ
C@

1/0ÿÿÿÿÿÿÿÿÿ

1/1ÿÿÿÿÿÿÿÿÿÓ
J__inference_lstm_cell_19_layer_call_and_return_conditional_losses_32583141)*+¢
y¢v
!
inputsÿÿÿÿÿÿÿÿÿ
M¢J
# 
states/0ÿÿÿÿÿÿÿÿÿ×
# 
states/1ÿÿÿÿÿÿÿÿÿ×
p 
ª "v¢s
l¢i

0/0ÿÿÿÿÿÿÿÿÿ×
GD
 
0/1/0ÿÿÿÿÿÿÿÿÿ×
 
0/1/1ÿÿÿÿÿÿÿÿÿ×
 Ó
J__inference_lstm_cell_19_layer_call_and_return_conditional_losses_32583173)*+¢
y¢v
!
inputsÿÿÿÿÿÿÿÿÿ
M¢J
# 
states/0ÿÿÿÿÿÿÿÿÿ×
# 
states/1ÿÿÿÿÿÿÿÿÿ×
p
ª "v¢s
l¢i

0/0ÿÿÿÿÿÿÿÿÿ×
GD
 
0/1/0ÿÿÿÿÿÿÿÿÿ×
 
0/1/1ÿÿÿÿÿÿÿÿÿ×
 ¨
/__inference_lstm_cell_19_layer_call_fn_32583190ô)*+¢
y¢v
!
inputsÿÿÿÿÿÿÿÿÿ
M¢J
# 
states/0ÿÿÿÿÿÿÿÿÿ×
# 
states/1ÿÿÿÿÿÿÿÿÿ×
p 
ª "f¢c

0ÿÿÿÿÿÿÿÿÿ×
C@

1/0ÿÿÿÿÿÿÿÿÿ×

1/1ÿÿÿÿÿÿÿÿÿ×¨
/__inference_lstm_cell_19_layer_call_fn_32583207ô)*+¢
y¢v
!
inputsÿÿÿÿÿÿÿÿÿ
M¢J
# 
states/0ÿÿÿÿÿÿÿÿÿ×
# 
states/1ÿÿÿÿÿÿÿÿÿ×
p
ª "f¢c

0ÿÿÿÿÿÿÿÿÿ×
C@

1/0ÿÿÿÿÿÿÿÿÿ×

1/1ÿÿÿÿÿÿÿÿÿ×Ç
J__inference_sequential_9_layer_call_and_return_conditional_losses_32580863y&'()*+ !B¢?
8¢5
+(
lstm_18_inputÿÿÿÿÿÿÿÿÿ]
p 

 
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ
 Ç
J__inference_sequential_9_layer_call_and_return_conditional_losses_32580888y&'()*+ !B¢?
8¢5
+(
lstm_18_inputÿÿÿÿÿÿÿÿÿ]
p

 
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ
 À
J__inference_sequential_9_layer_call_and_return_conditional_losses_32581238r&'()*+ !;¢8
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
J__inference_sequential_9_layer_call_and_return_conditional_losses_32581579r&'()*+ !;¢8
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
/__inference_sequential_9_layer_call_fn_32580348l&'()*+ !B¢?
8¢5
+(
lstm_18_inputÿÿÿÿÿÿÿÿÿ]
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
/__inference_sequential_9_layer_call_fn_32580838l&'()*+ !B¢?
8¢5
+(
lstm_18_inputÿÿÿÿÿÿÿÿÿ]
p

 
ª "ÿÿÿÿÿÿÿÿÿ
/__inference_sequential_9_layer_call_fn_32581600e&'()*+ !;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿ]
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
/__inference_sequential_9_layer_call_fn_32581621e&'()*+ !;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿ]
p

 
ª "ÿÿÿÿÿÿÿÿÿ¹
&__inference_signature_wrapper_32580911&'()*+ !K¢H
¢ 
Aª>
<
lstm_18_input+(
lstm_18_inputÿÿÿÿÿÿÿÿÿ]"5ª2
0
dense_9%"
dense_9ÿÿÿÿÿÿÿÿÿ