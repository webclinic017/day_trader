ù'
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
"serve*2.6.02v2.6.0-rc2-32-g919f693420e8«&
{
dense_10/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	Å* 
shared_namedense_10/kernel
t
#dense_10/kernel/Read/ReadVariableOpReadVariableOpdense_10/kernel*
_output_shapes
:	Å*
dtype0
r
dense_10/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_10/bias
k
!dense_10/bias/Read/ReadVariableOpReadVariableOpdense_10/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0

lstm_20/lstm_cell_20/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	]*,
shared_namelstm_20/lstm_cell_20/kernel

/lstm_20/lstm_cell_20/kernel/Read/ReadVariableOpReadVariableOplstm_20/lstm_cell_20/kernel*
_output_shapes
:	]*
dtype0
¨
%lstm_20/lstm_cell_20/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*6
shared_name'%lstm_20/lstm_cell_20/recurrent_kernel
¡
9lstm_20/lstm_cell_20/recurrent_kernel/Read/ReadVariableOpReadVariableOp%lstm_20/lstm_cell_20/recurrent_kernel* 
_output_shapes
:
*
dtype0

lstm_20/lstm_cell_20/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_namelstm_20/lstm_cell_20/bias

-lstm_20/lstm_cell_20/bias/Read/ReadVariableOpReadVariableOplstm_20/lstm_cell_20/bias*
_output_shapes	
:*
dtype0

lstm_21/lstm_cell_21/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:

*,
shared_namelstm_21/lstm_cell_21/kernel

/lstm_21/lstm_cell_21/kernel/Read/ReadVariableOpReadVariableOplstm_21/lstm_cell_21/kernel* 
_output_shapes
:

*
dtype0
¨
%lstm_21/lstm_cell_21/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
Å
*6
shared_name'%lstm_21/lstm_cell_21/recurrent_kernel
¡
9lstm_21/lstm_cell_21/recurrent_kernel/Read/ReadVariableOpReadVariableOp%lstm_21/lstm_cell_21/recurrent_kernel* 
_output_shapes
:
Å
*
dtype0

lstm_21/lstm_cell_21/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
**
shared_namelstm_21/lstm_cell_21/bias

-lstm_21/lstm_cell_21/bias/Read/ReadVariableOpReadVariableOplstm_21/lstm_cell_21/bias*
_output_shapes	
:
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

Adam/dense_10/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	Å*'
shared_nameAdam/dense_10/kernel/m

*Adam/dense_10/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_10/kernel/m*
_output_shapes
:	Å*
dtype0

Adam/dense_10/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_10/bias/m
y
(Adam/dense_10/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_10/bias/m*
_output_shapes
:*
dtype0
¡
"Adam/lstm_20/lstm_cell_20/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	]*3
shared_name$"Adam/lstm_20/lstm_cell_20/kernel/m

6Adam/lstm_20/lstm_cell_20/kernel/m/Read/ReadVariableOpReadVariableOp"Adam/lstm_20/lstm_cell_20/kernel/m*
_output_shapes
:	]*
dtype0
¶
,Adam/lstm_20/lstm_cell_20/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*=
shared_name.,Adam/lstm_20/lstm_cell_20/recurrent_kernel/m
¯
@Adam/lstm_20/lstm_cell_20/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp,Adam/lstm_20/lstm_cell_20/recurrent_kernel/m* 
_output_shapes
:
*
dtype0

 Adam/lstm_20/lstm_cell_20/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" Adam/lstm_20/lstm_cell_20/bias/m

4Adam/lstm_20/lstm_cell_20/bias/m/Read/ReadVariableOpReadVariableOp Adam/lstm_20/lstm_cell_20/bias/m*
_output_shapes	
:*
dtype0
¢
"Adam/lstm_21/lstm_cell_21/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:

*3
shared_name$"Adam/lstm_21/lstm_cell_21/kernel/m

6Adam/lstm_21/lstm_cell_21/kernel/m/Read/ReadVariableOpReadVariableOp"Adam/lstm_21/lstm_cell_21/kernel/m* 
_output_shapes
:

*
dtype0
¶
,Adam/lstm_21/lstm_cell_21/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
Å
*=
shared_name.,Adam/lstm_21/lstm_cell_21/recurrent_kernel/m
¯
@Adam/lstm_21/lstm_cell_21/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp,Adam/lstm_21/lstm_cell_21/recurrent_kernel/m* 
_output_shapes
:
Å
*
dtype0

 Adam/lstm_21/lstm_cell_21/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*1
shared_name" Adam/lstm_21/lstm_cell_21/bias/m

4Adam/lstm_21/lstm_cell_21/bias/m/Read/ReadVariableOpReadVariableOp Adam/lstm_21/lstm_cell_21/bias/m*
_output_shapes	
:
*
dtype0

Adam/dense_10/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	Å*'
shared_nameAdam/dense_10/kernel/v

*Adam/dense_10/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_10/kernel/v*
_output_shapes
:	Å*
dtype0

Adam/dense_10/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_10/bias/v
y
(Adam/dense_10/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_10/bias/v*
_output_shapes
:*
dtype0
¡
"Adam/lstm_20/lstm_cell_20/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	]*3
shared_name$"Adam/lstm_20/lstm_cell_20/kernel/v

6Adam/lstm_20/lstm_cell_20/kernel/v/Read/ReadVariableOpReadVariableOp"Adam/lstm_20/lstm_cell_20/kernel/v*
_output_shapes
:	]*
dtype0
¶
,Adam/lstm_20/lstm_cell_20/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*=
shared_name.,Adam/lstm_20/lstm_cell_20/recurrent_kernel/v
¯
@Adam/lstm_20/lstm_cell_20/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp,Adam/lstm_20/lstm_cell_20/recurrent_kernel/v* 
_output_shapes
:
*
dtype0

 Adam/lstm_20/lstm_cell_20/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" Adam/lstm_20/lstm_cell_20/bias/v

4Adam/lstm_20/lstm_cell_20/bias/v/Read/ReadVariableOpReadVariableOp Adam/lstm_20/lstm_cell_20/bias/v*
_output_shapes	
:*
dtype0
¢
"Adam/lstm_21/lstm_cell_21/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:

*3
shared_name$"Adam/lstm_21/lstm_cell_21/kernel/v

6Adam/lstm_21/lstm_cell_21/kernel/v/Read/ReadVariableOpReadVariableOp"Adam/lstm_21/lstm_cell_21/kernel/v* 
_output_shapes
:

*
dtype0
¶
,Adam/lstm_21/lstm_cell_21/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
Å
*=
shared_name.,Adam/lstm_21/lstm_cell_21/recurrent_kernel/v
¯
@Adam/lstm_21/lstm_cell_21/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp,Adam/lstm_21/lstm_cell_21/recurrent_kernel/v* 
_output_shapes
:
Å
*
dtype0

 Adam/lstm_21/lstm_cell_21/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*1
shared_name" Adam/lstm_21/lstm_cell_21/bias/v

4Adam/lstm_21/lstm_cell_21/bias/v/Read/ReadVariableOpReadVariableOp Adam/lstm_21/lstm_cell_21/bias/v*
_output_shapes	
:
*
dtype0

NoOpNoOp
þ7
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*¹7
value¯7B¬7 B¥7
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
Ð
&iter

'beta_1

(beta_2
	)decay
*learning_rate mp!mq+mr,ms-mt.mu/mv0mw vx!vy+vz,v{-v|.v}/v~0v
8
+0
,1
-2
.3
/4
05
 6
!7
8
+0
,1
-2
.3
/4
05
 6
!7
 
­
trainable_variables
1layer_metrics
2layer_regularization_losses
	variables

3layers
4non_trainable_variables
	regularization_losses
5metrics
 

6
state_size

+kernel
,recurrent_kernel
-bias
7trainable_variables
8	variables
9regularization_losses
:	keras_api
 

+0
,1
-2

+0
,1
-2
 
¹
trainable_variables
;layer_metrics
<layer_regularization_losses
	variables

=layers
>non_trainable_variables
regularization_losses

?states
@metrics
 
 
 
­
trainable_variables
Alayer_metrics
Blayer_regularization_losses
	variables

Clayers
Dnon_trainable_variables
regularization_losses
Emetrics

F
state_size

.kernel
/recurrent_kernel
0bias
Gtrainable_variables
H	variables
Iregularization_losses
J	keras_api
 

.0
/1
02

.0
/1
02
 
¹
trainable_variables
Klayer_metrics
Llayer_regularization_losses
	variables

Mlayers
Nnon_trainable_variables
regularization_losses

Ostates
Pmetrics
 
 
 
­
trainable_variables
Qlayer_metrics
Rlayer_regularization_losses
	variables

Slayers
Tnon_trainable_variables
regularization_losses
Umetrics
[Y
VARIABLE_VALUEdense_10/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_10/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

 0
!1

 0
!1
 
­
"trainable_variables
Vlayer_metrics
Wlayer_regularization_losses
#	variables

Xlayers
Ynon_trainable_variables
$regularization_losses
Zmetrics
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUElstm_20/lstm_cell_20/kernel0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUE
ki
VARIABLE_VALUE%lstm_20/lstm_cell_20/recurrent_kernel0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUElstm_20/lstm_cell_20/bias0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUElstm_21/lstm_cell_21/kernel0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUE
ki
VARIABLE_VALUE%lstm_21/lstm_cell_21/recurrent_kernel0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUElstm_21/lstm_cell_21/bias0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUE
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
[0
\1
 

+0
,1
-2

+0
,1
-2
 
­
7trainable_variables
]layer_metrics
^layer_regularization_losses
8	variables

_layers
`non_trainable_variables
9regularization_losses
ametrics
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
.0
/1
02

.0
/1
02
 
­
Gtrainable_variables
blayer_metrics
clayer_regularization_losses
H	variables

dlayers
enon_trainable_variables
Iregularization_losses
fmetrics
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
	gtotal
	hcount
i	variables
j	keras_api
D
	ktotal
	lcount
m
_fn_kwargs
n	variables
o	keras_api
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
g0
h1

i	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

k0
l1

n	variables
~|
VARIABLE_VALUEAdam/dense_10/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_10/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE"Adam/lstm_20/lstm_cell_20/kernel/mLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE,Adam/lstm_20/lstm_cell_20/recurrent_kernel/mLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE Adam/lstm_20/lstm_cell_20/bias/mLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE"Adam/lstm_21/lstm_cell_21/kernel/mLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE,Adam/lstm_21/lstm_cell_21/recurrent_kernel/mLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE Adam/lstm_21/lstm_cell_21/bias/mLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_10/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_10/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE"Adam/lstm_20/lstm_cell_20/kernel/vLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE,Adam/lstm_20/lstm_cell_20/recurrent_kernel/vLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE Adam/lstm_20/lstm_cell_20/bias/vLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE"Adam/lstm_21/lstm_cell_21/kernel/vLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE,Adam/lstm_21/lstm_cell_21/recurrent_kernel/vLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE Adam/lstm_21/lstm_cell_21/bias/vLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

serving_default_lstm_20_inputPlaceholder*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]*
dtype0* 
shape:ÿÿÿÿÿÿÿÿÿ]
¬
StatefulPartitionedCallStatefulPartitionedCallserving_default_lstm_20_inputlstm_20/lstm_cell_20/kernel%lstm_20/lstm_cell_20/recurrent_kernellstm_20/lstm_cell_20/biaslstm_21/lstm_cell_21/kernel%lstm_21/lstm_cell_21/recurrent_kernellstm_21/lstm_cell_21/biasdense_10/kerneldense_10/bias*
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
&__inference_signature_wrapper_39103348
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
Á
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#dense_10/kernel/Read/ReadVariableOp!dense_10/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp/lstm_20/lstm_cell_20/kernel/Read/ReadVariableOp9lstm_20/lstm_cell_20/recurrent_kernel/Read/ReadVariableOp-lstm_20/lstm_cell_20/bias/Read/ReadVariableOp/lstm_21/lstm_cell_21/kernel/Read/ReadVariableOp9lstm_21/lstm_cell_21/recurrent_kernel/Read/ReadVariableOp-lstm_21/lstm_cell_21/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp*Adam/dense_10/kernel/m/Read/ReadVariableOp(Adam/dense_10/bias/m/Read/ReadVariableOp6Adam/lstm_20/lstm_cell_20/kernel/m/Read/ReadVariableOp@Adam/lstm_20/lstm_cell_20/recurrent_kernel/m/Read/ReadVariableOp4Adam/lstm_20/lstm_cell_20/bias/m/Read/ReadVariableOp6Adam/lstm_21/lstm_cell_21/kernel/m/Read/ReadVariableOp@Adam/lstm_21/lstm_cell_21/recurrent_kernel/m/Read/ReadVariableOp4Adam/lstm_21/lstm_cell_21/bias/m/Read/ReadVariableOp*Adam/dense_10/kernel/v/Read/ReadVariableOp(Adam/dense_10/bias/v/Read/ReadVariableOp6Adam/lstm_20/lstm_cell_20/kernel/v/Read/ReadVariableOp@Adam/lstm_20/lstm_cell_20/recurrent_kernel/v/Read/ReadVariableOp4Adam/lstm_20/lstm_cell_20/bias/v/Read/ReadVariableOp6Adam/lstm_21/lstm_cell_21/kernel/v/Read/ReadVariableOp@Adam/lstm_21/lstm_cell_21/recurrent_kernel/v/Read/ReadVariableOp4Adam/lstm_21/lstm_cell_21/bias/v/Read/ReadVariableOpConst*.
Tin'
%2#	*
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
!__inference__traced_save_39105766
¨	
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_10/kerneldense_10/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratelstm_20/lstm_cell_20/kernel%lstm_20/lstm_cell_20/recurrent_kernellstm_20/lstm_cell_20/biaslstm_21/lstm_cell_21/kernel%lstm_21/lstm_cell_21/recurrent_kernellstm_21/lstm_cell_21/biastotalcounttotal_1count_1Adam/dense_10/kernel/mAdam/dense_10/bias/m"Adam/lstm_20/lstm_cell_20/kernel/m,Adam/lstm_20/lstm_cell_20/recurrent_kernel/m Adam/lstm_20/lstm_cell_20/bias/m"Adam/lstm_21/lstm_cell_21/kernel/m,Adam/lstm_21/lstm_cell_21/recurrent_kernel/m Adam/lstm_21/lstm_cell_21/bias/mAdam/dense_10/kernel/vAdam/dense_10/bias/v"Adam/lstm_20/lstm_cell_20/kernel/v,Adam/lstm_20/lstm_cell_20/recurrent_kernel/v Adam/lstm_20/lstm_cell_20/bias/v"Adam/lstm_21/lstm_cell_21/kernel/v,Adam/lstm_21/lstm_cell_21/recurrent_kernel/v Adam/lstm_21/lstm_cell_21/bias/v*-
Tin&
$2"*
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
$__inference__traced_restore_39105875ÿ$
Ò

$__inference__traced_restore_39105875
file_prefix3
 assignvariableop_dense_10_kernel:	Å.
 assignvariableop_1_dense_10_bias:&
assignvariableop_2_adam_iter:	 (
assignvariableop_3_adam_beta_1: (
assignvariableop_4_adam_beta_2: '
assignvariableop_5_adam_decay: /
%assignvariableop_6_adam_learning_rate: A
.assignvariableop_7_lstm_20_lstm_cell_20_kernel:	]L
8assignvariableop_8_lstm_20_lstm_cell_20_recurrent_kernel:
;
,assignvariableop_9_lstm_20_lstm_cell_20_bias:	C
/assignvariableop_10_lstm_21_lstm_cell_21_kernel:

M
9assignvariableop_11_lstm_21_lstm_cell_21_recurrent_kernel:
Å
<
-assignvariableop_12_lstm_21_lstm_cell_21_bias:	
#
assignvariableop_13_total: #
assignvariableop_14_count: %
assignvariableop_15_total_1: %
assignvariableop_16_count_1: =
*assignvariableop_17_adam_dense_10_kernel_m:	Å6
(assignvariableop_18_adam_dense_10_bias_m:I
6assignvariableop_19_adam_lstm_20_lstm_cell_20_kernel_m:	]T
@assignvariableop_20_adam_lstm_20_lstm_cell_20_recurrent_kernel_m:
C
4assignvariableop_21_adam_lstm_20_lstm_cell_20_bias_m:	J
6assignvariableop_22_adam_lstm_21_lstm_cell_21_kernel_m:

T
@assignvariableop_23_adam_lstm_21_lstm_cell_21_recurrent_kernel_m:
Å
C
4assignvariableop_24_adam_lstm_21_lstm_cell_21_bias_m:	
=
*assignvariableop_25_adam_dense_10_kernel_v:	Å6
(assignvariableop_26_adam_dense_10_bias_v:I
6assignvariableop_27_adam_lstm_20_lstm_cell_20_kernel_v:	]T
@assignvariableop_28_adam_lstm_20_lstm_cell_20_recurrent_kernel_v:
C
4assignvariableop_29_adam_lstm_20_lstm_cell_20_bias_v:	J
6assignvariableop_30_adam_lstm_21_lstm_cell_21_kernel_v:

T
@assignvariableop_31_adam_lstm_21_lstm_cell_21_recurrent_kernel_v:
Å
C
4assignvariableop_32_adam_lstm_21_lstm_cell_21_bias_v:	

identity_34¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_24¢AssignVariableOp_25¢AssignVariableOp_26¢AssignVariableOp_27¢AssignVariableOp_28¢AssignVariableOp_29¢AssignVariableOp_3¢AssignVariableOp_30¢AssignVariableOp_31¢AssignVariableOp_32¢AssignVariableOp_4¢AssignVariableOp_5¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9ò
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:"*
dtype0*þ
valueôBñ"B6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_namesÒ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:"*
dtype0*W
valueNBL"B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slicesØ
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*
_output_shapes
::::::::::::::::::::::::::::::::::*0
dtypes&
$2"	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity
AssignVariableOpAssignVariableOp assignvariableop_dense_10_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1¥
AssignVariableOp_1AssignVariableOp assignvariableop_1_dense_10_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_2¡
AssignVariableOp_2AssignVariableOpassignvariableop_2_adam_iterIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3£
AssignVariableOp_3AssignVariableOpassignvariableop_3_adam_beta_1Identity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4£
AssignVariableOp_4AssignVariableOpassignvariableop_4_adam_beta_2Identity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5¢
AssignVariableOp_5AssignVariableOpassignvariableop_5_adam_decayIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6ª
AssignVariableOp_6AssignVariableOp%assignvariableop_6_adam_learning_rateIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7³
AssignVariableOp_7AssignVariableOp.assignvariableop_7_lstm_20_lstm_cell_20_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8½
AssignVariableOp_8AssignVariableOp8assignvariableop_8_lstm_20_lstm_cell_20_recurrent_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9±
AssignVariableOp_9AssignVariableOp,assignvariableop_9_lstm_20_lstm_cell_20_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10·
AssignVariableOp_10AssignVariableOp/assignvariableop_10_lstm_21_lstm_cell_21_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11Á
AssignVariableOp_11AssignVariableOp9assignvariableop_11_lstm_21_lstm_cell_21_recurrent_kernelIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12µ
AssignVariableOp_12AssignVariableOp-assignvariableop_12_lstm_21_lstm_cell_21_biasIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13¡
AssignVariableOp_13AssignVariableOpassignvariableop_13_totalIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14¡
AssignVariableOp_14AssignVariableOpassignvariableop_14_countIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15£
AssignVariableOp_15AssignVariableOpassignvariableop_15_total_1Identity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16£
AssignVariableOp_16AssignVariableOpassignvariableop_16_count_1Identity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17²
AssignVariableOp_17AssignVariableOp*assignvariableop_17_adam_dense_10_kernel_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18°
AssignVariableOp_18AssignVariableOp(assignvariableop_18_adam_dense_10_bias_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19¾
AssignVariableOp_19AssignVariableOp6assignvariableop_19_adam_lstm_20_lstm_cell_20_kernel_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20È
AssignVariableOp_20AssignVariableOp@assignvariableop_20_adam_lstm_20_lstm_cell_20_recurrent_kernel_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21¼
AssignVariableOp_21AssignVariableOp4assignvariableop_21_adam_lstm_20_lstm_cell_20_bias_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22¾
AssignVariableOp_22AssignVariableOp6assignvariableop_22_adam_lstm_21_lstm_cell_21_kernel_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23È
AssignVariableOp_23AssignVariableOp@assignvariableop_23_adam_lstm_21_lstm_cell_21_recurrent_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24¼
AssignVariableOp_24AssignVariableOp4assignvariableop_24_adam_lstm_21_lstm_cell_21_bias_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25²
AssignVariableOp_25AssignVariableOp*assignvariableop_25_adam_dense_10_kernel_vIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26°
AssignVariableOp_26AssignVariableOp(assignvariableop_26_adam_dense_10_bias_vIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27¾
AssignVariableOp_27AssignVariableOp6assignvariableop_27_adam_lstm_20_lstm_cell_20_kernel_vIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28È
AssignVariableOp_28AssignVariableOp@assignvariableop_28_adam_lstm_20_lstm_cell_20_recurrent_kernel_vIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29¼
AssignVariableOp_29AssignVariableOp4assignvariableop_29_adam_lstm_20_lstm_cell_20_bias_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30¾
AssignVariableOp_30AssignVariableOp6assignvariableop_30_adam_lstm_21_lstm_cell_21_kernel_vIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31È
AssignVariableOp_31AssignVariableOp@assignvariableop_31_adam_lstm_21_lstm_cell_21_recurrent_kernel_vIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32¼
AssignVariableOp_32AssignVariableOp4assignvariableop_32_adam_lstm_21_lstm_cell_21_bias_vIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_329
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp´
Identity_33Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_33f
Identity_34IdentityIdentity_33:output:0^NoOp_1*
T0*
_output_shapes
: 2
Identity_34
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 2
NoOp_1"#
identity_34Identity_34:output:0*W
_input_shapesF
D: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322(
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
×
g
H__inference_dropout_21_layer_call_and_return_conditional_losses_39105398

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
:ÿÿÿÿÿÿÿÿÿÅ2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape¹
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ*
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
:ÿÿÿÿÿÿÿÿÿÅ2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2
dropout/Mul_1j
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿÅ:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ
 
_user_specified_nameinputs
®

Ô
0__inference_sequential_10_layer_call_fn_39102779
lstm_20_input
unknown:	]
	unknown_0:

	unknown_1:	
	unknown_2:


	unknown_3:
Å

	unknown_4:	

	unknown_5:	Å
	unknown_6:
identity¢StatefulPartitionedCallÔ
StatefulPartitionedCallStatefulPartitionedCalllstm_20_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
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
K__inference_sequential_10_layer_call_and_return_conditional_losses_391027602
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
_user_specified_namelstm_20_input
ã
Í
while_cond_39101423
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_39101423___redundant_placeholder06
2while_while_cond_39101423___redundant_placeholder16
2while_while_cond_39101423___redundant_placeholder26
2while_while_cond_39101423___redundant_placeholder3
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
B: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
ã
Í
while_cond_39105252
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_39105252___redundant_placeholder06
2while_while_cond_39105252___redundant_placeholder16
2while_while_cond_39105252___redundant_placeholder26
2while_while_cond_39105252___redundant_placeholder3
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
B: : : : :ÿÿÿÿÿÿÿÿÿÅ:ÿÿÿÿÿÿÿÿÿÅ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ:

_output_shapes
: :

_output_shapes
:


J__inference_lstm_cell_20_layer_call_and_return_conditional_losses_39105480

inputs
states_0
states_11
matmul_readvariableop_resource:	]4
 matmul_1_readvariableop_resource:
.
biasadd_readvariableop_resource:	
identity

identity_1

identity_2¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	]*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul_1/ReadVariableOp|
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
add
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
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
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2
split`
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
Sigmoidd
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	Sigmoid_1]
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mulW
ReluRelusplit:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relui
mul_1MulSigmoid:y:0Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mul_1^
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
add_1d
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	Sigmoid_2V
Relu_1Relu	add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu_1m
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mul_2e
IdentityIdentity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityi

Identity_1Identity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1i

Identity_2Identity	add_1:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

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
A:ÿÿÿÿÿÿÿÿÿ]:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
 
_user_specified_nameinputs:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
states/0:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
states/1
¹
¹
*__inference_lstm_21_layer_call_fn_39105381

inputs
unknown:


	unknown_0:
Å

	unknown_1:	
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
:ÿÿÿÿÿÿÿÿÿÅ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_lstm_21_layer_call_and_return_conditional_losses_391029762
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:ÿÿÿÿÿÿÿÿÿ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ü	
Ê
&__inference_signature_wrapper_39103348
lstm_20_input
unknown:	]
	unknown_0:

	unknown_1:	
	unknown_2:


	unknown_3:
Å

	unknown_4:	

	unknown_5:	Å
	unknown_6:
identity¢StatefulPartitionedCall¬
StatefulPartitionedCallStatefulPartitionedCalllstm_20_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
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
#__inference__wrapped_model_391011252
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
_user_specified_namelstm_20_input
ã
»
*__inference_lstm_21_layer_call_fn_39105348
inputs_0
unknown:


	unknown_0:
Å

	unknown_1:	
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
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÅ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_lstm_21_layer_call_and_return_conditional_losses_391019132
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÅ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0
ã
Í
while_cond_39104275
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_39104275___redundant_placeholder06
2while_while_cond_39104275___redundant_placeholder16
2while_while_cond_39104275___redundant_placeholder26
2while_while_cond_39104275___redundant_placeholder3
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
B: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
×
g
H__inference_dropout_21_layer_call_and_return_conditional_losses_39102809

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
:ÿÿÿÿÿÿÿÿÿÅ2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape¹
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ*
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
:ÿÿÿÿÿÿÿÿÿÅ2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2
dropout/Mul_1j
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿÅ:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ
 
_user_specified_nameinputs
×
g
H__inference_dropout_20_layer_call_and_return_conditional_losses_39103005

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
:ÿÿÿÿÿÿÿÿÿ2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape¹
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
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
:ÿÿÿÿÿÿÿÿÿ2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/Mul_1j
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

f
H__inference_dropout_21_layer_call_and_return_conditional_losses_39105386

inputs

identity_1_
IdentityIdentityinputs*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2

Identityn

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿÅ:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ
 
_user_specified_nameinputs
Ç
ù
/__inference_lstm_cell_20_layer_call_fn_39105546

inputs
states_0
states_1
unknown:	]
	unknown_0:

	unknown_1:	
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
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_lstm_cell_20_layer_call_and_return_conditional_losses_391013462
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

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
A:ÿÿÿÿÿÿÿÿÿ]:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
 
_user_specified_nameinputs:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
states/0:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
states/1
Ê
ú
/__inference_lstm_cell_21_layer_call_fn_39105627

inputs
states_0
states_1
unknown:


	unknown_0:
Å

	unknown_1:	
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
<:ÿÿÿÿÿÿÿÿÿÅ:ÿÿÿÿÿÿÿÿÿÅ:ÿÿÿÿÿÿÿÿÿÅ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_lstm_cell_21_layer_call_and_return_conditional_losses_391018302
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2

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
B:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿÅ:ÿÿÿÿÿÿÿÿÿÅ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ
"
_user_specified_name
states/0:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ
"
_user_specified_name
states/1
ã
Í
while_cond_39102053
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_39102053___redundant_placeholder06
2while_while_cond_39102053___redundant_placeholder16
2while_while_cond_39102053___redundant_placeholder26
2while_while_cond_39102053___redundant_placeholder3
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
B: : : : :ÿÿÿÿÿÿÿÿÿÅ:ÿÿÿÿÿÿÿÿÿÅ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ:

_output_shapes
: :

_output_shapes
:
ã
»
*__inference_lstm_21_layer_call_fn_39105359
inputs_0
unknown:


	unknown_0:
Å

	unknown_1:	
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
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÅ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_lstm_21_layer_call_and_return_conditional_losses_391021232
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÅ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0


Í
0__inference_sequential_10_layer_call_fn_39104058

inputs
unknown:	]
	unknown_0:

	unknown_1:	
	unknown_2:


	unknown_3:
Å

	unknown_4:	

	unknown_5:	Å
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
K__inference_sequential_10_layer_call_and_return_conditional_losses_391032292
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
-__inference_dropout_21_layer_call_fn_39105408

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
:ÿÿÿÿÿÿÿÿÿÅ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_dropout_21_layer_call_and_return_conditional_losses_391028092
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿÅ22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ
 
_user_specified_nameinputs
ã
Í
while_cond_39102457
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_39102457___redundant_placeholder06
2while_while_cond_39102457___redundant_placeholder16
2while_while_cond_39102457___redundant_placeholder26
2while_while_cond_39102457___redundant_placeholder3
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
B: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
Ê
Â
K__inference_sequential_10_layer_call_and_return_conditional_losses_39103319
lstm_20_input#
lstm_20_39103297:	]$
lstm_20_39103299:

lstm_20_39103301:	$
lstm_21_39103305:

$
lstm_21_39103307:
Å

lstm_21_39103309:	
$
dense_10_39103313:	Å
dense_10_39103315:
identity¢ dense_10/StatefulPartitionedCall¢"dropout_20/StatefulPartitionedCall¢"dropout_21/StatefulPartitionedCall¢lstm_20/StatefulPartitionedCall¢lstm_21/StatefulPartitionedCallµ
lstm_20/StatefulPartitionedCallStatefulPartitionedCalllstm_20_inputlstm_20_39103297lstm_20_39103299lstm_20_39103301*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_lstm_20_layer_call_and_return_conditional_losses_391031722!
lstm_20/StatefulPartitionedCall
"dropout_20/StatefulPartitionedCallStatefulPartitionedCall(lstm_20/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_dropout_20_layer_call_and_return_conditional_losses_391030052$
"dropout_20/StatefulPartitionedCallÓ
lstm_21/StatefulPartitionedCallStatefulPartitionedCall+dropout_20/StatefulPartitionedCall:output:0lstm_21_39103305lstm_21_39103307lstm_21_39103309*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_lstm_21_layer_call_and_return_conditional_losses_391029762!
lstm_21/StatefulPartitionedCallÀ
"dropout_21/StatefulPartitionedCallStatefulPartitionedCall(lstm_21/StatefulPartitionedCall:output:0#^dropout_20/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_dropout_21_layer_call_and_return_conditional_losses_391028092$
"dropout_21/StatefulPartitionedCallÃ
 dense_10/StatefulPartitionedCallStatefulPartitionedCall+dropout_21/StatefulPartitionedCall:output:0dense_10_39103313dense_10_39103315*
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
F__inference_dense_10_layer_call_and_return_conditional_losses_391027532"
 dense_10/StatefulPartitionedCall
IdentityIdentity)dense_10/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityÿ
NoOpNoOp!^dense_10/StatefulPartitionedCall#^dropout_20/StatefulPartitionedCall#^dropout_21/StatefulPartitionedCall ^lstm_20/StatefulPartitionedCall ^lstm_21/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ]: : : : : : : : 2D
 dense_10/StatefulPartitionedCall dense_10/StatefulPartitionedCall2H
"dropout_20/StatefulPartitionedCall"dropout_20/StatefulPartitionedCall2H
"dropout_21/StatefulPartitionedCall"dropout_21/StatefulPartitionedCall2B
lstm_20/StatefulPartitionedCalllstm_20/StatefulPartitionedCall2B
lstm_21/StatefulPartitionedCalllstm_21/StatefulPartitionedCall:Z V
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
'
_user_specified_namelstm_20_input
ã
Í
while_cond_39101213
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_39101213___redundant_placeholder06
2while_while_cond_39101213___redundant_placeholder16
2while_while_cond_39101213___redundant_placeholder26
2while_while_cond_39101213___redundant_placeholder3
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
B: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
ù

)sequential_10_lstm_21_while_cond_39101012H
Dsequential_10_lstm_21_while_sequential_10_lstm_21_while_loop_counterN
Jsequential_10_lstm_21_while_sequential_10_lstm_21_while_maximum_iterations+
'sequential_10_lstm_21_while_placeholder-
)sequential_10_lstm_21_while_placeholder_1-
)sequential_10_lstm_21_while_placeholder_2-
)sequential_10_lstm_21_while_placeholder_3J
Fsequential_10_lstm_21_while_less_sequential_10_lstm_21_strided_slice_1b
^sequential_10_lstm_21_while_sequential_10_lstm_21_while_cond_39101012___redundant_placeholder0b
^sequential_10_lstm_21_while_sequential_10_lstm_21_while_cond_39101012___redundant_placeholder1b
^sequential_10_lstm_21_while_sequential_10_lstm_21_while_cond_39101012___redundant_placeholder2b
^sequential_10_lstm_21_while_sequential_10_lstm_21_while_cond_39101012___redundant_placeholder3(
$sequential_10_lstm_21_while_identity
Þ
 sequential_10/lstm_21/while/LessLess'sequential_10_lstm_21_while_placeholderFsequential_10_lstm_21_while_less_sequential_10_lstm_21_strided_slice_1*
T0*
_output_shapes
: 2"
 sequential_10/lstm_21/while/Less
$sequential_10/lstm_21/while/IdentityIdentity$sequential_10/lstm_21/while/Less:z:0*
T0
*
_output_shapes
: 2&
$sequential_10/lstm_21/while/Identity"U
$sequential_10_lstm_21_while_identity-sequential_10/lstm_21/while/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :ÿÿÿÿÿÿÿÿÿÅ:ÿÿÿÿÿÿÿÿÿÅ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ:

_output_shapes
: :

_output_shapes
:
´?
Ö
while_body_39105253
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
3while_lstm_cell_21_matmul_readvariableop_resource_0:

I
5while_lstm_cell_21_matmul_1_readvariableop_resource_0:
Å
C
4while_lstm_cell_21_biasadd_readvariableop_resource_0:	

while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
1while_lstm_cell_21_matmul_readvariableop_resource:

G
3while_lstm_cell_21_matmul_1_readvariableop_resource:
Å
A
2while_lstm_cell_21_biasadd_readvariableop_resource:	
¢)while/lstm_cell_21/BiasAdd/ReadVariableOp¢(while/lstm_cell_21/MatMul/ReadVariableOp¢*while/lstm_cell_21/MatMul_1/ReadVariableOpÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ  29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÔ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemÊ
(while/lstm_cell_21/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_21_matmul_readvariableop_resource_0* 
_output_shapes
:

*
dtype02*
(while/lstm_cell_21/MatMul/ReadVariableOp×
while/lstm_cell_21/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_21/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
while/lstm_cell_21/MatMulÐ
*while/lstm_cell_21/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_21_matmul_1_readvariableop_resource_0* 
_output_shapes
:
Å
*
dtype02,
*while/lstm_cell_21/MatMul_1/ReadVariableOpÀ
while/lstm_cell_21/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_21/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
while/lstm_cell_21/MatMul_1¸
while/lstm_cell_21/addAddV2#while/lstm_cell_21/MatMul:product:0%while/lstm_cell_21/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
while/lstm_cell_21/addÈ
)while/lstm_cell_21/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_21_biasadd_readvariableop_resource_0*
_output_shapes	
:
*
dtype02+
)while/lstm_cell_21/BiasAdd/ReadVariableOpÅ
while/lstm_cell_21/BiasAddBiasAddwhile/lstm_cell_21/add:z:01while/lstm_cell_21/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
while/lstm_cell_21/BiasAdd
"while/lstm_cell_21/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_21/split/split_dim
while/lstm_cell_21/splitSplit+while/lstm_cell_21/split/split_dim:output:0#while/lstm_cell_21/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿÅ:ÿÿÿÿÿÿÿÿÿÅ:ÿÿÿÿÿÿÿÿÿÅ:ÿÿÿÿÿÿÿÿÿÅ*
	num_split2
while/lstm_cell_21/split
while/lstm_cell_21/SigmoidSigmoid!while/lstm_cell_21/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2
while/lstm_cell_21/Sigmoid
while/lstm_cell_21/Sigmoid_1Sigmoid!while/lstm_cell_21/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2
while/lstm_cell_21/Sigmoid_1¡
while/lstm_cell_21/mulMul while/lstm_cell_21/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2
while/lstm_cell_21/mul
while/lstm_cell_21/ReluRelu!while/lstm_cell_21/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2
while/lstm_cell_21/Reluµ
while/lstm_cell_21/mul_1Mulwhile/lstm_cell_21/Sigmoid:y:0%while/lstm_cell_21/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2
while/lstm_cell_21/mul_1ª
while/lstm_cell_21/add_1AddV2while/lstm_cell_21/mul:z:0while/lstm_cell_21/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2
while/lstm_cell_21/add_1
while/lstm_cell_21/Sigmoid_2Sigmoid!while/lstm_cell_21/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2
while/lstm_cell_21/Sigmoid_2
while/lstm_cell_21/Relu_1Reluwhile/lstm_cell_21/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2
while/lstm_cell_21/Relu_1¹
while/lstm_cell_21/mul_2Mul while/lstm_cell_21/Sigmoid_2:y:0'while/lstm_cell_21/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2
while/lstm_cell_21/mul_2à
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_21/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_21/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_21/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2
while/Identity_5Þ

while/NoOpNoOp*^while/lstm_cell_21/BiasAdd/ReadVariableOp)^while/lstm_cell_21/MatMul/ReadVariableOp+^while/lstm_cell_21/MatMul_1/ReadVariableOp*"
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
2while_lstm_cell_21_biasadd_readvariableop_resource4while_lstm_cell_21_biasadd_readvariableop_resource_0"l
3while_lstm_cell_21_matmul_1_readvariableop_resource5while_lstm_cell_21_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_21_matmul_readvariableop_resource3while_lstm_cell_21_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿÅ:ÿÿÿÿÿÿÿÿÿÅ: : : : : 2V
)while/lstm_cell_21/BiasAdd/ReadVariableOp)while/lstm_cell_21/BiasAdd/ReadVariableOp2T
(while/lstm_cell_21/MatMul/ReadVariableOp(while/lstm_cell_21/MatMul/ReadVariableOp2X
*while/lstm_cell_21/MatMul_1/ReadVariableOp*while/lstm_cell_21/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ:

_output_shapes
: :

_output_shapes
: 


J__inference_lstm_cell_21_layer_call_and_return_conditional_losses_39105578

inputs
states_0
states_12
matmul_readvariableop_resource:

4
 matmul_1_readvariableop_resource:
Å
.
biasadd_readvariableop_resource:	

identity

identity_1

identity_2¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:

*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
MatMul
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
Å
*
dtype02
MatMul_1/ReadVariableOp|
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
add
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:
*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
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
P:ÿÿÿÿÿÿÿÿÿÅ:ÿÿÿÿÿÿÿÿÿÅ:ÿÿÿÿÿÿÿÿÿÅ:ÿÿÿÿÿÿÿÿÿÅ*
	num_split2
split`
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2	
Sigmoidd
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2
	Sigmoid_1]
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2
mulW
ReluRelusplit:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2
Relui
mul_1MulSigmoid:y:0Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2
mul_1^
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2
add_1d
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2
	Sigmoid_2V
Relu_1Relu	add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2
Relu_1m
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2
mul_2e
IdentityIdentity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2

Identityi

Identity_1Identity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2

Identity_1i

Identity_2Identity	add_1:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2

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
B:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿÅ:ÿÿÿÿÿÿÿÿÿÅ: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ
"
_user_specified_name
states/0:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ
"
_user_specified_name
states/1
Ô

í
lstm_20_while_cond_39103414,
(lstm_20_while_lstm_20_while_loop_counter2
.lstm_20_while_lstm_20_while_maximum_iterations
lstm_20_while_placeholder
lstm_20_while_placeholder_1
lstm_20_while_placeholder_2
lstm_20_while_placeholder_3.
*lstm_20_while_less_lstm_20_strided_slice_1F
Blstm_20_while_lstm_20_while_cond_39103414___redundant_placeholder0F
Blstm_20_while_lstm_20_while_cond_39103414___redundant_placeholder1F
Blstm_20_while_lstm_20_while_cond_39103414___redundant_placeholder2F
Blstm_20_while_lstm_20_while_cond_39103414___redundant_placeholder3
lstm_20_while_identity

lstm_20/while/LessLesslstm_20_while_placeholder*lstm_20_while_less_lstm_20_strided_slice_1*
T0*
_output_shapes
: 2
lstm_20/while/Lessu
lstm_20/while/IdentityIdentitylstm_20/while/Less:z:0*
T0
*
_output_shapes
: 2
lstm_20/while/Identity"9
lstm_20_while_identitylstm_20/while/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
ã
Í
while_cond_39102622
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_39102622___redundant_placeholder06
2while_while_cond_39102622___redundant_placeholder16
2while_while_cond_39102622___redundant_placeholder26
2while_while_cond_39102622___redundant_placeholder3
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
B: : : : :ÿÿÿÿÿÿÿÿÿÅ:ÿÿÿÿÿÿÿÿÿÅ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ:

_output_shapes
: :

_output_shapes
:
¨¹
ç	
#__inference__wrapped_model_39101125
lstm_20_inputT
Asequential_10_lstm_20_lstm_cell_20_matmul_readvariableop_resource:	]W
Csequential_10_lstm_20_lstm_cell_20_matmul_1_readvariableop_resource:
Q
Bsequential_10_lstm_20_lstm_cell_20_biasadd_readvariableop_resource:	U
Asequential_10_lstm_21_lstm_cell_21_matmul_readvariableop_resource:

W
Csequential_10_lstm_21_lstm_cell_21_matmul_1_readvariableop_resource:
Å
Q
Bsequential_10_lstm_21_lstm_cell_21_biasadd_readvariableop_resource:	
K
8sequential_10_dense_10_tensordot_readvariableop_resource:	ÅD
6sequential_10_dense_10_biasadd_readvariableop_resource:
identity¢-sequential_10/dense_10/BiasAdd/ReadVariableOp¢/sequential_10/dense_10/Tensordot/ReadVariableOp¢9sequential_10/lstm_20/lstm_cell_20/BiasAdd/ReadVariableOp¢8sequential_10/lstm_20/lstm_cell_20/MatMul/ReadVariableOp¢:sequential_10/lstm_20/lstm_cell_20/MatMul_1/ReadVariableOp¢sequential_10/lstm_20/while¢9sequential_10/lstm_21/lstm_cell_21/BiasAdd/ReadVariableOp¢8sequential_10/lstm_21/lstm_cell_21/MatMul/ReadVariableOp¢:sequential_10/lstm_21/lstm_cell_21/MatMul_1/ReadVariableOp¢sequential_10/lstm_21/whilew
sequential_10/lstm_20/ShapeShapelstm_20_input*
T0*
_output_shapes
:2
sequential_10/lstm_20/Shape 
)sequential_10/lstm_20/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)sequential_10/lstm_20/strided_slice/stack¤
+sequential_10/lstm_20/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential_10/lstm_20/strided_slice/stack_1¤
+sequential_10/lstm_20/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential_10/lstm_20/strided_slice/stack_2æ
#sequential_10/lstm_20/strided_sliceStridedSlice$sequential_10/lstm_20/Shape:output:02sequential_10/lstm_20/strided_slice/stack:output:04sequential_10/lstm_20/strided_slice/stack_1:output:04sequential_10/lstm_20/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#sequential_10/lstm_20/strided_slice
!sequential_10/lstm_20/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :2#
!sequential_10/lstm_20/zeros/mul/yÄ
sequential_10/lstm_20/zeros/mulMul,sequential_10/lstm_20/strided_slice:output:0*sequential_10/lstm_20/zeros/mul/y:output:0*
T0*
_output_shapes
: 2!
sequential_10/lstm_20/zeros/mul
"sequential_10/lstm_20/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2$
"sequential_10/lstm_20/zeros/Less/y¿
 sequential_10/lstm_20/zeros/LessLess#sequential_10/lstm_20/zeros/mul:z:0+sequential_10/lstm_20/zeros/Less/y:output:0*
T0*
_output_shapes
: 2"
 sequential_10/lstm_20/zeros/Less
$sequential_10/lstm_20/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :2&
$sequential_10/lstm_20/zeros/packed/1Û
"sequential_10/lstm_20/zeros/packedPack,sequential_10/lstm_20/strided_slice:output:0-sequential_10/lstm_20/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2$
"sequential_10/lstm_20/zeros/packed
!sequential_10/lstm_20/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!sequential_10/lstm_20/zeros/ConstÎ
sequential_10/lstm_20/zerosFill+sequential_10/lstm_20/zeros/packed:output:0*sequential_10/lstm_20/zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential_10/lstm_20/zeros
#sequential_10/lstm_20/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :2%
#sequential_10/lstm_20/zeros_1/mul/yÊ
!sequential_10/lstm_20/zeros_1/mulMul,sequential_10/lstm_20/strided_slice:output:0,sequential_10/lstm_20/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2#
!sequential_10/lstm_20/zeros_1/mul
$sequential_10/lstm_20/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2&
$sequential_10/lstm_20/zeros_1/Less/yÇ
"sequential_10/lstm_20/zeros_1/LessLess%sequential_10/lstm_20/zeros_1/mul:z:0-sequential_10/lstm_20/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2$
"sequential_10/lstm_20/zeros_1/Less
&sequential_10/lstm_20/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :2(
&sequential_10/lstm_20/zeros_1/packed/1á
$sequential_10/lstm_20/zeros_1/packedPack,sequential_10/lstm_20/strided_slice:output:0/sequential_10/lstm_20/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2&
$sequential_10/lstm_20/zeros_1/packed
#sequential_10/lstm_20/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2%
#sequential_10/lstm_20/zeros_1/ConstÖ
sequential_10/lstm_20/zeros_1Fill-sequential_10/lstm_20/zeros_1/packed:output:0,sequential_10/lstm_20/zeros_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential_10/lstm_20/zeros_1¡
$sequential_10/lstm_20/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2&
$sequential_10/lstm_20/transpose/permÃ
sequential_10/lstm_20/transpose	Transposelstm_20_input-sequential_10/lstm_20/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]2!
sequential_10/lstm_20/transpose
sequential_10/lstm_20/Shape_1Shape#sequential_10/lstm_20/transpose:y:0*
T0*
_output_shapes
:2
sequential_10/lstm_20/Shape_1¤
+sequential_10/lstm_20/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2-
+sequential_10/lstm_20/strided_slice_1/stack¨
-sequential_10/lstm_20/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_10/lstm_20/strided_slice_1/stack_1¨
-sequential_10/lstm_20/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_10/lstm_20/strided_slice_1/stack_2ò
%sequential_10/lstm_20/strided_slice_1StridedSlice&sequential_10/lstm_20/Shape_1:output:04sequential_10/lstm_20/strided_slice_1/stack:output:06sequential_10/lstm_20/strided_slice_1/stack_1:output:06sequential_10/lstm_20/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%sequential_10/lstm_20/strided_slice_1±
1sequential_10/lstm_20/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ23
1sequential_10/lstm_20/TensorArrayV2/element_shape
#sequential_10/lstm_20/TensorArrayV2TensorListReserve:sequential_10/lstm_20/TensorArrayV2/element_shape:output:0.sequential_10/lstm_20/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02%
#sequential_10/lstm_20/TensorArrayV2ë
Ksequential_10/lstm_20/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ]   2M
Ksequential_10/lstm_20/TensorArrayUnstack/TensorListFromTensor/element_shapeÐ
=sequential_10/lstm_20/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor#sequential_10/lstm_20/transpose:y:0Tsequential_10/lstm_20/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02?
=sequential_10/lstm_20/TensorArrayUnstack/TensorListFromTensor¤
+sequential_10/lstm_20/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2-
+sequential_10/lstm_20/strided_slice_2/stack¨
-sequential_10/lstm_20/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_10/lstm_20/strided_slice_2/stack_1¨
-sequential_10/lstm_20/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_10/lstm_20/strided_slice_2/stack_2
%sequential_10/lstm_20/strided_slice_2StridedSlice#sequential_10/lstm_20/transpose:y:04sequential_10/lstm_20/strided_slice_2/stack:output:06sequential_10/lstm_20/strided_slice_2/stack_1:output:06sequential_10/lstm_20/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]*
shrink_axis_mask2'
%sequential_10/lstm_20/strided_slice_2÷
8sequential_10/lstm_20/lstm_cell_20/MatMul/ReadVariableOpReadVariableOpAsequential_10_lstm_20_lstm_cell_20_matmul_readvariableop_resource*
_output_shapes
:	]*
dtype02:
8sequential_10/lstm_20/lstm_cell_20/MatMul/ReadVariableOp
)sequential_10/lstm_20/lstm_cell_20/MatMulMatMul.sequential_10/lstm_20/strided_slice_2:output:0@sequential_10/lstm_20/lstm_cell_20/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2+
)sequential_10/lstm_20/lstm_cell_20/MatMulþ
:sequential_10/lstm_20/lstm_cell_20/MatMul_1/ReadVariableOpReadVariableOpCsequential_10_lstm_20_lstm_cell_20_matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype02<
:sequential_10/lstm_20/lstm_cell_20/MatMul_1/ReadVariableOp
+sequential_10/lstm_20/lstm_cell_20/MatMul_1MatMul$sequential_10/lstm_20/zeros:output:0Bsequential_10/lstm_20/lstm_cell_20/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2-
+sequential_10/lstm_20/lstm_cell_20/MatMul_1ø
&sequential_10/lstm_20/lstm_cell_20/addAddV23sequential_10/lstm_20/lstm_cell_20/MatMul:product:05sequential_10/lstm_20/lstm_cell_20/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&sequential_10/lstm_20/lstm_cell_20/addö
9sequential_10/lstm_20/lstm_cell_20/BiasAdd/ReadVariableOpReadVariableOpBsequential_10_lstm_20_lstm_cell_20_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02;
9sequential_10/lstm_20/lstm_cell_20/BiasAdd/ReadVariableOp
*sequential_10/lstm_20/lstm_cell_20/BiasAddBiasAdd*sequential_10/lstm_20/lstm_cell_20/add:z:0Asequential_10/lstm_20/lstm_cell_20/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2,
*sequential_10/lstm_20/lstm_cell_20/BiasAddª
2sequential_10/lstm_20/lstm_cell_20/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :24
2sequential_10/lstm_20/lstm_cell_20/split/split_dimÏ
(sequential_10/lstm_20/lstm_cell_20/splitSplit;sequential_10/lstm_20/lstm_cell_20/split/split_dim:output:03sequential_10/lstm_20/lstm_cell_20/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2*
(sequential_10/lstm_20/lstm_cell_20/splitÉ
*sequential_10/lstm_20/lstm_cell_20/SigmoidSigmoid1sequential_10/lstm_20/lstm_cell_20/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2,
*sequential_10/lstm_20/lstm_cell_20/SigmoidÍ
,sequential_10/lstm_20/lstm_cell_20/Sigmoid_1Sigmoid1sequential_10/lstm_20/lstm_cell_20/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2.
,sequential_10/lstm_20/lstm_cell_20/Sigmoid_1ä
&sequential_10/lstm_20/lstm_cell_20/mulMul0sequential_10/lstm_20/lstm_cell_20/Sigmoid_1:y:0&sequential_10/lstm_20/zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&sequential_10/lstm_20/lstm_cell_20/mulÀ
'sequential_10/lstm_20/lstm_cell_20/ReluRelu1sequential_10/lstm_20/lstm_cell_20/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'sequential_10/lstm_20/lstm_cell_20/Reluõ
(sequential_10/lstm_20/lstm_cell_20/mul_1Mul.sequential_10/lstm_20/lstm_cell_20/Sigmoid:y:05sequential_10/lstm_20/lstm_cell_20/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
(sequential_10/lstm_20/lstm_cell_20/mul_1ê
(sequential_10/lstm_20/lstm_cell_20/add_1AddV2*sequential_10/lstm_20/lstm_cell_20/mul:z:0,sequential_10/lstm_20/lstm_cell_20/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
(sequential_10/lstm_20/lstm_cell_20/add_1Í
,sequential_10/lstm_20/lstm_cell_20/Sigmoid_2Sigmoid1sequential_10/lstm_20/lstm_cell_20/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2.
,sequential_10/lstm_20/lstm_cell_20/Sigmoid_2¿
)sequential_10/lstm_20/lstm_cell_20/Relu_1Relu,sequential_10/lstm_20/lstm_cell_20/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2+
)sequential_10/lstm_20/lstm_cell_20/Relu_1ù
(sequential_10/lstm_20/lstm_cell_20/mul_2Mul0sequential_10/lstm_20/lstm_cell_20/Sigmoid_2:y:07sequential_10/lstm_20/lstm_cell_20/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
(sequential_10/lstm_20/lstm_cell_20/mul_2»
3sequential_10/lstm_20/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ  25
3sequential_10/lstm_20/TensorArrayV2_1/element_shape
%sequential_10/lstm_20/TensorArrayV2_1TensorListReserve<sequential_10/lstm_20/TensorArrayV2_1/element_shape:output:0.sequential_10/lstm_20/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02'
%sequential_10/lstm_20/TensorArrayV2_1z
sequential_10/lstm_20/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
sequential_10/lstm_20/time«
.sequential_10/lstm_20/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ20
.sequential_10/lstm_20/while/maximum_iterations
(sequential_10/lstm_20/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2*
(sequential_10/lstm_20/while/loop_counterÝ
sequential_10/lstm_20/whileWhile1sequential_10/lstm_20/while/loop_counter:output:07sequential_10/lstm_20/while/maximum_iterations:output:0#sequential_10/lstm_20/time:output:0.sequential_10/lstm_20/TensorArrayV2_1:handle:0$sequential_10/lstm_20/zeros:output:0&sequential_10/lstm_20/zeros_1:output:0.sequential_10/lstm_20/strided_slice_1:output:0Msequential_10/lstm_20/TensorArrayUnstack/TensorListFromTensor:output_handle:0Asequential_10_lstm_20_lstm_cell_20_matmul_readvariableop_resourceCsequential_10_lstm_20_lstm_cell_20_matmul_1_readvariableop_resourceBsequential_10_lstm_20_lstm_cell_20_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *5
body-R+
)sequential_10_lstm_20_while_body_39100865*5
cond-R+
)sequential_10_lstm_20_while_cond_39100864*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 2
sequential_10/lstm_20/whileá
Fsequential_10/lstm_20/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ  2H
Fsequential_10/lstm_20/TensorArrayV2Stack/TensorListStack/element_shapeÁ
8sequential_10/lstm_20/TensorArrayV2Stack/TensorListStackTensorListStack$sequential_10/lstm_20/while:output:3Osequential_10/lstm_20/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype02:
8sequential_10/lstm_20/TensorArrayV2Stack/TensorListStack­
+sequential_10/lstm_20/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2-
+sequential_10/lstm_20/strided_slice_3/stack¨
-sequential_10/lstm_20/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2/
-sequential_10/lstm_20/strided_slice_3/stack_1¨
-sequential_10/lstm_20/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_10/lstm_20/strided_slice_3/stack_2
%sequential_10/lstm_20/strided_slice_3StridedSliceAsequential_10/lstm_20/TensorArrayV2Stack/TensorListStack:tensor:04sequential_10/lstm_20/strided_slice_3/stack:output:06sequential_10/lstm_20/strided_slice_3/stack_1:output:06sequential_10/lstm_20/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2'
%sequential_10/lstm_20/strided_slice_3¥
&sequential_10/lstm_20/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2(
&sequential_10/lstm_20/transpose_1/permþ
!sequential_10/lstm_20/transpose_1	TransposeAsequential_10/lstm_20/TensorArrayV2Stack/TensorListStack:tensor:0/sequential_10/lstm_20/transpose_1/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!sequential_10/lstm_20/transpose_1
sequential_10/lstm_20/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
sequential_10/lstm_20/runtime°
!sequential_10/dropout_20/IdentityIdentity%sequential_10/lstm_20/transpose_1:y:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!sequential_10/dropout_20/Identity
sequential_10/lstm_21/ShapeShape*sequential_10/dropout_20/Identity:output:0*
T0*
_output_shapes
:2
sequential_10/lstm_21/Shape 
)sequential_10/lstm_21/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)sequential_10/lstm_21/strided_slice/stack¤
+sequential_10/lstm_21/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential_10/lstm_21/strided_slice/stack_1¤
+sequential_10/lstm_21/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential_10/lstm_21/strided_slice/stack_2æ
#sequential_10/lstm_21/strided_sliceStridedSlice$sequential_10/lstm_21/Shape:output:02sequential_10/lstm_21/strided_slice/stack:output:04sequential_10/lstm_21/strided_slice/stack_1:output:04sequential_10/lstm_21/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#sequential_10/lstm_21/strided_slice
!sequential_10/lstm_21/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :Å2#
!sequential_10/lstm_21/zeros/mul/yÄ
sequential_10/lstm_21/zeros/mulMul,sequential_10/lstm_21/strided_slice:output:0*sequential_10/lstm_21/zeros/mul/y:output:0*
T0*
_output_shapes
: 2!
sequential_10/lstm_21/zeros/mul
"sequential_10/lstm_21/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2$
"sequential_10/lstm_21/zeros/Less/y¿
 sequential_10/lstm_21/zeros/LessLess#sequential_10/lstm_21/zeros/mul:z:0+sequential_10/lstm_21/zeros/Less/y:output:0*
T0*
_output_shapes
: 2"
 sequential_10/lstm_21/zeros/Less
$sequential_10/lstm_21/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :Å2&
$sequential_10/lstm_21/zeros/packed/1Û
"sequential_10/lstm_21/zeros/packedPack,sequential_10/lstm_21/strided_slice:output:0-sequential_10/lstm_21/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2$
"sequential_10/lstm_21/zeros/packed
!sequential_10/lstm_21/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!sequential_10/lstm_21/zeros/ConstÎ
sequential_10/lstm_21/zerosFill+sequential_10/lstm_21/zeros/packed:output:0*sequential_10/lstm_21/zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2
sequential_10/lstm_21/zeros
#sequential_10/lstm_21/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :Å2%
#sequential_10/lstm_21/zeros_1/mul/yÊ
!sequential_10/lstm_21/zeros_1/mulMul,sequential_10/lstm_21/strided_slice:output:0,sequential_10/lstm_21/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2#
!sequential_10/lstm_21/zeros_1/mul
$sequential_10/lstm_21/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2&
$sequential_10/lstm_21/zeros_1/Less/yÇ
"sequential_10/lstm_21/zeros_1/LessLess%sequential_10/lstm_21/zeros_1/mul:z:0-sequential_10/lstm_21/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2$
"sequential_10/lstm_21/zeros_1/Less
&sequential_10/lstm_21/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :Å2(
&sequential_10/lstm_21/zeros_1/packed/1á
$sequential_10/lstm_21/zeros_1/packedPack,sequential_10/lstm_21/strided_slice:output:0/sequential_10/lstm_21/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2&
$sequential_10/lstm_21/zeros_1/packed
#sequential_10/lstm_21/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2%
#sequential_10/lstm_21/zeros_1/ConstÖ
sequential_10/lstm_21/zeros_1Fill-sequential_10/lstm_21/zeros_1/packed:output:0,sequential_10/lstm_21/zeros_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2
sequential_10/lstm_21/zeros_1¡
$sequential_10/lstm_21/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2&
$sequential_10/lstm_21/transpose/permá
sequential_10/lstm_21/transpose	Transpose*sequential_10/dropout_20/Identity:output:0-sequential_10/lstm_21/transpose/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
sequential_10/lstm_21/transpose
sequential_10/lstm_21/Shape_1Shape#sequential_10/lstm_21/transpose:y:0*
T0*
_output_shapes
:2
sequential_10/lstm_21/Shape_1¤
+sequential_10/lstm_21/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2-
+sequential_10/lstm_21/strided_slice_1/stack¨
-sequential_10/lstm_21/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_10/lstm_21/strided_slice_1/stack_1¨
-sequential_10/lstm_21/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_10/lstm_21/strided_slice_1/stack_2ò
%sequential_10/lstm_21/strided_slice_1StridedSlice&sequential_10/lstm_21/Shape_1:output:04sequential_10/lstm_21/strided_slice_1/stack:output:06sequential_10/lstm_21/strided_slice_1/stack_1:output:06sequential_10/lstm_21/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%sequential_10/lstm_21/strided_slice_1±
1sequential_10/lstm_21/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ23
1sequential_10/lstm_21/TensorArrayV2/element_shape
#sequential_10/lstm_21/TensorArrayV2TensorListReserve:sequential_10/lstm_21/TensorArrayV2/element_shape:output:0.sequential_10/lstm_21/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02%
#sequential_10/lstm_21/TensorArrayV2ë
Ksequential_10/lstm_21/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ  2M
Ksequential_10/lstm_21/TensorArrayUnstack/TensorListFromTensor/element_shapeÐ
=sequential_10/lstm_21/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor#sequential_10/lstm_21/transpose:y:0Tsequential_10/lstm_21/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02?
=sequential_10/lstm_21/TensorArrayUnstack/TensorListFromTensor¤
+sequential_10/lstm_21/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2-
+sequential_10/lstm_21/strided_slice_2/stack¨
-sequential_10/lstm_21/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_10/lstm_21/strided_slice_2/stack_1¨
-sequential_10/lstm_21/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_10/lstm_21/strided_slice_2/stack_2
%sequential_10/lstm_21/strided_slice_2StridedSlice#sequential_10/lstm_21/transpose:y:04sequential_10/lstm_21/strided_slice_2/stack:output:06sequential_10/lstm_21/strided_slice_2/stack_1:output:06sequential_10/lstm_21/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2'
%sequential_10/lstm_21/strided_slice_2ø
8sequential_10/lstm_21/lstm_cell_21/MatMul/ReadVariableOpReadVariableOpAsequential_10_lstm_21_lstm_cell_21_matmul_readvariableop_resource* 
_output_shapes
:

*
dtype02:
8sequential_10/lstm_21/lstm_cell_21/MatMul/ReadVariableOp
)sequential_10/lstm_21/lstm_cell_21/MatMulMatMul.sequential_10/lstm_21/strided_slice_2:output:0@sequential_10/lstm_21/lstm_cell_21/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2+
)sequential_10/lstm_21/lstm_cell_21/MatMulþ
:sequential_10/lstm_21/lstm_cell_21/MatMul_1/ReadVariableOpReadVariableOpCsequential_10_lstm_21_lstm_cell_21_matmul_1_readvariableop_resource* 
_output_shapes
:
Å
*
dtype02<
:sequential_10/lstm_21/lstm_cell_21/MatMul_1/ReadVariableOp
+sequential_10/lstm_21/lstm_cell_21/MatMul_1MatMul$sequential_10/lstm_21/zeros:output:0Bsequential_10/lstm_21/lstm_cell_21/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2-
+sequential_10/lstm_21/lstm_cell_21/MatMul_1ø
&sequential_10/lstm_21/lstm_cell_21/addAddV23sequential_10/lstm_21/lstm_cell_21/MatMul:product:05sequential_10/lstm_21/lstm_cell_21/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2(
&sequential_10/lstm_21/lstm_cell_21/addö
9sequential_10/lstm_21/lstm_cell_21/BiasAdd/ReadVariableOpReadVariableOpBsequential_10_lstm_21_lstm_cell_21_biasadd_readvariableop_resource*
_output_shapes	
:
*
dtype02;
9sequential_10/lstm_21/lstm_cell_21/BiasAdd/ReadVariableOp
*sequential_10/lstm_21/lstm_cell_21/BiasAddBiasAdd*sequential_10/lstm_21/lstm_cell_21/add:z:0Asequential_10/lstm_21/lstm_cell_21/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2,
*sequential_10/lstm_21/lstm_cell_21/BiasAddª
2sequential_10/lstm_21/lstm_cell_21/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :24
2sequential_10/lstm_21/lstm_cell_21/split/split_dimÏ
(sequential_10/lstm_21/lstm_cell_21/splitSplit;sequential_10/lstm_21/lstm_cell_21/split/split_dim:output:03sequential_10/lstm_21/lstm_cell_21/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿÅ:ÿÿÿÿÿÿÿÿÿÅ:ÿÿÿÿÿÿÿÿÿÅ:ÿÿÿÿÿÿÿÿÿÅ*
	num_split2*
(sequential_10/lstm_21/lstm_cell_21/splitÉ
*sequential_10/lstm_21/lstm_cell_21/SigmoidSigmoid1sequential_10/lstm_21/lstm_cell_21/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2,
*sequential_10/lstm_21/lstm_cell_21/SigmoidÍ
,sequential_10/lstm_21/lstm_cell_21/Sigmoid_1Sigmoid1sequential_10/lstm_21/lstm_cell_21/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2.
,sequential_10/lstm_21/lstm_cell_21/Sigmoid_1ä
&sequential_10/lstm_21/lstm_cell_21/mulMul0sequential_10/lstm_21/lstm_cell_21/Sigmoid_1:y:0&sequential_10/lstm_21/zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2(
&sequential_10/lstm_21/lstm_cell_21/mulÀ
'sequential_10/lstm_21/lstm_cell_21/ReluRelu1sequential_10/lstm_21/lstm_cell_21/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2)
'sequential_10/lstm_21/lstm_cell_21/Reluõ
(sequential_10/lstm_21/lstm_cell_21/mul_1Mul.sequential_10/lstm_21/lstm_cell_21/Sigmoid:y:05sequential_10/lstm_21/lstm_cell_21/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2*
(sequential_10/lstm_21/lstm_cell_21/mul_1ê
(sequential_10/lstm_21/lstm_cell_21/add_1AddV2*sequential_10/lstm_21/lstm_cell_21/mul:z:0,sequential_10/lstm_21/lstm_cell_21/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2*
(sequential_10/lstm_21/lstm_cell_21/add_1Í
,sequential_10/lstm_21/lstm_cell_21/Sigmoid_2Sigmoid1sequential_10/lstm_21/lstm_cell_21/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2.
,sequential_10/lstm_21/lstm_cell_21/Sigmoid_2¿
)sequential_10/lstm_21/lstm_cell_21/Relu_1Relu,sequential_10/lstm_21/lstm_cell_21/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2+
)sequential_10/lstm_21/lstm_cell_21/Relu_1ù
(sequential_10/lstm_21/lstm_cell_21/mul_2Mul0sequential_10/lstm_21/lstm_cell_21/Sigmoid_2:y:07sequential_10/lstm_21/lstm_cell_21/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2*
(sequential_10/lstm_21/lstm_cell_21/mul_2»
3sequential_10/lstm_21/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿE  25
3sequential_10/lstm_21/TensorArrayV2_1/element_shape
%sequential_10/lstm_21/TensorArrayV2_1TensorListReserve<sequential_10/lstm_21/TensorArrayV2_1/element_shape:output:0.sequential_10/lstm_21/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02'
%sequential_10/lstm_21/TensorArrayV2_1z
sequential_10/lstm_21/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
sequential_10/lstm_21/time«
.sequential_10/lstm_21/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ20
.sequential_10/lstm_21/while/maximum_iterations
(sequential_10/lstm_21/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2*
(sequential_10/lstm_21/while/loop_counterÝ
sequential_10/lstm_21/whileWhile1sequential_10/lstm_21/while/loop_counter:output:07sequential_10/lstm_21/while/maximum_iterations:output:0#sequential_10/lstm_21/time:output:0.sequential_10/lstm_21/TensorArrayV2_1:handle:0$sequential_10/lstm_21/zeros:output:0&sequential_10/lstm_21/zeros_1:output:0.sequential_10/lstm_21/strided_slice_1:output:0Msequential_10/lstm_21/TensorArrayUnstack/TensorListFromTensor:output_handle:0Asequential_10_lstm_21_lstm_cell_21_matmul_readvariableop_resourceCsequential_10_lstm_21_lstm_cell_21_matmul_1_readvariableop_resourceBsequential_10_lstm_21_lstm_cell_21_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿÅ:ÿÿÿÿÿÿÿÿÿÅ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *5
body-R+
)sequential_10_lstm_21_while_body_39101013*5
cond-R+
)sequential_10_lstm_21_while_cond_39101012*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿÅ:ÿÿÿÿÿÿÿÿÿÅ: : : : : *
parallel_iterations 2
sequential_10/lstm_21/whileá
Fsequential_10/lstm_21/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿE  2H
Fsequential_10/lstm_21/TensorArrayV2Stack/TensorListStack/element_shapeÁ
8sequential_10/lstm_21/TensorArrayV2Stack/TensorListStackTensorListStack$sequential_10/lstm_21/while:output:3Osequential_10/lstm_21/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ*
element_dtype02:
8sequential_10/lstm_21/TensorArrayV2Stack/TensorListStack­
+sequential_10/lstm_21/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2-
+sequential_10/lstm_21/strided_slice_3/stack¨
-sequential_10/lstm_21/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2/
-sequential_10/lstm_21/strided_slice_3/stack_1¨
-sequential_10/lstm_21/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_10/lstm_21/strided_slice_3/stack_2
%sequential_10/lstm_21/strided_slice_3StridedSliceAsequential_10/lstm_21/TensorArrayV2Stack/TensorListStack:tensor:04sequential_10/lstm_21/strided_slice_3/stack:output:06sequential_10/lstm_21/strided_slice_3/stack_1:output:06sequential_10/lstm_21/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ*
shrink_axis_mask2'
%sequential_10/lstm_21/strided_slice_3¥
&sequential_10/lstm_21/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2(
&sequential_10/lstm_21/transpose_1/permþ
!sequential_10/lstm_21/transpose_1	TransposeAsequential_10/lstm_21/TensorArrayV2Stack/TensorListStack:tensor:0/sequential_10/lstm_21/transpose_1/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2#
!sequential_10/lstm_21/transpose_1
sequential_10/lstm_21/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
sequential_10/lstm_21/runtime°
!sequential_10/dropout_21/IdentityIdentity%sequential_10/lstm_21/transpose_1:y:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2#
!sequential_10/dropout_21/IdentityÜ
/sequential_10/dense_10/Tensordot/ReadVariableOpReadVariableOp8sequential_10_dense_10_tensordot_readvariableop_resource*
_output_shapes
:	Å*
dtype021
/sequential_10/dense_10/Tensordot/ReadVariableOp
%sequential_10/dense_10/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2'
%sequential_10/dense_10/Tensordot/axes
%sequential_10/dense_10/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2'
%sequential_10/dense_10/Tensordot/freeª
&sequential_10/dense_10/Tensordot/ShapeShape*sequential_10/dropout_21/Identity:output:0*
T0*
_output_shapes
:2(
&sequential_10/dense_10/Tensordot/Shape¢
.sequential_10/dense_10/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.sequential_10/dense_10/Tensordot/GatherV2/axisÄ
)sequential_10/dense_10/Tensordot/GatherV2GatherV2/sequential_10/dense_10/Tensordot/Shape:output:0.sequential_10/dense_10/Tensordot/free:output:07sequential_10/dense_10/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2+
)sequential_10/dense_10/Tensordot/GatherV2¦
0sequential_10/dense_10/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 22
0sequential_10/dense_10/Tensordot/GatherV2_1/axisÊ
+sequential_10/dense_10/Tensordot/GatherV2_1GatherV2/sequential_10/dense_10/Tensordot/Shape:output:0.sequential_10/dense_10/Tensordot/axes:output:09sequential_10/dense_10/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2-
+sequential_10/dense_10/Tensordot/GatherV2_1
&sequential_10/dense_10/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2(
&sequential_10/dense_10/Tensordot/ConstÜ
%sequential_10/dense_10/Tensordot/ProdProd2sequential_10/dense_10/Tensordot/GatherV2:output:0/sequential_10/dense_10/Tensordot/Const:output:0*
T0*
_output_shapes
: 2'
%sequential_10/dense_10/Tensordot/Prod
(sequential_10/dense_10/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(sequential_10/dense_10/Tensordot/Const_1ä
'sequential_10/dense_10/Tensordot/Prod_1Prod4sequential_10/dense_10/Tensordot/GatherV2_1:output:01sequential_10/dense_10/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2)
'sequential_10/dense_10/Tensordot/Prod_1
,sequential_10/dense_10/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,sequential_10/dense_10/Tensordot/concat/axis£
'sequential_10/dense_10/Tensordot/concatConcatV2.sequential_10/dense_10/Tensordot/free:output:0.sequential_10/dense_10/Tensordot/axes:output:05sequential_10/dense_10/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2)
'sequential_10/dense_10/Tensordot/concatè
&sequential_10/dense_10/Tensordot/stackPack.sequential_10/dense_10/Tensordot/Prod:output:00sequential_10/dense_10/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2(
&sequential_10/dense_10/Tensordot/stackú
*sequential_10/dense_10/Tensordot/transpose	Transpose*sequential_10/dropout_21/Identity:output:00sequential_10/dense_10/Tensordot/concat:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2,
*sequential_10/dense_10/Tensordot/transposeû
(sequential_10/dense_10/Tensordot/ReshapeReshape.sequential_10/dense_10/Tensordot/transpose:y:0/sequential_10/dense_10/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2*
(sequential_10/dense_10/Tensordot/Reshapeú
'sequential_10/dense_10/Tensordot/MatMulMatMul1sequential_10/dense_10/Tensordot/Reshape:output:07sequential_10/dense_10/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'sequential_10/dense_10/Tensordot/MatMul
(sequential_10/dense_10/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(sequential_10/dense_10/Tensordot/Const_2¢
.sequential_10/dense_10/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.sequential_10/dense_10/Tensordot/concat_1/axis°
)sequential_10/dense_10/Tensordot/concat_1ConcatV22sequential_10/dense_10/Tensordot/GatherV2:output:01sequential_10/dense_10/Tensordot/Const_2:output:07sequential_10/dense_10/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2+
)sequential_10/dense_10/Tensordot/concat_1ì
 sequential_10/dense_10/TensordotReshape1sequential_10/dense_10/Tensordot/MatMul:product:02sequential_10/dense_10/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 sequential_10/dense_10/TensordotÑ
-sequential_10/dense_10/BiasAdd/ReadVariableOpReadVariableOp6sequential_10_dense_10_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02/
-sequential_10/dense_10/BiasAdd/ReadVariableOpã
sequential_10/dense_10/BiasAddBiasAdd)sequential_10/dense_10/Tensordot:output:05sequential_10/dense_10/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
sequential_10/dense_10/BiasAddª
sequential_10/dense_10/SoftmaxSoftmax'sequential_10/dense_10/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
sequential_10/dense_10/Softmax
IdentityIdentity(sequential_10/dense_10/Softmax:softmax:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

IdentityÔ
NoOpNoOp.^sequential_10/dense_10/BiasAdd/ReadVariableOp0^sequential_10/dense_10/Tensordot/ReadVariableOp:^sequential_10/lstm_20/lstm_cell_20/BiasAdd/ReadVariableOp9^sequential_10/lstm_20/lstm_cell_20/MatMul/ReadVariableOp;^sequential_10/lstm_20/lstm_cell_20/MatMul_1/ReadVariableOp^sequential_10/lstm_20/while:^sequential_10/lstm_21/lstm_cell_21/BiasAdd/ReadVariableOp9^sequential_10/lstm_21/lstm_cell_21/MatMul/ReadVariableOp;^sequential_10/lstm_21/lstm_cell_21/MatMul_1/ReadVariableOp^sequential_10/lstm_21/while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ]: : : : : : : : 2^
-sequential_10/dense_10/BiasAdd/ReadVariableOp-sequential_10/dense_10/BiasAdd/ReadVariableOp2b
/sequential_10/dense_10/Tensordot/ReadVariableOp/sequential_10/dense_10/Tensordot/ReadVariableOp2v
9sequential_10/lstm_20/lstm_cell_20/BiasAdd/ReadVariableOp9sequential_10/lstm_20/lstm_cell_20/BiasAdd/ReadVariableOp2t
8sequential_10/lstm_20/lstm_cell_20/MatMul/ReadVariableOp8sequential_10/lstm_20/lstm_cell_20/MatMul/ReadVariableOp2x
:sequential_10/lstm_20/lstm_cell_20/MatMul_1/ReadVariableOp:sequential_10/lstm_20/lstm_cell_20/MatMul_1/ReadVariableOp2:
sequential_10/lstm_20/whilesequential_10/lstm_20/while2v
9sequential_10/lstm_21/lstm_cell_21/BiasAdd/ReadVariableOp9sequential_10/lstm_21/lstm_cell_21/BiasAdd/ReadVariableOp2t
8sequential_10/lstm_21/lstm_cell_21/MatMul/ReadVariableOp8sequential_10/lstm_21/lstm_cell_21/MatMul/ReadVariableOp2x
:sequential_10/lstm_21/lstm_cell_21/MatMul_1/ReadVariableOp:sequential_10/lstm_21/lstm_cell_21/MatMul_1/ReadVariableOp2:
sequential_10/lstm_21/whilesequential_10/lstm_21/while:Z V
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
'
_user_specified_namelstm_20_input
Ô

í
lstm_21_while_cond_39103562,
(lstm_21_while_lstm_21_while_loop_counter2
.lstm_21_while_lstm_21_while_maximum_iterations
lstm_21_while_placeholder
lstm_21_while_placeholder_1
lstm_21_while_placeholder_2
lstm_21_while_placeholder_3.
*lstm_21_while_less_lstm_21_strided_slice_1F
Blstm_21_while_lstm_21_while_cond_39103562___redundant_placeholder0F
Blstm_21_while_lstm_21_while_cond_39103562___redundant_placeholder1F
Blstm_21_while_lstm_21_while_cond_39103562___redundant_placeholder2F
Blstm_21_while_lstm_21_while_cond_39103562___redundant_placeholder3
lstm_21_while_identity

lstm_21/while/LessLesslstm_21_while_placeholder*lstm_21_while_less_lstm_21_strided_slice_1*
T0*
_output_shapes
: 2
lstm_21/while/Lessu
lstm_21/while/IdentityIdentitylstm_21/while/Less:z:0*
T0
*
_output_shapes
: 2
lstm_21/while/Identity"9
lstm_21_while_identitylstm_21/while/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :ÿÿÿÿÿÿÿÿÿÅ:ÿÿÿÿÿÿÿÿÿÅ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ:

_output_shapes
: :

_output_shapes
:
Õ!
þ
F__inference_dense_10_layer_call_and_return_conditional_losses_39102753

inputs4
!tensordot_readvariableop_resource:	Å-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢Tensordot/ReadVariableOp
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes
:	Å*
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
:ÿÿÿÿÿÿÿÿÿÅ2
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
:ÿÿÿÿÿÿÿÿÿÅ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ
 
_user_specified_nameinputs
´?
Ö
while_body_39104951
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
3while_lstm_cell_21_matmul_readvariableop_resource_0:

I
5while_lstm_cell_21_matmul_1_readvariableop_resource_0:
Å
C
4while_lstm_cell_21_biasadd_readvariableop_resource_0:	

while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
1while_lstm_cell_21_matmul_readvariableop_resource:

G
3while_lstm_cell_21_matmul_1_readvariableop_resource:
Å
A
2while_lstm_cell_21_biasadd_readvariableop_resource:	
¢)while/lstm_cell_21/BiasAdd/ReadVariableOp¢(while/lstm_cell_21/MatMul/ReadVariableOp¢*while/lstm_cell_21/MatMul_1/ReadVariableOpÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ  29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÔ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemÊ
(while/lstm_cell_21/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_21_matmul_readvariableop_resource_0* 
_output_shapes
:

*
dtype02*
(while/lstm_cell_21/MatMul/ReadVariableOp×
while/lstm_cell_21/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_21/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
while/lstm_cell_21/MatMulÐ
*while/lstm_cell_21/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_21_matmul_1_readvariableop_resource_0* 
_output_shapes
:
Å
*
dtype02,
*while/lstm_cell_21/MatMul_1/ReadVariableOpÀ
while/lstm_cell_21/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_21/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
while/lstm_cell_21/MatMul_1¸
while/lstm_cell_21/addAddV2#while/lstm_cell_21/MatMul:product:0%while/lstm_cell_21/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
while/lstm_cell_21/addÈ
)while/lstm_cell_21/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_21_biasadd_readvariableop_resource_0*
_output_shapes	
:
*
dtype02+
)while/lstm_cell_21/BiasAdd/ReadVariableOpÅ
while/lstm_cell_21/BiasAddBiasAddwhile/lstm_cell_21/add:z:01while/lstm_cell_21/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
while/lstm_cell_21/BiasAdd
"while/lstm_cell_21/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_21/split/split_dim
while/lstm_cell_21/splitSplit+while/lstm_cell_21/split/split_dim:output:0#while/lstm_cell_21/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿÅ:ÿÿÿÿÿÿÿÿÿÅ:ÿÿÿÿÿÿÿÿÿÅ:ÿÿÿÿÿÿÿÿÿÅ*
	num_split2
while/lstm_cell_21/split
while/lstm_cell_21/SigmoidSigmoid!while/lstm_cell_21/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2
while/lstm_cell_21/Sigmoid
while/lstm_cell_21/Sigmoid_1Sigmoid!while/lstm_cell_21/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2
while/lstm_cell_21/Sigmoid_1¡
while/lstm_cell_21/mulMul while/lstm_cell_21/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2
while/lstm_cell_21/mul
while/lstm_cell_21/ReluRelu!while/lstm_cell_21/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2
while/lstm_cell_21/Reluµ
while/lstm_cell_21/mul_1Mulwhile/lstm_cell_21/Sigmoid:y:0%while/lstm_cell_21/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2
while/lstm_cell_21/mul_1ª
while/lstm_cell_21/add_1AddV2while/lstm_cell_21/mul:z:0while/lstm_cell_21/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2
while/lstm_cell_21/add_1
while/lstm_cell_21/Sigmoid_2Sigmoid!while/lstm_cell_21/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2
while/lstm_cell_21/Sigmoid_2
while/lstm_cell_21/Relu_1Reluwhile/lstm_cell_21/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2
while/lstm_cell_21/Relu_1¹
while/lstm_cell_21/mul_2Mul while/lstm_cell_21/Sigmoid_2:y:0'while/lstm_cell_21/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2
while/lstm_cell_21/mul_2à
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_21/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_21/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_21/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2
while/Identity_5Þ

while/NoOpNoOp*^while/lstm_cell_21/BiasAdd/ReadVariableOp)^while/lstm_cell_21/MatMul/ReadVariableOp+^while/lstm_cell_21/MatMul_1/ReadVariableOp*"
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
2while_lstm_cell_21_biasadd_readvariableop_resource4while_lstm_cell_21_biasadd_readvariableop_resource_0"l
3while_lstm_cell_21_matmul_1_readvariableop_resource5while_lstm_cell_21_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_21_matmul_readvariableop_resource3while_lstm_cell_21_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿÅ:ÿÿÿÿÿÿÿÿÿÅ: : : : : 2V
)while/lstm_cell_21/BiasAdd/ReadVariableOp)while/lstm_cell_21/BiasAdd/ReadVariableOp2T
(while/lstm_cell_21/MatMul/ReadVariableOp(while/lstm_cell_21/MatMul/ReadVariableOp2X
*while/lstm_cell_21/MatMul_1/ReadVariableOp*while/lstm_cell_21/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ:

_output_shapes
: :

_output_shapes
: 
ä^

)sequential_10_lstm_20_while_body_39100865H
Dsequential_10_lstm_20_while_sequential_10_lstm_20_while_loop_counterN
Jsequential_10_lstm_20_while_sequential_10_lstm_20_while_maximum_iterations+
'sequential_10_lstm_20_while_placeholder-
)sequential_10_lstm_20_while_placeholder_1-
)sequential_10_lstm_20_while_placeholder_2-
)sequential_10_lstm_20_while_placeholder_3G
Csequential_10_lstm_20_while_sequential_10_lstm_20_strided_slice_1_0
sequential_10_lstm_20_while_tensorarrayv2read_tensorlistgetitem_sequential_10_lstm_20_tensorarrayunstack_tensorlistfromtensor_0\
Isequential_10_lstm_20_while_lstm_cell_20_matmul_readvariableop_resource_0:	]_
Ksequential_10_lstm_20_while_lstm_cell_20_matmul_1_readvariableop_resource_0:
Y
Jsequential_10_lstm_20_while_lstm_cell_20_biasadd_readvariableop_resource_0:	(
$sequential_10_lstm_20_while_identity*
&sequential_10_lstm_20_while_identity_1*
&sequential_10_lstm_20_while_identity_2*
&sequential_10_lstm_20_while_identity_3*
&sequential_10_lstm_20_while_identity_4*
&sequential_10_lstm_20_while_identity_5E
Asequential_10_lstm_20_while_sequential_10_lstm_20_strided_slice_1
}sequential_10_lstm_20_while_tensorarrayv2read_tensorlistgetitem_sequential_10_lstm_20_tensorarrayunstack_tensorlistfromtensorZ
Gsequential_10_lstm_20_while_lstm_cell_20_matmul_readvariableop_resource:	]]
Isequential_10_lstm_20_while_lstm_cell_20_matmul_1_readvariableop_resource:
W
Hsequential_10_lstm_20_while_lstm_cell_20_biasadd_readvariableop_resource:	¢?sequential_10/lstm_20/while/lstm_cell_20/BiasAdd/ReadVariableOp¢>sequential_10/lstm_20/while/lstm_cell_20/MatMul/ReadVariableOp¢@sequential_10/lstm_20/while/lstm_cell_20/MatMul_1/ReadVariableOpï
Msequential_10/lstm_20/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ]   2O
Msequential_10/lstm_20/while/TensorArrayV2Read/TensorListGetItem/element_shape×
?sequential_10/lstm_20/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemsequential_10_lstm_20_while_tensorarrayv2read_tensorlistgetitem_sequential_10_lstm_20_tensorarrayunstack_tensorlistfromtensor_0'sequential_10_lstm_20_while_placeholderVsequential_10/lstm_20/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]*
element_dtype02A
?sequential_10/lstm_20/while/TensorArrayV2Read/TensorListGetItem
>sequential_10/lstm_20/while/lstm_cell_20/MatMul/ReadVariableOpReadVariableOpIsequential_10_lstm_20_while_lstm_cell_20_matmul_readvariableop_resource_0*
_output_shapes
:	]*
dtype02@
>sequential_10/lstm_20/while/lstm_cell_20/MatMul/ReadVariableOp¯
/sequential_10/lstm_20/while/lstm_cell_20/MatMulMatMulFsequential_10/lstm_20/while/TensorArrayV2Read/TensorListGetItem:item:0Fsequential_10/lstm_20/while/lstm_cell_20/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ21
/sequential_10/lstm_20/while/lstm_cell_20/MatMul
@sequential_10/lstm_20/while/lstm_cell_20/MatMul_1/ReadVariableOpReadVariableOpKsequential_10_lstm_20_while_lstm_cell_20_matmul_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype02B
@sequential_10/lstm_20/while/lstm_cell_20/MatMul_1/ReadVariableOp
1sequential_10/lstm_20/while/lstm_cell_20/MatMul_1MatMul)sequential_10_lstm_20_while_placeholder_2Hsequential_10/lstm_20/while/lstm_cell_20/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ23
1sequential_10/lstm_20/while/lstm_cell_20/MatMul_1
,sequential_10/lstm_20/while/lstm_cell_20/addAddV29sequential_10/lstm_20/while/lstm_cell_20/MatMul:product:0;sequential_10/lstm_20/while/lstm_cell_20/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2.
,sequential_10/lstm_20/while/lstm_cell_20/add
?sequential_10/lstm_20/while/lstm_cell_20/BiasAdd/ReadVariableOpReadVariableOpJsequential_10_lstm_20_while_lstm_cell_20_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02A
?sequential_10/lstm_20/while/lstm_cell_20/BiasAdd/ReadVariableOp
0sequential_10/lstm_20/while/lstm_cell_20/BiasAddBiasAdd0sequential_10/lstm_20/while/lstm_cell_20/add:z:0Gsequential_10/lstm_20/while/lstm_cell_20/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
0sequential_10/lstm_20/while/lstm_cell_20/BiasAdd¶
8sequential_10/lstm_20/while/lstm_cell_20/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2:
8sequential_10/lstm_20/while/lstm_cell_20/split/split_dimç
.sequential_10/lstm_20/while/lstm_cell_20/splitSplitAsequential_10/lstm_20/while/lstm_cell_20/split/split_dim:output:09sequential_10/lstm_20/while/lstm_cell_20/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split20
.sequential_10/lstm_20/while/lstm_cell_20/splitÛ
0sequential_10/lstm_20/while/lstm_cell_20/SigmoidSigmoid7sequential_10/lstm_20/while/lstm_cell_20/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
0sequential_10/lstm_20/while/lstm_cell_20/Sigmoidß
2sequential_10/lstm_20/while/lstm_cell_20/Sigmoid_1Sigmoid7sequential_10/lstm_20/while/lstm_cell_20/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ24
2sequential_10/lstm_20/while/lstm_cell_20/Sigmoid_1ù
,sequential_10/lstm_20/while/lstm_cell_20/mulMul6sequential_10/lstm_20/while/lstm_cell_20/Sigmoid_1:y:0)sequential_10_lstm_20_while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2.
,sequential_10/lstm_20/while/lstm_cell_20/mulÒ
-sequential_10/lstm_20/while/lstm_cell_20/ReluRelu7sequential_10/lstm_20/while/lstm_cell_20/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2/
-sequential_10/lstm_20/while/lstm_cell_20/Relu
.sequential_10/lstm_20/while/lstm_cell_20/mul_1Mul4sequential_10/lstm_20/while/lstm_cell_20/Sigmoid:y:0;sequential_10/lstm_20/while/lstm_cell_20/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ20
.sequential_10/lstm_20/while/lstm_cell_20/mul_1
.sequential_10/lstm_20/while/lstm_cell_20/add_1AddV20sequential_10/lstm_20/while/lstm_cell_20/mul:z:02sequential_10/lstm_20/while/lstm_cell_20/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ20
.sequential_10/lstm_20/while/lstm_cell_20/add_1ß
2sequential_10/lstm_20/while/lstm_cell_20/Sigmoid_2Sigmoid7sequential_10/lstm_20/while/lstm_cell_20/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ24
2sequential_10/lstm_20/while/lstm_cell_20/Sigmoid_2Ñ
/sequential_10/lstm_20/while/lstm_cell_20/Relu_1Relu2sequential_10/lstm_20/while/lstm_cell_20/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ21
/sequential_10/lstm_20/while/lstm_cell_20/Relu_1
.sequential_10/lstm_20/while/lstm_cell_20/mul_2Mul6sequential_10/lstm_20/while/lstm_cell_20/Sigmoid_2:y:0=sequential_10/lstm_20/while/lstm_cell_20/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ20
.sequential_10/lstm_20/while/lstm_cell_20/mul_2Î
@sequential_10/lstm_20/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem)sequential_10_lstm_20_while_placeholder_1'sequential_10_lstm_20_while_placeholder2sequential_10/lstm_20/while/lstm_cell_20/mul_2:z:0*
_output_shapes
: *
element_dtype02B
@sequential_10/lstm_20/while/TensorArrayV2Write/TensorListSetItem
!sequential_10/lstm_20/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2#
!sequential_10/lstm_20/while/add/yÁ
sequential_10/lstm_20/while/addAddV2'sequential_10_lstm_20_while_placeholder*sequential_10/lstm_20/while/add/y:output:0*
T0*
_output_shapes
: 2!
sequential_10/lstm_20/while/add
#sequential_10/lstm_20/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2%
#sequential_10/lstm_20/while/add_1/yä
!sequential_10/lstm_20/while/add_1AddV2Dsequential_10_lstm_20_while_sequential_10_lstm_20_while_loop_counter,sequential_10/lstm_20/while/add_1/y:output:0*
T0*
_output_shapes
: 2#
!sequential_10/lstm_20/while/add_1Ã
$sequential_10/lstm_20/while/IdentityIdentity%sequential_10/lstm_20/while/add_1:z:0!^sequential_10/lstm_20/while/NoOp*
T0*
_output_shapes
: 2&
$sequential_10/lstm_20/while/Identityì
&sequential_10/lstm_20/while/Identity_1IdentityJsequential_10_lstm_20_while_sequential_10_lstm_20_while_maximum_iterations!^sequential_10/lstm_20/while/NoOp*
T0*
_output_shapes
: 2(
&sequential_10/lstm_20/while/Identity_1Å
&sequential_10/lstm_20/while/Identity_2Identity#sequential_10/lstm_20/while/add:z:0!^sequential_10/lstm_20/while/NoOp*
T0*
_output_shapes
: 2(
&sequential_10/lstm_20/while/Identity_2ò
&sequential_10/lstm_20/while/Identity_3IdentityPsequential_10/lstm_20/while/TensorArrayV2Write/TensorListSetItem:output_handle:0!^sequential_10/lstm_20/while/NoOp*
T0*
_output_shapes
: 2(
&sequential_10/lstm_20/while/Identity_3æ
&sequential_10/lstm_20/while/Identity_4Identity2sequential_10/lstm_20/while/lstm_cell_20/mul_2:z:0!^sequential_10/lstm_20/while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&sequential_10/lstm_20/while/Identity_4æ
&sequential_10/lstm_20/while/Identity_5Identity2sequential_10/lstm_20/while/lstm_cell_20/add_1:z:0!^sequential_10/lstm_20/while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&sequential_10/lstm_20/while/Identity_5Ì
 sequential_10/lstm_20/while/NoOpNoOp@^sequential_10/lstm_20/while/lstm_cell_20/BiasAdd/ReadVariableOp?^sequential_10/lstm_20/while/lstm_cell_20/MatMul/ReadVariableOpA^sequential_10/lstm_20/while/lstm_cell_20/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2"
 sequential_10/lstm_20/while/NoOp"U
$sequential_10_lstm_20_while_identity-sequential_10/lstm_20/while/Identity:output:0"Y
&sequential_10_lstm_20_while_identity_1/sequential_10/lstm_20/while/Identity_1:output:0"Y
&sequential_10_lstm_20_while_identity_2/sequential_10/lstm_20/while/Identity_2:output:0"Y
&sequential_10_lstm_20_while_identity_3/sequential_10/lstm_20/while/Identity_3:output:0"Y
&sequential_10_lstm_20_while_identity_4/sequential_10/lstm_20/while/Identity_4:output:0"Y
&sequential_10_lstm_20_while_identity_5/sequential_10/lstm_20/while/Identity_5:output:0"
Hsequential_10_lstm_20_while_lstm_cell_20_biasadd_readvariableop_resourceJsequential_10_lstm_20_while_lstm_cell_20_biasadd_readvariableop_resource_0"
Isequential_10_lstm_20_while_lstm_cell_20_matmul_1_readvariableop_resourceKsequential_10_lstm_20_while_lstm_cell_20_matmul_1_readvariableop_resource_0"
Gsequential_10_lstm_20_while_lstm_cell_20_matmul_readvariableop_resourceIsequential_10_lstm_20_while_lstm_cell_20_matmul_readvariableop_resource_0"
Asequential_10_lstm_20_while_sequential_10_lstm_20_strided_slice_1Csequential_10_lstm_20_while_sequential_10_lstm_20_strided_slice_1_0"
}sequential_10_lstm_20_while_tensorarrayv2read_tensorlistgetitem_sequential_10_lstm_20_tensorarrayunstack_tensorlistfromtensorsequential_10_lstm_20_while_tensorarrayv2read_tensorlistgetitem_sequential_10_lstm_20_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2
?sequential_10/lstm_20/while/lstm_cell_20/BiasAdd/ReadVariableOp?sequential_10/lstm_20/while/lstm_cell_20/BiasAdd/ReadVariableOp2
>sequential_10/lstm_20/while/lstm_cell_20/MatMul/ReadVariableOp>sequential_10/lstm_20/while/lstm_cell_20/MatMul/ReadVariableOp2
@sequential_10/lstm_20/while/lstm_cell_20/MatMul_1/ReadVariableOp@sequential_10/lstm_20/while/lstm_cell_20/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 

f
H__inference_dropout_20_layer_call_and_return_conditional_losses_39104711

inputs

identity_1_
IdentityIdentityinputs*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityn

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ÐF

E__inference_lstm_21_layer_call_and_return_conditional_losses_39101913

inputs)
lstm_cell_21_39101831:

)
lstm_cell_21_39101833:
Å
$
lstm_cell_21_39101835:	

identity¢$lstm_cell_21/StatefulPartitionedCall¢whileD
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
B :Å2
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
B :Å2
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
:ÿÿÿÿÿÿÿÿÿÅ2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :Å2
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
B :Å2
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
:ÿÿÿÿÿÿÿÿÿÅ2	
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
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
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
valueB"ÿÿÿÿ  27
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
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
strided_slice_2¨
$lstm_cell_21/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_21_39101831lstm_cell_21_39101833lstm_cell_21_39101835*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿÅ:ÿÿÿÿÿÿÿÿÿÅ:ÿÿÿÿÿÿÿÿÿÅ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_lstm_cell_21_layer_call_and_return_conditional_losses_391018302&
$lstm_cell_21/StatefulPartitionedCall
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿE  2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_21_39101831lstm_cell_21_39101833lstm_cell_21_39101835*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿÅ:ÿÿÿÿÿÿÿÿÿÅ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_39101844*
condR
while_cond_39101843*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿÅ:ÿÿÿÿÿÿÿÿÿÅ: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿE  22
0TensorArrayV2Stack/TensorListStack/element_shapeò
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÅ*
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
:ÿÿÿÿÿÿÿÿÿÅ*
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
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÅ2
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
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÅ2

Identity}
NoOpNoOp%^lstm_cell_21/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 2L
$lstm_cell_21/StatefulPartitionedCall$lstm_cell_21/StatefulPartitionedCall2
whilewhile:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ã
Í
while_cond_39103087
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_39103087___redundant_placeholder06
2while_while_cond_39103087___redundant_placeholder16
2while_while_cond_39103087___redundant_placeholder26
2while_while_cond_39103087___redundant_placeholder3
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
B: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
ã
Í
while_cond_39104799
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_39104799___redundant_placeholder06
2while_while_cond_39104799___redundant_placeholder16
2while_while_cond_39104799___redundant_placeholder26
2while_while_cond_39104799___redundant_placeholder3
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
B: : : : :ÿÿÿÿÿÿÿÿÿÅ:ÿÿÿÿÿÿÿÿÿÅ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ:

_output_shapes
: :

_output_shapes
:
åJ
Ô

lstm_20_while_body_39103742,
(lstm_20_while_lstm_20_while_loop_counter2
.lstm_20_while_lstm_20_while_maximum_iterations
lstm_20_while_placeholder
lstm_20_while_placeholder_1
lstm_20_while_placeholder_2
lstm_20_while_placeholder_3+
'lstm_20_while_lstm_20_strided_slice_1_0g
clstm_20_while_tensorarrayv2read_tensorlistgetitem_lstm_20_tensorarrayunstack_tensorlistfromtensor_0N
;lstm_20_while_lstm_cell_20_matmul_readvariableop_resource_0:	]Q
=lstm_20_while_lstm_cell_20_matmul_1_readvariableop_resource_0:
K
<lstm_20_while_lstm_cell_20_biasadd_readvariableop_resource_0:	
lstm_20_while_identity
lstm_20_while_identity_1
lstm_20_while_identity_2
lstm_20_while_identity_3
lstm_20_while_identity_4
lstm_20_while_identity_5)
%lstm_20_while_lstm_20_strided_slice_1e
alstm_20_while_tensorarrayv2read_tensorlistgetitem_lstm_20_tensorarrayunstack_tensorlistfromtensorL
9lstm_20_while_lstm_cell_20_matmul_readvariableop_resource:	]O
;lstm_20_while_lstm_cell_20_matmul_1_readvariableop_resource:
I
:lstm_20_while_lstm_cell_20_biasadd_readvariableop_resource:	¢1lstm_20/while/lstm_cell_20/BiasAdd/ReadVariableOp¢0lstm_20/while/lstm_cell_20/MatMul/ReadVariableOp¢2lstm_20/while/lstm_cell_20/MatMul_1/ReadVariableOpÓ
?lstm_20/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ]   2A
?lstm_20/while/TensorArrayV2Read/TensorListGetItem/element_shape
1lstm_20/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemclstm_20_while_tensorarrayv2read_tensorlistgetitem_lstm_20_tensorarrayunstack_tensorlistfromtensor_0lstm_20_while_placeholderHlstm_20/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]*
element_dtype023
1lstm_20/while/TensorArrayV2Read/TensorListGetItemá
0lstm_20/while/lstm_cell_20/MatMul/ReadVariableOpReadVariableOp;lstm_20_while_lstm_cell_20_matmul_readvariableop_resource_0*
_output_shapes
:	]*
dtype022
0lstm_20/while/lstm_cell_20/MatMul/ReadVariableOp÷
!lstm_20/while/lstm_cell_20/MatMulMatMul8lstm_20/while/TensorArrayV2Read/TensorListGetItem:item:08lstm_20/while/lstm_cell_20/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!lstm_20/while/lstm_cell_20/MatMulè
2lstm_20/while/lstm_cell_20/MatMul_1/ReadVariableOpReadVariableOp=lstm_20_while_lstm_cell_20_matmul_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype024
2lstm_20/while/lstm_cell_20/MatMul_1/ReadVariableOpà
#lstm_20/while/lstm_cell_20/MatMul_1MatMullstm_20_while_placeholder_2:lstm_20/while/lstm_cell_20/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2%
#lstm_20/while/lstm_cell_20/MatMul_1Ø
lstm_20/while/lstm_cell_20/addAddV2+lstm_20/while/lstm_cell_20/MatMul:product:0-lstm_20/while/lstm_cell_20/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
lstm_20/while/lstm_cell_20/addà
1lstm_20/while/lstm_cell_20/BiasAdd/ReadVariableOpReadVariableOp<lstm_20_while_lstm_cell_20_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype023
1lstm_20/while/lstm_cell_20/BiasAdd/ReadVariableOpå
"lstm_20/while/lstm_cell_20/BiasAddBiasAdd"lstm_20/while/lstm_cell_20/add:z:09lstm_20/while/lstm_cell_20/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2$
"lstm_20/while/lstm_cell_20/BiasAdd
*lstm_20/while/lstm_cell_20/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2,
*lstm_20/while/lstm_cell_20/split/split_dim¯
 lstm_20/while/lstm_cell_20/splitSplit3lstm_20/while/lstm_cell_20/split/split_dim:output:0+lstm_20/while/lstm_cell_20/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2"
 lstm_20/while/lstm_cell_20/split±
"lstm_20/while/lstm_cell_20/SigmoidSigmoid)lstm_20/while/lstm_cell_20/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2$
"lstm_20/while/lstm_cell_20/Sigmoidµ
$lstm_20/while/lstm_cell_20/Sigmoid_1Sigmoid)lstm_20/while/lstm_cell_20/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$lstm_20/while/lstm_cell_20/Sigmoid_1Á
lstm_20/while/lstm_cell_20/mulMul(lstm_20/while/lstm_cell_20/Sigmoid_1:y:0lstm_20_while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
lstm_20/while/lstm_cell_20/mul¨
lstm_20/while/lstm_cell_20/ReluRelu)lstm_20/while/lstm_cell_20/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
lstm_20/while/lstm_cell_20/ReluÕ
 lstm_20/while/lstm_cell_20/mul_1Mul&lstm_20/while/lstm_cell_20/Sigmoid:y:0-lstm_20/while/lstm_cell_20/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 lstm_20/while/lstm_cell_20/mul_1Ê
 lstm_20/while/lstm_cell_20/add_1AddV2"lstm_20/while/lstm_cell_20/mul:z:0$lstm_20/while/lstm_cell_20/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 lstm_20/while/lstm_cell_20/add_1µ
$lstm_20/while/lstm_cell_20/Sigmoid_2Sigmoid)lstm_20/while/lstm_cell_20/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$lstm_20/while/lstm_cell_20/Sigmoid_2§
!lstm_20/while/lstm_cell_20/Relu_1Relu$lstm_20/while/lstm_cell_20/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!lstm_20/while/lstm_cell_20/Relu_1Ù
 lstm_20/while/lstm_cell_20/mul_2Mul(lstm_20/while/lstm_cell_20/Sigmoid_2:y:0/lstm_20/while/lstm_cell_20/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 lstm_20/while/lstm_cell_20/mul_2
2lstm_20/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_20_while_placeholder_1lstm_20_while_placeholder$lstm_20/while/lstm_cell_20/mul_2:z:0*
_output_shapes
: *
element_dtype024
2lstm_20/while/TensorArrayV2Write/TensorListSetIteml
lstm_20/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_20/while/add/y
lstm_20/while/addAddV2lstm_20_while_placeholderlstm_20/while/add/y:output:0*
T0*
_output_shapes
: 2
lstm_20/while/addp
lstm_20/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_20/while/add_1/y
lstm_20/while/add_1AddV2(lstm_20_while_lstm_20_while_loop_counterlstm_20/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lstm_20/while/add_1
lstm_20/while/IdentityIdentitylstm_20/while/add_1:z:0^lstm_20/while/NoOp*
T0*
_output_shapes
: 2
lstm_20/while/Identity¦
lstm_20/while/Identity_1Identity.lstm_20_while_lstm_20_while_maximum_iterations^lstm_20/while/NoOp*
T0*
_output_shapes
: 2
lstm_20/while/Identity_1
lstm_20/while/Identity_2Identitylstm_20/while/add:z:0^lstm_20/while/NoOp*
T0*
_output_shapes
: 2
lstm_20/while/Identity_2º
lstm_20/while/Identity_3IdentityBlstm_20/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_20/while/NoOp*
T0*
_output_shapes
: 2
lstm_20/while/Identity_3®
lstm_20/while/Identity_4Identity$lstm_20/while/lstm_cell_20/mul_2:z:0^lstm_20/while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_20/while/Identity_4®
lstm_20/while/Identity_5Identity$lstm_20/while/lstm_cell_20/add_1:z:0^lstm_20/while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_20/while/Identity_5
lstm_20/while/NoOpNoOp2^lstm_20/while/lstm_cell_20/BiasAdd/ReadVariableOp1^lstm_20/while/lstm_cell_20/MatMul/ReadVariableOp3^lstm_20/while/lstm_cell_20/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
lstm_20/while/NoOp"9
lstm_20_while_identitylstm_20/while/Identity:output:0"=
lstm_20_while_identity_1!lstm_20/while/Identity_1:output:0"=
lstm_20_while_identity_2!lstm_20/while/Identity_2:output:0"=
lstm_20_while_identity_3!lstm_20/while/Identity_3:output:0"=
lstm_20_while_identity_4!lstm_20/while/Identity_4:output:0"=
lstm_20_while_identity_5!lstm_20/while/Identity_5:output:0"P
%lstm_20_while_lstm_20_strided_slice_1'lstm_20_while_lstm_20_strided_slice_1_0"z
:lstm_20_while_lstm_cell_20_biasadd_readvariableop_resource<lstm_20_while_lstm_cell_20_biasadd_readvariableop_resource_0"|
;lstm_20_while_lstm_cell_20_matmul_1_readvariableop_resource=lstm_20_while_lstm_cell_20_matmul_1_readvariableop_resource_0"x
9lstm_20_while_lstm_cell_20_matmul_readvariableop_resource;lstm_20_while_lstm_cell_20_matmul_readvariableop_resource_0"È
alstm_20_while_tensorarrayv2read_tensorlistgetitem_lstm_20_tensorarrayunstack_tensorlistfromtensorclstm_20_while_tensorarrayv2read_tensorlistgetitem_lstm_20_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2f
1lstm_20/while/lstm_cell_20/BiasAdd/ReadVariableOp1lstm_20/while/lstm_cell_20/BiasAdd/ReadVariableOp2d
0lstm_20/while/lstm_cell_20/MatMul/ReadVariableOp0lstm_20/while/lstm_cell_20/MatMul/ReadVariableOp2h
2lstm_20/while/lstm_cell_20/MatMul_1/ReadVariableOp2lstm_20/while/lstm_cell_20/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
»
f
-__inference_dropout_20_layer_call_fn_39104733

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
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_dropout_20_layer_call_and_return_conditional_losses_391030052
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ò

K__inference_sequential_10_layer_call_and_return_conditional_losses_39104016

inputsF
3lstm_20_lstm_cell_20_matmul_readvariableop_resource:	]I
5lstm_20_lstm_cell_20_matmul_1_readvariableop_resource:
C
4lstm_20_lstm_cell_20_biasadd_readvariableop_resource:	G
3lstm_21_lstm_cell_21_matmul_readvariableop_resource:

I
5lstm_21_lstm_cell_21_matmul_1_readvariableop_resource:
Å
C
4lstm_21_lstm_cell_21_biasadd_readvariableop_resource:	
=
*dense_10_tensordot_readvariableop_resource:	Å6
(dense_10_biasadd_readvariableop_resource:
identity¢dense_10/BiasAdd/ReadVariableOp¢!dense_10/Tensordot/ReadVariableOp¢+lstm_20/lstm_cell_20/BiasAdd/ReadVariableOp¢*lstm_20/lstm_cell_20/MatMul/ReadVariableOp¢,lstm_20/lstm_cell_20/MatMul_1/ReadVariableOp¢lstm_20/while¢+lstm_21/lstm_cell_21/BiasAdd/ReadVariableOp¢*lstm_21/lstm_cell_21/MatMul/ReadVariableOp¢,lstm_21/lstm_cell_21/MatMul_1/ReadVariableOp¢lstm_21/whileT
lstm_20/ShapeShapeinputs*
T0*
_output_shapes
:2
lstm_20/Shape
lstm_20/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_20/strided_slice/stack
lstm_20/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_20/strided_slice/stack_1
lstm_20/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_20/strided_slice/stack_2
lstm_20/strided_sliceStridedSlicelstm_20/Shape:output:0$lstm_20/strided_slice/stack:output:0&lstm_20/strided_slice/stack_1:output:0&lstm_20/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_20/strided_slicem
lstm_20/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :2
lstm_20/zeros/mul/y
lstm_20/zeros/mulMullstm_20/strided_slice:output:0lstm_20/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_20/zeros/mulo
lstm_20/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
lstm_20/zeros/Less/y
lstm_20/zeros/LessLesslstm_20/zeros/mul:z:0lstm_20/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_20/zeros/Lesss
lstm_20/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :2
lstm_20/zeros/packed/1£
lstm_20/zeros/packedPacklstm_20/strided_slice:output:0lstm_20/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_20/zeros/packedo
lstm_20/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_20/zeros/Const
lstm_20/zerosFilllstm_20/zeros/packed:output:0lstm_20/zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_20/zerosq
lstm_20/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :2
lstm_20/zeros_1/mul/y
lstm_20/zeros_1/mulMullstm_20/strided_slice:output:0lstm_20/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_20/zeros_1/muls
lstm_20/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
lstm_20/zeros_1/Less/y
lstm_20/zeros_1/LessLesslstm_20/zeros_1/mul:z:0lstm_20/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_20/zeros_1/Lessw
lstm_20/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :2
lstm_20/zeros_1/packed/1©
lstm_20/zeros_1/packedPacklstm_20/strided_slice:output:0!lstm_20/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_20/zeros_1/packeds
lstm_20/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_20/zeros_1/Const
lstm_20/zeros_1Filllstm_20/zeros_1/packed:output:0lstm_20/zeros_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_20/zeros_1
lstm_20/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_20/transpose/perm
lstm_20/transpose	Transposeinputslstm_20/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]2
lstm_20/transposeg
lstm_20/Shape_1Shapelstm_20/transpose:y:0*
T0*
_output_shapes
:2
lstm_20/Shape_1
lstm_20/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_20/strided_slice_1/stack
lstm_20/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_20/strided_slice_1/stack_1
lstm_20/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_20/strided_slice_1/stack_2
lstm_20/strided_slice_1StridedSlicelstm_20/Shape_1:output:0&lstm_20/strided_slice_1/stack:output:0(lstm_20/strided_slice_1/stack_1:output:0(lstm_20/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_20/strided_slice_1
#lstm_20/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2%
#lstm_20/TensorArrayV2/element_shapeÒ
lstm_20/TensorArrayV2TensorListReserve,lstm_20/TensorArrayV2/element_shape:output:0 lstm_20/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_20/TensorArrayV2Ï
=lstm_20/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ]   2?
=lstm_20/TensorArrayUnstack/TensorListFromTensor/element_shape
/lstm_20/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_20/transpose:y:0Flstm_20/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type021
/lstm_20/TensorArrayUnstack/TensorListFromTensor
lstm_20/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_20/strided_slice_2/stack
lstm_20/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_20/strided_slice_2/stack_1
lstm_20/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_20/strided_slice_2/stack_2¬
lstm_20/strided_slice_2StridedSlicelstm_20/transpose:y:0&lstm_20/strided_slice_2/stack:output:0(lstm_20/strided_slice_2/stack_1:output:0(lstm_20/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]*
shrink_axis_mask2
lstm_20/strided_slice_2Í
*lstm_20/lstm_cell_20/MatMul/ReadVariableOpReadVariableOp3lstm_20_lstm_cell_20_matmul_readvariableop_resource*
_output_shapes
:	]*
dtype02,
*lstm_20/lstm_cell_20/MatMul/ReadVariableOpÍ
lstm_20/lstm_cell_20/MatMulMatMul lstm_20/strided_slice_2:output:02lstm_20/lstm_cell_20/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_20/lstm_cell_20/MatMulÔ
,lstm_20/lstm_cell_20/MatMul_1/ReadVariableOpReadVariableOp5lstm_20_lstm_cell_20_matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype02.
,lstm_20/lstm_cell_20/MatMul_1/ReadVariableOpÉ
lstm_20/lstm_cell_20/MatMul_1MatMullstm_20/zeros:output:04lstm_20/lstm_cell_20/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_20/lstm_cell_20/MatMul_1À
lstm_20/lstm_cell_20/addAddV2%lstm_20/lstm_cell_20/MatMul:product:0'lstm_20/lstm_cell_20/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_20/lstm_cell_20/addÌ
+lstm_20/lstm_cell_20/BiasAdd/ReadVariableOpReadVariableOp4lstm_20_lstm_cell_20_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02-
+lstm_20/lstm_cell_20/BiasAdd/ReadVariableOpÍ
lstm_20/lstm_cell_20/BiasAddBiasAddlstm_20/lstm_cell_20/add:z:03lstm_20/lstm_cell_20/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_20/lstm_cell_20/BiasAdd
$lstm_20/lstm_cell_20/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2&
$lstm_20/lstm_cell_20/split/split_dim
lstm_20/lstm_cell_20/splitSplit-lstm_20/lstm_cell_20/split/split_dim:output:0%lstm_20/lstm_cell_20/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2
lstm_20/lstm_cell_20/split
lstm_20/lstm_cell_20/SigmoidSigmoid#lstm_20/lstm_cell_20/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_20/lstm_cell_20/Sigmoid£
lstm_20/lstm_cell_20/Sigmoid_1Sigmoid#lstm_20/lstm_cell_20/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
lstm_20/lstm_cell_20/Sigmoid_1¬
lstm_20/lstm_cell_20/mulMul"lstm_20/lstm_cell_20/Sigmoid_1:y:0lstm_20/zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_20/lstm_cell_20/mul
lstm_20/lstm_cell_20/ReluRelu#lstm_20/lstm_cell_20/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_20/lstm_cell_20/Relu½
lstm_20/lstm_cell_20/mul_1Mul lstm_20/lstm_cell_20/Sigmoid:y:0'lstm_20/lstm_cell_20/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_20/lstm_cell_20/mul_1²
lstm_20/lstm_cell_20/add_1AddV2lstm_20/lstm_cell_20/mul:z:0lstm_20/lstm_cell_20/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_20/lstm_cell_20/add_1£
lstm_20/lstm_cell_20/Sigmoid_2Sigmoid#lstm_20/lstm_cell_20/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
lstm_20/lstm_cell_20/Sigmoid_2
lstm_20/lstm_cell_20/Relu_1Relulstm_20/lstm_cell_20/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_20/lstm_cell_20/Relu_1Á
lstm_20/lstm_cell_20/mul_2Mul"lstm_20/lstm_cell_20/Sigmoid_2:y:0)lstm_20/lstm_cell_20/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_20/lstm_cell_20/mul_2
%lstm_20/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ  2'
%lstm_20/TensorArrayV2_1/element_shapeØ
lstm_20/TensorArrayV2_1TensorListReserve.lstm_20/TensorArrayV2_1/element_shape:output:0 lstm_20/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_20/TensorArrayV2_1^
lstm_20/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_20/time
 lstm_20/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2"
 lstm_20/while/maximum_iterationsz
lstm_20/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_20/while/loop_counter
lstm_20/whileWhile#lstm_20/while/loop_counter:output:0)lstm_20/while/maximum_iterations:output:0lstm_20/time:output:0 lstm_20/TensorArrayV2_1:handle:0lstm_20/zeros:output:0lstm_20/zeros_1:output:0 lstm_20/strided_slice_1:output:0?lstm_20/TensorArrayUnstack/TensorListFromTensor:output_handle:03lstm_20_lstm_cell_20_matmul_readvariableop_resource5lstm_20_lstm_cell_20_matmul_1_readvariableop_resource4lstm_20_lstm_cell_20_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *'
bodyR
lstm_20_while_body_39103742*'
condR
lstm_20_while_cond_39103741*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 2
lstm_20/whileÅ
8lstm_20/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ  2:
8lstm_20/TensorArrayV2Stack/TensorListStack/element_shape
*lstm_20/TensorArrayV2Stack/TensorListStackTensorListStacklstm_20/while:output:3Alstm_20/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype02,
*lstm_20/TensorArrayV2Stack/TensorListStack
lstm_20/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
lstm_20/strided_slice_3/stack
lstm_20/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2!
lstm_20/strided_slice_3/stack_1
lstm_20/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_20/strided_slice_3/stack_2Ë
lstm_20/strided_slice_3StridedSlice3lstm_20/TensorArrayV2Stack/TensorListStack:tensor:0&lstm_20/strided_slice_3/stack:output:0(lstm_20/strided_slice_3/stack_1:output:0(lstm_20/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
lstm_20/strided_slice_3
lstm_20/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_20/transpose_1/permÆ
lstm_20/transpose_1	Transpose3lstm_20/TensorArrayV2Stack/TensorListStack:tensor:0!lstm_20/transpose_1/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_20/transpose_1v
lstm_20/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_20/runtimey
dropout_20/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *«ªª?2
dropout_20/dropout/Constª
dropout_20/dropout/MulMullstm_20/transpose_1:y:0!dropout_20/dropout/Const:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout_20/dropout/Mul{
dropout_20/dropout/ShapeShapelstm_20/transpose_1:y:0*
T0*
_output_shapes
:2
dropout_20/dropout/ShapeÚ
/dropout_20/dropout/random_uniform/RandomUniformRandomUniform!dropout_20/dropout/Shape:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype021
/dropout_20/dropout/random_uniform/RandomUniform
!dropout_20/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  >2#
!dropout_20/dropout/GreaterEqual/yï
dropout_20/dropout/GreaterEqualGreaterEqual8dropout_20/dropout/random_uniform/RandomUniform:output:0*dropout_20/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
dropout_20/dropout/GreaterEqual¥
dropout_20/dropout/CastCast#dropout_20/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout_20/dropout/Cast«
dropout_20/dropout/Mul_1Muldropout_20/dropout/Mul:z:0dropout_20/dropout/Cast:y:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout_20/dropout/Mul_1j
lstm_21/ShapeShapedropout_20/dropout/Mul_1:z:0*
T0*
_output_shapes
:2
lstm_21/Shape
lstm_21/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_21/strided_slice/stack
lstm_21/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_21/strided_slice/stack_1
lstm_21/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_21/strided_slice/stack_2
lstm_21/strided_sliceStridedSlicelstm_21/Shape:output:0$lstm_21/strided_slice/stack:output:0&lstm_21/strided_slice/stack_1:output:0&lstm_21/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_21/strided_slicem
lstm_21/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :Å2
lstm_21/zeros/mul/y
lstm_21/zeros/mulMullstm_21/strided_slice:output:0lstm_21/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_21/zeros/mulo
lstm_21/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
lstm_21/zeros/Less/y
lstm_21/zeros/LessLesslstm_21/zeros/mul:z:0lstm_21/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_21/zeros/Lesss
lstm_21/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :Å2
lstm_21/zeros/packed/1£
lstm_21/zeros/packedPacklstm_21/strided_slice:output:0lstm_21/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_21/zeros/packedo
lstm_21/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_21/zeros/Const
lstm_21/zerosFilllstm_21/zeros/packed:output:0lstm_21/zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2
lstm_21/zerosq
lstm_21/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :Å2
lstm_21/zeros_1/mul/y
lstm_21/zeros_1/mulMullstm_21/strided_slice:output:0lstm_21/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_21/zeros_1/muls
lstm_21/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
lstm_21/zeros_1/Less/y
lstm_21/zeros_1/LessLesslstm_21/zeros_1/mul:z:0lstm_21/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_21/zeros_1/Lessw
lstm_21/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :Å2
lstm_21/zeros_1/packed/1©
lstm_21/zeros_1/packedPacklstm_21/strided_slice:output:0!lstm_21/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_21/zeros_1/packeds
lstm_21/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_21/zeros_1/Const
lstm_21/zeros_1Filllstm_21/zeros_1/packed:output:0lstm_21/zeros_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2
lstm_21/zeros_1
lstm_21/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_21/transpose/perm©
lstm_21/transpose	Transposedropout_20/dropout/Mul_1:z:0lstm_21/transpose/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_21/transposeg
lstm_21/Shape_1Shapelstm_21/transpose:y:0*
T0*
_output_shapes
:2
lstm_21/Shape_1
lstm_21/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_21/strided_slice_1/stack
lstm_21/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_21/strided_slice_1/stack_1
lstm_21/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_21/strided_slice_1/stack_2
lstm_21/strided_slice_1StridedSlicelstm_21/Shape_1:output:0&lstm_21/strided_slice_1/stack:output:0(lstm_21/strided_slice_1/stack_1:output:0(lstm_21/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_21/strided_slice_1
#lstm_21/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2%
#lstm_21/TensorArrayV2/element_shapeÒ
lstm_21/TensorArrayV2TensorListReserve,lstm_21/TensorArrayV2/element_shape:output:0 lstm_21/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_21/TensorArrayV2Ï
=lstm_21/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ  2?
=lstm_21/TensorArrayUnstack/TensorListFromTensor/element_shape
/lstm_21/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_21/transpose:y:0Flstm_21/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type021
/lstm_21/TensorArrayUnstack/TensorListFromTensor
lstm_21/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_21/strided_slice_2/stack
lstm_21/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_21/strided_slice_2/stack_1
lstm_21/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_21/strided_slice_2/stack_2­
lstm_21/strided_slice_2StridedSlicelstm_21/transpose:y:0&lstm_21/strided_slice_2/stack:output:0(lstm_21/strided_slice_2/stack_1:output:0(lstm_21/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
lstm_21/strided_slice_2Î
*lstm_21/lstm_cell_21/MatMul/ReadVariableOpReadVariableOp3lstm_21_lstm_cell_21_matmul_readvariableop_resource* 
_output_shapes
:

*
dtype02,
*lstm_21/lstm_cell_21/MatMul/ReadVariableOpÍ
lstm_21/lstm_cell_21/MatMulMatMul lstm_21/strided_slice_2:output:02lstm_21/lstm_cell_21/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
lstm_21/lstm_cell_21/MatMulÔ
,lstm_21/lstm_cell_21/MatMul_1/ReadVariableOpReadVariableOp5lstm_21_lstm_cell_21_matmul_1_readvariableop_resource* 
_output_shapes
:
Å
*
dtype02.
,lstm_21/lstm_cell_21/MatMul_1/ReadVariableOpÉ
lstm_21/lstm_cell_21/MatMul_1MatMullstm_21/zeros:output:04lstm_21/lstm_cell_21/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
lstm_21/lstm_cell_21/MatMul_1À
lstm_21/lstm_cell_21/addAddV2%lstm_21/lstm_cell_21/MatMul:product:0'lstm_21/lstm_cell_21/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
lstm_21/lstm_cell_21/addÌ
+lstm_21/lstm_cell_21/BiasAdd/ReadVariableOpReadVariableOp4lstm_21_lstm_cell_21_biasadd_readvariableop_resource*
_output_shapes	
:
*
dtype02-
+lstm_21/lstm_cell_21/BiasAdd/ReadVariableOpÍ
lstm_21/lstm_cell_21/BiasAddBiasAddlstm_21/lstm_cell_21/add:z:03lstm_21/lstm_cell_21/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
lstm_21/lstm_cell_21/BiasAdd
$lstm_21/lstm_cell_21/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2&
$lstm_21/lstm_cell_21/split/split_dim
lstm_21/lstm_cell_21/splitSplit-lstm_21/lstm_cell_21/split/split_dim:output:0%lstm_21/lstm_cell_21/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿÅ:ÿÿÿÿÿÿÿÿÿÅ:ÿÿÿÿÿÿÿÿÿÅ:ÿÿÿÿÿÿÿÿÿÅ*
	num_split2
lstm_21/lstm_cell_21/split
lstm_21/lstm_cell_21/SigmoidSigmoid#lstm_21/lstm_cell_21/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2
lstm_21/lstm_cell_21/Sigmoid£
lstm_21/lstm_cell_21/Sigmoid_1Sigmoid#lstm_21/lstm_cell_21/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2 
lstm_21/lstm_cell_21/Sigmoid_1¬
lstm_21/lstm_cell_21/mulMul"lstm_21/lstm_cell_21/Sigmoid_1:y:0lstm_21/zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2
lstm_21/lstm_cell_21/mul
lstm_21/lstm_cell_21/ReluRelu#lstm_21/lstm_cell_21/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2
lstm_21/lstm_cell_21/Relu½
lstm_21/lstm_cell_21/mul_1Mul lstm_21/lstm_cell_21/Sigmoid:y:0'lstm_21/lstm_cell_21/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2
lstm_21/lstm_cell_21/mul_1²
lstm_21/lstm_cell_21/add_1AddV2lstm_21/lstm_cell_21/mul:z:0lstm_21/lstm_cell_21/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2
lstm_21/lstm_cell_21/add_1£
lstm_21/lstm_cell_21/Sigmoid_2Sigmoid#lstm_21/lstm_cell_21/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2 
lstm_21/lstm_cell_21/Sigmoid_2
lstm_21/lstm_cell_21/Relu_1Relulstm_21/lstm_cell_21/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2
lstm_21/lstm_cell_21/Relu_1Á
lstm_21/lstm_cell_21/mul_2Mul"lstm_21/lstm_cell_21/Sigmoid_2:y:0)lstm_21/lstm_cell_21/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2
lstm_21/lstm_cell_21/mul_2
%lstm_21/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿE  2'
%lstm_21/TensorArrayV2_1/element_shapeØ
lstm_21/TensorArrayV2_1TensorListReserve.lstm_21/TensorArrayV2_1/element_shape:output:0 lstm_21/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_21/TensorArrayV2_1^
lstm_21/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_21/time
 lstm_21/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2"
 lstm_21/while/maximum_iterationsz
lstm_21/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_21/while/loop_counter
lstm_21/whileWhile#lstm_21/while/loop_counter:output:0)lstm_21/while/maximum_iterations:output:0lstm_21/time:output:0 lstm_21/TensorArrayV2_1:handle:0lstm_21/zeros:output:0lstm_21/zeros_1:output:0 lstm_21/strided_slice_1:output:0?lstm_21/TensorArrayUnstack/TensorListFromTensor:output_handle:03lstm_21_lstm_cell_21_matmul_readvariableop_resource5lstm_21_lstm_cell_21_matmul_1_readvariableop_resource4lstm_21_lstm_cell_21_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿÅ:ÿÿÿÿÿÿÿÿÿÅ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *'
bodyR
lstm_21_while_body_39103897*'
condR
lstm_21_while_cond_39103896*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿÅ:ÿÿÿÿÿÿÿÿÿÅ: : : : : *
parallel_iterations 2
lstm_21/whileÅ
8lstm_21/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿE  2:
8lstm_21/TensorArrayV2Stack/TensorListStack/element_shape
*lstm_21/TensorArrayV2Stack/TensorListStackTensorListStacklstm_21/while:output:3Alstm_21/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ*
element_dtype02,
*lstm_21/TensorArrayV2Stack/TensorListStack
lstm_21/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
lstm_21/strided_slice_3/stack
lstm_21/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2!
lstm_21/strided_slice_3/stack_1
lstm_21/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_21/strided_slice_3/stack_2Ë
lstm_21/strided_slice_3StridedSlice3lstm_21/TensorArrayV2Stack/TensorListStack:tensor:0&lstm_21/strided_slice_3/stack:output:0(lstm_21/strided_slice_3/stack_1:output:0(lstm_21/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ*
shrink_axis_mask2
lstm_21/strided_slice_3
lstm_21/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_21/transpose_1/permÆ
lstm_21/transpose_1	Transpose3lstm_21/TensorArrayV2Stack/TensorListStack:tensor:0!lstm_21/transpose_1/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2
lstm_21/transpose_1v
lstm_21/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_21/runtimey
dropout_21/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout_21/dropout/Constª
dropout_21/dropout/MulMullstm_21/transpose_1:y:0!dropout_21/dropout/Const:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2
dropout_21/dropout/Mul{
dropout_21/dropout/ShapeShapelstm_21/transpose_1:y:0*
T0*
_output_shapes
:2
dropout_21/dropout/ShapeÚ
/dropout_21/dropout/random_uniform/RandomUniformRandomUniform!dropout_21/dropout/Shape:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ*
dtype021
/dropout_21/dropout/random_uniform/RandomUniform
!dropout_21/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>2#
!dropout_21/dropout/GreaterEqual/yï
dropout_21/dropout/GreaterEqualGreaterEqual8dropout_21/dropout/random_uniform/RandomUniform:output:0*dropout_21/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2!
dropout_21/dropout/GreaterEqual¥
dropout_21/dropout/CastCast#dropout_21/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2
dropout_21/dropout/Cast«
dropout_21/dropout/Mul_1Muldropout_21/dropout/Mul:z:0dropout_21/dropout/Cast:y:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2
dropout_21/dropout/Mul_1²
!dense_10/Tensordot/ReadVariableOpReadVariableOp*dense_10_tensordot_readvariableop_resource*
_output_shapes
:	Å*
dtype02#
!dense_10/Tensordot/ReadVariableOp|
dense_10/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_10/Tensordot/axes
dense_10/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_10/Tensordot/free
dense_10/Tensordot/ShapeShapedropout_21/dropout/Mul_1:z:0*
T0*
_output_shapes
:2
dense_10/Tensordot/Shape
 dense_10/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_10/Tensordot/GatherV2/axisþ
dense_10/Tensordot/GatherV2GatherV2!dense_10/Tensordot/Shape:output:0 dense_10/Tensordot/free:output:0)dense_10/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_10/Tensordot/GatherV2
"dense_10/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"dense_10/Tensordot/GatherV2_1/axis
dense_10/Tensordot/GatherV2_1GatherV2!dense_10/Tensordot/Shape:output:0 dense_10/Tensordot/axes:output:0+dense_10/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_10/Tensordot/GatherV2_1~
dense_10/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_10/Tensordot/Const¤
dense_10/Tensordot/ProdProd$dense_10/Tensordot/GatherV2:output:0!dense_10/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_10/Tensordot/Prod
dense_10/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_10/Tensordot/Const_1¬
dense_10/Tensordot/Prod_1Prod&dense_10/Tensordot/GatherV2_1:output:0#dense_10/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_10/Tensordot/Prod_1
dense_10/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2 
dense_10/Tensordot/concat/axisÝ
dense_10/Tensordot/concatConcatV2 dense_10/Tensordot/free:output:0 dense_10/Tensordot/axes:output:0'dense_10/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_10/Tensordot/concat°
dense_10/Tensordot/stackPack dense_10/Tensordot/Prod:output:0"dense_10/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_10/Tensordot/stackÂ
dense_10/Tensordot/transpose	Transposedropout_21/dropout/Mul_1:z:0"dense_10/Tensordot/concat:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2
dense_10/Tensordot/transposeÃ
dense_10/Tensordot/ReshapeReshape dense_10/Tensordot/transpose:y:0!dense_10/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
dense_10/Tensordot/ReshapeÂ
dense_10/Tensordot/MatMulMatMul#dense_10/Tensordot/Reshape:output:0)dense_10/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_10/Tensordot/MatMul
dense_10/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
dense_10/Tensordot/Const_2
 dense_10/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_10/Tensordot/concat_1/axisê
dense_10/Tensordot/concat_1ConcatV2$dense_10/Tensordot/GatherV2:output:0#dense_10/Tensordot/Const_2:output:0)dense_10/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_10/Tensordot/concat_1´
dense_10/TensordotReshape#dense_10/Tensordot/MatMul:product:0$dense_10/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_10/Tensordot§
dense_10/BiasAdd/ReadVariableOpReadVariableOp(dense_10_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_10/BiasAdd/ReadVariableOp«
dense_10/BiasAddBiasAdddense_10/Tensordot:output:0'dense_10/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_10/BiasAdd
dense_10/SoftmaxSoftmaxdense_10/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_10/Softmaxy
IdentityIdentitydense_10/Softmax:softmax:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

IdentityÈ
NoOpNoOp ^dense_10/BiasAdd/ReadVariableOp"^dense_10/Tensordot/ReadVariableOp,^lstm_20/lstm_cell_20/BiasAdd/ReadVariableOp+^lstm_20/lstm_cell_20/MatMul/ReadVariableOp-^lstm_20/lstm_cell_20/MatMul_1/ReadVariableOp^lstm_20/while,^lstm_21/lstm_cell_21/BiasAdd/ReadVariableOp+^lstm_21/lstm_cell_21/MatMul/ReadVariableOp-^lstm_21/lstm_cell_21/MatMul_1/ReadVariableOp^lstm_21/while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ]: : : : : : : : 2B
dense_10/BiasAdd/ReadVariableOpdense_10/BiasAdd/ReadVariableOp2F
!dense_10/Tensordot/ReadVariableOp!dense_10/Tensordot/ReadVariableOp2Z
+lstm_20/lstm_cell_20/BiasAdd/ReadVariableOp+lstm_20/lstm_cell_20/BiasAdd/ReadVariableOp2X
*lstm_20/lstm_cell_20/MatMul/ReadVariableOp*lstm_20/lstm_cell_20/MatMul/ReadVariableOp2\
,lstm_20/lstm_cell_20/MatMul_1/ReadVariableOp,lstm_20/lstm_cell_20/MatMul_1/ReadVariableOp2
lstm_20/whilelstm_20/while2Z
+lstm_21/lstm_cell_21/BiasAdd/ReadVariableOp+lstm_21/lstm_cell_21/BiasAdd/ReadVariableOp2X
*lstm_21/lstm_cell_21/MatMul/ReadVariableOp*lstm_21/lstm_cell_21/MatMul/ReadVariableOp2\
,lstm_21/lstm_cell_21/MatMul_1/ReadVariableOp,lstm_21/lstm_cell_21/MatMul_1/ReadVariableOp2
lstm_21/whilelstm_21/while:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
 
_user_specified_nameinputs
Ã\
 
E__inference_lstm_20_layer_call_and_return_conditional_losses_39104209
inputs_0>
+lstm_cell_20_matmul_readvariableop_resource:	]A
-lstm_cell_20_matmul_1_readvariableop_resource:
;
,lstm_cell_20_biasadd_readvariableop_resource:	
identity¢#lstm_cell_20/BiasAdd/ReadVariableOp¢"lstm_cell_20/MatMul/ReadVariableOp¢$lstm_cell_20/MatMul_1/ReadVariableOp¢whileF
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
B :2
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
B :2
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
:ÿÿÿÿÿÿÿÿÿ2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :2
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
B :2
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
:ÿÿÿÿÿÿÿÿÿ2	
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
"lstm_cell_20/MatMul/ReadVariableOpReadVariableOp+lstm_cell_20_matmul_readvariableop_resource*
_output_shapes
:	]*
dtype02$
"lstm_cell_20/MatMul/ReadVariableOp­
lstm_cell_20/MatMulMatMulstrided_slice_2:output:0*lstm_cell_20/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_20/MatMul¼
$lstm_cell_20/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_20_matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype02&
$lstm_cell_20/MatMul_1/ReadVariableOp©
lstm_cell_20/MatMul_1MatMulzeros:output:0,lstm_cell_20/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_20/MatMul_1 
lstm_cell_20/addAddV2lstm_cell_20/MatMul:product:0lstm_cell_20/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_20/add´
#lstm_cell_20/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_20_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02%
#lstm_cell_20/BiasAdd/ReadVariableOp­
lstm_cell_20/BiasAddBiasAddlstm_cell_20/add:z:0+lstm_cell_20/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_20/BiasAdd~
lstm_cell_20/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_20/split/split_dim÷
lstm_cell_20/splitSplit%lstm_cell_20/split/split_dim:output:0lstm_cell_20/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2
lstm_cell_20/split
lstm_cell_20/SigmoidSigmoidlstm_cell_20/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_20/Sigmoid
lstm_cell_20/Sigmoid_1Sigmoidlstm_cell_20/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_20/Sigmoid_1
lstm_cell_20/mulMullstm_cell_20/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_20/mul~
lstm_cell_20/ReluRelulstm_cell_20/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_20/Relu
lstm_cell_20/mul_1Mullstm_cell_20/Sigmoid:y:0lstm_cell_20/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_20/mul_1
lstm_cell_20/add_1AddV2lstm_cell_20/mul:z:0lstm_cell_20/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_20/add_1
lstm_cell_20/Sigmoid_2Sigmoidlstm_cell_20/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_20/Sigmoid_2}
lstm_cell_20/Relu_1Relulstm_cell_20/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_20/Relu_1¡
lstm_cell_20/mul_2Mullstm_cell_20/Sigmoid_2:y:0!lstm_cell_20/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_20/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ  2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_20_matmul_readvariableop_resource-lstm_cell_20_matmul_1_readvariableop_resource,lstm_cell_20_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_39104125*
condR
while_cond_39104124*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ  22
0TensorArrayV2Stack/TensorListStack/element_shapeò
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
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
:ÿÿÿÿÿÿÿÿÿ*
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
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
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
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

IdentityÈ
NoOpNoOp$^lstm_cell_20/BiasAdd/ReadVariableOp#^lstm_cell_20/MatMul/ReadVariableOp%^lstm_cell_20/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ]: : : 2J
#lstm_cell_20/BiasAdd/ReadVariableOp#lstm_cell_20/BiasAdd/ReadVariableOp2H
"lstm_cell_20/MatMul/ReadVariableOp"lstm_cell_20/MatMul/ReadVariableOp2L
$lstm_cell_20/MatMul_1/ReadVariableOp$lstm_cell_20/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ]
"
_user_specified_name
inputs/0
à
º
*__inference_lstm_20_layer_call_fn_39104684
inputs_0
unknown:	]
	unknown_0:

	unknown_1:	
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_lstm_20_layer_call_and_return_conditional_losses_391014932
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

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
½
ø
K__inference_sequential_10_layer_call_and_return_conditional_losses_39103294
lstm_20_input#
lstm_20_39103272:	]$
lstm_20_39103274:

lstm_20_39103276:	$
lstm_21_39103280:

$
lstm_21_39103282:
Å

lstm_21_39103284:	
$
dense_10_39103288:	Å
dense_10_39103290:
identity¢ dense_10/StatefulPartitionedCall¢lstm_20/StatefulPartitionedCall¢lstm_21/StatefulPartitionedCallµ
lstm_20/StatefulPartitionedCallStatefulPartitionedCalllstm_20_inputlstm_20_39103272lstm_20_39103274lstm_20_39103276*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_lstm_20_layer_call_and_return_conditional_losses_391025422!
lstm_20/StatefulPartitionedCall
dropout_20/PartitionedCallPartitionedCall(lstm_20/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_dropout_20_layer_call_and_return_conditional_losses_391025552
dropout_20/PartitionedCallË
lstm_21/StatefulPartitionedCallStatefulPartitionedCall#dropout_20/PartitionedCall:output:0lstm_21_39103280lstm_21_39103282lstm_21_39103284*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_lstm_21_layer_call_and_return_conditional_losses_391027072!
lstm_21/StatefulPartitionedCall
dropout_21/PartitionedCallPartitionedCall(lstm_21/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_dropout_21_layer_call_and_return_conditional_losses_391027202
dropout_21/PartitionedCall»
 dense_10/StatefulPartitionedCallStatefulPartitionedCall#dropout_21/PartitionedCall:output:0dense_10_39103288dense_10_39103290*
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
F__inference_dense_10_layer_call_and_return_conditional_losses_391027532"
 dense_10/StatefulPartitionedCall
IdentityIdentity)dense_10/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityµ
NoOpNoOp!^dense_10/StatefulPartitionedCall ^lstm_20/StatefulPartitionedCall ^lstm_21/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ]: : : : : : : : 2D
 dense_10/StatefulPartitionedCall dense_10/StatefulPartitionedCall2B
lstm_20/StatefulPartitionedCalllstm_20/StatefulPartitionedCall2B
lstm_21/StatefulPartitionedCalllstm_21/StatefulPartitionedCall:Z V
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
'
_user_specified_namelstm_20_input
ã
Í
while_cond_39101843
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_39101843___redundant_placeholder06
2while_while_cond_39101843___redundant_placeholder16
2while_while_cond_39101843___redundant_placeholder26
2while_while_cond_39101843___redundant_placeholder3
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
B: : : : :ÿÿÿÿÿÿÿÿÿÅ:ÿÿÿÿÿÿÿÿÿÅ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ:

_output_shapes
: :

_output_shapes
:
ù

)sequential_10_lstm_20_while_cond_39100864H
Dsequential_10_lstm_20_while_sequential_10_lstm_20_while_loop_counterN
Jsequential_10_lstm_20_while_sequential_10_lstm_20_while_maximum_iterations+
'sequential_10_lstm_20_while_placeholder-
)sequential_10_lstm_20_while_placeholder_1-
)sequential_10_lstm_20_while_placeholder_2-
)sequential_10_lstm_20_while_placeholder_3J
Fsequential_10_lstm_20_while_less_sequential_10_lstm_20_strided_slice_1b
^sequential_10_lstm_20_while_sequential_10_lstm_20_while_cond_39100864___redundant_placeholder0b
^sequential_10_lstm_20_while_sequential_10_lstm_20_while_cond_39100864___redundant_placeholder1b
^sequential_10_lstm_20_while_sequential_10_lstm_20_while_cond_39100864___redundant_placeholder2b
^sequential_10_lstm_20_while_sequential_10_lstm_20_while_cond_39100864___redundant_placeholder3(
$sequential_10_lstm_20_while_identity
Þ
 sequential_10/lstm_20/while/LessLess'sequential_10_lstm_20_while_placeholderFsequential_10_lstm_20_while_less_sequential_10_lstm_20_strided_slice_1*
T0*
_output_shapes
: 2"
 sequential_10/lstm_20/while/Less
$sequential_10/lstm_20/while/IdentityIdentity$sequential_10/lstm_20/while/Less:z:0*
T0
*
_output_shapes
: 2&
$sequential_10/lstm_20/while/Identity"U
$sequential_10_lstm_20_while_identity-sequential_10/lstm_20/while/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
¶
¸
*__inference_lstm_20_layer_call_fn_39104706

inputs
unknown:	]
	unknown_0:

	unknown_1:	
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_lstm_20_layer_call_and_return_conditional_losses_391031722
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

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
\

E__inference_lstm_21_layer_call_and_return_conditional_losses_39102707

inputs?
+lstm_cell_21_matmul_readvariableop_resource:

A
-lstm_cell_21_matmul_1_readvariableop_resource:
Å
;
,lstm_cell_21_biasadd_readvariableop_resource:	

identity¢#lstm_cell_21/BiasAdd/ReadVariableOp¢"lstm_cell_21/MatMul/ReadVariableOp¢$lstm_cell_21/MatMul_1/ReadVariableOp¢whileD
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
B :Å2
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
B :Å2
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
:ÿÿÿÿÿÿÿÿÿÅ2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :Å2
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
B :Å2
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
:ÿÿÿÿÿÿÿÿÿÅ2	
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
:ÿÿÿÿÿÿÿÿÿ2
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
valueB"ÿÿÿÿ  27
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
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
strided_slice_2¶
"lstm_cell_21/MatMul/ReadVariableOpReadVariableOp+lstm_cell_21_matmul_readvariableop_resource* 
_output_shapes
:

*
dtype02$
"lstm_cell_21/MatMul/ReadVariableOp­
lstm_cell_21/MatMulMatMulstrided_slice_2:output:0*lstm_cell_21/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
lstm_cell_21/MatMul¼
$lstm_cell_21/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_21_matmul_1_readvariableop_resource* 
_output_shapes
:
Å
*
dtype02&
$lstm_cell_21/MatMul_1/ReadVariableOp©
lstm_cell_21/MatMul_1MatMulzeros:output:0,lstm_cell_21/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
lstm_cell_21/MatMul_1 
lstm_cell_21/addAddV2lstm_cell_21/MatMul:product:0lstm_cell_21/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
lstm_cell_21/add´
#lstm_cell_21/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_21_biasadd_readvariableop_resource*
_output_shapes	
:
*
dtype02%
#lstm_cell_21/BiasAdd/ReadVariableOp­
lstm_cell_21/BiasAddBiasAddlstm_cell_21/add:z:0+lstm_cell_21/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
lstm_cell_21/BiasAdd~
lstm_cell_21/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_21/split/split_dim÷
lstm_cell_21/splitSplit%lstm_cell_21/split/split_dim:output:0lstm_cell_21/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿÅ:ÿÿÿÿÿÿÿÿÿÅ:ÿÿÿÿÿÿÿÿÿÅ:ÿÿÿÿÿÿÿÿÿÅ*
	num_split2
lstm_cell_21/split
lstm_cell_21/SigmoidSigmoidlstm_cell_21/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2
lstm_cell_21/Sigmoid
lstm_cell_21/Sigmoid_1Sigmoidlstm_cell_21/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2
lstm_cell_21/Sigmoid_1
lstm_cell_21/mulMullstm_cell_21/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2
lstm_cell_21/mul~
lstm_cell_21/ReluRelulstm_cell_21/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2
lstm_cell_21/Relu
lstm_cell_21/mul_1Mullstm_cell_21/Sigmoid:y:0lstm_cell_21/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2
lstm_cell_21/mul_1
lstm_cell_21/add_1AddV2lstm_cell_21/mul:z:0lstm_cell_21/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2
lstm_cell_21/add_1
lstm_cell_21/Sigmoid_2Sigmoidlstm_cell_21/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2
lstm_cell_21/Sigmoid_2}
lstm_cell_21/Relu_1Relulstm_cell_21/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2
lstm_cell_21/Relu_1¡
lstm_cell_21/mul_2Mullstm_cell_21/Sigmoid_2:y:0!lstm_cell_21/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2
lstm_cell_21/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿE  2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_21_matmul_readvariableop_resource-lstm_cell_21_matmul_1_readvariableop_resource,lstm_cell_21_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿÅ:ÿÿÿÿÿÿÿÿÿÅ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_39102623*
condR
while_cond_39102622*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿÅ:ÿÿÿÿÿÿÿÿÿÅ: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿE  22
0TensorArrayV2Stack/TensorListStack/element_shapeé
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ*
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
:ÿÿÿÿÿÿÿÿÿÅ*
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
:ÿÿÿÿÿÿÿÿÿÅ2
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
:ÿÿÿÿÿÿÿÿÿÅ2

IdentityÈ
NoOpNoOp$^lstm_cell_21/BiasAdd/ReadVariableOp#^lstm_cell_21/MatMul/ReadVariableOp%^lstm_cell_21/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:ÿÿÿÿÿÿÿÿÿ: : : 2J
#lstm_cell_21/BiasAdd/ReadVariableOp#lstm_cell_21/BiasAdd/ReadVariableOp2H
"lstm_cell_21/MatMul/ReadVariableOp"lstm_cell_21/MatMul/ReadVariableOp2L
$lstm_cell_21/MatMul_1/ReadVariableOp$lstm_cell_21/MatMul_1/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
è^

)sequential_10_lstm_21_while_body_39101013H
Dsequential_10_lstm_21_while_sequential_10_lstm_21_while_loop_counterN
Jsequential_10_lstm_21_while_sequential_10_lstm_21_while_maximum_iterations+
'sequential_10_lstm_21_while_placeholder-
)sequential_10_lstm_21_while_placeholder_1-
)sequential_10_lstm_21_while_placeholder_2-
)sequential_10_lstm_21_while_placeholder_3G
Csequential_10_lstm_21_while_sequential_10_lstm_21_strided_slice_1_0
sequential_10_lstm_21_while_tensorarrayv2read_tensorlistgetitem_sequential_10_lstm_21_tensorarrayunstack_tensorlistfromtensor_0]
Isequential_10_lstm_21_while_lstm_cell_21_matmul_readvariableop_resource_0:

_
Ksequential_10_lstm_21_while_lstm_cell_21_matmul_1_readvariableop_resource_0:
Å
Y
Jsequential_10_lstm_21_while_lstm_cell_21_biasadd_readvariableop_resource_0:	
(
$sequential_10_lstm_21_while_identity*
&sequential_10_lstm_21_while_identity_1*
&sequential_10_lstm_21_while_identity_2*
&sequential_10_lstm_21_while_identity_3*
&sequential_10_lstm_21_while_identity_4*
&sequential_10_lstm_21_while_identity_5E
Asequential_10_lstm_21_while_sequential_10_lstm_21_strided_slice_1
}sequential_10_lstm_21_while_tensorarrayv2read_tensorlistgetitem_sequential_10_lstm_21_tensorarrayunstack_tensorlistfromtensor[
Gsequential_10_lstm_21_while_lstm_cell_21_matmul_readvariableop_resource:

]
Isequential_10_lstm_21_while_lstm_cell_21_matmul_1_readvariableop_resource:
Å
W
Hsequential_10_lstm_21_while_lstm_cell_21_biasadd_readvariableop_resource:	
¢?sequential_10/lstm_21/while/lstm_cell_21/BiasAdd/ReadVariableOp¢>sequential_10/lstm_21/while/lstm_cell_21/MatMul/ReadVariableOp¢@sequential_10/lstm_21/while/lstm_cell_21/MatMul_1/ReadVariableOpï
Msequential_10/lstm_21/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ  2O
Msequential_10/lstm_21/while/TensorArrayV2Read/TensorListGetItem/element_shapeØ
?sequential_10/lstm_21/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemsequential_10_lstm_21_while_tensorarrayv2read_tensorlistgetitem_sequential_10_lstm_21_tensorarrayunstack_tensorlistfromtensor_0'sequential_10_lstm_21_while_placeholderVsequential_10/lstm_21/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype02A
?sequential_10/lstm_21/while/TensorArrayV2Read/TensorListGetItem
>sequential_10/lstm_21/while/lstm_cell_21/MatMul/ReadVariableOpReadVariableOpIsequential_10_lstm_21_while_lstm_cell_21_matmul_readvariableop_resource_0* 
_output_shapes
:

*
dtype02@
>sequential_10/lstm_21/while/lstm_cell_21/MatMul/ReadVariableOp¯
/sequential_10/lstm_21/while/lstm_cell_21/MatMulMatMulFsequential_10/lstm_21/while/TensorArrayV2Read/TensorListGetItem:item:0Fsequential_10/lstm_21/while/lstm_cell_21/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
21
/sequential_10/lstm_21/while/lstm_cell_21/MatMul
@sequential_10/lstm_21/while/lstm_cell_21/MatMul_1/ReadVariableOpReadVariableOpKsequential_10_lstm_21_while_lstm_cell_21_matmul_1_readvariableop_resource_0* 
_output_shapes
:
Å
*
dtype02B
@sequential_10/lstm_21/while/lstm_cell_21/MatMul_1/ReadVariableOp
1sequential_10/lstm_21/while/lstm_cell_21/MatMul_1MatMul)sequential_10_lstm_21_while_placeholder_2Hsequential_10/lstm_21/while/lstm_cell_21/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
23
1sequential_10/lstm_21/while/lstm_cell_21/MatMul_1
,sequential_10/lstm_21/while/lstm_cell_21/addAddV29sequential_10/lstm_21/while/lstm_cell_21/MatMul:product:0;sequential_10/lstm_21/while/lstm_cell_21/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2.
,sequential_10/lstm_21/while/lstm_cell_21/add
?sequential_10/lstm_21/while/lstm_cell_21/BiasAdd/ReadVariableOpReadVariableOpJsequential_10_lstm_21_while_lstm_cell_21_biasadd_readvariableop_resource_0*
_output_shapes	
:
*
dtype02A
?sequential_10/lstm_21/while/lstm_cell_21/BiasAdd/ReadVariableOp
0sequential_10/lstm_21/while/lstm_cell_21/BiasAddBiasAdd0sequential_10/lstm_21/while/lstm_cell_21/add:z:0Gsequential_10/lstm_21/while/lstm_cell_21/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
22
0sequential_10/lstm_21/while/lstm_cell_21/BiasAdd¶
8sequential_10/lstm_21/while/lstm_cell_21/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2:
8sequential_10/lstm_21/while/lstm_cell_21/split/split_dimç
.sequential_10/lstm_21/while/lstm_cell_21/splitSplitAsequential_10/lstm_21/while/lstm_cell_21/split/split_dim:output:09sequential_10/lstm_21/while/lstm_cell_21/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿÅ:ÿÿÿÿÿÿÿÿÿÅ:ÿÿÿÿÿÿÿÿÿÅ:ÿÿÿÿÿÿÿÿÿÅ*
	num_split20
.sequential_10/lstm_21/while/lstm_cell_21/splitÛ
0sequential_10/lstm_21/while/lstm_cell_21/SigmoidSigmoid7sequential_10/lstm_21/while/lstm_cell_21/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ22
0sequential_10/lstm_21/while/lstm_cell_21/Sigmoidß
2sequential_10/lstm_21/while/lstm_cell_21/Sigmoid_1Sigmoid7sequential_10/lstm_21/while/lstm_cell_21/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ24
2sequential_10/lstm_21/while/lstm_cell_21/Sigmoid_1ù
,sequential_10/lstm_21/while/lstm_cell_21/mulMul6sequential_10/lstm_21/while/lstm_cell_21/Sigmoid_1:y:0)sequential_10_lstm_21_while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2.
,sequential_10/lstm_21/while/lstm_cell_21/mulÒ
-sequential_10/lstm_21/while/lstm_cell_21/ReluRelu7sequential_10/lstm_21/while/lstm_cell_21/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2/
-sequential_10/lstm_21/while/lstm_cell_21/Relu
.sequential_10/lstm_21/while/lstm_cell_21/mul_1Mul4sequential_10/lstm_21/while/lstm_cell_21/Sigmoid:y:0;sequential_10/lstm_21/while/lstm_cell_21/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ20
.sequential_10/lstm_21/while/lstm_cell_21/mul_1
.sequential_10/lstm_21/while/lstm_cell_21/add_1AddV20sequential_10/lstm_21/while/lstm_cell_21/mul:z:02sequential_10/lstm_21/while/lstm_cell_21/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ20
.sequential_10/lstm_21/while/lstm_cell_21/add_1ß
2sequential_10/lstm_21/while/lstm_cell_21/Sigmoid_2Sigmoid7sequential_10/lstm_21/while/lstm_cell_21/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ24
2sequential_10/lstm_21/while/lstm_cell_21/Sigmoid_2Ñ
/sequential_10/lstm_21/while/lstm_cell_21/Relu_1Relu2sequential_10/lstm_21/while/lstm_cell_21/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ21
/sequential_10/lstm_21/while/lstm_cell_21/Relu_1
.sequential_10/lstm_21/while/lstm_cell_21/mul_2Mul6sequential_10/lstm_21/while/lstm_cell_21/Sigmoid_2:y:0=sequential_10/lstm_21/while/lstm_cell_21/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ20
.sequential_10/lstm_21/while/lstm_cell_21/mul_2Î
@sequential_10/lstm_21/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem)sequential_10_lstm_21_while_placeholder_1'sequential_10_lstm_21_while_placeholder2sequential_10/lstm_21/while/lstm_cell_21/mul_2:z:0*
_output_shapes
: *
element_dtype02B
@sequential_10/lstm_21/while/TensorArrayV2Write/TensorListSetItem
!sequential_10/lstm_21/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2#
!sequential_10/lstm_21/while/add/yÁ
sequential_10/lstm_21/while/addAddV2'sequential_10_lstm_21_while_placeholder*sequential_10/lstm_21/while/add/y:output:0*
T0*
_output_shapes
: 2!
sequential_10/lstm_21/while/add
#sequential_10/lstm_21/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2%
#sequential_10/lstm_21/while/add_1/yä
!sequential_10/lstm_21/while/add_1AddV2Dsequential_10_lstm_21_while_sequential_10_lstm_21_while_loop_counter,sequential_10/lstm_21/while/add_1/y:output:0*
T0*
_output_shapes
: 2#
!sequential_10/lstm_21/while/add_1Ã
$sequential_10/lstm_21/while/IdentityIdentity%sequential_10/lstm_21/while/add_1:z:0!^sequential_10/lstm_21/while/NoOp*
T0*
_output_shapes
: 2&
$sequential_10/lstm_21/while/Identityì
&sequential_10/lstm_21/while/Identity_1IdentityJsequential_10_lstm_21_while_sequential_10_lstm_21_while_maximum_iterations!^sequential_10/lstm_21/while/NoOp*
T0*
_output_shapes
: 2(
&sequential_10/lstm_21/while/Identity_1Å
&sequential_10/lstm_21/while/Identity_2Identity#sequential_10/lstm_21/while/add:z:0!^sequential_10/lstm_21/while/NoOp*
T0*
_output_shapes
: 2(
&sequential_10/lstm_21/while/Identity_2ò
&sequential_10/lstm_21/while/Identity_3IdentityPsequential_10/lstm_21/while/TensorArrayV2Write/TensorListSetItem:output_handle:0!^sequential_10/lstm_21/while/NoOp*
T0*
_output_shapes
: 2(
&sequential_10/lstm_21/while/Identity_3æ
&sequential_10/lstm_21/while/Identity_4Identity2sequential_10/lstm_21/while/lstm_cell_21/mul_2:z:0!^sequential_10/lstm_21/while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2(
&sequential_10/lstm_21/while/Identity_4æ
&sequential_10/lstm_21/while/Identity_5Identity2sequential_10/lstm_21/while/lstm_cell_21/add_1:z:0!^sequential_10/lstm_21/while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2(
&sequential_10/lstm_21/while/Identity_5Ì
 sequential_10/lstm_21/while/NoOpNoOp@^sequential_10/lstm_21/while/lstm_cell_21/BiasAdd/ReadVariableOp?^sequential_10/lstm_21/while/lstm_cell_21/MatMul/ReadVariableOpA^sequential_10/lstm_21/while/lstm_cell_21/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2"
 sequential_10/lstm_21/while/NoOp"U
$sequential_10_lstm_21_while_identity-sequential_10/lstm_21/while/Identity:output:0"Y
&sequential_10_lstm_21_while_identity_1/sequential_10/lstm_21/while/Identity_1:output:0"Y
&sequential_10_lstm_21_while_identity_2/sequential_10/lstm_21/while/Identity_2:output:0"Y
&sequential_10_lstm_21_while_identity_3/sequential_10/lstm_21/while/Identity_3:output:0"Y
&sequential_10_lstm_21_while_identity_4/sequential_10/lstm_21/while/Identity_4:output:0"Y
&sequential_10_lstm_21_while_identity_5/sequential_10/lstm_21/while/Identity_5:output:0"
Hsequential_10_lstm_21_while_lstm_cell_21_biasadd_readvariableop_resourceJsequential_10_lstm_21_while_lstm_cell_21_biasadd_readvariableop_resource_0"
Isequential_10_lstm_21_while_lstm_cell_21_matmul_1_readvariableop_resourceKsequential_10_lstm_21_while_lstm_cell_21_matmul_1_readvariableop_resource_0"
Gsequential_10_lstm_21_while_lstm_cell_21_matmul_readvariableop_resourceIsequential_10_lstm_21_while_lstm_cell_21_matmul_readvariableop_resource_0"
Asequential_10_lstm_21_while_sequential_10_lstm_21_strided_slice_1Csequential_10_lstm_21_while_sequential_10_lstm_21_strided_slice_1_0"
}sequential_10_lstm_21_while_tensorarrayv2read_tensorlistgetitem_sequential_10_lstm_21_tensorarrayunstack_tensorlistfromtensorsequential_10_lstm_21_while_tensorarrayv2read_tensorlistgetitem_sequential_10_lstm_21_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿÅ:ÿÿÿÿÿÿÿÿÿÅ: : : : : 2
?sequential_10/lstm_21/while/lstm_cell_21/BiasAdd/ReadVariableOp?sequential_10/lstm_21/while/lstm_cell_21/BiasAdd/ReadVariableOp2
>sequential_10/lstm_21/while/lstm_cell_21/MatMul/ReadVariableOp>sequential_10/lstm_21/while/lstm_cell_21/MatMul/ReadVariableOp2
@sequential_10/lstm_21/while/lstm_cell_21/MatMul_1/ReadVariableOp@sequential_10/lstm_21/while/lstm_cell_21/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ:

_output_shapes
: :

_output_shapes
: 


J__inference_lstm_cell_20_layer_call_and_return_conditional_losses_39105512

inputs
states_0
states_11
matmul_readvariableop_resource:	]4
 matmul_1_readvariableop_resource:
.
biasadd_readvariableop_resource:	
identity

identity_1

identity_2¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	]*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul_1/ReadVariableOp|
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
add
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
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
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2
split`
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
Sigmoidd
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	Sigmoid_1]
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mulW
ReluRelusplit:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relui
mul_1MulSigmoid:y:0Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mul_1^
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
add_1d
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	Sigmoid_2V
Relu_1Relu	add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu_1m
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mul_2e
IdentityIdentity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityi

Identity_1Identity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1i

Identity_2Identity	add_1:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

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
A:ÿÿÿÿÿÿÿÿÿ]:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
 
_user_specified_nameinputs:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
states/0:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
states/1
Ø
I
-__inference_dropout_21_layer_call_fn_39105403

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
:ÿÿÿÿÿÿÿÿÿÅ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_dropout_21_layer_call_and_return_conditional_losses_391027202
PartitionedCallq
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿÅ:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ
 
_user_specified_nameinputs
ÐF

E__inference_lstm_21_layer_call_and_return_conditional_losses_39102123

inputs)
lstm_cell_21_39102041:

)
lstm_cell_21_39102043:
Å
$
lstm_cell_21_39102045:	

identity¢$lstm_cell_21/StatefulPartitionedCall¢whileD
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
B :Å2
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
B :Å2
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
:ÿÿÿÿÿÿÿÿÿÅ2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :Å2
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
B :Å2
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
:ÿÿÿÿÿÿÿÿÿÅ2	
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
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
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
valueB"ÿÿÿÿ  27
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
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
strided_slice_2¨
$lstm_cell_21/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_21_39102041lstm_cell_21_39102043lstm_cell_21_39102045*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿÅ:ÿÿÿÿÿÿÿÿÿÅ:ÿÿÿÿÿÿÿÿÿÅ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_lstm_cell_21_layer_call_and_return_conditional_losses_391019762&
$lstm_cell_21/StatefulPartitionedCall
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿE  2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_21_39102041lstm_cell_21_39102043lstm_cell_21_39102045*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿÅ:ÿÿÿÿÿÿÿÿÿÅ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_39102054*
condR
while_cond_39102053*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿÅ:ÿÿÿÿÿÿÿÿÿÅ: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿE  22
0TensorArrayV2Stack/TensorListStack/element_shapeò
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÅ*
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
:ÿÿÿÿÿÿÿÿÿÅ*
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
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÅ2
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
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÅ2

Identity}
NoOpNoOp%^lstm_cell_21/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 2L
$lstm_cell_21/StatefulPartitionedCall$lstm_cell_21/StatefulPartitionedCall2
whilewhile:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

f
H__inference_dropout_20_layer_call_and_return_conditional_losses_39102555

inputs

identity_1_
IdentityIdentityinputs*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityn

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
\

E__inference_lstm_21_layer_call_and_return_conditional_losses_39105186

inputs?
+lstm_cell_21_matmul_readvariableop_resource:

A
-lstm_cell_21_matmul_1_readvariableop_resource:
Å
;
,lstm_cell_21_biasadd_readvariableop_resource:	

identity¢#lstm_cell_21/BiasAdd/ReadVariableOp¢"lstm_cell_21/MatMul/ReadVariableOp¢$lstm_cell_21/MatMul_1/ReadVariableOp¢whileD
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
B :Å2
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
B :Å2
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
:ÿÿÿÿÿÿÿÿÿÅ2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :Å2
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
B :Å2
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
:ÿÿÿÿÿÿÿÿÿÅ2	
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
:ÿÿÿÿÿÿÿÿÿ2
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
valueB"ÿÿÿÿ  27
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
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
strided_slice_2¶
"lstm_cell_21/MatMul/ReadVariableOpReadVariableOp+lstm_cell_21_matmul_readvariableop_resource* 
_output_shapes
:

*
dtype02$
"lstm_cell_21/MatMul/ReadVariableOp­
lstm_cell_21/MatMulMatMulstrided_slice_2:output:0*lstm_cell_21/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
lstm_cell_21/MatMul¼
$lstm_cell_21/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_21_matmul_1_readvariableop_resource* 
_output_shapes
:
Å
*
dtype02&
$lstm_cell_21/MatMul_1/ReadVariableOp©
lstm_cell_21/MatMul_1MatMulzeros:output:0,lstm_cell_21/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
lstm_cell_21/MatMul_1 
lstm_cell_21/addAddV2lstm_cell_21/MatMul:product:0lstm_cell_21/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
lstm_cell_21/add´
#lstm_cell_21/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_21_biasadd_readvariableop_resource*
_output_shapes	
:
*
dtype02%
#lstm_cell_21/BiasAdd/ReadVariableOp­
lstm_cell_21/BiasAddBiasAddlstm_cell_21/add:z:0+lstm_cell_21/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
lstm_cell_21/BiasAdd~
lstm_cell_21/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_21/split/split_dim÷
lstm_cell_21/splitSplit%lstm_cell_21/split/split_dim:output:0lstm_cell_21/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿÅ:ÿÿÿÿÿÿÿÿÿÅ:ÿÿÿÿÿÿÿÿÿÅ:ÿÿÿÿÿÿÿÿÿÅ*
	num_split2
lstm_cell_21/split
lstm_cell_21/SigmoidSigmoidlstm_cell_21/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2
lstm_cell_21/Sigmoid
lstm_cell_21/Sigmoid_1Sigmoidlstm_cell_21/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2
lstm_cell_21/Sigmoid_1
lstm_cell_21/mulMullstm_cell_21/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2
lstm_cell_21/mul~
lstm_cell_21/ReluRelulstm_cell_21/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2
lstm_cell_21/Relu
lstm_cell_21/mul_1Mullstm_cell_21/Sigmoid:y:0lstm_cell_21/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2
lstm_cell_21/mul_1
lstm_cell_21/add_1AddV2lstm_cell_21/mul:z:0lstm_cell_21/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2
lstm_cell_21/add_1
lstm_cell_21/Sigmoid_2Sigmoidlstm_cell_21/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2
lstm_cell_21/Sigmoid_2}
lstm_cell_21/Relu_1Relulstm_cell_21/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2
lstm_cell_21/Relu_1¡
lstm_cell_21/mul_2Mullstm_cell_21/Sigmoid_2:y:0!lstm_cell_21/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2
lstm_cell_21/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿE  2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_21_matmul_readvariableop_resource-lstm_cell_21_matmul_1_readvariableop_resource,lstm_cell_21_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿÅ:ÿÿÿÿÿÿÿÿÿÅ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_39105102*
condR
while_cond_39105101*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿÅ:ÿÿÿÿÿÿÿÿÿÅ: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿE  22
0TensorArrayV2Stack/TensorListStack/element_shapeé
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ*
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
:ÿÿÿÿÿÿÿÿÿÅ*
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
:ÿÿÿÿÿÿÿÿÿÅ2
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
:ÿÿÿÿÿÿÿÿÿÅ2

IdentityÈ
NoOpNoOp$^lstm_cell_21/BiasAdd/ReadVariableOp#^lstm_cell_21/MatMul/ReadVariableOp%^lstm_cell_21/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:ÿÿÿÿÿÿÿÿÿ: : : 2J
#lstm_cell_21/BiasAdd/ReadVariableOp#lstm_cell_21/BiasAdd/ReadVariableOp2H
"lstm_cell_21/MatMul/ReadVariableOp"lstm_cell_21/MatMul/ReadVariableOp2L
$lstm_cell_21/MatMul_1/ReadVariableOp$lstm_cell_21/MatMul_1/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
×
g
H__inference_dropout_20_layer_call_and_return_conditional_losses_39104723

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
:ÿÿÿÿÿÿÿÿÿ2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape¹
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
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
:ÿÿÿÿÿÿÿÿÿ2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/Mul_1j
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ã\
 
E__inference_lstm_20_layer_call_and_return_conditional_losses_39104360
inputs_0>
+lstm_cell_20_matmul_readvariableop_resource:	]A
-lstm_cell_20_matmul_1_readvariableop_resource:
;
,lstm_cell_20_biasadd_readvariableop_resource:	
identity¢#lstm_cell_20/BiasAdd/ReadVariableOp¢"lstm_cell_20/MatMul/ReadVariableOp¢$lstm_cell_20/MatMul_1/ReadVariableOp¢whileF
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
B :2
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
B :2
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
:ÿÿÿÿÿÿÿÿÿ2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :2
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
B :2
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
:ÿÿÿÿÿÿÿÿÿ2	
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
"lstm_cell_20/MatMul/ReadVariableOpReadVariableOp+lstm_cell_20_matmul_readvariableop_resource*
_output_shapes
:	]*
dtype02$
"lstm_cell_20/MatMul/ReadVariableOp­
lstm_cell_20/MatMulMatMulstrided_slice_2:output:0*lstm_cell_20/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_20/MatMul¼
$lstm_cell_20/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_20_matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype02&
$lstm_cell_20/MatMul_1/ReadVariableOp©
lstm_cell_20/MatMul_1MatMulzeros:output:0,lstm_cell_20/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_20/MatMul_1 
lstm_cell_20/addAddV2lstm_cell_20/MatMul:product:0lstm_cell_20/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_20/add´
#lstm_cell_20/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_20_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02%
#lstm_cell_20/BiasAdd/ReadVariableOp­
lstm_cell_20/BiasAddBiasAddlstm_cell_20/add:z:0+lstm_cell_20/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_20/BiasAdd~
lstm_cell_20/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_20/split/split_dim÷
lstm_cell_20/splitSplit%lstm_cell_20/split/split_dim:output:0lstm_cell_20/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2
lstm_cell_20/split
lstm_cell_20/SigmoidSigmoidlstm_cell_20/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_20/Sigmoid
lstm_cell_20/Sigmoid_1Sigmoidlstm_cell_20/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_20/Sigmoid_1
lstm_cell_20/mulMullstm_cell_20/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_20/mul~
lstm_cell_20/ReluRelulstm_cell_20/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_20/Relu
lstm_cell_20/mul_1Mullstm_cell_20/Sigmoid:y:0lstm_cell_20/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_20/mul_1
lstm_cell_20/add_1AddV2lstm_cell_20/mul:z:0lstm_cell_20/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_20/add_1
lstm_cell_20/Sigmoid_2Sigmoidlstm_cell_20/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_20/Sigmoid_2}
lstm_cell_20/Relu_1Relulstm_cell_20/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_20/Relu_1¡
lstm_cell_20/mul_2Mullstm_cell_20/Sigmoid_2:y:0!lstm_cell_20/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_20/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ  2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_20_matmul_readvariableop_resource-lstm_cell_20_matmul_1_readvariableop_resource,lstm_cell_20_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_39104276*
condR
while_cond_39104275*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ  22
0TensorArrayV2Stack/TensorListStack/element_shapeò
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
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
:ÿÿÿÿÿÿÿÿÿ*
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
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
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
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

IdentityÈ
NoOpNoOp$^lstm_cell_20/BiasAdd/ReadVariableOp#^lstm_cell_20/MatMul/ReadVariableOp%^lstm_cell_20/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ]: : : 2J
#lstm_cell_20/BiasAdd/ReadVariableOp#lstm_cell_20/BiasAdd/ReadVariableOp2H
"lstm_cell_20/MatMul/ReadVariableOp"lstm_cell_20/MatMul/ReadVariableOp2L
$lstm_cell_20/MatMul_1/ReadVariableOp$lstm_cell_20/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ]
"
_user_specified_name
inputs/0
\

E__inference_lstm_20_layer_call_and_return_conditional_losses_39104511

inputs>
+lstm_cell_20_matmul_readvariableop_resource:	]A
-lstm_cell_20_matmul_1_readvariableop_resource:
;
,lstm_cell_20_biasadd_readvariableop_resource:	
identity¢#lstm_cell_20/BiasAdd/ReadVariableOp¢"lstm_cell_20/MatMul/ReadVariableOp¢$lstm_cell_20/MatMul_1/ReadVariableOp¢whileD
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
B :2
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
B :2
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
:ÿÿÿÿÿÿÿÿÿ2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :2
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
B :2
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
:ÿÿÿÿÿÿÿÿÿ2	
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
"lstm_cell_20/MatMul/ReadVariableOpReadVariableOp+lstm_cell_20_matmul_readvariableop_resource*
_output_shapes
:	]*
dtype02$
"lstm_cell_20/MatMul/ReadVariableOp­
lstm_cell_20/MatMulMatMulstrided_slice_2:output:0*lstm_cell_20/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_20/MatMul¼
$lstm_cell_20/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_20_matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype02&
$lstm_cell_20/MatMul_1/ReadVariableOp©
lstm_cell_20/MatMul_1MatMulzeros:output:0,lstm_cell_20/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_20/MatMul_1 
lstm_cell_20/addAddV2lstm_cell_20/MatMul:product:0lstm_cell_20/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_20/add´
#lstm_cell_20/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_20_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02%
#lstm_cell_20/BiasAdd/ReadVariableOp­
lstm_cell_20/BiasAddBiasAddlstm_cell_20/add:z:0+lstm_cell_20/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_20/BiasAdd~
lstm_cell_20/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_20/split/split_dim÷
lstm_cell_20/splitSplit%lstm_cell_20/split/split_dim:output:0lstm_cell_20/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2
lstm_cell_20/split
lstm_cell_20/SigmoidSigmoidlstm_cell_20/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_20/Sigmoid
lstm_cell_20/Sigmoid_1Sigmoidlstm_cell_20/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_20/Sigmoid_1
lstm_cell_20/mulMullstm_cell_20/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_20/mul~
lstm_cell_20/ReluRelulstm_cell_20/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_20/Relu
lstm_cell_20/mul_1Mullstm_cell_20/Sigmoid:y:0lstm_cell_20/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_20/mul_1
lstm_cell_20/add_1AddV2lstm_cell_20/mul:z:0lstm_cell_20/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_20/add_1
lstm_cell_20/Sigmoid_2Sigmoidlstm_cell_20/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_20/Sigmoid_2}
lstm_cell_20/Relu_1Relulstm_cell_20/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_20/Relu_1¡
lstm_cell_20/mul_2Mullstm_cell_20/Sigmoid_2:y:0!lstm_cell_20/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_20/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ  2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_20_matmul_readvariableop_resource-lstm_cell_20_matmul_1_readvariableop_resource,lstm_cell_20_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_39104427*
condR
while_cond_39104426*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ  22
0TensorArrayV2Stack/TensorListStack/element_shapeé
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
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
:ÿÿÿÿÿÿÿÿÿ*
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
:ÿÿÿÿÿÿÿÿÿ2
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
:ÿÿÿÿÿÿÿÿÿ2

IdentityÈ
NoOpNoOp$^lstm_cell_20/BiasAdd/ReadVariableOp#^lstm_cell_20/MatMul/ReadVariableOp%^lstm_cell_20/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ]: : : 2J
#lstm_cell_20/BiasAdd/ReadVariableOp#lstm_cell_20/BiasAdd/ReadVariableOp2H
"lstm_cell_20/MatMul/ReadVariableOp"lstm_cell_20/MatMul/ReadVariableOp2L
$lstm_cell_20/MatMul_1/ReadVariableOp$lstm_cell_20/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
 
_user_specified_nameinputs
´?
Ö
while_body_39105102
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
3while_lstm_cell_21_matmul_readvariableop_resource_0:

I
5while_lstm_cell_21_matmul_1_readvariableop_resource_0:
Å
C
4while_lstm_cell_21_biasadd_readvariableop_resource_0:	

while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
1while_lstm_cell_21_matmul_readvariableop_resource:

G
3while_lstm_cell_21_matmul_1_readvariableop_resource:
Å
A
2while_lstm_cell_21_biasadd_readvariableop_resource:	
¢)while/lstm_cell_21/BiasAdd/ReadVariableOp¢(while/lstm_cell_21/MatMul/ReadVariableOp¢*while/lstm_cell_21/MatMul_1/ReadVariableOpÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ  29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÔ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemÊ
(while/lstm_cell_21/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_21_matmul_readvariableop_resource_0* 
_output_shapes
:

*
dtype02*
(while/lstm_cell_21/MatMul/ReadVariableOp×
while/lstm_cell_21/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_21/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
while/lstm_cell_21/MatMulÐ
*while/lstm_cell_21/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_21_matmul_1_readvariableop_resource_0* 
_output_shapes
:
Å
*
dtype02,
*while/lstm_cell_21/MatMul_1/ReadVariableOpÀ
while/lstm_cell_21/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_21/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
while/lstm_cell_21/MatMul_1¸
while/lstm_cell_21/addAddV2#while/lstm_cell_21/MatMul:product:0%while/lstm_cell_21/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
while/lstm_cell_21/addÈ
)while/lstm_cell_21/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_21_biasadd_readvariableop_resource_0*
_output_shapes	
:
*
dtype02+
)while/lstm_cell_21/BiasAdd/ReadVariableOpÅ
while/lstm_cell_21/BiasAddBiasAddwhile/lstm_cell_21/add:z:01while/lstm_cell_21/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
while/lstm_cell_21/BiasAdd
"while/lstm_cell_21/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_21/split/split_dim
while/lstm_cell_21/splitSplit+while/lstm_cell_21/split/split_dim:output:0#while/lstm_cell_21/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿÅ:ÿÿÿÿÿÿÿÿÿÅ:ÿÿÿÿÿÿÿÿÿÅ:ÿÿÿÿÿÿÿÿÿÅ*
	num_split2
while/lstm_cell_21/split
while/lstm_cell_21/SigmoidSigmoid!while/lstm_cell_21/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2
while/lstm_cell_21/Sigmoid
while/lstm_cell_21/Sigmoid_1Sigmoid!while/lstm_cell_21/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2
while/lstm_cell_21/Sigmoid_1¡
while/lstm_cell_21/mulMul while/lstm_cell_21/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2
while/lstm_cell_21/mul
while/lstm_cell_21/ReluRelu!while/lstm_cell_21/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2
while/lstm_cell_21/Reluµ
while/lstm_cell_21/mul_1Mulwhile/lstm_cell_21/Sigmoid:y:0%while/lstm_cell_21/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2
while/lstm_cell_21/mul_1ª
while/lstm_cell_21/add_1AddV2while/lstm_cell_21/mul:z:0while/lstm_cell_21/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2
while/lstm_cell_21/add_1
while/lstm_cell_21/Sigmoid_2Sigmoid!while/lstm_cell_21/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2
while/lstm_cell_21/Sigmoid_2
while/lstm_cell_21/Relu_1Reluwhile/lstm_cell_21/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2
while/lstm_cell_21/Relu_1¹
while/lstm_cell_21/mul_2Mul while/lstm_cell_21/Sigmoid_2:y:0'while/lstm_cell_21/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2
while/lstm_cell_21/mul_2à
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_21/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_21/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_21/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2
while/Identity_5Þ

while/NoOpNoOp*^while/lstm_cell_21/BiasAdd/ReadVariableOp)^while/lstm_cell_21/MatMul/ReadVariableOp+^while/lstm_cell_21/MatMul_1/ReadVariableOp*"
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
2while_lstm_cell_21_biasadd_readvariableop_resource4while_lstm_cell_21_biasadd_readvariableop_resource_0"l
3while_lstm_cell_21_matmul_1_readvariableop_resource5while_lstm_cell_21_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_21_matmul_readvariableop_resource3while_lstm_cell_21_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿÅ:ÿÿÿÿÿÿÿÿÿÅ: : : : : 2V
)while/lstm_cell_21/BiasAdd/ReadVariableOp)while/lstm_cell_21/BiasAdd/ReadVariableOp2T
(while/lstm_cell_21/MatMul/ReadVariableOp(while/lstm_cell_21/MatMul/ReadVariableOp2X
*while/lstm_cell_21/MatMul_1/ReadVariableOp*while/lstm_cell_21/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ:

_output_shapes
: :

_output_shapes
: 
éJ
Ö

lstm_21_while_body_39103563,
(lstm_21_while_lstm_21_while_loop_counter2
.lstm_21_while_lstm_21_while_maximum_iterations
lstm_21_while_placeholder
lstm_21_while_placeholder_1
lstm_21_while_placeholder_2
lstm_21_while_placeholder_3+
'lstm_21_while_lstm_21_strided_slice_1_0g
clstm_21_while_tensorarrayv2read_tensorlistgetitem_lstm_21_tensorarrayunstack_tensorlistfromtensor_0O
;lstm_21_while_lstm_cell_21_matmul_readvariableop_resource_0:

Q
=lstm_21_while_lstm_cell_21_matmul_1_readvariableop_resource_0:
Å
K
<lstm_21_while_lstm_cell_21_biasadd_readvariableop_resource_0:	

lstm_21_while_identity
lstm_21_while_identity_1
lstm_21_while_identity_2
lstm_21_while_identity_3
lstm_21_while_identity_4
lstm_21_while_identity_5)
%lstm_21_while_lstm_21_strided_slice_1e
alstm_21_while_tensorarrayv2read_tensorlistgetitem_lstm_21_tensorarrayunstack_tensorlistfromtensorM
9lstm_21_while_lstm_cell_21_matmul_readvariableop_resource:

O
;lstm_21_while_lstm_cell_21_matmul_1_readvariableop_resource:
Å
I
:lstm_21_while_lstm_cell_21_biasadd_readvariableop_resource:	
¢1lstm_21/while/lstm_cell_21/BiasAdd/ReadVariableOp¢0lstm_21/while/lstm_cell_21/MatMul/ReadVariableOp¢2lstm_21/while/lstm_cell_21/MatMul_1/ReadVariableOpÓ
?lstm_21/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ  2A
?lstm_21/while/TensorArrayV2Read/TensorListGetItem/element_shape
1lstm_21/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemclstm_21_while_tensorarrayv2read_tensorlistgetitem_lstm_21_tensorarrayunstack_tensorlistfromtensor_0lstm_21_while_placeholderHlstm_21/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype023
1lstm_21/while/TensorArrayV2Read/TensorListGetItemâ
0lstm_21/while/lstm_cell_21/MatMul/ReadVariableOpReadVariableOp;lstm_21_while_lstm_cell_21_matmul_readvariableop_resource_0* 
_output_shapes
:

*
dtype022
0lstm_21/while/lstm_cell_21/MatMul/ReadVariableOp÷
!lstm_21/while/lstm_cell_21/MatMulMatMul8lstm_21/while/TensorArrayV2Read/TensorListGetItem:item:08lstm_21/while/lstm_cell_21/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2#
!lstm_21/while/lstm_cell_21/MatMulè
2lstm_21/while/lstm_cell_21/MatMul_1/ReadVariableOpReadVariableOp=lstm_21_while_lstm_cell_21_matmul_1_readvariableop_resource_0* 
_output_shapes
:
Å
*
dtype024
2lstm_21/while/lstm_cell_21/MatMul_1/ReadVariableOpà
#lstm_21/while/lstm_cell_21/MatMul_1MatMullstm_21_while_placeholder_2:lstm_21/while/lstm_cell_21/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2%
#lstm_21/while/lstm_cell_21/MatMul_1Ø
lstm_21/while/lstm_cell_21/addAddV2+lstm_21/while/lstm_cell_21/MatMul:product:0-lstm_21/while/lstm_cell_21/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2 
lstm_21/while/lstm_cell_21/addà
1lstm_21/while/lstm_cell_21/BiasAdd/ReadVariableOpReadVariableOp<lstm_21_while_lstm_cell_21_biasadd_readvariableop_resource_0*
_output_shapes	
:
*
dtype023
1lstm_21/while/lstm_cell_21/BiasAdd/ReadVariableOpå
"lstm_21/while/lstm_cell_21/BiasAddBiasAdd"lstm_21/while/lstm_cell_21/add:z:09lstm_21/while/lstm_cell_21/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2$
"lstm_21/while/lstm_cell_21/BiasAdd
*lstm_21/while/lstm_cell_21/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2,
*lstm_21/while/lstm_cell_21/split/split_dim¯
 lstm_21/while/lstm_cell_21/splitSplit3lstm_21/while/lstm_cell_21/split/split_dim:output:0+lstm_21/while/lstm_cell_21/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿÅ:ÿÿÿÿÿÿÿÿÿÅ:ÿÿÿÿÿÿÿÿÿÅ:ÿÿÿÿÿÿÿÿÿÅ*
	num_split2"
 lstm_21/while/lstm_cell_21/split±
"lstm_21/while/lstm_cell_21/SigmoidSigmoid)lstm_21/while/lstm_cell_21/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2$
"lstm_21/while/lstm_cell_21/Sigmoidµ
$lstm_21/while/lstm_cell_21/Sigmoid_1Sigmoid)lstm_21/while/lstm_cell_21/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2&
$lstm_21/while/lstm_cell_21/Sigmoid_1Á
lstm_21/while/lstm_cell_21/mulMul(lstm_21/while/lstm_cell_21/Sigmoid_1:y:0lstm_21_while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2 
lstm_21/while/lstm_cell_21/mul¨
lstm_21/while/lstm_cell_21/ReluRelu)lstm_21/while/lstm_cell_21/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2!
lstm_21/while/lstm_cell_21/ReluÕ
 lstm_21/while/lstm_cell_21/mul_1Mul&lstm_21/while/lstm_cell_21/Sigmoid:y:0-lstm_21/while/lstm_cell_21/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2"
 lstm_21/while/lstm_cell_21/mul_1Ê
 lstm_21/while/lstm_cell_21/add_1AddV2"lstm_21/while/lstm_cell_21/mul:z:0$lstm_21/while/lstm_cell_21/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2"
 lstm_21/while/lstm_cell_21/add_1µ
$lstm_21/while/lstm_cell_21/Sigmoid_2Sigmoid)lstm_21/while/lstm_cell_21/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2&
$lstm_21/while/lstm_cell_21/Sigmoid_2§
!lstm_21/while/lstm_cell_21/Relu_1Relu$lstm_21/while/lstm_cell_21/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2#
!lstm_21/while/lstm_cell_21/Relu_1Ù
 lstm_21/while/lstm_cell_21/mul_2Mul(lstm_21/while/lstm_cell_21/Sigmoid_2:y:0/lstm_21/while/lstm_cell_21/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2"
 lstm_21/while/lstm_cell_21/mul_2
2lstm_21/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_21_while_placeholder_1lstm_21_while_placeholder$lstm_21/while/lstm_cell_21/mul_2:z:0*
_output_shapes
: *
element_dtype024
2lstm_21/while/TensorArrayV2Write/TensorListSetIteml
lstm_21/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_21/while/add/y
lstm_21/while/addAddV2lstm_21_while_placeholderlstm_21/while/add/y:output:0*
T0*
_output_shapes
: 2
lstm_21/while/addp
lstm_21/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_21/while/add_1/y
lstm_21/while/add_1AddV2(lstm_21_while_lstm_21_while_loop_counterlstm_21/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lstm_21/while/add_1
lstm_21/while/IdentityIdentitylstm_21/while/add_1:z:0^lstm_21/while/NoOp*
T0*
_output_shapes
: 2
lstm_21/while/Identity¦
lstm_21/while/Identity_1Identity.lstm_21_while_lstm_21_while_maximum_iterations^lstm_21/while/NoOp*
T0*
_output_shapes
: 2
lstm_21/while/Identity_1
lstm_21/while/Identity_2Identitylstm_21/while/add:z:0^lstm_21/while/NoOp*
T0*
_output_shapes
: 2
lstm_21/while/Identity_2º
lstm_21/while/Identity_3IdentityBlstm_21/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_21/while/NoOp*
T0*
_output_shapes
: 2
lstm_21/while/Identity_3®
lstm_21/while/Identity_4Identity$lstm_21/while/lstm_cell_21/mul_2:z:0^lstm_21/while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2
lstm_21/while/Identity_4®
lstm_21/while/Identity_5Identity$lstm_21/while/lstm_cell_21/add_1:z:0^lstm_21/while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2
lstm_21/while/Identity_5
lstm_21/while/NoOpNoOp2^lstm_21/while/lstm_cell_21/BiasAdd/ReadVariableOp1^lstm_21/while/lstm_cell_21/MatMul/ReadVariableOp3^lstm_21/while/lstm_cell_21/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
lstm_21/while/NoOp"9
lstm_21_while_identitylstm_21/while/Identity:output:0"=
lstm_21_while_identity_1!lstm_21/while/Identity_1:output:0"=
lstm_21_while_identity_2!lstm_21/while/Identity_2:output:0"=
lstm_21_while_identity_3!lstm_21/while/Identity_3:output:0"=
lstm_21_while_identity_4!lstm_21/while/Identity_4:output:0"=
lstm_21_while_identity_5!lstm_21/while/Identity_5:output:0"P
%lstm_21_while_lstm_21_strided_slice_1'lstm_21_while_lstm_21_strided_slice_1_0"z
:lstm_21_while_lstm_cell_21_biasadd_readvariableop_resource<lstm_21_while_lstm_cell_21_biasadd_readvariableop_resource_0"|
;lstm_21_while_lstm_cell_21_matmul_1_readvariableop_resource=lstm_21_while_lstm_cell_21_matmul_1_readvariableop_resource_0"x
9lstm_21_while_lstm_cell_21_matmul_readvariableop_resource;lstm_21_while_lstm_cell_21_matmul_readvariableop_resource_0"È
alstm_21_while_tensorarrayv2read_tensorlistgetitem_lstm_21_tensorarrayunstack_tensorlistfromtensorclstm_21_while_tensorarrayv2read_tensorlistgetitem_lstm_21_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿÅ:ÿÿÿÿÿÿÿÿÿÅ: : : : : 2f
1lstm_21/while/lstm_cell_21/BiasAdd/ReadVariableOp1lstm_21/while/lstm_cell_21/BiasAdd/ReadVariableOp2d
0lstm_21/while/lstm_cell_21/MatMul/ReadVariableOp0lstm_21/while/lstm_cell_21/MatMul/ReadVariableOp2h
2lstm_21/while/lstm_cell_21/MatMul_1/ReadVariableOp2lstm_21/while/lstm_cell_21/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ:

_output_shapes
: :

_output_shapes
: 
°?
Ô
while_body_39102458
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_20_matmul_readvariableop_resource_0:	]I
5while_lstm_cell_20_matmul_1_readvariableop_resource_0:
C
4while_lstm_cell_20_biasadd_readvariableop_resource_0:	
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_20_matmul_readvariableop_resource:	]G
3while_lstm_cell_20_matmul_1_readvariableop_resource:
A
2while_lstm_cell_20_biasadd_readvariableop_resource:	¢)while/lstm_cell_20/BiasAdd/ReadVariableOp¢(while/lstm_cell_20/MatMul/ReadVariableOp¢*while/lstm_cell_20/MatMul_1/ReadVariableOpÃ
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
(while/lstm_cell_20/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_20_matmul_readvariableop_resource_0*
_output_shapes
:	]*
dtype02*
(while/lstm_cell_20/MatMul/ReadVariableOp×
while/lstm_cell_20/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_20/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_20/MatMulÐ
*while/lstm_cell_20/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_20_matmul_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype02,
*while/lstm_cell_20/MatMul_1/ReadVariableOpÀ
while/lstm_cell_20/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_20/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_20/MatMul_1¸
while/lstm_cell_20/addAddV2#while/lstm_cell_20/MatMul:product:0%while/lstm_cell_20/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_20/addÈ
)while/lstm_cell_20/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_20_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02+
)while/lstm_cell_20/BiasAdd/ReadVariableOpÅ
while/lstm_cell_20/BiasAddBiasAddwhile/lstm_cell_20/add:z:01while/lstm_cell_20/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_20/BiasAdd
"while/lstm_cell_20/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_20/split/split_dim
while/lstm_cell_20/splitSplit+while/lstm_cell_20/split/split_dim:output:0#while/lstm_cell_20/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2
while/lstm_cell_20/split
while/lstm_cell_20/SigmoidSigmoid!while/lstm_cell_20/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_20/Sigmoid
while/lstm_cell_20/Sigmoid_1Sigmoid!while/lstm_cell_20/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_20/Sigmoid_1¡
while/lstm_cell_20/mulMul while/lstm_cell_20/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_20/mul
while/lstm_cell_20/ReluRelu!while/lstm_cell_20/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_20/Reluµ
while/lstm_cell_20/mul_1Mulwhile/lstm_cell_20/Sigmoid:y:0%while/lstm_cell_20/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_20/mul_1ª
while/lstm_cell_20/add_1AddV2while/lstm_cell_20/mul:z:0while/lstm_cell_20/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_20/add_1
while/lstm_cell_20/Sigmoid_2Sigmoid!while/lstm_cell_20/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_20/Sigmoid_2
while/lstm_cell_20/Relu_1Reluwhile/lstm_cell_20/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_20/Relu_1¹
while/lstm_cell_20/mul_2Mul while/lstm_cell_20/Sigmoid_2:y:0'while/lstm_cell_20/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_20/mul_2à
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_20/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_20/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_20/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/Identity_5Þ

while/NoOpNoOp*^while/lstm_cell_20/BiasAdd/ReadVariableOp)^while/lstm_cell_20/MatMul/ReadVariableOp+^while/lstm_cell_20/MatMul_1/ReadVariableOp*"
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
2while_lstm_cell_20_biasadd_readvariableop_resource4while_lstm_cell_20_biasadd_readvariableop_resource_0"l
3while_lstm_cell_20_matmul_1_readvariableop_resource5while_lstm_cell_20_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_20_matmul_readvariableop_resource3while_lstm_cell_20_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2V
)while/lstm_cell_20/BiasAdd/ReadVariableOp)while/lstm_cell_20/BiasAdd/ReadVariableOp2T
(while/lstm_cell_20/MatMul/ReadVariableOp(while/lstm_cell_20/MatMul/ReadVariableOp2X
*while/lstm_cell_20/MatMul_1/ReadVariableOp*while/lstm_cell_20/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
ËF

E__inference_lstm_20_layer_call_and_return_conditional_losses_39101493

inputs(
lstm_cell_20_39101411:	])
lstm_cell_20_39101413:
$
lstm_cell_20_39101415:	
identity¢$lstm_cell_20/StatefulPartitionedCall¢whileD
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
B :2
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
B :2
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
:ÿÿÿÿÿÿÿÿÿ2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :2
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
B :2
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
:ÿÿÿÿÿÿÿÿÿ2	
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
$lstm_cell_20/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_20_39101411lstm_cell_20_39101413lstm_cell_20_39101415*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_lstm_cell_20_layer_call_and_return_conditional_losses_391013462&
$lstm_cell_20/StatefulPartitionedCall
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ  2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_20_39101411lstm_cell_20_39101413lstm_cell_20_39101415*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_39101424*
condR
while_cond_39101423*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ  22
0TensorArrayV2Stack/TensorListStack/element_shapeò
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
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
:ÿÿÿÿÿÿÿÿÿ*
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
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
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
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity}
NoOpNoOp%^lstm_cell_20/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ]: : : 2L
$lstm_cell_20/StatefulPartitionedCall$lstm_cell_20/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ]
 
_user_specified_nameinputs
°?
Ô
while_body_39104427
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_20_matmul_readvariableop_resource_0:	]I
5while_lstm_cell_20_matmul_1_readvariableop_resource_0:
C
4while_lstm_cell_20_biasadd_readvariableop_resource_0:	
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_20_matmul_readvariableop_resource:	]G
3while_lstm_cell_20_matmul_1_readvariableop_resource:
A
2while_lstm_cell_20_biasadd_readvariableop_resource:	¢)while/lstm_cell_20/BiasAdd/ReadVariableOp¢(while/lstm_cell_20/MatMul/ReadVariableOp¢*while/lstm_cell_20/MatMul_1/ReadVariableOpÃ
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
(while/lstm_cell_20/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_20_matmul_readvariableop_resource_0*
_output_shapes
:	]*
dtype02*
(while/lstm_cell_20/MatMul/ReadVariableOp×
while/lstm_cell_20/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_20/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_20/MatMulÐ
*while/lstm_cell_20/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_20_matmul_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype02,
*while/lstm_cell_20/MatMul_1/ReadVariableOpÀ
while/lstm_cell_20/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_20/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_20/MatMul_1¸
while/lstm_cell_20/addAddV2#while/lstm_cell_20/MatMul:product:0%while/lstm_cell_20/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_20/addÈ
)while/lstm_cell_20/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_20_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02+
)while/lstm_cell_20/BiasAdd/ReadVariableOpÅ
while/lstm_cell_20/BiasAddBiasAddwhile/lstm_cell_20/add:z:01while/lstm_cell_20/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_20/BiasAdd
"while/lstm_cell_20/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_20/split/split_dim
while/lstm_cell_20/splitSplit+while/lstm_cell_20/split/split_dim:output:0#while/lstm_cell_20/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2
while/lstm_cell_20/split
while/lstm_cell_20/SigmoidSigmoid!while/lstm_cell_20/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_20/Sigmoid
while/lstm_cell_20/Sigmoid_1Sigmoid!while/lstm_cell_20/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_20/Sigmoid_1¡
while/lstm_cell_20/mulMul while/lstm_cell_20/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_20/mul
while/lstm_cell_20/ReluRelu!while/lstm_cell_20/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_20/Reluµ
while/lstm_cell_20/mul_1Mulwhile/lstm_cell_20/Sigmoid:y:0%while/lstm_cell_20/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_20/mul_1ª
while/lstm_cell_20/add_1AddV2while/lstm_cell_20/mul:z:0while/lstm_cell_20/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_20/add_1
while/lstm_cell_20/Sigmoid_2Sigmoid!while/lstm_cell_20/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_20/Sigmoid_2
while/lstm_cell_20/Relu_1Reluwhile/lstm_cell_20/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_20/Relu_1¹
while/lstm_cell_20/mul_2Mul while/lstm_cell_20/Sigmoid_2:y:0'while/lstm_cell_20/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_20/mul_2à
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_20/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_20/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_20/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/Identity_5Þ

while/NoOpNoOp*^while/lstm_cell_20/BiasAdd/ReadVariableOp)^while/lstm_cell_20/MatMul/ReadVariableOp+^while/lstm_cell_20/MatMul_1/ReadVariableOp*"
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
2while_lstm_cell_20_biasadd_readvariableop_resource4while_lstm_cell_20_biasadd_readvariableop_resource_0"l
3while_lstm_cell_20_matmul_1_readvariableop_resource5while_lstm_cell_20_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_20_matmul_readvariableop_resource3while_lstm_cell_20_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2V
)while/lstm_cell_20/BiasAdd/ReadVariableOp)while/lstm_cell_20/BiasAdd/ReadVariableOp2T
(while/lstm_cell_20/MatMul/ReadVariableOp(while/lstm_cell_20/MatMul/ReadVariableOp2X
*while/lstm_cell_20/MatMul_1/ReadVariableOp*while/lstm_cell_20/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
¨
ñ
K__inference_sequential_10_layer_call_and_return_conditional_losses_39102760

inputs#
lstm_20_39102543:	]$
lstm_20_39102545:

lstm_20_39102547:	$
lstm_21_39102708:

$
lstm_21_39102710:
Å

lstm_21_39102712:	
$
dense_10_39102754:	Å
dense_10_39102756:
identity¢ dense_10/StatefulPartitionedCall¢lstm_20/StatefulPartitionedCall¢lstm_21/StatefulPartitionedCall®
lstm_20/StatefulPartitionedCallStatefulPartitionedCallinputslstm_20_39102543lstm_20_39102545lstm_20_39102547*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_lstm_20_layer_call_and_return_conditional_losses_391025422!
lstm_20/StatefulPartitionedCall
dropout_20/PartitionedCallPartitionedCall(lstm_20/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_dropout_20_layer_call_and_return_conditional_losses_391025552
dropout_20/PartitionedCallË
lstm_21/StatefulPartitionedCallStatefulPartitionedCall#dropout_20/PartitionedCall:output:0lstm_21_39102708lstm_21_39102710lstm_21_39102712*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_lstm_21_layer_call_and_return_conditional_losses_391027072!
lstm_21/StatefulPartitionedCall
dropout_21/PartitionedCallPartitionedCall(lstm_21/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_dropout_21_layer_call_and_return_conditional_losses_391027202
dropout_21/PartitionedCall»
 dense_10/StatefulPartitionedCallStatefulPartitionedCall#dropout_21/PartitionedCall:output:0dense_10_39102754dense_10_39102756*
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
F__inference_dense_10_layer_call_and_return_conditional_losses_391027532"
 dense_10/StatefulPartitionedCall
IdentityIdentity)dense_10/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityµ
NoOpNoOp!^dense_10/StatefulPartitionedCall ^lstm_20/StatefulPartitionedCall ^lstm_21/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ]: : : : : : : : 2D
 dense_10/StatefulPartitionedCall dense_10/StatefulPartitionedCall2B
lstm_20/StatefulPartitionedCalllstm_20/StatefulPartitionedCall2B
lstm_21/StatefulPartitionedCalllstm_21/StatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
 
_user_specified_nameinputs
ã
Í
while_cond_39104577
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_39104577___redundant_placeholder06
2while_while_cond_39104577___redundant_placeholder16
2while_while_cond_39104577___redundant_placeholder26
2while_while_cond_39104577___redundant_placeholder3
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
B: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
\

E__inference_lstm_21_layer_call_and_return_conditional_losses_39105337

inputs?
+lstm_cell_21_matmul_readvariableop_resource:

A
-lstm_cell_21_matmul_1_readvariableop_resource:
Å
;
,lstm_cell_21_biasadd_readvariableop_resource:	

identity¢#lstm_cell_21/BiasAdd/ReadVariableOp¢"lstm_cell_21/MatMul/ReadVariableOp¢$lstm_cell_21/MatMul_1/ReadVariableOp¢whileD
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
B :Å2
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
B :Å2
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
:ÿÿÿÿÿÿÿÿÿÅ2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :Å2
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
B :Å2
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
:ÿÿÿÿÿÿÿÿÿÅ2	
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
:ÿÿÿÿÿÿÿÿÿ2
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
valueB"ÿÿÿÿ  27
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
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
strided_slice_2¶
"lstm_cell_21/MatMul/ReadVariableOpReadVariableOp+lstm_cell_21_matmul_readvariableop_resource* 
_output_shapes
:

*
dtype02$
"lstm_cell_21/MatMul/ReadVariableOp­
lstm_cell_21/MatMulMatMulstrided_slice_2:output:0*lstm_cell_21/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
lstm_cell_21/MatMul¼
$lstm_cell_21/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_21_matmul_1_readvariableop_resource* 
_output_shapes
:
Å
*
dtype02&
$lstm_cell_21/MatMul_1/ReadVariableOp©
lstm_cell_21/MatMul_1MatMulzeros:output:0,lstm_cell_21/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
lstm_cell_21/MatMul_1 
lstm_cell_21/addAddV2lstm_cell_21/MatMul:product:0lstm_cell_21/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
lstm_cell_21/add´
#lstm_cell_21/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_21_biasadd_readvariableop_resource*
_output_shapes	
:
*
dtype02%
#lstm_cell_21/BiasAdd/ReadVariableOp­
lstm_cell_21/BiasAddBiasAddlstm_cell_21/add:z:0+lstm_cell_21/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
lstm_cell_21/BiasAdd~
lstm_cell_21/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_21/split/split_dim÷
lstm_cell_21/splitSplit%lstm_cell_21/split/split_dim:output:0lstm_cell_21/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿÅ:ÿÿÿÿÿÿÿÿÿÅ:ÿÿÿÿÿÿÿÿÿÅ:ÿÿÿÿÿÿÿÿÿÅ*
	num_split2
lstm_cell_21/split
lstm_cell_21/SigmoidSigmoidlstm_cell_21/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2
lstm_cell_21/Sigmoid
lstm_cell_21/Sigmoid_1Sigmoidlstm_cell_21/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2
lstm_cell_21/Sigmoid_1
lstm_cell_21/mulMullstm_cell_21/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2
lstm_cell_21/mul~
lstm_cell_21/ReluRelulstm_cell_21/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2
lstm_cell_21/Relu
lstm_cell_21/mul_1Mullstm_cell_21/Sigmoid:y:0lstm_cell_21/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2
lstm_cell_21/mul_1
lstm_cell_21/add_1AddV2lstm_cell_21/mul:z:0lstm_cell_21/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2
lstm_cell_21/add_1
lstm_cell_21/Sigmoid_2Sigmoidlstm_cell_21/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2
lstm_cell_21/Sigmoid_2}
lstm_cell_21/Relu_1Relulstm_cell_21/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2
lstm_cell_21/Relu_1¡
lstm_cell_21/mul_2Mullstm_cell_21/Sigmoid_2:y:0!lstm_cell_21/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2
lstm_cell_21/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿE  2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_21_matmul_readvariableop_resource-lstm_cell_21_matmul_1_readvariableop_resource,lstm_cell_21_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿÅ:ÿÿÿÿÿÿÿÿÿÅ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_39105253*
condR
while_cond_39105252*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿÅ:ÿÿÿÿÿÿÿÿÿÅ: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿE  22
0TensorArrayV2Stack/TensorListStack/element_shapeé
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ*
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
:ÿÿÿÿÿÿÿÿÿÅ*
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
:ÿÿÿÿÿÿÿÿÿÅ2
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
:ÿÿÿÿÿÿÿÿÿÅ2

IdentityÈ
NoOpNoOp$^lstm_cell_21/BiasAdd/ReadVariableOp#^lstm_cell_21/MatMul/ReadVariableOp%^lstm_cell_21/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:ÿÿÿÿÿÿÿÿÿ: : : 2J
#lstm_cell_21/BiasAdd/ReadVariableOp#lstm_cell_21/BiasAdd/ReadVariableOp2H
"lstm_cell_21/MatMul/ReadVariableOp"lstm_cell_21/MatMul/ReadVariableOp2L
$lstm_cell_21/MatMul_1/ReadVariableOp$lstm_cell_21/MatMul_1/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¶
¸
*__inference_lstm_20_layer_call_fn_39104695

inputs
unknown:	]
	unknown_0:

	unknown_1:	
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_lstm_20_layer_call_and_return_conditional_losses_391025422
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

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
\

E__inference_lstm_21_layer_call_and_return_conditional_losses_39102976

inputs?
+lstm_cell_21_matmul_readvariableop_resource:

A
-lstm_cell_21_matmul_1_readvariableop_resource:
Å
;
,lstm_cell_21_biasadd_readvariableop_resource:	

identity¢#lstm_cell_21/BiasAdd/ReadVariableOp¢"lstm_cell_21/MatMul/ReadVariableOp¢$lstm_cell_21/MatMul_1/ReadVariableOp¢whileD
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
B :Å2
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
B :Å2
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
:ÿÿÿÿÿÿÿÿÿÅ2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :Å2
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
B :Å2
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
:ÿÿÿÿÿÿÿÿÿÅ2	
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
:ÿÿÿÿÿÿÿÿÿ2
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
valueB"ÿÿÿÿ  27
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
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
strided_slice_2¶
"lstm_cell_21/MatMul/ReadVariableOpReadVariableOp+lstm_cell_21_matmul_readvariableop_resource* 
_output_shapes
:

*
dtype02$
"lstm_cell_21/MatMul/ReadVariableOp­
lstm_cell_21/MatMulMatMulstrided_slice_2:output:0*lstm_cell_21/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
lstm_cell_21/MatMul¼
$lstm_cell_21/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_21_matmul_1_readvariableop_resource* 
_output_shapes
:
Å
*
dtype02&
$lstm_cell_21/MatMul_1/ReadVariableOp©
lstm_cell_21/MatMul_1MatMulzeros:output:0,lstm_cell_21/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
lstm_cell_21/MatMul_1 
lstm_cell_21/addAddV2lstm_cell_21/MatMul:product:0lstm_cell_21/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
lstm_cell_21/add´
#lstm_cell_21/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_21_biasadd_readvariableop_resource*
_output_shapes	
:
*
dtype02%
#lstm_cell_21/BiasAdd/ReadVariableOp­
lstm_cell_21/BiasAddBiasAddlstm_cell_21/add:z:0+lstm_cell_21/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
lstm_cell_21/BiasAdd~
lstm_cell_21/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_21/split/split_dim÷
lstm_cell_21/splitSplit%lstm_cell_21/split/split_dim:output:0lstm_cell_21/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿÅ:ÿÿÿÿÿÿÿÿÿÅ:ÿÿÿÿÿÿÿÿÿÅ:ÿÿÿÿÿÿÿÿÿÅ*
	num_split2
lstm_cell_21/split
lstm_cell_21/SigmoidSigmoidlstm_cell_21/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2
lstm_cell_21/Sigmoid
lstm_cell_21/Sigmoid_1Sigmoidlstm_cell_21/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2
lstm_cell_21/Sigmoid_1
lstm_cell_21/mulMullstm_cell_21/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2
lstm_cell_21/mul~
lstm_cell_21/ReluRelulstm_cell_21/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2
lstm_cell_21/Relu
lstm_cell_21/mul_1Mullstm_cell_21/Sigmoid:y:0lstm_cell_21/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2
lstm_cell_21/mul_1
lstm_cell_21/add_1AddV2lstm_cell_21/mul:z:0lstm_cell_21/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2
lstm_cell_21/add_1
lstm_cell_21/Sigmoid_2Sigmoidlstm_cell_21/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2
lstm_cell_21/Sigmoid_2}
lstm_cell_21/Relu_1Relulstm_cell_21/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2
lstm_cell_21/Relu_1¡
lstm_cell_21/mul_2Mullstm_cell_21/Sigmoid_2:y:0!lstm_cell_21/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2
lstm_cell_21/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿE  2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_21_matmul_readvariableop_resource-lstm_cell_21_matmul_1_readvariableop_resource,lstm_cell_21_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿÅ:ÿÿÿÿÿÿÿÿÿÅ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_39102892*
condR
while_cond_39102891*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿÅ:ÿÿÿÿÿÿÿÿÿÅ: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿE  22
0TensorArrayV2Stack/TensorListStack/element_shapeé
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ*
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
:ÿÿÿÿÿÿÿÿÿÅ*
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
:ÿÿÿÿÿÿÿÿÿÅ2
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
:ÿÿÿÿÿÿÿÿÿÅ2

IdentityÈ
NoOpNoOp$^lstm_cell_21/BiasAdd/ReadVariableOp#^lstm_cell_21/MatMul/ReadVariableOp%^lstm_cell_21/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:ÿÿÿÿÿÿÿÿÿ: : : 2J
#lstm_cell_21/BiasAdd/ReadVariableOp#lstm_cell_21/BiasAdd/ReadVariableOp2H
"lstm_cell_21/MatMul/ReadVariableOp"lstm_cell_21/MatMul/ReadVariableOp2L
$lstm_cell_21/MatMul_1/ReadVariableOp$lstm_cell_21/MatMul_1/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


J__inference_lstm_cell_21_layer_call_and_return_conditional_losses_39105610

inputs
states_0
states_12
matmul_readvariableop_resource:

4
 matmul_1_readvariableop_resource:
Å
.
biasadd_readvariableop_resource:	

identity

identity_1

identity_2¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:

*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
MatMul
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
Å
*
dtype02
MatMul_1/ReadVariableOp|
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
add
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:
*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
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
P:ÿÿÿÿÿÿÿÿÿÅ:ÿÿÿÿÿÿÿÿÿÅ:ÿÿÿÿÿÿÿÿÿÅ:ÿÿÿÿÿÿÿÿÿÅ*
	num_split2
split`
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2	
Sigmoidd
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2
	Sigmoid_1]
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2
mulW
ReluRelusplit:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2
Relui
mul_1MulSigmoid:y:0Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2
mul_1^
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2
add_1d
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2
	Sigmoid_2V
Relu_1Relu	add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2
Relu_1m
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2
mul_2e
IdentityIdentity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2

Identityi

Identity_1Identity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2

Identity_1i

Identity_2Identity	add_1:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2

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
B:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿÅ:ÿÿÿÿÿÿÿÿÿÅ: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ
"
_user_specified_name
states/0:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ
"
_user_specified_name
states/1
´?
Ö
while_body_39102623
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
3while_lstm_cell_21_matmul_readvariableop_resource_0:

I
5while_lstm_cell_21_matmul_1_readvariableop_resource_0:
Å
C
4while_lstm_cell_21_biasadd_readvariableop_resource_0:	

while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
1while_lstm_cell_21_matmul_readvariableop_resource:

G
3while_lstm_cell_21_matmul_1_readvariableop_resource:
Å
A
2while_lstm_cell_21_biasadd_readvariableop_resource:	
¢)while/lstm_cell_21/BiasAdd/ReadVariableOp¢(while/lstm_cell_21/MatMul/ReadVariableOp¢*while/lstm_cell_21/MatMul_1/ReadVariableOpÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ  29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÔ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemÊ
(while/lstm_cell_21/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_21_matmul_readvariableop_resource_0* 
_output_shapes
:

*
dtype02*
(while/lstm_cell_21/MatMul/ReadVariableOp×
while/lstm_cell_21/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_21/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
while/lstm_cell_21/MatMulÐ
*while/lstm_cell_21/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_21_matmul_1_readvariableop_resource_0* 
_output_shapes
:
Å
*
dtype02,
*while/lstm_cell_21/MatMul_1/ReadVariableOpÀ
while/lstm_cell_21/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_21/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
while/lstm_cell_21/MatMul_1¸
while/lstm_cell_21/addAddV2#while/lstm_cell_21/MatMul:product:0%while/lstm_cell_21/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
while/lstm_cell_21/addÈ
)while/lstm_cell_21/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_21_biasadd_readvariableop_resource_0*
_output_shapes	
:
*
dtype02+
)while/lstm_cell_21/BiasAdd/ReadVariableOpÅ
while/lstm_cell_21/BiasAddBiasAddwhile/lstm_cell_21/add:z:01while/lstm_cell_21/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
while/lstm_cell_21/BiasAdd
"while/lstm_cell_21/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_21/split/split_dim
while/lstm_cell_21/splitSplit+while/lstm_cell_21/split/split_dim:output:0#while/lstm_cell_21/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿÅ:ÿÿÿÿÿÿÿÿÿÅ:ÿÿÿÿÿÿÿÿÿÅ:ÿÿÿÿÿÿÿÿÿÅ*
	num_split2
while/lstm_cell_21/split
while/lstm_cell_21/SigmoidSigmoid!while/lstm_cell_21/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2
while/lstm_cell_21/Sigmoid
while/lstm_cell_21/Sigmoid_1Sigmoid!while/lstm_cell_21/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2
while/lstm_cell_21/Sigmoid_1¡
while/lstm_cell_21/mulMul while/lstm_cell_21/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2
while/lstm_cell_21/mul
while/lstm_cell_21/ReluRelu!while/lstm_cell_21/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2
while/lstm_cell_21/Reluµ
while/lstm_cell_21/mul_1Mulwhile/lstm_cell_21/Sigmoid:y:0%while/lstm_cell_21/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2
while/lstm_cell_21/mul_1ª
while/lstm_cell_21/add_1AddV2while/lstm_cell_21/mul:z:0while/lstm_cell_21/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2
while/lstm_cell_21/add_1
while/lstm_cell_21/Sigmoid_2Sigmoid!while/lstm_cell_21/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2
while/lstm_cell_21/Sigmoid_2
while/lstm_cell_21/Relu_1Reluwhile/lstm_cell_21/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2
while/lstm_cell_21/Relu_1¹
while/lstm_cell_21/mul_2Mul while/lstm_cell_21/Sigmoid_2:y:0'while/lstm_cell_21/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2
while/lstm_cell_21/mul_2à
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_21/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_21/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_21/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2
while/Identity_5Þ

while/NoOpNoOp*^while/lstm_cell_21/BiasAdd/ReadVariableOp)^while/lstm_cell_21/MatMul/ReadVariableOp+^while/lstm_cell_21/MatMul_1/ReadVariableOp*"
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
2while_lstm_cell_21_biasadd_readvariableop_resource4while_lstm_cell_21_biasadd_readvariableop_resource_0"l
3while_lstm_cell_21_matmul_1_readvariableop_resource5while_lstm_cell_21_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_21_matmul_readvariableop_resource3while_lstm_cell_21_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿÅ:ÿÿÿÿÿÿÿÿÿÅ: : : : : 2V
)while/lstm_cell_21/BiasAdd/ReadVariableOp)while/lstm_cell_21/BiasAdd/ReadVariableOp2T
(while/lstm_cell_21/MatMul/ReadVariableOp(while/lstm_cell_21/MatMul/ReadVariableOp2X
*while/lstm_cell_21/MatMul_1/ReadVariableOp*while/lstm_cell_21/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ:

_output_shapes
: :

_output_shapes
: 
ã
Í
while_cond_39105101
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_39105101___redundant_placeholder06
2while_while_cond_39105101___redundant_placeholder16
2while_while_cond_39105101___redundant_placeholder26
2while_while_cond_39105101___redundant_placeholder3
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
B: : : : :ÿÿÿÿÿÿÿÿÿÅ:ÿÿÿÿÿÿÿÿÿÅ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ:

_output_shapes
: :

_output_shapes
:
à
º
*__inference_lstm_20_layer_call_fn_39104673
inputs_0
unknown:	]
	unknown_0:

	unknown_1:	
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_lstm_20_layer_call_and_return_conditional_losses_391012832
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

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
°?
Ô
while_body_39104125
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_20_matmul_readvariableop_resource_0:	]I
5while_lstm_cell_20_matmul_1_readvariableop_resource_0:
C
4while_lstm_cell_20_biasadd_readvariableop_resource_0:	
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_20_matmul_readvariableop_resource:	]G
3while_lstm_cell_20_matmul_1_readvariableop_resource:
A
2while_lstm_cell_20_biasadd_readvariableop_resource:	¢)while/lstm_cell_20/BiasAdd/ReadVariableOp¢(while/lstm_cell_20/MatMul/ReadVariableOp¢*while/lstm_cell_20/MatMul_1/ReadVariableOpÃ
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
(while/lstm_cell_20/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_20_matmul_readvariableop_resource_0*
_output_shapes
:	]*
dtype02*
(while/lstm_cell_20/MatMul/ReadVariableOp×
while/lstm_cell_20/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_20/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_20/MatMulÐ
*while/lstm_cell_20/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_20_matmul_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype02,
*while/lstm_cell_20/MatMul_1/ReadVariableOpÀ
while/lstm_cell_20/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_20/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_20/MatMul_1¸
while/lstm_cell_20/addAddV2#while/lstm_cell_20/MatMul:product:0%while/lstm_cell_20/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_20/addÈ
)while/lstm_cell_20/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_20_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02+
)while/lstm_cell_20/BiasAdd/ReadVariableOpÅ
while/lstm_cell_20/BiasAddBiasAddwhile/lstm_cell_20/add:z:01while/lstm_cell_20/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_20/BiasAdd
"while/lstm_cell_20/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_20/split/split_dim
while/lstm_cell_20/splitSplit+while/lstm_cell_20/split/split_dim:output:0#while/lstm_cell_20/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2
while/lstm_cell_20/split
while/lstm_cell_20/SigmoidSigmoid!while/lstm_cell_20/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_20/Sigmoid
while/lstm_cell_20/Sigmoid_1Sigmoid!while/lstm_cell_20/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_20/Sigmoid_1¡
while/lstm_cell_20/mulMul while/lstm_cell_20/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_20/mul
while/lstm_cell_20/ReluRelu!while/lstm_cell_20/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_20/Reluµ
while/lstm_cell_20/mul_1Mulwhile/lstm_cell_20/Sigmoid:y:0%while/lstm_cell_20/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_20/mul_1ª
while/lstm_cell_20/add_1AddV2while/lstm_cell_20/mul:z:0while/lstm_cell_20/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_20/add_1
while/lstm_cell_20/Sigmoid_2Sigmoid!while/lstm_cell_20/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_20/Sigmoid_2
while/lstm_cell_20/Relu_1Reluwhile/lstm_cell_20/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_20/Relu_1¹
while/lstm_cell_20/mul_2Mul while/lstm_cell_20/Sigmoid_2:y:0'while/lstm_cell_20/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_20/mul_2à
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_20/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_20/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_20/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/Identity_5Þ

while/NoOpNoOp*^while/lstm_cell_20/BiasAdd/ReadVariableOp)^while/lstm_cell_20/MatMul/ReadVariableOp+^while/lstm_cell_20/MatMul_1/ReadVariableOp*"
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
2while_lstm_cell_20_biasadd_readvariableop_resource4while_lstm_cell_20_biasadd_readvariableop_resource_0"l
3while_lstm_cell_20_matmul_1_readvariableop_resource5while_lstm_cell_20_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_20_matmul_readvariableop_resource3while_lstm_cell_20_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2V
)while/lstm_cell_20/BiasAdd/ReadVariableOp)while/lstm_cell_20/BiasAdd/ReadVariableOp2T
(while/lstm_cell_20/MatMul/ReadVariableOp(while/lstm_cell_20/MatMul/ReadVariableOp2X
*while/lstm_cell_20/MatMul_1/ReadVariableOp*while/lstm_cell_20/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
ã
Í
while_cond_39104426
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_39104426___redundant_placeholder06
2while_while_cond_39104426___redundant_placeholder16
2while_while_cond_39104426___redundant_placeholder26
2while_while_cond_39104426___redundant_placeholder3
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
B: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
ÏL
Ü
!__inference__traced_save_39105766
file_prefix.
*savev2_dense_10_kernel_read_readvariableop,
(savev2_dense_10_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop:
6savev2_lstm_20_lstm_cell_20_kernel_read_readvariableopD
@savev2_lstm_20_lstm_cell_20_recurrent_kernel_read_readvariableop8
4savev2_lstm_20_lstm_cell_20_bias_read_readvariableop:
6savev2_lstm_21_lstm_cell_21_kernel_read_readvariableopD
@savev2_lstm_21_lstm_cell_21_recurrent_kernel_read_readvariableop8
4savev2_lstm_21_lstm_cell_21_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop5
1savev2_adam_dense_10_kernel_m_read_readvariableop3
/savev2_adam_dense_10_bias_m_read_readvariableopA
=savev2_adam_lstm_20_lstm_cell_20_kernel_m_read_readvariableopK
Gsavev2_adam_lstm_20_lstm_cell_20_recurrent_kernel_m_read_readvariableop?
;savev2_adam_lstm_20_lstm_cell_20_bias_m_read_readvariableopA
=savev2_adam_lstm_21_lstm_cell_21_kernel_m_read_readvariableopK
Gsavev2_adam_lstm_21_lstm_cell_21_recurrent_kernel_m_read_readvariableop?
;savev2_adam_lstm_21_lstm_cell_21_bias_m_read_readvariableop5
1savev2_adam_dense_10_kernel_v_read_readvariableop3
/savev2_adam_dense_10_bias_v_read_readvariableopA
=savev2_adam_lstm_20_lstm_cell_20_kernel_v_read_readvariableopK
Gsavev2_adam_lstm_20_lstm_cell_20_recurrent_kernel_v_read_readvariableop?
;savev2_adam_lstm_20_lstm_cell_20_bias_v_read_readvariableopA
=savev2_adam_lstm_21_lstm_cell_21_kernel_v_read_readvariableopK
Gsavev2_adam_lstm_21_lstm_cell_21_recurrent_kernel_v_read_readvariableop?
;savev2_adam_lstm_21_lstm_cell_21_bias_v_read_readvariableop
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
ShardedFilenameì
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:"*
dtype0*þ
valueôBñ"B6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_namesÌ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:"*
dtype0*W
valueNBL"B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slicesÅ
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_dense_10_kernel_read_readvariableop(savev2_dense_10_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop6savev2_lstm_20_lstm_cell_20_kernel_read_readvariableop@savev2_lstm_20_lstm_cell_20_recurrent_kernel_read_readvariableop4savev2_lstm_20_lstm_cell_20_bias_read_readvariableop6savev2_lstm_21_lstm_cell_21_kernel_read_readvariableop@savev2_lstm_21_lstm_cell_21_recurrent_kernel_read_readvariableop4savev2_lstm_21_lstm_cell_21_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop1savev2_adam_dense_10_kernel_m_read_readvariableop/savev2_adam_dense_10_bias_m_read_readvariableop=savev2_adam_lstm_20_lstm_cell_20_kernel_m_read_readvariableopGsavev2_adam_lstm_20_lstm_cell_20_recurrent_kernel_m_read_readvariableop;savev2_adam_lstm_20_lstm_cell_20_bias_m_read_readvariableop=savev2_adam_lstm_21_lstm_cell_21_kernel_m_read_readvariableopGsavev2_adam_lstm_21_lstm_cell_21_recurrent_kernel_m_read_readvariableop;savev2_adam_lstm_21_lstm_cell_21_bias_m_read_readvariableop1savev2_adam_dense_10_kernel_v_read_readvariableop/savev2_adam_dense_10_bias_v_read_readvariableop=savev2_adam_lstm_20_lstm_cell_20_kernel_v_read_readvariableopGsavev2_adam_lstm_20_lstm_cell_20_recurrent_kernel_v_read_readvariableop;savev2_adam_lstm_20_lstm_cell_20_bias_v_read_readvariableop=savev2_adam_lstm_21_lstm_cell_21_kernel_v_read_readvariableopGsavev2_adam_lstm_21_lstm_cell_21_recurrent_kernel_v_read_readvariableop;savev2_adam_lstm_21_lstm_cell_21_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *0
dtypes&
$2"	2
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

identity_1Identity_1:output:0*
_input_shapes
: :	Å:: : : : : :	]:
::

:
Å
:
: : : : :	Å::	]:
::

:
Å
:
:	Å::	]:
::

:
Å
:
: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	Å: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	]:&	"
 
_output_shapes
:
:!


_output_shapes	
::&"
 
_output_shapes
:

:&"
 
_output_shapes
:
Å
:!

_output_shapes	
:
:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	Å: 

_output_shapes
::%!

_output_shapes
:	]:&"
 
_output_shapes
:
:!

_output_shapes	
::&"
 
_output_shapes
:

:&"
 
_output_shapes
:
Å
:!

_output_shapes	
:
:%!

_output_shapes
:	Å: 

_output_shapes
::%!

_output_shapes
:	]:&"
 
_output_shapes
:
:!

_output_shapes	
::&"
 
_output_shapes
:

:& "
 
_output_shapes
:
Å
:!!

_output_shapes	
:
:"

_output_shapes
: 
&
ó
while_body_39101424
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_00
while_lstm_cell_20_39101448_0:	]1
while_lstm_cell_20_39101450_0:
,
while_lstm_cell_20_39101452_0:	
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor.
while_lstm_cell_20_39101448:	]/
while_lstm_cell_20_39101450:
*
while_lstm_cell_20_39101452:	¢*while/lstm_cell_20/StatefulPartitionedCallÃ
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
*while/lstm_cell_20/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_20_39101448_0while_lstm_cell_20_39101450_0while_lstm_cell_20_39101452_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_lstm_cell_20_layer_call_and_return_conditional_losses_391013462,
*while/lstm_cell_20/StatefulPartitionedCall÷
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder3while/lstm_cell_20/StatefulPartitionedCall:output:0*
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
while/Identity_4Identity3while/lstm_cell_20/StatefulPartitionedCall:output:1^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/Identity_4¥
while/Identity_5Identity3while/lstm_cell_20/StatefulPartitionedCall:output:2^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/Identity_5

while/NoOpNoOp+^while/lstm_cell_20/StatefulPartitionedCall*"
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
while_lstm_cell_20_39101448while_lstm_cell_20_39101448_0"<
while_lstm_cell_20_39101450while_lstm_cell_20_39101450_0"<
while_lstm_cell_20_39101452while_lstm_cell_20_39101452_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2X
*while/lstm_cell_20/StatefulPartitionedCall*while/lstm_cell_20/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
Ç
ù
/__inference_lstm_cell_20_layer_call_fn_39105529

inputs
states_0
states_1
unknown:	]
	unknown_0:

	unknown_1:	
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
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_lstm_cell_20_layer_call_and_return_conditional_losses_391012002
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

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
A:ÿÿÿÿÿÿÿÿÿ]:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
 
_user_specified_nameinputs:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
states/0:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
states/1
ã
Í
while_cond_39104950
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_39104950___redundant_placeholder06
2while_while_cond_39104950___redundant_placeholder16
2while_while_cond_39104950___redundant_placeholder26
2while_while_cond_39104950___redundant_placeholder3
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
B: : : : :ÿÿÿÿÿÿÿÿÿÅ:ÿÿÿÿÿÿÿÿÿÅ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ:

_output_shapes
: :

_output_shapes
:
\

E__inference_lstm_20_layer_call_and_return_conditional_losses_39104662

inputs>
+lstm_cell_20_matmul_readvariableop_resource:	]A
-lstm_cell_20_matmul_1_readvariableop_resource:
;
,lstm_cell_20_biasadd_readvariableop_resource:	
identity¢#lstm_cell_20/BiasAdd/ReadVariableOp¢"lstm_cell_20/MatMul/ReadVariableOp¢$lstm_cell_20/MatMul_1/ReadVariableOp¢whileD
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
B :2
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
B :2
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
:ÿÿÿÿÿÿÿÿÿ2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :2
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
B :2
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
:ÿÿÿÿÿÿÿÿÿ2	
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
"lstm_cell_20/MatMul/ReadVariableOpReadVariableOp+lstm_cell_20_matmul_readvariableop_resource*
_output_shapes
:	]*
dtype02$
"lstm_cell_20/MatMul/ReadVariableOp­
lstm_cell_20/MatMulMatMulstrided_slice_2:output:0*lstm_cell_20/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_20/MatMul¼
$lstm_cell_20/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_20_matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype02&
$lstm_cell_20/MatMul_1/ReadVariableOp©
lstm_cell_20/MatMul_1MatMulzeros:output:0,lstm_cell_20/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_20/MatMul_1 
lstm_cell_20/addAddV2lstm_cell_20/MatMul:product:0lstm_cell_20/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_20/add´
#lstm_cell_20/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_20_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02%
#lstm_cell_20/BiasAdd/ReadVariableOp­
lstm_cell_20/BiasAddBiasAddlstm_cell_20/add:z:0+lstm_cell_20/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_20/BiasAdd~
lstm_cell_20/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_20/split/split_dim÷
lstm_cell_20/splitSplit%lstm_cell_20/split/split_dim:output:0lstm_cell_20/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2
lstm_cell_20/split
lstm_cell_20/SigmoidSigmoidlstm_cell_20/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_20/Sigmoid
lstm_cell_20/Sigmoid_1Sigmoidlstm_cell_20/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_20/Sigmoid_1
lstm_cell_20/mulMullstm_cell_20/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_20/mul~
lstm_cell_20/ReluRelulstm_cell_20/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_20/Relu
lstm_cell_20/mul_1Mullstm_cell_20/Sigmoid:y:0lstm_cell_20/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_20/mul_1
lstm_cell_20/add_1AddV2lstm_cell_20/mul:z:0lstm_cell_20/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_20/add_1
lstm_cell_20/Sigmoid_2Sigmoidlstm_cell_20/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_20/Sigmoid_2}
lstm_cell_20/Relu_1Relulstm_cell_20/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_20/Relu_1¡
lstm_cell_20/mul_2Mullstm_cell_20/Sigmoid_2:y:0!lstm_cell_20/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_20/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ  2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_20_matmul_readvariableop_resource-lstm_cell_20_matmul_1_readvariableop_resource,lstm_cell_20_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_39104578*
condR
while_cond_39104577*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ  22
0TensorArrayV2Stack/TensorListStack/element_shapeé
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
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
:ÿÿÿÿÿÿÿÿÿ*
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
:ÿÿÿÿÿÿÿÿÿ2
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
:ÿÿÿÿÿÿÿÿÿ2

IdentityÈ
NoOpNoOp$^lstm_cell_20/BiasAdd/ReadVariableOp#^lstm_cell_20/MatMul/ReadVariableOp%^lstm_cell_20/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ]: : : 2J
#lstm_cell_20/BiasAdd/ReadVariableOp#lstm_cell_20/BiasAdd/ReadVariableOp2H
"lstm_cell_20/MatMul/ReadVariableOp"lstm_cell_20/MatMul/ReadVariableOp2L
$lstm_cell_20/MatMul_1/ReadVariableOp$lstm_cell_20/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
 
_user_specified_nameinputs


Í
0__inference_sequential_10_layer_call_fn_39104037

inputs
unknown:	]
	unknown_0:

	unknown_1:	
	unknown_2:


	unknown_3:
Å

	unknown_4:	

	unknown_5:	Å
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
K__inference_sequential_10_layer_call_and_return_conditional_losses_391027602
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


J__inference_lstm_cell_20_layer_call_and_return_conditional_losses_39101200

inputs

states
states_11
matmul_readvariableop_resource:	]4
 matmul_1_readvariableop_resource:
.
biasadd_readvariableop_resource:	
identity

identity_1

identity_2¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	]*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
add
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
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
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2
split`
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
Sigmoidd
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	Sigmoid_1]
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mulW
ReluRelusplit:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relui
mul_1MulSigmoid:y:0Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mul_1^
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
add_1d
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	Sigmoid_2V
Relu_1Relu	add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu_1m
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mul_2e
IdentityIdentity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityi

Identity_1Identity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1i

Identity_2Identity	add_1:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

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
A:ÿÿÿÿÿÿÿÿÿ]:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
 
_user_specified_nameinputs:PL
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_namestates:PL
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_namestates
É\
¡
E__inference_lstm_21_layer_call_and_return_conditional_losses_39105035
inputs_0?
+lstm_cell_21_matmul_readvariableop_resource:

A
-lstm_cell_21_matmul_1_readvariableop_resource:
Å
;
,lstm_cell_21_biasadd_readvariableop_resource:	

identity¢#lstm_cell_21/BiasAdd/ReadVariableOp¢"lstm_cell_21/MatMul/ReadVariableOp¢$lstm_cell_21/MatMul_1/ReadVariableOp¢whileF
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
B :Å2
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
B :Å2
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
:ÿÿÿÿÿÿÿÿÿÅ2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :Å2
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
B :Å2
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
:ÿÿÿÿÿÿÿÿÿÅ2	
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
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
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
valueB"ÿÿÿÿ  27
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
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
strided_slice_2¶
"lstm_cell_21/MatMul/ReadVariableOpReadVariableOp+lstm_cell_21_matmul_readvariableop_resource* 
_output_shapes
:

*
dtype02$
"lstm_cell_21/MatMul/ReadVariableOp­
lstm_cell_21/MatMulMatMulstrided_slice_2:output:0*lstm_cell_21/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
lstm_cell_21/MatMul¼
$lstm_cell_21/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_21_matmul_1_readvariableop_resource* 
_output_shapes
:
Å
*
dtype02&
$lstm_cell_21/MatMul_1/ReadVariableOp©
lstm_cell_21/MatMul_1MatMulzeros:output:0,lstm_cell_21/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
lstm_cell_21/MatMul_1 
lstm_cell_21/addAddV2lstm_cell_21/MatMul:product:0lstm_cell_21/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
lstm_cell_21/add´
#lstm_cell_21/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_21_biasadd_readvariableop_resource*
_output_shapes	
:
*
dtype02%
#lstm_cell_21/BiasAdd/ReadVariableOp­
lstm_cell_21/BiasAddBiasAddlstm_cell_21/add:z:0+lstm_cell_21/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
lstm_cell_21/BiasAdd~
lstm_cell_21/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_21/split/split_dim÷
lstm_cell_21/splitSplit%lstm_cell_21/split/split_dim:output:0lstm_cell_21/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿÅ:ÿÿÿÿÿÿÿÿÿÅ:ÿÿÿÿÿÿÿÿÿÅ:ÿÿÿÿÿÿÿÿÿÅ*
	num_split2
lstm_cell_21/split
lstm_cell_21/SigmoidSigmoidlstm_cell_21/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2
lstm_cell_21/Sigmoid
lstm_cell_21/Sigmoid_1Sigmoidlstm_cell_21/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2
lstm_cell_21/Sigmoid_1
lstm_cell_21/mulMullstm_cell_21/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2
lstm_cell_21/mul~
lstm_cell_21/ReluRelulstm_cell_21/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2
lstm_cell_21/Relu
lstm_cell_21/mul_1Mullstm_cell_21/Sigmoid:y:0lstm_cell_21/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2
lstm_cell_21/mul_1
lstm_cell_21/add_1AddV2lstm_cell_21/mul:z:0lstm_cell_21/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2
lstm_cell_21/add_1
lstm_cell_21/Sigmoid_2Sigmoidlstm_cell_21/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2
lstm_cell_21/Sigmoid_2}
lstm_cell_21/Relu_1Relulstm_cell_21/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2
lstm_cell_21/Relu_1¡
lstm_cell_21/mul_2Mullstm_cell_21/Sigmoid_2:y:0!lstm_cell_21/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2
lstm_cell_21/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿE  2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_21_matmul_readvariableop_resource-lstm_cell_21_matmul_1_readvariableop_resource,lstm_cell_21_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿÅ:ÿÿÿÿÿÿÿÿÿÅ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_39104951*
condR
while_cond_39104950*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿÅ:ÿÿÿÿÿÿÿÿÿÅ: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿE  22
0TensorArrayV2Stack/TensorListStack/element_shapeò
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÅ*
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
:ÿÿÿÿÿÿÿÿÿÅ*
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
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÅ2
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
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÅ2

IdentityÈ
NoOpNoOp$^lstm_cell_21/BiasAdd/ReadVariableOp#^lstm_cell_21/MatMul/ReadVariableOp%^lstm_cell_21/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 2J
#lstm_cell_21/BiasAdd/ReadVariableOp#lstm_cell_21/BiasAdd/ReadVariableOp2H
"lstm_cell_21/MatMul/ReadVariableOp"lstm_cell_21/MatMul/ReadVariableOp2L
$lstm_cell_21/MatMul_1/ReadVariableOp$lstm_cell_21/MatMul_1/ReadVariableOp2
whilewhile:_ [
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0
&
õ
while_body_39102054
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_01
while_lstm_cell_21_39102078_0:

1
while_lstm_cell_21_39102080_0:
Å
,
while_lstm_cell_21_39102082_0:	

while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor/
while_lstm_cell_21_39102078:

/
while_lstm_cell_21_39102080:
Å
*
while_lstm_cell_21_39102082:	
¢*while/lstm_cell_21/StatefulPartitionedCallÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ  29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÔ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemì
*while/lstm_cell_21/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_21_39102078_0while_lstm_cell_21_39102080_0while_lstm_cell_21_39102082_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿÅ:ÿÿÿÿÿÿÿÿÿÅ:ÿÿÿÿÿÿÿÿÿÅ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_lstm_cell_21_layer_call_and_return_conditional_losses_391019762,
*while/lstm_cell_21/StatefulPartitionedCall÷
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder3while/lstm_cell_21/StatefulPartitionedCall:output:0*
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
while/Identity_4Identity3while/lstm_cell_21/StatefulPartitionedCall:output:1^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2
while/Identity_4¥
while/Identity_5Identity3while/lstm_cell_21/StatefulPartitionedCall:output:2^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2
while/Identity_5

while/NoOpNoOp+^while/lstm_cell_21/StatefulPartitionedCall*"
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
while_lstm_cell_21_39102078while_lstm_cell_21_39102078_0"<
while_lstm_cell_21_39102080while_lstm_cell_21_39102080_0"<
while_lstm_cell_21_39102082while_lstm_cell_21_39102082_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿÅ:ÿÿÿÿÿÿÿÿÿÅ: : : : : 2X
*while/lstm_cell_21/StatefulPartitionedCall*while/lstm_cell_21/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ:

_output_shapes
: :

_output_shapes
: 
Õ!
þ
F__inference_dense_10_layer_call_and_return_conditional_losses_39105439

inputs4
!tensordot_readvariableop_resource:	Å-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢Tensordot/ReadVariableOp
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes
:	Å*
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
:ÿÿÿÿÿÿÿÿÿÅ2
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
:ÿÿÿÿÿÿÿÿÿÅ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ
 
_user_specified_nameinputs
ËF

E__inference_lstm_20_layer_call_and_return_conditional_losses_39101283

inputs(
lstm_cell_20_39101201:	])
lstm_cell_20_39101203:
$
lstm_cell_20_39101205:	
identity¢$lstm_cell_20/StatefulPartitionedCall¢whileD
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
B :2
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
B :2
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
:ÿÿÿÿÿÿÿÿÿ2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :2
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
B :2
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
:ÿÿÿÿÿÿÿÿÿ2	
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
$lstm_cell_20/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_20_39101201lstm_cell_20_39101203lstm_cell_20_39101205*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_lstm_cell_20_layer_call_and_return_conditional_losses_391012002&
$lstm_cell_20/StatefulPartitionedCall
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ  2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_20_39101201lstm_cell_20_39101203lstm_cell_20_39101205*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_39101214*
condR
while_cond_39101213*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ  22
0TensorArrayV2Stack/TensorListStack/element_shapeò
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
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
:ÿÿÿÿÿÿÿÿÿ*
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
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
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
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity}
NoOpNoOp%^lstm_cell_20/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ]: : : 2L
$lstm_cell_20/StatefulPartitionedCall$lstm_cell_20/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ]
 
_user_specified_nameinputs
´?
Ö
while_body_39104800
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
3while_lstm_cell_21_matmul_readvariableop_resource_0:

I
5while_lstm_cell_21_matmul_1_readvariableop_resource_0:
Å
C
4while_lstm_cell_21_biasadd_readvariableop_resource_0:	

while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
1while_lstm_cell_21_matmul_readvariableop_resource:

G
3while_lstm_cell_21_matmul_1_readvariableop_resource:
Å
A
2while_lstm_cell_21_biasadd_readvariableop_resource:	
¢)while/lstm_cell_21/BiasAdd/ReadVariableOp¢(while/lstm_cell_21/MatMul/ReadVariableOp¢*while/lstm_cell_21/MatMul_1/ReadVariableOpÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ  29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÔ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemÊ
(while/lstm_cell_21/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_21_matmul_readvariableop_resource_0* 
_output_shapes
:

*
dtype02*
(while/lstm_cell_21/MatMul/ReadVariableOp×
while/lstm_cell_21/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_21/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
while/lstm_cell_21/MatMulÐ
*while/lstm_cell_21/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_21_matmul_1_readvariableop_resource_0* 
_output_shapes
:
Å
*
dtype02,
*while/lstm_cell_21/MatMul_1/ReadVariableOpÀ
while/lstm_cell_21/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_21/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
while/lstm_cell_21/MatMul_1¸
while/lstm_cell_21/addAddV2#while/lstm_cell_21/MatMul:product:0%while/lstm_cell_21/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
while/lstm_cell_21/addÈ
)while/lstm_cell_21/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_21_biasadd_readvariableop_resource_0*
_output_shapes	
:
*
dtype02+
)while/lstm_cell_21/BiasAdd/ReadVariableOpÅ
while/lstm_cell_21/BiasAddBiasAddwhile/lstm_cell_21/add:z:01while/lstm_cell_21/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
while/lstm_cell_21/BiasAdd
"while/lstm_cell_21/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_21/split/split_dim
while/lstm_cell_21/splitSplit+while/lstm_cell_21/split/split_dim:output:0#while/lstm_cell_21/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿÅ:ÿÿÿÿÿÿÿÿÿÅ:ÿÿÿÿÿÿÿÿÿÅ:ÿÿÿÿÿÿÿÿÿÅ*
	num_split2
while/lstm_cell_21/split
while/lstm_cell_21/SigmoidSigmoid!while/lstm_cell_21/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2
while/lstm_cell_21/Sigmoid
while/lstm_cell_21/Sigmoid_1Sigmoid!while/lstm_cell_21/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2
while/lstm_cell_21/Sigmoid_1¡
while/lstm_cell_21/mulMul while/lstm_cell_21/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2
while/lstm_cell_21/mul
while/lstm_cell_21/ReluRelu!while/lstm_cell_21/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2
while/lstm_cell_21/Reluµ
while/lstm_cell_21/mul_1Mulwhile/lstm_cell_21/Sigmoid:y:0%while/lstm_cell_21/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2
while/lstm_cell_21/mul_1ª
while/lstm_cell_21/add_1AddV2while/lstm_cell_21/mul:z:0while/lstm_cell_21/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2
while/lstm_cell_21/add_1
while/lstm_cell_21/Sigmoid_2Sigmoid!while/lstm_cell_21/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2
while/lstm_cell_21/Sigmoid_2
while/lstm_cell_21/Relu_1Reluwhile/lstm_cell_21/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2
while/lstm_cell_21/Relu_1¹
while/lstm_cell_21/mul_2Mul while/lstm_cell_21/Sigmoid_2:y:0'while/lstm_cell_21/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2
while/lstm_cell_21/mul_2à
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_21/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_21/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_21/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2
while/Identity_5Þ

while/NoOpNoOp*^while/lstm_cell_21/BiasAdd/ReadVariableOp)^while/lstm_cell_21/MatMul/ReadVariableOp+^while/lstm_cell_21/MatMul_1/ReadVariableOp*"
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
2while_lstm_cell_21_biasadd_readvariableop_resource4while_lstm_cell_21_biasadd_readvariableop_resource_0"l
3while_lstm_cell_21_matmul_1_readvariableop_resource5while_lstm_cell_21_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_21_matmul_readvariableop_resource3while_lstm_cell_21_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿÅ:ÿÿÿÿÿÿÿÿÿÅ: : : : : 2V
)while/lstm_cell_21/BiasAdd/ReadVariableOp)while/lstm_cell_21/BiasAdd/ReadVariableOp2T
(while/lstm_cell_21/MatMul/ReadVariableOp(while/lstm_cell_21/MatMul/ReadVariableOp2X
*while/lstm_cell_21/MatMul_1/ReadVariableOp*while/lstm_cell_21/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ:

_output_shapes
: :

_output_shapes
: 
Ê
ú
/__inference_lstm_cell_21_layer_call_fn_39105644

inputs
states_0
states_1
unknown:


	unknown_0:
Å

	unknown_1:	
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
<:ÿÿÿÿÿÿÿÿÿÅ:ÿÿÿÿÿÿÿÿÿÅ:ÿÿÿÿÿÿÿÿÿÅ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_lstm_cell_21_layer_call_and_return_conditional_losses_391019762
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2

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
B:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿÅ:ÿÿÿÿÿÿÿÿÿÅ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ
"
_user_specified_name
states/0:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ
"
_user_specified_name
states/1
°?
Ô
while_body_39104578
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_20_matmul_readvariableop_resource_0:	]I
5while_lstm_cell_20_matmul_1_readvariableop_resource_0:
C
4while_lstm_cell_20_biasadd_readvariableop_resource_0:	
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_20_matmul_readvariableop_resource:	]G
3while_lstm_cell_20_matmul_1_readvariableop_resource:
A
2while_lstm_cell_20_biasadd_readvariableop_resource:	¢)while/lstm_cell_20/BiasAdd/ReadVariableOp¢(while/lstm_cell_20/MatMul/ReadVariableOp¢*while/lstm_cell_20/MatMul_1/ReadVariableOpÃ
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
(while/lstm_cell_20/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_20_matmul_readvariableop_resource_0*
_output_shapes
:	]*
dtype02*
(while/lstm_cell_20/MatMul/ReadVariableOp×
while/lstm_cell_20/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_20/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_20/MatMulÐ
*while/lstm_cell_20/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_20_matmul_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype02,
*while/lstm_cell_20/MatMul_1/ReadVariableOpÀ
while/lstm_cell_20/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_20/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_20/MatMul_1¸
while/lstm_cell_20/addAddV2#while/lstm_cell_20/MatMul:product:0%while/lstm_cell_20/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_20/addÈ
)while/lstm_cell_20/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_20_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02+
)while/lstm_cell_20/BiasAdd/ReadVariableOpÅ
while/lstm_cell_20/BiasAddBiasAddwhile/lstm_cell_20/add:z:01while/lstm_cell_20/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_20/BiasAdd
"while/lstm_cell_20/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_20/split/split_dim
while/lstm_cell_20/splitSplit+while/lstm_cell_20/split/split_dim:output:0#while/lstm_cell_20/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2
while/lstm_cell_20/split
while/lstm_cell_20/SigmoidSigmoid!while/lstm_cell_20/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_20/Sigmoid
while/lstm_cell_20/Sigmoid_1Sigmoid!while/lstm_cell_20/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_20/Sigmoid_1¡
while/lstm_cell_20/mulMul while/lstm_cell_20/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_20/mul
while/lstm_cell_20/ReluRelu!while/lstm_cell_20/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_20/Reluµ
while/lstm_cell_20/mul_1Mulwhile/lstm_cell_20/Sigmoid:y:0%while/lstm_cell_20/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_20/mul_1ª
while/lstm_cell_20/add_1AddV2while/lstm_cell_20/mul:z:0while/lstm_cell_20/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_20/add_1
while/lstm_cell_20/Sigmoid_2Sigmoid!while/lstm_cell_20/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_20/Sigmoid_2
while/lstm_cell_20/Relu_1Reluwhile/lstm_cell_20/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_20/Relu_1¹
while/lstm_cell_20/mul_2Mul while/lstm_cell_20/Sigmoid_2:y:0'while/lstm_cell_20/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_20/mul_2à
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_20/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_20/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_20/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/Identity_5Þ

while/NoOpNoOp*^while/lstm_cell_20/BiasAdd/ReadVariableOp)^while/lstm_cell_20/MatMul/ReadVariableOp+^while/lstm_cell_20/MatMul_1/ReadVariableOp*"
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
2while_lstm_cell_20_biasadd_readvariableop_resource4while_lstm_cell_20_biasadd_readvariableop_resource_0"l
3while_lstm_cell_20_matmul_1_readvariableop_resource5while_lstm_cell_20_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_20_matmul_readvariableop_resource3while_lstm_cell_20_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2V
)while/lstm_cell_20/BiasAdd/ReadVariableOp)while/lstm_cell_20/BiasAdd/ReadVariableOp2T
(while/lstm_cell_20/MatMul/ReadVariableOp(while/lstm_cell_20/MatMul/ReadVariableOp2X
*while/lstm_cell_20/MatMul_1/ReadVariableOp*while/lstm_cell_20/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
Ô

í
lstm_21_while_cond_39103896,
(lstm_21_while_lstm_21_while_loop_counter2
.lstm_21_while_lstm_21_while_maximum_iterations
lstm_21_while_placeholder
lstm_21_while_placeholder_1
lstm_21_while_placeholder_2
lstm_21_while_placeholder_3.
*lstm_21_while_less_lstm_21_strided_slice_1F
Blstm_21_while_lstm_21_while_cond_39103896___redundant_placeholder0F
Blstm_21_while_lstm_21_while_cond_39103896___redundant_placeholder1F
Blstm_21_while_lstm_21_while_cond_39103896___redundant_placeholder2F
Blstm_21_while_lstm_21_while_cond_39103896___redundant_placeholder3
lstm_21_while_identity

lstm_21/while/LessLesslstm_21_while_placeholder*lstm_21_while_less_lstm_21_strided_slice_1*
T0*
_output_shapes
: 2
lstm_21/while/Lessu
lstm_21/while/IdentityIdentitylstm_21/while/Less:z:0*
T0
*
_output_shapes
: 2
lstm_21/while/Identity"9
lstm_21_while_identitylstm_21/while/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :ÿÿÿÿÿÿÿÿÿÅ:ÿÿÿÿÿÿÿÿÿÅ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ:

_output_shapes
: :

_output_shapes
:

f
H__inference_dropout_21_layer_call_and_return_conditional_losses_39102720

inputs

identity_1_
IdentityIdentityinputs*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2

Identityn

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿÅ:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ
 
_user_specified_nameinputs
µ
»
K__inference_sequential_10_layer_call_and_return_conditional_losses_39103229

inputs#
lstm_20_39103207:	]$
lstm_20_39103209:

lstm_20_39103211:	$
lstm_21_39103215:

$
lstm_21_39103217:
Å

lstm_21_39103219:	
$
dense_10_39103223:	Å
dense_10_39103225:
identity¢ dense_10/StatefulPartitionedCall¢"dropout_20/StatefulPartitionedCall¢"dropout_21/StatefulPartitionedCall¢lstm_20/StatefulPartitionedCall¢lstm_21/StatefulPartitionedCall®
lstm_20/StatefulPartitionedCallStatefulPartitionedCallinputslstm_20_39103207lstm_20_39103209lstm_20_39103211*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_lstm_20_layer_call_and_return_conditional_losses_391031722!
lstm_20/StatefulPartitionedCall
"dropout_20/StatefulPartitionedCallStatefulPartitionedCall(lstm_20/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_dropout_20_layer_call_and_return_conditional_losses_391030052$
"dropout_20/StatefulPartitionedCallÓ
lstm_21/StatefulPartitionedCallStatefulPartitionedCall+dropout_20/StatefulPartitionedCall:output:0lstm_21_39103215lstm_21_39103217lstm_21_39103219*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_lstm_21_layer_call_and_return_conditional_losses_391029762!
lstm_21/StatefulPartitionedCallÀ
"dropout_21/StatefulPartitionedCallStatefulPartitionedCall(lstm_21/StatefulPartitionedCall:output:0#^dropout_20/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_dropout_21_layer_call_and_return_conditional_losses_391028092$
"dropout_21/StatefulPartitionedCallÃ
 dense_10/StatefulPartitionedCallStatefulPartitionedCall+dropout_21/StatefulPartitionedCall:output:0dense_10_39103223dense_10_39103225*
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
F__inference_dense_10_layer_call_and_return_conditional_losses_391027532"
 dense_10/StatefulPartitionedCall
IdentityIdentity)dense_10/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityÿ
NoOpNoOp!^dense_10/StatefulPartitionedCall#^dropout_20/StatefulPartitionedCall#^dropout_21/StatefulPartitionedCall ^lstm_20/StatefulPartitionedCall ^lstm_21/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ]: : : : : : : : 2D
 dense_10/StatefulPartitionedCall dense_10/StatefulPartitionedCall2H
"dropout_20/StatefulPartitionedCall"dropout_20/StatefulPartitionedCall2H
"dropout_21/StatefulPartitionedCall"dropout_21/StatefulPartitionedCall2B
lstm_20/StatefulPartitionedCalllstm_20/StatefulPartitionedCall2B
lstm_21/StatefulPartitionedCalllstm_21/StatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
 
_user_specified_nameinputs


+__inference_dense_10_layer_call_fn_39105448

inputs
unknown:	Å
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
F__inference_dense_10_layer_call_and_return_conditional_losses_391027532
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
:ÿÿÿÿÿÿÿÿÿÅ: : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ
 
_user_specified_nameinputs
&
õ
while_body_39101844
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_01
while_lstm_cell_21_39101868_0:

1
while_lstm_cell_21_39101870_0:
Å
,
while_lstm_cell_21_39101872_0:	

while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor/
while_lstm_cell_21_39101868:

/
while_lstm_cell_21_39101870:
Å
*
while_lstm_cell_21_39101872:	
¢*while/lstm_cell_21/StatefulPartitionedCallÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ  29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÔ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemì
*while/lstm_cell_21/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_21_39101868_0while_lstm_cell_21_39101870_0while_lstm_cell_21_39101872_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿÅ:ÿÿÿÿÿÿÿÿÿÅ:ÿÿÿÿÿÿÿÿÿÅ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_lstm_cell_21_layer_call_and_return_conditional_losses_391018302,
*while/lstm_cell_21/StatefulPartitionedCall÷
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder3while/lstm_cell_21/StatefulPartitionedCall:output:0*
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
while/Identity_4Identity3while/lstm_cell_21/StatefulPartitionedCall:output:1^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2
while/Identity_4¥
while/Identity_5Identity3while/lstm_cell_21/StatefulPartitionedCall:output:2^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2
while/Identity_5

while/NoOpNoOp+^while/lstm_cell_21/StatefulPartitionedCall*"
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
while_lstm_cell_21_39101868while_lstm_cell_21_39101868_0"<
while_lstm_cell_21_39101870while_lstm_cell_21_39101870_0"<
while_lstm_cell_21_39101872while_lstm_cell_21_39101872_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿÅ:ÿÿÿÿÿÿÿÿÿÅ: : : : : 2X
*while/lstm_cell_21/StatefulPartitionedCall*while/lstm_cell_21/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ:

_output_shapes
: :

_output_shapes
: 
¹
¹
*__inference_lstm_21_layer_call_fn_39105370

inputs
unknown:


	unknown_0:
Å

	unknown_1:	
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
:ÿÿÿÿÿÿÿÿÿÅ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_lstm_21_layer_call_and_return_conditional_losses_391027072
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:ÿÿÿÿÿÿÿÿÿ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ã
Í
while_cond_39104124
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_39104124___redundant_placeholder06
2while_while_cond_39104124___redundant_placeholder16
2while_while_cond_39104124___redundant_placeholder26
2while_while_cond_39104124___redundant_placeholder3
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
B: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
\

E__inference_lstm_20_layer_call_and_return_conditional_losses_39103172

inputs>
+lstm_cell_20_matmul_readvariableop_resource:	]A
-lstm_cell_20_matmul_1_readvariableop_resource:
;
,lstm_cell_20_biasadd_readvariableop_resource:	
identity¢#lstm_cell_20/BiasAdd/ReadVariableOp¢"lstm_cell_20/MatMul/ReadVariableOp¢$lstm_cell_20/MatMul_1/ReadVariableOp¢whileD
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
B :2
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
B :2
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
:ÿÿÿÿÿÿÿÿÿ2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :2
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
B :2
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
:ÿÿÿÿÿÿÿÿÿ2	
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
"lstm_cell_20/MatMul/ReadVariableOpReadVariableOp+lstm_cell_20_matmul_readvariableop_resource*
_output_shapes
:	]*
dtype02$
"lstm_cell_20/MatMul/ReadVariableOp­
lstm_cell_20/MatMulMatMulstrided_slice_2:output:0*lstm_cell_20/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_20/MatMul¼
$lstm_cell_20/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_20_matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype02&
$lstm_cell_20/MatMul_1/ReadVariableOp©
lstm_cell_20/MatMul_1MatMulzeros:output:0,lstm_cell_20/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_20/MatMul_1 
lstm_cell_20/addAddV2lstm_cell_20/MatMul:product:0lstm_cell_20/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_20/add´
#lstm_cell_20/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_20_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02%
#lstm_cell_20/BiasAdd/ReadVariableOp­
lstm_cell_20/BiasAddBiasAddlstm_cell_20/add:z:0+lstm_cell_20/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_20/BiasAdd~
lstm_cell_20/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_20/split/split_dim÷
lstm_cell_20/splitSplit%lstm_cell_20/split/split_dim:output:0lstm_cell_20/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2
lstm_cell_20/split
lstm_cell_20/SigmoidSigmoidlstm_cell_20/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_20/Sigmoid
lstm_cell_20/Sigmoid_1Sigmoidlstm_cell_20/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_20/Sigmoid_1
lstm_cell_20/mulMullstm_cell_20/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_20/mul~
lstm_cell_20/ReluRelulstm_cell_20/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_20/Relu
lstm_cell_20/mul_1Mullstm_cell_20/Sigmoid:y:0lstm_cell_20/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_20/mul_1
lstm_cell_20/add_1AddV2lstm_cell_20/mul:z:0lstm_cell_20/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_20/add_1
lstm_cell_20/Sigmoid_2Sigmoidlstm_cell_20/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_20/Sigmoid_2}
lstm_cell_20/Relu_1Relulstm_cell_20/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_20/Relu_1¡
lstm_cell_20/mul_2Mullstm_cell_20/Sigmoid_2:y:0!lstm_cell_20/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_20/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ  2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_20_matmul_readvariableop_resource-lstm_cell_20_matmul_1_readvariableop_resource,lstm_cell_20_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_39103088*
condR
while_cond_39103087*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ  22
0TensorArrayV2Stack/TensorListStack/element_shapeé
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
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
:ÿÿÿÿÿÿÿÿÿ*
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
:ÿÿÿÿÿÿÿÿÿ2
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
:ÿÿÿÿÿÿÿÿÿ2

IdentityÈ
NoOpNoOp$^lstm_cell_20/BiasAdd/ReadVariableOp#^lstm_cell_20/MatMul/ReadVariableOp%^lstm_cell_20/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ]: : : 2J
#lstm_cell_20/BiasAdd/ReadVariableOp#lstm_cell_20/BiasAdd/ReadVariableOp2H
"lstm_cell_20/MatMul/ReadVariableOp"lstm_cell_20/MatMul/ReadVariableOp2L
$lstm_cell_20/MatMul_1/ReadVariableOp$lstm_cell_20/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
 
_user_specified_nameinputs
ã
Í
while_cond_39102891
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_39102891___redundant_placeholder06
2while_while_cond_39102891___redundant_placeholder16
2while_while_cond_39102891___redundant_placeholder26
2while_while_cond_39102891___redundant_placeholder3
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
B: : : : :ÿÿÿÿÿÿÿÿÿÅ:ÿÿÿÿÿÿÿÿÿÅ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ:

_output_shapes
: :

_output_shapes
:
É\
¡
E__inference_lstm_21_layer_call_and_return_conditional_losses_39104884
inputs_0?
+lstm_cell_21_matmul_readvariableop_resource:

A
-lstm_cell_21_matmul_1_readvariableop_resource:
Å
;
,lstm_cell_21_biasadd_readvariableop_resource:	

identity¢#lstm_cell_21/BiasAdd/ReadVariableOp¢"lstm_cell_21/MatMul/ReadVariableOp¢$lstm_cell_21/MatMul_1/ReadVariableOp¢whileF
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
B :Å2
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
B :Å2
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
:ÿÿÿÿÿÿÿÿÿÅ2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :Å2
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
B :Å2
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
:ÿÿÿÿÿÿÿÿÿÅ2	
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
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
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
valueB"ÿÿÿÿ  27
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
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
strided_slice_2¶
"lstm_cell_21/MatMul/ReadVariableOpReadVariableOp+lstm_cell_21_matmul_readvariableop_resource* 
_output_shapes
:

*
dtype02$
"lstm_cell_21/MatMul/ReadVariableOp­
lstm_cell_21/MatMulMatMulstrided_slice_2:output:0*lstm_cell_21/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
lstm_cell_21/MatMul¼
$lstm_cell_21/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_21_matmul_1_readvariableop_resource* 
_output_shapes
:
Å
*
dtype02&
$lstm_cell_21/MatMul_1/ReadVariableOp©
lstm_cell_21/MatMul_1MatMulzeros:output:0,lstm_cell_21/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
lstm_cell_21/MatMul_1 
lstm_cell_21/addAddV2lstm_cell_21/MatMul:product:0lstm_cell_21/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
lstm_cell_21/add´
#lstm_cell_21/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_21_biasadd_readvariableop_resource*
_output_shapes	
:
*
dtype02%
#lstm_cell_21/BiasAdd/ReadVariableOp­
lstm_cell_21/BiasAddBiasAddlstm_cell_21/add:z:0+lstm_cell_21/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
lstm_cell_21/BiasAdd~
lstm_cell_21/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_21/split/split_dim÷
lstm_cell_21/splitSplit%lstm_cell_21/split/split_dim:output:0lstm_cell_21/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿÅ:ÿÿÿÿÿÿÿÿÿÅ:ÿÿÿÿÿÿÿÿÿÅ:ÿÿÿÿÿÿÿÿÿÅ*
	num_split2
lstm_cell_21/split
lstm_cell_21/SigmoidSigmoidlstm_cell_21/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2
lstm_cell_21/Sigmoid
lstm_cell_21/Sigmoid_1Sigmoidlstm_cell_21/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2
lstm_cell_21/Sigmoid_1
lstm_cell_21/mulMullstm_cell_21/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2
lstm_cell_21/mul~
lstm_cell_21/ReluRelulstm_cell_21/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2
lstm_cell_21/Relu
lstm_cell_21/mul_1Mullstm_cell_21/Sigmoid:y:0lstm_cell_21/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2
lstm_cell_21/mul_1
lstm_cell_21/add_1AddV2lstm_cell_21/mul:z:0lstm_cell_21/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2
lstm_cell_21/add_1
lstm_cell_21/Sigmoid_2Sigmoidlstm_cell_21/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2
lstm_cell_21/Sigmoid_2}
lstm_cell_21/Relu_1Relulstm_cell_21/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2
lstm_cell_21/Relu_1¡
lstm_cell_21/mul_2Mullstm_cell_21/Sigmoid_2:y:0!lstm_cell_21/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2
lstm_cell_21/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿE  2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_21_matmul_readvariableop_resource-lstm_cell_21_matmul_1_readvariableop_resource,lstm_cell_21_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿÅ:ÿÿÿÿÿÿÿÿÿÅ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_39104800*
condR
while_cond_39104799*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿÅ:ÿÿÿÿÿÿÿÿÿÅ: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿE  22
0TensorArrayV2Stack/TensorListStack/element_shapeò
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÅ*
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
:ÿÿÿÿÿÿÿÿÿÅ*
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
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÅ2
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
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÅ2

IdentityÈ
NoOpNoOp$^lstm_cell_21/BiasAdd/ReadVariableOp#^lstm_cell_21/MatMul/ReadVariableOp%^lstm_cell_21/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 2J
#lstm_cell_21/BiasAdd/ReadVariableOp#lstm_cell_21/BiasAdd/ReadVariableOp2H
"lstm_cell_21/MatMul/ReadVariableOp"lstm_cell_21/MatMul/ReadVariableOp2L
$lstm_cell_21/MatMul_1/ReadVariableOp$lstm_cell_21/MatMul_1/ReadVariableOp2
whilewhile:_ [
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0
åJ
Ô

lstm_20_while_body_39103415,
(lstm_20_while_lstm_20_while_loop_counter2
.lstm_20_while_lstm_20_while_maximum_iterations
lstm_20_while_placeholder
lstm_20_while_placeholder_1
lstm_20_while_placeholder_2
lstm_20_while_placeholder_3+
'lstm_20_while_lstm_20_strided_slice_1_0g
clstm_20_while_tensorarrayv2read_tensorlistgetitem_lstm_20_tensorarrayunstack_tensorlistfromtensor_0N
;lstm_20_while_lstm_cell_20_matmul_readvariableop_resource_0:	]Q
=lstm_20_while_lstm_cell_20_matmul_1_readvariableop_resource_0:
K
<lstm_20_while_lstm_cell_20_biasadd_readvariableop_resource_0:	
lstm_20_while_identity
lstm_20_while_identity_1
lstm_20_while_identity_2
lstm_20_while_identity_3
lstm_20_while_identity_4
lstm_20_while_identity_5)
%lstm_20_while_lstm_20_strided_slice_1e
alstm_20_while_tensorarrayv2read_tensorlistgetitem_lstm_20_tensorarrayunstack_tensorlistfromtensorL
9lstm_20_while_lstm_cell_20_matmul_readvariableop_resource:	]O
;lstm_20_while_lstm_cell_20_matmul_1_readvariableop_resource:
I
:lstm_20_while_lstm_cell_20_biasadd_readvariableop_resource:	¢1lstm_20/while/lstm_cell_20/BiasAdd/ReadVariableOp¢0lstm_20/while/lstm_cell_20/MatMul/ReadVariableOp¢2lstm_20/while/lstm_cell_20/MatMul_1/ReadVariableOpÓ
?lstm_20/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ]   2A
?lstm_20/while/TensorArrayV2Read/TensorListGetItem/element_shape
1lstm_20/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemclstm_20_while_tensorarrayv2read_tensorlistgetitem_lstm_20_tensorarrayunstack_tensorlistfromtensor_0lstm_20_while_placeholderHlstm_20/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]*
element_dtype023
1lstm_20/while/TensorArrayV2Read/TensorListGetItemá
0lstm_20/while/lstm_cell_20/MatMul/ReadVariableOpReadVariableOp;lstm_20_while_lstm_cell_20_matmul_readvariableop_resource_0*
_output_shapes
:	]*
dtype022
0lstm_20/while/lstm_cell_20/MatMul/ReadVariableOp÷
!lstm_20/while/lstm_cell_20/MatMulMatMul8lstm_20/while/TensorArrayV2Read/TensorListGetItem:item:08lstm_20/while/lstm_cell_20/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!lstm_20/while/lstm_cell_20/MatMulè
2lstm_20/while/lstm_cell_20/MatMul_1/ReadVariableOpReadVariableOp=lstm_20_while_lstm_cell_20_matmul_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype024
2lstm_20/while/lstm_cell_20/MatMul_1/ReadVariableOpà
#lstm_20/while/lstm_cell_20/MatMul_1MatMullstm_20_while_placeholder_2:lstm_20/while/lstm_cell_20/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2%
#lstm_20/while/lstm_cell_20/MatMul_1Ø
lstm_20/while/lstm_cell_20/addAddV2+lstm_20/while/lstm_cell_20/MatMul:product:0-lstm_20/while/lstm_cell_20/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
lstm_20/while/lstm_cell_20/addà
1lstm_20/while/lstm_cell_20/BiasAdd/ReadVariableOpReadVariableOp<lstm_20_while_lstm_cell_20_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype023
1lstm_20/while/lstm_cell_20/BiasAdd/ReadVariableOpå
"lstm_20/while/lstm_cell_20/BiasAddBiasAdd"lstm_20/while/lstm_cell_20/add:z:09lstm_20/while/lstm_cell_20/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2$
"lstm_20/while/lstm_cell_20/BiasAdd
*lstm_20/while/lstm_cell_20/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2,
*lstm_20/while/lstm_cell_20/split/split_dim¯
 lstm_20/while/lstm_cell_20/splitSplit3lstm_20/while/lstm_cell_20/split/split_dim:output:0+lstm_20/while/lstm_cell_20/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2"
 lstm_20/while/lstm_cell_20/split±
"lstm_20/while/lstm_cell_20/SigmoidSigmoid)lstm_20/while/lstm_cell_20/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2$
"lstm_20/while/lstm_cell_20/Sigmoidµ
$lstm_20/while/lstm_cell_20/Sigmoid_1Sigmoid)lstm_20/while/lstm_cell_20/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$lstm_20/while/lstm_cell_20/Sigmoid_1Á
lstm_20/while/lstm_cell_20/mulMul(lstm_20/while/lstm_cell_20/Sigmoid_1:y:0lstm_20_while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
lstm_20/while/lstm_cell_20/mul¨
lstm_20/while/lstm_cell_20/ReluRelu)lstm_20/while/lstm_cell_20/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
lstm_20/while/lstm_cell_20/ReluÕ
 lstm_20/while/lstm_cell_20/mul_1Mul&lstm_20/while/lstm_cell_20/Sigmoid:y:0-lstm_20/while/lstm_cell_20/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 lstm_20/while/lstm_cell_20/mul_1Ê
 lstm_20/while/lstm_cell_20/add_1AddV2"lstm_20/while/lstm_cell_20/mul:z:0$lstm_20/while/lstm_cell_20/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 lstm_20/while/lstm_cell_20/add_1µ
$lstm_20/while/lstm_cell_20/Sigmoid_2Sigmoid)lstm_20/while/lstm_cell_20/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$lstm_20/while/lstm_cell_20/Sigmoid_2§
!lstm_20/while/lstm_cell_20/Relu_1Relu$lstm_20/while/lstm_cell_20/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!lstm_20/while/lstm_cell_20/Relu_1Ù
 lstm_20/while/lstm_cell_20/mul_2Mul(lstm_20/while/lstm_cell_20/Sigmoid_2:y:0/lstm_20/while/lstm_cell_20/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 lstm_20/while/lstm_cell_20/mul_2
2lstm_20/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_20_while_placeholder_1lstm_20_while_placeholder$lstm_20/while/lstm_cell_20/mul_2:z:0*
_output_shapes
: *
element_dtype024
2lstm_20/while/TensorArrayV2Write/TensorListSetIteml
lstm_20/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_20/while/add/y
lstm_20/while/addAddV2lstm_20_while_placeholderlstm_20/while/add/y:output:0*
T0*
_output_shapes
: 2
lstm_20/while/addp
lstm_20/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_20/while/add_1/y
lstm_20/while/add_1AddV2(lstm_20_while_lstm_20_while_loop_counterlstm_20/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lstm_20/while/add_1
lstm_20/while/IdentityIdentitylstm_20/while/add_1:z:0^lstm_20/while/NoOp*
T0*
_output_shapes
: 2
lstm_20/while/Identity¦
lstm_20/while/Identity_1Identity.lstm_20_while_lstm_20_while_maximum_iterations^lstm_20/while/NoOp*
T0*
_output_shapes
: 2
lstm_20/while/Identity_1
lstm_20/while/Identity_2Identitylstm_20/while/add:z:0^lstm_20/while/NoOp*
T0*
_output_shapes
: 2
lstm_20/while/Identity_2º
lstm_20/while/Identity_3IdentityBlstm_20/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_20/while/NoOp*
T0*
_output_shapes
: 2
lstm_20/while/Identity_3®
lstm_20/while/Identity_4Identity$lstm_20/while/lstm_cell_20/mul_2:z:0^lstm_20/while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_20/while/Identity_4®
lstm_20/while/Identity_5Identity$lstm_20/while/lstm_cell_20/add_1:z:0^lstm_20/while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_20/while/Identity_5
lstm_20/while/NoOpNoOp2^lstm_20/while/lstm_cell_20/BiasAdd/ReadVariableOp1^lstm_20/while/lstm_cell_20/MatMul/ReadVariableOp3^lstm_20/while/lstm_cell_20/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
lstm_20/while/NoOp"9
lstm_20_while_identitylstm_20/while/Identity:output:0"=
lstm_20_while_identity_1!lstm_20/while/Identity_1:output:0"=
lstm_20_while_identity_2!lstm_20/while/Identity_2:output:0"=
lstm_20_while_identity_3!lstm_20/while/Identity_3:output:0"=
lstm_20_while_identity_4!lstm_20/while/Identity_4:output:0"=
lstm_20_while_identity_5!lstm_20/while/Identity_5:output:0"P
%lstm_20_while_lstm_20_strided_slice_1'lstm_20_while_lstm_20_strided_slice_1_0"z
:lstm_20_while_lstm_cell_20_biasadd_readvariableop_resource<lstm_20_while_lstm_cell_20_biasadd_readvariableop_resource_0"|
;lstm_20_while_lstm_cell_20_matmul_1_readvariableop_resource=lstm_20_while_lstm_cell_20_matmul_1_readvariableop_resource_0"x
9lstm_20_while_lstm_cell_20_matmul_readvariableop_resource;lstm_20_while_lstm_cell_20_matmul_readvariableop_resource_0"È
alstm_20_while_tensorarrayv2read_tensorlistgetitem_lstm_20_tensorarrayunstack_tensorlistfromtensorclstm_20_while_tensorarrayv2read_tensorlistgetitem_lstm_20_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2f
1lstm_20/while/lstm_cell_20/BiasAdd/ReadVariableOp1lstm_20/while/lstm_cell_20/BiasAdd/ReadVariableOp2d
0lstm_20/while/lstm_cell_20/MatMul/ReadVariableOp0lstm_20/while/lstm_cell_20/MatMul/ReadVariableOp2h
2lstm_20/while/lstm_cell_20/MatMul_1/ReadVariableOp2lstm_20/while/lstm_cell_20/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 


J__inference_lstm_cell_20_layer_call_and_return_conditional_losses_39101346

inputs

states
states_11
matmul_readvariableop_resource:	]4
 matmul_1_readvariableop_resource:
.
biasadd_readvariableop_resource:	
identity

identity_1

identity_2¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	]*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
add
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
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
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2
split`
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
Sigmoidd
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	Sigmoid_1]
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mulW
ReluRelusplit:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relui
mul_1MulSigmoid:y:0Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mul_1^
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
add_1d
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	Sigmoid_2V
Relu_1Relu	add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu_1m
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mul_2e
IdentityIdentity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityi

Identity_1Identity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1i

Identity_2Identity	add_1:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

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
A:ÿÿÿÿÿÿÿÿÿ]:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
 
_user_specified_nameinputs:PL
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_namestates:PL
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_namestates
Ôù

K__inference_sequential_10_layer_call_and_return_conditional_losses_39103675

inputsF
3lstm_20_lstm_cell_20_matmul_readvariableop_resource:	]I
5lstm_20_lstm_cell_20_matmul_1_readvariableop_resource:
C
4lstm_20_lstm_cell_20_biasadd_readvariableop_resource:	G
3lstm_21_lstm_cell_21_matmul_readvariableop_resource:

I
5lstm_21_lstm_cell_21_matmul_1_readvariableop_resource:
Å
C
4lstm_21_lstm_cell_21_biasadd_readvariableop_resource:	
=
*dense_10_tensordot_readvariableop_resource:	Å6
(dense_10_biasadd_readvariableop_resource:
identity¢dense_10/BiasAdd/ReadVariableOp¢!dense_10/Tensordot/ReadVariableOp¢+lstm_20/lstm_cell_20/BiasAdd/ReadVariableOp¢*lstm_20/lstm_cell_20/MatMul/ReadVariableOp¢,lstm_20/lstm_cell_20/MatMul_1/ReadVariableOp¢lstm_20/while¢+lstm_21/lstm_cell_21/BiasAdd/ReadVariableOp¢*lstm_21/lstm_cell_21/MatMul/ReadVariableOp¢,lstm_21/lstm_cell_21/MatMul_1/ReadVariableOp¢lstm_21/whileT
lstm_20/ShapeShapeinputs*
T0*
_output_shapes
:2
lstm_20/Shape
lstm_20/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_20/strided_slice/stack
lstm_20/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_20/strided_slice/stack_1
lstm_20/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_20/strided_slice/stack_2
lstm_20/strided_sliceStridedSlicelstm_20/Shape:output:0$lstm_20/strided_slice/stack:output:0&lstm_20/strided_slice/stack_1:output:0&lstm_20/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_20/strided_slicem
lstm_20/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :2
lstm_20/zeros/mul/y
lstm_20/zeros/mulMullstm_20/strided_slice:output:0lstm_20/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_20/zeros/mulo
lstm_20/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
lstm_20/zeros/Less/y
lstm_20/zeros/LessLesslstm_20/zeros/mul:z:0lstm_20/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_20/zeros/Lesss
lstm_20/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :2
lstm_20/zeros/packed/1£
lstm_20/zeros/packedPacklstm_20/strided_slice:output:0lstm_20/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_20/zeros/packedo
lstm_20/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_20/zeros/Const
lstm_20/zerosFilllstm_20/zeros/packed:output:0lstm_20/zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_20/zerosq
lstm_20/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :2
lstm_20/zeros_1/mul/y
lstm_20/zeros_1/mulMullstm_20/strided_slice:output:0lstm_20/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_20/zeros_1/muls
lstm_20/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
lstm_20/zeros_1/Less/y
lstm_20/zeros_1/LessLesslstm_20/zeros_1/mul:z:0lstm_20/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_20/zeros_1/Lessw
lstm_20/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :2
lstm_20/zeros_1/packed/1©
lstm_20/zeros_1/packedPacklstm_20/strided_slice:output:0!lstm_20/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_20/zeros_1/packeds
lstm_20/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_20/zeros_1/Const
lstm_20/zeros_1Filllstm_20/zeros_1/packed:output:0lstm_20/zeros_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_20/zeros_1
lstm_20/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_20/transpose/perm
lstm_20/transpose	Transposeinputslstm_20/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]2
lstm_20/transposeg
lstm_20/Shape_1Shapelstm_20/transpose:y:0*
T0*
_output_shapes
:2
lstm_20/Shape_1
lstm_20/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_20/strided_slice_1/stack
lstm_20/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_20/strided_slice_1/stack_1
lstm_20/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_20/strided_slice_1/stack_2
lstm_20/strided_slice_1StridedSlicelstm_20/Shape_1:output:0&lstm_20/strided_slice_1/stack:output:0(lstm_20/strided_slice_1/stack_1:output:0(lstm_20/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_20/strided_slice_1
#lstm_20/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2%
#lstm_20/TensorArrayV2/element_shapeÒ
lstm_20/TensorArrayV2TensorListReserve,lstm_20/TensorArrayV2/element_shape:output:0 lstm_20/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_20/TensorArrayV2Ï
=lstm_20/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ]   2?
=lstm_20/TensorArrayUnstack/TensorListFromTensor/element_shape
/lstm_20/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_20/transpose:y:0Flstm_20/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type021
/lstm_20/TensorArrayUnstack/TensorListFromTensor
lstm_20/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_20/strided_slice_2/stack
lstm_20/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_20/strided_slice_2/stack_1
lstm_20/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_20/strided_slice_2/stack_2¬
lstm_20/strided_slice_2StridedSlicelstm_20/transpose:y:0&lstm_20/strided_slice_2/stack:output:0(lstm_20/strided_slice_2/stack_1:output:0(lstm_20/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]*
shrink_axis_mask2
lstm_20/strided_slice_2Í
*lstm_20/lstm_cell_20/MatMul/ReadVariableOpReadVariableOp3lstm_20_lstm_cell_20_matmul_readvariableop_resource*
_output_shapes
:	]*
dtype02,
*lstm_20/lstm_cell_20/MatMul/ReadVariableOpÍ
lstm_20/lstm_cell_20/MatMulMatMul lstm_20/strided_slice_2:output:02lstm_20/lstm_cell_20/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_20/lstm_cell_20/MatMulÔ
,lstm_20/lstm_cell_20/MatMul_1/ReadVariableOpReadVariableOp5lstm_20_lstm_cell_20_matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype02.
,lstm_20/lstm_cell_20/MatMul_1/ReadVariableOpÉ
lstm_20/lstm_cell_20/MatMul_1MatMullstm_20/zeros:output:04lstm_20/lstm_cell_20/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_20/lstm_cell_20/MatMul_1À
lstm_20/lstm_cell_20/addAddV2%lstm_20/lstm_cell_20/MatMul:product:0'lstm_20/lstm_cell_20/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_20/lstm_cell_20/addÌ
+lstm_20/lstm_cell_20/BiasAdd/ReadVariableOpReadVariableOp4lstm_20_lstm_cell_20_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02-
+lstm_20/lstm_cell_20/BiasAdd/ReadVariableOpÍ
lstm_20/lstm_cell_20/BiasAddBiasAddlstm_20/lstm_cell_20/add:z:03lstm_20/lstm_cell_20/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_20/lstm_cell_20/BiasAdd
$lstm_20/lstm_cell_20/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2&
$lstm_20/lstm_cell_20/split/split_dim
lstm_20/lstm_cell_20/splitSplit-lstm_20/lstm_cell_20/split/split_dim:output:0%lstm_20/lstm_cell_20/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2
lstm_20/lstm_cell_20/split
lstm_20/lstm_cell_20/SigmoidSigmoid#lstm_20/lstm_cell_20/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_20/lstm_cell_20/Sigmoid£
lstm_20/lstm_cell_20/Sigmoid_1Sigmoid#lstm_20/lstm_cell_20/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
lstm_20/lstm_cell_20/Sigmoid_1¬
lstm_20/lstm_cell_20/mulMul"lstm_20/lstm_cell_20/Sigmoid_1:y:0lstm_20/zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_20/lstm_cell_20/mul
lstm_20/lstm_cell_20/ReluRelu#lstm_20/lstm_cell_20/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_20/lstm_cell_20/Relu½
lstm_20/lstm_cell_20/mul_1Mul lstm_20/lstm_cell_20/Sigmoid:y:0'lstm_20/lstm_cell_20/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_20/lstm_cell_20/mul_1²
lstm_20/lstm_cell_20/add_1AddV2lstm_20/lstm_cell_20/mul:z:0lstm_20/lstm_cell_20/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_20/lstm_cell_20/add_1£
lstm_20/lstm_cell_20/Sigmoid_2Sigmoid#lstm_20/lstm_cell_20/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
lstm_20/lstm_cell_20/Sigmoid_2
lstm_20/lstm_cell_20/Relu_1Relulstm_20/lstm_cell_20/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_20/lstm_cell_20/Relu_1Á
lstm_20/lstm_cell_20/mul_2Mul"lstm_20/lstm_cell_20/Sigmoid_2:y:0)lstm_20/lstm_cell_20/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_20/lstm_cell_20/mul_2
%lstm_20/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ  2'
%lstm_20/TensorArrayV2_1/element_shapeØ
lstm_20/TensorArrayV2_1TensorListReserve.lstm_20/TensorArrayV2_1/element_shape:output:0 lstm_20/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_20/TensorArrayV2_1^
lstm_20/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_20/time
 lstm_20/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2"
 lstm_20/while/maximum_iterationsz
lstm_20/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_20/while/loop_counter
lstm_20/whileWhile#lstm_20/while/loop_counter:output:0)lstm_20/while/maximum_iterations:output:0lstm_20/time:output:0 lstm_20/TensorArrayV2_1:handle:0lstm_20/zeros:output:0lstm_20/zeros_1:output:0 lstm_20/strided_slice_1:output:0?lstm_20/TensorArrayUnstack/TensorListFromTensor:output_handle:03lstm_20_lstm_cell_20_matmul_readvariableop_resource5lstm_20_lstm_cell_20_matmul_1_readvariableop_resource4lstm_20_lstm_cell_20_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *'
bodyR
lstm_20_while_body_39103415*'
condR
lstm_20_while_cond_39103414*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 2
lstm_20/whileÅ
8lstm_20/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ  2:
8lstm_20/TensorArrayV2Stack/TensorListStack/element_shape
*lstm_20/TensorArrayV2Stack/TensorListStackTensorListStacklstm_20/while:output:3Alstm_20/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype02,
*lstm_20/TensorArrayV2Stack/TensorListStack
lstm_20/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
lstm_20/strided_slice_3/stack
lstm_20/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2!
lstm_20/strided_slice_3/stack_1
lstm_20/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_20/strided_slice_3/stack_2Ë
lstm_20/strided_slice_3StridedSlice3lstm_20/TensorArrayV2Stack/TensorListStack:tensor:0&lstm_20/strided_slice_3/stack:output:0(lstm_20/strided_slice_3/stack_1:output:0(lstm_20/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
lstm_20/strided_slice_3
lstm_20/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_20/transpose_1/permÆ
lstm_20/transpose_1	Transpose3lstm_20/TensorArrayV2Stack/TensorListStack:tensor:0!lstm_20/transpose_1/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_20/transpose_1v
lstm_20/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_20/runtime
dropout_20/IdentityIdentitylstm_20/transpose_1:y:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout_20/Identityj
lstm_21/ShapeShapedropout_20/Identity:output:0*
T0*
_output_shapes
:2
lstm_21/Shape
lstm_21/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_21/strided_slice/stack
lstm_21/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_21/strided_slice/stack_1
lstm_21/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_21/strided_slice/stack_2
lstm_21/strided_sliceStridedSlicelstm_21/Shape:output:0$lstm_21/strided_slice/stack:output:0&lstm_21/strided_slice/stack_1:output:0&lstm_21/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_21/strided_slicem
lstm_21/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :Å2
lstm_21/zeros/mul/y
lstm_21/zeros/mulMullstm_21/strided_slice:output:0lstm_21/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_21/zeros/mulo
lstm_21/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
lstm_21/zeros/Less/y
lstm_21/zeros/LessLesslstm_21/zeros/mul:z:0lstm_21/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_21/zeros/Lesss
lstm_21/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :Å2
lstm_21/zeros/packed/1£
lstm_21/zeros/packedPacklstm_21/strided_slice:output:0lstm_21/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_21/zeros/packedo
lstm_21/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_21/zeros/Const
lstm_21/zerosFilllstm_21/zeros/packed:output:0lstm_21/zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2
lstm_21/zerosq
lstm_21/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :Å2
lstm_21/zeros_1/mul/y
lstm_21/zeros_1/mulMullstm_21/strided_slice:output:0lstm_21/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_21/zeros_1/muls
lstm_21/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
lstm_21/zeros_1/Less/y
lstm_21/zeros_1/LessLesslstm_21/zeros_1/mul:z:0lstm_21/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_21/zeros_1/Lessw
lstm_21/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :Å2
lstm_21/zeros_1/packed/1©
lstm_21/zeros_1/packedPacklstm_21/strided_slice:output:0!lstm_21/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_21/zeros_1/packeds
lstm_21/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_21/zeros_1/Const
lstm_21/zeros_1Filllstm_21/zeros_1/packed:output:0lstm_21/zeros_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2
lstm_21/zeros_1
lstm_21/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_21/transpose/perm©
lstm_21/transpose	Transposedropout_20/Identity:output:0lstm_21/transpose/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_21/transposeg
lstm_21/Shape_1Shapelstm_21/transpose:y:0*
T0*
_output_shapes
:2
lstm_21/Shape_1
lstm_21/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_21/strided_slice_1/stack
lstm_21/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_21/strided_slice_1/stack_1
lstm_21/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_21/strided_slice_1/stack_2
lstm_21/strided_slice_1StridedSlicelstm_21/Shape_1:output:0&lstm_21/strided_slice_1/stack:output:0(lstm_21/strided_slice_1/stack_1:output:0(lstm_21/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_21/strided_slice_1
#lstm_21/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2%
#lstm_21/TensorArrayV2/element_shapeÒ
lstm_21/TensorArrayV2TensorListReserve,lstm_21/TensorArrayV2/element_shape:output:0 lstm_21/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_21/TensorArrayV2Ï
=lstm_21/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ  2?
=lstm_21/TensorArrayUnstack/TensorListFromTensor/element_shape
/lstm_21/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_21/transpose:y:0Flstm_21/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type021
/lstm_21/TensorArrayUnstack/TensorListFromTensor
lstm_21/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_21/strided_slice_2/stack
lstm_21/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_21/strided_slice_2/stack_1
lstm_21/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_21/strided_slice_2/stack_2­
lstm_21/strided_slice_2StridedSlicelstm_21/transpose:y:0&lstm_21/strided_slice_2/stack:output:0(lstm_21/strided_slice_2/stack_1:output:0(lstm_21/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
lstm_21/strided_slice_2Î
*lstm_21/lstm_cell_21/MatMul/ReadVariableOpReadVariableOp3lstm_21_lstm_cell_21_matmul_readvariableop_resource* 
_output_shapes
:

*
dtype02,
*lstm_21/lstm_cell_21/MatMul/ReadVariableOpÍ
lstm_21/lstm_cell_21/MatMulMatMul lstm_21/strided_slice_2:output:02lstm_21/lstm_cell_21/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
lstm_21/lstm_cell_21/MatMulÔ
,lstm_21/lstm_cell_21/MatMul_1/ReadVariableOpReadVariableOp5lstm_21_lstm_cell_21_matmul_1_readvariableop_resource* 
_output_shapes
:
Å
*
dtype02.
,lstm_21/lstm_cell_21/MatMul_1/ReadVariableOpÉ
lstm_21/lstm_cell_21/MatMul_1MatMullstm_21/zeros:output:04lstm_21/lstm_cell_21/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
lstm_21/lstm_cell_21/MatMul_1À
lstm_21/lstm_cell_21/addAddV2%lstm_21/lstm_cell_21/MatMul:product:0'lstm_21/lstm_cell_21/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
lstm_21/lstm_cell_21/addÌ
+lstm_21/lstm_cell_21/BiasAdd/ReadVariableOpReadVariableOp4lstm_21_lstm_cell_21_biasadd_readvariableop_resource*
_output_shapes	
:
*
dtype02-
+lstm_21/lstm_cell_21/BiasAdd/ReadVariableOpÍ
lstm_21/lstm_cell_21/BiasAddBiasAddlstm_21/lstm_cell_21/add:z:03lstm_21/lstm_cell_21/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
lstm_21/lstm_cell_21/BiasAdd
$lstm_21/lstm_cell_21/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2&
$lstm_21/lstm_cell_21/split/split_dim
lstm_21/lstm_cell_21/splitSplit-lstm_21/lstm_cell_21/split/split_dim:output:0%lstm_21/lstm_cell_21/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿÅ:ÿÿÿÿÿÿÿÿÿÅ:ÿÿÿÿÿÿÿÿÿÅ:ÿÿÿÿÿÿÿÿÿÅ*
	num_split2
lstm_21/lstm_cell_21/split
lstm_21/lstm_cell_21/SigmoidSigmoid#lstm_21/lstm_cell_21/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2
lstm_21/lstm_cell_21/Sigmoid£
lstm_21/lstm_cell_21/Sigmoid_1Sigmoid#lstm_21/lstm_cell_21/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2 
lstm_21/lstm_cell_21/Sigmoid_1¬
lstm_21/lstm_cell_21/mulMul"lstm_21/lstm_cell_21/Sigmoid_1:y:0lstm_21/zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2
lstm_21/lstm_cell_21/mul
lstm_21/lstm_cell_21/ReluRelu#lstm_21/lstm_cell_21/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2
lstm_21/lstm_cell_21/Relu½
lstm_21/lstm_cell_21/mul_1Mul lstm_21/lstm_cell_21/Sigmoid:y:0'lstm_21/lstm_cell_21/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2
lstm_21/lstm_cell_21/mul_1²
lstm_21/lstm_cell_21/add_1AddV2lstm_21/lstm_cell_21/mul:z:0lstm_21/lstm_cell_21/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2
lstm_21/lstm_cell_21/add_1£
lstm_21/lstm_cell_21/Sigmoid_2Sigmoid#lstm_21/lstm_cell_21/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2 
lstm_21/lstm_cell_21/Sigmoid_2
lstm_21/lstm_cell_21/Relu_1Relulstm_21/lstm_cell_21/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2
lstm_21/lstm_cell_21/Relu_1Á
lstm_21/lstm_cell_21/mul_2Mul"lstm_21/lstm_cell_21/Sigmoid_2:y:0)lstm_21/lstm_cell_21/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2
lstm_21/lstm_cell_21/mul_2
%lstm_21/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿE  2'
%lstm_21/TensorArrayV2_1/element_shapeØ
lstm_21/TensorArrayV2_1TensorListReserve.lstm_21/TensorArrayV2_1/element_shape:output:0 lstm_21/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_21/TensorArrayV2_1^
lstm_21/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_21/time
 lstm_21/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2"
 lstm_21/while/maximum_iterationsz
lstm_21/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_21/while/loop_counter
lstm_21/whileWhile#lstm_21/while/loop_counter:output:0)lstm_21/while/maximum_iterations:output:0lstm_21/time:output:0 lstm_21/TensorArrayV2_1:handle:0lstm_21/zeros:output:0lstm_21/zeros_1:output:0 lstm_21/strided_slice_1:output:0?lstm_21/TensorArrayUnstack/TensorListFromTensor:output_handle:03lstm_21_lstm_cell_21_matmul_readvariableop_resource5lstm_21_lstm_cell_21_matmul_1_readvariableop_resource4lstm_21_lstm_cell_21_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿÅ:ÿÿÿÿÿÿÿÿÿÅ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *'
bodyR
lstm_21_while_body_39103563*'
condR
lstm_21_while_cond_39103562*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿÅ:ÿÿÿÿÿÿÿÿÿÅ: : : : : *
parallel_iterations 2
lstm_21/whileÅ
8lstm_21/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿE  2:
8lstm_21/TensorArrayV2Stack/TensorListStack/element_shape
*lstm_21/TensorArrayV2Stack/TensorListStackTensorListStacklstm_21/while:output:3Alstm_21/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ*
element_dtype02,
*lstm_21/TensorArrayV2Stack/TensorListStack
lstm_21/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
lstm_21/strided_slice_3/stack
lstm_21/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2!
lstm_21/strided_slice_3/stack_1
lstm_21/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_21/strided_slice_3/stack_2Ë
lstm_21/strided_slice_3StridedSlice3lstm_21/TensorArrayV2Stack/TensorListStack:tensor:0&lstm_21/strided_slice_3/stack:output:0(lstm_21/strided_slice_3/stack_1:output:0(lstm_21/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ*
shrink_axis_mask2
lstm_21/strided_slice_3
lstm_21/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_21/transpose_1/permÆ
lstm_21/transpose_1	Transpose3lstm_21/TensorArrayV2Stack/TensorListStack:tensor:0!lstm_21/transpose_1/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2
lstm_21/transpose_1v
lstm_21/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_21/runtime
dropout_21/IdentityIdentitylstm_21/transpose_1:y:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2
dropout_21/Identity²
!dense_10/Tensordot/ReadVariableOpReadVariableOp*dense_10_tensordot_readvariableop_resource*
_output_shapes
:	Å*
dtype02#
!dense_10/Tensordot/ReadVariableOp|
dense_10/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_10/Tensordot/axes
dense_10/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_10/Tensordot/free
dense_10/Tensordot/ShapeShapedropout_21/Identity:output:0*
T0*
_output_shapes
:2
dense_10/Tensordot/Shape
 dense_10/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_10/Tensordot/GatherV2/axisþ
dense_10/Tensordot/GatherV2GatherV2!dense_10/Tensordot/Shape:output:0 dense_10/Tensordot/free:output:0)dense_10/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_10/Tensordot/GatherV2
"dense_10/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"dense_10/Tensordot/GatherV2_1/axis
dense_10/Tensordot/GatherV2_1GatherV2!dense_10/Tensordot/Shape:output:0 dense_10/Tensordot/axes:output:0+dense_10/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_10/Tensordot/GatherV2_1~
dense_10/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_10/Tensordot/Const¤
dense_10/Tensordot/ProdProd$dense_10/Tensordot/GatherV2:output:0!dense_10/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_10/Tensordot/Prod
dense_10/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_10/Tensordot/Const_1¬
dense_10/Tensordot/Prod_1Prod&dense_10/Tensordot/GatherV2_1:output:0#dense_10/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_10/Tensordot/Prod_1
dense_10/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2 
dense_10/Tensordot/concat/axisÝ
dense_10/Tensordot/concatConcatV2 dense_10/Tensordot/free:output:0 dense_10/Tensordot/axes:output:0'dense_10/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_10/Tensordot/concat°
dense_10/Tensordot/stackPack dense_10/Tensordot/Prod:output:0"dense_10/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_10/Tensordot/stackÂ
dense_10/Tensordot/transpose	Transposedropout_21/Identity:output:0"dense_10/Tensordot/concat:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2
dense_10/Tensordot/transposeÃ
dense_10/Tensordot/ReshapeReshape dense_10/Tensordot/transpose:y:0!dense_10/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
dense_10/Tensordot/ReshapeÂ
dense_10/Tensordot/MatMulMatMul#dense_10/Tensordot/Reshape:output:0)dense_10/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_10/Tensordot/MatMul
dense_10/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
dense_10/Tensordot/Const_2
 dense_10/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_10/Tensordot/concat_1/axisê
dense_10/Tensordot/concat_1ConcatV2$dense_10/Tensordot/GatherV2:output:0#dense_10/Tensordot/Const_2:output:0)dense_10/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_10/Tensordot/concat_1´
dense_10/TensordotReshape#dense_10/Tensordot/MatMul:product:0$dense_10/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_10/Tensordot§
dense_10/BiasAdd/ReadVariableOpReadVariableOp(dense_10_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_10/BiasAdd/ReadVariableOp«
dense_10/BiasAddBiasAdddense_10/Tensordot:output:0'dense_10/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_10/BiasAdd
dense_10/SoftmaxSoftmaxdense_10/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_10/Softmaxy
IdentityIdentitydense_10/Softmax:softmax:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

IdentityÈ
NoOpNoOp ^dense_10/BiasAdd/ReadVariableOp"^dense_10/Tensordot/ReadVariableOp,^lstm_20/lstm_cell_20/BiasAdd/ReadVariableOp+^lstm_20/lstm_cell_20/MatMul/ReadVariableOp-^lstm_20/lstm_cell_20/MatMul_1/ReadVariableOp^lstm_20/while,^lstm_21/lstm_cell_21/BiasAdd/ReadVariableOp+^lstm_21/lstm_cell_21/MatMul/ReadVariableOp-^lstm_21/lstm_cell_21/MatMul_1/ReadVariableOp^lstm_21/while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ]: : : : : : : : 2B
dense_10/BiasAdd/ReadVariableOpdense_10/BiasAdd/ReadVariableOp2F
!dense_10/Tensordot/ReadVariableOp!dense_10/Tensordot/ReadVariableOp2Z
+lstm_20/lstm_cell_20/BiasAdd/ReadVariableOp+lstm_20/lstm_cell_20/BiasAdd/ReadVariableOp2X
*lstm_20/lstm_cell_20/MatMul/ReadVariableOp*lstm_20/lstm_cell_20/MatMul/ReadVariableOp2\
,lstm_20/lstm_cell_20/MatMul_1/ReadVariableOp,lstm_20/lstm_cell_20/MatMul_1/ReadVariableOp2
lstm_20/whilelstm_20/while2Z
+lstm_21/lstm_cell_21/BiasAdd/ReadVariableOp+lstm_21/lstm_cell_21/BiasAdd/ReadVariableOp2X
*lstm_21/lstm_cell_21/MatMul/ReadVariableOp*lstm_21/lstm_cell_21/MatMul/ReadVariableOp2\
,lstm_21/lstm_cell_21/MatMul_1/ReadVariableOp,lstm_21/lstm_cell_21/MatMul_1/ReadVariableOp2
lstm_21/whilelstm_21/while:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
 
_user_specified_nameinputs


J__inference_lstm_cell_21_layer_call_and_return_conditional_losses_39101976

inputs

states
states_12
matmul_readvariableop_resource:

4
 matmul_1_readvariableop_resource:
Å
.
biasadd_readvariableop_resource:	

identity

identity_1

identity_2¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:

*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
MatMul
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
Å
*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
add
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:
*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
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
P:ÿÿÿÿÿÿÿÿÿÅ:ÿÿÿÿÿÿÿÿÿÅ:ÿÿÿÿÿÿÿÿÿÅ:ÿÿÿÿÿÿÿÿÿÅ*
	num_split2
split`
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2	
Sigmoidd
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2
	Sigmoid_1]
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2
mulW
ReluRelusplit:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2
Relui
mul_1MulSigmoid:y:0Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2
mul_1^
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2
add_1d
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2
	Sigmoid_2V
Relu_1Relu	add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2
Relu_1m
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2
mul_2e
IdentityIdentity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2

Identityi

Identity_1Identity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2

Identity_1i

Identity_2Identity	add_1:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2

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
B:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿÅ:ÿÿÿÿÿÿÿÿÿÅ: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:PL
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ
 
_user_specified_namestates:PL
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ
 
_user_specified_namestates
\

E__inference_lstm_20_layer_call_and_return_conditional_losses_39102542

inputs>
+lstm_cell_20_matmul_readvariableop_resource:	]A
-lstm_cell_20_matmul_1_readvariableop_resource:
;
,lstm_cell_20_biasadd_readvariableop_resource:	
identity¢#lstm_cell_20/BiasAdd/ReadVariableOp¢"lstm_cell_20/MatMul/ReadVariableOp¢$lstm_cell_20/MatMul_1/ReadVariableOp¢whileD
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
B :2
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
B :2
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
:ÿÿÿÿÿÿÿÿÿ2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :2
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
B :2
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
:ÿÿÿÿÿÿÿÿÿ2	
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
"lstm_cell_20/MatMul/ReadVariableOpReadVariableOp+lstm_cell_20_matmul_readvariableop_resource*
_output_shapes
:	]*
dtype02$
"lstm_cell_20/MatMul/ReadVariableOp­
lstm_cell_20/MatMulMatMulstrided_slice_2:output:0*lstm_cell_20/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_20/MatMul¼
$lstm_cell_20/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_20_matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype02&
$lstm_cell_20/MatMul_1/ReadVariableOp©
lstm_cell_20/MatMul_1MatMulzeros:output:0,lstm_cell_20/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_20/MatMul_1 
lstm_cell_20/addAddV2lstm_cell_20/MatMul:product:0lstm_cell_20/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_20/add´
#lstm_cell_20/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_20_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02%
#lstm_cell_20/BiasAdd/ReadVariableOp­
lstm_cell_20/BiasAddBiasAddlstm_cell_20/add:z:0+lstm_cell_20/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_20/BiasAdd~
lstm_cell_20/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_20/split/split_dim÷
lstm_cell_20/splitSplit%lstm_cell_20/split/split_dim:output:0lstm_cell_20/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2
lstm_cell_20/split
lstm_cell_20/SigmoidSigmoidlstm_cell_20/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_20/Sigmoid
lstm_cell_20/Sigmoid_1Sigmoidlstm_cell_20/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_20/Sigmoid_1
lstm_cell_20/mulMullstm_cell_20/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_20/mul~
lstm_cell_20/ReluRelulstm_cell_20/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_20/Relu
lstm_cell_20/mul_1Mullstm_cell_20/Sigmoid:y:0lstm_cell_20/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_20/mul_1
lstm_cell_20/add_1AddV2lstm_cell_20/mul:z:0lstm_cell_20/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_20/add_1
lstm_cell_20/Sigmoid_2Sigmoidlstm_cell_20/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_20/Sigmoid_2}
lstm_cell_20/Relu_1Relulstm_cell_20/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_20/Relu_1¡
lstm_cell_20/mul_2Mullstm_cell_20/Sigmoid_2:y:0!lstm_cell_20/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_20/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ  2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_20_matmul_readvariableop_resource-lstm_cell_20_matmul_1_readvariableop_resource,lstm_cell_20_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_39102458*
condR
while_cond_39102457*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ  22
0TensorArrayV2Stack/TensorListStack/element_shapeé
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
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
:ÿÿÿÿÿÿÿÿÿ*
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
:ÿÿÿÿÿÿÿÿÿ2
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
:ÿÿÿÿÿÿÿÿÿ2

IdentityÈ
NoOpNoOp$^lstm_cell_20/BiasAdd/ReadVariableOp#^lstm_cell_20/MatMul/ReadVariableOp%^lstm_cell_20/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ]: : : 2J
#lstm_cell_20/BiasAdd/ReadVariableOp#lstm_cell_20/BiasAdd/ReadVariableOp2H
"lstm_cell_20/MatMul/ReadVariableOp"lstm_cell_20/MatMul/ReadVariableOp2L
$lstm_cell_20/MatMul_1/ReadVariableOp$lstm_cell_20/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
 
_user_specified_nameinputs
°?
Ô
while_body_39104276
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_20_matmul_readvariableop_resource_0:	]I
5while_lstm_cell_20_matmul_1_readvariableop_resource_0:
C
4while_lstm_cell_20_biasadd_readvariableop_resource_0:	
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_20_matmul_readvariableop_resource:	]G
3while_lstm_cell_20_matmul_1_readvariableop_resource:
A
2while_lstm_cell_20_biasadd_readvariableop_resource:	¢)while/lstm_cell_20/BiasAdd/ReadVariableOp¢(while/lstm_cell_20/MatMul/ReadVariableOp¢*while/lstm_cell_20/MatMul_1/ReadVariableOpÃ
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
(while/lstm_cell_20/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_20_matmul_readvariableop_resource_0*
_output_shapes
:	]*
dtype02*
(while/lstm_cell_20/MatMul/ReadVariableOp×
while/lstm_cell_20/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_20/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_20/MatMulÐ
*while/lstm_cell_20/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_20_matmul_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype02,
*while/lstm_cell_20/MatMul_1/ReadVariableOpÀ
while/lstm_cell_20/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_20/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_20/MatMul_1¸
while/lstm_cell_20/addAddV2#while/lstm_cell_20/MatMul:product:0%while/lstm_cell_20/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_20/addÈ
)while/lstm_cell_20/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_20_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02+
)while/lstm_cell_20/BiasAdd/ReadVariableOpÅ
while/lstm_cell_20/BiasAddBiasAddwhile/lstm_cell_20/add:z:01while/lstm_cell_20/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_20/BiasAdd
"while/lstm_cell_20/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_20/split/split_dim
while/lstm_cell_20/splitSplit+while/lstm_cell_20/split/split_dim:output:0#while/lstm_cell_20/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2
while/lstm_cell_20/split
while/lstm_cell_20/SigmoidSigmoid!while/lstm_cell_20/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_20/Sigmoid
while/lstm_cell_20/Sigmoid_1Sigmoid!while/lstm_cell_20/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_20/Sigmoid_1¡
while/lstm_cell_20/mulMul while/lstm_cell_20/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_20/mul
while/lstm_cell_20/ReluRelu!while/lstm_cell_20/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_20/Reluµ
while/lstm_cell_20/mul_1Mulwhile/lstm_cell_20/Sigmoid:y:0%while/lstm_cell_20/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_20/mul_1ª
while/lstm_cell_20/add_1AddV2while/lstm_cell_20/mul:z:0while/lstm_cell_20/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_20/add_1
while/lstm_cell_20/Sigmoid_2Sigmoid!while/lstm_cell_20/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_20/Sigmoid_2
while/lstm_cell_20/Relu_1Reluwhile/lstm_cell_20/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_20/Relu_1¹
while/lstm_cell_20/mul_2Mul while/lstm_cell_20/Sigmoid_2:y:0'while/lstm_cell_20/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_20/mul_2à
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_20/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_20/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_20/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/Identity_5Þ

while/NoOpNoOp*^while/lstm_cell_20/BiasAdd/ReadVariableOp)^while/lstm_cell_20/MatMul/ReadVariableOp+^while/lstm_cell_20/MatMul_1/ReadVariableOp*"
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
2while_lstm_cell_20_biasadd_readvariableop_resource4while_lstm_cell_20_biasadd_readvariableop_resource_0"l
3while_lstm_cell_20_matmul_1_readvariableop_resource5while_lstm_cell_20_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_20_matmul_readvariableop_resource3while_lstm_cell_20_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2V
)while/lstm_cell_20/BiasAdd/ReadVariableOp)while/lstm_cell_20/BiasAdd/ReadVariableOp2T
(while/lstm_cell_20/MatMul/ReadVariableOp(while/lstm_cell_20/MatMul/ReadVariableOp2X
*while/lstm_cell_20/MatMul_1/ReadVariableOp*while/lstm_cell_20/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
Ø
I
-__inference_dropout_20_layer_call_fn_39104728

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
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_dropout_20_layer_call_and_return_conditional_losses_391025552
PartitionedCallq
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
®

Ô
0__inference_sequential_10_layer_call_fn_39103269
lstm_20_input
unknown:	]
	unknown_0:

	unknown_1:	
	unknown_2:


	unknown_3:
Å

	unknown_4:	

	unknown_5:	Å
	unknown_6:
identity¢StatefulPartitionedCallÔ
StatefulPartitionedCallStatefulPartitionedCalllstm_20_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
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
K__inference_sequential_10_layer_call_and_return_conditional_losses_391032292
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
_user_specified_namelstm_20_input
éJ
Ö

lstm_21_while_body_39103897,
(lstm_21_while_lstm_21_while_loop_counter2
.lstm_21_while_lstm_21_while_maximum_iterations
lstm_21_while_placeholder
lstm_21_while_placeholder_1
lstm_21_while_placeholder_2
lstm_21_while_placeholder_3+
'lstm_21_while_lstm_21_strided_slice_1_0g
clstm_21_while_tensorarrayv2read_tensorlistgetitem_lstm_21_tensorarrayunstack_tensorlistfromtensor_0O
;lstm_21_while_lstm_cell_21_matmul_readvariableop_resource_0:

Q
=lstm_21_while_lstm_cell_21_matmul_1_readvariableop_resource_0:
Å
K
<lstm_21_while_lstm_cell_21_biasadd_readvariableop_resource_0:	

lstm_21_while_identity
lstm_21_while_identity_1
lstm_21_while_identity_2
lstm_21_while_identity_3
lstm_21_while_identity_4
lstm_21_while_identity_5)
%lstm_21_while_lstm_21_strided_slice_1e
alstm_21_while_tensorarrayv2read_tensorlistgetitem_lstm_21_tensorarrayunstack_tensorlistfromtensorM
9lstm_21_while_lstm_cell_21_matmul_readvariableop_resource:

O
;lstm_21_while_lstm_cell_21_matmul_1_readvariableop_resource:
Å
I
:lstm_21_while_lstm_cell_21_biasadd_readvariableop_resource:	
¢1lstm_21/while/lstm_cell_21/BiasAdd/ReadVariableOp¢0lstm_21/while/lstm_cell_21/MatMul/ReadVariableOp¢2lstm_21/while/lstm_cell_21/MatMul_1/ReadVariableOpÓ
?lstm_21/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ  2A
?lstm_21/while/TensorArrayV2Read/TensorListGetItem/element_shape
1lstm_21/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemclstm_21_while_tensorarrayv2read_tensorlistgetitem_lstm_21_tensorarrayunstack_tensorlistfromtensor_0lstm_21_while_placeholderHlstm_21/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype023
1lstm_21/while/TensorArrayV2Read/TensorListGetItemâ
0lstm_21/while/lstm_cell_21/MatMul/ReadVariableOpReadVariableOp;lstm_21_while_lstm_cell_21_matmul_readvariableop_resource_0* 
_output_shapes
:

*
dtype022
0lstm_21/while/lstm_cell_21/MatMul/ReadVariableOp÷
!lstm_21/while/lstm_cell_21/MatMulMatMul8lstm_21/while/TensorArrayV2Read/TensorListGetItem:item:08lstm_21/while/lstm_cell_21/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2#
!lstm_21/while/lstm_cell_21/MatMulè
2lstm_21/while/lstm_cell_21/MatMul_1/ReadVariableOpReadVariableOp=lstm_21_while_lstm_cell_21_matmul_1_readvariableop_resource_0* 
_output_shapes
:
Å
*
dtype024
2lstm_21/while/lstm_cell_21/MatMul_1/ReadVariableOpà
#lstm_21/while/lstm_cell_21/MatMul_1MatMullstm_21_while_placeholder_2:lstm_21/while/lstm_cell_21/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2%
#lstm_21/while/lstm_cell_21/MatMul_1Ø
lstm_21/while/lstm_cell_21/addAddV2+lstm_21/while/lstm_cell_21/MatMul:product:0-lstm_21/while/lstm_cell_21/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2 
lstm_21/while/lstm_cell_21/addà
1lstm_21/while/lstm_cell_21/BiasAdd/ReadVariableOpReadVariableOp<lstm_21_while_lstm_cell_21_biasadd_readvariableop_resource_0*
_output_shapes	
:
*
dtype023
1lstm_21/while/lstm_cell_21/BiasAdd/ReadVariableOpå
"lstm_21/while/lstm_cell_21/BiasAddBiasAdd"lstm_21/while/lstm_cell_21/add:z:09lstm_21/while/lstm_cell_21/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2$
"lstm_21/while/lstm_cell_21/BiasAdd
*lstm_21/while/lstm_cell_21/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2,
*lstm_21/while/lstm_cell_21/split/split_dim¯
 lstm_21/while/lstm_cell_21/splitSplit3lstm_21/while/lstm_cell_21/split/split_dim:output:0+lstm_21/while/lstm_cell_21/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿÅ:ÿÿÿÿÿÿÿÿÿÅ:ÿÿÿÿÿÿÿÿÿÅ:ÿÿÿÿÿÿÿÿÿÅ*
	num_split2"
 lstm_21/while/lstm_cell_21/split±
"lstm_21/while/lstm_cell_21/SigmoidSigmoid)lstm_21/while/lstm_cell_21/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2$
"lstm_21/while/lstm_cell_21/Sigmoidµ
$lstm_21/while/lstm_cell_21/Sigmoid_1Sigmoid)lstm_21/while/lstm_cell_21/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2&
$lstm_21/while/lstm_cell_21/Sigmoid_1Á
lstm_21/while/lstm_cell_21/mulMul(lstm_21/while/lstm_cell_21/Sigmoid_1:y:0lstm_21_while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2 
lstm_21/while/lstm_cell_21/mul¨
lstm_21/while/lstm_cell_21/ReluRelu)lstm_21/while/lstm_cell_21/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2!
lstm_21/while/lstm_cell_21/ReluÕ
 lstm_21/while/lstm_cell_21/mul_1Mul&lstm_21/while/lstm_cell_21/Sigmoid:y:0-lstm_21/while/lstm_cell_21/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2"
 lstm_21/while/lstm_cell_21/mul_1Ê
 lstm_21/while/lstm_cell_21/add_1AddV2"lstm_21/while/lstm_cell_21/mul:z:0$lstm_21/while/lstm_cell_21/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2"
 lstm_21/while/lstm_cell_21/add_1µ
$lstm_21/while/lstm_cell_21/Sigmoid_2Sigmoid)lstm_21/while/lstm_cell_21/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2&
$lstm_21/while/lstm_cell_21/Sigmoid_2§
!lstm_21/while/lstm_cell_21/Relu_1Relu$lstm_21/while/lstm_cell_21/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2#
!lstm_21/while/lstm_cell_21/Relu_1Ù
 lstm_21/while/lstm_cell_21/mul_2Mul(lstm_21/while/lstm_cell_21/Sigmoid_2:y:0/lstm_21/while/lstm_cell_21/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2"
 lstm_21/while/lstm_cell_21/mul_2
2lstm_21/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_21_while_placeholder_1lstm_21_while_placeholder$lstm_21/while/lstm_cell_21/mul_2:z:0*
_output_shapes
: *
element_dtype024
2lstm_21/while/TensorArrayV2Write/TensorListSetIteml
lstm_21/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_21/while/add/y
lstm_21/while/addAddV2lstm_21_while_placeholderlstm_21/while/add/y:output:0*
T0*
_output_shapes
: 2
lstm_21/while/addp
lstm_21/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_21/while/add_1/y
lstm_21/while/add_1AddV2(lstm_21_while_lstm_21_while_loop_counterlstm_21/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lstm_21/while/add_1
lstm_21/while/IdentityIdentitylstm_21/while/add_1:z:0^lstm_21/while/NoOp*
T0*
_output_shapes
: 2
lstm_21/while/Identity¦
lstm_21/while/Identity_1Identity.lstm_21_while_lstm_21_while_maximum_iterations^lstm_21/while/NoOp*
T0*
_output_shapes
: 2
lstm_21/while/Identity_1
lstm_21/while/Identity_2Identitylstm_21/while/add:z:0^lstm_21/while/NoOp*
T0*
_output_shapes
: 2
lstm_21/while/Identity_2º
lstm_21/while/Identity_3IdentityBlstm_21/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_21/while/NoOp*
T0*
_output_shapes
: 2
lstm_21/while/Identity_3®
lstm_21/while/Identity_4Identity$lstm_21/while/lstm_cell_21/mul_2:z:0^lstm_21/while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2
lstm_21/while/Identity_4®
lstm_21/while/Identity_5Identity$lstm_21/while/lstm_cell_21/add_1:z:0^lstm_21/while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2
lstm_21/while/Identity_5
lstm_21/while/NoOpNoOp2^lstm_21/while/lstm_cell_21/BiasAdd/ReadVariableOp1^lstm_21/while/lstm_cell_21/MatMul/ReadVariableOp3^lstm_21/while/lstm_cell_21/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
lstm_21/while/NoOp"9
lstm_21_while_identitylstm_21/while/Identity:output:0"=
lstm_21_while_identity_1!lstm_21/while/Identity_1:output:0"=
lstm_21_while_identity_2!lstm_21/while/Identity_2:output:0"=
lstm_21_while_identity_3!lstm_21/while/Identity_3:output:0"=
lstm_21_while_identity_4!lstm_21/while/Identity_4:output:0"=
lstm_21_while_identity_5!lstm_21/while/Identity_5:output:0"P
%lstm_21_while_lstm_21_strided_slice_1'lstm_21_while_lstm_21_strided_slice_1_0"z
:lstm_21_while_lstm_cell_21_biasadd_readvariableop_resource<lstm_21_while_lstm_cell_21_biasadd_readvariableop_resource_0"|
;lstm_21_while_lstm_cell_21_matmul_1_readvariableop_resource=lstm_21_while_lstm_cell_21_matmul_1_readvariableop_resource_0"x
9lstm_21_while_lstm_cell_21_matmul_readvariableop_resource;lstm_21_while_lstm_cell_21_matmul_readvariableop_resource_0"È
alstm_21_while_tensorarrayv2read_tensorlistgetitem_lstm_21_tensorarrayunstack_tensorlistfromtensorclstm_21_while_tensorarrayv2read_tensorlistgetitem_lstm_21_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿÅ:ÿÿÿÿÿÿÿÿÿÅ: : : : : 2f
1lstm_21/while/lstm_cell_21/BiasAdd/ReadVariableOp1lstm_21/while/lstm_cell_21/BiasAdd/ReadVariableOp2d
0lstm_21/while/lstm_cell_21/MatMul/ReadVariableOp0lstm_21/while/lstm_cell_21/MatMul/ReadVariableOp2h
2lstm_21/while/lstm_cell_21/MatMul_1/ReadVariableOp2lstm_21/while/lstm_cell_21/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ:

_output_shapes
: :

_output_shapes
: 
&
ó
while_body_39101214
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_00
while_lstm_cell_20_39101238_0:	]1
while_lstm_cell_20_39101240_0:
,
while_lstm_cell_20_39101242_0:	
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor.
while_lstm_cell_20_39101238:	]/
while_lstm_cell_20_39101240:
*
while_lstm_cell_20_39101242:	¢*while/lstm_cell_20/StatefulPartitionedCallÃ
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
*while/lstm_cell_20/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_20_39101238_0while_lstm_cell_20_39101240_0while_lstm_cell_20_39101242_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_lstm_cell_20_layer_call_and_return_conditional_losses_391012002,
*while/lstm_cell_20/StatefulPartitionedCall÷
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder3while/lstm_cell_20/StatefulPartitionedCall:output:0*
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
while/Identity_4Identity3while/lstm_cell_20/StatefulPartitionedCall:output:1^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/Identity_4¥
while/Identity_5Identity3while/lstm_cell_20/StatefulPartitionedCall:output:2^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/Identity_5

while/NoOpNoOp+^while/lstm_cell_20/StatefulPartitionedCall*"
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
while_lstm_cell_20_39101238while_lstm_cell_20_39101238_0"<
while_lstm_cell_20_39101240while_lstm_cell_20_39101240_0"<
while_lstm_cell_20_39101242while_lstm_cell_20_39101242_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2X
*while/lstm_cell_20/StatefulPartitionedCall*while/lstm_cell_20/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
Ô

í
lstm_20_while_cond_39103741,
(lstm_20_while_lstm_20_while_loop_counter2
.lstm_20_while_lstm_20_while_maximum_iterations
lstm_20_while_placeholder
lstm_20_while_placeholder_1
lstm_20_while_placeholder_2
lstm_20_while_placeholder_3.
*lstm_20_while_less_lstm_20_strided_slice_1F
Blstm_20_while_lstm_20_while_cond_39103741___redundant_placeholder0F
Blstm_20_while_lstm_20_while_cond_39103741___redundant_placeholder1F
Blstm_20_while_lstm_20_while_cond_39103741___redundant_placeholder2F
Blstm_20_while_lstm_20_while_cond_39103741___redundant_placeholder3
lstm_20_while_identity

lstm_20/while/LessLesslstm_20_while_placeholder*lstm_20_while_less_lstm_20_strided_slice_1*
T0*
_output_shapes
: 2
lstm_20/while/Lessu
lstm_20/while/IdentityIdentitylstm_20/while/Less:z:0*
T0
*
_output_shapes
: 2
lstm_20/while/Identity"9
lstm_20_while_identitylstm_20/while/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:


J__inference_lstm_cell_21_layer_call_and_return_conditional_losses_39101830

inputs

states
states_12
matmul_readvariableop_resource:

4
 matmul_1_readvariableop_resource:
Å
.
biasadd_readvariableop_resource:	

identity

identity_1

identity_2¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:

*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
MatMul
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
Å
*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
add
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:
*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
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
P:ÿÿÿÿÿÿÿÿÿÅ:ÿÿÿÿÿÿÿÿÿÅ:ÿÿÿÿÿÿÿÿÿÅ:ÿÿÿÿÿÿÿÿÿÅ*
	num_split2
split`
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2	
Sigmoidd
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2
	Sigmoid_1]
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2
mulW
ReluRelusplit:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2
Relui
mul_1MulSigmoid:y:0Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2
mul_1^
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2
add_1d
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2
	Sigmoid_2V
Relu_1Relu	add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2
Relu_1m
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2
mul_2e
IdentityIdentity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2

Identityi

Identity_1Identity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2

Identity_1i

Identity_2Identity	add_1:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2

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
B:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿÅ:ÿÿÿÿÿÿÿÿÿÅ: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:PL
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ
 
_user_specified_namestates:PL
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ
 
_user_specified_namestates
°?
Ô
while_body_39103088
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_20_matmul_readvariableop_resource_0:	]I
5while_lstm_cell_20_matmul_1_readvariableop_resource_0:
C
4while_lstm_cell_20_biasadd_readvariableop_resource_0:	
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_20_matmul_readvariableop_resource:	]G
3while_lstm_cell_20_matmul_1_readvariableop_resource:
A
2while_lstm_cell_20_biasadd_readvariableop_resource:	¢)while/lstm_cell_20/BiasAdd/ReadVariableOp¢(while/lstm_cell_20/MatMul/ReadVariableOp¢*while/lstm_cell_20/MatMul_1/ReadVariableOpÃ
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
(while/lstm_cell_20/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_20_matmul_readvariableop_resource_0*
_output_shapes
:	]*
dtype02*
(while/lstm_cell_20/MatMul/ReadVariableOp×
while/lstm_cell_20/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_20/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_20/MatMulÐ
*while/lstm_cell_20/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_20_matmul_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype02,
*while/lstm_cell_20/MatMul_1/ReadVariableOpÀ
while/lstm_cell_20/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_20/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_20/MatMul_1¸
while/lstm_cell_20/addAddV2#while/lstm_cell_20/MatMul:product:0%while/lstm_cell_20/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_20/addÈ
)while/lstm_cell_20/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_20_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02+
)while/lstm_cell_20/BiasAdd/ReadVariableOpÅ
while/lstm_cell_20/BiasAddBiasAddwhile/lstm_cell_20/add:z:01while/lstm_cell_20/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_20/BiasAdd
"while/lstm_cell_20/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_20/split/split_dim
while/lstm_cell_20/splitSplit+while/lstm_cell_20/split/split_dim:output:0#while/lstm_cell_20/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
	num_split2
while/lstm_cell_20/split
while/lstm_cell_20/SigmoidSigmoid!while/lstm_cell_20/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_20/Sigmoid
while/lstm_cell_20/Sigmoid_1Sigmoid!while/lstm_cell_20/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_20/Sigmoid_1¡
while/lstm_cell_20/mulMul while/lstm_cell_20/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_20/mul
while/lstm_cell_20/ReluRelu!while/lstm_cell_20/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_20/Reluµ
while/lstm_cell_20/mul_1Mulwhile/lstm_cell_20/Sigmoid:y:0%while/lstm_cell_20/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_20/mul_1ª
while/lstm_cell_20/add_1AddV2while/lstm_cell_20/mul:z:0while/lstm_cell_20/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_20/add_1
while/lstm_cell_20/Sigmoid_2Sigmoid!while/lstm_cell_20/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_20/Sigmoid_2
while/lstm_cell_20/Relu_1Reluwhile/lstm_cell_20/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_20/Relu_1¹
while/lstm_cell_20/mul_2Mul while/lstm_cell_20/Sigmoid_2:y:0'while/lstm_cell_20/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_20/mul_2à
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_20/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_20/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_20/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/Identity_5Þ

while/NoOpNoOp*^while/lstm_cell_20/BiasAdd/ReadVariableOp)^while/lstm_cell_20/MatMul/ReadVariableOp+^while/lstm_cell_20/MatMul_1/ReadVariableOp*"
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
2while_lstm_cell_20_biasadd_readvariableop_resource4while_lstm_cell_20_biasadd_readvariableop_resource_0"l
3while_lstm_cell_20_matmul_1_readvariableop_resource5while_lstm_cell_20_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_20_matmul_readvariableop_resource3while_lstm_cell_20_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2V
)while/lstm_cell_20/BiasAdd/ReadVariableOp)while/lstm_cell_20/BiasAdd/ReadVariableOp2T
(while/lstm_cell_20/MatMul/ReadVariableOp(while/lstm_cell_20/MatMul/ReadVariableOp2X
*while/lstm_cell_20/MatMul_1/ReadVariableOp*while/lstm_cell_20/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
´?
Ö
while_body_39102892
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
3while_lstm_cell_21_matmul_readvariableop_resource_0:

I
5while_lstm_cell_21_matmul_1_readvariableop_resource_0:
Å
C
4while_lstm_cell_21_biasadd_readvariableop_resource_0:	

while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
1while_lstm_cell_21_matmul_readvariableop_resource:

G
3while_lstm_cell_21_matmul_1_readvariableop_resource:
Å
A
2while_lstm_cell_21_biasadd_readvariableop_resource:	
¢)while/lstm_cell_21/BiasAdd/ReadVariableOp¢(while/lstm_cell_21/MatMul/ReadVariableOp¢*while/lstm_cell_21/MatMul_1/ReadVariableOpÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ  29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÔ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemÊ
(while/lstm_cell_21/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_21_matmul_readvariableop_resource_0* 
_output_shapes
:

*
dtype02*
(while/lstm_cell_21/MatMul/ReadVariableOp×
while/lstm_cell_21/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_21/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
while/lstm_cell_21/MatMulÐ
*while/lstm_cell_21/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_21_matmul_1_readvariableop_resource_0* 
_output_shapes
:
Å
*
dtype02,
*while/lstm_cell_21/MatMul_1/ReadVariableOpÀ
while/lstm_cell_21/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_21/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
while/lstm_cell_21/MatMul_1¸
while/lstm_cell_21/addAddV2#while/lstm_cell_21/MatMul:product:0%while/lstm_cell_21/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
while/lstm_cell_21/addÈ
)while/lstm_cell_21/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_21_biasadd_readvariableop_resource_0*
_output_shapes	
:
*
dtype02+
)while/lstm_cell_21/BiasAdd/ReadVariableOpÅ
while/lstm_cell_21/BiasAddBiasAddwhile/lstm_cell_21/add:z:01while/lstm_cell_21/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
while/lstm_cell_21/BiasAdd
"while/lstm_cell_21/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_21/split/split_dim
while/lstm_cell_21/splitSplit+while/lstm_cell_21/split/split_dim:output:0#while/lstm_cell_21/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿÅ:ÿÿÿÿÿÿÿÿÿÅ:ÿÿÿÿÿÿÿÿÿÅ:ÿÿÿÿÿÿÿÿÿÅ*
	num_split2
while/lstm_cell_21/split
while/lstm_cell_21/SigmoidSigmoid!while/lstm_cell_21/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2
while/lstm_cell_21/Sigmoid
while/lstm_cell_21/Sigmoid_1Sigmoid!while/lstm_cell_21/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2
while/lstm_cell_21/Sigmoid_1¡
while/lstm_cell_21/mulMul while/lstm_cell_21/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2
while/lstm_cell_21/mul
while/lstm_cell_21/ReluRelu!while/lstm_cell_21/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2
while/lstm_cell_21/Reluµ
while/lstm_cell_21/mul_1Mulwhile/lstm_cell_21/Sigmoid:y:0%while/lstm_cell_21/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2
while/lstm_cell_21/mul_1ª
while/lstm_cell_21/add_1AddV2while/lstm_cell_21/mul:z:0while/lstm_cell_21/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2
while/lstm_cell_21/add_1
while/lstm_cell_21/Sigmoid_2Sigmoid!while/lstm_cell_21/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2
while/lstm_cell_21/Sigmoid_2
while/lstm_cell_21/Relu_1Reluwhile/lstm_cell_21/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2
while/lstm_cell_21/Relu_1¹
while/lstm_cell_21/mul_2Mul while/lstm_cell_21/Sigmoid_2:y:0'while/lstm_cell_21/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2
while/lstm_cell_21/mul_2à
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_21/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_21/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_21/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ2
while/Identity_5Þ

while/NoOpNoOp*^while/lstm_cell_21/BiasAdd/ReadVariableOp)^while/lstm_cell_21/MatMul/ReadVariableOp+^while/lstm_cell_21/MatMul_1/ReadVariableOp*"
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
2while_lstm_cell_21_biasadd_readvariableop_resource4while_lstm_cell_21_biasadd_readvariableop_resource_0"l
3while_lstm_cell_21_matmul_1_readvariableop_resource5while_lstm_cell_21_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_21_matmul_readvariableop_resource3while_lstm_cell_21_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿÅ:ÿÿÿÿÿÿÿÿÿÅ: : : : : 2V
)while/lstm_cell_21/BiasAdd/ReadVariableOp)while/lstm_cell_21/BiasAdd/ReadVariableOp2T
(while/lstm_cell_21/MatMul/ReadVariableOp(while/lstm_cell_21/MatMul/ReadVariableOp2X
*while/lstm_cell_21/MatMul_1/ReadVariableOp*while/lstm_cell_21/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ:

_output_shapes
: :

_output_shapes
: "¨L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*¿
serving_default«
K
lstm_20_input:
serving_default_lstm_20_input:0ÿÿÿÿÿÿÿÿÿ]@
dense_104
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:©¼
ø
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
_default_save_signature
+&call_and_return_all_conditional_losses
__call__"
_tf_keras_sequential
Å
cell

state_spec
trainable_variables
	variables
regularization_losses
	keras_api
+&call_and_return_all_conditional_losses
__call__"
_tf_keras_rnn_layer
§
trainable_variables
	variables
regularization_losses
	keras_api
+&call_and_return_all_conditional_losses
__call__"
_tf_keras_layer
Å
cell

state_spec
trainable_variables
	variables
regularization_losses
	keras_api
+&call_and_return_all_conditional_losses
__call__"
_tf_keras_rnn_layer
§
trainable_variables
	variables
regularization_losses
	keras_api
+&call_and_return_all_conditional_losses
__call__"
_tf_keras_layer
½

 kernel
!bias
"trainable_variables
#	variables
$regularization_losses
%	keras_api
+&call_and_return_all_conditional_losses
__call__"
_tf_keras_layer
ã
&iter

'beta_1

(beta_2
	)decay
*learning_rate mp!mq+mr,ms-mt.mu/mv0mw vx!vy+vz,v{-v|.v}/v~0v"
	optimizer
X
+0
,1
-2
.3
/4
05
 6
!7"
trackable_list_wrapper
X
+0
,1
-2
.3
/4
05
 6
!7"
trackable_list_wrapper
 "
trackable_list_wrapper
Î
trainable_variables
1layer_metrics
2layer_regularization_losses
	variables

3layers
4non_trainable_variables
	regularization_losses
5metrics
__call__
_default_save_signature
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
-
serving_default"
signature_map
ã
6
state_size

+kernel
,recurrent_kernel
-bias
7trainable_variables
8	variables
9regularization_losses
:	keras_api
+&call_and_return_all_conditional_losses
__call__"
_tf_keras_layer
 "
trackable_list_wrapper
5
+0
,1
-2"
trackable_list_wrapper
5
+0
,1
-2"
trackable_list_wrapper
 "
trackable_list_wrapper
¼
trainable_variables
;layer_metrics
<layer_regularization_losses
	variables

=layers
>non_trainable_variables
regularization_losses

?states
@metrics
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
°
trainable_variables
Alayer_metrics
Blayer_regularization_losses
	variables

Clayers
Dnon_trainable_variables
regularization_losses
Emetrics
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
ã
F
state_size

.kernel
/recurrent_kernel
0bias
Gtrainable_variables
H	variables
Iregularization_losses
J	keras_api
+&call_and_return_all_conditional_losses
__call__"
_tf_keras_layer
 "
trackable_list_wrapper
5
.0
/1
02"
trackable_list_wrapper
5
.0
/1
02"
trackable_list_wrapper
 "
trackable_list_wrapper
¼
trainable_variables
Klayer_metrics
Llayer_regularization_losses
	variables

Mlayers
Nnon_trainable_variables
regularization_losses

Ostates
Pmetrics
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
°
trainable_variables
Qlayer_metrics
Rlayer_regularization_losses
	variables

Slayers
Tnon_trainable_variables
regularization_losses
Umetrics
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
": 	Å2dense_10/kernel
:2dense_10/bias
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
°
"trainable_variables
Vlayer_metrics
Wlayer_regularization_losses
#	variables

Xlayers
Ynon_trainable_variables
$regularization_losses
Zmetrics
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
.:,	]2lstm_20/lstm_cell_20/kernel
9:7
2%lstm_20/lstm_cell_20/recurrent_kernel
(:&2lstm_20/lstm_cell_20/bias
/:-

2lstm_21/lstm_cell_21/kernel
9:7
Å
2%lstm_21/lstm_cell_21/recurrent_kernel
(:&
2lstm_21/lstm_cell_21/bias
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
[0
\1"
trackable_list_wrapper
 "
trackable_list_wrapper
5
+0
,1
-2"
trackable_list_wrapper
5
+0
,1
-2"
trackable_list_wrapper
 "
trackable_list_wrapper
°
7trainable_variables
]layer_metrics
^layer_regularization_losses
8	variables

_layers
`non_trainable_variables
9regularization_losses
ametrics
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
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
.0
/1
02"
trackable_list_wrapper
5
.0
/1
02"
trackable_list_wrapper
 "
trackable_list_wrapper
°
Gtrainable_variables
blayer_metrics
clayer_regularization_losses
H	variables

dlayers
enon_trainable_variables
Iregularization_losses
fmetrics
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
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
	gtotal
	hcount
i	variables
j	keras_api"
_tf_keras_metric
^
	ktotal
	lcount
m
_fn_kwargs
n	variables
o	keras_api"
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
g0
h1"
trackable_list_wrapper
-
i	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
k0
l1"
trackable_list_wrapper
-
n	variables"
_generic_user_object
':%	Å2Adam/dense_10/kernel/m
 :2Adam/dense_10/bias/m
3:1	]2"Adam/lstm_20/lstm_cell_20/kernel/m
>:<
2,Adam/lstm_20/lstm_cell_20/recurrent_kernel/m
-:+2 Adam/lstm_20/lstm_cell_20/bias/m
4:2

2"Adam/lstm_21/lstm_cell_21/kernel/m
>:<
Å
2,Adam/lstm_21/lstm_cell_21/recurrent_kernel/m
-:+
2 Adam/lstm_21/lstm_cell_21/bias/m
':%	Å2Adam/dense_10/kernel/v
 :2Adam/dense_10/bias/v
3:1	]2"Adam/lstm_20/lstm_cell_20/kernel/v
>:<
2,Adam/lstm_20/lstm_cell_20/recurrent_kernel/v
-:+2 Adam/lstm_20/lstm_cell_20/bias/v
4:2

2"Adam/lstm_21/lstm_cell_21/kernel/v
>:<
Å
2,Adam/lstm_21/lstm_cell_21/recurrent_kernel/v
-:+
2 Adam/lstm_21/lstm_cell_21/bias/v
ÔBÑ
#__inference__wrapped_model_39101125lstm_20_input"
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
ú2÷
K__inference_sequential_10_layer_call_and_return_conditional_losses_39103675
K__inference_sequential_10_layer_call_and_return_conditional_losses_39104016
K__inference_sequential_10_layer_call_and_return_conditional_losses_39103294
K__inference_sequential_10_layer_call_and_return_conditional_losses_39103319À
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
2
0__inference_sequential_10_layer_call_fn_39102779
0__inference_sequential_10_layer_call_fn_39104037
0__inference_sequential_10_layer_call_fn_39104058
0__inference_sequential_10_layer_call_fn_39103269À
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
E__inference_lstm_20_layer_call_and_return_conditional_losses_39104209
E__inference_lstm_20_layer_call_and_return_conditional_losses_39104360
E__inference_lstm_20_layer_call_and_return_conditional_losses_39104511
E__inference_lstm_20_layer_call_and_return_conditional_losses_39104662Õ
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
*__inference_lstm_20_layer_call_fn_39104673
*__inference_lstm_20_layer_call_fn_39104684
*__inference_lstm_20_layer_call_fn_39104695
*__inference_lstm_20_layer_call_fn_39104706Õ
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
H__inference_dropout_20_layer_call_and_return_conditional_losses_39104711
H__inference_dropout_20_layer_call_and_return_conditional_losses_39104723´
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
-__inference_dropout_20_layer_call_fn_39104728
-__inference_dropout_20_layer_call_fn_39104733´
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
E__inference_lstm_21_layer_call_and_return_conditional_losses_39104884
E__inference_lstm_21_layer_call_and_return_conditional_losses_39105035
E__inference_lstm_21_layer_call_and_return_conditional_losses_39105186
E__inference_lstm_21_layer_call_and_return_conditional_losses_39105337Õ
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
*__inference_lstm_21_layer_call_fn_39105348
*__inference_lstm_21_layer_call_fn_39105359
*__inference_lstm_21_layer_call_fn_39105370
*__inference_lstm_21_layer_call_fn_39105381Õ
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
H__inference_dropout_21_layer_call_and_return_conditional_losses_39105386
H__inference_dropout_21_layer_call_and_return_conditional_losses_39105398´
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
-__inference_dropout_21_layer_call_fn_39105403
-__inference_dropout_21_layer_call_fn_39105408´
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
ð2í
F__inference_dense_10_layer_call_and_return_conditional_losses_39105439¢
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
Õ2Ò
+__inference_dense_10_layer_call_fn_39105448¢
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
&__inference_signature_wrapper_39103348lstm_20_input"
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
J__inference_lstm_cell_20_layer_call_and_return_conditional_losses_39105480
J__inference_lstm_cell_20_layer_call_and_return_conditional_losses_39105512¾
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
/__inference_lstm_cell_20_layer_call_fn_39105529
/__inference_lstm_cell_20_layer_call_fn_39105546¾
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
J__inference_lstm_cell_21_layer_call_and_return_conditional_losses_39105578
J__inference_lstm_cell_21_layer_call_and_return_conditional_losses_39105610¾
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
/__inference_lstm_cell_21_layer_call_fn_39105627
/__inference_lstm_cell_21_layer_call_fn_39105644¾
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
#__inference__wrapped_model_39101125+,-./0 !:¢7
0¢-
+(
lstm_20_inputÿÿÿÿÿÿÿÿÿ]
ª "7ª4
2
dense_10&#
dense_10ÿÿÿÿÿÿÿÿÿ¯
F__inference_dense_10_layer_call_and_return_conditional_losses_39105439e !4¢1
*¢'
%"
inputsÿÿÿÿÿÿÿÿÿÅ
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ
 
+__inference_dense_10_layer_call_fn_39105448X !4¢1
*¢'
%"
inputsÿÿÿÿÿÿÿÿÿÅ
ª "ÿÿÿÿÿÿÿÿÿ²
H__inference_dropout_20_layer_call_and_return_conditional_losses_39104711f8¢5
.¢+
%"
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "*¢'
 
0ÿÿÿÿÿÿÿÿÿ
 ²
H__inference_dropout_20_layer_call_and_return_conditional_losses_39104723f8¢5
.¢+
%"
inputsÿÿÿÿÿÿÿÿÿ
p
ª "*¢'
 
0ÿÿÿÿÿÿÿÿÿ
 
-__inference_dropout_20_layer_call_fn_39104728Y8¢5
.¢+
%"
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "ÿÿÿÿÿÿÿÿÿ
-__inference_dropout_20_layer_call_fn_39104733Y8¢5
.¢+
%"
inputsÿÿÿÿÿÿÿÿÿ
p
ª "ÿÿÿÿÿÿÿÿÿ²
H__inference_dropout_21_layer_call_and_return_conditional_losses_39105386f8¢5
.¢+
%"
inputsÿÿÿÿÿÿÿÿÿÅ
p 
ª "*¢'
 
0ÿÿÿÿÿÿÿÿÿÅ
 ²
H__inference_dropout_21_layer_call_and_return_conditional_losses_39105398f8¢5
.¢+
%"
inputsÿÿÿÿÿÿÿÿÿÅ
p
ª "*¢'
 
0ÿÿÿÿÿÿÿÿÿÅ
 
-__inference_dropout_21_layer_call_fn_39105403Y8¢5
.¢+
%"
inputsÿÿÿÿÿÿÿÿÿÅ
p 
ª "ÿÿÿÿÿÿÿÿÿÅ
-__inference_dropout_21_layer_call_fn_39105408Y8¢5
.¢+
%"
inputsÿÿÿÿÿÿÿÿÿÅ
p
ª "ÿÿÿÿÿÿÿÿÿÅÕ
E__inference_lstm_20_layer_call_and_return_conditional_losses_39104209+,-O¢L
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
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Õ
E__inference_lstm_20_layer_call_and_return_conditional_losses_39104360+,-O¢L
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
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 »
E__inference_lstm_20_layer_call_and_return_conditional_losses_39104511r+,-?¢<
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
0ÿÿÿÿÿÿÿÿÿ
 »
E__inference_lstm_20_layer_call_and_return_conditional_losses_39104662r+,-?¢<
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
0ÿÿÿÿÿÿÿÿÿ
 ¬
*__inference_lstm_20_layer_call_fn_39104673~+,-O¢L
E¢B
41
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ]

 
p 

 
ª "&#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¬
*__inference_lstm_20_layer_call_fn_39104684~+,-O¢L
E¢B
41
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ]

 
p

 
ª "&#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
*__inference_lstm_20_layer_call_fn_39104695e+,-?¢<
5¢2
$!
inputsÿÿÿÿÿÿÿÿÿ]

 
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
*__inference_lstm_20_layer_call_fn_39104706e+,-?¢<
5¢2
$!
inputsÿÿÿÿÿÿÿÿÿ]

 
p

 
ª "ÿÿÿÿÿÿÿÿÿÖ
E__inference_lstm_21_layer_call_and_return_conditional_losses_39104884./0P¢M
F¢C
52
0-
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
p 

 
ª "3¢0
)&
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÅ
 Ö
E__inference_lstm_21_layer_call_and_return_conditional_losses_39105035./0P¢M
F¢C
52
0-
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
p

 
ª "3¢0
)&
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÅ
 ¼
E__inference_lstm_21_layer_call_and_return_conditional_losses_39105186s./0@¢=
6¢3
%"
inputsÿÿÿÿÿÿÿÿÿ

 
p 

 
ª "*¢'
 
0ÿÿÿÿÿÿÿÿÿÅ
 ¼
E__inference_lstm_21_layer_call_and_return_conditional_losses_39105337s./0@¢=
6¢3
%"
inputsÿÿÿÿÿÿÿÿÿ

 
p

 
ª "*¢'
 
0ÿÿÿÿÿÿÿÿÿÅ
 ­
*__inference_lstm_21_layer_call_fn_39105348./0P¢M
F¢C
52
0-
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
p 

 
ª "&#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÅ­
*__inference_lstm_21_layer_call_fn_39105359./0P¢M
F¢C
52
0-
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
p

 
ª "&#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÅ
*__inference_lstm_21_layer_call_fn_39105370f./0@¢=
6¢3
%"
inputsÿÿÿÿÿÿÿÿÿ

 
p 

 
ª "ÿÿÿÿÿÿÿÿÿÅ
*__inference_lstm_21_layer_call_fn_39105381f./0@¢=
6¢3
%"
inputsÿÿÿÿÿÿÿÿÿ

 
p

 
ª "ÿÿÿÿÿÿÿÿÿÅÑ
J__inference_lstm_cell_20_layer_call_and_return_conditional_losses_39105480+,-¢
x¢u
 
inputsÿÿÿÿÿÿÿÿÿ]
M¢J
# 
states/0ÿÿÿÿÿÿÿÿÿ
# 
states/1ÿÿÿÿÿÿÿÿÿ
p 
ª "v¢s
l¢i

0/0ÿÿÿÿÿÿÿÿÿ
GD
 
0/1/0ÿÿÿÿÿÿÿÿÿ
 
0/1/1ÿÿÿÿÿÿÿÿÿ
 Ñ
J__inference_lstm_cell_20_layer_call_and_return_conditional_losses_39105512+,-¢
x¢u
 
inputsÿÿÿÿÿÿÿÿÿ]
M¢J
# 
states/0ÿÿÿÿÿÿÿÿÿ
# 
states/1ÿÿÿÿÿÿÿÿÿ
p
ª "v¢s
l¢i

0/0ÿÿÿÿÿÿÿÿÿ
GD
 
0/1/0ÿÿÿÿÿÿÿÿÿ
 
0/1/1ÿÿÿÿÿÿÿÿÿ
 ¦
/__inference_lstm_cell_20_layer_call_fn_39105529ò+,-¢
x¢u
 
inputsÿÿÿÿÿÿÿÿÿ]
M¢J
# 
states/0ÿÿÿÿÿÿÿÿÿ
# 
states/1ÿÿÿÿÿÿÿÿÿ
p 
ª "f¢c

0ÿÿÿÿÿÿÿÿÿ
C@

1/0ÿÿÿÿÿÿÿÿÿ

1/1ÿÿÿÿÿÿÿÿÿ¦
/__inference_lstm_cell_20_layer_call_fn_39105546ò+,-¢
x¢u
 
inputsÿÿÿÿÿÿÿÿÿ]
M¢J
# 
states/0ÿÿÿÿÿÿÿÿÿ
# 
states/1ÿÿÿÿÿÿÿÿÿ
p
ª "f¢c

0ÿÿÿÿÿÿÿÿÿ
C@

1/0ÿÿÿÿÿÿÿÿÿ

1/1ÿÿÿÿÿÿÿÿÿÓ
J__inference_lstm_cell_21_layer_call_and_return_conditional_losses_39105578./0¢
y¢v
!
inputsÿÿÿÿÿÿÿÿÿ
M¢J
# 
states/0ÿÿÿÿÿÿÿÿÿÅ
# 
states/1ÿÿÿÿÿÿÿÿÿÅ
p 
ª "v¢s
l¢i

0/0ÿÿÿÿÿÿÿÿÿÅ
GD
 
0/1/0ÿÿÿÿÿÿÿÿÿÅ
 
0/1/1ÿÿÿÿÿÿÿÿÿÅ
 Ó
J__inference_lstm_cell_21_layer_call_and_return_conditional_losses_39105610./0¢
y¢v
!
inputsÿÿÿÿÿÿÿÿÿ
M¢J
# 
states/0ÿÿÿÿÿÿÿÿÿÅ
# 
states/1ÿÿÿÿÿÿÿÿÿÅ
p
ª "v¢s
l¢i

0/0ÿÿÿÿÿÿÿÿÿÅ
GD
 
0/1/0ÿÿÿÿÿÿÿÿÿÅ
 
0/1/1ÿÿÿÿÿÿÿÿÿÅ
 ¨
/__inference_lstm_cell_21_layer_call_fn_39105627ô./0¢
y¢v
!
inputsÿÿÿÿÿÿÿÿÿ
M¢J
# 
states/0ÿÿÿÿÿÿÿÿÿÅ
# 
states/1ÿÿÿÿÿÿÿÿÿÅ
p 
ª "f¢c

0ÿÿÿÿÿÿÿÿÿÅ
C@

1/0ÿÿÿÿÿÿÿÿÿÅ

1/1ÿÿÿÿÿÿÿÿÿÅ¨
/__inference_lstm_cell_21_layer_call_fn_39105644ô./0¢
y¢v
!
inputsÿÿÿÿÿÿÿÿÿ
M¢J
# 
states/0ÿÿÿÿÿÿÿÿÿÅ
# 
states/1ÿÿÿÿÿÿÿÿÿÅ
p
ª "f¢c

0ÿÿÿÿÿÿÿÿÿÅ
C@

1/0ÿÿÿÿÿÿÿÿÿÅ

1/1ÿÿÿÿÿÿÿÿÿÅÈ
K__inference_sequential_10_layer_call_and_return_conditional_losses_39103294y+,-./0 !B¢?
8¢5
+(
lstm_20_inputÿÿÿÿÿÿÿÿÿ]
p 

 
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ
 È
K__inference_sequential_10_layer_call_and_return_conditional_losses_39103319y+,-./0 !B¢?
8¢5
+(
lstm_20_inputÿÿÿÿÿÿÿÿÿ]
p

 
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ
 Á
K__inference_sequential_10_layer_call_and_return_conditional_losses_39103675r+,-./0 !;¢8
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
K__inference_sequential_10_layer_call_and_return_conditional_losses_39104016r+,-./0 !;¢8
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
0__inference_sequential_10_layer_call_fn_39102779l+,-./0 !B¢?
8¢5
+(
lstm_20_inputÿÿÿÿÿÿÿÿÿ]
p 

 
ª "ÿÿÿÿÿÿÿÿÿ 
0__inference_sequential_10_layer_call_fn_39103269l+,-./0 !B¢?
8¢5
+(
lstm_20_inputÿÿÿÿÿÿÿÿÿ]
p

 
ª "ÿÿÿÿÿÿÿÿÿ
0__inference_sequential_10_layer_call_fn_39104037e+,-./0 !;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿ]
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
0__inference_sequential_10_layer_call_fn_39104058e+,-./0 !;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿ]
p

 
ª "ÿÿÿÿÿÿÿÿÿ»
&__inference_signature_wrapper_39103348+,-./0 !K¢H
¢ 
Aª>
<
lstm_20_input+(
lstm_20_inputÿÿÿÿÿÿÿÿÿ]"7ª4
2
dense_10&#
dense_10ÿÿÿÿÿÿÿÿÿ