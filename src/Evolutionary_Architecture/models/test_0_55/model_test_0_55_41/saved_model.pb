²ñ'
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
"serve*2.6.02v2.6.0-rc2-32-g919f693420e8&
{
dense_12/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	Ú* 
shared_namedense_12/kernel
t
#dense_12/kernel/Read/ReadVariableOpReadVariableOpdense_12/kernel*
_output_shapes
:	Ú*
dtype0
r
dense_12/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_12/bias
k
!dense_12/bias/Read/ReadVariableOpReadVariableOpdense_12/bias*
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
lstm_24/lstm_cell_24/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	]*,
shared_namelstm_24/lstm_cell_24/kernel

/lstm_24/lstm_cell_24/kernel/Read/ReadVariableOpReadVariableOplstm_24/lstm_cell_24/kernel*
_output_shapes
:	]*
dtype0
§
%lstm_24/lstm_cell_24/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	`*6
shared_name'%lstm_24/lstm_cell_24/recurrent_kernel
 
9lstm_24/lstm_cell_24/recurrent_kernel/Read/ReadVariableOpReadVariableOp%lstm_24/lstm_cell_24/recurrent_kernel*
_output_shapes
:	`*
dtype0

lstm_24/lstm_cell_24/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_namelstm_24/lstm_cell_24/bias

-lstm_24/lstm_cell_24/bias/Read/ReadVariableOpReadVariableOplstm_24/lstm_cell_24/bias*
_output_shapes	
:*
dtype0

lstm_25/lstm_cell_25/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	`è
*,
shared_namelstm_25/lstm_cell_25/kernel

/lstm_25/lstm_cell_25/kernel/Read/ReadVariableOpReadVariableOplstm_25/lstm_cell_25/kernel*
_output_shapes
:	`è
*
dtype0
¨
%lstm_25/lstm_cell_25/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
Úè
*6
shared_name'%lstm_25/lstm_cell_25/recurrent_kernel
¡
9lstm_25/lstm_cell_25/recurrent_kernel/Read/ReadVariableOpReadVariableOp%lstm_25/lstm_cell_25/recurrent_kernel* 
_output_shapes
:
Úè
*
dtype0

lstm_25/lstm_cell_25/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:è
**
shared_namelstm_25/lstm_cell_25/bias

-lstm_25/lstm_cell_25/bias/Read/ReadVariableOpReadVariableOplstm_25/lstm_cell_25/bias*
_output_shapes	
:è
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
Adam/dense_12/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	Ú*'
shared_nameAdam/dense_12/kernel/m

*Adam/dense_12/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_12/kernel/m*
_output_shapes
:	Ú*
dtype0

Adam/dense_12/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_12/bias/m
y
(Adam/dense_12/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_12/bias/m*
_output_shapes
:*
dtype0
¡
"Adam/lstm_24/lstm_cell_24/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	]*3
shared_name$"Adam/lstm_24/lstm_cell_24/kernel/m

6Adam/lstm_24/lstm_cell_24/kernel/m/Read/ReadVariableOpReadVariableOp"Adam/lstm_24/lstm_cell_24/kernel/m*
_output_shapes
:	]*
dtype0
µ
,Adam/lstm_24/lstm_cell_24/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	`*=
shared_name.,Adam/lstm_24/lstm_cell_24/recurrent_kernel/m
®
@Adam/lstm_24/lstm_cell_24/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp,Adam/lstm_24/lstm_cell_24/recurrent_kernel/m*
_output_shapes
:	`*
dtype0

 Adam/lstm_24/lstm_cell_24/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" Adam/lstm_24/lstm_cell_24/bias/m

4Adam/lstm_24/lstm_cell_24/bias/m/Read/ReadVariableOpReadVariableOp Adam/lstm_24/lstm_cell_24/bias/m*
_output_shapes	
:*
dtype0
¡
"Adam/lstm_25/lstm_cell_25/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	`è
*3
shared_name$"Adam/lstm_25/lstm_cell_25/kernel/m

6Adam/lstm_25/lstm_cell_25/kernel/m/Read/ReadVariableOpReadVariableOp"Adam/lstm_25/lstm_cell_25/kernel/m*
_output_shapes
:	`è
*
dtype0
¶
,Adam/lstm_25/lstm_cell_25/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
Úè
*=
shared_name.,Adam/lstm_25/lstm_cell_25/recurrent_kernel/m
¯
@Adam/lstm_25/lstm_cell_25/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp,Adam/lstm_25/lstm_cell_25/recurrent_kernel/m* 
_output_shapes
:
Úè
*
dtype0

 Adam/lstm_25/lstm_cell_25/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:è
*1
shared_name" Adam/lstm_25/lstm_cell_25/bias/m

4Adam/lstm_25/lstm_cell_25/bias/m/Read/ReadVariableOpReadVariableOp Adam/lstm_25/lstm_cell_25/bias/m*
_output_shapes	
:è
*
dtype0

Adam/dense_12/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	Ú*'
shared_nameAdam/dense_12/kernel/v

*Adam/dense_12/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_12/kernel/v*
_output_shapes
:	Ú*
dtype0

Adam/dense_12/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_12/bias/v
y
(Adam/dense_12/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_12/bias/v*
_output_shapes
:*
dtype0
¡
"Adam/lstm_24/lstm_cell_24/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	]*3
shared_name$"Adam/lstm_24/lstm_cell_24/kernel/v

6Adam/lstm_24/lstm_cell_24/kernel/v/Read/ReadVariableOpReadVariableOp"Adam/lstm_24/lstm_cell_24/kernel/v*
_output_shapes
:	]*
dtype0
µ
,Adam/lstm_24/lstm_cell_24/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	`*=
shared_name.,Adam/lstm_24/lstm_cell_24/recurrent_kernel/v
®
@Adam/lstm_24/lstm_cell_24/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp,Adam/lstm_24/lstm_cell_24/recurrent_kernel/v*
_output_shapes
:	`*
dtype0

 Adam/lstm_24/lstm_cell_24/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" Adam/lstm_24/lstm_cell_24/bias/v

4Adam/lstm_24/lstm_cell_24/bias/v/Read/ReadVariableOpReadVariableOp Adam/lstm_24/lstm_cell_24/bias/v*
_output_shapes	
:*
dtype0
¡
"Adam/lstm_25/lstm_cell_25/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	`è
*3
shared_name$"Adam/lstm_25/lstm_cell_25/kernel/v

6Adam/lstm_25/lstm_cell_25/kernel/v/Read/ReadVariableOpReadVariableOp"Adam/lstm_25/lstm_cell_25/kernel/v*
_output_shapes
:	`è
*
dtype0
¶
,Adam/lstm_25/lstm_cell_25/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
Úè
*=
shared_name.,Adam/lstm_25/lstm_cell_25/recurrent_kernel/v
¯
@Adam/lstm_25/lstm_cell_25/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp,Adam/lstm_25/lstm_cell_25/recurrent_kernel/v* 
_output_shapes
:
Úè
*
dtype0

 Adam/lstm_25/lstm_cell_25/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:è
*1
shared_name" Adam/lstm_25/lstm_cell_25/bias/v

4Adam/lstm_25/lstm_cell_25/bias/v/Read/ReadVariableOpReadVariableOp Adam/lstm_25/lstm_cell_25/bias/v*
_output_shapes	
:è
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
VARIABLE_VALUEdense_12/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_12/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
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
VARIABLE_VALUElstm_24/lstm_cell_24/kernel0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUE
ki
VARIABLE_VALUE%lstm_24/lstm_cell_24/recurrent_kernel0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUElstm_24/lstm_cell_24/bias0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUElstm_25/lstm_cell_25/kernel0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUE
ki
VARIABLE_VALUE%lstm_25/lstm_cell_25/recurrent_kernel0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUElstm_25/lstm_cell_25/bias0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUE
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
VARIABLE_VALUEAdam/dense_12/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_12/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE"Adam/lstm_24/lstm_cell_24/kernel/mLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE,Adam/lstm_24/lstm_cell_24/recurrent_kernel/mLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE Adam/lstm_24/lstm_cell_24/bias/mLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE"Adam/lstm_25/lstm_cell_25/kernel/mLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE,Adam/lstm_25/lstm_cell_25/recurrent_kernel/mLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE Adam/lstm_25/lstm_cell_25/bias/mLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_12/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_12/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE"Adam/lstm_24/lstm_cell_24/kernel/vLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE,Adam/lstm_24/lstm_cell_24/recurrent_kernel/vLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE Adam/lstm_24/lstm_cell_24/bias/vLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE"Adam/lstm_25/lstm_cell_25/kernel/vLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE,Adam/lstm_25/lstm_cell_25/recurrent_kernel/vLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE Adam/lstm_25/lstm_cell_25/bias/vLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

serving_default_lstm_24_inputPlaceholder*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]*
dtype0* 
shape:ÿÿÿÿÿÿÿÿÿ]
¬
StatefulPartitionedCallStatefulPartitionedCallserving_default_lstm_24_inputlstm_24/lstm_cell_24/kernel%lstm_24/lstm_cell_24/recurrent_kernellstm_24/lstm_cell_24/biaslstm_25/lstm_cell_25/kernel%lstm_25/lstm_cell_25/recurrent_kernellstm_25/lstm_cell_25/biasdense_12/kerneldense_12/bias*
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
&__inference_signature_wrapper_39971850
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
Á
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#dense_12/kernel/Read/ReadVariableOp!dense_12/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp/lstm_24/lstm_cell_24/kernel/Read/ReadVariableOp9lstm_24/lstm_cell_24/recurrent_kernel/Read/ReadVariableOp-lstm_24/lstm_cell_24/bias/Read/ReadVariableOp/lstm_25/lstm_cell_25/kernel/Read/ReadVariableOp9lstm_25/lstm_cell_25/recurrent_kernel/Read/ReadVariableOp-lstm_25/lstm_cell_25/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp*Adam/dense_12/kernel/m/Read/ReadVariableOp(Adam/dense_12/bias/m/Read/ReadVariableOp6Adam/lstm_24/lstm_cell_24/kernel/m/Read/ReadVariableOp@Adam/lstm_24/lstm_cell_24/recurrent_kernel/m/Read/ReadVariableOp4Adam/lstm_24/lstm_cell_24/bias/m/Read/ReadVariableOp6Adam/lstm_25/lstm_cell_25/kernel/m/Read/ReadVariableOp@Adam/lstm_25/lstm_cell_25/recurrent_kernel/m/Read/ReadVariableOp4Adam/lstm_25/lstm_cell_25/bias/m/Read/ReadVariableOp*Adam/dense_12/kernel/v/Read/ReadVariableOp(Adam/dense_12/bias/v/Read/ReadVariableOp6Adam/lstm_24/lstm_cell_24/kernel/v/Read/ReadVariableOp@Adam/lstm_24/lstm_cell_24/recurrent_kernel/v/Read/ReadVariableOp4Adam/lstm_24/lstm_cell_24/bias/v/Read/ReadVariableOp6Adam/lstm_25/lstm_cell_25/kernel/v/Read/ReadVariableOp@Adam/lstm_25/lstm_cell_25/recurrent_kernel/v/Read/ReadVariableOp4Adam/lstm_25/lstm_cell_25/bias/v/Read/ReadVariableOpConst*.
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
!__inference__traced_save_39974268
¨	
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_12/kerneldense_12/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratelstm_24/lstm_cell_24/kernel%lstm_24/lstm_cell_24/recurrent_kernellstm_24/lstm_cell_24/biaslstm_25/lstm_cell_25/kernel%lstm_25/lstm_cell_25/recurrent_kernellstm_25/lstm_cell_25/biastotalcounttotal_1count_1Adam/dense_12/kernel/mAdam/dense_12/bias/m"Adam/lstm_24/lstm_cell_24/kernel/m,Adam/lstm_24/lstm_cell_24/recurrent_kernel/m Adam/lstm_24/lstm_cell_24/bias/m"Adam/lstm_25/lstm_cell_25/kernel/m,Adam/lstm_25/lstm_cell_25/recurrent_kernel/m Adam/lstm_25/lstm_cell_25/bias/mAdam/dense_12/kernel/vAdam/dense_12/bias/v"Adam/lstm_24/lstm_cell_24/kernel/v,Adam/lstm_24/lstm_cell_24/recurrent_kernel/v Adam/lstm_24/lstm_cell_24/bias/v"Adam/lstm_25/lstm_cell_25/kernel/v,Adam/lstm_25/lstm_cell_25/recurrent_kernel/v Adam/lstm_25/lstm_cell_25/bias/v*-
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
$__inference__traced_restore_39974377ø÷$
ç[

E__inference_lstm_24_layer_call_and_return_conditional_losses_39971674

inputs>
+lstm_cell_24_matmul_readvariableop_resource:	]@
-lstm_cell_24_matmul_1_readvariableop_resource:	`;
,lstm_cell_24_biasadd_readvariableop_resource:	
identity¢#lstm_cell_24/BiasAdd/ReadVariableOp¢"lstm_cell_24/MatMul/ReadVariableOp¢$lstm_cell_24/MatMul_1/ReadVariableOp¢whileD
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
value	B :`2
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
value	B :`2
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
:ÿÿÿÿÿÿÿÿÿ`2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :`2
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
value	B :`2
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
:ÿÿÿÿÿÿÿÿÿ`2	
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
"lstm_cell_24/MatMul/ReadVariableOpReadVariableOp+lstm_cell_24_matmul_readvariableop_resource*
_output_shapes
:	]*
dtype02$
"lstm_cell_24/MatMul/ReadVariableOp­
lstm_cell_24/MatMulMatMulstrided_slice_2:output:0*lstm_cell_24/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_24/MatMul»
$lstm_cell_24/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_24_matmul_1_readvariableop_resource*
_output_shapes
:	`*
dtype02&
$lstm_cell_24/MatMul_1/ReadVariableOp©
lstm_cell_24/MatMul_1MatMulzeros:output:0,lstm_cell_24/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_24/MatMul_1 
lstm_cell_24/addAddV2lstm_cell_24/MatMul:product:0lstm_cell_24/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_24/add´
#lstm_cell_24/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_24_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02%
#lstm_cell_24/BiasAdd/ReadVariableOp­
lstm_cell_24/BiasAddBiasAddlstm_cell_24/add:z:0+lstm_cell_24/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_24/BiasAdd~
lstm_cell_24/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_24/split/split_dimó
lstm_cell_24/splitSplit%lstm_cell_24/split/split_dim:output:0lstm_cell_24/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ`:ÿÿÿÿÿÿÿÿÿ`:ÿÿÿÿÿÿÿÿÿ`:ÿÿÿÿÿÿÿÿÿ`*
	num_split2
lstm_cell_24/split
lstm_cell_24/SigmoidSigmoidlstm_cell_24/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
lstm_cell_24/Sigmoid
lstm_cell_24/Sigmoid_1Sigmoidlstm_cell_24/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
lstm_cell_24/Sigmoid_1
lstm_cell_24/mulMullstm_cell_24/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
lstm_cell_24/mul}
lstm_cell_24/ReluRelulstm_cell_24/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
lstm_cell_24/Relu
lstm_cell_24/mul_1Mullstm_cell_24/Sigmoid:y:0lstm_cell_24/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
lstm_cell_24/mul_1
lstm_cell_24/add_1AddV2lstm_cell_24/mul:z:0lstm_cell_24/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
lstm_cell_24/add_1
lstm_cell_24/Sigmoid_2Sigmoidlstm_cell_24/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
lstm_cell_24/Sigmoid_2|
lstm_cell_24/Relu_1Relulstm_cell_24/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
lstm_cell_24/Relu_1 
lstm_cell_24/mul_2Mullstm_cell_24/Sigmoid_2:y:0!lstm_cell_24/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
lstm_cell_24/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ`   2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_24_matmul_readvariableop_resource-lstm_cell_24_matmul_1_readvariableop_resource,lstm_cell_24_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ`:ÿÿÿÿÿÿÿÿÿ`: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_39971590*
condR
while_cond_39971589*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ`:ÿÿÿÿÿÿÿÿÿ`: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ`   22
0TensorArrayV2Stack/TensorListStack/element_shapeè
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`*
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
:ÿÿÿÿÿÿÿÿÿ`*
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
:ÿÿÿÿÿÿÿÿÿ`2
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
:ÿÿÿÿÿÿÿÿÿ`2

IdentityÈ
NoOpNoOp$^lstm_cell_24/BiasAdd/ReadVariableOp#^lstm_cell_24/MatMul/ReadVariableOp%^lstm_cell_24/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ]: : : 2J
#lstm_cell_24/BiasAdd/ReadVariableOp#lstm_cell_24/BiasAdd/ReadVariableOp2H
"lstm_cell_24/MatMul/ReadVariableOp"lstm_cell_24/MatMul/ReadVariableOp2L
$lstm_cell_24/MatMul_1/ReadVariableOp$lstm_cell_24/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
 
_user_specified_nameinputs
åJ
Ô

lstm_25_while_body_39972399,
(lstm_25_while_lstm_25_while_loop_counter2
.lstm_25_while_lstm_25_while_maximum_iterations
lstm_25_while_placeholder
lstm_25_while_placeholder_1
lstm_25_while_placeholder_2
lstm_25_while_placeholder_3+
'lstm_25_while_lstm_25_strided_slice_1_0g
clstm_25_while_tensorarrayv2read_tensorlistgetitem_lstm_25_tensorarrayunstack_tensorlistfromtensor_0N
;lstm_25_while_lstm_cell_25_matmul_readvariableop_resource_0:	`è
Q
=lstm_25_while_lstm_cell_25_matmul_1_readvariableop_resource_0:
Úè
K
<lstm_25_while_lstm_cell_25_biasadd_readvariableop_resource_0:	è

lstm_25_while_identity
lstm_25_while_identity_1
lstm_25_while_identity_2
lstm_25_while_identity_3
lstm_25_while_identity_4
lstm_25_while_identity_5)
%lstm_25_while_lstm_25_strided_slice_1e
alstm_25_while_tensorarrayv2read_tensorlistgetitem_lstm_25_tensorarrayunstack_tensorlistfromtensorL
9lstm_25_while_lstm_cell_25_matmul_readvariableop_resource:	`è
O
;lstm_25_while_lstm_cell_25_matmul_1_readvariableop_resource:
Úè
I
:lstm_25_while_lstm_cell_25_biasadd_readvariableop_resource:	è
¢1lstm_25/while/lstm_cell_25/BiasAdd/ReadVariableOp¢0lstm_25/while/lstm_cell_25/MatMul/ReadVariableOp¢2lstm_25/while/lstm_cell_25/MatMul_1/ReadVariableOpÓ
?lstm_25/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ`   2A
?lstm_25/while/TensorArrayV2Read/TensorListGetItem/element_shape
1lstm_25/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemclstm_25_while_tensorarrayv2read_tensorlistgetitem_lstm_25_tensorarrayunstack_tensorlistfromtensor_0lstm_25_while_placeholderHlstm_25/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`*
element_dtype023
1lstm_25/while/TensorArrayV2Read/TensorListGetItemá
0lstm_25/while/lstm_cell_25/MatMul/ReadVariableOpReadVariableOp;lstm_25_while_lstm_cell_25_matmul_readvariableop_resource_0*
_output_shapes
:	`è
*
dtype022
0lstm_25/while/lstm_cell_25/MatMul/ReadVariableOp÷
!lstm_25/while/lstm_cell_25/MatMulMatMul8lstm_25/while/TensorArrayV2Read/TensorListGetItem:item:08lstm_25/while/lstm_cell_25/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè
2#
!lstm_25/while/lstm_cell_25/MatMulè
2lstm_25/while/lstm_cell_25/MatMul_1/ReadVariableOpReadVariableOp=lstm_25_while_lstm_cell_25_matmul_1_readvariableop_resource_0* 
_output_shapes
:
Úè
*
dtype024
2lstm_25/while/lstm_cell_25/MatMul_1/ReadVariableOpà
#lstm_25/while/lstm_cell_25/MatMul_1MatMullstm_25_while_placeholder_2:lstm_25/while/lstm_cell_25/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè
2%
#lstm_25/while/lstm_cell_25/MatMul_1Ø
lstm_25/while/lstm_cell_25/addAddV2+lstm_25/while/lstm_cell_25/MatMul:product:0-lstm_25/while/lstm_cell_25/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè
2 
lstm_25/while/lstm_cell_25/addà
1lstm_25/while/lstm_cell_25/BiasAdd/ReadVariableOpReadVariableOp<lstm_25_while_lstm_cell_25_biasadd_readvariableop_resource_0*
_output_shapes	
:è
*
dtype023
1lstm_25/while/lstm_cell_25/BiasAdd/ReadVariableOpå
"lstm_25/while/lstm_cell_25/BiasAddBiasAdd"lstm_25/while/lstm_cell_25/add:z:09lstm_25/while/lstm_cell_25/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè
2$
"lstm_25/while/lstm_cell_25/BiasAdd
*lstm_25/while/lstm_cell_25/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2,
*lstm_25/while/lstm_cell_25/split/split_dim¯
 lstm_25/while/lstm_cell_25/splitSplit3lstm_25/while/lstm_cell_25/split/split_dim:output:0+lstm_25/while/lstm_cell_25/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿÚ:ÿÿÿÿÿÿÿÿÿÚ:ÿÿÿÿÿÿÿÿÿÚ:ÿÿÿÿÿÿÿÿÿÚ*
	num_split2"
 lstm_25/while/lstm_cell_25/split±
"lstm_25/while/lstm_cell_25/SigmoidSigmoid)lstm_25/while/lstm_cell_25/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2$
"lstm_25/while/lstm_cell_25/Sigmoidµ
$lstm_25/while/lstm_cell_25/Sigmoid_1Sigmoid)lstm_25/while/lstm_cell_25/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2&
$lstm_25/while/lstm_cell_25/Sigmoid_1Á
lstm_25/while/lstm_cell_25/mulMul(lstm_25/while/lstm_cell_25/Sigmoid_1:y:0lstm_25_while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2 
lstm_25/while/lstm_cell_25/mul¨
lstm_25/while/lstm_cell_25/ReluRelu)lstm_25/while/lstm_cell_25/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2!
lstm_25/while/lstm_cell_25/ReluÕ
 lstm_25/while/lstm_cell_25/mul_1Mul&lstm_25/while/lstm_cell_25/Sigmoid:y:0-lstm_25/while/lstm_cell_25/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2"
 lstm_25/while/lstm_cell_25/mul_1Ê
 lstm_25/while/lstm_cell_25/add_1AddV2"lstm_25/while/lstm_cell_25/mul:z:0$lstm_25/while/lstm_cell_25/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2"
 lstm_25/while/lstm_cell_25/add_1µ
$lstm_25/while/lstm_cell_25/Sigmoid_2Sigmoid)lstm_25/while/lstm_cell_25/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2&
$lstm_25/while/lstm_cell_25/Sigmoid_2§
!lstm_25/while/lstm_cell_25/Relu_1Relu$lstm_25/while/lstm_cell_25/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2#
!lstm_25/while/lstm_cell_25/Relu_1Ù
 lstm_25/while/lstm_cell_25/mul_2Mul(lstm_25/while/lstm_cell_25/Sigmoid_2:y:0/lstm_25/while/lstm_cell_25/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2"
 lstm_25/while/lstm_cell_25/mul_2
2lstm_25/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_25_while_placeholder_1lstm_25_while_placeholder$lstm_25/while/lstm_cell_25/mul_2:z:0*
_output_shapes
: *
element_dtype024
2lstm_25/while/TensorArrayV2Write/TensorListSetIteml
lstm_25/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_25/while/add/y
lstm_25/while/addAddV2lstm_25_while_placeholderlstm_25/while/add/y:output:0*
T0*
_output_shapes
: 2
lstm_25/while/addp
lstm_25/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_25/while/add_1/y
lstm_25/while/add_1AddV2(lstm_25_while_lstm_25_while_loop_counterlstm_25/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lstm_25/while/add_1
lstm_25/while/IdentityIdentitylstm_25/while/add_1:z:0^lstm_25/while/NoOp*
T0*
_output_shapes
: 2
lstm_25/while/Identity¦
lstm_25/while/Identity_1Identity.lstm_25_while_lstm_25_while_maximum_iterations^lstm_25/while/NoOp*
T0*
_output_shapes
: 2
lstm_25/while/Identity_1
lstm_25/while/Identity_2Identitylstm_25/while/add:z:0^lstm_25/while/NoOp*
T0*
_output_shapes
: 2
lstm_25/while/Identity_2º
lstm_25/while/Identity_3IdentityBlstm_25/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_25/while/NoOp*
T0*
_output_shapes
: 2
lstm_25/while/Identity_3®
lstm_25/while/Identity_4Identity$lstm_25/while/lstm_cell_25/mul_2:z:0^lstm_25/while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
lstm_25/while/Identity_4®
lstm_25/while/Identity_5Identity$lstm_25/while/lstm_cell_25/add_1:z:0^lstm_25/while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
lstm_25/while/Identity_5
lstm_25/while/NoOpNoOp2^lstm_25/while/lstm_cell_25/BiasAdd/ReadVariableOp1^lstm_25/while/lstm_cell_25/MatMul/ReadVariableOp3^lstm_25/while/lstm_cell_25/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
lstm_25/while/NoOp"9
lstm_25_while_identitylstm_25/while/Identity:output:0"=
lstm_25_while_identity_1!lstm_25/while/Identity_1:output:0"=
lstm_25_while_identity_2!lstm_25/while/Identity_2:output:0"=
lstm_25_while_identity_3!lstm_25/while/Identity_3:output:0"=
lstm_25_while_identity_4!lstm_25/while/Identity_4:output:0"=
lstm_25_while_identity_5!lstm_25/while/Identity_5:output:0"P
%lstm_25_while_lstm_25_strided_slice_1'lstm_25_while_lstm_25_strided_slice_1_0"z
:lstm_25_while_lstm_cell_25_biasadd_readvariableop_resource<lstm_25_while_lstm_cell_25_biasadd_readvariableop_resource_0"|
;lstm_25_while_lstm_cell_25_matmul_1_readvariableop_resource=lstm_25_while_lstm_cell_25_matmul_1_readvariableop_resource_0"x
9lstm_25_while_lstm_cell_25_matmul_readvariableop_resource;lstm_25_while_lstm_cell_25_matmul_readvariableop_resource_0"È
alstm_25_while_tensorarrayv2read_tensorlistgetitem_lstm_25_tensorarrayunstack_tensorlistfromtensorclstm_25_while_tensorarrayv2read_tensorlistgetitem_lstm_25_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿÚ:ÿÿÿÿÿÿÿÿÿÚ: : : : : 2f
1lstm_25/while/lstm_cell_25/BiasAdd/ReadVariableOp1lstm_25/while/lstm_cell_25/BiasAdd/ReadVariableOp2d
0lstm_25/while/lstm_cell_25/MatMul/ReadVariableOp0lstm_25/while/lstm_cell_25/MatMul/ReadVariableOp2h
2lstm_25/while/lstm_cell_25/MatMul_1/ReadVariableOp2lstm_25/while/lstm_cell_25/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
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
Î^

)sequential_12_lstm_24_while_body_39969367H
Dsequential_12_lstm_24_while_sequential_12_lstm_24_while_loop_counterN
Jsequential_12_lstm_24_while_sequential_12_lstm_24_while_maximum_iterations+
'sequential_12_lstm_24_while_placeholder-
)sequential_12_lstm_24_while_placeholder_1-
)sequential_12_lstm_24_while_placeholder_2-
)sequential_12_lstm_24_while_placeholder_3G
Csequential_12_lstm_24_while_sequential_12_lstm_24_strided_slice_1_0
sequential_12_lstm_24_while_tensorarrayv2read_tensorlistgetitem_sequential_12_lstm_24_tensorarrayunstack_tensorlistfromtensor_0\
Isequential_12_lstm_24_while_lstm_cell_24_matmul_readvariableop_resource_0:	]^
Ksequential_12_lstm_24_while_lstm_cell_24_matmul_1_readvariableop_resource_0:	`Y
Jsequential_12_lstm_24_while_lstm_cell_24_biasadd_readvariableop_resource_0:	(
$sequential_12_lstm_24_while_identity*
&sequential_12_lstm_24_while_identity_1*
&sequential_12_lstm_24_while_identity_2*
&sequential_12_lstm_24_while_identity_3*
&sequential_12_lstm_24_while_identity_4*
&sequential_12_lstm_24_while_identity_5E
Asequential_12_lstm_24_while_sequential_12_lstm_24_strided_slice_1
}sequential_12_lstm_24_while_tensorarrayv2read_tensorlistgetitem_sequential_12_lstm_24_tensorarrayunstack_tensorlistfromtensorZ
Gsequential_12_lstm_24_while_lstm_cell_24_matmul_readvariableop_resource:	]\
Isequential_12_lstm_24_while_lstm_cell_24_matmul_1_readvariableop_resource:	`W
Hsequential_12_lstm_24_while_lstm_cell_24_biasadd_readvariableop_resource:	¢?sequential_12/lstm_24/while/lstm_cell_24/BiasAdd/ReadVariableOp¢>sequential_12/lstm_24/while/lstm_cell_24/MatMul/ReadVariableOp¢@sequential_12/lstm_24/while/lstm_cell_24/MatMul_1/ReadVariableOpï
Msequential_12/lstm_24/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ]   2O
Msequential_12/lstm_24/while/TensorArrayV2Read/TensorListGetItem/element_shape×
?sequential_12/lstm_24/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemsequential_12_lstm_24_while_tensorarrayv2read_tensorlistgetitem_sequential_12_lstm_24_tensorarrayunstack_tensorlistfromtensor_0'sequential_12_lstm_24_while_placeholderVsequential_12/lstm_24/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]*
element_dtype02A
?sequential_12/lstm_24/while/TensorArrayV2Read/TensorListGetItem
>sequential_12/lstm_24/while/lstm_cell_24/MatMul/ReadVariableOpReadVariableOpIsequential_12_lstm_24_while_lstm_cell_24_matmul_readvariableop_resource_0*
_output_shapes
:	]*
dtype02@
>sequential_12/lstm_24/while/lstm_cell_24/MatMul/ReadVariableOp¯
/sequential_12/lstm_24/while/lstm_cell_24/MatMulMatMulFsequential_12/lstm_24/while/TensorArrayV2Read/TensorListGetItem:item:0Fsequential_12/lstm_24/while/lstm_cell_24/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ21
/sequential_12/lstm_24/while/lstm_cell_24/MatMul
@sequential_12/lstm_24/while/lstm_cell_24/MatMul_1/ReadVariableOpReadVariableOpKsequential_12_lstm_24_while_lstm_cell_24_matmul_1_readvariableop_resource_0*
_output_shapes
:	`*
dtype02B
@sequential_12/lstm_24/while/lstm_cell_24/MatMul_1/ReadVariableOp
1sequential_12/lstm_24/while/lstm_cell_24/MatMul_1MatMul)sequential_12_lstm_24_while_placeholder_2Hsequential_12/lstm_24/while/lstm_cell_24/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ23
1sequential_12/lstm_24/while/lstm_cell_24/MatMul_1
,sequential_12/lstm_24/while/lstm_cell_24/addAddV29sequential_12/lstm_24/while/lstm_cell_24/MatMul:product:0;sequential_12/lstm_24/while/lstm_cell_24/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2.
,sequential_12/lstm_24/while/lstm_cell_24/add
?sequential_12/lstm_24/while/lstm_cell_24/BiasAdd/ReadVariableOpReadVariableOpJsequential_12_lstm_24_while_lstm_cell_24_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02A
?sequential_12/lstm_24/while/lstm_cell_24/BiasAdd/ReadVariableOp
0sequential_12/lstm_24/while/lstm_cell_24/BiasAddBiasAdd0sequential_12/lstm_24/while/lstm_cell_24/add:z:0Gsequential_12/lstm_24/while/lstm_cell_24/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
0sequential_12/lstm_24/while/lstm_cell_24/BiasAdd¶
8sequential_12/lstm_24/while/lstm_cell_24/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2:
8sequential_12/lstm_24/while/lstm_cell_24/split/split_dimã
.sequential_12/lstm_24/while/lstm_cell_24/splitSplitAsequential_12/lstm_24/while/lstm_cell_24/split/split_dim:output:09sequential_12/lstm_24/while/lstm_cell_24/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ`:ÿÿÿÿÿÿÿÿÿ`:ÿÿÿÿÿÿÿÿÿ`:ÿÿÿÿÿÿÿÿÿ`*
	num_split20
.sequential_12/lstm_24/while/lstm_cell_24/splitÚ
0sequential_12/lstm_24/while/lstm_cell_24/SigmoidSigmoid7sequential_12/lstm_24/while/lstm_cell_24/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`22
0sequential_12/lstm_24/while/lstm_cell_24/SigmoidÞ
2sequential_12/lstm_24/while/lstm_cell_24/Sigmoid_1Sigmoid7sequential_12/lstm_24/while/lstm_cell_24/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`24
2sequential_12/lstm_24/while/lstm_cell_24/Sigmoid_1ø
,sequential_12/lstm_24/while/lstm_cell_24/mulMul6sequential_12/lstm_24/while/lstm_cell_24/Sigmoid_1:y:0)sequential_12_lstm_24_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2.
,sequential_12/lstm_24/while/lstm_cell_24/mulÑ
-sequential_12/lstm_24/while/lstm_cell_24/ReluRelu7sequential_12/lstm_24/while/lstm_cell_24/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2/
-sequential_12/lstm_24/while/lstm_cell_24/Relu
.sequential_12/lstm_24/while/lstm_cell_24/mul_1Mul4sequential_12/lstm_24/while/lstm_cell_24/Sigmoid:y:0;sequential_12/lstm_24/while/lstm_cell_24/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`20
.sequential_12/lstm_24/while/lstm_cell_24/mul_1
.sequential_12/lstm_24/while/lstm_cell_24/add_1AddV20sequential_12/lstm_24/while/lstm_cell_24/mul:z:02sequential_12/lstm_24/while/lstm_cell_24/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`20
.sequential_12/lstm_24/while/lstm_cell_24/add_1Þ
2sequential_12/lstm_24/while/lstm_cell_24/Sigmoid_2Sigmoid7sequential_12/lstm_24/while/lstm_cell_24/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`24
2sequential_12/lstm_24/while/lstm_cell_24/Sigmoid_2Ð
/sequential_12/lstm_24/while/lstm_cell_24/Relu_1Relu2sequential_12/lstm_24/while/lstm_cell_24/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`21
/sequential_12/lstm_24/while/lstm_cell_24/Relu_1
.sequential_12/lstm_24/while/lstm_cell_24/mul_2Mul6sequential_12/lstm_24/while/lstm_cell_24/Sigmoid_2:y:0=sequential_12/lstm_24/while/lstm_cell_24/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`20
.sequential_12/lstm_24/while/lstm_cell_24/mul_2Î
@sequential_12/lstm_24/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem)sequential_12_lstm_24_while_placeholder_1'sequential_12_lstm_24_while_placeholder2sequential_12/lstm_24/while/lstm_cell_24/mul_2:z:0*
_output_shapes
: *
element_dtype02B
@sequential_12/lstm_24/while/TensorArrayV2Write/TensorListSetItem
!sequential_12/lstm_24/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2#
!sequential_12/lstm_24/while/add/yÁ
sequential_12/lstm_24/while/addAddV2'sequential_12_lstm_24_while_placeholder*sequential_12/lstm_24/while/add/y:output:0*
T0*
_output_shapes
: 2!
sequential_12/lstm_24/while/add
#sequential_12/lstm_24/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2%
#sequential_12/lstm_24/while/add_1/yä
!sequential_12/lstm_24/while/add_1AddV2Dsequential_12_lstm_24_while_sequential_12_lstm_24_while_loop_counter,sequential_12/lstm_24/while/add_1/y:output:0*
T0*
_output_shapes
: 2#
!sequential_12/lstm_24/while/add_1Ã
$sequential_12/lstm_24/while/IdentityIdentity%sequential_12/lstm_24/while/add_1:z:0!^sequential_12/lstm_24/while/NoOp*
T0*
_output_shapes
: 2&
$sequential_12/lstm_24/while/Identityì
&sequential_12/lstm_24/while/Identity_1IdentityJsequential_12_lstm_24_while_sequential_12_lstm_24_while_maximum_iterations!^sequential_12/lstm_24/while/NoOp*
T0*
_output_shapes
: 2(
&sequential_12/lstm_24/while/Identity_1Å
&sequential_12/lstm_24/while/Identity_2Identity#sequential_12/lstm_24/while/add:z:0!^sequential_12/lstm_24/while/NoOp*
T0*
_output_shapes
: 2(
&sequential_12/lstm_24/while/Identity_2ò
&sequential_12/lstm_24/while/Identity_3IdentityPsequential_12/lstm_24/while/TensorArrayV2Write/TensorListSetItem:output_handle:0!^sequential_12/lstm_24/while/NoOp*
T0*
_output_shapes
: 2(
&sequential_12/lstm_24/while/Identity_3å
&sequential_12/lstm_24/while/Identity_4Identity2sequential_12/lstm_24/while/lstm_cell_24/mul_2:z:0!^sequential_12/lstm_24/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2(
&sequential_12/lstm_24/while/Identity_4å
&sequential_12/lstm_24/while/Identity_5Identity2sequential_12/lstm_24/while/lstm_cell_24/add_1:z:0!^sequential_12/lstm_24/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2(
&sequential_12/lstm_24/while/Identity_5Ì
 sequential_12/lstm_24/while/NoOpNoOp@^sequential_12/lstm_24/while/lstm_cell_24/BiasAdd/ReadVariableOp?^sequential_12/lstm_24/while/lstm_cell_24/MatMul/ReadVariableOpA^sequential_12/lstm_24/while/lstm_cell_24/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2"
 sequential_12/lstm_24/while/NoOp"U
$sequential_12_lstm_24_while_identity-sequential_12/lstm_24/while/Identity:output:0"Y
&sequential_12_lstm_24_while_identity_1/sequential_12/lstm_24/while/Identity_1:output:0"Y
&sequential_12_lstm_24_while_identity_2/sequential_12/lstm_24/while/Identity_2:output:0"Y
&sequential_12_lstm_24_while_identity_3/sequential_12/lstm_24/while/Identity_3:output:0"Y
&sequential_12_lstm_24_while_identity_4/sequential_12/lstm_24/while/Identity_4:output:0"Y
&sequential_12_lstm_24_while_identity_5/sequential_12/lstm_24/while/Identity_5:output:0"
Hsequential_12_lstm_24_while_lstm_cell_24_biasadd_readvariableop_resourceJsequential_12_lstm_24_while_lstm_cell_24_biasadd_readvariableop_resource_0"
Isequential_12_lstm_24_while_lstm_cell_24_matmul_1_readvariableop_resourceKsequential_12_lstm_24_while_lstm_cell_24_matmul_1_readvariableop_resource_0"
Gsequential_12_lstm_24_while_lstm_cell_24_matmul_readvariableop_resourceIsequential_12_lstm_24_while_lstm_cell_24_matmul_readvariableop_resource_0"
Asequential_12_lstm_24_while_sequential_12_lstm_24_strided_slice_1Csequential_12_lstm_24_while_sequential_12_lstm_24_strided_slice_1_0"
}sequential_12_lstm_24_while_tensorarrayv2read_tensorlistgetitem_sequential_12_lstm_24_tensorarrayunstack_tensorlistfromtensorsequential_12_lstm_24_while_tensorarrayv2read_tensorlistgetitem_sequential_12_lstm_24_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ`:ÿÿÿÿÿÿÿÿÿ`: : : : : 2
?sequential_12/lstm_24/while/lstm_cell_24/BiasAdd/ReadVariableOp?sequential_12/lstm_24/while/lstm_cell_24/BiasAdd/ReadVariableOp2
>sequential_12/lstm_24/while/lstm_cell_24/MatMul/ReadVariableOp>sequential_12/lstm_24/while/lstm_cell_24/MatMul/ReadVariableOp2
@sequential_12/lstm_24/while/lstm_cell_24/MatMul_1/ReadVariableOp@sequential_12/lstm_24/while/lstm_cell_24/MatMul_1/ReadVariableOp: 

_output_shapes
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
:ÿÿÿÿÿÿÿÿÿ`:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`:

_output_shapes
: :

_output_shapes
: 
¶
¸
*__inference_lstm_25_layer_call_fn_39973872

inputs
unknown:	`è
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
E__inference_lstm_25_layer_call_and_return_conditional_losses_399712092
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
:ÿÿÿÿÿÿÿÿÿ`: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
 
_user_specified_nameinputs

f
H__inference_dropout_24_layer_call_and_return_conditional_losses_39971057

inputs

identity_1^
IdentityIdentityinputs*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2

Identitym

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ`:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
 
_user_specified_nameinputs
º
ø
/__inference_lstm_cell_24_layer_call_fn_39974048

inputs
states_0
states_1
unknown:	]
	unknown_0:	`
	unknown_1:	
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
9:ÿÿÿÿÿÿÿÿÿ`:ÿÿÿÿÿÿÿÿÿ`:ÿÿÿÿÿÿÿÿÿ`*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_lstm_cell_24_layer_call_and_return_conditional_losses_399698482
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2

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
?:ÿÿÿÿÿÿÿÿÿ]:ÿÿÿÿÿÿÿÿÿ`:ÿÿÿÿÿÿÿÿÿ`: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
 
_user_specified_nameinputs:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
"
_user_specified_name
states/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
"
_user_specified_name
states/1
°?
Ô
while_body_39971394
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_25_matmul_readvariableop_resource_0:	`è
I
5while_lstm_cell_25_matmul_1_readvariableop_resource_0:
Úè
C
4while_lstm_cell_25_biasadd_readvariableop_resource_0:	è

while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_25_matmul_readvariableop_resource:	`è
G
3while_lstm_cell_25_matmul_1_readvariableop_resource:
Úè
A
2while_lstm_cell_25_biasadd_readvariableop_resource:	è
¢)while/lstm_cell_25/BiasAdd/ReadVariableOp¢(while/lstm_cell_25/MatMul/ReadVariableOp¢*while/lstm_cell_25/MatMul_1/ReadVariableOpÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ`   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÓ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemÉ
(while/lstm_cell_25/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_25_matmul_readvariableop_resource_0*
_output_shapes
:	`è
*
dtype02*
(while/lstm_cell_25/MatMul/ReadVariableOp×
while/lstm_cell_25/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_25/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè
2
while/lstm_cell_25/MatMulÐ
*while/lstm_cell_25/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_25_matmul_1_readvariableop_resource_0* 
_output_shapes
:
Úè
*
dtype02,
*while/lstm_cell_25/MatMul_1/ReadVariableOpÀ
while/lstm_cell_25/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_25/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè
2
while/lstm_cell_25/MatMul_1¸
while/lstm_cell_25/addAddV2#while/lstm_cell_25/MatMul:product:0%while/lstm_cell_25/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè
2
while/lstm_cell_25/addÈ
)while/lstm_cell_25/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_25_biasadd_readvariableop_resource_0*
_output_shapes	
:è
*
dtype02+
)while/lstm_cell_25/BiasAdd/ReadVariableOpÅ
while/lstm_cell_25/BiasAddBiasAddwhile/lstm_cell_25/add:z:01while/lstm_cell_25/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè
2
while/lstm_cell_25/BiasAdd
"while/lstm_cell_25/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_25/split/split_dim
while/lstm_cell_25/splitSplit+while/lstm_cell_25/split/split_dim:output:0#while/lstm_cell_25/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿÚ:ÿÿÿÿÿÿÿÿÿÚ:ÿÿÿÿÿÿÿÿÿÚ:ÿÿÿÿÿÿÿÿÿÚ*
	num_split2
while/lstm_cell_25/split
while/lstm_cell_25/SigmoidSigmoid!while/lstm_cell_25/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
while/lstm_cell_25/Sigmoid
while/lstm_cell_25/Sigmoid_1Sigmoid!while/lstm_cell_25/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
while/lstm_cell_25/Sigmoid_1¡
while/lstm_cell_25/mulMul while/lstm_cell_25/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
while/lstm_cell_25/mul
while/lstm_cell_25/ReluRelu!while/lstm_cell_25/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
while/lstm_cell_25/Reluµ
while/lstm_cell_25/mul_1Mulwhile/lstm_cell_25/Sigmoid:y:0%while/lstm_cell_25/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
while/lstm_cell_25/mul_1ª
while/lstm_cell_25/add_1AddV2while/lstm_cell_25/mul:z:0while/lstm_cell_25/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
while/lstm_cell_25/add_1
while/lstm_cell_25/Sigmoid_2Sigmoid!while/lstm_cell_25/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
while/lstm_cell_25/Sigmoid_2
while/lstm_cell_25/Relu_1Reluwhile/lstm_cell_25/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
while/lstm_cell_25/Relu_1¹
while/lstm_cell_25/mul_2Mul while/lstm_cell_25/Sigmoid_2:y:0'while/lstm_cell_25/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
while/lstm_cell_25/mul_2à
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_25/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_25/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_25/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
while/Identity_5Þ

while/NoOpNoOp*^while/lstm_cell_25/BiasAdd/ReadVariableOp)^while/lstm_cell_25/MatMul/ReadVariableOp+^while/lstm_cell_25/MatMul_1/ReadVariableOp*"
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
2while_lstm_cell_25_biasadd_readvariableop_resource4while_lstm_cell_25_biasadd_readvariableop_resource_0"l
3while_lstm_cell_25_matmul_1_readvariableop_resource5while_lstm_cell_25_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_25_matmul_readvariableop_resource3while_lstm_cell_25_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿÚ:ÿÿÿÿÿÿÿÿÿÚ: : : : : 2V
)while/lstm_cell_25/BiasAdd/ReadVariableOp)while/lstm_cell_25/BiasAdd/ReadVariableOp2T
(while/lstm_cell_25/MatMul/ReadVariableOp(while/lstm_cell_25/MatMul/ReadVariableOp2X
*while/lstm_cell_25/MatMul_1/ReadVariableOp*while/lstm_cell_25/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
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
ú%
ñ
while_body_39969716
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_00
while_lstm_cell_24_39969740_0:	]0
while_lstm_cell_24_39969742_0:	`,
while_lstm_cell_24_39969744_0:	
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor.
while_lstm_cell_24_39969740:	].
while_lstm_cell_24_39969742:	`*
while_lstm_cell_24_39969744:	¢*while/lstm_cell_24/StatefulPartitionedCallÃ
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
)while/TensorArrayV2Read/TensorListGetItemé
*while/lstm_cell_24/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_24_39969740_0while_lstm_cell_24_39969742_0while_lstm_cell_24_39969744_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ`:ÿÿÿÿÿÿÿÿÿ`:ÿÿÿÿÿÿÿÿÿ`*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_lstm_cell_24_layer_call_and_return_conditional_losses_399697022,
*while/lstm_cell_24/StatefulPartitionedCall÷
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder3while/lstm_cell_24/StatefulPartitionedCall:output:0*
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
while/Identity_4Identity3while/lstm_cell_24/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
while/Identity_4¤
while/Identity_5Identity3while/lstm_cell_24/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
while/Identity_5

while/NoOpNoOp+^while/lstm_cell_24/StatefulPartitionedCall*"
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
while_lstm_cell_24_39969740while_lstm_cell_24_39969740_0"<
while_lstm_cell_24_39969742while_lstm_cell_24_39969742_0"<
while_lstm_cell_24_39969744while_lstm_cell_24_39969744_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ`:ÿÿÿÿÿÿÿÿÿ`: : : : : 2X
*while/lstm_cell_24/StatefulPartitionedCall*while/lstm_cell_24/StatefulPartitionedCall: 

_output_shapes
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
:ÿÿÿÿÿÿÿÿÿ`:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`:

_output_shapes
: :

_output_shapes
: 
°?
Ô
while_body_39973453
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_25_matmul_readvariableop_resource_0:	`è
I
5while_lstm_cell_25_matmul_1_readvariableop_resource_0:
Úè
C
4while_lstm_cell_25_biasadd_readvariableop_resource_0:	è

while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_25_matmul_readvariableop_resource:	`è
G
3while_lstm_cell_25_matmul_1_readvariableop_resource:
Úè
A
2while_lstm_cell_25_biasadd_readvariableop_resource:	è
¢)while/lstm_cell_25/BiasAdd/ReadVariableOp¢(while/lstm_cell_25/MatMul/ReadVariableOp¢*while/lstm_cell_25/MatMul_1/ReadVariableOpÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ`   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÓ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemÉ
(while/lstm_cell_25/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_25_matmul_readvariableop_resource_0*
_output_shapes
:	`è
*
dtype02*
(while/lstm_cell_25/MatMul/ReadVariableOp×
while/lstm_cell_25/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_25/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè
2
while/lstm_cell_25/MatMulÐ
*while/lstm_cell_25/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_25_matmul_1_readvariableop_resource_0* 
_output_shapes
:
Úè
*
dtype02,
*while/lstm_cell_25/MatMul_1/ReadVariableOpÀ
while/lstm_cell_25/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_25/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè
2
while/lstm_cell_25/MatMul_1¸
while/lstm_cell_25/addAddV2#while/lstm_cell_25/MatMul:product:0%while/lstm_cell_25/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè
2
while/lstm_cell_25/addÈ
)while/lstm_cell_25/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_25_biasadd_readvariableop_resource_0*
_output_shapes	
:è
*
dtype02+
)while/lstm_cell_25/BiasAdd/ReadVariableOpÅ
while/lstm_cell_25/BiasAddBiasAddwhile/lstm_cell_25/add:z:01while/lstm_cell_25/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè
2
while/lstm_cell_25/BiasAdd
"while/lstm_cell_25/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_25/split/split_dim
while/lstm_cell_25/splitSplit+while/lstm_cell_25/split/split_dim:output:0#while/lstm_cell_25/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿÚ:ÿÿÿÿÿÿÿÿÿÚ:ÿÿÿÿÿÿÿÿÿÚ:ÿÿÿÿÿÿÿÿÿÚ*
	num_split2
while/lstm_cell_25/split
while/lstm_cell_25/SigmoidSigmoid!while/lstm_cell_25/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
while/lstm_cell_25/Sigmoid
while/lstm_cell_25/Sigmoid_1Sigmoid!while/lstm_cell_25/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
while/lstm_cell_25/Sigmoid_1¡
while/lstm_cell_25/mulMul while/lstm_cell_25/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
while/lstm_cell_25/mul
while/lstm_cell_25/ReluRelu!while/lstm_cell_25/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
while/lstm_cell_25/Reluµ
while/lstm_cell_25/mul_1Mulwhile/lstm_cell_25/Sigmoid:y:0%while/lstm_cell_25/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
while/lstm_cell_25/mul_1ª
while/lstm_cell_25/add_1AddV2while/lstm_cell_25/mul:z:0while/lstm_cell_25/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
while/lstm_cell_25/add_1
while/lstm_cell_25/Sigmoid_2Sigmoid!while/lstm_cell_25/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
while/lstm_cell_25/Sigmoid_2
while/lstm_cell_25/Relu_1Reluwhile/lstm_cell_25/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
while/lstm_cell_25/Relu_1¹
while/lstm_cell_25/mul_2Mul while/lstm_cell_25/Sigmoid_2:y:0'while/lstm_cell_25/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
while/lstm_cell_25/mul_2à
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_25/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_25/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_25/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
while/Identity_5Þ

while/NoOpNoOp*^while/lstm_cell_25/BiasAdd/ReadVariableOp)^while/lstm_cell_25/MatMul/ReadVariableOp+^while/lstm_cell_25/MatMul_1/ReadVariableOp*"
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
2while_lstm_cell_25_biasadd_readvariableop_resource4while_lstm_cell_25_biasadd_readvariableop_resource_0"l
3while_lstm_cell_25_matmul_1_readvariableop_resource5while_lstm_cell_25_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_25_matmul_readvariableop_resource3while_lstm_cell_25_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿÚ:ÿÿÿÿÿÿÿÿÿÚ: : : : : 2V
)while/lstm_cell_25/BiasAdd/ReadVariableOp)while/lstm_cell_25/BiasAdd/ReadVariableOp2T
(while/lstm_cell_25/MatMul/ReadVariableOp(while/lstm_cell_25/MatMul/ReadVariableOp2X
*while/lstm_cell_25/MatMul_1/ReadVariableOp*while/lstm_cell_25/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
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
²
·
*__inference_lstm_24_layer_call_fn_39973208

inputs
unknown:	]
	unknown_0:	`
	unknown_1:	
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_lstm_24_layer_call_and_return_conditional_losses_399716742
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2

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


J__inference_lstm_cell_25_layer_call_and_return_conditional_losses_39970478

inputs

states
states_11
matmul_readvariableop_resource:	`è
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
:	`è
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
A:ÿÿÿÿÿÿÿÿÿ`:ÿÿÿÿÿÿÿÿÿÚ:ÿÿÿÿÿÿÿÿÿÚ: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
 
_user_specified_nameinputs:PL
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ
 
_user_specified_namestates:PL
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ
 
_user_specified_namestates
à
º
*__inference_lstm_25_layer_call_fn_39973861
inputs_0
unknown:	`è
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
E__inference_lstm_25_layer_call_and_return_conditional_losses_399706252
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
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`
"
_user_specified_name
inputs/0
Æ
À
K__inference_sequential_12_layer_call_and_return_conditional_losses_39971821
lstm_24_input#
lstm_24_39971799:	]#
lstm_24_39971801:	`
lstm_24_39971803:	#
lstm_25_39971807:	`è
$
lstm_25_39971809:
Úè

lstm_25_39971811:	è
$
dense_12_39971815:	Ú
dense_12_39971817:
identity¢ dense_12/StatefulPartitionedCall¢"dropout_24/StatefulPartitionedCall¢"dropout_25/StatefulPartitionedCall¢lstm_24/StatefulPartitionedCall¢lstm_25/StatefulPartitionedCall´
lstm_24/StatefulPartitionedCallStatefulPartitionedCalllstm_24_inputlstm_24_39971799lstm_24_39971801lstm_24_39971803*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_lstm_24_layer_call_and_return_conditional_losses_399716742!
lstm_24/StatefulPartitionedCall
"dropout_24/StatefulPartitionedCallStatefulPartitionedCall(lstm_24/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_dropout_24_layer_call_and_return_conditional_losses_399715072$
"dropout_24/StatefulPartitionedCallÓ
lstm_25/StatefulPartitionedCallStatefulPartitionedCall+dropout_24/StatefulPartitionedCall:output:0lstm_25_39971807lstm_25_39971809lstm_25_39971811*
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
E__inference_lstm_25_layer_call_and_return_conditional_losses_399714782!
lstm_25/StatefulPartitionedCallÀ
"dropout_25/StatefulPartitionedCallStatefulPartitionedCall(lstm_25/StatefulPartitionedCall:output:0#^dropout_24/StatefulPartitionedCall*
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
H__inference_dropout_25_layer_call_and_return_conditional_losses_399713112$
"dropout_25/StatefulPartitionedCallÃ
 dense_12/StatefulPartitionedCallStatefulPartitionedCall+dropout_25/StatefulPartitionedCall:output:0dense_12_39971815dense_12_39971817*
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
F__inference_dense_12_layer_call_and_return_conditional_losses_399712552"
 dense_12/StatefulPartitionedCall
IdentityIdentity)dense_12/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityÿ
NoOpNoOp!^dense_12/StatefulPartitionedCall#^dropout_24/StatefulPartitionedCall#^dropout_25/StatefulPartitionedCall ^lstm_24/StatefulPartitionedCall ^lstm_25/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ]: : : : : : : : 2D
 dense_12/StatefulPartitionedCall dense_12/StatefulPartitionedCall2H
"dropout_24/StatefulPartitionedCall"dropout_24/StatefulPartitionedCall2H
"dropout_25/StatefulPartitionedCall"dropout_25/StatefulPartitionedCall2B
lstm_24/StatefulPartitionedCalllstm_24/StatefulPartitionedCall2B
lstm_25/StatefulPartitionedCalllstm_25/StatefulPartitionedCall:Z V
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
'
_user_specified_namelstm_24_input
Ø
I
-__inference_dropout_25_layer_call_fn_39973905

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
H__inference_dropout_25_layer_call_and_return_conditional_losses_399712222
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
³ù

K__inference_sequential_12_layer_call_and_return_conditional_losses_39972177

inputsF
3lstm_24_lstm_cell_24_matmul_readvariableop_resource:	]H
5lstm_24_lstm_cell_24_matmul_1_readvariableop_resource:	`C
4lstm_24_lstm_cell_24_biasadd_readvariableop_resource:	F
3lstm_25_lstm_cell_25_matmul_readvariableop_resource:	`è
I
5lstm_25_lstm_cell_25_matmul_1_readvariableop_resource:
Úè
C
4lstm_25_lstm_cell_25_biasadd_readvariableop_resource:	è
=
*dense_12_tensordot_readvariableop_resource:	Ú6
(dense_12_biasadd_readvariableop_resource:
identity¢dense_12/BiasAdd/ReadVariableOp¢!dense_12/Tensordot/ReadVariableOp¢+lstm_24/lstm_cell_24/BiasAdd/ReadVariableOp¢*lstm_24/lstm_cell_24/MatMul/ReadVariableOp¢,lstm_24/lstm_cell_24/MatMul_1/ReadVariableOp¢lstm_24/while¢+lstm_25/lstm_cell_25/BiasAdd/ReadVariableOp¢*lstm_25/lstm_cell_25/MatMul/ReadVariableOp¢,lstm_25/lstm_cell_25/MatMul_1/ReadVariableOp¢lstm_25/whileT
lstm_24/ShapeShapeinputs*
T0*
_output_shapes
:2
lstm_24/Shape
lstm_24/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_24/strided_slice/stack
lstm_24/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_24/strided_slice/stack_1
lstm_24/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_24/strided_slice/stack_2
lstm_24/strided_sliceStridedSlicelstm_24/Shape:output:0$lstm_24/strided_slice/stack:output:0&lstm_24/strided_slice/stack_1:output:0&lstm_24/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_24/strided_slicel
lstm_24/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :`2
lstm_24/zeros/mul/y
lstm_24/zeros/mulMullstm_24/strided_slice:output:0lstm_24/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_24/zeros/mulo
lstm_24/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
lstm_24/zeros/Less/y
lstm_24/zeros/LessLesslstm_24/zeros/mul:z:0lstm_24/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_24/zeros/Lessr
lstm_24/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :`2
lstm_24/zeros/packed/1£
lstm_24/zeros/packedPacklstm_24/strided_slice:output:0lstm_24/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_24/zeros/packedo
lstm_24/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_24/zeros/Const
lstm_24/zerosFilllstm_24/zeros/packed:output:0lstm_24/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
lstm_24/zerosp
lstm_24/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :`2
lstm_24/zeros_1/mul/y
lstm_24/zeros_1/mulMullstm_24/strided_slice:output:0lstm_24/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_24/zeros_1/muls
lstm_24/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
lstm_24/zeros_1/Less/y
lstm_24/zeros_1/LessLesslstm_24/zeros_1/mul:z:0lstm_24/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_24/zeros_1/Lessv
lstm_24/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :`2
lstm_24/zeros_1/packed/1©
lstm_24/zeros_1/packedPacklstm_24/strided_slice:output:0!lstm_24/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_24/zeros_1/packeds
lstm_24/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_24/zeros_1/Const
lstm_24/zeros_1Filllstm_24/zeros_1/packed:output:0lstm_24/zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
lstm_24/zeros_1
lstm_24/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_24/transpose/perm
lstm_24/transpose	Transposeinputslstm_24/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]2
lstm_24/transposeg
lstm_24/Shape_1Shapelstm_24/transpose:y:0*
T0*
_output_shapes
:2
lstm_24/Shape_1
lstm_24/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_24/strided_slice_1/stack
lstm_24/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_24/strided_slice_1/stack_1
lstm_24/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_24/strided_slice_1/stack_2
lstm_24/strided_slice_1StridedSlicelstm_24/Shape_1:output:0&lstm_24/strided_slice_1/stack:output:0(lstm_24/strided_slice_1/stack_1:output:0(lstm_24/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_24/strided_slice_1
#lstm_24/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2%
#lstm_24/TensorArrayV2/element_shapeÒ
lstm_24/TensorArrayV2TensorListReserve,lstm_24/TensorArrayV2/element_shape:output:0 lstm_24/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_24/TensorArrayV2Ï
=lstm_24/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ]   2?
=lstm_24/TensorArrayUnstack/TensorListFromTensor/element_shape
/lstm_24/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_24/transpose:y:0Flstm_24/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type021
/lstm_24/TensorArrayUnstack/TensorListFromTensor
lstm_24/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_24/strided_slice_2/stack
lstm_24/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_24/strided_slice_2/stack_1
lstm_24/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_24/strided_slice_2/stack_2¬
lstm_24/strided_slice_2StridedSlicelstm_24/transpose:y:0&lstm_24/strided_slice_2/stack:output:0(lstm_24/strided_slice_2/stack_1:output:0(lstm_24/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]*
shrink_axis_mask2
lstm_24/strided_slice_2Í
*lstm_24/lstm_cell_24/MatMul/ReadVariableOpReadVariableOp3lstm_24_lstm_cell_24_matmul_readvariableop_resource*
_output_shapes
:	]*
dtype02,
*lstm_24/lstm_cell_24/MatMul/ReadVariableOpÍ
lstm_24/lstm_cell_24/MatMulMatMul lstm_24/strided_slice_2:output:02lstm_24/lstm_cell_24/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_24/lstm_cell_24/MatMulÓ
,lstm_24/lstm_cell_24/MatMul_1/ReadVariableOpReadVariableOp5lstm_24_lstm_cell_24_matmul_1_readvariableop_resource*
_output_shapes
:	`*
dtype02.
,lstm_24/lstm_cell_24/MatMul_1/ReadVariableOpÉ
lstm_24/lstm_cell_24/MatMul_1MatMullstm_24/zeros:output:04lstm_24/lstm_cell_24/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_24/lstm_cell_24/MatMul_1À
lstm_24/lstm_cell_24/addAddV2%lstm_24/lstm_cell_24/MatMul:product:0'lstm_24/lstm_cell_24/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_24/lstm_cell_24/addÌ
+lstm_24/lstm_cell_24/BiasAdd/ReadVariableOpReadVariableOp4lstm_24_lstm_cell_24_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02-
+lstm_24/lstm_cell_24/BiasAdd/ReadVariableOpÍ
lstm_24/lstm_cell_24/BiasAddBiasAddlstm_24/lstm_cell_24/add:z:03lstm_24/lstm_cell_24/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_24/lstm_cell_24/BiasAdd
$lstm_24/lstm_cell_24/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2&
$lstm_24/lstm_cell_24/split/split_dim
lstm_24/lstm_cell_24/splitSplit-lstm_24/lstm_cell_24/split/split_dim:output:0%lstm_24/lstm_cell_24/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ`:ÿÿÿÿÿÿÿÿÿ`:ÿÿÿÿÿÿÿÿÿ`:ÿÿÿÿÿÿÿÿÿ`*
	num_split2
lstm_24/lstm_cell_24/split
lstm_24/lstm_cell_24/SigmoidSigmoid#lstm_24/lstm_cell_24/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
lstm_24/lstm_cell_24/Sigmoid¢
lstm_24/lstm_cell_24/Sigmoid_1Sigmoid#lstm_24/lstm_cell_24/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2 
lstm_24/lstm_cell_24/Sigmoid_1«
lstm_24/lstm_cell_24/mulMul"lstm_24/lstm_cell_24/Sigmoid_1:y:0lstm_24/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
lstm_24/lstm_cell_24/mul
lstm_24/lstm_cell_24/ReluRelu#lstm_24/lstm_cell_24/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
lstm_24/lstm_cell_24/Relu¼
lstm_24/lstm_cell_24/mul_1Mul lstm_24/lstm_cell_24/Sigmoid:y:0'lstm_24/lstm_cell_24/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
lstm_24/lstm_cell_24/mul_1±
lstm_24/lstm_cell_24/add_1AddV2lstm_24/lstm_cell_24/mul:z:0lstm_24/lstm_cell_24/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
lstm_24/lstm_cell_24/add_1¢
lstm_24/lstm_cell_24/Sigmoid_2Sigmoid#lstm_24/lstm_cell_24/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2 
lstm_24/lstm_cell_24/Sigmoid_2
lstm_24/lstm_cell_24/Relu_1Relulstm_24/lstm_cell_24/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
lstm_24/lstm_cell_24/Relu_1À
lstm_24/lstm_cell_24/mul_2Mul"lstm_24/lstm_cell_24/Sigmoid_2:y:0)lstm_24/lstm_cell_24/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
lstm_24/lstm_cell_24/mul_2
%lstm_24/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ`   2'
%lstm_24/TensorArrayV2_1/element_shapeØ
lstm_24/TensorArrayV2_1TensorListReserve.lstm_24/TensorArrayV2_1/element_shape:output:0 lstm_24/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_24/TensorArrayV2_1^
lstm_24/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_24/time
 lstm_24/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2"
 lstm_24/while/maximum_iterationsz
lstm_24/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_24/while/loop_counter
lstm_24/whileWhile#lstm_24/while/loop_counter:output:0)lstm_24/while/maximum_iterations:output:0lstm_24/time:output:0 lstm_24/TensorArrayV2_1:handle:0lstm_24/zeros:output:0lstm_24/zeros_1:output:0 lstm_24/strided_slice_1:output:0?lstm_24/TensorArrayUnstack/TensorListFromTensor:output_handle:03lstm_24_lstm_cell_24_matmul_readvariableop_resource5lstm_24_lstm_cell_24_matmul_1_readvariableop_resource4lstm_24_lstm_cell_24_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ`:ÿÿÿÿÿÿÿÿÿ`: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *'
bodyR
lstm_24_while_body_39971917*'
condR
lstm_24_while_cond_39971916*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ`:ÿÿÿÿÿÿÿÿÿ`: : : : : *
parallel_iterations 2
lstm_24/whileÅ
8lstm_24/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ`   2:
8lstm_24/TensorArrayV2Stack/TensorListStack/element_shape
*lstm_24/TensorArrayV2Stack/TensorListStackTensorListStacklstm_24/while:output:3Alstm_24/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`*
element_dtype02,
*lstm_24/TensorArrayV2Stack/TensorListStack
lstm_24/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
lstm_24/strided_slice_3/stack
lstm_24/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2!
lstm_24/strided_slice_3/stack_1
lstm_24/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_24/strided_slice_3/stack_2Ê
lstm_24/strided_slice_3StridedSlice3lstm_24/TensorArrayV2Stack/TensorListStack:tensor:0&lstm_24/strided_slice_3/stack:output:0(lstm_24/strided_slice_3/stack_1:output:0(lstm_24/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`*
shrink_axis_mask2
lstm_24/strided_slice_3
lstm_24/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_24/transpose_1/permÅ
lstm_24/transpose_1	Transpose3lstm_24/TensorArrayV2Stack/TensorListStack:tensor:0!lstm_24/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
lstm_24/transpose_1v
lstm_24/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_24/runtime
dropout_24/IdentityIdentitylstm_24/transpose_1:y:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
dropout_24/Identityj
lstm_25/ShapeShapedropout_24/Identity:output:0*
T0*
_output_shapes
:2
lstm_25/Shape
lstm_25/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_25/strided_slice/stack
lstm_25/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_25/strided_slice/stack_1
lstm_25/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_25/strided_slice/stack_2
lstm_25/strided_sliceStridedSlicelstm_25/Shape:output:0$lstm_25/strided_slice/stack:output:0&lstm_25/strided_slice/stack_1:output:0&lstm_25/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_25/strided_slicem
lstm_25/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :Ú2
lstm_25/zeros/mul/y
lstm_25/zeros/mulMullstm_25/strided_slice:output:0lstm_25/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_25/zeros/mulo
lstm_25/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
lstm_25/zeros/Less/y
lstm_25/zeros/LessLesslstm_25/zeros/mul:z:0lstm_25/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_25/zeros/Lesss
lstm_25/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :Ú2
lstm_25/zeros/packed/1£
lstm_25/zeros/packedPacklstm_25/strided_slice:output:0lstm_25/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_25/zeros/packedo
lstm_25/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_25/zeros/Const
lstm_25/zerosFilllstm_25/zeros/packed:output:0lstm_25/zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
lstm_25/zerosq
lstm_25/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :Ú2
lstm_25/zeros_1/mul/y
lstm_25/zeros_1/mulMullstm_25/strided_slice:output:0lstm_25/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_25/zeros_1/muls
lstm_25/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
lstm_25/zeros_1/Less/y
lstm_25/zeros_1/LessLesslstm_25/zeros_1/mul:z:0lstm_25/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_25/zeros_1/Lessw
lstm_25/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :Ú2
lstm_25/zeros_1/packed/1©
lstm_25/zeros_1/packedPacklstm_25/strided_slice:output:0!lstm_25/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_25/zeros_1/packeds
lstm_25/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_25/zeros_1/Const
lstm_25/zeros_1Filllstm_25/zeros_1/packed:output:0lstm_25/zeros_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
lstm_25/zeros_1
lstm_25/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_25/transpose/perm¨
lstm_25/transpose	Transposedropout_24/Identity:output:0lstm_25/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
lstm_25/transposeg
lstm_25/Shape_1Shapelstm_25/transpose:y:0*
T0*
_output_shapes
:2
lstm_25/Shape_1
lstm_25/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_25/strided_slice_1/stack
lstm_25/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_25/strided_slice_1/stack_1
lstm_25/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_25/strided_slice_1/stack_2
lstm_25/strided_slice_1StridedSlicelstm_25/Shape_1:output:0&lstm_25/strided_slice_1/stack:output:0(lstm_25/strided_slice_1/stack_1:output:0(lstm_25/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_25/strided_slice_1
#lstm_25/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2%
#lstm_25/TensorArrayV2/element_shapeÒ
lstm_25/TensorArrayV2TensorListReserve,lstm_25/TensorArrayV2/element_shape:output:0 lstm_25/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_25/TensorArrayV2Ï
=lstm_25/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ`   2?
=lstm_25/TensorArrayUnstack/TensorListFromTensor/element_shape
/lstm_25/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_25/transpose:y:0Flstm_25/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type021
/lstm_25/TensorArrayUnstack/TensorListFromTensor
lstm_25/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_25/strided_slice_2/stack
lstm_25/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_25/strided_slice_2/stack_1
lstm_25/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_25/strided_slice_2/stack_2¬
lstm_25/strided_slice_2StridedSlicelstm_25/transpose:y:0&lstm_25/strided_slice_2/stack:output:0(lstm_25/strided_slice_2/stack_1:output:0(lstm_25/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`*
shrink_axis_mask2
lstm_25/strided_slice_2Í
*lstm_25/lstm_cell_25/MatMul/ReadVariableOpReadVariableOp3lstm_25_lstm_cell_25_matmul_readvariableop_resource*
_output_shapes
:	`è
*
dtype02,
*lstm_25/lstm_cell_25/MatMul/ReadVariableOpÍ
lstm_25/lstm_cell_25/MatMulMatMul lstm_25/strided_slice_2:output:02lstm_25/lstm_cell_25/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè
2
lstm_25/lstm_cell_25/MatMulÔ
,lstm_25/lstm_cell_25/MatMul_1/ReadVariableOpReadVariableOp5lstm_25_lstm_cell_25_matmul_1_readvariableop_resource* 
_output_shapes
:
Úè
*
dtype02.
,lstm_25/lstm_cell_25/MatMul_1/ReadVariableOpÉ
lstm_25/lstm_cell_25/MatMul_1MatMullstm_25/zeros:output:04lstm_25/lstm_cell_25/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè
2
lstm_25/lstm_cell_25/MatMul_1À
lstm_25/lstm_cell_25/addAddV2%lstm_25/lstm_cell_25/MatMul:product:0'lstm_25/lstm_cell_25/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè
2
lstm_25/lstm_cell_25/addÌ
+lstm_25/lstm_cell_25/BiasAdd/ReadVariableOpReadVariableOp4lstm_25_lstm_cell_25_biasadd_readvariableop_resource*
_output_shapes	
:è
*
dtype02-
+lstm_25/lstm_cell_25/BiasAdd/ReadVariableOpÍ
lstm_25/lstm_cell_25/BiasAddBiasAddlstm_25/lstm_cell_25/add:z:03lstm_25/lstm_cell_25/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè
2
lstm_25/lstm_cell_25/BiasAdd
$lstm_25/lstm_cell_25/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2&
$lstm_25/lstm_cell_25/split/split_dim
lstm_25/lstm_cell_25/splitSplit-lstm_25/lstm_cell_25/split/split_dim:output:0%lstm_25/lstm_cell_25/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿÚ:ÿÿÿÿÿÿÿÿÿÚ:ÿÿÿÿÿÿÿÿÿÚ:ÿÿÿÿÿÿÿÿÿÚ*
	num_split2
lstm_25/lstm_cell_25/split
lstm_25/lstm_cell_25/SigmoidSigmoid#lstm_25/lstm_cell_25/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
lstm_25/lstm_cell_25/Sigmoid£
lstm_25/lstm_cell_25/Sigmoid_1Sigmoid#lstm_25/lstm_cell_25/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2 
lstm_25/lstm_cell_25/Sigmoid_1¬
lstm_25/lstm_cell_25/mulMul"lstm_25/lstm_cell_25/Sigmoid_1:y:0lstm_25/zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
lstm_25/lstm_cell_25/mul
lstm_25/lstm_cell_25/ReluRelu#lstm_25/lstm_cell_25/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
lstm_25/lstm_cell_25/Relu½
lstm_25/lstm_cell_25/mul_1Mul lstm_25/lstm_cell_25/Sigmoid:y:0'lstm_25/lstm_cell_25/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
lstm_25/lstm_cell_25/mul_1²
lstm_25/lstm_cell_25/add_1AddV2lstm_25/lstm_cell_25/mul:z:0lstm_25/lstm_cell_25/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
lstm_25/lstm_cell_25/add_1£
lstm_25/lstm_cell_25/Sigmoid_2Sigmoid#lstm_25/lstm_cell_25/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2 
lstm_25/lstm_cell_25/Sigmoid_2
lstm_25/lstm_cell_25/Relu_1Relulstm_25/lstm_cell_25/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
lstm_25/lstm_cell_25/Relu_1Á
lstm_25/lstm_cell_25/mul_2Mul"lstm_25/lstm_cell_25/Sigmoid_2:y:0)lstm_25/lstm_cell_25/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
lstm_25/lstm_cell_25/mul_2
%lstm_25/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿZ  2'
%lstm_25/TensorArrayV2_1/element_shapeØ
lstm_25/TensorArrayV2_1TensorListReserve.lstm_25/TensorArrayV2_1/element_shape:output:0 lstm_25/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_25/TensorArrayV2_1^
lstm_25/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_25/time
 lstm_25/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2"
 lstm_25/while/maximum_iterationsz
lstm_25/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_25/while/loop_counter
lstm_25/whileWhile#lstm_25/while/loop_counter:output:0)lstm_25/while/maximum_iterations:output:0lstm_25/time:output:0 lstm_25/TensorArrayV2_1:handle:0lstm_25/zeros:output:0lstm_25/zeros_1:output:0 lstm_25/strided_slice_1:output:0?lstm_25/TensorArrayUnstack/TensorListFromTensor:output_handle:03lstm_25_lstm_cell_25_matmul_readvariableop_resource5lstm_25_lstm_cell_25_matmul_1_readvariableop_resource4lstm_25_lstm_cell_25_biasadd_readvariableop_resource*
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
lstm_25_while_body_39972065*'
condR
lstm_25_while_cond_39972064*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿÚ:ÿÿÿÿÿÿÿÿÿÚ: : : : : *
parallel_iterations 2
lstm_25/whileÅ
8lstm_25/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿZ  2:
8lstm_25/TensorArrayV2Stack/TensorListStack/element_shape
*lstm_25/TensorArrayV2Stack/TensorListStackTensorListStacklstm_25/while:output:3Alstm_25/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ*
element_dtype02,
*lstm_25/TensorArrayV2Stack/TensorListStack
lstm_25/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
lstm_25/strided_slice_3/stack
lstm_25/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2!
lstm_25/strided_slice_3/stack_1
lstm_25/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_25/strided_slice_3/stack_2Ë
lstm_25/strided_slice_3StridedSlice3lstm_25/TensorArrayV2Stack/TensorListStack:tensor:0&lstm_25/strided_slice_3/stack:output:0(lstm_25/strided_slice_3/stack_1:output:0(lstm_25/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ*
shrink_axis_mask2
lstm_25/strided_slice_3
lstm_25/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_25/transpose_1/permÆ
lstm_25/transpose_1	Transpose3lstm_25/TensorArrayV2Stack/TensorListStack:tensor:0!lstm_25/transpose_1/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
lstm_25/transpose_1v
lstm_25/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_25/runtime
dropout_25/IdentityIdentitylstm_25/transpose_1:y:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
dropout_25/Identity²
!dense_12/Tensordot/ReadVariableOpReadVariableOp*dense_12_tensordot_readvariableop_resource*
_output_shapes
:	Ú*
dtype02#
!dense_12/Tensordot/ReadVariableOp|
dense_12/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_12/Tensordot/axes
dense_12/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_12/Tensordot/free
dense_12/Tensordot/ShapeShapedropout_25/Identity:output:0*
T0*
_output_shapes
:2
dense_12/Tensordot/Shape
 dense_12/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_12/Tensordot/GatherV2/axisþ
dense_12/Tensordot/GatherV2GatherV2!dense_12/Tensordot/Shape:output:0 dense_12/Tensordot/free:output:0)dense_12/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_12/Tensordot/GatherV2
"dense_12/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"dense_12/Tensordot/GatherV2_1/axis
dense_12/Tensordot/GatherV2_1GatherV2!dense_12/Tensordot/Shape:output:0 dense_12/Tensordot/axes:output:0+dense_12/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_12/Tensordot/GatherV2_1~
dense_12/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_12/Tensordot/Const¤
dense_12/Tensordot/ProdProd$dense_12/Tensordot/GatherV2:output:0!dense_12/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_12/Tensordot/Prod
dense_12/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_12/Tensordot/Const_1¬
dense_12/Tensordot/Prod_1Prod&dense_12/Tensordot/GatherV2_1:output:0#dense_12/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_12/Tensordot/Prod_1
dense_12/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2 
dense_12/Tensordot/concat/axisÝ
dense_12/Tensordot/concatConcatV2 dense_12/Tensordot/free:output:0 dense_12/Tensordot/axes:output:0'dense_12/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_12/Tensordot/concat°
dense_12/Tensordot/stackPack dense_12/Tensordot/Prod:output:0"dense_12/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_12/Tensordot/stackÂ
dense_12/Tensordot/transpose	Transposedropout_25/Identity:output:0"dense_12/Tensordot/concat:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
dense_12/Tensordot/transposeÃ
dense_12/Tensordot/ReshapeReshape dense_12/Tensordot/transpose:y:0!dense_12/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
dense_12/Tensordot/ReshapeÂ
dense_12/Tensordot/MatMulMatMul#dense_12/Tensordot/Reshape:output:0)dense_12/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_12/Tensordot/MatMul
dense_12/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
dense_12/Tensordot/Const_2
 dense_12/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_12/Tensordot/concat_1/axisê
dense_12/Tensordot/concat_1ConcatV2$dense_12/Tensordot/GatherV2:output:0#dense_12/Tensordot/Const_2:output:0)dense_12/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_12/Tensordot/concat_1´
dense_12/TensordotReshape#dense_12/Tensordot/MatMul:product:0$dense_12/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_12/Tensordot§
dense_12/BiasAdd/ReadVariableOpReadVariableOp(dense_12_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_12/BiasAdd/ReadVariableOp«
dense_12/BiasAddBiasAdddense_12/Tensordot:output:0'dense_12/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_12/BiasAdd
dense_12/SoftmaxSoftmaxdense_12/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_12/Softmaxy
IdentityIdentitydense_12/Softmax:softmax:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

IdentityÈ
NoOpNoOp ^dense_12/BiasAdd/ReadVariableOp"^dense_12/Tensordot/ReadVariableOp,^lstm_24/lstm_cell_24/BiasAdd/ReadVariableOp+^lstm_24/lstm_cell_24/MatMul/ReadVariableOp-^lstm_24/lstm_cell_24/MatMul_1/ReadVariableOp^lstm_24/while,^lstm_25/lstm_cell_25/BiasAdd/ReadVariableOp+^lstm_25/lstm_cell_25/MatMul/ReadVariableOp-^lstm_25/lstm_cell_25/MatMul_1/ReadVariableOp^lstm_25/while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ]: : : : : : : : 2B
dense_12/BiasAdd/ReadVariableOpdense_12/BiasAdd/ReadVariableOp2F
!dense_12/Tensordot/ReadVariableOp!dense_12/Tensordot/ReadVariableOp2Z
+lstm_24/lstm_cell_24/BiasAdd/ReadVariableOp+lstm_24/lstm_cell_24/BiasAdd/ReadVariableOp2X
*lstm_24/lstm_cell_24/MatMul/ReadVariableOp*lstm_24/lstm_cell_24/MatMul/ReadVariableOp2\
,lstm_24/lstm_cell_24/MatMul_1/ReadVariableOp,lstm_24/lstm_cell_24/MatMul_1/ReadVariableOp2
lstm_24/whilelstm_24/while2Z
+lstm_25/lstm_cell_25/BiasAdd/ReadVariableOp+lstm_25/lstm_cell_25/BiasAdd/ReadVariableOp2X
*lstm_25/lstm_cell_25/MatMul/ReadVariableOp*lstm_25/lstm_cell_25/MatMul/ReadVariableOp2\
,lstm_25/lstm_cell_25/MatMul_1/ReadVariableOp,lstm_25/lstm_cell_25/MatMul_1/ReadVariableOp2
lstm_25/whilelstm_25/while:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
 
_user_specified_nameinputs
åJ
Ô

lstm_25_while_body_39972065,
(lstm_25_while_lstm_25_while_loop_counter2
.lstm_25_while_lstm_25_while_maximum_iterations
lstm_25_while_placeholder
lstm_25_while_placeholder_1
lstm_25_while_placeholder_2
lstm_25_while_placeholder_3+
'lstm_25_while_lstm_25_strided_slice_1_0g
clstm_25_while_tensorarrayv2read_tensorlistgetitem_lstm_25_tensorarrayunstack_tensorlistfromtensor_0N
;lstm_25_while_lstm_cell_25_matmul_readvariableop_resource_0:	`è
Q
=lstm_25_while_lstm_cell_25_matmul_1_readvariableop_resource_0:
Úè
K
<lstm_25_while_lstm_cell_25_biasadd_readvariableop_resource_0:	è

lstm_25_while_identity
lstm_25_while_identity_1
lstm_25_while_identity_2
lstm_25_while_identity_3
lstm_25_while_identity_4
lstm_25_while_identity_5)
%lstm_25_while_lstm_25_strided_slice_1e
alstm_25_while_tensorarrayv2read_tensorlistgetitem_lstm_25_tensorarrayunstack_tensorlistfromtensorL
9lstm_25_while_lstm_cell_25_matmul_readvariableop_resource:	`è
O
;lstm_25_while_lstm_cell_25_matmul_1_readvariableop_resource:
Úè
I
:lstm_25_while_lstm_cell_25_biasadd_readvariableop_resource:	è
¢1lstm_25/while/lstm_cell_25/BiasAdd/ReadVariableOp¢0lstm_25/while/lstm_cell_25/MatMul/ReadVariableOp¢2lstm_25/while/lstm_cell_25/MatMul_1/ReadVariableOpÓ
?lstm_25/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ`   2A
?lstm_25/while/TensorArrayV2Read/TensorListGetItem/element_shape
1lstm_25/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemclstm_25_while_tensorarrayv2read_tensorlistgetitem_lstm_25_tensorarrayunstack_tensorlistfromtensor_0lstm_25_while_placeholderHlstm_25/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`*
element_dtype023
1lstm_25/while/TensorArrayV2Read/TensorListGetItemá
0lstm_25/while/lstm_cell_25/MatMul/ReadVariableOpReadVariableOp;lstm_25_while_lstm_cell_25_matmul_readvariableop_resource_0*
_output_shapes
:	`è
*
dtype022
0lstm_25/while/lstm_cell_25/MatMul/ReadVariableOp÷
!lstm_25/while/lstm_cell_25/MatMulMatMul8lstm_25/while/TensorArrayV2Read/TensorListGetItem:item:08lstm_25/while/lstm_cell_25/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè
2#
!lstm_25/while/lstm_cell_25/MatMulè
2lstm_25/while/lstm_cell_25/MatMul_1/ReadVariableOpReadVariableOp=lstm_25_while_lstm_cell_25_matmul_1_readvariableop_resource_0* 
_output_shapes
:
Úè
*
dtype024
2lstm_25/while/lstm_cell_25/MatMul_1/ReadVariableOpà
#lstm_25/while/lstm_cell_25/MatMul_1MatMullstm_25_while_placeholder_2:lstm_25/while/lstm_cell_25/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè
2%
#lstm_25/while/lstm_cell_25/MatMul_1Ø
lstm_25/while/lstm_cell_25/addAddV2+lstm_25/while/lstm_cell_25/MatMul:product:0-lstm_25/while/lstm_cell_25/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè
2 
lstm_25/while/lstm_cell_25/addà
1lstm_25/while/lstm_cell_25/BiasAdd/ReadVariableOpReadVariableOp<lstm_25_while_lstm_cell_25_biasadd_readvariableop_resource_0*
_output_shapes	
:è
*
dtype023
1lstm_25/while/lstm_cell_25/BiasAdd/ReadVariableOpå
"lstm_25/while/lstm_cell_25/BiasAddBiasAdd"lstm_25/while/lstm_cell_25/add:z:09lstm_25/while/lstm_cell_25/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè
2$
"lstm_25/while/lstm_cell_25/BiasAdd
*lstm_25/while/lstm_cell_25/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2,
*lstm_25/while/lstm_cell_25/split/split_dim¯
 lstm_25/while/lstm_cell_25/splitSplit3lstm_25/while/lstm_cell_25/split/split_dim:output:0+lstm_25/while/lstm_cell_25/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿÚ:ÿÿÿÿÿÿÿÿÿÚ:ÿÿÿÿÿÿÿÿÿÚ:ÿÿÿÿÿÿÿÿÿÚ*
	num_split2"
 lstm_25/while/lstm_cell_25/split±
"lstm_25/while/lstm_cell_25/SigmoidSigmoid)lstm_25/while/lstm_cell_25/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2$
"lstm_25/while/lstm_cell_25/Sigmoidµ
$lstm_25/while/lstm_cell_25/Sigmoid_1Sigmoid)lstm_25/while/lstm_cell_25/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2&
$lstm_25/while/lstm_cell_25/Sigmoid_1Á
lstm_25/while/lstm_cell_25/mulMul(lstm_25/while/lstm_cell_25/Sigmoid_1:y:0lstm_25_while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2 
lstm_25/while/lstm_cell_25/mul¨
lstm_25/while/lstm_cell_25/ReluRelu)lstm_25/while/lstm_cell_25/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2!
lstm_25/while/lstm_cell_25/ReluÕ
 lstm_25/while/lstm_cell_25/mul_1Mul&lstm_25/while/lstm_cell_25/Sigmoid:y:0-lstm_25/while/lstm_cell_25/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2"
 lstm_25/while/lstm_cell_25/mul_1Ê
 lstm_25/while/lstm_cell_25/add_1AddV2"lstm_25/while/lstm_cell_25/mul:z:0$lstm_25/while/lstm_cell_25/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2"
 lstm_25/while/lstm_cell_25/add_1µ
$lstm_25/while/lstm_cell_25/Sigmoid_2Sigmoid)lstm_25/while/lstm_cell_25/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2&
$lstm_25/while/lstm_cell_25/Sigmoid_2§
!lstm_25/while/lstm_cell_25/Relu_1Relu$lstm_25/while/lstm_cell_25/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2#
!lstm_25/while/lstm_cell_25/Relu_1Ù
 lstm_25/while/lstm_cell_25/mul_2Mul(lstm_25/while/lstm_cell_25/Sigmoid_2:y:0/lstm_25/while/lstm_cell_25/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2"
 lstm_25/while/lstm_cell_25/mul_2
2lstm_25/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_25_while_placeholder_1lstm_25_while_placeholder$lstm_25/while/lstm_cell_25/mul_2:z:0*
_output_shapes
: *
element_dtype024
2lstm_25/while/TensorArrayV2Write/TensorListSetIteml
lstm_25/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_25/while/add/y
lstm_25/while/addAddV2lstm_25_while_placeholderlstm_25/while/add/y:output:0*
T0*
_output_shapes
: 2
lstm_25/while/addp
lstm_25/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_25/while/add_1/y
lstm_25/while/add_1AddV2(lstm_25_while_lstm_25_while_loop_counterlstm_25/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lstm_25/while/add_1
lstm_25/while/IdentityIdentitylstm_25/while/add_1:z:0^lstm_25/while/NoOp*
T0*
_output_shapes
: 2
lstm_25/while/Identity¦
lstm_25/while/Identity_1Identity.lstm_25_while_lstm_25_while_maximum_iterations^lstm_25/while/NoOp*
T0*
_output_shapes
: 2
lstm_25/while/Identity_1
lstm_25/while/Identity_2Identitylstm_25/while/add:z:0^lstm_25/while/NoOp*
T0*
_output_shapes
: 2
lstm_25/while/Identity_2º
lstm_25/while/Identity_3IdentityBlstm_25/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_25/while/NoOp*
T0*
_output_shapes
: 2
lstm_25/while/Identity_3®
lstm_25/while/Identity_4Identity$lstm_25/while/lstm_cell_25/mul_2:z:0^lstm_25/while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
lstm_25/while/Identity_4®
lstm_25/while/Identity_5Identity$lstm_25/while/lstm_cell_25/add_1:z:0^lstm_25/while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
lstm_25/while/Identity_5
lstm_25/while/NoOpNoOp2^lstm_25/while/lstm_cell_25/BiasAdd/ReadVariableOp1^lstm_25/while/lstm_cell_25/MatMul/ReadVariableOp3^lstm_25/while/lstm_cell_25/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
lstm_25/while/NoOp"9
lstm_25_while_identitylstm_25/while/Identity:output:0"=
lstm_25_while_identity_1!lstm_25/while/Identity_1:output:0"=
lstm_25_while_identity_2!lstm_25/while/Identity_2:output:0"=
lstm_25_while_identity_3!lstm_25/while/Identity_3:output:0"=
lstm_25_while_identity_4!lstm_25/while/Identity_4:output:0"=
lstm_25_while_identity_5!lstm_25/while/Identity_5:output:0"P
%lstm_25_while_lstm_25_strided_slice_1'lstm_25_while_lstm_25_strided_slice_1_0"z
:lstm_25_while_lstm_cell_25_biasadd_readvariableop_resource<lstm_25_while_lstm_cell_25_biasadd_readvariableop_resource_0"|
;lstm_25_while_lstm_cell_25_matmul_1_readvariableop_resource=lstm_25_while_lstm_cell_25_matmul_1_readvariableop_resource_0"x
9lstm_25_while_lstm_cell_25_matmul_readvariableop_resource;lstm_25_while_lstm_cell_25_matmul_readvariableop_resource_0"È
alstm_25_while_tensorarrayv2read_tensorlistgetitem_lstm_25_tensorarrayunstack_tensorlistfromtensorclstm_25_while_tensorarrayv2read_tensorlistgetitem_lstm_25_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿÚ:ÿÿÿÿÿÿÿÿÿÚ: : : : : 2f
1lstm_25/while/lstm_cell_25/BiasAdd/ReadVariableOp1lstm_25/while/lstm_cell_25/BiasAdd/ReadVariableOp2d
0lstm_25/while/lstm_cell_25/MatMul/ReadVariableOp0lstm_25/while/lstm_cell_25/MatMul/ReadVariableOp2h
2lstm_25/while/lstm_cell_25/MatMul_1/ReadVariableOp2lstm_25/while/lstm_cell_25/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
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
H__inference_dropout_25_layer_call_and_return_conditional_losses_39971222

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
ËF

E__inference_lstm_25_layer_call_and_return_conditional_losses_39970415

inputs(
lstm_cell_25_39970333:	`è
)
lstm_cell_25_39970335:
Úè
$
lstm_cell_25_39970337:	è

identity¢$lstm_cell_25/StatefulPartitionedCall¢whileD
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
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`2
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
valueB"ÿÿÿÿ`   27
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
:ÿÿÿÿÿÿÿÿÿ`*
shrink_axis_mask2
strided_slice_2¨
$lstm_cell_25/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_25_39970333lstm_cell_25_39970335lstm_cell_25_39970337*
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
J__inference_lstm_cell_25_layer_call_and_return_conditional_losses_399703322&
$lstm_cell_25/StatefulPartitionedCall
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_25_39970333lstm_cell_25_39970335lstm_cell_25_39970337*
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
while_body_39970346*
condR
while_cond_39970345*M
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
NoOpNoOp%^lstm_cell_25/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`: : : 2L
$lstm_cell_25/StatefulPartitionedCall$lstm_cell_25/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`
 
_user_specified_nameinputs
õ

J__inference_lstm_cell_24_layer_call_and_return_conditional_losses_39969848

inputs

states
states_11
matmul_readvariableop_resource:	]3
 matmul_1_readvariableop_resource:	`.
biasadd_readvariableop_resource:	
identity

identity_1

identity_2¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	]*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	`*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
add
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
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
L:ÿÿÿÿÿÿÿÿÿ`:ÿÿÿÿÿÿÿÿÿ`:ÿÿÿÿÿÿÿÿÿ`:ÿÿÿÿÿÿÿÿÿ`*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
	Sigmoid_1\
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
mulV
ReluRelusplit:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
	Sigmoid_2U
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
mul_2d
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2

Identityh

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2

Identity_1h

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2

Identity_2
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
?:ÿÿÿÿÿÿÿÿÿ]:ÿÿÿÿÿÿÿÿÿ`:ÿÿÿÿÿÿÿÿÿ`: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
 
_user_specified_namestates:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
 
_user_specified_namestates
°?
Ô
while_body_39973755
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_25_matmul_readvariableop_resource_0:	`è
I
5while_lstm_cell_25_matmul_1_readvariableop_resource_0:
Úè
C
4while_lstm_cell_25_biasadd_readvariableop_resource_0:	è

while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_25_matmul_readvariableop_resource:	`è
G
3while_lstm_cell_25_matmul_1_readvariableop_resource:
Úè
A
2while_lstm_cell_25_biasadd_readvariableop_resource:	è
¢)while/lstm_cell_25/BiasAdd/ReadVariableOp¢(while/lstm_cell_25/MatMul/ReadVariableOp¢*while/lstm_cell_25/MatMul_1/ReadVariableOpÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ`   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÓ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemÉ
(while/lstm_cell_25/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_25_matmul_readvariableop_resource_0*
_output_shapes
:	`è
*
dtype02*
(while/lstm_cell_25/MatMul/ReadVariableOp×
while/lstm_cell_25/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_25/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè
2
while/lstm_cell_25/MatMulÐ
*while/lstm_cell_25/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_25_matmul_1_readvariableop_resource_0* 
_output_shapes
:
Úè
*
dtype02,
*while/lstm_cell_25/MatMul_1/ReadVariableOpÀ
while/lstm_cell_25/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_25/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè
2
while/lstm_cell_25/MatMul_1¸
while/lstm_cell_25/addAddV2#while/lstm_cell_25/MatMul:product:0%while/lstm_cell_25/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè
2
while/lstm_cell_25/addÈ
)while/lstm_cell_25/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_25_biasadd_readvariableop_resource_0*
_output_shapes	
:è
*
dtype02+
)while/lstm_cell_25/BiasAdd/ReadVariableOpÅ
while/lstm_cell_25/BiasAddBiasAddwhile/lstm_cell_25/add:z:01while/lstm_cell_25/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè
2
while/lstm_cell_25/BiasAdd
"while/lstm_cell_25/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_25/split/split_dim
while/lstm_cell_25/splitSplit+while/lstm_cell_25/split/split_dim:output:0#while/lstm_cell_25/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿÚ:ÿÿÿÿÿÿÿÿÿÚ:ÿÿÿÿÿÿÿÿÿÚ:ÿÿÿÿÿÿÿÿÿÚ*
	num_split2
while/lstm_cell_25/split
while/lstm_cell_25/SigmoidSigmoid!while/lstm_cell_25/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
while/lstm_cell_25/Sigmoid
while/lstm_cell_25/Sigmoid_1Sigmoid!while/lstm_cell_25/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
while/lstm_cell_25/Sigmoid_1¡
while/lstm_cell_25/mulMul while/lstm_cell_25/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
while/lstm_cell_25/mul
while/lstm_cell_25/ReluRelu!while/lstm_cell_25/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
while/lstm_cell_25/Reluµ
while/lstm_cell_25/mul_1Mulwhile/lstm_cell_25/Sigmoid:y:0%while/lstm_cell_25/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
while/lstm_cell_25/mul_1ª
while/lstm_cell_25/add_1AddV2while/lstm_cell_25/mul:z:0while/lstm_cell_25/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
while/lstm_cell_25/add_1
while/lstm_cell_25/Sigmoid_2Sigmoid!while/lstm_cell_25/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
while/lstm_cell_25/Sigmoid_2
while/lstm_cell_25/Relu_1Reluwhile/lstm_cell_25/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
while/lstm_cell_25/Relu_1¹
while/lstm_cell_25/mul_2Mul while/lstm_cell_25/Sigmoid_2:y:0'while/lstm_cell_25/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
while/lstm_cell_25/mul_2à
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_25/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_25/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_25/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
while/Identity_5Þ

while/NoOpNoOp*^while/lstm_cell_25/BiasAdd/ReadVariableOp)^while/lstm_cell_25/MatMul/ReadVariableOp+^while/lstm_cell_25/MatMul_1/ReadVariableOp*"
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
2while_lstm_cell_25_biasadd_readvariableop_resource4while_lstm_cell_25_biasadd_readvariableop_resource_0"l
3while_lstm_cell_25_matmul_1_readvariableop_resource5while_lstm_cell_25_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_25_matmul_readvariableop_resource3while_lstm_cell_25_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿÚ:ÿÿÿÿÿÿÿÿÿÚ: : : : : 2V
)while/lstm_cell_25/BiasAdd/ReadVariableOp)while/lstm_cell_25/BiasAdd/ReadVariableOp2T
(while/lstm_cell_25/MatMul/ReadVariableOp(while/lstm_cell_25/MatMul/ReadVariableOp2X
*while/lstm_cell_25/MatMul_1/ReadVariableOp*while/lstm_cell_25/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
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
F__inference_dense_12_layer_call_and_return_conditional_losses_39973941

inputs4
!tensordot_readvariableop_resource:	Ú-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢Tensordot/ReadVariableOp
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes
:	Ú*
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
:ÿÿÿÿÿÿÿÿÿÚ2
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
:ÿÿÿÿÿÿÿÿÿÚ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ
 
_user_specified_nameinputs
à
º
*__inference_lstm_25_layer_call_fn_39973850
inputs_0
unknown:	`è
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
E__inference_lstm_25_layer_call_and_return_conditional_losses_399704152
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
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`
"
_user_specified_name
inputs/0
ß
Í
while_cond_39971589
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_39971589___redundant_placeholder06
2while_while_cond_39971589___redundant_placeholder16
2while_while_cond_39971589___redundant_placeholder26
2while_while_cond_39971589___redundant_placeholder3
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
@: : : : :ÿÿÿÿÿÿÿÿÿ`:ÿÿÿÿÿÿÿÿÿ`: ::::: 

_output_shapes
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
:ÿÿÿÿÿÿÿÿÿ`:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`:

_output_shapes
: :

_output_shapes
:
Ï
g
H__inference_dropout_24_layer_call_and_return_conditional_losses_39973225

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *«ªª?2
dropout/Constw
dropout/MulMulinputsdropout/Const:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape¸
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  >2
dropout/GreaterEqual/yÂ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
dropout/Cast~
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
dropout/Mul_1i
IdentityIdentitydropout/Mul_1:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ`:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
 
_user_specified_nameinputs
¬

Ò
0__inference_sequential_12_layer_call_fn_39971771
lstm_24_input
unknown:	]
	unknown_0:	`
	unknown_1:	
	unknown_2:	`è

	unknown_3:
Úè

	unknown_4:	è

	unknown_5:	Ú
	unknown_6:
identity¢StatefulPartitionedCallÔ
StatefulPartitionedCallStatefulPartitionedCalllstm_24_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
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
K__inference_sequential_12_layer_call_and_return_conditional_losses_399717312
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
_user_specified_namelstm_24_input
¹
ö
K__inference_sequential_12_layer_call_and_return_conditional_losses_39971796
lstm_24_input#
lstm_24_39971774:	]#
lstm_24_39971776:	`
lstm_24_39971778:	#
lstm_25_39971782:	`è
$
lstm_25_39971784:
Úè

lstm_25_39971786:	è
$
dense_12_39971790:	Ú
dense_12_39971792:
identity¢ dense_12/StatefulPartitionedCall¢lstm_24/StatefulPartitionedCall¢lstm_25/StatefulPartitionedCall´
lstm_24/StatefulPartitionedCallStatefulPartitionedCalllstm_24_inputlstm_24_39971774lstm_24_39971776lstm_24_39971778*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_lstm_24_layer_call_and_return_conditional_losses_399710442!
lstm_24/StatefulPartitionedCall
dropout_24/PartitionedCallPartitionedCall(lstm_24/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_dropout_24_layer_call_and_return_conditional_losses_399710572
dropout_24/PartitionedCallË
lstm_25/StatefulPartitionedCallStatefulPartitionedCall#dropout_24/PartitionedCall:output:0lstm_25_39971782lstm_25_39971784lstm_25_39971786*
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
E__inference_lstm_25_layer_call_and_return_conditional_losses_399712092!
lstm_25/StatefulPartitionedCall
dropout_25/PartitionedCallPartitionedCall(lstm_25/StatefulPartitionedCall:output:0*
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
H__inference_dropout_25_layer_call_and_return_conditional_losses_399712222
dropout_25/PartitionedCall»
 dense_12/StatefulPartitionedCallStatefulPartitionedCall#dropout_25/PartitionedCall:output:0dense_12_39971790dense_12_39971792*
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
F__inference_dense_12_layer_call_and_return_conditional_losses_399712552"
 dense_12/StatefulPartitionedCall
IdentityIdentity)dense_12/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityµ
NoOpNoOp!^dense_12/StatefulPartitionedCall ^lstm_24/StatefulPartitionedCall ^lstm_25/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ]: : : : : : : : 2D
 dense_12/StatefulPartitionedCall dense_12/StatefulPartitionedCall2B
lstm_24/StatefulPartitionedCalllstm_24/StatefulPartitionedCall2B
lstm_25/StatefulPartitionedCalllstm_25/StatefulPartitionedCall:Z V
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
'
_user_specified_namelstm_24_input

f
H__inference_dropout_24_layer_call_and_return_conditional_losses_39973213

inputs

identity_1^
IdentityIdentityinputs*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2

Identitym

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ`:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
 
_user_specified_nameinputs
Ð

í
lstm_24_while_cond_39971916,
(lstm_24_while_lstm_24_while_loop_counter2
.lstm_24_while_lstm_24_while_maximum_iterations
lstm_24_while_placeholder
lstm_24_while_placeholder_1
lstm_24_while_placeholder_2
lstm_24_while_placeholder_3.
*lstm_24_while_less_lstm_24_strided_slice_1F
Blstm_24_while_lstm_24_while_cond_39971916___redundant_placeholder0F
Blstm_24_while_lstm_24_while_cond_39971916___redundant_placeholder1F
Blstm_24_while_lstm_24_while_cond_39971916___redundant_placeholder2F
Blstm_24_while_lstm_24_while_cond_39971916___redundant_placeholder3
lstm_24_while_identity

lstm_24/while/LessLesslstm_24_while_placeholder*lstm_24_while_less_lstm_24_strided_slice_1*
T0*
_output_shapes
: 2
lstm_24/while/Lessu
lstm_24/while/IdentityIdentitylstm_24/while/Less:z:0*
T0
*
_output_shapes
: 2
lstm_24/while/Identity"9
lstm_24_while_identitylstm_24/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿ`:ÿÿÿÿÿÿÿÿÿ`: ::::: 

_output_shapes
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
:ÿÿÿÿÿÿÿÿÿ`:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`:

_output_shapes
: :

_output_shapes
:
ß
Í
while_cond_39970959
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_39970959___redundant_placeholder06
2while_while_cond_39970959___redundant_placeholder16
2while_while_cond_39970959___redundant_placeholder26
2while_while_cond_39970959___redundant_placeholder3
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
@: : : : :ÿÿÿÿÿÿÿÿÿ`:ÿÿÿÿÿÿÿÿÿ`: ::::: 

_output_shapes
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
:ÿÿÿÿÿÿÿÿÿ`:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`:

_output_shapes
: :

_output_shapes
:
ß
Í
while_cond_39969925
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_39969925___redundant_placeholder06
2while_while_cond_39969925___redundant_placeholder16
2while_while_cond_39969925___redundant_placeholder26
2while_while_cond_39969925___redundant_placeholder3
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
@: : : : :ÿÿÿÿÿÿÿÿÿ`:ÿÿÿÿÿÿÿÿÿ`: ::::: 

_output_shapes
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
:ÿÿÿÿÿÿÿÿÿ`:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`:

_output_shapes
: :

_output_shapes
:
ã
Í
while_cond_39973301
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_39973301___redundant_placeholder06
2while_while_cond_39973301___redundant_placeholder16
2while_while_cond_39973301___redundant_placeholder26
2while_while_cond_39973301___redundant_placeholder3
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
*__inference_lstm_25_layer_call_fn_39973883

inputs
unknown:	`è
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
E__inference_lstm_25_layer_call_and_return_conditional_losses_399714782
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
:ÿÿÿÿÿÿÿÿÿ`: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
 
_user_specified_nameinputs
?
Ò
while_body_39972778
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_24_matmul_readvariableop_resource_0:	]H
5while_lstm_cell_24_matmul_1_readvariableop_resource_0:	`C
4while_lstm_cell_24_biasadd_readvariableop_resource_0:	
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_24_matmul_readvariableop_resource:	]F
3while_lstm_cell_24_matmul_1_readvariableop_resource:	`A
2while_lstm_cell_24_biasadd_readvariableop_resource:	¢)while/lstm_cell_24/BiasAdd/ReadVariableOp¢(while/lstm_cell_24/MatMul/ReadVariableOp¢*while/lstm_cell_24/MatMul_1/ReadVariableOpÃ
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
(while/lstm_cell_24/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_24_matmul_readvariableop_resource_0*
_output_shapes
:	]*
dtype02*
(while/lstm_cell_24/MatMul/ReadVariableOp×
while/lstm_cell_24/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_24/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_24/MatMulÏ
*while/lstm_cell_24/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_24_matmul_1_readvariableop_resource_0*
_output_shapes
:	`*
dtype02,
*while/lstm_cell_24/MatMul_1/ReadVariableOpÀ
while/lstm_cell_24/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_24/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_24/MatMul_1¸
while/lstm_cell_24/addAddV2#while/lstm_cell_24/MatMul:product:0%while/lstm_cell_24/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_24/addÈ
)while/lstm_cell_24/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_24_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02+
)while/lstm_cell_24/BiasAdd/ReadVariableOpÅ
while/lstm_cell_24/BiasAddBiasAddwhile/lstm_cell_24/add:z:01while/lstm_cell_24/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_24/BiasAdd
"while/lstm_cell_24/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_24/split/split_dim
while/lstm_cell_24/splitSplit+while/lstm_cell_24/split/split_dim:output:0#while/lstm_cell_24/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ`:ÿÿÿÿÿÿÿÿÿ`:ÿÿÿÿÿÿÿÿÿ`:ÿÿÿÿÿÿÿÿÿ`*
	num_split2
while/lstm_cell_24/split
while/lstm_cell_24/SigmoidSigmoid!while/lstm_cell_24/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
while/lstm_cell_24/Sigmoid
while/lstm_cell_24/Sigmoid_1Sigmoid!while/lstm_cell_24/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
while/lstm_cell_24/Sigmoid_1 
while/lstm_cell_24/mulMul while/lstm_cell_24/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
while/lstm_cell_24/mul
while/lstm_cell_24/ReluRelu!while/lstm_cell_24/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
while/lstm_cell_24/Relu´
while/lstm_cell_24/mul_1Mulwhile/lstm_cell_24/Sigmoid:y:0%while/lstm_cell_24/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
while/lstm_cell_24/mul_1©
while/lstm_cell_24/add_1AddV2while/lstm_cell_24/mul:z:0while/lstm_cell_24/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
while/lstm_cell_24/add_1
while/lstm_cell_24/Sigmoid_2Sigmoid!while/lstm_cell_24/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
while/lstm_cell_24/Sigmoid_2
while/lstm_cell_24/Relu_1Reluwhile/lstm_cell_24/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
while/lstm_cell_24/Relu_1¸
while/lstm_cell_24/mul_2Mul while/lstm_cell_24/Sigmoid_2:y:0'while/lstm_cell_24/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
while/lstm_cell_24/mul_2à
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_24/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_24/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_24/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
while/Identity_5Þ

while/NoOpNoOp*^while/lstm_cell_24/BiasAdd/ReadVariableOp)^while/lstm_cell_24/MatMul/ReadVariableOp+^while/lstm_cell_24/MatMul_1/ReadVariableOp*"
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
2while_lstm_cell_24_biasadd_readvariableop_resource4while_lstm_cell_24_biasadd_readvariableop_resource_0"l
3while_lstm_cell_24_matmul_1_readvariableop_resource5while_lstm_cell_24_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_24_matmul_readvariableop_resource3while_lstm_cell_24_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ`:ÿÿÿÿÿÿÿÿÿ`: : : : : 2V
)while/lstm_cell_24/BiasAdd/ReadVariableOp)while/lstm_cell_24/BiasAdd/ReadVariableOp2T
(while/lstm_cell_24/MatMul/ReadVariableOp(while/lstm_cell_24/MatMul/ReadVariableOp2X
*while/lstm_cell_24/MatMul_1/ReadVariableOp*while/lstm_cell_24/MatMul_1/ReadVariableOp: 

_output_shapes
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
:ÿÿÿÿÿÿÿÿÿ`:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`:

_output_shapes
: :

_output_shapes
: 
?
Ò
while_body_39971590
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_24_matmul_readvariableop_resource_0:	]H
5while_lstm_cell_24_matmul_1_readvariableop_resource_0:	`C
4while_lstm_cell_24_biasadd_readvariableop_resource_0:	
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_24_matmul_readvariableop_resource:	]F
3while_lstm_cell_24_matmul_1_readvariableop_resource:	`A
2while_lstm_cell_24_biasadd_readvariableop_resource:	¢)while/lstm_cell_24/BiasAdd/ReadVariableOp¢(while/lstm_cell_24/MatMul/ReadVariableOp¢*while/lstm_cell_24/MatMul_1/ReadVariableOpÃ
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
(while/lstm_cell_24/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_24_matmul_readvariableop_resource_0*
_output_shapes
:	]*
dtype02*
(while/lstm_cell_24/MatMul/ReadVariableOp×
while/lstm_cell_24/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_24/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_24/MatMulÏ
*while/lstm_cell_24/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_24_matmul_1_readvariableop_resource_0*
_output_shapes
:	`*
dtype02,
*while/lstm_cell_24/MatMul_1/ReadVariableOpÀ
while/lstm_cell_24/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_24/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_24/MatMul_1¸
while/lstm_cell_24/addAddV2#while/lstm_cell_24/MatMul:product:0%while/lstm_cell_24/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_24/addÈ
)while/lstm_cell_24/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_24_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02+
)while/lstm_cell_24/BiasAdd/ReadVariableOpÅ
while/lstm_cell_24/BiasAddBiasAddwhile/lstm_cell_24/add:z:01while/lstm_cell_24/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_24/BiasAdd
"while/lstm_cell_24/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_24/split/split_dim
while/lstm_cell_24/splitSplit+while/lstm_cell_24/split/split_dim:output:0#while/lstm_cell_24/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ`:ÿÿÿÿÿÿÿÿÿ`:ÿÿÿÿÿÿÿÿÿ`:ÿÿÿÿÿÿÿÿÿ`*
	num_split2
while/lstm_cell_24/split
while/lstm_cell_24/SigmoidSigmoid!while/lstm_cell_24/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
while/lstm_cell_24/Sigmoid
while/lstm_cell_24/Sigmoid_1Sigmoid!while/lstm_cell_24/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
while/lstm_cell_24/Sigmoid_1 
while/lstm_cell_24/mulMul while/lstm_cell_24/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
while/lstm_cell_24/mul
while/lstm_cell_24/ReluRelu!while/lstm_cell_24/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
while/lstm_cell_24/Relu´
while/lstm_cell_24/mul_1Mulwhile/lstm_cell_24/Sigmoid:y:0%while/lstm_cell_24/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
while/lstm_cell_24/mul_1©
while/lstm_cell_24/add_1AddV2while/lstm_cell_24/mul:z:0while/lstm_cell_24/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
while/lstm_cell_24/add_1
while/lstm_cell_24/Sigmoid_2Sigmoid!while/lstm_cell_24/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
while/lstm_cell_24/Sigmoid_2
while/lstm_cell_24/Relu_1Reluwhile/lstm_cell_24/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
while/lstm_cell_24/Relu_1¸
while/lstm_cell_24/mul_2Mul while/lstm_cell_24/Sigmoid_2:y:0'while/lstm_cell_24/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
while/lstm_cell_24/mul_2à
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_24/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_24/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_24/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
while/Identity_5Þ

while/NoOpNoOp*^while/lstm_cell_24/BiasAdd/ReadVariableOp)^while/lstm_cell_24/MatMul/ReadVariableOp+^while/lstm_cell_24/MatMul_1/ReadVariableOp*"
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
2while_lstm_cell_24_biasadd_readvariableop_resource4while_lstm_cell_24_biasadd_readvariableop_resource_0"l
3while_lstm_cell_24_matmul_1_readvariableop_resource5while_lstm_cell_24_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_24_matmul_readvariableop_resource3while_lstm_cell_24_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ`:ÿÿÿÿÿÿÿÿÿ`: : : : : 2V
)while/lstm_cell_24/BiasAdd/ReadVariableOp)while/lstm_cell_24/BiasAdd/ReadVariableOp2T
(while/lstm_cell_24/MatMul/ReadVariableOp(while/lstm_cell_24/MatMul/ReadVariableOp2X
*while/lstm_cell_24/MatMul_1/ReadVariableOp*while/lstm_cell_24/MatMul_1/ReadVariableOp: 

_output_shapes
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
:ÿÿÿÿÿÿÿÿÿ`:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`:

_output_shapes
: :

_output_shapes
: 
Ç
ù
/__inference_lstm_cell_25_layer_call_fn_39974146

inputs
states_0
states_1
unknown:	`è
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
J__inference_lstm_cell_25_layer_call_and_return_conditional_losses_399704782
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
A:ÿÿÿÿÿÿÿÿÿ`:ÿÿÿÿÿÿÿÿÿÚ:ÿÿÿÿÿÿÿÿÿÚ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
 
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
+__inference_dense_12_layer_call_fn_39973950

inputs
unknown:	Ú
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
F__inference_dense_12_layer_call_and_return_conditional_losses_399712552
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
:ÿÿÿÿÿÿÿÿÿÚ: : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ
 
_user_specified_nameinputs
ß
Í
while_cond_39972928
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_39972928___redundant_placeholder06
2while_while_cond_39972928___redundant_placeholder16
2while_while_cond_39972928___redundant_placeholder26
2while_while_cond_39972928___redundant_placeholder3
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
@: : : : :ÿÿÿÿÿÿÿÿÿ`:ÿÿÿÿÿÿÿÿÿ`: ::::: 

_output_shapes
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
:ÿÿÿÿÿÿÿÿÿ`:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`:

_output_shapes
: :

_output_shapes
:
²
·
*__inference_lstm_24_layer_call_fn_39973197

inputs
unknown:	]
	unknown_0:	`
	unknown_1:	
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_lstm_24_layer_call_and_return_conditional_losses_399710442
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2

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
while_cond_39973452
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_39973452___redundant_placeholder06
2while_while_cond_39973452___redundant_placeholder16
2while_while_cond_39973452___redundant_placeholder26
2while_while_cond_39973452___redundant_placeholder3
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
while_cond_39973603
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_39973603___redundant_placeholder06
2while_while_cond_39973603___redundant_placeholder16
2while_while_cond_39973603___redundant_placeholder26
2while_while_cond_39973603___redundant_placeholder3
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
ÃL
Ü
!__inference__traced_save_39974268
file_prefix.
*savev2_dense_12_kernel_read_readvariableop,
(savev2_dense_12_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop:
6savev2_lstm_24_lstm_cell_24_kernel_read_readvariableopD
@savev2_lstm_24_lstm_cell_24_recurrent_kernel_read_readvariableop8
4savev2_lstm_24_lstm_cell_24_bias_read_readvariableop:
6savev2_lstm_25_lstm_cell_25_kernel_read_readvariableopD
@savev2_lstm_25_lstm_cell_25_recurrent_kernel_read_readvariableop8
4savev2_lstm_25_lstm_cell_25_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop5
1savev2_adam_dense_12_kernel_m_read_readvariableop3
/savev2_adam_dense_12_bias_m_read_readvariableopA
=savev2_adam_lstm_24_lstm_cell_24_kernel_m_read_readvariableopK
Gsavev2_adam_lstm_24_lstm_cell_24_recurrent_kernel_m_read_readvariableop?
;savev2_adam_lstm_24_lstm_cell_24_bias_m_read_readvariableopA
=savev2_adam_lstm_25_lstm_cell_25_kernel_m_read_readvariableopK
Gsavev2_adam_lstm_25_lstm_cell_25_recurrent_kernel_m_read_readvariableop?
;savev2_adam_lstm_25_lstm_cell_25_bias_m_read_readvariableop5
1savev2_adam_dense_12_kernel_v_read_readvariableop3
/savev2_adam_dense_12_bias_v_read_readvariableopA
=savev2_adam_lstm_24_lstm_cell_24_kernel_v_read_readvariableopK
Gsavev2_adam_lstm_24_lstm_cell_24_recurrent_kernel_v_read_readvariableop?
;savev2_adam_lstm_24_lstm_cell_24_bias_v_read_readvariableopA
=savev2_adam_lstm_25_lstm_cell_25_kernel_v_read_readvariableopK
Gsavev2_adam_lstm_25_lstm_cell_25_recurrent_kernel_v_read_readvariableop?
;savev2_adam_lstm_25_lstm_cell_25_bias_v_read_readvariableop
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
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_dense_12_kernel_read_readvariableop(savev2_dense_12_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop6savev2_lstm_24_lstm_cell_24_kernel_read_readvariableop@savev2_lstm_24_lstm_cell_24_recurrent_kernel_read_readvariableop4savev2_lstm_24_lstm_cell_24_bias_read_readvariableop6savev2_lstm_25_lstm_cell_25_kernel_read_readvariableop@savev2_lstm_25_lstm_cell_25_recurrent_kernel_read_readvariableop4savev2_lstm_25_lstm_cell_25_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop1savev2_adam_dense_12_kernel_m_read_readvariableop/savev2_adam_dense_12_bias_m_read_readvariableop=savev2_adam_lstm_24_lstm_cell_24_kernel_m_read_readvariableopGsavev2_adam_lstm_24_lstm_cell_24_recurrent_kernel_m_read_readvariableop;savev2_adam_lstm_24_lstm_cell_24_bias_m_read_readvariableop=savev2_adam_lstm_25_lstm_cell_25_kernel_m_read_readvariableopGsavev2_adam_lstm_25_lstm_cell_25_recurrent_kernel_m_read_readvariableop;savev2_adam_lstm_25_lstm_cell_25_bias_m_read_readvariableop1savev2_adam_dense_12_kernel_v_read_readvariableop/savev2_adam_dense_12_bias_v_read_readvariableop=savev2_adam_lstm_24_lstm_cell_24_kernel_v_read_readvariableopGsavev2_adam_lstm_24_lstm_cell_24_recurrent_kernel_v_read_readvariableop;savev2_adam_lstm_24_lstm_cell_24_bias_v_read_readvariableop=savev2_adam_lstm_25_lstm_cell_25_kernel_v_read_readvariableopGsavev2_adam_lstm_25_lstm_cell_25_recurrent_kernel_v_read_readvariableop;savev2_adam_lstm_25_lstm_cell_25_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
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

identity_1Identity_1:output:0*
_input_shapesý
ú: :	Ú:: : : : : :	]:	`::	`è
:
Úè
:è
: : : : :	Ú::	]:	`::	`è
:
Úè
:è
:	Ú::	]:	`::	`è
:
Úè
:è
: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	Ú: 
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
:	]:%	!

_output_shapes
:	`:!


_output_shapes	
::%!

_output_shapes
:	`è
:&"
 
_output_shapes
:
Úè
:!

_output_shapes	
:è
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
:	Ú: 

_output_shapes
::%!

_output_shapes
:	]:%!

_output_shapes
:	`:!

_output_shapes	
::%!

_output_shapes
:	`è
:&"
 
_output_shapes
:
Úè
:!

_output_shapes	
:è
:%!

_output_shapes
:	Ú: 

_output_shapes
::%!

_output_shapes
:	]:%!

_output_shapes
:	`:!

_output_shapes	
::%!

_output_shapes
:	`è
:& "
 
_output_shapes
:
Úè
:!!

_output_shapes	
:è
:"

_output_shapes
: 
ÏJ
Ò

lstm_24_while_body_39971917,
(lstm_24_while_lstm_24_while_loop_counter2
.lstm_24_while_lstm_24_while_maximum_iterations
lstm_24_while_placeholder
lstm_24_while_placeholder_1
lstm_24_while_placeholder_2
lstm_24_while_placeholder_3+
'lstm_24_while_lstm_24_strided_slice_1_0g
clstm_24_while_tensorarrayv2read_tensorlistgetitem_lstm_24_tensorarrayunstack_tensorlistfromtensor_0N
;lstm_24_while_lstm_cell_24_matmul_readvariableop_resource_0:	]P
=lstm_24_while_lstm_cell_24_matmul_1_readvariableop_resource_0:	`K
<lstm_24_while_lstm_cell_24_biasadd_readvariableop_resource_0:	
lstm_24_while_identity
lstm_24_while_identity_1
lstm_24_while_identity_2
lstm_24_while_identity_3
lstm_24_while_identity_4
lstm_24_while_identity_5)
%lstm_24_while_lstm_24_strided_slice_1e
alstm_24_while_tensorarrayv2read_tensorlistgetitem_lstm_24_tensorarrayunstack_tensorlistfromtensorL
9lstm_24_while_lstm_cell_24_matmul_readvariableop_resource:	]N
;lstm_24_while_lstm_cell_24_matmul_1_readvariableop_resource:	`I
:lstm_24_while_lstm_cell_24_biasadd_readvariableop_resource:	¢1lstm_24/while/lstm_cell_24/BiasAdd/ReadVariableOp¢0lstm_24/while/lstm_cell_24/MatMul/ReadVariableOp¢2lstm_24/while/lstm_cell_24/MatMul_1/ReadVariableOpÓ
?lstm_24/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ]   2A
?lstm_24/while/TensorArrayV2Read/TensorListGetItem/element_shape
1lstm_24/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemclstm_24_while_tensorarrayv2read_tensorlistgetitem_lstm_24_tensorarrayunstack_tensorlistfromtensor_0lstm_24_while_placeholderHlstm_24/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]*
element_dtype023
1lstm_24/while/TensorArrayV2Read/TensorListGetItemá
0lstm_24/while/lstm_cell_24/MatMul/ReadVariableOpReadVariableOp;lstm_24_while_lstm_cell_24_matmul_readvariableop_resource_0*
_output_shapes
:	]*
dtype022
0lstm_24/while/lstm_cell_24/MatMul/ReadVariableOp÷
!lstm_24/while/lstm_cell_24/MatMulMatMul8lstm_24/while/TensorArrayV2Read/TensorListGetItem:item:08lstm_24/while/lstm_cell_24/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!lstm_24/while/lstm_cell_24/MatMulç
2lstm_24/while/lstm_cell_24/MatMul_1/ReadVariableOpReadVariableOp=lstm_24_while_lstm_cell_24_matmul_1_readvariableop_resource_0*
_output_shapes
:	`*
dtype024
2lstm_24/while/lstm_cell_24/MatMul_1/ReadVariableOpà
#lstm_24/while/lstm_cell_24/MatMul_1MatMullstm_24_while_placeholder_2:lstm_24/while/lstm_cell_24/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2%
#lstm_24/while/lstm_cell_24/MatMul_1Ø
lstm_24/while/lstm_cell_24/addAddV2+lstm_24/while/lstm_cell_24/MatMul:product:0-lstm_24/while/lstm_cell_24/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
lstm_24/while/lstm_cell_24/addà
1lstm_24/while/lstm_cell_24/BiasAdd/ReadVariableOpReadVariableOp<lstm_24_while_lstm_cell_24_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype023
1lstm_24/while/lstm_cell_24/BiasAdd/ReadVariableOpå
"lstm_24/while/lstm_cell_24/BiasAddBiasAdd"lstm_24/while/lstm_cell_24/add:z:09lstm_24/while/lstm_cell_24/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2$
"lstm_24/while/lstm_cell_24/BiasAdd
*lstm_24/while/lstm_cell_24/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2,
*lstm_24/while/lstm_cell_24/split/split_dim«
 lstm_24/while/lstm_cell_24/splitSplit3lstm_24/while/lstm_cell_24/split/split_dim:output:0+lstm_24/while/lstm_cell_24/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ`:ÿÿÿÿÿÿÿÿÿ`:ÿÿÿÿÿÿÿÿÿ`:ÿÿÿÿÿÿÿÿÿ`*
	num_split2"
 lstm_24/while/lstm_cell_24/split°
"lstm_24/while/lstm_cell_24/SigmoidSigmoid)lstm_24/while/lstm_cell_24/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2$
"lstm_24/while/lstm_cell_24/Sigmoid´
$lstm_24/while/lstm_cell_24/Sigmoid_1Sigmoid)lstm_24/while/lstm_cell_24/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2&
$lstm_24/while/lstm_cell_24/Sigmoid_1À
lstm_24/while/lstm_cell_24/mulMul(lstm_24/while/lstm_cell_24/Sigmoid_1:y:0lstm_24_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2 
lstm_24/while/lstm_cell_24/mul§
lstm_24/while/lstm_cell_24/ReluRelu)lstm_24/while/lstm_cell_24/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2!
lstm_24/while/lstm_cell_24/ReluÔ
 lstm_24/while/lstm_cell_24/mul_1Mul&lstm_24/while/lstm_cell_24/Sigmoid:y:0-lstm_24/while/lstm_cell_24/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2"
 lstm_24/while/lstm_cell_24/mul_1É
 lstm_24/while/lstm_cell_24/add_1AddV2"lstm_24/while/lstm_cell_24/mul:z:0$lstm_24/while/lstm_cell_24/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2"
 lstm_24/while/lstm_cell_24/add_1´
$lstm_24/while/lstm_cell_24/Sigmoid_2Sigmoid)lstm_24/while/lstm_cell_24/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2&
$lstm_24/while/lstm_cell_24/Sigmoid_2¦
!lstm_24/while/lstm_cell_24/Relu_1Relu$lstm_24/while/lstm_cell_24/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2#
!lstm_24/while/lstm_cell_24/Relu_1Ø
 lstm_24/while/lstm_cell_24/mul_2Mul(lstm_24/while/lstm_cell_24/Sigmoid_2:y:0/lstm_24/while/lstm_cell_24/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2"
 lstm_24/while/lstm_cell_24/mul_2
2lstm_24/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_24_while_placeholder_1lstm_24_while_placeholder$lstm_24/while/lstm_cell_24/mul_2:z:0*
_output_shapes
: *
element_dtype024
2lstm_24/while/TensorArrayV2Write/TensorListSetIteml
lstm_24/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_24/while/add/y
lstm_24/while/addAddV2lstm_24_while_placeholderlstm_24/while/add/y:output:0*
T0*
_output_shapes
: 2
lstm_24/while/addp
lstm_24/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_24/while/add_1/y
lstm_24/while/add_1AddV2(lstm_24_while_lstm_24_while_loop_counterlstm_24/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lstm_24/while/add_1
lstm_24/while/IdentityIdentitylstm_24/while/add_1:z:0^lstm_24/while/NoOp*
T0*
_output_shapes
: 2
lstm_24/while/Identity¦
lstm_24/while/Identity_1Identity.lstm_24_while_lstm_24_while_maximum_iterations^lstm_24/while/NoOp*
T0*
_output_shapes
: 2
lstm_24/while/Identity_1
lstm_24/while/Identity_2Identitylstm_24/while/add:z:0^lstm_24/while/NoOp*
T0*
_output_shapes
: 2
lstm_24/while/Identity_2º
lstm_24/while/Identity_3IdentityBlstm_24/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_24/while/NoOp*
T0*
_output_shapes
: 2
lstm_24/while/Identity_3­
lstm_24/while/Identity_4Identity$lstm_24/while/lstm_cell_24/mul_2:z:0^lstm_24/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
lstm_24/while/Identity_4­
lstm_24/while/Identity_5Identity$lstm_24/while/lstm_cell_24/add_1:z:0^lstm_24/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
lstm_24/while/Identity_5
lstm_24/while/NoOpNoOp2^lstm_24/while/lstm_cell_24/BiasAdd/ReadVariableOp1^lstm_24/while/lstm_cell_24/MatMul/ReadVariableOp3^lstm_24/while/lstm_cell_24/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
lstm_24/while/NoOp"9
lstm_24_while_identitylstm_24/while/Identity:output:0"=
lstm_24_while_identity_1!lstm_24/while/Identity_1:output:0"=
lstm_24_while_identity_2!lstm_24/while/Identity_2:output:0"=
lstm_24_while_identity_3!lstm_24/while/Identity_3:output:0"=
lstm_24_while_identity_4!lstm_24/while/Identity_4:output:0"=
lstm_24_while_identity_5!lstm_24/while/Identity_5:output:0"P
%lstm_24_while_lstm_24_strided_slice_1'lstm_24_while_lstm_24_strided_slice_1_0"z
:lstm_24_while_lstm_cell_24_biasadd_readvariableop_resource<lstm_24_while_lstm_cell_24_biasadd_readvariableop_resource_0"|
;lstm_24_while_lstm_cell_24_matmul_1_readvariableop_resource=lstm_24_while_lstm_cell_24_matmul_1_readvariableop_resource_0"x
9lstm_24_while_lstm_cell_24_matmul_readvariableop_resource;lstm_24_while_lstm_cell_24_matmul_readvariableop_resource_0"È
alstm_24_while_tensorarrayv2read_tensorlistgetitem_lstm_24_tensorarrayunstack_tensorlistfromtensorclstm_24_while_tensorarrayv2read_tensorlistgetitem_lstm_24_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ`:ÿÿÿÿÿÿÿÿÿ`: : : : : 2f
1lstm_24/while/lstm_cell_24/BiasAdd/ReadVariableOp1lstm_24/while/lstm_cell_24/BiasAdd/ReadVariableOp2d
0lstm_24/while/lstm_cell_24/MatMul/ReadVariableOp0lstm_24/while/lstm_cell_24/MatMul/ReadVariableOp2h
2lstm_24/while/lstm_cell_24/MatMul_1/ReadVariableOp2lstm_24/while/lstm_cell_24/MatMul_1/ReadVariableOp: 

_output_shapes
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
:ÿÿÿÿÿÿÿÿÿ`:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`:

_output_shapes
: :

_output_shapes
: 
¦\

E__inference_lstm_24_layer_call_and_return_conditional_losses_39972862
inputs_0>
+lstm_cell_24_matmul_readvariableop_resource:	]@
-lstm_cell_24_matmul_1_readvariableop_resource:	`;
,lstm_cell_24_biasadd_readvariableop_resource:	
identity¢#lstm_cell_24/BiasAdd/ReadVariableOp¢"lstm_cell_24/MatMul/ReadVariableOp¢$lstm_cell_24/MatMul_1/ReadVariableOp¢whileF
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
value	B :`2
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
value	B :`2
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
:ÿÿÿÿÿÿÿÿÿ`2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :`2
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
value	B :`2
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
:ÿÿÿÿÿÿÿÿÿ`2	
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
"lstm_cell_24/MatMul/ReadVariableOpReadVariableOp+lstm_cell_24_matmul_readvariableop_resource*
_output_shapes
:	]*
dtype02$
"lstm_cell_24/MatMul/ReadVariableOp­
lstm_cell_24/MatMulMatMulstrided_slice_2:output:0*lstm_cell_24/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_24/MatMul»
$lstm_cell_24/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_24_matmul_1_readvariableop_resource*
_output_shapes
:	`*
dtype02&
$lstm_cell_24/MatMul_1/ReadVariableOp©
lstm_cell_24/MatMul_1MatMulzeros:output:0,lstm_cell_24/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_24/MatMul_1 
lstm_cell_24/addAddV2lstm_cell_24/MatMul:product:0lstm_cell_24/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_24/add´
#lstm_cell_24/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_24_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02%
#lstm_cell_24/BiasAdd/ReadVariableOp­
lstm_cell_24/BiasAddBiasAddlstm_cell_24/add:z:0+lstm_cell_24/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_24/BiasAdd~
lstm_cell_24/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_24/split/split_dimó
lstm_cell_24/splitSplit%lstm_cell_24/split/split_dim:output:0lstm_cell_24/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ`:ÿÿÿÿÿÿÿÿÿ`:ÿÿÿÿÿÿÿÿÿ`:ÿÿÿÿÿÿÿÿÿ`*
	num_split2
lstm_cell_24/split
lstm_cell_24/SigmoidSigmoidlstm_cell_24/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
lstm_cell_24/Sigmoid
lstm_cell_24/Sigmoid_1Sigmoidlstm_cell_24/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
lstm_cell_24/Sigmoid_1
lstm_cell_24/mulMullstm_cell_24/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
lstm_cell_24/mul}
lstm_cell_24/ReluRelulstm_cell_24/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
lstm_cell_24/Relu
lstm_cell_24/mul_1Mullstm_cell_24/Sigmoid:y:0lstm_cell_24/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
lstm_cell_24/mul_1
lstm_cell_24/add_1AddV2lstm_cell_24/mul:z:0lstm_cell_24/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
lstm_cell_24/add_1
lstm_cell_24/Sigmoid_2Sigmoidlstm_cell_24/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
lstm_cell_24/Sigmoid_2|
lstm_cell_24/Relu_1Relulstm_cell_24/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
lstm_cell_24/Relu_1 
lstm_cell_24/mul_2Mullstm_cell_24/Sigmoid_2:y:0!lstm_cell_24/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
lstm_cell_24/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ`   2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_24_matmul_readvariableop_resource-lstm_cell_24_matmul_1_readvariableop_resource,lstm_cell_24_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ`:ÿÿÿÿÿÿÿÿÿ`: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_39972778*
condR
while_cond_39972777*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ`:ÿÿÿÿÿÿÿÿÿ`: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ`   22
0TensorArrayV2Stack/TensorListStack/element_shapeñ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`*
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
:ÿÿÿÿÿÿÿÿÿ`*
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
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`2
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
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`2

IdentityÈ
NoOpNoOp$^lstm_cell_24/BiasAdd/ReadVariableOp#^lstm_cell_24/MatMul/ReadVariableOp%^lstm_cell_24/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ]: : : 2J
#lstm_cell_24/BiasAdd/ReadVariableOp#lstm_cell_24/BiasAdd/ReadVariableOp2H
"lstm_cell_24/MatMul/ReadVariableOp"lstm_cell_24/MatMul/ReadVariableOp2L
$lstm_cell_24/MatMul_1/ReadVariableOp$lstm_cell_24/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ]
"
_user_specified_name
inputs/0
\

E__inference_lstm_25_layer_call_and_return_conditional_losses_39973839

inputs>
+lstm_cell_25_matmul_readvariableop_resource:	`è
A
-lstm_cell_25_matmul_1_readvariableop_resource:
Úè
;
,lstm_cell_25_biasadd_readvariableop_resource:	è

identity¢#lstm_cell_25/BiasAdd/ReadVariableOp¢"lstm_cell_25/MatMul/ReadVariableOp¢$lstm_cell_25/MatMul_1/ReadVariableOp¢whileD
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
:ÿÿÿÿÿÿÿÿÿ`2
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
valueB"ÿÿÿÿ`   27
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
:ÿÿÿÿÿÿÿÿÿ`*
shrink_axis_mask2
strided_slice_2µ
"lstm_cell_25/MatMul/ReadVariableOpReadVariableOp+lstm_cell_25_matmul_readvariableop_resource*
_output_shapes
:	`è
*
dtype02$
"lstm_cell_25/MatMul/ReadVariableOp­
lstm_cell_25/MatMulMatMulstrided_slice_2:output:0*lstm_cell_25/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè
2
lstm_cell_25/MatMul¼
$lstm_cell_25/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_25_matmul_1_readvariableop_resource* 
_output_shapes
:
Úè
*
dtype02&
$lstm_cell_25/MatMul_1/ReadVariableOp©
lstm_cell_25/MatMul_1MatMulzeros:output:0,lstm_cell_25/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè
2
lstm_cell_25/MatMul_1 
lstm_cell_25/addAddV2lstm_cell_25/MatMul:product:0lstm_cell_25/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè
2
lstm_cell_25/add´
#lstm_cell_25/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_25_biasadd_readvariableop_resource*
_output_shapes	
:è
*
dtype02%
#lstm_cell_25/BiasAdd/ReadVariableOp­
lstm_cell_25/BiasAddBiasAddlstm_cell_25/add:z:0+lstm_cell_25/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè
2
lstm_cell_25/BiasAdd~
lstm_cell_25/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_25/split/split_dim÷
lstm_cell_25/splitSplit%lstm_cell_25/split/split_dim:output:0lstm_cell_25/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿÚ:ÿÿÿÿÿÿÿÿÿÚ:ÿÿÿÿÿÿÿÿÿÚ:ÿÿÿÿÿÿÿÿÿÚ*
	num_split2
lstm_cell_25/split
lstm_cell_25/SigmoidSigmoidlstm_cell_25/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
lstm_cell_25/Sigmoid
lstm_cell_25/Sigmoid_1Sigmoidlstm_cell_25/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
lstm_cell_25/Sigmoid_1
lstm_cell_25/mulMullstm_cell_25/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
lstm_cell_25/mul~
lstm_cell_25/ReluRelulstm_cell_25/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
lstm_cell_25/Relu
lstm_cell_25/mul_1Mullstm_cell_25/Sigmoid:y:0lstm_cell_25/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
lstm_cell_25/mul_1
lstm_cell_25/add_1AddV2lstm_cell_25/mul:z:0lstm_cell_25/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
lstm_cell_25/add_1
lstm_cell_25/Sigmoid_2Sigmoidlstm_cell_25/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
lstm_cell_25/Sigmoid_2}
lstm_cell_25/Relu_1Relulstm_cell_25/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
lstm_cell_25/Relu_1¡
lstm_cell_25/mul_2Mullstm_cell_25/Sigmoid_2:y:0!lstm_cell_25/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
lstm_cell_25/mul_2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_25_matmul_readvariableop_resource-lstm_cell_25_matmul_1_readvariableop_resource,lstm_cell_25_biasadd_readvariableop_resource*
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
while_body_39973755*
condR
while_cond_39973754*M
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
NoOpNoOp$^lstm_cell_25/BiasAdd/ReadVariableOp#^lstm_cell_25/MatMul/ReadVariableOp%^lstm_cell_25/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ`: : : 2J
#lstm_cell_25/BiasAdd/ReadVariableOp#lstm_cell_25/BiasAdd/ReadVariableOp2H
"lstm_cell_25/MatMul/ReadVariableOp"lstm_cell_25/MatMul/ReadVariableOp2L
$lstm_cell_25/MatMul_1/ReadVariableOp$lstm_cell_25/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
 
_user_specified_nameinputs
×
g
H__inference_dropout_25_layer_call_and_return_conditional_losses_39971311

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
 *ÍÌL>2
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
Ô

í
lstm_25_while_cond_39972398,
(lstm_25_while_lstm_25_while_loop_counter2
.lstm_25_while_lstm_25_while_maximum_iterations
lstm_25_while_placeholder
lstm_25_while_placeholder_1
lstm_25_while_placeholder_2
lstm_25_while_placeholder_3.
*lstm_25_while_less_lstm_25_strided_slice_1F
Blstm_25_while_lstm_25_while_cond_39972398___redundant_placeholder0F
Blstm_25_while_lstm_25_while_cond_39972398___redundant_placeholder1F
Blstm_25_while_lstm_25_while_cond_39972398___redundant_placeholder2F
Blstm_25_while_lstm_25_while_cond_39972398___redundant_placeholder3
lstm_25_while_identity

lstm_25/while/LessLesslstm_25_while_placeholder*lstm_25_while_less_lstm_25_strided_slice_1*
T0*
_output_shapes
: 2
lstm_25/while/Lessu
lstm_25/while/IdentityIdentitylstm_25/while/Less:z:0*
T0
*
_output_shapes
: 2
lstm_25/while/Identity"9
lstm_25_while_identitylstm_25/while/Identity:output:0*(
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
while_body_39971125
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_25_matmul_readvariableop_resource_0:	`è
I
5while_lstm_cell_25_matmul_1_readvariableop_resource_0:
Úè
C
4while_lstm_cell_25_biasadd_readvariableop_resource_0:	è

while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_25_matmul_readvariableop_resource:	`è
G
3while_lstm_cell_25_matmul_1_readvariableop_resource:
Úè
A
2while_lstm_cell_25_biasadd_readvariableop_resource:	è
¢)while/lstm_cell_25/BiasAdd/ReadVariableOp¢(while/lstm_cell_25/MatMul/ReadVariableOp¢*while/lstm_cell_25/MatMul_1/ReadVariableOpÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ`   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÓ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemÉ
(while/lstm_cell_25/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_25_matmul_readvariableop_resource_0*
_output_shapes
:	`è
*
dtype02*
(while/lstm_cell_25/MatMul/ReadVariableOp×
while/lstm_cell_25/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_25/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè
2
while/lstm_cell_25/MatMulÐ
*while/lstm_cell_25/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_25_matmul_1_readvariableop_resource_0* 
_output_shapes
:
Úè
*
dtype02,
*while/lstm_cell_25/MatMul_1/ReadVariableOpÀ
while/lstm_cell_25/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_25/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè
2
while/lstm_cell_25/MatMul_1¸
while/lstm_cell_25/addAddV2#while/lstm_cell_25/MatMul:product:0%while/lstm_cell_25/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè
2
while/lstm_cell_25/addÈ
)while/lstm_cell_25/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_25_biasadd_readvariableop_resource_0*
_output_shapes	
:è
*
dtype02+
)while/lstm_cell_25/BiasAdd/ReadVariableOpÅ
while/lstm_cell_25/BiasAddBiasAddwhile/lstm_cell_25/add:z:01while/lstm_cell_25/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè
2
while/lstm_cell_25/BiasAdd
"while/lstm_cell_25/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_25/split/split_dim
while/lstm_cell_25/splitSplit+while/lstm_cell_25/split/split_dim:output:0#while/lstm_cell_25/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿÚ:ÿÿÿÿÿÿÿÿÿÚ:ÿÿÿÿÿÿÿÿÿÚ:ÿÿÿÿÿÿÿÿÿÚ*
	num_split2
while/lstm_cell_25/split
while/lstm_cell_25/SigmoidSigmoid!while/lstm_cell_25/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
while/lstm_cell_25/Sigmoid
while/lstm_cell_25/Sigmoid_1Sigmoid!while/lstm_cell_25/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
while/lstm_cell_25/Sigmoid_1¡
while/lstm_cell_25/mulMul while/lstm_cell_25/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
while/lstm_cell_25/mul
while/lstm_cell_25/ReluRelu!while/lstm_cell_25/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
while/lstm_cell_25/Reluµ
while/lstm_cell_25/mul_1Mulwhile/lstm_cell_25/Sigmoid:y:0%while/lstm_cell_25/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
while/lstm_cell_25/mul_1ª
while/lstm_cell_25/add_1AddV2while/lstm_cell_25/mul:z:0while/lstm_cell_25/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
while/lstm_cell_25/add_1
while/lstm_cell_25/Sigmoid_2Sigmoid!while/lstm_cell_25/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
while/lstm_cell_25/Sigmoid_2
while/lstm_cell_25/Relu_1Reluwhile/lstm_cell_25/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
while/lstm_cell_25/Relu_1¹
while/lstm_cell_25/mul_2Mul while/lstm_cell_25/Sigmoid_2:y:0'while/lstm_cell_25/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
while/lstm_cell_25/mul_2à
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_25/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_25/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_25/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
while/Identity_5Þ

while/NoOpNoOp*^while/lstm_cell_25/BiasAdd/ReadVariableOp)^while/lstm_cell_25/MatMul/ReadVariableOp+^while/lstm_cell_25/MatMul_1/ReadVariableOp*"
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
2while_lstm_cell_25_biasadd_readvariableop_resource4while_lstm_cell_25_biasadd_readvariableop_resource_0"l
3while_lstm_cell_25_matmul_1_readvariableop_resource5while_lstm_cell_25_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_25_matmul_readvariableop_resource3while_lstm_cell_25_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿÚ:ÿÿÿÿÿÿÿÿÿÚ: : : : : 2V
)while/lstm_cell_25/BiasAdd/ReadVariableOp)while/lstm_cell_25/BiasAdd/ReadVariableOp2T
(while/lstm_cell_25/MatMul/ReadVariableOp(while/lstm_cell_25/MatMul/ReadVariableOp2X
*while/lstm_cell_25/MatMul_1/ReadVariableOp*while/lstm_cell_25/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
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
while_cond_39971393
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_39971393___redundant_placeholder06
2while_while_cond_39971393___redundant_placeholder16
2while_while_cond_39971393___redundant_placeholder26
2while_while_cond_39971393___redundant_placeholder3
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
¹F

E__inference_lstm_24_layer_call_and_return_conditional_losses_39969995

inputs(
lstm_cell_24_39969913:	](
lstm_cell_24_39969915:	`$
lstm_cell_24_39969917:	
identity¢$lstm_cell_24/StatefulPartitionedCall¢whileD
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
value	B :`2
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
value	B :`2
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
:ÿÿÿÿÿÿÿÿÿ`2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :`2
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
value	B :`2
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
:ÿÿÿÿÿÿÿÿÿ`2	
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
strided_slice_2¥
$lstm_cell_24/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_24_39969913lstm_cell_24_39969915lstm_cell_24_39969917*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ`:ÿÿÿÿÿÿÿÿÿ`:ÿÿÿÿÿÿÿÿÿ`*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_lstm_cell_24_layer_call_and_return_conditional_losses_399698482&
$lstm_cell_24/StatefulPartitionedCall
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ`   2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_24_39969913lstm_cell_24_39969915lstm_cell_24_39969917*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ`:ÿÿÿÿÿÿÿÿÿ`: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_39969926*
condR
while_cond_39969925*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ`:ÿÿÿÿÿÿÿÿÿ`: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ`   22
0TensorArrayV2Stack/TensorListStack/element_shapeñ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`*
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
:ÿÿÿÿÿÿÿÿÿ`*
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
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`2
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
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`2

Identity}
NoOpNoOp%^lstm_cell_24/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ]: : : 2L
$lstm_cell_24/StatefulPartitionedCall$lstm_cell_24/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ]
 
_user_specified_nameinputs
Ý
¹
*__inference_lstm_24_layer_call_fn_39973186
inputs_0
unknown:	]
	unknown_0:	`
	unknown_1:	
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_lstm_24_layer_call_and_return_conditional_losses_399699952
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`2

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
×
g
H__inference_dropout_25_layer_call_and_return_conditional_losses_39973900

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
 *ÍÌL>2
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
ß
Í
while_cond_39972777
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_39972777___redundant_placeholder06
2while_while_cond_39972777___redundant_placeholder16
2while_while_cond_39972777___redundant_placeholder26
2while_while_cond_39972777___redundant_placeholder3
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
@: : : : :ÿÿÿÿÿÿÿÿÿ`:ÿÿÿÿÿÿÿÿÿ`: ::::: 

_output_shapes
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
:ÿÿÿÿÿÿÿÿÿ`:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`:

_output_shapes
: :

_output_shapes
:

f
H__inference_dropout_25_layer_call_and_return_conditional_losses_39973888

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
õ

J__inference_lstm_cell_24_layer_call_and_return_conditional_losses_39969702

inputs

states
states_11
matmul_readvariableop_resource:	]3
 matmul_1_readvariableop_resource:	`.
biasadd_readvariableop_resource:	
identity

identity_1

identity_2¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	]*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	`*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
add
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
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
L:ÿÿÿÿÿÿÿÿÿ`:ÿÿÿÿÿÿÿÿÿ`:ÿÿÿÿÿÿÿÿÿ`:ÿÿÿÿÿÿÿÿÿ`*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
	Sigmoid_1\
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
mulV
ReluRelusplit:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
	Sigmoid_2U
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
mul_2d
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2

Identityh

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2

Identity_1h

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2

Identity_2
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
?:ÿÿÿÿÿÿÿÿÿ]:ÿÿÿÿÿÿÿÿÿ`:ÿÿÿÿÿÿÿÿÿ`: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
 
_user_specified_namestates:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
 
_user_specified_namestates
?
Ò
while_body_39972929
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_24_matmul_readvariableop_resource_0:	]H
5while_lstm_cell_24_matmul_1_readvariableop_resource_0:	`C
4while_lstm_cell_24_biasadd_readvariableop_resource_0:	
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_24_matmul_readvariableop_resource:	]F
3while_lstm_cell_24_matmul_1_readvariableop_resource:	`A
2while_lstm_cell_24_biasadd_readvariableop_resource:	¢)while/lstm_cell_24/BiasAdd/ReadVariableOp¢(while/lstm_cell_24/MatMul/ReadVariableOp¢*while/lstm_cell_24/MatMul_1/ReadVariableOpÃ
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
(while/lstm_cell_24/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_24_matmul_readvariableop_resource_0*
_output_shapes
:	]*
dtype02*
(while/lstm_cell_24/MatMul/ReadVariableOp×
while/lstm_cell_24/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_24/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_24/MatMulÏ
*while/lstm_cell_24/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_24_matmul_1_readvariableop_resource_0*
_output_shapes
:	`*
dtype02,
*while/lstm_cell_24/MatMul_1/ReadVariableOpÀ
while/lstm_cell_24/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_24/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_24/MatMul_1¸
while/lstm_cell_24/addAddV2#while/lstm_cell_24/MatMul:product:0%while/lstm_cell_24/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_24/addÈ
)while/lstm_cell_24/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_24_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02+
)while/lstm_cell_24/BiasAdd/ReadVariableOpÅ
while/lstm_cell_24/BiasAddBiasAddwhile/lstm_cell_24/add:z:01while/lstm_cell_24/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_24/BiasAdd
"while/lstm_cell_24/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_24/split/split_dim
while/lstm_cell_24/splitSplit+while/lstm_cell_24/split/split_dim:output:0#while/lstm_cell_24/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ`:ÿÿÿÿÿÿÿÿÿ`:ÿÿÿÿÿÿÿÿÿ`:ÿÿÿÿÿÿÿÿÿ`*
	num_split2
while/lstm_cell_24/split
while/lstm_cell_24/SigmoidSigmoid!while/lstm_cell_24/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
while/lstm_cell_24/Sigmoid
while/lstm_cell_24/Sigmoid_1Sigmoid!while/lstm_cell_24/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
while/lstm_cell_24/Sigmoid_1 
while/lstm_cell_24/mulMul while/lstm_cell_24/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
while/lstm_cell_24/mul
while/lstm_cell_24/ReluRelu!while/lstm_cell_24/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
while/lstm_cell_24/Relu´
while/lstm_cell_24/mul_1Mulwhile/lstm_cell_24/Sigmoid:y:0%while/lstm_cell_24/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
while/lstm_cell_24/mul_1©
while/lstm_cell_24/add_1AddV2while/lstm_cell_24/mul:z:0while/lstm_cell_24/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
while/lstm_cell_24/add_1
while/lstm_cell_24/Sigmoid_2Sigmoid!while/lstm_cell_24/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
while/lstm_cell_24/Sigmoid_2
while/lstm_cell_24/Relu_1Reluwhile/lstm_cell_24/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
while/lstm_cell_24/Relu_1¸
while/lstm_cell_24/mul_2Mul while/lstm_cell_24/Sigmoid_2:y:0'while/lstm_cell_24/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
while/lstm_cell_24/mul_2à
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_24/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_24/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_24/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
while/Identity_5Þ

while/NoOpNoOp*^while/lstm_cell_24/BiasAdd/ReadVariableOp)^while/lstm_cell_24/MatMul/ReadVariableOp+^while/lstm_cell_24/MatMul_1/ReadVariableOp*"
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
2while_lstm_cell_24_biasadd_readvariableop_resource4while_lstm_cell_24_biasadd_readvariableop_resource_0"l
3while_lstm_cell_24_matmul_1_readvariableop_resource5while_lstm_cell_24_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_24_matmul_readvariableop_resource3while_lstm_cell_24_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ`:ÿÿÿÿÿÿÿÿÿ`: : : : : 2V
)while/lstm_cell_24/BiasAdd/ReadVariableOp)while/lstm_cell_24/BiasAdd/ReadVariableOp2T
(while/lstm_cell_24/MatMul/ReadVariableOp(while/lstm_cell_24/MatMul/ReadVariableOp2X
*while/lstm_cell_24/MatMul_1/ReadVariableOp*while/lstm_cell_24/MatMul_1/ReadVariableOp: 

_output_shapes
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
:ÿÿÿÿÿÿÿÿÿ`:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`:

_output_shapes
: :

_output_shapes
: 
ÏJ
Ò

lstm_24_while_body_39972244,
(lstm_24_while_lstm_24_while_loop_counter2
.lstm_24_while_lstm_24_while_maximum_iterations
lstm_24_while_placeholder
lstm_24_while_placeholder_1
lstm_24_while_placeholder_2
lstm_24_while_placeholder_3+
'lstm_24_while_lstm_24_strided_slice_1_0g
clstm_24_while_tensorarrayv2read_tensorlistgetitem_lstm_24_tensorarrayunstack_tensorlistfromtensor_0N
;lstm_24_while_lstm_cell_24_matmul_readvariableop_resource_0:	]P
=lstm_24_while_lstm_cell_24_matmul_1_readvariableop_resource_0:	`K
<lstm_24_while_lstm_cell_24_biasadd_readvariableop_resource_0:	
lstm_24_while_identity
lstm_24_while_identity_1
lstm_24_while_identity_2
lstm_24_while_identity_3
lstm_24_while_identity_4
lstm_24_while_identity_5)
%lstm_24_while_lstm_24_strided_slice_1e
alstm_24_while_tensorarrayv2read_tensorlistgetitem_lstm_24_tensorarrayunstack_tensorlistfromtensorL
9lstm_24_while_lstm_cell_24_matmul_readvariableop_resource:	]N
;lstm_24_while_lstm_cell_24_matmul_1_readvariableop_resource:	`I
:lstm_24_while_lstm_cell_24_biasadd_readvariableop_resource:	¢1lstm_24/while/lstm_cell_24/BiasAdd/ReadVariableOp¢0lstm_24/while/lstm_cell_24/MatMul/ReadVariableOp¢2lstm_24/while/lstm_cell_24/MatMul_1/ReadVariableOpÓ
?lstm_24/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ]   2A
?lstm_24/while/TensorArrayV2Read/TensorListGetItem/element_shape
1lstm_24/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemclstm_24_while_tensorarrayv2read_tensorlistgetitem_lstm_24_tensorarrayunstack_tensorlistfromtensor_0lstm_24_while_placeholderHlstm_24/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]*
element_dtype023
1lstm_24/while/TensorArrayV2Read/TensorListGetItemá
0lstm_24/while/lstm_cell_24/MatMul/ReadVariableOpReadVariableOp;lstm_24_while_lstm_cell_24_matmul_readvariableop_resource_0*
_output_shapes
:	]*
dtype022
0lstm_24/while/lstm_cell_24/MatMul/ReadVariableOp÷
!lstm_24/while/lstm_cell_24/MatMulMatMul8lstm_24/while/TensorArrayV2Read/TensorListGetItem:item:08lstm_24/while/lstm_cell_24/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!lstm_24/while/lstm_cell_24/MatMulç
2lstm_24/while/lstm_cell_24/MatMul_1/ReadVariableOpReadVariableOp=lstm_24_while_lstm_cell_24_matmul_1_readvariableop_resource_0*
_output_shapes
:	`*
dtype024
2lstm_24/while/lstm_cell_24/MatMul_1/ReadVariableOpà
#lstm_24/while/lstm_cell_24/MatMul_1MatMullstm_24_while_placeholder_2:lstm_24/while/lstm_cell_24/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2%
#lstm_24/while/lstm_cell_24/MatMul_1Ø
lstm_24/while/lstm_cell_24/addAddV2+lstm_24/while/lstm_cell_24/MatMul:product:0-lstm_24/while/lstm_cell_24/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
lstm_24/while/lstm_cell_24/addà
1lstm_24/while/lstm_cell_24/BiasAdd/ReadVariableOpReadVariableOp<lstm_24_while_lstm_cell_24_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype023
1lstm_24/while/lstm_cell_24/BiasAdd/ReadVariableOpå
"lstm_24/while/lstm_cell_24/BiasAddBiasAdd"lstm_24/while/lstm_cell_24/add:z:09lstm_24/while/lstm_cell_24/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2$
"lstm_24/while/lstm_cell_24/BiasAdd
*lstm_24/while/lstm_cell_24/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2,
*lstm_24/while/lstm_cell_24/split/split_dim«
 lstm_24/while/lstm_cell_24/splitSplit3lstm_24/while/lstm_cell_24/split/split_dim:output:0+lstm_24/while/lstm_cell_24/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ`:ÿÿÿÿÿÿÿÿÿ`:ÿÿÿÿÿÿÿÿÿ`:ÿÿÿÿÿÿÿÿÿ`*
	num_split2"
 lstm_24/while/lstm_cell_24/split°
"lstm_24/while/lstm_cell_24/SigmoidSigmoid)lstm_24/while/lstm_cell_24/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2$
"lstm_24/while/lstm_cell_24/Sigmoid´
$lstm_24/while/lstm_cell_24/Sigmoid_1Sigmoid)lstm_24/while/lstm_cell_24/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2&
$lstm_24/while/lstm_cell_24/Sigmoid_1À
lstm_24/while/lstm_cell_24/mulMul(lstm_24/while/lstm_cell_24/Sigmoid_1:y:0lstm_24_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2 
lstm_24/while/lstm_cell_24/mul§
lstm_24/while/lstm_cell_24/ReluRelu)lstm_24/while/lstm_cell_24/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2!
lstm_24/while/lstm_cell_24/ReluÔ
 lstm_24/while/lstm_cell_24/mul_1Mul&lstm_24/while/lstm_cell_24/Sigmoid:y:0-lstm_24/while/lstm_cell_24/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2"
 lstm_24/while/lstm_cell_24/mul_1É
 lstm_24/while/lstm_cell_24/add_1AddV2"lstm_24/while/lstm_cell_24/mul:z:0$lstm_24/while/lstm_cell_24/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2"
 lstm_24/while/lstm_cell_24/add_1´
$lstm_24/while/lstm_cell_24/Sigmoid_2Sigmoid)lstm_24/while/lstm_cell_24/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2&
$lstm_24/while/lstm_cell_24/Sigmoid_2¦
!lstm_24/while/lstm_cell_24/Relu_1Relu$lstm_24/while/lstm_cell_24/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2#
!lstm_24/while/lstm_cell_24/Relu_1Ø
 lstm_24/while/lstm_cell_24/mul_2Mul(lstm_24/while/lstm_cell_24/Sigmoid_2:y:0/lstm_24/while/lstm_cell_24/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2"
 lstm_24/while/lstm_cell_24/mul_2
2lstm_24/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_24_while_placeholder_1lstm_24_while_placeholder$lstm_24/while/lstm_cell_24/mul_2:z:0*
_output_shapes
: *
element_dtype024
2lstm_24/while/TensorArrayV2Write/TensorListSetIteml
lstm_24/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_24/while/add/y
lstm_24/while/addAddV2lstm_24_while_placeholderlstm_24/while/add/y:output:0*
T0*
_output_shapes
: 2
lstm_24/while/addp
lstm_24/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_24/while/add_1/y
lstm_24/while/add_1AddV2(lstm_24_while_lstm_24_while_loop_counterlstm_24/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lstm_24/while/add_1
lstm_24/while/IdentityIdentitylstm_24/while/add_1:z:0^lstm_24/while/NoOp*
T0*
_output_shapes
: 2
lstm_24/while/Identity¦
lstm_24/while/Identity_1Identity.lstm_24_while_lstm_24_while_maximum_iterations^lstm_24/while/NoOp*
T0*
_output_shapes
: 2
lstm_24/while/Identity_1
lstm_24/while/Identity_2Identitylstm_24/while/add:z:0^lstm_24/while/NoOp*
T0*
_output_shapes
: 2
lstm_24/while/Identity_2º
lstm_24/while/Identity_3IdentityBlstm_24/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_24/while/NoOp*
T0*
_output_shapes
: 2
lstm_24/while/Identity_3­
lstm_24/while/Identity_4Identity$lstm_24/while/lstm_cell_24/mul_2:z:0^lstm_24/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
lstm_24/while/Identity_4­
lstm_24/while/Identity_5Identity$lstm_24/while/lstm_cell_24/add_1:z:0^lstm_24/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
lstm_24/while/Identity_5
lstm_24/while/NoOpNoOp2^lstm_24/while/lstm_cell_24/BiasAdd/ReadVariableOp1^lstm_24/while/lstm_cell_24/MatMul/ReadVariableOp3^lstm_24/while/lstm_cell_24/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
lstm_24/while/NoOp"9
lstm_24_while_identitylstm_24/while/Identity:output:0"=
lstm_24_while_identity_1!lstm_24/while/Identity_1:output:0"=
lstm_24_while_identity_2!lstm_24/while/Identity_2:output:0"=
lstm_24_while_identity_3!lstm_24/while/Identity_3:output:0"=
lstm_24_while_identity_4!lstm_24/while/Identity_4:output:0"=
lstm_24_while_identity_5!lstm_24/while/Identity_5:output:0"P
%lstm_24_while_lstm_24_strided_slice_1'lstm_24_while_lstm_24_strided_slice_1_0"z
:lstm_24_while_lstm_cell_24_biasadd_readvariableop_resource<lstm_24_while_lstm_cell_24_biasadd_readvariableop_resource_0"|
;lstm_24_while_lstm_cell_24_matmul_1_readvariableop_resource=lstm_24_while_lstm_cell_24_matmul_1_readvariableop_resource_0"x
9lstm_24_while_lstm_cell_24_matmul_readvariableop_resource;lstm_24_while_lstm_cell_24_matmul_readvariableop_resource_0"È
alstm_24_while_tensorarrayv2read_tensorlistgetitem_lstm_24_tensorarrayunstack_tensorlistfromtensorclstm_24_while_tensorarrayv2read_tensorlistgetitem_lstm_24_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ`:ÿÿÿÿÿÿÿÿÿ`: : : : : 2f
1lstm_24/while/lstm_cell_24/BiasAdd/ReadVariableOp1lstm_24/while/lstm_cell_24/BiasAdd/ReadVariableOp2d
0lstm_24/while/lstm_cell_24/MatMul/ReadVariableOp0lstm_24/while/lstm_cell_24/MatMul/ReadVariableOp2h
2lstm_24/while/lstm_cell_24/MatMul_1/ReadVariableOp2lstm_24/while/lstm_cell_24/MatMul_1/ReadVariableOp: 

_output_shapes
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
:ÿÿÿÿÿÿÿÿÿ`:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`:

_output_shapes
: :

_output_shapes
: 
ú	
È
&__inference_signature_wrapper_39971850
lstm_24_input
unknown:	]
	unknown_0:	`
	unknown_1:	
	unknown_2:	`è

	unknown_3:
Úè

	unknown_4:	è

	unknown_5:	Ú
	unknown_6:
identity¢StatefulPartitionedCall¬
StatefulPartitionedCallStatefulPartitionedCalllstm_24_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
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
#__inference__wrapped_model_399696272
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
_user_specified_namelstm_24_input
ù

)sequential_12_lstm_25_while_cond_39969514H
Dsequential_12_lstm_25_while_sequential_12_lstm_25_while_loop_counterN
Jsequential_12_lstm_25_while_sequential_12_lstm_25_while_maximum_iterations+
'sequential_12_lstm_25_while_placeholder-
)sequential_12_lstm_25_while_placeholder_1-
)sequential_12_lstm_25_while_placeholder_2-
)sequential_12_lstm_25_while_placeholder_3J
Fsequential_12_lstm_25_while_less_sequential_12_lstm_25_strided_slice_1b
^sequential_12_lstm_25_while_sequential_12_lstm_25_while_cond_39969514___redundant_placeholder0b
^sequential_12_lstm_25_while_sequential_12_lstm_25_while_cond_39969514___redundant_placeholder1b
^sequential_12_lstm_25_while_sequential_12_lstm_25_while_cond_39969514___redundant_placeholder2b
^sequential_12_lstm_25_while_sequential_12_lstm_25_while_cond_39969514___redundant_placeholder3(
$sequential_12_lstm_25_while_identity
Þ
 sequential_12/lstm_25/while/LessLess'sequential_12_lstm_25_while_placeholderFsequential_12_lstm_25_while_less_sequential_12_lstm_25_strided_slice_1*
T0*
_output_shapes
: 2"
 sequential_12/lstm_25/while/Less
$sequential_12/lstm_25/while/IdentityIdentity$sequential_12/lstm_25/while/Less:z:0*
T0
*
_output_shapes
: 2&
$sequential_12/lstm_25/while/Identity"U
$sequential_12_lstm_25_while_identity-sequential_12/lstm_25/while/Identity:output:0*(
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
?
Ò
while_body_39973080
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_24_matmul_readvariableop_resource_0:	]H
5while_lstm_cell_24_matmul_1_readvariableop_resource_0:	`C
4while_lstm_cell_24_biasadd_readvariableop_resource_0:	
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_24_matmul_readvariableop_resource:	]F
3while_lstm_cell_24_matmul_1_readvariableop_resource:	`A
2while_lstm_cell_24_biasadd_readvariableop_resource:	¢)while/lstm_cell_24/BiasAdd/ReadVariableOp¢(while/lstm_cell_24/MatMul/ReadVariableOp¢*while/lstm_cell_24/MatMul_1/ReadVariableOpÃ
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
(while/lstm_cell_24/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_24_matmul_readvariableop_resource_0*
_output_shapes
:	]*
dtype02*
(while/lstm_cell_24/MatMul/ReadVariableOp×
while/lstm_cell_24/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_24/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_24/MatMulÏ
*while/lstm_cell_24/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_24_matmul_1_readvariableop_resource_0*
_output_shapes
:	`*
dtype02,
*while/lstm_cell_24/MatMul_1/ReadVariableOpÀ
while/lstm_cell_24/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_24/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_24/MatMul_1¸
while/lstm_cell_24/addAddV2#while/lstm_cell_24/MatMul:product:0%while/lstm_cell_24/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_24/addÈ
)while/lstm_cell_24/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_24_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02+
)while/lstm_cell_24/BiasAdd/ReadVariableOpÅ
while/lstm_cell_24/BiasAddBiasAddwhile/lstm_cell_24/add:z:01while/lstm_cell_24/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_24/BiasAdd
"while/lstm_cell_24/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_24/split/split_dim
while/lstm_cell_24/splitSplit+while/lstm_cell_24/split/split_dim:output:0#while/lstm_cell_24/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ`:ÿÿÿÿÿÿÿÿÿ`:ÿÿÿÿÿÿÿÿÿ`:ÿÿÿÿÿÿÿÿÿ`*
	num_split2
while/lstm_cell_24/split
while/lstm_cell_24/SigmoidSigmoid!while/lstm_cell_24/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
while/lstm_cell_24/Sigmoid
while/lstm_cell_24/Sigmoid_1Sigmoid!while/lstm_cell_24/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
while/lstm_cell_24/Sigmoid_1 
while/lstm_cell_24/mulMul while/lstm_cell_24/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
while/lstm_cell_24/mul
while/lstm_cell_24/ReluRelu!while/lstm_cell_24/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
while/lstm_cell_24/Relu´
while/lstm_cell_24/mul_1Mulwhile/lstm_cell_24/Sigmoid:y:0%while/lstm_cell_24/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
while/lstm_cell_24/mul_1©
while/lstm_cell_24/add_1AddV2while/lstm_cell_24/mul:z:0while/lstm_cell_24/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
while/lstm_cell_24/add_1
while/lstm_cell_24/Sigmoid_2Sigmoid!while/lstm_cell_24/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
while/lstm_cell_24/Sigmoid_2
while/lstm_cell_24/Relu_1Reluwhile/lstm_cell_24/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
while/lstm_cell_24/Relu_1¸
while/lstm_cell_24/mul_2Mul while/lstm_cell_24/Sigmoid_2:y:0'while/lstm_cell_24/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
while/lstm_cell_24/mul_2à
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_24/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_24/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_24/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
while/Identity_5Þ

while/NoOpNoOp*^while/lstm_cell_24/BiasAdd/ReadVariableOp)^while/lstm_cell_24/MatMul/ReadVariableOp+^while/lstm_cell_24/MatMul_1/ReadVariableOp*"
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
2while_lstm_cell_24_biasadd_readvariableop_resource4while_lstm_cell_24_biasadd_readvariableop_resource_0"l
3while_lstm_cell_24_matmul_1_readvariableop_resource5while_lstm_cell_24_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_24_matmul_readvariableop_resource3while_lstm_cell_24_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ`:ÿÿÿÿÿÿÿÿÿ`: : : : : 2V
)while/lstm_cell_24/BiasAdd/ReadVariableOp)while/lstm_cell_24/BiasAdd/ReadVariableOp2T
(while/lstm_cell_24/MatMul/ReadVariableOp(while/lstm_cell_24/MatMul/ReadVariableOp2X
*while/lstm_cell_24/MatMul_1/ReadVariableOp*while/lstm_cell_24/MatMul_1/ReadVariableOp: 

_output_shapes
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
:ÿÿÿÿÿÿÿÿÿ`:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`:

_output_shapes
: :

_output_shapes
: 
¦\

E__inference_lstm_24_layer_call_and_return_conditional_losses_39972711
inputs_0>
+lstm_cell_24_matmul_readvariableop_resource:	]@
-lstm_cell_24_matmul_1_readvariableop_resource:	`;
,lstm_cell_24_biasadd_readvariableop_resource:	
identity¢#lstm_cell_24/BiasAdd/ReadVariableOp¢"lstm_cell_24/MatMul/ReadVariableOp¢$lstm_cell_24/MatMul_1/ReadVariableOp¢whileF
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
value	B :`2
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
value	B :`2
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
:ÿÿÿÿÿÿÿÿÿ`2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :`2
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
value	B :`2
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
:ÿÿÿÿÿÿÿÿÿ`2	
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
"lstm_cell_24/MatMul/ReadVariableOpReadVariableOp+lstm_cell_24_matmul_readvariableop_resource*
_output_shapes
:	]*
dtype02$
"lstm_cell_24/MatMul/ReadVariableOp­
lstm_cell_24/MatMulMatMulstrided_slice_2:output:0*lstm_cell_24/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_24/MatMul»
$lstm_cell_24/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_24_matmul_1_readvariableop_resource*
_output_shapes
:	`*
dtype02&
$lstm_cell_24/MatMul_1/ReadVariableOp©
lstm_cell_24/MatMul_1MatMulzeros:output:0,lstm_cell_24/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_24/MatMul_1 
lstm_cell_24/addAddV2lstm_cell_24/MatMul:product:0lstm_cell_24/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_24/add´
#lstm_cell_24/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_24_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02%
#lstm_cell_24/BiasAdd/ReadVariableOp­
lstm_cell_24/BiasAddBiasAddlstm_cell_24/add:z:0+lstm_cell_24/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_24/BiasAdd~
lstm_cell_24/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_24/split/split_dimó
lstm_cell_24/splitSplit%lstm_cell_24/split/split_dim:output:0lstm_cell_24/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ`:ÿÿÿÿÿÿÿÿÿ`:ÿÿÿÿÿÿÿÿÿ`:ÿÿÿÿÿÿÿÿÿ`*
	num_split2
lstm_cell_24/split
lstm_cell_24/SigmoidSigmoidlstm_cell_24/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
lstm_cell_24/Sigmoid
lstm_cell_24/Sigmoid_1Sigmoidlstm_cell_24/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
lstm_cell_24/Sigmoid_1
lstm_cell_24/mulMullstm_cell_24/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
lstm_cell_24/mul}
lstm_cell_24/ReluRelulstm_cell_24/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
lstm_cell_24/Relu
lstm_cell_24/mul_1Mullstm_cell_24/Sigmoid:y:0lstm_cell_24/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
lstm_cell_24/mul_1
lstm_cell_24/add_1AddV2lstm_cell_24/mul:z:0lstm_cell_24/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
lstm_cell_24/add_1
lstm_cell_24/Sigmoid_2Sigmoidlstm_cell_24/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
lstm_cell_24/Sigmoid_2|
lstm_cell_24/Relu_1Relulstm_cell_24/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
lstm_cell_24/Relu_1 
lstm_cell_24/mul_2Mullstm_cell_24/Sigmoid_2:y:0!lstm_cell_24/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
lstm_cell_24/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ`   2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_24_matmul_readvariableop_resource-lstm_cell_24_matmul_1_readvariableop_resource,lstm_cell_24_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ`:ÿÿÿÿÿÿÿÿÿ`: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_39972627*
condR
while_cond_39972626*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ`:ÿÿÿÿÿÿÿÿÿ`: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ`   22
0TensorArrayV2Stack/TensorListStack/element_shapeñ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`*
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
:ÿÿÿÿÿÿÿÿÿ`*
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
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`2
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
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`2

IdentityÈ
NoOpNoOp$^lstm_cell_24/BiasAdd/ReadVariableOp#^lstm_cell_24/MatMul/ReadVariableOp%^lstm_cell_24/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ]: : : 2J
#lstm_cell_24/BiasAdd/ReadVariableOp#lstm_cell_24/BiasAdd/ReadVariableOp2H
"lstm_cell_24/MatMul/ReadVariableOp"lstm_cell_24/MatMul/ReadVariableOp2L
$lstm_cell_24/MatMul_1/ReadVariableOp$lstm_cell_24/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ]
"
_user_specified_name
inputs/0
ß
Í
while_cond_39973079
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_39973079___redundant_placeholder06
2while_while_cond_39973079___redundant_placeholder16
2while_while_cond_39973079___redundant_placeholder26
2while_while_cond_39973079___redundant_placeholder3
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
@: : : : :ÿÿÿÿÿÿÿÿÿ`:ÿÿÿÿÿÿÿÿÿ`: ::::: 

_output_shapes
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
:ÿÿÿÿÿÿÿÿÿ`:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`:

_output_shapes
: :

_output_shapes
:
»
f
-__inference_dropout_25_layer_call_fn_39973910

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
H__inference_dropout_25_layer_call_and_return_conditional_losses_399713112
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
¹
å	
#__inference__wrapped_model_39969627
lstm_24_inputT
Asequential_12_lstm_24_lstm_cell_24_matmul_readvariableop_resource:	]V
Csequential_12_lstm_24_lstm_cell_24_matmul_1_readvariableop_resource:	`Q
Bsequential_12_lstm_24_lstm_cell_24_biasadd_readvariableop_resource:	T
Asequential_12_lstm_25_lstm_cell_25_matmul_readvariableop_resource:	`è
W
Csequential_12_lstm_25_lstm_cell_25_matmul_1_readvariableop_resource:
Úè
Q
Bsequential_12_lstm_25_lstm_cell_25_biasadd_readvariableop_resource:	è
K
8sequential_12_dense_12_tensordot_readvariableop_resource:	ÚD
6sequential_12_dense_12_biasadd_readvariableop_resource:
identity¢-sequential_12/dense_12/BiasAdd/ReadVariableOp¢/sequential_12/dense_12/Tensordot/ReadVariableOp¢9sequential_12/lstm_24/lstm_cell_24/BiasAdd/ReadVariableOp¢8sequential_12/lstm_24/lstm_cell_24/MatMul/ReadVariableOp¢:sequential_12/lstm_24/lstm_cell_24/MatMul_1/ReadVariableOp¢sequential_12/lstm_24/while¢9sequential_12/lstm_25/lstm_cell_25/BiasAdd/ReadVariableOp¢8sequential_12/lstm_25/lstm_cell_25/MatMul/ReadVariableOp¢:sequential_12/lstm_25/lstm_cell_25/MatMul_1/ReadVariableOp¢sequential_12/lstm_25/whilew
sequential_12/lstm_24/ShapeShapelstm_24_input*
T0*
_output_shapes
:2
sequential_12/lstm_24/Shape 
)sequential_12/lstm_24/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)sequential_12/lstm_24/strided_slice/stack¤
+sequential_12/lstm_24/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential_12/lstm_24/strided_slice/stack_1¤
+sequential_12/lstm_24/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential_12/lstm_24/strided_slice/stack_2æ
#sequential_12/lstm_24/strided_sliceStridedSlice$sequential_12/lstm_24/Shape:output:02sequential_12/lstm_24/strided_slice/stack:output:04sequential_12/lstm_24/strided_slice/stack_1:output:04sequential_12/lstm_24/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#sequential_12/lstm_24/strided_slice
!sequential_12/lstm_24/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :`2#
!sequential_12/lstm_24/zeros/mul/yÄ
sequential_12/lstm_24/zeros/mulMul,sequential_12/lstm_24/strided_slice:output:0*sequential_12/lstm_24/zeros/mul/y:output:0*
T0*
_output_shapes
: 2!
sequential_12/lstm_24/zeros/mul
"sequential_12/lstm_24/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2$
"sequential_12/lstm_24/zeros/Less/y¿
 sequential_12/lstm_24/zeros/LessLess#sequential_12/lstm_24/zeros/mul:z:0+sequential_12/lstm_24/zeros/Less/y:output:0*
T0*
_output_shapes
: 2"
 sequential_12/lstm_24/zeros/Less
$sequential_12/lstm_24/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :`2&
$sequential_12/lstm_24/zeros/packed/1Û
"sequential_12/lstm_24/zeros/packedPack,sequential_12/lstm_24/strided_slice:output:0-sequential_12/lstm_24/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2$
"sequential_12/lstm_24/zeros/packed
!sequential_12/lstm_24/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!sequential_12/lstm_24/zeros/ConstÍ
sequential_12/lstm_24/zerosFill+sequential_12/lstm_24/zeros/packed:output:0*sequential_12/lstm_24/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
sequential_12/lstm_24/zeros
#sequential_12/lstm_24/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :`2%
#sequential_12/lstm_24/zeros_1/mul/yÊ
!sequential_12/lstm_24/zeros_1/mulMul,sequential_12/lstm_24/strided_slice:output:0,sequential_12/lstm_24/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2#
!sequential_12/lstm_24/zeros_1/mul
$sequential_12/lstm_24/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2&
$sequential_12/lstm_24/zeros_1/Less/yÇ
"sequential_12/lstm_24/zeros_1/LessLess%sequential_12/lstm_24/zeros_1/mul:z:0-sequential_12/lstm_24/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2$
"sequential_12/lstm_24/zeros_1/Less
&sequential_12/lstm_24/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :`2(
&sequential_12/lstm_24/zeros_1/packed/1á
$sequential_12/lstm_24/zeros_1/packedPack,sequential_12/lstm_24/strided_slice:output:0/sequential_12/lstm_24/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2&
$sequential_12/lstm_24/zeros_1/packed
#sequential_12/lstm_24/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2%
#sequential_12/lstm_24/zeros_1/ConstÕ
sequential_12/lstm_24/zeros_1Fill-sequential_12/lstm_24/zeros_1/packed:output:0,sequential_12/lstm_24/zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
sequential_12/lstm_24/zeros_1¡
$sequential_12/lstm_24/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2&
$sequential_12/lstm_24/transpose/permÃ
sequential_12/lstm_24/transpose	Transposelstm_24_input-sequential_12/lstm_24/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]2!
sequential_12/lstm_24/transpose
sequential_12/lstm_24/Shape_1Shape#sequential_12/lstm_24/transpose:y:0*
T0*
_output_shapes
:2
sequential_12/lstm_24/Shape_1¤
+sequential_12/lstm_24/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2-
+sequential_12/lstm_24/strided_slice_1/stack¨
-sequential_12/lstm_24/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_12/lstm_24/strided_slice_1/stack_1¨
-sequential_12/lstm_24/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_12/lstm_24/strided_slice_1/stack_2ò
%sequential_12/lstm_24/strided_slice_1StridedSlice&sequential_12/lstm_24/Shape_1:output:04sequential_12/lstm_24/strided_slice_1/stack:output:06sequential_12/lstm_24/strided_slice_1/stack_1:output:06sequential_12/lstm_24/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%sequential_12/lstm_24/strided_slice_1±
1sequential_12/lstm_24/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ23
1sequential_12/lstm_24/TensorArrayV2/element_shape
#sequential_12/lstm_24/TensorArrayV2TensorListReserve:sequential_12/lstm_24/TensorArrayV2/element_shape:output:0.sequential_12/lstm_24/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02%
#sequential_12/lstm_24/TensorArrayV2ë
Ksequential_12/lstm_24/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ]   2M
Ksequential_12/lstm_24/TensorArrayUnstack/TensorListFromTensor/element_shapeÐ
=sequential_12/lstm_24/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor#sequential_12/lstm_24/transpose:y:0Tsequential_12/lstm_24/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02?
=sequential_12/lstm_24/TensorArrayUnstack/TensorListFromTensor¤
+sequential_12/lstm_24/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2-
+sequential_12/lstm_24/strided_slice_2/stack¨
-sequential_12/lstm_24/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_12/lstm_24/strided_slice_2/stack_1¨
-sequential_12/lstm_24/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_12/lstm_24/strided_slice_2/stack_2
%sequential_12/lstm_24/strided_slice_2StridedSlice#sequential_12/lstm_24/transpose:y:04sequential_12/lstm_24/strided_slice_2/stack:output:06sequential_12/lstm_24/strided_slice_2/stack_1:output:06sequential_12/lstm_24/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]*
shrink_axis_mask2'
%sequential_12/lstm_24/strided_slice_2÷
8sequential_12/lstm_24/lstm_cell_24/MatMul/ReadVariableOpReadVariableOpAsequential_12_lstm_24_lstm_cell_24_matmul_readvariableop_resource*
_output_shapes
:	]*
dtype02:
8sequential_12/lstm_24/lstm_cell_24/MatMul/ReadVariableOp
)sequential_12/lstm_24/lstm_cell_24/MatMulMatMul.sequential_12/lstm_24/strided_slice_2:output:0@sequential_12/lstm_24/lstm_cell_24/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2+
)sequential_12/lstm_24/lstm_cell_24/MatMulý
:sequential_12/lstm_24/lstm_cell_24/MatMul_1/ReadVariableOpReadVariableOpCsequential_12_lstm_24_lstm_cell_24_matmul_1_readvariableop_resource*
_output_shapes
:	`*
dtype02<
:sequential_12/lstm_24/lstm_cell_24/MatMul_1/ReadVariableOp
+sequential_12/lstm_24/lstm_cell_24/MatMul_1MatMul$sequential_12/lstm_24/zeros:output:0Bsequential_12/lstm_24/lstm_cell_24/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2-
+sequential_12/lstm_24/lstm_cell_24/MatMul_1ø
&sequential_12/lstm_24/lstm_cell_24/addAddV23sequential_12/lstm_24/lstm_cell_24/MatMul:product:05sequential_12/lstm_24/lstm_cell_24/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&sequential_12/lstm_24/lstm_cell_24/addö
9sequential_12/lstm_24/lstm_cell_24/BiasAdd/ReadVariableOpReadVariableOpBsequential_12_lstm_24_lstm_cell_24_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02;
9sequential_12/lstm_24/lstm_cell_24/BiasAdd/ReadVariableOp
*sequential_12/lstm_24/lstm_cell_24/BiasAddBiasAdd*sequential_12/lstm_24/lstm_cell_24/add:z:0Asequential_12/lstm_24/lstm_cell_24/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2,
*sequential_12/lstm_24/lstm_cell_24/BiasAddª
2sequential_12/lstm_24/lstm_cell_24/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :24
2sequential_12/lstm_24/lstm_cell_24/split/split_dimË
(sequential_12/lstm_24/lstm_cell_24/splitSplit;sequential_12/lstm_24/lstm_cell_24/split/split_dim:output:03sequential_12/lstm_24/lstm_cell_24/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ`:ÿÿÿÿÿÿÿÿÿ`:ÿÿÿÿÿÿÿÿÿ`:ÿÿÿÿÿÿÿÿÿ`*
	num_split2*
(sequential_12/lstm_24/lstm_cell_24/splitÈ
*sequential_12/lstm_24/lstm_cell_24/SigmoidSigmoid1sequential_12/lstm_24/lstm_cell_24/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2,
*sequential_12/lstm_24/lstm_cell_24/SigmoidÌ
,sequential_12/lstm_24/lstm_cell_24/Sigmoid_1Sigmoid1sequential_12/lstm_24/lstm_cell_24/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2.
,sequential_12/lstm_24/lstm_cell_24/Sigmoid_1ã
&sequential_12/lstm_24/lstm_cell_24/mulMul0sequential_12/lstm_24/lstm_cell_24/Sigmoid_1:y:0&sequential_12/lstm_24/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2(
&sequential_12/lstm_24/lstm_cell_24/mul¿
'sequential_12/lstm_24/lstm_cell_24/ReluRelu1sequential_12/lstm_24/lstm_cell_24/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2)
'sequential_12/lstm_24/lstm_cell_24/Reluô
(sequential_12/lstm_24/lstm_cell_24/mul_1Mul.sequential_12/lstm_24/lstm_cell_24/Sigmoid:y:05sequential_12/lstm_24/lstm_cell_24/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2*
(sequential_12/lstm_24/lstm_cell_24/mul_1é
(sequential_12/lstm_24/lstm_cell_24/add_1AddV2*sequential_12/lstm_24/lstm_cell_24/mul:z:0,sequential_12/lstm_24/lstm_cell_24/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2*
(sequential_12/lstm_24/lstm_cell_24/add_1Ì
,sequential_12/lstm_24/lstm_cell_24/Sigmoid_2Sigmoid1sequential_12/lstm_24/lstm_cell_24/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2.
,sequential_12/lstm_24/lstm_cell_24/Sigmoid_2¾
)sequential_12/lstm_24/lstm_cell_24/Relu_1Relu,sequential_12/lstm_24/lstm_cell_24/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2+
)sequential_12/lstm_24/lstm_cell_24/Relu_1ø
(sequential_12/lstm_24/lstm_cell_24/mul_2Mul0sequential_12/lstm_24/lstm_cell_24/Sigmoid_2:y:07sequential_12/lstm_24/lstm_cell_24/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2*
(sequential_12/lstm_24/lstm_cell_24/mul_2»
3sequential_12/lstm_24/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ`   25
3sequential_12/lstm_24/TensorArrayV2_1/element_shape
%sequential_12/lstm_24/TensorArrayV2_1TensorListReserve<sequential_12/lstm_24/TensorArrayV2_1/element_shape:output:0.sequential_12/lstm_24/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02'
%sequential_12/lstm_24/TensorArrayV2_1z
sequential_12/lstm_24/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
sequential_12/lstm_24/time«
.sequential_12/lstm_24/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ20
.sequential_12/lstm_24/while/maximum_iterations
(sequential_12/lstm_24/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2*
(sequential_12/lstm_24/while/loop_counterÙ
sequential_12/lstm_24/whileWhile1sequential_12/lstm_24/while/loop_counter:output:07sequential_12/lstm_24/while/maximum_iterations:output:0#sequential_12/lstm_24/time:output:0.sequential_12/lstm_24/TensorArrayV2_1:handle:0$sequential_12/lstm_24/zeros:output:0&sequential_12/lstm_24/zeros_1:output:0.sequential_12/lstm_24/strided_slice_1:output:0Msequential_12/lstm_24/TensorArrayUnstack/TensorListFromTensor:output_handle:0Asequential_12_lstm_24_lstm_cell_24_matmul_readvariableop_resourceCsequential_12_lstm_24_lstm_cell_24_matmul_1_readvariableop_resourceBsequential_12_lstm_24_lstm_cell_24_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ`:ÿÿÿÿÿÿÿÿÿ`: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *5
body-R+
)sequential_12_lstm_24_while_body_39969367*5
cond-R+
)sequential_12_lstm_24_while_cond_39969366*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ`:ÿÿÿÿÿÿÿÿÿ`: : : : : *
parallel_iterations 2
sequential_12/lstm_24/whileá
Fsequential_12/lstm_24/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ`   2H
Fsequential_12/lstm_24/TensorArrayV2Stack/TensorListStack/element_shapeÀ
8sequential_12/lstm_24/TensorArrayV2Stack/TensorListStackTensorListStack$sequential_12/lstm_24/while:output:3Osequential_12/lstm_24/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`*
element_dtype02:
8sequential_12/lstm_24/TensorArrayV2Stack/TensorListStack­
+sequential_12/lstm_24/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2-
+sequential_12/lstm_24/strided_slice_3/stack¨
-sequential_12/lstm_24/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2/
-sequential_12/lstm_24/strided_slice_3/stack_1¨
-sequential_12/lstm_24/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_12/lstm_24/strided_slice_3/stack_2
%sequential_12/lstm_24/strided_slice_3StridedSliceAsequential_12/lstm_24/TensorArrayV2Stack/TensorListStack:tensor:04sequential_12/lstm_24/strided_slice_3/stack:output:06sequential_12/lstm_24/strided_slice_3/stack_1:output:06sequential_12/lstm_24/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`*
shrink_axis_mask2'
%sequential_12/lstm_24/strided_slice_3¥
&sequential_12/lstm_24/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2(
&sequential_12/lstm_24/transpose_1/permý
!sequential_12/lstm_24/transpose_1	TransposeAsequential_12/lstm_24/TensorArrayV2Stack/TensorListStack:tensor:0/sequential_12/lstm_24/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2#
!sequential_12/lstm_24/transpose_1
sequential_12/lstm_24/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
sequential_12/lstm_24/runtime¯
!sequential_12/dropout_24/IdentityIdentity%sequential_12/lstm_24/transpose_1:y:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2#
!sequential_12/dropout_24/Identity
sequential_12/lstm_25/ShapeShape*sequential_12/dropout_24/Identity:output:0*
T0*
_output_shapes
:2
sequential_12/lstm_25/Shape 
)sequential_12/lstm_25/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)sequential_12/lstm_25/strided_slice/stack¤
+sequential_12/lstm_25/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential_12/lstm_25/strided_slice/stack_1¤
+sequential_12/lstm_25/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential_12/lstm_25/strided_slice/stack_2æ
#sequential_12/lstm_25/strided_sliceStridedSlice$sequential_12/lstm_25/Shape:output:02sequential_12/lstm_25/strided_slice/stack:output:04sequential_12/lstm_25/strided_slice/stack_1:output:04sequential_12/lstm_25/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#sequential_12/lstm_25/strided_slice
!sequential_12/lstm_25/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :Ú2#
!sequential_12/lstm_25/zeros/mul/yÄ
sequential_12/lstm_25/zeros/mulMul,sequential_12/lstm_25/strided_slice:output:0*sequential_12/lstm_25/zeros/mul/y:output:0*
T0*
_output_shapes
: 2!
sequential_12/lstm_25/zeros/mul
"sequential_12/lstm_25/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2$
"sequential_12/lstm_25/zeros/Less/y¿
 sequential_12/lstm_25/zeros/LessLess#sequential_12/lstm_25/zeros/mul:z:0+sequential_12/lstm_25/zeros/Less/y:output:0*
T0*
_output_shapes
: 2"
 sequential_12/lstm_25/zeros/Less
$sequential_12/lstm_25/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :Ú2&
$sequential_12/lstm_25/zeros/packed/1Û
"sequential_12/lstm_25/zeros/packedPack,sequential_12/lstm_25/strided_slice:output:0-sequential_12/lstm_25/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2$
"sequential_12/lstm_25/zeros/packed
!sequential_12/lstm_25/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!sequential_12/lstm_25/zeros/ConstÎ
sequential_12/lstm_25/zerosFill+sequential_12/lstm_25/zeros/packed:output:0*sequential_12/lstm_25/zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
sequential_12/lstm_25/zeros
#sequential_12/lstm_25/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :Ú2%
#sequential_12/lstm_25/zeros_1/mul/yÊ
!sequential_12/lstm_25/zeros_1/mulMul,sequential_12/lstm_25/strided_slice:output:0,sequential_12/lstm_25/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2#
!sequential_12/lstm_25/zeros_1/mul
$sequential_12/lstm_25/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2&
$sequential_12/lstm_25/zeros_1/Less/yÇ
"sequential_12/lstm_25/zeros_1/LessLess%sequential_12/lstm_25/zeros_1/mul:z:0-sequential_12/lstm_25/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2$
"sequential_12/lstm_25/zeros_1/Less
&sequential_12/lstm_25/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :Ú2(
&sequential_12/lstm_25/zeros_1/packed/1á
$sequential_12/lstm_25/zeros_1/packedPack,sequential_12/lstm_25/strided_slice:output:0/sequential_12/lstm_25/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2&
$sequential_12/lstm_25/zeros_1/packed
#sequential_12/lstm_25/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2%
#sequential_12/lstm_25/zeros_1/ConstÖ
sequential_12/lstm_25/zeros_1Fill-sequential_12/lstm_25/zeros_1/packed:output:0,sequential_12/lstm_25/zeros_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
sequential_12/lstm_25/zeros_1¡
$sequential_12/lstm_25/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2&
$sequential_12/lstm_25/transpose/permà
sequential_12/lstm_25/transpose	Transpose*sequential_12/dropout_24/Identity:output:0-sequential_12/lstm_25/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2!
sequential_12/lstm_25/transpose
sequential_12/lstm_25/Shape_1Shape#sequential_12/lstm_25/transpose:y:0*
T0*
_output_shapes
:2
sequential_12/lstm_25/Shape_1¤
+sequential_12/lstm_25/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2-
+sequential_12/lstm_25/strided_slice_1/stack¨
-sequential_12/lstm_25/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_12/lstm_25/strided_slice_1/stack_1¨
-sequential_12/lstm_25/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_12/lstm_25/strided_slice_1/stack_2ò
%sequential_12/lstm_25/strided_slice_1StridedSlice&sequential_12/lstm_25/Shape_1:output:04sequential_12/lstm_25/strided_slice_1/stack:output:06sequential_12/lstm_25/strided_slice_1/stack_1:output:06sequential_12/lstm_25/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%sequential_12/lstm_25/strided_slice_1±
1sequential_12/lstm_25/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ23
1sequential_12/lstm_25/TensorArrayV2/element_shape
#sequential_12/lstm_25/TensorArrayV2TensorListReserve:sequential_12/lstm_25/TensorArrayV2/element_shape:output:0.sequential_12/lstm_25/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02%
#sequential_12/lstm_25/TensorArrayV2ë
Ksequential_12/lstm_25/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ`   2M
Ksequential_12/lstm_25/TensorArrayUnstack/TensorListFromTensor/element_shapeÐ
=sequential_12/lstm_25/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor#sequential_12/lstm_25/transpose:y:0Tsequential_12/lstm_25/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02?
=sequential_12/lstm_25/TensorArrayUnstack/TensorListFromTensor¤
+sequential_12/lstm_25/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2-
+sequential_12/lstm_25/strided_slice_2/stack¨
-sequential_12/lstm_25/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_12/lstm_25/strided_slice_2/stack_1¨
-sequential_12/lstm_25/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_12/lstm_25/strided_slice_2/stack_2
%sequential_12/lstm_25/strided_slice_2StridedSlice#sequential_12/lstm_25/transpose:y:04sequential_12/lstm_25/strided_slice_2/stack:output:06sequential_12/lstm_25/strided_slice_2/stack_1:output:06sequential_12/lstm_25/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`*
shrink_axis_mask2'
%sequential_12/lstm_25/strided_slice_2÷
8sequential_12/lstm_25/lstm_cell_25/MatMul/ReadVariableOpReadVariableOpAsequential_12_lstm_25_lstm_cell_25_matmul_readvariableop_resource*
_output_shapes
:	`è
*
dtype02:
8sequential_12/lstm_25/lstm_cell_25/MatMul/ReadVariableOp
)sequential_12/lstm_25/lstm_cell_25/MatMulMatMul.sequential_12/lstm_25/strided_slice_2:output:0@sequential_12/lstm_25/lstm_cell_25/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè
2+
)sequential_12/lstm_25/lstm_cell_25/MatMulþ
:sequential_12/lstm_25/lstm_cell_25/MatMul_1/ReadVariableOpReadVariableOpCsequential_12_lstm_25_lstm_cell_25_matmul_1_readvariableop_resource* 
_output_shapes
:
Úè
*
dtype02<
:sequential_12/lstm_25/lstm_cell_25/MatMul_1/ReadVariableOp
+sequential_12/lstm_25/lstm_cell_25/MatMul_1MatMul$sequential_12/lstm_25/zeros:output:0Bsequential_12/lstm_25/lstm_cell_25/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè
2-
+sequential_12/lstm_25/lstm_cell_25/MatMul_1ø
&sequential_12/lstm_25/lstm_cell_25/addAddV23sequential_12/lstm_25/lstm_cell_25/MatMul:product:05sequential_12/lstm_25/lstm_cell_25/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè
2(
&sequential_12/lstm_25/lstm_cell_25/addö
9sequential_12/lstm_25/lstm_cell_25/BiasAdd/ReadVariableOpReadVariableOpBsequential_12_lstm_25_lstm_cell_25_biasadd_readvariableop_resource*
_output_shapes	
:è
*
dtype02;
9sequential_12/lstm_25/lstm_cell_25/BiasAdd/ReadVariableOp
*sequential_12/lstm_25/lstm_cell_25/BiasAddBiasAdd*sequential_12/lstm_25/lstm_cell_25/add:z:0Asequential_12/lstm_25/lstm_cell_25/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè
2,
*sequential_12/lstm_25/lstm_cell_25/BiasAddª
2sequential_12/lstm_25/lstm_cell_25/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :24
2sequential_12/lstm_25/lstm_cell_25/split/split_dimÏ
(sequential_12/lstm_25/lstm_cell_25/splitSplit;sequential_12/lstm_25/lstm_cell_25/split/split_dim:output:03sequential_12/lstm_25/lstm_cell_25/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿÚ:ÿÿÿÿÿÿÿÿÿÚ:ÿÿÿÿÿÿÿÿÿÚ:ÿÿÿÿÿÿÿÿÿÚ*
	num_split2*
(sequential_12/lstm_25/lstm_cell_25/splitÉ
*sequential_12/lstm_25/lstm_cell_25/SigmoidSigmoid1sequential_12/lstm_25/lstm_cell_25/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2,
*sequential_12/lstm_25/lstm_cell_25/SigmoidÍ
,sequential_12/lstm_25/lstm_cell_25/Sigmoid_1Sigmoid1sequential_12/lstm_25/lstm_cell_25/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2.
,sequential_12/lstm_25/lstm_cell_25/Sigmoid_1ä
&sequential_12/lstm_25/lstm_cell_25/mulMul0sequential_12/lstm_25/lstm_cell_25/Sigmoid_1:y:0&sequential_12/lstm_25/zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2(
&sequential_12/lstm_25/lstm_cell_25/mulÀ
'sequential_12/lstm_25/lstm_cell_25/ReluRelu1sequential_12/lstm_25/lstm_cell_25/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2)
'sequential_12/lstm_25/lstm_cell_25/Reluõ
(sequential_12/lstm_25/lstm_cell_25/mul_1Mul.sequential_12/lstm_25/lstm_cell_25/Sigmoid:y:05sequential_12/lstm_25/lstm_cell_25/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2*
(sequential_12/lstm_25/lstm_cell_25/mul_1ê
(sequential_12/lstm_25/lstm_cell_25/add_1AddV2*sequential_12/lstm_25/lstm_cell_25/mul:z:0,sequential_12/lstm_25/lstm_cell_25/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2*
(sequential_12/lstm_25/lstm_cell_25/add_1Í
,sequential_12/lstm_25/lstm_cell_25/Sigmoid_2Sigmoid1sequential_12/lstm_25/lstm_cell_25/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2.
,sequential_12/lstm_25/lstm_cell_25/Sigmoid_2¿
)sequential_12/lstm_25/lstm_cell_25/Relu_1Relu,sequential_12/lstm_25/lstm_cell_25/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2+
)sequential_12/lstm_25/lstm_cell_25/Relu_1ù
(sequential_12/lstm_25/lstm_cell_25/mul_2Mul0sequential_12/lstm_25/lstm_cell_25/Sigmoid_2:y:07sequential_12/lstm_25/lstm_cell_25/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2*
(sequential_12/lstm_25/lstm_cell_25/mul_2»
3sequential_12/lstm_25/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿZ  25
3sequential_12/lstm_25/TensorArrayV2_1/element_shape
%sequential_12/lstm_25/TensorArrayV2_1TensorListReserve<sequential_12/lstm_25/TensorArrayV2_1/element_shape:output:0.sequential_12/lstm_25/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02'
%sequential_12/lstm_25/TensorArrayV2_1z
sequential_12/lstm_25/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
sequential_12/lstm_25/time«
.sequential_12/lstm_25/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ20
.sequential_12/lstm_25/while/maximum_iterations
(sequential_12/lstm_25/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2*
(sequential_12/lstm_25/while/loop_counterÝ
sequential_12/lstm_25/whileWhile1sequential_12/lstm_25/while/loop_counter:output:07sequential_12/lstm_25/while/maximum_iterations:output:0#sequential_12/lstm_25/time:output:0.sequential_12/lstm_25/TensorArrayV2_1:handle:0$sequential_12/lstm_25/zeros:output:0&sequential_12/lstm_25/zeros_1:output:0.sequential_12/lstm_25/strided_slice_1:output:0Msequential_12/lstm_25/TensorArrayUnstack/TensorListFromTensor:output_handle:0Asequential_12_lstm_25_lstm_cell_25_matmul_readvariableop_resourceCsequential_12_lstm_25_lstm_cell_25_matmul_1_readvariableop_resourceBsequential_12_lstm_25_lstm_cell_25_biasadd_readvariableop_resource*
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
)sequential_12_lstm_25_while_body_39969515*5
cond-R+
)sequential_12_lstm_25_while_cond_39969514*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿÚ:ÿÿÿÿÿÿÿÿÿÚ: : : : : *
parallel_iterations 2
sequential_12/lstm_25/whileá
Fsequential_12/lstm_25/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿZ  2H
Fsequential_12/lstm_25/TensorArrayV2Stack/TensorListStack/element_shapeÁ
8sequential_12/lstm_25/TensorArrayV2Stack/TensorListStackTensorListStack$sequential_12/lstm_25/while:output:3Osequential_12/lstm_25/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ*
element_dtype02:
8sequential_12/lstm_25/TensorArrayV2Stack/TensorListStack­
+sequential_12/lstm_25/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2-
+sequential_12/lstm_25/strided_slice_3/stack¨
-sequential_12/lstm_25/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2/
-sequential_12/lstm_25/strided_slice_3/stack_1¨
-sequential_12/lstm_25/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_12/lstm_25/strided_slice_3/stack_2
%sequential_12/lstm_25/strided_slice_3StridedSliceAsequential_12/lstm_25/TensorArrayV2Stack/TensorListStack:tensor:04sequential_12/lstm_25/strided_slice_3/stack:output:06sequential_12/lstm_25/strided_slice_3/stack_1:output:06sequential_12/lstm_25/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ*
shrink_axis_mask2'
%sequential_12/lstm_25/strided_slice_3¥
&sequential_12/lstm_25/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2(
&sequential_12/lstm_25/transpose_1/permþ
!sequential_12/lstm_25/transpose_1	TransposeAsequential_12/lstm_25/TensorArrayV2Stack/TensorListStack:tensor:0/sequential_12/lstm_25/transpose_1/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2#
!sequential_12/lstm_25/transpose_1
sequential_12/lstm_25/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
sequential_12/lstm_25/runtime°
!sequential_12/dropout_25/IdentityIdentity%sequential_12/lstm_25/transpose_1:y:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2#
!sequential_12/dropout_25/IdentityÜ
/sequential_12/dense_12/Tensordot/ReadVariableOpReadVariableOp8sequential_12_dense_12_tensordot_readvariableop_resource*
_output_shapes
:	Ú*
dtype021
/sequential_12/dense_12/Tensordot/ReadVariableOp
%sequential_12/dense_12/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2'
%sequential_12/dense_12/Tensordot/axes
%sequential_12/dense_12/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2'
%sequential_12/dense_12/Tensordot/freeª
&sequential_12/dense_12/Tensordot/ShapeShape*sequential_12/dropout_25/Identity:output:0*
T0*
_output_shapes
:2(
&sequential_12/dense_12/Tensordot/Shape¢
.sequential_12/dense_12/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.sequential_12/dense_12/Tensordot/GatherV2/axisÄ
)sequential_12/dense_12/Tensordot/GatherV2GatherV2/sequential_12/dense_12/Tensordot/Shape:output:0.sequential_12/dense_12/Tensordot/free:output:07sequential_12/dense_12/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2+
)sequential_12/dense_12/Tensordot/GatherV2¦
0sequential_12/dense_12/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 22
0sequential_12/dense_12/Tensordot/GatherV2_1/axisÊ
+sequential_12/dense_12/Tensordot/GatherV2_1GatherV2/sequential_12/dense_12/Tensordot/Shape:output:0.sequential_12/dense_12/Tensordot/axes:output:09sequential_12/dense_12/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2-
+sequential_12/dense_12/Tensordot/GatherV2_1
&sequential_12/dense_12/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2(
&sequential_12/dense_12/Tensordot/ConstÜ
%sequential_12/dense_12/Tensordot/ProdProd2sequential_12/dense_12/Tensordot/GatherV2:output:0/sequential_12/dense_12/Tensordot/Const:output:0*
T0*
_output_shapes
: 2'
%sequential_12/dense_12/Tensordot/Prod
(sequential_12/dense_12/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(sequential_12/dense_12/Tensordot/Const_1ä
'sequential_12/dense_12/Tensordot/Prod_1Prod4sequential_12/dense_12/Tensordot/GatherV2_1:output:01sequential_12/dense_12/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2)
'sequential_12/dense_12/Tensordot/Prod_1
,sequential_12/dense_12/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,sequential_12/dense_12/Tensordot/concat/axis£
'sequential_12/dense_12/Tensordot/concatConcatV2.sequential_12/dense_12/Tensordot/free:output:0.sequential_12/dense_12/Tensordot/axes:output:05sequential_12/dense_12/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2)
'sequential_12/dense_12/Tensordot/concatè
&sequential_12/dense_12/Tensordot/stackPack.sequential_12/dense_12/Tensordot/Prod:output:00sequential_12/dense_12/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2(
&sequential_12/dense_12/Tensordot/stackú
*sequential_12/dense_12/Tensordot/transpose	Transpose*sequential_12/dropout_25/Identity:output:00sequential_12/dense_12/Tensordot/concat:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2,
*sequential_12/dense_12/Tensordot/transposeû
(sequential_12/dense_12/Tensordot/ReshapeReshape.sequential_12/dense_12/Tensordot/transpose:y:0/sequential_12/dense_12/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2*
(sequential_12/dense_12/Tensordot/Reshapeú
'sequential_12/dense_12/Tensordot/MatMulMatMul1sequential_12/dense_12/Tensordot/Reshape:output:07sequential_12/dense_12/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'sequential_12/dense_12/Tensordot/MatMul
(sequential_12/dense_12/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(sequential_12/dense_12/Tensordot/Const_2¢
.sequential_12/dense_12/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.sequential_12/dense_12/Tensordot/concat_1/axis°
)sequential_12/dense_12/Tensordot/concat_1ConcatV22sequential_12/dense_12/Tensordot/GatherV2:output:01sequential_12/dense_12/Tensordot/Const_2:output:07sequential_12/dense_12/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2+
)sequential_12/dense_12/Tensordot/concat_1ì
 sequential_12/dense_12/TensordotReshape1sequential_12/dense_12/Tensordot/MatMul:product:02sequential_12/dense_12/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 sequential_12/dense_12/TensordotÑ
-sequential_12/dense_12/BiasAdd/ReadVariableOpReadVariableOp6sequential_12_dense_12_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02/
-sequential_12/dense_12/BiasAdd/ReadVariableOpã
sequential_12/dense_12/BiasAddBiasAdd)sequential_12/dense_12/Tensordot:output:05sequential_12/dense_12/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
sequential_12/dense_12/BiasAddª
sequential_12/dense_12/SoftmaxSoftmax'sequential_12/dense_12/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
sequential_12/dense_12/Softmax
IdentityIdentity(sequential_12/dense_12/Softmax:softmax:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

IdentityÔ
NoOpNoOp.^sequential_12/dense_12/BiasAdd/ReadVariableOp0^sequential_12/dense_12/Tensordot/ReadVariableOp:^sequential_12/lstm_24/lstm_cell_24/BiasAdd/ReadVariableOp9^sequential_12/lstm_24/lstm_cell_24/MatMul/ReadVariableOp;^sequential_12/lstm_24/lstm_cell_24/MatMul_1/ReadVariableOp^sequential_12/lstm_24/while:^sequential_12/lstm_25/lstm_cell_25/BiasAdd/ReadVariableOp9^sequential_12/lstm_25/lstm_cell_25/MatMul/ReadVariableOp;^sequential_12/lstm_25/lstm_cell_25/MatMul_1/ReadVariableOp^sequential_12/lstm_25/while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ]: : : : : : : : 2^
-sequential_12/dense_12/BiasAdd/ReadVariableOp-sequential_12/dense_12/BiasAdd/ReadVariableOp2b
/sequential_12/dense_12/Tensordot/ReadVariableOp/sequential_12/dense_12/Tensordot/ReadVariableOp2v
9sequential_12/lstm_24/lstm_cell_24/BiasAdd/ReadVariableOp9sequential_12/lstm_24/lstm_cell_24/BiasAdd/ReadVariableOp2t
8sequential_12/lstm_24/lstm_cell_24/MatMul/ReadVariableOp8sequential_12/lstm_24/lstm_cell_24/MatMul/ReadVariableOp2x
:sequential_12/lstm_24/lstm_cell_24/MatMul_1/ReadVariableOp:sequential_12/lstm_24/lstm_cell_24/MatMul_1/ReadVariableOp2:
sequential_12/lstm_24/whilesequential_12/lstm_24/while2v
9sequential_12/lstm_25/lstm_cell_25/BiasAdd/ReadVariableOp9sequential_12/lstm_25/lstm_cell_25/BiasAdd/ReadVariableOp2t
8sequential_12/lstm_25/lstm_cell_25/MatMul/ReadVariableOp8sequential_12/lstm_25/lstm_cell_25/MatMul/ReadVariableOp2x
:sequential_12/lstm_25/lstm_cell_25/MatMul_1/ReadVariableOp:sequential_12/lstm_25/lstm_cell_25/MatMul_1/ReadVariableOp2:
sequential_12/lstm_25/whilesequential_12/lstm_25/while:Z V
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
'
_user_specified_namelstm_24_input
¹F

E__inference_lstm_24_layer_call_and_return_conditional_losses_39969785

inputs(
lstm_cell_24_39969703:	](
lstm_cell_24_39969705:	`$
lstm_cell_24_39969707:	
identity¢$lstm_cell_24/StatefulPartitionedCall¢whileD
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
value	B :`2
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
value	B :`2
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
:ÿÿÿÿÿÿÿÿÿ`2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :`2
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
value	B :`2
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
:ÿÿÿÿÿÿÿÿÿ`2	
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
strided_slice_2¥
$lstm_cell_24/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_24_39969703lstm_cell_24_39969705lstm_cell_24_39969707*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ`:ÿÿÿÿÿÿÿÿÿ`:ÿÿÿÿÿÿÿÿÿ`*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_lstm_cell_24_layer_call_and_return_conditional_losses_399697022&
$lstm_cell_24/StatefulPartitionedCall
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ`   2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_24_39969703lstm_cell_24_39969705lstm_cell_24_39969707*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ`:ÿÿÿÿÿÿÿÿÿ`: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_39969716*
condR
while_cond_39969715*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ`:ÿÿÿÿÿÿÿÿÿ`: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ`   22
0TensorArrayV2Stack/TensorListStack/element_shapeñ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`*
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
:ÿÿÿÿÿÿÿÿÿ`*
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
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`2
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
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`2

Identity}
NoOpNoOp%^lstm_cell_24/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ]: : : 2L
$lstm_cell_24/StatefulPartitionedCall$lstm_cell_24/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ]
 
_user_specified_nameinputs
ã
Í
while_cond_39970555
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_39970555___redundant_placeholder06
2while_while_cond_39970555___redundant_placeholder16
2while_while_cond_39970555___redundant_placeholder26
2while_while_cond_39970555___redundant_placeholder3
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
\

E__inference_lstm_25_layer_call_and_return_conditional_losses_39971209

inputs>
+lstm_cell_25_matmul_readvariableop_resource:	`è
A
-lstm_cell_25_matmul_1_readvariableop_resource:
Úè
;
,lstm_cell_25_biasadd_readvariableop_resource:	è

identity¢#lstm_cell_25/BiasAdd/ReadVariableOp¢"lstm_cell_25/MatMul/ReadVariableOp¢$lstm_cell_25/MatMul_1/ReadVariableOp¢whileD
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
:ÿÿÿÿÿÿÿÿÿ`2
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
valueB"ÿÿÿÿ`   27
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
:ÿÿÿÿÿÿÿÿÿ`*
shrink_axis_mask2
strided_slice_2µ
"lstm_cell_25/MatMul/ReadVariableOpReadVariableOp+lstm_cell_25_matmul_readvariableop_resource*
_output_shapes
:	`è
*
dtype02$
"lstm_cell_25/MatMul/ReadVariableOp­
lstm_cell_25/MatMulMatMulstrided_slice_2:output:0*lstm_cell_25/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè
2
lstm_cell_25/MatMul¼
$lstm_cell_25/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_25_matmul_1_readvariableop_resource* 
_output_shapes
:
Úè
*
dtype02&
$lstm_cell_25/MatMul_1/ReadVariableOp©
lstm_cell_25/MatMul_1MatMulzeros:output:0,lstm_cell_25/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè
2
lstm_cell_25/MatMul_1 
lstm_cell_25/addAddV2lstm_cell_25/MatMul:product:0lstm_cell_25/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè
2
lstm_cell_25/add´
#lstm_cell_25/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_25_biasadd_readvariableop_resource*
_output_shapes	
:è
*
dtype02%
#lstm_cell_25/BiasAdd/ReadVariableOp­
lstm_cell_25/BiasAddBiasAddlstm_cell_25/add:z:0+lstm_cell_25/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè
2
lstm_cell_25/BiasAdd~
lstm_cell_25/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_25/split/split_dim÷
lstm_cell_25/splitSplit%lstm_cell_25/split/split_dim:output:0lstm_cell_25/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿÚ:ÿÿÿÿÿÿÿÿÿÚ:ÿÿÿÿÿÿÿÿÿÚ:ÿÿÿÿÿÿÿÿÿÚ*
	num_split2
lstm_cell_25/split
lstm_cell_25/SigmoidSigmoidlstm_cell_25/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
lstm_cell_25/Sigmoid
lstm_cell_25/Sigmoid_1Sigmoidlstm_cell_25/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
lstm_cell_25/Sigmoid_1
lstm_cell_25/mulMullstm_cell_25/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
lstm_cell_25/mul~
lstm_cell_25/ReluRelulstm_cell_25/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
lstm_cell_25/Relu
lstm_cell_25/mul_1Mullstm_cell_25/Sigmoid:y:0lstm_cell_25/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
lstm_cell_25/mul_1
lstm_cell_25/add_1AddV2lstm_cell_25/mul:z:0lstm_cell_25/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
lstm_cell_25/add_1
lstm_cell_25/Sigmoid_2Sigmoidlstm_cell_25/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
lstm_cell_25/Sigmoid_2}
lstm_cell_25/Relu_1Relulstm_cell_25/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
lstm_cell_25/Relu_1¡
lstm_cell_25/mul_2Mullstm_cell_25/Sigmoid_2:y:0!lstm_cell_25/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
lstm_cell_25/mul_2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_25_matmul_readvariableop_resource-lstm_cell_25_matmul_1_readvariableop_resource,lstm_cell_25_biasadd_readvariableop_resource*
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
while_body_39971125*
condR
while_cond_39971124*M
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
NoOpNoOp$^lstm_cell_25/BiasAdd/ReadVariableOp#^lstm_cell_25/MatMul/ReadVariableOp%^lstm_cell_25/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ`: : : 2J
#lstm_cell_25/BiasAdd/ReadVariableOp#lstm_cell_25/BiasAdd/ReadVariableOp2H
"lstm_cell_25/MatMul/ReadVariableOp"lstm_cell_25/MatMul/ReadVariableOp2L
$lstm_cell_25/MatMul_1/ReadVariableOp$lstm_cell_25/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
 
_user_specified_nameinputs
°?
Ô
while_body_39973604
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_25_matmul_readvariableop_resource_0:	`è
I
5while_lstm_cell_25_matmul_1_readvariableop_resource_0:
Úè
C
4while_lstm_cell_25_biasadd_readvariableop_resource_0:	è

while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_25_matmul_readvariableop_resource:	`è
G
3while_lstm_cell_25_matmul_1_readvariableop_resource:
Úè
A
2while_lstm_cell_25_biasadd_readvariableop_resource:	è
¢)while/lstm_cell_25/BiasAdd/ReadVariableOp¢(while/lstm_cell_25/MatMul/ReadVariableOp¢*while/lstm_cell_25/MatMul_1/ReadVariableOpÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ`   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÓ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemÉ
(while/lstm_cell_25/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_25_matmul_readvariableop_resource_0*
_output_shapes
:	`è
*
dtype02*
(while/lstm_cell_25/MatMul/ReadVariableOp×
while/lstm_cell_25/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_25/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè
2
while/lstm_cell_25/MatMulÐ
*while/lstm_cell_25/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_25_matmul_1_readvariableop_resource_0* 
_output_shapes
:
Úè
*
dtype02,
*while/lstm_cell_25/MatMul_1/ReadVariableOpÀ
while/lstm_cell_25/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_25/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè
2
while/lstm_cell_25/MatMul_1¸
while/lstm_cell_25/addAddV2#while/lstm_cell_25/MatMul:product:0%while/lstm_cell_25/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè
2
while/lstm_cell_25/addÈ
)while/lstm_cell_25/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_25_biasadd_readvariableop_resource_0*
_output_shapes	
:è
*
dtype02+
)while/lstm_cell_25/BiasAdd/ReadVariableOpÅ
while/lstm_cell_25/BiasAddBiasAddwhile/lstm_cell_25/add:z:01while/lstm_cell_25/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè
2
while/lstm_cell_25/BiasAdd
"while/lstm_cell_25/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_25/split/split_dim
while/lstm_cell_25/splitSplit+while/lstm_cell_25/split/split_dim:output:0#while/lstm_cell_25/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿÚ:ÿÿÿÿÿÿÿÿÿÚ:ÿÿÿÿÿÿÿÿÿÚ:ÿÿÿÿÿÿÿÿÿÚ*
	num_split2
while/lstm_cell_25/split
while/lstm_cell_25/SigmoidSigmoid!while/lstm_cell_25/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
while/lstm_cell_25/Sigmoid
while/lstm_cell_25/Sigmoid_1Sigmoid!while/lstm_cell_25/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
while/lstm_cell_25/Sigmoid_1¡
while/lstm_cell_25/mulMul while/lstm_cell_25/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
while/lstm_cell_25/mul
while/lstm_cell_25/ReluRelu!while/lstm_cell_25/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
while/lstm_cell_25/Reluµ
while/lstm_cell_25/mul_1Mulwhile/lstm_cell_25/Sigmoid:y:0%while/lstm_cell_25/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
while/lstm_cell_25/mul_1ª
while/lstm_cell_25/add_1AddV2while/lstm_cell_25/mul:z:0while/lstm_cell_25/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
while/lstm_cell_25/add_1
while/lstm_cell_25/Sigmoid_2Sigmoid!while/lstm_cell_25/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
while/lstm_cell_25/Sigmoid_2
while/lstm_cell_25/Relu_1Reluwhile/lstm_cell_25/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
while/lstm_cell_25/Relu_1¹
while/lstm_cell_25/mul_2Mul while/lstm_cell_25/Sigmoid_2:y:0'while/lstm_cell_25/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
while/lstm_cell_25/mul_2à
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_25/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_25/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_25/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
while/Identity_5Þ

while/NoOpNoOp*^while/lstm_cell_25/BiasAdd/ReadVariableOp)^while/lstm_cell_25/MatMul/ReadVariableOp+^while/lstm_cell_25/MatMul_1/ReadVariableOp*"
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
2while_lstm_cell_25_biasadd_readvariableop_resource4while_lstm_cell_25_biasadd_readvariableop_resource_0"l
3while_lstm_cell_25_matmul_1_readvariableop_resource5while_lstm_cell_25_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_25_matmul_readvariableop_resource3while_lstm_cell_25_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿÚ:ÿÿÿÿÿÿÿÿÿÚ: : : : : 2V
)while/lstm_cell_25/BiasAdd/ReadVariableOp)while/lstm_cell_25/BiasAdd/ReadVariableOp2T
(while/lstm_cell_25/MatMul/ReadVariableOp(while/lstm_cell_25/MatMul/ReadVariableOp2X
*while/lstm_cell_25/MatMul_1/ReadVariableOp*while/lstm_cell_25/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
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
±
¹
K__inference_sequential_12_layer_call_and_return_conditional_losses_39971731

inputs#
lstm_24_39971709:	]#
lstm_24_39971711:	`
lstm_24_39971713:	#
lstm_25_39971717:	`è
$
lstm_25_39971719:
Úè

lstm_25_39971721:	è
$
dense_12_39971725:	Ú
dense_12_39971727:
identity¢ dense_12/StatefulPartitionedCall¢"dropout_24/StatefulPartitionedCall¢"dropout_25/StatefulPartitionedCall¢lstm_24/StatefulPartitionedCall¢lstm_25/StatefulPartitionedCall­
lstm_24/StatefulPartitionedCallStatefulPartitionedCallinputslstm_24_39971709lstm_24_39971711lstm_24_39971713*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_lstm_24_layer_call_and_return_conditional_losses_399716742!
lstm_24/StatefulPartitionedCall
"dropout_24/StatefulPartitionedCallStatefulPartitionedCall(lstm_24/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_dropout_24_layer_call_and_return_conditional_losses_399715072$
"dropout_24/StatefulPartitionedCallÓ
lstm_25/StatefulPartitionedCallStatefulPartitionedCall+dropout_24/StatefulPartitionedCall:output:0lstm_25_39971717lstm_25_39971719lstm_25_39971721*
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
E__inference_lstm_25_layer_call_and_return_conditional_losses_399714782!
lstm_25/StatefulPartitionedCallÀ
"dropout_25/StatefulPartitionedCallStatefulPartitionedCall(lstm_25/StatefulPartitionedCall:output:0#^dropout_24/StatefulPartitionedCall*
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
H__inference_dropout_25_layer_call_and_return_conditional_losses_399713112$
"dropout_25/StatefulPartitionedCallÃ
 dense_12/StatefulPartitionedCallStatefulPartitionedCall+dropout_25/StatefulPartitionedCall:output:0dense_12_39971725dense_12_39971727*
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
F__inference_dense_12_layer_call_and_return_conditional_losses_399712552"
 dense_12/StatefulPartitionedCall
IdentityIdentity)dense_12/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityÿ
NoOpNoOp!^dense_12/StatefulPartitionedCall#^dropout_24/StatefulPartitionedCall#^dropout_25/StatefulPartitionedCall ^lstm_24/StatefulPartitionedCall ^lstm_25/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ]: : : : : : : : 2D
 dense_12/StatefulPartitionedCall dense_12/StatefulPartitionedCall2H
"dropout_24/StatefulPartitionedCall"dropout_24/StatefulPartitionedCall2H
"dropout_25/StatefulPartitionedCall"dropout_25/StatefulPartitionedCall2B
lstm_24/StatefulPartitionedCalllstm_24/StatefulPartitionedCall2B
lstm_25/StatefulPartitionedCalllstm_25/StatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
 
_user_specified_nameinputs
ý

J__inference_lstm_cell_24_layer_call_and_return_conditional_losses_39974014

inputs
states_0
states_11
matmul_readvariableop_resource:	]3
 matmul_1_readvariableop_resource:	`.
biasadd_readvariableop_resource:	
identity

identity_1

identity_2¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	]*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	`*
dtype02
MatMul_1/ReadVariableOp|
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
add
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
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
L:ÿÿÿÿÿÿÿÿÿ`:ÿÿÿÿÿÿÿÿÿ`:ÿÿÿÿÿÿÿÿÿ`:ÿÿÿÿÿÿÿÿÿ`*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
	Sigmoid_1\
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
mulV
ReluRelusplit:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
	Sigmoid_2U
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
mul_2d
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2

Identityh

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2

Identity_1h

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2

Identity_2
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
?:ÿÿÿÿÿÿÿÿÿ]:ÿÿÿÿÿÿÿÿÿ`:ÿÿÿÿÿÿÿÿÿ`: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
 
_user_specified_nameinputs:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
"
_user_specified_name
states/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
"
_user_specified_name
states/1
¤
ï
K__inference_sequential_12_layer_call_and_return_conditional_losses_39971262

inputs#
lstm_24_39971045:	]#
lstm_24_39971047:	`
lstm_24_39971049:	#
lstm_25_39971210:	`è
$
lstm_25_39971212:
Úè

lstm_25_39971214:	è
$
dense_12_39971256:	Ú
dense_12_39971258:
identity¢ dense_12/StatefulPartitionedCall¢lstm_24/StatefulPartitionedCall¢lstm_25/StatefulPartitionedCall­
lstm_24/StatefulPartitionedCallStatefulPartitionedCallinputslstm_24_39971045lstm_24_39971047lstm_24_39971049*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_lstm_24_layer_call_and_return_conditional_losses_399710442!
lstm_24/StatefulPartitionedCall
dropout_24/PartitionedCallPartitionedCall(lstm_24/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_dropout_24_layer_call_and_return_conditional_losses_399710572
dropout_24/PartitionedCallË
lstm_25/StatefulPartitionedCallStatefulPartitionedCall#dropout_24/PartitionedCall:output:0lstm_25_39971210lstm_25_39971212lstm_25_39971214*
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
E__inference_lstm_25_layer_call_and_return_conditional_losses_399712092!
lstm_25/StatefulPartitionedCall
dropout_25/PartitionedCallPartitionedCall(lstm_25/StatefulPartitionedCall:output:0*
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
H__inference_dropout_25_layer_call_and_return_conditional_losses_399712222
dropout_25/PartitionedCall»
 dense_12/StatefulPartitionedCallStatefulPartitionedCall#dropout_25/PartitionedCall:output:0dense_12_39971256dense_12_39971258*
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
F__inference_dense_12_layer_call_and_return_conditional_losses_399712552"
 dense_12/StatefulPartitionedCall
IdentityIdentity)dense_12/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityµ
NoOpNoOp!^dense_12/StatefulPartitionedCall ^lstm_24/StatefulPartitionedCall ^lstm_25/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ]: : : : : : : : 2D
 dense_12/StatefulPartitionedCall dense_12/StatefulPartitionedCall2B
lstm_24/StatefulPartitionedCalllstm_24/StatefulPartitionedCall2B
lstm_25/StatefulPartitionedCalllstm_25/StatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
 
_user_specified_nameinputs
ß
Í
while_cond_39972626
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_39972626___redundant_placeholder06
2while_while_cond_39972626___redundant_placeholder16
2while_while_cond_39972626___redundant_placeholder26
2while_while_cond_39972626___redundant_placeholder3
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
@: : : : :ÿÿÿÿÿÿÿÿÿ`:ÿÿÿÿÿÿÿÿÿ`: ::::: 

_output_shapes
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
:ÿÿÿÿÿÿÿÿÿ`:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`:

_output_shapes
: :

_output_shapes
:
?
Ò
while_body_39970960
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_24_matmul_readvariableop_resource_0:	]H
5while_lstm_cell_24_matmul_1_readvariableop_resource_0:	`C
4while_lstm_cell_24_biasadd_readvariableop_resource_0:	
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_24_matmul_readvariableop_resource:	]F
3while_lstm_cell_24_matmul_1_readvariableop_resource:	`A
2while_lstm_cell_24_biasadd_readvariableop_resource:	¢)while/lstm_cell_24/BiasAdd/ReadVariableOp¢(while/lstm_cell_24/MatMul/ReadVariableOp¢*while/lstm_cell_24/MatMul_1/ReadVariableOpÃ
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
(while/lstm_cell_24/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_24_matmul_readvariableop_resource_0*
_output_shapes
:	]*
dtype02*
(while/lstm_cell_24/MatMul/ReadVariableOp×
while/lstm_cell_24/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_24/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_24/MatMulÏ
*while/lstm_cell_24/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_24_matmul_1_readvariableop_resource_0*
_output_shapes
:	`*
dtype02,
*while/lstm_cell_24/MatMul_1/ReadVariableOpÀ
while/lstm_cell_24/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_24/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_24/MatMul_1¸
while/lstm_cell_24/addAddV2#while/lstm_cell_24/MatMul:product:0%while/lstm_cell_24/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_24/addÈ
)while/lstm_cell_24/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_24_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02+
)while/lstm_cell_24/BiasAdd/ReadVariableOpÅ
while/lstm_cell_24/BiasAddBiasAddwhile/lstm_cell_24/add:z:01while/lstm_cell_24/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_24/BiasAdd
"while/lstm_cell_24/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_24/split/split_dim
while/lstm_cell_24/splitSplit+while/lstm_cell_24/split/split_dim:output:0#while/lstm_cell_24/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ`:ÿÿÿÿÿÿÿÿÿ`:ÿÿÿÿÿÿÿÿÿ`:ÿÿÿÿÿÿÿÿÿ`*
	num_split2
while/lstm_cell_24/split
while/lstm_cell_24/SigmoidSigmoid!while/lstm_cell_24/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
while/lstm_cell_24/Sigmoid
while/lstm_cell_24/Sigmoid_1Sigmoid!while/lstm_cell_24/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
while/lstm_cell_24/Sigmoid_1 
while/lstm_cell_24/mulMul while/lstm_cell_24/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
while/lstm_cell_24/mul
while/lstm_cell_24/ReluRelu!while/lstm_cell_24/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
while/lstm_cell_24/Relu´
while/lstm_cell_24/mul_1Mulwhile/lstm_cell_24/Sigmoid:y:0%while/lstm_cell_24/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
while/lstm_cell_24/mul_1©
while/lstm_cell_24/add_1AddV2while/lstm_cell_24/mul:z:0while/lstm_cell_24/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
while/lstm_cell_24/add_1
while/lstm_cell_24/Sigmoid_2Sigmoid!while/lstm_cell_24/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
while/lstm_cell_24/Sigmoid_2
while/lstm_cell_24/Relu_1Reluwhile/lstm_cell_24/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
while/lstm_cell_24/Relu_1¸
while/lstm_cell_24/mul_2Mul while/lstm_cell_24/Sigmoid_2:y:0'while/lstm_cell_24/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
while/lstm_cell_24/mul_2à
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_24/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_24/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_24/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
while/Identity_5Þ

while/NoOpNoOp*^while/lstm_cell_24/BiasAdd/ReadVariableOp)^while/lstm_cell_24/MatMul/ReadVariableOp+^while/lstm_cell_24/MatMul_1/ReadVariableOp*"
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
2while_lstm_cell_24_biasadd_readvariableop_resource4while_lstm_cell_24_biasadd_readvariableop_resource_0"l
3while_lstm_cell_24_matmul_1_readvariableop_resource5while_lstm_cell_24_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_24_matmul_readvariableop_resource3while_lstm_cell_24_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ`:ÿÿÿÿÿÿÿÿÿ`: : : : : 2V
)while/lstm_cell_24/BiasAdd/ReadVariableOp)while/lstm_cell_24/BiasAdd/ReadVariableOp2T
(while/lstm_cell_24/MatMul/ReadVariableOp(while/lstm_cell_24/MatMul/ReadVariableOp2X
*while/lstm_cell_24/MatMul_1/ReadVariableOp*while/lstm_cell_24/MatMul_1/ReadVariableOp: 

_output_shapes
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
:ÿÿÿÿÿÿÿÿÿ`:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`:

_output_shapes
: :

_output_shapes
: 


J__inference_lstm_cell_25_layer_call_and_return_conditional_losses_39974112

inputs
states_0
states_11
matmul_readvariableop_resource:	`è
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
:	`è
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
A:ÿÿÿÿÿÿÿÿÿ`:ÿÿÿÿÿÿÿÿÿÚ:ÿÿÿÿÿÿÿÿÿÚ: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
 
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
ý

J__inference_lstm_cell_24_layer_call_and_return_conditional_losses_39973982

inputs
states_0
states_11
matmul_readvariableop_resource:	]3
 matmul_1_readvariableop_resource:	`.
biasadd_readvariableop_resource:	
identity

identity_1

identity_2¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	]*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	`*
dtype02
MatMul_1/ReadVariableOp|
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
add
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
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
L:ÿÿÿÿÿÿÿÿÿ`:ÿÿÿÿÿÿÿÿÿ`:ÿÿÿÿÿÿÿÿÿ`:ÿÿÿÿÿÿÿÿÿ`*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
	Sigmoid_1\
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
mulV
ReluRelusplit:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
	Sigmoid_2U
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
mul_2d
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2

Identityh

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2

Identity_1h

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2

Identity_2
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
?:ÿÿÿÿÿÿÿÿÿ]:ÿÿÿÿÿÿÿÿÿ`:ÿÿÿÿÿÿÿÿÿ`: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
 
_user_specified_nameinputs:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
"
_user_specified_name
states/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
"
_user_specified_name
states/1
\

E__inference_lstm_25_layer_call_and_return_conditional_losses_39971478

inputs>
+lstm_cell_25_matmul_readvariableop_resource:	`è
A
-lstm_cell_25_matmul_1_readvariableop_resource:
Úè
;
,lstm_cell_25_biasadd_readvariableop_resource:	è

identity¢#lstm_cell_25/BiasAdd/ReadVariableOp¢"lstm_cell_25/MatMul/ReadVariableOp¢$lstm_cell_25/MatMul_1/ReadVariableOp¢whileD
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
:ÿÿÿÿÿÿÿÿÿ`2
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
valueB"ÿÿÿÿ`   27
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
:ÿÿÿÿÿÿÿÿÿ`*
shrink_axis_mask2
strided_slice_2µ
"lstm_cell_25/MatMul/ReadVariableOpReadVariableOp+lstm_cell_25_matmul_readvariableop_resource*
_output_shapes
:	`è
*
dtype02$
"lstm_cell_25/MatMul/ReadVariableOp­
lstm_cell_25/MatMulMatMulstrided_slice_2:output:0*lstm_cell_25/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè
2
lstm_cell_25/MatMul¼
$lstm_cell_25/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_25_matmul_1_readvariableop_resource* 
_output_shapes
:
Úè
*
dtype02&
$lstm_cell_25/MatMul_1/ReadVariableOp©
lstm_cell_25/MatMul_1MatMulzeros:output:0,lstm_cell_25/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè
2
lstm_cell_25/MatMul_1 
lstm_cell_25/addAddV2lstm_cell_25/MatMul:product:0lstm_cell_25/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè
2
lstm_cell_25/add´
#lstm_cell_25/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_25_biasadd_readvariableop_resource*
_output_shapes	
:è
*
dtype02%
#lstm_cell_25/BiasAdd/ReadVariableOp­
lstm_cell_25/BiasAddBiasAddlstm_cell_25/add:z:0+lstm_cell_25/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè
2
lstm_cell_25/BiasAdd~
lstm_cell_25/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_25/split/split_dim÷
lstm_cell_25/splitSplit%lstm_cell_25/split/split_dim:output:0lstm_cell_25/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿÚ:ÿÿÿÿÿÿÿÿÿÚ:ÿÿÿÿÿÿÿÿÿÚ:ÿÿÿÿÿÿÿÿÿÚ*
	num_split2
lstm_cell_25/split
lstm_cell_25/SigmoidSigmoidlstm_cell_25/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
lstm_cell_25/Sigmoid
lstm_cell_25/Sigmoid_1Sigmoidlstm_cell_25/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
lstm_cell_25/Sigmoid_1
lstm_cell_25/mulMullstm_cell_25/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
lstm_cell_25/mul~
lstm_cell_25/ReluRelulstm_cell_25/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
lstm_cell_25/Relu
lstm_cell_25/mul_1Mullstm_cell_25/Sigmoid:y:0lstm_cell_25/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
lstm_cell_25/mul_1
lstm_cell_25/add_1AddV2lstm_cell_25/mul:z:0lstm_cell_25/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
lstm_cell_25/add_1
lstm_cell_25/Sigmoid_2Sigmoidlstm_cell_25/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
lstm_cell_25/Sigmoid_2}
lstm_cell_25/Relu_1Relulstm_cell_25/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
lstm_cell_25/Relu_1¡
lstm_cell_25/mul_2Mullstm_cell_25/Sigmoid_2:y:0!lstm_cell_25/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
lstm_cell_25/mul_2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_25_matmul_readvariableop_resource-lstm_cell_25_matmul_1_readvariableop_resource,lstm_cell_25_biasadd_readvariableop_resource*
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
while_body_39971394*
condR
while_cond_39971393*M
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
NoOpNoOp$^lstm_cell_25/BiasAdd/ReadVariableOp#^lstm_cell_25/MatMul/ReadVariableOp%^lstm_cell_25/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ`: : : 2J
#lstm_cell_25/BiasAdd/ReadVariableOp#lstm_cell_25/BiasAdd/ReadVariableOp2H
"lstm_cell_25/MatMul/ReadVariableOp"lstm_cell_25/MatMul/ReadVariableOp2L
$lstm_cell_25/MatMul_1/ReadVariableOp$lstm_cell_25/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
 
_user_specified_nameinputs
&
ó
while_body_39970346
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_00
while_lstm_cell_25_39970370_0:	`è
1
while_lstm_cell_25_39970372_0:
Úè
,
while_lstm_cell_25_39970374_0:	è

while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor.
while_lstm_cell_25_39970370:	`è
/
while_lstm_cell_25_39970372:
Úè
*
while_lstm_cell_25_39970374:	è
¢*while/lstm_cell_25/StatefulPartitionedCallÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ`   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÓ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemì
*while/lstm_cell_25/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_25_39970370_0while_lstm_cell_25_39970372_0while_lstm_cell_25_39970374_0*
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
J__inference_lstm_cell_25_layer_call_and_return_conditional_losses_399703322,
*while/lstm_cell_25/StatefulPartitionedCall÷
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder3while/lstm_cell_25/StatefulPartitionedCall:output:0*
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
while/Identity_4Identity3while/lstm_cell_25/StatefulPartitionedCall:output:1^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
while/Identity_4¥
while/Identity_5Identity3while/lstm_cell_25/StatefulPartitionedCall:output:2^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
while/Identity_5

while/NoOpNoOp+^while/lstm_cell_25/StatefulPartitionedCall*"
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
while_lstm_cell_25_39970370while_lstm_cell_25_39970370_0"<
while_lstm_cell_25_39970372while_lstm_cell_25_39970372_0"<
while_lstm_cell_25_39970374while_lstm_cell_25_39970374_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿÚ:ÿÿÿÿÿÿÿÿÿÚ: : : : : 2X
*while/lstm_cell_25/StatefulPartitionedCall*while/lstm_cell_25/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
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
ä^

)sequential_12_lstm_25_while_body_39969515H
Dsequential_12_lstm_25_while_sequential_12_lstm_25_while_loop_counterN
Jsequential_12_lstm_25_while_sequential_12_lstm_25_while_maximum_iterations+
'sequential_12_lstm_25_while_placeholder-
)sequential_12_lstm_25_while_placeholder_1-
)sequential_12_lstm_25_while_placeholder_2-
)sequential_12_lstm_25_while_placeholder_3G
Csequential_12_lstm_25_while_sequential_12_lstm_25_strided_slice_1_0
sequential_12_lstm_25_while_tensorarrayv2read_tensorlistgetitem_sequential_12_lstm_25_tensorarrayunstack_tensorlistfromtensor_0\
Isequential_12_lstm_25_while_lstm_cell_25_matmul_readvariableop_resource_0:	`è
_
Ksequential_12_lstm_25_while_lstm_cell_25_matmul_1_readvariableop_resource_0:
Úè
Y
Jsequential_12_lstm_25_while_lstm_cell_25_biasadd_readvariableop_resource_0:	è
(
$sequential_12_lstm_25_while_identity*
&sequential_12_lstm_25_while_identity_1*
&sequential_12_lstm_25_while_identity_2*
&sequential_12_lstm_25_while_identity_3*
&sequential_12_lstm_25_while_identity_4*
&sequential_12_lstm_25_while_identity_5E
Asequential_12_lstm_25_while_sequential_12_lstm_25_strided_slice_1
}sequential_12_lstm_25_while_tensorarrayv2read_tensorlistgetitem_sequential_12_lstm_25_tensorarrayunstack_tensorlistfromtensorZ
Gsequential_12_lstm_25_while_lstm_cell_25_matmul_readvariableop_resource:	`è
]
Isequential_12_lstm_25_while_lstm_cell_25_matmul_1_readvariableop_resource:
Úè
W
Hsequential_12_lstm_25_while_lstm_cell_25_biasadd_readvariableop_resource:	è
¢?sequential_12/lstm_25/while/lstm_cell_25/BiasAdd/ReadVariableOp¢>sequential_12/lstm_25/while/lstm_cell_25/MatMul/ReadVariableOp¢@sequential_12/lstm_25/while/lstm_cell_25/MatMul_1/ReadVariableOpï
Msequential_12/lstm_25/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ`   2O
Msequential_12/lstm_25/while/TensorArrayV2Read/TensorListGetItem/element_shape×
?sequential_12/lstm_25/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemsequential_12_lstm_25_while_tensorarrayv2read_tensorlistgetitem_sequential_12_lstm_25_tensorarrayunstack_tensorlistfromtensor_0'sequential_12_lstm_25_while_placeholderVsequential_12/lstm_25/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`*
element_dtype02A
?sequential_12/lstm_25/while/TensorArrayV2Read/TensorListGetItem
>sequential_12/lstm_25/while/lstm_cell_25/MatMul/ReadVariableOpReadVariableOpIsequential_12_lstm_25_while_lstm_cell_25_matmul_readvariableop_resource_0*
_output_shapes
:	`è
*
dtype02@
>sequential_12/lstm_25/while/lstm_cell_25/MatMul/ReadVariableOp¯
/sequential_12/lstm_25/while/lstm_cell_25/MatMulMatMulFsequential_12/lstm_25/while/TensorArrayV2Read/TensorListGetItem:item:0Fsequential_12/lstm_25/while/lstm_cell_25/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè
21
/sequential_12/lstm_25/while/lstm_cell_25/MatMul
@sequential_12/lstm_25/while/lstm_cell_25/MatMul_1/ReadVariableOpReadVariableOpKsequential_12_lstm_25_while_lstm_cell_25_matmul_1_readvariableop_resource_0* 
_output_shapes
:
Úè
*
dtype02B
@sequential_12/lstm_25/while/lstm_cell_25/MatMul_1/ReadVariableOp
1sequential_12/lstm_25/while/lstm_cell_25/MatMul_1MatMul)sequential_12_lstm_25_while_placeholder_2Hsequential_12/lstm_25/while/lstm_cell_25/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè
23
1sequential_12/lstm_25/while/lstm_cell_25/MatMul_1
,sequential_12/lstm_25/while/lstm_cell_25/addAddV29sequential_12/lstm_25/while/lstm_cell_25/MatMul:product:0;sequential_12/lstm_25/while/lstm_cell_25/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè
2.
,sequential_12/lstm_25/while/lstm_cell_25/add
?sequential_12/lstm_25/while/lstm_cell_25/BiasAdd/ReadVariableOpReadVariableOpJsequential_12_lstm_25_while_lstm_cell_25_biasadd_readvariableop_resource_0*
_output_shapes	
:è
*
dtype02A
?sequential_12/lstm_25/while/lstm_cell_25/BiasAdd/ReadVariableOp
0sequential_12/lstm_25/while/lstm_cell_25/BiasAddBiasAdd0sequential_12/lstm_25/while/lstm_cell_25/add:z:0Gsequential_12/lstm_25/while/lstm_cell_25/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè
22
0sequential_12/lstm_25/while/lstm_cell_25/BiasAdd¶
8sequential_12/lstm_25/while/lstm_cell_25/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2:
8sequential_12/lstm_25/while/lstm_cell_25/split/split_dimç
.sequential_12/lstm_25/while/lstm_cell_25/splitSplitAsequential_12/lstm_25/while/lstm_cell_25/split/split_dim:output:09sequential_12/lstm_25/while/lstm_cell_25/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿÚ:ÿÿÿÿÿÿÿÿÿÚ:ÿÿÿÿÿÿÿÿÿÚ:ÿÿÿÿÿÿÿÿÿÚ*
	num_split20
.sequential_12/lstm_25/while/lstm_cell_25/splitÛ
0sequential_12/lstm_25/while/lstm_cell_25/SigmoidSigmoid7sequential_12/lstm_25/while/lstm_cell_25/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ22
0sequential_12/lstm_25/while/lstm_cell_25/Sigmoidß
2sequential_12/lstm_25/while/lstm_cell_25/Sigmoid_1Sigmoid7sequential_12/lstm_25/while/lstm_cell_25/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ24
2sequential_12/lstm_25/while/lstm_cell_25/Sigmoid_1ù
,sequential_12/lstm_25/while/lstm_cell_25/mulMul6sequential_12/lstm_25/while/lstm_cell_25/Sigmoid_1:y:0)sequential_12_lstm_25_while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2.
,sequential_12/lstm_25/while/lstm_cell_25/mulÒ
-sequential_12/lstm_25/while/lstm_cell_25/ReluRelu7sequential_12/lstm_25/while/lstm_cell_25/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2/
-sequential_12/lstm_25/while/lstm_cell_25/Relu
.sequential_12/lstm_25/while/lstm_cell_25/mul_1Mul4sequential_12/lstm_25/while/lstm_cell_25/Sigmoid:y:0;sequential_12/lstm_25/while/lstm_cell_25/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ20
.sequential_12/lstm_25/while/lstm_cell_25/mul_1
.sequential_12/lstm_25/while/lstm_cell_25/add_1AddV20sequential_12/lstm_25/while/lstm_cell_25/mul:z:02sequential_12/lstm_25/while/lstm_cell_25/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ20
.sequential_12/lstm_25/while/lstm_cell_25/add_1ß
2sequential_12/lstm_25/while/lstm_cell_25/Sigmoid_2Sigmoid7sequential_12/lstm_25/while/lstm_cell_25/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ24
2sequential_12/lstm_25/while/lstm_cell_25/Sigmoid_2Ñ
/sequential_12/lstm_25/while/lstm_cell_25/Relu_1Relu2sequential_12/lstm_25/while/lstm_cell_25/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ21
/sequential_12/lstm_25/while/lstm_cell_25/Relu_1
.sequential_12/lstm_25/while/lstm_cell_25/mul_2Mul6sequential_12/lstm_25/while/lstm_cell_25/Sigmoid_2:y:0=sequential_12/lstm_25/while/lstm_cell_25/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ20
.sequential_12/lstm_25/while/lstm_cell_25/mul_2Î
@sequential_12/lstm_25/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem)sequential_12_lstm_25_while_placeholder_1'sequential_12_lstm_25_while_placeholder2sequential_12/lstm_25/while/lstm_cell_25/mul_2:z:0*
_output_shapes
: *
element_dtype02B
@sequential_12/lstm_25/while/TensorArrayV2Write/TensorListSetItem
!sequential_12/lstm_25/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2#
!sequential_12/lstm_25/while/add/yÁ
sequential_12/lstm_25/while/addAddV2'sequential_12_lstm_25_while_placeholder*sequential_12/lstm_25/while/add/y:output:0*
T0*
_output_shapes
: 2!
sequential_12/lstm_25/while/add
#sequential_12/lstm_25/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2%
#sequential_12/lstm_25/while/add_1/yä
!sequential_12/lstm_25/while/add_1AddV2Dsequential_12_lstm_25_while_sequential_12_lstm_25_while_loop_counter,sequential_12/lstm_25/while/add_1/y:output:0*
T0*
_output_shapes
: 2#
!sequential_12/lstm_25/while/add_1Ã
$sequential_12/lstm_25/while/IdentityIdentity%sequential_12/lstm_25/while/add_1:z:0!^sequential_12/lstm_25/while/NoOp*
T0*
_output_shapes
: 2&
$sequential_12/lstm_25/while/Identityì
&sequential_12/lstm_25/while/Identity_1IdentityJsequential_12_lstm_25_while_sequential_12_lstm_25_while_maximum_iterations!^sequential_12/lstm_25/while/NoOp*
T0*
_output_shapes
: 2(
&sequential_12/lstm_25/while/Identity_1Å
&sequential_12/lstm_25/while/Identity_2Identity#sequential_12/lstm_25/while/add:z:0!^sequential_12/lstm_25/while/NoOp*
T0*
_output_shapes
: 2(
&sequential_12/lstm_25/while/Identity_2ò
&sequential_12/lstm_25/while/Identity_3IdentityPsequential_12/lstm_25/while/TensorArrayV2Write/TensorListSetItem:output_handle:0!^sequential_12/lstm_25/while/NoOp*
T0*
_output_shapes
: 2(
&sequential_12/lstm_25/while/Identity_3æ
&sequential_12/lstm_25/while/Identity_4Identity2sequential_12/lstm_25/while/lstm_cell_25/mul_2:z:0!^sequential_12/lstm_25/while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2(
&sequential_12/lstm_25/while/Identity_4æ
&sequential_12/lstm_25/while/Identity_5Identity2sequential_12/lstm_25/while/lstm_cell_25/add_1:z:0!^sequential_12/lstm_25/while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2(
&sequential_12/lstm_25/while/Identity_5Ì
 sequential_12/lstm_25/while/NoOpNoOp@^sequential_12/lstm_25/while/lstm_cell_25/BiasAdd/ReadVariableOp?^sequential_12/lstm_25/while/lstm_cell_25/MatMul/ReadVariableOpA^sequential_12/lstm_25/while/lstm_cell_25/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2"
 sequential_12/lstm_25/while/NoOp"U
$sequential_12_lstm_25_while_identity-sequential_12/lstm_25/while/Identity:output:0"Y
&sequential_12_lstm_25_while_identity_1/sequential_12/lstm_25/while/Identity_1:output:0"Y
&sequential_12_lstm_25_while_identity_2/sequential_12/lstm_25/while/Identity_2:output:0"Y
&sequential_12_lstm_25_while_identity_3/sequential_12/lstm_25/while/Identity_3:output:0"Y
&sequential_12_lstm_25_while_identity_4/sequential_12/lstm_25/while/Identity_4:output:0"Y
&sequential_12_lstm_25_while_identity_5/sequential_12/lstm_25/while/Identity_5:output:0"
Hsequential_12_lstm_25_while_lstm_cell_25_biasadd_readvariableop_resourceJsequential_12_lstm_25_while_lstm_cell_25_biasadd_readvariableop_resource_0"
Isequential_12_lstm_25_while_lstm_cell_25_matmul_1_readvariableop_resourceKsequential_12_lstm_25_while_lstm_cell_25_matmul_1_readvariableop_resource_0"
Gsequential_12_lstm_25_while_lstm_cell_25_matmul_readvariableop_resourceIsequential_12_lstm_25_while_lstm_cell_25_matmul_readvariableop_resource_0"
Asequential_12_lstm_25_while_sequential_12_lstm_25_strided_slice_1Csequential_12_lstm_25_while_sequential_12_lstm_25_strided_slice_1_0"
}sequential_12_lstm_25_while_tensorarrayv2read_tensorlistgetitem_sequential_12_lstm_25_tensorarrayunstack_tensorlistfromtensorsequential_12_lstm_25_while_tensorarrayv2read_tensorlistgetitem_sequential_12_lstm_25_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿÚ:ÿÿÿÿÿÿÿÿÿÚ: : : : : 2
?sequential_12/lstm_25/while/lstm_cell_25/BiasAdd/ReadVariableOp?sequential_12/lstm_25/while/lstm_cell_25/BiasAdd/ReadVariableOp2
>sequential_12/lstm_25/while/lstm_cell_25/MatMul/ReadVariableOp>sequential_12/lstm_25/while/lstm_cell_25/MatMul/ReadVariableOp2
@sequential_12/lstm_25/while/lstm_cell_25/MatMul_1/ReadVariableOp@sequential_12/lstm_25/while/lstm_cell_25/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
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
F__inference_dense_12_layer_call_and_return_conditional_losses_39971255

inputs4
!tensordot_readvariableop_resource:	Ú-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢Tensordot/ReadVariableOp
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes
:	Ú*
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
:ÿÿÿÿÿÿÿÿÿÚ2
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
:ÿÿÿÿÿÿÿÿÿÚ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ
 
_user_specified_nameinputs
ËF

E__inference_lstm_25_layer_call_and_return_conditional_losses_39970625

inputs(
lstm_cell_25_39970543:	`è
)
lstm_cell_25_39970545:
Úè
$
lstm_cell_25_39970547:	è

identity¢$lstm_cell_25/StatefulPartitionedCall¢whileD
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
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`2
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
valueB"ÿÿÿÿ`   27
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
:ÿÿÿÿÿÿÿÿÿ`*
shrink_axis_mask2
strided_slice_2¨
$lstm_cell_25/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_25_39970543lstm_cell_25_39970545lstm_cell_25_39970547*
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
J__inference_lstm_cell_25_layer_call_and_return_conditional_losses_399704782&
$lstm_cell_25/StatefulPartitionedCall
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_25_39970543lstm_cell_25_39970545lstm_cell_25_39970547*
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
while_body_39970556*
condR
while_cond_39970555*M
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
NoOpNoOp%^lstm_cell_25/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`: : : 2L
$lstm_cell_25/StatefulPartitionedCall$lstm_cell_25/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`
 
_user_specified_nameinputs
?
Ò
while_body_39972627
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_24_matmul_readvariableop_resource_0:	]H
5while_lstm_cell_24_matmul_1_readvariableop_resource_0:	`C
4while_lstm_cell_24_biasadd_readvariableop_resource_0:	
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_24_matmul_readvariableop_resource:	]F
3while_lstm_cell_24_matmul_1_readvariableop_resource:	`A
2while_lstm_cell_24_biasadd_readvariableop_resource:	¢)while/lstm_cell_24/BiasAdd/ReadVariableOp¢(while/lstm_cell_24/MatMul/ReadVariableOp¢*while/lstm_cell_24/MatMul_1/ReadVariableOpÃ
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
(while/lstm_cell_24/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_24_matmul_readvariableop_resource_0*
_output_shapes
:	]*
dtype02*
(while/lstm_cell_24/MatMul/ReadVariableOp×
while/lstm_cell_24/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_24/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_24/MatMulÏ
*while/lstm_cell_24/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_24_matmul_1_readvariableop_resource_0*
_output_shapes
:	`*
dtype02,
*while/lstm_cell_24/MatMul_1/ReadVariableOpÀ
while/lstm_cell_24/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_24/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_24/MatMul_1¸
while/lstm_cell_24/addAddV2#while/lstm_cell_24/MatMul:product:0%while/lstm_cell_24/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_24/addÈ
)while/lstm_cell_24/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_24_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02+
)while/lstm_cell_24/BiasAdd/ReadVariableOpÅ
while/lstm_cell_24/BiasAddBiasAddwhile/lstm_cell_24/add:z:01while/lstm_cell_24/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_24/BiasAdd
"while/lstm_cell_24/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_24/split/split_dim
while/lstm_cell_24/splitSplit+while/lstm_cell_24/split/split_dim:output:0#while/lstm_cell_24/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ`:ÿÿÿÿÿÿÿÿÿ`:ÿÿÿÿÿÿÿÿÿ`:ÿÿÿÿÿÿÿÿÿ`*
	num_split2
while/lstm_cell_24/split
while/lstm_cell_24/SigmoidSigmoid!while/lstm_cell_24/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
while/lstm_cell_24/Sigmoid
while/lstm_cell_24/Sigmoid_1Sigmoid!while/lstm_cell_24/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
while/lstm_cell_24/Sigmoid_1 
while/lstm_cell_24/mulMul while/lstm_cell_24/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
while/lstm_cell_24/mul
while/lstm_cell_24/ReluRelu!while/lstm_cell_24/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
while/lstm_cell_24/Relu´
while/lstm_cell_24/mul_1Mulwhile/lstm_cell_24/Sigmoid:y:0%while/lstm_cell_24/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
while/lstm_cell_24/mul_1©
while/lstm_cell_24/add_1AddV2while/lstm_cell_24/mul:z:0while/lstm_cell_24/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
while/lstm_cell_24/add_1
while/lstm_cell_24/Sigmoid_2Sigmoid!while/lstm_cell_24/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
while/lstm_cell_24/Sigmoid_2
while/lstm_cell_24/Relu_1Reluwhile/lstm_cell_24/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
while/lstm_cell_24/Relu_1¸
while/lstm_cell_24/mul_2Mul while/lstm_cell_24/Sigmoid_2:y:0'while/lstm_cell_24/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
while/lstm_cell_24/mul_2à
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_24/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_24/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_24/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
while/Identity_5Þ

while/NoOpNoOp*^while/lstm_cell_24/BiasAdd/ReadVariableOp)^while/lstm_cell_24/MatMul/ReadVariableOp+^while/lstm_cell_24/MatMul_1/ReadVariableOp*"
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
2while_lstm_cell_24_biasadd_readvariableop_resource4while_lstm_cell_24_biasadd_readvariableop_resource_0"l
3while_lstm_cell_24_matmul_1_readvariableop_resource5while_lstm_cell_24_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_24_matmul_readvariableop_resource3while_lstm_cell_24_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ`:ÿÿÿÿÿÿÿÿÿ`: : : : : 2V
)while/lstm_cell_24/BiasAdd/ReadVariableOp)while/lstm_cell_24/BiasAdd/ReadVariableOp2T
(while/lstm_cell_24/MatMul/ReadVariableOp(while/lstm_cell_24/MatMul/ReadVariableOp2X
*while/lstm_cell_24/MatMul_1/ReadVariableOp*while/lstm_cell_24/MatMul_1/ReadVariableOp: 

_output_shapes
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
:ÿÿÿÿÿÿÿÿÿ`:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`:

_output_shapes
: :

_output_shapes
: 
Ï
g
H__inference_dropout_24_layer_call_and_return_conditional_losses_39971507

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *«ªª?2
dropout/Constw
dropout/MulMulinputsdropout/Const:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape¸
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  >2
dropout/GreaterEqual/yÂ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
dropout/Cast~
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
dropout/Mul_1i
IdentityIdentitydropout/Mul_1:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ`:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
 
_user_specified_nameinputs
Ô
I
-__inference_dropout_24_layer_call_fn_39973230

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
:ÿÿÿÿÿÿÿÿÿ`* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_dropout_24_layer_call_and_return_conditional_losses_399710572
PartitionedCallp
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ`:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
 
_user_specified_nameinputs
º
ø
/__inference_lstm_cell_24_layer_call_fn_39974031

inputs
states_0
states_1
unknown:	]
	unknown_0:	`
	unknown_1:	
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
9:ÿÿÿÿÿÿÿÿÿ`:ÿÿÿÿÿÿÿÿÿ`:ÿÿÿÿÿÿÿÿÿ`*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_lstm_cell_24_layer_call_and_return_conditional_losses_399697022
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2

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
?:ÿÿÿÿÿÿÿÿÿ]:ÿÿÿÿÿÿÿÿÿ`:ÿÿÿÿÿÿÿÿÿ`: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
 
_user_specified_nameinputs:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
"
_user_specified_name
states/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
"
_user_specified_name
states/1
ã
Í
while_cond_39970345
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_39970345___redundant_placeholder06
2while_while_cond_39970345___redundant_placeholder16
2while_while_cond_39970345___redundant_placeholder26
2while_while_cond_39970345___redundant_placeholder3
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
Ì

$__inference__traced_restore_39974377
file_prefix3
 assignvariableop_dense_12_kernel:	Ú.
 assignvariableop_1_dense_12_bias:&
assignvariableop_2_adam_iter:	 (
assignvariableop_3_adam_beta_1: (
assignvariableop_4_adam_beta_2: '
assignvariableop_5_adam_decay: /
%assignvariableop_6_adam_learning_rate: A
.assignvariableop_7_lstm_24_lstm_cell_24_kernel:	]K
8assignvariableop_8_lstm_24_lstm_cell_24_recurrent_kernel:	`;
,assignvariableop_9_lstm_24_lstm_cell_24_bias:	B
/assignvariableop_10_lstm_25_lstm_cell_25_kernel:	`è
M
9assignvariableop_11_lstm_25_lstm_cell_25_recurrent_kernel:
Úè
<
-assignvariableop_12_lstm_25_lstm_cell_25_bias:	è
#
assignvariableop_13_total: #
assignvariableop_14_count: %
assignvariableop_15_total_1: %
assignvariableop_16_count_1: =
*assignvariableop_17_adam_dense_12_kernel_m:	Ú6
(assignvariableop_18_adam_dense_12_bias_m:I
6assignvariableop_19_adam_lstm_24_lstm_cell_24_kernel_m:	]S
@assignvariableop_20_adam_lstm_24_lstm_cell_24_recurrent_kernel_m:	`C
4assignvariableop_21_adam_lstm_24_lstm_cell_24_bias_m:	I
6assignvariableop_22_adam_lstm_25_lstm_cell_25_kernel_m:	`è
T
@assignvariableop_23_adam_lstm_25_lstm_cell_25_recurrent_kernel_m:
Úè
C
4assignvariableop_24_adam_lstm_25_lstm_cell_25_bias_m:	è
=
*assignvariableop_25_adam_dense_12_kernel_v:	Ú6
(assignvariableop_26_adam_dense_12_bias_v:I
6assignvariableop_27_adam_lstm_24_lstm_cell_24_kernel_v:	]S
@assignvariableop_28_adam_lstm_24_lstm_cell_24_recurrent_kernel_v:	`C
4assignvariableop_29_adam_lstm_24_lstm_cell_24_bias_v:	I
6assignvariableop_30_adam_lstm_25_lstm_cell_25_kernel_v:	`è
T
@assignvariableop_31_adam_lstm_25_lstm_cell_25_recurrent_kernel_v:
Úè
C
4assignvariableop_32_adam_lstm_25_lstm_cell_25_bias_v:	è
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
AssignVariableOpAssignVariableOp assignvariableop_dense_12_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1¥
AssignVariableOp_1AssignVariableOp assignvariableop_1_dense_12_biasIdentity_1:output:0"/device:CPU:0*
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
AssignVariableOp_7AssignVariableOp.assignvariableop_7_lstm_24_lstm_cell_24_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8½
AssignVariableOp_8AssignVariableOp8assignvariableop_8_lstm_24_lstm_cell_24_recurrent_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9±
AssignVariableOp_9AssignVariableOp,assignvariableop_9_lstm_24_lstm_cell_24_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10·
AssignVariableOp_10AssignVariableOp/assignvariableop_10_lstm_25_lstm_cell_25_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11Á
AssignVariableOp_11AssignVariableOp9assignvariableop_11_lstm_25_lstm_cell_25_recurrent_kernelIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12µ
AssignVariableOp_12AssignVariableOp-assignvariableop_12_lstm_25_lstm_cell_25_biasIdentity_12:output:0"/device:CPU:0*
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
AssignVariableOp_17AssignVariableOp*assignvariableop_17_adam_dense_12_kernel_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18°
AssignVariableOp_18AssignVariableOp(assignvariableop_18_adam_dense_12_bias_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19¾
AssignVariableOp_19AssignVariableOp6assignvariableop_19_adam_lstm_24_lstm_cell_24_kernel_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20È
AssignVariableOp_20AssignVariableOp@assignvariableop_20_adam_lstm_24_lstm_cell_24_recurrent_kernel_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21¼
AssignVariableOp_21AssignVariableOp4assignvariableop_21_adam_lstm_24_lstm_cell_24_bias_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22¾
AssignVariableOp_22AssignVariableOp6assignvariableop_22_adam_lstm_25_lstm_cell_25_kernel_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23È
AssignVariableOp_23AssignVariableOp@assignvariableop_23_adam_lstm_25_lstm_cell_25_recurrent_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24¼
AssignVariableOp_24AssignVariableOp4assignvariableop_24_adam_lstm_25_lstm_cell_25_bias_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25²
AssignVariableOp_25AssignVariableOp*assignvariableop_25_adam_dense_12_kernel_vIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26°
AssignVariableOp_26AssignVariableOp(assignvariableop_26_adam_dense_12_bias_vIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27¾
AssignVariableOp_27AssignVariableOp6assignvariableop_27_adam_lstm_24_lstm_cell_24_kernel_vIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28È
AssignVariableOp_28AssignVariableOp@assignvariableop_28_adam_lstm_24_lstm_cell_24_recurrent_kernel_vIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29¼
AssignVariableOp_29AssignVariableOp4assignvariableop_29_adam_lstm_24_lstm_cell_24_bias_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30¾
AssignVariableOp_30AssignVariableOp6assignvariableop_30_adam_lstm_25_lstm_cell_25_kernel_vIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31È
AssignVariableOp_31AssignVariableOp@assignvariableop_31_adam_lstm_25_lstm_cell_25_recurrent_kernel_vIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32¼
AssignVariableOp_32AssignVariableOp4assignvariableop_32_adam_lstm_25_lstm_cell_25_bias_vIdentity_32:output:0"/device:CPU:0*
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
ß
Í
while_cond_39969715
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_39969715___redundant_placeholder06
2while_while_cond_39969715___redundant_placeholder16
2while_while_cond_39969715___redundant_placeholder26
2while_while_cond_39969715___redundant_placeholder3
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
@: : : : :ÿÿÿÿÿÿÿÿÿ`:ÿÿÿÿÿÿÿÿÿ`: ::::: 

_output_shapes
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
:ÿÿÿÿÿÿÿÿÿ`:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`:

_output_shapes
: :

_output_shapes
:
Í

K__inference_sequential_12_layer_call_and_return_conditional_losses_39972518

inputsF
3lstm_24_lstm_cell_24_matmul_readvariableop_resource:	]H
5lstm_24_lstm_cell_24_matmul_1_readvariableop_resource:	`C
4lstm_24_lstm_cell_24_biasadd_readvariableop_resource:	F
3lstm_25_lstm_cell_25_matmul_readvariableop_resource:	`è
I
5lstm_25_lstm_cell_25_matmul_1_readvariableop_resource:
Úè
C
4lstm_25_lstm_cell_25_biasadd_readvariableop_resource:	è
=
*dense_12_tensordot_readvariableop_resource:	Ú6
(dense_12_biasadd_readvariableop_resource:
identity¢dense_12/BiasAdd/ReadVariableOp¢!dense_12/Tensordot/ReadVariableOp¢+lstm_24/lstm_cell_24/BiasAdd/ReadVariableOp¢*lstm_24/lstm_cell_24/MatMul/ReadVariableOp¢,lstm_24/lstm_cell_24/MatMul_1/ReadVariableOp¢lstm_24/while¢+lstm_25/lstm_cell_25/BiasAdd/ReadVariableOp¢*lstm_25/lstm_cell_25/MatMul/ReadVariableOp¢,lstm_25/lstm_cell_25/MatMul_1/ReadVariableOp¢lstm_25/whileT
lstm_24/ShapeShapeinputs*
T0*
_output_shapes
:2
lstm_24/Shape
lstm_24/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_24/strided_slice/stack
lstm_24/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_24/strided_slice/stack_1
lstm_24/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_24/strided_slice/stack_2
lstm_24/strided_sliceStridedSlicelstm_24/Shape:output:0$lstm_24/strided_slice/stack:output:0&lstm_24/strided_slice/stack_1:output:0&lstm_24/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_24/strided_slicel
lstm_24/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :`2
lstm_24/zeros/mul/y
lstm_24/zeros/mulMullstm_24/strided_slice:output:0lstm_24/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_24/zeros/mulo
lstm_24/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
lstm_24/zeros/Less/y
lstm_24/zeros/LessLesslstm_24/zeros/mul:z:0lstm_24/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_24/zeros/Lessr
lstm_24/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :`2
lstm_24/zeros/packed/1£
lstm_24/zeros/packedPacklstm_24/strided_slice:output:0lstm_24/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_24/zeros/packedo
lstm_24/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_24/zeros/Const
lstm_24/zerosFilllstm_24/zeros/packed:output:0lstm_24/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
lstm_24/zerosp
lstm_24/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :`2
lstm_24/zeros_1/mul/y
lstm_24/zeros_1/mulMullstm_24/strided_slice:output:0lstm_24/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_24/zeros_1/muls
lstm_24/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
lstm_24/zeros_1/Less/y
lstm_24/zeros_1/LessLesslstm_24/zeros_1/mul:z:0lstm_24/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_24/zeros_1/Lessv
lstm_24/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :`2
lstm_24/zeros_1/packed/1©
lstm_24/zeros_1/packedPacklstm_24/strided_slice:output:0!lstm_24/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_24/zeros_1/packeds
lstm_24/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_24/zeros_1/Const
lstm_24/zeros_1Filllstm_24/zeros_1/packed:output:0lstm_24/zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
lstm_24/zeros_1
lstm_24/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_24/transpose/perm
lstm_24/transpose	Transposeinputslstm_24/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]2
lstm_24/transposeg
lstm_24/Shape_1Shapelstm_24/transpose:y:0*
T0*
_output_shapes
:2
lstm_24/Shape_1
lstm_24/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_24/strided_slice_1/stack
lstm_24/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_24/strided_slice_1/stack_1
lstm_24/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_24/strided_slice_1/stack_2
lstm_24/strided_slice_1StridedSlicelstm_24/Shape_1:output:0&lstm_24/strided_slice_1/stack:output:0(lstm_24/strided_slice_1/stack_1:output:0(lstm_24/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_24/strided_slice_1
#lstm_24/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2%
#lstm_24/TensorArrayV2/element_shapeÒ
lstm_24/TensorArrayV2TensorListReserve,lstm_24/TensorArrayV2/element_shape:output:0 lstm_24/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_24/TensorArrayV2Ï
=lstm_24/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ]   2?
=lstm_24/TensorArrayUnstack/TensorListFromTensor/element_shape
/lstm_24/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_24/transpose:y:0Flstm_24/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type021
/lstm_24/TensorArrayUnstack/TensorListFromTensor
lstm_24/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_24/strided_slice_2/stack
lstm_24/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_24/strided_slice_2/stack_1
lstm_24/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_24/strided_slice_2/stack_2¬
lstm_24/strided_slice_2StridedSlicelstm_24/transpose:y:0&lstm_24/strided_slice_2/stack:output:0(lstm_24/strided_slice_2/stack_1:output:0(lstm_24/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]*
shrink_axis_mask2
lstm_24/strided_slice_2Í
*lstm_24/lstm_cell_24/MatMul/ReadVariableOpReadVariableOp3lstm_24_lstm_cell_24_matmul_readvariableop_resource*
_output_shapes
:	]*
dtype02,
*lstm_24/lstm_cell_24/MatMul/ReadVariableOpÍ
lstm_24/lstm_cell_24/MatMulMatMul lstm_24/strided_slice_2:output:02lstm_24/lstm_cell_24/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_24/lstm_cell_24/MatMulÓ
,lstm_24/lstm_cell_24/MatMul_1/ReadVariableOpReadVariableOp5lstm_24_lstm_cell_24_matmul_1_readvariableop_resource*
_output_shapes
:	`*
dtype02.
,lstm_24/lstm_cell_24/MatMul_1/ReadVariableOpÉ
lstm_24/lstm_cell_24/MatMul_1MatMullstm_24/zeros:output:04lstm_24/lstm_cell_24/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_24/lstm_cell_24/MatMul_1À
lstm_24/lstm_cell_24/addAddV2%lstm_24/lstm_cell_24/MatMul:product:0'lstm_24/lstm_cell_24/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_24/lstm_cell_24/addÌ
+lstm_24/lstm_cell_24/BiasAdd/ReadVariableOpReadVariableOp4lstm_24_lstm_cell_24_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02-
+lstm_24/lstm_cell_24/BiasAdd/ReadVariableOpÍ
lstm_24/lstm_cell_24/BiasAddBiasAddlstm_24/lstm_cell_24/add:z:03lstm_24/lstm_cell_24/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_24/lstm_cell_24/BiasAdd
$lstm_24/lstm_cell_24/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2&
$lstm_24/lstm_cell_24/split/split_dim
lstm_24/lstm_cell_24/splitSplit-lstm_24/lstm_cell_24/split/split_dim:output:0%lstm_24/lstm_cell_24/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ`:ÿÿÿÿÿÿÿÿÿ`:ÿÿÿÿÿÿÿÿÿ`:ÿÿÿÿÿÿÿÿÿ`*
	num_split2
lstm_24/lstm_cell_24/split
lstm_24/lstm_cell_24/SigmoidSigmoid#lstm_24/lstm_cell_24/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
lstm_24/lstm_cell_24/Sigmoid¢
lstm_24/lstm_cell_24/Sigmoid_1Sigmoid#lstm_24/lstm_cell_24/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2 
lstm_24/lstm_cell_24/Sigmoid_1«
lstm_24/lstm_cell_24/mulMul"lstm_24/lstm_cell_24/Sigmoid_1:y:0lstm_24/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
lstm_24/lstm_cell_24/mul
lstm_24/lstm_cell_24/ReluRelu#lstm_24/lstm_cell_24/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
lstm_24/lstm_cell_24/Relu¼
lstm_24/lstm_cell_24/mul_1Mul lstm_24/lstm_cell_24/Sigmoid:y:0'lstm_24/lstm_cell_24/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
lstm_24/lstm_cell_24/mul_1±
lstm_24/lstm_cell_24/add_1AddV2lstm_24/lstm_cell_24/mul:z:0lstm_24/lstm_cell_24/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
lstm_24/lstm_cell_24/add_1¢
lstm_24/lstm_cell_24/Sigmoid_2Sigmoid#lstm_24/lstm_cell_24/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2 
lstm_24/lstm_cell_24/Sigmoid_2
lstm_24/lstm_cell_24/Relu_1Relulstm_24/lstm_cell_24/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
lstm_24/lstm_cell_24/Relu_1À
lstm_24/lstm_cell_24/mul_2Mul"lstm_24/lstm_cell_24/Sigmoid_2:y:0)lstm_24/lstm_cell_24/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
lstm_24/lstm_cell_24/mul_2
%lstm_24/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ`   2'
%lstm_24/TensorArrayV2_1/element_shapeØ
lstm_24/TensorArrayV2_1TensorListReserve.lstm_24/TensorArrayV2_1/element_shape:output:0 lstm_24/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_24/TensorArrayV2_1^
lstm_24/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_24/time
 lstm_24/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2"
 lstm_24/while/maximum_iterationsz
lstm_24/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_24/while/loop_counter
lstm_24/whileWhile#lstm_24/while/loop_counter:output:0)lstm_24/while/maximum_iterations:output:0lstm_24/time:output:0 lstm_24/TensorArrayV2_1:handle:0lstm_24/zeros:output:0lstm_24/zeros_1:output:0 lstm_24/strided_slice_1:output:0?lstm_24/TensorArrayUnstack/TensorListFromTensor:output_handle:03lstm_24_lstm_cell_24_matmul_readvariableop_resource5lstm_24_lstm_cell_24_matmul_1_readvariableop_resource4lstm_24_lstm_cell_24_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ`:ÿÿÿÿÿÿÿÿÿ`: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *'
bodyR
lstm_24_while_body_39972244*'
condR
lstm_24_while_cond_39972243*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ`:ÿÿÿÿÿÿÿÿÿ`: : : : : *
parallel_iterations 2
lstm_24/whileÅ
8lstm_24/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ`   2:
8lstm_24/TensorArrayV2Stack/TensorListStack/element_shape
*lstm_24/TensorArrayV2Stack/TensorListStackTensorListStacklstm_24/while:output:3Alstm_24/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`*
element_dtype02,
*lstm_24/TensorArrayV2Stack/TensorListStack
lstm_24/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
lstm_24/strided_slice_3/stack
lstm_24/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2!
lstm_24/strided_slice_3/stack_1
lstm_24/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_24/strided_slice_3/stack_2Ê
lstm_24/strided_slice_3StridedSlice3lstm_24/TensorArrayV2Stack/TensorListStack:tensor:0&lstm_24/strided_slice_3/stack:output:0(lstm_24/strided_slice_3/stack_1:output:0(lstm_24/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`*
shrink_axis_mask2
lstm_24/strided_slice_3
lstm_24/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_24/transpose_1/permÅ
lstm_24/transpose_1	Transpose3lstm_24/TensorArrayV2Stack/TensorListStack:tensor:0!lstm_24/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
lstm_24/transpose_1v
lstm_24/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_24/runtimey
dropout_24/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *«ªª?2
dropout_24/dropout/Const©
dropout_24/dropout/MulMullstm_24/transpose_1:y:0!dropout_24/dropout/Const:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
dropout_24/dropout/Mul{
dropout_24/dropout/ShapeShapelstm_24/transpose_1:y:0*
T0*
_output_shapes
:2
dropout_24/dropout/ShapeÙ
/dropout_24/dropout/random_uniform/RandomUniformRandomUniform!dropout_24/dropout/Shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`*
dtype021
/dropout_24/dropout/random_uniform/RandomUniform
!dropout_24/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  >2#
!dropout_24/dropout/GreaterEqual/yî
dropout_24/dropout/GreaterEqualGreaterEqual8dropout_24/dropout/random_uniform/RandomUniform:output:0*dropout_24/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2!
dropout_24/dropout/GreaterEqual¤
dropout_24/dropout/CastCast#dropout_24/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
dropout_24/dropout/Castª
dropout_24/dropout/Mul_1Muldropout_24/dropout/Mul:z:0dropout_24/dropout/Cast:y:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
dropout_24/dropout/Mul_1j
lstm_25/ShapeShapedropout_24/dropout/Mul_1:z:0*
T0*
_output_shapes
:2
lstm_25/Shape
lstm_25/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_25/strided_slice/stack
lstm_25/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_25/strided_slice/stack_1
lstm_25/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_25/strided_slice/stack_2
lstm_25/strided_sliceStridedSlicelstm_25/Shape:output:0$lstm_25/strided_slice/stack:output:0&lstm_25/strided_slice/stack_1:output:0&lstm_25/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_25/strided_slicem
lstm_25/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :Ú2
lstm_25/zeros/mul/y
lstm_25/zeros/mulMullstm_25/strided_slice:output:0lstm_25/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_25/zeros/mulo
lstm_25/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
lstm_25/zeros/Less/y
lstm_25/zeros/LessLesslstm_25/zeros/mul:z:0lstm_25/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_25/zeros/Lesss
lstm_25/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :Ú2
lstm_25/zeros/packed/1£
lstm_25/zeros/packedPacklstm_25/strided_slice:output:0lstm_25/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_25/zeros/packedo
lstm_25/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_25/zeros/Const
lstm_25/zerosFilllstm_25/zeros/packed:output:0lstm_25/zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
lstm_25/zerosq
lstm_25/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :Ú2
lstm_25/zeros_1/mul/y
lstm_25/zeros_1/mulMullstm_25/strided_slice:output:0lstm_25/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_25/zeros_1/muls
lstm_25/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
lstm_25/zeros_1/Less/y
lstm_25/zeros_1/LessLesslstm_25/zeros_1/mul:z:0lstm_25/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_25/zeros_1/Lessw
lstm_25/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :Ú2
lstm_25/zeros_1/packed/1©
lstm_25/zeros_1/packedPacklstm_25/strided_slice:output:0!lstm_25/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_25/zeros_1/packeds
lstm_25/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_25/zeros_1/Const
lstm_25/zeros_1Filllstm_25/zeros_1/packed:output:0lstm_25/zeros_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
lstm_25/zeros_1
lstm_25/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_25/transpose/perm¨
lstm_25/transpose	Transposedropout_24/dropout/Mul_1:z:0lstm_25/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
lstm_25/transposeg
lstm_25/Shape_1Shapelstm_25/transpose:y:0*
T0*
_output_shapes
:2
lstm_25/Shape_1
lstm_25/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_25/strided_slice_1/stack
lstm_25/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_25/strided_slice_1/stack_1
lstm_25/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_25/strided_slice_1/stack_2
lstm_25/strided_slice_1StridedSlicelstm_25/Shape_1:output:0&lstm_25/strided_slice_1/stack:output:0(lstm_25/strided_slice_1/stack_1:output:0(lstm_25/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_25/strided_slice_1
#lstm_25/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2%
#lstm_25/TensorArrayV2/element_shapeÒ
lstm_25/TensorArrayV2TensorListReserve,lstm_25/TensorArrayV2/element_shape:output:0 lstm_25/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_25/TensorArrayV2Ï
=lstm_25/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ`   2?
=lstm_25/TensorArrayUnstack/TensorListFromTensor/element_shape
/lstm_25/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_25/transpose:y:0Flstm_25/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type021
/lstm_25/TensorArrayUnstack/TensorListFromTensor
lstm_25/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_25/strided_slice_2/stack
lstm_25/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_25/strided_slice_2/stack_1
lstm_25/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_25/strided_slice_2/stack_2¬
lstm_25/strided_slice_2StridedSlicelstm_25/transpose:y:0&lstm_25/strided_slice_2/stack:output:0(lstm_25/strided_slice_2/stack_1:output:0(lstm_25/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`*
shrink_axis_mask2
lstm_25/strided_slice_2Í
*lstm_25/lstm_cell_25/MatMul/ReadVariableOpReadVariableOp3lstm_25_lstm_cell_25_matmul_readvariableop_resource*
_output_shapes
:	`è
*
dtype02,
*lstm_25/lstm_cell_25/MatMul/ReadVariableOpÍ
lstm_25/lstm_cell_25/MatMulMatMul lstm_25/strided_slice_2:output:02lstm_25/lstm_cell_25/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè
2
lstm_25/lstm_cell_25/MatMulÔ
,lstm_25/lstm_cell_25/MatMul_1/ReadVariableOpReadVariableOp5lstm_25_lstm_cell_25_matmul_1_readvariableop_resource* 
_output_shapes
:
Úè
*
dtype02.
,lstm_25/lstm_cell_25/MatMul_1/ReadVariableOpÉ
lstm_25/lstm_cell_25/MatMul_1MatMullstm_25/zeros:output:04lstm_25/lstm_cell_25/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè
2
lstm_25/lstm_cell_25/MatMul_1À
lstm_25/lstm_cell_25/addAddV2%lstm_25/lstm_cell_25/MatMul:product:0'lstm_25/lstm_cell_25/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè
2
lstm_25/lstm_cell_25/addÌ
+lstm_25/lstm_cell_25/BiasAdd/ReadVariableOpReadVariableOp4lstm_25_lstm_cell_25_biasadd_readvariableop_resource*
_output_shapes	
:è
*
dtype02-
+lstm_25/lstm_cell_25/BiasAdd/ReadVariableOpÍ
lstm_25/lstm_cell_25/BiasAddBiasAddlstm_25/lstm_cell_25/add:z:03lstm_25/lstm_cell_25/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè
2
lstm_25/lstm_cell_25/BiasAdd
$lstm_25/lstm_cell_25/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2&
$lstm_25/lstm_cell_25/split/split_dim
lstm_25/lstm_cell_25/splitSplit-lstm_25/lstm_cell_25/split/split_dim:output:0%lstm_25/lstm_cell_25/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿÚ:ÿÿÿÿÿÿÿÿÿÚ:ÿÿÿÿÿÿÿÿÿÚ:ÿÿÿÿÿÿÿÿÿÚ*
	num_split2
lstm_25/lstm_cell_25/split
lstm_25/lstm_cell_25/SigmoidSigmoid#lstm_25/lstm_cell_25/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
lstm_25/lstm_cell_25/Sigmoid£
lstm_25/lstm_cell_25/Sigmoid_1Sigmoid#lstm_25/lstm_cell_25/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2 
lstm_25/lstm_cell_25/Sigmoid_1¬
lstm_25/lstm_cell_25/mulMul"lstm_25/lstm_cell_25/Sigmoid_1:y:0lstm_25/zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
lstm_25/lstm_cell_25/mul
lstm_25/lstm_cell_25/ReluRelu#lstm_25/lstm_cell_25/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
lstm_25/lstm_cell_25/Relu½
lstm_25/lstm_cell_25/mul_1Mul lstm_25/lstm_cell_25/Sigmoid:y:0'lstm_25/lstm_cell_25/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
lstm_25/lstm_cell_25/mul_1²
lstm_25/lstm_cell_25/add_1AddV2lstm_25/lstm_cell_25/mul:z:0lstm_25/lstm_cell_25/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
lstm_25/lstm_cell_25/add_1£
lstm_25/lstm_cell_25/Sigmoid_2Sigmoid#lstm_25/lstm_cell_25/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2 
lstm_25/lstm_cell_25/Sigmoid_2
lstm_25/lstm_cell_25/Relu_1Relulstm_25/lstm_cell_25/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
lstm_25/lstm_cell_25/Relu_1Á
lstm_25/lstm_cell_25/mul_2Mul"lstm_25/lstm_cell_25/Sigmoid_2:y:0)lstm_25/lstm_cell_25/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
lstm_25/lstm_cell_25/mul_2
%lstm_25/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿZ  2'
%lstm_25/TensorArrayV2_1/element_shapeØ
lstm_25/TensorArrayV2_1TensorListReserve.lstm_25/TensorArrayV2_1/element_shape:output:0 lstm_25/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_25/TensorArrayV2_1^
lstm_25/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_25/time
 lstm_25/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2"
 lstm_25/while/maximum_iterationsz
lstm_25/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_25/while/loop_counter
lstm_25/whileWhile#lstm_25/while/loop_counter:output:0)lstm_25/while/maximum_iterations:output:0lstm_25/time:output:0 lstm_25/TensorArrayV2_1:handle:0lstm_25/zeros:output:0lstm_25/zeros_1:output:0 lstm_25/strided_slice_1:output:0?lstm_25/TensorArrayUnstack/TensorListFromTensor:output_handle:03lstm_25_lstm_cell_25_matmul_readvariableop_resource5lstm_25_lstm_cell_25_matmul_1_readvariableop_resource4lstm_25_lstm_cell_25_biasadd_readvariableop_resource*
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
lstm_25_while_body_39972399*'
condR
lstm_25_while_cond_39972398*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿÚ:ÿÿÿÿÿÿÿÿÿÚ: : : : : *
parallel_iterations 2
lstm_25/whileÅ
8lstm_25/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿZ  2:
8lstm_25/TensorArrayV2Stack/TensorListStack/element_shape
*lstm_25/TensorArrayV2Stack/TensorListStackTensorListStacklstm_25/while:output:3Alstm_25/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ*
element_dtype02,
*lstm_25/TensorArrayV2Stack/TensorListStack
lstm_25/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
lstm_25/strided_slice_3/stack
lstm_25/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2!
lstm_25/strided_slice_3/stack_1
lstm_25/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_25/strided_slice_3/stack_2Ë
lstm_25/strided_slice_3StridedSlice3lstm_25/TensorArrayV2Stack/TensorListStack:tensor:0&lstm_25/strided_slice_3/stack:output:0(lstm_25/strided_slice_3/stack_1:output:0(lstm_25/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ*
shrink_axis_mask2
lstm_25/strided_slice_3
lstm_25/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_25/transpose_1/permÆ
lstm_25/transpose_1	Transpose3lstm_25/TensorArrayV2Stack/TensorListStack:tensor:0!lstm_25/transpose_1/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
lstm_25/transpose_1v
lstm_25/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_25/runtimey
dropout_25/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout_25/dropout/Constª
dropout_25/dropout/MulMullstm_25/transpose_1:y:0!dropout_25/dropout/Const:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
dropout_25/dropout/Mul{
dropout_25/dropout/ShapeShapelstm_25/transpose_1:y:0*
T0*
_output_shapes
:2
dropout_25/dropout/ShapeÚ
/dropout_25/dropout/random_uniform/RandomUniformRandomUniform!dropout_25/dropout/Shape:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ*
dtype021
/dropout_25/dropout/random_uniform/RandomUniform
!dropout_25/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>2#
!dropout_25/dropout/GreaterEqual/yï
dropout_25/dropout/GreaterEqualGreaterEqual8dropout_25/dropout/random_uniform/RandomUniform:output:0*dropout_25/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2!
dropout_25/dropout/GreaterEqual¥
dropout_25/dropout/CastCast#dropout_25/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
dropout_25/dropout/Cast«
dropout_25/dropout/Mul_1Muldropout_25/dropout/Mul:z:0dropout_25/dropout/Cast:y:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
dropout_25/dropout/Mul_1²
!dense_12/Tensordot/ReadVariableOpReadVariableOp*dense_12_tensordot_readvariableop_resource*
_output_shapes
:	Ú*
dtype02#
!dense_12/Tensordot/ReadVariableOp|
dense_12/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_12/Tensordot/axes
dense_12/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_12/Tensordot/free
dense_12/Tensordot/ShapeShapedropout_25/dropout/Mul_1:z:0*
T0*
_output_shapes
:2
dense_12/Tensordot/Shape
 dense_12/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_12/Tensordot/GatherV2/axisþ
dense_12/Tensordot/GatherV2GatherV2!dense_12/Tensordot/Shape:output:0 dense_12/Tensordot/free:output:0)dense_12/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_12/Tensordot/GatherV2
"dense_12/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"dense_12/Tensordot/GatherV2_1/axis
dense_12/Tensordot/GatherV2_1GatherV2!dense_12/Tensordot/Shape:output:0 dense_12/Tensordot/axes:output:0+dense_12/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_12/Tensordot/GatherV2_1~
dense_12/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_12/Tensordot/Const¤
dense_12/Tensordot/ProdProd$dense_12/Tensordot/GatherV2:output:0!dense_12/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_12/Tensordot/Prod
dense_12/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_12/Tensordot/Const_1¬
dense_12/Tensordot/Prod_1Prod&dense_12/Tensordot/GatherV2_1:output:0#dense_12/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_12/Tensordot/Prod_1
dense_12/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2 
dense_12/Tensordot/concat/axisÝ
dense_12/Tensordot/concatConcatV2 dense_12/Tensordot/free:output:0 dense_12/Tensordot/axes:output:0'dense_12/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_12/Tensordot/concat°
dense_12/Tensordot/stackPack dense_12/Tensordot/Prod:output:0"dense_12/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_12/Tensordot/stackÂ
dense_12/Tensordot/transpose	Transposedropout_25/dropout/Mul_1:z:0"dense_12/Tensordot/concat:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
dense_12/Tensordot/transposeÃ
dense_12/Tensordot/ReshapeReshape dense_12/Tensordot/transpose:y:0!dense_12/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
dense_12/Tensordot/ReshapeÂ
dense_12/Tensordot/MatMulMatMul#dense_12/Tensordot/Reshape:output:0)dense_12/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_12/Tensordot/MatMul
dense_12/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
dense_12/Tensordot/Const_2
 dense_12/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_12/Tensordot/concat_1/axisê
dense_12/Tensordot/concat_1ConcatV2$dense_12/Tensordot/GatherV2:output:0#dense_12/Tensordot/Const_2:output:0)dense_12/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_12/Tensordot/concat_1´
dense_12/TensordotReshape#dense_12/Tensordot/MatMul:product:0$dense_12/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_12/Tensordot§
dense_12/BiasAdd/ReadVariableOpReadVariableOp(dense_12_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_12/BiasAdd/ReadVariableOp«
dense_12/BiasAddBiasAdddense_12/Tensordot:output:0'dense_12/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_12/BiasAdd
dense_12/SoftmaxSoftmaxdense_12/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_12/Softmaxy
IdentityIdentitydense_12/Softmax:softmax:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

IdentityÈ
NoOpNoOp ^dense_12/BiasAdd/ReadVariableOp"^dense_12/Tensordot/ReadVariableOp,^lstm_24/lstm_cell_24/BiasAdd/ReadVariableOp+^lstm_24/lstm_cell_24/MatMul/ReadVariableOp-^lstm_24/lstm_cell_24/MatMul_1/ReadVariableOp^lstm_24/while,^lstm_25/lstm_cell_25/BiasAdd/ReadVariableOp+^lstm_25/lstm_cell_25/MatMul/ReadVariableOp-^lstm_25/lstm_cell_25/MatMul_1/ReadVariableOp^lstm_25/while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ]: : : : : : : : 2B
dense_12/BiasAdd/ReadVariableOpdense_12/BiasAdd/ReadVariableOp2F
!dense_12/Tensordot/ReadVariableOp!dense_12/Tensordot/ReadVariableOp2Z
+lstm_24/lstm_cell_24/BiasAdd/ReadVariableOp+lstm_24/lstm_cell_24/BiasAdd/ReadVariableOp2X
*lstm_24/lstm_cell_24/MatMul/ReadVariableOp*lstm_24/lstm_cell_24/MatMul/ReadVariableOp2\
,lstm_24/lstm_cell_24/MatMul_1/ReadVariableOp,lstm_24/lstm_cell_24/MatMul_1/ReadVariableOp2
lstm_24/whilelstm_24/while2Z
+lstm_25/lstm_cell_25/BiasAdd/ReadVariableOp+lstm_25/lstm_cell_25/BiasAdd/ReadVariableOp2X
*lstm_25/lstm_cell_25/MatMul/ReadVariableOp*lstm_25/lstm_cell_25/MatMul/ReadVariableOp2\
,lstm_25/lstm_cell_25/MatMul_1/ReadVariableOp,lstm_25/lstm_cell_25/MatMul_1/ReadVariableOp2
lstm_25/whilelstm_25/while:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
 
_user_specified_nameinputs
ã
Í
while_cond_39971124
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_39971124___redundant_placeholder06
2while_while_cond_39971124___redundant_placeholder16
2while_while_cond_39971124___redundant_placeholder26
2while_while_cond_39971124___redundant_placeholder3
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
J__inference_lstm_cell_25_layer_call_and_return_conditional_losses_39970332

inputs

states
states_11
matmul_readvariableop_resource:	`è
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
:	`è
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
A:ÿÿÿÿÿÿÿÿÿ`:ÿÿÿÿÿÿÿÿÿÚ:ÿÿÿÿÿÿÿÿÿÚ: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
 
_user_specified_nameinputs:PL
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ
 
_user_specified_namestates:PL
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ
 
_user_specified_namestates
ç[

E__inference_lstm_24_layer_call_and_return_conditional_losses_39971044

inputs>
+lstm_cell_24_matmul_readvariableop_resource:	]@
-lstm_cell_24_matmul_1_readvariableop_resource:	`;
,lstm_cell_24_biasadd_readvariableop_resource:	
identity¢#lstm_cell_24/BiasAdd/ReadVariableOp¢"lstm_cell_24/MatMul/ReadVariableOp¢$lstm_cell_24/MatMul_1/ReadVariableOp¢whileD
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
value	B :`2
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
value	B :`2
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
:ÿÿÿÿÿÿÿÿÿ`2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :`2
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
value	B :`2
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
:ÿÿÿÿÿÿÿÿÿ`2	
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
"lstm_cell_24/MatMul/ReadVariableOpReadVariableOp+lstm_cell_24_matmul_readvariableop_resource*
_output_shapes
:	]*
dtype02$
"lstm_cell_24/MatMul/ReadVariableOp­
lstm_cell_24/MatMulMatMulstrided_slice_2:output:0*lstm_cell_24/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_24/MatMul»
$lstm_cell_24/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_24_matmul_1_readvariableop_resource*
_output_shapes
:	`*
dtype02&
$lstm_cell_24/MatMul_1/ReadVariableOp©
lstm_cell_24/MatMul_1MatMulzeros:output:0,lstm_cell_24/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_24/MatMul_1 
lstm_cell_24/addAddV2lstm_cell_24/MatMul:product:0lstm_cell_24/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_24/add´
#lstm_cell_24/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_24_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02%
#lstm_cell_24/BiasAdd/ReadVariableOp­
lstm_cell_24/BiasAddBiasAddlstm_cell_24/add:z:0+lstm_cell_24/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_24/BiasAdd~
lstm_cell_24/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_24/split/split_dimó
lstm_cell_24/splitSplit%lstm_cell_24/split/split_dim:output:0lstm_cell_24/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ`:ÿÿÿÿÿÿÿÿÿ`:ÿÿÿÿÿÿÿÿÿ`:ÿÿÿÿÿÿÿÿÿ`*
	num_split2
lstm_cell_24/split
lstm_cell_24/SigmoidSigmoidlstm_cell_24/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
lstm_cell_24/Sigmoid
lstm_cell_24/Sigmoid_1Sigmoidlstm_cell_24/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
lstm_cell_24/Sigmoid_1
lstm_cell_24/mulMullstm_cell_24/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
lstm_cell_24/mul}
lstm_cell_24/ReluRelulstm_cell_24/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
lstm_cell_24/Relu
lstm_cell_24/mul_1Mullstm_cell_24/Sigmoid:y:0lstm_cell_24/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
lstm_cell_24/mul_1
lstm_cell_24/add_1AddV2lstm_cell_24/mul:z:0lstm_cell_24/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
lstm_cell_24/add_1
lstm_cell_24/Sigmoid_2Sigmoidlstm_cell_24/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
lstm_cell_24/Sigmoid_2|
lstm_cell_24/Relu_1Relulstm_cell_24/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
lstm_cell_24/Relu_1 
lstm_cell_24/mul_2Mullstm_cell_24/Sigmoid_2:y:0!lstm_cell_24/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
lstm_cell_24/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ`   2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_24_matmul_readvariableop_resource-lstm_cell_24_matmul_1_readvariableop_resource,lstm_cell_24_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ`:ÿÿÿÿÿÿÿÿÿ`: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_39970960*
condR
while_cond_39970959*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ`:ÿÿÿÿÿÿÿÿÿ`: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ`   22
0TensorArrayV2Stack/TensorListStack/element_shapeè
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`*
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
:ÿÿÿÿÿÿÿÿÿ`*
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
:ÿÿÿÿÿÿÿÿÿ`2
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
:ÿÿÿÿÿÿÿÿÿ`2

IdentityÈ
NoOpNoOp$^lstm_cell_24/BiasAdd/ReadVariableOp#^lstm_cell_24/MatMul/ReadVariableOp%^lstm_cell_24/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ]: : : 2J
#lstm_cell_24/BiasAdd/ReadVariableOp#lstm_cell_24/BiasAdd/ReadVariableOp2H
"lstm_cell_24/MatMul/ReadVariableOp"lstm_cell_24/MatMul/ReadVariableOp2L
$lstm_cell_24/MatMul_1/ReadVariableOp$lstm_cell_24/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
 
_user_specified_nameinputs
ú%
ñ
while_body_39969926
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_00
while_lstm_cell_24_39969950_0:	]0
while_lstm_cell_24_39969952_0:	`,
while_lstm_cell_24_39969954_0:	
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor.
while_lstm_cell_24_39969950:	].
while_lstm_cell_24_39969952:	`*
while_lstm_cell_24_39969954:	¢*while/lstm_cell_24/StatefulPartitionedCallÃ
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
)while/TensorArrayV2Read/TensorListGetItemé
*while/lstm_cell_24/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_24_39969950_0while_lstm_cell_24_39969952_0while_lstm_cell_24_39969954_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ`:ÿÿÿÿÿÿÿÿÿ`:ÿÿÿÿÿÿÿÿÿ`*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_lstm_cell_24_layer_call_and_return_conditional_losses_399698482,
*while/lstm_cell_24/StatefulPartitionedCall÷
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder3while/lstm_cell_24/StatefulPartitionedCall:output:0*
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
while/Identity_4Identity3while/lstm_cell_24/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
while/Identity_4¤
while/Identity_5Identity3while/lstm_cell_24/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
while/Identity_5

while/NoOpNoOp+^while/lstm_cell_24/StatefulPartitionedCall*"
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
while_lstm_cell_24_39969950while_lstm_cell_24_39969950_0"<
while_lstm_cell_24_39969952while_lstm_cell_24_39969952_0"<
while_lstm_cell_24_39969954while_lstm_cell_24_39969954_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ`:ÿÿÿÿÿÿÿÿÿ`: : : : : 2X
*while/lstm_cell_24/StatefulPartitionedCall*while/lstm_cell_24/StatefulPartitionedCall: 

_output_shapes
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
:ÿÿÿÿÿÿÿÿÿ`:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`:

_output_shapes
: :

_output_shapes
: 
Ç
ù
/__inference_lstm_cell_25_layer_call_fn_39974129

inputs
states_0
states_1
unknown:	`è
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
J__inference_lstm_cell_25_layer_call_and_return_conditional_losses_399703322
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
A:ÿÿÿÿÿÿÿÿÿ`:ÿÿÿÿÿÿÿÿÿÚ:ÿÿÿÿÿÿÿÿÿÚ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
 
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
Ã\
 
E__inference_lstm_25_layer_call_and_return_conditional_losses_39973537
inputs_0>
+lstm_cell_25_matmul_readvariableop_resource:	`è
A
-lstm_cell_25_matmul_1_readvariableop_resource:
Úè
;
,lstm_cell_25_biasadd_readvariableop_resource:	è

identity¢#lstm_cell_25/BiasAdd/ReadVariableOp¢"lstm_cell_25/MatMul/ReadVariableOp¢$lstm_cell_25/MatMul_1/ReadVariableOp¢whileF
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
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`2
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
valueB"ÿÿÿÿ`   27
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
:ÿÿÿÿÿÿÿÿÿ`*
shrink_axis_mask2
strided_slice_2µ
"lstm_cell_25/MatMul/ReadVariableOpReadVariableOp+lstm_cell_25_matmul_readvariableop_resource*
_output_shapes
:	`è
*
dtype02$
"lstm_cell_25/MatMul/ReadVariableOp­
lstm_cell_25/MatMulMatMulstrided_slice_2:output:0*lstm_cell_25/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè
2
lstm_cell_25/MatMul¼
$lstm_cell_25/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_25_matmul_1_readvariableop_resource* 
_output_shapes
:
Úè
*
dtype02&
$lstm_cell_25/MatMul_1/ReadVariableOp©
lstm_cell_25/MatMul_1MatMulzeros:output:0,lstm_cell_25/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè
2
lstm_cell_25/MatMul_1 
lstm_cell_25/addAddV2lstm_cell_25/MatMul:product:0lstm_cell_25/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè
2
lstm_cell_25/add´
#lstm_cell_25/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_25_biasadd_readvariableop_resource*
_output_shapes	
:è
*
dtype02%
#lstm_cell_25/BiasAdd/ReadVariableOp­
lstm_cell_25/BiasAddBiasAddlstm_cell_25/add:z:0+lstm_cell_25/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè
2
lstm_cell_25/BiasAdd~
lstm_cell_25/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_25/split/split_dim÷
lstm_cell_25/splitSplit%lstm_cell_25/split/split_dim:output:0lstm_cell_25/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿÚ:ÿÿÿÿÿÿÿÿÿÚ:ÿÿÿÿÿÿÿÿÿÚ:ÿÿÿÿÿÿÿÿÿÚ*
	num_split2
lstm_cell_25/split
lstm_cell_25/SigmoidSigmoidlstm_cell_25/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
lstm_cell_25/Sigmoid
lstm_cell_25/Sigmoid_1Sigmoidlstm_cell_25/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
lstm_cell_25/Sigmoid_1
lstm_cell_25/mulMullstm_cell_25/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
lstm_cell_25/mul~
lstm_cell_25/ReluRelulstm_cell_25/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
lstm_cell_25/Relu
lstm_cell_25/mul_1Mullstm_cell_25/Sigmoid:y:0lstm_cell_25/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
lstm_cell_25/mul_1
lstm_cell_25/add_1AddV2lstm_cell_25/mul:z:0lstm_cell_25/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
lstm_cell_25/add_1
lstm_cell_25/Sigmoid_2Sigmoidlstm_cell_25/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
lstm_cell_25/Sigmoid_2}
lstm_cell_25/Relu_1Relulstm_cell_25/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
lstm_cell_25/Relu_1¡
lstm_cell_25/mul_2Mullstm_cell_25/Sigmoid_2:y:0!lstm_cell_25/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
lstm_cell_25/mul_2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_25_matmul_readvariableop_resource-lstm_cell_25_matmul_1_readvariableop_resource,lstm_cell_25_biasadd_readvariableop_resource*
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
while_body_39973453*
condR
while_cond_39973452*M
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
NoOpNoOp$^lstm_cell_25/BiasAdd/ReadVariableOp#^lstm_cell_25/MatMul/ReadVariableOp%^lstm_cell_25/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`: : : 2J
#lstm_cell_25/BiasAdd/ReadVariableOp#lstm_cell_25/BiasAdd/ReadVariableOp2H
"lstm_cell_25/MatMul/ReadVariableOp"lstm_cell_25/MatMul/ReadVariableOp2L
$lstm_cell_25/MatMul_1/ReadVariableOp$lstm_cell_25/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`
"
_user_specified_name
inputs/0
&
ó
while_body_39970556
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_00
while_lstm_cell_25_39970580_0:	`è
1
while_lstm_cell_25_39970582_0:
Úè
,
while_lstm_cell_25_39970584_0:	è

while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor.
while_lstm_cell_25_39970580:	`è
/
while_lstm_cell_25_39970582:
Úè
*
while_lstm_cell_25_39970584:	è
¢*while/lstm_cell_25/StatefulPartitionedCallÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ`   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÓ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemì
*while/lstm_cell_25/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_25_39970580_0while_lstm_cell_25_39970582_0while_lstm_cell_25_39970584_0*
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
J__inference_lstm_cell_25_layer_call_and_return_conditional_losses_399704782,
*while/lstm_cell_25/StatefulPartitionedCall÷
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder3while/lstm_cell_25/StatefulPartitionedCall:output:0*
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
while/Identity_4Identity3while/lstm_cell_25/StatefulPartitionedCall:output:1^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
while/Identity_4¥
while/Identity_5Identity3while/lstm_cell_25/StatefulPartitionedCall:output:2^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
while/Identity_5

while/NoOpNoOp+^while/lstm_cell_25/StatefulPartitionedCall*"
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
while_lstm_cell_25_39970580while_lstm_cell_25_39970580_0"<
while_lstm_cell_25_39970582while_lstm_cell_25_39970582_0"<
while_lstm_cell_25_39970584while_lstm_cell_25_39970584_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿÚ:ÿÿÿÿÿÿÿÿÿÚ: : : : : 2X
*while/lstm_cell_25/StatefulPartitionedCall*while/lstm_cell_25/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
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
ç[

E__inference_lstm_24_layer_call_and_return_conditional_losses_39973013

inputs>
+lstm_cell_24_matmul_readvariableop_resource:	]@
-lstm_cell_24_matmul_1_readvariableop_resource:	`;
,lstm_cell_24_biasadd_readvariableop_resource:	
identity¢#lstm_cell_24/BiasAdd/ReadVariableOp¢"lstm_cell_24/MatMul/ReadVariableOp¢$lstm_cell_24/MatMul_1/ReadVariableOp¢whileD
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
value	B :`2
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
value	B :`2
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
:ÿÿÿÿÿÿÿÿÿ`2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :`2
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
value	B :`2
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
:ÿÿÿÿÿÿÿÿÿ`2	
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
"lstm_cell_24/MatMul/ReadVariableOpReadVariableOp+lstm_cell_24_matmul_readvariableop_resource*
_output_shapes
:	]*
dtype02$
"lstm_cell_24/MatMul/ReadVariableOp­
lstm_cell_24/MatMulMatMulstrided_slice_2:output:0*lstm_cell_24/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_24/MatMul»
$lstm_cell_24/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_24_matmul_1_readvariableop_resource*
_output_shapes
:	`*
dtype02&
$lstm_cell_24/MatMul_1/ReadVariableOp©
lstm_cell_24/MatMul_1MatMulzeros:output:0,lstm_cell_24/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_24/MatMul_1 
lstm_cell_24/addAddV2lstm_cell_24/MatMul:product:0lstm_cell_24/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_24/add´
#lstm_cell_24/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_24_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02%
#lstm_cell_24/BiasAdd/ReadVariableOp­
lstm_cell_24/BiasAddBiasAddlstm_cell_24/add:z:0+lstm_cell_24/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_24/BiasAdd~
lstm_cell_24/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_24/split/split_dimó
lstm_cell_24/splitSplit%lstm_cell_24/split/split_dim:output:0lstm_cell_24/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ`:ÿÿÿÿÿÿÿÿÿ`:ÿÿÿÿÿÿÿÿÿ`:ÿÿÿÿÿÿÿÿÿ`*
	num_split2
lstm_cell_24/split
lstm_cell_24/SigmoidSigmoidlstm_cell_24/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
lstm_cell_24/Sigmoid
lstm_cell_24/Sigmoid_1Sigmoidlstm_cell_24/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
lstm_cell_24/Sigmoid_1
lstm_cell_24/mulMullstm_cell_24/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
lstm_cell_24/mul}
lstm_cell_24/ReluRelulstm_cell_24/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
lstm_cell_24/Relu
lstm_cell_24/mul_1Mullstm_cell_24/Sigmoid:y:0lstm_cell_24/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
lstm_cell_24/mul_1
lstm_cell_24/add_1AddV2lstm_cell_24/mul:z:0lstm_cell_24/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
lstm_cell_24/add_1
lstm_cell_24/Sigmoid_2Sigmoidlstm_cell_24/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
lstm_cell_24/Sigmoid_2|
lstm_cell_24/Relu_1Relulstm_cell_24/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
lstm_cell_24/Relu_1 
lstm_cell_24/mul_2Mullstm_cell_24/Sigmoid_2:y:0!lstm_cell_24/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
lstm_cell_24/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ`   2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_24_matmul_readvariableop_resource-lstm_cell_24_matmul_1_readvariableop_resource,lstm_cell_24_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ`:ÿÿÿÿÿÿÿÿÿ`: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_39972929*
condR
while_cond_39972928*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ`:ÿÿÿÿÿÿÿÿÿ`: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ`   22
0TensorArrayV2Stack/TensorListStack/element_shapeè
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`*
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
:ÿÿÿÿÿÿÿÿÿ`*
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
:ÿÿÿÿÿÿÿÿÿ`2
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
:ÿÿÿÿÿÿÿÿÿ`2

IdentityÈ
NoOpNoOp$^lstm_cell_24/BiasAdd/ReadVariableOp#^lstm_cell_24/MatMul/ReadVariableOp%^lstm_cell_24/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ]: : : 2J
#lstm_cell_24/BiasAdd/ReadVariableOp#lstm_cell_24/BiasAdd/ReadVariableOp2H
"lstm_cell_24/MatMul/ReadVariableOp"lstm_cell_24/MatMul/ReadVariableOp2L
$lstm_cell_24/MatMul_1/ReadVariableOp$lstm_cell_24/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
 
_user_specified_nameinputs
Ã\
 
E__inference_lstm_25_layer_call_and_return_conditional_losses_39973386
inputs_0>
+lstm_cell_25_matmul_readvariableop_resource:	`è
A
-lstm_cell_25_matmul_1_readvariableop_resource:
Úè
;
,lstm_cell_25_biasadd_readvariableop_resource:	è

identity¢#lstm_cell_25/BiasAdd/ReadVariableOp¢"lstm_cell_25/MatMul/ReadVariableOp¢$lstm_cell_25/MatMul_1/ReadVariableOp¢whileF
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
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`2
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
valueB"ÿÿÿÿ`   27
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
:ÿÿÿÿÿÿÿÿÿ`*
shrink_axis_mask2
strided_slice_2µ
"lstm_cell_25/MatMul/ReadVariableOpReadVariableOp+lstm_cell_25_matmul_readvariableop_resource*
_output_shapes
:	`è
*
dtype02$
"lstm_cell_25/MatMul/ReadVariableOp­
lstm_cell_25/MatMulMatMulstrided_slice_2:output:0*lstm_cell_25/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè
2
lstm_cell_25/MatMul¼
$lstm_cell_25/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_25_matmul_1_readvariableop_resource* 
_output_shapes
:
Úè
*
dtype02&
$lstm_cell_25/MatMul_1/ReadVariableOp©
lstm_cell_25/MatMul_1MatMulzeros:output:0,lstm_cell_25/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè
2
lstm_cell_25/MatMul_1 
lstm_cell_25/addAddV2lstm_cell_25/MatMul:product:0lstm_cell_25/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè
2
lstm_cell_25/add´
#lstm_cell_25/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_25_biasadd_readvariableop_resource*
_output_shapes	
:è
*
dtype02%
#lstm_cell_25/BiasAdd/ReadVariableOp­
lstm_cell_25/BiasAddBiasAddlstm_cell_25/add:z:0+lstm_cell_25/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè
2
lstm_cell_25/BiasAdd~
lstm_cell_25/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_25/split/split_dim÷
lstm_cell_25/splitSplit%lstm_cell_25/split/split_dim:output:0lstm_cell_25/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿÚ:ÿÿÿÿÿÿÿÿÿÚ:ÿÿÿÿÿÿÿÿÿÚ:ÿÿÿÿÿÿÿÿÿÚ*
	num_split2
lstm_cell_25/split
lstm_cell_25/SigmoidSigmoidlstm_cell_25/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
lstm_cell_25/Sigmoid
lstm_cell_25/Sigmoid_1Sigmoidlstm_cell_25/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
lstm_cell_25/Sigmoid_1
lstm_cell_25/mulMullstm_cell_25/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
lstm_cell_25/mul~
lstm_cell_25/ReluRelulstm_cell_25/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
lstm_cell_25/Relu
lstm_cell_25/mul_1Mullstm_cell_25/Sigmoid:y:0lstm_cell_25/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
lstm_cell_25/mul_1
lstm_cell_25/add_1AddV2lstm_cell_25/mul:z:0lstm_cell_25/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
lstm_cell_25/add_1
lstm_cell_25/Sigmoid_2Sigmoidlstm_cell_25/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
lstm_cell_25/Sigmoid_2}
lstm_cell_25/Relu_1Relulstm_cell_25/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
lstm_cell_25/Relu_1¡
lstm_cell_25/mul_2Mullstm_cell_25/Sigmoid_2:y:0!lstm_cell_25/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
lstm_cell_25/mul_2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_25_matmul_readvariableop_resource-lstm_cell_25_matmul_1_readvariableop_resource,lstm_cell_25_biasadd_readvariableop_resource*
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
while_body_39973302*
condR
while_cond_39973301*M
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
NoOpNoOp$^lstm_cell_25/BiasAdd/ReadVariableOp#^lstm_cell_25/MatMul/ReadVariableOp%^lstm_cell_25/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`: : : 2J
#lstm_cell_25/BiasAdd/ReadVariableOp#lstm_cell_25/BiasAdd/ReadVariableOp2H
"lstm_cell_25/MatMul/ReadVariableOp"lstm_cell_25/MatMul/ReadVariableOp2L
$lstm_cell_25/MatMul_1/ReadVariableOp$lstm_cell_25/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`
"
_user_specified_name
inputs/0
Ý
¹
*__inference_lstm_24_layer_call_fn_39973175
inputs_0
unknown:	]
	unknown_0:	`
	unknown_1:	
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_lstm_24_layer_call_and_return_conditional_losses_399697852
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`2

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
õ

)sequential_12_lstm_24_while_cond_39969366H
Dsequential_12_lstm_24_while_sequential_12_lstm_24_while_loop_counterN
Jsequential_12_lstm_24_while_sequential_12_lstm_24_while_maximum_iterations+
'sequential_12_lstm_24_while_placeholder-
)sequential_12_lstm_24_while_placeholder_1-
)sequential_12_lstm_24_while_placeholder_2-
)sequential_12_lstm_24_while_placeholder_3J
Fsequential_12_lstm_24_while_less_sequential_12_lstm_24_strided_slice_1b
^sequential_12_lstm_24_while_sequential_12_lstm_24_while_cond_39969366___redundant_placeholder0b
^sequential_12_lstm_24_while_sequential_12_lstm_24_while_cond_39969366___redundant_placeholder1b
^sequential_12_lstm_24_while_sequential_12_lstm_24_while_cond_39969366___redundant_placeholder2b
^sequential_12_lstm_24_while_sequential_12_lstm_24_while_cond_39969366___redundant_placeholder3(
$sequential_12_lstm_24_while_identity
Þ
 sequential_12/lstm_24/while/LessLess'sequential_12_lstm_24_while_placeholderFsequential_12_lstm_24_while_less_sequential_12_lstm_24_strided_slice_1*
T0*
_output_shapes
: 2"
 sequential_12/lstm_24/while/Less
$sequential_12/lstm_24/while/IdentityIdentity$sequential_12/lstm_24/while/Less:z:0*
T0
*
_output_shapes
: 2&
$sequential_12/lstm_24/while/Identity"U
$sequential_12_lstm_24_while_identity-sequential_12/lstm_24/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿ`:ÿÿÿÿÿÿÿÿÿ`: ::::: 

_output_shapes
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
:ÿÿÿÿÿÿÿÿÿ`:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`:

_output_shapes
: :

_output_shapes
:


J__inference_lstm_cell_25_layer_call_and_return_conditional_losses_39974080

inputs
states_0
states_11
matmul_readvariableop_resource:	`è
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
:	`è
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
A:ÿÿÿÿÿÿÿÿÿ`:ÿÿÿÿÿÿÿÿÿÚ:ÿÿÿÿÿÿÿÿÿÚ: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
 
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
¬

Ò
0__inference_sequential_12_layer_call_fn_39971281
lstm_24_input
unknown:	]
	unknown_0:	`
	unknown_1:	
	unknown_2:	`è

	unknown_3:
Úè

	unknown_4:	è

	unknown_5:	Ú
	unknown_6:
identity¢StatefulPartitionedCallÔ
StatefulPartitionedCallStatefulPartitionedCalllstm_24_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
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
K__inference_sequential_12_layer_call_and_return_conditional_losses_399712622
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
_user_specified_namelstm_24_input
\

E__inference_lstm_25_layer_call_and_return_conditional_losses_39973688

inputs>
+lstm_cell_25_matmul_readvariableop_resource:	`è
A
-lstm_cell_25_matmul_1_readvariableop_resource:
Úè
;
,lstm_cell_25_biasadd_readvariableop_resource:	è

identity¢#lstm_cell_25/BiasAdd/ReadVariableOp¢"lstm_cell_25/MatMul/ReadVariableOp¢$lstm_cell_25/MatMul_1/ReadVariableOp¢whileD
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
:ÿÿÿÿÿÿÿÿÿ`2
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
valueB"ÿÿÿÿ`   27
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
:ÿÿÿÿÿÿÿÿÿ`*
shrink_axis_mask2
strided_slice_2µ
"lstm_cell_25/MatMul/ReadVariableOpReadVariableOp+lstm_cell_25_matmul_readvariableop_resource*
_output_shapes
:	`è
*
dtype02$
"lstm_cell_25/MatMul/ReadVariableOp­
lstm_cell_25/MatMulMatMulstrided_slice_2:output:0*lstm_cell_25/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè
2
lstm_cell_25/MatMul¼
$lstm_cell_25/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_25_matmul_1_readvariableop_resource* 
_output_shapes
:
Úè
*
dtype02&
$lstm_cell_25/MatMul_1/ReadVariableOp©
lstm_cell_25/MatMul_1MatMulzeros:output:0,lstm_cell_25/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè
2
lstm_cell_25/MatMul_1 
lstm_cell_25/addAddV2lstm_cell_25/MatMul:product:0lstm_cell_25/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè
2
lstm_cell_25/add´
#lstm_cell_25/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_25_biasadd_readvariableop_resource*
_output_shapes	
:è
*
dtype02%
#lstm_cell_25/BiasAdd/ReadVariableOp­
lstm_cell_25/BiasAddBiasAddlstm_cell_25/add:z:0+lstm_cell_25/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè
2
lstm_cell_25/BiasAdd~
lstm_cell_25/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_25/split/split_dim÷
lstm_cell_25/splitSplit%lstm_cell_25/split/split_dim:output:0lstm_cell_25/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿÚ:ÿÿÿÿÿÿÿÿÿÚ:ÿÿÿÿÿÿÿÿÿÚ:ÿÿÿÿÿÿÿÿÿÚ*
	num_split2
lstm_cell_25/split
lstm_cell_25/SigmoidSigmoidlstm_cell_25/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
lstm_cell_25/Sigmoid
lstm_cell_25/Sigmoid_1Sigmoidlstm_cell_25/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
lstm_cell_25/Sigmoid_1
lstm_cell_25/mulMullstm_cell_25/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
lstm_cell_25/mul~
lstm_cell_25/ReluRelulstm_cell_25/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
lstm_cell_25/Relu
lstm_cell_25/mul_1Mullstm_cell_25/Sigmoid:y:0lstm_cell_25/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
lstm_cell_25/mul_1
lstm_cell_25/add_1AddV2lstm_cell_25/mul:z:0lstm_cell_25/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
lstm_cell_25/add_1
lstm_cell_25/Sigmoid_2Sigmoidlstm_cell_25/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
lstm_cell_25/Sigmoid_2}
lstm_cell_25/Relu_1Relulstm_cell_25/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
lstm_cell_25/Relu_1¡
lstm_cell_25/mul_2Mullstm_cell_25/Sigmoid_2:y:0!lstm_cell_25/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
lstm_cell_25/mul_2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_25_matmul_readvariableop_resource-lstm_cell_25_matmul_1_readvariableop_resource,lstm_cell_25_biasadd_readvariableop_resource*
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
while_body_39973604*
condR
while_cond_39973603*M
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
NoOpNoOp$^lstm_cell_25/BiasAdd/ReadVariableOp#^lstm_cell_25/MatMul/ReadVariableOp%^lstm_cell_25/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ`: : : 2J
#lstm_cell_25/BiasAdd/ReadVariableOp#lstm_cell_25/BiasAdd/ReadVariableOp2H
"lstm_cell_25/MatMul/ReadVariableOp"lstm_cell_25/MatMul/ReadVariableOp2L
$lstm_cell_25/MatMul_1/ReadVariableOp$lstm_cell_25/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
 
_user_specified_nameinputs
ã
Í
while_cond_39973754
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_39973754___redundant_placeholder06
2while_while_cond_39973754___redundant_placeholder16
2while_while_cond_39973754___redundant_placeholder26
2while_while_cond_39973754___redundant_placeholder3
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


Ë
0__inference_sequential_12_layer_call_fn_39972560

inputs
unknown:	]
	unknown_0:	`
	unknown_1:	
	unknown_2:	`è

	unknown_3:
Úè

	unknown_4:	è

	unknown_5:	Ú
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
K__inference_sequential_12_layer_call_and_return_conditional_losses_399717312
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
°?
Ô
while_body_39973302
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_25_matmul_readvariableop_resource_0:	`è
I
5while_lstm_cell_25_matmul_1_readvariableop_resource_0:
Úè
C
4while_lstm_cell_25_biasadd_readvariableop_resource_0:	è

while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_25_matmul_readvariableop_resource:	`è
G
3while_lstm_cell_25_matmul_1_readvariableop_resource:
Úè
A
2while_lstm_cell_25_biasadd_readvariableop_resource:	è
¢)while/lstm_cell_25/BiasAdd/ReadVariableOp¢(while/lstm_cell_25/MatMul/ReadVariableOp¢*while/lstm_cell_25/MatMul_1/ReadVariableOpÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ`   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÓ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemÉ
(while/lstm_cell_25/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_25_matmul_readvariableop_resource_0*
_output_shapes
:	`è
*
dtype02*
(while/lstm_cell_25/MatMul/ReadVariableOp×
while/lstm_cell_25/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_25/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè
2
while/lstm_cell_25/MatMulÐ
*while/lstm_cell_25/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_25_matmul_1_readvariableop_resource_0* 
_output_shapes
:
Úè
*
dtype02,
*while/lstm_cell_25/MatMul_1/ReadVariableOpÀ
while/lstm_cell_25/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_25/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè
2
while/lstm_cell_25/MatMul_1¸
while/lstm_cell_25/addAddV2#while/lstm_cell_25/MatMul:product:0%while/lstm_cell_25/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè
2
while/lstm_cell_25/addÈ
)while/lstm_cell_25/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_25_biasadd_readvariableop_resource_0*
_output_shapes	
:è
*
dtype02+
)while/lstm_cell_25/BiasAdd/ReadVariableOpÅ
while/lstm_cell_25/BiasAddBiasAddwhile/lstm_cell_25/add:z:01while/lstm_cell_25/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿè
2
while/lstm_cell_25/BiasAdd
"while/lstm_cell_25/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_25/split/split_dim
while/lstm_cell_25/splitSplit+while/lstm_cell_25/split/split_dim:output:0#while/lstm_cell_25/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿÚ:ÿÿÿÿÿÿÿÿÿÚ:ÿÿÿÿÿÿÿÿÿÚ:ÿÿÿÿÿÿÿÿÿÚ*
	num_split2
while/lstm_cell_25/split
while/lstm_cell_25/SigmoidSigmoid!while/lstm_cell_25/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
while/lstm_cell_25/Sigmoid
while/lstm_cell_25/Sigmoid_1Sigmoid!while/lstm_cell_25/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
while/lstm_cell_25/Sigmoid_1¡
while/lstm_cell_25/mulMul while/lstm_cell_25/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
while/lstm_cell_25/mul
while/lstm_cell_25/ReluRelu!while/lstm_cell_25/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
while/lstm_cell_25/Reluµ
while/lstm_cell_25/mul_1Mulwhile/lstm_cell_25/Sigmoid:y:0%while/lstm_cell_25/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
while/lstm_cell_25/mul_1ª
while/lstm_cell_25/add_1AddV2while/lstm_cell_25/mul:z:0while/lstm_cell_25/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
while/lstm_cell_25/add_1
while/lstm_cell_25/Sigmoid_2Sigmoid!while/lstm_cell_25/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
while/lstm_cell_25/Sigmoid_2
while/lstm_cell_25/Relu_1Reluwhile/lstm_cell_25/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
while/lstm_cell_25/Relu_1¹
while/lstm_cell_25/mul_2Mul while/lstm_cell_25/Sigmoid_2:y:0'while/lstm_cell_25/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
while/lstm_cell_25/mul_2à
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_25/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_25/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_25/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ2
while/Identity_5Þ

while/NoOpNoOp*^while/lstm_cell_25/BiasAdd/ReadVariableOp)^while/lstm_cell_25/MatMul/ReadVariableOp+^while/lstm_cell_25/MatMul_1/ReadVariableOp*"
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
2while_lstm_cell_25_biasadd_readvariableop_resource4while_lstm_cell_25_biasadd_readvariableop_resource_0"l
3while_lstm_cell_25_matmul_1_readvariableop_resource5while_lstm_cell_25_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_25_matmul_readvariableop_resource3while_lstm_cell_25_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿÚ:ÿÿÿÿÿÿÿÿÿÚ: : : : : 2V
)while/lstm_cell_25/BiasAdd/ReadVariableOp)while/lstm_cell_25/BiasAdd/ReadVariableOp2T
(while/lstm_cell_25/MatMul/ReadVariableOp(while/lstm_cell_25/MatMul/ReadVariableOp2X
*while/lstm_cell_25/MatMul_1/ReadVariableOp*while/lstm_cell_25/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
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

Ë
0__inference_sequential_12_layer_call_fn_39972539

inputs
unknown:	]
	unknown_0:	`
	unknown_1:	
	unknown_2:	`è

	unknown_3:
Úè

	unknown_4:	è

	unknown_5:	Ú
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
K__inference_sequential_12_layer_call_and_return_conditional_losses_399712622
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
ç[

E__inference_lstm_24_layer_call_and_return_conditional_losses_39973164

inputs>
+lstm_cell_24_matmul_readvariableop_resource:	]@
-lstm_cell_24_matmul_1_readvariableop_resource:	`;
,lstm_cell_24_biasadd_readvariableop_resource:	
identity¢#lstm_cell_24/BiasAdd/ReadVariableOp¢"lstm_cell_24/MatMul/ReadVariableOp¢$lstm_cell_24/MatMul_1/ReadVariableOp¢whileD
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
value	B :`2
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
value	B :`2
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
:ÿÿÿÿÿÿÿÿÿ`2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :`2
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
value	B :`2
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
:ÿÿÿÿÿÿÿÿÿ`2	
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
"lstm_cell_24/MatMul/ReadVariableOpReadVariableOp+lstm_cell_24_matmul_readvariableop_resource*
_output_shapes
:	]*
dtype02$
"lstm_cell_24/MatMul/ReadVariableOp­
lstm_cell_24/MatMulMatMulstrided_slice_2:output:0*lstm_cell_24/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_24/MatMul»
$lstm_cell_24/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_24_matmul_1_readvariableop_resource*
_output_shapes
:	`*
dtype02&
$lstm_cell_24/MatMul_1/ReadVariableOp©
lstm_cell_24/MatMul_1MatMulzeros:output:0,lstm_cell_24/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_24/MatMul_1 
lstm_cell_24/addAddV2lstm_cell_24/MatMul:product:0lstm_cell_24/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_24/add´
#lstm_cell_24/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_24_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02%
#lstm_cell_24/BiasAdd/ReadVariableOp­
lstm_cell_24/BiasAddBiasAddlstm_cell_24/add:z:0+lstm_cell_24/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_24/BiasAdd~
lstm_cell_24/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_24/split/split_dimó
lstm_cell_24/splitSplit%lstm_cell_24/split/split_dim:output:0lstm_cell_24/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ`:ÿÿÿÿÿÿÿÿÿ`:ÿÿÿÿÿÿÿÿÿ`:ÿÿÿÿÿÿÿÿÿ`*
	num_split2
lstm_cell_24/split
lstm_cell_24/SigmoidSigmoidlstm_cell_24/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
lstm_cell_24/Sigmoid
lstm_cell_24/Sigmoid_1Sigmoidlstm_cell_24/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
lstm_cell_24/Sigmoid_1
lstm_cell_24/mulMullstm_cell_24/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
lstm_cell_24/mul}
lstm_cell_24/ReluRelulstm_cell_24/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
lstm_cell_24/Relu
lstm_cell_24/mul_1Mullstm_cell_24/Sigmoid:y:0lstm_cell_24/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
lstm_cell_24/mul_1
lstm_cell_24/add_1AddV2lstm_cell_24/mul:z:0lstm_cell_24/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
lstm_cell_24/add_1
lstm_cell_24/Sigmoid_2Sigmoidlstm_cell_24/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
lstm_cell_24/Sigmoid_2|
lstm_cell_24/Relu_1Relulstm_cell_24/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
lstm_cell_24/Relu_1 
lstm_cell_24/mul_2Mullstm_cell_24/Sigmoid_2:y:0!lstm_cell_24/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2
lstm_cell_24/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ`   2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_24_matmul_readvariableop_resource-lstm_cell_24_matmul_1_readvariableop_resource,lstm_cell_24_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ`:ÿÿÿÿÿÿÿÿÿ`: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_39973080*
condR
while_cond_39973079*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ`:ÿÿÿÿÿÿÿÿÿ`: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ`   22
0TensorArrayV2Stack/TensorListStack/element_shapeè
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`*
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
:ÿÿÿÿÿÿÿÿÿ`*
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
:ÿÿÿÿÿÿÿÿÿ`2
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
:ÿÿÿÿÿÿÿÿÿ`2

IdentityÈ
NoOpNoOp$^lstm_cell_24/BiasAdd/ReadVariableOp#^lstm_cell_24/MatMul/ReadVariableOp%^lstm_cell_24/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ]: : : 2J
#lstm_cell_24/BiasAdd/ReadVariableOp#lstm_cell_24/BiasAdd/ReadVariableOp2H
"lstm_cell_24/MatMul/ReadVariableOp"lstm_cell_24/MatMul/ReadVariableOp2L
$lstm_cell_24/MatMul_1/ReadVariableOp$lstm_cell_24/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
 
_user_specified_nameinputs
Ô

í
lstm_25_while_cond_39972064,
(lstm_25_while_lstm_25_while_loop_counter2
.lstm_25_while_lstm_25_while_maximum_iterations
lstm_25_while_placeholder
lstm_25_while_placeholder_1
lstm_25_while_placeholder_2
lstm_25_while_placeholder_3.
*lstm_25_while_less_lstm_25_strided_slice_1F
Blstm_25_while_lstm_25_while_cond_39972064___redundant_placeholder0F
Blstm_25_while_lstm_25_while_cond_39972064___redundant_placeholder1F
Blstm_25_while_lstm_25_while_cond_39972064___redundant_placeholder2F
Blstm_25_while_lstm_25_while_cond_39972064___redundant_placeholder3
lstm_25_while_identity

lstm_25/while/LessLesslstm_25_while_placeholder*lstm_25_while_less_lstm_25_strided_slice_1*
T0*
_output_shapes
: 2
lstm_25/while/Lessu
lstm_25/while/IdentityIdentitylstm_25/while/Less:z:0*
T0
*
_output_shapes
: 2
lstm_25/while/Identity"9
lstm_25_while_identitylstm_25/while/Identity:output:0*(
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
¶
f
-__inference_dropout_24_layer_call_fn_39973235

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
:ÿÿÿÿÿÿÿÿÿ`* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_dropout_24_layer_call_and_return_conditional_losses_399715072
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ`22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
 
_user_specified_nameinputs
Ð

í
lstm_24_while_cond_39972243,
(lstm_24_while_lstm_24_while_loop_counter2
.lstm_24_while_lstm_24_while_maximum_iterations
lstm_24_while_placeholder
lstm_24_while_placeholder_1
lstm_24_while_placeholder_2
lstm_24_while_placeholder_3.
*lstm_24_while_less_lstm_24_strided_slice_1F
Blstm_24_while_lstm_24_while_cond_39972243___redundant_placeholder0F
Blstm_24_while_lstm_24_while_cond_39972243___redundant_placeholder1F
Blstm_24_while_lstm_24_while_cond_39972243___redundant_placeholder2F
Blstm_24_while_lstm_24_while_cond_39972243___redundant_placeholder3
lstm_24_while_identity

lstm_24/while/LessLesslstm_24_while_placeholder*lstm_24_while_less_lstm_24_strided_slice_1*
T0*
_output_shapes
: 2
lstm_24/while/Lessu
lstm_24/while/IdentityIdentitylstm_24/while/Less:z:0*
T0
*
_output_shapes
: 2
lstm_24/while/Identity"9
lstm_24_while_identitylstm_24/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿ`:ÿÿÿÿÿÿÿÿÿ`: ::::: 

_output_shapes
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
:ÿÿÿÿÿÿÿÿÿ`:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`:
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
lstm_24_input:
serving_default_lstm_24_input:0ÿÿÿÿÿÿÿÿÿ]@
dense_124
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:ï»
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
": 	Ú2dense_12/kernel
:2dense_12/bias
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
.:,	]2lstm_24/lstm_cell_24/kernel
8:6	`2%lstm_24/lstm_cell_24/recurrent_kernel
(:&2lstm_24/lstm_cell_24/bias
.:,	`è
2lstm_25/lstm_cell_25/kernel
9:7
Úè
2%lstm_25/lstm_cell_25/recurrent_kernel
(:&è
2lstm_25/lstm_cell_25/bias
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
':%	Ú2Adam/dense_12/kernel/m
 :2Adam/dense_12/bias/m
3:1	]2"Adam/lstm_24/lstm_cell_24/kernel/m
=:;	`2,Adam/lstm_24/lstm_cell_24/recurrent_kernel/m
-:+2 Adam/lstm_24/lstm_cell_24/bias/m
3:1	`è
2"Adam/lstm_25/lstm_cell_25/kernel/m
>:<
Úè
2,Adam/lstm_25/lstm_cell_25/recurrent_kernel/m
-:+è
2 Adam/lstm_25/lstm_cell_25/bias/m
':%	Ú2Adam/dense_12/kernel/v
 :2Adam/dense_12/bias/v
3:1	]2"Adam/lstm_24/lstm_cell_24/kernel/v
=:;	`2,Adam/lstm_24/lstm_cell_24/recurrent_kernel/v
-:+2 Adam/lstm_24/lstm_cell_24/bias/v
3:1	`è
2"Adam/lstm_25/lstm_cell_25/kernel/v
>:<
Úè
2,Adam/lstm_25/lstm_cell_25/recurrent_kernel/v
-:+è
2 Adam/lstm_25/lstm_cell_25/bias/v
ÔBÑ
#__inference__wrapped_model_39969627lstm_24_input"
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
K__inference_sequential_12_layer_call_and_return_conditional_losses_39972177
K__inference_sequential_12_layer_call_and_return_conditional_losses_39972518
K__inference_sequential_12_layer_call_and_return_conditional_losses_39971796
K__inference_sequential_12_layer_call_and_return_conditional_losses_39971821À
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
0__inference_sequential_12_layer_call_fn_39971281
0__inference_sequential_12_layer_call_fn_39972539
0__inference_sequential_12_layer_call_fn_39972560
0__inference_sequential_12_layer_call_fn_39971771À
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
E__inference_lstm_24_layer_call_and_return_conditional_losses_39972711
E__inference_lstm_24_layer_call_and_return_conditional_losses_39972862
E__inference_lstm_24_layer_call_and_return_conditional_losses_39973013
E__inference_lstm_24_layer_call_and_return_conditional_losses_39973164Õ
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
*__inference_lstm_24_layer_call_fn_39973175
*__inference_lstm_24_layer_call_fn_39973186
*__inference_lstm_24_layer_call_fn_39973197
*__inference_lstm_24_layer_call_fn_39973208Õ
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
H__inference_dropout_24_layer_call_and_return_conditional_losses_39973213
H__inference_dropout_24_layer_call_and_return_conditional_losses_39973225´
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
-__inference_dropout_24_layer_call_fn_39973230
-__inference_dropout_24_layer_call_fn_39973235´
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
E__inference_lstm_25_layer_call_and_return_conditional_losses_39973386
E__inference_lstm_25_layer_call_and_return_conditional_losses_39973537
E__inference_lstm_25_layer_call_and_return_conditional_losses_39973688
E__inference_lstm_25_layer_call_and_return_conditional_losses_39973839Õ
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
*__inference_lstm_25_layer_call_fn_39973850
*__inference_lstm_25_layer_call_fn_39973861
*__inference_lstm_25_layer_call_fn_39973872
*__inference_lstm_25_layer_call_fn_39973883Õ
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
H__inference_dropout_25_layer_call_and_return_conditional_losses_39973888
H__inference_dropout_25_layer_call_and_return_conditional_losses_39973900´
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
-__inference_dropout_25_layer_call_fn_39973905
-__inference_dropout_25_layer_call_fn_39973910´
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
F__inference_dense_12_layer_call_and_return_conditional_losses_39973941¢
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
+__inference_dense_12_layer_call_fn_39973950¢
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
&__inference_signature_wrapper_39971850lstm_24_input"
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
J__inference_lstm_cell_24_layer_call_and_return_conditional_losses_39973982
J__inference_lstm_cell_24_layer_call_and_return_conditional_losses_39974014¾
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
/__inference_lstm_cell_24_layer_call_fn_39974031
/__inference_lstm_cell_24_layer_call_fn_39974048¾
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
J__inference_lstm_cell_25_layer_call_and_return_conditional_losses_39974080
J__inference_lstm_cell_25_layer_call_and_return_conditional_losses_39974112¾
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
/__inference_lstm_cell_25_layer_call_fn_39974129
/__inference_lstm_cell_25_layer_call_fn_39974146¾
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
#__inference__wrapped_model_39969627+,-./0 !:¢7
0¢-
+(
lstm_24_inputÿÿÿÿÿÿÿÿÿ]
ª "7ª4
2
dense_12&#
dense_12ÿÿÿÿÿÿÿÿÿ¯
F__inference_dense_12_layer_call_and_return_conditional_losses_39973941e !4¢1
*¢'
%"
inputsÿÿÿÿÿÿÿÿÿÚ
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ
 
+__inference_dense_12_layer_call_fn_39973950X !4¢1
*¢'
%"
inputsÿÿÿÿÿÿÿÿÿÚ
ª "ÿÿÿÿÿÿÿÿÿ°
H__inference_dropout_24_layer_call_and_return_conditional_losses_39973213d7¢4
-¢*
$!
inputsÿÿÿÿÿÿÿÿÿ`
p 
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ`
 °
H__inference_dropout_24_layer_call_and_return_conditional_losses_39973225d7¢4
-¢*
$!
inputsÿÿÿÿÿÿÿÿÿ`
p
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ`
 
-__inference_dropout_24_layer_call_fn_39973230W7¢4
-¢*
$!
inputsÿÿÿÿÿÿÿÿÿ`
p 
ª "ÿÿÿÿÿÿÿÿÿ`
-__inference_dropout_24_layer_call_fn_39973235W7¢4
-¢*
$!
inputsÿÿÿÿÿÿÿÿÿ`
p
ª "ÿÿÿÿÿÿÿÿÿ`²
H__inference_dropout_25_layer_call_and_return_conditional_losses_39973888f8¢5
.¢+
%"
inputsÿÿÿÿÿÿÿÿÿÚ
p 
ª "*¢'
 
0ÿÿÿÿÿÿÿÿÿÚ
 ²
H__inference_dropout_25_layer_call_and_return_conditional_losses_39973900f8¢5
.¢+
%"
inputsÿÿÿÿÿÿÿÿÿÚ
p
ª "*¢'
 
0ÿÿÿÿÿÿÿÿÿÚ
 
-__inference_dropout_25_layer_call_fn_39973905Y8¢5
.¢+
%"
inputsÿÿÿÿÿÿÿÿÿÚ
p 
ª "ÿÿÿÿÿÿÿÿÿÚ
-__inference_dropout_25_layer_call_fn_39973910Y8¢5
.¢+
%"
inputsÿÿÿÿÿÿÿÿÿÚ
p
ª "ÿÿÿÿÿÿÿÿÿÚÔ
E__inference_lstm_24_layer_call_and_return_conditional_losses_39972711+,-O¢L
E¢B
41
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ]

 
p 

 
ª "2¢/
(%
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`
 Ô
E__inference_lstm_24_layer_call_and_return_conditional_losses_39972862+,-O¢L
E¢B
41
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ]

 
p

 
ª "2¢/
(%
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`
 º
E__inference_lstm_24_layer_call_and_return_conditional_losses_39973013q+,-?¢<
5¢2
$!
inputsÿÿÿÿÿÿÿÿÿ]

 
p 

 
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ`
 º
E__inference_lstm_24_layer_call_and_return_conditional_losses_39973164q+,-?¢<
5¢2
$!
inputsÿÿÿÿÿÿÿÿÿ]

 
p

 
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ`
 «
*__inference_lstm_24_layer_call_fn_39973175}+,-O¢L
E¢B
41
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ]

 
p 

 
ª "%"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`«
*__inference_lstm_24_layer_call_fn_39973186}+,-O¢L
E¢B
41
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ]

 
p

 
ª "%"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`
*__inference_lstm_24_layer_call_fn_39973197d+,-?¢<
5¢2
$!
inputsÿÿÿÿÿÿÿÿÿ]

 
p 

 
ª "ÿÿÿÿÿÿÿÿÿ`
*__inference_lstm_24_layer_call_fn_39973208d+,-?¢<
5¢2
$!
inputsÿÿÿÿÿÿÿÿÿ]

 
p

 
ª "ÿÿÿÿÿÿÿÿÿ`Õ
E__inference_lstm_25_layer_call_and_return_conditional_losses_39973386./0O¢L
E¢B
41
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`

 
p 

 
ª "3¢0
)&
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÚ
 Õ
E__inference_lstm_25_layer_call_and_return_conditional_losses_39973537./0O¢L
E¢B
41
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`

 
p

 
ª "3¢0
)&
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÚ
 »
E__inference_lstm_25_layer_call_and_return_conditional_losses_39973688r./0?¢<
5¢2
$!
inputsÿÿÿÿÿÿÿÿÿ`

 
p 

 
ª "*¢'
 
0ÿÿÿÿÿÿÿÿÿÚ
 »
E__inference_lstm_25_layer_call_and_return_conditional_losses_39973839r./0?¢<
5¢2
$!
inputsÿÿÿÿÿÿÿÿÿ`

 
p

 
ª "*¢'
 
0ÿÿÿÿÿÿÿÿÿÚ
 ¬
*__inference_lstm_25_layer_call_fn_39973850~./0O¢L
E¢B
41
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`

 
p 

 
ª "&#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÚ¬
*__inference_lstm_25_layer_call_fn_39973861~./0O¢L
E¢B
41
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`

 
p

 
ª "&#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÚ
*__inference_lstm_25_layer_call_fn_39973872e./0?¢<
5¢2
$!
inputsÿÿÿÿÿÿÿÿÿ`

 
p 

 
ª "ÿÿÿÿÿÿÿÿÿÚ
*__inference_lstm_25_layer_call_fn_39973883e./0?¢<
5¢2
$!
inputsÿÿÿÿÿÿÿÿÿ`

 
p

 
ª "ÿÿÿÿÿÿÿÿÿÚÌ
J__inference_lstm_cell_24_layer_call_and_return_conditional_losses_39973982ý+,-¢}
v¢s
 
inputsÿÿÿÿÿÿÿÿÿ]
K¢H
"
states/0ÿÿÿÿÿÿÿÿÿ`
"
states/1ÿÿÿÿÿÿÿÿÿ`
p 
ª "s¢p
i¢f

0/0ÿÿÿÿÿÿÿÿÿ`
EB

0/1/0ÿÿÿÿÿÿÿÿÿ`

0/1/1ÿÿÿÿÿÿÿÿÿ`
 Ì
J__inference_lstm_cell_24_layer_call_and_return_conditional_losses_39974014ý+,-¢}
v¢s
 
inputsÿÿÿÿÿÿÿÿÿ]
K¢H
"
states/0ÿÿÿÿÿÿÿÿÿ`
"
states/1ÿÿÿÿÿÿÿÿÿ`
p
ª "s¢p
i¢f

0/0ÿÿÿÿÿÿÿÿÿ`
EB

0/1/0ÿÿÿÿÿÿÿÿÿ`

0/1/1ÿÿÿÿÿÿÿÿÿ`
 ¡
/__inference_lstm_cell_24_layer_call_fn_39974031í+,-¢}
v¢s
 
inputsÿÿÿÿÿÿÿÿÿ]
K¢H
"
states/0ÿÿÿÿÿÿÿÿÿ`
"
states/1ÿÿÿÿÿÿÿÿÿ`
p 
ª "c¢`

0ÿÿÿÿÿÿÿÿÿ`
A>

1/0ÿÿÿÿÿÿÿÿÿ`

1/1ÿÿÿÿÿÿÿÿÿ`¡
/__inference_lstm_cell_24_layer_call_fn_39974048í+,-¢}
v¢s
 
inputsÿÿÿÿÿÿÿÿÿ]
K¢H
"
states/0ÿÿÿÿÿÿÿÿÿ`
"
states/1ÿÿÿÿÿÿÿÿÿ`
p
ª "c¢`

0ÿÿÿÿÿÿÿÿÿ`
A>

1/0ÿÿÿÿÿÿÿÿÿ`

1/1ÿÿÿÿÿÿÿÿÿ`Ñ
J__inference_lstm_cell_25_layer_call_and_return_conditional_losses_39974080./0¢
x¢u
 
inputsÿÿÿÿÿÿÿÿÿ`
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
J__inference_lstm_cell_25_layer_call_and_return_conditional_losses_39974112./0¢
x¢u
 
inputsÿÿÿÿÿÿÿÿÿ`
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
/__inference_lstm_cell_25_layer_call_fn_39974129ò./0¢
x¢u
 
inputsÿÿÿÿÿÿÿÿÿ`
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
/__inference_lstm_cell_25_layer_call_fn_39974146ò./0¢
x¢u
 
inputsÿÿÿÿÿÿÿÿÿ`
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
1/1ÿÿÿÿÿÿÿÿÿÚÈ
K__inference_sequential_12_layer_call_and_return_conditional_losses_39971796y+,-./0 !B¢?
8¢5
+(
lstm_24_inputÿÿÿÿÿÿÿÿÿ]
p 

 
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ
 È
K__inference_sequential_12_layer_call_and_return_conditional_losses_39971821y+,-./0 !B¢?
8¢5
+(
lstm_24_inputÿÿÿÿÿÿÿÿÿ]
p

 
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ
 Á
K__inference_sequential_12_layer_call_and_return_conditional_losses_39972177r+,-./0 !;¢8
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
K__inference_sequential_12_layer_call_and_return_conditional_losses_39972518r+,-./0 !;¢8
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
0__inference_sequential_12_layer_call_fn_39971281l+,-./0 !B¢?
8¢5
+(
lstm_24_inputÿÿÿÿÿÿÿÿÿ]
p 

 
ª "ÿÿÿÿÿÿÿÿÿ 
0__inference_sequential_12_layer_call_fn_39971771l+,-./0 !B¢?
8¢5
+(
lstm_24_inputÿÿÿÿÿÿÿÿÿ]
p

 
ª "ÿÿÿÿÿÿÿÿÿ
0__inference_sequential_12_layer_call_fn_39972539e+,-./0 !;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿ]
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
0__inference_sequential_12_layer_call_fn_39972560e+,-./0 !;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿ]
p

 
ª "ÿÿÿÿÿÿÿÿÿ»
&__inference_signature_wrapper_39971850+,-./0 !K¢H
¢ 
Aª>
<
lstm_24_input+(
lstm_24_inputÿÿÿÿÿÿÿÿÿ]"7ª4
2
dense_12&#
dense_12ÿÿÿÿÿÿÿÿÿ