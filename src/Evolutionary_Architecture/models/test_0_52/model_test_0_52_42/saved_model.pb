¬ò'
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
"serve*2.6.02v2.6.0-rc2-32-g919f693420e8&
z
dense_12/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:8* 
shared_namedense_12/kernel
s
#dense_12/kernel/Read/ReadVariableOpReadVariableOpdense_12/kernel*
_output_shapes

:8*
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
shape:	]Ð*,
shared_namelstm_24/lstm_cell_24/kernel

/lstm_24/lstm_cell_24/kernel/Read/ReadVariableOpReadVariableOplstm_24/lstm_cell_24/kernel*
_output_shapes
:	]Ð*
dtype0
¨
%lstm_24/lstm_cell_24/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ôÐ*6
shared_name'%lstm_24/lstm_cell_24/recurrent_kernel
¡
9lstm_24/lstm_cell_24/recurrent_kernel/Read/ReadVariableOpReadVariableOp%lstm_24/lstm_cell_24/recurrent_kernel* 
_output_shapes
:
ôÐ*
dtype0

lstm_24/lstm_cell_24/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:Ð**
shared_namelstm_24/lstm_cell_24/bias

-lstm_24/lstm_cell_24/bias/Read/ReadVariableOpReadVariableOplstm_24/lstm_cell_24/bias*
_output_shapes	
:Ð*
dtype0

lstm_25/lstm_cell_25/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ôà*,
shared_namelstm_25/lstm_cell_25/kernel

/lstm_25/lstm_cell_25/kernel/Read/ReadVariableOpReadVariableOplstm_25/lstm_cell_25/kernel* 
_output_shapes
:
ôà*
dtype0
§
%lstm_25/lstm_cell_25/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	8à*6
shared_name'%lstm_25/lstm_cell_25/recurrent_kernel
 
9lstm_25/lstm_cell_25/recurrent_kernel/Read/ReadVariableOpReadVariableOp%lstm_25/lstm_cell_25/recurrent_kernel*
_output_shapes
:	8à*
dtype0

lstm_25/lstm_cell_25/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:à**
shared_namelstm_25/lstm_cell_25/bias

-lstm_25/lstm_cell_25/bias/Read/ReadVariableOpReadVariableOplstm_25/lstm_cell_25/bias*
_output_shapes	
:à*
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

Adam/dense_12/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:8*'
shared_nameAdam/dense_12/kernel/m

*Adam/dense_12/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_12/kernel/m*
_output_shapes

:8*
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
shape:	]Ð*3
shared_name$"Adam/lstm_24/lstm_cell_24/kernel/m

6Adam/lstm_24/lstm_cell_24/kernel/m/Read/ReadVariableOpReadVariableOp"Adam/lstm_24/lstm_cell_24/kernel/m*
_output_shapes
:	]Ð*
dtype0
¶
,Adam/lstm_24/lstm_cell_24/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ôÐ*=
shared_name.,Adam/lstm_24/lstm_cell_24/recurrent_kernel/m
¯
@Adam/lstm_24/lstm_cell_24/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp,Adam/lstm_24/lstm_cell_24/recurrent_kernel/m* 
_output_shapes
:
ôÐ*
dtype0

 Adam/lstm_24/lstm_cell_24/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:Ð*1
shared_name" Adam/lstm_24/lstm_cell_24/bias/m

4Adam/lstm_24/lstm_cell_24/bias/m/Read/ReadVariableOpReadVariableOp Adam/lstm_24/lstm_cell_24/bias/m*
_output_shapes	
:Ð*
dtype0
¢
"Adam/lstm_25/lstm_cell_25/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ôà*3
shared_name$"Adam/lstm_25/lstm_cell_25/kernel/m

6Adam/lstm_25/lstm_cell_25/kernel/m/Read/ReadVariableOpReadVariableOp"Adam/lstm_25/lstm_cell_25/kernel/m* 
_output_shapes
:
ôà*
dtype0
µ
,Adam/lstm_25/lstm_cell_25/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	8à*=
shared_name.,Adam/lstm_25/lstm_cell_25/recurrent_kernel/m
®
@Adam/lstm_25/lstm_cell_25/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp,Adam/lstm_25/lstm_cell_25/recurrent_kernel/m*
_output_shapes
:	8à*
dtype0

 Adam/lstm_25/lstm_cell_25/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:à*1
shared_name" Adam/lstm_25/lstm_cell_25/bias/m

4Adam/lstm_25/lstm_cell_25/bias/m/Read/ReadVariableOpReadVariableOp Adam/lstm_25/lstm_cell_25/bias/m*
_output_shapes	
:à*
dtype0

Adam/dense_12/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:8*'
shared_nameAdam/dense_12/kernel/v

*Adam/dense_12/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_12/kernel/v*
_output_shapes

:8*
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
shape:	]Ð*3
shared_name$"Adam/lstm_24/lstm_cell_24/kernel/v

6Adam/lstm_24/lstm_cell_24/kernel/v/Read/ReadVariableOpReadVariableOp"Adam/lstm_24/lstm_cell_24/kernel/v*
_output_shapes
:	]Ð*
dtype0
¶
,Adam/lstm_24/lstm_cell_24/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ôÐ*=
shared_name.,Adam/lstm_24/lstm_cell_24/recurrent_kernel/v
¯
@Adam/lstm_24/lstm_cell_24/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp,Adam/lstm_24/lstm_cell_24/recurrent_kernel/v* 
_output_shapes
:
ôÐ*
dtype0

 Adam/lstm_24/lstm_cell_24/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:Ð*1
shared_name" Adam/lstm_24/lstm_cell_24/bias/v

4Adam/lstm_24/lstm_cell_24/bias/v/Read/ReadVariableOpReadVariableOp Adam/lstm_24/lstm_cell_24/bias/v*
_output_shapes	
:Ð*
dtype0
¢
"Adam/lstm_25/lstm_cell_25/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ôà*3
shared_name$"Adam/lstm_25/lstm_cell_25/kernel/v

6Adam/lstm_25/lstm_cell_25/kernel/v/Read/ReadVariableOpReadVariableOp"Adam/lstm_25/lstm_cell_25/kernel/v* 
_output_shapes
:
ôà*
dtype0
µ
,Adam/lstm_25/lstm_cell_25/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	8à*=
shared_name.,Adam/lstm_25/lstm_cell_25/recurrent_kernel/v
®
@Adam/lstm_25/lstm_cell_25/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp,Adam/lstm_25/lstm_cell_25/recurrent_kernel/v*
_output_shapes
:	8à*
dtype0

 Adam/lstm_25/lstm_cell_25/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:à*1
shared_name" Adam/lstm_25/lstm_cell_25/bias/v

4Adam/lstm_25/lstm_cell_25/bias/v/Read/ReadVariableOpReadVariableOp Adam/lstm_25/lstm_cell_25/bias/v*
_output_shapes	
:à*
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
 
8
+0
,1
-2
.3
/4
05
 6
!7
­
1layer_metrics
trainable_variables

2layers
3layer_regularization_losses
4metrics
regularization_losses
5non_trainable_variables
		variables
 

6
state_size

+kernel
,recurrent_kernel
-bias
7trainable_variables
8regularization_losses
9	variables
:	keras_api
 

+0
,1
-2
 

+0
,1
-2
¹
;layer_metrics

<states
trainable_variables

=layers
>layer_regularization_losses
?metrics
regularization_losses
@non_trainable_variables
	variables
 
 
 
­
Alayer_metrics
trainable_variables

Blayers
Clayer_regularization_losses
Dmetrics
regularization_losses
Enon_trainable_variables
	variables

F
state_size

.kernel
/recurrent_kernel
0bias
Gtrainable_variables
Hregularization_losses
I	variables
J	keras_api
 

.0
/1
02
 

.0
/1
02
¹
Klayer_metrics

Lstates
trainable_variables

Mlayers
Nlayer_regularization_losses
Ometrics
regularization_losses
Pnon_trainable_variables
	variables
 
 
 
­
Qlayer_metrics
trainable_variables

Rlayers
Slayer_regularization_losses
Tmetrics
regularization_losses
Unon_trainable_variables
	variables
[Y
VARIABLE_VALUEdense_12/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_12/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

 0
!1
 

 0
!1
­
Vlayer_metrics
"trainable_variables

Wlayers
Xlayer_regularization_losses
Ymetrics
#regularization_losses
Znon_trainable_variables
$	variables
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
 

+0
,1
-2
 

+0
,1
-2
­
]layer_metrics
7trainable_variables

^layers
_layer_regularization_losses
`metrics
8regularization_losses
anon_trainable_variables
9	variables
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
 

.0
/1
02
­
blayer_metrics
Gtrainable_variables

clayers
dlayer_regularization_losses
emetrics
Hregularization_losses
fnon_trainable_variables
I	variables
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
&__inference_signature_wrapper_40078066
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
!__inference__traced_save_40080484
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
$__inference__traced_restore_40080593èø$
à
º
*__inference_lstm_25_layer_call_fn_40080077
inputs_0
unknown:
ôà
	unknown_0:	8à
	unknown_1:	à
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ8*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_lstm_25_layer_call_and_return_conditional_losses_400768412
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ82

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿô: : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿô
"
_user_specified_name
inputs/0
´ù

K__inference_sequential_12_layer_call_and_return_conditional_losses_40078393

inputsF
3lstm_24_lstm_cell_24_matmul_readvariableop_resource:	]ÐI
5lstm_24_lstm_cell_24_matmul_1_readvariableop_resource:
ôÐC
4lstm_24_lstm_cell_24_biasadd_readvariableop_resource:	ÐG
3lstm_25_lstm_cell_25_matmul_readvariableop_resource:
ôàH
5lstm_25_lstm_cell_25_matmul_1_readvariableop_resource:	8àC
4lstm_25_lstm_cell_25_biasadd_readvariableop_resource:	à<
*dense_12_tensordot_readvariableop_resource:86
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
lstm_24/strided_slicem
lstm_24/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :ô2
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
lstm_24/zeros/Lesss
lstm_24/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :ô2
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
lstm_24/zeros/Const
lstm_24/zerosFilllstm_24/zeros/packed:output:0lstm_24/zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2
lstm_24/zerosq
lstm_24/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :ô2
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
lstm_24/zeros_1/Lessw
lstm_24/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :ô2
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
lstm_24/zeros_1/Const
lstm_24/zeros_1Filllstm_24/zeros_1/packed:output:0lstm_24/zeros_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2
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
:	]Ð*
dtype02,
*lstm_24/lstm_cell_24/MatMul/ReadVariableOpÍ
lstm_24/lstm_cell_24/MatMulMatMul lstm_24/strided_slice_2:output:02lstm_24/lstm_cell_24/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
lstm_24/lstm_cell_24/MatMulÔ
,lstm_24/lstm_cell_24/MatMul_1/ReadVariableOpReadVariableOp5lstm_24_lstm_cell_24_matmul_1_readvariableop_resource* 
_output_shapes
:
ôÐ*
dtype02.
,lstm_24/lstm_cell_24/MatMul_1/ReadVariableOpÉ
lstm_24/lstm_cell_24/MatMul_1MatMullstm_24/zeros:output:04lstm_24/lstm_cell_24/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
lstm_24/lstm_cell_24/MatMul_1À
lstm_24/lstm_cell_24/addAddV2%lstm_24/lstm_cell_24/MatMul:product:0'lstm_24/lstm_cell_24/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
lstm_24/lstm_cell_24/addÌ
+lstm_24/lstm_cell_24/BiasAdd/ReadVariableOpReadVariableOp4lstm_24_lstm_cell_24_biasadd_readvariableop_resource*
_output_shapes	
:Ð*
dtype02-
+lstm_24/lstm_cell_24/BiasAdd/ReadVariableOpÍ
lstm_24/lstm_cell_24/BiasAddBiasAddlstm_24/lstm_cell_24/add:z:03lstm_24/lstm_cell_24/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
lstm_24/lstm_cell_24/BiasAdd
$lstm_24/lstm_cell_24/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2&
$lstm_24/lstm_cell_24/split/split_dim
lstm_24/lstm_cell_24/splitSplit-lstm_24/lstm_cell_24/split/split_dim:output:0%lstm_24/lstm_cell_24/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿô:ÿÿÿÿÿÿÿÿÿô:ÿÿÿÿÿÿÿÿÿô:ÿÿÿÿÿÿÿÿÿô*
	num_split2
lstm_24/lstm_cell_24/split
lstm_24/lstm_cell_24/SigmoidSigmoid#lstm_24/lstm_cell_24/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2
lstm_24/lstm_cell_24/Sigmoid£
lstm_24/lstm_cell_24/Sigmoid_1Sigmoid#lstm_24/lstm_cell_24/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2 
lstm_24/lstm_cell_24/Sigmoid_1¬
lstm_24/lstm_cell_24/mulMul"lstm_24/lstm_cell_24/Sigmoid_1:y:0lstm_24/zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2
lstm_24/lstm_cell_24/mul
lstm_24/lstm_cell_24/ReluRelu#lstm_24/lstm_cell_24/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2
lstm_24/lstm_cell_24/Relu½
lstm_24/lstm_cell_24/mul_1Mul lstm_24/lstm_cell_24/Sigmoid:y:0'lstm_24/lstm_cell_24/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2
lstm_24/lstm_cell_24/mul_1²
lstm_24/lstm_cell_24/add_1AddV2lstm_24/lstm_cell_24/mul:z:0lstm_24/lstm_cell_24/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2
lstm_24/lstm_cell_24/add_1£
lstm_24/lstm_cell_24/Sigmoid_2Sigmoid#lstm_24/lstm_cell_24/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2 
lstm_24/lstm_cell_24/Sigmoid_2
lstm_24/lstm_cell_24/Relu_1Relulstm_24/lstm_cell_24/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2
lstm_24/lstm_cell_24/Relu_1Á
lstm_24/lstm_cell_24/mul_2Mul"lstm_24/lstm_cell_24/Sigmoid_2:y:0)lstm_24/lstm_cell_24/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2
lstm_24/lstm_cell_24/mul_2
%lstm_24/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿô  2'
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
lstm_24/while/loop_counter
lstm_24/whileWhile#lstm_24/while/loop_counter:output:0)lstm_24/while/maximum_iterations:output:0lstm_24/time:output:0 lstm_24/TensorArrayV2_1:handle:0lstm_24/zeros:output:0lstm_24/zeros_1:output:0 lstm_24/strided_slice_1:output:0?lstm_24/TensorArrayUnstack/TensorListFromTensor:output_handle:03lstm_24_lstm_cell_24_matmul_readvariableop_resource5lstm_24_lstm_cell_24_matmul_1_readvariableop_resource4lstm_24_lstm_cell_24_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿô:ÿÿÿÿÿÿÿÿÿô: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *'
bodyR
lstm_24_while_body_40078133*'
condR
lstm_24_while_cond_40078132*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿô:ÿÿÿÿÿÿÿÿÿô: : : : : *
parallel_iterations 2
lstm_24/whileÅ
8lstm_24/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿô  2:
8lstm_24/TensorArrayV2Stack/TensorListStack/element_shape
*lstm_24/TensorArrayV2Stack/TensorListStackTensorListStacklstm_24/while:output:3Alstm_24/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿô*
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
lstm_24/strided_slice_3/stack_2Ë
lstm_24/strided_slice_3StridedSlice3lstm_24/TensorArrayV2Stack/TensorListStack:tensor:0&lstm_24/strided_slice_3/stack:output:0(lstm_24/strided_slice_3/stack_1:output:0(lstm_24/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô*
shrink_axis_mask2
lstm_24/strided_slice_3
lstm_24/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_24/transpose_1/permÆ
lstm_24/transpose_1	Transpose3lstm_24/TensorArrayV2Stack/TensorListStack:tensor:0!lstm_24/transpose_1/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2
lstm_24/transpose_1v
lstm_24/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_24/runtime
dropout_24/IdentityIdentitylstm_24/transpose_1:y:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2
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
lstm_25/strided_slicel
lstm_25/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :82
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
lstm_25/zeros/Lessr
lstm_25/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :82
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
lstm_25/zeros/Const
lstm_25/zerosFilllstm_25/zeros/packed:output:0lstm_25/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82
lstm_25/zerosp
lstm_25/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :82
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
lstm_25/zeros_1/Lessv
lstm_25/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :82
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
lstm_25/zeros_1/Const
lstm_25/zeros_1Filllstm_25/zeros_1/packed:output:0lstm_25/zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82
lstm_25/zeros_1
lstm_25/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_25/transpose/perm©
lstm_25/transpose	Transposedropout_24/Identity:output:0lstm_25/transpose/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2
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
valueB"ÿÿÿÿô  2?
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
lstm_25/strided_slice_2/stack_2­
lstm_25/strided_slice_2StridedSlicelstm_25/transpose:y:0&lstm_25/strided_slice_2/stack:output:0(lstm_25/strided_slice_2/stack_1:output:0(lstm_25/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô*
shrink_axis_mask2
lstm_25/strided_slice_2Î
*lstm_25/lstm_cell_25/MatMul/ReadVariableOpReadVariableOp3lstm_25_lstm_cell_25_matmul_readvariableop_resource* 
_output_shapes
:
ôà*
dtype02,
*lstm_25/lstm_cell_25/MatMul/ReadVariableOpÍ
lstm_25/lstm_cell_25/MatMulMatMul lstm_25/strided_slice_2:output:02lstm_25/lstm_cell_25/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà2
lstm_25/lstm_cell_25/MatMulÓ
,lstm_25/lstm_cell_25/MatMul_1/ReadVariableOpReadVariableOp5lstm_25_lstm_cell_25_matmul_1_readvariableop_resource*
_output_shapes
:	8à*
dtype02.
,lstm_25/lstm_cell_25/MatMul_1/ReadVariableOpÉ
lstm_25/lstm_cell_25/MatMul_1MatMullstm_25/zeros:output:04lstm_25/lstm_cell_25/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà2
lstm_25/lstm_cell_25/MatMul_1À
lstm_25/lstm_cell_25/addAddV2%lstm_25/lstm_cell_25/MatMul:product:0'lstm_25/lstm_cell_25/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà2
lstm_25/lstm_cell_25/addÌ
+lstm_25/lstm_cell_25/BiasAdd/ReadVariableOpReadVariableOp4lstm_25_lstm_cell_25_biasadd_readvariableop_resource*
_output_shapes	
:à*
dtype02-
+lstm_25/lstm_cell_25/BiasAdd/ReadVariableOpÍ
lstm_25/lstm_cell_25/BiasAddBiasAddlstm_25/lstm_cell_25/add:z:03lstm_25/lstm_cell_25/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà2
lstm_25/lstm_cell_25/BiasAdd
$lstm_25/lstm_cell_25/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2&
$lstm_25/lstm_cell_25/split/split_dim
lstm_25/lstm_cell_25/splitSplit-lstm_25/lstm_cell_25/split/split_dim:output:0%lstm_25/lstm_cell_25/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ8:ÿÿÿÿÿÿÿÿÿ8:ÿÿÿÿÿÿÿÿÿ8:ÿÿÿÿÿÿÿÿÿ8*
	num_split2
lstm_25/lstm_cell_25/split
lstm_25/lstm_cell_25/SigmoidSigmoid#lstm_25/lstm_cell_25/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82
lstm_25/lstm_cell_25/Sigmoid¢
lstm_25/lstm_cell_25/Sigmoid_1Sigmoid#lstm_25/lstm_cell_25/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82 
lstm_25/lstm_cell_25/Sigmoid_1«
lstm_25/lstm_cell_25/mulMul"lstm_25/lstm_cell_25/Sigmoid_1:y:0lstm_25/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82
lstm_25/lstm_cell_25/mul
lstm_25/lstm_cell_25/ReluRelu#lstm_25/lstm_cell_25/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82
lstm_25/lstm_cell_25/Relu¼
lstm_25/lstm_cell_25/mul_1Mul lstm_25/lstm_cell_25/Sigmoid:y:0'lstm_25/lstm_cell_25/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82
lstm_25/lstm_cell_25/mul_1±
lstm_25/lstm_cell_25/add_1AddV2lstm_25/lstm_cell_25/mul:z:0lstm_25/lstm_cell_25/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82
lstm_25/lstm_cell_25/add_1¢
lstm_25/lstm_cell_25/Sigmoid_2Sigmoid#lstm_25/lstm_cell_25/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82 
lstm_25/lstm_cell_25/Sigmoid_2
lstm_25/lstm_cell_25/Relu_1Relulstm_25/lstm_cell_25/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82
lstm_25/lstm_cell_25/Relu_1À
lstm_25/lstm_cell_25/mul_2Mul"lstm_25/lstm_cell_25/Sigmoid_2:y:0)lstm_25/lstm_cell_25/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82
lstm_25/lstm_cell_25/mul_2
%lstm_25/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ8   2'
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
lstm_25/while/loop_counter
lstm_25/whileWhile#lstm_25/while/loop_counter:output:0)lstm_25/while/maximum_iterations:output:0lstm_25/time:output:0 lstm_25/TensorArrayV2_1:handle:0lstm_25/zeros:output:0lstm_25/zeros_1:output:0 lstm_25/strided_slice_1:output:0?lstm_25/TensorArrayUnstack/TensorListFromTensor:output_handle:03lstm_25_lstm_cell_25_matmul_readvariableop_resource5lstm_25_lstm_cell_25_matmul_1_readvariableop_resource4lstm_25_lstm_cell_25_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ8:ÿÿÿÿÿÿÿÿÿ8: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *'
bodyR
lstm_25_while_body_40078281*'
condR
lstm_25_while_cond_40078280*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ8:ÿÿÿÿÿÿÿÿÿ8: : : : : *
parallel_iterations 2
lstm_25/whileÅ
8lstm_25/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ8   2:
8lstm_25/TensorArrayV2Stack/TensorListStack/element_shape
*lstm_25/TensorArrayV2Stack/TensorListStackTensorListStacklstm_25/while:output:3Alstm_25/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ8*
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
lstm_25/strided_slice_3/stack_2Ê
lstm_25/strided_slice_3StridedSlice3lstm_25/TensorArrayV2Stack/TensorListStack:tensor:0&lstm_25/strided_slice_3/stack:output:0(lstm_25/strided_slice_3/stack_1:output:0(lstm_25/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ8*
shrink_axis_mask2
lstm_25/strided_slice_3
lstm_25/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_25/transpose_1/permÅ
lstm_25/transpose_1	Transpose3lstm_25/TensorArrayV2Stack/TensorListStack:tensor:0!lstm_25/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82
lstm_25/transpose_1v
lstm_25/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_25/runtime
dropout_25/IdentityIdentitylstm_25/transpose_1:y:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82
dropout_25/Identity±
!dense_12/Tensordot/ReadVariableOpReadVariableOp*dense_12_tensordot_readvariableop_resource*
_output_shapes

:8*
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
dense_12/Tensordot/stackÁ
dense_12/Tensordot/transpose	Transposedropout_25/Identity:output:0"dense_12/Tensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82
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
Î

K__inference_sequential_12_layer_call_and_return_conditional_losses_40078734

inputsF
3lstm_24_lstm_cell_24_matmul_readvariableop_resource:	]ÐI
5lstm_24_lstm_cell_24_matmul_1_readvariableop_resource:
ôÐC
4lstm_24_lstm_cell_24_biasadd_readvariableop_resource:	ÐG
3lstm_25_lstm_cell_25_matmul_readvariableop_resource:
ôàH
5lstm_25_lstm_cell_25_matmul_1_readvariableop_resource:	8àC
4lstm_25_lstm_cell_25_biasadd_readvariableop_resource:	à<
*dense_12_tensordot_readvariableop_resource:86
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
lstm_24/strided_slicem
lstm_24/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :ô2
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
lstm_24/zeros/Lesss
lstm_24/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :ô2
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
lstm_24/zeros/Const
lstm_24/zerosFilllstm_24/zeros/packed:output:0lstm_24/zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2
lstm_24/zerosq
lstm_24/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :ô2
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
lstm_24/zeros_1/Lessw
lstm_24/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :ô2
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
lstm_24/zeros_1/Const
lstm_24/zeros_1Filllstm_24/zeros_1/packed:output:0lstm_24/zeros_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2
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
:	]Ð*
dtype02,
*lstm_24/lstm_cell_24/MatMul/ReadVariableOpÍ
lstm_24/lstm_cell_24/MatMulMatMul lstm_24/strided_slice_2:output:02lstm_24/lstm_cell_24/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
lstm_24/lstm_cell_24/MatMulÔ
,lstm_24/lstm_cell_24/MatMul_1/ReadVariableOpReadVariableOp5lstm_24_lstm_cell_24_matmul_1_readvariableop_resource* 
_output_shapes
:
ôÐ*
dtype02.
,lstm_24/lstm_cell_24/MatMul_1/ReadVariableOpÉ
lstm_24/lstm_cell_24/MatMul_1MatMullstm_24/zeros:output:04lstm_24/lstm_cell_24/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
lstm_24/lstm_cell_24/MatMul_1À
lstm_24/lstm_cell_24/addAddV2%lstm_24/lstm_cell_24/MatMul:product:0'lstm_24/lstm_cell_24/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
lstm_24/lstm_cell_24/addÌ
+lstm_24/lstm_cell_24/BiasAdd/ReadVariableOpReadVariableOp4lstm_24_lstm_cell_24_biasadd_readvariableop_resource*
_output_shapes	
:Ð*
dtype02-
+lstm_24/lstm_cell_24/BiasAdd/ReadVariableOpÍ
lstm_24/lstm_cell_24/BiasAddBiasAddlstm_24/lstm_cell_24/add:z:03lstm_24/lstm_cell_24/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
lstm_24/lstm_cell_24/BiasAdd
$lstm_24/lstm_cell_24/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2&
$lstm_24/lstm_cell_24/split/split_dim
lstm_24/lstm_cell_24/splitSplit-lstm_24/lstm_cell_24/split/split_dim:output:0%lstm_24/lstm_cell_24/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿô:ÿÿÿÿÿÿÿÿÿô:ÿÿÿÿÿÿÿÿÿô:ÿÿÿÿÿÿÿÿÿô*
	num_split2
lstm_24/lstm_cell_24/split
lstm_24/lstm_cell_24/SigmoidSigmoid#lstm_24/lstm_cell_24/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2
lstm_24/lstm_cell_24/Sigmoid£
lstm_24/lstm_cell_24/Sigmoid_1Sigmoid#lstm_24/lstm_cell_24/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2 
lstm_24/lstm_cell_24/Sigmoid_1¬
lstm_24/lstm_cell_24/mulMul"lstm_24/lstm_cell_24/Sigmoid_1:y:0lstm_24/zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2
lstm_24/lstm_cell_24/mul
lstm_24/lstm_cell_24/ReluRelu#lstm_24/lstm_cell_24/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2
lstm_24/lstm_cell_24/Relu½
lstm_24/lstm_cell_24/mul_1Mul lstm_24/lstm_cell_24/Sigmoid:y:0'lstm_24/lstm_cell_24/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2
lstm_24/lstm_cell_24/mul_1²
lstm_24/lstm_cell_24/add_1AddV2lstm_24/lstm_cell_24/mul:z:0lstm_24/lstm_cell_24/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2
lstm_24/lstm_cell_24/add_1£
lstm_24/lstm_cell_24/Sigmoid_2Sigmoid#lstm_24/lstm_cell_24/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2 
lstm_24/lstm_cell_24/Sigmoid_2
lstm_24/lstm_cell_24/Relu_1Relulstm_24/lstm_cell_24/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2
lstm_24/lstm_cell_24/Relu_1Á
lstm_24/lstm_cell_24/mul_2Mul"lstm_24/lstm_cell_24/Sigmoid_2:y:0)lstm_24/lstm_cell_24/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2
lstm_24/lstm_cell_24/mul_2
%lstm_24/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿô  2'
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
lstm_24/while/loop_counter
lstm_24/whileWhile#lstm_24/while/loop_counter:output:0)lstm_24/while/maximum_iterations:output:0lstm_24/time:output:0 lstm_24/TensorArrayV2_1:handle:0lstm_24/zeros:output:0lstm_24/zeros_1:output:0 lstm_24/strided_slice_1:output:0?lstm_24/TensorArrayUnstack/TensorListFromTensor:output_handle:03lstm_24_lstm_cell_24_matmul_readvariableop_resource5lstm_24_lstm_cell_24_matmul_1_readvariableop_resource4lstm_24_lstm_cell_24_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿô:ÿÿÿÿÿÿÿÿÿô: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *'
bodyR
lstm_24_while_body_40078460*'
condR
lstm_24_while_cond_40078459*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿô:ÿÿÿÿÿÿÿÿÿô: : : : : *
parallel_iterations 2
lstm_24/whileÅ
8lstm_24/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿô  2:
8lstm_24/TensorArrayV2Stack/TensorListStack/element_shape
*lstm_24/TensorArrayV2Stack/TensorListStackTensorListStacklstm_24/while:output:3Alstm_24/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿô*
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
lstm_24/strided_slice_3/stack_2Ë
lstm_24/strided_slice_3StridedSlice3lstm_24/TensorArrayV2Stack/TensorListStack:tensor:0&lstm_24/strided_slice_3/stack:output:0(lstm_24/strided_slice_3/stack_1:output:0(lstm_24/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô*
shrink_axis_mask2
lstm_24/strided_slice_3
lstm_24/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_24/transpose_1/permÆ
lstm_24/transpose_1	Transpose3lstm_24/TensorArrayV2Stack/TensorListStack:tensor:0!lstm_24/transpose_1/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2
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
dropout_24/dropout/Constª
dropout_24/dropout/MulMullstm_24/transpose_1:y:0!dropout_24/dropout/Const:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2
dropout_24/dropout/Mul{
dropout_24/dropout/ShapeShapelstm_24/transpose_1:y:0*
T0*
_output_shapes
:2
dropout_24/dropout/ShapeÚ
/dropout_24/dropout/random_uniform/RandomUniformRandomUniform!dropout_24/dropout/Shape:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿô*
dtype021
/dropout_24/dropout/random_uniform/RandomUniform
!dropout_24/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  >2#
!dropout_24/dropout/GreaterEqual/yï
dropout_24/dropout/GreaterEqualGreaterEqual8dropout_24/dropout/random_uniform/RandomUniform:output:0*dropout_24/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2!
dropout_24/dropout/GreaterEqual¥
dropout_24/dropout/CastCast#dropout_24/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2
dropout_24/dropout/Cast«
dropout_24/dropout/Mul_1Muldropout_24/dropout/Mul:z:0dropout_24/dropout/Cast:y:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2
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
lstm_25/strided_slicel
lstm_25/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :82
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
lstm_25/zeros/Lessr
lstm_25/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :82
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
lstm_25/zeros/Const
lstm_25/zerosFilllstm_25/zeros/packed:output:0lstm_25/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82
lstm_25/zerosp
lstm_25/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :82
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
lstm_25/zeros_1/Lessv
lstm_25/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :82
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
lstm_25/zeros_1/Const
lstm_25/zeros_1Filllstm_25/zeros_1/packed:output:0lstm_25/zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82
lstm_25/zeros_1
lstm_25/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_25/transpose/perm©
lstm_25/transpose	Transposedropout_24/dropout/Mul_1:z:0lstm_25/transpose/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2
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
valueB"ÿÿÿÿô  2?
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
lstm_25/strided_slice_2/stack_2­
lstm_25/strided_slice_2StridedSlicelstm_25/transpose:y:0&lstm_25/strided_slice_2/stack:output:0(lstm_25/strided_slice_2/stack_1:output:0(lstm_25/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô*
shrink_axis_mask2
lstm_25/strided_slice_2Î
*lstm_25/lstm_cell_25/MatMul/ReadVariableOpReadVariableOp3lstm_25_lstm_cell_25_matmul_readvariableop_resource* 
_output_shapes
:
ôà*
dtype02,
*lstm_25/lstm_cell_25/MatMul/ReadVariableOpÍ
lstm_25/lstm_cell_25/MatMulMatMul lstm_25/strided_slice_2:output:02lstm_25/lstm_cell_25/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà2
lstm_25/lstm_cell_25/MatMulÓ
,lstm_25/lstm_cell_25/MatMul_1/ReadVariableOpReadVariableOp5lstm_25_lstm_cell_25_matmul_1_readvariableop_resource*
_output_shapes
:	8à*
dtype02.
,lstm_25/lstm_cell_25/MatMul_1/ReadVariableOpÉ
lstm_25/lstm_cell_25/MatMul_1MatMullstm_25/zeros:output:04lstm_25/lstm_cell_25/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà2
lstm_25/lstm_cell_25/MatMul_1À
lstm_25/lstm_cell_25/addAddV2%lstm_25/lstm_cell_25/MatMul:product:0'lstm_25/lstm_cell_25/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà2
lstm_25/lstm_cell_25/addÌ
+lstm_25/lstm_cell_25/BiasAdd/ReadVariableOpReadVariableOp4lstm_25_lstm_cell_25_biasadd_readvariableop_resource*
_output_shapes	
:à*
dtype02-
+lstm_25/lstm_cell_25/BiasAdd/ReadVariableOpÍ
lstm_25/lstm_cell_25/BiasAddBiasAddlstm_25/lstm_cell_25/add:z:03lstm_25/lstm_cell_25/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà2
lstm_25/lstm_cell_25/BiasAdd
$lstm_25/lstm_cell_25/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2&
$lstm_25/lstm_cell_25/split/split_dim
lstm_25/lstm_cell_25/splitSplit-lstm_25/lstm_cell_25/split/split_dim:output:0%lstm_25/lstm_cell_25/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ8:ÿÿÿÿÿÿÿÿÿ8:ÿÿÿÿÿÿÿÿÿ8:ÿÿÿÿÿÿÿÿÿ8*
	num_split2
lstm_25/lstm_cell_25/split
lstm_25/lstm_cell_25/SigmoidSigmoid#lstm_25/lstm_cell_25/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82
lstm_25/lstm_cell_25/Sigmoid¢
lstm_25/lstm_cell_25/Sigmoid_1Sigmoid#lstm_25/lstm_cell_25/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82 
lstm_25/lstm_cell_25/Sigmoid_1«
lstm_25/lstm_cell_25/mulMul"lstm_25/lstm_cell_25/Sigmoid_1:y:0lstm_25/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82
lstm_25/lstm_cell_25/mul
lstm_25/lstm_cell_25/ReluRelu#lstm_25/lstm_cell_25/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82
lstm_25/lstm_cell_25/Relu¼
lstm_25/lstm_cell_25/mul_1Mul lstm_25/lstm_cell_25/Sigmoid:y:0'lstm_25/lstm_cell_25/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82
lstm_25/lstm_cell_25/mul_1±
lstm_25/lstm_cell_25/add_1AddV2lstm_25/lstm_cell_25/mul:z:0lstm_25/lstm_cell_25/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82
lstm_25/lstm_cell_25/add_1¢
lstm_25/lstm_cell_25/Sigmoid_2Sigmoid#lstm_25/lstm_cell_25/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82 
lstm_25/lstm_cell_25/Sigmoid_2
lstm_25/lstm_cell_25/Relu_1Relulstm_25/lstm_cell_25/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82
lstm_25/lstm_cell_25/Relu_1À
lstm_25/lstm_cell_25/mul_2Mul"lstm_25/lstm_cell_25/Sigmoid_2:y:0)lstm_25/lstm_cell_25/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82
lstm_25/lstm_cell_25/mul_2
%lstm_25/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ8   2'
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
lstm_25/while/loop_counter
lstm_25/whileWhile#lstm_25/while/loop_counter:output:0)lstm_25/while/maximum_iterations:output:0lstm_25/time:output:0 lstm_25/TensorArrayV2_1:handle:0lstm_25/zeros:output:0lstm_25/zeros_1:output:0 lstm_25/strided_slice_1:output:0?lstm_25/TensorArrayUnstack/TensorListFromTensor:output_handle:03lstm_25_lstm_cell_25_matmul_readvariableop_resource5lstm_25_lstm_cell_25_matmul_1_readvariableop_resource4lstm_25_lstm_cell_25_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ8:ÿÿÿÿÿÿÿÿÿ8: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *'
bodyR
lstm_25_while_body_40078615*'
condR
lstm_25_while_cond_40078614*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ8:ÿÿÿÿÿÿÿÿÿ8: : : : : *
parallel_iterations 2
lstm_25/whileÅ
8lstm_25/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ8   2:
8lstm_25/TensorArrayV2Stack/TensorListStack/element_shape
*lstm_25/TensorArrayV2Stack/TensorListStackTensorListStacklstm_25/while:output:3Alstm_25/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ8*
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
lstm_25/strided_slice_3/stack_2Ê
lstm_25/strided_slice_3StridedSlice3lstm_25/TensorArrayV2Stack/TensorListStack:tensor:0&lstm_25/strided_slice_3/stack:output:0(lstm_25/strided_slice_3/stack_1:output:0(lstm_25/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ8*
shrink_axis_mask2
lstm_25/strided_slice_3
lstm_25/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_25/transpose_1/permÅ
lstm_25/transpose_1	Transpose3lstm_25/TensorArrayV2Stack/TensorListStack:tensor:0!lstm_25/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82
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
dropout_25/dropout/Const©
dropout_25/dropout/MulMullstm_25/transpose_1:y:0!dropout_25/dropout/Const:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82
dropout_25/dropout/Mul{
dropout_25/dropout/ShapeShapelstm_25/transpose_1:y:0*
T0*
_output_shapes
:2
dropout_25/dropout/ShapeÙ
/dropout_25/dropout/random_uniform/RandomUniformRandomUniform!dropout_25/dropout/Shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ8*
dtype021
/dropout_25/dropout/random_uniform/RandomUniform
!dropout_25/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>2#
!dropout_25/dropout/GreaterEqual/yî
dropout_25/dropout/GreaterEqualGreaterEqual8dropout_25/dropout/random_uniform/RandomUniform:output:0*dropout_25/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82!
dropout_25/dropout/GreaterEqual¤
dropout_25/dropout/CastCast#dropout_25/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82
dropout_25/dropout/Castª
dropout_25/dropout/Mul_1Muldropout_25/dropout/Mul:z:0dropout_25/dropout/Cast:y:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82
dropout_25/dropout/Mul_1±
!dense_12/Tensordot/ReadVariableOpReadVariableOp*dense_12_tensordot_readvariableop_resource*
_output_shapes

:8*
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
dense_12/Tensordot/stackÁ
dense_12/Tensordot/transpose	Transposedropout_25/dropout/Mul_1:z:0"dense_12/Tensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82
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
°?
Ô
while_body_40079145
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_24_matmul_readvariableop_resource_0:	]ÐI
5while_lstm_cell_24_matmul_1_readvariableop_resource_0:
ôÐC
4while_lstm_cell_24_biasadd_readvariableop_resource_0:	Ð
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_24_matmul_readvariableop_resource:	]ÐG
3while_lstm_cell_24_matmul_1_readvariableop_resource:
ôÐA
2while_lstm_cell_24_biasadd_readvariableop_resource:	Ð¢)while/lstm_cell_24/BiasAdd/ReadVariableOp¢(while/lstm_cell_24/MatMul/ReadVariableOp¢*while/lstm_cell_24/MatMul_1/ReadVariableOpÃ
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
:	]Ð*
dtype02*
(while/lstm_cell_24/MatMul/ReadVariableOp×
while/lstm_cell_24/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_24/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
while/lstm_cell_24/MatMulÐ
*while/lstm_cell_24/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_24_matmul_1_readvariableop_resource_0* 
_output_shapes
:
ôÐ*
dtype02,
*while/lstm_cell_24/MatMul_1/ReadVariableOpÀ
while/lstm_cell_24/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_24/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
while/lstm_cell_24/MatMul_1¸
while/lstm_cell_24/addAddV2#while/lstm_cell_24/MatMul:product:0%while/lstm_cell_24/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
while/lstm_cell_24/addÈ
)while/lstm_cell_24/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_24_biasadd_readvariableop_resource_0*
_output_shapes	
:Ð*
dtype02+
)while/lstm_cell_24/BiasAdd/ReadVariableOpÅ
while/lstm_cell_24/BiasAddBiasAddwhile/lstm_cell_24/add:z:01while/lstm_cell_24/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
while/lstm_cell_24/BiasAdd
"while/lstm_cell_24/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_24/split/split_dim
while/lstm_cell_24/splitSplit+while/lstm_cell_24/split/split_dim:output:0#while/lstm_cell_24/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿô:ÿÿÿÿÿÿÿÿÿô:ÿÿÿÿÿÿÿÿÿô:ÿÿÿÿÿÿÿÿÿô*
	num_split2
while/lstm_cell_24/split
while/lstm_cell_24/SigmoidSigmoid!while/lstm_cell_24/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2
while/lstm_cell_24/Sigmoid
while/lstm_cell_24/Sigmoid_1Sigmoid!while/lstm_cell_24/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2
while/lstm_cell_24/Sigmoid_1¡
while/lstm_cell_24/mulMul while/lstm_cell_24/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2
while/lstm_cell_24/mul
while/lstm_cell_24/ReluRelu!while/lstm_cell_24/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2
while/lstm_cell_24/Reluµ
while/lstm_cell_24/mul_1Mulwhile/lstm_cell_24/Sigmoid:y:0%while/lstm_cell_24/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2
while/lstm_cell_24/mul_1ª
while/lstm_cell_24/add_1AddV2while/lstm_cell_24/mul:z:0while/lstm_cell_24/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2
while/lstm_cell_24/add_1
while/lstm_cell_24/Sigmoid_2Sigmoid!while/lstm_cell_24/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2
while/lstm_cell_24/Sigmoid_2
while/lstm_cell_24/Relu_1Reluwhile/lstm_cell_24/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2
while/lstm_cell_24/Relu_1¹
while/lstm_cell_24/mul_2Mul while/lstm_cell_24/Sigmoid_2:y:0'while/lstm_cell_24/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2
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
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_24/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_24/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2
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
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿô:ÿÿÿÿÿÿÿÿÿô: : : : : 2V
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
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô:

_output_shapes
: :

_output_shapes
: 


Ë
0__inference_sequential_12_layer_call_fn_40078776

inputs
unknown:	]Ð
	unknown_0:
ôÐ
	unknown_1:	Ð
	unknown_2:
ôà
	unknown_3:	8à
	unknown_4:	à
	unknown_5:8
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
K__inference_sequential_12_layer_call_and_return_conditional_losses_400779472
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
Ø
I
-__inference_dropout_24_layer_call_fn_40079446

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
:ÿÿÿÿÿÿÿÿÿô* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_dropout_24_layer_call_and_return_conditional_losses_400772732
PartitionedCallq
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿô:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿô
 
_user_specified_nameinputs
ã
Í
while_cond_40079295
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_40079295___redundant_placeholder06
2while_while_cond_40079295___redundant_placeholder16
2while_while_cond_40079295___redundant_placeholder26
2while_while_cond_40079295___redundant_placeholder3
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
B: : : : :ÿÿÿÿÿÿÿÿÿô:ÿÿÿÿÿÿÿÿÿô: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô:

_output_shapes
: :

_output_shapes
:
?
Ô
while_body_40079971
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
3while_lstm_cell_25_matmul_readvariableop_resource_0:
ôàH
5while_lstm_cell_25_matmul_1_readvariableop_resource_0:	8àC
4while_lstm_cell_25_biasadd_readvariableop_resource_0:	à
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
1while_lstm_cell_25_matmul_readvariableop_resource:
ôàF
3while_lstm_cell_25_matmul_1_readvariableop_resource:	8àA
2while_lstm_cell_25_biasadd_readvariableop_resource:	à¢)while/lstm_cell_25/BiasAdd/ReadVariableOp¢(while/lstm_cell_25/MatMul/ReadVariableOp¢*while/lstm_cell_25/MatMul_1/ReadVariableOpÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿô  29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÔ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemÊ
(while/lstm_cell_25/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_25_matmul_readvariableop_resource_0* 
_output_shapes
:
ôà*
dtype02*
(while/lstm_cell_25/MatMul/ReadVariableOp×
while/lstm_cell_25/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_25/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà2
while/lstm_cell_25/MatMulÏ
*while/lstm_cell_25/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_25_matmul_1_readvariableop_resource_0*
_output_shapes
:	8à*
dtype02,
*while/lstm_cell_25/MatMul_1/ReadVariableOpÀ
while/lstm_cell_25/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_25/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà2
while/lstm_cell_25/MatMul_1¸
while/lstm_cell_25/addAddV2#while/lstm_cell_25/MatMul:product:0%while/lstm_cell_25/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà2
while/lstm_cell_25/addÈ
)while/lstm_cell_25/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_25_biasadd_readvariableop_resource_0*
_output_shapes	
:à*
dtype02+
)while/lstm_cell_25/BiasAdd/ReadVariableOpÅ
while/lstm_cell_25/BiasAddBiasAddwhile/lstm_cell_25/add:z:01while/lstm_cell_25/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà2
while/lstm_cell_25/BiasAdd
"while/lstm_cell_25/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_25/split/split_dim
while/lstm_cell_25/splitSplit+while/lstm_cell_25/split/split_dim:output:0#while/lstm_cell_25/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ8:ÿÿÿÿÿÿÿÿÿ8:ÿÿÿÿÿÿÿÿÿ8:ÿÿÿÿÿÿÿÿÿ8*
	num_split2
while/lstm_cell_25/split
while/lstm_cell_25/SigmoidSigmoid!while/lstm_cell_25/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82
while/lstm_cell_25/Sigmoid
while/lstm_cell_25/Sigmoid_1Sigmoid!while/lstm_cell_25/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82
while/lstm_cell_25/Sigmoid_1 
while/lstm_cell_25/mulMul while/lstm_cell_25/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82
while/lstm_cell_25/mul
while/lstm_cell_25/ReluRelu!while/lstm_cell_25/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82
while/lstm_cell_25/Relu´
while/lstm_cell_25/mul_1Mulwhile/lstm_cell_25/Sigmoid:y:0%while/lstm_cell_25/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82
while/lstm_cell_25/mul_1©
while/lstm_cell_25/add_1AddV2while/lstm_cell_25/mul:z:0while/lstm_cell_25/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82
while/lstm_cell_25/add_1
while/lstm_cell_25/Sigmoid_2Sigmoid!while/lstm_cell_25/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82
while/lstm_cell_25/Sigmoid_2
while/lstm_cell_25/Relu_1Reluwhile/lstm_cell_25/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82
while/lstm_cell_25/Relu_1¸
while/lstm_cell_25/mul_2Mul while/lstm_cell_25/Sigmoid_2:y:0'while/lstm_cell_25/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82
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
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_25/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_25/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82
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
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ8:ÿÿÿÿÿÿÿÿÿ8: : : : : 2V
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
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ8:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ8:

_output_shapes
: :

_output_shapes
: 

f
H__inference_dropout_25_layer_call_and_return_conditional_losses_40080104

inputs

identity_1^
IdentityIdentityinputs*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82

Identitym

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ8:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ8
 
_user_specified_nameinputs
ù

J__inference_lstm_cell_25_layer_call_and_return_conditional_losses_40076548

inputs

states
states_12
matmul_readvariableop_resource:
ôà3
 matmul_1_readvariableop_resource:	8à.
biasadd_readvariableop_resource:	à
identity

identity_1

identity_2¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
ôà*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà2
MatMul
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	8à*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà2
add
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:à*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà2	
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
L:ÿÿÿÿÿÿÿÿÿ8:ÿÿÿÿÿÿÿÿÿ8:ÿÿÿÿÿÿÿÿÿ8:ÿÿÿÿÿÿÿÿÿ8*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82
	Sigmoid_1\
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82
mulV
ReluRelusplit:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82
	Sigmoid_2U
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82
mul_2d
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82

Identityh

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82

Identity_1h

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82

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
@:ÿÿÿÿÿÿÿÿÿô:ÿÿÿÿÿÿÿÿÿ8:ÿÿÿÿÿÿÿÿÿ8: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ8
 
_user_specified_namestates:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ8
 
_user_specified_namestates
ß
Í
while_cond_40076561
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_40076561___redundant_placeholder06
2while_while_cond_40076561___redundant_placeholder16
2while_while_cond_40076561___redundant_placeholder26
2while_while_cond_40076561___redundant_placeholder3
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
@: : : : :ÿÿÿÿÿÿÿÿÿ8:ÿÿÿÿÿÿÿÿÿ8: ::::: 

_output_shapes
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
:ÿÿÿÿÿÿÿÿÿ8:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ8:

_output_shapes
: :

_output_shapes
:
Ô

í
lstm_24_while_cond_40078459,
(lstm_24_while_lstm_24_while_loop_counter2
.lstm_24_while_lstm_24_while_maximum_iterations
lstm_24_while_placeholder
lstm_24_while_placeholder_1
lstm_24_while_placeholder_2
lstm_24_while_placeholder_3.
*lstm_24_while_less_lstm_24_strided_slice_1F
Blstm_24_while_lstm_24_while_cond_40078459___redundant_placeholder0F
Blstm_24_while_lstm_24_while_cond_40078459___redundant_placeholder1F
Blstm_24_while_lstm_24_while_cond_40078459___redundant_placeholder2F
Blstm_24_while_lstm_24_while_cond_40078459___redundant_placeholder3
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
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :ÿÿÿÿÿÿÿÿÿô:ÿÿÿÿÿÿÿÿÿô: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô:

_output_shapes
: :

_output_shapes
:
í[

E__inference_lstm_25_layer_call_and_return_conditional_losses_40077425

inputs?
+lstm_cell_25_matmul_readvariableop_resource:
ôà@
-lstm_cell_25_matmul_1_readvariableop_resource:	8à;
,lstm_cell_25_biasadd_readvariableop_resource:	à
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
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :82
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
value	B :82
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
:ÿÿÿÿÿÿÿÿÿ82
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :82
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
value	B :82
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
:ÿÿÿÿÿÿÿÿÿ82	
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
:ÿÿÿÿÿÿÿÿÿô2
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
valueB"ÿÿÿÿô  27
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
:ÿÿÿÿÿÿÿÿÿô*
shrink_axis_mask2
strided_slice_2¶
"lstm_cell_25/MatMul/ReadVariableOpReadVariableOp+lstm_cell_25_matmul_readvariableop_resource* 
_output_shapes
:
ôà*
dtype02$
"lstm_cell_25/MatMul/ReadVariableOp­
lstm_cell_25/MatMulMatMulstrided_slice_2:output:0*lstm_cell_25/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà2
lstm_cell_25/MatMul»
$lstm_cell_25/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_25_matmul_1_readvariableop_resource*
_output_shapes
:	8à*
dtype02&
$lstm_cell_25/MatMul_1/ReadVariableOp©
lstm_cell_25/MatMul_1MatMulzeros:output:0,lstm_cell_25/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà2
lstm_cell_25/MatMul_1 
lstm_cell_25/addAddV2lstm_cell_25/MatMul:product:0lstm_cell_25/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà2
lstm_cell_25/add´
#lstm_cell_25/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_25_biasadd_readvariableop_resource*
_output_shapes	
:à*
dtype02%
#lstm_cell_25/BiasAdd/ReadVariableOp­
lstm_cell_25/BiasAddBiasAddlstm_cell_25/add:z:0+lstm_cell_25/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà2
lstm_cell_25/BiasAdd~
lstm_cell_25/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_25/split/split_dimó
lstm_cell_25/splitSplit%lstm_cell_25/split/split_dim:output:0lstm_cell_25/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ8:ÿÿÿÿÿÿÿÿÿ8:ÿÿÿÿÿÿÿÿÿ8:ÿÿÿÿÿÿÿÿÿ8*
	num_split2
lstm_cell_25/split
lstm_cell_25/SigmoidSigmoidlstm_cell_25/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82
lstm_cell_25/Sigmoid
lstm_cell_25/Sigmoid_1Sigmoidlstm_cell_25/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82
lstm_cell_25/Sigmoid_1
lstm_cell_25/mulMullstm_cell_25/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82
lstm_cell_25/mul}
lstm_cell_25/ReluRelulstm_cell_25/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82
lstm_cell_25/Relu
lstm_cell_25/mul_1Mullstm_cell_25/Sigmoid:y:0lstm_cell_25/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82
lstm_cell_25/mul_1
lstm_cell_25/add_1AddV2lstm_cell_25/mul:z:0lstm_cell_25/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82
lstm_cell_25/add_1
lstm_cell_25/Sigmoid_2Sigmoidlstm_cell_25/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82
lstm_cell_25/Sigmoid_2|
lstm_cell_25/Relu_1Relulstm_cell_25/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82
lstm_cell_25/Relu_1 
lstm_cell_25/mul_2Mullstm_cell_25/Sigmoid_2:y:0!lstm_cell_25/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82
lstm_cell_25/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ8   2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_25_matmul_readvariableop_resource-lstm_cell_25_matmul_1_readvariableop_resource,lstm_cell_25_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ8:ÿÿÿÿÿÿÿÿÿ8: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_40077341*
condR
while_cond_40077340*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ8:ÿÿÿÿÿÿÿÿÿ8: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ8   22
0TensorArrayV2Stack/TensorListStack/element_shapeè
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ8*
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
:ÿÿÿÿÿÿÿÿÿ8*
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
:ÿÿÿÿÿÿÿÿÿ82
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
:ÿÿÿÿÿÿÿÿÿ82

IdentityÈ
NoOpNoOp$^lstm_cell_25/BiasAdd/ReadVariableOp#^lstm_cell_25/MatMul/ReadVariableOp%^lstm_cell_25/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:ÿÿÿÿÿÿÿÿÿô: : : 2J
#lstm_cell_25/BiasAdd/ReadVariableOp#lstm_cell_25/BiasAdd/ReadVariableOp2H
"lstm_cell_25/MatMul/ReadVariableOp"lstm_cell_25/MatMul/ReadVariableOp2L
$lstm_cell_25/MatMul_1/ReadVariableOp$lstm_cell_25/MatMul_1/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿô
 
_user_specified_nameinputs
Ò^

)sequential_12_lstm_25_while_body_40075731H
Dsequential_12_lstm_25_while_sequential_12_lstm_25_while_loop_counterN
Jsequential_12_lstm_25_while_sequential_12_lstm_25_while_maximum_iterations+
'sequential_12_lstm_25_while_placeholder-
)sequential_12_lstm_25_while_placeholder_1-
)sequential_12_lstm_25_while_placeholder_2-
)sequential_12_lstm_25_while_placeholder_3G
Csequential_12_lstm_25_while_sequential_12_lstm_25_strided_slice_1_0
sequential_12_lstm_25_while_tensorarrayv2read_tensorlistgetitem_sequential_12_lstm_25_tensorarrayunstack_tensorlistfromtensor_0]
Isequential_12_lstm_25_while_lstm_cell_25_matmul_readvariableop_resource_0:
ôà^
Ksequential_12_lstm_25_while_lstm_cell_25_matmul_1_readvariableop_resource_0:	8àY
Jsequential_12_lstm_25_while_lstm_cell_25_biasadd_readvariableop_resource_0:	à(
$sequential_12_lstm_25_while_identity*
&sequential_12_lstm_25_while_identity_1*
&sequential_12_lstm_25_while_identity_2*
&sequential_12_lstm_25_while_identity_3*
&sequential_12_lstm_25_while_identity_4*
&sequential_12_lstm_25_while_identity_5E
Asequential_12_lstm_25_while_sequential_12_lstm_25_strided_slice_1
}sequential_12_lstm_25_while_tensorarrayv2read_tensorlistgetitem_sequential_12_lstm_25_tensorarrayunstack_tensorlistfromtensor[
Gsequential_12_lstm_25_while_lstm_cell_25_matmul_readvariableop_resource:
ôà\
Isequential_12_lstm_25_while_lstm_cell_25_matmul_1_readvariableop_resource:	8àW
Hsequential_12_lstm_25_while_lstm_cell_25_biasadd_readvariableop_resource:	à¢?sequential_12/lstm_25/while/lstm_cell_25/BiasAdd/ReadVariableOp¢>sequential_12/lstm_25/while/lstm_cell_25/MatMul/ReadVariableOp¢@sequential_12/lstm_25/while/lstm_cell_25/MatMul_1/ReadVariableOpï
Msequential_12/lstm_25/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿô  2O
Msequential_12/lstm_25/while/TensorArrayV2Read/TensorListGetItem/element_shapeØ
?sequential_12/lstm_25/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemsequential_12_lstm_25_while_tensorarrayv2read_tensorlistgetitem_sequential_12_lstm_25_tensorarrayunstack_tensorlistfromtensor_0'sequential_12_lstm_25_while_placeholderVsequential_12/lstm_25/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô*
element_dtype02A
?sequential_12/lstm_25/while/TensorArrayV2Read/TensorListGetItem
>sequential_12/lstm_25/while/lstm_cell_25/MatMul/ReadVariableOpReadVariableOpIsequential_12_lstm_25_while_lstm_cell_25_matmul_readvariableop_resource_0* 
_output_shapes
:
ôà*
dtype02@
>sequential_12/lstm_25/while/lstm_cell_25/MatMul/ReadVariableOp¯
/sequential_12/lstm_25/while/lstm_cell_25/MatMulMatMulFsequential_12/lstm_25/while/TensorArrayV2Read/TensorListGetItem:item:0Fsequential_12/lstm_25/while/lstm_cell_25/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà21
/sequential_12/lstm_25/while/lstm_cell_25/MatMul
@sequential_12/lstm_25/while/lstm_cell_25/MatMul_1/ReadVariableOpReadVariableOpKsequential_12_lstm_25_while_lstm_cell_25_matmul_1_readvariableop_resource_0*
_output_shapes
:	8à*
dtype02B
@sequential_12/lstm_25/while/lstm_cell_25/MatMul_1/ReadVariableOp
1sequential_12/lstm_25/while/lstm_cell_25/MatMul_1MatMul)sequential_12_lstm_25_while_placeholder_2Hsequential_12/lstm_25/while/lstm_cell_25/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà23
1sequential_12/lstm_25/while/lstm_cell_25/MatMul_1
,sequential_12/lstm_25/while/lstm_cell_25/addAddV29sequential_12/lstm_25/while/lstm_cell_25/MatMul:product:0;sequential_12/lstm_25/while/lstm_cell_25/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà2.
,sequential_12/lstm_25/while/lstm_cell_25/add
?sequential_12/lstm_25/while/lstm_cell_25/BiasAdd/ReadVariableOpReadVariableOpJsequential_12_lstm_25_while_lstm_cell_25_biasadd_readvariableop_resource_0*
_output_shapes	
:à*
dtype02A
?sequential_12/lstm_25/while/lstm_cell_25/BiasAdd/ReadVariableOp
0sequential_12/lstm_25/while/lstm_cell_25/BiasAddBiasAdd0sequential_12/lstm_25/while/lstm_cell_25/add:z:0Gsequential_12/lstm_25/while/lstm_cell_25/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà22
0sequential_12/lstm_25/while/lstm_cell_25/BiasAdd¶
8sequential_12/lstm_25/while/lstm_cell_25/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2:
8sequential_12/lstm_25/while/lstm_cell_25/split/split_dimã
.sequential_12/lstm_25/while/lstm_cell_25/splitSplitAsequential_12/lstm_25/while/lstm_cell_25/split/split_dim:output:09sequential_12/lstm_25/while/lstm_cell_25/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ8:ÿÿÿÿÿÿÿÿÿ8:ÿÿÿÿÿÿÿÿÿ8:ÿÿÿÿÿÿÿÿÿ8*
	num_split20
.sequential_12/lstm_25/while/lstm_cell_25/splitÚ
0sequential_12/lstm_25/while/lstm_cell_25/SigmoidSigmoid7sequential_12/lstm_25/while/lstm_cell_25/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ822
0sequential_12/lstm_25/while/lstm_cell_25/SigmoidÞ
2sequential_12/lstm_25/while/lstm_cell_25/Sigmoid_1Sigmoid7sequential_12/lstm_25/while/lstm_cell_25/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ824
2sequential_12/lstm_25/while/lstm_cell_25/Sigmoid_1ø
,sequential_12/lstm_25/while/lstm_cell_25/mulMul6sequential_12/lstm_25/while/lstm_cell_25/Sigmoid_1:y:0)sequential_12_lstm_25_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82.
,sequential_12/lstm_25/while/lstm_cell_25/mulÑ
-sequential_12/lstm_25/while/lstm_cell_25/ReluRelu7sequential_12/lstm_25/while/lstm_cell_25/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82/
-sequential_12/lstm_25/while/lstm_cell_25/Relu
.sequential_12/lstm_25/while/lstm_cell_25/mul_1Mul4sequential_12/lstm_25/while/lstm_cell_25/Sigmoid:y:0;sequential_12/lstm_25/while/lstm_cell_25/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ820
.sequential_12/lstm_25/while/lstm_cell_25/mul_1
.sequential_12/lstm_25/while/lstm_cell_25/add_1AddV20sequential_12/lstm_25/while/lstm_cell_25/mul:z:02sequential_12/lstm_25/while/lstm_cell_25/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ820
.sequential_12/lstm_25/while/lstm_cell_25/add_1Þ
2sequential_12/lstm_25/while/lstm_cell_25/Sigmoid_2Sigmoid7sequential_12/lstm_25/while/lstm_cell_25/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ824
2sequential_12/lstm_25/while/lstm_cell_25/Sigmoid_2Ð
/sequential_12/lstm_25/while/lstm_cell_25/Relu_1Relu2sequential_12/lstm_25/while/lstm_cell_25/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ821
/sequential_12/lstm_25/while/lstm_cell_25/Relu_1
.sequential_12/lstm_25/while/lstm_cell_25/mul_2Mul6sequential_12/lstm_25/while/lstm_cell_25/Sigmoid_2:y:0=sequential_12/lstm_25/while/lstm_cell_25/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ820
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
&sequential_12/lstm_25/while/Identity_3å
&sequential_12/lstm_25/while/Identity_4Identity2sequential_12/lstm_25/while/lstm_cell_25/mul_2:z:0!^sequential_12/lstm_25/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82(
&sequential_12/lstm_25/while/Identity_4å
&sequential_12/lstm_25/while/Identity_5Identity2sequential_12/lstm_25/while/lstm_cell_25/add_1:z:0!^sequential_12/lstm_25/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82(
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
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ8:ÿÿÿÿÿÿÿÿÿ8: : : : : 2
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
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ8:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ8:

_output_shapes
: :

_output_shapes
: 
Ã\
 
E__inference_lstm_24_layer_call_and_return_conditional_losses_40078927
inputs_0>
+lstm_cell_24_matmul_readvariableop_resource:	]ÐA
-lstm_cell_24_matmul_1_readvariableop_resource:
ôÐ;
,lstm_cell_24_biasadd_readvariableop_resource:	Ð
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
strided_slice]
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :ô2
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
B :ô2
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
:ÿÿÿÿÿÿÿÿÿô2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :ô2
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
B :ô2
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
:ÿÿÿÿÿÿÿÿÿô2	
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
:	]Ð*
dtype02$
"lstm_cell_24/MatMul/ReadVariableOp­
lstm_cell_24/MatMulMatMulstrided_slice_2:output:0*lstm_cell_24/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
lstm_cell_24/MatMul¼
$lstm_cell_24/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_24_matmul_1_readvariableop_resource* 
_output_shapes
:
ôÐ*
dtype02&
$lstm_cell_24/MatMul_1/ReadVariableOp©
lstm_cell_24/MatMul_1MatMulzeros:output:0,lstm_cell_24/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
lstm_cell_24/MatMul_1 
lstm_cell_24/addAddV2lstm_cell_24/MatMul:product:0lstm_cell_24/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
lstm_cell_24/add´
#lstm_cell_24/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_24_biasadd_readvariableop_resource*
_output_shapes	
:Ð*
dtype02%
#lstm_cell_24/BiasAdd/ReadVariableOp­
lstm_cell_24/BiasAddBiasAddlstm_cell_24/add:z:0+lstm_cell_24/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
lstm_cell_24/BiasAdd~
lstm_cell_24/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_24/split/split_dim÷
lstm_cell_24/splitSplit%lstm_cell_24/split/split_dim:output:0lstm_cell_24/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿô:ÿÿÿÿÿÿÿÿÿô:ÿÿÿÿÿÿÿÿÿô:ÿÿÿÿÿÿÿÿÿô*
	num_split2
lstm_cell_24/split
lstm_cell_24/SigmoidSigmoidlstm_cell_24/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2
lstm_cell_24/Sigmoid
lstm_cell_24/Sigmoid_1Sigmoidlstm_cell_24/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2
lstm_cell_24/Sigmoid_1
lstm_cell_24/mulMullstm_cell_24/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2
lstm_cell_24/mul~
lstm_cell_24/ReluRelulstm_cell_24/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2
lstm_cell_24/Relu
lstm_cell_24/mul_1Mullstm_cell_24/Sigmoid:y:0lstm_cell_24/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2
lstm_cell_24/mul_1
lstm_cell_24/add_1AddV2lstm_cell_24/mul:z:0lstm_cell_24/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2
lstm_cell_24/add_1
lstm_cell_24/Sigmoid_2Sigmoidlstm_cell_24/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2
lstm_cell_24/Sigmoid_2}
lstm_cell_24/Relu_1Relulstm_cell_24/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2
lstm_cell_24/Relu_1¡
lstm_cell_24/mul_2Mullstm_cell_24/Sigmoid_2:y:0!lstm_cell_24/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2
lstm_cell_24/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿô  2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_24_matmul_readvariableop_resource-lstm_cell_24_matmul_1_readvariableop_resource,lstm_cell_24_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿô:ÿÿÿÿÿÿÿÿÿô: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_40078843*
condR
while_cond_40078842*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿô:ÿÿÿÿÿÿÿÿÿô: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿô  22
0TensorArrayV2Stack/TensorListStack/element_shapeò
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿô*
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
:ÿÿÿÿÿÿÿÿÿô*
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
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿô2
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
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿô2

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
ä^

)sequential_12_lstm_24_while_body_40075583H
Dsequential_12_lstm_24_while_sequential_12_lstm_24_while_loop_counterN
Jsequential_12_lstm_24_while_sequential_12_lstm_24_while_maximum_iterations+
'sequential_12_lstm_24_while_placeholder-
)sequential_12_lstm_24_while_placeholder_1-
)sequential_12_lstm_24_while_placeholder_2-
)sequential_12_lstm_24_while_placeholder_3G
Csequential_12_lstm_24_while_sequential_12_lstm_24_strided_slice_1_0
sequential_12_lstm_24_while_tensorarrayv2read_tensorlistgetitem_sequential_12_lstm_24_tensorarrayunstack_tensorlistfromtensor_0\
Isequential_12_lstm_24_while_lstm_cell_24_matmul_readvariableop_resource_0:	]Ð_
Ksequential_12_lstm_24_while_lstm_cell_24_matmul_1_readvariableop_resource_0:
ôÐY
Jsequential_12_lstm_24_while_lstm_cell_24_biasadd_readvariableop_resource_0:	Ð(
$sequential_12_lstm_24_while_identity*
&sequential_12_lstm_24_while_identity_1*
&sequential_12_lstm_24_while_identity_2*
&sequential_12_lstm_24_while_identity_3*
&sequential_12_lstm_24_while_identity_4*
&sequential_12_lstm_24_while_identity_5E
Asequential_12_lstm_24_while_sequential_12_lstm_24_strided_slice_1
}sequential_12_lstm_24_while_tensorarrayv2read_tensorlistgetitem_sequential_12_lstm_24_tensorarrayunstack_tensorlistfromtensorZ
Gsequential_12_lstm_24_while_lstm_cell_24_matmul_readvariableop_resource:	]Ð]
Isequential_12_lstm_24_while_lstm_cell_24_matmul_1_readvariableop_resource:
ôÐW
Hsequential_12_lstm_24_while_lstm_cell_24_biasadd_readvariableop_resource:	Ð¢?sequential_12/lstm_24/while/lstm_cell_24/BiasAdd/ReadVariableOp¢>sequential_12/lstm_24/while/lstm_cell_24/MatMul/ReadVariableOp¢@sequential_12/lstm_24/while/lstm_cell_24/MatMul_1/ReadVariableOpï
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
:	]Ð*
dtype02@
>sequential_12/lstm_24/while/lstm_cell_24/MatMul/ReadVariableOp¯
/sequential_12/lstm_24/while/lstm_cell_24/MatMulMatMulFsequential_12/lstm_24/while/TensorArrayV2Read/TensorListGetItem:item:0Fsequential_12/lstm_24/while/lstm_cell_24/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ21
/sequential_12/lstm_24/while/lstm_cell_24/MatMul
@sequential_12/lstm_24/while/lstm_cell_24/MatMul_1/ReadVariableOpReadVariableOpKsequential_12_lstm_24_while_lstm_cell_24_matmul_1_readvariableop_resource_0* 
_output_shapes
:
ôÐ*
dtype02B
@sequential_12/lstm_24/while/lstm_cell_24/MatMul_1/ReadVariableOp
1sequential_12/lstm_24/while/lstm_cell_24/MatMul_1MatMul)sequential_12_lstm_24_while_placeholder_2Hsequential_12/lstm_24/while/lstm_cell_24/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ23
1sequential_12/lstm_24/while/lstm_cell_24/MatMul_1
,sequential_12/lstm_24/while/lstm_cell_24/addAddV29sequential_12/lstm_24/while/lstm_cell_24/MatMul:product:0;sequential_12/lstm_24/while/lstm_cell_24/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2.
,sequential_12/lstm_24/while/lstm_cell_24/add
?sequential_12/lstm_24/while/lstm_cell_24/BiasAdd/ReadVariableOpReadVariableOpJsequential_12_lstm_24_while_lstm_cell_24_biasadd_readvariableop_resource_0*
_output_shapes	
:Ð*
dtype02A
?sequential_12/lstm_24/while/lstm_cell_24/BiasAdd/ReadVariableOp
0sequential_12/lstm_24/while/lstm_cell_24/BiasAddBiasAdd0sequential_12/lstm_24/while/lstm_cell_24/add:z:0Gsequential_12/lstm_24/while/lstm_cell_24/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ22
0sequential_12/lstm_24/while/lstm_cell_24/BiasAdd¶
8sequential_12/lstm_24/while/lstm_cell_24/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2:
8sequential_12/lstm_24/while/lstm_cell_24/split/split_dimç
.sequential_12/lstm_24/while/lstm_cell_24/splitSplitAsequential_12/lstm_24/while/lstm_cell_24/split/split_dim:output:09sequential_12/lstm_24/while/lstm_cell_24/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿô:ÿÿÿÿÿÿÿÿÿô:ÿÿÿÿÿÿÿÿÿô:ÿÿÿÿÿÿÿÿÿô*
	num_split20
.sequential_12/lstm_24/while/lstm_cell_24/splitÛ
0sequential_12/lstm_24/while/lstm_cell_24/SigmoidSigmoid7sequential_12/lstm_24/while/lstm_cell_24/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô22
0sequential_12/lstm_24/while/lstm_cell_24/Sigmoidß
2sequential_12/lstm_24/while/lstm_cell_24/Sigmoid_1Sigmoid7sequential_12/lstm_24/while/lstm_cell_24/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô24
2sequential_12/lstm_24/while/lstm_cell_24/Sigmoid_1ù
,sequential_12/lstm_24/while/lstm_cell_24/mulMul6sequential_12/lstm_24/while/lstm_cell_24/Sigmoid_1:y:0)sequential_12_lstm_24_while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2.
,sequential_12/lstm_24/while/lstm_cell_24/mulÒ
-sequential_12/lstm_24/while/lstm_cell_24/ReluRelu7sequential_12/lstm_24/while/lstm_cell_24/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2/
-sequential_12/lstm_24/while/lstm_cell_24/Relu
.sequential_12/lstm_24/while/lstm_cell_24/mul_1Mul4sequential_12/lstm_24/while/lstm_cell_24/Sigmoid:y:0;sequential_12/lstm_24/while/lstm_cell_24/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô20
.sequential_12/lstm_24/while/lstm_cell_24/mul_1
.sequential_12/lstm_24/while/lstm_cell_24/add_1AddV20sequential_12/lstm_24/while/lstm_cell_24/mul:z:02sequential_12/lstm_24/while/lstm_cell_24/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô20
.sequential_12/lstm_24/while/lstm_cell_24/add_1ß
2sequential_12/lstm_24/while/lstm_cell_24/Sigmoid_2Sigmoid7sequential_12/lstm_24/while/lstm_cell_24/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô24
2sequential_12/lstm_24/while/lstm_cell_24/Sigmoid_2Ñ
/sequential_12/lstm_24/while/lstm_cell_24/Relu_1Relu2sequential_12/lstm_24/while/lstm_cell_24/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô21
/sequential_12/lstm_24/while/lstm_cell_24/Relu_1
.sequential_12/lstm_24/while/lstm_cell_24/mul_2Mul6sequential_12/lstm_24/while/lstm_cell_24/Sigmoid_2:y:0=sequential_12/lstm_24/while/lstm_cell_24/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô20
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
&sequential_12/lstm_24/while/Identity_3æ
&sequential_12/lstm_24/while/Identity_4Identity2sequential_12/lstm_24/while/lstm_cell_24/mul_2:z:0!^sequential_12/lstm_24/while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2(
&sequential_12/lstm_24/while/Identity_4æ
&sequential_12/lstm_24/while/Identity_5Identity2sequential_12/lstm_24/while/lstm_cell_24/add_1:z:0!^sequential_12/lstm_24/while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2(
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
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿô:ÿÿÿÿÿÿÿÿÿô: : : : : 2
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
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô:

_output_shapes
: :

_output_shapes
: 
Ð

í
lstm_25_while_cond_40078614,
(lstm_25_while_lstm_25_while_loop_counter2
.lstm_25_while_lstm_25_while_maximum_iterations
lstm_25_while_placeholder
lstm_25_while_placeholder_1
lstm_25_while_placeholder_2
lstm_25_while_placeholder_3.
*lstm_25_while_less_lstm_25_strided_slice_1F
Blstm_25_while_lstm_25_while_cond_40078614___redundant_placeholder0F
Blstm_25_while_lstm_25_while_cond_40078614___redundant_placeholder1F
Blstm_25_while_lstm_25_while_cond_40078614___redundant_placeholder2F
Blstm_25_while_lstm_25_while_cond_40078614___redundant_placeholder3
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
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿ8:ÿÿÿÿÿÿÿÿÿ8: ::::: 

_output_shapes
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
:ÿÿÿÿÿÿÿÿÿ8:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ8:

_output_shapes
: :

_output_shapes
:


J__inference_lstm_cell_24_layer_call_and_return_conditional_losses_40080230

inputs
states_0
states_11
matmul_readvariableop_resource:	]Ð4
 matmul_1_readvariableop_resource:
ôÐ.
biasadd_readvariableop_resource:	Ð
identity

identity_1

identity_2¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	]Ð*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
MatMul
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
ôÐ*
dtype02
MatMul_1/ReadVariableOp|
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
add
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:Ð*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2	
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
P:ÿÿÿÿÿÿÿÿÿô:ÿÿÿÿÿÿÿÿÿô:ÿÿÿÿÿÿÿÿÿô:ÿÿÿÿÿÿÿÿÿô*
	num_split2
split`
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2	
Sigmoidd
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2
	Sigmoid_1]
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2
mulW
ReluRelusplit:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2
Relui
mul_1MulSigmoid:y:0Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2
mul_1^
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2
add_1d
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2
	Sigmoid_2V
Relu_1Relu	add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2
Relu_1m
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2
mul_2e
IdentityIdentity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2

Identityi

Identity_1Identity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2

Identity_1i

Identity_2Identity	add_1:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2

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
A:ÿÿÿÿÿÿÿÿÿ]:ÿÿÿÿÿÿÿÿÿô:ÿÿÿÿÿÿÿÿÿô: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
 
_user_specified_nameinputs:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô
"
_user_specified_name
states/0:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô
"
_user_specified_name
states/1
\

E__inference_lstm_24_layer_call_and_return_conditional_losses_40077260

inputs>
+lstm_cell_24_matmul_readvariableop_resource:	]ÐA
-lstm_cell_24_matmul_1_readvariableop_resource:
ôÐ;
,lstm_cell_24_biasadd_readvariableop_resource:	Ð
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
strided_slice]
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :ô2
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
B :ô2
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
:ÿÿÿÿÿÿÿÿÿô2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :ô2
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
B :ô2
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
:ÿÿÿÿÿÿÿÿÿô2	
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
:	]Ð*
dtype02$
"lstm_cell_24/MatMul/ReadVariableOp­
lstm_cell_24/MatMulMatMulstrided_slice_2:output:0*lstm_cell_24/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
lstm_cell_24/MatMul¼
$lstm_cell_24/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_24_matmul_1_readvariableop_resource* 
_output_shapes
:
ôÐ*
dtype02&
$lstm_cell_24/MatMul_1/ReadVariableOp©
lstm_cell_24/MatMul_1MatMulzeros:output:0,lstm_cell_24/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
lstm_cell_24/MatMul_1 
lstm_cell_24/addAddV2lstm_cell_24/MatMul:product:0lstm_cell_24/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
lstm_cell_24/add´
#lstm_cell_24/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_24_biasadd_readvariableop_resource*
_output_shapes	
:Ð*
dtype02%
#lstm_cell_24/BiasAdd/ReadVariableOp­
lstm_cell_24/BiasAddBiasAddlstm_cell_24/add:z:0+lstm_cell_24/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
lstm_cell_24/BiasAdd~
lstm_cell_24/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_24/split/split_dim÷
lstm_cell_24/splitSplit%lstm_cell_24/split/split_dim:output:0lstm_cell_24/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿô:ÿÿÿÿÿÿÿÿÿô:ÿÿÿÿÿÿÿÿÿô:ÿÿÿÿÿÿÿÿÿô*
	num_split2
lstm_cell_24/split
lstm_cell_24/SigmoidSigmoidlstm_cell_24/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2
lstm_cell_24/Sigmoid
lstm_cell_24/Sigmoid_1Sigmoidlstm_cell_24/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2
lstm_cell_24/Sigmoid_1
lstm_cell_24/mulMullstm_cell_24/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2
lstm_cell_24/mul~
lstm_cell_24/ReluRelulstm_cell_24/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2
lstm_cell_24/Relu
lstm_cell_24/mul_1Mullstm_cell_24/Sigmoid:y:0lstm_cell_24/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2
lstm_cell_24/mul_1
lstm_cell_24/add_1AddV2lstm_cell_24/mul:z:0lstm_cell_24/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2
lstm_cell_24/add_1
lstm_cell_24/Sigmoid_2Sigmoidlstm_cell_24/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2
lstm_cell_24/Sigmoid_2}
lstm_cell_24/Relu_1Relulstm_cell_24/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2
lstm_cell_24/Relu_1¡
lstm_cell_24/mul_2Mullstm_cell_24/Sigmoid_2:y:0!lstm_cell_24/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2
lstm_cell_24/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿô  2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_24_matmul_readvariableop_resource-lstm_cell_24_matmul_1_readvariableop_resource,lstm_cell_24_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿô:ÿÿÿÿÿÿÿÿÿô: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_40077176*
condR
while_cond_40077175*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿô:ÿÿÿÿÿÿÿÿÿô: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿô  22
0TensorArrayV2Stack/TensorListStack/element_shapeé
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿô*
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
:ÿÿÿÿÿÿÿÿÿô*
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
:ÿÿÿÿÿÿÿÿÿô2
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
:ÿÿÿÿÿÿÿÿÿô2

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
à
º
*__inference_lstm_24_layer_call_fn_40079402
inputs_0
unknown:	]Ð
	unknown_0:
ôÐ
	unknown_1:	Ð
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿô*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_lstm_24_layer_call_and_return_conditional_losses_400762112
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿô2

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
J__inference_lstm_cell_24_layer_call_and_return_conditional_losses_40075918

inputs

states
states_11
matmul_readvariableop_resource:	]Ð4
 matmul_1_readvariableop_resource:
ôÐ.
biasadd_readvariableop_resource:	Ð
identity

identity_1

identity_2¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	]Ð*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
MatMul
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
ôÐ*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
add
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:Ð*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2	
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
P:ÿÿÿÿÿÿÿÿÿô:ÿÿÿÿÿÿÿÿÿô:ÿÿÿÿÿÿÿÿÿô:ÿÿÿÿÿÿÿÿÿô*
	num_split2
split`
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2	
Sigmoidd
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2
	Sigmoid_1]
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2
mulW
ReluRelusplit:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2
Relui
mul_1MulSigmoid:y:0Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2
mul_1^
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2
add_1d
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2
	Sigmoid_2V
Relu_1Relu	add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2
Relu_1m
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2
mul_2e
IdentityIdentity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2

Identityi

Identity_1Identity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2

Identity_1i

Identity_2Identity	add_1:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2

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
A:ÿÿÿÿÿÿÿÿÿ]:ÿÿÿÿÿÿÿÿÿô:ÿÿÿÿÿÿÿÿÿô: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
 
_user_specified_nameinputs:PL
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô
 
_user_specified_namestates:PL
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô
 
_user_specified_namestates
ã
Í
while_cond_40076141
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_40076141___redundant_placeholder06
2while_while_cond_40076141___redundant_placeholder16
2while_while_cond_40076141___redundant_placeholder26
2while_while_cond_40076141___redundant_placeholder3
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
B: : : : :ÿÿÿÿÿÿÿÿÿô:ÿÿÿÿÿÿÿÿÿô: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô:

_output_shapes
: :

_output_shapes
:
&
ó
while_body_40076142
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_00
while_lstm_cell_24_40076166_0:	]Ð1
while_lstm_cell_24_40076168_0:
ôÐ,
while_lstm_cell_24_40076170_0:	Ð
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor.
while_lstm_cell_24_40076166:	]Ð/
while_lstm_cell_24_40076168:
ôÐ*
while_lstm_cell_24_40076170:	Ð¢*while/lstm_cell_24/StatefulPartitionedCallÃ
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
*while/lstm_cell_24/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_24_40076166_0while_lstm_cell_24_40076168_0while_lstm_cell_24_40076170_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿô:ÿÿÿÿÿÿÿÿÿô:ÿÿÿÿÿÿÿÿÿô*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_lstm_cell_24_layer_call_and_return_conditional_losses_400760642,
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
while/Identity_3¥
while/Identity_4Identity3while/lstm_cell_24/StatefulPartitionedCall:output:1^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2
while/Identity_4¥
while/Identity_5Identity3while/lstm_cell_24/StatefulPartitionedCall:output:2^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2
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
while_lstm_cell_24_40076166while_lstm_cell_24_40076166_0"<
while_lstm_cell_24_40076168while_lstm_cell_24_40076168_0"<
while_lstm_cell_24_40076170while_lstm_cell_24_40076170_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿô:ÿÿÿÿÿÿÿÿÿô: : : : : 2X
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
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô:

_output_shapes
: :

_output_shapes
: 
Ç
ù
/__inference_lstm_cell_24_layer_call_fn_40080247

inputs
states_0
states_1
unknown:	]Ð
	unknown_0:
ôÐ
	unknown_1:	Ð
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
<:ÿÿÿÿÿÿÿÿÿô:ÿÿÿÿÿÿÿÿÿô:ÿÿÿÿÿÿÿÿÿô*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_lstm_cell_24_layer_call_and_return_conditional_losses_400759182
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2

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
A:ÿÿÿÿÿÿÿÿÿ]:ÿÿÿÿÿÿÿÿÿô:ÿÿÿÿÿÿÿÿÿô: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
 
_user_specified_nameinputs:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô
"
_user_specified_name
states/0:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô
"
_user_specified_name
states/1


J__inference_lstm_cell_24_layer_call_and_return_conditional_losses_40076064

inputs

states
states_11
matmul_readvariableop_resource:	]Ð4
 matmul_1_readvariableop_resource:
ôÐ.
biasadd_readvariableop_resource:	Ð
identity

identity_1

identity_2¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	]Ð*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
MatMul
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
ôÐ*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
add
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:Ð*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2	
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
P:ÿÿÿÿÿÿÿÿÿô:ÿÿÿÿÿÿÿÿÿô:ÿÿÿÿÿÿÿÿÿô:ÿÿÿÿÿÿÿÿÿô*
	num_split2
split`
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2	
Sigmoidd
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2
	Sigmoid_1]
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2
mulW
ReluRelusplit:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2
Relui
mul_1MulSigmoid:y:0Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2
mul_1^
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2
add_1d
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2
	Sigmoid_2V
Relu_1Relu	add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2
Relu_1m
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2
mul_2e
IdentityIdentity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2

Identityi

Identity_1Identity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2

Identity_1i

Identity_2Identity	add_1:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2

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
A:ÿÿÿÿÿÿÿÿÿ]:ÿÿÿÿÿÿÿÿÿô:ÿÿÿÿÿÿÿÿÿô: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
 
_user_specified_nameinputs:PL
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô
 
_user_specified_namestates:PL
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô
 
_user_specified_namestates
ß
Í
while_cond_40079970
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_40079970___redundant_placeholder06
2while_while_cond_40079970___redundant_placeholder16
2while_while_cond_40079970___redundant_placeholder26
2while_while_cond_40079970___redundant_placeholder3
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
@: : : : :ÿÿÿÿÿÿÿÿÿ8:ÿÿÿÿÿÿÿÿÿ8: ::::: 

_output_shapes
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
:ÿÿÿÿÿÿÿÿÿ8:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ8:

_output_shapes
: :

_output_shapes
:

f
H__inference_dropout_24_layer_call_and_return_conditional_losses_40079429

inputs

identity_1_
IdentityIdentityinputs*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2

Identityn

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿô:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿô
 
_user_specified_nameinputs
Ï
g
H__inference_dropout_25_layer_call_and_return_conditional_losses_40080116

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
:ÿÿÿÿÿÿÿÿÿ82
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape¸
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ8*
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
:ÿÿÿÿÿÿÿÿÿ82
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82
dropout/Cast~
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82
dropout/Mul_1i
IdentityIdentitydropout/Mul_1:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ8:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ8
 
_user_specified_nameinputs
ã
Í
while_cond_40078993
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_40078993___redundant_placeholder06
2while_while_cond_40078993___redundant_placeholder16
2while_while_cond_40078993___redundant_placeholder26
2while_while_cond_40078993___redundant_placeholder3
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
B: : : : :ÿÿÿÿÿÿÿÿÿô:ÿÿÿÿÿÿÿÿÿô: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô:

_output_shapes
: :

_output_shapes
:
í[

E__inference_lstm_25_layer_call_and_return_conditional_losses_40077694

inputs?
+lstm_cell_25_matmul_readvariableop_resource:
ôà@
-lstm_cell_25_matmul_1_readvariableop_resource:	8à;
,lstm_cell_25_biasadd_readvariableop_resource:	à
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
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :82
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
value	B :82
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
:ÿÿÿÿÿÿÿÿÿ82
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :82
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
value	B :82
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
:ÿÿÿÿÿÿÿÿÿ82	
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
:ÿÿÿÿÿÿÿÿÿô2
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
valueB"ÿÿÿÿô  27
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
:ÿÿÿÿÿÿÿÿÿô*
shrink_axis_mask2
strided_slice_2¶
"lstm_cell_25/MatMul/ReadVariableOpReadVariableOp+lstm_cell_25_matmul_readvariableop_resource* 
_output_shapes
:
ôà*
dtype02$
"lstm_cell_25/MatMul/ReadVariableOp­
lstm_cell_25/MatMulMatMulstrided_slice_2:output:0*lstm_cell_25/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà2
lstm_cell_25/MatMul»
$lstm_cell_25/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_25_matmul_1_readvariableop_resource*
_output_shapes
:	8à*
dtype02&
$lstm_cell_25/MatMul_1/ReadVariableOp©
lstm_cell_25/MatMul_1MatMulzeros:output:0,lstm_cell_25/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà2
lstm_cell_25/MatMul_1 
lstm_cell_25/addAddV2lstm_cell_25/MatMul:product:0lstm_cell_25/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà2
lstm_cell_25/add´
#lstm_cell_25/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_25_biasadd_readvariableop_resource*
_output_shapes	
:à*
dtype02%
#lstm_cell_25/BiasAdd/ReadVariableOp­
lstm_cell_25/BiasAddBiasAddlstm_cell_25/add:z:0+lstm_cell_25/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà2
lstm_cell_25/BiasAdd~
lstm_cell_25/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_25/split/split_dimó
lstm_cell_25/splitSplit%lstm_cell_25/split/split_dim:output:0lstm_cell_25/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ8:ÿÿÿÿÿÿÿÿÿ8:ÿÿÿÿÿÿÿÿÿ8:ÿÿÿÿÿÿÿÿÿ8*
	num_split2
lstm_cell_25/split
lstm_cell_25/SigmoidSigmoidlstm_cell_25/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82
lstm_cell_25/Sigmoid
lstm_cell_25/Sigmoid_1Sigmoidlstm_cell_25/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82
lstm_cell_25/Sigmoid_1
lstm_cell_25/mulMullstm_cell_25/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82
lstm_cell_25/mul}
lstm_cell_25/ReluRelulstm_cell_25/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82
lstm_cell_25/Relu
lstm_cell_25/mul_1Mullstm_cell_25/Sigmoid:y:0lstm_cell_25/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82
lstm_cell_25/mul_1
lstm_cell_25/add_1AddV2lstm_cell_25/mul:z:0lstm_cell_25/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82
lstm_cell_25/add_1
lstm_cell_25/Sigmoid_2Sigmoidlstm_cell_25/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82
lstm_cell_25/Sigmoid_2|
lstm_cell_25/Relu_1Relulstm_cell_25/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82
lstm_cell_25/Relu_1 
lstm_cell_25/mul_2Mullstm_cell_25/Sigmoid_2:y:0!lstm_cell_25/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82
lstm_cell_25/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ8   2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_25_matmul_readvariableop_resource-lstm_cell_25_matmul_1_readvariableop_resource,lstm_cell_25_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ8:ÿÿÿÿÿÿÿÿÿ8: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_40077610*
condR
while_cond_40077609*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ8:ÿÿÿÿÿÿÿÿÿ8: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ8   22
0TensorArrayV2Stack/TensorListStack/element_shapeè
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ8*
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
:ÿÿÿÿÿÿÿÿÿ8*
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
:ÿÿÿÿÿÿÿÿÿ82
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
:ÿÿÿÿÿÿÿÿÿ82

IdentityÈ
NoOpNoOp$^lstm_cell_25/BiasAdd/ReadVariableOp#^lstm_cell_25/MatMul/ReadVariableOp%^lstm_cell_25/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:ÿÿÿÿÿÿÿÿÿô: : : 2J
#lstm_cell_25/BiasAdd/ReadVariableOp#lstm_cell_25/BiasAdd/ReadVariableOp2H
"lstm_cell_25/MatMul/ReadVariableOp"lstm_cell_25/MatMul/ReadVariableOp2L
$lstm_cell_25/MatMul_1/ReadVariableOp$lstm_cell_25/MatMul_1/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿô
 
_user_specified_nameinputs
¹
ö
K__inference_sequential_12_layer_call_and_return_conditional_losses_40078012
lstm_24_input#
lstm_24_40077990:	]Ð$
lstm_24_40077992:
ôÐ
lstm_24_40077994:	Ð$
lstm_25_40077998:
ôà#
lstm_25_40078000:	8à
lstm_25_40078002:	à#
dense_12_40078006:8
dense_12_40078008:
identity¢ dense_12/StatefulPartitionedCall¢lstm_24/StatefulPartitionedCall¢lstm_25/StatefulPartitionedCallµ
lstm_24/StatefulPartitionedCallStatefulPartitionedCalllstm_24_inputlstm_24_40077990lstm_24_40077992lstm_24_40077994*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿô*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_lstm_24_layer_call_and_return_conditional_losses_400772602!
lstm_24/StatefulPartitionedCall
dropout_24/PartitionedCallPartitionedCall(lstm_24/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿô* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_dropout_24_layer_call_and_return_conditional_losses_400772732
dropout_24/PartitionedCallÊ
lstm_25/StatefulPartitionedCallStatefulPartitionedCall#dropout_24/PartitionedCall:output:0lstm_25_40077998lstm_25_40078000lstm_25_40078002*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ8*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_lstm_25_layer_call_and_return_conditional_losses_400774252!
lstm_25/StatefulPartitionedCall
dropout_25/PartitionedCallPartitionedCall(lstm_25/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ8* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_dropout_25_layer_call_and_return_conditional_losses_400774382
dropout_25/PartitionedCall»
 dense_12/StatefulPartitionedCallStatefulPartitionedCall#dropout_25/PartitionedCall:output:0dense_12_40078006dense_12_40078008*
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
F__inference_dense_12_layer_call_and_return_conditional_losses_400774712"
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
ý%
ó
while_body_40076562
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_01
while_lstm_cell_25_40076586_0:
ôà0
while_lstm_cell_25_40076588_0:	8à,
while_lstm_cell_25_40076590_0:	à
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor/
while_lstm_cell_25_40076586:
ôà.
while_lstm_cell_25_40076588:	8à*
while_lstm_cell_25_40076590:	à¢*while/lstm_cell_25/StatefulPartitionedCallÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿô  29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÔ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemé
*while/lstm_cell_25/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_25_40076586_0while_lstm_cell_25_40076588_0while_lstm_cell_25_40076590_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ8:ÿÿÿÿÿÿÿÿÿ8:ÿÿÿÿÿÿÿÿÿ8*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_lstm_cell_25_layer_call_and_return_conditional_losses_400765482,
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
while/Identity_3¤
while/Identity_4Identity3while/lstm_cell_25/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82
while/Identity_4¤
while/Identity_5Identity3while/lstm_cell_25/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82
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
while_lstm_cell_25_40076586while_lstm_cell_25_40076586_0"<
while_lstm_cell_25_40076588while_lstm_cell_25_40076588_0"<
while_lstm_cell_25_40076590while_lstm_cell_25_40076590_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ8:ÿÿÿÿÿÿÿÿÿ8: : : : : 2X
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
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ8:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ8:

_output_shapes
: :

_output_shapes
: 
?
Ô
while_body_40077341
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
3while_lstm_cell_25_matmul_readvariableop_resource_0:
ôàH
5while_lstm_cell_25_matmul_1_readvariableop_resource_0:	8àC
4while_lstm_cell_25_biasadd_readvariableop_resource_0:	à
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
1while_lstm_cell_25_matmul_readvariableop_resource:
ôàF
3while_lstm_cell_25_matmul_1_readvariableop_resource:	8àA
2while_lstm_cell_25_biasadd_readvariableop_resource:	à¢)while/lstm_cell_25/BiasAdd/ReadVariableOp¢(while/lstm_cell_25/MatMul/ReadVariableOp¢*while/lstm_cell_25/MatMul_1/ReadVariableOpÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿô  29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÔ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemÊ
(while/lstm_cell_25/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_25_matmul_readvariableop_resource_0* 
_output_shapes
:
ôà*
dtype02*
(while/lstm_cell_25/MatMul/ReadVariableOp×
while/lstm_cell_25/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_25/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà2
while/lstm_cell_25/MatMulÏ
*while/lstm_cell_25/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_25_matmul_1_readvariableop_resource_0*
_output_shapes
:	8à*
dtype02,
*while/lstm_cell_25/MatMul_1/ReadVariableOpÀ
while/lstm_cell_25/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_25/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà2
while/lstm_cell_25/MatMul_1¸
while/lstm_cell_25/addAddV2#while/lstm_cell_25/MatMul:product:0%while/lstm_cell_25/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà2
while/lstm_cell_25/addÈ
)while/lstm_cell_25/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_25_biasadd_readvariableop_resource_0*
_output_shapes	
:à*
dtype02+
)while/lstm_cell_25/BiasAdd/ReadVariableOpÅ
while/lstm_cell_25/BiasAddBiasAddwhile/lstm_cell_25/add:z:01while/lstm_cell_25/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà2
while/lstm_cell_25/BiasAdd
"while/lstm_cell_25/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_25/split/split_dim
while/lstm_cell_25/splitSplit+while/lstm_cell_25/split/split_dim:output:0#while/lstm_cell_25/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ8:ÿÿÿÿÿÿÿÿÿ8:ÿÿÿÿÿÿÿÿÿ8:ÿÿÿÿÿÿÿÿÿ8*
	num_split2
while/lstm_cell_25/split
while/lstm_cell_25/SigmoidSigmoid!while/lstm_cell_25/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82
while/lstm_cell_25/Sigmoid
while/lstm_cell_25/Sigmoid_1Sigmoid!while/lstm_cell_25/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82
while/lstm_cell_25/Sigmoid_1 
while/lstm_cell_25/mulMul while/lstm_cell_25/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82
while/lstm_cell_25/mul
while/lstm_cell_25/ReluRelu!while/lstm_cell_25/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82
while/lstm_cell_25/Relu´
while/lstm_cell_25/mul_1Mulwhile/lstm_cell_25/Sigmoid:y:0%while/lstm_cell_25/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82
while/lstm_cell_25/mul_1©
while/lstm_cell_25/add_1AddV2while/lstm_cell_25/mul:z:0while/lstm_cell_25/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82
while/lstm_cell_25/add_1
while/lstm_cell_25/Sigmoid_2Sigmoid!while/lstm_cell_25/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82
while/lstm_cell_25/Sigmoid_2
while/lstm_cell_25/Relu_1Reluwhile/lstm_cell_25/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82
while/lstm_cell_25/Relu_1¸
while/lstm_cell_25/mul_2Mul while/lstm_cell_25/Sigmoid_2:y:0'while/lstm_cell_25/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82
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
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_25/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_25/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82
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
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ8:ÿÿÿÿÿÿÿÿÿ8: : : : : 2V
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
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ8:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ8:

_output_shapes
: :

_output_shapes
: 
Ï
g
H__inference_dropout_25_layer_call_and_return_conditional_losses_40077527

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
:ÿÿÿÿÿÿÿÿÿ82
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape¸
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ8*
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
:ÿÿÿÿÿÿÿÿÿ82
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82
dropout/Cast~
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82
dropout/Mul_1i
IdentityIdentitydropout/Mul_1:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ8:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ8
 
_user_specified_nameinputs
¬\
 
E__inference_lstm_25_layer_call_and_return_conditional_losses_40079602
inputs_0?
+lstm_cell_25_matmul_readvariableop_resource:
ôà@
-lstm_cell_25_matmul_1_readvariableop_resource:	8à;
,lstm_cell_25_biasadd_readvariableop_resource:	à
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
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :82
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
value	B :82
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
:ÿÿÿÿÿÿÿÿÿ82
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :82
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
value	B :82
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
:ÿÿÿÿÿÿÿÿÿ82	
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
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿô2
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
valueB"ÿÿÿÿô  27
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
:ÿÿÿÿÿÿÿÿÿô*
shrink_axis_mask2
strided_slice_2¶
"lstm_cell_25/MatMul/ReadVariableOpReadVariableOp+lstm_cell_25_matmul_readvariableop_resource* 
_output_shapes
:
ôà*
dtype02$
"lstm_cell_25/MatMul/ReadVariableOp­
lstm_cell_25/MatMulMatMulstrided_slice_2:output:0*lstm_cell_25/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà2
lstm_cell_25/MatMul»
$lstm_cell_25/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_25_matmul_1_readvariableop_resource*
_output_shapes
:	8à*
dtype02&
$lstm_cell_25/MatMul_1/ReadVariableOp©
lstm_cell_25/MatMul_1MatMulzeros:output:0,lstm_cell_25/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà2
lstm_cell_25/MatMul_1 
lstm_cell_25/addAddV2lstm_cell_25/MatMul:product:0lstm_cell_25/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà2
lstm_cell_25/add´
#lstm_cell_25/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_25_biasadd_readvariableop_resource*
_output_shapes	
:à*
dtype02%
#lstm_cell_25/BiasAdd/ReadVariableOp­
lstm_cell_25/BiasAddBiasAddlstm_cell_25/add:z:0+lstm_cell_25/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà2
lstm_cell_25/BiasAdd~
lstm_cell_25/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_25/split/split_dimó
lstm_cell_25/splitSplit%lstm_cell_25/split/split_dim:output:0lstm_cell_25/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ8:ÿÿÿÿÿÿÿÿÿ8:ÿÿÿÿÿÿÿÿÿ8:ÿÿÿÿÿÿÿÿÿ8*
	num_split2
lstm_cell_25/split
lstm_cell_25/SigmoidSigmoidlstm_cell_25/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82
lstm_cell_25/Sigmoid
lstm_cell_25/Sigmoid_1Sigmoidlstm_cell_25/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82
lstm_cell_25/Sigmoid_1
lstm_cell_25/mulMullstm_cell_25/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82
lstm_cell_25/mul}
lstm_cell_25/ReluRelulstm_cell_25/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82
lstm_cell_25/Relu
lstm_cell_25/mul_1Mullstm_cell_25/Sigmoid:y:0lstm_cell_25/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82
lstm_cell_25/mul_1
lstm_cell_25/add_1AddV2lstm_cell_25/mul:z:0lstm_cell_25/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82
lstm_cell_25/add_1
lstm_cell_25/Sigmoid_2Sigmoidlstm_cell_25/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82
lstm_cell_25/Sigmoid_2|
lstm_cell_25/Relu_1Relulstm_cell_25/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82
lstm_cell_25/Relu_1 
lstm_cell_25/mul_2Mullstm_cell_25/Sigmoid_2:y:0!lstm_cell_25/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82
lstm_cell_25/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ8   2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_25_matmul_readvariableop_resource-lstm_cell_25_matmul_1_readvariableop_resource,lstm_cell_25_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ8:ÿÿÿÿÿÿÿÿÿ8: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_40079518*
condR
while_cond_40079517*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ8:ÿÿÿÿÿÿÿÿÿ8: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ8   22
0TensorArrayV2Stack/TensorListStack/element_shapeñ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ8*
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
:ÿÿÿÿÿÿÿÿÿ8*
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
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ82
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
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ82

IdentityÈ
NoOpNoOp$^lstm_cell_25/BiasAdd/ReadVariableOp#^lstm_cell_25/MatMul/ReadVariableOp%^lstm_cell_25/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿô: : : 2J
#lstm_cell_25/BiasAdd/ReadVariableOp#lstm_cell_25/BiasAdd/ReadVariableOp2H
"lstm_cell_25/MatMul/ReadVariableOp"lstm_cell_25/MatMul/ReadVariableOp2L
$lstm_cell_25/MatMul_1/ReadVariableOp$lstm_cell_25/MatMul_1/ReadVariableOp2
whilewhile:_ [
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿô
"
_user_specified_name
inputs/0
Ì

$__inference__traced_restore_40080593
file_prefix2
 assignvariableop_dense_12_kernel:8.
 assignvariableop_1_dense_12_bias:&
assignvariableop_2_adam_iter:	 (
assignvariableop_3_adam_beta_1: (
assignvariableop_4_adam_beta_2: '
assignvariableop_5_adam_decay: /
%assignvariableop_6_adam_learning_rate: A
.assignvariableop_7_lstm_24_lstm_cell_24_kernel:	]ÐL
8assignvariableop_8_lstm_24_lstm_cell_24_recurrent_kernel:
ôÐ;
,assignvariableop_9_lstm_24_lstm_cell_24_bias:	ÐC
/assignvariableop_10_lstm_25_lstm_cell_25_kernel:
ôàL
9assignvariableop_11_lstm_25_lstm_cell_25_recurrent_kernel:	8à<
-assignvariableop_12_lstm_25_lstm_cell_25_bias:	à#
assignvariableop_13_total: #
assignvariableop_14_count: %
assignvariableop_15_total_1: %
assignvariableop_16_count_1: <
*assignvariableop_17_adam_dense_12_kernel_m:86
(assignvariableop_18_adam_dense_12_bias_m:I
6assignvariableop_19_adam_lstm_24_lstm_cell_24_kernel_m:	]ÐT
@assignvariableop_20_adam_lstm_24_lstm_cell_24_recurrent_kernel_m:
ôÐC
4assignvariableop_21_adam_lstm_24_lstm_cell_24_bias_m:	ÐJ
6assignvariableop_22_adam_lstm_25_lstm_cell_25_kernel_m:
ôàS
@assignvariableop_23_adam_lstm_25_lstm_cell_25_recurrent_kernel_m:	8àC
4assignvariableop_24_adam_lstm_25_lstm_cell_25_bias_m:	à<
*assignvariableop_25_adam_dense_12_kernel_v:86
(assignvariableop_26_adam_dense_12_bias_v:I
6assignvariableop_27_adam_lstm_24_lstm_cell_24_kernel_v:	]ÐT
@assignvariableop_28_adam_lstm_24_lstm_cell_24_recurrent_kernel_v:
ôÐC
4assignvariableop_29_adam_lstm_24_lstm_cell_24_bias_v:	ÐJ
6assignvariableop_30_adam_lstm_25_lstm_cell_25_kernel_v:
ôàS
@assignvariableop_31_adam_lstm_25_lstm_cell_25_recurrent_kernel_v:	8àC
4assignvariableop_32_adam_lstm_25_lstm_cell_25_bias_v:	à
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
à
º
*__inference_lstm_24_layer_call_fn_40079391
inputs_0
unknown:	]Ð
	unknown_0:
ôÐ
	unknown_1:	Ð
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿô*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_lstm_24_layer_call_and_return_conditional_losses_400760012
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿô2

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
while_body_40079296
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_24_matmul_readvariableop_resource_0:	]ÐI
5while_lstm_cell_24_matmul_1_readvariableop_resource_0:
ôÐC
4while_lstm_cell_24_biasadd_readvariableop_resource_0:	Ð
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_24_matmul_readvariableop_resource:	]ÐG
3while_lstm_cell_24_matmul_1_readvariableop_resource:
ôÐA
2while_lstm_cell_24_biasadd_readvariableop_resource:	Ð¢)while/lstm_cell_24/BiasAdd/ReadVariableOp¢(while/lstm_cell_24/MatMul/ReadVariableOp¢*while/lstm_cell_24/MatMul_1/ReadVariableOpÃ
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
:	]Ð*
dtype02*
(while/lstm_cell_24/MatMul/ReadVariableOp×
while/lstm_cell_24/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_24/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
while/lstm_cell_24/MatMulÐ
*while/lstm_cell_24/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_24_matmul_1_readvariableop_resource_0* 
_output_shapes
:
ôÐ*
dtype02,
*while/lstm_cell_24/MatMul_1/ReadVariableOpÀ
while/lstm_cell_24/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_24/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
while/lstm_cell_24/MatMul_1¸
while/lstm_cell_24/addAddV2#while/lstm_cell_24/MatMul:product:0%while/lstm_cell_24/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
while/lstm_cell_24/addÈ
)while/lstm_cell_24/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_24_biasadd_readvariableop_resource_0*
_output_shapes	
:Ð*
dtype02+
)while/lstm_cell_24/BiasAdd/ReadVariableOpÅ
while/lstm_cell_24/BiasAddBiasAddwhile/lstm_cell_24/add:z:01while/lstm_cell_24/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
while/lstm_cell_24/BiasAdd
"while/lstm_cell_24/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_24/split/split_dim
while/lstm_cell_24/splitSplit+while/lstm_cell_24/split/split_dim:output:0#while/lstm_cell_24/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿô:ÿÿÿÿÿÿÿÿÿô:ÿÿÿÿÿÿÿÿÿô:ÿÿÿÿÿÿÿÿÿô*
	num_split2
while/lstm_cell_24/split
while/lstm_cell_24/SigmoidSigmoid!while/lstm_cell_24/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2
while/lstm_cell_24/Sigmoid
while/lstm_cell_24/Sigmoid_1Sigmoid!while/lstm_cell_24/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2
while/lstm_cell_24/Sigmoid_1¡
while/lstm_cell_24/mulMul while/lstm_cell_24/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2
while/lstm_cell_24/mul
while/lstm_cell_24/ReluRelu!while/lstm_cell_24/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2
while/lstm_cell_24/Reluµ
while/lstm_cell_24/mul_1Mulwhile/lstm_cell_24/Sigmoid:y:0%while/lstm_cell_24/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2
while/lstm_cell_24/mul_1ª
while/lstm_cell_24/add_1AddV2while/lstm_cell_24/mul:z:0while/lstm_cell_24/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2
while/lstm_cell_24/add_1
while/lstm_cell_24/Sigmoid_2Sigmoid!while/lstm_cell_24/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2
while/lstm_cell_24/Sigmoid_2
while/lstm_cell_24/Relu_1Reluwhile/lstm_cell_24/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2
while/lstm_cell_24/Relu_1¹
while/lstm_cell_24/mul_2Mul while/lstm_cell_24/Sigmoid_2:y:0'while/lstm_cell_24/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2
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
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_24/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_24/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2
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
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿô:ÿÿÿÿÿÿÿÿÿô: : : : : 2V
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
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô:

_output_shapes
: :

_output_shapes
: 
ý%
ó
while_body_40076772
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_01
while_lstm_cell_25_40076796_0:
ôà0
while_lstm_cell_25_40076798_0:	8à,
while_lstm_cell_25_40076800_0:	à
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor/
while_lstm_cell_25_40076796:
ôà.
while_lstm_cell_25_40076798:	8à*
while_lstm_cell_25_40076800:	à¢*while/lstm_cell_25/StatefulPartitionedCallÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿô  29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÔ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemé
*while/lstm_cell_25/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_25_40076796_0while_lstm_cell_25_40076798_0while_lstm_cell_25_40076800_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ8:ÿÿÿÿÿÿÿÿÿ8:ÿÿÿÿÿÿÿÿÿ8*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_lstm_cell_25_layer_call_and_return_conditional_losses_400766942,
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
while/Identity_3¤
while/Identity_4Identity3while/lstm_cell_25/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82
while/Identity_4¤
while/Identity_5Identity3while/lstm_cell_25/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82
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
while_lstm_cell_25_40076796while_lstm_cell_25_40076796_0"<
while_lstm_cell_25_40076798while_lstm_cell_25_40076798_0"<
while_lstm_cell_25_40076800while_lstm_cell_25_40076800_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ8:ÿÿÿÿÿÿÿÿÿ8: : : : : 2X
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
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ8:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ8:

_output_shapes
: :

_output_shapes
: 
µ
¸
*__inference_lstm_25_layer_call_fn_40080099

inputs
unknown:
ôà
	unknown_0:	8à
	unknown_1:	à
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ8*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_lstm_25_layer_call_and_return_conditional_losses_400776942
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:ÿÿÿÿÿÿÿÿÿô: : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿô
 
_user_specified_nameinputs


J__inference_lstm_cell_25_layer_call_and_return_conditional_losses_40080328

inputs
states_0
states_12
matmul_readvariableop_resource:
ôà3
 matmul_1_readvariableop_resource:	8à.
biasadd_readvariableop_resource:	à
identity

identity_1

identity_2¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
ôà*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà2
MatMul
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	8à*
dtype02
MatMul_1/ReadVariableOp|
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà2
add
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:à*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà2	
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
L:ÿÿÿÿÿÿÿÿÿ8:ÿÿÿÿÿÿÿÿÿ8:ÿÿÿÿÿÿÿÿÿ8:ÿÿÿÿÿÿÿÿÿ8*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82
	Sigmoid_1\
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82
mulV
ReluRelusplit:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82
	Sigmoid_2U
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82
mul_2d
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82

Identityh

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82

Identity_1h

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82

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
@:ÿÿÿÿÿÿÿÿÿô:ÿÿÿÿÿÿÿÿÿ8:ÿÿÿÿÿÿÿÿÿ8: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô
 
_user_specified_nameinputs:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ8
"
_user_specified_name
states/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ8
"
_user_specified_name
states/1
ã
Í
while_cond_40077175
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_40077175___redundant_placeholder06
2while_while_cond_40077175___redundant_placeholder16
2while_while_cond_40077175___redundant_placeholder26
2while_while_cond_40077175___redundant_placeholder3
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
B: : : : :ÿÿÿÿÿÿÿÿÿô:ÿÿÿÿÿÿÿÿÿô: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô:

_output_shapes
: :

_output_shapes
:
Æ
À
K__inference_sequential_12_layer_call_and_return_conditional_losses_40078037
lstm_24_input#
lstm_24_40078015:	]Ð$
lstm_24_40078017:
ôÐ
lstm_24_40078019:	Ð$
lstm_25_40078023:
ôà#
lstm_25_40078025:	8à
lstm_25_40078027:	à#
dense_12_40078031:8
dense_12_40078033:
identity¢ dense_12/StatefulPartitionedCall¢"dropout_24/StatefulPartitionedCall¢"dropout_25/StatefulPartitionedCall¢lstm_24/StatefulPartitionedCall¢lstm_25/StatefulPartitionedCallµ
lstm_24/StatefulPartitionedCallStatefulPartitionedCalllstm_24_inputlstm_24_40078015lstm_24_40078017lstm_24_40078019*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿô*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_lstm_24_layer_call_and_return_conditional_losses_400778902!
lstm_24/StatefulPartitionedCall
"dropout_24/StatefulPartitionedCallStatefulPartitionedCall(lstm_24/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿô* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_dropout_24_layer_call_and_return_conditional_losses_400777232$
"dropout_24/StatefulPartitionedCallÒ
lstm_25/StatefulPartitionedCallStatefulPartitionedCall+dropout_24/StatefulPartitionedCall:output:0lstm_25_40078023lstm_25_40078025lstm_25_40078027*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ8*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_lstm_25_layer_call_and_return_conditional_losses_400776942!
lstm_25/StatefulPartitionedCall¿
"dropout_25/StatefulPartitionedCallStatefulPartitionedCall(lstm_25/StatefulPartitionedCall:output:0#^dropout_24/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ8* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_dropout_25_layer_call_and_return_conditional_losses_400775272$
"dropout_25/StatefulPartitionedCallÃ
 dense_12/StatefulPartitionedCallStatefulPartitionedCall+dropout_25/StatefulPartitionedCall:output:0dense_12_40078031dense_12_40078033*
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
F__inference_dense_12_layer_call_and_return_conditional_losses_400774712"
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
ËF

E__inference_lstm_24_layer_call_and_return_conditional_losses_40076211

inputs(
lstm_cell_24_40076129:	]Ð)
lstm_cell_24_40076131:
ôÐ$
lstm_cell_24_40076133:	Ð
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
strided_slice]
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :ô2
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
B :ô2
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
:ÿÿÿÿÿÿÿÿÿô2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :ô2
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
B :ô2
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
:ÿÿÿÿÿÿÿÿÿô2	
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
$lstm_cell_24/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_24_40076129lstm_cell_24_40076131lstm_cell_24_40076133*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿô:ÿÿÿÿÿÿÿÿÿô:ÿÿÿÿÿÿÿÿÿô*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_lstm_cell_24_layer_call_and_return_conditional_losses_400760642&
$lstm_cell_24/StatefulPartitionedCall
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿô  2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_24_40076129lstm_cell_24_40076131lstm_cell_24_40076133*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿô:ÿÿÿÿÿÿÿÿÿô: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_40076142*
condR
while_cond_40076141*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿô:ÿÿÿÿÿÿÿÿÿô: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿô  22
0TensorArrayV2Stack/TensorListStack/element_shapeò
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿô*
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
:ÿÿÿÿÿÿÿÿÿô*
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
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿô2
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
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿô2

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
½
ù
/__inference_lstm_cell_25_layer_call_fn_40080345

inputs
states_0
states_1
unknown:
ôà
	unknown_0:	8à
	unknown_1:	à
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
9:ÿÿÿÿÿÿÿÿÿ8:ÿÿÿÿÿÿÿÿÿ8:ÿÿÿÿÿÿÿÿÿ8*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_lstm_cell_25_layer_call_and_return_conditional_losses_400765482
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82

Identity

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82

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
@:ÿÿÿÿÿÿÿÿÿô:ÿÿÿÿÿÿÿÿÿ8:ÿÿÿÿÿÿÿÿÿ8: : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô
 
_user_specified_nameinputs:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ8
"
_user_specified_name
states/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ8
"
_user_specified_name
states/1


J__inference_lstm_cell_25_layer_call_and_return_conditional_losses_40080296

inputs
states_0
states_12
matmul_readvariableop_resource:
ôà3
 matmul_1_readvariableop_resource:	8à.
biasadd_readvariableop_resource:	à
identity

identity_1

identity_2¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
ôà*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà2
MatMul
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	8à*
dtype02
MatMul_1/ReadVariableOp|
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà2
add
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:à*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà2	
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
L:ÿÿÿÿÿÿÿÿÿ8:ÿÿÿÿÿÿÿÿÿ8:ÿÿÿÿÿÿÿÿÿ8:ÿÿÿÿÿÿÿÿÿ8*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82
	Sigmoid_1\
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82
mulV
ReluRelusplit:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82
	Sigmoid_2U
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82
mul_2d
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82

Identityh

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82

Identity_1h

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82

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
@:ÿÿÿÿÿÿÿÿÿô:ÿÿÿÿÿÿÿÿÿ8:ÿÿÿÿÿÿÿÿÿ8: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô
 
_user_specified_nameinputs:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ8
"
_user_specified_name
states/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ8
"
_user_specified_name
states/1
µ
¸
*__inference_lstm_25_layer_call_fn_40080088

inputs
unknown:
ôà
	unknown_0:	8à
	unknown_1:	à
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ8*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_lstm_25_layer_call_and_return_conditional_losses_400774252
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:ÿÿÿÿÿÿÿÿÿô: : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿô
 
_user_specified_nameinputs


J__inference_lstm_cell_24_layer_call_and_return_conditional_losses_40080198

inputs
states_0
states_11
matmul_readvariableop_resource:	]Ð4
 matmul_1_readvariableop_resource:
ôÐ.
biasadd_readvariableop_resource:	Ð
identity

identity_1

identity_2¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	]Ð*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
MatMul
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
ôÐ*
dtype02
MatMul_1/ReadVariableOp|
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
add
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:Ð*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2	
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
P:ÿÿÿÿÿÿÿÿÿô:ÿÿÿÿÿÿÿÿÿô:ÿÿÿÿÿÿÿÿÿô:ÿÿÿÿÿÿÿÿÿô*
	num_split2
split`
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2	
Sigmoidd
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2
	Sigmoid_1]
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2
mulW
ReluRelusplit:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2
Relui
mul_1MulSigmoid:y:0Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2
mul_1^
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2
add_1d
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2
	Sigmoid_2V
Relu_1Relu	add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2
Relu_1m
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2
mul_2e
IdentityIdentity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2

Identityi

Identity_1Identity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2

Identity_1i

Identity_2Identity	add_1:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2

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
A:ÿÿÿÿÿÿÿÿÿ]:ÿÿÿÿÿÿÿÿÿô:ÿÿÿÿÿÿÿÿÿô: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
 
_user_specified_nameinputs:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô
"
_user_specified_name
states/0:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô
"
_user_specified_name
states/1
ß
Í
while_cond_40076771
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_40076771___redundant_placeholder06
2while_while_cond_40076771___redundant_placeholder16
2while_while_cond_40076771___redundant_placeholder26
2while_while_cond_40076771___redundant_placeholder3
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
@: : : : :ÿÿÿÿÿÿÿÿÿ8:ÿÿÿÿÿÿÿÿÿ8: ::::: 

_output_shapes
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
:ÿÿÿÿÿÿÿÿÿ8:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ8:

_output_shapes
: :

_output_shapes
:
ß
Í
while_cond_40079819
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_40079819___redundant_placeholder06
2while_while_cond_40079819___redundant_placeholder16
2while_while_cond_40079819___redundant_placeholder26
2while_while_cond_40079819___redundant_placeholder3
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
@: : : : :ÿÿÿÿÿÿÿÿÿ8:ÿÿÿÿÿÿÿÿÿ8: ::::: 

_output_shapes
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
:ÿÿÿÿÿÿÿÿÿ8:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ8:

_output_shapes
: :

_output_shapes
:
\

E__inference_lstm_24_layer_call_and_return_conditional_losses_40077890

inputs>
+lstm_cell_24_matmul_readvariableop_resource:	]ÐA
-lstm_cell_24_matmul_1_readvariableop_resource:
ôÐ;
,lstm_cell_24_biasadd_readvariableop_resource:	Ð
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
strided_slice]
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :ô2
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
B :ô2
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
:ÿÿÿÿÿÿÿÿÿô2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :ô2
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
B :ô2
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
:ÿÿÿÿÿÿÿÿÿô2	
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
:	]Ð*
dtype02$
"lstm_cell_24/MatMul/ReadVariableOp­
lstm_cell_24/MatMulMatMulstrided_slice_2:output:0*lstm_cell_24/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
lstm_cell_24/MatMul¼
$lstm_cell_24/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_24_matmul_1_readvariableop_resource* 
_output_shapes
:
ôÐ*
dtype02&
$lstm_cell_24/MatMul_1/ReadVariableOp©
lstm_cell_24/MatMul_1MatMulzeros:output:0,lstm_cell_24/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
lstm_cell_24/MatMul_1 
lstm_cell_24/addAddV2lstm_cell_24/MatMul:product:0lstm_cell_24/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
lstm_cell_24/add´
#lstm_cell_24/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_24_biasadd_readvariableop_resource*
_output_shapes	
:Ð*
dtype02%
#lstm_cell_24/BiasAdd/ReadVariableOp­
lstm_cell_24/BiasAddBiasAddlstm_cell_24/add:z:0+lstm_cell_24/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
lstm_cell_24/BiasAdd~
lstm_cell_24/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_24/split/split_dim÷
lstm_cell_24/splitSplit%lstm_cell_24/split/split_dim:output:0lstm_cell_24/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿô:ÿÿÿÿÿÿÿÿÿô:ÿÿÿÿÿÿÿÿÿô:ÿÿÿÿÿÿÿÿÿô*
	num_split2
lstm_cell_24/split
lstm_cell_24/SigmoidSigmoidlstm_cell_24/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2
lstm_cell_24/Sigmoid
lstm_cell_24/Sigmoid_1Sigmoidlstm_cell_24/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2
lstm_cell_24/Sigmoid_1
lstm_cell_24/mulMullstm_cell_24/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2
lstm_cell_24/mul~
lstm_cell_24/ReluRelulstm_cell_24/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2
lstm_cell_24/Relu
lstm_cell_24/mul_1Mullstm_cell_24/Sigmoid:y:0lstm_cell_24/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2
lstm_cell_24/mul_1
lstm_cell_24/add_1AddV2lstm_cell_24/mul:z:0lstm_cell_24/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2
lstm_cell_24/add_1
lstm_cell_24/Sigmoid_2Sigmoidlstm_cell_24/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2
lstm_cell_24/Sigmoid_2}
lstm_cell_24/Relu_1Relulstm_cell_24/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2
lstm_cell_24/Relu_1¡
lstm_cell_24/mul_2Mullstm_cell_24/Sigmoid_2:y:0!lstm_cell_24/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2
lstm_cell_24/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿô  2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_24_matmul_readvariableop_resource-lstm_cell_24_matmul_1_readvariableop_resource,lstm_cell_24_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿô:ÿÿÿÿÿÿÿÿÿô: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_40077806*
condR
while_cond_40077805*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿô:ÿÿÿÿÿÿÿÿÿô: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿô  22
0TensorArrayV2Stack/TensorListStack/element_shapeé
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿô*
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
:ÿÿÿÿÿÿÿÿÿô*
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
:ÿÿÿÿÿÿÿÿÿô2
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
:ÿÿÿÿÿÿÿÿÿô2

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
°?
Ô
while_body_40077806
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_24_matmul_readvariableop_resource_0:	]ÐI
5while_lstm_cell_24_matmul_1_readvariableop_resource_0:
ôÐC
4while_lstm_cell_24_biasadd_readvariableop_resource_0:	Ð
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_24_matmul_readvariableop_resource:	]ÐG
3while_lstm_cell_24_matmul_1_readvariableop_resource:
ôÐA
2while_lstm_cell_24_biasadd_readvariableop_resource:	Ð¢)while/lstm_cell_24/BiasAdd/ReadVariableOp¢(while/lstm_cell_24/MatMul/ReadVariableOp¢*while/lstm_cell_24/MatMul_1/ReadVariableOpÃ
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
:	]Ð*
dtype02*
(while/lstm_cell_24/MatMul/ReadVariableOp×
while/lstm_cell_24/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_24/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
while/lstm_cell_24/MatMulÐ
*while/lstm_cell_24/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_24_matmul_1_readvariableop_resource_0* 
_output_shapes
:
ôÐ*
dtype02,
*while/lstm_cell_24/MatMul_1/ReadVariableOpÀ
while/lstm_cell_24/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_24/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
while/lstm_cell_24/MatMul_1¸
while/lstm_cell_24/addAddV2#while/lstm_cell_24/MatMul:product:0%while/lstm_cell_24/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
while/lstm_cell_24/addÈ
)while/lstm_cell_24/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_24_biasadd_readvariableop_resource_0*
_output_shapes	
:Ð*
dtype02+
)while/lstm_cell_24/BiasAdd/ReadVariableOpÅ
while/lstm_cell_24/BiasAddBiasAddwhile/lstm_cell_24/add:z:01while/lstm_cell_24/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
while/lstm_cell_24/BiasAdd
"while/lstm_cell_24/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_24/split/split_dim
while/lstm_cell_24/splitSplit+while/lstm_cell_24/split/split_dim:output:0#while/lstm_cell_24/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿô:ÿÿÿÿÿÿÿÿÿô:ÿÿÿÿÿÿÿÿÿô:ÿÿÿÿÿÿÿÿÿô*
	num_split2
while/lstm_cell_24/split
while/lstm_cell_24/SigmoidSigmoid!while/lstm_cell_24/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2
while/lstm_cell_24/Sigmoid
while/lstm_cell_24/Sigmoid_1Sigmoid!while/lstm_cell_24/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2
while/lstm_cell_24/Sigmoid_1¡
while/lstm_cell_24/mulMul while/lstm_cell_24/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2
while/lstm_cell_24/mul
while/lstm_cell_24/ReluRelu!while/lstm_cell_24/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2
while/lstm_cell_24/Reluµ
while/lstm_cell_24/mul_1Mulwhile/lstm_cell_24/Sigmoid:y:0%while/lstm_cell_24/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2
while/lstm_cell_24/mul_1ª
while/lstm_cell_24/add_1AddV2while/lstm_cell_24/mul:z:0while/lstm_cell_24/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2
while/lstm_cell_24/add_1
while/lstm_cell_24/Sigmoid_2Sigmoid!while/lstm_cell_24/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2
while/lstm_cell_24/Sigmoid_2
while/lstm_cell_24/Relu_1Reluwhile/lstm_cell_24/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2
while/lstm_cell_24/Relu_1¹
while/lstm_cell_24/mul_2Mul while/lstm_cell_24/Sigmoid_2:y:0'while/lstm_cell_24/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2
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
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_24/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_24/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2
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
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿô:ÿÿÿÿÿÿÿÿÿô: : : : : 2V
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
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô:

_output_shapes
: :

_output_shapes
: 
±
¹
K__inference_sequential_12_layer_call_and_return_conditional_losses_40077947

inputs#
lstm_24_40077925:	]Ð$
lstm_24_40077927:
ôÐ
lstm_24_40077929:	Ð$
lstm_25_40077933:
ôà#
lstm_25_40077935:	8à
lstm_25_40077937:	à#
dense_12_40077941:8
dense_12_40077943:
identity¢ dense_12/StatefulPartitionedCall¢"dropout_24/StatefulPartitionedCall¢"dropout_25/StatefulPartitionedCall¢lstm_24/StatefulPartitionedCall¢lstm_25/StatefulPartitionedCall®
lstm_24/StatefulPartitionedCallStatefulPartitionedCallinputslstm_24_40077925lstm_24_40077927lstm_24_40077929*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿô*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_lstm_24_layer_call_and_return_conditional_losses_400778902!
lstm_24/StatefulPartitionedCall
"dropout_24/StatefulPartitionedCallStatefulPartitionedCall(lstm_24/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿô* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_dropout_24_layer_call_and_return_conditional_losses_400777232$
"dropout_24/StatefulPartitionedCallÒ
lstm_25/StatefulPartitionedCallStatefulPartitionedCall+dropout_24/StatefulPartitionedCall:output:0lstm_25_40077933lstm_25_40077935lstm_25_40077937*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ8*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_lstm_25_layer_call_and_return_conditional_losses_400776942!
lstm_25/StatefulPartitionedCall¿
"dropout_25/StatefulPartitionedCallStatefulPartitionedCall(lstm_25/StatefulPartitionedCall:output:0#^dropout_24/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ8* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_dropout_25_layer_call_and_return_conditional_losses_400775272$
"dropout_25/StatefulPartitionedCallÃ
 dense_12/StatefulPartitionedCallStatefulPartitionedCall+dropout_25/StatefulPartitionedCall:output:0dense_12_40077941dense_12_40077943*
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
F__inference_dense_12_layer_call_and_return_conditional_losses_400774712"
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
ù

)sequential_12_lstm_24_while_cond_40075582H
Dsequential_12_lstm_24_while_sequential_12_lstm_24_while_loop_counterN
Jsequential_12_lstm_24_while_sequential_12_lstm_24_while_maximum_iterations+
'sequential_12_lstm_24_while_placeholder-
)sequential_12_lstm_24_while_placeholder_1-
)sequential_12_lstm_24_while_placeholder_2-
)sequential_12_lstm_24_while_placeholder_3J
Fsequential_12_lstm_24_while_less_sequential_12_lstm_24_strided_slice_1b
^sequential_12_lstm_24_while_sequential_12_lstm_24_while_cond_40075582___redundant_placeholder0b
^sequential_12_lstm_24_while_sequential_12_lstm_24_while_cond_40075582___redundant_placeholder1b
^sequential_12_lstm_24_while_sequential_12_lstm_24_while_cond_40075582___redundant_placeholder2b
^sequential_12_lstm_24_while_sequential_12_lstm_24_while_cond_40075582___redundant_placeholder3(
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
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :ÿÿÿÿÿÿÿÿÿô:ÿÿÿÿÿÿÿÿÿô: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô:

_output_shapes
: :

_output_shapes
:
½
ù
/__inference_lstm_cell_25_layer_call_fn_40080362

inputs
states_0
states_1
unknown:
ôà
	unknown_0:	8à
	unknown_1:	à
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
9:ÿÿÿÿÿÿÿÿÿ8:ÿÿÿÿÿÿÿÿÿ8:ÿÿÿÿÿÿÿÿÿ8*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_lstm_cell_25_layer_call_and_return_conditional_losses_400766942
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82

Identity

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82

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
@:ÿÿÿÿÿÿÿÿÿô:ÿÿÿÿÿÿÿÿÿ8:ÿÿÿÿÿÿÿÿÿ8: : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô
 
_user_specified_nameinputs:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ8
"
_user_specified_name
states/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ8
"
_user_specified_name
states/1
¶
¸
*__inference_lstm_24_layer_call_fn_40079424

inputs
unknown:	]Ð
	unknown_0:
ôÐ
	unknown_1:	Ð
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿô*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_lstm_24_layer_call_and_return_conditional_losses_400778902
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2

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

f
H__inference_dropout_24_layer_call_and_return_conditional_losses_40077273

inputs

identity_1_
IdentityIdentityinputs*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2

Identityn

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿô:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿô
 
_user_specified_nameinputs
¬\
 
E__inference_lstm_25_layer_call_and_return_conditional_losses_40079753
inputs_0?
+lstm_cell_25_matmul_readvariableop_resource:
ôà@
-lstm_cell_25_matmul_1_readvariableop_resource:	8à;
,lstm_cell_25_biasadd_readvariableop_resource:	à
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
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :82
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
value	B :82
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
:ÿÿÿÿÿÿÿÿÿ82
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :82
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
value	B :82
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
:ÿÿÿÿÿÿÿÿÿ82	
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
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿô2
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
valueB"ÿÿÿÿô  27
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
:ÿÿÿÿÿÿÿÿÿô*
shrink_axis_mask2
strided_slice_2¶
"lstm_cell_25/MatMul/ReadVariableOpReadVariableOp+lstm_cell_25_matmul_readvariableop_resource* 
_output_shapes
:
ôà*
dtype02$
"lstm_cell_25/MatMul/ReadVariableOp­
lstm_cell_25/MatMulMatMulstrided_slice_2:output:0*lstm_cell_25/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà2
lstm_cell_25/MatMul»
$lstm_cell_25/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_25_matmul_1_readvariableop_resource*
_output_shapes
:	8à*
dtype02&
$lstm_cell_25/MatMul_1/ReadVariableOp©
lstm_cell_25/MatMul_1MatMulzeros:output:0,lstm_cell_25/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà2
lstm_cell_25/MatMul_1 
lstm_cell_25/addAddV2lstm_cell_25/MatMul:product:0lstm_cell_25/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà2
lstm_cell_25/add´
#lstm_cell_25/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_25_biasadd_readvariableop_resource*
_output_shapes	
:à*
dtype02%
#lstm_cell_25/BiasAdd/ReadVariableOp­
lstm_cell_25/BiasAddBiasAddlstm_cell_25/add:z:0+lstm_cell_25/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà2
lstm_cell_25/BiasAdd~
lstm_cell_25/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_25/split/split_dimó
lstm_cell_25/splitSplit%lstm_cell_25/split/split_dim:output:0lstm_cell_25/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ8:ÿÿÿÿÿÿÿÿÿ8:ÿÿÿÿÿÿÿÿÿ8:ÿÿÿÿÿÿÿÿÿ8*
	num_split2
lstm_cell_25/split
lstm_cell_25/SigmoidSigmoidlstm_cell_25/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82
lstm_cell_25/Sigmoid
lstm_cell_25/Sigmoid_1Sigmoidlstm_cell_25/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82
lstm_cell_25/Sigmoid_1
lstm_cell_25/mulMullstm_cell_25/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82
lstm_cell_25/mul}
lstm_cell_25/ReluRelulstm_cell_25/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82
lstm_cell_25/Relu
lstm_cell_25/mul_1Mullstm_cell_25/Sigmoid:y:0lstm_cell_25/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82
lstm_cell_25/mul_1
lstm_cell_25/add_1AddV2lstm_cell_25/mul:z:0lstm_cell_25/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82
lstm_cell_25/add_1
lstm_cell_25/Sigmoid_2Sigmoidlstm_cell_25/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82
lstm_cell_25/Sigmoid_2|
lstm_cell_25/Relu_1Relulstm_cell_25/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82
lstm_cell_25/Relu_1 
lstm_cell_25/mul_2Mullstm_cell_25/Sigmoid_2:y:0!lstm_cell_25/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82
lstm_cell_25/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ8   2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_25_matmul_readvariableop_resource-lstm_cell_25_matmul_1_readvariableop_resource,lstm_cell_25_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ8:ÿÿÿÿÿÿÿÿÿ8: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_40079669*
condR
while_cond_40079668*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ8:ÿÿÿÿÿÿÿÿÿ8: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ8   22
0TensorArrayV2Stack/TensorListStack/element_shapeñ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ8*
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
:ÿÿÿÿÿÿÿÿÿ8*
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
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ82
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
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ82

IdentityÈ
NoOpNoOp$^lstm_cell_25/BiasAdd/ReadVariableOp#^lstm_cell_25/MatMul/ReadVariableOp%^lstm_cell_25/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿô: : : 2J
#lstm_cell_25/BiasAdd/ReadVariableOp#lstm_cell_25/BiasAdd/ReadVariableOp2H
"lstm_cell_25/MatMul/ReadVariableOp"lstm_cell_25/MatMul/ReadVariableOp2L
$lstm_cell_25/MatMul_1/ReadVariableOp$lstm_cell_25/MatMul_1/ReadVariableOp2
whilewhile:_ [
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿô
"
_user_specified_name
inputs/0


Ë
0__inference_sequential_12_layer_call_fn_40078755

inputs
unknown:	]Ð
	unknown_0:
ôÐ
	unknown_1:	Ð
	unknown_2:
ôà
	unknown_3:	8à
	unknown_4:	à
	unknown_5:8
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
K__inference_sequential_12_layer_call_and_return_conditional_losses_400774782
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
ã
Í
while_cond_40075931
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_40075931___redundant_placeholder06
2while_while_cond_40075931___redundant_placeholder16
2while_while_cond_40075931___redundant_placeholder26
2while_while_cond_40075931___redundant_placeholder3
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
B: : : : :ÿÿÿÿÿÿÿÿÿô:ÿÿÿÿÿÿÿÿÿô: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô:

_output_shapes
: :

_output_shapes
:
\

E__inference_lstm_24_layer_call_and_return_conditional_losses_40079380

inputs>
+lstm_cell_24_matmul_readvariableop_resource:	]ÐA
-lstm_cell_24_matmul_1_readvariableop_resource:
ôÐ;
,lstm_cell_24_biasadd_readvariableop_resource:	Ð
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
strided_slice]
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :ô2
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
B :ô2
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
:ÿÿÿÿÿÿÿÿÿô2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :ô2
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
B :ô2
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
:ÿÿÿÿÿÿÿÿÿô2	
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
:	]Ð*
dtype02$
"lstm_cell_24/MatMul/ReadVariableOp­
lstm_cell_24/MatMulMatMulstrided_slice_2:output:0*lstm_cell_24/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
lstm_cell_24/MatMul¼
$lstm_cell_24/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_24_matmul_1_readvariableop_resource* 
_output_shapes
:
ôÐ*
dtype02&
$lstm_cell_24/MatMul_1/ReadVariableOp©
lstm_cell_24/MatMul_1MatMulzeros:output:0,lstm_cell_24/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
lstm_cell_24/MatMul_1 
lstm_cell_24/addAddV2lstm_cell_24/MatMul:product:0lstm_cell_24/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
lstm_cell_24/add´
#lstm_cell_24/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_24_biasadd_readvariableop_resource*
_output_shapes	
:Ð*
dtype02%
#lstm_cell_24/BiasAdd/ReadVariableOp­
lstm_cell_24/BiasAddBiasAddlstm_cell_24/add:z:0+lstm_cell_24/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
lstm_cell_24/BiasAdd~
lstm_cell_24/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_24/split/split_dim÷
lstm_cell_24/splitSplit%lstm_cell_24/split/split_dim:output:0lstm_cell_24/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿô:ÿÿÿÿÿÿÿÿÿô:ÿÿÿÿÿÿÿÿÿô:ÿÿÿÿÿÿÿÿÿô*
	num_split2
lstm_cell_24/split
lstm_cell_24/SigmoidSigmoidlstm_cell_24/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2
lstm_cell_24/Sigmoid
lstm_cell_24/Sigmoid_1Sigmoidlstm_cell_24/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2
lstm_cell_24/Sigmoid_1
lstm_cell_24/mulMullstm_cell_24/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2
lstm_cell_24/mul~
lstm_cell_24/ReluRelulstm_cell_24/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2
lstm_cell_24/Relu
lstm_cell_24/mul_1Mullstm_cell_24/Sigmoid:y:0lstm_cell_24/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2
lstm_cell_24/mul_1
lstm_cell_24/add_1AddV2lstm_cell_24/mul:z:0lstm_cell_24/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2
lstm_cell_24/add_1
lstm_cell_24/Sigmoid_2Sigmoidlstm_cell_24/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2
lstm_cell_24/Sigmoid_2}
lstm_cell_24/Relu_1Relulstm_cell_24/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2
lstm_cell_24/Relu_1¡
lstm_cell_24/mul_2Mullstm_cell_24/Sigmoid_2:y:0!lstm_cell_24/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2
lstm_cell_24/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿô  2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_24_matmul_readvariableop_resource-lstm_cell_24_matmul_1_readvariableop_resource,lstm_cell_24_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿô:ÿÿÿÿÿÿÿÿÿô: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_40079296*
condR
while_cond_40079295*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿô:ÿÿÿÿÿÿÿÿÿô: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿô  22
0TensorArrayV2Stack/TensorListStack/element_shapeé
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿô*
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
:ÿÿÿÿÿÿÿÿÿô*
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
:ÿÿÿÿÿÿÿÿÿô2
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
:ÿÿÿÿÿÿÿÿÿô2

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
?
Ô
while_body_40079820
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
3while_lstm_cell_25_matmul_readvariableop_resource_0:
ôàH
5while_lstm_cell_25_matmul_1_readvariableop_resource_0:	8àC
4while_lstm_cell_25_biasadd_readvariableop_resource_0:	à
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
1while_lstm_cell_25_matmul_readvariableop_resource:
ôàF
3while_lstm_cell_25_matmul_1_readvariableop_resource:	8àA
2while_lstm_cell_25_biasadd_readvariableop_resource:	à¢)while/lstm_cell_25/BiasAdd/ReadVariableOp¢(while/lstm_cell_25/MatMul/ReadVariableOp¢*while/lstm_cell_25/MatMul_1/ReadVariableOpÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿô  29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÔ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemÊ
(while/lstm_cell_25/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_25_matmul_readvariableop_resource_0* 
_output_shapes
:
ôà*
dtype02*
(while/lstm_cell_25/MatMul/ReadVariableOp×
while/lstm_cell_25/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_25/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà2
while/lstm_cell_25/MatMulÏ
*while/lstm_cell_25/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_25_matmul_1_readvariableop_resource_0*
_output_shapes
:	8à*
dtype02,
*while/lstm_cell_25/MatMul_1/ReadVariableOpÀ
while/lstm_cell_25/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_25/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà2
while/lstm_cell_25/MatMul_1¸
while/lstm_cell_25/addAddV2#while/lstm_cell_25/MatMul:product:0%while/lstm_cell_25/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà2
while/lstm_cell_25/addÈ
)while/lstm_cell_25/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_25_biasadd_readvariableop_resource_0*
_output_shapes	
:à*
dtype02+
)while/lstm_cell_25/BiasAdd/ReadVariableOpÅ
while/lstm_cell_25/BiasAddBiasAddwhile/lstm_cell_25/add:z:01while/lstm_cell_25/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà2
while/lstm_cell_25/BiasAdd
"while/lstm_cell_25/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_25/split/split_dim
while/lstm_cell_25/splitSplit+while/lstm_cell_25/split/split_dim:output:0#while/lstm_cell_25/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ8:ÿÿÿÿÿÿÿÿÿ8:ÿÿÿÿÿÿÿÿÿ8:ÿÿÿÿÿÿÿÿÿ8*
	num_split2
while/lstm_cell_25/split
while/lstm_cell_25/SigmoidSigmoid!while/lstm_cell_25/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82
while/lstm_cell_25/Sigmoid
while/lstm_cell_25/Sigmoid_1Sigmoid!while/lstm_cell_25/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82
while/lstm_cell_25/Sigmoid_1 
while/lstm_cell_25/mulMul while/lstm_cell_25/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82
while/lstm_cell_25/mul
while/lstm_cell_25/ReluRelu!while/lstm_cell_25/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82
while/lstm_cell_25/Relu´
while/lstm_cell_25/mul_1Mulwhile/lstm_cell_25/Sigmoid:y:0%while/lstm_cell_25/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82
while/lstm_cell_25/mul_1©
while/lstm_cell_25/add_1AddV2while/lstm_cell_25/mul:z:0while/lstm_cell_25/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82
while/lstm_cell_25/add_1
while/lstm_cell_25/Sigmoid_2Sigmoid!while/lstm_cell_25/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82
while/lstm_cell_25/Sigmoid_2
while/lstm_cell_25/Relu_1Reluwhile/lstm_cell_25/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82
while/lstm_cell_25/Relu_1¸
while/lstm_cell_25/mul_2Mul while/lstm_cell_25/Sigmoid_2:y:0'while/lstm_cell_25/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82
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
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_25/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_25/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82
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
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ8:ÿÿÿÿÿÿÿÿÿ8: : : : : 2V
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
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ8:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ8:

_output_shapes
: :

_output_shapes
: 
ËF

E__inference_lstm_24_layer_call_and_return_conditional_losses_40076001

inputs(
lstm_cell_24_40075919:	]Ð)
lstm_cell_24_40075921:
ôÐ$
lstm_cell_24_40075923:	Ð
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
strided_slice]
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :ô2
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
B :ô2
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
:ÿÿÿÿÿÿÿÿÿô2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :ô2
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
B :ô2
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
:ÿÿÿÿÿÿÿÿÿô2	
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
$lstm_cell_24/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_24_40075919lstm_cell_24_40075921lstm_cell_24_40075923*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿô:ÿÿÿÿÿÿÿÿÿô:ÿÿÿÿÿÿÿÿÿô*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_lstm_cell_24_layer_call_and_return_conditional_losses_400759182&
$lstm_cell_24/StatefulPartitionedCall
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿô  2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_24_40075919lstm_cell_24_40075921lstm_cell_24_40075923*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿô:ÿÿÿÿÿÿÿÿÿô: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_40075932*
condR
while_cond_40075931*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿô:ÿÿÿÿÿÿÿÿÿô: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿô  22
0TensorArrayV2Stack/TensorListStack/element_shapeò
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿô*
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
:ÿÿÿÿÿÿÿÿÿô*
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
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿô2
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
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿô2

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
¹
å	
#__inference__wrapped_model_40075843
lstm_24_inputT
Asequential_12_lstm_24_lstm_cell_24_matmul_readvariableop_resource:	]ÐW
Csequential_12_lstm_24_lstm_cell_24_matmul_1_readvariableop_resource:
ôÐQ
Bsequential_12_lstm_24_lstm_cell_24_biasadd_readvariableop_resource:	ÐU
Asequential_12_lstm_25_lstm_cell_25_matmul_readvariableop_resource:
ôàV
Csequential_12_lstm_25_lstm_cell_25_matmul_1_readvariableop_resource:	8àQ
Bsequential_12_lstm_25_lstm_cell_25_biasadd_readvariableop_resource:	àJ
8sequential_12_dense_12_tensordot_readvariableop_resource:8D
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
#sequential_12/lstm_24/strided_slice
!sequential_12/lstm_24/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :ô2#
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
 sequential_12/lstm_24/zeros/Less
$sequential_12/lstm_24/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :ô2&
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
!sequential_12/lstm_24/zeros/ConstÎ
sequential_12/lstm_24/zerosFill+sequential_12/lstm_24/zeros/packed:output:0*sequential_12/lstm_24/zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2
sequential_12/lstm_24/zeros
#sequential_12/lstm_24/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :ô2%
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
"sequential_12/lstm_24/zeros_1/Less
&sequential_12/lstm_24/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :ô2(
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
#sequential_12/lstm_24/zeros_1/ConstÖ
sequential_12/lstm_24/zeros_1Fill-sequential_12/lstm_24/zeros_1/packed:output:0,sequential_12/lstm_24/zeros_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2
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
:	]Ð*
dtype02:
8sequential_12/lstm_24/lstm_cell_24/MatMul/ReadVariableOp
)sequential_12/lstm_24/lstm_cell_24/MatMulMatMul.sequential_12/lstm_24/strided_slice_2:output:0@sequential_12/lstm_24/lstm_cell_24/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2+
)sequential_12/lstm_24/lstm_cell_24/MatMulþ
:sequential_12/lstm_24/lstm_cell_24/MatMul_1/ReadVariableOpReadVariableOpCsequential_12_lstm_24_lstm_cell_24_matmul_1_readvariableop_resource* 
_output_shapes
:
ôÐ*
dtype02<
:sequential_12/lstm_24/lstm_cell_24/MatMul_1/ReadVariableOp
+sequential_12/lstm_24/lstm_cell_24/MatMul_1MatMul$sequential_12/lstm_24/zeros:output:0Bsequential_12/lstm_24/lstm_cell_24/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2-
+sequential_12/lstm_24/lstm_cell_24/MatMul_1ø
&sequential_12/lstm_24/lstm_cell_24/addAddV23sequential_12/lstm_24/lstm_cell_24/MatMul:product:05sequential_12/lstm_24/lstm_cell_24/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2(
&sequential_12/lstm_24/lstm_cell_24/addö
9sequential_12/lstm_24/lstm_cell_24/BiasAdd/ReadVariableOpReadVariableOpBsequential_12_lstm_24_lstm_cell_24_biasadd_readvariableop_resource*
_output_shapes	
:Ð*
dtype02;
9sequential_12/lstm_24/lstm_cell_24/BiasAdd/ReadVariableOp
*sequential_12/lstm_24/lstm_cell_24/BiasAddBiasAdd*sequential_12/lstm_24/lstm_cell_24/add:z:0Asequential_12/lstm_24/lstm_cell_24/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2,
*sequential_12/lstm_24/lstm_cell_24/BiasAddª
2sequential_12/lstm_24/lstm_cell_24/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :24
2sequential_12/lstm_24/lstm_cell_24/split/split_dimÏ
(sequential_12/lstm_24/lstm_cell_24/splitSplit;sequential_12/lstm_24/lstm_cell_24/split/split_dim:output:03sequential_12/lstm_24/lstm_cell_24/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿô:ÿÿÿÿÿÿÿÿÿô:ÿÿÿÿÿÿÿÿÿô:ÿÿÿÿÿÿÿÿÿô*
	num_split2*
(sequential_12/lstm_24/lstm_cell_24/splitÉ
*sequential_12/lstm_24/lstm_cell_24/SigmoidSigmoid1sequential_12/lstm_24/lstm_cell_24/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2,
*sequential_12/lstm_24/lstm_cell_24/SigmoidÍ
,sequential_12/lstm_24/lstm_cell_24/Sigmoid_1Sigmoid1sequential_12/lstm_24/lstm_cell_24/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2.
,sequential_12/lstm_24/lstm_cell_24/Sigmoid_1ä
&sequential_12/lstm_24/lstm_cell_24/mulMul0sequential_12/lstm_24/lstm_cell_24/Sigmoid_1:y:0&sequential_12/lstm_24/zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2(
&sequential_12/lstm_24/lstm_cell_24/mulÀ
'sequential_12/lstm_24/lstm_cell_24/ReluRelu1sequential_12/lstm_24/lstm_cell_24/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2)
'sequential_12/lstm_24/lstm_cell_24/Reluõ
(sequential_12/lstm_24/lstm_cell_24/mul_1Mul.sequential_12/lstm_24/lstm_cell_24/Sigmoid:y:05sequential_12/lstm_24/lstm_cell_24/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2*
(sequential_12/lstm_24/lstm_cell_24/mul_1ê
(sequential_12/lstm_24/lstm_cell_24/add_1AddV2*sequential_12/lstm_24/lstm_cell_24/mul:z:0,sequential_12/lstm_24/lstm_cell_24/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2*
(sequential_12/lstm_24/lstm_cell_24/add_1Í
,sequential_12/lstm_24/lstm_cell_24/Sigmoid_2Sigmoid1sequential_12/lstm_24/lstm_cell_24/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2.
,sequential_12/lstm_24/lstm_cell_24/Sigmoid_2¿
)sequential_12/lstm_24/lstm_cell_24/Relu_1Relu,sequential_12/lstm_24/lstm_cell_24/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2+
)sequential_12/lstm_24/lstm_cell_24/Relu_1ù
(sequential_12/lstm_24/lstm_cell_24/mul_2Mul0sequential_12/lstm_24/lstm_cell_24/Sigmoid_2:y:07sequential_12/lstm_24/lstm_cell_24/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2*
(sequential_12/lstm_24/lstm_cell_24/mul_2»
3sequential_12/lstm_24/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿô  25
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
(sequential_12/lstm_24/while/loop_counterÝ
sequential_12/lstm_24/whileWhile1sequential_12/lstm_24/while/loop_counter:output:07sequential_12/lstm_24/while/maximum_iterations:output:0#sequential_12/lstm_24/time:output:0.sequential_12/lstm_24/TensorArrayV2_1:handle:0$sequential_12/lstm_24/zeros:output:0&sequential_12/lstm_24/zeros_1:output:0.sequential_12/lstm_24/strided_slice_1:output:0Msequential_12/lstm_24/TensorArrayUnstack/TensorListFromTensor:output_handle:0Asequential_12_lstm_24_lstm_cell_24_matmul_readvariableop_resourceCsequential_12_lstm_24_lstm_cell_24_matmul_1_readvariableop_resourceBsequential_12_lstm_24_lstm_cell_24_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿô:ÿÿÿÿÿÿÿÿÿô: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *5
body-R+
)sequential_12_lstm_24_while_body_40075583*5
cond-R+
)sequential_12_lstm_24_while_cond_40075582*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿô:ÿÿÿÿÿÿÿÿÿô: : : : : *
parallel_iterations 2
sequential_12/lstm_24/whileá
Fsequential_12/lstm_24/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿô  2H
Fsequential_12/lstm_24/TensorArrayV2Stack/TensorListStack/element_shapeÁ
8sequential_12/lstm_24/TensorArrayV2Stack/TensorListStackTensorListStack$sequential_12/lstm_24/while:output:3Osequential_12/lstm_24/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿô*
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
-sequential_12/lstm_24/strided_slice_3/stack_2
%sequential_12/lstm_24/strided_slice_3StridedSliceAsequential_12/lstm_24/TensorArrayV2Stack/TensorListStack:tensor:04sequential_12/lstm_24/strided_slice_3/stack:output:06sequential_12/lstm_24/strided_slice_3/stack_1:output:06sequential_12/lstm_24/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô*
shrink_axis_mask2'
%sequential_12/lstm_24/strided_slice_3¥
&sequential_12/lstm_24/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2(
&sequential_12/lstm_24/transpose_1/permþ
!sequential_12/lstm_24/transpose_1	TransposeAsequential_12/lstm_24/TensorArrayV2Stack/TensorListStack:tensor:0/sequential_12/lstm_24/transpose_1/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2#
!sequential_12/lstm_24/transpose_1
sequential_12/lstm_24/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
sequential_12/lstm_24/runtime°
!sequential_12/dropout_24/IdentityIdentity%sequential_12/lstm_24/transpose_1:y:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2#
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
#sequential_12/lstm_25/strided_slice
!sequential_12/lstm_25/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :82#
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
 sequential_12/lstm_25/zeros/Less
$sequential_12/lstm_25/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :82&
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
!sequential_12/lstm_25/zeros/ConstÍ
sequential_12/lstm_25/zerosFill+sequential_12/lstm_25/zeros/packed:output:0*sequential_12/lstm_25/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82
sequential_12/lstm_25/zeros
#sequential_12/lstm_25/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :82%
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
"sequential_12/lstm_25/zeros_1/Less
&sequential_12/lstm_25/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :82(
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
#sequential_12/lstm_25/zeros_1/ConstÕ
sequential_12/lstm_25/zeros_1Fill-sequential_12/lstm_25/zeros_1/packed:output:0,sequential_12/lstm_25/zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82
sequential_12/lstm_25/zeros_1¡
$sequential_12/lstm_25/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2&
$sequential_12/lstm_25/transpose/permá
sequential_12/lstm_25/transpose	Transpose*sequential_12/dropout_24/Identity:output:0-sequential_12/lstm_25/transpose/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2!
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
valueB"ÿÿÿÿô  2M
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
-sequential_12/lstm_25/strided_slice_2/stack_2
%sequential_12/lstm_25/strided_slice_2StridedSlice#sequential_12/lstm_25/transpose:y:04sequential_12/lstm_25/strided_slice_2/stack:output:06sequential_12/lstm_25/strided_slice_2/stack_1:output:06sequential_12/lstm_25/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô*
shrink_axis_mask2'
%sequential_12/lstm_25/strided_slice_2ø
8sequential_12/lstm_25/lstm_cell_25/MatMul/ReadVariableOpReadVariableOpAsequential_12_lstm_25_lstm_cell_25_matmul_readvariableop_resource* 
_output_shapes
:
ôà*
dtype02:
8sequential_12/lstm_25/lstm_cell_25/MatMul/ReadVariableOp
)sequential_12/lstm_25/lstm_cell_25/MatMulMatMul.sequential_12/lstm_25/strided_slice_2:output:0@sequential_12/lstm_25/lstm_cell_25/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà2+
)sequential_12/lstm_25/lstm_cell_25/MatMulý
:sequential_12/lstm_25/lstm_cell_25/MatMul_1/ReadVariableOpReadVariableOpCsequential_12_lstm_25_lstm_cell_25_matmul_1_readvariableop_resource*
_output_shapes
:	8à*
dtype02<
:sequential_12/lstm_25/lstm_cell_25/MatMul_1/ReadVariableOp
+sequential_12/lstm_25/lstm_cell_25/MatMul_1MatMul$sequential_12/lstm_25/zeros:output:0Bsequential_12/lstm_25/lstm_cell_25/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà2-
+sequential_12/lstm_25/lstm_cell_25/MatMul_1ø
&sequential_12/lstm_25/lstm_cell_25/addAddV23sequential_12/lstm_25/lstm_cell_25/MatMul:product:05sequential_12/lstm_25/lstm_cell_25/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà2(
&sequential_12/lstm_25/lstm_cell_25/addö
9sequential_12/lstm_25/lstm_cell_25/BiasAdd/ReadVariableOpReadVariableOpBsequential_12_lstm_25_lstm_cell_25_biasadd_readvariableop_resource*
_output_shapes	
:à*
dtype02;
9sequential_12/lstm_25/lstm_cell_25/BiasAdd/ReadVariableOp
*sequential_12/lstm_25/lstm_cell_25/BiasAddBiasAdd*sequential_12/lstm_25/lstm_cell_25/add:z:0Asequential_12/lstm_25/lstm_cell_25/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà2,
*sequential_12/lstm_25/lstm_cell_25/BiasAddª
2sequential_12/lstm_25/lstm_cell_25/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :24
2sequential_12/lstm_25/lstm_cell_25/split/split_dimË
(sequential_12/lstm_25/lstm_cell_25/splitSplit;sequential_12/lstm_25/lstm_cell_25/split/split_dim:output:03sequential_12/lstm_25/lstm_cell_25/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ8:ÿÿÿÿÿÿÿÿÿ8:ÿÿÿÿÿÿÿÿÿ8:ÿÿÿÿÿÿÿÿÿ8*
	num_split2*
(sequential_12/lstm_25/lstm_cell_25/splitÈ
*sequential_12/lstm_25/lstm_cell_25/SigmoidSigmoid1sequential_12/lstm_25/lstm_cell_25/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82,
*sequential_12/lstm_25/lstm_cell_25/SigmoidÌ
,sequential_12/lstm_25/lstm_cell_25/Sigmoid_1Sigmoid1sequential_12/lstm_25/lstm_cell_25/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82.
,sequential_12/lstm_25/lstm_cell_25/Sigmoid_1ã
&sequential_12/lstm_25/lstm_cell_25/mulMul0sequential_12/lstm_25/lstm_cell_25/Sigmoid_1:y:0&sequential_12/lstm_25/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82(
&sequential_12/lstm_25/lstm_cell_25/mul¿
'sequential_12/lstm_25/lstm_cell_25/ReluRelu1sequential_12/lstm_25/lstm_cell_25/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82)
'sequential_12/lstm_25/lstm_cell_25/Reluô
(sequential_12/lstm_25/lstm_cell_25/mul_1Mul.sequential_12/lstm_25/lstm_cell_25/Sigmoid:y:05sequential_12/lstm_25/lstm_cell_25/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82*
(sequential_12/lstm_25/lstm_cell_25/mul_1é
(sequential_12/lstm_25/lstm_cell_25/add_1AddV2*sequential_12/lstm_25/lstm_cell_25/mul:z:0,sequential_12/lstm_25/lstm_cell_25/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82*
(sequential_12/lstm_25/lstm_cell_25/add_1Ì
,sequential_12/lstm_25/lstm_cell_25/Sigmoid_2Sigmoid1sequential_12/lstm_25/lstm_cell_25/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82.
,sequential_12/lstm_25/lstm_cell_25/Sigmoid_2¾
)sequential_12/lstm_25/lstm_cell_25/Relu_1Relu,sequential_12/lstm_25/lstm_cell_25/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82+
)sequential_12/lstm_25/lstm_cell_25/Relu_1ø
(sequential_12/lstm_25/lstm_cell_25/mul_2Mul0sequential_12/lstm_25/lstm_cell_25/Sigmoid_2:y:07sequential_12/lstm_25/lstm_cell_25/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82*
(sequential_12/lstm_25/lstm_cell_25/mul_2»
3sequential_12/lstm_25/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ8   25
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
(sequential_12/lstm_25/while/loop_counterÙ
sequential_12/lstm_25/whileWhile1sequential_12/lstm_25/while/loop_counter:output:07sequential_12/lstm_25/while/maximum_iterations:output:0#sequential_12/lstm_25/time:output:0.sequential_12/lstm_25/TensorArrayV2_1:handle:0$sequential_12/lstm_25/zeros:output:0&sequential_12/lstm_25/zeros_1:output:0.sequential_12/lstm_25/strided_slice_1:output:0Msequential_12/lstm_25/TensorArrayUnstack/TensorListFromTensor:output_handle:0Asequential_12_lstm_25_lstm_cell_25_matmul_readvariableop_resourceCsequential_12_lstm_25_lstm_cell_25_matmul_1_readvariableop_resourceBsequential_12_lstm_25_lstm_cell_25_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ8:ÿÿÿÿÿÿÿÿÿ8: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *5
body-R+
)sequential_12_lstm_25_while_body_40075731*5
cond-R+
)sequential_12_lstm_25_while_cond_40075730*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ8:ÿÿÿÿÿÿÿÿÿ8: : : : : *
parallel_iterations 2
sequential_12/lstm_25/whileá
Fsequential_12/lstm_25/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ8   2H
Fsequential_12/lstm_25/TensorArrayV2Stack/TensorListStack/element_shapeÀ
8sequential_12/lstm_25/TensorArrayV2Stack/TensorListStackTensorListStack$sequential_12/lstm_25/while:output:3Osequential_12/lstm_25/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ8*
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
-sequential_12/lstm_25/strided_slice_3/stack_2
%sequential_12/lstm_25/strided_slice_3StridedSliceAsequential_12/lstm_25/TensorArrayV2Stack/TensorListStack:tensor:04sequential_12/lstm_25/strided_slice_3/stack:output:06sequential_12/lstm_25/strided_slice_3/stack_1:output:06sequential_12/lstm_25/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ8*
shrink_axis_mask2'
%sequential_12/lstm_25/strided_slice_3¥
&sequential_12/lstm_25/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2(
&sequential_12/lstm_25/transpose_1/permý
!sequential_12/lstm_25/transpose_1	TransposeAsequential_12/lstm_25/TensorArrayV2Stack/TensorListStack:tensor:0/sequential_12/lstm_25/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82#
!sequential_12/lstm_25/transpose_1
sequential_12/lstm_25/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
sequential_12/lstm_25/runtime¯
!sequential_12/dropout_25/IdentityIdentity%sequential_12/lstm_25/transpose_1:y:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82#
!sequential_12/dropout_25/IdentityÛ
/sequential_12/dense_12/Tensordot/ReadVariableOpReadVariableOp8sequential_12_dense_12_tensordot_readvariableop_resource*
_output_shapes

:8*
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
&sequential_12/dense_12/Tensordot/stackù
*sequential_12/dense_12/Tensordot/transpose	Transpose*sequential_12/dropout_25/Identity:output:00sequential_12/dense_12/Tensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82,
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
¶
¸
*__inference_lstm_24_layer_call_fn_40079413

inputs
unknown:	]Ð
	unknown_0:
ôÐ
	unknown_1:	Ð
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿô*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_lstm_24_layer_call_and_return_conditional_losses_400772602
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2

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
while_cond_40079668
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_40079668___redundant_placeholder06
2while_while_cond_40079668___redundant_placeholder16
2while_while_cond_40079668___redundant_placeholder26
2while_while_cond_40079668___redundant_placeholder3
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
@: : : : :ÿÿÿÿÿÿÿÿÿ8:ÿÿÿÿÿÿÿÿÿ8: ::::: 

_output_shapes
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
:ÿÿÿÿÿÿÿÿÿ8:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ8:

_output_shapes
: :

_output_shapes
:
¶
f
-__inference_dropout_25_layer_call_fn_40080126

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
:ÿÿÿÿÿÿÿÿÿ8* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_dropout_25_layer_call_and_return_conditional_losses_400775272
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ822
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ8
 
_user_specified_nameinputs
ù

J__inference_lstm_cell_25_layer_call_and_return_conditional_losses_40076694

inputs

states
states_12
matmul_readvariableop_resource:
ôà3
 matmul_1_readvariableop_resource:	8à.
biasadd_readvariableop_resource:	à
identity

identity_1

identity_2¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
ôà*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà2
MatMul
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	8à*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà2
add
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:à*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà2	
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
L:ÿÿÿÿÿÿÿÿÿ8:ÿÿÿÿÿÿÿÿÿ8:ÿÿÿÿÿÿÿÿÿ8:ÿÿÿÿÿÿÿÿÿ8*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82
	Sigmoid_1\
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82
mulV
ReluRelusplit:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82
	Sigmoid_2U
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82
mul_2d
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82

Identityh

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82

Identity_1h

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82

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
@:ÿÿÿÿÿÿÿÿÿô:ÿÿÿÿÿÿÿÿÿ8:ÿÿÿÿÿÿÿÿÿ8: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ8
 
_user_specified_namestates:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ8
 
_user_specified_namestates
í[

E__inference_lstm_25_layer_call_and_return_conditional_losses_40079904

inputs?
+lstm_cell_25_matmul_readvariableop_resource:
ôà@
-lstm_cell_25_matmul_1_readvariableop_resource:	8à;
,lstm_cell_25_biasadd_readvariableop_resource:	à
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
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :82
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
value	B :82
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
:ÿÿÿÿÿÿÿÿÿ82
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :82
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
value	B :82
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
:ÿÿÿÿÿÿÿÿÿ82	
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
:ÿÿÿÿÿÿÿÿÿô2
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
valueB"ÿÿÿÿô  27
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
:ÿÿÿÿÿÿÿÿÿô*
shrink_axis_mask2
strided_slice_2¶
"lstm_cell_25/MatMul/ReadVariableOpReadVariableOp+lstm_cell_25_matmul_readvariableop_resource* 
_output_shapes
:
ôà*
dtype02$
"lstm_cell_25/MatMul/ReadVariableOp­
lstm_cell_25/MatMulMatMulstrided_slice_2:output:0*lstm_cell_25/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà2
lstm_cell_25/MatMul»
$lstm_cell_25/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_25_matmul_1_readvariableop_resource*
_output_shapes
:	8à*
dtype02&
$lstm_cell_25/MatMul_1/ReadVariableOp©
lstm_cell_25/MatMul_1MatMulzeros:output:0,lstm_cell_25/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà2
lstm_cell_25/MatMul_1 
lstm_cell_25/addAddV2lstm_cell_25/MatMul:product:0lstm_cell_25/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà2
lstm_cell_25/add´
#lstm_cell_25/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_25_biasadd_readvariableop_resource*
_output_shapes	
:à*
dtype02%
#lstm_cell_25/BiasAdd/ReadVariableOp­
lstm_cell_25/BiasAddBiasAddlstm_cell_25/add:z:0+lstm_cell_25/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà2
lstm_cell_25/BiasAdd~
lstm_cell_25/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_25/split/split_dimó
lstm_cell_25/splitSplit%lstm_cell_25/split/split_dim:output:0lstm_cell_25/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ8:ÿÿÿÿÿÿÿÿÿ8:ÿÿÿÿÿÿÿÿÿ8:ÿÿÿÿÿÿÿÿÿ8*
	num_split2
lstm_cell_25/split
lstm_cell_25/SigmoidSigmoidlstm_cell_25/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82
lstm_cell_25/Sigmoid
lstm_cell_25/Sigmoid_1Sigmoidlstm_cell_25/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82
lstm_cell_25/Sigmoid_1
lstm_cell_25/mulMullstm_cell_25/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82
lstm_cell_25/mul}
lstm_cell_25/ReluRelulstm_cell_25/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82
lstm_cell_25/Relu
lstm_cell_25/mul_1Mullstm_cell_25/Sigmoid:y:0lstm_cell_25/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82
lstm_cell_25/mul_1
lstm_cell_25/add_1AddV2lstm_cell_25/mul:z:0lstm_cell_25/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82
lstm_cell_25/add_1
lstm_cell_25/Sigmoid_2Sigmoidlstm_cell_25/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82
lstm_cell_25/Sigmoid_2|
lstm_cell_25/Relu_1Relulstm_cell_25/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82
lstm_cell_25/Relu_1 
lstm_cell_25/mul_2Mullstm_cell_25/Sigmoid_2:y:0!lstm_cell_25/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82
lstm_cell_25/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ8   2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_25_matmul_readvariableop_resource-lstm_cell_25_matmul_1_readvariableop_resource,lstm_cell_25_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ8:ÿÿÿÿÿÿÿÿÿ8: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_40079820*
condR
while_cond_40079819*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ8:ÿÿÿÿÿÿÿÿÿ8: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ8   22
0TensorArrayV2Stack/TensorListStack/element_shapeè
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ8*
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
:ÿÿÿÿÿÿÿÿÿ8*
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
:ÿÿÿÿÿÿÿÿÿ82
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
:ÿÿÿÿÿÿÿÿÿ82

IdentityÈ
NoOpNoOp$^lstm_cell_25/BiasAdd/ReadVariableOp#^lstm_cell_25/MatMul/ReadVariableOp%^lstm_cell_25/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:ÿÿÿÿÿÿÿÿÿô: : : 2J
#lstm_cell_25/BiasAdd/ReadVariableOp#lstm_cell_25/BiasAdd/ReadVariableOp2H
"lstm_cell_25/MatMul/ReadVariableOp"lstm_cell_25/MatMul/ReadVariableOp2L
$lstm_cell_25/MatMul_1/ReadVariableOp$lstm_cell_25/MatMul_1/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿô
 
_user_specified_nameinputs
¬

Ò
0__inference_sequential_12_layer_call_fn_40077987
lstm_24_input
unknown:	]Ð
	unknown_0:
ôÐ
	unknown_1:	Ð
	unknown_2:
ôà
	unknown_3:	8à
	unknown_4:	à
	unknown_5:8
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
K__inference_sequential_12_layer_call_and_return_conditional_losses_400779472
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
ß
Í
while_cond_40077340
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_40077340___redundant_placeholder06
2while_while_cond_40077340___redundant_placeholder16
2while_while_cond_40077340___redundant_placeholder26
2while_while_cond_40077340___redundant_placeholder3
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
@: : : : :ÿÿÿÿÿÿÿÿÿ8:ÿÿÿÿÿÿÿÿÿ8: ::::: 

_output_shapes
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
:ÿÿÿÿÿÿÿÿÿ8:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ8:

_output_shapes
: :

_output_shapes
:
ÃL
Ü
!__inference__traced_save_40080484
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
ú: :8:: : : : : :	]Ð:
ôÐ:Ð:
ôà:	8à:à: : : : :8::	]Ð:
ôÐ:Ð:
ôà:	8à:à:8::	]Ð:
ôÐ:Ð:
ôà:	8à:à: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:8: 
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
:	]Ð:&	"
 
_output_shapes
:
ôÐ:!


_output_shapes	
:Ð:&"
 
_output_shapes
:
ôà:%!

_output_shapes
:	8à:!

_output_shapes	
:à:
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
: :$ 

_output_shapes

:8: 

_output_shapes
::%!

_output_shapes
:	]Ð:&"
 
_output_shapes
:
ôÐ:!

_output_shapes	
:Ð:&"
 
_output_shapes
:
ôà:%!

_output_shapes
:	8à:!

_output_shapes	
:à:$ 

_output_shapes

:8: 

_output_shapes
::%!

_output_shapes
:	]Ð:&"
 
_output_shapes
:
ôÐ:!

_output_shapes	
:Ð:&"
 
_output_shapes
:
ôà:% !

_output_shapes
:	8à:!!

_output_shapes	
:à:"

_output_shapes
: 
ÓJ
Ô

lstm_25_while_body_40078281,
(lstm_25_while_lstm_25_while_loop_counter2
.lstm_25_while_lstm_25_while_maximum_iterations
lstm_25_while_placeholder
lstm_25_while_placeholder_1
lstm_25_while_placeholder_2
lstm_25_while_placeholder_3+
'lstm_25_while_lstm_25_strided_slice_1_0g
clstm_25_while_tensorarrayv2read_tensorlistgetitem_lstm_25_tensorarrayunstack_tensorlistfromtensor_0O
;lstm_25_while_lstm_cell_25_matmul_readvariableop_resource_0:
ôàP
=lstm_25_while_lstm_cell_25_matmul_1_readvariableop_resource_0:	8àK
<lstm_25_while_lstm_cell_25_biasadd_readvariableop_resource_0:	à
lstm_25_while_identity
lstm_25_while_identity_1
lstm_25_while_identity_2
lstm_25_while_identity_3
lstm_25_while_identity_4
lstm_25_while_identity_5)
%lstm_25_while_lstm_25_strided_slice_1e
alstm_25_while_tensorarrayv2read_tensorlistgetitem_lstm_25_tensorarrayunstack_tensorlistfromtensorM
9lstm_25_while_lstm_cell_25_matmul_readvariableop_resource:
ôàN
;lstm_25_while_lstm_cell_25_matmul_1_readvariableop_resource:	8àI
:lstm_25_while_lstm_cell_25_biasadd_readvariableop_resource:	à¢1lstm_25/while/lstm_cell_25/BiasAdd/ReadVariableOp¢0lstm_25/while/lstm_cell_25/MatMul/ReadVariableOp¢2lstm_25/while/lstm_cell_25/MatMul_1/ReadVariableOpÓ
?lstm_25/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿô  2A
?lstm_25/while/TensorArrayV2Read/TensorListGetItem/element_shape
1lstm_25/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemclstm_25_while_tensorarrayv2read_tensorlistgetitem_lstm_25_tensorarrayunstack_tensorlistfromtensor_0lstm_25_while_placeholderHlstm_25/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô*
element_dtype023
1lstm_25/while/TensorArrayV2Read/TensorListGetItemâ
0lstm_25/while/lstm_cell_25/MatMul/ReadVariableOpReadVariableOp;lstm_25_while_lstm_cell_25_matmul_readvariableop_resource_0* 
_output_shapes
:
ôà*
dtype022
0lstm_25/while/lstm_cell_25/MatMul/ReadVariableOp÷
!lstm_25/while/lstm_cell_25/MatMulMatMul8lstm_25/while/TensorArrayV2Read/TensorListGetItem:item:08lstm_25/while/lstm_cell_25/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà2#
!lstm_25/while/lstm_cell_25/MatMulç
2lstm_25/while/lstm_cell_25/MatMul_1/ReadVariableOpReadVariableOp=lstm_25_while_lstm_cell_25_matmul_1_readvariableop_resource_0*
_output_shapes
:	8à*
dtype024
2lstm_25/while/lstm_cell_25/MatMul_1/ReadVariableOpà
#lstm_25/while/lstm_cell_25/MatMul_1MatMullstm_25_while_placeholder_2:lstm_25/while/lstm_cell_25/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà2%
#lstm_25/while/lstm_cell_25/MatMul_1Ø
lstm_25/while/lstm_cell_25/addAddV2+lstm_25/while/lstm_cell_25/MatMul:product:0-lstm_25/while/lstm_cell_25/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà2 
lstm_25/while/lstm_cell_25/addà
1lstm_25/while/lstm_cell_25/BiasAdd/ReadVariableOpReadVariableOp<lstm_25_while_lstm_cell_25_biasadd_readvariableop_resource_0*
_output_shapes	
:à*
dtype023
1lstm_25/while/lstm_cell_25/BiasAdd/ReadVariableOpå
"lstm_25/while/lstm_cell_25/BiasAddBiasAdd"lstm_25/while/lstm_cell_25/add:z:09lstm_25/while/lstm_cell_25/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà2$
"lstm_25/while/lstm_cell_25/BiasAdd
*lstm_25/while/lstm_cell_25/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2,
*lstm_25/while/lstm_cell_25/split/split_dim«
 lstm_25/while/lstm_cell_25/splitSplit3lstm_25/while/lstm_cell_25/split/split_dim:output:0+lstm_25/while/lstm_cell_25/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ8:ÿÿÿÿÿÿÿÿÿ8:ÿÿÿÿÿÿÿÿÿ8:ÿÿÿÿÿÿÿÿÿ8*
	num_split2"
 lstm_25/while/lstm_cell_25/split°
"lstm_25/while/lstm_cell_25/SigmoidSigmoid)lstm_25/while/lstm_cell_25/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82$
"lstm_25/while/lstm_cell_25/Sigmoid´
$lstm_25/while/lstm_cell_25/Sigmoid_1Sigmoid)lstm_25/while/lstm_cell_25/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82&
$lstm_25/while/lstm_cell_25/Sigmoid_1À
lstm_25/while/lstm_cell_25/mulMul(lstm_25/while/lstm_cell_25/Sigmoid_1:y:0lstm_25_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82 
lstm_25/while/lstm_cell_25/mul§
lstm_25/while/lstm_cell_25/ReluRelu)lstm_25/while/lstm_cell_25/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82!
lstm_25/while/lstm_cell_25/ReluÔ
 lstm_25/while/lstm_cell_25/mul_1Mul&lstm_25/while/lstm_cell_25/Sigmoid:y:0-lstm_25/while/lstm_cell_25/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82"
 lstm_25/while/lstm_cell_25/mul_1É
 lstm_25/while/lstm_cell_25/add_1AddV2"lstm_25/while/lstm_cell_25/mul:z:0$lstm_25/while/lstm_cell_25/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82"
 lstm_25/while/lstm_cell_25/add_1´
$lstm_25/while/lstm_cell_25/Sigmoid_2Sigmoid)lstm_25/while/lstm_cell_25/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82&
$lstm_25/while/lstm_cell_25/Sigmoid_2¦
!lstm_25/while/lstm_cell_25/Relu_1Relu$lstm_25/while/lstm_cell_25/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82#
!lstm_25/while/lstm_cell_25/Relu_1Ø
 lstm_25/while/lstm_cell_25/mul_2Mul(lstm_25/while/lstm_cell_25/Sigmoid_2:y:0/lstm_25/while/lstm_cell_25/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82"
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
lstm_25/while/Identity_3­
lstm_25/while/Identity_4Identity$lstm_25/while/lstm_cell_25/mul_2:z:0^lstm_25/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82
lstm_25/while/Identity_4­
lstm_25/while/Identity_5Identity$lstm_25/while/lstm_cell_25/add_1:z:0^lstm_25/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82
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
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ8:ÿÿÿÿÿÿÿÿÿ8: : : : : 2f
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
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ8:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ8:

_output_shapes
: :

_output_shapes
: 

f
H__inference_dropout_25_layer_call_and_return_conditional_losses_40077438

inputs

identity_1^
IdentityIdentityinputs*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82

Identitym

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ8:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ8
 
_user_specified_nameinputs
°?
Ô
while_body_40078994
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_24_matmul_readvariableop_resource_0:	]ÐI
5while_lstm_cell_24_matmul_1_readvariableop_resource_0:
ôÐC
4while_lstm_cell_24_biasadd_readvariableop_resource_0:	Ð
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_24_matmul_readvariableop_resource:	]ÐG
3while_lstm_cell_24_matmul_1_readvariableop_resource:
ôÐA
2while_lstm_cell_24_biasadd_readvariableop_resource:	Ð¢)while/lstm_cell_24/BiasAdd/ReadVariableOp¢(while/lstm_cell_24/MatMul/ReadVariableOp¢*while/lstm_cell_24/MatMul_1/ReadVariableOpÃ
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
:	]Ð*
dtype02*
(while/lstm_cell_24/MatMul/ReadVariableOp×
while/lstm_cell_24/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_24/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
while/lstm_cell_24/MatMulÐ
*while/lstm_cell_24/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_24_matmul_1_readvariableop_resource_0* 
_output_shapes
:
ôÐ*
dtype02,
*while/lstm_cell_24/MatMul_1/ReadVariableOpÀ
while/lstm_cell_24/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_24/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
while/lstm_cell_24/MatMul_1¸
while/lstm_cell_24/addAddV2#while/lstm_cell_24/MatMul:product:0%while/lstm_cell_24/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
while/lstm_cell_24/addÈ
)while/lstm_cell_24/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_24_biasadd_readvariableop_resource_0*
_output_shapes	
:Ð*
dtype02+
)while/lstm_cell_24/BiasAdd/ReadVariableOpÅ
while/lstm_cell_24/BiasAddBiasAddwhile/lstm_cell_24/add:z:01while/lstm_cell_24/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
while/lstm_cell_24/BiasAdd
"while/lstm_cell_24/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_24/split/split_dim
while/lstm_cell_24/splitSplit+while/lstm_cell_24/split/split_dim:output:0#while/lstm_cell_24/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿô:ÿÿÿÿÿÿÿÿÿô:ÿÿÿÿÿÿÿÿÿô:ÿÿÿÿÿÿÿÿÿô*
	num_split2
while/lstm_cell_24/split
while/lstm_cell_24/SigmoidSigmoid!while/lstm_cell_24/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2
while/lstm_cell_24/Sigmoid
while/lstm_cell_24/Sigmoid_1Sigmoid!while/lstm_cell_24/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2
while/lstm_cell_24/Sigmoid_1¡
while/lstm_cell_24/mulMul while/lstm_cell_24/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2
while/lstm_cell_24/mul
while/lstm_cell_24/ReluRelu!while/lstm_cell_24/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2
while/lstm_cell_24/Reluµ
while/lstm_cell_24/mul_1Mulwhile/lstm_cell_24/Sigmoid:y:0%while/lstm_cell_24/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2
while/lstm_cell_24/mul_1ª
while/lstm_cell_24/add_1AddV2while/lstm_cell_24/mul:z:0while/lstm_cell_24/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2
while/lstm_cell_24/add_1
while/lstm_cell_24/Sigmoid_2Sigmoid!while/lstm_cell_24/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2
while/lstm_cell_24/Sigmoid_2
while/lstm_cell_24/Relu_1Reluwhile/lstm_cell_24/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2
while/lstm_cell_24/Relu_1¹
while/lstm_cell_24/mul_2Mul while/lstm_cell_24/Sigmoid_2:y:0'while/lstm_cell_24/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2
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
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_24/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_24/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2
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
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿô:ÿÿÿÿÿÿÿÿÿô: : : : : 2V
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
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô:

_output_shapes
: :

_output_shapes
: 


+__inference_dense_12_layer_call_fn_40080166

inputs
unknown:8
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
F__inference_dense_12_layer_call_and_return_conditional_losses_400774712
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
:ÿÿÿÿÿÿÿÿÿ8: : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ8
 
_user_specified_nameinputs
ú	
È
&__inference_signature_wrapper_40078066
lstm_24_input
unknown:	]Ð
	unknown_0:
ôÐ
	unknown_1:	Ð
	unknown_2:
ôà
	unknown_3:	8à
	unknown_4:	à
	unknown_5:8
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
#__inference__wrapped_model_400758432
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
»
f
-__inference_dropout_24_layer_call_fn_40079451

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
:ÿÿÿÿÿÿÿÿÿô* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_dropout_24_layer_call_and_return_conditional_losses_400777232
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿô22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿô
 
_user_specified_nameinputs
\

E__inference_lstm_24_layer_call_and_return_conditional_losses_40079229

inputs>
+lstm_cell_24_matmul_readvariableop_resource:	]ÐA
-lstm_cell_24_matmul_1_readvariableop_resource:
ôÐ;
,lstm_cell_24_biasadd_readvariableop_resource:	Ð
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
strided_slice]
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :ô2
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
B :ô2
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
:ÿÿÿÿÿÿÿÿÿô2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :ô2
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
B :ô2
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
:ÿÿÿÿÿÿÿÿÿô2	
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
:	]Ð*
dtype02$
"lstm_cell_24/MatMul/ReadVariableOp­
lstm_cell_24/MatMulMatMulstrided_slice_2:output:0*lstm_cell_24/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
lstm_cell_24/MatMul¼
$lstm_cell_24/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_24_matmul_1_readvariableop_resource* 
_output_shapes
:
ôÐ*
dtype02&
$lstm_cell_24/MatMul_1/ReadVariableOp©
lstm_cell_24/MatMul_1MatMulzeros:output:0,lstm_cell_24/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
lstm_cell_24/MatMul_1 
lstm_cell_24/addAddV2lstm_cell_24/MatMul:product:0lstm_cell_24/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
lstm_cell_24/add´
#lstm_cell_24/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_24_biasadd_readvariableop_resource*
_output_shapes	
:Ð*
dtype02%
#lstm_cell_24/BiasAdd/ReadVariableOp­
lstm_cell_24/BiasAddBiasAddlstm_cell_24/add:z:0+lstm_cell_24/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
lstm_cell_24/BiasAdd~
lstm_cell_24/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_24/split/split_dim÷
lstm_cell_24/splitSplit%lstm_cell_24/split/split_dim:output:0lstm_cell_24/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿô:ÿÿÿÿÿÿÿÿÿô:ÿÿÿÿÿÿÿÿÿô:ÿÿÿÿÿÿÿÿÿô*
	num_split2
lstm_cell_24/split
lstm_cell_24/SigmoidSigmoidlstm_cell_24/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2
lstm_cell_24/Sigmoid
lstm_cell_24/Sigmoid_1Sigmoidlstm_cell_24/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2
lstm_cell_24/Sigmoid_1
lstm_cell_24/mulMullstm_cell_24/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2
lstm_cell_24/mul~
lstm_cell_24/ReluRelulstm_cell_24/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2
lstm_cell_24/Relu
lstm_cell_24/mul_1Mullstm_cell_24/Sigmoid:y:0lstm_cell_24/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2
lstm_cell_24/mul_1
lstm_cell_24/add_1AddV2lstm_cell_24/mul:z:0lstm_cell_24/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2
lstm_cell_24/add_1
lstm_cell_24/Sigmoid_2Sigmoidlstm_cell_24/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2
lstm_cell_24/Sigmoid_2}
lstm_cell_24/Relu_1Relulstm_cell_24/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2
lstm_cell_24/Relu_1¡
lstm_cell_24/mul_2Mullstm_cell_24/Sigmoid_2:y:0!lstm_cell_24/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2
lstm_cell_24/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿô  2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_24_matmul_readvariableop_resource-lstm_cell_24_matmul_1_readvariableop_resource,lstm_cell_24_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿô:ÿÿÿÿÿÿÿÿÿô: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_40079145*
condR
while_cond_40079144*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿô:ÿÿÿÿÿÿÿÿÿô: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿô  22
0TensorArrayV2Stack/TensorListStack/element_shapeé
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿô*
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
:ÿÿÿÿÿÿÿÿÿô*
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
:ÿÿÿÿÿÿÿÿÿô2
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
:ÿÿÿÿÿÿÿÿÿô2

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
ã
Í
while_cond_40079144
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_40079144___redundant_placeholder06
2while_while_cond_40079144___redundant_placeholder16
2while_while_cond_40079144___redundant_placeholder26
2while_while_cond_40079144___redundant_placeholder3
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
B: : : : :ÿÿÿÿÿÿÿÿÿô:ÿÿÿÿÿÿÿÿÿô: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô:

_output_shapes
: :

_output_shapes
:
¬

Ò
0__inference_sequential_12_layer_call_fn_40077497
lstm_24_input
unknown:	]Ð
	unknown_0:
ôÐ
	unknown_1:	Ð
	unknown_2:
ôà
	unknown_3:	8à
	unknown_4:	à
	unknown_5:8
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
K__inference_sequential_12_layer_call_and_return_conditional_losses_400774782
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
?
Ô
while_body_40079518
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
3while_lstm_cell_25_matmul_readvariableop_resource_0:
ôàH
5while_lstm_cell_25_matmul_1_readvariableop_resource_0:	8àC
4while_lstm_cell_25_biasadd_readvariableop_resource_0:	à
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
1while_lstm_cell_25_matmul_readvariableop_resource:
ôàF
3while_lstm_cell_25_matmul_1_readvariableop_resource:	8àA
2while_lstm_cell_25_biasadd_readvariableop_resource:	à¢)while/lstm_cell_25/BiasAdd/ReadVariableOp¢(while/lstm_cell_25/MatMul/ReadVariableOp¢*while/lstm_cell_25/MatMul_1/ReadVariableOpÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿô  29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÔ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemÊ
(while/lstm_cell_25/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_25_matmul_readvariableop_resource_0* 
_output_shapes
:
ôà*
dtype02*
(while/lstm_cell_25/MatMul/ReadVariableOp×
while/lstm_cell_25/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_25/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà2
while/lstm_cell_25/MatMulÏ
*while/lstm_cell_25/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_25_matmul_1_readvariableop_resource_0*
_output_shapes
:	8à*
dtype02,
*while/lstm_cell_25/MatMul_1/ReadVariableOpÀ
while/lstm_cell_25/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_25/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà2
while/lstm_cell_25/MatMul_1¸
while/lstm_cell_25/addAddV2#while/lstm_cell_25/MatMul:product:0%while/lstm_cell_25/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà2
while/lstm_cell_25/addÈ
)while/lstm_cell_25/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_25_biasadd_readvariableop_resource_0*
_output_shapes	
:à*
dtype02+
)while/lstm_cell_25/BiasAdd/ReadVariableOpÅ
while/lstm_cell_25/BiasAddBiasAddwhile/lstm_cell_25/add:z:01while/lstm_cell_25/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà2
while/lstm_cell_25/BiasAdd
"while/lstm_cell_25/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_25/split/split_dim
while/lstm_cell_25/splitSplit+while/lstm_cell_25/split/split_dim:output:0#while/lstm_cell_25/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ8:ÿÿÿÿÿÿÿÿÿ8:ÿÿÿÿÿÿÿÿÿ8:ÿÿÿÿÿÿÿÿÿ8*
	num_split2
while/lstm_cell_25/split
while/lstm_cell_25/SigmoidSigmoid!while/lstm_cell_25/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82
while/lstm_cell_25/Sigmoid
while/lstm_cell_25/Sigmoid_1Sigmoid!while/lstm_cell_25/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82
while/lstm_cell_25/Sigmoid_1 
while/lstm_cell_25/mulMul while/lstm_cell_25/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82
while/lstm_cell_25/mul
while/lstm_cell_25/ReluRelu!while/lstm_cell_25/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82
while/lstm_cell_25/Relu´
while/lstm_cell_25/mul_1Mulwhile/lstm_cell_25/Sigmoid:y:0%while/lstm_cell_25/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82
while/lstm_cell_25/mul_1©
while/lstm_cell_25/add_1AddV2while/lstm_cell_25/mul:z:0while/lstm_cell_25/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82
while/lstm_cell_25/add_1
while/lstm_cell_25/Sigmoid_2Sigmoid!while/lstm_cell_25/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82
while/lstm_cell_25/Sigmoid_2
while/lstm_cell_25/Relu_1Reluwhile/lstm_cell_25/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82
while/lstm_cell_25/Relu_1¸
while/lstm_cell_25/mul_2Mul while/lstm_cell_25/Sigmoid_2:y:0'while/lstm_cell_25/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82
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
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_25/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_25/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82
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
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ8:ÿÿÿÿÿÿÿÿÿ8: : : : : 2V
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
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ8:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ8:

_output_shapes
: :

_output_shapes
: 
&
ó
while_body_40075932
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_00
while_lstm_cell_24_40075956_0:	]Ð1
while_lstm_cell_24_40075958_0:
ôÐ,
while_lstm_cell_24_40075960_0:	Ð
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor.
while_lstm_cell_24_40075956:	]Ð/
while_lstm_cell_24_40075958:
ôÐ*
while_lstm_cell_24_40075960:	Ð¢*while/lstm_cell_24/StatefulPartitionedCallÃ
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
*while/lstm_cell_24/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_24_40075956_0while_lstm_cell_24_40075958_0while_lstm_cell_24_40075960_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿô:ÿÿÿÿÿÿÿÿÿô:ÿÿÿÿÿÿÿÿÿô*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_lstm_cell_24_layer_call_and_return_conditional_losses_400759182,
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
while/Identity_3¥
while/Identity_4Identity3while/lstm_cell_24/StatefulPartitionedCall:output:1^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2
while/Identity_4¥
while/Identity_5Identity3while/lstm_cell_24/StatefulPartitionedCall:output:2^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2
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
while_lstm_cell_24_40075956while_lstm_cell_24_40075956_0"<
while_lstm_cell_24_40075958while_lstm_cell_24_40075958_0"<
while_lstm_cell_24_40075960while_lstm_cell_24_40075960_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿô:ÿÿÿÿÿÿÿÿÿô: : : : : 2X
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
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô:

_output_shapes
: :

_output_shapes
: 
Ç
ù
/__inference_lstm_cell_24_layer_call_fn_40080264

inputs
states_0
states_1
unknown:	]Ð
	unknown_0:
ôÐ
	unknown_1:	Ð
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
<:ÿÿÿÿÿÿÿÿÿô:ÿÿÿÿÿÿÿÿÿô:ÿÿÿÿÿÿÿÿÿô*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_lstm_cell_24_layer_call_and_return_conditional_losses_400760642
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2

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
A:ÿÿÿÿÿÿÿÿÿ]:ÿÿÿÿÿÿÿÿÿô:ÿÿÿÿÿÿÿÿÿô: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
 
_user_specified_nameinputs:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô
"
_user_specified_name
states/0:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô
"
_user_specified_name
states/1
ã
Í
while_cond_40078842
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_40078842___redundant_placeholder06
2while_while_cond_40078842___redundant_placeholder16
2while_while_cond_40078842___redundant_placeholder26
2while_while_cond_40078842___redundant_placeholder3
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
B: : : : :ÿÿÿÿÿÿÿÿÿô:ÿÿÿÿÿÿÿÿÿô: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô:

_output_shapes
: :

_output_shapes
:
Ð!
ý
F__inference_dense_12_layer_call_and_return_conditional_losses_40077471

inputs3
!tensordot_readvariableop_resource:8-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢Tensordot/ReadVariableOp
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:8*
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
:ÿÿÿÿÿÿÿÿÿ82
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
:ÿÿÿÿÿÿÿÿÿ8: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ8
 
_user_specified_nameinputs
?
Ô
while_body_40077610
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
3while_lstm_cell_25_matmul_readvariableop_resource_0:
ôàH
5while_lstm_cell_25_matmul_1_readvariableop_resource_0:	8àC
4while_lstm_cell_25_biasadd_readvariableop_resource_0:	à
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
1while_lstm_cell_25_matmul_readvariableop_resource:
ôàF
3while_lstm_cell_25_matmul_1_readvariableop_resource:	8àA
2while_lstm_cell_25_biasadd_readvariableop_resource:	à¢)while/lstm_cell_25/BiasAdd/ReadVariableOp¢(while/lstm_cell_25/MatMul/ReadVariableOp¢*while/lstm_cell_25/MatMul_1/ReadVariableOpÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿô  29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÔ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemÊ
(while/lstm_cell_25/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_25_matmul_readvariableop_resource_0* 
_output_shapes
:
ôà*
dtype02*
(while/lstm_cell_25/MatMul/ReadVariableOp×
while/lstm_cell_25/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_25/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà2
while/lstm_cell_25/MatMulÏ
*while/lstm_cell_25/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_25_matmul_1_readvariableop_resource_0*
_output_shapes
:	8à*
dtype02,
*while/lstm_cell_25/MatMul_1/ReadVariableOpÀ
while/lstm_cell_25/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_25/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà2
while/lstm_cell_25/MatMul_1¸
while/lstm_cell_25/addAddV2#while/lstm_cell_25/MatMul:product:0%while/lstm_cell_25/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà2
while/lstm_cell_25/addÈ
)while/lstm_cell_25/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_25_biasadd_readvariableop_resource_0*
_output_shapes	
:à*
dtype02+
)while/lstm_cell_25/BiasAdd/ReadVariableOpÅ
while/lstm_cell_25/BiasAddBiasAddwhile/lstm_cell_25/add:z:01while/lstm_cell_25/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà2
while/lstm_cell_25/BiasAdd
"while/lstm_cell_25/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_25/split/split_dim
while/lstm_cell_25/splitSplit+while/lstm_cell_25/split/split_dim:output:0#while/lstm_cell_25/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ8:ÿÿÿÿÿÿÿÿÿ8:ÿÿÿÿÿÿÿÿÿ8:ÿÿÿÿÿÿÿÿÿ8*
	num_split2
while/lstm_cell_25/split
while/lstm_cell_25/SigmoidSigmoid!while/lstm_cell_25/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82
while/lstm_cell_25/Sigmoid
while/lstm_cell_25/Sigmoid_1Sigmoid!while/lstm_cell_25/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82
while/lstm_cell_25/Sigmoid_1 
while/lstm_cell_25/mulMul while/lstm_cell_25/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82
while/lstm_cell_25/mul
while/lstm_cell_25/ReluRelu!while/lstm_cell_25/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82
while/lstm_cell_25/Relu´
while/lstm_cell_25/mul_1Mulwhile/lstm_cell_25/Sigmoid:y:0%while/lstm_cell_25/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82
while/lstm_cell_25/mul_1©
while/lstm_cell_25/add_1AddV2while/lstm_cell_25/mul:z:0while/lstm_cell_25/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82
while/lstm_cell_25/add_1
while/lstm_cell_25/Sigmoid_2Sigmoid!while/lstm_cell_25/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82
while/lstm_cell_25/Sigmoid_2
while/lstm_cell_25/Relu_1Reluwhile/lstm_cell_25/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82
while/lstm_cell_25/Relu_1¸
while/lstm_cell_25/mul_2Mul while/lstm_cell_25/Sigmoid_2:y:0'while/lstm_cell_25/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82
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
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_25/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_25/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82
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
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ8:ÿÿÿÿÿÿÿÿÿ8: : : : : 2V
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
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ8:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ8:

_output_shapes
: :

_output_shapes
: 
¤
ï
K__inference_sequential_12_layer_call_and_return_conditional_losses_40077478

inputs#
lstm_24_40077261:	]Ð$
lstm_24_40077263:
ôÐ
lstm_24_40077265:	Ð$
lstm_25_40077426:
ôà#
lstm_25_40077428:	8à
lstm_25_40077430:	à#
dense_12_40077472:8
dense_12_40077474:
identity¢ dense_12/StatefulPartitionedCall¢lstm_24/StatefulPartitionedCall¢lstm_25/StatefulPartitionedCall®
lstm_24/StatefulPartitionedCallStatefulPartitionedCallinputslstm_24_40077261lstm_24_40077263lstm_24_40077265*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿô*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_lstm_24_layer_call_and_return_conditional_losses_400772602!
lstm_24/StatefulPartitionedCall
dropout_24/PartitionedCallPartitionedCall(lstm_24/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿô* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_dropout_24_layer_call_and_return_conditional_losses_400772732
dropout_24/PartitionedCallÊ
lstm_25/StatefulPartitionedCallStatefulPartitionedCall#dropout_24/PartitionedCall:output:0lstm_25_40077426lstm_25_40077428lstm_25_40077430*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ8*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_lstm_25_layer_call_and_return_conditional_losses_400774252!
lstm_25/StatefulPartitionedCall
dropout_25/PartitionedCallPartitionedCall(lstm_25/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ8* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_dropout_25_layer_call_and_return_conditional_losses_400774382
dropout_25/PartitionedCall»
 dense_12/StatefulPartitionedCallStatefulPartitionedCall#dropout_25/PartitionedCall:output:0dense_12_40077472dense_12_40077474*
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
F__inference_dense_12_layer_call_and_return_conditional_losses_400774712"
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
ã
Í
while_cond_40077805
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_40077805___redundant_placeholder06
2while_while_cond_40077805___redundant_placeholder16
2while_while_cond_40077805___redundant_placeholder26
2while_while_cond_40077805___redundant_placeholder3
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
B: : : : :ÿÿÿÿÿÿÿÿÿô:ÿÿÿÿÿÿÿÿÿô: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô:

_output_shapes
: :

_output_shapes
:
¾F

E__inference_lstm_25_layer_call_and_return_conditional_losses_40076841

inputs)
lstm_cell_25_40076759:
ôà(
lstm_cell_25_40076761:	8à$
lstm_cell_25_40076763:	à
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
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :82
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
value	B :82
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
:ÿÿÿÿÿÿÿÿÿ82
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :82
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
value	B :82
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
:ÿÿÿÿÿÿÿÿÿ82	
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
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿô2
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
valueB"ÿÿÿÿô  27
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
:ÿÿÿÿÿÿÿÿÿô*
shrink_axis_mask2
strided_slice_2¥
$lstm_cell_25/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_25_40076759lstm_cell_25_40076761lstm_cell_25_40076763*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ8:ÿÿÿÿÿÿÿÿÿ8:ÿÿÿÿÿÿÿÿÿ8*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_lstm_cell_25_layer_call_and_return_conditional_losses_400766942&
$lstm_cell_25/StatefulPartitionedCall
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ8   2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_25_40076759lstm_cell_25_40076761lstm_cell_25_40076763*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ8:ÿÿÿÿÿÿÿÿÿ8: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_40076772*
condR
while_cond_40076771*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ8:ÿÿÿÿÿÿÿÿÿ8: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ8   22
0TensorArrayV2Stack/TensorListStack/element_shapeñ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ8*
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
:ÿÿÿÿÿÿÿÿÿ8*
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
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ82
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
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ82

Identity}
NoOpNoOp%^lstm_cell_25/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿô: : : 2L
$lstm_cell_25/StatefulPartitionedCall$lstm_cell_25/StatefulPartitionedCall2
whilewhile:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿô
 
_user_specified_nameinputs
Ã\
 
E__inference_lstm_24_layer_call_and_return_conditional_losses_40079078
inputs_0>
+lstm_cell_24_matmul_readvariableop_resource:	]ÐA
-lstm_cell_24_matmul_1_readvariableop_resource:
ôÐ;
,lstm_cell_24_biasadd_readvariableop_resource:	Ð
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
strided_slice]
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :ô2
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
B :ô2
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
:ÿÿÿÿÿÿÿÿÿô2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :ô2
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
B :ô2
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
:ÿÿÿÿÿÿÿÿÿô2	
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
:	]Ð*
dtype02$
"lstm_cell_24/MatMul/ReadVariableOp­
lstm_cell_24/MatMulMatMulstrided_slice_2:output:0*lstm_cell_24/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
lstm_cell_24/MatMul¼
$lstm_cell_24/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_24_matmul_1_readvariableop_resource* 
_output_shapes
:
ôÐ*
dtype02&
$lstm_cell_24/MatMul_1/ReadVariableOp©
lstm_cell_24/MatMul_1MatMulzeros:output:0,lstm_cell_24/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
lstm_cell_24/MatMul_1 
lstm_cell_24/addAddV2lstm_cell_24/MatMul:product:0lstm_cell_24/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
lstm_cell_24/add´
#lstm_cell_24/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_24_biasadd_readvariableop_resource*
_output_shapes	
:Ð*
dtype02%
#lstm_cell_24/BiasAdd/ReadVariableOp­
lstm_cell_24/BiasAddBiasAddlstm_cell_24/add:z:0+lstm_cell_24/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
lstm_cell_24/BiasAdd~
lstm_cell_24/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_24/split/split_dim÷
lstm_cell_24/splitSplit%lstm_cell_24/split/split_dim:output:0lstm_cell_24/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿô:ÿÿÿÿÿÿÿÿÿô:ÿÿÿÿÿÿÿÿÿô:ÿÿÿÿÿÿÿÿÿô*
	num_split2
lstm_cell_24/split
lstm_cell_24/SigmoidSigmoidlstm_cell_24/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2
lstm_cell_24/Sigmoid
lstm_cell_24/Sigmoid_1Sigmoidlstm_cell_24/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2
lstm_cell_24/Sigmoid_1
lstm_cell_24/mulMullstm_cell_24/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2
lstm_cell_24/mul~
lstm_cell_24/ReluRelulstm_cell_24/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2
lstm_cell_24/Relu
lstm_cell_24/mul_1Mullstm_cell_24/Sigmoid:y:0lstm_cell_24/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2
lstm_cell_24/mul_1
lstm_cell_24/add_1AddV2lstm_cell_24/mul:z:0lstm_cell_24/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2
lstm_cell_24/add_1
lstm_cell_24/Sigmoid_2Sigmoidlstm_cell_24/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2
lstm_cell_24/Sigmoid_2}
lstm_cell_24/Relu_1Relulstm_cell_24/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2
lstm_cell_24/Relu_1¡
lstm_cell_24/mul_2Mullstm_cell_24/Sigmoid_2:y:0!lstm_cell_24/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2
lstm_cell_24/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿô  2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_24_matmul_readvariableop_resource-lstm_cell_24_matmul_1_readvariableop_resource,lstm_cell_24_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿô:ÿÿÿÿÿÿÿÿÿô: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_40078994*
condR
while_cond_40078993*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿô:ÿÿÿÿÿÿÿÿÿô: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿô  22
0TensorArrayV2Stack/TensorListStack/element_shapeò
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿô*
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
:ÿÿÿÿÿÿÿÿÿô*
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
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿô2
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
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿô2

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
¾F

E__inference_lstm_25_layer_call_and_return_conditional_losses_40076631

inputs)
lstm_cell_25_40076549:
ôà(
lstm_cell_25_40076551:	8à$
lstm_cell_25_40076553:	à
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
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :82
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
value	B :82
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
:ÿÿÿÿÿÿÿÿÿ82
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :82
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
value	B :82
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
:ÿÿÿÿÿÿÿÿÿ82	
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
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿô2
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
valueB"ÿÿÿÿô  27
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
:ÿÿÿÿÿÿÿÿÿô*
shrink_axis_mask2
strided_slice_2¥
$lstm_cell_25/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_25_40076549lstm_cell_25_40076551lstm_cell_25_40076553*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ8:ÿÿÿÿÿÿÿÿÿ8:ÿÿÿÿÿÿÿÿÿ8*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_lstm_cell_25_layer_call_and_return_conditional_losses_400765482&
$lstm_cell_25/StatefulPartitionedCall
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ8   2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_25_40076549lstm_cell_25_40076551lstm_cell_25_40076553*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ8:ÿÿÿÿÿÿÿÿÿ8: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_40076562*
condR
while_cond_40076561*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ8:ÿÿÿÿÿÿÿÿÿ8: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ8   22
0TensorArrayV2Stack/TensorListStack/element_shapeñ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ8*
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
:ÿÿÿÿÿÿÿÿÿ8*
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
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ82
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
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ82

Identity}
NoOpNoOp%^lstm_cell_25/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿô: : : 2L
$lstm_cell_25/StatefulPartitionedCall$lstm_cell_25/StatefulPartitionedCall2
whilewhile:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿô
 
_user_specified_nameinputs
°?
Ô
while_body_40077176
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_24_matmul_readvariableop_resource_0:	]ÐI
5while_lstm_cell_24_matmul_1_readvariableop_resource_0:
ôÐC
4while_lstm_cell_24_biasadd_readvariableop_resource_0:	Ð
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_24_matmul_readvariableop_resource:	]ÐG
3while_lstm_cell_24_matmul_1_readvariableop_resource:
ôÐA
2while_lstm_cell_24_biasadd_readvariableop_resource:	Ð¢)while/lstm_cell_24/BiasAdd/ReadVariableOp¢(while/lstm_cell_24/MatMul/ReadVariableOp¢*while/lstm_cell_24/MatMul_1/ReadVariableOpÃ
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
:	]Ð*
dtype02*
(while/lstm_cell_24/MatMul/ReadVariableOp×
while/lstm_cell_24/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_24/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
while/lstm_cell_24/MatMulÐ
*while/lstm_cell_24/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_24_matmul_1_readvariableop_resource_0* 
_output_shapes
:
ôÐ*
dtype02,
*while/lstm_cell_24/MatMul_1/ReadVariableOpÀ
while/lstm_cell_24/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_24/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
while/lstm_cell_24/MatMul_1¸
while/lstm_cell_24/addAddV2#while/lstm_cell_24/MatMul:product:0%while/lstm_cell_24/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
while/lstm_cell_24/addÈ
)while/lstm_cell_24/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_24_biasadd_readvariableop_resource_0*
_output_shapes	
:Ð*
dtype02+
)while/lstm_cell_24/BiasAdd/ReadVariableOpÅ
while/lstm_cell_24/BiasAddBiasAddwhile/lstm_cell_24/add:z:01while/lstm_cell_24/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
while/lstm_cell_24/BiasAdd
"while/lstm_cell_24/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_24/split/split_dim
while/lstm_cell_24/splitSplit+while/lstm_cell_24/split/split_dim:output:0#while/lstm_cell_24/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿô:ÿÿÿÿÿÿÿÿÿô:ÿÿÿÿÿÿÿÿÿô:ÿÿÿÿÿÿÿÿÿô*
	num_split2
while/lstm_cell_24/split
while/lstm_cell_24/SigmoidSigmoid!while/lstm_cell_24/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2
while/lstm_cell_24/Sigmoid
while/lstm_cell_24/Sigmoid_1Sigmoid!while/lstm_cell_24/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2
while/lstm_cell_24/Sigmoid_1¡
while/lstm_cell_24/mulMul while/lstm_cell_24/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2
while/lstm_cell_24/mul
while/lstm_cell_24/ReluRelu!while/lstm_cell_24/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2
while/lstm_cell_24/Reluµ
while/lstm_cell_24/mul_1Mulwhile/lstm_cell_24/Sigmoid:y:0%while/lstm_cell_24/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2
while/lstm_cell_24/mul_1ª
while/lstm_cell_24/add_1AddV2while/lstm_cell_24/mul:z:0while/lstm_cell_24/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2
while/lstm_cell_24/add_1
while/lstm_cell_24/Sigmoid_2Sigmoid!while/lstm_cell_24/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2
while/lstm_cell_24/Sigmoid_2
while/lstm_cell_24/Relu_1Reluwhile/lstm_cell_24/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2
while/lstm_cell_24/Relu_1¹
while/lstm_cell_24/mul_2Mul while/lstm_cell_24/Sigmoid_2:y:0'while/lstm_cell_24/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2
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
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_24/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_24/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2
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
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿô:ÿÿÿÿÿÿÿÿÿô: : : : : 2V
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
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô:

_output_shapes
: :

_output_shapes
: 
åJ
Ô

lstm_24_while_body_40078133,
(lstm_24_while_lstm_24_while_loop_counter2
.lstm_24_while_lstm_24_while_maximum_iterations
lstm_24_while_placeholder
lstm_24_while_placeholder_1
lstm_24_while_placeholder_2
lstm_24_while_placeholder_3+
'lstm_24_while_lstm_24_strided_slice_1_0g
clstm_24_while_tensorarrayv2read_tensorlistgetitem_lstm_24_tensorarrayunstack_tensorlistfromtensor_0N
;lstm_24_while_lstm_cell_24_matmul_readvariableop_resource_0:	]ÐQ
=lstm_24_while_lstm_cell_24_matmul_1_readvariableop_resource_0:
ôÐK
<lstm_24_while_lstm_cell_24_biasadd_readvariableop_resource_0:	Ð
lstm_24_while_identity
lstm_24_while_identity_1
lstm_24_while_identity_2
lstm_24_while_identity_3
lstm_24_while_identity_4
lstm_24_while_identity_5)
%lstm_24_while_lstm_24_strided_slice_1e
alstm_24_while_tensorarrayv2read_tensorlistgetitem_lstm_24_tensorarrayunstack_tensorlistfromtensorL
9lstm_24_while_lstm_cell_24_matmul_readvariableop_resource:	]ÐO
;lstm_24_while_lstm_cell_24_matmul_1_readvariableop_resource:
ôÐI
:lstm_24_while_lstm_cell_24_biasadd_readvariableop_resource:	Ð¢1lstm_24/while/lstm_cell_24/BiasAdd/ReadVariableOp¢0lstm_24/while/lstm_cell_24/MatMul/ReadVariableOp¢2lstm_24/while/lstm_cell_24/MatMul_1/ReadVariableOpÓ
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
:	]Ð*
dtype022
0lstm_24/while/lstm_cell_24/MatMul/ReadVariableOp÷
!lstm_24/while/lstm_cell_24/MatMulMatMul8lstm_24/while/TensorArrayV2Read/TensorListGetItem:item:08lstm_24/while/lstm_cell_24/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2#
!lstm_24/while/lstm_cell_24/MatMulè
2lstm_24/while/lstm_cell_24/MatMul_1/ReadVariableOpReadVariableOp=lstm_24_while_lstm_cell_24_matmul_1_readvariableop_resource_0* 
_output_shapes
:
ôÐ*
dtype024
2lstm_24/while/lstm_cell_24/MatMul_1/ReadVariableOpà
#lstm_24/while/lstm_cell_24/MatMul_1MatMullstm_24_while_placeholder_2:lstm_24/while/lstm_cell_24/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2%
#lstm_24/while/lstm_cell_24/MatMul_1Ø
lstm_24/while/lstm_cell_24/addAddV2+lstm_24/while/lstm_cell_24/MatMul:product:0-lstm_24/while/lstm_cell_24/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2 
lstm_24/while/lstm_cell_24/addà
1lstm_24/while/lstm_cell_24/BiasAdd/ReadVariableOpReadVariableOp<lstm_24_while_lstm_cell_24_biasadd_readvariableop_resource_0*
_output_shapes	
:Ð*
dtype023
1lstm_24/while/lstm_cell_24/BiasAdd/ReadVariableOpå
"lstm_24/while/lstm_cell_24/BiasAddBiasAdd"lstm_24/while/lstm_cell_24/add:z:09lstm_24/while/lstm_cell_24/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2$
"lstm_24/while/lstm_cell_24/BiasAdd
*lstm_24/while/lstm_cell_24/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2,
*lstm_24/while/lstm_cell_24/split/split_dim¯
 lstm_24/while/lstm_cell_24/splitSplit3lstm_24/while/lstm_cell_24/split/split_dim:output:0+lstm_24/while/lstm_cell_24/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿô:ÿÿÿÿÿÿÿÿÿô:ÿÿÿÿÿÿÿÿÿô:ÿÿÿÿÿÿÿÿÿô*
	num_split2"
 lstm_24/while/lstm_cell_24/split±
"lstm_24/while/lstm_cell_24/SigmoidSigmoid)lstm_24/while/lstm_cell_24/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2$
"lstm_24/while/lstm_cell_24/Sigmoidµ
$lstm_24/while/lstm_cell_24/Sigmoid_1Sigmoid)lstm_24/while/lstm_cell_24/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2&
$lstm_24/while/lstm_cell_24/Sigmoid_1Á
lstm_24/while/lstm_cell_24/mulMul(lstm_24/while/lstm_cell_24/Sigmoid_1:y:0lstm_24_while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2 
lstm_24/while/lstm_cell_24/mul¨
lstm_24/while/lstm_cell_24/ReluRelu)lstm_24/while/lstm_cell_24/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2!
lstm_24/while/lstm_cell_24/ReluÕ
 lstm_24/while/lstm_cell_24/mul_1Mul&lstm_24/while/lstm_cell_24/Sigmoid:y:0-lstm_24/while/lstm_cell_24/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2"
 lstm_24/while/lstm_cell_24/mul_1Ê
 lstm_24/while/lstm_cell_24/add_1AddV2"lstm_24/while/lstm_cell_24/mul:z:0$lstm_24/while/lstm_cell_24/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2"
 lstm_24/while/lstm_cell_24/add_1µ
$lstm_24/while/lstm_cell_24/Sigmoid_2Sigmoid)lstm_24/while/lstm_cell_24/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2&
$lstm_24/while/lstm_cell_24/Sigmoid_2§
!lstm_24/while/lstm_cell_24/Relu_1Relu$lstm_24/while/lstm_cell_24/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2#
!lstm_24/while/lstm_cell_24/Relu_1Ù
 lstm_24/while/lstm_cell_24/mul_2Mul(lstm_24/while/lstm_cell_24/Sigmoid_2:y:0/lstm_24/while/lstm_cell_24/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2"
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
lstm_24/while/Identity_3®
lstm_24/while/Identity_4Identity$lstm_24/while/lstm_cell_24/mul_2:z:0^lstm_24/while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2
lstm_24/while/Identity_4®
lstm_24/while/Identity_5Identity$lstm_24/while/lstm_cell_24/add_1:z:0^lstm_24/while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2
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
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿô:ÿÿÿÿÿÿÿÿÿô: : : : : 2f
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
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô:

_output_shapes
: :

_output_shapes
: 
Ð!
ý
F__inference_dense_12_layer_call_and_return_conditional_losses_40080157

inputs3
!tensordot_readvariableop_resource:8-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢Tensordot/ReadVariableOp
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:8*
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
:ÿÿÿÿÿÿÿÿÿ82
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
:ÿÿÿÿÿÿÿÿÿ8: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ8
 
_user_specified_nameinputs
í[

E__inference_lstm_25_layer_call_and_return_conditional_losses_40080055

inputs?
+lstm_cell_25_matmul_readvariableop_resource:
ôà@
-lstm_cell_25_matmul_1_readvariableop_resource:	8à;
,lstm_cell_25_biasadd_readvariableop_resource:	à
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
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :82
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
value	B :82
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
:ÿÿÿÿÿÿÿÿÿ82
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :82
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
value	B :82
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
:ÿÿÿÿÿÿÿÿÿ82	
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
:ÿÿÿÿÿÿÿÿÿô2
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
valueB"ÿÿÿÿô  27
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
:ÿÿÿÿÿÿÿÿÿô*
shrink_axis_mask2
strided_slice_2¶
"lstm_cell_25/MatMul/ReadVariableOpReadVariableOp+lstm_cell_25_matmul_readvariableop_resource* 
_output_shapes
:
ôà*
dtype02$
"lstm_cell_25/MatMul/ReadVariableOp­
lstm_cell_25/MatMulMatMulstrided_slice_2:output:0*lstm_cell_25/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà2
lstm_cell_25/MatMul»
$lstm_cell_25/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_25_matmul_1_readvariableop_resource*
_output_shapes
:	8à*
dtype02&
$lstm_cell_25/MatMul_1/ReadVariableOp©
lstm_cell_25/MatMul_1MatMulzeros:output:0,lstm_cell_25/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà2
lstm_cell_25/MatMul_1 
lstm_cell_25/addAddV2lstm_cell_25/MatMul:product:0lstm_cell_25/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà2
lstm_cell_25/add´
#lstm_cell_25/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_25_biasadd_readvariableop_resource*
_output_shapes	
:à*
dtype02%
#lstm_cell_25/BiasAdd/ReadVariableOp­
lstm_cell_25/BiasAddBiasAddlstm_cell_25/add:z:0+lstm_cell_25/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà2
lstm_cell_25/BiasAdd~
lstm_cell_25/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_25/split/split_dimó
lstm_cell_25/splitSplit%lstm_cell_25/split/split_dim:output:0lstm_cell_25/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ8:ÿÿÿÿÿÿÿÿÿ8:ÿÿÿÿÿÿÿÿÿ8:ÿÿÿÿÿÿÿÿÿ8*
	num_split2
lstm_cell_25/split
lstm_cell_25/SigmoidSigmoidlstm_cell_25/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82
lstm_cell_25/Sigmoid
lstm_cell_25/Sigmoid_1Sigmoidlstm_cell_25/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82
lstm_cell_25/Sigmoid_1
lstm_cell_25/mulMullstm_cell_25/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82
lstm_cell_25/mul}
lstm_cell_25/ReluRelulstm_cell_25/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82
lstm_cell_25/Relu
lstm_cell_25/mul_1Mullstm_cell_25/Sigmoid:y:0lstm_cell_25/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82
lstm_cell_25/mul_1
lstm_cell_25/add_1AddV2lstm_cell_25/mul:z:0lstm_cell_25/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82
lstm_cell_25/add_1
lstm_cell_25/Sigmoid_2Sigmoidlstm_cell_25/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82
lstm_cell_25/Sigmoid_2|
lstm_cell_25/Relu_1Relulstm_cell_25/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82
lstm_cell_25/Relu_1 
lstm_cell_25/mul_2Mullstm_cell_25/Sigmoid_2:y:0!lstm_cell_25/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82
lstm_cell_25/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ8   2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_25_matmul_readvariableop_resource-lstm_cell_25_matmul_1_readvariableop_resource,lstm_cell_25_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ8:ÿÿÿÿÿÿÿÿÿ8: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_40079971*
condR
while_cond_40079970*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ8:ÿÿÿÿÿÿÿÿÿ8: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ8   22
0TensorArrayV2Stack/TensorListStack/element_shapeè
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ8*
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
:ÿÿÿÿÿÿÿÿÿ8*
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
:ÿÿÿÿÿÿÿÿÿ82
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
:ÿÿÿÿÿÿÿÿÿ82

IdentityÈ
NoOpNoOp$^lstm_cell_25/BiasAdd/ReadVariableOp#^lstm_cell_25/MatMul/ReadVariableOp%^lstm_cell_25/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:ÿÿÿÿÿÿÿÿÿô: : : 2J
#lstm_cell_25/BiasAdd/ReadVariableOp#lstm_cell_25/BiasAdd/ReadVariableOp2H
"lstm_cell_25/MatMul/ReadVariableOp"lstm_cell_25/MatMul/ReadVariableOp2L
$lstm_cell_25/MatMul_1/ReadVariableOp$lstm_cell_25/MatMul_1/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿô
 
_user_specified_nameinputs
×
g
H__inference_dropout_24_layer_call_and_return_conditional_losses_40077723

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
:ÿÿÿÿÿÿÿÿÿô2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape¹
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿô*
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
:ÿÿÿÿÿÿÿÿÿô2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2
dropout/Mul_1j
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿô:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿô
 
_user_specified_nameinputs
Ô

í
lstm_24_while_cond_40078132,
(lstm_24_while_lstm_24_while_loop_counter2
.lstm_24_while_lstm_24_while_maximum_iterations
lstm_24_while_placeholder
lstm_24_while_placeholder_1
lstm_24_while_placeholder_2
lstm_24_while_placeholder_3.
*lstm_24_while_less_lstm_24_strided_slice_1F
Blstm_24_while_lstm_24_while_cond_40078132___redundant_placeholder0F
Blstm_24_while_lstm_24_while_cond_40078132___redundant_placeholder1F
Blstm_24_while_lstm_24_while_cond_40078132___redundant_placeholder2F
Blstm_24_while_lstm_24_while_cond_40078132___redundant_placeholder3
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
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :ÿÿÿÿÿÿÿÿÿô:ÿÿÿÿÿÿÿÿÿô: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô:

_output_shapes
: :

_output_shapes
:
ß
Í
while_cond_40077609
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_40077609___redundant_placeholder06
2while_while_cond_40077609___redundant_placeholder16
2while_while_cond_40077609___redundant_placeholder26
2while_while_cond_40077609___redundant_placeholder3
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
@: : : : :ÿÿÿÿÿÿÿÿÿ8:ÿÿÿÿÿÿÿÿÿ8: ::::: 

_output_shapes
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
:ÿÿÿÿÿÿÿÿÿ8:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ8:

_output_shapes
: :

_output_shapes
:
Ð

í
lstm_25_while_cond_40078280,
(lstm_25_while_lstm_25_while_loop_counter2
.lstm_25_while_lstm_25_while_maximum_iterations
lstm_25_while_placeholder
lstm_25_while_placeholder_1
lstm_25_while_placeholder_2
lstm_25_while_placeholder_3.
*lstm_25_while_less_lstm_25_strided_slice_1F
Blstm_25_while_lstm_25_while_cond_40078280___redundant_placeholder0F
Blstm_25_while_lstm_25_while_cond_40078280___redundant_placeholder1F
Blstm_25_while_lstm_25_while_cond_40078280___redundant_placeholder2F
Blstm_25_while_lstm_25_while_cond_40078280___redundant_placeholder3
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
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿ8:ÿÿÿÿÿÿÿÿÿ8: ::::: 

_output_shapes
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
:ÿÿÿÿÿÿÿÿÿ8:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ8:

_output_shapes
: :

_output_shapes
:
×
g
H__inference_dropout_24_layer_call_and_return_conditional_losses_40079441

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
:ÿÿÿÿÿÿÿÿÿô2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape¹
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿô*
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
:ÿÿÿÿÿÿÿÿÿô2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2
dropout/Mul_1j
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿô:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿô
 
_user_specified_nameinputs
°?
Ô
while_body_40078843
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_24_matmul_readvariableop_resource_0:	]ÐI
5while_lstm_cell_24_matmul_1_readvariableop_resource_0:
ôÐC
4while_lstm_cell_24_biasadd_readvariableop_resource_0:	Ð
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_24_matmul_readvariableop_resource:	]ÐG
3while_lstm_cell_24_matmul_1_readvariableop_resource:
ôÐA
2while_lstm_cell_24_biasadd_readvariableop_resource:	Ð¢)while/lstm_cell_24/BiasAdd/ReadVariableOp¢(while/lstm_cell_24/MatMul/ReadVariableOp¢*while/lstm_cell_24/MatMul_1/ReadVariableOpÃ
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
:	]Ð*
dtype02*
(while/lstm_cell_24/MatMul/ReadVariableOp×
while/lstm_cell_24/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_24/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
while/lstm_cell_24/MatMulÐ
*while/lstm_cell_24/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_24_matmul_1_readvariableop_resource_0* 
_output_shapes
:
ôÐ*
dtype02,
*while/lstm_cell_24/MatMul_1/ReadVariableOpÀ
while/lstm_cell_24/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_24/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
while/lstm_cell_24/MatMul_1¸
while/lstm_cell_24/addAddV2#while/lstm_cell_24/MatMul:product:0%while/lstm_cell_24/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
while/lstm_cell_24/addÈ
)while/lstm_cell_24/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_24_biasadd_readvariableop_resource_0*
_output_shapes	
:Ð*
dtype02+
)while/lstm_cell_24/BiasAdd/ReadVariableOpÅ
while/lstm_cell_24/BiasAddBiasAddwhile/lstm_cell_24/add:z:01while/lstm_cell_24/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2
while/lstm_cell_24/BiasAdd
"while/lstm_cell_24/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_24/split/split_dim
while/lstm_cell_24/splitSplit+while/lstm_cell_24/split/split_dim:output:0#while/lstm_cell_24/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿô:ÿÿÿÿÿÿÿÿÿô:ÿÿÿÿÿÿÿÿÿô:ÿÿÿÿÿÿÿÿÿô*
	num_split2
while/lstm_cell_24/split
while/lstm_cell_24/SigmoidSigmoid!while/lstm_cell_24/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2
while/lstm_cell_24/Sigmoid
while/lstm_cell_24/Sigmoid_1Sigmoid!while/lstm_cell_24/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2
while/lstm_cell_24/Sigmoid_1¡
while/lstm_cell_24/mulMul while/lstm_cell_24/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2
while/lstm_cell_24/mul
while/lstm_cell_24/ReluRelu!while/lstm_cell_24/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2
while/lstm_cell_24/Reluµ
while/lstm_cell_24/mul_1Mulwhile/lstm_cell_24/Sigmoid:y:0%while/lstm_cell_24/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2
while/lstm_cell_24/mul_1ª
while/lstm_cell_24/add_1AddV2while/lstm_cell_24/mul:z:0while/lstm_cell_24/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2
while/lstm_cell_24/add_1
while/lstm_cell_24/Sigmoid_2Sigmoid!while/lstm_cell_24/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2
while/lstm_cell_24/Sigmoid_2
while/lstm_cell_24/Relu_1Reluwhile/lstm_cell_24/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2
while/lstm_cell_24/Relu_1¹
while/lstm_cell_24/mul_2Mul while/lstm_cell_24/Sigmoid_2:y:0'while/lstm_cell_24/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2
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
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_24/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_24/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2
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
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿô:ÿÿÿÿÿÿÿÿÿô: : : : : 2V
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
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô:

_output_shapes
: :

_output_shapes
: 
Ô
I
-__inference_dropout_25_layer_call_fn_40080121

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
:ÿÿÿÿÿÿÿÿÿ8* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_dropout_25_layer_call_and_return_conditional_losses_400774382
PartitionedCallp
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ8:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ8
 
_user_specified_nameinputs
à
º
*__inference_lstm_25_layer_call_fn_40080066
inputs_0
unknown:
ôà
	unknown_0:	8à
	unknown_1:	à
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ8*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_lstm_25_layer_call_and_return_conditional_losses_400766312
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ82

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿô: : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿô
"
_user_specified_name
inputs/0
?
Ô
while_body_40079669
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
3while_lstm_cell_25_matmul_readvariableop_resource_0:
ôàH
5while_lstm_cell_25_matmul_1_readvariableop_resource_0:	8àC
4while_lstm_cell_25_biasadd_readvariableop_resource_0:	à
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
1while_lstm_cell_25_matmul_readvariableop_resource:
ôàF
3while_lstm_cell_25_matmul_1_readvariableop_resource:	8àA
2while_lstm_cell_25_biasadd_readvariableop_resource:	à¢)while/lstm_cell_25/BiasAdd/ReadVariableOp¢(while/lstm_cell_25/MatMul/ReadVariableOp¢*while/lstm_cell_25/MatMul_1/ReadVariableOpÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿô  29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÔ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemÊ
(while/lstm_cell_25/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_25_matmul_readvariableop_resource_0* 
_output_shapes
:
ôà*
dtype02*
(while/lstm_cell_25/MatMul/ReadVariableOp×
while/lstm_cell_25/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_25/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà2
while/lstm_cell_25/MatMulÏ
*while/lstm_cell_25/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_25_matmul_1_readvariableop_resource_0*
_output_shapes
:	8à*
dtype02,
*while/lstm_cell_25/MatMul_1/ReadVariableOpÀ
while/lstm_cell_25/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_25/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà2
while/lstm_cell_25/MatMul_1¸
while/lstm_cell_25/addAddV2#while/lstm_cell_25/MatMul:product:0%while/lstm_cell_25/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà2
while/lstm_cell_25/addÈ
)while/lstm_cell_25/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_25_biasadd_readvariableop_resource_0*
_output_shapes	
:à*
dtype02+
)while/lstm_cell_25/BiasAdd/ReadVariableOpÅ
while/lstm_cell_25/BiasAddBiasAddwhile/lstm_cell_25/add:z:01while/lstm_cell_25/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà2
while/lstm_cell_25/BiasAdd
"while/lstm_cell_25/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_25/split/split_dim
while/lstm_cell_25/splitSplit+while/lstm_cell_25/split/split_dim:output:0#while/lstm_cell_25/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ8:ÿÿÿÿÿÿÿÿÿ8:ÿÿÿÿÿÿÿÿÿ8:ÿÿÿÿÿÿÿÿÿ8*
	num_split2
while/lstm_cell_25/split
while/lstm_cell_25/SigmoidSigmoid!while/lstm_cell_25/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82
while/lstm_cell_25/Sigmoid
while/lstm_cell_25/Sigmoid_1Sigmoid!while/lstm_cell_25/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82
while/lstm_cell_25/Sigmoid_1 
while/lstm_cell_25/mulMul while/lstm_cell_25/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82
while/lstm_cell_25/mul
while/lstm_cell_25/ReluRelu!while/lstm_cell_25/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82
while/lstm_cell_25/Relu´
while/lstm_cell_25/mul_1Mulwhile/lstm_cell_25/Sigmoid:y:0%while/lstm_cell_25/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82
while/lstm_cell_25/mul_1©
while/lstm_cell_25/add_1AddV2while/lstm_cell_25/mul:z:0while/lstm_cell_25/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82
while/lstm_cell_25/add_1
while/lstm_cell_25/Sigmoid_2Sigmoid!while/lstm_cell_25/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82
while/lstm_cell_25/Sigmoid_2
while/lstm_cell_25/Relu_1Reluwhile/lstm_cell_25/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82
while/lstm_cell_25/Relu_1¸
while/lstm_cell_25/mul_2Mul while/lstm_cell_25/Sigmoid_2:y:0'while/lstm_cell_25/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82
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
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_25/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_25/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82
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
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ8:ÿÿÿÿÿÿÿÿÿ8: : : : : 2V
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
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ8:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ8:

_output_shapes
: :

_output_shapes
: 
åJ
Ô

lstm_24_while_body_40078460,
(lstm_24_while_lstm_24_while_loop_counter2
.lstm_24_while_lstm_24_while_maximum_iterations
lstm_24_while_placeholder
lstm_24_while_placeholder_1
lstm_24_while_placeholder_2
lstm_24_while_placeholder_3+
'lstm_24_while_lstm_24_strided_slice_1_0g
clstm_24_while_tensorarrayv2read_tensorlistgetitem_lstm_24_tensorarrayunstack_tensorlistfromtensor_0N
;lstm_24_while_lstm_cell_24_matmul_readvariableop_resource_0:	]ÐQ
=lstm_24_while_lstm_cell_24_matmul_1_readvariableop_resource_0:
ôÐK
<lstm_24_while_lstm_cell_24_biasadd_readvariableop_resource_0:	Ð
lstm_24_while_identity
lstm_24_while_identity_1
lstm_24_while_identity_2
lstm_24_while_identity_3
lstm_24_while_identity_4
lstm_24_while_identity_5)
%lstm_24_while_lstm_24_strided_slice_1e
alstm_24_while_tensorarrayv2read_tensorlistgetitem_lstm_24_tensorarrayunstack_tensorlistfromtensorL
9lstm_24_while_lstm_cell_24_matmul_readvariableop_resource:	]ÐO
;lstm_24_while_lstm_cell_24_matmul_1_readvariableop_resource:
ôÐI
:lstm_24_while_lstm_cell_24_biasadd_readvariableop_resource:	Ð¢1lstm_24/while/lstm_cell_24/BiasAdd/ReadVariableOp¢0lstm_24/while/lstm_cell_24/MatMul/ReadVariableOp¢2lstm_24/while/lstm_cell_24/MatMul_1/ReadVariableOpÓ
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
:	]Ð*
dtype022
0lstm_24/while/lstm_cell_24/MatMul/ReadVariableOp÷
!lstm_24/while/lstm_cell_24/MatMulMatMul8lstm_24/while/TensorArrayV2Read/TensorListGetItem:item:08lstm_24/while/lstm_cell_24/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2#
!lstm_24/while/lstm_cell_24/MatMulè
2lstm_24/while/lstm_cell_24/MatMul_1/ReadVariableOpReadVariableOp=lstm_24_while_lstm_cell_24_matmul_1_readvariableop_resource_0* 
_output_shapes
:
ôÐ*
dtype024
2lstm_24/while/lstm_cell_24/MatMul_1/ReadVariableOpà
#lstm_24/while/lstm_cell_24/MatMul_1MatMullstm_24_while_placeholder_2:lstm_24/while/lstm_cell_24/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2%
#lstm_24/while/lstm_cell_24/MatMul_1Ø
lstm_24/while/lstm_cell_24/addAddV2+lstm_24/while/lstm_cell_24/MatMul:product:0-lstm_24/while/lstm_cell_24/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2 
lstm_24/while/lstm_cell_24/addà
1lstm_24/while/lstm_cell_24/BiasAdd/ReadVariableOpReadVariableOp<lstm_24_while_lstm_cell_24_biasadd_readvariableop_resource_0*
_output_shapes	
:Ð*
dtype023
1lstm_24/while/lstm_cell_24/BiasAdd/ReadVariableOpå
"lstm_24/while/lstm_cell_24/BiasAddBiasAdd"lstm_24/while/lstm_cell_24/add:z:09lstm_24/while/lstm_cell_24/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ2$
"lstm_24/while/lstm_cell_24/BiasAdd
*lstm_24/while/lstm_cell_24/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2,
*lstm_24/while/lstm_cell_24/split/split_dim¯
 lstm_24/while/lstm_cell_24/splitSplit3lstm_24/while/lstm_cell_24/split/split_dim:output:0+lstm_24/while/lstm_cell_24/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿô:ÿÿÿÿÿÿÿÿÿô:ÿÿÿÿÿÿÿÿÿô:ÿÿÿÿÿÿÿÿÿô*
	num_split2"
 lstm_24/while/lstm_cell_24/split±
"lstm_24/while/lstm_cell_24/SigmoidSigmoid)lstm_24/while/lstm_cell_24/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2$
"lstm_24/while/lstm_cell_24/Sigmoidµ
$lstm_24/while/lstm_cell_24/Sigmoid_1Sigmoid)lstm_24/while/lstm_cell_24/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2&
$lstm_24/while/lstm_cell_24/Sigmoid_1Á
lstm_24/while/lstm_cell_24/mulMul(lstm_24/while/lstm_cell_24/Sigmoid_1:y:0lstm_24_while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2 
lstm_24/while/lstm_cell_24/mul¨
lstm_24/while/lstm_cell_24/ReluRelu)lstm_24/while/lstm_cell_24/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2!
lstm_24/while/lstm_cell_24/ReluÕ
 lstm_24/while/lstm_cell_24/mul_1Mul&lstm_24/while/lstm_cell_24/Sigmoid:y:0-lstm_24/while/lstm_cell_24/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2"
 lstm_24/while/lstm_cell_24/mul_1Ê
 lstm_24/while/lstm_cell_24/add_1AddV2"lstm_24/while/lstm_cell_24/mul:z:0$lstm_24/while/lstm_cell_24/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2"
 lstm_24/while/lstm_cell_24/add_1µ
$lstm_24/while/lstm_cell_24/Sigmoid_2Sigmoid)lstm_24/while/lstm_cell_24/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2&
$lstm_24/while/lstm_cell_24/Sigmoid_2§
!lstm_24/while/lstm_cell_24/Relu_1Relu$lstm_24/while/lstm_cell_24/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2#
!lstm_24/while/lstm_cell_24/Relu_1Ù
 lstm_24/while/lstm_cell_24/mul_2Mul(lstm_24/while/lstm_cell_24/Sigmoid_2:y:0/lstm_24/while/lstm_cell_24/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2"
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
lstm_24/while/Identity_3®
lstm_24/while/Identity_4Identity$lstm_24/while/lstm_cell_24/mul_2:z:0^lstm_24/while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2
lstm_24/while/Identity_4®
lstm_24/while/Identity_5Identity$lstm_24/while/lstm_cell_24/add_1:z:0^lstm_24/while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô2
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
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿô:ÿÿÿÿÿÿÿÿÿô: : : : : 2f
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
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô:

_output_shapes
: :

_output_shapes
: 
õ

)sequential_12_lstm_25_while_cond_40075730H
Dsequential_12_lstm_25_while_sequential_12_lstm_25_while_loop_counterN
Jsequential_12_lstm_25_while_sequential_12_lstm_25_while_maximum_iterations+
'sequential_12_lstm_25_while_placeholder-
)sequential_12_lstm_25_while_placeholder_1-
)sequential_12_lstm_25_while_placeholder_2-
)sequential_12_lstm_25_while_placeholder_3J
Fsequential_12_lstm_25_while_less_sequential_12_lstm_25_strided_slice_1b
^sequential_12_lstm_25_while_sequential_12_lstm_25_while_cond_40075730___redundant_placeholder0b
^sequential_12_lstm_25_while_sequential_12_lstm_25_while_cond_40075730___redundant_placeholder1b
^sequential_12_lstm_25_while_sequential_12_lstm_25_while_cond_40075730___redundant_placeholder2b
^sequential_12_lstm_25_while_sequential_12_lstm_25_while_cond_40075730___redundant_placeholder3(
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
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿ8:ÿÿÿÿÿÿÿÿÿ8: ::::: 

_output_shapes
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
:ÿÿÿÿÿÿÿÿÿ8:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ8:

_output_shapes
: :

_output_shapes
:
ÓJ
Ô

lstm_25_while_body_40078615,
(lstm_25_while_lstm_25_while_loop_counter2
.lstm_25_while_lstm_25_while_maximum_iterations
lstm_25_while_placeholder
lstm_25_while_placeholder_1
lstm_25_while_placeholder_2
lstm_25_while_placeholder_3+
'lstm_25_while_lstm_25_strided_slice_1_0g
clstm_25_while_tensorarrayv2read_tensorlistgetitem_lstm_25_tensorarrayunstack_tensorlistfromtensor_0O
;lstm_25_while_lstm_cell_25_matmul_readvariableop_resource_0:
ôàP
=lstm_25_while_lstm_cell_25_matmul_1_readvariableop_resource_0:	8àK
<lstm_25_while_lstm_cell_25_biasadd_readvariableop_resource_0:	à
lstm_25_while_identity
lstm_25_while_identity_1
lstm_25_while_identity_2
lstm_25_while_identity_3
lstm_25_while_identity_4
lstm_25_while_identity_5)
%lstm_25_while_lstm_25_strided_slice_1e
alstm_25_while_tensorarrayv2read_tensorlistgetitem_lstm_25_tensorarrayunstack_tensorlistfromtensorM
9lstm_25_while_lstm_cell_25_matmul_readvariableop_resource:
ôàN
;lstm_25_while_lstm_cell_25_matmul_1_readvariableop_resource:	8àI
:lstm_25_while_lstm_cell_25_biasadd_readvariableop_resource:	à¢1lstm_25/while/lstm_cell_25/BiasAdd/ReadVariableOp¢0lstm_25/while/lstm_cell_25/MatMul/ReadVariableOp¢2lstm_25/while/lstm_cell_25/MatMul_1/ReadVariableOpÓ
?lstm_25/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿô  2A
?lstm_25/while/TensorArrayV2Read/TensorListGetItem/element_shape
1lstm_25/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemclstm_25_while_tensorarrayv2read_tensorlistgetitem_lstm_25_tensorarrayunstack_tensorlistfromtensor_0lstm_25_while_placeholderHlstm_25/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿô*
element_dtype023
1lstm_25/while/TensorArrayV2Read/TensorListGetItemâ
0lstm_25/while/lstm_cell_25/MatMul/ReadVariableOpReadVariableOp;lstm_25_while_lstm_cell_25_matmul_readvariableop_resource_0* 
_output_shapes
:
ôà*
dtype022
0lstm_25/while/lstm_cell_25/MatMul/ReadVariableOp÷
!lstm_25/while/lstm_cell_25/MatMulMatMul8lstm_25/while/TensorArrayV2Read/TensorListGetItem:item:08lstm_25/while/lstm_cell_25/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà2#
!lstm_25/while/lstm_cell_25/MatMulç
2lstm_25/while/lstm_cell_25/MatMul_1/ReadVariableOpReadVariableOp=lstm_25_while_lstm_cell_25_matmul_1_readvariableop_resource_0*
_output_shapes
:	8à*
dtype024
2lstm_25/while/lstm_cell_25/MatMul_1/ReadVariableOpà
#lstm_25/while/lstm_cell_25/MatMul_1MatMullstm_25_while_placeholder_2:lstm_25/while/lstm_cell_25/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà2%
#lstm_25/while/lstm_cell_25/MatMul_1Ø
lstm_25/while/lstm_cell_25/addAddV2+lstm_25/while/lstm_cell_25/MatMul:product:0-lstm_25/while/lstm_cell_25/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà2 
lstm_25/while/lstm_cell_25/addà
1lstm_25/while/lstm_cell_25/BiasAdd/ReadVariableOpReadVariableOp<lstm_25_while_lstm_cell_25_biasadd_readvariableop_resource_0*
_output_shapes	
:à*
dtype023
1lstm_25/while/lstm_cell_25/BiasAdd/ReadVariableOpå
"lstm_25/while/lstm_cell_25/BiasAddBiasAdd"lstm_25/while/lstm_cell_25/add:z:09lstm_25/while/lstm_cell_25/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿà2$
"lstm_25/while/lstm_cell_25/BiasAdd
*lstm_25/while/lstm_cell_25/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2,
*lstm_25/while/lstm_cell_25/split/split_dim«
 lstm_25/while/lstm_cell_25/splitSplit3lstm_25/while/lstm_cell_25/split/split_dim:output:0+lstm_25/while/lstm_cell_25/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ8:ÿÿÿÿÿÿÿÿÿ8:ÿÿÿÿÿÿÿÿÿ8:ÿÿÿÿÿÿÿÿÿ8*
	num_split2"
 lstm_25/while/lstm_cell_25/split°
"lstm_25/while/lstm_cell_25/SigmoidSigmoid)lstm_25/while/lstm_cell_25/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82$
"lstm_25/while/lstm_cell_25/Sigmoid´
$lstm_25/while/lstm_cell_25/Sigmoid_1Sigmoid)lstm_25/while/lstm_cell_25/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82&
$lstm_25/while/lstm_cell_25/Sigmoid_1À
lstm_25/while/lstm_cell_25/mulMul(lstm_25/while/lstm_cell_25/Sigmoid_1:y:0lstm_25_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82 
lstm_25/while/lstm_cell_25/mul§
lstm_25/while/lstm_cell_25/ReluRelu)lstm_25/while/lstm_cell_25/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82!
lstm_25/while/lstm_cell_25/ReluÔ
 lstm_25/while/lstm_cell_25/mul_1Mul&lstm_25/while/lstm_cell_25/Sigmoid:y:0-lstm_25/while/lstm_cell_25/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82"
 lstm_25/while/lstm_cell_25/mul_1É
 lstm_25/while/lstm_cell_25/add_1AddV2"lstm_25/while/lstm_cell_25/mul:z:0$lstm_25/while/lstm_cell_25/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82"
 lstm_25/while/lstm_cell_25/add_1´
$lstm_25/while/lstm_cell_25/Sigmoid_2Sigmoid)lstm_25/while/lstm_cell_25/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82&
$lstm_25/while/lstm_cell_25/Sigmoid_2¦
!lstm_25/while/lstm_cell_25/Relu_1Relu$lstm_25/while/lstm_cell_25/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82#
!lstm_25/while/lstm_cell_25/Relu_1Ø
 lstm_25/while/lstm_cell_25/mul_2Mul(lstm_25/while/lstm_cell_25/Sigmoid_2:y:0/lstm_25/while/lstm_cell_25/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82"
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
lstm_25/while/Identity_3­
lstm_25/while/Identity_4Identity$lstm_25/while/lstm_cell_25/mul_2:z:0^lstm_25/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82
lstm_25/while/Identity_4­
lstm_25/while/Identity_5Identity$lstm_25/while/lstm_cell_25/add_1:z:0^lstm_25/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ82
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
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ8:ÿÿÿÿÿÿÿÿÿ8: : : : : 2f
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
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ8:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ8:

_output_shapes
: :

_output_shapes
: 
ß
Í
while_cond_40079517
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_40079517___redundant_placeholder06
2while_while_cond_40079517___redundant_placeholder16
2while_while_cond_40079517___redundant_placeholder26
2while_while_cond_40079517___redundant_placeholder3
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
@: : : : :ÿÿÿÿÿÿÿÿÿ8:ÿÿÿÿÿÿÿÿÿ8: ::::: 

_output_shapes
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
:ÿÿÿÿÿÿÿÿÿ8:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ8:
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
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:ù»
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
regularization_losses
		variables
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
regularization_losses
	variables
	keras_api
+&call_and_return_all_conditional_losses
__call__"
_tf_keras_rnn_layer
§
trainable_variables
regularization_losses
	variables
	keras_api
+&call_and_return_all_conditional_losses
__call__"
_tf_keras_layer
Å
cell

state_spec
trainable_variables
regularization_losses
	variables
	keras_api
+&call_and_return_all_conditional_losses
__call__"
_tf_keras_rnn_layer
§
trainable_variables
regularization_losses
	variables
	keras_api
+&call_and_return_all_conditional_losses
__call__"
_tf_keras_layer
½

 kernel
!bias
"trainable_variables
#regularization_losses
$	variables
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
 "
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
Î
1layer_metrics
trainable_variables

2layers
3layer_regularization_losses
4metrics
regularization_losses
5non_trainable_variables
		variables
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
8regularization_losses
9	variables
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
 "
trackable_list_wrapper
5
+0
,1
-2"
trackable_list_wrapper
¼
;layer_metrics

<states
trainable_variables

=layers
>layer_regularization_losses
?metrics
regularization_losses
@non_trainable_variables
	variables
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
Alayer_metrics
trainable_variables

Blayers
Clayer_regularization_losses
Dmetrics
regularization_losses
Enon_trainable_variables
	variables
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
Hregularization_losses
I	variables
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
 "
trackable_list_wrapper
5
.0
/1
02"
trackable_list_wrapper
¼
Klayer_metrics

Lstates
trainable_variables

Mlayers
Nlayer_regularization_losses
Ometrics
regularization_losses
Pnon_trainable_variables
	variables
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
Qlayer_metrics
trainable_variables

Rlayers
Slayer_regularization_losses
Tmetrics
regularization_losses
Unon_trainable_variables
	variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
!:82dense_12/kernel
:2dense_12/bias
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
°
Vlayer_metrics
"trainable_variables

Wlayers
Xlayer_regularization_losses
Ymetrics
#regularization_losses
Znon_trainable_variables
$	variables
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
.:,	]Ð2lstm_24/lstm_cell_24/kernel
9:7
ôÐ2%lstm_24/lstm_cell_24/recurrent_kernel
(:&Ð2lstm_24/lstm_cell_24/bias
/:-
ôà2lstm_25/lstm_cell_25/kernel
8:6	8à2%lstm_25/lstm_cell_25/recurrent_kernel
(:&à2lstm_25/lstm_cell_25/bias
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
[0
\1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
5
+0
,1
-2"
trackable_list_wrapper
 "
trackable_list_wrapper
5
+0
,1
-2"
trackable_list_wrapper
°
]layer_metrics
7trainable_variables

^layers
_layer_regularization_losses
`metrics
8regularization_losses
anon_trainable_variables
9	variables
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
 "
trackable_list_wrapper
5
.0
/1
02"
trackable_list_wrapper
°
blayer_metrics
Gtrainable_variables

clayers
dlayer_regularization_losses
emetrics
Hregularization_losses
fnon_trainable_variables
I	variables
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
&:$82Adam/dense_12/kernel/m
 :2Adam/dense_12/bias/m
3:1	]Ð2"Adam/lstm_24/lstm_cell_24/kernel/m
>:<
ôÐ2,Adam/lstm_24/lstm_cell_24/recurrent_kernel/m
-:+Ð2 Adam/lstm_24/lstm_cell_24/bias/m
4:2
ôà2"Adam/lstm_25/lstm_cell_25/kernel/m
=:;	8à2,Adam/lstm_25/lstm_cell_25/recurrent_kernel/m
-:+à2 Adam/lstm_25/lstm_cell_25/bias/m
&:$82Adam/dense_12/kernel/v
 :2Adam/dense_12/bias/v
3:1	]Ð2"Adam/lstm_24/lstm_cell_24/kernel/v
>:<
ôÐ2,Adam/lstm_24/lstm_cell_24/recurrent_kernel/v
-:+Ð2 Adam/lstm_24/lstm_cell_24/bias/v
4:2
ôà2"Adam/lstm_25/lstm_cell_25/kernel/v
=:;	8à2,Adam/lstm_25/lstm_cell_25/recurrent_kernel/v
-:+à2 Adam/lstm_25/lstm_cell_25/bias/v
ÔBÑ
#__inference__wrapped_model_40075843lstm_24_input"
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
K__inference_sequential_12_layer_call_and_return_conditional_losses_40078393
K__inference_sequential_12_layer_call_and_return_conditional_losses_40078734
K__inference_sequential_12_layer_call_and_return_conditional_losses_40078012
K__inference_sequential_12_layer_call_and_return_conditional_losses_40078037À
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
0__inference_sequential_12_layer_call_fn_40077497
0__inference_sequential_12_layer_call_fn_40078755
0__inference_sequential_12_layer_call_fn_40078776
0__inference_sequential_12_layer_call_fn_40077987À
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
E__inference_lstm_24_layer_call_and_return_conditional_losses_40078927
E__inference_lstm_24_layer_call_and_return_conditional_losses_40079078
E__inference_lstm_24_layer_call_and_return_conditional_losses_40079229
E__inference_lstm_24_layer_call_and_return_conditional_losses_40079380Õ
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
*__inference_lstm_24_layer_call_fn_40079391
*__inference_lstm_24_layer_call_fn_40079402
*__inference_lstm_24_layer_call_fn_40079413
*__inference_lstm_24_layer_call_fn_40079424Õ
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
H__inference_dropout_24_layer_call_and_return_conditional_losses_40079429
H__inference_dropout_24_layer_call_and_return_conditional_losses_40079441´
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
-__inference_dropout_24_layer_call_fn_40079446
-__inference_dropout_24_layer_call_fn_40079451´
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
E__inference_lstm_25_layer_call_and_return_conditional_losses_40079602
E__inference_lstm_25_layer_call_and_return_conditional_losses_40079753
E__inference_lstm_25_layer_call_and_return_conditional_losses_40079904
E__inference_lstm_25_layer_call_and_return_conditional_losses_40080055Õ
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
*__inference_lstm_25_layer_call_fn_40080066
*__inference_lstm_25_layer_call_fn_40080077
*__inference_lstm_25_layer_call_fn_40080088
*__inference_lstm_25_layer_call_fn_40080099Õ
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
H__inference_dropout_25_layer_call_and_return_conditional_losses_40080104
H__inference_dropout_25_layer_call_and_return_conditional_losses_40080116´
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
-__inference_dropout_25_layer_call_fn_40080121
-__inference_dropout_25_layer_call_fn_40080126´
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
F__inference_dense_12_layer_call_and_return_conditional_losses_40080157¢
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
+__inference_dense_12_layer_call_fn_40080166¢
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
&__inference_signature_wrapper_40078066lstm_24_input"
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
J__inference_lstm_cell_24_layer_call_and_return_conditional_losses_40080198
J__inference_lstm_cell_24_layer_call_and_return_conditional_losses_40080230¾
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
/__inference_lstm_cell_24_layer_call_fn_40080247
/__inference_lstm_cell_24_layer_call_fn_40080264¾
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
J__inference_lstm_cell_25_layer_call_and_return_conditional_losses_40080296
J__inference_lstm_cell_25_layer_call_and_return_conditional_losses_40080328¾
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
/__inference_lstm_cell_25_layer_call_fn_40080345
/__inference_lstm_cell_25_layer_call_fn_40080362¾
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
#__inference__wrapped_model_40075843+,-./0 !:¢7
0¢-
+(
lstm_24_inputÿÿÿÿÿÿÿÿÿ]
ª "7ª4
2
dense_12&#
dense_12ÿÿÿÿÿÿÿÿÿ®
F__inference_dense_12_layer_call_and_return_conditional_losses_40080157d !3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿ8
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ
 
+__inference_dense_12_layer_call_fn_40080166W !3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿ8
ª "ÿÿÿÿÿÿÿÿÿ²
H__inference_dropout_24_layer_call_and_return_conditional_losses_40079429f8¢5
.¢+
%"
inputsÿÿÿÿÿÿÿÿÿô
p 
ª "*¢'
 
0ÿÿÿÿÿÿÿÿÿô
 ²
H__inference_dropout_24_layer_call_and_return_conditional_losses_40079441f8¢5
.¢+
%"
inputsÿÿÿÿÿÿÿÿÿô
p
ª "*¢'
 
0ÿÿÿÿÿÿÿÿÿô
 
-__inference_dropout_24_layer_call_fn_40079446Y8¢5
.¢+
%"
inputsÿÿÿÿÿÿÿÿÿô
p 
ª "ÿÿÿÿÿÿÿÿÿô
-__inference_dropout_24_layer_call_fn_40079451Y8¢5
.¢+
%"
inputsÿÿÿÿÿÿÿÿÿô
p
ª "ÿÿÿÿÿÿÿÿÿô°
H__inference_dropout_25_layer_call_and_return_conditional_losses_40080104d7¢4
-¢*
$!
inputsÿÿÿÿÿÿÿÿÿ8
p 
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ8
 °
H__inference_dropout_25_layer_call_and_return_conditional_losses_40080116d7¢4
-¢*
$!
inputsÿÿÿÿÿÿÿÿÿ8
p
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ8
 
-__inference_dropout_25_layer_call_fn_40080121W7¢4
-¢*
$!
inputsÿÿÿÿÿÿÿÿÿ8
p 
ª "ÿÿÿÿÿÿÿÿÿ8
-__inference_dropout_25_layer_call_fn_40080126W7¢4
-¢*
$!
inputsÿÿÿÿÿÿÿÿÿ8
p
ª "ÿÿÿÿÿÿÿÿÿ8Õ
E__inference_lstm_24_layer_call_and_return_conditional_losses_40078927+,-O¢L
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
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿô
 Õ
E__inference_lstm_24_layer_call_and_return_conditional_losses_40079078+,-O¢L
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
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿô
 »
E__inference_lstm_24_layer_call_and_return_conditional_losses_40079229r+,-?¢<
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
0ÿÿÿÿÿÿÿÿÿô
 »
E__inference_lstm_24_layer_call_and_return_conditional_losses_40079380r+,-?¢<
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
0ÿÿÿÿÿÿÿÿÿô
 ¬
*__inference_lstm_24_layer_call_fn_40079391~+,-O¢L
E¢B
41
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ]

 
p 

 
ª "&#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿô¬
*__inference_lstm_24_layer_call_fn_40079402~+,-O¢L
E¢B
41
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ]

 
p

 
ª "&#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿô
*__inference_lstm_24_layer_call_fn_40079413e+,-?¢<
5¢2
$!
inputsÿÿÿÿÿÿÿÿÿ]

 
p 

 
ª "ÿÿÿÿÿÿÿÿÿô
*__inference_lstm_24_layer_call_fn_40079424e+,-?¢<
5¢2
$!
inputsÿÿÿÿÿÿÿÿÿ]

 
p

 
ª "ÿÿÿÿÿÿÿÿÿôÕ
E__inference_lstm_25_layer_call_and_return_conditional_losses_40079602./0P¢M
F¢C
52
0-
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿô

 
p 

 
ª "2¢/
(%
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ8
 Õ
E__inference_lstm_25_layer_call_and_return_conditional_losses_40079753./0P¢M
F¢C
52
0-
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿô

 
p

 
ª "2¢/
(%
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ8
 »
E__inference_lstm_25_layer_call_and_return_conditional_losses_40079904r./0@¢=
6¢3
%"
inputsÿÿÿÿÿÿÿÿÿô

 
p 

 
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ8
 »
E__inference_lstm_25_layer_call_and_return_conditional_losses_40080055r./0@¢=
6¢3
%"
inputsÿÿÿÿÿÿÿÿÿô

 
p

 
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ8
 ¬
*__inference_lstm_25_layer_call_fn_40080066~./0P¢M
F¢C
52
0-
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿô

 
p 

 
ª "%"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ8¬
*__inference_lstm_25_layer_call_fn_40080077~./0P¢M
F¢C
52
0-
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿô

 
p

 
ª "%"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ8
*__inference_lstm_25_layer_call_fn_40080088e./0@¢=
6¢3
%"
inputsÿÿÿÿÿÿÿÿÿô

 
p 

 
ª "ÿÿÿÿÿÿÿÿÿ8
*__inference_lstm_25_layer_call_fn_40080099e./0@¢=
6¢3
%"
inputsÿÿÿÿÿÿÿÿÿô

 
p

 
ª "ÿÿÿÿÿÿÿÿÿ8Ñ
J__inference_lstm_cell_24_layer_call_and_return_conditional_losses_40080198+,-¢
x¢u
 
inputsÿÿÿÿÿÿÿÿÿ]
M¢J
# 
states/0ÿÿÿÿÿÿÿÿÿô
# 
states/1ÿÿÿÿÿÿÿÿÿô
p 
ª "v¢s
l¢i

0/0ÿÿÿÿÿÿÿÿÿô
GD
 
0/1/0ÿÿÿÿÿÿÿÿÿô
 
0/1/1ÿÿÿÿÿÿÿÿÿô
 Ñ
J__inference_lstm_cell_24_layer_call_and_return_conditional_losses_40080230+,-¢
x¢u
 
inputsÿÿÿÿÿÿÿÿÿ]
M¢J
# 
states/0ÿÿÿÿÿÿÿÿÿô
# 
states/1ÿÿÿÿÿÿÿÿÿô
p
ª "v¢s
l¢i

0/0ÿÿÿÿÿÿÿÿÿô
GD
 
0/1/0ÿÿÿÿÿÿÿÿÿô
 
0/1/1ÿÿÿÿÿÿÿÿÿô
 ¦
/__inference_lstm_cell_24_layer_call_fn_40080247ò+,-¢
x¢u
 
inputsÿÿÿÿÿÿÿÿÿ]
M¢J
# 
states/0ÿÿÿÿÿÿÿÿÿô
# 
states/1ÿÿÿÿÿÿÿÿÿô
p 
ª "f¢c

0ÿÿÿÿÿÿÿÿÿô
C@

1/0ÿÿÿÿÿÿÿÿÿô

1/1ÿÿÿÿÿÿÿÿÿô¦
/__inference_lstm_cell_24_layer_call_fn_40080264ò+,-¢
x¢u
 
inputsÿÿÿÿÿÿÿÿÿ]
M¢J
# 
states/0ÿÿÿÿÿÿÿÿÿô
# 
states/1ÿÿÿÿÿÿÿÿÿô
p
ª "f¢c

0ÿÿÿÿÿÿÿÿÿô
C@

1/0ÿÿÿÿÿÿÿÿÿô

1/1ÿÿÿÿÿÿÿÿÿôÍ
J__inference_lstm_cell_25_layer_call_and_return_conditional_losses_40080296þ./0¢~
w¢t
!
inputsÿÿÿÿÿÿÿÿÿô
K¢H
"
states/0ÿÿÿÿÿÿÿÿÿ8
"
states/1ÿÿÿÿÿÿÿÿÿ8
p 
ª "s¢p
i¢f

0/0ÿÿÿÿÿÿÿÿÿ8
EB

0/1/0ÿÿÿÿÿÿÿÿÿ8

0/1/1ÿÿÿÿÿÿÿÿÿ8
 Í
J__inference_lstm_cell_25_layer_call_and_return_conditional_losses_40080328þ./0¢~
w¢t
!
inputsÿÿÿÿÿÿÿÿÿô
K¢H
"
states/0ÿÿÿÿÿÿÿÿÿ8
"
states/1ÿÿÿÿÿÿÿÿÿ8
p
ª "s¢p
i¢f

0/0ÿÿÿÿÿÿÿÿÿ8
EB

0/1/0ÿÿÿÿÿÿÿÿÿ8

0/1/1ÿÿÿÿÿÿÿÿÿ8
 ¢
/__inference_lstm_cell_25_layer_call_fn_40080345î./0¢~
w¢t
!
inputsÿÿÿÿÿÿÿÿÿô
K¢H
"
states/0ÿÿÿÿÿÿÿÿÿ8
"
states/1ÿÿÿÿÿÿÿÿÿ8
p 
ª "c¢`

0ÿÿÿÿÿÿÿÿÿ8
A>

1/0ÿÿÿÿÿÿÿÿÿ8

1/1ÿÿÿÿÿÿÿÿÿ8¢
/__inference_lstm_cell_25_layer_call_fn_40080362î./0¢~
w¢t
!
inputsÿÿÿÿÿÿÿÿÿô
K¢H
"
states/0ÿÿÿÿÿÿÿÿÿ8
"
states/1ÿÿÿÿÿÿÿÿÿ8
p
ª "c¢`

0ÿÿÿÿÿÿÿÿÿ8
A>

1/0ÿÿÿÿÿÿÿÿÿ8

1/1ÿÿÿÿÿÿÿÿÿ8È
K__inference_sequential_12_layer_call_and_return_conditional_losses_40078012y+,-./0 !B¢?
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
K__inference_sequential_12_layer_call_and_return_conditional_losses_40078037y+,-./0 !B¢?
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
K__inference_sequential_12_layer_call_and_return_conditional_losses_40078393r+,-./0 !;¢8
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
K__inference_sequential_12_layer_call_and_return_conditional_losses_40078734r+,-./0 !;¢8
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
0__inference_sequential_12_layer_call_fn_40077497l+,-./0 !B¢?
8¢5
+(
lstm_24_inputÿÿÿÿÿÿÿÿÿ]
p 

 
ª "ÿÿÿÿÿÿÿÿÿ 
0__inference_sequential_12_layer_call_fn_40077987l+,-./0 !B¢?
8¢5
+(
lstm_24_inputÿÿÿÿÿÿÿÿÿ]
p

 
ª "ÿÿÿÿÿÿÿÿÿ
0__inference_sequential_12_layer_call_fn_40078755e+,-./0 !;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿ]
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
0__inference_sequential_12_layer_call_fn_40078776e+,-./0 !;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿ]
p

 
ª "ÿÿÿÿÿÿÿÿÿ»
&__inference_signature_wrapper_40078066+,-./0 !K¢H
¢ 
Aª>
<
lstm_24_input+(
lstm_24_inputÿÿÿÿÿÿÿÿÿ]"7ª4
2
dense_12&#
dense_12ÿÿÿÿÿÿÿÿÿ