??'
??
D
AddV2
x"T
y"T
z"T"
Ttype:
2	??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
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
?
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
delete_old_dirsbool(?
?
Mul
x"T
y"T
z"T"
Ttype:
2	?
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
?
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
dtypetype?
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
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
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
?
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
executor_typestring ?
@
StaticRegexFullMatch	
input

output
"
patternstring
?
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
?
TensorListFromTensor
tensor"element_dtype
element_shape"
shape_type*
output_handle??element_dtype"
element_dtypetype"

shape_typetype:
2	
?
TensorListReserve
element_shape"
shape_type
num_elements#
handle??element_dtype"
element_dtypetype"

shape_typetype:
2	
?
TensorListStack
input_handle
element_shape
tensor"element_dtype"
element_dtypetype" 
num_elementsint?????????
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?
?
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
?"serve*2.6.02v2.6.0-rc2-32-g919f693420e8??&
y
dense_8/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*
shared_namedense_8/kernel
r
"dense_8/kernel/Read/ReadVariableOpReadVariableOpdense_8/kernel*
_output_shapes
:	?*
dtype0
p
dense_8/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_8/bias
i
 dense_8/bias/Read/ReadVariableOpReadVariableOpdense_8/bias*
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
?
lstm_16/lstm_cell_16/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	]?*,
shared_namelstm_16/lstm_cell_16/kernel
?
/lstm_16/lstm_cell_16/kernel/Read/ReadVariableOpReadVariableOplstm_16/lstm_cell_16/kernel*
_output_shapes
:	]?*
dtype0
?
%lstm_16/lstm_cell_16/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*6
shared_name'%lstm_16/lstm_cell_16/recurrent_kernel
?
9lstm_16/lstm_cell_16/recurrent_kernel/Read/ReadVariableOpReadVariableOp%lstm_16/lstm_cell_16/recurrent_kernel* 
_output_shapes
:
??*
dtype0
?
lstm_16/lstm_cell_16/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?**
shared_namelstm_16/lstm_cell_16/bias
?
-lstm_16/lstm_cell_16/bias/Read/ReadVariableOpReadVariableOplstm_16/lstm_cell_16/bias*
_output_shapes	
:?*
dtype0
?
lstm_17/lstm_cell_17/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*,
shared_namelstm_17/lstm_cell_17/kernel
?
/lstm_17/lstm_cell_17/kernel/Read/ReadVariableOpReadVariableOplstm_17/lstm_cell_17/kernel* 
_output_shapes
:
??*
dtype0
?
%lstm_17/lstm_cell_17/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*6
shared_name'%lstm_17/lstm_cell_17/recurrent_kernel
?
9lstm_17/lstm_cell_17/recurrent_kernel/Read/ReadVariableOpReadVariableOp%lstm_17/lstm_cell_17/recurrent_kernel* 
_output_shapes
:
??*
dtype0
?
lstm_17/lstm_cell_17/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?**
shared_namelstm_17/lstm_cell_17/bias
?
-lstm_17/lstm_cell_17/bias/Read/ReadVariableOpReadVariableOplstm_17/lstm_cell_17/bias*
_output_shapes	
:?*
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
?
Adam/dense_8/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*&
shared_nameAdam/dense_8/kernel/m
?
)Adam/dense_8/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_8/kernel/m*
_output_shapes
:	?*
dtype0
~
Adam/dense_8/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_8/bias/m
w
'Adam/dense_8/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_8/bias/m*
_output_shapes
:*
dtype0
?
"Adam/lstm_16/lstm_cell_16/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	]?*3
shared_name$"Adam/lstm_16/lstm_cell_16/kernel/m
?
6Adam/lstm_16/lstm_cell_16/kernel/m/Read/ReadVariableOpReadVariableOp"Adam/lstm_16/lstm_cell_16/kernel/m*
_output_shapes
:	]?*
dtype0
?
,Adam/lstm_16/lstm_cell_16/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*=
shared_name.,Adam/lstm_16/lstm_cell_16/recurrent_kernel/m
?
@Adam/lstm_16/lstm_cell_16/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp,Adam/lstm_16/lstm_cell_16/recurrent_kernel/m* 
_output_shapes
:
??*
dtype0
?
 Adam/lstm_16/lstm_cell_16/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*1
shared_name" Adam/lstm_16/lstm_cell_16/bias/m
?
4Adam/lstm_16/lstm_cell_16/bias/m/Read/ReadVariableOpReadVariableOp Adam/lstm_16/lstm_cell_16/bias/m*
_output_shapes	
:?*
dtype0
?
"Adam/lstm_17/lstm_cell_17/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*3
shared_name$"Adam/lstm_17/lstm_cell_17/kernel/m
?
6Adam/lstm_17/lstm_cell_17/kernel/m/Read/ReadVariableOpReadVariableOp"Adam/lstm_17/lstm_cell_17/kernel/m* 
_output_shapes
:
??*
dtype0
?
,Adam/lstm_17/lstm_cell_17/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*=
shared_name.,Adam/lstm_17/lstm_cell_17/recurrent_kernel/m
?
@Adam/lstm_17/lstm_cell_17/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp,Adam/lstm_17/lstm_cell_17/recurrent_kernel/m* 
_output_shapes
:
??*
dtype0
?
 Adam/lstm_17/lstm_cell_17/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*1
shared_name" Adam/lstm_17/lstm_cell_17/bias/m
?
4Adam/lstm_17/lstm_cell_17/bias/m/Read/ReadVariableOpReadVariableOp Adam/lstm_17/lstm_cell_17/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/dense_8/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*&
shared_nameAdam/dense_8/kernel/v
?
)Adam/dense_8/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_8/kernel/v*
_output_shapes
:	?*
dtype0
~
Adam/dense_8/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_8/bias/v
w
'Adam/dense_8/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_8/bias/v*
_output_shapes
:*
dtype0
?
"Adam/lstm_16/lstm_cell_16/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	]?*3
shared_name$"Adam/lstm_16/lstm_cell_16/kernel/v
?
6Adam/lstm_16/lstm_cell_16/kernel/v/Read/ReadVariableOpReadVariableOp"Adam/lstm_16/lstm_cell_16/kernel/v*
_output_shapes
:	]?*
dtype0
?
,Adam/lstm_16/lstm_cell_16/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*=
shared_name.,Adam/lstm_16/lstm_cell_16/recurrent_kernel/v
?
@Adam/lstm_16/lstm_cell_16/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp,Adam/lstm_16/lstm_cell_16/recurrent_kernel/v* 
_output_shapes
:
??*
dtype0
?
 Adam/lstm_16/lstm_cell_16/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*1
shared_name" Adam/lstm_16/lstm_cell_16/bias/v
?
4Adam/lstm_16/lstm_cell_16/bias/v/Read/ReadVariableOpReadVariableOp Adam/lstm_16/lstm_cell_16/bias/v*
_output_shapes	
:?*
dtype0
?
"Adam/lstm_17/lstm_cell_17/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*3
shared_name$"Adam/lstm_17/lstm_cell_17/kernel/v
?
6Adam/lstm_17/lstm_cell_17/kernel/v/Read/ReadVariableOpReadVariableOp"Adam/lstm_17/lstm_cell_17/kernel/v* 
_output_shapes
:
??*
dtype0
?
,Adam/lstm_17/lstm_cell_17/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*=
shared_name.,Adam/lstm_17/lstm_cell_17/recurrent_kernel/v
?
@Adam/lstm_17/lstm_cell_17/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp,Adam/lstm_17/lstm_cell_17/recurrent_kernel/v* 
_output_shapes
:
??*
dtype0
?
 Adam/lstm_17/lstm_cell_17/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*1
shared_name" Adam/lstm_17/lstm_cell_17/bias/v
?
4Adam/lstm_17/lstm_cell_17/bias/v/Read/ReadVariableOpReadVariableOp Adam/lstm_17/lstm_cell_17/bias/v*
_output_shapes	
:?*
dtype0

NoOpNoOp
?7
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?7
value?7B?7 B?7
?
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
?
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
?
1metrics
trainable_variables
regularization_losses
2layer_regularization_losses
3non_trainable_variables

4layers
		variables
5layer_metrics
 
?
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
?
;metrics
trainable_variables

<states
=layer_regularization_losses
>non_trainable_variables
regularization_losses

?layers
	variables
@layer_metrics
 
 
 
?
Ametrics
trainable_variables
Blayer_regularization_losses
Cnon_trainable_variables
regularization_losses

Dlayers
	variables
Elayer_metrics
?
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
?
Kmetrics
trainable_variables

Lstates
Mlayer_regularization_losses
Nnon_trainable_variables
regularization_losses

Olayers
	variables
Player_metrics
 
 
 
?
Qmetrics
trainable_variables
Rlayer_regularization_losses
Snon_trainable_variables
regularization_losses

Tlayers
	variables
Ulayer_metrics
ZX
VARIABLE_VALUEdense_8/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_8/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

 0
!1
 

 0
!1
?
Vmetrics
"trainable_variables
Wlayer_regularization_losses
Xnon_trainable_variables
#regularization_losses

Ylayers
$	variables
Zlayer_metrics
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
VARIABLE_VALUElstm_16/lstm_cell_16/kernel0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUE
ki
VARIABLE_VALUE%lstm_16/lstm_cell_16/recurrent_kernel0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUElstm_16/lstm_cell_16/bias0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUElstm_17/lstm_cell_17/kernel0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUE
ki
VARIABLE_VALUE%lstm_17/lstm_cell_17/recurrent_kernel0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUElstm_17/lstm_cell_17/bias0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUE

[0
\1
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
+0
,1
-2
 

+0
,1
-2
?
]metrics
7trainable_variables
^layer_regularization_losses
_non_trainable_variables
8regularization_losses

`layers
9	variables
alayer_metrics
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
.0
/1
02
 

.0
/1
02
?
bmetrics
Gtrainable_variables
clayer_regularization_losses
dnon_trainable_variables
Hregularization_losses

elayers
I	variables
flayer_metrics
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
}{
VARIABLE_VALUEAdam/dense_8/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_8/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"Adam/lstm_16/lstm_cell_16/kernel/mLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE,Adam/lstm_16/lstm_cell_16/recurrent_kernel/mLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE Adam/lstm_16/lstm_cell_16/bias/mLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"Adam/lstm_17/lstm_cell_17/kernel/mLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE,Adam/lstm_17/lstm_cell_17/recurrent_kernel/mLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE Adam/lstm_17/lstm_cell_17/bias/mLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_8/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_8/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"Adam/lstm_16/lstm_cell_16/kernel/vLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE,Adam/lstm_16/lstm_cell_16/recurrent_kernel/vLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE Adam/lstm_16/lstm_cell_16/bias/vLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"Adam/lstm_17/lstm_cell_17/kernel/vLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE,Adam/lstm_17/lstm_cell_17/recurrent_kernel/vLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE Adam/lstm_17/lstm_cell_17/bias/vLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
serving_default_lstm_16_inputPlaceholder*+
_output_shapes
:?????????]*
dtype0* 
shape:?????????]
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_lstm_16_inputlstm_16/lstm_cell_16/kernel%lstm_16/lstm_cell_16/recurrent_kernellstm_16/lstm_cell_16/biaslstm_17/lstm_cell_17/kernel%lstm_17/lstm_cell_17/recurrent_kernellstm_17/lstm_cell_17/biasdense_8/kerneldense_8/bias*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? */
f*R(
&__inference_signature_wrapper_32575686
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename"dense_8/kernel/Read/ReadVariableOp dense_8/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp/lstm_16/lstm_cell_16/kernel/Read/ReadVariableOp9lstm_16/lstm_cell_16/recurrent_kernel/Read/ReadVariableOp-lstm_16/lstm_cell_16/bias/Read/ReadVariableOp/lstm_17/lstm_cell_17/kernel/Read/ReadVariableOp9lstm_17/lstm_cell_17/recurrent_kernel/Read/ReadVariableOp-lstm_17/lstm_cell_17/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp)Adam/dense_8/kernel/m/Read/ReadVariableOp'Adam/dense_8/bias/m/Read/ReadVariableOp6Adam/lstm_16/lstm_cell_16/kernel/m/Read/ReadVariableOp@Adam/lstm_16/lstm_cell_16/recurrent_kernel/m/Read/ReadVariableOp4Adam/lstm_16/lstm_cell_16/bias/m/Read/ReadVariableOp6Adam/lstm_17/lstm_cell_17/kernel/m/Read/ReadVariableOp@Adam/lstm_17/lstm_cell_17/recurrent_kernel/m/Read/ReadVariableOp4Adam/lstm_17/lstm_cell_17/bias/m/Read/ReadVariableOp)Adam/dense_8/kernel/v/Read/ReadVariableOp'Adam/dense_8/bias/v/Read/ReadVariableOp6Adam/lstm_16/lstm_cell_16/kernel/v/Read/ReadVariableOp@Adam/lstm_16/lstm_cell_16/recurrent_kernel/v/Read/ReadVariableOp4Adam/lstm_16/lstm_cell_16/bias/v/Read/ReadVariableOp6Adam/lstm_17/lstm_cell_17/kernel/v/Read/ReadVariableOp@Adam/lstm_17/lstm_cell_17/recurrent_kernel/v/Read/ReadVariableOp4Adam/lstm_17/lstm_cell_17/bias/v/Read/ReadVariableOpConst*.
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
GPU 2J 8? **
f%R#
!__inference__traced_save_32578104
?	
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_8/kerneldense_8/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratelstm_16/lstm_cell_16/kernel%lstm_16/lstm_cell_16/recurrent_kernellstm_16/lstm_cell_16/biaslstm_17/lstm_cell_17/kernel%lstm_17/lstm_cell_17/recurrent_kernellstm_17/lstm_cell_17/biastotalcounttotal_1count_1Adam/dense_8/kernel/mAdam/dense_8/bias/m"Adam/lstm_16/lstm_cell_16/kernel/m,Adam/lstm_16/lstm_cell_16/recurrent_kernel/m Adam/lstm_16/lstm_cell_16/bias/m"Adam/lstm_17/lstm_cell_17/kernel/m,Adam/lstm_17/lstm_cell_17/recurrent_kernel/m Adam/lstm_17/lstm_cell_17/bias/mAdam/dense_8/kernel/vAdam/dense_8/bias/v"Adam/lstm_16/lstm_cell_16/kernel/v,Adam/lstm_16/lstm_cell_16/recurrent_kernel/v Adam/lstm_16/lstm_cell_16/bias/v"Adam/lstm_17/lstm_cell_17/kernel/v,Adam/lstm_17/lstm_cell_17/recurrent_kernel/v Adam/lstm_17/lstm_cell_17/bias/v*-
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
GPU 2J 8? *-
f(R&
$__inference__traced_restore_32578213??$
??
?
while_body_32575426
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_16_matmul_readvariableop_resource_0:	]?I
5while_lstm_cell_16_matmul_1_readvariableop_resource_0:
??C
4while_lstm_cell_16_biasadd_readvariableop_resource_0:	?
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_16_matmul_readvariableop_resource:	]?G
3while_lstm_cell_16_matmul_1_readvariableop_resource:
??A
2while_lstm_cell_16_biasadd_readvariableop_resource:	???)while/lstm_cell_16/BiasAdd/ReadVariableOp?(while/lstm_cell_16/MatMul/ReadVariableOp?*while/lstm_cell_16/MatMul_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????]   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????]*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
(while/lstm_cell_16/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_16_matmul_readvariableop_resource_0*
_output_shapes
:	]?*
dtype02*
(while/lstm_cell_16/MatMul/ReadVariableOp?
while/lstm_cell_16/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_16/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_16/MatMul?
*while/lstm_cell_16/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_16_matmul_1_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02,
*while/lstm_cell_16/MatMul_1/ReadVariableOp?
while/lstm_cell_16/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_16/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_16/MatMul_1?
while/lstm_cell_16/addAddV2#while/lstm_cell_16/MatMul:product:0%while/lstm_cell_16/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_16/add?
)while/lstm_cell_16/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_16_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype02+
)while/lstm_cell_16/BiasAdd/ReadVariableOp?
while/lstm_cell_16/BiasAddBiasAddwhile/lstm_cell_16/add:z:01while/lstm_cell_16/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_16/BiasAdd?
"while/lstm_cell_16/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_16/split/split_dim?
while/lstm_cell_16/splitSplit+while/lstm_cell_16/split/split_dim:output:0#while/lstm_cell_16/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split2
while/lstm_cell_16/split?
while/lstm_cell_16/SigmoidSigmoid!while/lstm_cell_16/split:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_16/Sigmoid?
while/lstm_cell_16/Sigmoid_1Sigmoid!while/lstm_cell_16/split:output:1*
T0*(
_output_shapes
:??????????2
while/lstm_cell_16/Sigmoid_1?
while/lstm_cell_16/mulMul while/lstm_cell_16/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:??????????2
while/lstm_cell_16/mul?
while/lstm_cell_16/ReluRelu!while/lstm_cell_16/split:output:2*
T0*(
_output_shapes
:??????????2
while/lstm_cell_16/Relu?
while/lstm_cell_16/mul_1Mulwhile/lstm_cell_16/Sigmoid:y:0%while/lstm_cell_16/Relu:activations:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_16/mul_1?
while/lstm_cell_16/add_1AddV2while/lstm_cell_16/mul:z:0while/lstm_cell_16/mul_1:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_16/add_1?
while/lstm_cell_16/Sigmoid_2Sigmoid!while/lstm_cell_16/split:output:3*
T0*(
_output_shapes
:??????????2
while/lstm_cell_16/Sigmoid_2?
while/lstm_cell_16/Relu_1Reluwhile/lstm_cell_16/add_1:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_16/Relu_1?
while/lstm_cell_16/mul_2Mul while/lstm_cell_16/Sigmoid_2:y:0'while/lstm_cell_16/Relu_1:activations:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_16/mul_2?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_16/mul_2:z:0*
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
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identitywhile/lstm_cell_16/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:??????????2
while/Identity_4?
while/Identity_5Identitywhile/lstm_cell_16/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:??????????2
while/Identity_5?

while/NoOpNoOp*^while/lstm_cell_16/BiasAdd/ReadVariableOp)^while/lstm_cell_16/MatMul/ReadVariableOp+^while/lstm_cell_16/MatMul_1/ReadVariableOp*"
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
2while_lstm_cell_16_biasadd_readvariableop_resource4while_lstm_cell_16_biasadd_readvariableop_resource_0"l
3while_lstm_cell_16_matmul_1_readvariableop_resource5while_lstm_cell_16_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_16_matmul_readvariableop_resource3while_lstm_cell_16_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :??????????:??????????: : : : : 2V
)while/lstm_cell_16/BiasAdd/ReadVariableOp)while/lstm_cell_16/BiasAdd/ReadVariableOp2T
(while/lstm_cell_16/MatMul/ReadVariableOp(while/lstm_cell_16/MatMul/ReadVariableOp2X
*while/lstm_cell_16/MatMul_1/ReadVariableOp*while/lstm_cell_16/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?F
?
E__inference_lstm_17_layer_call_and_return_conditional_losses_32574251

inputs)
lstm_cell_17_32574169:
??)
lstm_cell_17_32574171:
??$
lstm_cell_17_32574173:	?
identity??$lstm_cell_17/StatefulPartitionedCall?whileD
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
strided_slice/stack_2?
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
B :?2
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
B :?2
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
B :?2
zeros/packed/1?
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
:??????????2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
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
B :?2
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
B :?2
zeros_1/packed/1?
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
:??????????2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm?
	transpose	Transposeinputstranspose/perm:output:0*
T0*5
_output_shapes#
!:???????????????????2
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
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????  27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
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
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_2?
$lstm_cell_17/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_17_32574169lstm_cell_17_32574171lstm_cell_17_32574173*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:??????????:??????????:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_lstm_cell_17_layer_call_and_return_conditional_losses_325741682&
$lstm_cell_17/StatefulPartitionedCall?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2
TensorArrayV2_1/element_shape?
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
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_17_32574169lstm_cell_17_32574171lstm_cell_17_32574173*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_32574182*
condR
while_cond_32574181*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:???????????????????*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
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
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:???????????????????2
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
!:???????????????????2

Identity}
NoOpNoOp%^lstm_cell_17/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':???????????????????: : : 2L
$lstm_cell_17/StatefulPartitionedCall$lstm_cell_17/StatefulPartitionedCall2
whilewhile:] Y
5
_output_shapes#
!:???????????????????
 
_user_specified_nameinputs
??
?
while_body_32576765
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_16_matmul_readvariableop_resource_0:	]?I
5while_lstm_cell_16_matmul_1_readvariableop_resource_0:
??C
4while_lstm_cell_16_biasadd_readvariableop_resource_0:	?
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_16_matmul_readvariableop_resource:	]?G
3while_lstm_cell_16_matmul_1_readvariableop_resource:
??A
2while_lstm_cell_16_biasadd_readvariableop_resource:	???)while/lstm_cell_16/BiasAdd/ReadVariableOp?(while/lstm_cell_16/MatMul/ReadVariableOp?*while/lstm_cell_16/MatMul_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????]   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????]*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
(while/lstm_cell_16/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_16_matmul_readvariableop_resource_0*
_output_shapes
:	]?*
dtype02*
(while/lstm_cell_16/MatMul/ReadVariableOp?
while/lstm_cell_16/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_16/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_16/MatMul?
*while/lstm_cell_16/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_16_matmul_1_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02,
*while/lstm_cell_16/MatMul_1/ReadVariableOp?
while/lstm_cell_16/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_16/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_16/MatMul_1?
while/lstm_cell_16/addAddV2#while/lstm_cell_16/MatMul:product:0%while/lstm_cell_16/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_16/add?
)while/lstm_cell_16/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_16_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype02+
)while/lstm_cell_16/BiasAdd/ReadVariableOp?
while/lstm_cell_16/BiasAddBiasAddwhile/lstm_cell_16/add:z:01while/lstm_cell_16/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_16/BiasAdd?
"while/lstm_cell_16/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_16/split/split_dim?
while/lstm_cell_16/splitSplit+while/lstm_cell_16/split/split_dim:output:0#while/lstm_cell_16/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split2
while/lstm_cell_16/split?
while/lstm_cell_16/SigmoidSigmoid!while/lstm_cell_16/split:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_16/Sigmoid?
while/lstm_cell_16/Sigmoid_1Sigmoid!while/lstm_cell_16/split:output:1*
T0*(
_output_shapes
:??????????2
while/lstm_cell_16/Sigmoid_1?
while/lstm_cell_16/mulMul while/lstm_cell_16/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:??????????2
while/lstm_cell_16/mul?
while/lstm_cell_16/ReluRelu!while/lstm_cell_16/split:output:2*
T0*(
_output_shapes
:??????????2
while/lstm_cell_16/Relu?
while/lstm_cell_16/mul_1Mulwhile/lstm_cell_16/Sigmoid:y:0%while/lstm_cell_16/Relu:activations:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_16/mul_1?
while/lstm_cell_16/add_1AddV2while/lstm_cell_16/mul:z:0while/lstm_cell_16/mul_1:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_16/add_1?
while/lstm_cell_16/Sigmoid_2Sigmoid!while/lstm_cell_16/split:output:3*
T0*(
_output_shapes
:??????????2
while/lstm_cell_16/Sigmoid_2?
while/lstm_cell_16/Relu_1Reluwhile/lstm_cell_16/add_1:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_16/Relu_1?
while/lstm_cell_16/mul_2Mul while/lstm_cell_16/Sigmoid_2:y:0'while/lstm_cell_16/Relu_1:activations:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_16/mul_2?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_16/mul_2:z:0*
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
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identitywhile/lstm_cell_16/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:??????????2
while/Identity_4?
while/Identity_5Identitywhile/lstm_cell_16/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:??????????2
while/Identity_5?

while/NoOpNoOp*^while/lstm_cell_16/BiasAdd/ReadVariableOp)^while/lstm_cell_16/MatMul/ReadVariableOp+^while/lstm_cell_16/MatMul_1/ReadVariableOp*"
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
2while_lstm_cell_16_biasadd_readvariableop_resource4while_lstm_cell_16_biasadd_readvariableop_resource_0"l
3while_lstm_cell_16_matmul_1_readvariableop_resource5while_lstm_cell_16_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_16_matmul_readvariableop_resource3while_lstm_cell_16_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :??????????:??????????: : : : : 2V
)while/lstm_cell_16/BiasAdd/ReadVariableOp)while/lstm_cell_16/BiasAdd/ReadVariableOp2T
(while/lstm_cell_16/MatMul/ReadVariableOp(while/lstm_cell_16/MatMul/ReadVariableOp2X
*while/lstm_cell_16/MatMul_1/ReadVariableOp*while/lstm_cell_16/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?&
?
while_body_32573552
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_00
while_lstm_cell_16_32573576_0:	]?1
while_lstm_cell_16_32573578_0:
??,
while_lstm_cell_16_32573580_0:	?
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor.
while_lstm_cell_16_32573576:	]?/
while_lstm_cell_16_32573578:
??*
while_lstm_cell_16_32573580:	???*while/lstm_cell_16/StatefulPartitionedCall?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????]   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????]*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
*while/lstm_cell_16/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_16_32573576_0while_lstm_cell_16_32573578_0while_lstm_cell_16_32573580_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:??????????:??????????:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_lstm_cell_16_layer_call_and_return_conditional_losses_325735382,
*while/lstm_cell_16/StatefulPartitionedCall?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder3while/lstm_cell_16/StatefulPartitionedCall:output:0*
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
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identity3while/lstm_cell_16/StatefulPartitionedCall:output:1^while/NoOp*
T0*(
_output_shapes
:??????????2
while/Identity_4?
while/Identity_5Identity3while/lstm_cell_16/StatefulPartitionedCall:output:2^while/NoOp*
T0*(
_output_shapes
:??????????2
while/Identity_5?

while/NoOpNoOp+^while/lstm_cell_16/StatefulPartitionedCall*"
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
while_lstm_cell_16_32573576while_lstm_cell_16_32573576_0"<
while_lstm_cell_16_32573578while_lstm_cell_16_32573578_0"<
while_lstm_cell_16_32573580while_lstm_cell_16_32573580_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :??????????:??????????: : : : : 2X
*while/lstm_cell_16/StatefulPartitionedCall*while/lstm_cell_16/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
??
?
while_body_32576916
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_16_matmul_readvariableop_resource_0:	]?I
5while_lstm_cell_16_matmul_1_readvariableop_resource_0:
??C
4while_lstm_cell_16_biasadd_readvariableop_resource_0:	?
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_16_matmul_readvariableop_resource:	]?G
3while_lstm_cell_16_matmul_1_readvariableop_resource:
??A
2while_lstm_cell_16_biasadd_readvariableop_resource:	???)while/lstm_cell_16/BiasAdd/ReadVariableOp?(while/lstm_cell_16/MatMul/ReadVariableOp?*while/lstm_cell_16/MatMul_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????]   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????]*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
(while/lstm_cell_16/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_16_matmul_readvariableop_resource_0*
_output_shapes
:	]?*
dtype02*
(while/lstm_cell_16/MatMul/ReadVariableOp?
while/lstm_cell_16/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_16/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_16/MatMul?
*while/lstm_cell_16/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_16_matmul_1_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02,
*while/lstm_cell_16/MatMul_1/ReadVariableOp?
while/lstm_cell_16/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_16/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_16/MatMul_1?
while/lstm_cell_16/addAddV2#while/lstm_cell_16/MatMul:product:0%while/lstm_cell_16/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_16/add?
)while/lstm_cell_16/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_16_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype02+
)while/lstm_cell_16/BiasAdd/ReadVariableOp?
while/lstm_cell_16/BiasAddBiasAddwhile/lstm_cell_16/add:z:01while/lstm_cell_16/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_16/BiasAdd?
"while/lstm_cell_16/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_16/split/split_dim?
while/lstm_cell_16/splitSplit+while/lstm_cell_16/split/split_dim:output:0#while/lstm_cell_16/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split2
while/lstm_cell_16/split?
while/lstm_cell_16/SigmoidSigmoid!while/lstm_cell_16/split:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_16/Sigmoid?
while/lstm_cell_16/Sigmoid_1Sigmoid!while/lstm_cell_16/split:output:1*
T0*(
_output_shapes
:??????????2
while/lstm_cell_16/Sigmoid_1?
while/lstm_cell_16/mulMul while/lstm_cell_16/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:??????????2
while/lstm_cell_16/mul?
while/lstm_cell_16/ReluRelu!while/lstm_cell_16/split:output:2*
T0*(
_output_shapes
:??????????2
while/lstm_cell_16/Relu?
while/lstm_cell_16/mul_1Mulwhile/lstm_cell_16/Sigmoid:y:0%while/lstm_cell_16/Relu:activations:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_16/mul_1?
while/lstm_cell_16/add_1AddV2while/lstm_cell_16/mul:z:0while/lstm_cell_16/mul_1:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_16/add_1?
while/lstm_cell_16/Sigmoid_2Sigmoid!while/lstm_cell_16/split:output:3*
T0*(
_output_shapes
:??????????2
while/lstm_cell_16/Sigmoid_2?
while/lstm_cell_16/Relu_1Reluwhile/lstm_cell_16/add_1:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_16/Relu_1?
while/lstm_cell_16/mul_2Mul while/lstm_cell_16/Sigmoid_2:y:0'while/lstm_cell_16/Relu_1:activations:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_16/mul_2?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_16/mul_2:z:0*
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
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identitywhile/lstm_cell_16/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:??????????2
while/Identity_4?
while/Identity_5Identitywhile/lstm_cell_16/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:??????????2
while/Identity_5?

while/NoOpNoOp*^while/lstm_cell_16/BiasAdd/ReadVariableOp)^while/lstm_cell_16/MatMul/ReadVariableOp+^while/lstm_cell_16/MatMul_1/ReadVariableOp*"
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
2while_lstm_cell_16_biasadd_readvariableop_resource4while_lstm_cell_16_biasadd_readvariableop_resource_0"l
3while_lstm_cell_16_matmul_1_readvariableop_resource5while_lstm_cell_16_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_16_matmul_readvariableop_resource3while_lstm_cell_16_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :??????????:??????????: : : : : 2V
)while/lstm_cell_16/BiasAdd/ReadVariableOp)while/lstm_cell_16/BiasAdd/ReadVariableOp2T
(while/lstm_cell_16/MatMul/ReadVariableOp(while/lstm_cell_16/MatMul/ReadVariableOp2X
*while/lstm_cell_16/MatMul_1/ReadVariableOp*while/lstm_cell_16/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?
?
while_cond_32574181
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_32574181___redundant_placeholder06
2while_while_cond_32574181___redundant_placeholder16
2while_while_cond_32574181___redundant_placeholder26
2while_while_cond_32574181___redundant_placeholder3
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
B: : : : :??????????:??????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?
f
H__inference_dropout_17_layer_call_and_return_conditional_losses_32575058

inputs

identity_1_
IdentityIdentityinputs*
T0*,
_output_shapes
:??????????2

Identityn

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:??????????2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?

?
/__inference_sequential_8_layer_call_fn_32575607
lstm_16_input
unknown:	]?
	unknown_0:
??
	unknown_1:	?
	unknown_2:
??
	unknown_3:
??
	unknown_4:	?
	unknown_5:	?
	unknown_6:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalllstm_16_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_sequential_8_layer_call_and_return_conditional_losses_325755672
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????]: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
+
_output_shapes
:?????????]
'
_user_specified_namelstm_16_input
?F
?
E__inference_lstm_16_layer_call_and_return_conditional_losses_32573831

inputs(
lstm_cell_16_32573749:	]?)
lstm_cell_16_32573751:
??$
lstm_cell_16_32573753:	?
identity??$lstm_cell_16/StatefulPartitionedCall?whileD
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
strided_slice/stack_2?
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
B :?2
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
B :?2
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
B :?2
zeros/packed/1?
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
:??????????2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
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
B :?2
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
B :?2
zeros_1/packed/1?
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
:??????????2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm?
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????]2
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
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????]   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
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
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????]*
shrink_axis_mask2
strided_slice_2?
$lstm_cell_16/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_16_32573749lstm_cell_16_32573751lstm_cell_16_32573753*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:??????????:??????????:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_lstm_cell_16_layer_call_and_return_conditional_losses_325736842&
$lstm_cell_16/StatefulPartitionedCall?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????  2
TensorArrayV2_1/element_shape?
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
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_16_32573749lstm_cell_16_32573751lstm_cell_16_32573753*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_32573762*
condR
while_cond_32573761*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????  22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:???????????????????*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
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
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:???????????????????2
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
!:???????????????????2

Identity}
NoOpNoOp%^lstm_cell_16/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????????????]: : : 2L
$lstm_cell_16/StatefulPartitionedCall$lstm_cell_16/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :??????????????????]
 
_user_specified_nameinputs
?
?
while_cond_32576613
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_32576613___redundant_placeholder06
2while_while_cond_32576613___redundant_placeholder16
2while_while_cond_32576613___redundant_placeholder26
2while_while_cond_32576613___redundant_placeholder3
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
B: : : : :??????????:??????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?
?
J__inference_lstm_cell_16_layer_call_and_return_conditional_losses_32573538

inputs

states
states_11
matmul_readvariableop_resource:	]?4
 matmul_1_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity

identity_1

identity_2??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?MatMul_1/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	]?*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
add?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split2
split`
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:??????????2	
Sigmoidd
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:??????????2
	Sigmoid_1]
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:??????????2
mulW
ReluRelusplit:output:2*
T0*(
_output_shapes
:??????????2
Relui
mul_1MulSigmoid:y:0Relu:activations:0*
T0*(
_output_shapes
:??????????2
mul_1^
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:??????????2
add_1d
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:??????????2
	Sigmoid_2V
Relu_1Relu	add_1:z:0*
T0*(
_output_shapes
:??????????2
Relu_1m
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*(
_output_shapes
:??????????2
mul_2e
IdentityIdentity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:??????????2

Identityi

Identity_1Identity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity_1i

Identity_2Identity	add_1:z:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity_2?
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
A:?????????]:??????????:??????????: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:?????????]
 
_user_specified_nameinputs:PL
(
_output_shapes
:??????????
 
_user_specified_namestates:PL
(
_output_shapes
:??????????
 
_user_specified_namestates
?
?
while_cond_32575229
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_32575229___redundant_placeholder06
2while_while_cond_32575229___redundant_placeholder16
2while_while_cond_32575229___redundant_placeholder26
2while_while_cond_32575229___redundant_placeholder3
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
B: : : : :??????????:??????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?

?
lstm_16_while_cond_32575752,
(lstm_16_while_lstm_16_while_loop_counter2
.lstm_16_while_lstm_16_while_maximum_iterations
lstm_16_while_placeholder
lstm_16_while_placeholder_1
lstm_16_while_placeholder_2
lstm_16_while_placeholder_3.
*lstm_16_while_less_lstm_16_strided_slice_1F
Blstm_16_while_lstm_16_while_cond_32575752___redundant_placeholder0F
Blstm_16_while_lstm_16_while_cond_32575752___redundant_placeholder1F
Blstm_16_while_lstm_16_while_cond_32575752___redundant_placeholder2F
Blstm_16_while_lstm_16_while_cond_32575752___redundant_placeholder3
lstm_16_while_identity
?
lstm_16/while/LessLesslstm_16_while_placeholder*lstm_16_while_less_lstm_16_strided_slice_1*
T0*
_output_shapes
: 2
lstm_16/while/Lessu
lstm_16/while/IdentityIdentitylstm_16/while/Less:z:0*
T0
*
_output_shapes
: 2
lstm_16/while/Identity"9
lstm_16_while_identitylstm_16/while/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :??????????:??????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?!
?
E__inference_dense_8_layer_call_and_return_conditional_losses_32577777

inputs4
!tensordot_readvariableop_resource:	?-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Tensordot/ReadVariableOp?
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes
:	?*
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
Tensordot/GatherV2/axis?
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
Tensordot/GatherV2_1/axis?
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
Tensordot/Const?
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
Tensordot/Const_1?
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
Tensordot/concat/axis?
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat?
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack?
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*,
_output_shapes
:??????????2
Tensordot/transpose?
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
Tensordot/Reshape?
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
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
Tensordot/concat_1/axis?
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1?
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????2
	Tensordot?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????2	
BiasAdde
SoftmaxSoftmaxBiasAdd:output:0*
T0*+
_output_shapes
:?????????2	
Softmaxp
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*+
_output_shapes
:?????????2

Identity?
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?\
?
E__inference_lstm_16_layer_call_and_return_conditional_losses_32576849

inputs>
+lstm_cell_16_matmul_readvariableop_resource:	]?A
-lstm_cell_16_matmul_1_readvariableop_resource:
??;
,lstm_cell_16_biasadd_readvariableop_resource:	?
identity??#lstm_cell_16/BiasAdd/ReadVariableOp?"lstm_cell_16/MatMul/ReadVariableOp?$lstm_cell_16/MatMul_1/ReadVariableOp?whileD
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
strided_slice/stack_2?
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
B :?2
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
B :?2
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
B :?2
zeros/packed/1?
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
:??????????2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
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
B :?2
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
B :?2
zeros_1/packed/1?
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
:??????????2	
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
:?????????]2
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
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????]   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
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
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????]*
shrink_axis_mask2
strided_slice_2?
"lstm_cell_16/MatMul/ReadVariableOpReadVariableOp+lstm_cell_16_matmul_readvariableop_resource*
_output_shapes
:	]?*
dtype02$
"lstm_cell_16/MatMul/ReadVariableOp?
lstm_cell_16/MatMulMatMulstrided_slice_2:output:0*lstm_cell_16/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_16/MatMul?
$lstm_cell_16/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_16_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02&
$lstm_cell_16/MatMul_1/ReadVariableOp?
lstm_cell_16/MatMul_1MatMulzeros:output:0,lstm_cell_16/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_16/MatMul_1?
lstm_cell_16/addAddV2lstm_cell_16/MatMul:product:0lstm_cell_16/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell_16/add?
#lstm_cell_16/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_16_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02%
#lstm_cell_16/BiasAdd/ReadVariableOp?
lstm_cell_16/BiasAddBiasAddlstm_cell_16/add:z:0+lstm_cell_16/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_16/BiasAdd~
lstm_cell_16/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_16/split/split_dim?
lstm_cell_16/splitSplit%lstm_cell_16/split/split_dim:output:0lstm_cell_16/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split2
lstm_cell_16/split?
lstm_cell_16/SigmoidSigmoidlstm_cell_16/split:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_16/Sigmoid?
lstm_cell_16/Sigmoid_1Sigmoidlstm_cell_16/split:output:1*
T0*(
_output_shapes
:??????????2
lstm_cell_16/Sigmoid_1?
lstm_cell_16/mulMullstm_cell_16/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_16/mul~
lstm_cell_16/ReluRelulstm_cell_16/split:output:2*
T0*(
_output_shapes
:??????????2
lstm_cell_16/Relu?
lstm_cell_16/mul_1Mullstm_cell_16/Sigmoid:y:0lstm_cell_16/Relu:activations:0*
T0*(
_output_shapes
:??????????2
lstm_cell_16/mul_1?
lstm_cell_16/add_1AddV2lstm_cell_16/mul:z:0lstm_cell_16/mul_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell_16/add_1?
lstm_cell_16/Sigmoid_2Sigmoidlstm_cell_16/split:output:3*
T0*(
_output_shapes
:??????????2
lstm_cell_16/Sigmoid_2}
lstm_cell_16/Relu_1Relulstm_cell_16/add_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell_16/Relu_1?
lstm_cell_16/mul_2Mullstm_cell_16/Sigmoid_2:y:0!lstm_cell_16/Relu_1:activations:0*
T0*(
_output_shapes
:??????????2
lstm_cell_16/mul_2?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????  2
TensorArrayV2_1/element_shape?
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
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_16_matmul_readvariableop_resource-lstm_cell_16_matmul_1_readvariableop_resource,lstm_cell_16_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_32576765*
condR
while_cond_32576764*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????  22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:??????????*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
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
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:??????????2
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
:??????????2

Identity?
NoOpNoOp$^lstm_cell_16/BiasAdd/ReadVariableOp#^lstm_cell_16/MatMul/ReadVariableOp%^lstm_cell_16/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????]: : : 2J
#lstm_cell_16/BiasAdd/ReadVariableOp#lstm_cell_16/BiasAdd/ReadVariableOp2H
"lstm_cell_16/MatMul/ReadVariableOp"lstm_cell_16/MatMul/ReadVariableOp2L
$lstm_cell_16/MatMul_1/ReadVariableOp$lstm_cell_16/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:?????????]
 
_user_specified_nameinputs
?
g
H__inference_dropout_17_layer_call_and_return_conditional_losses_32577736

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout/Constx
dropout/MulMulinputsdropout/Const:output:0*
T0*,
_output_shapes
:??????????2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:??????????*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:??????????2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:??????????2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:??????????2
dropout/Mul_1j
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
f
H__inference_dropout_17_layer_call_and_return_conditional_losses_32577724

inputs

identity_1_
IdentityIdentityinputs*
T0*,
_output_shapes
:??????????2

Identityn

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:??????????2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
J__inference_sequential_8_layer_call_and_return_conditional_losses_32575632
lstm_16_input#
lstm_16_32575610:	]?$
lstm_16_32575612:
??
lstm_16_32575614:	?$
lstm_17_32575618:
??$
lstm_17_32575620:
??
lstm_17_32575622:	?#
dense_8_32575626:	?
dense_8_32575628:
identity??dense_8/StatefulPartitionedCall?lstm_16/StatefulPartitionedCall?lstm_17/StatefulPartitionedCall?
lstm_16/StatefulPartitionedCallStatefulPartitionedCalllstm_16_inputlstm_16_32575610lstm_16_32575612lstm_16_32575614*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_lstm_16_layer_call_and_return_conditional_losses_325748802!
lstm_16/StatefulPartitionedCall?
dropout_16/PartitionedCallPartitionedCall(lstm_16/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_dropout_16_layer_call_and_return_conditional_losses_325748932
dropout_16/PartitionedCall?
lstm_17/StatefulPartitionedCallStatefulPartitionedCall#dropout_16/PartitionedCall:output:0lstm_17_32575618lstm_17_32575620lstm_17_32575622*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_lstm_17_layer_call_and_return_conditional_losses_325750452!
lstm_17/StatefulPartitionedCall?
dropout_17/PartitionedCallPartitionedCall(lstm_17/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_dropout_17_layer_call_and_return_conditional_losses_325750582
dropout_17/PartitionedCall?
dense_8/StatefulPartitionedCallStatefulPartitionedCall#dropout_17/PartitionedCall:output:0dense_8_32575626dense_8_32575628*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_8_layer_call_and_return_conditional_losses_325750912!
dense_8/StatefulPartitionedCall?
IdentityIdentity(dense_8/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????2

Identity?
NoOpNoOp ^dense_8/StatefulPartitionedCall ^lstm_16/StatefulPartitionedCall ^lstm_17/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????]: : : : : : : : 2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall2B
lstm_16/StatefulPartitionedCalllstm_16/StatefulPartitionedCall2B
lstm_17/StatefulPartitionedCalllstm_17/StatefulPartitionedCall:Z V
+
_output_shapes
:?????????]
'
_user_specified_namelstm_16_input
?J
?

lstm_17_while_body_32576235,
(lstm_17_while_lstm_17_while_loop_counter2
.lstm_17_while_lstm_17_while_maximum_iterations
lstm_17_while_placeholder
lstm_17_while_placeholder_1
lstm_17_while_placeholder_2
lstm_17_while_placeholder_3+
'lstm_17_while_lstm_17_strided_slice_1_0g
clstm_17_while_tensorarrayv2read_tensorlistgetitem_lstm_17_tensorarrayunstack_tensorlistfromtensor_0O
;lstm_17_while_lstm_cell_17_matmul_readvariableop_resource_0:
??Q
=lstm_17_while_lstm_cell_17_matmul_1_readvariableop_resource_0:
??K
<lstm_17_while_lstm_cell_17_biasadd_readvariableop_resource_0:	?
lstm_17_while_identity
lstm_17_while_identity_1
lstm_17_while_identity_2
lstm_17_while_identity_3
lstm_17_while_identity_4
lstm_17_while_identity_5)
%lstm_17_while_lstm_17_strided_slice_1e
alstm_17_while_tensorarrayv2read_tensorlistgetitem_lstm_17_tensorarrayunstack_tensorlistfromtensorM
9lstm_17_while_lstm_cell_17_matmul_readvariableop_resource:
??O
;lstm_17_while_lstm_cell_17_matmul_1_readvariableop_resource:
??I
:lstm_17_while_lstm_cell_17_biasadd_readvariableop_resource:	???1lstm_17/while/lstm_cell_17/BiasAdd/ReadVariableOp?0lstm_17/while/lstm_cell_17/MatMul/ReadVariableOp?2lstm_17/while/lstm_cell_17/MatMul_1/ReadVariableOp?
?lstm_17/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????  2A
?lstm_17/while/TensorArrayV2Read/TensorListGetItem/element_shape?
1lstm_17/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemclstm_17_while_tensorarrayv2read_tensorlistgetitem_lstm_17_tensorarrayunstack_tensorlistfromtensor_0lstm_17_while_placeholderHlstm_17/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:??????????*
element_dtype023
1lstm_17/while/TensorArrayV2Read/TensorListGetItem?
0lstm_17/while/lstm_cell_17/MatMul/ReadVariableOpReadVariableOp;lstm_17_while_lstm_cell_17_matmul_readvariableop_resource_0* 
_output_shapes
:
??*
dtype022
0lstm_17/while/lstm_cell_17/MatMul/ReadVariableOp?
!lstm_17/while/lstm_cell_17/MatMulMatMul8lstm_17/while/TensorArrayV2Read/TensorListGetItem:item:08lstm_17/while/lstm_cell_17/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2#
!lstm_17/while/lstm_cell_17/MatMul?
2lstm_17/while/lstm_cell_17/MatMul_1/ReadVariableOpReadVariableOp=lstm_17_while_lstm_cell_17_matmul_1_readvariableop_resource_0* 
_output_shapes
:
??*
dtype024
2lstm_17/while/lstm_cell_17/MatMul_1/ReadVariableOp?
#lstm_17/while/lstm_cell_17/MatMul_1MatMullstm_17_while_placeholder_2:lstm_17/while/lstm_cell_17/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2%
#lstm_17/while/lstm_cell_17/MatMul_1?
lstm_17/while/lstm_cell_17/addAddV2+lstm_17/while/lstm_cell_17/MatMul:product:0-lstm_17/while/lstm_cell_17/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2 
lstm_17/while/lstm_cell_17/add?
1lstm_17/while/lstm_cell_17/BiasAdd/ReadVariableOpReadVariableOp<lstm_17_while_lstm_cell_17_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype023
1lstm_17/while/lstm_cell_17/BiasAdd/ReadVariableOp?
"lstm_17/while/lstm_cell_17/BiasAddBiasAdd"lstm_17/while/lstm_cell_17/add:z:09lstm_17/while/lstm_cell_17/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2$
"lstm_17/while/lstm_cell_17/BiasAdd?
*lstm_17/while/lstm_cell_17/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2,
*lstm_17/while/lstm_cell_17/split/split_dim?
 lstm_17/while/lstm_cell_17/splitSplit3lstm_17/while/lstm_cell_17/split/split_dim:output:0+lstm_17/while/lstm_cell_17/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split2"
 lstm_17/while/lstm_cell_17/split?
"lstm_17/while/lstm_cell_17/SigmoidSigmoid)lstm_17/while/lstm_cell_17/split:output:0*
T0*(
_output_shapes
:??????????2$
"lstm_17/while/lstm_cell_17/Sigmoid?
$lstm_17/while/lstm_cell_17/Sigmoid_1Sigmoid)lstm_17/while/lstm_cell_17/split:output:1*
T0*(
_output_shapes
:??????????2&
$lstm_17/while/lstm_cell_17/Sigmoid_1?
lstm_17/while/lstm_cell_17/mulMul(lstm_17/while/lstm_cell_17/Sigmoid_1:y:0lstm_17_while_placeholder_3*
T0*(
_output_shapes
:??????????2 
lstm_17/while/lstm_cell_17/mul?
lstm_17/while/lstm_cell_17/ReluRelu)lstm_17/while/lstm_cell_17/split:output:2*
T0*(
_output_shapes
:??????????2!
lstm_17/while/lstm_cell_17/Relu?
 lstm_17/while/lstm_cell_17/mul_1Mul&lstm_17/while/lstm_cell_17/Sigmoid:y:0-lstm_17/while/lstm_cell_17/Relu:activations:0*
T0*(
_output_shapes
:??????????2"
 lstm_17/while/lstm_cell_17/mul_1?
 lstm_17/while/lstm_cell_17/add_1AddV2"lstm_17/while/lstm_cell_17/mul:z:0$lstm_17/while/lstm_cell_17/mul_1:z:0*
T0*(
_output_shapes
:??????????2"
 lstm_17/while/lstm_cell_17/add_1?
$lstm_17/while/lstm_cell_17/Sigmoid_2Sigmoid)lstm_17/while/lstm_cell_17/split:output:3*
T0*(
_output_shapes
:??????????2&
$lstm_17/while/lstm_cell_17/Sigmoid_2?
!lstm_17/while/lstm_cell_17/Relu_1Relu$lstm_17/while/lstm_cell_17/add_1:z:0*
T0*(
_output_shapes
:??????????2#
!lstm_17/while/lstm_cell_17/Relu_1?
 lstm_17/while/lstm_cell_17/mul_2Mul(lstm_17/while/lstm_cell_17/Sigmoid_2:y:0/lstm_17/while/lstm_cell_17/Relu_1:activations:0*
T0*(
_output_shapes
:??????????2"
 lstm_17/while/lstm_cell_17/mul_2?
2lstm_17/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_17_while_placeholder_1lstm_17_while_placeholder$lstm_17/while/lstm_cell_17/mul_2:z:0*
_output_shapes
: *
element_dtype024
2lstm_17/while/TensorArrayV2Write/TensorListSetIteml
lstm_17/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_17/while/add/y?
lstm_17/while/addAddV2lstm_17_while_placeholderlstm_17/while/add/y:output:0*
T0*
_output_shapes
: 2
lstm_17/while/addp
lstm_17/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_17/while/add_1/y?
lstm_17/while/add_1AddV2(lstm_17_while_lstm_17_while_loop_counterlstm_17/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lstm_17/while/add_1?
lstm_17/while/IdentityIdentitylstm_17/while/add_1:z:0^lstm_17/while/NoOp*
T0*
_output_shapes
: 2
lstm_17/while/Identity?
lstm_17/while/Identity_1Identity.lstm_17_while_lstm_17_while_maximum_iterations^lstm_17/while/NoOp*
T0*
_output_shapes
: 2
lstm_17/while/Identity_1?
lstm_17/while/Identity_2Identitylstm_17/while/add:z:0^lstm_17/while/NoOp*
T0*
_output_shapes
: 2
lstm_17/while/Identity_2?
lstm_17/while/Identity_3IdentityBlstm_17/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_17/while/NoOp*
T0*
_output_shapes
: 2
lstm_17/while/Identity_3?
lstm_17/while/Identity_4Identity$lstm_17/while/lstm_cell_17/mul_2:z:0^lstm_17/while/NoOp*
T0*(
_output_shapes
:??????????2
lstm_17/while/Identity_4?
lstm_17/while/Identity_5Identity$lstm_17/while/lstm_cell_17/add_1:z:0^lstm_17/while/NoOp*
T0*(
_output_shapes
:??????????2
lstm_17/while/Identity_5?
lstm_17/while/NoOpNoOp2^lstm_17/while/lstm_cell_17/BiasAdd/ReadVariableOp1^lstm_17/while/lstm_cell_17/MatMul/ReadVariableOp3^lstm_17/while/lstm_cell_17/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
lstm_17/while/NoOp"9
lstm_17_while_identitylstm_17/while/Identity:output:0"=
lstm_17_while_identity_1!lstm_17/while/Identity_1:output:0"=
lstm_17_while_identity_2!lstm_17/while/Identity_2:output:0"=
lstm_17_while_identity_3!lstm_17/while/Identity_3:output:0"=
lstm_17_while_identity_4!lstm_17/while/Identity_4:output:0"=
lstm_17_while_identity_5!lstm_17/while/Identity_5:output:0"P
%lstm_17_while_lstm_17_strided_slice_1'lstm_17_while_lstm_17_strided_slice_1_0"z
:lstm_17_while_lstm_cell_17_biasadd_readvariableop_resource<lstm_17_while_lstm_cell_17_biasadd_readvariableop_resource_0"|
;lstm_17_while_lstm_cell_17_matmul_1_readvariableop_resource=lstm_17_while_lstm_cell_17_matmul_1_readvariableop_resource_0"x
9lstm_17_while_lstm_cell_17_matmul_readvariableop_resource;lstm_17_while_lstm_cell_17_matmul_readvariableop_resource_0"?
alstm_17_while_tensorarrayv2read_tensorlistgetitem_lstm_17_tensorarrayunstack_tensorlistfromtensorclstm_17_while_tensorarrayv2read_tensorlistgetitem_lstm_17_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :??????????:??????????: : : : : 2f
1lstm_17/while/lstm_cell_17/BiasAdd/ReadVariableOp1lstm_17/while/lstm_cell_17/BiasAdd/ReadVariableOp2d
0lstm_17/while/lstm_cell_17/MatMul/ReadVariableOp0lstm_17/while/lstm_cell_17/MatMul/ReadVariableOp2h
2lstm_17/while/lstm_cell_17/MatMul_1/ReadVariableOp2lstm_17/while/lstm_cell_17/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?\
?
E__inference_lstm_17_layer_call_and_return_conditional_losses_32575314

inputs?
+lstm_cell_17_matmul_readvariableop_resource:
??A
-lstm_cell_17_matmul_1_readvariableop_resource:
??;
,lstm_cell_17_biasadd_readvariableop_resource:	?
identity??#lstm_cell_17/BiasAdd/ReadVariableOp?"lstm_cell_17/MatMul/ReadVariableOp?$lstm_cell_17/MatMul_1/ReadVariableOp?whileD
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
strided_slice/stack_2?
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
B :?2
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
B :?2
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
B :?2
zeros/packed/1?
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
:??????????2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
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
B :?2
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
B :?2
zeros_1/packed/1?
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
:??????????2	
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
:??????????2
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
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????  27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
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
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_2?
"lstm_cell_17/MatMul/ReadVariableOpReadVariableOp+lstm_cell_17_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02$
"lstm_cell_17/MatMul/ReadVariableOp?
lstm_cell_17/MatMulMatMulstrided_slice_2:output:0*lstm_cell_17/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_17/MatMul?
$lstm_cell_17/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_17_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02&
$lstm_cell_17/MatMul_1/ReadVariableOp?
lstm_cell_17/MatMul_1MatMulzeros:output:0,lstm_cell_17/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_17/MatMul_1?
lstm_cell_17/addAddV2lstm_cell_17/MatMul:product:0lstm_cell_17/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell_17/add?
#lstm_cell_17/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_17_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02%
#lstm_cell_17/BiasAdd/ReadVariableOp?
lstm_cell_17/BiasAddBiasAddlstm_cell_17/add:z:0+lstm_cell_17/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_17/BiasAdd~
lstm_cell_17/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_17/split/split_dim?
lstm_cell_17/splitSplit%lstm_cell_17/split/split_dim:output:0lstm_cell_17/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split2
lstm_cell_17/split?
lstm_cell_17/SigmoidSigmoidlstm_cell_17/split:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_17/Sigmoid?
lstm_cell_17/Sigmoid_1Sigmoidlstm_cell_17/split:output:1*
T0*(
_output_shapes
:??????????2
lstm_cell_17/Sigmoid_1?
lstm_cell_17/mulMullstm_cell_17/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_17/mul~
lstm_cell_17/ReluRelulstm_cell_17/split:output:2*
T0*(
_output_shapes
:??????????2
lstm_cell_17/Relu?
lstm_cell_17/mul_1Mullstm_cell_17/Sigmoid:y:0lstm_cell_17/Relu:activations:0*
T0*(
_output_shapes
:??????????2
lstm_cell_17/mul_1?
lstm_cell_17/add_1AddV2lstm_cell_17/mul:z:0lstm_cell_17/mul_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell_17/add_1?
lstm_cell_17/Sigmoid_2Sigmoidlstm_cell_17/split:output:3*
T0*(
_output_shapes
:??????????2
lstm_cell_17/Sigmoid_2}
lstm_cell_17/Relu_1Relulstm_cell_17/add_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell_17/Relu_1?
lstm_cell_17/mul_2Mullstm_cell_17/Sigmoid_2:y:0!lstm_cell_17/Relu_1:activations:0*
T0*(
_output_shapes
:??????????2
lstm_cell_17/mul_2?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2
TensorArrayV2_1/element_shape?
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
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_17_matmul_readvariableop_resource-lstm_cell_17_matmul_1_readvariableop_resource,lstm_cell_17_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_32575230*
condR
while_cond_32575229*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:??????????*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
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
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:??????????2
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
:??????????2

Identity?
NoOpNoOp$^lstm_cell_17/BiasAdd/ReadVariableOp#^lstm_cell_17/MatMul/ReadVariableOp%^lstm_cell_17/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:??????????: : : 2J
#lstm_cell_17/BiasAdd/ReadVariableOp#lstm_cell_17/BiasAdd/ReadVariableOp2H
"lstm_cell_17/MatMul/ReadVariableOp"lstm_cell_17/MatMul/ReadVariableOp2L
$lstm_cell_17/MatMul_1/ReadVariableOp$lstm_cell_17/MatMul_1/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?!
?
E__inference_dense_8_layer_call_and_return_conditional_losses_32575091

inputs4
!tensordot_readvariableop_resource:	?-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Tensordot/ReadVariableOp?
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes
:	?*
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
Tensordot/GatherV2/axis?
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
Tensordot/GatherV2_1/axis?
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
Tensordot/Const?
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
Tensordot/Const_1?
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
Tensordot/concat/axis?
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat?
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack?
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*,
_output_shapes
:??????????2
Tensordot/transpose?
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
Tensordot/Reshape?
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
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
Tensordot/concat_1/axis?
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1?
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????2
	Tensordot?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????2	
BiasAdde
SoftmaxSoftmaxBiasAdd:output:0*
T0*+
_output_shapes
:?????????2	
Softmaxp
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*+
_output_shapes
:?????????2

Identity?
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
J__inference_lstm_cell_16_layer_call_and_return_conditional_losses_32577818

inputs
states_0
states_11
matmul_readvariableop_resource:	]?4
 matmul_1_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity

identity_1

identity_2??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?MatMul_1/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	]?*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul_1/ReadVariableOp|
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
add?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split2
split`
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:??????????2	
Sigmoidd
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:??????????2
	Sigmoid_1]
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:??????????2
mulW
ReluRelusplit:output:2*
T0*(
_output_shapes
:??????????2
Relui
mul_1MulSigmoid:y:0Relu:activations:0*
T0*(
_output_shapes
:??????????2
mul_1^
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:??????????2
add_1d
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:??????????2
	Sigmoid_2V
Relu_1Relu	add_1:z:0*
T0*(
_output_shapes
:??????????2
Relu_1m
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*(
_output_shapes
:??????????2
mul_2e
IdentityIdentity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:??????????2

Identityi

Identity_1Identity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity_1i

Identity_2Identity	add_1:z:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity_2?
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
A:?????????]:??????????:??????????: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:?????????]
 
_user_specified_nameinputs:RN
(
_output_shapes
:??????????
"
_user_specified_name
states/0:RN
(
_output_shapes
:??????????
"
_user_specified_name
states/1
??
?
while_body_32576614
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_16_matmul_readvariableop_resource_0:	]?I
5while_lstm_cell_16_matmul_1_readvariableop_resource_0:
??C
4while_lstm_cell_16_biasadd_readvariableop_resource_0:	?
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_16_matmul_readvariableop_resource:	]?G
3while_lstm_cell_16_matmul_1_readvariableop_resource:
??A
2while_lstm_cell_16_biasadd_readvariableop_resource:	???)while/lstm_cell_16/BiasAdd/ReadVariableOp?(while/lstm_cell_16/MatMul/ReadVariableOp?*while/lstm_cell_16/MatMul_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????]   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????]*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
(while/lstm_cell_16/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_16_matmul_readvariableop_resource_0*
_output_shapes
:	]?*
dtype02*
(while/lstm_cell_16/MatMul/ReadVariableOp?
while/lstm_cell_16/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_16/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_16/MatMul?
*while/lstm_cell_16/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_16_matmul_1_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02,
*while/lstm_cell_16/MatMul_1/ReadVariableOp?
while/lstm_cell_16/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_16/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_16/MatMul_1?
while/lstm_cell_16/addAddV2#while/lstm_cell_16/MatMul:product:0%while/lstm_cell_16/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_16/add?
)while/lstm_cell_16/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_16_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype02+
)while/lstm_cell_16/BiasAdd/ReadVariableOp?
while/lstm_cell_16/BiasAddBiasAddwhile/lstm_cell_16/add:z:01while/lstm_cell_16/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_16/BiasAdd?
"while/lstm_cell_16/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_16/split/split_dim?
while/lstm_cell_16/splitSplit+while/lstm_cell_16/split/split_dim:output:0#while/lstm_cell_16/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split2
while/lstm_cell_16/split?
while/lstm_cell_16/SigmoidSigmoid!while/lstm_cell_16/split:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_16/Sigmoid?
while/lstm_cell_16/Sigmoid_1Sigmoid!while/lstm_cell_16/split:output:1*
T0*(
_output_shapes
:??????????2
while/lstm_cell_16/Sigmoid_1?
while/lstm_cell_16/mulMul while/lstm_cell_16/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:??????????2
while/lstm_cell_16/mul?
while/lstm_cell_16/ReluRelu!while/lstm_cell_16/split:output:2*
T0*(
_output_shapes
:??????????2
while/lstm_cell_16/Relu?
while/lstm_cell_16/mul_1Mulwhile/lstm_cell_16/Sigmoid:y:0%while/lstm_cell_16/Relu:activations:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_16/mul_1?
while/lstm_cell_16/add_1AddV2while/lstm_cell_16/mul:z:0while/lstm_cell_16/mul_1:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_16/add_1?
while/lstm_cell_16/Sigmoid_2Sigmoid!while/lstm_cell_16/split:output:3*
T0*(
_output_shapes
:??????????2
while/lstm_cell_16/Sigmoid_2?
while/lstm_cell_16/Relu_1Reluwhile/lstm_cell_16/add_1:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_16/Relu_1?
while/lstm_cell_16/mul_2Mul while/lstm_cell_16/Sigmoid_2:y:0'while/lstm_cell_16/Relu_1:activations:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_16/mul_2?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_16/mul_2:z:0*
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
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identitywhile/lstm_cell_16/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:??????????2
while/Identity_4?
while/Identity_5Identitywhile/lstm_cell_16/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:??????????2
while/Identity_5?

while/NoOpNoOp*^while/lstm_cell_16/BiasAdd/ReadVariableOp)^while/lstm_cell_16/MatMul/ReadVariableOp+^while/lstm_cell_16/MatMul_1/ReadVariableOp*"
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
2while_lstm_cell_16_biasadd_readvariableop_resource4while_lstm_cell_16_biasadd_readvariableop_resource_0"l
3while_lstm_cell_16_matmul_1_readvariableop_resource5while_lstm_cell_16_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_16_matmul_readvariableop_resource3while_lstm_cell_16_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :??????????:??????????: : : : : 2V
)while/lstm_cell_16/BiasAdd/ReadVariableOp)while/lstm_cell_16/BiasAdd/ReadVariableOp2T
(while/lstm_cell_16/MatMul/ReadVariableOp(while/lstm_cell_16/MatMul/ReadVariableOp2X
*while/lstm_cell_16/MatMul_1/ReadVariableOp*while/lstm_cell_16/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?
?
while_cond_32574795
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_32574795___redundant_placeholder06
2while_while_cond_32574795___redundant_placeholder16
2while_while_cond_32574795___redundant_placeholder26
2while_while_cond_32574795___redundant_placeholder3
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
B: : : : :??????????:??????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?
?
/__inference_lstm_cell_17_layer_call_fn_32577965

inputs
states_0
states_1
unknown:
??
	unknown_0:
??
	unknown_1:	?
identity

identity_1

identity_2??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:??????????:??????????:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_lstm_cell_17_layer_call_and_return_conditional_losses_325741682
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:??????????2

Identity_1?

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:??????????2

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
B:??????????:??????????:??????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:RN
(
_output_shapes
:??????????
"
_user_specified_name
states/0:RN
(
_output_shapes
:??????????
"
_user_specified_name
states/1
?&
?
while_body_32574182
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_01
while_lstm_cell_17_32574206_0:
??1
while_lstm_cell_17_32574208_0:
??,
while_lstm_cell_17_32574210_0:	?
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor/
while_lstm_cell_17_32574206:
??/
while_lstm_cell_17_32574208:
??*
while_lstm_cell_17_32574210:	???*while/lstm_cell_17/StatefulPartitionedCall?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????  29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:??????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
*while/lstm_cell_17/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_17_32574206_0while_lstm_cell_17_32574208_0while_lstm_cell_17_32574210_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:??????????:??????????:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_lstm_cell_17_layer_call_and_return_conditional_losses_325741682,
*while/lstm_cell_17/StatefulPartitionedCall?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder3while/lstm_cell_17/StatefulPartitionedCall:output:0*
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
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identity3while/lstm_cell_17/StatefulPartitionedCall:output:1^while/NoOp*
T0*(
_output_shapes
:??????????2
while/Identity_4?
while/Identity_5Identity3while/lstm_cell_17/StatefulPartitionedCall:output:2^while/NoOp*
T0*(
_output_shapes
:??????????2
while/Identity_5?

while/NoOpNoOp+^while/lstm_cell_17/StatefulPartitionedCall*"
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
while_lstm_cell_17_32574206while_lstm_cell_17_32574206_0"<
while_lstm_cell_17_32574208while_lstm_cell_17_32574208_0"<
while_lstm_cell_17_32574210while_lstm_cell_17_32574210_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :??????????:??????????: : : : : 2X
*while/lstm_cell_17/StatefulPartitionedCall*while/lstm_cell_17/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
??
?
while_body_32577138
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
3while_lstm_cell_17_matmul_readvariableop_resource_0:
??I
5while_lstm_cell_17_matmul_1_readvariableop_resource_0:
??C
4while_lstm_cell_17_biasadd_readvariableop_resource_0:	?
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
1while_lstm_cell_17_matmul_readvariableop_resource:
??G
3while_lstm_cell_17_matmul_1_readvariableop_resource:
??A
2while_lstm_cell_17_biasadd_readvariableop_resource:	???)while/lstm_cell_17/BiasAdd/ReadVariableOp?(while/lstm_cell_17/MatMul/ReadVariableOp?*while/lstm_cell_17/MatMul_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????  29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:??????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
(while/lstm_cell_17/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_17_matmul_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02*
(while/lstm_cell_17/MatMul/ReadVariableOp?
while/lstm_cell_17/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_17/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_17/MatMul?
*while/lstm_cell_17/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_17_matmul_1_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02,
*while/lstm_cell_17/MatMul_1/ReadVariableOp?
while/lstm_cell_17/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_17/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_17/MatMul_1?
while/lstm_cell_17/addAddV2#while/lstm_cell_17/MatMul:product:0%while/lstm_cell_17/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_17/add?
)while/lstm_cell_17/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_17_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype02+
)while/lstm_cell_17/BiasAdd/ReadVariableOp?
while/lstm_cell_17/BiasAddBiasAddwhile/lstm_cell_17/add:z:01while/lstm_cell_17/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_17/BiasAdd?
"while/lstm_cell_17/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_17/split/split_dim?
while/lstm_cell_17/splitSplit+while/lstm_cell_17/split/split_dim:output:0#while/lstm_cell_17/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split2
while/lstm_cell_17/split?
while/lstm_cell_17/SigmoidSigmoid!while/lstm_cell_17/split:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_17/Sigmoid?
while/lstm_cell_17/Sigmoid_1Sigmoid!while/lstm_cell_17/split:output:1*
T0*(
_output_shapes
:??????????2
while/lstm_cell_17/Sigmoid_1?
while/lstm_cell_17/mulMul while/lstm_cell_17/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:??????????2
while/lstm_cell_17/mul?
while/lstm_cell_17/ReluRelu!while/lstm_cell_17/split:output:2*
T0*(
_output_shapes
:??????????2
while/lstm_cell_17/Relu?
while/lstm_cell_17/mul_1Mulwhile/lstm_cell_17/Sigmoid:y:0%while/lstm_cell_17/Relu:activations:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_17/mul_1?
while/lstm_cell_17/add_1AddV2while/lstm_cell_17/mul:z:0while/lstm_cell_17/mul_1:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_17/add_1?
while/lstm_cell_17/Sigmoid_2Sigmoid!while/lstm_cell_17/split:output:3*
T0*(
_output_shapes
:??????????2
while/lstm_cell_17/Sigmoid_2?
while/lstm_cell_17/Relu_1Reluwhile/lstm_cell_17/add_1:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_17/Relu_1?
while/lstm_cell_17/mul_2Mul while/lstm_cell_17/Sigmoid_2:y:0'while/lstm_cell_17/Relu_1:activations:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_17/mul_2?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_17/mul_2:z:0*
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
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identitywhile/lstm_cell_17/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:??????????2
while/Identity_4?
while/Identity_5Identitywhile/lstm_cell_17/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:??????????2
while/Identity_5?

while/NoOpNoOp*^while/lstm_cell_17/BiasAdd/ReadVariableOp)^while/lstm_cell_17/MatMul/ReadVariableOp+^while/lstm_cell_17/MatMul_1/ReadVariableOp*"
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
2while_lstm_cell_17_biasadd_readvariableop_resource4while_lstm_cell_17_biasadd_readvariableop_resource_0"l
3while_lstm_cell_17_matmul_1_readvariableop_resource5while_lstm_cell_17_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_17_matmul_readvariableop_resource3while_lstm_cell_17_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :??????????:??????????: : : : : 2V
)while/lstm_cell_17/BiasAdd/ReadVariableOp)while/lstm_cell_17/BiasAdd/ReadVariableOp2T
(while/lstm_cell_17/MatMul/ReadVariableOp(while/lstm_cell_17/MatMul/ReadVariableOp2X
*while/lstm_cell_17/MatMul_1/ReadVariableOp*while/lstm_cell_17/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?
I
-__inference_dropout_16_layer_call_fn_32577066

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_dropout_16_layer_call_and_return_conditional_losses_325748932
PartitionedCallq
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
while_cond_32577590
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_32577590___redundant_placeholder06
2while_while_cond_32577590___redundant_placeholder16
2while_while_cond_32577590___redundant_placeholder26
2while_while_cond_32577590___redundant_placeholder3
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
B: : : : :??????????:??????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?
?
while_cond_32576764
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_32576764___redundant_placeholder06
2while_while_cond_32576764___redundant_placeholder16
2while_while_cond_32576764___redundant_placeholder26
2while_while_cond_32576764___redundant_placeholder3
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
B: : : : :??????????:??????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?
?
while_cond_32574960
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_32574960___redundant_placeholder06
2while_while_cond_32574960___redundant_placeholder16
2while_while_cond_32574960___redundant_placeholder26
2while_while_cond_32574960___redundant_placeholder3
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
B: : : : :??????????:??????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?

?
/__inference_sequential_8_layer_call_fn_32575117
lstm_16_input
unknown:	]?
	unknown_0:
??
	unknown_1:	?
	unknown_2:
??
	unknown_3:
??
	unknown_4:	?
	unknown_5:	?
	unknown_6:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalllstm_16_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_sequential_8_layer_call_and_return_conditional_losses_325750982
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????]: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
+
_output_shapes
:?????????]
'
_user_specified_namelstm_16_input
?
?
(sequential_8_lstm_17_while_cond_32573350F
Bsequential_8_lstm_17_while_sequential_8_lstm_17_while_loop_counterL
Hsequential_8_lstm_17_while_sequential_8_lstm_17_while_maximum_iterations*
&sequential_8_lstm_17_while_placeholder,
(sequential_8_lstm_17_while_placeholder_1,
(sequential_8_lstm_17_while_placeholder_2,
(sequential_8_lstm_17_while_placeholder_3H
Dsequential_8_lstm_17_while_less_sequential_8_lstm_17_strided_slice_1`
\sequential_8_lstm_17_while_sequential_8_lstm_17_while_cond_32573350___redundant_placeholder0`
\sequential_8_lstm_17_while_sequential_8_lstm_17_while_cond_32573350___redundant_placeholder1`
\sequential_8_lstm_17_while_sequential_8_lstm_17_while_cond_32573350___redundant_placeholder2`
\sequential_8_lstm_17_while_sequential_8_lstm_17_while_cond_32573350___redundant_placeholder3'
#sequential_8_lstm_17_while_identity
?
sequential_8/lstm_17/while/LessLess&sequential_8_lstm_17_while_placeholderDsequential_8_lstm_17_while_less_sequential_8_lstm_17_strided_slice_1*
T0*
_output_shapes
: 2!
sequential_8/lstm_17/while/Less?
#sequential_8/lstm_17/while/IdentityIdentity#sequential_8/lstm_17/while/Less:z:0*
T0
*
_output_shapes
: 2%
#sequential_8/lstm_17/while/Identity"S
#sequential_8_lstm_17_while_identity,sequential_8/lstm_17/while/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :??????????:??????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?
?
J__inference_sequential_8_layer_call_and_return_conditional_losses_32575098

inputs#
lstm_16_32574881:	]?$
lstm_16_32574883:
??
lstm_16_32574885:	?$
lstm_17_32575046:
??$
lstm_17_32575048:
??
lstm_17_32575050:	?#
dense_8_32575092:	?
dense_8_32575094:
identity??dense_8/StatefulPartitionedCall?lstm_16/StatefulPartitionedCall?lstm_17/StatefulPartitionedCall?
lstm_16/StatefulPartitionedCallStatefulPartitionedCallinputslstm_16_32574881lstm_16_32574883lstm_16_32574885*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_lstm_16_layer_call_and_return_conditional_losses_325748802!
lstm_16/StatefulPartitionedCall?
dropout_16/PartitionedCallPartitionedCall(lstm_16/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_dropout_16_layer_call_and_return_conditional_losses_325748932
dropout_16/PartitionedCall?
lstm_17/StatefulPartitionedCallStatefulPartitionedCall#dropout_16/PartitionedCall:output:0lstm_17_32575046lstm_17_32575048lstm_17_32575050*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_lstm_17_layer_call_and_return_conditional_losses_325750452!
lstm_17/StatefulPartitionedCall?
dropout_17/PartitionedCallPartitionedCall(lstm_17/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_dropout_17_layer_call_and_return_conditional_losses_325750582
dropout_17/PartitionedCall?
dense_8/StatefulPartitionedCallStatefulPartitionedCall#dropout_17/PartitionedCall:output:0dense_8_32575092dense_8_32575094*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_8_layer_call_and_return_conditional_losses_325750912!
dense_8/StatefulPartitionedCall?
IdentityIdentity(dense_8/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????2

Identity?
NoOpNoOp ^dense_8/StatefulPartitionedCall ^lstm_16/StatefulPartitionedCall ^lstm_17/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????]: : : : : : : : 2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall2B
lstm_16/StatefulPartitionedCalllstm_16/StatefulPartitionedCall2B
lstm_17/StatefulPartitionedCalllstm_17/StatefulPartitionedCall:S O
+
_output_shapes
:?????????]
 
_user_specified_nameinputs
?
?
J__inference_lstm_cell_16_layer_call_and_return_conditional_losses_32573684

inputs

states
states_11
matmul_readvariableop_resource:	]?4
 matmul_1_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity

identity_1

identity_2??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?MatMul_1/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	]?*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
add?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split2
split`
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:??????????2	
Sigmoidd
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:??????????2
	Sigmoid_1]
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:??????????2
mulW
ReluRelusplit:output:2*
T0*(
_output_shapes
:??????????2
Relui
mul_1MulSigmoid:y:0Relu:activations:0*
T0*(
_output_shapes
:??????????2
mul_1^
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:??????????2
add_1d
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:??????????2
	Sigmoid_2V
Relu_1Relu	add_1:z:0*
T0*(
_output_shapes
:??????????2
Relu_1m
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*(
_output_shapes
:??????????2
mul_2e
IdentityIdentity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:??????????2

Identityi

Identity_1Identity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity_1i

Identity_2Identity	add_1:z:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity_2?
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
A:?????????]:??????????:??????????: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:?????????]
 
_user_specified_nameinputs:PL
(
_output_shapes
:??????????
 
_user_specified_namestates:PL
(
_output_shapes
:??????????
 
_user_specified_namestates
?\
?
E__inference_lstm_16_layer_call_and_return_conditional_losses_32577000

inputs>
+lstm_cell_16_matmul_readvariableop_resource:	]?A
-lstm_cell_16_matmul_1_readvariableop_resource:
??;
,lstm_cell_16_biasadd_readvariableop_resource:	?
identity??#lstm_cell_16/BiasAdd/ReadVariableOp?"lstm_cell_16/MatMul/ReadVariableOp?$lstm_cell_16/MatMul_1/ReadVariableOp?whileD
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
strided_slice/stack_2?
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
B :?2
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
B :?2
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
B :?2
zeros/packed/1?
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
:??????????2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
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
B :?2
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
B :?2
zeros_1/packed/1?
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
:??????????2	
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
:?????????]2
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
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????]   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
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
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????]*
shrink_axis_mask2
strided_slice_2?
"lstm_cell_16/MatMul/ReadVariableOpReadVariableOp+lstm_cell_16_matmul_readvariableop_resource*
_output_shapes
:	]?*
dtype02$
"lstm_cell_16/MatMul/ReadVariableOp?
lstm_cell_16/MatMulMatMulstrided_slice_2:output:0*lstm_cell_16/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_16/MatMul?
$lstm_cell_16/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_16_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02&
$lstm_cell_16/MatMul_1/ReadVariableOp?
lstm_cell_16/MatMul_1MatMulzeros:output:0,lstm_cell_16/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_16/MatMul_1?
lstm_cell_16/addAddV2lstm_cell_16/MatMul:product:0lstm_cell_16/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell_16/add?
#lstm_cell_16/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_16_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02%
#lstm_cell_16/BiasAdd/ReadVariableOp?
lstm_cell_16/BiasAddBiasAddlstm_cell_16/add:z:0+lstm_cell_16/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_16/BiasAdd~
lstm_cell_16/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_16/split/split_dim?
lstm_cell_16/splitSplit%lstm_cell_16/split/split_dim:output:0lstm_cell_16/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split2
lstm_cell_16/split?
lstm_cell_16/SigmoidSigmoidlstm_cell_16/split:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_16/Sigmoid?
lstm_cell_16/Sigmoid_1Sigmoidlstm_cell_16/split:output:1*
T0*(
_output_shapes
:??????????2
lstm_cell_16/Sigmoid_1?
lstm_cell_16/mulMullstm_cell_16/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_16/mul~
lstm_cell_16/ReluRelulstm_cell_16/split:output:2*
T0*(
_output_shapes
:??????????2
lstm_cell_16/Relu?
lstm_cell_16/mul_1Mullstm_cell_16/Sigmoid:y:0lstm_cell_16/Relu:activations:0*
T0*(
_output_shapes
:??????????2
lstm_cell_16/mul_1?
lstm_cell_16/add_1AddV2lstm_cell_16/mul:z:0lstm_cell_16/mul_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell_16/add_1?
lstm_cell_16/Sigmoid_2Sigmoidlstm_cell_16/split:output:3*
T0*(
_output_shapes
:??????????2
lstm_cell_16/Sigmoid_2}
lstm_cell_16/Relu_1Relulstm_cell_16/add_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell_16/Relu_1?
lstm_cell_16/mul_2Mullstm_cell_16/Sigmoid_2:y:0!lstm_cell_16/Relu_1:activations:0*
T0*(
_output_shapes
:??????????2
lstm_cell_16/mul_2?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????  2
TensorArrayV2_1/element_shape?
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
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_16_matmul_readvariableop_resource-lstm_cell_16_matmul_1_readvariableop_resource,lstm_cell_16_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_32576916*
condR
while_cond_32576915*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????  22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:??????????*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
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
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:??????????2
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
:??????????2

Identity?
NoOpNoOp$^lstm_cell_16/BiasAdd/ReadVariableOp#^lstm_cell_16/MatMul/ReadVariableOp%^lstm_cell_16/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????]: : : 2J
#lstm_cell_16/BiasAdd/ReadVariableOp#lstm_cell_16/BiasAdd/ReadVariableOp2H
"lstm_cell_16/MatMul/ReadVariableOp"lstm_cell_16/MatMul/ReadVariableOp2L
$lstm_cell_16/MatMul_1/ReadVariableOp$lstm_cell_16/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:?????????]
 
_user_specified_nameinputs
?
?
while_cond_32573551
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_32573551___redundant_placeholder06
2while_while_cond_32573551___redundant_placeholder16
2while_while_cond_32573551___redundant_placeholder26
2while_while_cond_32573551___redundant_placeholder3
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
B: : : : :??????????:??????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?
f
H__inference_dropout_16_layer_call_and_return_conditional_losses_32574893

inputs

identity_1_
IdentityIdentityinputs*
T0*,
_output_shapes
:??????????2

Identityn

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:??????????2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?

?
lstm_17_while_cond_32575900,
(lstm_17_while_lstm_17_while_loop_counter2
.lstm_17_while_lstm_17_while_maximum_iterations
lstm_17_while_placeholder
lstm_17_while_placeholder_1
lstm_17_while_placeholder_2
lstm_17_while_placeholder_3.
*lstm_17_while_less_lstm_17_strided_slice_1F
Blstm_17_while_lstm_17_while_cond_32575900___redundant_placeholder0F
Blstm_17_while_lstm_17_while_cond_32575900___redundant_placeholder1F
Blstm_17_while_lstm_17_while_cond_32575900___redundant_placeholder2F
Blstm_17_while_lstm_17_while_cond_32575900___redundant_placeholder3
lstm_17_while_identity
?
lstm_17/while/LessLesslstm_17_while_placeholder*lstm_17_while_less_lstm_17_strided_slice_1*
T0*
_output_shapes
: 2
lstm_17/while/Lessu
lstm_17/while/IdentityIdentitylstm_17/while/Less:z:0*
T0
*
_output_shapes
: 2
lstm_17/while/Identity"9
lstm_17_while_identitylstm_17/while/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :??????????:??????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?
g
H__inference_dropout_16_layer_call_and_return_conditional_losses_32575343

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2
dropout/Constx
dropout/MulMulinputsdropout/Const:output:0*
T0*,
_output_shapes
:??????????2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:??????????*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?>2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:??????????2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:??????????2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:??????????2
dropout/Mul_1j
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
*__inference_lstm_16_layer_call_fn_32577022
inputs_0
unknown:	]?
	unknown_0:
??
	unknown_1:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:???????????????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_lstm_16_layer_call_and_return_conditional_losses_325738312
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:???????????????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????????????]: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :??????????????????]
"
_user_specified_name
inputs/0
?	
?
&__inference_signature_wrapper_32575686
lstm_16_input
unknown:	]?
	unknown_0:
??
	unknown_1:	?
	unknown_2:
??
	unknown_3:
??
	unknown_4:	?
	unknown_5:	?
	unknown_6:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalllstm_16_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *,
f'R%
#__inference__wrapped_model_325734632
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????]: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
+
_output_shapes
:?????????]
'
_user_specified_namelstm_16_input
?J
?

lstm_16_while_body_32575753,
(lstm_16_while_lstm_16_while_loop_counter2
.lstm_16_while_lstm_16_while_maximum_iterations
lstm_16_while_placeholder
lstm_16_while_placeholder_1
lstm_16_while_placeholder_2
lstm_16_while_placeholder_3+
'lstm_16_while_lstm_16_strided_slice_1_0g
clstm_16_while_tensorarrayv2read_tensorlistgetitem_lstm_16_tensorarrayunstack_tensorlistfromtensor_0N
;lstm_16_while_lstm_cell_16_matmul_readvariableop_resource_0:	]?Q
=lstm_16_while_lstm_cell_16_matmul_1_readvariableop_resource_0:
??K
<lstm_16_while_lstm_cell_16_biasadd_readvariableop_resource_0:	?
lstm_16_while_identity
lstm_16_while_identity_1
lstm_16_while_identity_2
lstm_16_while_identity_3
lstm_16_while_identity_4
lstm_16_while_identity_5)
%lstm_16_while_lstm_16_strided_slice_1e
alstm_16_while_tensorarrayv2read_tensorlistgetitem_lstm_16_tensorarrayunstack_tensorlistfromtensorL
9lstm_16_while_lstm_cell_16_matmul_readvariableop_resource:	]?O
;lstm_16_while_lstm_cell_16_matmul_1_readvariableop_resource:
??I
:lstm_16_while_lstm_cell_16_biasadd_readvariableop_resource:	???1lstm_16/while/lstm_cell_16/BiasAdd/ReadVariableOp?0lstm_16/while/lstm_cell_16/MatMul/ReadVariableOp?2lstm_16/while/lstm_cell_16/MatMul_1/ReadVariableOp?
?lstm_16/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????]   2A
?lstm_16/while/TensorArrayV2Read/TensorListGetItem/element_shape?
1lstm_16/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemclstm_16_while_tensorarrayv2read_tensorlistgetitem_lstm_16_tensorarrayunstack_tensorlistfromtensor_0lstm_16_while_placeholderHlstm_16/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????]*
element_dtype023
1lstm_16/while/TensorArrayV2Read/TensorListGetItem?
0lstm_16/while/lstm_cell_16/MatMul/ReadVariableOpReadVariableOp;lstm_16_while_lstm_cell_16_matmul_readvariableop_resource_0*
_output_shapes
:	]?*
dtype022
0lstm_16/while/lstm_cell_16/MatMul/ReadVariableOp?
!lstm_16/while/lstm_cell_16/MatMulMatMul8lstm_16/while/TensorArrayV2Read/TensorListGetItem:item:08lstm_16/while/lstm_cell_16/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2#
!lstm_16/while/lstm_cell_16/MatMul?
2lstm_16/while/lstm_cell_16/MatMul_1/ReadVariableOpReadVariableOp=lstm_16_while_lstm_cell_16_matmul_1_readvariableop_resource_0* 
_output_shapes
:
??*
dtype024
2lstm_16/while/lstm_cell_16/MatMul_1/ReadVariableOp?
#lstm_16/while/lstm_cell_16/MatMul_1MatMullstm_16_while_placeholder_2:lstm_16/while/lstm_cell_16/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2%
#lstm_16/while/lstm_cell_16/MatMul_1?
lstm_16/while/lstm_cell_16/addAddV2+lstm_16/while/lstm_cell_16/MatMul:product:0-lstm_16/while/lstm_cell_16/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2 
lstm_16/while/lstm_cell_16/add?
1lstm_16/while/lstm_cell_16/BiasAdd/ReadVariableOpReadVariableOp<lstm_16_while_lstm_cell_16_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype023
1lstm_16/while/lstm_cell_16/BiasAdd/ReadVariableOp?
"lstm_16/while/lstm_cell_16/BiasAddBiasAdd"lstm_16/while/lstm_cell_16/add:z:09lstm_16/while/lstm_cell_16/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2$
"lstm_16/while/lstm_cell_16/BiasAdd?
*lstm_16/while/lstm_cell_16/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2,
*lstm_16/while/lstm_cell_16/split/split_dim?
 lstm_16/while/lstm_cell_16/splitSplit3lstm_16/while/lstm_cell_16/split/split_dim:output:0+lstm_16/while/lstm_cell_16/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split2"
 lstm_16/while/lstm_cell_16/split?
"lstm_16/while/lstm_cell_16/SigmoidSigmoid)lstm_16/while/lstm_cell_16/split:output:0*
T0*(
_output_shapes
:??????????2$
"lstm_16/while/lstm_cell_16/Sigmoid?
$lstm_16/while/lstm_cell_16/Sigmoid_1Sigmoid)lstm_16/while/lstm_cell_16/split:output:1*
T0*(
_output_shapes
:??????????2&
$lstm_16/while/lstm_cell_16/Sigmoid_1?
lstm_16/while/lstm_cell_16/mulMul(lstm_16/while/lstm_cell_16/Sigmoid_1:y:0lstm_16_while_placeholder_3*
T0*(
_output_shapes
:??????????2 
lstm_16/while/lstm_cell_16/mul?
lstm_16/while/lstm_cell_16/ReluRelu)lstm_16/while/lstm_cell_16/split:output:2*
T0*(
_output_shapes
:??????????2!
lstm_16/while/lstm_cell_16/Relu?
 lstm_16/while/lstm_cell_16/mul_1Mul&lstm_16/while/lstm_cell_16/Sigmoid:y:0-lstm_16/while/lstm_cell_16/Relu:activations:0*
T0*(
_output_shapes
:??????????2"
 lstm_16/while/lstm_cell_16/mul_1?
 lstm_16/while/lstm_cell_16/add_1AddV2"lstm_16/while/lstm_cell_16/mul:z:0$lstm_16/while/lstm_cell_16/mul_1:z:0*
T0*(
_output_shapes
:??????????2"
 lstm_16/while/lstm_cell_16/add_1?
$lstm_16/while/lstm_cell_16/Sigmoid_2Sigmoid)lstm_16/while/lstm_cell_16/split:output:3*
T0*(
_output_shapes
:??????????2&
$lstm_16/while/lstm_cell_16/Sigmoid_2?
!lstm_16/while/lstm_cell_16/Relu_1Relu$lstm_16/while/lstm_cell_16/add_1:z:0*
T0*(
_output_shapes
:??????????2#
!lstm_16/while/lstm_cell_16/Relu_1?
 lstm_16/while/lstm_cell_16/mul_2Mul(lstm_16/while/lstm_cell_16/Sigmoid_2:y:0/lstm_16/while/lstm_cell_16/Relu_1:activations:0*
T0*(
_output_shapes
:??????????2"
 lstm_16/while/lstm_cell_16/mul_2?
2lstm_16/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_16_while_placeholder_1lstm_16_while_placeholder$lstm_16/while/lstm_cell_16/mul_2:z:0*
_output_shapes
: *
element_dtype024
2lstm_16/while/TensorArrayV2Write/TensorListSetIteml
lstm_16/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_16/while/add/y?
lstm_16/while/addAddV2lstm_16_while_placeholderlstm_16/while/add/y:output:0*
T0*
_output_shapes
: 2
lstm_16/while/addp
lstm_16/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_16/while/add_1/y?
lstm_16/while/add_1AddV2(lstm_16_while_lstm_16_while_loop_counterlstm_16/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lstm_16/while/add_1?
lstm_16/while/IdentityIdentitylstm_16/while/add_1:z:0^lstm_16/while/NoOp*
T0*
_output_shapes
: 2
lstm_16/while/Identity?
lstm_16/while/Identity_1Identity.lstm_16_while_lstm_16_while_maximum_iterations^lstm_16/while/NoOp*
T0*
_output_shapes
: 2
lstm_16/while/Identity_1?
lstm_16/while/Identity_2Identitylstm_16/while/add:z:0^lstm_16/while/NoOp*
T0*
_output_shapes
: 2
lstm_16/while/Identity_2?
lstm_16/while/Identity_3IdentityBlstm_16/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_16/while/NoOp*
T0*
_output_shapes
: 2
lstm_16/while/Identity_3?
lstm_16/while/Identity_4Identity$lstm_16/while/lstm_cell_16/mul_2:z:0^lstm_16/while/NoOp*
T0*(
_output_shapes
:??????????2
lstm_16/while/Identity_4?
lstm_16/while/Identity_5Identity$lstm_16/while/lstm_cell_16/add_1:z:0^lstm_16/while/NoOp*
T0*(
_output_shapes
:??????????2
lstm_16/while/Identity_5?
lstm_16/while/NoOpNoOp2^lstm_16/while/lstm_cell_16/BiasAdd/ReadVariableOp1^lstm_16/while/lstm_cell_16/MatMul/ReadVariableOp3^lstm_16/while/lstm_cell_16/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
lstm_16/while/NoOp"9
lstm_16_while_identitylstm_16/while/Identity:output:0"=
lstm_16_while_identity_1!lstm_16/while/Identity_1:output:0"=
lstm_16_while_identity_2!lstm_16/while/Identity_2:output:0"=
lstm_16_while_identity_3!lstm_16/while/Identity_3:output:0"=
lstm_16_while_identity_4!lstm_16/while/Identity_4:output:0"=
lstm_16_while_identity_5!lstm_16/while/Identity_5:output:0"P
%lstm_16_while_lstm_16_strided_slice_1'lstm_16_while_lstm_16_strided_slice_1_0"z
:lstm_16_while_lstm_cell_16_biasadd_readvariableop_resource<lstm_16_while_lstm_cell_16_biasadd_readvariableop_resource_0"|
;lstm_16_while_lstm_cell_16_matmul_1_readvariableop_resource=lstm_16_while_lstm_cell_16_matmul_1_readvariableop_resource_0"x
9lstm_16_while_lstm_cell_16_matmul_readvariableop_resource;lstm_16_while_lstm_cell_16_matmul_readvariableop_resource_0"?
alstm_16_while_tensorarrayv2read_tensorlistgetitem_lstm_16_tensorarrayunstack_tensorlistfromtensorclstm_16_while_tensorarrayv2read_tensorlistgetitem_lstm_16_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :??????????:??????????: : : : : 2f
1lstm_16/while/lstm_cell_16/BiasAdd/ReadVariableOp1lstm_16/while/lstm_cell_16/BiasAdd/ReadVariableOp2d
0lstm_16/while/lstm_cell_16/MatMul/ReadVariableOp0lstm_16/while/lstm_cell_16/MatMul/ReadVariableOp2h
2lstm_16/while/lstm_cell_16/MatMul_1/ReadVariableOp2lstm_16/while/lstm_cell_16/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?
?
/__inference_lstm_cell_17_layer_call_fn_32577982

inputs
states_0
states_1
unknown:
??
	unknown_0:
??
	unknown_1:	?
identity

identity_1

identity_2??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:??????????:??????????:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_lstm_cell_17_layer_call_and_return_conditional_losses_325743142
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:??????????2

Identity_1?

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:??????????2

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
B:??????????:??????????:??????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:RN
(
_output_shapes
:??????????
"
_user_specified_name
states/0:RN
(
_output_shapes
:??????????
"
_user_specified_name
states/1
?
f
-__inference_dropout_17_layer_call_fn_32577746

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_dropout_17_layer_call_and_return_conditional_losses_325751472
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:??????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
??
?
J__inference_sequential_8_layer_call_and_return_conditional_losses_32576013

inputsF
3lstm_16_lstm_cell_16_matmul_readvariableop_resource:	]?I
5lstm_16_lstm_cell_16_matmul_1_readvariableop_resource:
??C
4lstm_16_lstm_cell_16_biasadd_readvariableop_resource:	?G
3lstm_17_lstm_cell_17_matmul_readvariableop_resource:
??I
5lstm_17_lstm_cell_17_matmul_1_readvariableop_resource:
??C
4lstm_17_lstm_cell_17_biasadd_readvariableop_resource:	?<
)dense_8_tensordot_readvariableop_resource:	?5
'dense_8_biasadd_readvariableop_resource:
identity??dense_8/BiasAdd/ReadVariableOp? dense_8/Tensordot/ReadVariableOp?+lstm_16/lstm_cell_16/BiasAdd/ReadVariableOp?*lstm_16/lstm_cell_16/MatMul/ReadVariableOp?,lstm_16/lstm_cell_16/MatMul_1/ReadVariableOp?lstm_16/while?+lstm_17/lstm_cell_17/BiasAdd/ReadVariableOp?*lstm_17/lstm_cell_17/MatMul/ReadVariableOp?,lstm_17/lstm_cell_17/MatMul_1/ReadVariableOp?lstm_17/whileT
lstm_16/ShapeShapeinputs*
T0*
_output_shapes
:2
lstm_16/Shape?
lstm_16/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_16/strided_slice/stack?
lstm_16/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_16/strided_slice/stack_1?
lstm_16/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_16/strided_slice/stack_2?
lstm_16/strided_sliceStridedSlicelstm_16/Shape:output:0$lstm_16/strided_slice/stack:output:0&lstm_16/strided_slice/stack_1:output:0&lstm_16/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_16/strided_slicem
lstm_16/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
lstm_16/zeros/mul/y?
lstm_16/zeros/mulMullstm_16/strided_slice:output:0lstm_16/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_16/zeros/mulo
lstm_16/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
lstm_16/zeros/Less/y?
lstm_16/zeros/LessLesslstm_16/zeros/mul:z:0lstm_16/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_16/zeros/Lesss
lstm_16/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
lstm_16/zeros/packed/1?
lstm_16/zeros/packedPacklstm_16/strided_slice:output:0lstm_16/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_16/zeros/packedo
lstm_16/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_16/zeros/Const?
lstm_16/zerosFilllstm_16/zeros/packed:output:0lstm_16/zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
lstm_16/zerosq
lstm_16/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
lstm_16/zeros_1/mul/y?
lstm_16/zeros_1/mulMullstm_16/strided_slice:output:0lstm_16/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_16/zeros_1/muls
lstm_16/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
lstm_16/zeros_1/Less/y?
lstm_16/zeros_1/LessLesslstm_16/zeros_1/mul:z:0lstm_16/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_16/zeros_1/Lessw
lstm_16/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
lstm_16/zeros_1/packed/1?
lstm_16/zeros_1/packedPacklstm_16/strided_slice:output:0!lstm_16/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_16/zeros_1/packeds
lstm_16/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_16/zeros_1/Const?
lstm_16/zeros_1Filllstm_16/zeros_1/packed:output:0lstm_16/zeros_1/Const:output:0*
T0*(
_output_shapes
:??????????2
lstm_16/zeros_1?
lstm_16/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_16/transpose/perm?
lstm_16/transpose	Transposeinputslstm_16/transpose/perm:output:0*
T0*+
_output_shapes
:?????????]2
lstm_16/transposeg
lstm_16/Shape_1Shapelstm_16/transpose:y:0*
T0*
_output_shapes
:2
lstm_16/Shape_1?
lstm_16/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_16/strided_slice_1/stack?
lstm_16/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_16/strided_slice_1/stack_1?
lstm_16/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_16/strided_slice_1/stack_2?
lstm_16/strided_slice_1StridedSlicelstm_16/Shape_1:output:0&lstm_16/strided_slice_1/stack:output:0(lstm_16/strided_slice_1/stack_1:output:0(lstm_16/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_16/strided_slice_1?
#lstm_16/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2%
#lstm_16/TensorArrayV2/element_shape?
lstm_16/TensorArrayV2TensorListReserve,lstm_16/TensorArrayV2/element_shape:output:0 lstm_16/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_16/TensorArrayV2?
=lstm_16/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????]   2?
=lstm_16/TensorArrayUnstack/TensorListFromTensor/element_shape?
/lstm_16/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_16/transpose:y:0Flstm_16/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type021
/lstm_16/TensorArrayUnstack/TensorListFromTensor?
lstm_16/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_16/strided_slice_2/stack?
lstm_16/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_16/strided_slice_2/stack_1?
lstm_16/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_16/strided_slice_2/stack_2?
lstm_16/strided_slice_2StridedSlicelstm_16/transpose:y:0&lstm_16/strided_slice_2/stack:output:0(lstm_16/strided_slice_2/stack_1:output:0(lstm_16/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????]*
shrink_axis_mask2
lstm_16/strided_slice_2?
*lstm_16/lstm_cell_16/MatMul/ReadVariableOpReadVariableOp3lstm_16_lstm_cell_16_matmul_readvariableop_resource*
_output_shapes
:	]?*
dtype02,
*lstm_16/lstm_cell_16/MatMul/ReadVariableOp?
lstm_16/lstm_cell_16/MatMulMatMul lstm_16/strided_slice_2:output:02lstm_16/lstm_cell_16/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_16/lstm_cell_16/MatMul?
,lstm_16/lstm_cell_16/MatMul_1/ReadVariableOpReadVariableOp5lstm_16_lstm_cell_16_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02.
,lstm_16/lstm_cell_16/MatMul_1/ReadVariableOp?
lstm_16/lstm_cell_16/MatMul_1MatMullstm_16/zeros:output:04lstm_16/lstm_cell_16/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_16/lstm_cell_16/MatMul_1?
lstm_16/lstm_cell_16/addAddV2%lstm_16/lstm_cell_16/MatMul:product:0'lstm_16/lstm_cell_16/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
lstm_16/lstm_cell_16/add?
+lstm_16/lstm_cell_16/BiasAdd/ReadVariableOpReadVariableOp4lstm_16_lstm_cell_16_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02-
+lstm_16/lstm_cell_16/BiasAdd/ReadVariableOp?
lstm_16/lstm_cell_16/BiasAddBiasAddlstm_16/lstm_cell_16/add:z:03lstm_16/lstm_cell_16/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_16/lstm_cell_16/BiasAdd?
$lstm_16/lstm_cell_16/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2&
$lstm_16/lstm_cell_16/split/split_dim?
lstm_16/lstm_cell_16/splitSplit-lstm_16/lstm_cell_16/split/split_dim:output:0%lstm_16/lstm_cell_16/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split2
lstm_16/lstm_cell_16/split?
lstm_16/lstm_cell_16/SigmoidSigmoid#lstm_16/lstm_cell_16/split:output:0*
T0*(
_output_shapes
:??????????2
lstm_16/lstm_cell_16/Sigmoid?
lstm_16/lstm_cell_16/Sigmoid_1Sigmoid#lstm_16/lstm_cell_16/split:output:1*
T0*(
_output_shapes
:??????????2 
lstm_16/lstm_cell_16/Sigmoid_1?
lstm_16/lstm_cell_16/mulMul"lstm_16/lstm_cell_16/Sigmoid_1:y:0lstm_16/zeros_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_16/lstm_cell_16/mul?
lstm_16/lstm_cell_16/ReluRelu#lstm_16/lstm_cell_16/split:output:2*
T0*(
_output_shapes
:??????????2
lstm_16/lstm_cell_16/Relu?
lstm_16/lstm_cell_16/mul_1Mul lstm_16/lstm_cell_16/Sigmoid:y:0'lstm_16/lstm_cell_16/Relu:activations:0*
T0*(
_output_shapes
:??????????2
lstm_16/lstm_cell_16/mul_1?
lstm_16/lstm_cell_16/add_1AddV2lstm_16/lstm_cell_16/mul:z:0lstm_16/lstm_cell_16/mul_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_16/lstm_cell_16/add_1?
lstm_16/lstm_cell_16/Sigmoid_2Sigmoid#lstm_16/lstm_cell_16/split:output:3*
T0*(
_output_shapes
:??????????2 
lstm_16/lstm_cell_16/Sigmoid_2?
lstm_16/lstm_cell_16/Relu_1Relulstm_16/lstm_cell_16/add_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_16/lstm_cell_16/Relu_1?
lstm_16/lstm_cell_16/mul_2Mul"lstm_16/lstm_cell_16/Sigmoid_2:y:0)lstm_16/lstm_cell_16/Relu_1:activations:0*
T0*(
_output_shapes
:??????????2
lstm_16/lstm_cell_16/mul_2?
%lstm_16/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????  2'
%lstm_16/TensorArrayV2_1/element_shape?
lstm_16/TensorArrayV2_1TensorListReserve.lstm_16/TensorArrayV2_1/element_shape:output:0 lstm_16/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_16/TensorArrayV2_1^
lstm_16/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_16/time?
 lstm_16/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2"
 lstm_16/while/maximum_iterationsz
lstm_16/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_16/while/loop_counter?
lstm_16/whileWhile#lstm_16/while/loop_counter:output:0)lstm_16/while/maximum_iterations:output:0lstm_16/time:output:0 lstm_16/TensorArrayV2_1:handle:0lstm_16/zeros:output:0lstm_16/zeros_1:output:0 lstm_16/strided_slice_1:output:0?lstm_16/TensorArrayUnstack/TensorListFromTensor:output_handle:03lstm_16_lstm_cell_16_matmul_readvariableop_resource5lstm_16_lstm_cell_16_matmul_1_readvariableop_resource4lstm_16_lstm_cell_16_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *'
bodyR
lstm_16_while_body_32575753*'
condR
lstm_16_while_cond_32575752*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations 2
lstm_16/while?
8lstm_16/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????  2:
8lstm_16/TensorArrayV2Stack/TensorListStack/element_shape?
*lstm_16/TensorArrayV2Stack/TensorListStackTensorListStacklstm_16/while:output:3Alstm_16/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:??????????*
element_dtype02,
*lstm_16/TensorArrayV2Stack/TensorListStack?
lstm_16/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
lstm_16/strided_slice_3/stack?
lstm_16/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2!
lstm_16/strided_slice_3/stack_1?
lstm_16/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_16/strided_slice_3/stack_2?
lstm_16/strided_slice_3StridedSlice3lstm_16/TensorArrayV2Stack/TensorListStack:tensor:0&lstm_16/strided_slice_3/stack:output:0(lstm_16/strided_slice_3/stack_1:output:0(lstm_16/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
lstm_16/strided_slice_3?
lstm_16/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_16/transpose_1/perm?
lstm_16/transpose_1	Transpose3lstm_16/TensorArrayV2Stack/TensorListStack:tensor:0!lstm_16/transpose_1/perm:output:0*
T0*,
_output_shapes
:??????????2
lstm_16/transpose_1v
lstm_16/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_16/runtime?
dropout_16/IdentityIdentitylstm_16/transpose_1:y:0*
T0*,
_output_shapes
:??????????2
dropout_16/Identityj
lstm_17/ShapeShapedropout_16/Identity:output:0*
T0*
_output_shapes
:2
lstm_17/Shape?
lstm_17/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_17/strided_slice/stack?
lstm_17/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_17/strided_slice/stack_1?
lstm_17/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_17/strided_slice/stack_2?
lstm_17/strided_sliceStridedSlicelstm_17/Shape:output:0$lstm_17/strided_slice/stack:output:0&lstm_17/strided_slice/stack_1:output:0&lstm_17/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_17/strided_slicem
lstm_17/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
lstm_17/zeros/mul/y?
lstm_17/zeros/mulMullstm_17/strided_slice:output:0lstm_17/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_17/zeros/mulo
lstm_17/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
lstm_17/zeros/Less/y?
lstm_17/zeros/LessLesslstm_17/zeros/mul:z:0lstm_17/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_17/zeros/Lesss
lstm_17/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
lstm_17/zeros/packed/1?
lstm_17/zeros/packedPacklstm_17/strided_slice:output:0lstm_17/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_17/zeros/packedo
lstm_17/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_17/zeros/Const?
lstm_17/zerosFilllstm_17/zeros/packed:output:0lstm_17/zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
lstm_17/zerosq
lstm_17/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
lstm_17/zeros_1/mul/y?
lstm_17/zeros_1/mulMullstm_17/strided_slice:output:0lstm_17/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_17/zeros_1/muls
lstm_17/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
lstm_17/zeros_1/Less/y?
lstm_17/zeros_1/LessLesslstm_17/zeros_1/mul:z:0lstm_17/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_17/zeros_1/Lessw
lstm_17/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
lstm_17/zeros_1/packed/1?
lstm_17/zeros_1/packedPacklstm_17/strided_slice:output:0!lstm_17/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_17/zeros_1/packeds
lstm_17/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_17/zeros_1/Const?
lstm_17/zeros_1Filllstm_17/zeros_1/packed:output:0lstm_17/zeros_1/Const:output:0*
T0*(
_output_shapes
:??????????2
lstm_17/zeros_1?
lstm_17/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_17/transpose/perm?
lstm_17/transpose	Transposedropout_16/Identity:output:0lstm_17/transpose/perm:output:0*
T0*,
_output_shapes
:??????????2
lstm_17/transposeg
lstm_17/Shape_1Shapelstm_17/transpose:y:0*
T0*
_output_shapes
:2
lstm_17/Shape_1?
lstm_17/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_17/strided_slice_1/stack?
lstm_17/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_17/strided_slice_1/stack_1?
lstm_17/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_17/strided_slice_1/stack_2?
lstm_17/strided_slice_1StridedSlicelstm_17/Shape_1:output:0&lstm_17/strided_slice_1/stack:output:0(lstm_17/strided_slice_1/stack_1:output:0(lstm_17/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_17/strided_slice_1?
#lstm_17/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2%
#lstm_17/TensorArrayV2/element_shape?
lstm_17/TensorArrayV2TensorListReserve,lstm_17/TensorArrayV2/element_shape:output:0 lstm_17/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_17/TensorArrayV2?
=lstm_17/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????  2?
=lstm_17/TensorArrayUnstack/TensorListFromTensor/element_shape?
/lstm_17/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_17/transpose:y:0Flstm_17/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type021
/lstm_17/TensorArrayUnstack/TensorListFromTensor?
lstm_17/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_17/strided_slice_2/stack?
lstm_17/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_17/strided_slice_2/stack_1?
lstm_17/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_17/strided_slice_2/stack_2?
lstm_17/strided_slice_2StridedSlicelstm_17/transpose:y:0&lstm_17/strided_slice_2/stack:output:0(lstm_17/strided_slice_2/stack_1:output:0(lstm_17/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
lstm_17/strided_slice_2?
*lstm_17/lstm_cell_17/MatMul/ReadVariableOpReadVariableOp3lstm_17_lstm_cell_17_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02,
*lstm_17/lstm_cell_17/MatMul/ReadVariableOp?
lstm_17/lstm_cell_17/MatMulMatMul lstm_17/strided_slice_2:output:02lstm_17/lstm_cell_17/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_17/lstm_cell_17/MatMul?
,lstm_17/lstm_cell_17/MatMul_1/ReadVariableOpReadVariableOp5lstm_17_lstm_cell_17_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02.
,lstm_17/lstm_cell_17/MatMul_1/ReadVariableOp?
lstm_17/lstm_cell_17/MatMul_1MatMullstm_17/zeros:output:04lstm_17/lstm_cell_17/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_17/lstm_cell_17/MatMul_1?
lstm_17/lstm_cell_17/addAddV2%lstm_17/lstm_cell_17/MatMul:product:0'lstm_17/lstm_cell_17/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
lstm_17/lstm_cell_17/add?
+lstm_17/lstm_cell_17/BiasAdd/ReadVariableOpReadVariableOp4lstm_17_lstm_cell_17_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02-
+lstm_17/lstm_cell_17/BiasAdd/ReadVariableOp?
lstm_17/lstm_cell_17/BiasAddBiasAddlstm_17/lstm_cell_17/add:z:03lstm_17/lstm_cell_17/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_17/lstm_cell_17/BiasAdd?
$lstm_17/lstm_cell_17/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2&
$lstm_17/lstm_cell_17/split/split_dim?
lstm_17/lstm_cell_17/splitSplit-lstm_17/lstm_cell_17/split/split_dim:output:0%lstm_17/lstm_cell_17/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split2
lstm_17/lstm_cell_17/split?
lstm_17/lstm_cell_17/SigmoidSigmoid#lstm_17/lstm_cell_17/split:output:0*
T0*(
_output_shapes
:??????????2
lstm_17/lstm_cell_17/Sigmoid?
lstm_17/lstm_cell_17/Sigmoid_1Sigmoid#lstm_17/lstm_cell_17/split:output:1*
T0*(
_output_shapes
:??????????2 
lstm_17/lstm_cell_17/Sigmoid_1?
lstm_17/lstm_cell_17/mulMul"lstm_17/lstm_cell_17/Sigmoid_1:y:0lstm_17/zeros_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_17/lstm_cell_17/mul?
lstm_17/lstm_cell_17/ReluRelu#lstm_17/lstm_cell_17/split:output:2*
T0*(
_output_shapes
:??????????2
lstm_17/lstm_cell_17/Relu?
lstm_17/lstm_cell_17/mul_1Mul lstm_17/lstm_cell_17/Sigmoid:y:0'lstm_17/lstm_cell_17/Relu:activations:0*
T0*(
_output_shapes
:??????????2
lstm_17/lstm_cell_17/mul_1?
lstm_17/lstm_cell_17/add_1AddV2lstm_17/lstm_cell_17/mul:z:0lstm_17/lstm_cell_17/mul_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_17/lstm_cell_17/add_1?
lstm_17/lstm_cell_17/Sigmoid_2Sigmoid#lstm_17/lstm_cell_17/split:output:3*
T0*(
_output_shapes
:??????????2 
lstm_17/lstm_cell_17/Sigmoid_2?
lstm_17/lstm_cell_17/Relu_1Relulstm_17/lstm_cell_17/add_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_17/lstm_cell_17/Relu_1?
lstm_17/lstm_cell_17/mul_2Mul"lstm_17/lstm_cell_17/Sigmoid_2:y:0)lstm_17/lstm_cell_17/Relu_1:activations:0*
T0*(
_output_shapes
:??????????2
lstm_17/lstm_cell_17/mul_2?
%lstm_17/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2'
%lstm_17/TensorArrayV2_1/element_shape?
lstm_17/TensorArrayV2_1TensorListReserve.lstm_17/TensorArrayV2_1/element_shape:output:0 lstm_17/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_17/TensorArrayV2_1^
lstm_17/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_17/time?
 lstm_17/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2"
 lstm_17/while/maximum_iterationsz
lstm_17/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_17/while/loop_counter?
lstm_17/whileWhile#lstm_17/while/loop_counter:output:0)lstm_17/while/maximum_iterations:output:0lstm_17/time:output:0 lstm_17/TensorArrayV2_1:handle:0lstm_17/zeros:output:0lstm_17/zeros_1:output:0 lstm_17/strided_slice_1:output:0?lstm_17/TensorArrayUnstack/TensorListFromTensor:output_handle:03lstm_17_lstm_cell_17_matmul_readvariableop_resource5lstm_17_lstm_cell_17_matmul_1_readvariableop_resource4lstm_17_lstm_cell_17_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *'
bodyR
lstm_17_while_body_32575901*'
condR
lstm_17_while_cond_32575900*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations 2
lstm_17/while?
8lstm_17/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2:
8lstm_17/TensorArrayV2Stack/TensorListStack/element_shape?
*lstm_17/TensorArrayV2Stack/TensorListStackTensorListStacklstm_17/while:output:3Alstm_17/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:??????????*
element_dtype02,
*lstm_17/TensorArrayV2Stack/TensorListStack?
lstm_17/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
lstm_17/strided_slice_3/stack?
lstm_17/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2!
lstm_17/strided_slice_3/stack_1?
lstm_17/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_17/strided_slice_3/stack_2?
lstm_17/strided_slice_3StridedSlice3lstm_17/TensorArrayV2Stack/TensorListStack:tensor:0&lstm_17/strided_slice_3/stack:output:0(lstm_17/strided_slice_3/stack_1:output:0(lstm_17/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
lstm_17/strided_slice_3?
lstm_17/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_17/transpose_1/perm?
lstm_17/transpose_1	Transpose3lstm_17/TensorArrayV2Stack/TensorListStack:tensor:0!lstm_17/transpose_1/perm:output:0*
T0*,
_output_shapes
:??????????2
lstm_17/transpose_1v
lstm_17/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_17/runtime?
dropout_17/IdentityIdentitylstm_17/transpose_1:y:0*
T0*,
_output_shapes
:??????????2
dropout_17/Identity?
 dense_8/Tensordot/ReadVariableOpReadVariableOp)dense_8_tensordot_readvariableop_resource*
_output_shapes
:	?*
dtype02"
 dense_8/Tensordot/ReadVariableOpz
dense_8/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_8/Tensordot/axes?
dense_8/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_8/Tensordot/free~
dense_8/Tensordot/ShapeShapedropout_17/Identity:output:0*
T0*
_output_shapes
:2
dense_8/Tensordot/Shape?
dense_8/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_8/Tensordot/GatherV2/axis?
dense_8/Tensordot/GatherV2GatherV2 dense_8/Tensordot/Shape:output:0dense_8/Tensordot/free:output:0(dense_8/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_8/Tensordot/GatherV2?
!dense_8/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_8/Tensordot/GatherV2_1/axis?
dense_8/Tensordot/GatherV2_1GatherV2 dense_8/Tensordot/Shape:output:0dense_8/Tensordot/axes:output:0*dense_8/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_8/Tensordot/GatherV2_1|
dense_8/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_8/Tensordot/Const?
dense_8/Tensordot/ProdProd#dense_8/Tensordot/GatherV2:output:0 dense_8/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_8/Tensordot/Prod?
dense_8/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_8/Tensordot/Const_1?
dense_8/Tensordot/Prod_1Prod%dense_8/Tensordot/GatherV2_1:output:0"dense_8/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_8/Tensordot/Prod_1?
dense_8/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense_8/Tensordot/concat/axis?
dense_8/Tensordot/concatConcatV2dense_8/Tensordot/free:output:0dense_8/Tensordot/axes:output:0&dense_8/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_8/Tensordot/concat?
dense_8/Tensordot/stackPackdense_8/Tensordot/Prod:output:0!dense_8/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_8/Tensordot/stack?
dense_8/Tensordot/transpose	Transposedropout_17/Identity:output:0!dense_8/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????2
dense_8/Tensordot/transpose?
dense_8/Tensordot/ReshapeReshapedense_8/Tensordot/transpose:y:0 dense_8/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense_8/Tensordot/Reshape?
dense_8/Tensordot/MatMulMatMul"dense_8/Tensordot/Reshape:output:0(dense_8/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_8/Tensordot/MatMul?
dense_8/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
dense_8/Tensordot/Const_2?
dense_8/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_8/Tensordot/concat_1/axis?
dense_8/Tensordot/concat_1ConcatV2#dense_8/Tensordot/GatherV2:output:0"dense_8/Tensordot/Const_2:output:0(dense_8/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_8/Tensordot/concat_1?
dense_8/TensordotReshape"dense_8/Tensordot/MatMul:product:0#dense_8/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????2
dense_8/Tensordot?
dense_8/BiasAdd/ReadVariableOpReadVariableOp'dense_8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_8/BiasAdd/ReadVariableOp?
dense_8/BiasAddBiasAdddense_8/Tensordot:output:0&dense_8/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????2
dense_8/BiasAdd}
dense_8/SoftmaxSoftmaxdense_8/BiasAdd:output:0*
T0*+
_output_shapes
:?????????2
dense_8/Softmaxx
IdentityIdentitydense_8/Softmax:softmax:0^NoOp*
T0*+
_output_shapes
:?????????2

Identity?
NoOpNoOp^dense_8/BiasAdd/ReadVariableOp!^dense_8/Tensordot/ReadVariableOp,^lstm_16/lstm_cell_16/BiasAdd/ReadVariableOp+^lstm_16/lstm_cell_16/MatMul/ReadVariableOp-^lstm_16/lstm_cell_16/MatMul_1/ReadVariableOp^lstm_16/while,^lstm_17/lstm_cell_17/BiasAdd/ReadVariableOp+^lstm_17/lstm_cell_17/MatMul/ReadVariableOp-^lstm_17/lstm_cell_17/MatMul_1/ReadVariableOp^lstm_17/while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????]: : : : : : : : 2@
dense_8/BiasAdd/ReadVariableOpdense_8/BiasAdd/ReadVariableOp2D
 dense_8/Tensordot/ReadVariableOp dense_8/Tensordot/ReadVariableOp2Z
+lstm_16/lstm_cell_16/BiasAdd/ReadVariableOp+lstm_16/lstm_cell_16/BiasAdd/ReadVariableOp2X
*lstm_16/lstm_cell_16/MatMul/ReadVariableOp*lstm_16/lstm_cell_16/MatMul/ReadVariableOp2\
,lstm_16/lstm_cell_16/MatMul_1/ReadVariableOp,lstm_16/lstm_cell_16/MatMul_1/ReadVariableOp2
lstm_16/whilelstm_16/while2Z
+lstm_17/lstm_cell_17/BiasAdd/ReadVariableOp+lstm_17/lstm_cell_17/BiasAdd/ReadVariableOp2X
*lstm_17/lstm_cell_17/MatMul/ReadVariableOp*lstm_17/lstm_cell_17/MatMul/ReadVariableOp2\
,lstm_17/lstm_cell_17/MatMul_1/ReadVariableOp,lstm_17/lstm_cell_17/MatMul_1/ReadVariableOp2
lstm_17/whilelstm_17/while:S O
+
_output_shapes
:?????????]
 
_user_specified_nameinputs
?
?
/__inference_lstm_cell_16_layer_call_fn_32577884

inputs
states_0
states_1
unknown:	]?
	unknown_0:
??
	unknown_1:	?
identity

identity_1

identity_2??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:??????????:??????????:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_lstm_cell_16_layer_call_and_return_conditional_losses_325736842
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:??????????2

Identity_1?

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:??????????2

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
A:?????????]:??????????:??????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????]
 
_user_specified_nameinputs:RN
(
_output_shapes
:??????????
"
_user_specified_name
states/0:RN
(
_output_shapes
:??????????
"
_user_specified_name
states/1
?
?
while_cond_32574391
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_32574391___redundant_placeholder06
2while_while_cond_32574391___redundant_placeholder16
2while_while_cond_32574391___redundant_placeholder26
2while_while_cond_32574391___redundant_placeholder3
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
B: : : : :??????????:??????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?
g
H__inference_dropout_17_layer_call_and_return_conditional_losses_32575147

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout/Constx
dropout/MulMulinputsdropout/Const:output:0*
T0*,
_output_shapes
:??????????2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:??????????*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:??????????2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:??????????2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:??????????2
dropout/Mul_1j
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
??
?
J__inference_sequential_8_layer_call_and_return_conditional_losses_32576354

inputsF
3lstm_16_lstm_cell_16_matmul_readvariableop_resource:	]?I
5lstm_16_lstm_cell_16_matmul_1_readvariableop_resource:
??C
4lstm_16_lstm_cell_16_biasadd_readvariableop_resource:	?G
3lstm_17_lstm_cell_17_matmul_readvariableop_resource:
??I
5lstm_17_lstm_cell_17_matmul_1_readvariableop_resource:
??C
4lstm_17_lstm_cell_17_biasadd_readvariableop_resource:	?<
)dense_8_tensordot_readvariableop_resource:	?5
'dense_8_biasadd_readvariableop_resource:
identity??dense_8/BiasAdd/ReadVariableOp? dense_8/Tensordot/ReadVariableOp?+lstm_16/lstm_cell_16/BiasAdd/ReadVariableOp?*lstm_16/lstm_cell_16/MatMul/ReadVariableOp?,lstm_16/lstm_cell_16/MatMul_1/ReadVariableOp?lstm_16/while?+lstm_17/lstm_cell_17/BiasAdd/ReadVariableOp?*lstm_17/lstm_cell_17/MatMul/ReadVariableOp?,lstm_17/lstm_cell_17/MatMul_1/ReadVariableOp?lstm_17/whileT
lstm_16/ShapeShapeinputs*
T0*
_output_shapes
:2
lstm_16/Shape?
lstm_16/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_16/strided_slice/stack?
lstm_16/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_16/strided_slice/stack_1?
lstm_16/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_16/strided_slice/stack_2?
lstm_16/strided_sliceStridedSlicelstm_16/Shape:output:0$lstm_16/strided_slice/stack:output:0&lstm_16/strided_slice/stack_1:output:0&lstm_16/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_16/strided_slicem
lstm_16/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
lstm_16/zeros/mul/y?
lstm_16/zeros/mulMullstm_16/strided_slice:output:0lstm_16/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_16/zeros/mulo
lstm_16/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
lstm_16/zeros/Less/y?
lstm_16/zeros/LessLesslstm_16/zeros/mul:z:0lstm_16/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_16/zeros/Lesss
lstm_16/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
lstm_16/zeros/packed/1?
lstm_16/zeros/packedPacklstm_16/strided_slice:output:0lstm_16/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_16/zeros/packedo
lstm_16/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_16/zeros/Const?
lstm_16/zerosFilllstm_16/zeros/packed:output:0lstm_16/zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
lstm_16/zerosq
lstm_16/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
lstm_16/zeros_1/mul/y?
lstm_16/zeros_1/mulMullstm_16/strided_slice:output:0lstm_16/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_16/zeros_1/muls
lstm_16/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
lstm_16/zeros_1/Less/y?
lstm_16/zeros_1/LessLesslstm_16/zeros_1/mul:z:0lstm_16/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_16/zeros_1/Lessw
lstm_16/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
lstm_16/zeros_1/packed/1?
lstm_16/zeros_1/packedPacklstm_16/strided_slice:output:0!lstm_16/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_16/zeros_1/packeds
lstm_16/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_16/zeros_1/Const?
lstm_16/zeros_1Filllstm_16/zeros_1/packed:output:0lstm_16/zeros_1/Const:output:0*
T0*(
_output_shapes
:??????????2
lstm_16/zeros_1?
lstm_16/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_16/transpose/perm?
lstm_16/transpose	Transposeinputslstm_16/transpose/perm:output:0*
T0*+
_output_shapes
:?????????]2
lstm_16/transposeg
lstm_16/Shape_1Shapelstm_16/transpose:y:0*
T0*
_output_shapes
:2
lstm_16/Shape_1?
lstm_16/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_16/strided_slice_1/stack?
lstm_16/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_16/strided_slice_1/stack_1?
lstm_16/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_16/strided_slice_1/stack_2?
lstm_16/strided_slice_1StridedSlicelstm_16/Shape_1:output:0&lstm_16/strided_slice_1/stack:output:0(lstm_16/strided_slice_1/stack_1:output:0(lstm_16/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_16/strided_slice_1?
#lstm_16/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2%
#lstm_16/TensorArrayV2/element_shape?
lstm_16/TensorArrayV2TensorListReserve,lstm_16/TensorArrayV2/element_shape:output:0 lstm_16/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_16/TensorArrayV2?
=lstm_16/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????]   2?
=lstm_16/TensorArrayUnstack/TensorListFromTensor/element_shape?
/lstm_16/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_16/transpose:y:0Flstm_16/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type021
/lstm_16/TensorArrayUnstack/TensorListFromTensor?
lstm_16/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_16/strided_slice_2/stack?
lstm_16/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_16/strided_slice_2/stack_1?
lstm_16/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_16/strided_slice_2/stack_2?
lstm_16/strided_slice_2StridedSlicelstm_16/transpose:y:0&lstm_16/strided_slice_2/stack:output:0(lstm_16/strided_slice_2/stack_1:output:0(lstm_16/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????]*
shrink_axis_mask2
lstm_16/strided_slice_2?
*lstm_16/lstm_cell_16/MatMul/ReadVariableOpReadVariableOp3lstm_16_lstm_cell_16_matmul_readvariableop_resource*
_output_shapes
:	]?*
dtype02,
*lstm_16/lstm_cell_16/MatMul/ReadVariableOp?
lstm_16/lstm_cell_16/MatMulMatMul lstm_16/strided_slice_2:output:02lstm_16/lstm_cell_16/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_16/lstm_cell_16/MatMul?
,lstm_16/lstm_cell_16/MatMul_1/ReadVariableOpReadVariableOp5lstm_16_lstm_cell_16_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02.
,lstm_16/lstm_cell_16/MatMul_1/ReadVariableOp?
lstm_16/lstm_cell_16/MatMul_1MatMullstm_16/zeros:output:04lstm_16/lstm_cell_16/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_16/lstm_cell_16/MatMul_1?
lstm_16/lstm_cell_16/addAddV2%lstm_16/lstm_cell_16/MatMul:product:0'lstm_16/lstm_cell_16/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
lstm_16/lstm_cell_16/add?
+lstm_16/lstm_cell_16/BiasAdd/ReadVariableOpReadVariableOp4lstm_16_lstm_cell_16_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02-
+lstm_16/lstm_cell_16/BiasAdd/ReadVariableOp?
lstm_16/lstm_cell_16/BiasAddBiasAddlstm_16/lstm_cell_16/add:z:03lstm_16/lstm_cell_16/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_16/lstm_cell_16/BiasAdd?
$lstm_16/lstm_cell_16/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2&
$lstm_16/lstm_cell_16/split/split_dim?
lstm_16/lstm_cell_16/splitSplit-lstm_16/lstm_cell_16/split/split_dim:output:0%lstm_16/lstm_cell_16/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split2
lstm_16/lstm_cell_16/split?
lstm_16/lstm_cell_16/SigmoidSigmoid#lstm_16/lstm_cell_16/split:output:0*
T0*(
_output_shapes
:??????????2
lstm_16/lstm_cell_16/Sigmoid?
lstm_16/lstm_cell_16/Sigmoid_1Sigmoid#lstm_16/lstm_cell_16/split:output:1*
T0*(
_output_shapes
:??????????2 
lstm_16/lstm_cell_16/Sigmoid_1?
lstm_16/lstm_cell_16/mulMul"lstm_16/lstm_cell_16/Sigmoid_1:y:0lstm_16/zeros_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_16/lstm_cell_16/mul?
lstm_16/lstm_cell_16/ReluRelu#lstm_16/lstm_cell_16/split:output:2*
T0*(
_output_shapes
:??????????2
lstm_16/lstm_cell_16/Relu?
lstm_16/lstm_cell_16/mul_1Mul lstm_16/lstm_cell_16/Sigmoid:y:0'lstm_16/lstm_cell_16/Relu:activations:0*
T0*(
_output_shapes
:??????????2
lstm_16/lstm_cell_16/mul_1?
lstm_16/lstm_cell_16/add_1AddV2lstm_16/lstm_cell_16/mul:z:0lstm_16/lstm_cell_16/mul_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_16/lstm_cell_16/add_1?
lstm_16/lstm_cell_16/Sigmoid_2Sigmoid#lstm_16/lstm_cell_16/split:output:3*
T0*(
_output_shapes
:??????????2 
lstm_16/lstm_cell_16/Sigmoid_2?
lstm_16/lstm_cell_16/Relu_1Relulstm_16/lstm_cell_16/add_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_16/lstm_cell_16/Relu_1?
lstm_16/lstm_cell_16/mul_2Mul"lstm_16/lstm_cell_16/Sigmoid_2:y:0)lstm_16/lstm_cell_16/Relu_1:activations:0*
T0*(
_output_shapes
:??????????2
lstm_16/lstm_cell_16/mul_2?
%lstm_16/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????  2'
%lstm_16/TensorArrayV2_1/element_shape?
lstm_16/TensorArrayV2_1TensorListReserve.lstm_16/TensorArrayV2_1/element_shape:output:0 lstm_16/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_16/TensorArrayV2_1^
lstm_16/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_16/time?
 lstm_16/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2"
 lstm_16/while/maximum_iterationsz
lstm_16/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_16/while/loop_counter?
lstm_16/whileWhile#lstm_16/while/loop_counter:output:0)lstm_16/while/maximum_iterations:output:0lstm_16/time:output:0 lstm_16/TensorArrayV2_1:handle:0lstm_16/zeros:output:0lstm_16/zeros_1:output:0 lstm_16/strided_slice_1:output:0?lstm_16/TensorArrayUnstack/TensorListFromTensor:output_handle:03lstm_16_lstm_cell_16_matmul_readvariableop_resource5lstm_16_lstm_cell_16_matmul_1_readvariableop_resource4lstm_16_lstm_cell_16_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *'
bodyR
lstm_16_while_body_32576080*'
condR
lstm_16_while_cond_32576079*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations 2
lstm_16/while?
8lstm_16/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????  2:
8lstm_16/TensorArrayV2Stack/TensorListStack/element_shape?
*lstm_16/TensorArrayV2Stack/TensorListStackTensorListStacklstm_16/while:output:3Alstm_16/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:??????????*
element_dtype02,
*lstm_16/TensorArrayV2Stack/TensorListStack?
lstm_16/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
lstm_16/strided_slice_3/stack?
lstm_16/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2!
lstm_16/strided_slice_3/stack_1?
lstm_16/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_16/strided_slice_3/stack_2?
lstm_16/strided_slice_3StridedSlice3lstm_16/TensorArrayV2Stack/TensorListStack:tensor:0&lstm_16/strided_slice_3/stack:output:0(lstm_16/strided_slice_3/stack_1:output:0(lstm_16/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
lstm_16/strided_slice_3?
lstm_16/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_16/transpose_1/perm?
lstm_16/transpose_1	Transpose3lstm_16/TensorArrayV2Stack/TensorListStack:tensor:0!lstm_16/transpose_1/perm:output:0*
T0*,
_output_shapes
:??????????2
lstm_16/transpose_1v
lstm_16/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_16/runtimey
dropout_16/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2
dropout_16/dropout/Const?
dropout_16/dropout/MulMullstm_16/transpose_1:y:0!dropout_16/dropout/Const:output:0*
T0*,
_output_shapes
:??????????2
dropout_16/dropout/Mul{
dropout_16/dropout/ShapeShapelstm_16/transpose_1:y:0*
T0*
_output_shapes
:2
dropout_16/dropout/Shape?
/dropout_16/dropout/random_uniform/RandomUniformRandomUniform!dropout_16/dropout/Shape:output:0*
T0*,
_output_shapes
:??????????*
dtype021
/dropout_16/dropout/random_uniform/RandomUniform?
!dropout_16/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?>2#
!dropout_16/dropout/GreaterEqual/y?
dropout_16/dropout/GreaterEqualGreaterEqual8dropout_16/dropout/random_uniform/RandomUniform:output:0*dropout_16/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:??????????2!
dropout_16/dropout/GreaterEqual?
dropout_16/dropout/CastCast#dropout_16/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:??????????2
dropout_16/dropout/Cast?
dropout_16/dropout/Mul_1Muldropout_16/dropout/Mul:z:0dropout_16/dropout/Cast:y:0*
T0*,
_output_shapes
:??????????2
dropout_16/dropout/Mul_1j
lstm_17/ShapeShapedropout_16/dropout/Mul_1:z:0*
T0*
_output_shapes
:2
lstm_17/Shape?
lstm_17/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_17/strided_slice/stack?
lstm_17/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_17/strided_slice/stack_1?
lstm_17/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_17/strided_slice/stack_2?
lstm_17/strided_sliceStridedSlicelstm_17/Shape:output:0$lstm_17/strided_slice/stack:output:0&lstm_17/strided_slice/stack_1:output:0&lstm_17/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_17/strided_slicem
lstm_17/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
lstm_17/zeros/mul/y?
lstm_17/zeros/mulMullstm_17/strided_slice:output:0lstm_17/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_17/zeros/mulo
lstm_17/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
lstm_17/zeros/Less/y?
lstm_17/zeros/LessLesslstm_17/zeros/mul:z:0lstm_17/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_17/zeros/Lesss
lstm_17/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
lstm_17/zeros/packed/1?
lstm_17/zeros/packedPacklstm_17/strided_slice:output:0lstm_17/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_17/zeros/packedo
lstm_17/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_17/zeros/Const?
lstm_17/zerosFilllstm_17/zeros/packed:output:0lstm_17/zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
lstm_17/zerosq
lstm_17/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
lstm_17/zeros_1/mul/y?
lstm_17/zeros_1/mulMullstm_17/strided_slice:output:0lstm_17/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_17/zeros_1/muls
lstm_17/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
lstm_17/zeros_1/Less/y?
lstm_17/zeros_1/LessLesslstm_17/zeros_1/mul:z:0lstm_17/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_17/zeros_1/Lessw
lstm_17/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
lstm_17/zeros_1/packed/1?
lstm_17/zeros_1/packedPacklstm_17/strided_slice:output:0!lstm_17/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_17/zeros_1/packeds
lstm_17/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_17/zeros_1/Const?
lstm_17/zeros_1Filllstm_17/zeros_1/packed:output:0lstm_17/zeros_1/Const:output:0*
T0*(
_output_shapes
:??????????2
lstm_17/zeros_1?
lstm_17/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_17/transpose/perm?
lstm_17/transpose	Transposedropout_16/dropout/Mul_1:z:0lstm_17/transpose/perm:output:0*
T0*,
_output_shapes
:??????????2
lstm_17/transposeg
lstm_17/Shape_1Shapelstm_17/transpose:y:0*
T0*
_output_shapes
:2
lstm_17/Shape_1?
lstm_17/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_17/strided_slice_1/stack?
lstm_17/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_17/strided_slice_1/stack_1?
lstm_17/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_17/strided_slice_1/stack_2?
lstm_17/strided_slice_1StridedSlicelstm_17/Shape_1:output:0&lstm_17/strided_slice_1/stack:output:0(lstm_17/strided_slice_1/stack_1:output:0(lstm_17/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_17/strided_slice_1?
#lstm_17/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2%
#lstm_17/TensorArrayV2/element_shape?
lstm_17/TensorArrayV2TensorListReserve,lstm_17/TensorArrayV2/element_shape:output:0 lstm_17/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_17/TensorArrayV2?
=lstm_17/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????  2?
=lstm_17/TensorArrayUnstack/TensorListFromTensor/element_shape?
/lstm_17/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_17/transpose:y:0Flstm_17/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type021
/lstm_17/TensorArrayUnstack/TensorListFromTensor?
lstm_17/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_17/strided_slice_2/stack?
lstm_17/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_17/strided_slice_2/stack_1?
lstm_17/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_17/strided_slice_2/stack_2?
lstm_17/strided_slice_2StridedSlicelstm_17/transpose:y:0&lstm_17/strided_slice_2/stack:output:0(lstm_17/strided_slice_2/stack_1:output:0(lstm_17/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
lstm_17/strided_slice_2?
*lstm_17/lstm_cell_17/MatMul/ReadVariableOpReadVariableOp3lstm_17_lstm_cell_17_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02,
*lstm_17/lstm_cell_17/MatMul/ReadVariableOp?
lstm_17/lstm_cell_17/MatMulMatMul lstm_17/strided_slice_2:output:02lstm_17/lstm_cell_17/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_17/lstm_cell_17/MatMul?
,lstm_17/lstm_cell_17/MatMul_1/ReadVariableOpReadVariableOp5lstm_17_lstm_cell_17_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02.
,lstm_17/lstm_cell_17/MatMul_1/ReadVariableOp?
lstm_17/lstm_cell_17/MatMul_1MatMullstm_17/zeros:output:04lstm_17/lstm_cell_17/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_17/lstm_cell_17/MatMul_1?
lstm_17/lstm_cell_17/addAddV2%lstm_17/lstm_cell_17/MatMul:product:0'lstm_17/lstm_cell_17/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
lstm_17/lstm_cell_17/add?
+lstm_17/lstm_cell_17/BiasAdd/ReadVariableOpReadVariableOp4lstm_17_lstm_cell_17_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02-
+lstm_17/lstm_cell_17/BiasAdd/ReadVariableOp?
lstm_17/lstm_cell_17/BiasAddBiasAddlstm_17/lstm_cell_17/add:z:03lstm_17/lstm_cell_17/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_17/lstm_cell_17/BiasAdd?
$lstm_17/lstm_cell_17/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2&
$lstm_17/lstm_cell_17/split/split_dim?
lstm_17/lstm_cell_17/splitSplit-lstm_17/lstm_cell_17/split/split_dim:output:0%lstm_17/lstm_cell_17/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split2
lstm_17/lstm_cell_17/split?
lstm_17/lstm_cell_17/SigmoidSigmoid#lstm_17/lstm_cell_17/split:output:0*
T0*(
_output_shapes
:??????????2
lstm_17/lstm_cell_17/Sigmoid?
lstm_17/lstm_cell_17/Sigmoid_1Sigmoid#lstm_17/lstm_cell_17/split:output:1*
T0*(
_output_shapes
:??????????2 
lstm_17/lstm_cell_17/Sigmoid_1?
lstm_17/lstm_cell_17/mulMul"lstm_17/lstm_cell_17/Sigmoid_1:y:0lstm_17/zeros_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_17/lstm_cell_17/mul?
lstm_17/lstm_cell_17/ReluRelu#lstm_17/lstm_cell_17/split:output:2*
T0*(
_output_shapes
:??????????2
lstm_17/lstm_cell_17/Relu?
lstm_17/lstm_cell_17/mul_1Mul lstm_17/lstm_cell_17/Sigmoid:y:0'lstm_17/lstm_cell_17/Relu:activations:0*
T0*(
_output_shapes
:??????????2
lstm_17/lstm_cell_17/mul_1?
lstm_17/lstm_cell_17/add_1AddV2lstm_17/lstm_cell_17/mul:z:0lstm_17/lstm_cell_17/mul_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_17/lstm_cell_17/add_1?
lstm_17/lstm_cell_17/Sigmoid_2Sigmoid#lstm_17/lstm_cell_17/split:output:3*
T0*(
_output_shapes
:??????????2 
lstm_17/lstm_cell_17/Sigmoid_2?
lstm_17/lstm_cell_17/Relu_1Relulstm_17/lstm_cell_17/add_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_17/lstm_cell_17/Relu_1?
lstm_17/lstm_cell_17/mul_2Mul"lstm_17/lstm_cell_17/Sigmoid_2:y:0)lstm_17/lstm_cell_17/Relu_1:activations:0*
T0*(
_output_shapes
:??????????2
lstm_17/lstm_cell_17/mul_2?
%lstm_17/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2'
%lstm_17/TensorArrayV2_1/element_shape?
lstm_17/TensorArrayV2_1TensorListReserve.lstm_17/TensorArrayV2_1/element_shape:output:0 lstm_17/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_17/TensorArrayV2_1^
lstm_17/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_17/time?
 lstm_17/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2"
 lstm_17/while/maximum_iterationsz
lstm_17/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_17/while/loop_counter?
lstm_17/whileWhile#lstm_17/while/loop_counter:output:0)lstm_17/while/maximum_iterations:output:0lstm_17/time:output:0 lstm_17/TensorArrayV2_1:handle:0lstm_17/zeros:output:0lstm_17/zeros_1:output:0 lstm_17/strided_slice_1:output:0?lstm_17/TensorArrayUnstack/TensorListFromTensor:output_handle:03lstm_17_lstm_cell_17_matmul_readvariableop_resource5lstm_17_lstm_cell_17_matmul_1_readvariableop_resource4lstm_17_lstm_cell_17_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *'
bodyR
lstm_17_while_body_32576235*'
condR
lstm_17_while_cond_32576234*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations 2
lstm_17/while?
8lstm_17/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2:
8lstm_17/TensorArrayV2Stack/TensorListStack/element_shape?
*lstm_17/TensorArrayV2Stack/TensorListStackTensorListStacklstm_17/while:output:3Alstm_17/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:??????????*
element_dtype02,
*lstm_17/TensorArrayV2Stack/TensorListStack?
lstm_17/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
lstm_17/strided_slice_3/stack?
lstm_17/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2!
lstm_17/strided_slice_3/stack_1?
lstm_17/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_17/strided_slice_3/stack_2?
lstm_17/strided_slice_3StridedSlice3lstm_17/TensorArrayV2Stack/TensorListStack:tensor:0&lstm_17/strided_slice_3/stack:output:0(lstm_17/strided_slice_3/stack_1:output:0(lstm_17/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
lstm_17/strided_slice_3?
lstm_17/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_17/transpose_1/perm?
lstm_17/transpose_1	Transpose3lstm_17/TensorArrayV2Stack/TensorListStack:tensor:0!lstm_17/transpose_1/perm:output:0*
T0*,
_output_shapes
:??????????2
lstm_17/transpose_1v
lstm_17/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_17/runtimey
dropout_17/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout_17/dropout/Const?
dropout_17/dropout/MulMullstm_17/transpose_1:y:0!dropout_17/dropout/Const:output:0*
T0*,
_output_shapes
:??????????2
dropout_17/dropout/Mul{
dropout_17/dropout/ShapeShapelstm_17/transpose_1:y:0*
T0*
_output_shapes
:2
dropout_17/dropout/Shape?
/dropout_17/dropout/random_uniform/RandomUniformRandomUniform!dropout_17/dropout/Shape:output:0*
T0*,
_output_shapes
:??????????*
dtype021
/dropout_17/dropout/random_uniform/RandomUniform?
!dropout_17/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2#
!dropout_17/dropout/GreaterEqual/y?
dropout_17/dropout/GreaterEqualGreaterEqual8dropout_17/dropout/random_uniform/RandomUniform:output:0*dropout_17/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:??????????2!
dropout_17/dropout/GreaterEqual?
dropout_17/dropout/CastCast#dropout_17/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:??????????2
dropout_17/dropout/Cast?
dropout_17/dropout/Mul_1Muldropout_17/dropout/Mul:z:0dropout_17/dropout/Cast:y:0*
T0*,
_output_shapes
:??????????2
dropout_17/dropout/Mul_1?
 dense_8/Tensordot/ReadVariableOpReadVariableOp)dense_8_tensordot_readvariableop_resource*
_output_shapes
:	?*
dtype02"
 dense_8/Tensordot/ReadVariableOpz
dense_8/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_8/Tensordot/axes?
dense_8/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_8/Tensordot/free~
dense_8/Tensordot/ShapeShapedropout_17/dropout/Mul_1:z:0*
T0*
_output_shapes
:2
dense_8/Tensordot/Shape?
dense_8/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_8/Tensordot/GatherV2/axis?
dense_8/Tensordot/GatherV2GatherV2 dense_8/Tensordot/Shape:output:0dense_8/Tensordot/free:output:0(dense_8/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_8/Tensordot/GatherV2?
!dense_8/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_8/Tensordot/GatherV2_1/axis?
dense_8/Tensordot/GatherV2_1GatherV2 dense_8/Tensordot/Shape:output:0dense_8/Tensordot/axes:output:0*dense_8/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_8/Tensordot/GatherV2_1|
dense_8/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_8/Tensordot/Const?
dense_8/Tensordot/ProdProd#dense_8/Tensordot/GatherV2:output:0 dense_8/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_8/Tensordot/Prod?
dense_8/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_8/Tensordot/Const_1?
dense_8/Tensordot/Prod_1Prod%dense_8/Tensordot/GatherV2_1:output:0"dense_8/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_8/Tensordot/Prod_1?
dense_8/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense_8/Tensordot/concat/axis?
dense_8/Tensordot/concatConcatV2dense_8/Tensordot/free:output:0dense_8/Tensordot/axes:output:0&dense_8/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_8/Tensordot/concat?
dense_8/Tensordot/stackPackdense_8/Tensordot/Prod:output:0!dense_8/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_8/Tensordot/stack?
dense_8/Tensordot/transpose	Transposedropout_17/dropout/Mul_1:z:0!dense_8/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????2
dense_8/Tensordot/transpose?
dense_8/Tensordot/ReshapeReshapedense_8/Tensordot/transpose:y:0 dense_8/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense_8/Tensordot/Reshape?
dense_8/Tensordot/MatMulMatMul"dense_8/Tensordot/Reshape:output:0(dense_8/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_8/Tensordot/MatMul?
dense_8/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
dense_8/Tensordot/Const_2?
dense_8/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_8/Tensordot/concat_1/axis?
dense_8/Tensordot/concat_1ConcatV2#dense_8/Tensordot/GatherV2:output:0"dense_8/Tensordot/Const_2:output:0(dense_8/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_8/Tensordot/concat_1?
dense_8/TensordotReshape"dense_8/Tensordot/MatMul:product:0#dense_8/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????2
dense_8/Tensordot?
dense_8/BiasAdd/ReadVariableOpReadVariableOp'dense_8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_8/BiasAdd/ReadVariableOp?
dense_8/BiasAddBiasAdddense_8/Tensordot:output:0&dense_8/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????2
dense_8/BiasAdd}
dense_8/SoftmaxSoftmaxdense_8/BiasAdd:output:0*
T0*+
_output_shapes
:?????????2
dense_8/Softmaxx
IdentityIdentitydense_8/Softmax:softmax:0^NoOp*
T0*+
_output_shapes
:?????????2

Identity?
NoOpNoOp^dense_8/BiasAdd/ReadVariableOp!^dense_8/Tensordot/ReadVariableOp,^lstm_16/lstm_cell_16/BiasAdd/ReadVariableOp+^lstm_16/lstm_cell_16/MatMul/ReadVariableOp-^lstm_16/lstm_cell_16/MatMul_1/ReadVariableOp^lstm_16/while,^lstm_17/lstm_cell_17/BiasAdd/ReadVariableOp+^lstm_17/lstm_cell_17/MatMul/ReadVariableOp-^lstm_17/lstm_cell_17/MatMul_1/ReadVariableOp^lstm_17/while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????]: : : : : : : : 2@
dense_8/BiasAdd/ReadVariableOpdense_8/BiasAdd/ReadVariableOp2D
 dense_8/Tensordot/ReadVariableOp dense_8/Tensordot/ReadVariableOp2Z
+lstm_16/lstm_cell_16/BiasAdd/ReadVariableOp+lstm_16/lstm_cell_16/BiasAdd/ReadVariableOp2X
*lstm_16/lstm_cell_16/MatMul/ReadVariableOp*lstm_16/lstm_cell_16/MatMul/ReadVariableOp2\
,lstm_16/lstm_cell_16/MatMul_1/ReadVariableOp,lstm_16/lstm_cell_16/MatMul_1/ReadVariableOp2
lstm_16/whilelstm_16/while2Z
+lstm_17/lstm_cell_17/BiasAdd/ReadVariableOp+lstm_17/lstm_cell_17/BiasAdd/ReadVariableOp2X
*lstm_17/lstm_cell_17/MatMul/ReadVariableOp*lstm_17/lstm_cell_17/MatMul/ReadVariableOp2\
,lstm_17/lstm_cell_17/MatMul_1/ReadVariableOp,lstm_17/lstm_cell_17/MatMul_1/ReadVariableOp2
lstm_17/whilelstm_17/while:S O
+
_output_shapes
:?????????]
 
_user_specified_nameinputs
?
?
J__inference_sequential_8_layer_call_and_return_conditional_losses_32575657
lstm_16_input#
lstm_16_32575635:	]?$
lstm_16_32575637:
??
lstm_16_32575639:	?$
lstm_17_32575643:
??$
lstm_17_32575645:
??
lstm_17_32575647:	?#
dense_8_32575651:	?
dense_8_32575653:
identity??dense_8/StatefulPartitionedCall?"dropout_16/StatefulPartitionedCall?"dropout_17/StatefulPartitionedCall?lstm_16/StatefulPartitionedCall?lstm_17/StatefulPartitionedCall?
lstm_16/StatefulPartitionedCallStatefulPartitionedCalllstm_16_inputlstm_16_32575635lstm_16_32575637lstm_16_32575639*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_lstm_16_layer_call_and_return_conditional_losses_325755102!
lstm_16/StatefulPartitionedCall?
"dropout_16/StatefulPartitionedCallStatefulPartitionedCall(lstm_16/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_dropout_16_layer_call_and_return_conditional_losses_325753432$
"dropout_16/StatefulPartitionedCall?
lstm_17/StatefulPartitionedCallStatefulPartitionedCall+dropout_16/StatefulPartitionedCall:output:0lstm_17_32575643lstm_17_32575645lstm_17_32575647*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_lstm_17_layer_call_and_return_conditional_losses_325753142!
lstm_17/StatefulPartitionedCall?
"dropout_17/StatefulPartitionedCallStatefulPartitionedCall(lstm_17/StatefulPartitionedCall:output:0#^dropout_16/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_dropout_17_layer_call_and_return_conditional_losses_325751472$
"dropout_17/StatefulPartitionedCall?
dense_8/StatefulPartitionedCallStatefulPartitionedCall+dropout_17/StatefulPartitionedCall:output:0dense_8_32575651dense_8_32575653*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_8_layer_call_and_return_conditional_losses_325750912!
dense_8/StatefulPartitionedCall?
IdentityIdentity(dense_8/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????2

Identity?
NoOpNoOp ^dense_8/StatefulPartitionedCall#^dropout_16/StatefulPartitionedCall#^dropout_17/StatefulPartitionedCall ^lstm_16/StatefulPartitionedCall ^lstm_17/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????]: : : : : : : : 2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall2H
"dropout_16/StatefulPartitionedCall"dropout_16/StatefulPartitionedCall2H
"dropout_17/StatefulPartitionedCall"dropout_17/StatefulPartitionedCall2B
lstm_16/StatefulPartitionedCalllstm_16/StatefulPartitionedCall2B
lstm_17/StatefulPartitionedCalllstm_17/StatefulPartitionedCall:Z V
+
_output_shapes
:?????????]
'
_user_specified_namelstm_16_input
?
?
(sequential_8_lstm_16_while_cond_32573202F
Bsequential_8_lstm_16_while_sequential_8_lstm_16_while_loop_counterL
Hsequential_8_lstm_16_while_sequential_8_lstm_16_while_maximum_iterations*
&sequential_8_lstm_16_while_placeholder,
(sequential_8_lstm_16_while_placeholder_1,
(sequential_8_lstm_16_while_placeholder_2,
(sequential_8_lstm_16_while_placeholder_3H
Dsequential_8_lstm_16_while_less_sequential_8_lstm_16_strided_slice_1`
\sequential_8_lstm_16_while_sequential_8_lstm_16_while_cond_32573202___redundant_placeholder0`
\sequential_8_lstm_16_while_sequential_8_lstm_16_while_cond_32573202___redundant_placeholder1`
\sequential_8_lstm_16_while_sequential_8_lstm_16_while_cond_32573202___redundant_placeholder2`
\sequential_8_lstm_16_while_sequential_8_lstm_16_while_cond_32573202___redundant_placeholder3'
#sequential_8_lstm_16_while_identity
?
sequential_8/lstm_16/while/LessLess&sequential_8_lstm_16_while_placeholderDsequential_8_lstm_16_while_less_sequential_8_lstm_16_strided_slice_1*
T0*
_output_shapes
: 2!
sequential_8/lstm_16/while/Less?
#sequential_8/lstm_16/while/IdentityIdentity#sequential_8/lstm_16/while/Less:z:0*
T0
*
_output_shapes
: 2%
#sequential_8/lstm_16/while/Identity"S
#sequential_8_lstm_16_while_identity,sequential_8/lstm_16/while/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :??????????:??????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?&
?
while_body_32573762
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_00
while_lstm_cell_16_32573786_0:	]?1
while_lstm_cell_16_32573788_0:
??,
while_lstm_cell_16_32573790_0:	?
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor.
while_lstm_cell_16_32573786:	]?/
while_lstm_cell_16_32573788:
??*
while_lstm_cell_16_32573790:	???*while/lstm_cell_16/StatefulPartitionedCall?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????]   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????]*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
*while/lstm_cell_16/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_16_32573786_0while_lstm_cell_16_32573788_0while_lstm_cell_16_32573790_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:??????????:??????????:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_lstm_cell_16_layer_call_and_return_conditional_losses_325736842,
*while/lstm_cell_16/StatefulPartitionedCall?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder3while/lstm_cell_16/StatefulPartitionedCall:output:0*
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
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identity3while/lstm_cell_16/StatefulPartitionedCall:output:1^while/NoOp*
T0*(
_output_shapes
:??????????2
while/Identity_4?
while/Identity_5Identity3while/lstm_cell_16/StatefulPartitionedCall:output:2^while/NoOp*
T0*(
_output_shapes
:??????????2
while/Identity_5?

while/NoOpNoOp+^while/lstm_cell_16/StatefulPartitionedCall*"
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
while_lstm_cell_16_32573786while_lstm_cell_16_32573786_0"<
while_lstm_cell_16_32573788while_lstm_cell_16_32573788_0"<
while_lstm_cell_16_32573790while_lstm_cell_16_32573790_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :??????????:??????????: : : : : 2X
*while/lstm_cell_16/StatefulPartitionedCall*while/lstm_cell_16/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
ƒ
?
$__inference__traced_restore_32578213
file_prefix2
assignvariableop_dense_8_kernel:	?-
assignvariableop_1_dense_8_bias:&
assignvariableop_2_adam_iter:	 (
assignvariableop_3_adam_beta_1: (
assignvariableop_4_adam_beta_2: '
assignvariableop_5_adam_decay: /
%assignvariableop_6_adam_learning_rate: A
.assignvariableop_7_lstm_16_lstm_cell_16_kernel:	]?L
8assignvariableop_8_lstm_16_lstm_cell_16_recurrent_kernel:
??;
,assignvariableop_9_lstm_16_lstm_cell_16_bias:	?C
/assignvariableop_10_lstm_17_lstm_cell_17_kernel:
??M
9assignvariableop_11_lstm_17_lstm_cell_17_recurrent_kernel:
??<
-assignvariableop_12_lstm_17_lstm_cell_17_bias:	?#
assignvariableop_13_total: #
assignvariableop_14_count: %
assignvariableop_15_total_1: %
assignvariableop_16_count_1: <
)assignvariableop_17_adam_dense_8_kernel_m:	?5
'assignvariableop_18_adam_dense_8_bias_m:I
6assignvariableop_19_adam_lstm_16_lstm_cell_16_kernel_m:	]?T
@assignvariableop_20_adam_lstm_16_lstm_cell_16_recurrent_kernel_m:
??C
4assignvariableop_21_adam_lstm_16_lstm_cell_16_bias_m:	?J
6assignvariableop_22_adam_lstm_17_lstm_cell_17_kernel_m:
??T
@assignvariableop_23_adam_lstm_17_lstm_cell_17_recurrent_kernel_m:
??C
4assignvariableop_24_adam_lstm_17_lstm_cell_17_bias_m:	?<
)assignvariableop_25_adam_dense_8_kernel_v:	?5
'assignvariableop_26_adam_dense_8_bias_v:I
6assignvariableop_27_adam_lstm_16_lstm_cell_16_kernel_v:	]?T
@assignvariableop_28_adam_lstm_16_lstm_cell_16_recurrent_kernel_v:
??C
4assignvariableop_29_adam_lstm_16_lstm_cell_16_bias_v:	?J
6assignvariableop_30_adam_lstm_17_lstm_cell_17_kernel_v:
??T
@assignvariableop_31_adam_lstm_17_lstm_cell_17_recurrent_kernel_v:
??C
4assignvariableop_32_adam_lstm_17_lstm_cell_17_bias_v:	?
identity_34??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:"*
dtype0*?
value?B?"B6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:"*
dtype0*W
valueNBL"B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?::::::::::::::::::::::::::::::::::*0
dtypes&
$2"	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOpassignvariableop_dense_8_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_8_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOpassignvariableop_2_adam_iterIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOpassignvariableop_3_adam_beta_1Identity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOpassignvariableop_4_adam_beta_2Identity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOpassignvariableop_5_adam_decayIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp%assignvariableop_6_adam_learning_rateIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp.assignvariableop_7_lstm_16_lstm_cell_16_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp8assignvariableop_8_lstm_16_lstm_cell_16_recurrent_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp,assignvariableop_9_lstm_16_lstm_cell_16_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp/assignvariableop_10_lstm_17_lstm_cell_17_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOp9assignvariableop_11_lstm_17_lstm_cell_17_recurrent_kernelIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOp-assignvariableop_12_lstm_17_lstm_cell_17_biasIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOpassignvariableop_13_totalIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOpassignvariableop_14_countIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOpassignvariableop_15_total_1Identity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOpassignvariableop_16_count_1Identity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOp)assignvariableop_17_adam_dense_8_kernel_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOp'assignvariableop_18_adam_dense_8_bias_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOp6assignvariableop_19_adam_lstm_16_lstm_cell_16_kernel_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOp@assignvariableop_20_adam_lstm_16_lstm_cell_16_recurrent_kernel_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOp4assignvariableop_21_adam_lstm_16_lstm_cell_16_bias_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOp6assignvariableop_22_adam_lstm_17_lstm_cell_17_kernel_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOp@assignvariableop_23_adam_lstm_17_lstm_cell_17_recurrent_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOp4assignvariableop_24_adam_lstm_17_lstm_cell_17_bias_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOp)assignvariableop_25_adam_dense_8_kernel_vIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOp'assignvariableop_26_adam_dense_8_bias_vIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOp6assignvariableop_27_adam_lstm_16_lstm_cell_16_kernel_vIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOp@assignvariableop_28_adam_lstm_16_lstm_cell_16_recurrent_kernel_vIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOp4assignvariableop_29_adam_lstm_16_lstm_cell_16_bias_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOp6assignvariableop_30_adam_lstm_17_lstm_cell_17_kernel_vIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31?
AssignVariableOp_31AssignVariableOp@assignvariableop_31_adam_lstm_17_lstm_cell_17_recurrent_kernel_vIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32?
AssignVariableOp_32AssignVariableOp4assignvariableop_32_adam_lstm_17_lstm_cell_17_bias_vIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_329
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_33Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_33f
Identity_34IdentityIdentity_33:output:0^NoOp_1*
T0*
_output_shapes
: 2
Identity_34?
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
?
?
while_cond_32573761
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_32573761___redundant_placeholder06
2while_while_cond_32573761___redundant_placeholder16
2while_while_cond_32573761___redundant_placeholder26
2while_while_cond_32573761___redundant_placeholder3
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
B: : : : :??????????:??????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?
?
J__inference_lstm_cell_17_layer_call_and_return_conditional_losses_32577916

inputs
states_0
states_12
matmul_readvariableop_resource:
??4
 matmul_1_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity

identity_1

identity_2??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?MatMul_1/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul_1/ReadVariableOp|
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
add?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split2
split`
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:??????????2	
Sigmoidd
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:??????????2
	Sigmoid_1]
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:??????????2
mulW
ReluRelusplit:output:2*
T0*(
_output_shapes
:??????????2
Relui
mul_1MulSigmoid:y:0Relu:activations:0*
T0*(
_output_shapes
:??????????2
mul_1^
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:??????????2
add_1d
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:??????????2
	Sigmoid_2V
Relu_1Relu	add_1:z:0*
T0*(
_output_shapes
:??????????2
Relu_1m
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*(
_output_shapes
:??????????2
mul_2e
IdentityIdentity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:??????????2

Identityi

Identity_1Identity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity_1i

Identity_2Identity	add_1:z:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity_2?
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
B:??????????:??????????:??????????: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:RN
(
_output_shapes
:??????????
"
_user_specified_name
states/0:RN
(
_output_shapes
:??????????
"
_user_specified_name
states/1
?
?
while_cond_32577288
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_32577288___redundant_placeholder06
2while_while_cond_32577288___redundant_placeholder16
2while_while_cond_32577288___redundant_placeholder26
2while_while_cond_32577288___redundant_placeholder3
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
B: : : : :??????????:??????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?
?
while_cond_32577439
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_32577439___redundant_placeholder06
2while_while_cond_32577439___redundant_placeholder16
2while_while_cond_32577439___redundant_placeholder26
2while_while_cond_32577439___redundant_placeholder3
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
B: : : : :??????????:??????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?&
?
while_body_32574392
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_01
while_lstm_cell_17_32574416_0:
??1
while_lstm_cell_17_32574418_0:
??,
while_lstm_cell_17_32574420_0:	?
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor/
while_lstm_cell_17_32574416:
??/
while_lstm_cell_17_32574418:
??*
while_lstm_cell_17_32574420:	???*while/lstm_cell_17/StatefulPartitionedCall?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????  29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:??????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
*while/lstm_cell_17/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_17_32574416_0while_lstm_cell_17_32574418_0while_lstm_cell_17_32574420_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:??????????:??????????:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_lstm_cell_17_layer_call_and_return_conditional_losses_325743142,
*while/lstm_cell_17/StatefulPartitionedCall?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder3while/lstm_cell_17/StatefulPartitionedCall:output:0*
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
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identity3while/lstm_cell_17/StatefulPartitionedCall:output:1^while/NoOp*
T0*(
_output_shapes
:??????????2
while/Identity_4?
while/Identity_5Identity3while/lstm_cell_17/StatefulPartitionedCall:output:2^while/NoOp*
T0*(
_output_shapes
:??????????2
while/Identity_5?

while/NoOpNoOp+^while/lstm_cell_17/StatefulPartitionedCall*"
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
while_lstm_cell_17_32574416while_lstm_cell_17_32574416_0"<
while_lstm_cell_17_32574418while_lstm_cell_17_32574418_0"<
while_lstm_cell_17_32574420while_lstm_cell_17_32574420_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :??????????:??????????: : : : : 2X
*while/lstm_cell_17/StatefulPartitionedCall*while/lstm_cell_17/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?
?
while_cond_32576462
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_32576462___redundant_placeholder06
2while_while_cond_32576462___redundant_placeholder16
2while_while_cond_32576462___redundant_placeholder26
2while_while_cond_32576462___redundant_placeholder3
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
B: : : : :??????????:??????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
??
?
while_body_32576463
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_16_matmul_readvariableop_resource_0:	]?I
5while_lstm_cell_16_matmul_1_readvariableop_resource_0:
??C
4while_lstm_cell_16_biasadd_readvariableop_resource_0:	?
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_16_matmul_readvariableop_resource:	]?G
3while_lstm_cell_16_matmul_1_readvariableop_resource:
??A
2while_lstm_cell_16_biasadd_readvariableop_resource:	???)while/lstm_cell_16/BiasAdd/ReadVariableOp?(while/lstm_cell_16/MatMul/ReadVariableOp?*while/lstm_cell_16/MatMul_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????]   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????]*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
(while/lstm_cell_16/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_16_matmul_readvariableop_resource_0*
_output_shapes
:	]?*
dtype02*
(while/lstm_cell_16/MatMul/ReadVariableOp?
while/lstm_cell_16/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_16/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_16/MatMul?
*while/lstm_cell_16/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_16_matmul_1_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02,
*while/lstm_cell_16/MatMul_1/ReadVariableOp?
while/lstm_cell_16/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_16/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_16/MatMul_1?
while/lstm_cell_16/addAddV2#while/lstm_cell_16/MatMul:product:0%while/lstm_cell_16/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_16/add?
)while/lstm_cell_16/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_16_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype02+
)while/lstm_cell_16/BiasAdd/ReadVariableOp?
while/lstm_cell_16/BiasAddBiasAddwhile/lstm_cell_16/add:z:01while/lstm_cell_16/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_16/BiasAdd?
"while/lstm_cell_16/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_16/split/split_dim?
while/lstm_cell_16/splitSplit+while/lstm_cell_16/split/split_dim:output:0#while/lstm_cell_16/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split2
while/lstm_cell_16/split?
while/lstm_cell_16/SigmoidSigmoid!while/lstm_cell_16/split:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_16/Sigmoid?
while/lstm_cell_16/Sigmoid_1Sigmoid!while/lstm_cell_16/split:output:1*
T0*(
_output_shapes
:??????????2
while/lstm_cell_16/Sigmoid_1?
while/lstm_cell_16/mulMul while/lstm_cell_16/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:??????????2
while/lstm_cell_16/mul?
while/lstm_cell_16/ReluRelu!while/lstm_cell_16/split:output:2*
T0*(
_output_shapes
:??????????2
while/lstm_cell_16/Relu?
while/lstm_cell_16/mul_1Mulwhile/lstm_cell_16/Sigmoid:y:0%while/lstm_cell_16/Relu:activations:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_16/mul_1?
while/lstm_cell_16/add_1AddV2while/lstm_cell_16/mul:z:0while/lstm_cell_16/mul_1:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_16/add_1?
while/lstm_cell_16/Sigmoid_2Sigmoid!while/lstm_cell_16/split:output:3*
T0*(
_output_shapes
:??????????2
while/lstm_cell_16/Sigmoid_2?
while/lstm_cell_16/Relu_1Reluwhile/lstm_cell_16/add_1:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_16/Relu_1?
while/lstm_cell_16/mul_2Mul while/lstm_cell_16/Sigmoid_2:y:0'while/lstm_cell_16/Relu_1:activations:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_16/mul_2?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_16/mul_2:z:0*
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
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identitywhile/lstm_cell_16/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:??????????2
while/Identity_4?
while/Identity_5Identitywhile/lstm_cell_16/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:??????????2
while/Identity_5?

while/NoOpNoOp*^while/lstm_cell_16/BiasAdd/ReadVariableOp)^while/lstm_cell_16/MatMul/ReadVariableOp+^while/lstm_cell_16/MatMul_1/ReadVariableOp*"
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
2while_lstm_cell_16_biasadd_readvariableop_resource4while_lstm_cell_16_biasadd_readvariableop_resource_0"l
3while_lstm_cell_16_matmul_1_readvariableop_resource5while_lstm_cell_16_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_16_matmul_readvariableop_resource3while_lstm_cell_16_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :??????????:??????????: : : : : 2V
)while/lstm_cell_16/BiasAdd/ReadVariableOp)while/lstm_cell_16/BiasAdd/ReadVariableOp2T
(while/lstm_cell_16/MatMul/ReadVariableOp(while/lstm_cell_16/MatMul/ReadVariableOp2X
*while/lstm_cell_16/MatMul_1/ReadVariableOp*while/lstm_cell_16/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?
?
while_cond_32577137
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_32577137___redundant_placeholder06
2while_while_cond_32577137___redundant_placeholder16
2while_while_cond_32577137___redundant_placeholder26
2while_while_cond_32577137___redundant_placeholder3
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
B: : : : :??????????:??????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?\
?
E__inference_lstm_17_layer_call_and_return_conditional_losses_32577675

inputs?
+lstm_cell_17_matmul_readvariableop_resource:
??A
-lstm_cell_17_matmul_1_readvariableop_resource:
??;
,lstm_cell_17_biasadd_readvariableop_resource:	?
identity??#lstm_cell_17/BiasAdd/ReadVariableOp?"lstm_cell_17/MatMul/ReadVariableOp?$lstm_cell_17/MatMul_1/ReadVariableOp?whileD
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
strided_slice/stack_2?
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
B :?2
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
B :?2
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
B :?2
zeros/packed/1?
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
:??????????2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
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
B :?2
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
B :?2
zeros_1/packed/1?
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
:??????????2	
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
:??????????2
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
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????  27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
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
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_2?
"lstm_cell_17/MatMul/ReadVariableOpReadVariableOp+lstm_cell_17_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02$
"lstm_cell_17/MatMul/ReadVariableOp?
lstm_cell_17/MatMulMatMulstrided_slice_2:output:0*lstm_cell_17/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_17/MatMul?
$lstm_cell_17/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_17_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02&
$lstm_cell_17/MatMul_1/ReadVariableOp?
lstm_cell_17/MatMul_1MatMulzeros:output:0,lstm_cell_17/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_17/MatMul_1?
lstm_cell_17/addAddV2lstm_cell_17/MatMul:product:0lstm_cell_17/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell_17/add?
#lstm_cell_17/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_17_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02%
#lstm_cell_17/BiasAdd/ReadVariableOp?
lstm_cell_17/BiasAddBiasAddlstm_cell_17/add:z:0+lstm_cell_17/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_17/BiasAdd~
lstm_cell_17/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_17/split/split_dim?
lstm_cell_17/splitSplit%lstm_cell_17/split/split_dim:output:0lstm_cell_17/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split2
lstm_cell_17/split?
lstm_cell_17/SigmoidSigmoidlstm_cell_17/split:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_17/Sigmoid?
lstm_cell_17/Sigmoid_1Sigmoidlstm_cell_17/split:output:1*
T0*(
_output_shapes
:??????????2
lstm_cell_17/Sigmoid_1?
lstm_cell_17/mulMullstm_cell_17/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_17/mul~
lstm_cell_17/ReluRelulstm_cell_17/split:output:2*
T0*(
_output_shapes
:??????????2
lstm_cell_17/Relu?
lstm_cell_17/mul_1Mullstm_cell_17/Sigmoid:y:0lstm_cell_17/Relu:activations:0*
T0*(
_output_shapes
:??????????2
lstm_cell_17/mul_1?
lstm_cell_17/add_1AddV2lstm_cell_17/mul:z:0lstm_cell_17/mul_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell_17/add_1?
lstm_cell_17/Sigmoid_2Sigmoidlstm_cell_17/split:output:3*
T0*(
_output_shapes
:??????????2
lstm_cell_17/Sigmoid_2}
lstm_cell_17/Relu_1Relulstm_cell_17/add_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell_17/Relu_1?
lstm_cell_17/mul_2Mullstm_cell_17/Sigmoid_2:y:0!lstm_cell_17/Relu_1:activations:0*
T0*(
_output_shapes
:??????????2
lstm_cell_17/mul_2?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2
TensorArrayV2_1/element_shape?
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
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_17_matmul_readvariableop_resource-lstm_cell_17_matmul_1_readvariableop_resource,lstm_cell_17_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_32577591*
condR
while_cond_32577590*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:??????????*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
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
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:??????????2
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
:??????????2

Identity?
NoOpNoOp$^lstm_cell_17/BiasAdd/ReadVariableOp#^lstm_cell_17/MatMul/ReadVariableOp%^lstm_cell_17/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:??????????: : : 2J
#lstm_cell_17/BiasAdd/ReadVariableOp#lstm_cell_17/BiasAdd/ReadVariableOp2H
"lstm_cell_17/MatMul/ReadVariableOp"lstm_cell_17/MatMul/ReadVariableOp2L
$lstm_cell_17/MatMul_1/ReadVariableOp$lstm_cell_17/MatMul_1/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?\
?
E__inference_lstm_16_layer_call_and_return_conditional_losses_32576547
inputs_0>
+lstm_cell_16_matmul_readvariableop_resource:	]?A
-lstm_cell_16_matmul_1_readvariableop_resource:
??;
,lstm_cell_16_biasadd_readvariableop_resource:	?
identity??#lstm_cell_16/BiasAdd/ReadVariableOp?"lstm_cell_16/MatMul/ReadVariableOp?$lstm_cell_16/MatMul_1/ReadVariableOp?whileF
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
strided_slice/stack_2?
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
B :?2
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
B :?2
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
B :?2
zeros/packed/1?
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
:??????????2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
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
B :?2
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
B :?2
zeros_1/packed/1?
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
:??????????2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm?
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????]2
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
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????]   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
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
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????]*
shrink_axis_mask2
strided_slice_2?
"lstm_cell_16/MatMul/ReadVariableOpReadVariableOp+lstm_cell_16_matmul_readvariableop_resource*
_output_shapes
:	]?*
dtype02$
"lstm_cell_16/MatMul/ReadVariableOp?
lstm_cell_16/MatMulMatMulstrided_slice_2:output:0*lstm_cell_16/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_16/MatMul?
$lstm_cell_16/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_16_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02&
$lstm_cell_16/MatMul_1/ReadVariableOp?
lstm_cell_16/MatMul_1MatMulzeros:output:0,lstm_cell_16/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_16/MatMul_1?
lstm_cell_16/addAddV2lstm_cell_16/MatMul:product:0lstm_cell_16/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell_16/add?
#lstm_cell_16/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_16_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02%
#lstm_cell_16/BiasAdd/ReadVariableOp?
lstm_cell_16/BiasAddBiasAddlstm_cell_16/add:z:0+lstm_cell_16/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_16/BiasAdd~
lstm_cell_16/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_16/split/split_dim?
lstm_cell_16/splitSplit%lstm_cell_16/split/split_dim:output:0lstm_cell_16/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split2
lstm_cell_16/split?
lstm_cell_16/SigmoidSigmoidlstm_cell_16/split:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_16/Sigmoid?
lstm_cell_16/Sigmoid_1Sigmoidlstm_cell_16/split:output:1*
T0*(
_output_shapes
:??????????2
lstm_cell_16/Sigmoid_1?
lstm_cell_16/mulMullstm_cell_16/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_16/mul~
lstm_cell_16/ReluRelulstm_cell_16/split:output:2*
T0*(
_output_shapes
:??????????2
lstm_cell_16/Relu?
lstm_cell_16/mul_1Mullstm_cell_16/Sigmoid:y:0lstm_cell_16/Relu:activations:0*
T0*(
_output_shapes
:??????????2
lstm_cell_16/mul_1?
lstm_cell_16/add_1AddV2lstm_cell_16/mul:z:0lstm_cell_16/mul_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell_16/add_1?
lstm_cell_16/Sigmoid_2Sigmoidlstm_cell_16/split:output:3*
T0*(
_output_shapes
:??????????2
lstm_cell_16/Sigmoid_2}
lstm_cell_16/Relu_1Relulstm_cell_16/add_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell_16/Relu_1?
lstm_cell_16/mul_2Mullstm_cell_16/Sigmoid_2:y:0!lstm_cell_16/Relu_1:activations:0*
T0*(
_output_shapes
:??????????2
lstm_cell_16/mul_2?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????  2
TensorArrayV2_1/element_shape?
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
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_16_matmul_readvariableop_resource-lstm_cell_16_matmul_1_readvariableop_resource,lstm_cell_16_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_32576463*
condR
while_cond_32576462*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????  22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:???????????????????*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
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
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:???????????????????2
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
!:???????????????????2

Identity?
NoOpNoOp$^lstm_cell_16/BiasAdd/ReadVariableOp#^lstm_cell_16/MatMul/ReadVariableOp%^lstm_cell_16/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????????????]: : : 2J
#lstm_cell_16/BiasAdd/ReadVariableOp#lstm_cell_16/BiasAdd/ReadVariableOp2H
"lstm_cell_16/MatMul/ReadVariableOp"lstm_cell_16/MatMul/ReadVariableOp2L
$lstm_cell_16/MatMul_1/ReadVariableOp$lstm_cell_16/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :??????????????????]
"
_user_specified_name
inputs/0
?
?
*__inference_lstm_16_layer_call_fn_32577044

inputs
unknown:	]?
	unknown_0:
??
	unknown_1:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_lstm_16_layer_call_and_return_conditional_losses_325755102
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:??????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????]: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????]
 
_user_specified_nameinputs
?
g
H__inference_dropout_16_layer_call_and_return_conditional_losses_32577061

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2
dropout/Constx
dropout/MulMulinputsdropout/Const:output:0*
T0*,
_output_shapes
:??????????2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:??????????*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?>2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:??????????2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:??????????2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:??????????2
dropout/Mul_1j
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
*__inference_dense_8_layer_call_fn_32577786

inputs
unknown:	?
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_8_layer_call_and_return_conditional_losses_325750912
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?\
?
E__inference_lstm_16_layer_call_and_return_conditional_losses_32576698
inputs_0>
+lstm_cell_16_matmul_readvariableop_resource:	]?A
-lstm_cell_16_matmul_1_readvariableop_resource:
??;
,lstm_cell_16_biasadd_readvariableop_resource:	?
identity??#lstm_cell_16/BiasAdd/ReadVariableOp?"lstm_cell_16/MatMul/ReadVariableOp?$lstm_cell_16/MatMul_1/ReadVariableOp?whileF
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
strided_slice/stack_2?
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
B :?2
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
B :?2
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
B :?2
zeros/packed/1?
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
:??????????2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
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
B :?2
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
B :?2
zeros_1/packed/1?
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
:??????????2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm?
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????]2
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
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????]   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
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
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????]*
shrink_axis_mask2
strided_slice_2?
"lstm_cell_16/MatMul/ReadVariableOpReadVariableOp+lstm_cell_16_matmul_readvariableop_resource*
_output_shapes
:	]?*
dtype02$
"lstm_cell_16/MatMul/ReadVariableOp?
lstm_cell_16/MatMulMatMulstrided_slice_2:output:0*lstm_cell_16/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_16/MatMul?
$lstm_cell_16/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_16_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02&
$lstm_cell_16/MatMul_1/ReadVariableOp?
lstm_cell_16/MatMul_1MatMulzeros:output:0,lstm_cell_16/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_16/MatMul_1?
lstm_cell_16/addAddV2lstm_cell_16/MatMul:product:0lstm_cell_16/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell_16/add?
#lstm_cell_16/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_16_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02%
#lstm_cell_16/BiasAdd/ReadVariableOp?
lstm_cell_16/BiasAddBiasAddlstm_cell_16/add:z:0+lstm_cell_16/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_16/BiasAdd~
lstm_cell_16/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_16/split/split_dim?
lstm_cell_16/splitSplit%lstm_cell_16/split/split_dim:output:0lstm_cell_16/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split2
lstm_cell_16/split?
lstm_cell_16/SigmoidSigmoidlstm_cell_16/split:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_16/Sigmoid?
lstm_cell_16/Sigmoid_1Sigmoidlstm_cell_16/split:output:1*
T0*(
_output_shapes
:??????????2
lstm_cell_16/Sigmoid_1?
lstm_cell_16/mulMullstm_cell_16/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_16/mul~
lstm_cell_16/ReluRelulstm_cell_16/split:output:2*
T0*(
_output_shapes
:??????????2
lstm_cell_16/Relu?
lstm_cell_16/mul_1Mullstm_cell_16/Sigmoid:y:0lstm_cell_16/Relu:activations:0*
T0*(
_output_shapes
:??????????2
lstm_cell_16/mul_1?
lstm_cell_16/add_1AddV2lstm_cell_16/mul:z:0lstm_cell_16/mul_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell_16/add_1?
lstm_cell_16/Sigmoid_2Sigmoidlstm_cell_16/split:output:3*
T0*(
_output_shapes
:??????????2
lstm_cell_16/Sigmoid_2}
lstm_cell_16/Relu_1Relulstm_cell_16/add_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell_16/Relu_1?
lstm_cell_16/mul_2Mullstm_cell_16/Sigmoid_2:y:0!lstm_cell_16/Relu_1:activations:0*
T0*(
_output_shapes
:??????????2
lstm_cell_16/mul_2?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????  2
TensorArrayV2_1/element_shape?
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
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_16_matmul_readvariableop_resource-lstm_cell_16_matmul_1_readvariableop_resource,lstm_cell_16_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_32576614*
condR
while_cond_32576613*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????  22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:???????????????????*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
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
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:???????????????????2
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
!:???????????????????2

Identity?
NoOpNoOp$^lstm_cell_16/BiasAdd/ReadVariableOp#^lstm_cell_16/MatMul/ReadVariableOp%^lstm_cell_16/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????????????]: : : 2J
#lstm_cell_16/BiasAdd/ReadVariableOp#lstm_cell_16/BiasAdd/ReadVariableOp2H
"lstm_cell_16/MatMul/ReadVariableOp"lstm_cell_16/MatMul/ReadVariableOp2L
$lstm_cell_16/MatMul_1/ReadVariableOp$lstm_cell_16/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :??????????????????]
"
_user_specified_name
inputs/0
??
?
while_body_32577440
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
3while_lstm_cell_17_matmul_readvariableop_resource_0:
??I
5while_lstm_cell_17_matmul_1_readvariableop_resource_0:
??C
4while_lstm_cell_17_biasadd_readvariableop_resource_0:	?
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
1while_lstm_cell_17_matmul_readvariableop_resource:
??G
3while_lstm_cell_17_matmul_1_readvariableop_resource:
??A
2while_lstm_cell_17_biasadd_readvariableop_resource:	???)while/lstm_cell_17/BiasAdd/ReadVariableOp?(while/lstm_cell_17/MatMul/ReadVariableOp?*while/lstm_cell_17/MatMul_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????  29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:??????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
(while/lstm_cell_17/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_17_matmul_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02*
(while/lstm_cell_17/MatMul/ReadVariableOp?
while/lstm_cell_17/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_17/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_17/MatMul?
*while/lstm_cell_17/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_17_matmul_1_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02,
*while/lstm_cell_17/MatMul_1/ReadVariableOp?
while/lstm_cell_17/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_17/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_17/MatMul_1?
while/lstm_cell_17/addAddV2#while/lstm_cell_17/MatMul:product:0%while/lstm_cell_17/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_17/add?
)while/lstm_cell_17/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_17_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype02+
)while/lstm_cell_17/BiasAdd/ReadVariableOp?
while/lstm_cell_17/BiasAddBiasAddwhile/lstm_cell_17/add:z:01while/lstm_cell_17/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_17/BiasAdd?
"while/lstm_cell_17/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_17/split/split_dim?
while/lstm_cell_17/splitSplit+while/lstm_cell_17/split/split_dim:output:0#while/lstm_cell_17/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split2
while/lstm_cell_17/split?
while/lstm_cell_17/SigmoidSigmoid!while/lstm_cell_17/split:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_17/Sigmoid?
while/lstm_cell_17/Sigmoid_1Sigmoid!while/lstm_cell_17/split:output:1*
T0*(
_output_shapes
:??????????2
while/lstm_cell_17/Sigmoid_1?
while/lstm_cell_17/mulMul while/lstm_cell_17/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:??????????2
while/lstm_cell_17/mul?
while/lstm_cell_17/ReluRelu!while/lstm_cell_17/split:output:2*
T0*(
_output_shapes
:??????????2
while/lstm_cell_17/Relu?
while/lstm_cell_17/mul_1Mulwhile/lstm_cell_17/Sigmoid:y:0%while/lstm_cell_17/Relu:activations:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_17/mul_1?
while/lstm_cell_17/add_1AddV2while/lstm_cell_17/mul:z:0while/lstm_cell_17/mul_1:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_17/add_1?
while/lstm_cell_17/Sigmoid_2Sigmoid!while/lstm_cell_17/split:output:3*
T0*(
_output_shapes
:??????????2
while/lstm_cell_17/Sigmoid_2?
while/lstm_cell_17/Relu_1Reluwhile/lstm_cell_17/add_1:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_17/Relu_1?
while/lstm_cell_17/mul_2Mul while/lstm_cell_17/Sigmoid_2:y:0'while/lstm_cell_17/Relu_1:activations:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_17/mul_2?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_17/mul_2:z:0*
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
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identitywhile/lstm_cell_17/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:??????????2
while/Identity_4?
while/Identity_5Identitywhile/lstm_cell_17/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:??????????2
while/Identity_5?

while/NoOpNoOp*^while/lstm_cell_17/BiasAdd/ReadVariableOp)^while/lstm_cell_17/MatMul/ReadVariableOp+^while/lstm_cell_17/MatMul_1/ReadVariableOp*"
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
2while_lstm_cell_17_biasadd_readvariableop_resource4while_lstm_cell_17_biasadd_readvariableop_resource_0"l
3while_lstm_cell_17_matmul_1_readvariableop_resource5while_lstm_cell_17_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_17_matmul_readvariableop_resource3while_lstm_cell_17_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :??????????:??????????: : : : : 2V
)while/lstm_cell_17/BiasAdd/ReadVariableOp)while/lstm_cell_17/BiasAdd/ReadVariableOp2T
(while/lstm_cell_17/MatMul/ReadVariableOp(while/lstm_cell_17/MatMul/ReadVariableOp2X
*while/lstm_cell_17/MatMul_1/ReadVariableOp*while/lstm_cell_17/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
??
?
while_body_32577591
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
3while_lstm_cell_17_matmul_readvariableop_resource_0:
??I
5while_lstm_cell_17_matmul_1_readvariableop_resource_0:
??C
4while_lstm_cell_17_biasadd_readvariableop_resource_0:	?
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
1while_lstm_cell_17_matmul_readvariableop_resource:
??G
3while_lstm_cell_17_matmul_1_readvariableop_resource:
??A
2while_lstm_cell_17_biasadd_readvariableop_resource:	???)while/lstm_cell_17/BiasAdd/ReadVariableOp?(while/lstm_cell_17/MatMul/ReadVariableOp?*while/lstm_cell_17/MatMul_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????  29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:??????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
(while/lstm_cell_17/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_17_matmul_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02*
(while/lstm_cell_17/MatMul/ReadVariableOp?
while/lstm_cell_17/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_17/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_17/MatMul?
*while/lstm_cell_17/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_17_matmul_1_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02,
*while/lstm_cell_17/MatMul_1/ReadVariableOp?
while/lstm_cell_17/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_17/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_17/MatMul_1?
while/lstm_cell_17/addAddV2#while/lstm_cell_17/MatMul:product:0%while/lstm_cell_17/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_17/add?
)while/lstm_cell_17/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_17_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype02+
)while/lstm_cell_17/BiasAdd/ReadVariableOp?
while/lstm_cell_17/BiasAddBiasAddwhile/lstm_cell_17/add:z:01while/lstm_cell_17/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_17/BiasAdd?
"while/lstm_cell_17/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_17/split/split_dim?
while/lstm_cell_17/splitSplit+while/lstm_cell_17/split/split_dim:output:0#while/lstm_cell_17/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split2
while/lstm_cell_17/split?
while/lstm_cell_17/SigmoidSigmoid!while/lstm_cell_17/split:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_17/Sigmoid?
while/lstm_cell_17/Sigmoid_1Sigmoid!while/lstm_cell_17/split:output:1*
T0*(
_output_shapes
:??????????2
while/lstm_cell_17/Sigmoid_1?
while/lstm_cell_17/mulMul while/lstm_cell_17/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:??????????2
while/lstm_cell_17/mul?
while/lstm_cell_17/ReluRelu!while/lstm_cell_17/split:output:2*
T0*(
_output_shapes
:??????????2
while/lstm_cell_17/Relu?
while/lstm_cell_17/mul_1Mulwhile/lstm_cell_17/Sigmoid:y:0%while/lstm_cell_17/Relu:activations:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_17/mul_1?
while/lstm_cell_17/add_1AddV2while/lstm_cell_17/mul:z:0while/lstm_cell_17/mul_1:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_17/add_1?
while/lstm_cell_17/Sigmoid_2Sigmoid!while/lstm_cell_17/split:output:3*
T0*(
_output_shapes
:??????????2
while/lstm_cell_17/Sigmoid_2?
while/lstm_cell_17/Relu_1Reluwhile/lstm_cell_17/add_1:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_17/Relu_1?
while/lstm_cell_17/mul_2Mul while/lstm_cell_17/Sigmoid_2:y:0'while/lstm_cell_17/Relu_1:activations:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_17/mul_2?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_17/mul_2:z:0*
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
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identitywhile/lstm_cell_17/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:??????????2
while/Identity_4?
while/Identity_5Identitywhile/lstm_cell_17/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:??????????2
while/Identity_5?

while/NoOpNoOp*^while/lstm_cell_17/BiasAdd/ReadVariableOp)^while/lstm_cell_17/MatMul/ReadVariableOp+^while/lstm_cell_17/MatMul_1/ReadVariableOp*"
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
2while_lstm_cell_17_biasadd_readvariableop_resource4while_lstm_cell_17_biasadd_readvariableop_resource_0"l
3while_lstm_cell_17_matmul_1_readvariableop_resource5while_lstm_cell_17_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_17_matmul_readvariableop_resource3while_lstm_cell_17_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :??????????:??????????: : : : : 2V
)while/lstm_cell_17/BiasAdd/ReadVariableOp)while/lstm_cell_17/BiasAdd/ReadVariableOp2T
(while/lstm_cell_17/MatMul/ReadVariableOp(while/lstm_cell_17/MatMul/ReadVariableOp2X
*while/lstm_cell_17/MatMul_1/ReadVariableOp*while/lstm_cell_17/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?
?
*__inference_lstm_16_layer_call_fn_32577011
inputs_0
unknown:	]?
	unknown_0:
??
	unknown_1:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:???????????????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_lstm_16_layer_call_and_return_conditional_losses_325736212
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:???????????????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????????????]: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :??????????????????]
"
_user_specified_name
inputs/0
?
f
-__inference_dropout_16_layer_call_fn_32577071

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_dropout_16_layer_call_and_return_conditional_losses_325753432
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:??????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?\
?
E__inference_lstm_16_layer_call_and_return_conditional_losses_32575510

inputs>
+lstm_cell_16_matmul_readvariableop_resource:	]?A
-lstm_cell_16_matmul_1_readvariableop_resource:
??;
,lstm_cell_16_biasadd_readvariableop_resource:	?
identity??#lstm_cell_16/BiasAdd/ReadVariableOp?"lstm_cell_16/MatMul/ReadVariableOp?$lstm_cell_16/MatMul_1/ReadVariableOp?whileD
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
strided_slice/stack_2?
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
B :?2
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
B :?2
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
B :?2
zeros/packed/1?
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
:??????????2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
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
B :?2
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
B :?2
zeros_1/packed/1?
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
:??????????2	
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
:?????????]2
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
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????]   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
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
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????]*
shrink_axis_mask2
strided_slice_2?
"lstm_cell_16/MatMul/ReadVariableOpReadVariableOp+lstm_cell_16_matmul_readvariableop_resource*
_output_shapes
:	]?*
dtype02$
"lstm_cell_16/MatMul/ReadVariableOp?
lstm_cell_16/MatMulMatMulstrided_slice_2:output:0*lstm_cell_16/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_16/MatMul?
$lstm_cell_16/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_16_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02&
$lstm_cell_16/MatMul_1/ReadVariableOp?
lstm_cell_16/MatMul_1MatMulzeros:output:0,lstm_cell_16/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_16/MatMul_1?
lstm_cell_16/addAddV2lstm_cell_16/MatMul:product:0lstm_cell_16/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell_16/add?
#lstm_cell_16/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_16_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02%
#lstm_cell_16/BiasAdd/ReadVariableOp?
lstm_cell_16/BiasAddBiasAddlstm_cell_16/add:z:0+lstm_cell_16/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_16/BiasAdd~
lstm_cell_16/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_16/split/split_dim?
lstm_cell_16/splitSplit%lstm_cell_16/split/split_dim:output:0lstm_cell_16/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split2
lstm_cell_16/split?
lstm_cell_16/SigmoidSigmoidlstm_cell_16/split:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_16/Sigmoid?
lstm_cell_16/Sigmoid_1Sigmoidlstm_cell_16/split:output:1*
T0*(
_output_shapes
:??????????2
lstm_cell_16/Sigmoid_1?
lstm_cell_16/mulMullstm_cell_16/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_16/mul~
lstm_cell_16/ReluRelulstm_cell_16/split:output:2*
T0*(
_output_shapes
:??????????2
lstm_cell_16/Relu?
lstm_cell_16/mul_1Mullstm_cell_16/Sigmoid:y:0lstm_cell_16/Relu:activations:0*
T0*(
_output_shapes
:??????????2
lstm_cell_16/mul_1?
lstm_cell_16/add_1AddV2lstm_cell_16/mul:z:0lstm_cell_16/mul_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell_16/add_1?
lstm_cell_16/Sigmoid_2Sigmoidlstm_cell_16/split:output:3*
T0*(
_output_shapes
:??????????2
lstm_cell_16/Sigmoid_2}
lstm_cell_16/Relu_1Relulstm_cell_16/add_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell_16/Relu_1?
lstm_cell_16/mul_2Mullstm_cell_16/Sigmoid_2:y:0!lstm_cell_16/Relu_1:activations:0*
T0*(
_output_shapes
:??????????2
lstm_cell_16/mul_2?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????  2
TensorArrayV2_1/element_shape?
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
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_16_matmul_readvariableop_resource-lstm_cell_16_matmul_1_readvariableop_resource,lstm_cell_16_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_32575426*
condR
while_cond_32575425*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????  22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:??????????*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
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
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:??????????2
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
:??????????2

Identity?
NoOpNoOp$^lstm_cell_16/BiasAdd/ReadVariableOp#^lstm_cell_16/MatMul/ReadVariableOp%^lstm_cell_16/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????]: : : 2J
#lstm_cell_16/BiasAdd/ReadVariableOp#lstm_cell_16/BiasAdd/ReadVariableOp2H
"lstm_cell_16/MatMul/ReadVariableOp"lstm_cell_16/MatMul/ReadVariableOp2L
$lstm_cell_16/MatMul_1/ReadVariableOp$lstm_cell_16/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:?????????]
 
_user_specified_nameinputs
?
?
*__inference_lstm_17_layer_call_fn_32577719

inputs
unknown:
??
	unknown_0:
??
	unknown_1:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_lstm_17_layer_call_and_return_conditional_losses_325753142
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:??????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:??????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?]
?
(sequential_8_lstm_16_while_body_32573203F
Bsequential_8_lstm_16_while_sequential_8_lstm_16_while_loop_counterL
Hsequential_8_lstm_16_while_sequential_8_lstm_16_while_maximum_iterations*
&sequential_8_lstm_16_while_placeholder,
(sequential_8_lstm_16_while_placeholder_1,
(sequential_8_lstm_16_while_placeholder_2,
(sequential_8_lstm_16_while_placeholder_3E
Asequential_8_lstm_16_while_sequential_8_lstm_16_strided_slice_1_0?
}sequential_8_lstm_16_while_tensorarrayv2read_tensorlistgetitem_sequential_8_lstm_16_tensorarrayunstack_tensorlistfromtensor_0[
Hsequential_8_lstm_16_while_lstm_cell_16_matmul_readvariableop_resource_0:	]?^
Jsequential_8_lstm_16_while_lstm_cell_16_matmul_1_readvariableop_resource_0:
??X
Isequential_8_lstm_16_while_lstm_cell_16_biasadd_readvariableop_resource_0:	?'
#sequential_8_lstm_16_while_identity)
%sequential_8_lstm_16_while_identity_1)
%sequential_8_lstm_16_while_identity_2)
%sequential_8_lstm_16_while_identity_3)
%sequential_8_lstm_16_while_identity_4)
%sequential_8_lstm_16_while_identity_5C
?sequential_8_lstm_16_while_sequential_8_lstm_16_strided_slice_1
{sequential_8_lstm_16_while_tensorarrayv2read_tensorlistgetitem_sequential_8_lstm_16_tensorarrayunstack_tensorlistfromtensorY
Fsequential_8_lstm_16_while_lstm_cell_16_matmul_readvariableop_resource:	]?\
Hsequential_8_lstm_16_while_lstm_cell_16_matmul_1_readvariableop_resource:
??V
Gsequential_8_lstm_16_while_lstm_cell_16_biasadd_readvariableop_resource:	???>sequential_8/lstm_16/while/lstm_cell_16/BiasAdd/ReadVariableOp?=sequential_8/lstm_16/while/lstm_cell_16/MatMul/ReadVariableOp??sequential_8/lstm_16/while/lstm_cell_16/MatMul_1/ReadVariableOp?
Lsequential_8/lstm_16/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????]   2N
Lsequential_8/lstm_16/while/TensorArrayV2Read/TensorListGetItem/element_shape?
>sequential_8/lstm_16/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem}sequential_8_lstm_16_while_tensorarrayv2read_tensorlistgetitem_sequential_8_lstm_16_tensorarrayunstack_tensorlistfromtensor_0&sequential_8_lstm_16_while_placeholderUsequential_8/lstm_16/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????]*
element_dtype02@
>sequential_8/lstm_16/while/TensorArrayV2Read/TensorListGetItem?
=sequential_8/lstm_16/while/lstm_cell_16/MatMul/ReadVariableOpReadVariableOpHsequential_8_lstm_16_while_lstm_cell_16_matmul_readvariableop_resource_0*
_output_shapes
:	]?*
dtype02?
=sequential_8/lstm_16/while/lstm_cell_16/MatMul/ReadVariableOp?
.sequential_8/lstm_16/while/lstm_cell_16/MatMulMatMulEsequential_8/lstm_16/while/TensorArrayV2Read/TensorListGetItem:item:0Esequential_8/lstm_16/while/lstm_cell_16/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????20
.sequential_8/lstm_16/while/lstm_cell_16/MatMul?
?sequential_8/lstm_16/while/lstm_cell_16/MatMul_1/ReadVariableOpReadVariableOpJsequential_8_lstm_16_while_lstm_cell_16_matmul_1_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02A
?sequential_8/lstm_16/while/lstm_cell_16/MatMul_1/ReadVariableOp?
0sequential_8/lstm_16/while/lstm_cell_16/MatMul_1MatMul(sequential_8_lstm_16_while_placeholder_2Gsequential_8/lstm_16/while/lstm_cell_16/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????22
0sequential_8/lstm_16/while/lstm_cell_16/MatMul_1?
+sequential_8/lstm_16/while/lstm_cell_16/addAddV28sequential_8/lstm_16/while/lstm_cell_16/MatMul:product:0:sequential_8/lstm_16/while/lstm_cell_16/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2-
+sequential_8/lstm_16/while/lstm_cell_16/add?
>sequential_8/lstm_16/while/lstm_cell_16/BiasAdd/ReadVariableOpReadVariableOpIsequential_8_lstm_16_while_lstm_cell_16_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype02@
>sequential_8/lstm_16/while/lstm_cell_16/BiasAdd/ReadVariableOp?
/sequential_8/lstm_16/while/lstm_cell_16/BiasAddBiasAdd/sequential_8/lstm_16/while/lstm_cell_16/add:z:0Fsequential_8/lstm_16/while/lstm_cell_16/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????21
/sequential_8/lstm_16/while/lstm_cell_16/BiasAdd?
7sequential_8/lstm_16/while/lstm_cell_16/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :29
7sequential_8/lstm_16/while/lstm_cell_16/split/split_dim?
-sequential_8/lstm_16/while/lstm_cell_16/splitSplit@sequential_8/lstm_16/while/lstm_cell_16/split/split_dim:output:08sequential_8/lstm_16/while/lstm_cell_16/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split2/
-sequential_8/lstm_16/while/lstm_cell_16/split?
/sequential_8/lstm_16/while/lstm_cell_16/SigmoidSigmoid6sequential_8/lstm_16/while/lstm_cell_16/split:output:0*
T0*(
_output_shapes
:??????????21
/sequential_8/lstm_16/while/lstm_cell_16/Sigmoid?
1sequential_8/lstm_16/while/lstm_cell_16/Sigmoid_1Sigmoid6sequential_8/lstm_16/while/lstm_cell_16/split:output:1*
T0*(
_output_shapes
:??????????23
1sequential_8/lstm_16/while/lstm_cell_16/Sigmoid_1?
+sequential_8/lstm_16/while/lstm_cell_16/mulMul5sequential_8/lstm_16/while/lstm_cell_16/Sigmoid_1:y:0(sequential_8_lstm_16_while_placeholder_3*
T0*(
_output_shapes
:??????????2-
+sequential_8/lstm_16/while/lstm_cell_16/mul?
,sequential_8/lstm_16/while/lstm_cell_16/ReluRelu6sequential_8/lstm_16/while/lstm_cell_16/split:output:2*
T0*(
_output_shapes
:??????????2.
,sequential_8/lstm_16/while/lstm_cell_16/Relu?
-sequential_8/lstm_16/while/lstm_cell_16/mul_1Mul3sequential_8/lstm_16/while/lstm_cell_16/Sigmoid:y:0:sequential_8/lstm_16/while/lstm_cell_16/Relu:activations:0*
T0*(
_output_shapes
:??????????2/
-sequential_8/lstm_16/while/lstm_cell_16/mul_1?
-sequential_8/lstm_16/while/lstm_cell_16/add_1AddV2/sequential_8/lstm_16/while/lstm_cell_16/mul:z:01sequential_8/lstm_16/while/lstm_cell_16/mul_1:z:0*
T0*(
_output_shapes
:??????????2/
-sequential_8/lstm_16/while/lstm_cell_16/add_1?
1sequential_8/lstm_16/while/lstm_cell_16/Sigmoid_2Sigmoid6sequential_8/lstm_16/while/lstm_cell_16/split:output:3*
T0*(
_output_shapes
:??????????23
1sequential_8/lstm_16/while/lstm_cell_16/Sigmoid_2?
.sequential_8/lstm_16/while/lstm_cell_16/Relu_1Relu1sequential_8/lstm_16/while/lstm_cell_16/add_1:z:0*
T0*(
_output_shapes
:??????????20
.sequential_8/lstm_16/while/lstm_cell_16/Relu_1?
-sequential_8/lstm_16/while/lstm_cell_16/mul_2Mul5sequential_8/lstm_16/while/lstm_cell_16/Sigmoid_2:y:0<sequential_8/lstm_16/while/lstm_cell_16/Relu_1:activations:0*
T0*(
_output_shapes
:??????????2/
-sequential_8/lstm_16/while/lstm_cell_16/mul_2?
?sequential_8/lstm_16/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem(sequential_8_lstm_16_while_placeholder_1&sequential_8_lstm_16_while_placeholder1sequential_8/lstm_16/while/lstm_cell_16/mul_2:z:0*
_output_shapes
: *
element_dtype02A
?sequential_8/lstm_16/while/TensorArrayV2Write/TensorListSetItem?
 sequential_8/lstm_16/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2"
 sequential_8/lstm_16/while/add/y?
sequential_8/lstm_16/while/addAddV2&sequential_8_lstm_16_while_placeholder)sequential_8/lstm_16/while/add/y:output:0*
T0*
_output_shapes
: 2 
sequential_8/lstm_16/while/add?
"sequential_8/lstm_16/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2$
"sequential_8/lstm_16/while/add_1/y?
 sequential_8/lstm_16/while/add_1AddV2Bsequential_8_lstm_16_while_sequential_8_lstm_16_while_loop_counter+sequential_8/lstm_16/while/add_1/y:output:0*
T0*
_output_shapes
: 2"
 sequential_8/lstm_16/while/add_1?
#sequential_8/lstm_16/while/IdentityIdentity$sequential_8/lstm_16/while/add_1:z:0 ^sequential_8/lstm_16/while/NoOp*
T0*
_output_shapes
: 2%
#sequential_8/lstm_16/while/Identity?
%sequential_8/lstm_16/while/Identity_1IdentityHsequential_8_lstm_16_while_sequential_8_lstm_16_while_maximum_iterations ^sequential_8/lstm_16/while/NoOp*
T0*
_output_shapes
: 2'
%sequential_8/lstm_16/while/Identity_1?
%sequential_8/lstm_16/while/Identity_2Identity"sequential_8/lstm_16/while/add:z:0 ^sequential_8/lstm_16/while/NoOp*
T0*
_output_shapes
: 2'
%sequential_8/lstm_16/while/Identity_2?
%sequential_8/lstm_16/while/Identity_3IdentityOsequential_8/lstm_16/while/TensorArrayV2Write/TensorListSetItem:output_handle:0 ^sequential_8/lstm_16/while/NoOp*
T0*
_output_shapes
: 2'
%sequential_8/lstm_16/while/Identity_3?
%sequential_8/lstm_16/while/Identity_4Identity1sequential_8/lstm_16/while/lstm_cell_16/mul_2:z:0 ^sequential_8/lstm_16/while/NoOp*
T0*(
_output_shapes
:??????????2'
%sequential_8/lstm_16/while/Identity_4?
%sequential_8/lstm_16/while/Identity_5Identity1sequential_8/lstm_16/while/lstm_cell_16/add_1:z:0 ^sequential_8/lstm_16/while/NoOp*
T0*(
_output_shapes
:??????????2'
%sequential_8/lstm_16/while/Identity_5?
sequential_8/lstm_16/while/NoOpNoOp?^sequential_8/lstm_16/while/lstm_cell_16/BiasAdd/ReadVariableOp>^sequential_8/lstm_16/while/lstm_cell_16/MatMul/ReadVariableOp@^sequential_8/lstm_16/while/lstm_cell_16/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2!
sequential_8/lstm_16/while/NoOp"S
#sequential_8_lstm_16_while_identity,sequential_8/lstm_16/while/Identity:output:0"W
%sequential_8_lstm_16_while_identity_1.sequential_8/lstm_16/while/Identity_1:output:0"W
%sequential_8_lstm_16_while_identity_2.sequential_8/lstm_16/while/Identity_2:output:0"W
%sequential_8_lstm_16_while_identity_3.sequential_8/lstm_16/while/Identity_3:output:0"W
%sequential_8_lstm_16_while_identity_4.sequential_8/lstm_16/while/Identity_4:output:0"W
%sequential_8_lstm_16_while_identity_5.sequential_8/lstm_16/while/Identity_5:output:0"?
Gsequential_8_lstm_16_while_lstm_cell_16_biasadd_readvariableop_resourceIsequential_8_lstm_16_while_lstm_cell_16_biasadd_readvariableop_resource_0"?
Hsequential_8_lstm_16_while_lstm_cell_16_matmul_1_readvariableop_resourceJsequential_8_lstm_16_while_lstm_cell_16_matmul_1_readvariableop_resource_0"?
Fsequential_8_lstm_16_while_lstm_cell_16_matmul_readvariableop_resourceHsequential_8_lstm_16_while_lstm_cell_16_matmul_readvariableop_resource_0"?
?sequential_8_lstm_16_while_sequential_8_lstm_16_strided_slice_1Asequential_8_lstm_16_while_sequential_8_lstm_16_strided_slice_1_0"?
{sequential_8_lstm_16_while_tensorarrayv2read_tensorlistgetitem_sequential_8_lstm_16_tensorarrayunstack_tensorlistfromtensor}sequential_8_lstm_16_while_tensorarrayv2read_tensorlistgetitem_sequential_8_lstm_16_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :??????????:??????????: : : : : 2?
>sequential_8/lstm_16/while/lstm_cell_16/BiasAdd/ReadVariableOp>sequential_8/lstm_16/while/lstm_cell_16/BiasAdd/ReadVariableOp2~
=sequential_8/lstm_16/while/lstm_cell_16/MatMul/ReadVariableOp=sequential_8/lstm_16/while/lstm_cell_16/MatMul/ReadVariableOp2?
?sequential_8/lstm_16/while/lstm_cell_16/MatMul_1/ReadVariableOp?sequential_8/lstm_16/while/lstm_cell_16/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?

?
lstm_17_while_cond_32576234,
(lstm_17_while_lstm_17_while_loop_counter2
.lstm_17_while_lstm_17_while_maximum_iterations
lstm_17_while_placeholder
lstm_17_while_placeholder_1
lstm_17_while_placeholder_2
lstm_17_while_placeholder_3.
*lstm_17_while_less_lstm_17_strided_slice_1F
Blstm_17_while_lstm_17_while_cond_32576234___redundant_placeholder0F
Blstm_17_while_lstm_17_while_cond_32576234___redundant_placeholder1F
Blstm_17_while_lstm_17_while_cond_32576234___redundant_placeholder2F
Blstm_17_while_lstm_17_while_cond_32576234___redundant_placeholder3
lstm_17_while_identity
?
lstm_17/while/LessLesslstm_17_while_placeholder*lstm_17_while_less_lstm_17_strided_slice_1*
T0*
_output_shapes
: 2
lstm_17/while/Lessu
lstm_17/while/IdentityIdentitylstm_17/while/Less:z:0*
T0
*
_output_shapes
: 2
lstm_17/while/Identity"9
lstm_17_while_identitylstm_17/while/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :??????????:??????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?
?
*__inference_lstm_17_layer_call_fn_32577697
inputs_0
unknown:
??
	unknown_0:
??
	unknown_1:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:???????????????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_lstm_17_layer_call_and_return_conditional_losses_325744612
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:???????????????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':???????????????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
5
_output_shapes#
!:???????????????????
"
_user_specified_name
inputs/0
?\
?
E__inference_lstm_17_layer_call_and_return_conditional_losses_32577373
inputs_0?
+lstm_cell_17_matmul_readvariableop_resource:
??A
-lstm_cell_17_matmul_1_readvariableop_resource:
??;
,lstm_cell_17_biasadd_readvariableop_resource:	?
identity??#lstm_cell_17/BiasAdd/ReadVariableOp?"lstm_cell_17/MatMul/ReadVariableOp?$lstm_cell_17/MatMul_1/ReadVariableOp?whileF
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
strided_slice/stack_2?
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
B :?2
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
B :?2
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
B :?2
zeros/packed/1?
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
:??????????2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
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
B :?2
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
B :?2
zeros_1/packed/1?
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
:??????????2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm?
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*5
_output_shapes#
!:???????????????????2
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
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????  27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
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
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_2?
"lstm_cell_17/MatMul/ReadVariableOpReadVariableOp+lstm_cell_17_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02$
"lstm_cell_17/MatMul/ReadVariableOp?
lstm_cell_17/MatMulMatMulstrided_slice_2:output:0*lstm_cell_17/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_17/MatMul?
$lstm_cell_17/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_17_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02&
$lstm_cell_17/MatMul_1/ReadVariableOp?
lstm_cell_17/MatMul_1MatMulzeros:output:0,lstm_cell_17/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_17/MatMul_1?
lstm_cell_17/addAddV2lstm_cell_17/MatMul:product:0lstm_cell_17/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell_17/add?
#lstm_cell_17/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_17_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02%
#lstm_cell_17/BiasAdd/ReadVariableOp?
lstm_cell_17/BiasAddBiasAddlstm_cell_17/add:z:0+lstm_cell_17/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_17/BiasAdd~
lstm_cell_17/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_17/split/split_dim?
lstm_cell_17/splitSplit%lstm_cell_17/split/split_dim:output:0lstm_cell_17/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split2
lstm_cell_17/split?
lstm_cell_17/SigmoidSigmoidlstm_cell_17/split:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_17/Sigmoid?
lstm_cell_17/Sigmoid_1Sigmoidlstm_cell_17/split:output:1*
T0*(
_output_shapes
:??????????2
lstm_cell_17/Sigmoid_1?
lstm_cell_17/mulMullstm_cell_17/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_17/mul~
lstm_cell_17/ReluRelulstm_cell_17/split:output:2*
T0*(
_output_shapes
:??????????2
lstm_cell_17/Relu?
lstm_cell_17/mul_1Mullstm_cell_17/Sigmoid:y:0lstm_cell_17/Relu:activations:0*
T0*(
_output_shapes
:??????????2
lstm_cell_17/mul_1?
lstm_cell_17/add_1AddV2lstm_cell_17/mul:z:0lstm_cell_17/mul_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell_17/add_1?
lstm_cell_17/Sigmoid_2Sigmoidlstm_cell_17/split:output:3*
T0*(
_output_shapes
:??????????2
lstm_cell_17/Sigmoid_2}
lstm_cell_17/Relu_1Relulstm_cell_17/add_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell_17/Relu_1?
lstm_cell_17/mul_2Mullstm_cell_17/Sigmoid_2:y:0!lstm_cell_17/Relu_1:activations:0*
T0*(
_output_shapes
:??????????2
lstm_cell_17/mul_2?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2
TensorArrayV2_1/element_shape?
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
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_17_matmul_readvariableop_resource-lstm_cell_17_matmul_1_readvariableop_resource,lstm_cell_17_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_32577289*
condR
while_cond_32577288*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:???????????????????*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
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
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:???????????????????2
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
!:???????????????????2

Identity?
NoOpNoOp$^lstm_cell_17/BiasAdd/ReadVariableOp#^lstm_cell_17/MatMul/ReadVariableOp%^lstm_cell_17/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':???????????????????: : : 2J
#lstm_cell_17/BiasAdd/ReadVariableOp#lstm_cell_17/BiasAdd/ReadVariableOp2H
"lstm_cell_17/MatMul/ReadVariableOp"lstm_cell_17/MatMul/ReadVariableOp2L
$lstm_cell_17/MatMul_1/ReadVariableOp$lstm_cell_17/MatMul_1/ReadVariableOp2
whilewhile:_ [
5
_output_shapes#
!:???????????????????
"
_user_specified_name
inputs/0
?
?
while_cond_32576915
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_32576915___redundant_placeholder06
2while_while_cond_32576915___redundant_placeholder16
2while_while_cond_32576915___redundant_placeholder26
2while_while_cond_32576915___redundant_placeholder3
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
B: : : : :??????????:??????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?]
?
(sequential_8_lstm_17_while_body_32573351F
Bsequential_8_lstm_17_while_sequential_8_lstm_17_while_loop_counterL
Hsequential_8_lstm_17_while_sequential_8_lstm_17_while_maximum_iterations*
&sequential_8_lstm_17_while_placeholder,
(sequential_8_lstm_17_while_placeholder_1,
(sequential_8_lstm_17_while_placeholder_2,
(sequential_8_lstm_17_while_placeholder_3E
Asequential_8_lstm_17_while_sequential_8_lstm_17_strided_slice_1_0?
}sequential_8_lstm_17_while_tensorarrayv2read_tensorlistgetitem_sequential_8_lstm_17_tensorarrayunstack_tensorlistfromtensor_0\
Hsequential_8_lstm_17_while_lstm_cell_17_matmul_readvariableop_resource_0:
??^
Jsequential_8_lstm_17_while_lstm_cell_17_matmul_1_readvariableop_resource_0:
??X
Isequential_8_lstm_17_while_lstm_cell_17_biasadd_readvariableop_resource_0:	?'
#sequential_8_lstm_17_while_identity)
%sequential_8_lstm_17_while_identity_1)
%sequential_8_lstm_17_while_identity_2)
%sequential_8_lstm_17_while_identity_3)
%sequential_8_lstm_17_while_identity_4)
%sequential_8_lstm_17_while_identity_5C
?sequential_8_lstm_17_while_sequential_8_lstm_17_strided_slice_1
{sequential_8_lstm_17_while_tensorarrayv2read_tensorlistgetitem_sequential_8_lstm_17_tensorarrayunstack_tensorlistfromtensorZ
Fsequential_8_lstm_17_while_lstm_cell_17_matmul_readvariableop_resource:
??\
Hsequential_8_lstm_17_while_lstm_cell_17_matmul_1_readvariableop_resource:
??V
Gsequential_8_lstm_17_while_lstm_cell_17_biasadd_readvariableop_resource:	???>sequential_8/lstm_17/while/lstm_cell_17/BiasAdd/ReadVariableOp?=sequential_8/lstm_17/while/lstm_cell_17/MatMul/ReadVariableOp??sequential_8/lstm_17/while/lstm_cell_17/MatMul_1/ReadVariableOp?
Lsequential_8/lstm_17/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????  2N
Lsequential_8/lstm_17/while/TensorArrayV2Read/TensorListGetItem/element_shape?
>sequential_8/lstm_17/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem}sequential_8_lstm_17_while_tensorarrayv2read_tensorlistgetitem_sequential_8_lstm_17_tensorarrayunstack_tensorlistfromtensor_0&sequential_8_lstm_17_while_placeholderUsequential_8/lstm_17/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:??????????*
element_dtype02@
>sequential_8/lstm_17/while/TensorArrayV2Read/TensorListGetItem?
=sequential_8/lstm_17/while/lstm_cell_17/MatMul/ReadVariableOpReadVariableOpHsequential_8_lstm_17_while_lstm_cell_17_matmul_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02?
=sequential_8/lstm_17/while/lstm_cell_17/MatMul/ReadVariableOp?
.sequential_8/lstm_17/while/lstm_cell_17/MatMulMatMulEsequential_8/lstm_17/while/TensorArrayV2Read/TensorListGetItem:item:0Esequential_8/lstm_17/while/lstm_cell_17/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????20
.sequential_8/lstm_17/while/lstm_cell_17/MatMul?
?sequential_8/lstm_17/while/lstm_cell_17/MatMul_1/ReadVariableOpReadVariableOpJsequential_8_lstm_17_while_lstm_cell_17_matmul_1_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02A
?sequential_8/lstm_17/while/lstm_cell_17/MatMul_1/ReadVariableOp?
0sequential_8/lstm_17/while/lstm_cell_17/MatMul_1MatMul(sequential_8_lstm_17_while_placeholder_2Gsequential_8/lstm_17/while/lstm_cell_17/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????22
0sequential_8/lstm_17/while/lstm_cell_17/MatMul_1?
+sequential_8/lstm_17/while/lstm_cell_17/addAddV28sequential_8/lstm_17/while/lstm_cell_17/MatMul:product:0:sequential_8/lstm_17/while/lstm_cell_17/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2-
+sequential_8/lstm_17/while/lstm_cell_17/add?
>sequential_8/lstm_17/while/lstm_cell_17/BiasAdd/ReadVariableOpReadVariableOpIsequential_8_lstm_17_while_lstm_cell_17_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype02@
>sequential_8/lstm_17/while/lstm_cell_17/BiasAdd/ReadVariableOp?
/sequential_8/lstm_17/while/lstm_cell_17/BiasAddBiasAdd/sequential_8/lstm_17/while/lstm_cell_17/add:z:0Fsequential_8/lstm_17/while/lstm_cell_17/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????21
/sequential_8/lstm_17/while/lstm_cell_17/BiasAdd?
7sequential_8/lstm_17/while/lstm_cell_17/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :29
7sequential_8/lstm_17/while/lstm_cell_17/split/split_dim?
-sequential_8/lstm_17/while/lstm_cell_17/splitSplit@sequential_8/lstm_17/while/lstm_cell_17/split/split_dim:output:08sequential_8/lstm_17/while/lstm_cell_17/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split2/
-sequential_8/lstm_17/while/lstm_cell_17/split?
/sequential_8/lstm_17/while/lstm_cell_17/SigmoidSigmoid6sequential_8/lstm_17/while/lstm_cell_17/split:output:0*
T0*(
_output_shapes
:??????????21
/sequential_8/lstm_17/while/lstm_cell_17/Sigmoid?
1sequential_8/lstm_17/while/lstm_cell_17/Sigmoid_1Sigmoid6sequential_8/lstm_17/while/lstm_cell_17/split:output:1*
T0*(
_output_shapes
:??????????23
1sequential_8/lstm_17/while/lstm_cell_17/Sigmoid_1?
+sequential_8/lstm_17/while/lstm_cell_17/mulMul5sequential_8/lstm_17/while/lstm_cell_17/Sigmoid_1:y:0(sequential_8_lstm_17_while_placeholder_3*
T0*(
_output_shapes
:??????????2-
+sequential_8/lstm_17/while/lstm_cell_17/mul?
,sequential_8/lstm_17/while/lstm_cell_17/ReluRelu6sequential_8/lstm_17/while/lstm_cell_17/split:output:2*
T0*(
_output_shapes
:??????????2.
,sequential_8/lstm_17/while/lstm_cell_17/Relu?
-sequential_8/lstm_17/while/lstm_cell_17/mul_1Mul3sequential_8/lstm_17/while/lstm_cell_17/Sigmoid:y:0:sequential_8/lstm_17/while/lstm_cell_17/Relu:activations:0*
T0*(
_output_shapes
:??????????2/
-sequential_8/lstm_17/while/lstm_cell_17/mul_1?
-sequential_8/lstm_17/while/lstm_cell_17/add_1AddV2/sequential_8/lstm_17/while/lstm_cell_17/mul:z:01sequential_8/lstm_17/while/lstm_cell_17/mul_1:z:0*
T0*(
_output_shapes
:??????????2/
-sequential_8/lstm_17/while/lstm_cell_17/add_1?
1sequential_8/lstm_17/while/lstm_cell_17/Sigmoid_2Sigmoid6sequential_8/lstm_17/while/lstm_cell_17/split:output:3*
T0*(
_output_shapes
:??????????23
1sequential_8/lstm_17/while/lstm_cell_17/Sigmoid_2?
.sequential_8/lstm_17/while/lstm_cell_17/Relu_1Relu1sequential_8/lstm_17/while/lstm_cell_17/add_1:z:0*
T0*(
_output_shapes
:??????????20
.sequential_8/lstm_17/while/lstm_cell_17/Relu_1?
-sequential_8/lstm_17/while/lstm_cell_17/mul_2Mul5sequential_8/lstm_17/while/lstm_cell_17/Sigmoid_2:y:0<sequential_8/lstm_17/while/lstm_cell_17/Relu_1:activations:0*
T0*(
_output_shapes
:??????????2/
-sequential_8/lstm_17/while/lstm_cell_17/mul_2?
?sequential_8/lstm_17/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem(sequential_8_lstm_17_while_placeholder_1&sequential_8_lstm_17_while_placeholder1sequential_8/lstm_17/while/lstm_cell_17/mul_2:z:0*
_output_shapes
: *
element_dtype02A
?sequential_8/lstm_17/while/TensorArrayV2Write/TensorListSetItem?
 sequential_8/lstm_17/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2"
 sequential_8/lstm_17/while/add/y?
sequential_8/lstm_17/while/addAddV2&sequential_8_lstm_17_while_placeholder)sequential_8/lstm_17/while/add/y:output:0*
T0*
_output_shapes
: 2 
sequential_8/lstm_17/while/add?
"sequential_8/lstm_17/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2$
"sequential_8/lstm_17/while/add_1/y?
 sequential_8/lstm_17/while/add_1AddV2Bsequential_8_lstm_17_while_sequential_8_lstm_17_while_loop_counter+sequential_8/lstm_17/while/add_1/y:output:0*
T0*
_output_shapes
: 2"
 sequential_8/lstm_17/while/add_1?
#sequential_8/lstm_17/while/IdentityIdentity$sequential_8/lstm_17/while/add_1:z:0 ^sequential_8/lstm_17/while/NoOp*
T0*
_output_shapes
: 2%
#sequential_8/lstm_17/while/Identity?
%sequential_8/lstm_17/while/Identity_1IdentityHsequential_8_lstm_17_while_sequential_8_lstm_17_while_maximum_iterations ^sequential_8/lstm_17/while/NoOp*
T0*
_output_shapes
: 2'
%sequential_8/lstm_17/while/Identity_1?
%sequential_8/lstm_17/while/Identity_2Identity"sequential_8/lstm_17/while/add:z:0 ^sequential_8/lstm_17/while/NoOp*
T0*
_output_shapes
: 2'
%sequential_8/lstm_17/while/Identity_2?
%sequential_8/lstm_17/while/Identity_3IdentityOsequential_8/lstm_17/while/TensorArrayV2Write/TensorListSetItem:output_handle:0 ^sequential_8/lstm_17/while/NoOp*
T0*
_output_shapes
: 2'
%sequential_8/lstm_17/while/Identity_3?
%sequential_8/lstm_17/while/Identity_4Identity1sequential_8/lstm_17/while/lstm_cell_17/mul_2:z:0 ^sequential_8/lstm_17/while/NoOp*
T0*(
_output_shapes
:??????????2'
%sequential_8/lstm_17/while/Identity_4?
%sequential_8/lstm_17/while/Identity_5Identity1sequential_8/lstm_17/while/lstm_cell_17/add_1:z:0 ^sequential_8/lstm_17/while/NoOp*
T0*(
_output_shapes
:??????????2'
%sequential_8/lstm_17/while/Identity_5?
sequential_8/lstm_17/while/NoOpNoOp?^sequential_8/lstm_17/while/lstm_cell_17/BiasAdd/ReadVariableOp>^sequential_8/lstm_17/while/lstm_cell_17/MatMul/ReadVariableOp@^sequential_8/lstm_17/while/lstm_cell_17/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2!
sequential_8/lstm_17/while/NoOp"S
#sequential_8_lstm_17_while_identity,sequential_8/lstm_17/while/Identity:output:0"W
%sequential_8_lstm_17_while_identity_1.sequential_8/lstm_17/while/Identity_1:output:0"W
%sequential_8_lstm_17_while_identity_2.sequential_8/lstm_17/while/Identity_2:output:0"W
%sequential_8_lstm_17_while_identity_3.sequential_8/lstm_17/while/Identity_3:output:0"W
%sequential_8_lstm_17_while_identity_4.sequential_8/lstm_17/while/Identity_4:output:0"W
%sequential_8_lstm_17_while_identity_5.sequential_8/lstm_17/while/Identity_5:output:0"?
Gsequential_8_lstm_17_while_lstm_cell_17_biasadd_readvariableop_resourceIsequential_8_lstm_17_while_lstm_cell_17_biasadd_readvariableop_resource_0"?
Hsequential_8_lstm_17_while_lstm_cell_17_matmul_1_readvariableop_resourceJsequential_8_lstm_17_while_lstm_cell_17_matmul_1_readvariableop_resource_0"?
Fsequential_8_lstm_17_while_lstm_cell_17_matmul_readvariableop_resourceHsequential_8_lstm_17_while_lstm_cell_17_matmul_readvariableop_resource_0"?
?sequential_8_lstm_17_while_sequential_8_lstm_17_strided_slice_1Asequential_8_lstm_17_while_sequential_8_lstm_17_strided_slice_1_0"?
{sequential_8_lstm_17_while_tensorarrayv2read_tensorlistgetitem_sequential_8_lstm_17_tensorarrayunstack_tensorlistfromtensor}sequential_8_lstm_17_while_tensorarrayv2read_tensorlistgetitem_sequential_8_lstm_17_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :??????????:??????????: : : : : 2?
>sequential_8/lstm_17/while/lstm_cell_17/BiasAdd/ReadVariableOp>sequential_8/lstm_17/while/lstm_cell_17/BiasAdd/ReadVariableOp2~
=sequential_8/lstm_17/while/lstm_cell_17/MatMul/ReadVariableOp=sequential_8/lstm_17/while/lstm_cell_17/MatMul/ReadVariableOp2?
?sequential_8/lstm_17/while/lstm_cell_17/MatMul_1/ReadVariableOp?sequential_8/lstm_17/while/lstm_cell_17/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?
I
-__inference_dropout_17_layer_call_fn_32577741

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_dropout_17_layer_call_and_return_conditional_losses_325750582
PartitionedCallq
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?F
?
E__inference_lstm_16_layer_call_and_return_conditional_losses_32573621

inputs(
lstm_cell_16_32573539:	]?)
lstm_cell_16_32573541:
??$
lstm_cell_16_32573543:	?
identity??$lstm_cell_16/StatefulPartitionedCall?whileD
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
strided_slice/stack_2?
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
B :?2
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
B :?2
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
B :?2
zeros/packed/1?
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
:??????????2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
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
B :?2
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
B :?2
zeros_1/packed/1?
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
:??????????2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm?
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????]2
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
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????]   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
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
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????]*
shrink_axis_mask2
strided_slice_2?
$lstm_cell_16/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_16_32573539lstm_cell_16_32573541lstm_cell_16_32573543*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:??????????:??????????:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_lstm_cell_16_layer_call_and_return_conditional_losses_325735382&
$lstm_cell_16/StatefulPartitionedCall?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????  2
TensorArrayV2_1/element_shape?
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
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_16_32573539lstm_cell_16_32573541lstm_cell_16_32573543*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_32573552*
condR
while_cond_32573551*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????  22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:???????????????????*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
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
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:???????????????????2
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
!:???????????????????2

Identity}
NoOpNoOp%^lstm_cell_16/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????????????]: : : 2L
$lstm_cell_16/StatefulPartitionedCall$lstm_cell_16/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :??????????????????]
 
_user_specified_nameinputs
??
?
while_body_32574961
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
3while_lstm_cell_17_matmul_readvariableop_resource_0:
??I
5while_lstm_cell_17_matmul_1_readvariableop_resource_0:
??C
4while_lstm_cell_17_biasadd_readvariableop_resource_0:	?
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
1while_lstm_cell_17_matmul_readvariableop_resource:
??G
3while_lstm_cell_17_matmul_1_readvariableop_resource:
??A
2while_lstm_cell_17_biasadd_readvariableop_resource:	???)while/lstm_cell_17/BiasAdd/ReadVariableOp?(while/lstm_cell_17/MatMul/ReadVariableOp?*while/lstm_cell_17/MatMul_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????  29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:??????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
(while/lstm_cell_17/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_17_matmul_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02*
(while/lstm_cell_17/MatMul/ReadVariableOp?
while/lstm_cell_17/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_17/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_17/MatMul?
*while/lstm_cell_17/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_17_matmul_1_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02,
*while/lstm_cell_17/MatMul_1/ReadVariableOp?
while/lstm_cell_17/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_17/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_17/MatMul_1?
while/lstm_cell_17/addAddV2#while/lstm_cell_17/MatMul:product:0%while/lstm_cell_17/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_17/add?
)while/lstm_cell_17/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_17_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype02+
)while/lstm_cell_17/BiasAdd/ReadVariableOp?
while/lstm_cell_17/BiasAddBiasAddwhile/lstm_cell_17/add:z:01while/lstm_cell_17/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_17/BiasAdd?
"while/lstm_cell_17/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_17/split/split_dim?
while/lstm_cell_17/splitSplit+while/lstm_cell_17/split/split_dim:output:0#while/lstm_cell_17/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split2
while/lstm_cell_17/split?
while/lstm_cell_17/SigmoidSigmoid!while/lstm_cell_17/split:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_17/Sigmoid?
while/lstm_cell_17/Sigmoid_1Sigmoid!while/lstm_cell_17/split:output:1*
T0*(
_output_shapes
:??????????2
while/lstm_cell_17/Sigmoid_1?
while/lstm_cell_17/mulMul while/lstm_cell_17/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:??????????2
while/lstm_cell_17/mul?
while/lstm_cell_17/ReluRelu!while/lstm_cell_17/split:output:2*
T0*(
_output_shapes
:??????????2
while/lstm_cell_17/Relu?
while/lstm_cell_17/mul_1Mulwhile/lstm_cell_17/Sigmoid:y:0%while/lstm_cell_17/Relu:activations:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_17/mul_1?
while/lstm_cell_17/add_1AddV2while/lstm_cell_17/mul:z:0while/lstm_cell_17/mul_1:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_17/add_1?
while/lstm_cell_17/Sigmoid_2Sigmoid!while/lstm_cell_17/split:output:3*
T0*(
_output_shapes
:??????????2
while/lstm_cell_17/Sigmoid_2?
while/lstm_cell_17/Relu_1Reluwhile/lstm_cell_17/add_1:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_17/Relu_1?
while/lstm_cell_17/mul_2Mul while/lstm_cell_17/Sigmoid_2:y:0'while/lstm_cell_17/Relu_1:activations:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_17/mul_2?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_17/mul_2:z:0*
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
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identitywhile/lstm_cell_17/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:??????????2
while/Identity_4?
while/Identity_5Identitywhile/lstm_cell_17/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:??????????2
while/Identity_5?

while/NoOpNoOp*^while/lstm_cell_17/BiasAdd/ReadVariableOp)^while/lstm_cell_17/MatMul/ReadVariableOp+^while/lstm_cell_17/MatMul_1/ReadVariableOp*"
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
2while_lstm_cell_17_biasadd_readvariableop_resource4while_lstm_cell_17_biasadd_readvariableop_resource_0"l
3while_lstm_cell_17_matmul_1_readvariableop_resource5while_lstm_cell_17_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_17_matmul_readvariableop_resource3while_lstm_cell_17_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :??????????:??????????: : : : : 2V
)while/lstm_cell_17/BiasAdd/ReadVariableOp)while/lstm_cell_17/BiasAdd/ReadVariableOp2T
(while/lstm_cell_17/MatMul/ReadVariableOp(while/lstm_cell_17/MatMul/ReadVariableOp2X
*while/lstm_cell_17/MatMul_1/ReadVariableOp*while/lstm_cell_17/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?\
?
E__inference_lstm_17_layer_call_and_return_conditional_losses_32575045

inputs?
+lstm_cell_17_matmul_readvariableop_resource:
??A
-lstm_cell_17_matmul_1_readvariableop_resource:
??;
,lstm_cell_17_biasadd_readvariableop_resource:	?
identity??#lstm_cell_17/BiasAdd/ReadVariableOp?"lstm_cell_17/MatMul/ReadVariableOp?$lstm_cell_17/MatMul_1/ReadVariableOp?whileD
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
strided_slice/stack_2?
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
B :?2
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
B :?2
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
B :?2
zeros/packed/1?
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
:??????????2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
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
B :?2
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
B :?2
zeros_1/packed/1?
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
:??????????2	
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
:??????????2
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
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????  27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
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
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_2?
"lstm_cell_17/MatMul/ReadVariableOpReadVariableOp+lstm_cell_17_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02$
"lstm_cell_17/MatMul/ReadVariableOp?
lstm_cell_17/MatMulMatMulstrided_slice_2:output:0*lstm_cell_17/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_17/MatMul?
$lstm_cell_17/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_17_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02&
$lstm_cell_17/MatMul_1/ReadVariableOp?
lstm_cell_17/MatMul_1MatMulzeros:output:0,lstm_cell_17/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_17/MatMul_1?
lstm_cell_17/addAddV2lstm_cell_17/MatMul:product:0lstm_cell_17/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell_17/add?
#lstm_cell_17/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_17_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02%
#lstm_cell_17/BiasAdd/ReadVariableOp?
lstm_cell_17/BiasAddBiasAddlstm_cell_17/add:z:0+lstm_cell_17/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_17/BiasAdd~
lstm_cell_17/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_17/split/split_dim?
lstm_cell_17/splitSplit%lstm_cell_17/split/split_dim:output:0lstm_cell_17/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split2
lstm_cell_17/split?
lstm_cell_17/SigmoidSigmoidlstm_cell_17/split:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_17/Sigmoid?
lstm_cell_17/Sigmoid_1Sigmoidlstm_cell_17/split:output:1*
T0*(
_output_shapes
:??????????2
lstm_cell_17/Sigmoid_1?
lstm_cell_17/mulMullstm_cell_17/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_17/mul~
lstm_cell_17/ReluRelulstm_cell_17/split:output:2*
T0*(
_output_shapes
:??????????2
lstm_cell_17/Relu?
lstm_cell_17/mul_1Mullstm_cell_17/Sigmoid:y:0lstm_cell_17/Relu:activations:0*
T0*(
_output_shapes
:??????????2
lstm_cell_17/mul_1?
lstm_cell_17/add_1AddV2lstm_cell_17/mul:z:0lstm_cell_17/mul_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell_17/add_1?
lstm_cell_17/Sigmoid_2Sigmoidlstm_cell_17/split:output:3*
T0*(
_output_shapes
:??????????2
lstm_cell_17/Sigmoid_2}
lstm_cell_17/Relu_1Relulstm_cell_17/add_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell_17/Relu_1?
lstm_cell_17/mul_2Mullstm_cell_17/Sigmoid_2:y:0!lstm_cell_17/Relu_1:activations:0*
T0*(
_output_shapes
:??????????2
lstm_cell_17/mul_2?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2
TensorArrayV2_1/element_shape?
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
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_17_matmul_readvariableop_resource-lstm_cell_17_matmul_1_readvariableop_resource,lstm_cell_17_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_32574961*
condR
while_cond_32574960*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:??????????*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
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
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:??????????2
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
:??????????2

Identity?
NoOpNoOp$^lstm_cell_17/BiasAdd/ReadVariableOp#^lstm_cell_17/MatMul/ReadVariableOp%^lstm_cell_17/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:??????????: : : 2J
#lstm_cell_17/BiasAdd/ReadVariableOp#lstm_cell_17/BiasAdd/ReadVariableOp2H
"lstm_cell_17/MatMul/ReadVariableOp"lstm_cell_17/MatMul/ReadVariableOp2L
$lstm_cell_17/MatMul_1/ReadVariableOp$lstm_cell_17/MatMul_1/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
*__inference_lstm_17_layer_call_fn_32577708

inputs
unknown:
??
	unknown_0:
??
	unknown_1:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_lstm_17_layer_call_and_return_conditional_losses_325750452
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:??????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:??????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
/__inference_lstm_cell_16_layer_call_fn_32577867

inputs
states_0
states_1
unknown:	]?
	unknown_0:
??
	unknown_1:	?
identity

identity_1

identity_2??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:??????????:??????????:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_lstm_cell_16_layer_call_and_return_conditional_losses_325735382
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:??????????2

Identity_1?

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:??????????2

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
A:?????????]:??????????:??????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????]
 
_user_specified_nameinputs:RN
(
_output_shapes
:??????????
"
_user_specified_name
states/0:RN
(
_output_shapes
:??????????
"
_user_specified_name
states/1
?
?
J__inference_lstm_cell_17_layer_call_and_return_conditional_losses_32574314

inputs

states
states_12
matmul_readvariableop_resource:
??4
 matmul_1_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity

identity_1

identity_2??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?MatMul_1/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
add?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split2
split`
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:??????????2	
Sigmoidd
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:??????????2
	Sigmoid_1]
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:??????????2
mulW
ReluRelusplit:output:2*
T0*(
_output_shapes
:??????????2
Relui
mul_1MulSigmoid:y:0Relu:activations:0*
T0*(
_output_shapes
:??????????2
mul_1^
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:??????????2
add_1d
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:??????????2
	Sigmoid_2V
Relu_1Relu	add_1:z:0*
T0*(
_output_shapes
:??????????2
Relu_1m
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*(
_output_shapes
:??????????2
mul_2e
IdentityIdentity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:??????????2

Identityi

Identity_1Identity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity_1i

Identity_2Identity	add_1:z:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity_2?
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
B:??????????:??????????:??????????: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:PL
(
_output_shapes
:??????????
 
_user_specified_namestates:PL
(
_output_shapes
:??????????
 
_user_specified_namestates
?

?
/__inference_sequential_8_layer_call_fn_32576375

inputs
unknown:	]?
	unknown_0:
??
	unknown_1:	?
	unknown_2:
??
	unknown_3:
??
	unknown_4:	?
	unknown_5:	?
	unknown_6:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_sequential_8_layer_call_and_return_conditional_losses_325750982
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????]: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????]
 
_user_specified_nameinputs
?
?
J__inference_lstm_cell_16_layer_call_and_return_conditional_losses_32577850

inputs
states_0
states_11
matmul_readvariableop_resource:	]?4
 matmul_1_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity

identity_1

identity_2??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?MatMul_1/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	]?*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul_1/ReadVariableOp|
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
add?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split2
split`
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:??????????2	
Sigmoidd
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:??????????2
	Sigmoid_1]
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:??????????2
mulW
ReluRelusplit:output:2*
T0*(
_output_shapes
:??????????2
Relui
mul_1MulSigmoid:y:0Relu:activations:0*
T0*(
_output_shapes
:??????????2
mul_1^
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:??????????2
add_1d
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:??????????2
	Sigmoid_2V
Relu_1Relu	add_1:z:0*
T0*(
_output_shapes
:??????????2
Relu_1m
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*(
_output_shapes
:??????????2
mul_2e
IdentityIdentity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:??????????2

Identityi

Identity_1Identity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity_1i

Identity_2Identity	add_1:z:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity_2?
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
A:?????????]:??????????:??????????: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:?????????]
 
_user_specified_nameinputs:RN
(
_output_shapes
:??????????
"
_user_specified_name
states/0:RN
(
_output_shapes
:??????????
"
_user_specified_name
states/1
?L
?
!__inference__traced_save_32578104
file_prefix-
)savev2_dense_8_kernel_read_readvariableop+
'savev2_dense_8_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop:
6savev2_lstm_16_lstm_cell_16_kernel_read_readvariableopD
@savev2_lstm_16_lstm_cell_16_recurrent_kernel_read_readvariableop8
4savev2_lstm_16_lstm_cell_16_bias_read_readvariableop:
6savev2_lstm_17_lstm_cell_17_kernel_read_readvariableopD
@savev2_lstm_17_lstm_cell_17_recurrent_kernel_read_readvariableop8
4savev2_lstm_17_lstm_cell_17_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop4
0savev2_adam_dense_8_kernel_m_read_readvariableop2
.savev2_adam_dense_8_bias_m_read_readvariableopA
=savev2_adam_lstm_16_lstm_cell_16_kernel_m_read_readvariableopK
Gsavev2_adam_lstm_16_lstm_cell_16_recurrent_kernel_m_read_readvariableop?
;savev2_adam_lstm_16_lstm_cell_16_bias_m_read_readvariableopA
=savev2_adam_lstm_17_lstm_cell_17_kernel_m_read_readvariableopK
Gsavev2_adam_lstm_17_lstm_cell_17_recurrent_kernel_m_read_readvariableop?
;savev2_adam_lstm_17_lstm_cell_17_bias_m_read_readvariableop4
0savev2_adam_dense_8_kernel_v_read_readvariableop2
.savev2_adam_dense_8_bias_v_read_readvariableopA
=savev2_adam_lstm_16_lstm_cell_16_kernel_v_read_readvariableopK
Gsavev2_adam_lstm_16_lstm_cell_16_recurrent_kernel_v_read_readvariableop?
;savev2_adam_lstm_16_lstm_cell_16_bias_v_read_readvariableopA
=savev2_adam_lstm_17_lstm_cell_17_kernel_v_read_readvariableopK
Gsavev2_adam_lstm_17_lstm_cell_17_recurrent_kernel_v_read_readvariableop?
;savev2_adam_lstm_17_lstm_cell_17_bias_v_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
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
Const_1?
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
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:"*
dtype0*?
value?B?"B6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:"*
dtype0*W
valueNBL"B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0)savev2_dense_8_kernel_read_readvariableop'savev2_dense_8_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop6savev2_lstm_16_lstm_cell_16_kernel_read_readvariableop@savev2_lstm_16_lstm_cell_16_recurrent_kernel_read_readvariableop4savev2_lstm_16_lstm_cell_16_bias_read_readvariableop6savev2_lstm_17_lstm_cell_17_kernel_read_readvariableop@savev2_lstm_17_lstm_cell_17_recurrent_kernel_read_readvariableop4savev2_lstm_17_lstm_cell_17_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop0savev2_adam_dense_8_kernel_m_read_readvariableop.savev2_adam_dense_8_bias_m_read_readvariableop=savev2_adam_lstm_16_lstm_cell_16_kernel_m_read_readvariableopGsavev2_adam_lstm_16_lstm_cell_16_recurrent_kernel_m_read_readvariableop;savev2_adam_lstm_16_lstm_cell_16_bias_m_read_readvariableop=savev2_adam_lstm_17_lstm_cell_17_kernel_m_read_readvariableopGsavev2_adam_lstm_17_lstm_cell_17_recurrent_kernel_m_read_readvariableop;savev2_adam_lstm_17_lstm_cell_17_bias_m_read_readvariableop0savev2_adam_dense_8_kernel_v_read_readvariableop.savev2_adam_dense_8_bias_v_read_readvariableop=savev2_adam_lstm_16_lstm_cell_16_kernel_v_read_readvariableopGsavev2_adam_lstm_16_lstm_cell_16_recurrent_kernel_v_read_readvariableop;savev2_adam_lstm_16_lstm_cell_16_bias_v_read_readvariableop=savev2_adam_lstm_17_lstm_cell_17_kernel_v_read_readvariableopGsavev2_adam_lstm_17_lstm_cell_17_recurrent_kernel_v_read_readvariableop;savev2_adam_lstm_17_lstm_cell_17_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *0
dtypes&
$2"	2
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
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

identity_1Identity_1:output:0*?
_input_shapes?
?: :	?:: : : : : :	]?:
??:?:
??:
??:?: : : : :	?::	]?:
??:?:
??:
??:?:	?::	]?:
??:?:
??:
??:?: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	?: 
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
:	]?:&	"
 
_output_shapes
:
??:!


_output_shapes	
:?:&"
 
_output_shapes
:
??:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:
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
:	?: 

_output_shapes
::%!

_output_shapes
:	]?:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:&"
 
_output_shapes
:
??:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:%!

_output_shapes
:	?: 

_output_shapes
::%!

_output_shapes
:	]?:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:&"
 
_output_shapes
:
??:& "
 
_output_shapes
:
??:!!

_output_shapes	
:?:"

_output_shapes
: 
?\
?
E__inference_lstm_16_layer_call_and_return_conditional_losses_32574880

inputs>
+lstm_cell_16_matmul_readvariableop_resource:	]?A
-lstm_cell_16_matmul_1_readvariableop_resource:
??;
,lstm_cell_16_biasadd_readvariableop_resource:	?
identity??#lstm_cell_16/BiasAdd/ReadVariableOp?"lstm_cell_16/MatMul/ReadVariableOp?$lstm_cell_16/MatMul_1/ReadVariableOp?whileD
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
strided_slice/stack_2?
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
B :?2
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
B :?2
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
B :?2
zeros/packed/1?
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
:??????????2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
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
B :?2
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
B :?2
zeros_1/packed/1?
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
:??????????2	
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
:?????????]2
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
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????]   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
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
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????]*
shrink_axis_mask2
strided_slice_2?
"lstm_cell_16/MatMul/ReadVariableOpReadVariableOp+lstm_cell_16_matmul_readvariableop_resource*
_output_shapes
:	]?*
dtype02$
"lstm_cell_16/MatMul/ReadVariableOp?
lstm_cell_16/MatMulMatMulstrided_slice_2:output:0*lstm_cell_16/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_16/MatMul?
$lstm_cell_16/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_16_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02&
$lstm_cell_16/MatMul_1/ReadVariableOp?
lstm_cell_16/MatMul_1MatMulzeros:output:0,lstm_cell_16/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_16/MatMul_1?
lstm_cell_16/addAddV2lstm_cell_16/MatMul:product:0lstm_cell_16/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell_16/add?
#lstm_cell_16/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_16_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02%
#lstm_cell_16/BiasAdd/ReadVariableOp?
lstm_cell_16/BiasAddBiasAddlstm_cell_16/add:z:0+lstm_cell_16/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_16/BiasAdd~
lstm_cell_16/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_16/split/split_dim?
lstm_cell_16/splitSplit%lstm_cell_16/split/split_dim:output:0lstm_cell_16/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split2
lstm_cell_16/split?
lstm_cell_16/SigmoidSigmoidlstm_cell_16/split:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_16/Sigmoid?
lstm_cell_16/Sigmoid_1Sigmoidlstm_cell_16/split:output:1*
T0*(
_output_shapes
:??????????2
lstm_cell_16/Sigmoid_1?
lstm_cell_16/mulMullstm_cell_16/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_16/mul~
lstm_cell_16/ReluRelulstm_cell_16/split:output:2*
T0*(
_output_shapes
:??????????2
lstm_cell_16/Relu?
lstm_cell_16/mul_1Mullstm_cell_16/Sigmoid:y:0lstm_cell_16/Relu:activations:0*
T0*(
_output_shapes
:??????????2
lstm_cell_16/mul_1?
lstm_cell_16/add_1AddV2lstm_cell_16/mul:z:0lstm_cell_16/mul_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell_16/add_1?
lstm_cell_16/Sigmoid_2Sigmoidlstm_cell_16/split:output:3*
T0*(
_output_shapes
:??????????2
lstm_cell_16/Sigmoid_2}
lstm_cell_16/Relu_1Relulstm_cell_16/add_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell_16/Relu_1?
lstm_cell_16/mul_2Mullstm_cell_16/Sigmoid_2:y:0!lstm_cell_16/Relu_1:activations:0*
T0*(
_output_shapes
:??????????2
lstm_cell_16/mul_2?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????  2
TensorArrayV2_1/element_shape?
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
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_16_matmul_readvariableop_resource-lstm_cell_16_matmul_1_readvariableop_resource,lstm_cell_16_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_32574796*
condR
while_cond_32574795*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????  22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:??????????*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
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
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:??????????2
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
:??????????2

Identity?
NoOpNoOp$^lstm_cell_16/BiasAdd/ReadVariableOp#^lstm_cell_16/MatMul/ReadVariableOp%^lstm_cell_16/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????]: : : 2J
#lstm_cell_16/BiasAdd/ReadVariableOp#lstm_cell_16/BiasAdd/ReadVariableOp2H
"lstm_cell_16/MatMul/ReadVariableOp"lstm_cell_16/MatMul/ReadVariableOp2L
$lstm_cell_16/MatMul_1/ReadVariableOp$lstm_cell_16/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:?????????]
 
_user_specified_nameinputs
??
?
while_body_32574796
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_16_matmul_readvariableop_resource_0:	]?I
5while_lstm_cell_16_matmul_1_readvariableop_resource_0:
??C
4while_lstm_cell_16_biasadd_readvariableop_resource_0:	?
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_16_matmul_readvariableop_resource:	]?G
3while_lstm_cell_16_matmul_1_readvariableop_resource:
??A
2while_lstm_cell_16_biasadd_readvariableop_resource:	???)while/lstm_cell_16/BiasAdd/ReadVariableOp?(while/lstm_cell_16/MatMul/ReadVariableOp?*while/lstm_cell_16/MatMul_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????]   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????]*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
(while/lstm_cell_16/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_16_matmul_readvariableop_resource_0*
_output_shapes
:	]?*
dtype02*
(while/lstm_cell_16/MatMul/ReadVariableOp?
while/lstm_cell_16/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_16/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_16/MatMul?
*while/lstm_cell_16/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_16_matmul_1_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02,
*while/lstm_cell_16/MatMul_1/ReadVariableOp?
while/lstm_cell_16/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_16/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_16/MatMul_1?
while/lstm_cell_16/addAddV2#while/lstm_cell_16/MatMul:product:0%while/lstm_cell_16/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_16/add?
)while/lstm_cell_16/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_16_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype02+
)while/lstm_cell_16/BiasAdd/ReadVariableOp?
while/lstm_cell_16/BiasAddBiasAddwhile/lstm_cell_16/add:z:01while/lstm_cell_16/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_16/BiasAdd?
"while/lstm_cell_16/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_16/split/split_dim?
while/lstm_cell_16/splitSplit+while/lstm_cell_16/split/split_dim:output:0#while/lstm_cell_16/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split2
while/lstm_cell_16/split?
while/lstm_cell_16/SigmoidSigmoid!while/lstm_cell_16/split:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_16/Sigmoid?
while/lstm_cell_16/Sigmoid_1Sigmoid!while/lstm_cell_16/split:output:1*
T0*(
_output_shapes
:??????????2
while/lstm_cell_16/Sigmoid_1?
while/lstm_cell_16/mulMul while/lstm_cell_16/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:??????????2
while/lstm_cell_16/mul?
while/lstm_cell_16/ReluRelu!while/lstm_cell_16/split:output:2*
T0*(
_output_shapes
:??????????2
while/lstm_cell_16/Relu?
while/lstm_cell_16/mul_1Mulwhile/lstm_cell_16/Sigmoid:y:0%while/lstm_cell_16/Relu:activations:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_16/mul_1?
while/lstm_cell_16/add_1AddV2while/lstm_cell_16/mul:z:0while/lstm_cell_16/mul_1:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_16/add_1?
while/lstm_cell_16/Sigmoid_2Sigmoid!while/lstm_cell_16/split:output:3*
T0*(
_output_shapes
:??????????2
while/lstm_cell_16/Sigmoid_2?
while/lstm_cell_16/Relu_1Reluwhile/lstm_cell_16/add_1:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_16/Relu_1?
while/lstm_cell_16/mul_2Mul while/lstm_cell_16/Sigmoid_2:y:0'while/lstm_cell_16/Relu_1:activations:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_16/mul_2?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_16/mul_2:z:0*
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
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identitywhile/lstm_cell_16/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:??????????2
while/Identity_4?
while/Identity_5Identitywhile/lstm_cell_16/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:??????????2
while/Identity_5?

while/NoOpNoOp*^while/lstm_cell_16/BiasAdd/ReadVariableOp)^while/lstm_cell_16/MatMul/ReadVariableOp+^while/lstm_cell_16/MatMul_1/ReadVariableOp*"
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
2while_lstm_cell_16_biasadd_readvariableop_resource4while_lstm_cell_16_biasadd_readvariableop_resource_0"l
3while_lstm_cell_16_matmul_1_readvariableop_resource5while_lstm_cell_16_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_16_matmul_readvariableop_resource3while_lstm_cell_16_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :??????????:??????????: : : : : 2V
)while/lstm_cell_16/BiasAdd/ReadVariableOp)while/lstm_cell_16/BiasAdd/ReadVariableOp2T
(while/lstm_cell_16/MatMul/ReadVariableOp(while/lstm_cell_16/MatMul/ReadVariableOp2X
*while/lstm_cell_16/MatMul_1/ReadVariableOp*while/lstm_cell_16/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?
?
while_cond_32575425
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_32575425___redundant_placeholder06
2while_while_cond_32575425___redundant_placeholder16
2while_while_cond_32575425___redundant_placeholder26
2while_while_cond_32575425___redundant_placeholder3
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
B: : : : :??????????:??????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?J
?

lstm_16_while_body_32576080,
(lstm_16_while_lstm_16_while_loop_counter2
.lstm_16_while_lstm_16_while_maximum_iterations
lstm_16_while_placeholder
lstm_16_while_placeholder_1
lstm_16_while_placeholder_2
lstm_16_while_placeholder_3+
'lstm_16_while_lstm_16_strided_slice_1_0g
clstm_16_while_tensorarrayv2read_tensorlistgetitem_lstm_16_tensorarrayunstack_tensorlistfromtensor_0N
;lstm_16_while_lstm_cell_16_matmul_readvariableop_resource_0:	]?Q
=lstm_16_while_lstm_cell_16_matmul_1_readvariableop_resource_0:
??K
<lstm_16_while_lstm_cell_16_biasadd_readvariableop_resource_0:	?
lstm_16_while_identity
lstm_16_while_identity_1
lstm_16_while_identity_2
lstm_16_while_identity_3
lstm_16_while_identity_4
lstm_16_while_identity_5)
%lstm_16_while_lstm_16_strided_slice_1e
alstm_16_while_tensorarrayv2read_tensorlistgetitem_lstm_16_tensorarrayunstack_tensorlistfromtensorL
9lstm_16_while_lstm_cell_16_matmul_readvariableop_resource:	]?O
;lstm_16_while_lstm_cell_16_matmul_1_readvariableop_resource:
??I
:lstm_16_while_lstm_cell_16_biasadd_readvariableop_resource:	???1lstm_16/while/lstm_cell_16/BiasAdd/ReadVariableOp?0lstm_16/while/lstm_cell_16/MatMul/ReadVariableOp?2lstm_16/while/lstm_cell_16/MatMul_1/ReadVariableOp?
?lstm_16/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????]   2A
?lstm_16/while/TensorArrayV2Read/TensorListGetItem/element_shape?
1lstm_16/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemclstm_16_while_tensorarrayv2read_tensorlistgetitem_lstm_16_tensorarrayunstack_tensorlistfromtensor_0lstm_16_while_placeholderHlstm_16/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????]*
element_dtype023
1lstm_16/while/TensorArrayV2Read/TensorListGetItem?
0lstm_16/while/lstm_cell_16/MatMul/ReadVariableOpReadVariableOp;lstm_16_while_lstm_cell_16_matmul_readvariableop_resource_0*
_output_shapes
:	]?*
dtype022
0lstm_16/while/lstm_cell_16/MatMul/ReadVariableOp?
!lstm_16/while/lstm_cell_16/MatMulMatMul8lstm_16/while/TensorArrayV2Read/TensorListGetItem:item:08lstm_16/while/lstm_cell_16/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2#
!lstm_16/while/lstm_cell_16/MatMul?
2lstm_16/while/lstm_cell_16/MatMul_1/ReadVariableOpReadVariableOp=lstm_16_while_lstm_cell_16_matmul_1_readvariableop_resource_0* 
_output_shapes
:
??*
dtype024
2lstm_16/while/lstm_cell_16/MatMul_1/ReadVariableOp?
#lstm_16/while/lstm_cell_16/MatMul_1MatMullstm_16_while_placeholder_2:lstm_16/while/lstm_cell_16/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2%
#lstm_16/while/lstm_cell_16/MatMul_1?
lstm_16/while/lstm_cell_16/addAddV2+lstm_16/while/lstm_cell_16/MatMul:product:0-lstm_16/while/lstm_cell_16/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2 
lstm_16/while/lstm_cell_16/add?
1lstm_16/while/lstm_cell_16/BiasAdd/ReadVariableOpReadVariableOp<lstm_16_while_lstm_cell_16_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype023
1lstm_16/while/lstm_cell_16/BiasAdd/ReadVariableOp?
"lstm_16/while/lstm_cell_16/BiasAddBiasAdd"lstm_16/while/lstm_cell_16/add:z:09lstm_16/while/lstm_cell_16/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2$
"lstm_16/while/lstm_cell_16/BiasAdd?
*lstm_16/while/lstm_cell_16/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2,
*lstm_16/while/lstm_cell_16/split/split_dim?
 lstm_16/while/lstm_cell_16/splitSplit3lstm_16/while/lstm_cell_16/split/split_dim:output:0+lstm_16/while/lstm_cell_16/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split2"
 lstm_16/while/lstm_cell_16/split?
"lstm_16/while/lstm_cell_16/SigmoidSigmoid)lstm_16/while/lstm_cell_16/split:output:0*
T0*(
_output_shapes
:??????????2$
"lstm_16/while/lstm_cell_16/Sigmoid?
$lstm_16/while/lstm_cell_16/Sigmoid_1Sigmoid)lstm_16/while/lstm_cell_16/split:output:1*
T0*(
_output_shapes
:??????????2&
$lstm_16/while/lstm_cell_16/Sigmoid_1?
lstm_16/while/lstm_cell_16/mulMul(lstm_16/while/lstm_cell_16/Sigmoid_1:y:0lstm_16_while_placeholder_3*
T0*(
_output_shapes
:??????????2 
lstm_16/while/lstm_cell_16/mul?
lstm_16/while/lstm_cell_16/ReluRelu)lstm_16/while/lstm_cell_16/split:output:2*
T0*(
_output_shapes
:??????????2!
lstm_16/while/lstm_cell_16/Relu?
 lstm_16/while/lstm_cell_16/mul_1Mul&lstm_16/while/lstm_cell_16/Sigmoid:y:0-lstm_16/while/lstm_cell_16/Relu:activations:0*
T0*(
_output_shapes
:??????????2"
 lstm_16/while/lstm_cell_16/mul_1?
 lstm_16/while/lstm_cell_16/add_1AddV2"lstm_16/while/lstm_cell_16/mul:z:0$lstm_16/while/lstm_cell_16/mul_1:z:0*
T0*(
_output_shapes
:??????????2"
 lstm_16/while/lstm_cell_16/add_1?
$lstm_16/while/lstm_cell_16/Sigmoid_2Sigmoid)lstm_16/while/lstm_cell_16/split:output:3*
T0*(
_output_shapes
:??????????2&
$lstm_16/while/lstm_cell_16/Sigmoid_2?
!lstm_16/while/lstm_cell_16/Relu_1Relu$lstm_16/while/lstm_cell_16/add_1:z:0*
T0*(
_output_shapes
:??????????2#
!lstm_16/while/lstm_cell_16/Relu_1?
 lstm_16/while/lstm_cell_16/mul_2Mul(lstm_16/while/lstm_cell_16/Sigmoid_2:y:0/lstm_16/while/lstm_cell_16/Relu_1:activations:0*
T0*(
_output_shapes
:??????????2"
 lstm_16/while/lstm_cell_16/mul_2?
2lstm_16/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_16_while_placeholder_1lstm_16_while_placeholder$lstm_16/while/lstm_cell_16/mul_2:z:0*
_output_shapes
: *
element_dtype024
2lstm_16/while/TensorArrayV2Write/TensorListSetIteml
lstm_16/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_16/while/add/y?
lstm_16/while/addAddV2lstm_16_while_placeholderlstm_16/while/add/y:output:0*
T0*
_output_shapes
: 2
lstm_16/while/addp
lstm_16/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_16/while/add_1/y?
lstm_16/while/add_1AddV2(lstm_16_while_lstm_16_while_loop_counterlstm_16/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lstm_16/while/add_1?
lstm_16/while/IdentityIdentitylstm_16/while/add_1:z:0^lstm_16/while/NoOp*
T0*
_output_shapes
: 2
lstm_16/while/Identity?
lstm_16/while/Identity_1Identity.lstm_16_while_lstm_16_while_maximum_iterations^lstm_16/while/NoOp*
T0*
_output_shapes
: 2
lstm_16/while/Identity_1?
lstm_16/while/Identity_2Identitylstm_16/while/add:z:0^lstm_16/while/NoOp*
T0*
_output_shapes
: 2
lstm_16/while/Identity_2?
lstm_16/while/Identity_3IdentityBlstm_16/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_16/while/NoOp*
T0*
_output_shapes
: 2
lstm_16/while/Identity_3?
lstm_16/while/Identity_4Identity$lstm_16/while/lstm_cell_16/mul_2:z:0^lstm_16/while/NoOp*
T0*(
_output_shapes
:??????????2
lstm_16/while/Identity_4?
lstm_16/while/Identity_5Identity$lstm_16/while/lstm_cell_16/add_1:z:0^lstm_16/while/NoOp*
T0*(
_output_shapes
:??????????2
lstm_16/while/Identity_5?
lstm_16/while/NoOpNoOp2^lstm_16/while/lstm_cell_16/BiasAdd/ReadVariableOp1^lstm_16/while/lstm_cell_16/MatMul/ReadVariableOp3^lstm_16/while/lstm_cell_16/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
lstm_16/while/NoOp"9
lstm_16_while_identitylstm_16/while/Identity:output:0"=
lstm_16_while_identity_1!lstm_16/while/Identity_1:output:0"=
lstm_16_while_identity_2!lstm_16/while/Identity_2:output:0"=
lstm_16_while_identity_3!lstm_16/while/Identity_3:output:0"=
lstm_16_while_identity_4!lstm_16/while/Identity_4:output:0"=
lstm_16_while_identity_5!lstm_16/while/Identity_5:output:0"P
%lstm_16_while_lstm_16_strided_slice_1'lstm_16_while_lstm_16_strided_slice_1_0"z
:lstm_16_while_lstm_cell_16_biasadd_readvariableop_resource<lstm_16_while_lstm_cell_16_biasadd_readvariableop_resource_0"|
;lstm_16_while_lstm_cell_16_matmul_1_readvariableop_resource=lstm_16_while_lstm_cell_16_matmul_1_readvariableop_resource_0"x
9lstm_16_while_lstm_cell_16_matmul_readvariableop_resource;lstm_16_while_lstm_cell_16_matmul_readvariableop_resource_0"?
alstm_16_while_tensorarrayv2read_tensorlistgetitem_lstm_16_tensorarrayunstack_tensorlistfromtensorclstm_16_while_tensorarrayv2read_tensorlistgetitem_lstm_16_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :??????????:??????????: : : : : 2f
1lstm_16/while/lstm_cell_16/BiasAdd/ReadVariableOp1lstm_16/while/lstm_cell_16/BiasAdd/ReadVariableOp2d
0lstm_16/while/lstm_cell_16/MatMul/ReadVariableOp0lstm_16/while/lstm_cell_16/MatMul/ReadVariableOp2h
2lstm_16/while/lstm_cell_16/MatMul_1/ReadVariableOp2lstm_16/while/lstm_cell_16/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?
?
*__inference_lstm_16_layer_call_fn_32577033

inputs
unknown:	]?
	unknown_0:
??
	unknown_1:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_lstm_16_layer_call_and_return_conditional_losses_325748802
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:??????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????]: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????]
 
_user_specified_nameinputs
?
?
J__inference_lstm_cell_17_layer_call_and_return_conditional_losses_32574168

inputs

states
states_12
matmul_readvariableop_resource:
??4
 matmul_1_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity

identity_1

identity_2??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?MatMul_1/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
add?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split2
split`
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:??????????2	
Sigmoidd
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:??????????2
	Sigmoid_1]
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:??????????2
mulW
ReluRelusplit:output:2*
T0*(
_output_shapes
:??????????2
Relui
mul_1MulSigmoid:y:0Relu:activations:0*
T0*(
_output_shapes
:??????????2
mul_1^
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:??????????2
add_1d
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:??????????2
	Sigmoid_2V
Relu_1Relu	add_1:z:0*
T0*(
_output_shapes
:??????????2
Relu_1m
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*(
_output_shapes
:??????????2
mul_2e
IdentityIdentity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:??????????2

Identityi

Identity_1Identity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity_1i

Identity_2Identity	add_1:z:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity_2?
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
B:??????????:??????????:??????????: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:PL
(
_output_shapes
:??????????
 
_user_specified_namestates:PL
(
_output_shapes
:??????????
 
_user_specified_namestates
?

?
lstm_16_while_cond_32576079,
(lstm_16_while_lstm_16_while_loop_counter2
.lstm_16_while_lstm_16_while_maximum_iterations
lstm_16_while_placeholder
lstm_16_while_placeholder_1
lstm_16_while_placeholder_2
lstm_16_while_placeholder_3.
*lstm_16_while_less_lstm_16_strided_slice_1F
Blstm_16_while_lstm_16_while_cond_32576079___redundant_placeholder0F
Blstm_16_while_lstm_16_while_cond_32576079___redundant_placeholder1F
Blstm_16_while_lstm_16_while_cond_32576079___redundant_placeholder2F
Blstm_16_while_lstm_16_while_cond_32576079___redundant_placeholder3
lstm_16_while_identity
?
lstm_16/while/LessLesslstm_16_while_placeholder*lstm_16_while_less_lstm_16_strided_slice_1*
T0*
_output_shapes
: 2
lstm_16/while/Lessu
lstm_16/while/IdentityIdentitylstm_16/while/Less:z:0*
T0
*
_output_shapes
: 2
lstm_16/while/Identity"9
lstm_16_while_identitylstm_16/while/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :??????????:??????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
:
?\
?
E__inference_lstm_17_layer_call_and_return_conditional_losses_32577222
inputs_0?
+lstm_cell_17_matmul_readvariableop_resource:
??A
-lstm_cell_17_matmul_1_readvariableop_resource:
??;
,lstm_cell_17_biasadd_readvariableop_resource:	?
identity??#lstm_cell_17/BiasAdd/ReadVariableOp?"lstm_cell_17/MatMul/ReadVariableOp?$lstm_cell_17/MatMul_1/ReadVariableOp?whileF
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
strided_slice/stack_2?
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
B :?2
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
B :?2
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
B :?2
zeros/packed/1?
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
:??????????2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
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
B :?2
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
B :?2
zeros_1/packed/1?
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
:??????????2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm?
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*5
_output_shapes#
!:???????????????????2
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
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????  27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
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
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_2?
"lstm_cell_17/MatMul/ReadVariableOpReadVariableOp+lstm_cell_17_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02$
"lstm_cell_17/MatMul/ReadVariableOp?
lstm_cell_17/MatMulMatMulstrided_slice_2:output:0*lstm_cell_17/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_17/MatMul?
$lstm_cell_17/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_17_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02&
$lstm_cell_17/MatMul_1/ReadVariableOp?
lstm_cell_17/MatMul_1MatMulzeros:output:0,lstm_cell_17/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_17/MatMul_1?
lstm_cell_17/addAddV2lstm_cell_17/MatMul:product:0lstm_cell_17/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell_17/add?
#lstm_cell_17/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_17_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02%
#lstm_cell_17/BiasAdd/ReadVariableOp?
lstm_cell_17/BiasAddBiasAddlstm_cell_17/add:z:0+lstm_cell_17/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_17/BiasAdd~
lstm_cell_17/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_17/split/split_dim?
lstm_cell_17/splitSplit%lstm_cell_17/split/split_dim:output:0lstm_cell_17/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split2
lstm_cell_17/split?
lstm_cell_17/SigmoidSigmoidlstm_cell_17/split:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_17/Sigmoid?
lstm_cell_17/Sigmoid_1Sigmoidlstm_cell_17/split:output:1*
T0*(
_output_shapes
:??????????2
lstm_cell_17/Sigmoid_1?
lstm_cell_17/mulMullstm_cell_17/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_17/mul~
lstm_cell_17/ReluRelulstm_cell_17/split:output:2*
T0*(
_output_shapes
:??????????2
lstm_cell_17/Relu?
lstm_cell_17/mul_1Mullstm_cell_17/Sigmoid:y:0lstm_cell_17/Relu:activations:0*
T0*(
_output_shapes
:??????????2
lstm_cell_17/mul_1?
lstm_cell_17/add_1AddV2lstm_cell_17/mul:z:0lstm_cell_17/mul_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell_17/add_1?
lstm_cell_17/Sigmoid_2Sigmoidlstm_cell_17/split:output:3*
T0*(
_output_shapes
:??????????2
lstm_cell_17/Sigmoid_2}
lstm_cell_17/Relu_1Relulstm_cell_17/add_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell_17/Relu_1?
lstm_cell_17/mul_2Mullstm_cell_17/Sigmoid_2:y:0!lstm_cell_17/Relu_1:activations:0*
T0*(
_output_shapes
:??????????2
lstm_cell_17/mul_2?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2
TensorArrayV2_1/element_shape?
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
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_17_matmul_readvariableop_resource-lstm_cell_17_matmul_1_readvariableop_resource,lstm_cell_17_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_32577138*
condR
while_cond_32577137*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:???????????????????*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
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
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:???????????????????2
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
!:???????????????????2

Identity?
NoOpNoOp$^lstm_cell_17/BiasAdd/ReadVariableOp#^lstm_cell_17/MatMul/ReadVariableOp%^lstm_cell_17/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':???????????????????: : : 2J
#lstm_cell_17/BiasAdd/ReadVariableOp#lstm_cell_17/BiasAdd/ReadVariableOp2H
"lstm_cell_17/MatMul/ReadVariableOp"lstm_cell_17/MatMul/ReadVariableOp2L
$lstm_cell_17/MatMul_1/ReadVariableOp$lstm_cell_17/MatMul_1/ReadVariableOp2
whilewhile:_ [
5
_output_shapes#
!:???????????????????
"
_user_specified_name
inputs/0
?

?
/__inference_sequential_8_layer_call_fn_32576396

inputs
unknown:	]?
	unknown_0:
??
	unknown_1:	?
	unknown_2:
??
	unknown_3:
??
	unknown_4:	?
	unknown_5:	?
	unknown_6:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_sequential_8_layer_call_and_return_conditional_losses_325755672
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????]: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????]
 
_user_specified_nameinputs
?
f
H__inference_dropout_16_layer_call_and_return_conditional_losses_32577049

inputs

identity_1_
IdentityIdentityinputs*
T0*,
_output_shapes
:??????????2

Identityn

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:??????????2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?\
?
E__inference_lstm_17_layer_call_and_return_conditional_losses_32577524

inputs?
+lstm_cell_17_matmul_readvariableop_resource:
??A
-lstm_cell_17_matmul_1_readvariableop_resource:
??;
,lstm_cell_17_biasadd_readvariableop_resource:	?
identity??#lstm_cell_17/BiasAdd/ReadVariableOp?"lstm_cell_17/MatMul/ReadVariableOp?$lstm_cell_17/MatMul_1/ReadVariableOp?whileD
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
strided_slice/stack_2?
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
B :?2
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
B :?2
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
B :?2
zeros/packed/1?
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
:??????????2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
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
B :?2
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
B :?2
zeros_1/packed/1?
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
:??????????2	
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
:??????????2
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
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????  27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
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
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_2?
"lstm_cell_17/MatMul/ReadVariableOpReadVariableOp+lstm_cell_17_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02$
"lstm_cell_17/MatMul/ReadVariableOp?
lstm_cell_17/MatMulMatMulstrided_slice_2:output:0*lstm_cell_17/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_17/MatMul?
$lstm_cell_17/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_17_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02&
$lstm_cell_17/MatMul_1/ReadVariableOp?
lstm_cell_17/MatMul_1MatMulzeros:output:0,lstm_cell_17/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_17/MatMul_1?
lstm_cell_17/addAddV2lstm_cell_17/MatMul:product:0lstm_cell_17/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell_17/add?
#lstm_cell_17/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_17_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02%
#lstm_cell_17/BiasAdd/ReadVariableOp?
lstm_cell_17/BiasAddBiasAddlstm_cell_17/add:z:0+lstm_cell_17/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_17/BiasAdd~
lstm_cell_17/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_17/split/split_dim?
lstm_cell_17/splitSplit%lstm_cell_17/split/split_dim:output:0lstm_cell_17/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split2
lstm_cell_17/split?
lstm_cell_17/SigmoidSigmoidlstm_cell_17/split:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_17/Sigmoid?
lstm_cell_17/Sigmoid_1Sigmoidlstm_cell_17/split:output:1*
T0*(
_output_shapes
:??????????2
lstm_cell_17/Sigmoid_1?
lstm_cell_17/mulMullstm_cell_17/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_17/mul~
lstm_cell_17/ReluRelulstm_cell_17/split:output:2*
T0*(
_output_shapes
:??????????2
lstm_cell_17/Relu?
lstm_cell_17/mul_1Mullstm_cell_17/Sigmoid:y:0lstm_cell_17/Relu:activations:0*
T0*(
_output_shapes
:??????????2
lstm_cell_17/mul_1?
lstm_cell_17/add_1AddV2lstm_cell_17/mul:z:0lstm_cell_17/mul_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell_17/add_1?
lstm_cell_17/Sigmoid_2Sigmoidlstm_cell_17/split:output:3*
T0*(
_output_shapes
:??????????2
lstm_cell_17/Sigmoid_2}
lstm_cell_17/Relu_1Relulstm_cell_17/add_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell_17/Relu_1?
lstm_cell_17/mul_2Mullstm_cell_17/Sigmoid_2:y:0!lstm_cell_17/Relu_1:activations:0*
T0*(
_output_shapes
:??????????2
lstm_cell_17/mul_2?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2
TensorArrayV2_1/element_shape?
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
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_17_matmul_readvariableop_resource-lstm_cell_17_matmul_1_readvariableop_resource,lstm_cell_17_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_32577440*
condR
while_cond_32577439*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:??????????*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
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
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:??????????2
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
:??????????2

Identity?
NoOpNoOp$^lstm_cell_17/BiasAdd/ReadVariableOp#^lstm_cell_17/MatMul/ReadVariableOp%^lstm_cell_17/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:??????????: : : 2J
#lstm_cell_17/BiasAdd/ReadVariableOp#lstm_cell_17/BiasAdd/ReadVariableOp2H
"lstm_cell_17/MatMul/ReadVariableOp"lstm_cell_17/MatMul/ReadVariableOp2L
$lstm_cell_17/MatMul_1/ReadVariableOp$lstm_cell_17/MatMul_1/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
*__inference_lstm_17_layer_call_fn_32577686
inputs_0
unknown:
??
	unknown_0:
??
	unknown_1:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:???????????????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_lstm_17_layer_call_and_return_conditional_losses_325742512
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:???????????????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':???????????????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
5
_output_shapes#
!:???????????????????
"
_user_specified_name
inputs/0
?
?
J__inference_lstm_cell_17_layer_call_and_return_conditional_losses_32577948

inputs
states_0
states_12
matmul_readvariableop_resource:
??4
 matmul_1_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity

identity_1

identity_2??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?MatMul_1/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul_1/ReadVariableOp|
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
add?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split2
split`
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:??????????2	
Sigmoidd
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:??????????2
	Sigmoid_1]
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:??????????2
mulW
ReluRelusplit:output:2*
T0*(
_output_shapes
:??????????2
Relui
mul_1MulSigmoid:y:0Relu:activations:0*
T0*(
_output_shapes
:??????????2
mul_1^
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:??????????2
add_1d
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:??????????2
	Sigmoid_2V
Relu_1Relu	add_1:z:0*
T0*(
_output_shapes
:??????????2
Relu_1m
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*(
_output_shapes
:??????????2
mul_2e
IdentityIdentity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:??????????2

Identityi

Identity_1Identity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity_1i

Identity_2Identity	add_1:z:0^NoOp*
T0*(
_output_shapes
:??????????2

Identity_2?
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
B:??????????:??????????:??????????: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:RN
(
_output_shapes
:??????????
"
_user_specified_name
states/0:RN
(
_output_shapes
:??????????
"
_user_specified_name
states/1
?J
?

lstm_17_while_body_32575901,
(lstm_17_while_lstm_17_while_loop_counter2
.lstm_17_while_lstm_17_while_maximum_iterations
lstm_17_while_placeholder
lstm_17_while_placeholder_1
lstm_17_while_placeholder_2
lstm_17_while_placeholder_3+
'lstm_17_while_lstm_17_strided_slice_1_0g
clstm_17_while_tensorarrayv2read_tensorlistgetitem_lstm_17_tensorarrayunstack_tensorlistfromtensor_0O
;lstm_17_while_lstm_cell_17_matmul_readvariableop_resource_0:
??Q
=lstm_17_while_lstm_cell_17_matmul_1_readvariableop_resource_0:
??K
<lstm_17_while_lstm_cell_17_biasadd_readvariableop_resource_0:	?
lstm_17_while_identity
lstm_17_while_identity_1
lstm_17_while_identity_2
lstm_17_while_identity_3
lstm_17_while_identity_4
lstm_17_while_identity_5)
%lstm_17_while_lstm_17_strided_slice_1e
alstm_17_while_tensorarrayv2read_tensorlistgetitem_lstm_17_tensorarrayunstack_tensorlistfromtensorM
9lstm_17_while_lstm_cell_17_matmul_readvariableop_resource:
??O
;lstm_17_while_lstm_cell_17_matmul_1_readvariableop_resource:
??I
:lstm_17_while_lstm_cell_17_biasadd_readvariableop_resource:	???1lstm_17/while/lstm_cell_17/BiasAdd/ReadVariableOp?0lstm_17/while/lstm_cell_17/MatMul/ReadVariableOp?2lstm_17/while/lstm_cell_17/MatMul_1/ReadVariableOp?
?lstm_17/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????  2A
?lstm_17/while/TensorArrayV2Read/TensorListGetItem/element_shape?
1lstm_17/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemclstm_17_while_tensorarrayv2read_tensorlistgetitem_lstm_17_tensorarrayunstack_tensorlistfromtensor_0lstm_17_while_placeholderHlstm_17/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:??????????*
element_dtype023
1lstm_17/while/TensorArrayV2Read/TensorListGetItem?
0lstm_17/while/lstm_cell_17/MatMul/ReadVariableOpReadVariableOp;lstm_17_while_lstm_cell_17_matmul_readvariableop_resource_0* 
_output_shapes
:
??*
dtype022
0lstm_17/while/lstm_cell_17/MatMul/ReadVariableOp?
!lstm_17/while/lstm_cell_17/MatMulMatMul8lstm_17/while/TensorArrayV2Read/TensorListGetItem:item:08lstm_17/while/lstm_cell_17/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2#
!lstm_17/while/lstm_cell_17/MatMul?
2lstm_17/while/lstm_cell_17/MatMul_1/ReadVariableOpReadVariableOp=lstm_17_while_lstm_cell_17_matmul_1_readvariableop_resource_0* 
_output_shapes
:
??*
dtype024
2lstm_17/while/lstm_cell_17/MatMul_1/ReadVariableOp?
#lstm_17/while/lstm_cell_17/MatMul_1MatMullstm_17_while_placeholder_2:lstm_17/while/lstm_cell_17/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2%
#lstm_17/while/lstm_cell_17/MatMul_1?
lstm_17/while/lstm_cell_17/addAddV2+lstm_17/while/lstm_cell_17/MatMul:product:0-lstm_17/while/lstm_cell_17/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2 
lstm_17/while/lstm_cell_17/add?
1lstm_17/while/lstm_cell_17/BiasAdd/ReadVariableOpReadVariableOp<lstm_17_while_lstm_cell_17_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype023
1lstm_17/while/lstm_cell_17/BiasAdd/ReadVariableOp?
"lstm_17/while/lstm_cell_17/BiasAddBiasAdd"lstm_17/while/lstm_cell_17/add:z:09lstm_17/while/lstm_cell_17/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2$
"lstm_17/while/lstm_cell_17/BiasAdd?
*lstm_17/while/lstm_cell_17/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2,
*lstm_17/while/lstm_cell_17/split/split_dim?
 lstm_17/while/lstm_cell_17/splitSplit3lstm_17/while/lstm_cell_17/split/split_dim:output:0+lstm_17/while/lstm_cell_17/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split2"
 lstm_17/while/lstm_cell_17/split?
"lstm_17/while/lstm_cell_17/SigmoidSigmoid)lstm_17/while/lstm_cell_17/split:output:0*
T0*(
_output_shapes
:??????????2$
"lstm_17/while/lstm_cell_17/Sigmoid?
$lstm_17/while/lstm_cell_17/Sigmoid_1Sigmoid)lstm_17/while/lstm_cell_17/split:output:1*
T0*(
_output_shapes
:??????????2&
$lstm_17/while/lstm_cell_17/Sigmoid_1?
lstm_17/while/lstm_cell_17/mulMul(lstm_17/while/lstm_cell_17/Sigmoid_1:y:0lstm_17_while_placeholder_3*
T0*(
_output_shapes
:??????????2 
lstm_17/while/lstm_cell_17/mul?
lstm_17/while/lstm_cell_17/ReluRelu)lstm_17/while/lstm_cell_17/split:output:2*
T0*(
_output_shapes
:??????????2!
lstm_17/while/lstm_cell_17/Relu?
 lstm_17/while/lstm_cell_17/mul_1Mul&lstm_17/while/lstm_cell_17/Sigmoid:y:0-lstm_17/while/lstm_cell_17/Relu:activations:0*
T0*(
_output_shapes
:??????????2"
 lstm_17/while/lstm_cell_17/mul_1?
 lstm_17/while/lstm_cell_17/add_1AddV2"lstm_17/while/lstm_cell_17/mul:z:0$lstm_17/while/lstm_cell_17/mul_1:z:0*
T0*(
_output_shapes
:??????????2"
 lstm_17/while/lstm_cell_17/add_1?
$lstm_17/while/lstm_cell_17/Sigmoid_2Sigmoid)lstm_17/while/lstm_cell_17/split:output:3*
T0*(
_output_shapes
:??????????2&
$lstm_17/while/lstm_cell_17/Sigmoid_2?
!lstm_17/while/lstm_cell_17/Relu_1Relu$lstm_17/while/lstm_cell_17/add_1:z:0*
T0*(
_output_shapes
:??????????2#
!lstm_17/while/lstm_cell_17/Relu_1?
 lstm_17/while/lstm_cell_17/mul_2Mul(lstm_17/while/lstm_cell_17/Sigmoid_2:y:0/lstm_17/while/lstm_cell_17/Relu_1:activations:0*
T0*(
_output_shapes
:??????????2"
 lstm_17/while/lstm_cell_17/mul_2?
2lstm_17/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_17_while_placeholder_1lstm_17_while_placeholder$lstm_17/while/lstm_cell_17/mul_2:z:0*
_output_shapes
: *
element_dtype024
2lstm_17/while/TensorArrayV2Write/TensorListSetIteml
lstm_17/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_17/while/add/y?
lstm_17/while/addAddV2lstm_17_while_placeholderlstm_17/while/add/y:output:0*
T0*
_output_shapes
: 2
lstm_17/while/addp
lstm_17/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_17/while/add_1/y?
lstm_17/while/add_1AddV2(lstm_17_while_lstm_17_while_loop_counterlstm_17/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lstm_17/while/add_1?
lstm_17/while/IdentityIdentitylstm_17/while/add_1:z:0^lstm_17/while/NoOp*
T0*
_output_shapes
: 2
lstm_17/while/Identity?
lstm_17/while/Identity_1Identity.lstm_17_while_lstm_17_while_maximum_iterations^lstm_17/while/NoOp*
T0*
_output_shapes
: 2
lstm_17/while/Identity_1?
lstm_17/while/Identity_2Identitylstm_17/while/add:z:0^lstm_17/while/NoOp*
T0*
_output_shapes
: 2
lstm_17/while/Identity_2?
lstm_17/while/Identity_3IdentityBlstm_17/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_17/while/NoOp*
T0*
_output_shapes
: 2
lstm_17/while/Identity_3?
lstm_17/while/Identity_4Identity$lstm_17/while/lstm_cell_17/mul_2:z:0^lstm_17/while/NoOp*
T0*(
_output_shapes
:??????????2
lstm_17/while/Identity_4?
lstm_17/while/Identity_5Identity$lstm_17/while/lstm_cell_17/add_1:z:0^lstm_17/while/NoOp*
T0*(
_output_shapes
:??????????2
lstm_17/while/Identity_5?
lstm_17/while/NoOpNoOp2^lstm_17/while/lstm_cell_17/BiasAdd/ReadVariableOp1^lstm_17/while/lstm_cell_17/MatMul/ReadVariableOp3^lstm_17/while/lstm_cell_17/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
lstm_17/while/NoOp"9
lstm_17_while_identitylstm_17/while/Identity:output:0"=
lstm_17_while_identity_1!lstm_17/while/Identity_1:output:0"=
lstm_17_while_identity_2!lstm_17/while/Identity_2:output:0"=
lstm_17_while_identity_3!lstm_17/while/Identity_3:output:0"=
lstm_17_while_identity_4!lstm_17/while/Identity_4:output:0"=
lstm_17_while_identity_5!lstm_17/while/Identity_5:output:0"P
%lstm_17_while_lstm_17_strided_slice_1'lstm_17_while_lstm_17_strided_slice_1_0"z
:lstm_17_while_lstm_cell_17_biasadd_readvariableop_resource<lstm_17_while_lstm_cell_17_biasadd_readvariableop_resource_0"|
;lstm_17_while_lstm_cell_17_matmul_1_readvariableop_resource=lstm_17_while_lstm_cell_17_matmul_1_readvariableop_resource_0"x
9lstm_17_while_lstm_cell_17_matmul_readvariableop_resource;lstm_17_while_lstm_cell_17_matmul_readvariableop_resource_0"?
alstm_17_while_tensorarrayv2read_tensorlistgetitem_lstm_17_tensorarrayunstack_tensorlistfromtensorclstm_17_while_tensorarrayv2read_tensorlistgetitem_lstm_17_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :??????????:??????????: : : : : 2f
1lstm_17/while/lstm_cell_17/BiasAdd/ReadVariableOp1lstm_17/while/lstm_cell_17/BiasAdd/ReadVariableOp2d
0lstm_17/while/lstm_cell_17/MatMul/ReadVariableOp0lstm_17/while/lstm_cell_17/MatMul/ReadVariableOp2h
2lstm_17/while/lstm_cell_17/MatMul_1/ReadVariableOp2lstm_17/while/lstm_cell_17/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
??
?
while_body_32575230
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
3while_lstm_cell_17_matmul_readvariableop_resource_0:
??I
5while_lstm_cell_17_matmul_1_readvariableop_resource_0:
??C
4while_lstm_cell_17_biasadd_readvariableop_resource_0:	?
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
1while_lstm_cell_17_matmul_readvariableop_resource:
??G
3while_lstm_cell_17_matmul_1_readvariableop_resource:
??A
2while_lstm_cell_17_biasadd_readvariableop_resource:	???)while/lstm_cell_17/BiasAdd/ReadVariableOp?(while/lstm_cell_17/MatMul/ReadVariableOp?*while/lstm_cell_17/MatMul_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????  29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:??????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
(while/lstm_cell_17/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_17_matmul_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02*
(while/lstm_cell_17/MatMul/ReadVariableOp?
while/lstm_cell_17/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_17/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_17/MatMul?
*while/lstm_cell_17/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_17_matmul_1_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02,
*while/lstm_cell_17/MatMul_1/ReadVariableOp?
while/lstm_cell_17/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_17/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_17/MatMul_1?
while/lstm_cell_17/addAddV2#while/lstm_cell_17/MatMul:product:0%while/lstm_cell_17/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_17/add?
)while/lstm_cell_17/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_17_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype02+
)while/lstm_cell_17/BiasAdd/ReadVariableOp?
while/lstm_cell_17/BiasAddBiasAddwhile/lstm_cell_17/add:z:01while/lstm_cell_17/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_17/BiasAdd?
"while/lstm_cell_17/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_17/split/split_dim?
while/lstm_cell_17/splitSplit+while/lstm_cell_17/split/split_dim:output:0#while/lstm_cell_17/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split2
while/lstm_cell_17/split?
while/lstm_cell_17/SigmoidSigmoid!while/lstm_cell_17/split:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_17/Sigmoid?
while/lstm_cell_17/Sigmoid_1Sigmoid!while/lstm_cell_17/split:output:1*
T0*(
_output_shapes
:??????????2
while/lstm_cell_17/Sigmoid_1?
while/lstm_cell_17/mulMul while/lstm_cell_17/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:??????????2
while/lstm_cell_17/mul?
while/lstm_cell_17/ReluRelu!while/lstm_cell_17/split:output:2*
T0*(
_output_shapes
:??????????2
while/lstm_cell_17/Relu?
while/lstm_cell_17/mul_1Mulwhile/lstm_cell_17/Sigmoid:y:0%while/lstm_cell_17/Relu:activations:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_17/mul_1?
while/lstm_cell_17/add_1AddV2while/lstm_cell_17/mul:z:0while/lstm_cell_17/mul_1:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_17/add_1?
while/lstm_cell_17/Sigmoid_2Sigmoid!while/lstm_cell_17/split:output:3*
T0*(
_output_shapes
:??????????2
while/lstm_cell_17/Sigmoid_2?
while/lstm_cell_17/Relu_1Reluwhile/lstm_cell_17/add_1:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_17/Relu_1?
while/lstm_cell_17/mul_2Mul while/lstm_cell_17/Sigmoid_2:y:0'while/lstm_cell_17/Relu_1:activations:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_17/mul_2?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_17/mul_2:z:0*
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
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identitywhile/lstm_cell_17/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:??????????2
while/Identity_4?
while/Identity_5Identitywhile/lstm_cell_17/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:??????????2
while/Identity_5?

while/NoOpNoOp*^while/lstm_cell_17/BiasAdd/ReadVariableOp)^while/lstm_cell_17/MatMul/ReadVariableOp+^while/lstm_cell_17/MatMul_1/ReadVariableOp*"
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
2while_lstm_cell_17_biasadd_readvariableop_resource4while_lstm_cell_17_biasadd_readvariableop_resource_0"l
3while_lstm_cell_17_matmul_1_readvariableop_resource5while_lstm_cell_17_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_17_matmul_readvariableop_resource3while_lstm_cell_17_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :??????????:??????????: : : : : 2V
)while/lstm_cell_17/BiasAdd/ReadVariableOp)while/lstm_cell_17/BiasAdd/ReadVariableOp2T
(while/lstm_cell_17/MatMul/ReadVariableOp(while/lstm_cell_17/MatMul/ReadVariableOp2X
*while/lstm_cell_17/MatMul_1/ReadVariableOp*while/lstm_cell_17/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
?
?
J__inference_sequential_8_layer_call_and_return_conditional_losses_32575567

inputs#
lstm_16_32575545:	]?$
lstm_16_32575547:
??
lstm_16_32575549:	?$
lstm_17_32575553:
??$
lstm_17_32575555:
??
lstm_17_32575557:	?#
dense_8_32575561:	?
dense_8_32575563:
identity??dense_8/StatefulPartitionedCall?"dropout_16/StatefulPartitionedCall?"dropout_17/StatefulPartitionedCall?lstm_16/StatefulPartitionedCall?lstm_17/StatefulPartitionedCall?
lstm_16/StatefulPartitionedCallStatefulPartitionedCallinputslstm_16_32575545lstm_16_32575547lstm_16_32575549*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_lstm_16_layer_call_and_return_conditional_losses_325755102!
lstm_16/StatefulPartitionedCall?
"dropout_16/StatefulPartitionedCallStatefulPartitionedCall(lstm_16/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_dropout_16_layer_call_and_return_conditional_losses_325753432$
"dropout_16/StatefulPartitionedCall?
lstm_17/StatefulPartitionedCallStatefulPartitionedCall+dropout_16/StatefulPartitionedCall:output:0lstm_17_32575553lstm_17_32575555lstm_17_32575557*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_lstm_17_layer_call_and_return_conditional_losses_325753142!
lstm_17/StatefulPartitionedCall?
"dropout_17/StatefulPartitionedCallStatefulPartitionedCall(lstm_17/StatefulPartitionedCall:output:0#^dropout_16/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_dropout_17_layer_call_and_return_conditional_losses_325751472$
"dropout_17/StatefulPartitionedCall?
dense_8/StatefulPartitionedCallStatefulPartitionedCall+dropout_17/StatefulPartitionedCall:output:0dense_8_32575561dense_8_32575563*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_8_layer_call_and_return_conditional_losses_325750912!
dense_8/StatefulPartitionedCall?
IdentityIdentity(dense_8/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????2

Identity?
NoOpNoOp ^dense_8/StatefulPartitionedCall#^dropout_16/StatefulPartitionedCall#^dropout_17/StatefulPartitionedCall ^lstm_16/StatefulPartitionedCall ^lstm_17/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????]: : : : : : : : 2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall2H
"dropout_16/StatefulPartitionedCall"dropout_16/StatefulPartitionedCall2H
"dropout_17/StatefulPartitionedCall"dropout_17/StatefulPartitionedCall2B
lstm_16/StatefulPartitionedCalllstm_16/StatefulPartitionedCall2B
lstm_17/StatefulPartitionedCalllstm_17/StatefulPartitionedCall:S O
+
_output_shapes
:?????????]
 
_user_specified_nameinputs
?F
?
E__inference_lstm_17_layer_call_and_return_conditional_losses_32574461

inputs)
lstm_cell_17_32574379:
??)
lstm_cell_17_32574381:
??$
lstm_cell_17_32574383:	?
identity??$lstm_cell_17/StatefulPartitionedCall?whileD
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
strided_slice/stack_2?
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
B :?2
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
B :?2
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
B :?2
zeros/packed/1?
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
:??????????2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
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
B :?2
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
B :?2
zeros_1/packed/1?
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
:??????????2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm?
	transpose	Transposeinputstranspose/perm:output:0*
T0*5
_output_shapes#
!:???????????????????2
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
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????  27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
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
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_2?
$lstm_cell_17/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_17_32574379lstm_cell_17_32574381lstm_cell_17_32574383*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:??????????:??????????:??????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_lstm_cell_17_layer_call_and_return_conditional_losses_325743142&
$lstm_cell_17/StatefulPartitionedCall?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2
TensorArrayV2_1/element_shape?
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
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_17_32574379lstm_cell_17_32574381lstm_cell_17_32574383*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_32574392*
condR
while_cond_32574391*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:???????????????????*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
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
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:???????????????????2
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
!:???????????????????2

Identity}
NoOpNoOp%^lstm_cell_17/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':???????????????????: : : 2L
$lstm_cell_17/StatefulPartitionedCall$lstm_cell_17/StatefulPartitionedCall2
whilewhile:] Y
5
_output_shapes#
!:???????????????????
 
_user_specified_nameinputs
??
?
while_body_32577289
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
3while_lstm_cell_17_matmul_readvariableop_resource_0:
??I
5while_lstm_cell_17_matmul_1_readvariableop_resource_0:
??C
4while_lstm_cell_17_biasadd_readvariableop_resource_0:	?
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
1while_lstm_cell_17_matmul_readvariableop_resource:
??G
3while_lstm_cell_17_matmul_1_readvariableop_resource:
??A
2while_lstm_cell_17_biasadd_readvariableop_resource:	???)while/lstm_cell_17/BiasAdd/ReadVariableOp?(while/lstm_cell_17/MatMul/ReadVariableOp?*while/lstm_cell_17/MatMul_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????  29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:??????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
(while/lstm_cell_17/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_17_matmul_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02*
(while/lstm_cell_17/MatMul/ReadVariableOp?
while/lstm_cell_17/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_17/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_17/MatMul?
*while/lstm_cell_17/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_17_matmul_1_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02,
*while/lstm_cell_17/MatMul_1/ReadVariableOp?
while/lstm_cell_17/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_17/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_17/MatMul_1?
while/lstm_cell_17/addAddV2#while/lstm_cell_17/MatMul:product:0%while/lstm_cell_17/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_17/add?
)while/lstm_cell_17/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_17_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype02+
)while/lstm_cell_17/BiasAdd/ReadVariableOp?
while/lstm_cell_17/BiasAddBiasAddwhile/lstm_cell_17/add:z:01while/lstm_cell_17/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_17/BiasAdd?
"while/lstm_cell_17/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_17/split/split_dim?
while/lstm_cell_17/splitSplit+while/lstm_cell_17/split/split_dim:output:0#while/lstm_cell_17/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split2
while/lstm_cell_17/split?
while/lstm_cell_17/SigmoidSigmoid!while/lstm_cell_17/split:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_17/Sigmoid?
while/lstm_cell_17/Sigmoid_1Sigmoid!while/lstm_cell_17/split:output:1*
T0*(
_output_shapes
:??????????2
while/lstm_cell_17/Sigmoid_1?
while/lstm_cell_17/mulMul while/lstm_cell_17/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:??????????2
while/lstm_cell_17/mul?
while/lstm_cell_17/ReluRelu!while/lstm_cell_17/split:output:2*
T0*(
_output_shapes
:??????????2
while/lstm_cell_17/Relu?
while/lstm_cell_17/mul_1Mulwhile/lstm_cell_17/Sigmoid:y:0%while/lstm_cell_17/Relu:activations:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_17/mul_1?
while/lstm_cell_17/add_1AddV2while/lstm_cell_17/mul:z:0while/lstm_cell_17/mul_1:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_17/add_1?
while/lstm_cell_17/Sigmoid_2Sigmoid!while/lstm_cell_17/split:output:3*
T0*(
_output_shapes
:??????????2
while/lstm_cell_17/Sigmoid_2?
while/lstm_cell_17/Relu_1Reluwhile/lstm_cell_17/add_1:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_17/Relu_1?
while/lstm_cell_17/mul_2Mul while/lstm_cell_17/Sigmoid_2:y:0'while/lstm_cell_17/Relu_1:activations:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_17/mul_2?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_17/mul_2:z:0*
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
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identitywhile/lstm_cell_17/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:??????????2
while/Identity_4?
while/Identity_5Identitywhile/lstm_cell_17/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:??????????2
while/Identity_5?

while/NoOpNoOp*^while/lstm_cell_17/BiasAdd/ReadVariableOp)^while/lstm_cell_17/MatMul/ReadVariableOp+^while/lstm_cell_17/MatMul_1/ReadVariableOp*"
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
2while_lstm_cell_17_biasadd_readvariableop_resource4while_lstm_cell_17_biasadd_readvariableop_resource_0"l
3while_lstm_cell_17_matmul_1_readvariableop_resource5while_lstm_cell_17_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_17_matmul_readvariableop_resource3while_lstm_cell_17_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :??????????:??????????: : : : : 2V
)while/lstm_cell_17/BiasAdd/ReadVariableOp)while/lstm_cell_17/BiasAdd/ReadVariableOp2T
(while/lstm_cell_17/MatMul/ReadVariableOp(while/lstm_cell_17/MatMul/ReadVariableOp2X
*while/lstm_cell_17/MatMul_1/ReadVariableOp*while/lstm_cell_17/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:??????????:.*
(
_output_shapes
:??????????:

_output_shapes
: :

_output_shapes
: 
??
?	
#__inference__wrapped_model_32573463
lstm_16_inputS
@sequential_8_lstm_16_lstm_cell_16_matmul_readvariableop_resource:	]?V
Bsequential_8_lstm_16_lstm_cell_16_matmul_1_readvariableop_resource:
??P
Asequential_8_lstm_16_lstm_cell_16_biasadd_readvariableop_resource:	?T
@sequential_8_lstm_17_lstm_cell_17_matmul_readvariableop_resource:
??V
Bsequential_8_lstm_17_lstm_cell_17_matmul_1_readvariableop_resource:
??P
Asequential_8_lstm_17_lstm_cell_17_biasadd_readvariableop_resource:	?I
6sequential_8_dense_8_tensordot_readvariableop_resource:	?B
4sequential_8_dense_8_biasadd_readvariableop_resource:
identity??+sequential_8/dense_8/BiasAdd/ReadVariableOp?-sequential_8/dense_8/Tensordot/ReadVariableOp?8sequential_8/lstm_16/lstm_cell_16/BiasAdd/ReadVariableOp?7sequential_8/lstm_16/lstm_cell_16/MatMul/ReadVariableOp?9sequential_8/lstm_16/lstm_cell_16/MatMul_1/ReadVariableOp?sequential_8/lstm_16/while?8sequential_8/lstm_17/lstm_cell_17/BiasAdd/ReadVariableOp?7sequential_8/lstm_17/lstm_cell_17/MatMul/ReadVariableOp?9sequential_8/lstm_17/lstm_cell_17/MatMul_1/ReadVariableOp?sequential_8/lstm_17/whileu
sequential_8/lstm_16/ShapeShapelstm_16_input*
T0*
_output_shapes
:2
sequential_8/lstm_16/Shape?
(sequential_8/lstm_16/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(sequential_8/lstm_16/strided_slice/stack?
*sequential_8/lstm_16/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*sequential_8/lstm_16/strided_slice/stack_1?
*sequential_8/lstm_16/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*sequential_8/lstm_16/strided_slice/stack_2?
"sequential_8/lstm_16/strided_sliceStridedSlice#sequential_8/lstm_16/Shape:output:01sequential_8/lstm_16/strided_slice/stack:output:03sequential_8/lstm_16/strided_slice/stack_1:output:03sequential_8/lstm_16/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"sequential_8/lstm_16/strided_slice?
 sequential_8/lstm_16/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2"
 sequential_8/lstm_16/zeros/mul/y?
sequential_8/lstm_16/zeros/mulMul+sequential_8/lstm_16/strided_slice:output:0)sequential_8/lstm_16/zeros/mul/y:output:0*
T0*
_output_shapes
: 2 
sequential_8/lstm_16/zeros/mul?
!sequential_8/lstm_16/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2#
!sequential_8/lstm_16/zeros/Less/y?
sequential_8/lstm_16/zeros/LessLess"sequential_8/lstm_16/zeros/mul:z:0*sequential_8/lstm_16/zeros/Less/y:output:0*
T0*
_output_shapes
: 2!
sequential_8/lstm_16/zeros/Less?
#sequential_8/lstm_16/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2%
#sequential_8/lstm_16/zeros/packed/1?
!sequential_8/lstm_16/zeros/packedPack+sequential_8/lstm_16/strided_slice:output:0,sequential_8/lstm_16/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2#
!sequential_8/lstm_16/zeros/packed?
 sequential_8/lstm_16/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 sequential_8/lstm_16/zeros/Const?
sequential_8/lstm_16/zerosFill*sequential_8/lstm_16/zeros/packed:output:0)sequential_8/lstm_16/zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
sequential_8/lstm_16/zeros?
"sequential_8/lstm_16/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2$
"sequential_8/lstm_16/zeros_1/mul/y?
 sequential_8/lstm_16/zeros_1/mulMul+sequential_8/lstm_16/strided_slice:output:0+sequential_8/lstm_16/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2"
 sequential_8/lstm_16/zeros_1/mul?
#sequential_8/lstm_16/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2%
#sequential_8/lstm_16/zeros_1/Less/y?
!sequential_8/lstm_16/zeros_1/LessLess$sequential_8/lstm_16/zeros_1/mul:z:0,sequential_8/lstm_16/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2#
!sequential_8/lstm_16/zeros_1/Less?
%sequential_8/lstm_16/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2'
%sequential_8/lstm_16/zeros_1/packed/1?
#sequential_8/lstm_16/zeros_1/packedPack+sequential_8/lstm_16/strided_slice:output:0.sequential_8/lstm_16/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2%
#sequential_8/lstm_16/zeros_1/packed?
"sequential_8/lstm_16/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"sequential_8/lstm_16/zeros_1/Const?
sequential_8/lstm_16/zeros_1Fill,sequential_8/lstm_16/zeros_1/packed:output:0+sequential_8/lstm_16/zeros_1/Const:output:0*
T0*(
_output_shapes
:??????????2
sequential_8/lstm_16/zeros_1?
#sequential_8/lstm_16/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2%
#sequential_8/lstm_16/transpose/perm?
sequential_8/lstm_16/transpose	Transposelstm_16_input,sequential_8/lstm_16/transpose/perm:output:0*
T0*+
_output_shapes
:?????????]2 
sequential_8/lstm_16/transpose?
sequential_8/lstm_16/Shape_1Shape"sequential_8/lstm_16/transpose:y:0*
T0*
_output_shapes
:2
sequential_8/lstm_16/Shape_1?
*sequential_8/lstm_16/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2,
*sequential_8/lstm_16/strided_slice_1/stack?
,sequential_8/lstm_16/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_8/lstm_16/strided_slice_1/stack_1?
,sequential_8/lstm_16/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_8/lstm_16/strided_slice_1/stack_2?
$sequential_8/lstm_16/strided_slice_1StridedSlice%sequential_8/lstm_16/Shape_1:output:03sequential_8/lstm_16/strided_slice_1/stack:output:05sequential_8/lstm_16/strided_slice_1/stack_1:output:05sequential_8/lstm_16/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2&
$sequential_8/lstm_16/strided_slice_1?
0sequential_8/lstm_16/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????22
0sequential_8/lstm_16/TensorArrayV2/element_shape?
"sequential_8/lstm_16/TensorArrayV2TensorListReserve9sequential_8/lstm_16/TensorArrayV2/element_shape:output:0-sequential_8/lstm_16/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02$
"sequential_8/lstm_16/TensorArrayV2?
Jsequential_8/lstm_16/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????]   2L
Jsequential_8/lstm_16/TensorArrayUnstack/TensorListFromTensor/element_shape?
<sequential_8/lstm_16/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor"sequential_8/lstm_16/transpose:y:0Ssequential_8/lstm_16/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02>
<sequential_8/lstm_16/TensorArrayUnstack/TensorListFromTensor?
*sequential_8/lstm_16/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2,
*sequential_8/lstm_16/strided_slice_2/stack?
,sequential_8/lstm_16/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_8/lstm_16/strided_slice_2/stack_1?
,sequential_8/lstm_16/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_8/lstm_16/strided_slice_2/stack_2?
$sequential_8/lstm_16/strided_slice_2StridedSlice"sequential_8/lstm_16/transpose:y:03sequential_8/lstm_16/strided_slice_2/stack:output:05sequential_8/lstm_16/strided_slice_2/stack_1:output:05sequential_8/lstm_16/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????]*
shrink_axis_mask2&
$sequential_8/lstm_16/strided_slice_2?
7sequential_8/lstm_16/lstm_cell_16/MatMul/ReadVariableOpReadVariableOp@sequential_8_lstm_16_lstm_cell_16_matmul_readvariableop_resource*
_output_shapes
:	]?*
dtype029
7sequential_8/lstm_16/lstm_cell_16/MatMul/ReadVariableOp?
(sequential_8/lstm_16/lstm_cell_16/MatMulMatMul-sequential_8/lstm_16/strided_slice_2:output:0?sequential_8/lstm_16/lstm_cell_16/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2*
(sequential_8/lstm_16/lstm_cell_16/MatMul?
9sequential_8/lstm_16/lstm_cell_16/MatMul_1/ReadVariableOpReadVariableOpBsequential_8_lstm_16_lstm_cell_16_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02;
9sequential_8/lstm_16/lstm_cell_16/MatMul_1/ReadVariableOp?
*sequential_8/lstm_16/lstm_cell_16/MatMul_1MatMul#sequential_8/lstm_16/zeros:output:0Asequential_8/lstm_16/lstm_cell_16/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2,
*sequential_8/lstm_16/lstm_cell_16/MatMul_1?
%sequential_8/lstm_16/lstm_cell_16/addAddV22sequential_8/lstm_16/lstm_cell_16/MatMul:product:04sequential_8/lstm_16/lstm_cell_16/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2'
%sequential_8/lstm_16/lstm_cell_16/add?
8sequential_8/lstm_16/lstm_cell_16/BiasAdd/ReadVariableOpReadVariableOpAsequential_8_lstm_16_lstm_cell_16_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02:
8sequential_8/lstm_16/lstm_cell_16/BiasAdd/ReadVariableOp?
)sequential_8/lstm_16/lstm_cell_16/BiasAddBiasAdd)sequential_8/lstm_16/lstm_cell_16/add:z:0@sequential_8/lstm_16/lstm_cell_16/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2+
)sequential_8/lstm_16/lstm_cell_16/BiasAdd?
1sequential_8/lstm_16/lstm_cell_16/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :23
1sequential_8/lstm_16/lstm_cell_16/split/split_dim?
'sequential_8/lstm_16/lstm_cell_16/splitSplit:sequential_8/lstm_16/lstm_cell_16/split/split_dim:output:02sequential_8/lstm_16/lstm_cell_16/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split2)
'sequential_8/lstm_16/lstm_cell_16/split?
)sequential_8/lstm_16/lstm_cell_16/SigmoidSigmoid0sequential_8/lstm_16/lstm_cell_16/split:output:0*
T0*(
_output_shapes
:??????????2+
)sequential_8/lstm_16/lstm_cell_16/Sigmoid?
+sequential_8/lstm_16/lstm_cell_16/Sigmoid_1Sigmoid0sequential_8/lstm_16/lstm_cell_16/split:output:1*
T0*(
_output_shapes
:??????????2-
+sequential_8/lstm_16/lstm_cell_16/Sigmoid_1?
%sequential_8/lstm_16/lstm_cell_16/mulMul/sequential_8/lstm_16/lstm_cell_16/Sigmoid_1:y:0%sequential_8/lstm_16/zeros_1:output:0*
T0*(
_output_shapes
:??????????2'
%sequential_8/lstm_16/lstm_cell_16/mul?
&sequential_8/lstm_16/lstm_cell_16/ReluRelu0sequential_8/lstm_16/lstm_cell_16/split:output:2*
T0*(
_output_shapes
:??????????2(
&sequential_8/lstm_16/lstm_cell_16/Relu?
'sequential_8/lstm_16/lstm_cell_16/mul_1Mul-sequential_8/lstm_16/lstm_cell_16/Sigmoid:y:04sequential_8/lstm_16/lstm_cell_16/Relu:activations:0*
T0*(
_output_shapes
:??????????2)
'sequential_8/lstm_16/lstm_cell_16/mul_1?
'sequential_8/lstm_16/lstm_cell_16/add_1AddV2)sequential_8/lstm_16/lstm_cell_16/mul:z:0+sequential_8/lstm_16/lstm_cell_16/mul_1:z:0*
T0*(
_output_shapes
:??????????2)
'sequential_8/lstm_16/lstm_cell_16/add_1?
+sequential_8/lstm_16/lstm_cell_16/Sigmoid_2Sigmoid0sequential_8/lstm_16/lstm_cell_16/split:output:3*
T0*(
_output_shapes
:??????????2-
+sequential_8/lstm_16/lstm_cell_16/Sigmoid_2?
(sequential_8/lstm_16/lstm_cell_16/Relu_1Relu+sequential_8/lstm_16/lstm_cell_16/add_1:z:0*
T0*(
_output_shapes
:??????????2*
(sequential_8/lstm_16/lstm_cell_16/Relu_1?
'sequential_8/lstm_16/lstm_cell_16/mul_2Mul/sequential_8/lstm_16/lstm_cell_16/Sigmoid_2:y:06sequential_8/lstm_16/lstm_cell_16/Relu_1:activations:0*
T0*(
_output_shapes
:??????????2)
'sequential_8/lstm_16/lstm_cell_16/mul_2?
2sequential_8/lstm_16/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????  24
2sequential_8/lstm_16/TensorArrayV2_1/element_shape?
$sequential_8/lstm_16/TensorArrayV2_1TensorListReserve;sequential_8/lstm_16/TensorArrayV2_1/element_shape:output:0-sequential_8/lstm_16/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02&
$sequential_8/lstm_16/TensorArrayV2_1x
sequential_8/lstm_16/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
sequential_8/lstm_16/time?
-sequential_8/lstm_16/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2/
-sequential_8/lstm_16/while/maximum_iterations?
'sequential_8/lstm_16/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2)
'sequential_8/lstm_16/while/loop_counter?
sequential_8/lstm_16/whileWhile0sequential_8/lstm_16/while/loop_counter:output:06sequential_8/lstm_16/while/maximum_iterations:output:0"sequential_8/lstm_16/time:output:0-sequential_8/lstm_16/TensorArrayV2_1:handle:0#sequential_8/lstm_16/zeros:output:0%sequential_8/lstm_16/zeros_1:output:0-sequential_8/lstm_16/strided_slice_1:output:0Lsequential_8/lstm_16/TensorArrayUnstack/TensorListFromTensor:output_handle:0@sequential_8_lstm_16_lstm_cell_16_matmul_readvariableop_resourceBsequential_8_lstm_16_lstm_cell_16_matmul_1_readvariableop_resourceAsequential_8_lstm_16_lstm_cell_16_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *4
body,R*
(sequential_8_lstm_16_while_body_32573203*4
cond,R*
(sequential_8_lstm_16_while_cond_32573202*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations 2
sequential_8/lstm_16/while?
Esequential_8/lstm_16/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????  2G
Esequential_8/lstm_16/TensorArrayV2Stack/TensorListStack/element_shape?
7sequential_8/lstm_16/TensorArrayV2Stack/TensorListStackTensorListStack#sequential_8/lstm_16/while:output:3Nsequential_8/lstm_16/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:??????????*
element_dtype029
7sequential_8/lstm_16/TensorArrayV2Stack/TensorListStack?
*sequential_8/lstm_16/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2,
*sequential_8/lstm_16/strided_slice_3/stack?
,sequential_8/lstm_16/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2.
,sequential_8/lstm_16/strided_slice_3/stack_1?
,sequential_8/lstm_16/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_8/lstm_16/strided_slice_3/stack_2?
$sequential_8/lstm_16/strided_slice_3StridedSlice@sequential_8/lstm_16/TensorArrayV2Stack/TensorListStack:tensor:03sequential_8/lstm_16/strided_slice_3/stack:output:05sequential_8/lstm_16/strided_slice_3/stack_1:output:05sequential_8/lstm_16/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2&
$sequential_8/lstm_16/strided_slice_3?
%sequential_8/lstm_16/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2'
%sequential_8/lstm_16/transpose_1/perm?
 sequential_8/lstm_16/transpose_1	Transpose@sequential_8/lstm_16/TensorArrayV2Stack/TensorListStack:tensor:0.sequential_8/lstm_16/transpose_1/perm:output:0*
T0*,
_output_shapes
:??????????2"
 sequential_8/lstm_16/transpose_1?
sequential_8/lstm_16/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
sequential_8/lstm_16/runtime?
 sequential_8/dropout_16/IdentityIdentity$sequential_8/lstm_16/transpose_1:y:0*
T0*,
_output_shapes
:??????????2"
 sequential_8/dropout_16/Identity?
sequential_8/lstm_17/ShapeShape)sequential_8/dropout_16/Identity:output:0*
T0*
_output_shapes
:2
sequential_8/lstm_17/Shape?
(sequential_8/lstm_17/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(sequential_8/lstm_17/strided_slice/stack?
*sequential_8/lstm_17/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*sequential_8/lstm_17/strided_slice/stack_1?
*sequential_8/lstm_17/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*sequential_8/lstm_17/strided_slice/stack_2?
"sequential_8/lstm_17/strided_sliceStridedSlice#sequential_8/lstm_17/Shape:output:01sequential_8/lstm_17/strided_slice/stack:output:03sequential_8/lstm_17/strided_slice/stack_1:output:03sequential_8/lstm_17/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"sequential_8/lstm_17/strided_slice?
 sequential_8/lstm_17/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2"
 sequential_8/lstm_17/zeros/mul/y?
sequential_8/lstm_17/zeros/mulMul+sequential_8/lstm_17/strided_slice:output:0)sequential_8/lstm_17/zeros/mul/y:output:0*
T0*
_output_shapes
: 2 
sequential_8/lstm_17/zeros/mul?
!sequential_8/lstm_17/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2#
!sequential_8/lstm_17/zeros/Less/y?
sequential_8/lstm_17/zeros/LessLess"sequential_8/lstm_17/zeros/mul:z:0*sequential_8/lstm_17/zeros/Less/y:output:0*
T0*
_output_shapes
: 2!
sequential_8/lstm_17/zeros/Less?
#sequential_8/lstm_17/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2%
#sequential_8/lstm_17/zeros/packed/1?
!sequential_8/lstm_17/zeros/packedPack+sequential_8/lstm_17/strided_slice:output:0,sequential_8/lstm_17/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2#
!sequential_8/lstm_17/zeros/packed?
 sequential_8/lstm_17/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 sequential_8/lstm_17/zeros/Const?
sequential_8/lstm_17/zerosFill*sequential_8/lstm_17/zeros/packed:output:0)sequential_8/lstm_17/zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
sequential_8/lstm_17/zeros?
"sequential_8/lstm_17/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2$
"sequential_8/lstm_17/zeros_1/mul/y?
 sequential_8/lstm_17/zeros_1/mulMul+sequential_8/lstm_17/strided_slice:output:0+sequential_8/lstm_17/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2"
 sequential_8/lstm_17/zeros_1/mul?
#sequential_8/lstm_17/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2%
#sequential_8/lstm_17/zeros_1/Less/y?
!sequential_8/lstm_17/zeros_1/LessLess$sequential_8/lstm_17/zeros_1/mul:z:0,sequential_8/lstm_17/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2#
!sequential_8/lstm_17/zeros_1/Less?
%sequential_8/lstm_17/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2'
%sequential_8/lstm_17/zeros_1/packed/1?
#sequential_8/lstm_17/zeros_1/packedPack+sequential_8/lstm_17/strided_slice:output:0.sequential_8/lstm_17/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2%
#sequential_8/lstm_17/zeros_1/packed?
"sequential_8/lstm_17/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"sequential_8/lstm_17/zeros_1/Const?
sequential_8/lstm_17/zeros_1Fill,sequential_8/lstm_17/zeros_1/packed:output:0+sequential_8/lstm_17/zeros_1/Const:output:0*
T0*(
_output_shapes
:??????????2
sequential_8/lstm_17/zeros_1?
#sequential_8/lstm_17/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2%
#sequential_8/lstm_17/transpose/perm?
sequential_8/lstm_17/transpose	Transpose)sequential_8/dropout_16/Identity:output:0,sequential_8/lstm_17/transpose/perm:output:0*
T0*,
_output_shapes
:??????????2 
sequential_8/lstm_17/transpose?
sequential_8/lstm_17/Shape_1Shape"sequential_8/lstm_17/transpose:y:0*
T0*
_output_shapes
:2
sequential_8/lstm_17/Shape_1?
*sequential_8/lstm_17/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2,
*sequential_8/lstm_17/strided_slice_1/stack?
,sequential_8/lstm_17/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_8/lstm_17/strided_slice_1/stack_1?
,sequential_8/lstm_17/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_8/lstm_17/strided_slice_1/stack_2?
$sequential_8/lstm_17/strided_slice_1StridedSlice%sequential_8/lstm_17/Shape_1:output:03sequential_8/lstm_17/strided_slice_1/stack:output:05sequential_8/lstm_17/strided_slice_1/stack_1:output:05sequential_8/lstm_17/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2&
$sequential_8/lstm_17/strided_slice_1?
0sequential_8/lstm_17/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????22
0sequential_8/lstm_17/TensorArrayV2/element_shape?
"sequential_8/lstm_17/TensorArrayV2TensorListReserve9sequential_8/lstm_17/TensorArrayV2/element_shape:output:0-sequential_8/lstm_17/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02$
"sequential_8/lstm_17/TensorArrayV2?
Jsequential_8/lstm_17/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????  2L
Jsequential_8/lstm_17/TensorArrayUnstack/TensorListFromTensor/element_shape?
<sequential_8/lstm_17/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor"sequential_8/lstm_17/transpose:y:0Ssequential_8/lstm_17/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02>
<sequential_8/lstm_17/TensorArrayUnstack/TensorListFromTensor?
*sequential_8/lstm_17/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2,
*sequential_8/lstm_17/strided_slice_2/stack?
,sequential_8/lstm_17/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_8/lstm_17/strided_slice_2/stack_1?
,sequential_8/lstm_17/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_8/lstm_17/strided_slice_2/stack_2?
$sequential_8/lstm_17/strided_slice_2StridedSlice"sequential_8/lstm_17/transpose:y:03sequential_8/lstm_17/strided_slice_2/stack:output:05sequential_8/lstm_17/strided_slice_2/stack_1:output:05sequential_8/lstm_17/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2&
$sequential_8/lstm_17/strided_slice_2?
7sequential_8/lstm_17/lstm_cell_17/MatMul/ReadVariableOpReadVariableOp@sequential_8_lstm_17_lstm_cell_17_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype029
7sequential_8/lstm_17/lstm_cell_17/MatMul/ReadVariableOp?
(sequential_8/lstm_17/lstm_cell_17/MatMulMatMul-sequential_8/lstm_17/strided_slice_2:output:0?sequential_8/lstm_17/lstm_cell_17/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2*
(sequential_8/lstm_17/lstm_cell_17/MatMul?
9sequential_8/lstm_17/lstm_cell_17/MatMul_1/ReadVariableOpReadVariableOpBsequential_8_lstm_17_lstm_cell_17_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02;
9sequential_8/lstm_17/lstm_cell_17/MatMul_1/ReadVariableOp?
*sequential_8/lstm_17/lstm_cell_17/MatMul_1MatMul#sequential_8/lstm_17/zeros:output:0Asequential_8/lstm_17/lstm_cell_17/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2,
*sequential_8/lstm_17/lstm_cell_17/MatMul_1?
%sequential_8/lstm_17/lstm_cell_17/addAddV22sequential_8/lstm_17/lstm_cell_17/MatMul:product:04sequential_8/lstm_17/lstm_cell_17/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2'
%sequential_8/lstm_17/lstm_cell_17/add?
8sequential_8/lstm_17/lstm_cell_17/BiasAdd/ReadVariableOpReadVariableOpAsequential_8_lstm_17_lstm_cell_17_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02:
8sequential_8/lstm_17/lstm_cell_17/BiasAdd/ReadVariableOp?
)sequential_8/lstm_17/lstm_cell_17/BiasAddBiasAdd)sequential_8/lstm_17/lstm_cell_17/add:z:0@sequential_8/lstm_17/lstm_cell_17/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2+
)sequential_8/lstm_17/lstm_cell_17/BiasAdd?
1sequential_8/lstm_17/lstm_cell_17/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :23
1sequential_8/lstm_17/lstm_cell_17/split/split_dim?
'sequential_8/lstm_17/lstm_cell_17/splitSplit:sequential_8/lstm_17/lstm_cell_17/split/split_dim:output:02sequential_8/lstm_17/lstm_cell_17/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split2)
'sequential_8/lstm_17/lstm_cell_17/split?
)sequential_8/lstm_17/lstm_cell_17/SigmoidSigmoid0sequential_8/lstm_17/lstm_cell_17/split:output:0*
T0*(
_output_shapes
:??????????2+
)sequential_8/lstm_17/lstm_cell_17/Sigmoid?
+sequential_8/lstm_17/lstm_cell_17/Sigmoid_1Sigmoid0sequential_8/lstm_17/lstm_cell_17/split:output:1*
T0*(
_output_shapes
:??????????2-
+sequential_8/lstm_17/lstm_cell_17/Sigmoid_1?
%sequential_8/lstm_17/lstm_cell_17/mulMul/sequential_8/lstm_17/lstm_cell_17/Sigmoid_1:y:0%sequential_8/lstm_17/zeros_1:output:0*
T0*(
_output_shapes
:??????????2'
%sequential_8/lstm_17/lstm_cell_17/mul?
&sequential_8/lstm_17/lstm_cell_17/ReluRelu0sequential_8/lstm_17/lstm_cell_17/split:output:2*
T0*(
_output_shapes
:??????????2(
&sequential_8/lstm_17/lstm_cell_17/Relu?
'sequential_8/lstm_17/lstm_cell_17/mul_1Mul-sequential_8/lstm_17/lstm_cell_17/Sigmoid:y:04sequential_8/lstm_17/lstm_cell_17/Relu:activations:0*
T0*(
_output_shapes
:??????????2)
'sequential_8/lstm_17/lstm_cell_17/mul_1?
'sequential_8/lstm_17/lstm_cell_17/add_1AddV2)sequential_8/lstm_17/lstm_cell_17/mul:z:0+sequential_8/lstm_17/lstm_cell_17/mul_1:z:0*
T0*(
_output_shapes
:??????????2)
'sequential_8/lstm_17/lstm_cell_17/add_1?
+sequential_8/lstm_17/lstm_cell_17/Sigmoid_2Sigmoid0sequential_8/lstm_17/lstm_cell_17/split:output:3*
T0*(
_output_shapes
:??????????2-
+sequential_8/lstm_17/lstm_cell_17/Sigmoid_2?
(sequential_8/lstm_17/lstm_cell_17/Relu_1Relu+sequential_8/lstm_17/lstm_cell_17/add_1:z:0*
T0*(
_output_shapes
:??????????2*
(sequential_8/lstm_17/lstm_cell_17/Relu_1?
'sequential_8/lstm_17/lstm_cell_17/mul_2Mul/sequential_8/lstm_17/lstm_cell_17/Sigmoid_2:y:06sequential_8/lstm_17/lstm_cell_17/Relu_1:activations:0*
T0*(
_output_shapes
:??????????2)
'sequential_8/lstm_17/lstm_cell_17/mul_2?
2sequential_8/lstm_17/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   24
2sequential_8/lstm_17/TensorArrayV2_1/element_shape?
$sequential_8/lstm_17/TensorArrayV2_1TensorListReserve;sequential_8/lstm_17/TensorArrayV2_1/element_shape:output:0-sequential_8/lstm_17/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02&
$sequential_8/lstm_17/TensorArrayV2_1x
sequential_8/lstm_17/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
sequential_8/lstm_17/time?
-sequential_8/lstm_17/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2/
-sequential_8/lstm_17/while/maximum_iterations?
'sequential_8/lstm_17/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2)
'sequential_8/lstm_17/while/loop_counter?
sequential_8/lstm_17/whileWhile0sequential_8/lstm_17/while/loop_counter:output:06sequential_8/lstm_17/while/maximum_iterations:output:0"sequential_8/lstm_17/time:output:0-sequential_8/lstm_17/TensorArrayV2_1:handle:0#sequential_8/lstm_17/zeros:output:0%sequential_8/lstm_17/zeros_1:output:0-sequential_8/lstm_17/strided_slice_1:output:0Lsequential_8/lstm_17/TensorArrayUnstack/TensorListFromTensor:output_handle:0@sequential_8_lstm_17_lstm_cell_17_matmul_readvariableop_resourceBsequential_8_lstm_17_lstm_cell_17_matmul_1_readvariableop_resourceAsequential_8_lstm_17_lstm_cell_17_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :??????????:??????????: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *4
body,R*
(sequential_8_lstm_17_while_body_32573351*4
cond,R*
(sequential_8_lstm_17_while_cond_32573350*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations 2
sequential_8/lstm_17/while?
Esequential_8/lstm_17/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2G
Esequential_8/lstm_17/TensorArrayV2Stack/TensorListStack/element_shape?
7sequential_8/lstm_17/TensorArrayV2Stack/TensorListStackTensorListStack#sequential_8/lstm_17/while:output:3Nsequential_8/lstm_17/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:??????????*
element_dtype029
7sequential_8/lstm_17/TensorArrayV2Stack/TensorListStack?
*sequential_8/lstm_17/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2,
*sequential_8/lstm_17/strided_slice_3/stack?
,sequential_8/lstm_17/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2.
,sequential_8/lstm_17/strided_slice_3/stack_1?
,sequential_8/lstm_17/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_8/lstm_17/strided_slice_3/stack_2?
$sequential_8/lstm_17/strided_slice_3StridedSlice@sequential_8/lstm_17/TensorArrayV2Stack/TensorListStack:tensor:03sequential_8/lstm_17/strided_slice_3/stack:output:05sequential_8/lstm_17/strided_slice_3/stack_1:output:05sequential_8/lstm_17/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2&
$sequential_8/lstm_17/strided_slice_3?
%sequential_8/lstm_17/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2'
%sequential_8/lstm_17/transpose_1/perm?
 sequential_8/lstm_17/transpose_1	Transpose@sequential_8/lstm_17/TensorArrayV2Stack/TensorListStack:tensor:0.sequential_8/lstm_17/transpose_1/perm:output:0*
T0*,
_output_shapes
:??????????2"
 sequential_8/lstm_17/transpose_1?
sequential_8/lstm_17/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
sequential_8/lstm_17/runtime?
 sequential_8/dropout_17/IdentityIdentity$sequential_8/lstm_17/transpose_1:y:0*
T0*,
_output_shapes
:??????????2"
 sequential_8/dropout_17/Identity?
-sequential_8/dense_8/Tensordot/ReadVariableOpReadVariableOp6sequential_8_dense_8_tensordot_readvariableop_resource*
_output_shapes
:	?*
dtype02/
-sequential_8/dense_8/Tensordot/ReadVariableOp?
#sequential_8/dense_8/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2%
#sequential_8/dense_8/Tensordot/axes?
#sequential_8/dense_8/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2%
#sequential_8/dense_8/Tensordot/free?
$sequential_8/dense_8/Tensordot/ShapeShape)sequential_8/dropout_17/Identity:output:0*
T0*
_output_shapes
:2&
$sequential_8/dense_8/Tensordot/Shape?
,sequential_8/dense_8/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,sequential_8/dense_8/Tensordot/GatherV2/axis?
'sequential_8/dense_8/Tensordot/GatherV2GatherV2-sequential_8/dense_8/Tensordot/Shape:output:0,sequential_8/dense_8/Tensordot/free:output:05sequential_8/dense_8/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2)
'sequential_8/dense_8/Tensordot/GatherV2?
.sequential_8/dense_8/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.sequential_8/dense_8/Tensordot/GatherV2_1/axis?
)sequential_8/dense_8/Tensordot/GatherV2_1GatherV2-sequential_8/dense_8/Tensordot/Shape:output:0,sequential_8/dense_8/Tensordot/axes:output:07sequential_8/dense_8/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2+
)sequential_8/dense_8/Tensordot/GatherV2_1?
$sequential_8/dense_8/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2&
$sequential_8/dense_8/Tensordot/Const?
#sequential_8/dense_8/Tensordot/ProdProd0sequential_8/dense_8/Tensordot/GatherV2:output:0-sequential_8/dense_8/Tensordot/Const:output:0*
T0*
_output_shapes
: 2%
#sequential_8/dense_8/Tensordot/Prod?
&sequential_8/dense_8/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2(
&sequential_8/dense_8/Tensordot/Const_1?
%sequential_8/dense_8/Tensordot/Prod_1Prod2sequential_8/dense_8/Tensordot/GatherV2_1:output:0/sequential_8/dense_8/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2'
%sequential_8/dense_8/Tensordot/Prod_1?
*sequential_8/dense_8/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2,
*sequential_8/dense_8/Tensordot/concat/axis?
%sequential_8/dense_8/Tensordot/concatConcatV2,sequential_8/dense_8/Tensordot/free:output:0,sequential_8/dense_8/Tensordot/axes:output:03sequential_8/dense_8/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2'
%sequential_8/dense_8/Tensordot/concat?
$sequential_8/dense_8/Tensordot/stackPack,sequential_8/dense_8/Tensordot/Prod:output:0.sequential_8/dense_8/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2&
$sequential_8/dense_8/Tensordot/stack?
(sequential_8/dense_8/Tensordot/transpose	Transpose)sequential_8/dropout_17/Identity:output:0.sequential_8/dense_8/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????2*
(sequential_8/dense_8/Tensordot/transpose?
&sequential_8/dense_8/Tensordot/ReshapeReshape,sequential_8/dense_8/Tensordot/transpose:y:0-sequential_8/dense_8/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2(
&sequential_8/dense_8/Tensordot/Reshape?
%sequential_8/dense_8/Tensordot/MatMulMatMul/sequential_8/dense_8/Tensordot/Reshape:output:05sequential_8/dense_8/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2'
%sequential_8/dense_8/Tensordot/MatMul?
&sequential_8/dense_8/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&sequential_8/dense_8/Tensordot/Const_2?
,sequential_8/dense_8/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,sequential_8/dense_8/Tensordot/concat_1/axis?
'sequential_8/dense_8/Tensordot/concat_1ConcatV20sequential_8/dense_8/Tensordot/GatherV2:output:0/sequential_8/dense_8/Tensordot/Const_2:output:05sequential_8/dense_8/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2)
'sequential_8/dense_8/Tensordot/concat_1?
sequential_8/dense_8/TensordotReshape/sequential_8/dense_8/Tensordot/MatMul:product:00sequential_8/dense_8/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????2 
sequential_8/dense_8/Tensordot?
+sequential_8/dense_8/BiasAdd/ReadVariableOpReadVariableOp4sequential_8_dense_8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02-
+sequential_8/dense_8/BiasAdd/ReadVariableOp?
sequential_8/dense_8/BiasAddBiasAdd'sequential_8/dense_8/Tensordot:output:03sequential_8/dense_8/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????2
sequential_8/dense_8/BiasAdd?
sequential_8/dense_8/SoftmaxSoftmax%sequential_8/dense_8/BiasAdd:output:0*
T0*+
_output_shapes
:?????????2
sequential_8/dense_8/Softmax?
IdentityIdentity&sequential_8/dense_8/Softmax:softmax:0^NoOp*
T0*+
_output_shapes
:?????????2

Identity?
NoOpNoOp,^sequential_8/dense_8/BiasAdd/ReadVariableOp.^sequential_8/dense_8/Tensordot/ReadVariableOp9^sequential_8/lstm_16/lstm_cell_16/BiasAdd/ReadVariableOp8^sequential_8/lstm_16/lstm_cell_16/MatMul/ReadVariableOp:^sequential_8/lstm_16/lstm_cell_16/MatMul_1/ReadVariableOp^sequential_8/lstm_16/while9^sequential_8/lstm_17/lstm_cell_17/BiasAdd/ReadVariableOp8^sequential_8/lstm_17/lstm_cell_17/MatMul/ReadVariableOp:^sequential_8/lstm_17/lstm_cell_17/MatMul_1/ReadVariableOp^sequential_8/lstm_17/while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????]: : : : : : : : 2Z
+sequential_8/dense_8/BiasAdd/ReadVariableOp+sequential_8/dense_8/BiasAdd/ReadVariableOp2^
-sequential_8/dense_8/Tensordot/ReadVariableOp-sequential_8/dense_8/Tensordot/ReadVariableOp2t
8sequential_8/lstm_16/lstm_cell_16/BiasAdd/ReadVariableOp8sequential_8/lstm_16/lstm_cell_16/BiasAdd/ReadVariableOp2r
7sequential_8/lstm_16/lstm_cell_16/MatMul/ReadVariableOp7sequential_8/lstm_16/lstm_cell_16/MatMul/ReadVariableOp2v
9sequential_8/lstm_16/lstm_cell_16/MatMul_1/ReadVariableOp9sequential_8/lstm_16/lstm_cell_16/MatMul_1/ReadVariableOp28
sequential_8/lstm_16/whilesequential_8/lstm_16/while2t
8sequential_8/lstm_17/lstm_cell_17/BiasAdd/ReadVariableOp8sequential_8/lstm_17/lstm_cell_17/BiasAdd/ReadVariableOp2r
7sequential_8/lstm_17/lstm_cell_17/MatMul/ReadVariableOp7sequential_8/lstm_17/lstm_cell_17/MatMul/ReadVariableOp2v
9sequential_8/lstm_17/lstm_cell_17/MatMul_1/ReadVariableOp9sequential_8/lstm_17/lstm_cell_17/MatMul_1/ReadVariableOp28
sequential_8/lstm_17/whilesequential_8/lstm_17/while:Z V
+
_output_shapes
:?????????]
'
_user_specified_namelstm_16_input"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
K
lstm_16_input:
serving_default_lstm_16_input:0?????????]?
dense_84
StatefulPartitionedCall:0?????????tensorflow/serving/predict:??
?
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
?_default_save_signature
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_sequential
?
cell

state_spec
trainable_variables
regularization_losses
	variables
	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_rnn_layer
?
trainable_variables
regularization_losses
	variables
	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
cell

state_spec
trainable_variables
regularization_losses
	variables
	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_rnn_layer
?
trainable_variables
regularization_losses
	variables
	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?

 kernel
!bias
"trainable_variables
#regularization_losses
$	variables
%	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
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
?
1metrics
trainable_variables
regularization_losses
2layer_regularization_losses
3non_trainable_variables

4layers
		variables
5layer_metrics
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
?
6
state_size

+kernel
,recurrent_kernel
-bias
7trainable_variables
8regularization_losses
9	variables
:	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
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
?
;metrics
trainable_variables

<states
=layer_regularization_losses
>non_trainable_variables
regularization_losses

?layers
	variables
@layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
Ametrics
trainable_variables
Blayer_regularization_losses
Cnon_trainable_variables
regularization_losses

Dlayers
	variables
Elayer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
F
state_size

.kernel
/recurrent_kernel
0bias
Gtrainable_variables
Hregularization_losses
I	variables
J	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
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
?
Kmetrics
trainable_variables

Lstates
Mlayer_regularization_losses
Nnon_trainable_variables
regularization_losses

Olayers
	variables
Player_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
Qmetrics
trainable_variables
Rlayer_regularization_losses
Snon_trainable_variables
regularization_losses

Tlayers
	variables
Ulayer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
!:	?2dense_8/kernel
:2dense_8/bias
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
?
Vmetrics
"trainable_variables
Wlayer_regularization_losses
Xnon_trainable_variables
#regularization_losses

Ylayers
$	variables
Zlayer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
.:,	]?2lstm_16/lstm_cell_16/kernel
9:7
??2%lstm_16/lstm_cell_16/recurrent_kernel
(:&?2lstm_16/lstm_cell_16/bias
/:-
??2lstm_17/lstm_cell_17/kernel
9:7
??2%lstm_17/lstm_cell_17/recurrent_kernel
(:&?2lstm_17/lstm_cell_17/bias
.
[0
\1"
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
?
]metrics
7trainable_variables
^layer_regularization_losses
_non_trainable_variables
8regularization_losses

`layers
9	variables
alayer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
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
?
bmetrics
Gtrainable_variables
clayer_regularization_losses
dnon_trainable_variables
Hregularization_losses

elayers
I	variables
flayer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
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
&:$	?2Adam/dense_8/kernel/m
:2Adam/dense_8/bias/m
3:1	]?2"Adam/lstm_16/lstm_cell_16/kernel/m
>:<
??2,Adam/lstm_16/lstm_cell_16/recurrent_kernel/m
-:+?2 Adam/lstm_16/lstm_cell_16/bias/m
4:2
??2"Adam/lstm_17/lstm_cell_17/kernel/m
>:<
??2,Adam/lstm_17/lstm_cell_17/recurrent_kernel/m
-:+?2 Adam/lstm_17/lstm_cell_17/bias/m
&:$	?2Adam/dense_8/kernel/v
:2Adam/dense_8/bias/v
3:1	]?2"Adam/lstm_16/lstm_cell_16/kernel/v
>:<
??2,Adam/lstm_16/lstm_cell_16/recurrent_kernel/v
-:+?2 Adam/lstm_16/lstm_cell_16/bias/v
4:2
??2"Adam/lstm_17/lstm_cell_17/kernel/v
>:<
??2,Adam/lstm_17/lstm_cell_17/recurrent_kernel/v
-:+?2 Adam/lstm_17/lstm_cell_17/bias/v
?B?
#__inference__wrapped_model_32573463lstm_16_input"?
???
FullArgSpec
args? 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
J__inference_sequential_8_layer_call_and_return_conditional_losses_32576013
J__inference_sequential_8_layer_call_and_return_conditional_losses_32576354
J__inference_sequential_8_layer_call_and_return_conditional_losses_32575632
J__inference_sequential_8_layer_call_and_return_conditional_losses_32575657?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
/__inference_sequential_8_layer_call_fn_32575117
/__inference_sequential_8_layer_call_fn_32576375
/__inference_sequential_8_layer_call_fn_32576396
/__inference_sequential_8_layer_call_fn_32575607?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
E__inference_lstm_16_layer_call_and_return_conditional_losses_32576547
E__inference_lstm_16_layer_call_and_return_conditional_losses_32576698
E__inference_lstm_16_layer_call_and_return_conditional_losses_32576849
E__inference_lstm_16_layer_call_and_return_conditional_losses_32577000?
???
FullArgSpecB
args:?7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults?

 
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
*__inference_lstm_16_layer_call_fn_32577011
*__inference_lstm_16_layer_call_fn_32577022
*__inference_lstm_16_layer_call_fn_32577033
*__inference_lstm_16_layer_call_fn_32577044?
???
FullArgSpecB
args:?7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults?

 
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
H__inference_dropout_16_layer_call_and_return_conditional_losses_32577049
H__inference_dropout_16_layer_call_and_return_conditional_losses_32577061?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
-__inference_dropout_16_layer_call_fn_32577066
-__inference_dropout_16_layer_call_fn_32577071?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
E__inference_lstm_17_layer_call_and_return_conditional_losses_32577222
E__inference_lstm_17_layer_call_and_return_conditional_losses_32577373
E__inference_lstm_17_layer_call_and_return_conditional_losses_32577524
E__inference_lstm_17_layer_call_and_return_conditional_losses_32577675?
???
FullArgSpecB
args:?7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults?

 
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
*__inference_lstm_17_layer_call_fn_32577686
*__inference_lstm_17_layer_call_fn_32577697
*__inference_lstm_17_layer_call_fn_32577708
*__inference_lstm_17_layer_call_fn_32577719?
???
FullArgSpecB
args:?7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults?

 
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
H__inference_dropout_17_layer_call_and_return_conditional_losses_32577724
H__inference_dropout_17_layer_call_and_return_conditional_losses_32577736?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
-__inference_dropout_17_layer_call_fn_32577741
-__inference_dropout_17_layer_call_fn_32577746?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
E__inference_dense_8_layer_call_and_return_conditional_losses_32577777?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_dense_8_layer_call_fn_32577786?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
&__inference_signature_wrapper_32575686lstm_16_input"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
J__inference_lstm_cell_16_layer_call_and_return_conditional_losses_32577818
J__inference_lstm_cell_16_layer_call_and_return_conditional_losses_32577850?
???
FullArgSpec3
args+?(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
/__inference_lstm_cell_16_layer_call_fn_32577867
/__inference_lstm_cell_16_layer_call_fn_32577884?
???
FullArgSpec3
args+?(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
J__inference_lstm_cell_17_layer_call_and_return_conditional_losses_32577916
J__inference_lstm_cell_17_layer_call_and_return_conditional_losses_32577948?
???
FullArgSpec3
args+?(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
/__inference_lstm_cell_17_layer_call_fn_32577965
/__inference_lstm_cell_17_layer_call_fn_32577982?
???
FullArgSpec3
args+?(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 ?
#__inference__wrapped_model_32573463}+,-./0 !:?7
0?-
+?(
lstm_16_input?????????]
? "5?2
0
dense_8%?"
dense_8??????????
E__inference_dense_8_layer_call_and_return_conditional_losses_32577777e !4?1
*?'
%?"
inputs??????????
? ")?&
?
0?????????
? ?
*__inference_dense_8_layer_call_fn_32577786X !4?1
*?'
%?"
inputs??????????
? "???????????
H__inference_dropout_16_layer_call_and_return_conditional_losses_32577049f8?5
.?+
%?"
inputs??????????
p 
? "*?'
 ?
0??????????
? ?
H__inference_dropout_16_layer_call_and_return_conditional_losses_32577061f8?5
.?+
%?"
inputs??????????
p
? "*?'
 ?
0??????????
? ?
-__inference_dropout_16_layer_call_fn_32577066Y8?5
.?+
%?"
inputs??????????
p 
? "????????????
-__inference_dropout_16_layer_call_fn_32577071Y8?5
.?+
%?"
inputs??????????
p
? "????????????
H__inference_dropout_17_layer_call_and_return_conditional_losses_32577724f8?5
.?+
%?"
inputs??????????
p 
? "*?'
 ?
0??????????
? ?
H__inference_dropout_17_layer_call_and_return_conditional_losses_32577736f8?5
.?+
%?"
inputs??????????
p
? "*?'
 ?
0??????????
? ?
-__inference_dropout_17_layer_call_fn_32577741Y8?5
.?+
%?"
inputs??????????
p 
? "????????????
-__inference_dropout_17_layer_call_fn_32577746Y8?5
.?+
%?"
inputs??????????
p
? "????????????
E__inference_lstm_16_layer_call_and_return_conditional_losses_32576547?+,-O?L
E?B
4?1
/?,
inputs/0??????????????????]

 
p 

 
? "3?0
)?&
0???????????????????
? ?
E__inference_lstm_16_layer_call_and_return_conditional_losses_32576698?+,-O?L
E?B
4?1
/?,
inputs/0??????????????????]

 
p

 
? "3?0
)?&
0???????????????????
? ?
E__inference_lstm_16_layer_call_and_return_conditional_losses_32576849r+,-??<
5?2
$?!
inputs?????????]

 
p 

 
? "*?'
 ?
0??????????
? ?
E__inference_lstm_16_layer_call_and_return_conditional_losses_32577000r+,-??<
5?2
$?!
inputs?????????]

 
p

 
? "*?'
 ?
0??????????
? ?
*__inference_lstm_16_layer_call_fn_32577011~+,-O?L
E?B
4?1
/?,
inputs/0??????????????????]

 
p 

 
? "&?#????????????????????
*__inference_lstm_16_layer_call_fn_32577022~+,-O?L
E?B
4?1
/?,
inputs/0??????????????????]

 
p

 
? "&?#????????????????????
*__inference_lstm_16_layer_call_fn_32577033e+,-??<
5?2
$?!
inputs?????????]

 
p 

 
? "????????????
*__inference_lstm_16_layer_call_fn_32577044e+,-??<
5?2
$?!
inputs?????????]

 
p

 
? "????????????
E__inference_lstm_17_layer_call_and_return_conditional_losses_32577222?./0P?M
F?C
5?2
0?-
inputs/0???????????????????

 
p 

 
? "3?0
)?&
0???????????????????
? ?
E__inference_lstm_17_layer_call_and_return_conditional_losses_32577373?./0P?M
F?C
5?2
0?-
inputs/0???????????????????

 
p

 
? "3?0
)?&
0???????????????????
? ?
E__inference_lstm_17_layer_call_and_return_conditional_losses_32577524s./0@?=
6?3
%?"
inputs??????????

 
p 

 
? "*?'
 ?
0??????????
? ?
E__inference_lstm_17_layer_call_and_return_conditional_losses_32577675s./0@?=
6?3
%?"
inputs??????????

 
p

 
? "*?'
 ?
0??????????
? ?
*__inference_lstm_17_layer_call_fn_32577686./0P?M
F?C
5?2
0?-
inputs/0???????????????????

 
p 

 
? "&?#????????????????????
*__inference_lstm_17_layer_call_fn_32577697./0P?M
F?C
5?2
0?-
inputs/0???????????????????

 
p

 
? "&?#????????????????????
*__inference_lstm_17_layer_call_fn_32577708f./0@?=
6?3
%?"
inputs??????????

 
p 

 
? "????????????
*__inference_lstm_17_layer_call_fn_32577719f./0@?=
6?3
%?"
inputs??????????

 
p

 
? "????????????
J__inference_lstm_cell_16_layer_call_and_return_conditional_losses_32577818?+,-??
x?u
 ?
inputs?????????]
M?J
#? 
states/0??????????
#? 
states/1??????????
p 
? "v?s
l?i
?
0/0??????????
G?D
 ?
0/1/0??????????
 ?
0/1/1??????????
? ?
J__inference_lstm_cell_16_layer_call_and_return_conditional_losses_32577850?+,-??
x?u
 ?
inputs?????????]
M?J
#? 
states/0??????????
#? 
states/1??????????
p
? "v?s
l?i
?
0/0??????????
G?D
 ?
0/1/0??????????
 ?
0/1/1??????????
? ?
/__inference_lstm_cell_16_layer_call_fn_32577867?+,-??
x?u
 ?
inputs?????????]
M?J
#? 
states/0??????????
#? 
states/1??????????
p 
? "f?c
?
0??????????
C?@
?
1/0??????????
?
1/1???????????
/__inference_lstm_cell_16_layer_call_fn_32577884?+,-??
x?u
 ?
inputs?????????]
M?J
#? 
states/0??????????
#? 
states/1??????????
p
? "f?c
?
0??????????
C?@
?
1/0??????????
?
1/1???????????
J__inference_lstm_cell_17_layer_call_and_return_conditional_losses_32577916?./0???
y?v
!?
inputs??????????
M?J
#? 
states/0??????????
#? 
states/1??????????
p 
? "v?s
l?i
?
0/0??????????
G?D
 ?
0/1/0??????????
 ?
0/1/1??????????
? ?
J__inference_lstm_cell_17_layer_call_and_return_conditional_losses_32577948?./0???
y?v
!?
inputs??????????
M?J
#? 
states/0??????????
#? 
states/1??????????
p
? "v?s
l?i
?
0/0??????????
G?D
 ?
0/1/0??????????
 ?
0/1/1??????????
? ?
/__inference_lstm_cell_17_layer_call_fn_32577965?./0???
y?v
!?
inputs??????????
M?J
#? 
states/0??????????
#? 
states/1??????????
p 
? "f?c
?
0??????????
C?@
?
1/0??????????
?
1/1???????????
/__inference_lstm_cell_17_layer_call_fn_32577982?./0???
y?v
!?
inputs??????????
M?J
#? 
states/0??????????
#? 
states/1??????????
p
? "f?c
?
0??????????
C?@
?
1/0??????????
?
1/1???????????
J__inference_sequential_8_layer_call_and_return_conditional_losses_32575632y+,-./0 !B??
8?5
+?(
lstm_16_input?????????]
p 

 
? ")?&
?
0?????????
? ?
J__inference_sequential_8_layer_call_and_return_conditional_losses_32575657y+,-./0 !B??
8?5
+?(
lstm_16_input?????????]
p

 
? ")?&
?
0?????????
? ?
J__inference_sequential_8_layer_call_and_return_conditional_losses_32576013r+,-./0 !;?8
1?.
$?!
inputs?????????]
p 

 
? ")?&
?
0?????????
? ?
J__inference_sequential_8_layer_call_and_return_conditional_losses_32576354r+,-./0 !;?8
1?.
$?!
inputs?????????]
p

 
? ")?&
?
0?????????
? ?
/__inference_sequential_8_layer_call_fn_32575117l+,-./0 !B??
8?5
+?(
lstm_16_input?????????]
p 

 
? "???????????
/__inference_sequential_8_layer_call_fn_32575607l+,-./0 !B??
8?5
+?(
lstm_16_input?????????]
p

 
? "???????????
/__inference_sequential_8_layer_call_fn_32576375e+,-./0 !;?8
1?.
$?!
inputs?????????]
p 

 
? "???????????
/__inference_sequential_8_layer_call_fn_32576396e+,-./0 !;?8
1?.
$?!
inputs?????????]
p

 
? "???????????
&__inference_signature_wrapper_32575686?+,-./0 !K?H
? 
A?>
<
lstm_16_input+?(
lstm_16_input?????????]"5?2
0
dense_8%?"
dense_8?????????