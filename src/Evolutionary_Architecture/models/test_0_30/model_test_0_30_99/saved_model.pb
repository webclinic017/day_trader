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
dense_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*
shared_namedense_6/kernel
r
"dense_6/kernel/Read/ReadVariableOpReadVariableOpdense_6/kernel*
_output_shapes
:	?*
dtype0
p
dense_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_6/bias
i
 dense_6/bias/Read/ReadVariableOpReadVariableOpdense_6/bias*
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
lstm_12/lstm_cell_12/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	]?*,
shared_namelstm_12/lstm_cell_12/kernel
?
/lstm_12/lstm_cell_12/kernel/Read/ReadVariableOpReadVariableOplstm_12/lstm_cell_12/kernel*
_output_shapes
:	]?*
dtype0
?
%lstm_12/lstm_cell_12/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*6
shared_name'%lstm_12/lstm_cell_12/recurrent_kernel
?
9lstm_12/lstm_cell_12/recurrent_kernel/Read/ReadVariableOpReadVariableOp%lstm_12/lstm_cell_12/recurrent_kernel* 
_output_shapes
:
??*
dtype0
?
lstm_12/lstm_cell_12/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?**
shared_namelstm_12/lstm_cell_12/bias
?
-lstm_12/lstm_cell_12/bias/Read/ReadVariableOpReadVariableOplstm_12/lstm_cell_12/bias*
_output_shapes	
:?*
dtype0
?
lstm_13/lstm_cell_13/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*,
shared_namelstm_13/lstm_cell_13/kernel
?
/lstm_13/lstm_cell_13/kernel/Read/ReadVariableOpReadVariableOplstm_13/lstm_cell_13/kernel* 
_output_shapes
:
??*
dtype0
?
%lstm_13/lstm_cell_13/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*6
shared_name'%lstm_13/lstm_cell_13/recurrent_kernel
?
9lstm_13/lstm_cell_13/recurrent_kernel/Read/ReadVariableOpReadVariableOp%lstm_13/lstm_cell_13/recurrent_kernel* 
_output_shapes
:
??*
dtype0
?
lstm_13/lstm_cell_13/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?**
shared_namelstm_13/lstm_cell_13/bias
?
-lstm_13/lstm_cell_13/bias/Read/ReadVariableOpReadVariableOplstm_13/lstm_cell_13/bias*
_output_shapes	
:?*
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
Adam/dense_6/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*&
shared_nameAdam/dense_6/kernel/m
?
)Adam/dense_6/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_6/kernel/m*
_output_shapes
:	?*
dtype0
~
Adam/dense_6/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_6/bias/m
w
'Adam/dense_6/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_6/bias/m*
_output_shapes
:*
dtype0
?
"Adam/lstm_12/lstm_cell_12/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	]?*3
shared_name$"Adam/lstm_12/lstm_cell_12/kernel/m
?
6Adam/lstm_12/lstm_cell_12/kernel/m/Read/ReadVariableOpReadVariableOp"Adam/lstm_12/lstm_cell_12/kernel/m*
_output_shapes
:	]?*
dtype0
?
,Adam/lstm_12/lstm_cell_12/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*=
shared_name.,Adam/lstm_12/lstm_cell_12/recurrent_kernel/m
?
@Adam/lstm_12/lstm_cell_12/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp,Adam/lstm_12/lstm_cell_12/recurrent_kernel/m* 
_output_shapes
:
??*
dtype0
?
 Adam/lstm_12/lstm_cell_12/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*1
shared_name" Adam/lstm_12/lstm_cell_12/bias/m
?
4Adam/lstm_12/lstm_cell_12/bias/m/Read/ReadVariableOpReadVariableOp Adam/lstm_12/lstm_cell_12/bias/m*
_output_shapes	
:?*
dtype0
?
"Adam/lstm_13/lstm_cell_13/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*3
shared_name$"Adam/lstm_13/lstm_cell_13/kernel/m
?
6Adam/lstm_13/lstm_cell_13/kernel/m/Read/ReadVariableOpReadVariableOp"Adam/lstm_13/lstm_cell_13/kernel/m* 
_output_shapes
:
??*
dtype0
?
,Adam/lstm_13/lstm_cell_13/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*=
shared_name.,Adam/lstm_13/lstm_cell_13/recurrent_kernel/m
?
@Adam/lstm_13/lstm_cell_13/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp,Adam/lstm_13/lstm_cell_13/recurrent_kernel/m* 
_output_shapes
:
??*
dtype0
?
 Adam/lstm_13/lstm_cell_13/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*1
shared_name" Adam/lstm_13/lstm_cell_13/bias/m
?
4Adam/lstm_13/lstm_cell_13/bias/m/Read/ReadVariableOpReadVariableOp Adam/lstm_13/lstm_cell_13/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/dense_6/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*&
shared_nameAdam/dense_6/kernel/v
?
)Adam/dense_6/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_6/kernel/v*
_output_shapes
:	?*
dtype0
~
Adam/dense_6/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_6/bias/v
w
'Adam/dense_6/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_6/bias/v*
_output_shapes
:*
dtype0
?
"Adam/lstm_12/lstm_cell_12/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	]?*3
shared_name$"Adam/lstm_12/lstm_cell_12/kernel/v
?
6Adam/lstm_12/lstm_cell_12/kernel/v/Read/ReadVariableOpReadVariableOp"Adam/lstm_12/lstm_cell_12/kernel/v*
_output_shapes
:	]?*
dtype0
?
,Adam/lstm_12/lstm_cell_12/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*=
shared_name.,Adam/lstm_12/lstm_cell_12/recurrent_kernel/v
?
@Adam/lstm_12/lstm_cell_12/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp,Adam/lstm_12/lstm_cell_12/recurrent_kernel/v* 
_output_shapes
:
??*
dtype0
?
 Adam/lstm_12/lstm_cell_12/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*1
shared_name" Adam/lstm_12/lstm_cell_12/bias/v
?
4Adam/lstm_12/lstm_cell_12/bias/v/Read/ReadVariableOpReadVariableOp Adam/lstm_12/lstm_cell_12/bias/v*
_output_shapes	
:?*
dtype0
?
"Adam/lstm_13/lstm_cell_13/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*3
shared_name$"Adam/lstm_13/lstm_cell_13/kernel/v
?
6Adam/lstm_13/lstm_cell_13/kernel/v/Read/ReadVariableOpReadVariableOp"Adam/lstm_13/lstm_cell_13/kernel/v* 
_output_shapes
:
??*
dtype0
?
,Adam/lstm_13/lstm_cell_13/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*=
shared_name.,Adam/lstm_13/lstm_cell_13/recurrent_kernel/v
?
@Adam/lstm_13/lstm_cell_13/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp,Adam/lstm_13/lstm_cell_13/recurrent_kernel/v* 
_output_shapes
:
??*
dtype0
?
 Adam/lstm_13/lstm_cell_13/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*1
shared_name" Adam/lstm_13/lstm_cell_13/bias/v
?
4Adam/lstm_13/lstm_cell_13/bias/v/Read/ReadVariableOpReadVariableOp Adam/lstm_13/lstm_cell_13/bias/v*
_output_shapes	
:?*
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

1layers
trainable_variables
regularization_losses
		variables
2layer_metrics
3non_trainable_variables
4metrics
5layer_regularization_losses
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

;layers
trainable_variables
regularization_losses
	variables

<states
=layer_metrics
>non_trainable_variables
?metrics
@layer_regularization_losses
 
 
 
?

Alayers
trainable_variables
regularization_losses
	variables
Blayer_metrics
Cnon_trainable_variables
Dmetrics
Elayer_regularization_losses
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

Klayers
trainable_variables
regularization_losses
	variables

Lstates
Mlayer_metrics
Nnon_trainable_variables
Ometrics
Player_regularization_losses
 
 
 
?

Qlayers
trainable_variables
regularization_losses
	variables
Rlayer_metrics
Snon_trainable_variables
Tmetrics
Ulayer_regularization_losses
ZX
VARIABLE_VALUEdense_6/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_6/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

 0
!1
 

 0
!1
?

Vlayers
"trainable_variables
#regularization_losses
$	variables
Wlayer_metrics
Xnon_trainable_variables
Ymetrics
Zlayer_regularization_losses
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
VARIABLE_VALUElstm_12/lstm_cell_12/kernel0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUE
ki
VARIABLE_VALUE%lstm_12/lstm_cell_12/recurrent_kernel0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUElstm_12/lstm_cell_12/bias0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUElstm_13/lstm_cell_13/kernel0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUE
ki
VARIABLE_VALUE%lstm_13/lstm_cell_13/recurrent_kernel0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUElstm_13/lstm_cell_13/bias0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUE
#
0
1
2
3
4
 
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
?

]layers
7trainable_variables
8regularization_losses
9	variables
^layer_metrics
_non_trainable_variables
`metrics
alayer_regularization_losses
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
.0
/1
02
 

.0
/1
02
?

blayers
Gtrainable_variables
Hregularization_losses
I	variables
clayer_metrics
dnon_trainable_variables
emetrics
flayer_regularization_losses
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
VARIABLE_VALUEAdam/dense_6/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_6/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"Adam/lstm_12/lstm_cell_12/kernel/mLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE,Adam/lstm_12/lstm_cell_12/recurrent_kernel/mLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE Adam/lstm_12/lstm_cell_12/bias/mLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"Adam/lstm_13/lstm_cell_13/kernel/mLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE,Adam/lstm_13/lstm_cell_13/recurrent_kernel/mLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE Adam/lstm_13/lstm_cell_13/bias/mLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_6/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_6/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"Adam/lstm_12/lstm_cell_12/kernel/vLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE,Adam/lstm_12/lstm_cell_12/recurrent_kernel/vLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE Adam/lstm_12/lstm_cell_12/bias/vLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"Adam/lstm_13/lstm_cell_13/kernel/vLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE,Adam/lstm_13/lstm_cell_13/recurrent_kernel/vLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE Adam/lstm_13/lstm_cell_13/bias/vLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
serving_default_lstm_12_inputPlaceholder*+
_output_shapes
:?????????]*
dtype0* 
shape:?????????]
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_lstm_12_inputlstm_12/lstm_cell_12/kernel%lstm_12/lstm_cell_12/recurrent_kernellstm_12/lstm_cell_12/biaslstm_13/lstm_cell_13/kernel%lstm_13/lstm_cell_13/recurrent_kernellstm_13/lstm_cell_13/biasdense_6/kerneldense_6/bias*
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
&__inference_signature_wrapper_26066784
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename"dense_6/kernel/Read/ReadVariableOp dense_6/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp/lstm_12/lstm_cell_12/kernel/Read/ReadVariableOp9lstm_12/lstm_cell_12/recurrent_kernel/Read/ReadVariableOp-lstm_12/lstm_cell_12/bias/Read/ReadVariableOp/lstm_13/lstm_cell_13/kernel/Read/ReadVariableOp9lstm_13/lstm_cell_13/recurrent_kernel/Read/ReadVariableOp-lstm_13/lstm_cell_13/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp)Adam/dense_6/kernel/m/Read/ReadVariableOp'Adam/dense_6/bias/m/Read/ReadVariableOp6Adam/lstm_12/lstm_cell_12/kernel/m/Read/ReadVariableOp@Adam/lstm_12/lstm_cell_12/recurrent_kernel/m/Read/ReadVariableOp4Adam/lstm_12/lstm_cell_12/bias/m/Read/ReadVariableOp6Adam/lstm_13/lstm_cell_13/kernel/m/Read/ReadVariableOp@Adam/lstm_13/lstm_cell_13/recurrent_kernel/m/Read/ReadVariableOp4Adam/lstm_13/lstm_cell_13/bias/m/Read/ReadVariableOp)Adam/dense_6/kernel/v/Read/ReadVariableOp'Adam/dense_6/bias/v/Read/ReadVariableOp6Adam/lstm_12/lstm_cell_12/kernel/v/Read/ReadVariableOp@Adam/lstm_12/lstm_cell_12/recurrent_kernel/v/Read/ReadVariableOp4Adam/lstm_12/lstm_cell_12/bias/v/Read/ReadVariableOp6Adam/lstm_13/lstm_cell_13/kernel/v/Read/ReadVariableOp@Adam/lstm_13/lstm_cell_13/recurrent_kernel/v/Read/ReadVariableOp4Adam/lstm_13/lstm_cell_13/bias/v/Read/ReadVariableOpConst*.
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
!__inference__traced_save_26069202
?	
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_6/kerneldense_6/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratelstm_12/lstm_cell_12/kernel%lstm_12/lstm_cell_12/recurrent_kernellstm_12/lstm_cell_12/biaslstm_13/lstm_cell_13/kernel%lstm_13/lstm_cell_13/recurrent_kernellstm_13/lstm_cell_13/biastotalcounttotal_1count_1Adam/dense_6/kernel/mAdam/dense_6/bias/m"Adam/lstm_12/lstm_cell_12/kernel/m,Adam/lstm_12/lstm_cell_12/recurrent_kernel/m Adam/lstm_12/lstm_cell_12/bias/m"Adam/lstm_13/lstm_cell_13/kernel/m,Adam/lstm_13/lstm_cell_13/recurrent_kernel/m Adam/lstm_13/lstm_cell_13/bias/mAdam/dense_6/kernel/vAdam/dense_6/bias/v"Adam/lstm_12/lstm_cell_12/kernel/v,Adam/lstm_12/lstm_cell_12/recurrent_kernel/v Adam/lstm_12/lstm_cell_12/bias/v"Adam/lstm_13/lstm_cell_13/kernel/v,Adam/lstm_13/lstm_cell_13/recurrent_kernel/v Adam/lstm_13/lstm_cell_13/bias/v*-
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
$__inference__traced_restore_26069311??$
?
?
*__inference_lstm_12_layer_call_fn_26068109
inputs_0
unknown:	]?
	unknown_0:
??
	unknown_1:	?
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
E__inference_lstm_12_layer_call_and_return_conditional_losses_260647192
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
?!
?
E__inference_dense_6_layer_call_and_return_conditional_losses_26068875

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
?
?
*__inference_lstm_13_layer_call_fn_26068784
inputs_0
unknown:
??
	unknown_0:
??
	unknown_1:	?
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
E__inference_lstm_13_layer_call_and_return_conditional_losses_260653492
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
':???????????????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
5
_output_shapes#
!:???????????????????
"
_user_specified_name
inputs/0
?
?
while_cond_26068013
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_26068013___redundant_placeholder06
2while_while_cond_26068013___redundant_placeholder16
2while_while_cond_26068013___redundant_placeholder26
2while_while_cond_26068013___redundant_placeholder3
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
/__inference_sequential_6_layer_call_fn_26067494

inputs
unknown:	]?
	unknown_0:
??
	unknown_1:	?
	unknown_2:
??
	unknown_3:
??
	unknown_4:	?
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
J__inference_sequential_6_layer_call_and_return_conditional_losses_260666652
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
?
J__inference_sequential_6_layer_call_and_return_conditional_losses_26066665

inputs#
lstm_12_26066643:	]?$
lstm_12_26066645:
??
lstm_12_26066647:	?$
lstm_13_26066651:
??$
lstm_13_26066653:
??
lstm_13_26066655:	?#
dense_6_26066659:	?
dense_6_26066661:
identity??dense_6/StatefulPartitionedCall?"dropout_12/StatefulPartitionedCall?"dropout_13/StatefulPartitionedCall?lstm_12/StatefulPartitionedCall?lstm_13/StatefulPartitionedCall?
lstm_12/StatefulPartitionedCallStatefulPartitionedCallinputslstm_12_26066643lstm_12_26066645lstm_12_26066647*
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
E__inference_lstm_12_layer_call_and_return_conditional_losses_260666082!
lstm_12/StatefulPartitionedCall?
"dropout_12/StatefulPartitionedCallStatefulPartitionedCall(lstm_12/StatefulPartitionedCall:output:0*
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
H__inference_dropout_12_layer_call_and_return_conditional_losses_260664412$
"dropout_12/StatefulPartitionedCall?
lstm_13/StatefulPartitionedCallStatefulPartitionedCall+dropout_12/StatefulPartitionedCall:output:0lstm_13_26066651lstm_13_26066653lstm_13_26066655*
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
E__inference_lstm_13_layer_call_and_return_conditional_losses_260664122!
lstm_13/StatefulPartitionedCall?
"dropout_13/StatefulPartitionedCallStatefulPartitionedCall(lstm_13/StatefulPartitionedCall:output:0#^dropout_12/StatefulPartitionedCall*
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
H__inference_dropout_13_layer_call_and_return_conditional_losses_260662452$
"dropout_13/StatefulPartitionedCall?
dense_6/StatefulPartitionedCallStatefulPartitionedCall+dropout_13/StatefulPartitionedCall:output:0dense_6_26066659dense_6_26066661*
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
E__inference_dense_6_layer_call_and_return_conditional_losses_260661892!
dense_6/StatefulPartitionedCall?
IdentityIdentity(dense_6/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????2

Identity?
NoOpNoOp ^dense_6/StatefulPartitionedCall#^dropout_12/StatefulPartitionedCall#^dropout_13/StatefulPartitionedCall ^lstm_12/StatefulPartitionedCall ^lstm_13/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????]: : : : : : : : 2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2H
"dropout_12/StatefulPartitionedCall"dropout_12/StatefulPartitionedCall2H
"dropout_13/StatefulPartitionedCall"dropout_13/StatefulPartitionedCall2B
lstm_12/StatefulPartitionedCalllstm_12/StatefulPartitionedCall2B
lstm_13/StatefulPartitionedCalllstm_13/StatefulPartitionedCall:S O
+
_output_shapes
:?????????]
 
_user_specified_nameinputs
?
f
H__inference_dropout_12_layer_call_and_return_conditional_losses_26068147

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
??
?
while_body_26068689
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
3while_lstm_cell_13_matmul_readvariableop_resource_0:
??I
5while_lstm_cell_13_matmul_1_readvariableop_resource_0:
??C
4while_lstm_cell_13_biasadd_readvariableop_resource_0:	?
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
1while_lstm_cell_13_matmul_readvariableop_resource:
??G
3while_lstm_cell_13_matmul_1_readvariableop_resource:
??A
2while_lstm_cell_13_biasadd_readvariableop_resource:	???)while/lstm_cell_13/BiasAdd/ReadVariableOp?(while/lstm_cell_13/MatMul/ReadVariableOp?*while/lstm_cell_13/MatMul_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:??????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
(while/lstm_cell_13/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_13_matmul_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02*
(while/lstm_cell_13/MatMul/ReadVariableOp?
while/lstm_cell_13/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_13/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_13/MatMul?
*while/lstm_cell_13/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_13_matmul_1_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02,
*while/lstm_cell_13/MatMul_1/ReadVariableOp?
while/lstm_cell_13/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_13/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_13/MatMul_1?
while/lstm_cell_13/addAddV2#while/lstm_cell_13/MatMul:product:0%while/lstm_cell_13/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_13/add?
)while/lstm_cell_13/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_13_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype02+
)while/lstm_cell_13/BiasAdd/ReadVariableOp?
while/lstm_cell_13/BiasAddBiasAddwhile/lstm_cell_13/add:z:01while/lstm_cell_13/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_13/BiasAdd?
"while/lstm_cell_13/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_13/split/split_dim?
while/lstm_cell_13/splitSplit+while/lstm_cell_13/split/split_dim:output:0#while/lstm_cell_13/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split2
while/lstm_cell_13/split?
while/lstm_cell_13/SigmoidSigmoid!while/lstm_cell_13/split:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_13/Sigmoid?
while/lstm_cell_13/Sigmoid_1Sigmoid!while/lstm_cell_13/split:output:1*
T0*(
_output_shapes
:??????????2
while/lstm_cell_13/Sigmoid_1?
while/lstm_cell_13/mulMul while/lstm_cell_13/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:??????????2
while/lstm_cell_13/mul?
while/lstm_cell_13/ReluRelu!while/lstm_cell_13/split:output:2*
T0*(
_output_shapes
:??????????2
while/lstm_cell_13/Relu?
while/lstm_cell_13/mul_1Mulwhile/lstm_cell_13/Sigmoid:y:0%while/lstm_cell_13/Relu:activations:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_13/mul_1?
while/lstm_cell_13/add_1AddV2while/lstm_cell_13/mul:z:0while/lstm_cell_13/mul_1:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_13/add_1?
while/lstm_cell_13/Sigmoid_2Sigmoid!while/lstm_cell_13/split:output:3*
T0*(
_output_shapes
:??????????2
while/lstm_cell_13/Sigmoid_2?
while/lstm_cell_13/Relu_1Reluwhile/lstm_cell_13/add_1:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_13/Relu_1?
while/lstm_cell_13/mul_2Mul while/lstm_cell_13/Sigmoid_2:y:0'while/lstm_cell_13/Relu_1:activations:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_13/mul_2?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_13/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_13/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:??????????2
while/Identity_4?
while/Identity_5Identitywhile/lstm_cell_13/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:??????????2
while/Identity_5?

while/NoOpNoOp*^while/lstm_cell_13/BiasAdd/ReadVariableOp)^while/lstm_cell_13/MatMul/ReadVariableOp+^while/lstm_cell_13/MatMul_1/ReadVariableOp*"
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
2while_lstm_cell_13_biasadd_readvariableop_resource4while_lstm_cell_13_biasadd_readvariableop_resource_0"l
3while_lstm_cell_13_matmul_1_readvariableop_resource5while_lstm_cell_13_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_13_matmul_readvariableop_resource3while_lstm_cell_13_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :??????????:??????????: : : : : 2V
)while/lstm_cell_13/BiasAdd/ReadVariableOp)while/lstm_cell_13/BiasAdd/ReadVariableOp2T
(while/lstm_cell_13/MatMul/ReadVariableOp(while/lstm_cell_13/MatMul/ReadVariableOp2X
*while/lstm_cell_13/MatMul_1/ReadVariableOp*while/lstm_cell_13/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
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
while_body_26068014
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_12_matmul_readvariableop_resource_0:	]?I
5while_lstm_cell_12_matmul_1_readvariableop_resource_0:
??C
4while_lstm_cell_12_biasadd_readvariableop_resource_0:	?
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_12_matmul_readvariableop_resource:	]?G
3while_lstm_cell_12_matmul_1_readvariableop_resource:
??A
2while_lstm_cell_12_biasadd_readvariableop_resource:	???)while/lstm_cell_12/BiasAdd/ReadVariableOp?(while/lstm_cell_12/MatMul/ReadVariableOp?*while/lstm_cell_12/MatMul_1/ReadVariableOp?
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
(while/lstm_cell_12/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_12_matmul_readvariableop_resource_0*
_output_shapes
:	]?*
dtype02*
(while/lstm_cell_12/MatMul/ReadVariableOp?
while/lstm_cell_12/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_12/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_12/MatMul?
*while/lstm_cell_12/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_12_matmul_1_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02,
*while/lstm_cell_12/MatMul_1/ReadVariableOp?
while/lstm_cell_12/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_12/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_12/MatMul_1?
while/lstm_cell_12/addAddV2#while/lstm_cell_12/MatMul:product:0%while/lstm_cell_12/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_12/add?
)while/lstm_cell_12/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_12_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype02+
)while/lstm_cell_12/BiasAdd/ReadVariableOp?
while/lstm_cell_12/BiasAddBiasAddwhile/lstm_cell_12/add:z:01while/lstm_cell_12/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_12/BiasAdd?
"while/lstm_cell_12/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_12/split/split_dim?
while/lstm_cell_12/splitSplit+while/lstm_cell_12/split/split_dim:output:0#while/lstm_cell_12/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split2
while/lstm_cell_12/split?
while/lstm_cell_12/SigmoidSigmoid!while/lstm_cell_12/split:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_12/Sigmoid?
while/lstm_cell_12/Sigmoid_1Sigmoid!while/lstm_cell_12/split:output:1*
T0*(
_output_shapes
:??????????2
while/lstm_cell_12/Sigmoid_1?
while/lstm_cell_12/mulMul while/lstm_cell_12/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:??????????2
while/lstm_cell_12/mul?
while/lstm_cell_12/ReluRelu!while/lstm_cell_12/split:output:2*
T0*(
_output_shapes
:??????????2
while/lstm_cell_12/Relu?
while/lstm_cell_12/mul_1Mulwhile/lstm_cell_12/Sigmoid:y:0%while/lstm_cell_12/Relu:activations:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_12/mul_1?
while/lstm_cell_12/add_1AddV2while/lstm_cell_12/mul:z:0while/lstm_cell_12/mul_1:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_12/add_1?
while/lstm_cell_12/Sigmoid_2Sigmoid!while/lstm_cell_12/split:output:3*
T0*(
_output_shapes
:??????????2
while/lstm_cell_12/Sigmoid_2?
while/lstm_cell_12/Relu_1Reluwhile/lstm_cell_12/add_1:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_12/Relu_1?
while/lstm_cell_12/mul_2Mul while/lstm_cell_12/Sigmoid_2:y:0'while/lstm_cell_12/Relu_1:activations:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_12/mul_2?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_12/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_12/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:??????????2
while/Identity_4?
while/Identity_5Identitywhile/lstm_cell_12/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:??????????2
while/Identity_5?

while/NoOpNoOp*^while/lstm_cell_12/BiasAdd/ReadVariableOp)^while/lstm_cell_12/MatMul/ReadVariableOp+^while/lstm_cell_12/MatMul_1/ReadVariableOp*"
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
2while_lstm_cell_12_biasadd_readvariableop_resource4while_lstm_cell_12_biasadd_readvariableop_resource_0"l
3while_lstm_cell_12_matmul_1_readvariableop_resource5while_lstm_cell_12_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_12_matmul_readvariableop_resource3while_lstm_cell_12_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :??????????:??????????: : : : : 2V
)while/lstm_cell_12/BiasAdd/ReadVariableOp)while/lstm_cell_12/BiasAdd/ReadVariableOp2T
(while/lstm_cell_12/MatMul/ReadVariableOp(while/lstm_cell_12/MatMul/ReadVariableOp2X
*while/lstm_cell_12/MatMul_1/ReadVariableOp*while/lstm_cell_12/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
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
?
J__inference_lstm_cell_13_layer_call_and_return_conditional_losses_26069014

inputs
states_0
states_12
matmul_readvariableop_resource:
??4
 matmul_1_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity

identity_1

identity_2??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?MatMul_1/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul_1/ReadVariableOp|
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
add?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
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
B:??????????:??????????:??????????: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:P L
(
_output_shapes
:??????????
 
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
while_cond_26067560
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_26067560___redundant_placeholder06
2while_while_cond_26067560___redundant_placeholder16
2while_while_cond_26067560___redundant_placeholder26
2while_while_cond_26067560___redundant_placeholder3
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
?
?
J__inference_sequential_6_layer_call_and_return_conditional_losses_26066730
lstm_12_input#
lstm_12_26066708:	]?$
lstm_12_26066710:
??
lstm_12_26066712:	?$
lstm_13_26066716:
??$
lstm_13_26066718:
??
lstm_13_26066720:	?#
dense_6_26066724:	?
dense_6_26066726:
identity??dense_6/StatefulPartitionedCall?lstm_12/StatefulPartitionedCall?lstm_13/StatefulPartitionedCall?
lstm_12/StatefulPartitionedCallStatefulPartitionedCalllstm_12_inputlstm_12_26066708lstm_12_26066710lstm_12_26066712*
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
E__inference_lstm_12_layer_call_and_return_conditional_losses_260659782!
lstm_12/StatefulPartitionedCall?
dropout_12/PartitionedCallPartitionedCall(lstm_12/StatefulPartitionedCall:output:0*
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
H__inference_dropout_12_layer_call_and_return_conditional_losses_260659912
dropout_12/PartitionedCall?
lstm_13/StatefulPartitionedCallStatefulPartitionedCall#dropout_12/PartitionedCall:output:0lstm_13_26066716lstm_13_26066718lstm_13_26066720*
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
E__inference_lstm_13_layer_call_and_return_conditional_losses_260661432!
lstm_13/StatefulPartitionedCall?
dropout_13/PartitionedCallPartitionedCall(lstm_13/StatefulPartitionedCall:output:0*
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
H__inference_dropout_13_layer_call_and_return_conditional_losses_260661562
dropout_13/PartitionedCall?
dense_6/StatefulPartitionedCallStatefulPartitionedCall#dropout_13/PartitionedCall:output:0dense_6_26066724dense_6_26066726*
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
E__inference_dense_6_layer_call_and_return_conditional_losses_260661892!
dense_6/StatefulPartitionedCall?
IdentityIdentity(dense_6/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????2

Identity?
NoOpNoOp ^dense_6/StatefulPartitionedCall ^lstm_12/StatefulPartitionedCall ^lstm_13/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????]: : : : : : : : 2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2B
lstm_12/StatefulPartitionedCalllstm_12/StatefulPartitionedCall2B
lstm_13/StatefulPartitionedCalllstm_13/StatefulPartitionedCall:Z V
+
_output_shapes
:?????????]
'
_user_specified_namelstm_12_input
?F
?
E__inference_lstm_12_layer_call_and_return_conditional_losses_26064719

inputs(
lstm_cell_12_26064637:	]?)
lstm_cell_12_26064639:
??$
lstm_cell_12_26064641:	?
identity??$lstm_cell_12/StatefulPartitionedCall?whileD
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
$lstm_cell_12/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_12_26064637lstm_cell_12_26064639lstm_cell_12_26064641*
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
J__inference_lstm_cell_12_layer_call_and_return_conditional_losses_260646362&
$lstm_cell_12/StatefulPartitionedCall?
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_12_26064637lstm_cell_12_26064639lstm_cell_12_26064641*
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
while_body_26064650*
condR
while_cond_26064649*M
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
NoOpNoOp%^lstm_cell_12/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????????????]: : : 2L
$lstm_cell_12/StatefulPartitionedCall$lstm_cell_12/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :??????????????????]
 
_user_specified_nameinputs
?

?
lstm_13_while_cond_26066998,
(lstm_13_while_lstm_13_while_loop_counter2
.lstm_13_while_lstm_13_while_maximum_iterations
lstm_13_while_placeholder
lstm_13_while_placeholder_1
lstm_13_while_placeholder_2
lstm_13_while_placeholder_3.
*lstm_13_while_less_lstm_13_strided_slice_1F
Blstm_13_while_lstm_13_while_cond_26066998___redundant_placeholder0F
Blstm_13_while_lstm_13_while_cond_26066998___redundant_placeholder1F
Blstm_13_while_lstm_13_while_cond_26066998___redundant_placeholder2F
Blstm_13_while_lstm_13_while_cond_26066998___redundant_placeholder3
lstm_13_while_identity
?
lstm_13/while/LessLesslstm_13_while_placeholder*lstm_13_while_less_lstm_13_strided_slice_1*
T0*
_output_shapes
: 2
lstm_13/while/Lessu
lstm_13/while/IdentityIdentitylstm_13/while/Less:z:0*
T0
*
_output_shapes
: 2
lstm_13/while/Identity"9
lstm_13_while_identitylstm_13/while/Identity:output:0*(
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
E__inference_lstm_12_layer_call_and_return_conditional_losses_26068098

inputs>
+lstm_cell_12_matmul_readvariableop_resource:	]?A
-lstm_cell_12_matmul_1_readvariableop_resource:
??;
,lstm_cell_12_biasadd_readvariableop_resource:	?
identity??#lstm_cell_12/BiasAdd/ReadVariableOp?"lstm_cell_12/MatMul/ReadVariableOp?$lstm_cell_12/MatMul_1/ReadVariableOp?whileD
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
"lstm_cell_12/MatMul/ReadVariableOpReadVariableOp+lstm_cell_12_matmul_readvariableop_resource*
_output_shapes
:	]?*
dtype02$
"lstm_cell_12/MatMul/ReadVariableOp?
lstm_cell_12/MatMulMatMulstrided_slice_2:output:0*lstm_cell_12/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_12/MatMul?
$lstm_cell_12/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_12_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02&
$lstm_cell_12/MatMul_1/ReadVariableOp?
lstm_cell_12/MatMul_1MatMulzeros:output:0,lstm_cell_12/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_12/MatMul_1?
lstm_cell_12/addAddV2lstm_cell_12/MatMul:product:0lstm_cell_12/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell_12/add?
#lstm_cell_12/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_12_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02%
#lstm_cell_12/BiasAdd/ReadVariableOp?
lstm_cell_12/BiasAddBiasAddlstm_cell_12/add:z:0+lstm_cell_12/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_12/BiasAdd~
lstm_cell_12/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_12/split/split_dim?
lstm_cell_12/splitSplit%lstm_cell_12/split/split_dim:output:0lstm_cell_12/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split2
lstm_cell_12/split?
lstm_cell_12/SigmoidSigmoidlstm_cell_12/split:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_12/Sigmoid?
lstm_cell_12/Sigmoid_1Sigmoidlstm_cell_12/split:output:1*
T0*(
_output_shapes
:??????????2
lstm_cell_12/Sigmoid_1?
lstm_cell_12/mulMullstm_cell_12/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_12/mul~
lstm_cell_12/ReluRelulstm_cell_12/split:output:2*
T0*(
_output_shapes
:??????????2
lstm_cell_12/Relu?
lstm_cell_12/mul_1Mullstm_cell_12/Sigmoid:y:0lstm_cell_12/Relu:activations:0*
T0*(
_output_shapes
:??????????2
lstm_cell_12/mul_1?
lstm_cell_12/add_1AddV2lstm_cell_12/mul:z:0lstm_cell_12/mul_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell_12/add_1?
lstm_cell_12/Sigmoid_2Sigmoidlstm_cell_12/split:output:3*
T0*(
_output_shapes
:??????????2
lstm_cell_12/Sigmoid_2}
lstm_cell_12/Relu_1Relulstm_cell_12/add_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell_12/Relu_1?
lstm_cell_12/mul_2Mullstm_cell_12/Sigmoid_2:y:0!lstm_cell_12/Relu_1:activations:0*
T0*(
_output_shapes
:??????????2
lstm_cell_12/mul_2?
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_12_matmul_readvariableop_resource-lstm_cell_12_matmul_1_readvariableop_resource,lstm_cell_12_biasadd_readvariableop_resource*
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
while_body_26068014*
condR
while_cond_26068013*M
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
NoOpNoOp$^lstm_cell_12/BiasAdd/ReadVariableOp#^lstm_cell_12/MatMul/ReadVariableOp%^lstm_cell_12/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????]: : : 2J
#lstm_cell_12/BiasAdd/ReadVariableOp#lstm_cell_12/BiasAdd/ReadVariableOp2H
"lstm_cell_12/MatMul/ReadVariableOp"lstm_cell_12/MatMul/ReadVariableOp2L
$lstm_cell_12/MatMul_1/ReadVariableOp$lstm_cell_12/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:?????????]
 
_user_specified_nameinputs
??
?
while_body_26066328
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
3while_lstm_cell_13_matmul_readvariableop_resource_0:
??I
5while_lstm_cell_13_matmul_1_readvariableop_resource_0:
??C
4while_lstm_cell_13_biasadd_readvariableop_resource_0:	?
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
1while_lstm_cell_13_matmul_readvariableop_resource:
??G
3while_lstm_cell_13_matmul_1_readvariableop_resource:
??A
2while_lstm_cell_13_biasadd_readvariableop_resource:	???)while/lstm_cell_13/BiasAdd/ReadVariableOp?(while/lstm_cell_13/MatMul/ReadVariableOp?*while/lstm_cell_13/MatMul_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:??????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
(while/lstm_cell_13/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_13_matmul_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02*
(while/lstm_cell_13/MatMul/ReadVariableOp?
while/lstm_cell_13/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_13/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_13/MatMul?
*while/lstm_cell_13/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_13_matmul_1_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02,
*while/lstm_cell_13/MatMul_1/ReadVariableOp?
while/lstm_cell_13/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_13/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_13/MatMul_1?
while/lstm_cell_13/addAddV2#while/lstm_cell_13/MatMul:product:0%while/lstm_cell_13/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_13/add?
)while/lstm_cell_13/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_13_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype02+
)while/lstm_cell_13/BiasAdd/ReadVariableOp?
while/lstm_cell_13/BiasAddBiasAddwhile/lstm_cell_13/add:z:01while/lstm_cell_13/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_13/BiasAdd?
"while/lstm_cell_13/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_13/split/split_dim?
while/lstm_cell_13/splitSplit+while/lstm_cell_13/split/split_dim:output:0#while/lstm_cell_13/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split2
while/lstm_cell_13/split?
while/lstm_cell_13/SigmoidSigmoid!while/lstm_cell_13/split:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_13/Sigmoid?
while/lstm_cell_13/Sigmoid_1Sigmoid!while/lstm_cell_13/split:output:1*
T0*(
_output_shapes
:??????????2
while/lstm_cell_13/Sigmoid_1?
while/lstm_cell_13/mulMul while/lstm_cell_13/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:??????????2
while/lstm_cell_13/mul?
while/lstm_cell_13/ReluRelu!while/lstm_cell_13/split:output:2*
T0*(
_output_shapes
:??????????2
while/lstm_cell_13/Relu?
while/lstm_cell_13/mul_1Mulwhile/lstm_cell_13/Sigmoid:y:0%while/lstm_cell_13/Relu:activations:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_13/mul_1?
while/lstm_cell_13/add_1AddV2while/lstm_cell_13/mul:z:0while/lstm_cell_13/mul_1:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_13/add_1?
while/lstm_cell_13/Sigmoid_2Sigmoid!while/lstm_cell_13/split:output:3*
T0*(
_output_shapes
:??????????2
while/lstm_cell_13/Sigmoid_2?
while/lstm_cell_13/Relu_1Reluwhile/lstm_cell_13/add_1:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_13/Relu_1?
while/lstm_cell_13/mul_2Mul while/lstm_cell_13/Sigmoid_2:y:0'while/lstm_cell_13/Relu_1:activations:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_13/mul_2?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_13/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_13/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:??????????2
while/Identity_4?
while/Identity_5Identitywhile/lstm_cell_13/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:??????????2
while/Identity_5?

while/NoOpNoOp*^while/lstm_cell_13/BiasAdd/ReadVariableOp)^while/lstm_cell_13/MatMul/ReadVariableOp+^while/lstm_cell_13/MatMul_1/ReadVariableOp*"
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
2while_lstm_cell_13_biasadd_readvariableop_resource4while_lstm_cell_13_biasadd_readvariableop_resource_0"l
3while_lstm_cell_13_matmul_1_readvariableop_resource5while_lstm_cell_13_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_13_matmul_readvariableop_resource3while_lstm_cell_13_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :??????????:??????????: : : : : 2V
)while/lstm_cell_13/BiasAdd/ReadVariableOp)while/lstm_cell_13/BiasAdd/ReadVariableOp2T
(while/lstm_cell_13/MatMul/ReadVariableOp(while/lstm_cell_13/MatMul/ReadVariableOp2X
*while/lstm_cell_13/MatMul_1/ReadVariableOp*while/lstm_cell_13/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
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
?J
?

lstm_13_while_body_26067333,
(lstm_13_while_lstm_13_while_loop_counter2
.lstm_13_while_lstm_13_while_maximum_iterations
lstm_13_while_placeholder
lstm_13_while_placeholder_1
lstm_13_while_placeholder_2
lstm_13_while_placeholder_3+
'lstm_13_while_lstm_13_strided_slice_1_0g
clstm_13_while_tensorarrayv2read_tensorlistgetitem_lstm_13_tensorarrayunstack_tensorlistfromtensor_0O
;lstm_13_while_lstm_cell_13_matmul_readvariableop_resource_0:
??Q
=lstm_13_while_lstm_cell_13_matmul_1_readvariableop_resource_0:
??K
<lstm_13_while_lstm_cell_13_biasadd_readvariableop_resource_0:	?
lstm_13_while_identity
lstm_13_while_identity_1
lstm_13_while_identity_2
lstm_13_while_identity_3
lstm_13_while_identity_4
lstm_13_while_identity_5)
%lstm_13_while_lstm_13_strided_slice_1e
alstm_13_while_tensorarrayv2read_tensorlistgetitem_lstm_13_tensorarrayunstack_tensorlistfromtensorM
9lstm_13_while_lstm_cell_13_matmul_readvariableop_resource:
??O
;lstm_13_while_lstm_cell_13_matmul_1_readvariableop_resource:
??I
:lstm_13_while_lstm_cell_13_biasadd_readvariableop_resource:	???1lstm_13/while/lstm_cell_13/BiasAdd/ReadVariableOp?0lstm_13/while/lstm_cell_13/MatMul/ReadVariableOp?2lstm_13/while/lstm_cell_13/MatMul_1/ReadVariableOp?
?lstm_13/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2A
?lstm_13/while/TensorArrayV2Read/TensorListGetItem/element_shape?
1lstm_13/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemclstm_13_while_tensorarrayv2read_tensorlistgetitem_lstm_13_tensorarrayunstack_tensorlistfromtensor_0lstm_13_while_placeholderHlstm_13/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:??????????*
element_dtype023
1lstm_13/while/TensorArrayV2Read/TensorListGetItem?
0lstm_13/while/lstm_cell_13/MatMul/ReadVariableOpReadVariableOp;lstm_13_while_lstm_cell_13_matmul_readvariableop_resource_0* 
_output_shapes
:
??*
dtype022
0lstm_13/while/lstm_cell_13/MatMul/ReadVariableOp?
!lstm_13/while/lstm_cell_13/MatMulMatMul8lstm_13/while/TensorArrayV2Read/TensorListGetItem:item:08lstm_13/while/lstm_cell_13/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2#
!lstm_13/while/lstm_cell_13/MatMul?
2lstm_13/while/lstm_cell_13/MatMul_1/ReadVariableOpReadVariableOp=lstm_13_while_lstm_cell_13_matmul_1_readvariableop_resource_0* 
_output_shapes
:
??*
dtype024
2lstm_13/while/lstm_cell_13/MatMul_1/ReadVariableOp?
#lstm_13/while/lstm_cell_13/MatMul_1MatMullstm_13_while_placeholder_2:lstm_13/while/lstm_cell_13/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2%
#lstm_13/while/lstm_cell_13/MatMul_1?
lstm_13/while/lstm_cell_13/addAddV2+lstm_13/while/lstm_cell_13/MatMul:product:0-lstm_13/while/lstm_cell_13/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2 
lstm_13/while/lstm_cell_13/add?
1lstm_13/while/lstm_cell_13/BiasAdd/ReadVariableOpReadVariableOp<lstm_13_while_lstm_cell_13_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype023
1lstm_13/while/lstm_cell_13/BiasAdd/ReadVariableOp?
"lstm_13/while/lstm_cell_13/BiasAddBiasAdd"lstm_13/while/lstm_cell_13/add:z:09lstm_13/while/lstm_cell_13/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2$
"lstm_13/while/lstm_cell_13/BiasAdd?
*lstm_13/while/lstm_cell_13/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2,
*lstm_13/while/lstm_cell_13/split/split_dim?
 lstm_13/while/lstm_cell_13/splitSplit3lstm_13/while/lstm_cell_13/split/split_dim:output:0+lstm_13/while/lstm_cell_13/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split2"
 lstm_13/while/lstm_cell_13/split?
"lstm_13/while/lstm_cell_13/SigmoidSigmoid)lstm_13/while/lstm_cell_13/split:output:0*
T0*(
_output_shapes
:??????????2$
"lstm_13/while/lstm_cell_13/Sigmoid?
$lstm_13/while/lstm_cell_13/Sigmoid_1Sigmoid)lstm_13/while/lstm_cell_13/split:output:1*
T0*(
_output_shapes
:??????????2&
$lstm_13/while/lstm_cell_13/Sigmoid_1?
lstm_13/while/lstm_cell_13/mulMul(lstm_13/while/lstm_cell_13/Sigmoid_1:y:0lstm_13_while_placeholder_3*
T0*(
_output_shapes
:??????????2 
lstm_13/while/lstm_cell_13/mul?
lstm_13/while/lstm_cell_13/ReluRelu)lstm_13/while/lstm_cell_13/split:output:2*
T0*(
_output_shapes
:??????????2!
lstm_13/while/lstm_cell_13/Relu?
 lstm_13/while/lstm_cell_13/mul_1Mul&lstm_13/while/lstm_cell_13/Sigmoid:y:0-lstm_13/while/lstm_cell_13/Relu:activations:0*
T0*(
_output_shapes
:??????????2"
 lstm_13/while/lstm_cell_13/mul_1?
 lstm_13/while/lstm_cell_13/add_1AddV2"lstm_13/while/lstm_cell_13/mul:z:0$lstm_13/while/lstm_cell_13/mul_1:z:0*
T0*(
_output_shapes
:??????????2"
 lstm_13/while/lstm_cell_13/add_1?
$lstm_13/while/lstm_cell_13/Sigmoid_2Sigmoid)lstm_13/while/lstm_cell_13/split:output:3*
T0*(
_output_shapes
:??????????2&
$lstm_13/while/lstm_cell_13/Sigmoid_2?
!lstm_13/while/lstm_cell_13/Relu_1Relu$lstm_13/while/lstm_cell_13/add_1:z:0*
T0*(
_output_shapes
:??????????2#
!lstm_13/while/lstm_cell_13/Relu_1?
 lstm_13/while/lstm_cell_13/mul_2Mul(lstm_13/while/lstm_cell_13/Sigmoid_2:y:0/lstm_13/while/lstm_cell_13/Relu_1:activations:0*
T0*(
_output_shapes
:??????????2"
 lstm_13/while/lstm_cell_13/mul_2?
2lstm_13/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_13_while_placeholder_1lstm_13_while_placeholder$lstm_13/while/lstm_cell_13/mul_2:z:0*
_output_shapes
: *
element_dtype024
2lstm_13/while/TensorArrayV2Write/TensorListSetIteml
lstm_13/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_13/while/add/y?
lstm_13/while/addAddV2lstm_13_while_placeholderlstm_13/while/add/y:output:0*
T0*
_output_shapes
: 2
lstm_13/while/addp
lstm_13/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_13/while/add_1/y?
lstm_13/while/add_1AddV2(lstm_13_while_lstm_13_while_loop_counterlstm_13/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lstm_13/while/add_1?
lstm_13/while/IdentityIdentitylstm_13/while/add_1:z:0^lstm_13/while/NoOp*
T0*
_output_shapes
: 2
lstm_13/while/Identity?
lstm_13/while/Identity_1Identity.lstm_13_while_lstm_13_while_maximum_iterations^lstm_13/while/NoOp*
T0*
_output_shapes
: 2
lstm_13/while/Identity_1?
lstm_13/while/Identity_2Identitylstm_13/while/add:z:0^lstm_13/while/NoOp*
T0*
_output_shapes
: 2
lstm_13/while/Identity_2?
lstm_13/while/Identity_3IdentityBlstm_13/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_13/while/NoOp*
T0*
_output_shapes
: 2
lstm_13/while/Identity_3?
lstm_13/while/Identity_4Identity$lstm_13/while/lstm_cell_13/mul_2:z:0^lstm_13/while/NoOp*
T0*(
_output_shapes
:??????????2
lstm_13/while/Identity_4?
lstm_13/while/Identity_5Identity$lstm_13/while/lstm_cell_13/add_1:z:0^lstm_13/while/NoOp*
T0*(
_output_shapes
:??????????2
lstm_13/while/Identity_5?
lstm_13/while/NoOpNoOp2^lstm_13/while/lstm_cell_13/BiasAdd/ReadVariableOp1^lstm_13/while/lstm_cell_13/MatMul/ReadVariableOp3^lstm_13/while/lstm_cell_13/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
lstm_13/while/NoOp"9
lstm_13_while_identitylstm_13/while/Identity:output:0"=
lstm_13_while_identity_1!lstm_13/while/Identity_1:output:0"=
lstm_13_while_identity_2!lstm_13/while/Identity_2:output:0"=
lstm_13_while_identity_3!lstm_13/while/Identity_3:output:0"=
lstm_13_while_identity_4!lstm_13/while/Identity_4:output:0"=
lstm_13_while_identity_5!lstm_13/while/Identity_5:output:0"P
%lstm_13_while_lstm_13_strided_slice_1'lstm_13_while_lstm_13_strided_slice_1_0"z
:lstm_13_while_lstm_cell_13_biasadd_readvariableop_resource<lstm_13_while_lstm_cell_13_biasadd_readvariableop_resource_0"|
;lstm_13_while_lstm_cell_13_matmul_1_readvariableop_resource=lstm_13_while_lstm_cell_13_matmul_1_readvariableop_resource_0"x
9lstm_13_while_lstm_cell_13_matmul_readvariableop_resource;lstm_13_while_lstm_cell_13_matmul_readvariableop_resource_0"?
alstm_13_while_tensorarrayv2read_tensorlistgetitem_lstm_13_tensorarrayunstack_tensorlistfromtensorclstm_13_while_tensorarrayv2read_tensorlistgetitem_lstm_13_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :??????????:??????????: : : : : 2f
1lstm_13/while/lstm_cell_13/BiasAdd/ReadVariableOp1lstm_13/while/lstm_cell_13/BiasAdd/ReadVariableOp2d
0lstm_13/while/lstm_cell_13/MatMul/ReadVariableOp0lstm_13/while/lstm_cell_13/MatMul/ReadVariableOp2h
2lstm_13/while/lstm_cell_13/MatMul_1/ReadVariableOp2lstm_13/while/lstm_cell_13/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
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
?
(sequential_6_lstm_13_while_cond_26064448F
Bsequential_6_lstm_13_while_sequential_6_lstm_13_while_loop_counterL
Hsequential_6_lstm_13_while_sequential_6_lstm_13_while_maximum_iterations*
&sequential_6_lstm_13_while_placeholder,
(sequential_6_lstm_13_while_placeholder_1,
(sequential_6_lstm_13_while_placeholder_2,
(sequential_6_lstm_13_while_placeholder_3H
Dsequential_6_lstm_13_while_less_sequential_6_lstm_13_strided_slice_1`
\sequential_6_lstm_13_while_sequential_6_lstm_13_while_cond_26064448___redundant_placeholder0`
\sequential_6_lstm_13_while_sequential_6_lstm_13_while_cond_26064448___redundant_placeholder1`
\sequential_6_lstm_13_while_sequential_6_lstm_13_while_cond_26064448___redundant_placeholder2`
\sequential_6_lstm_13_while_sequential_6_lstm_13_while_cond_26064448___redundant_placeholder3'
#sequential_6_lstm_13_while_identity
?
sequential_6/lstm_13/while/LessLess&sequential_6_lstm_13_while_placeholderDsequential_6_lstm_13_while_less_sequential_6_lstm_13_strided_slice_1*
T0*
_output_shapes
: 2!
sequential_6/lstm_13/while/Less?
#sequential_6/lstm_13/while/IdentityIdentity#sequential_6/lstm_13/while/Less:z:0*
T0
*
_output_shapes
: 2%
#sequential_6/lstm_13/while/Identity"S
#sequential_6_lstm_13_while_identity,sequential_6/lstm_13/while/Identity:output:0*(
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
??
?
while_body_26068236
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
3while_lstm_cell_13_matmul_readvariableop_resource_0:
??I
5while_lstm_cell_13_matmul_1_readvariableop_resource_0:
??C
4while_lstm_cell_13_biasadd_readvariableop_resource_0:	?
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
1while_lstm_cell_13_matmul_readvariableop_resource:
??G
3while_lstm_cell_13_matmul_1_readvariableop_resource:
??A
2while_lstm_cell_13_biasadd_readvariableop_resource:	???)while/lstm_cell_13/BiasAdd/ReadVariableOp?(while/lstm_cell_13/MatMul/ReadVariableOp?*while/lstm_cell_13/MatMul_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:??????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
(while/lstm_cell_13/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_13_matmul_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02*
(while/lstm_cell_13/MatMul/ReadVariableOp?
while/lstm_cell_13/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_13/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_13/MatMul?
*while/lstm_cell_13/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_13_matmul_1_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02,
*while/lstm_cell_13/MatMul_1/ReadVariableOp?
while/lstm_cell_13/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_13/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_13/MatMul_1?
while/lstm_cell_13/addAddV2#while/lstm_cell_13/MatMul:product:0%while/lstm_cell_13/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_13/add?
)while/lstm_cell_13/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_13_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype02+
)while/lstm_cell_13/BiasAdd/ReadVariableOp?
while/lstm_cell_13/BiasAddBiasAddwhile/lstm_cell_13/add:z:01while/lstm_cell_13/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_13/BiasAdd?
"while/lstm_cell_13/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_13/split/split_dim?
while/lstm_cell_13/splitSplit+while/lstm_cell_13/split/split_dim:output:0#while/lstm_cell_13/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split2
while/lstm_cell_13/split?
while/lstm_cell_13/SigmoidSigmoid!while/lstm_cell_13/split:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_13/Sigmoid?
while/lstm_cell_13/Sigmoid_1Sigmoid!while/lstm_cell_13/split:output:1*
T0*(
_output_shapes
:??????????2
while/lstm_cell_13/Sigmoid_1?
while/lstm_cell_13/mulMul while/lstm_cell_13/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:??????????2
while/lstm_cell_13/mul?
while/lstm_cell_13/ReluRelu!while/lstm_cell_13/split:output:2*
T0*(
_output_shapes
:??????????2
while/lstm_cell_13/Relu?
while/lstm_cell_13/mul_1Mulwhile/lstm_cell_13/Sigmoid:y:0%while/lstm_cell_13/Relu:activations:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_13/mul_1?
while/lstm_cell_13/add_1AddV2while/lstm_cell_13/mul:z:0while/lstm_cell_13/mul_1:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_13/add_1?
while/lstm_cell_13/Sigmoid_2Sigmoid!while/lstm_cell_13/split:output:3*
T0*(
_output_shapes
:??????????2
while/lstm_cell_13/Sigmoid_2?
while/lstm_cell_13/Relu_1Reluwhile/lstm_cell_13/add_1:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_13/Relu_1?
while/lstm_cell_13/mul_2Mul while/lstm_cell_13/Sigmoid_2:y:0'while/lstm_cell_13/Relu_1:activations:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_13/mul_2?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_13/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_13/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:??????????2
while/Identity_4?
while/Identity_5Identitywhile/lstm_cell_13/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:??????????2
while/Identity_5?

while/NoOpNoOp*^while/lstm_cell_13/BiasAdd/ReadVariableOp)^while/lstm_cell_13/MatMul/ReadVariableOp+^while/lstm_cell_13/MatMul_1/ReadVariableOp*"
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
2while_lstm_cell_13_biasadd_readvariableop_resource4while_lstm_cell_13_biasadd_readvariableop_resource_0"l
3while_lstm_cell_13_matmul_1_readvariableop_resource5while_lstm_cell_13_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_13_matmul_readvariableop_resource3while_lstm_cell_13_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :??????????:??????????: : : : : 2V
)while/lstm_cell_13/BiasAdd/ReadVariableOp)while/lstm_cell_13/BiasAdd/ReadVariableOp2T
(while/lstm_cell_13/MatMul/ReadVariableOp(while/lstm_cell_13/MatMul/ReadVariableOp2X
*while/lstm_cell_13/MatMul_1/ReadVariableOp*while/lstm_cell_13/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
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
?
f
H__inference_dropout_13_layer_call_and_return_conditional_losses_26068822

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
??
?
J__inference_sequential_6_layer_call_and_return_conditional_losses_26067452

inputsF
3lstm_12_lstm_cell_12_matmul_readvariableop_resource:	]?I
5lstm_12_lstm_cell_12_matmul_1_readvariableop_resource:
??C
4lstm_12_lstm_cell_12_biasadd_readvariableop_resource:	?G
3lstm_13_lstm_cell_13_matmul_readvariableop_resource:
??I
5lstm_13_lstm_cell_13_matmul_1_readvariableop_resource:
??C
4lstm_13_lstm_cell_13_biasadd_readvariableop_resource:	?<
)dense_6_tensordot_readvariableop_resource:	?5
'dense_6_biasadd_readvariableop_resource:
identity??dense_6/BiasAdd/ReadVariableOp? dense_6/Tensordot/ReadVariableOp?+lstm_12/lstm_cell_12/BiasAdd/ReadVariableOp?*lstm_12/lstm_cell_12/MatMul/ReadVariableOp?,lstm_12/lstm_cell_12/MatMul_1/ReadVariableOp?lstm_12/while?+lstm_13/lstm_cell_13/BiasAdd/ReadVariableOp?*lstm_13/lstm_cell_13/MatMul/ReadVariableOp?,lstm_13/lstm_cell_13/MatMul_1/ReadVariableOp?lstm_13/whileT
lstm_12/ShapeShapeinputs*
T0*
_output_shapes
:2
lstm_12/Shape?
lstm_12/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_12/strided_slice/stack?
lstm_12/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_12/strided_slice/stack_1?
lstm_12/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_12/strided_slice/stack_2?
lstm_12/strided_sliceStridedSlicelstm_12/Shape:output:0$lstm_12/strided_slice/stack:output:0&lstm_12/strided_slice/stack_1:output:0&lstm_12/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_12/strided_slicem
lstm_12/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
lstm_12/zeros/mul/y?
lstm_12/zeros/mulMullstm_12/strided_slice:output:0lstm_12/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_12/zeros/mulo
lstm_12/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
lstm_12/zeros/Less/y?
lstm_12/zeros/LessLesslstm_12/zeros/mul:z:0lstm_12/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_12/zeros/Lesss
lstm_12/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
lstm_12/zeros/packed/1?
lstm_12/zeros/packedPacklstm_12/strided_slice:output:0lstm_12/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_12/zeros/packedo
lstm_12/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_12/zeros/Const?
lstm_12/zerosFilllstm_12/zeros/packed:output:0lstm_12/zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
lstm_12/zerosq
lstm_12/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
lstm_12/zeros_1/mul/y?
lstm_12/zeros_1/mulMullstm_12/strided_slice:output:0lstm_12/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_12/zeros_1/muls
lstm_12/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
lstm_12/zeros_1/Less/y?
lstm_12/zeros_1/LessLesslstm_12/zeros_1/mul:z:0lstm_12/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_12/zeros_1/Lessw
lstm_12/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
lstm_12/zeros_1/packed/1?
lstm_12/zeros_1/packedPacklstm_12/strided_slice:output:0!lstm_12/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_12/zeros_1/packeds
lstm_12/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_12/zeros_1/Const?
lstm_12/zeros_1Filllstm_12/zeros_1/packed:output:0lstm_12/zeros_1/Const:output:0*
T0*(
_output_shapes
:??????????2
lstm_12/zeros_1?
lstm_12/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_12/transpose/perm?
lstm_12/transpose	Transposeinputslstm_12/transpose/perm:output:0*
T0*+
_output_shapes
:?????????]2
lstm_12/transposeg
lstm_12/Shape_1Shapelstm_12/transpose:y:0*
T0*
_output_shapes
:2
lstm_12/Shape_1?
lstm_12/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_12/strided_slice_1/stack?
lstm_12/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_12/strided_slice_1/stack_1?
lstm_12/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_12/strided_slice_1/stack_2?
lstm_12/strided_slice_1StridedSlicelstm_12/Shape_1:output:0&lstm_12/strided_slice_1/stack:output:0(lstm_12/strided_slice_1/stack_1:output:0(lstm_12/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_12/strided_slice_1?
#lstm_12/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2%
#lstm_12/TensorArrayV2/element_shape?
lstm_12/TensorArrayV2TensorListReserve,lstm_12/TensorArrayV2/element_shape:output:0 lstm_12/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_12/TensorArrayV2?
=lstm_12/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????]   2?
=lstm_12/TensorArrayUnstack/TensorListFromTensor/element_shape?
/lstm_12/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_12/transpose:y:0Flstm_12/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type021
/lstm_12/TensorArrayUnstack/TensorListFromTensor?
lstm_12/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_12/strided_slice_2/stack?
lstm_12/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_12/strided_slice_2/stack_1?
lstm_12/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_12/strided_slice_2/stack_2?
lstm_12/strided_slice_2StridedSlicelstm_12/transpose:y:0&lstm_12/strided_slice_2/stack:output:0(lstm_12/strided_slice_2/stack_1:output:0(lstm_12/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????]*
shrink_axis_mask2
lstm_12/strided_slice_2?
*lstm_12/lstm_cell_12/MatMul/ReadVariableOpReadVariableOp3lstm_12_lstm_cell_12_matmul_readvariableop_resource*
_output_shapes
:	]?*
dtype02,
*lstm_12/lstm_cell_12/MatMul/ReadVariableOp?
lstm_12/lstm_cell_12/MatMulMatMul lstm_12/strided_slice_2:output:02lstm_12/lstm_cell_12/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_12/lstm_cell_12/MatMul?
,lstm_12/lstm_cell_12/MatMul_1/ReadVariableOpReadVariableOp5lstm_12_lstm_cell_12_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02.
,lstm_12/lstm_cell_12/MatMul_1/ReadVariableOp?
lstm_12/lstm_cell_12/MatMul_1MatMullstm_12/zeros:output:04lstm_12/lstm_cell_12/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_12/lstm_cell_12/MatMul_1?
lstm_12/lstm_cell_12/addAddV2%lstm_12/lstm_cell_12/MatMul:product:0'lstm_12/lstm_cell_12/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
lstm_12/lstm_cell_12/add?
+lstm_12/lstm_cell_12/BiasAdd/ReadVariableOpReadVariableOp4lstm_12_lstm_cell_12_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02-
+lstm_12/lstm_cell_12/BiasAdd/ReadVariableOp?
lstm_12/lstm_cell_12/BiasAddBiasAddlstm_12/lstm_cell_12/add:z:03lstm_12/lstm_cell_12/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_12/lstm_cell_12/BiasAdd?
$lstm_12/lstm_cell_12/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2&
$lstm_12/lstm_cell_12/split/split_dim?
lstm_12/lstm_cell_12/splitSplit-lstm_12/lstm_cell_12/split/split_dim:output:0%lstm_12/lstm_cell_12/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split2
lstm_12/lstm_cell_12/split?
lstm_12/lstm_cell_12/SigmoidSigmoid#lstm_12/lstm_cell_12/split:output:0*
T0*(
_output_shapes
:??????????2
lstm_12/lstm_cell_12/Sigmoid?
lstm_12/lstm_cell_12/Sigmoid_1Sigmoid#lstm_12/lstm_cell_12/split:output:1*
T0*(
_output_shapes
:??????????2 
lstm_12/lstm_cell_12/Sigmoid_1?
lstm_12/lstm_cell_12/mulMul"lstm_12/lstm_cell_12/Sigmoid_1:y:0lstm_12/zeros_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_12/lstm_cell_12/mul?
lstm_12/lstm_cell_12/ReluRelu#lstm_12/lstm_cell_12/split:output:2*
T0*(
_output_shapes
:??????????2
lstm_12/lstm_cell_12/Relu?
lstm_12/lstm_cell_12/mul_1Mul lstm_12/lstm_cell_12/Sigmoid:y:0'lstm_12/lstm_cell_12/Relu:activations:0*
T0*(
_output_shapes
:??????????2
lstm_12/lstm_cell_12/mul_1?
lstm_12/lstm_cell_12/add_1AddV2lstm_12/lstm_cell_12/mul:z:0lstm_12/lstm_cell_12/mul_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_12/lstm_cell_12/add_1?
lstm_12/lstm_cell_12/Sigmoid_2Sigmoid#lstm_12/lstm_cell_12/split:output:3*
T0*(
_output_shapes
:??????????2 
lstm_12/lstm_cell_12/Sigmoid_2?
lstm_12/lstm_cell_12/Relu_1Relulstm_12/lstm_cell_12/add_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_12/lstm_cell_12/Relu_1?
lstm_12/lstm_cell_12/mul_2Mul"lstm_12/lstm_cell_12/Sigmoid_2:y:0)lstm_12/lstm_cell_12/Relu_1:activations:0*
T0*(
_output_shapes
:??????????2
lstm_12/lstm_cell_12/mul_2?
%lstm_12/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2'
%lstm_12/TensorArrayV2_1/element_shape?
lstm_12/TensorArrayV2_1TensorListReserve.lstm_12/TensorArrayV2_1/element_shape:output:0 lstm_12/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_12/TensorArrayV2_1^
lstm_12/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_12/time?
 lstm_12/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2"
 lstm_12/while/maximum_iterationsz
lstm_12/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_12/while/loop_counter?
lstm_12/whileWhile#lstm_12/while/loop_counter:output:0)lstm_12/while/maximum_iterations:output:0lstm_12/time:output:0 lstm_12/TensorArrayV2_1:handle:0lstm_12/zeros:output:0lstm_12/zeros_1:output:0 lstm_12/strided_slice_1:output:0?lstm_12/TensorArrayUnstack/TensorListFromTensor:output_handle:03lstm_12_lstm_cell_12_matmul_readvariableop_resource5lstm_12_lstm_cell_12_matmul_1_readvariableop_resource4lstm_12_lstm_cell_12_biasadd_readvariableop_resource*
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
lstm_12_while_body_26067178*'
condR
lstm_12_while_cond_26067177*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations 2
lstm_12/while?
8lstm_12/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2:
8lstm_12/TensorArrayV2Stack/TensorListStack/element_shape?
*lstm_12/TensorArrayV2Stack/TensorListStackTensorListStacklstm_12/while:output:3Alstm_12/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:??????????*
element_dtype02,
*lstm_12/TensorArrayV2Stack/TensorListStack?
lstm_12/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
lstm_12/strided_slice_3/stack?
lstm_12/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2!
lstm_12/strided_slice_3/stack_1?
lstm_12/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_12/strided_slice_3/stack_2?
lstm_12/strided_slice_3StridedSlice3lstm_12/TensorArrayV2Stack/TensorListStack:tensor:0&lstm_12/strided_slice_3/stack:output:0(lstm_12/strided_slice_3/stack_1:output:0(lstm_12/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
lstm_12/strided_slice_3?
lstm_12/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_12/transpose_1/perm?
lstm_12/transpose_1	Transpose3lstm_12/TensorArrayV2Stack/TensorListStack:tensor:0!lstm_12/transpose_1/perm:output:0*
T0*,
_output_shapes
:??????????2
lstm_12/transpose_1v
lstm_12/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_12/runtimey
dropout_12/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2
dropout_12/dropout/Const?
dropout_12/dropout/MulMullstm_12/transpose_1:y:0!dropout_12/dropout/Const:output:0*
T0*,
_output_shapes
:??????????2
dropout_12/dropout/Mul{
dropout_12/dropout/ShapeShapelstm_12/transpose_1:y:0*
T0*
_output_shapes
:2
dropout_12/dropout/Shape?
/dropout_12/dropout/random_uniform/RandomUniformRandomUniform!dropout_12/dropout/Shape:output:0*
T0*,
_output_shapes
:??????????*
dtype021
/dropout_12/dropout/random_uniform/RandomUniform?
!dropout_12/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?>2#
!dropout_12/dropout/GreaterEqual/y?
dropout_12/dropout/GreaterEqualGreaterEqual8dropout_12/dropout/random_uniform/RandomUniform:output:0*dropout_12/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:??????????2!
dropout_12/dropout/GreaterEqual?
dropout_12/dropout/CastCast#dropout_12/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:??????????2
dropout_12/dropout/Cast?
dropout_12/dropout/Mul_1Muldropout_12/dropout/Mul:z:0dropout_12/dropout/Cast:y:0*
T0*,
_output_shapes
:??????????2
dropout_12/dropout/Mul_1j
lstm_13/ShapeShapedropout_12/dropout/Mul_1:z:0*
T0*
_output_shapes
:2
lstm_13/Shape?
lstm_13/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_13/strided_slice/stack?
lstm_13/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_13/strided_slice/stack_1?
lstm_13/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_13/strided_slice/stack_2?
lstm_13/strided_sliceStridedSlicelstm_13/Shape:output:0$lstm_13/strided_slice/stack:output:0&lstm_13/strided_slice/stack_1:output:0&lstm_13/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_13/strided_slicem
lstm_13/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
lstm_13/zeros/mul/y?
lstm_13/zeros/mulMullstm_13/strided_slice:output:0lstm_13/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_13/zeros/mulo
lstm_13/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
lstm_13/zeros/Less/y?
lstm_13/zeros/LessLesslstm_13/zeros/mul:z:0lstm_13/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_13/zeros/Lesss
lstm_13/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
lstm_13/zeros/packed/1?
lstm_13/zeros/packedPacklstm_13/strided_slice:output:0lstm_13/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_13/zeros/packedo
lstm_13/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_13/zeros/Const?
lstm_13/zerosFilllstm_13/zeros/packed:output:0lstm_13/zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
lstm_13/zerosq
lstm_13/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
lstm_13/zeros_1/mul/y?
lstm_13/zeros_1/mulMullstm_13/strided_slice:output:0lstm_13/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_13/zeros_1/muls
lstm_13/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
lstm_13/zeros_1/Less/y?
lstm_13/zeros_1/LessLesslstm_13/zeros_1/mul:z:0lstm_13/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_13/zeros_1/Lessw
lstm_13/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
lstm_13/zeros_1/packed/1?
lstm_13/zeros_1/packedPacklstm_13/strided_slice:output:0!lstm_13/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_13/zeros_1/packeds
lstm_13/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_13/zeros_1/Const?
lstm_13/zeros_1Filllstm_13/zeros_1/packed:output:0lstm_13/zeros_1/Const:output:0*
T0*(
_output_shapes
:??????????2
lstm_13/zeros_1?
lstm_13/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_13/transpose/perm?
lstm_13/transpose	Transposedropout_12/dropout/Mul_1:z:0lstm_13/transpose/perm:output:0*
T0*,
_output_shapes
:??????????2
lstm_13/transposeg
lstm_13/Shape_1Shapelstm_13/transpose:y:0*
T0*
_output_shapes
:2
lstm_13/Shape_1?
lstm_13/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_13/strided_slice_1/stack?
lstm_13/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_13/strided_slice_1/stack_1?
lstm_13/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_13/strided_slice_1/stack_2?
lstm_13/strided_slice_1StridedSlicelstm_13/Shape_1:output:0&lstm_13/strided_slice_1/stack:output:0(lstm_13/strided_slice_1/stack_1:output:0(lstm_13/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_13/strided_slice_1?
#lstm_13/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2%
#lstm_13/TensorArrayV2/element_shape?
lstm_13/TensorArrayV2TensorListReserve,lstm_13/TensorArrayV2/element_shape:output:0 lstm_13/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_13/TensorArrayV2?
=lstm_13/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2?
=lstm_13/TensorArrayUnstack/TensorListFromTensor/element_shape?
/lstm_13/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_13/transpose:y:0Flstm_13/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type021
/lstm_13/TensorArrayUnstack/TensorListFromTensor?
lstm_13/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_13/strided_slice_2/stack?
lstm_13/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_13/strided_slice_2/stack_1?
lstm_13/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_13/strided_slice_2/stack_2?
lstm_13/strided_slice_2StridedSlicelstm_13/transpose:y:0&lstm_13/strided_slice_2/stack:output:0(lstm_13/strided_slice_2/stack_1:output:0(lstm_13/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
lstm_13/strided_slice_2?
*lstm_13/lstm_cell_13/MatMul/ReadVariableOpReadVariableOp3lstm_13_lstm_cell_13_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02,
*lstm_13/lstm_cell_13/MatMul/ReadVariableOp?
lstm_13/lstm_cell_13/MatMulMatMul lstm_13/strided_slice_2:output:02lstm_13/lstm_cell_13/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_13/lstm_cell_13/MatMul?
,lstm_13/lstm_cell_13/MatMul_1/ReadVariableOpReadVariableOp5lstm_13_lstm_cell_13_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02.
,lstm_13/lstm_cell_13/MatMul_1/ReadVariableOp?
lstm_13/lstm_cell_13/MatMul_1MatMullstm_13/zeros:output:04lstm_13/lstm_cell_13/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_13/lstm_cell_13/MatMul_1?
lstm_13/lstm_cell_13/addAddV2%lstm_13/lstm_cell_13/MatMul:product:0'lstm_13/lstm_cell_13/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
lstm_13/lstm_cell_13/add?
+lstm_13/lstm_cell_13/BiasAdd/ReadVariableOpReadVariableOp4lstm_13_lstm_cell_13_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02-
+lstm_13/lstm_cell_13/BiasAdd/ReadVariableOp?
lstm_13/lstm_cell_13/BiasAddBiasAddlstm_13/lstm_cell_13/add:z:03lstm_13/lstm_cell_13/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_13/lstm_cell_13/BiasAdd?
$lstm_13/lstm_cell_13/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2&
$lstm_13/lstm_cell_13/split/split_dim?
lstm_13/lstm_cell_13/splitSplit-lstm_13/lstm_cell_13/split/split_dim:output:0%lstm_13/lstm_cell_13/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split2
lstm_13/lstm_cell_13/split?
lstm_13/lstm_cell_13/SigmoidSigmoid#lstm_13/lstm_cell_13/split:output:0*
T0*(
_output_shapes
:??????????2
lstm_13/lstm_cell_13/Sigmoid?
lstm_13/lstm_cell_13/Sigmoid_1Sigmoid#lstm_13/lstm_cell_13/split:output:1*
T0*(
_output_shapes
:??????????2 
lstm_13/lstm_cell_13/Sigmoid_1?
lstm_13/lstm_cell_13/mulMul"lstm_13/lstm_cell_13/Sigmoid_1:y:0lstm_13/zeros_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_13/lstm_cell_13/mul?
lstm_13/lstm_cell_13/ReluRelu#lstm_13/lstm_cell_13/split:output:2*
T0*(
_output_shapes
:??????????2
lstm_13/lstm_cell_13/Relu?
lstm_13/lstm_cell_13/mul_1Mul lstm_13/lstm_cell_13/Sigmoid:y:0'lstm_13/lstm_cell_13/Relu:activations:0*
T0*(
_output_shapes
:??????????2
lstm_13/lstm_cell_13/mul_1?
lstm_13/lstm_cell_13/add_1AddV2lstm_13/lstm_cell_13/mul:z:0lstm_13/lstm_cell_13/mul_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_13/lstm_cell_13/add_1?
lstm_13/lstm_cell_13/Sigmoid_2Sigmoid#lstm_13/lstm_cell_13/split:output:3*
T0*(
_output_shapes
:??????????2 
lstm_13/lstm_cell_13/Sigmoid_2?
lstm_13/lstm_cell_13/Relu_1Relulstm_13/lstm_cell_13/add_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_13/lstm_cell_13/Relu_1?
lstm_13/lstm_cell_13/mul_2Mul"lstm_13/lstm_cell_13/Sigmoid_2:y:0)lstm_13/lstm_cell_13/Relu_1:activations:0*
T0*(
_output_shapes
:??????????2
lstm_13/lstm_cell_13/mul_2?
%lstm_13/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2'
%lstm_13/TensorArrayV2_1/element_shape?
lstm_13/TensorArrayV2_1TensorListReserve.lstm_13/TensorArrayV2_1/element_shape:output:0 lstm_13/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_13/TensorArrayV2_1^
lstm_13/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_13/time?
 lstm_13/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2"
 lstm_13/while/maximum_iterationsz
lstm_13/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_13/while/loop_counter?
lstm_13/whileWhile#lstm_13/while/loop_counter:output:0)lstm_13/while/maximum_iterations:output:0lstm_13/time:output:0 lstm_13/TensorArrayV2_1:handle:0lstm_13/zeros:output:0lstm_13/zeros_1:output:0 lstm_13/strided_slice_1:output:0?lstm_13/TensorArrayUnstack/TensorListFromTensor:output_handle:03lstm_13_lstm_cell_13_matmul_readvariableop_resource5lstm_13_lstm_cell_13_matmul_1_readvariableop_resource4lstm_13_lstm_cell_13_biasadd_readvariableop_resource*
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
lstm_13_while_body_26067333*'
condR
lstm_13_while_cond_26067332*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations 2
lstm_13/while?
8lstm_13/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2:
8lstm_13/TensorArrayV2Stack/TensorListStack/element_shape?
*lstm_13/TensorArrayV2Stack/TensorListStackTensorListStacklstm_13/while:output:3Alstm_13/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:??????????*
element_dtype02,
*lstm_13/TensorArrayV2Stack/TensorListStack?
lstm_13/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
lstm_13/strided_slice_3/stack?
lstm_13/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2!
lstm_13/strided_slice_3/stack_1?
lstm_13/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_13/strided_slice_3/stack_2?
lstm_13/strided_slice_3StridedSlice3lstm_13/TensorArrayV2Stack/TensorListStack:tensor:0&lstm_13/strided_slice_3/stack:output:0(lstm_13/strided_slice_3/stack_1:output:0(lstm_13/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
lstm_13/strided_slice_3?
lstm_13/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_13/transpose_1/perm?
lstm_13/transpose_1	Transpose3lstm_13/TensorArrayV2Stack/TensorListStack:tensor:0!lstm_13/transpose_1/perm:output:0*
T0*,
_output_shapes
:??????????2
lstm_13/transpose_1v
lstm_13/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_13/runtimey
dropout_13/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout_13/dropout/Const?
dropout_13/dropout/MulMullstm_13/transpose_1:y:0!dropout_13/dropout/Const:output:0*
T0*,
_output_shapes
:??????????2
dropout_13/dropout/Mul{
dropout_13/dropout/ShapeShapelstm_13/transpose_1:y:0*
T0*
_output_shapes
:2
dropout_13/dropout/Shape?
/dropout_13/dropout/random_uniform/RandomUniformRandomUniform!dropout_13/dropout/Shape:output:0*
T0*,
_output_shapes
:??????????*
dtype021
/dropout_13/dropout/random_uniform/RandomUniform?
!dropout_13/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2#
!dropout_13/dropout/GreaterEqual/y?
dropout_13/dropout/GreaterEqualGreaterEqual8dropout_13/dropout/random_uniform/RandomUniform:output:0*dropout_13/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:??????????2!
dropout_13/dropout/GreaterEqual?
dropout_13/dropout/CastCast#dropout_13/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:??????????2
dropout_13/dropout/Cast?
dropout_13/dropout/Mul_1Muldropout_13/dropout/Mul:z:0dropout_13/dropout/Cast:y:0*
T0*,
_output_shapes
:??????????2
dropout_13/dropout/Mul_1?
 dense_6/Tensordot/ReadVariableOpReadVariableOp)dense_6_tensordot_readvariableop_resource*
_output_shapes
:	?*
dtype02"
 dense_6/Tensordot/ReadVariableOpz
dense_6/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_6/Tensordot/axes?
dense_6/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_6/Tensordot/free~
dense_6/Tensordot/ShapeShapedropout_13/dropout/Mul_1:z:0*
T0*
_output_shapes
:2
dense_6/Tensordot/Shape?
dense_6/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_6/Tensordot/GatherV2/axis?
dense_6/Tensordot/GatherV2GatherV2 dense_6/Tensordot/Shape:output:0dense_6/Tensordot/free:output:0(dense_6/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_6/Tensordot/GatherV2?
!dense_6/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_6/Tensordot/GatherV2_1/axis?
dense_6/Tensordot/GatherV2_1GatherV2 dense_6/Tensordot/Shape:output:0dense_6/Tensordot/axes:output:0*dense_6/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_6/Tensordot/GatherV2_1|
dense_6/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_6/Tensordot/Const?
dense_6/Tensordot/ProdProd#dense_6/Tensordot/GatherV2:output:0 dense_6/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_6/Tensordot/Prod?
dense_6/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_6/Tensordot/Const_1?
dense_6/Tensordot/Prod_1Prod%dense_6/Tensordot/GatherV2_1:output:0"dense_6/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_6/Tensordot/Prod_1?
dense_6/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense_6/Tensordot/concat/axis?
dense_6/Tensordot/concatConcatV2dense_6/Tensordot/free:output:0dense_6/Tensordot/axes:output:0&dense_6/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_6/Tensordot/concat?
dense_6/Tensordot/stackPackdense_6/Tensordot/Prod:output:0!dense_6/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_6/Tensordot/stack?
dense_6/Tensordot/transpose	Transposedropout_13/dropout/Mul_1:z:0!dense_6/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????2
dense_6/Tensordot/transpose?
dense_6/Tensordot/ReshapeReshapedense_6/Tensordot/transpose:y:0 dense_6/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense_6/Tensordot/Reshape?
dense_6/Tensordot/MatMulMatMul"dense_6/Tensordot/Reshape:output:0(dense_6/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_6/Tensordot/MatMul?
dense_6/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
dense_6/Tensordot/Const_2?
dense_6/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_6/Tensordot/concat_1/axis?
dense_6/Tensordot/concat_1ConcatV2#dense_6/Tensordot/GatherV2:output:0"dense_6/Tensordot/Const_2:output:0(dense_6/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_6/Tensordot/concat_1?
dense_6/TensordotReshape"dense_6/Tensordot/MatMul:product:0#dense_6/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????2
dense_6/Tensordot?
dense_6/BiasAdd/ReadVariableOpReadVariableOp'dense_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_6/BiasAdd/ReadVariableOp?
dense_6/BiasAddBiasAdddense_6/Tensordot:output:0&dense_6/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????2
dense_6/BiasAdd}
dense_6/SoftmaxSoftmaxdense_6/BiasAdd:output:0*
T0*+
_output_shapes
:?????????2
dense_6/Softmaxx
IdentityIdentitydense_6/Softmax:softmax:0^NoOp*
T0*+
_output_shapes
:?????????2

Identity?
NoOpNoOp^dense_6/BiasAdd/ReadVariableOp!^dense_6/Tensordot/ReadVariableOp,^lstm_12/lstm_cell_12/BiasAdd/ReadVariableOp+^lstm_12/lstm_cell_12/MatMul/ReadVariableOp-^lstm_12/lstm_cell_12/MatMul_1/ReadVariableOp^lstm_12/while,^lstm_13/lstm_cell_13/BiasAdd/ReadVariableOp+^lstm_13/lstm_cell_13/MatMul/ReadVariableOp-^lstm_13/lstm_cell_13/MatMul_1/ReadVariableOp^lstm_13/while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????]: : : : : : : : 2@
dense_6/BiasAdd/ReadVariableOpdense_6/BiasAdd/ReadVariableOp2D
 dense_6/Tensordot/ReadVariableOp dense_6/Tensordot/ReadVariableOp2Z
+lstm_12/lstm_cell_12/BiasAdd/ReadVariableOp+lstm_12/lstm_cell_12/BiasAdd/ReadVariableOp2X
*lstm_12/lstm_cell_12/MatMul/ReadVariableOp*lstm_12/lstm_cell_12/MatMul/ReadVariableOp2\
,lstm_12/lstm_cell_12/MatMul_1/ReadVariableOp,lstm_12/lstm_cell_12/MatMul_1/ReadVariableOp2
lstm_12/whilelstm_12/while2Z
+lstm_13/lstm_cell_13/BiasAdd/ReadVariableOp+lstm_13/lstm_cell_13/BiasAdd/ReadVariableOp2X
*lstm_13/lstm_cell_13/MatMul/ReadVariableOp*lstm_13/lstm_cell_13/MatMul/ReadVariableOp2\
,lstm_13/lstm_cell_13/MatMul_1/ReadVariableOp,lstm_13/lstm_cell_13/MatMul_1/ReadVariableOp2
lstm_13/whilelstm_13/while:S O
+
_output_shapes
:?????????]
 
_user_specified_nameinputs
?]
?
(sequential_6_lstm_13_while_body_26064449F
Bsequential_6_lstm_13_while_sequential_6_lstm_13_while_loop_counterL
Hsequential_6_lstm_13_while_sequential_6_lstm_13_while_maximum_iterations*
&sequential_6_lstm_13_while_placeholder,
(sequential_6_lstm_13_while_placeholder_1,
(sequential_6_lstm_13_while_placeholder_2,
(sequential_6_lstm_13_while_placeholder_3E
Asequential_6_lstm_13_while_sequential_6_lstm_13_strided_slice_1_0?
}sequential_6_lstm_13_while_tensorarrayv2read_tensorlistgetitem_sequential_6_lstm_13_tensorarrayunstack_tensorlistfromtensor_0\
Hsequential_6_lstm_13_while_lstm_cell_13_matmul_readvariableop_resource_0:
??^
Jsequential_6_lstm_13_while_lstm_cell_13_matmul_1_readvariableop_resource_0:
??X
Isequential_6_lstm_13_while_lstm_cell_13_biasadd_readvariableop_resource_0:	?'
#sequential_6_lstm_13_while_identity)
%sequential_6_lstm_13_while_identity_1)
%sequential_6_lstm_13_while_identity_2)
%sequential_6_lstm_13_while_identity_3)
%sequential_6_lstm_13_while_identity_4)
%sequential_6_lstm_13_while_identity_5C
?sequential_6_lstm_13_while_sequential_6_lstm_13_strided_slice_1
{sequential_6_lstm_13_while_tensorarrayv2read_tensorlistgetitem_sequential_6_lstm_13_tensorarrayunstack_tensorlistfromtensorZ
Fsequential_6_lstm_13_while_lstm_cell_13_matmul_readvariableop_resource:
??\
Hsequential_6_lstm_13_while_lstm_cell_13_matmul_1_readvariableop_resource:
??V
Gsequential_6_lstm_13_while_lstm_cell_13_biasadd_readvariableop_resource:	???>sequential_6/lstm_13/while/lstm_cell_13/BiasAdd/ReadVariableOp?=sequential_6/lstm_13/while/lstm_cell_13/MatMul/ReadVariableOp??sequential_6/lstm_13/while/lstm_cell_13/MatMul_1/ReadVariableOp?
Lsequential_6/lstm_13/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2N
Lsequential_6/lstm_13/while/TensorArrayV2Read/TensorListGetItem/element_shape?
>sequential_6/lstm_13/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem}sequential_6_lstm_13_while_tensorarrayv2read_tensorlistgetitem_sequential_6_lstm_13_tensorarrayunstack_tensorlistfromtensor_0&sequential_6_lstm_13_while_placeholderUsequential_6/lstm_13/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:??????????*
element_dtype02@
>sequential_6/lstm_13/while/TensorArrayV2Read/TensorListGetItem?
=sequential_6/lstm_13/while/lstm_cell_13/MatMul/ReadVariableOpReadVariableOpHsequential_6_lstm_13_while_lstm_cell_13_matmul_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02?
=sequential_6/lstm_13/while/lstm_cell_13/MatMul/ReadVariableOp?
.sequential_6/lstm_13/while/lstm_cell_13/MatMulMatMulEsequential_6/lstm_13/while/TensorArrayV2Read/TensorListGetItem:item:0Esequential_6/lstm_13/while/lstm_cell_13/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????20
.sequential_6/lstm_13/while/lstm_cell_13/MatMul?
?sequential_6/lstm_13/while/lstm_cell_13/MatMul_1/ReadVariableOpReadVariableOpJsequential_6_lstm_13_while_lstm_cell_13_matmul_1_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02A
?sequential_6/lstm_13/while/lstm_cell_13/MatMul_1/ReadVariableOp?
0sequential_6/lstm_13/while/lstm_cell_13/MatMul_1MatMul(sequential_6_lstm_13_while_placeholder_2Gsequential_6/lstm_13/while/lstm_cell_13/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????22
0sequential_6/lstm_13/while/lstm_cell_13/MatMul_1?
+sequential_6/lstm_13/while/lstm_cell_13/addAddV28sequential_6/lstm_13/while/lstm_cell_13/MatMul:product:0:sequential_6/lstm_13/while/lstm_cell_13/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2-
+sequential_6/lstm_13/while/lstm_cell_13/add?
>sequential_6/lstm_13/while/lstm_cell_13/BiasAdd/ReadVariableOpReadVariableOpIsequential_6_lstm_13_while_lstm_cell_13_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype02@
>sequential_6/lstm_13/while/lstm_cell_13/BiasAdd/ReadVariableOp?
/sequential_6/lstm_13/while/lstm_cell_13/BiasAddBiasAdd/sequential_6/lstm_13/while/lstm_cell_13/add:z:0Fsequential_6/lstm_13/while/lstm_cell_13/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????21
/sequential_6/lstm_13/while/lstm_cell_13/BiasAdd?
7sequential_6/lstm_13/while/lstm_cell_13/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :29
7sequential_6/lstm_13/while/lstm_cell_13/split/split_dim?
-sequential_6/lstm_13/while/lstm_cell_13/splitSplit@sequential_6/lstm_13/while/lstm_cell_13/split/split_dim:output:08sequential_6/lstm_13/while/lstm_cell_13/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split2/
-sequential_6/lstm_13/while/lstm_cell_13/split?
/sequential_6/lstm_13/while/lstm_cell_13/SigmoidSigmoid6sequential_6/lstm_13/while/lstm_cell_13/split:output:0*
T0*(
_output_shapes
:??????????21
/sequential_6/lstm_13/while/lstm_cell_13/Sigmoid?
1sequential_6/lstm_13/while/lstm_cell_13/Sigmoid_1Sigmoid6sequential_6/lstm_13/while/lstm_cell_13/split:output:1*
T0*(
_output_shapes
:??????????23
1sequential_6/lstm_13/while/lstm_cell_13/Sigmoid_1?
+sequential_6/lstm_13/while/lstm_cell_13/mulMul5sequential_6/lstm_13/while/lstm_cell_13/Sigmoid_1:y:0(sequential_6_lstm_13_while_placeholder_3*
T0*(
_output_shapes
:??????????2-
+sequential_6/lstm_13/while/lstm_cell_13/mul?
,sequential_6/lstm_13/while/lstm_cell_13/ReluRelu6sequential_6/lstm_13/while/lstm_cell_13/split:output:2*
T0*(
_output_shapes
:??????????2.
,sequential_6/lstm_13/while/lstm_cell_13/Relu?
-sequential_6/lstm_13/while/lstm_cell_13/mul_1Mul3sequential_6/lstm_13/while/lstm_cell_13/Sigmoid:y:0:sequential_6/lstm_13/while/lstm_cell_13/Relu:activations:0*
T0*(
_output_shapes
:??????????2/
-sequential_6/lstm_13/while/lstm_cell_13/mul_1?
-sequential_6/lstm_13/while/lstm_cell_13/add_1AddV2/sequential_6/lstm_13/while/lstm_cell_13/mul:z:01sequential_6/lstm_13/while/lstm_cell_13/mul_1:z:0*
T0*(
_output_shapes
:??????????2/
-sequential_6/lstm_13/while/lstm_cell_13/add_1?
1sequential_6/lstm_13/while/lstm_cell_13/Sigmoid_2Sigmoid6sequential_6/lstm_13/while/lstm_cell_13/split:output:3*
T0*(
_output_shapes
:??????????23
1sequential_6/lstm_13/while/lstm_cell_13/Sigmoid_2?
.sequential_6/lstm_13/while/lstm_cell_13/Relu_1Relu1sequential_6/lstm_13/while/lstm_cell_13/add_1:z:0*
T0*(
_output_shapes
:??????????20
.sequential_6/lstm_13/while/lstm_cell_13/Relu_1?
-sequential_6/lstm_13/while/lstm_cell_13/mul_2Mul5sequential_6/lstm_13/while/lstm_cell_13/Sigmoid_2:y:0<sequential_6/lstm_13/while/lstm_cell_13/Relu_1:activations:0*
T0*(
_output_shapes
:??????????2/
-sequential_6/lstm_13/while/lstm_cell_13/mul_2?
?sequential_6/lstm_13/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem(sequential_6_lstm_13_while_placeholder_1&sequential_6_lstm_13_while_placeholder1sequential_6/lstm_13/while/lstm_cell_13/mul_2:z:0*
_output_shapes
: *
element_dtype02A
?sequential_6/lstm_13/while/TensorArrayV2Write/TensorListSetItem?
 sequential_6/lstm_13/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2"
 sequential_6/lstm_13/while/add/y?
sequential_6/lstm_13/while/addAddV2&sequential_6_lstm_13_while_placeholder)sequential_6/lstm_13/while/add/y:output:0*
T0*
_output_shapes
: 2 
sequential_6/lstm_13/while/add?
"sequential_6/lstm_13/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2$
"sequential_6/lstm_13/while/add_1/y?
 sequential_6/lstm_13/while/add_1AddV2Bsequential_6_lstm_13_while_sequential_6_lstm_13_while_loop_counter+sequential_6/lstm_13/while/add_1/y:output:0*
T0*
_output_shapes
: 2"
 sequential_6/lstm_13/while/add_1?
#sequential_6/lstm_13/while/IdentityIdentity$sequential_6/lstm_13/while/add_1:z:0 ^sequential_6/lstm_13/while/NoOp*
T0*
_output_shapes
: 2%
#sequential_6/lstm_13/while/Identity?
%sequential_6/lstm_13/while/Identity_1IdentityHsequential_6_lstm_13_while_sequential_6_lstm_13_while_maximum_iterations ^sequential_6/lstm_13/while/NoOp*
T0*
_output_shapes
: 2'
%sequential_6/lstm_13/while/Identity_1?
%sequential_6/lstm_13/while/Identity_2Identity"sequential_6/lstm_13/while/add:z:0 ^sequential_6/lstm_13/while/NoOp*
T0*
_output_shapes
: 2'
%sequential_6/lstm_13/while/Identity_2?
%sequential_6/lstm_13/while/Identity_3IdentityOsequential_6/lstm_13/while/TensorArrayV2Write/TensorListSetItem:output_handle:0 ^sequential_6/lstm_13/while/NoOp*
T0*
_output_shapes
: 2'
%sequential_6/lstm_13/while/Identity_3?
%sequential_6/lstm_13/while/Identity_4Identity1sequential_6/lstm_13/while/lstm_cell_13/mul_2:z:0 ^sequential_6/lstm_13/while/NoOp*
T0*(
_output_shapes
:??????????2'
%sequential_6/lstm_13/while/Identity_4?
%sequential_6/lstm_13/while/Identity_5Identity1sequential_6/lstm_13/while/lstm_cell_13/add_1:z:0 ^sequential_6/lstm_13/while/NoOp*
T0*(
_output_shapes
:??????????2'
%sequential_6/lstm_13/while/Identity_5?
sequential_6/lstm_13/while/NoOpNoOp?^sequential_6/lstm_13/while/lstm_cell_13/BiasAdd/ReadVariableOp>^sequential_6/lstm_13/while/lstm_cell_13/MatMul/ReadVariableOp@^sequential_6/lstm_13/while/lstm_cell_13/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2!
sequential_6/lstm_13/while/NoOp"S
#sequential_6_lstm_13_while_identity,sequential_6/lstm_13/while/Identity:output:0"W
%sequential_6_lstm_13_while_identity_1.sequential_6/lstm_13/while/Identity_1:output:0"W
%sequential_6_lstm_13_while_identity_2.sequential_6/lstm_13/while/Identity_2:output:0"W
%sequential_6_lstm_13_while_identity_3.sequential_6/lstm_13/while/Identity_3:output:0"W
%sequential_6_lstm_13_while_identity_4.sequential_6/lstm_13/while/Identity_4:output:0"W
%sequential_6_lstm_13_while_identity_5.sequential_6/lstm_13/while/Identity_5:output:0"?
Gsequential_6_lstm_13_while_lstm_cell_13_biasadd_readvariableop_resourceIsequential_6_lstm_13_while_lstm_cell_13_biasadd_readvariableop_resource_0"?
Hsequential_6_lstm_13_while_lstm_cell_13_matmul_1_readvariableop_resourceJsequential_6_lstm_13_while_lstm_cell_13_matmul_1_readvariableop_resource_0"?
Fsequential_6_lstm_13_while_lstm_cell_13_matmul_readvariableop_resourceHsequential_6_lstm_13_while_lstm_cell_13_matmul_readvariableop_resource_0"?
?sequential_6_lstm_13_while_sequential_6_lstm_13_strided_slice_1Asequential_6_lstm_13_while_sequential_6_lstm_13_strided_slice_1_0"?
{sequential_6_lstm_13_while_tensorarrayv2read_tensorlistgetitem_sequential_6_lstm_13_tensorarrayunstack_tensorlistfromtensor}sequential_6_lstm_13_while_tensorarrayv2read_tensorlistgetitem_sequential_6_lstm_13_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :??????????:??????????: : : : : 2?
>sequential_6/lstm_13/while/lstm_cell_13/BiasAdd/ReadVariableOp>sequential_6/lstm_13/while/lstm_cell_13/BiasAdd/ReadVariableOp2~
=sequential_6/lstm_13/while/lstm_cell_13/MatMul/ReadVariableOp=sequential_6/lstm_13/while/lstm_cell_13/MatMul/ReadVariableOp2?
?sequential_6/lstm_13/while/lstm_cell_13/MatMul_1/ReadVariableOp?sequential_6/lstm_13/while/lstm_cell_13/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
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
lstm_13_while_cond_26067332,
(lstm_13_while_lstm_13_while_loop_counter2
.lstm_13_while_lstm_13_while_maximum_iterations
lstm_13_while_placeholder
lstm_13_while_placeholder_1
lstm_13_while_placeholder_2
lstm_13_while_placeholder_3.
*lstm_13_while_less_lstm_13_strided_slice_1F
Blstm_13_while_lstm_13_while_cond_26067332___redundant_placeholder0F
Blstm_13_while_lstm_13_while_cond_26067332___redundant_placeholder1F
Blstm_13_while_lstm_13_while_cond_26067332___redundant_placeholder2F
Blstm_13_while_lstm_13_while_cond_26067332___redundant_placeholder3
lstm_13_while_identity
?
lstm_13/while/LessLesslstm_13_while_placeholder*lstm_13_while_less_lstm_13_strided_slice_1*
T0*
_output_shapes
: 2
lstm_13/while/Lessu
lstm_13/while/IdentityIdentitylstm_13/while/Less:z:0*
T0
*
_output_shapes
: 2
lstm_13/while/Identity"9
lstm_13_while_identitylstm_13/while/Identity:output:0*(
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
?
J__inference_lstm_cell_12_layer_call_and_return_conditional_losses_26064782

inputs

states
states_11
matmul_readvariableop_resource:	]?4
 matmul_1_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity

identity_1

identity_2??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?MatMul_1/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	]?*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
add?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
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
_construction_contextkEagerRuntime*T
_input_shapesC
A:?????????]:??????????:??????????: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:?????????]
 
_user_specified_nameinputs:PL
(
_output_shapes
:??????????
 
_user_specified_namestates:PL
(
_output_shapes
:??????????
 
_user_specified_namestates
?\
?
E__inference_lstm_12_layer_call_and_return_conditional_losses_26066608

inputs>
+lstm_cell_12_matmul_readvariableop_resource:	]?A
-lstm_cell_12_matmul_1_readvariableop_resource:
??;
,lstm_cell_12_biasadd_readvariableop_resource:	?
identity??#lstm_cell_12/BiasAdd/ReadVariableOp?"lstm_cell_12/MatMul/ReadVariableOp?$lstm_cell_12/MatMul_1/ReadVariableOp?whileD
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
"lstm_cell_12/MatMul/ReadVariableOpReadVariableOp+lstm_cell_12_matmul_readvariableop_resource*
_output_shapes
:	]?*
dtype02$
"lstm_cell_12/MatMul/ReadVariableOp?
lstm_cell_12/MatMulMatMulstrided_slice_2:output:0*lstm_cell_12/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_12/MatMul?
$lstm_cell_12/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_12_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02&
$lstm_cell_12/MatMul_1/ReadVariableOp?
lstm_cell_12/MatMul_1MatMulzeros:output:0,lstm_cell_12/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_12/MatMul_1?
lstm_cell_12/addAddV2lstm_cell_12/MatMul:product:0lstm_cell_12/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell_12/add?
#lstm_cell_12/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_12_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02%
#lstm_cell_12/BiasAdd/ReadVariableOp?
lstm_cell_12/BiasAddBiasAddlstm_cell_12/add:z:0+lstm_cell_12/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_12/BiasAdd~
lstm_cell_12/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_12/split/split_dim?
lstm_cell_12/splitSplit%lstm_cell_12/split/split_dim:output:0lstm_cell_12/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split2
lstm_cell_12/split?
lstm_cell_12/SigmoidSigmoidlstm_cell_12/split:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_12/Sigmoid?
lstm_cell_12/Sigmoid_1Sigmoidlstm_cell_12/split:output:1*
T0*(
_output_shapes
:??????????2
lstm_cell_12/Sigmoid_1?
lstm_cell_12/mulMullstm_cell_12/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_12/mul~
lstm_cell_12/ReluRelulstm_cell_12/split:output:2*
T0*(
_output_shapes
:??????????2
lstm_cell_12/Relu?
lstm_cell_12/mul_1Mullstm_cell_12/Sigmoid:y:0lstm_cell_12/Relu:activations:0*
T0*(
_output_shapes
:??????????2
lstm_cell_12/mul_1?
lstm_cell_12/add_1AddV2lstm_cell_12/mul:z:0lstm_cell_12/mul_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell_12/add_1?
lstm_cell_12/Sigmoid_2Sigmoidlstm_cell_12/split:output:3*
T0*(
_output_shapes
:??????????2
lstm_cell_12/Sigmoid_2}
lstm_cell_12/Relu_1Relulstm_cell_12/add_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell_12/Relu_1?
lstm_cell_12/mul_2Mullstm_cell_12/Sigmoid_2:y:0!lstm_cell_12/Relu_1:activations:0*
T0*(
_output_shapes
:??????????2
lstm_cell_12/mul_2?
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_12_matmul_readvariableop_resource-lstm_cell_12_matmul_1_readvariableop_resource,lstm_cell_12_biasadd_readvariableop_resource*
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
while_body_26066524*
condR
while_cond_26066523*M
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
NoOpNoOp$^lstm_cell_12/BiasAdd/ReadVariableOp#^lstm_cell_12/MatMul/ReadVariableOp%^lstm_cell_12/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????]: : : 2J
#lstm_cell_12/BiasAdd/ReadVariableOp#lstm_cell_12/BiasAdd/ReadVariableOp2H
"lstm_cell_12/MatMul/ReadVariableOp"lstm_cell_12/MatMul/ReadVariableOp2L
$lstm_cell_12/MatMul_1/ReadVariableOp$lstm_cell_12/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:?????????]
 
_user_specified_nameinputs
?

?
/__inference_sequential_6_layer_call_fn_26067473

inputs
unknown:	]?
	unknown_0:
??
	unknown_1:	?
	unknown_2:
??
	unknown_3:
??
	unknown_4:	?
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
J__inference_sequential_6_layer_call_and_return_conditional_losses_260661962
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
?\
?
E__inference_lstm_12_layer_call_and_return_conditional_losses_26067796
inputs_0>
+lstm_cell_12_matmul_readvariableop_resource:	]?A
-lstm_cell_12_matmul_1_readvariableop_resource:
??;
,lstm_cell_12_biasadd_readvariableop_resource:	?
identity??#lstm_cell_12/BiasAdd/ReadVariableOp?"lstm_cell_12/MatMul/ReadVariableOp?$lstm_cell_12/MatMul_1/ReadVariableOp?whileF
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
"lstm_cell_12/MatMul/ReadVariableOpReadVariableOp+lstm_cell_12_matmul_readvariableop_resource*
_output_shapes
:	]?*
dtype02$
"lstm_cell_12/MatMul/ReadVariableOp?
lstm_cell_12/MatMulMatMulstrided_slice_2:output:0*lstm_cell_12/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_12/MatMul?
$lstm_cell_12/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_12_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02&
$lstm_cell_12/MatMul_1/ReadVariableOp?
lstm_cell_12/MatMul_1MatMulzeros:output:0,lstm_cell_12/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_12/MatMul_1?
lstm_cell_12/addAddV2lstm_cell_12/MatMul:product:0lstm_cell_12/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell_12/add?
#lstm_cell_12/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_12_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02%
#lstm_cell_12/BiasAdd/ReadVariableOp?
lstm_cell_12/BiasAddBiasAddlstm_cell_12/add:z:0+lstm_cell_12/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_12/BiasAdd~
lstm_cell_12/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_12/split/split_dim?
lstm_cell_12/splitSplit%lstm_cell_12/split/split_dim:output:0lstm_cell_12/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split2
lstm_cell_12/split?
lstm_cell_12/SigmoidSigmoidlstm_cell_12/split:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_12/Sigmoid?
lstm_cell_12/Sigmoid_1Sigmoidlstm_cell_12/split:output:1*
T0*(
_output_shapes
:??????????2
lstm_cell_12/Sigmoid_1?
lstm_cell_12/mulMullstm_cell_12/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_12/mul~
lstm_cell_12/ReluRelulstm_cell_12/split:output:2*
T0*(
_output_shapes
:??????????2
lstm_cell_12/Relu?
lstm_cell_12/mul_1Mullstm_cell_12/Sigmoid:y:0lstm_cell_12/Relu:activations:0*
T0*(
_output_shapes
:??????????2
lstm_cell_12/mul_1?
lstm_cell_12/add_1AddV2lstm_cell_12/mul:z:0lstm_cell_12/mul_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell_12/add_1?
lstm_cell_12/Sigmoid_2Sigmoidlstm_cell_12/split:output:3*
T0*(
_output_shapes
:??????????2
lstm_cell_12/Sigmoid_2}
lstm_cell_12/Relu_1Relulstm_cell_12/add_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell_12/Relu_1?
lstm_cell_12/mul_2Mullstm_cell_12/Sigmoid_2:y:0!lstm_cell_12/Relu_1:activations:0*
T0*(
_output_shapes
:??????????2
lstm_cell_12/mul_2?
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_12_matmul_readvariableop_resource-lstm_cell_12_matmul_1_readvariableop_resource,lstm_cell_12_biasadd_readvariableop_resource*
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
while_body_26067712*
condR
while_cond_26067711*M
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
NoOpNoOp$^lstm_cell_12/BiasAdd/ReadVariableOp#^lstm_cell_12/MatMul/ReadVariableOp%^lstm_cell_12/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????????????]: : : 2J
#lstm_cell_12/BiasAdd/ReadVariableOp#lstm_cell_12/BiasAdd/ReadVariableOp2H
"lstm_cell_12/MatMul/ReadVariableOp"lstm_cell_12/MatMul/ReadVariableOp2L
$lstm_cell_12/MatMul_1/ReadVariableOp$lstm_cell_12/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :??????????????????]
"
_user_specified_name
inputs/0
?\
?
E__inference_lstm_12_layer_call_and_return_conditional_losses_26067645
inputs_0>
+lstm_cell_12_matmul_readvariableop_resource:	]?A
-lstm_cell_12_matmul_1_readvariableop_resource:
??;
,lstm_cell_12_biasadd_readvariableop_resource:	?
identity??#lstm_cell_12/BiasAdd/ReadVariableOp?"lstm_cell_12/MatMul/ReadVariableOp?$lstm_cell_12/MatMul_1/ReadVariableOp?whileF
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
"lstm_cell_12/MatMul/ReadVariableOpReadVariableOp+lstm_cell_12_matmul_readvariableop_resource*
_output_shapes
:	]?*
dtype02$
"lstm_cell_12/MatMul/ReadVariableOp?
lstm_cell_12/MatMulMatMulstrided_slice_2:output:0*lstm_cell_12/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_12/MatMul?
$lstm_cell_12/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_12_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02&
$lstm_cell_12/MatMul_1/ReadVariableOp?
lstm_cell_12/MatMul_1MatMulzeros:output:0,lstm_cell_12/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_12/MatMul_1?
lstm_cell_12/addAddV2lstm_cell_12/MatMul:product:0lstm_cell_12/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell_12/add?
#lstm_cell_12/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_12_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02%
#lstm_cell_12/BiasAdd/ReadVariableOp?
lstm_cell_12/BiasAddBiasAddlstm_cell_12/add:z:0+lstm_cell_12/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_12/BiasAdd~
lstm_cell_12/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_12/split/split_dim?
lstm_cell_12/splitSplit%lstm_cell_12/split/split_dim:output:0lstm_cell_12/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split2
lstm_cell_12/split?
lstm_cell_12/SigmoidSigmoidlstm_cell_12/split:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_12/Sigmoid?
lstm_cell_12/Sigmoid_1Sigmoidlstm_cell_12/split:output:1*
T0*(
_output_shapes
:??????????2
lstm_cell_12/Sigmoid_1?
lstm_cell_12/mulMullstm_cell_12/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_12/mul~
lstm_cell_12/ReluRelulstm_cell_12/split:output:2*
T0*(
_output_shapes
:??????????2
lstm_cell_12/Relu?
lstm_cell_12/mul_1Mullstm_cell_12/Sigmoid:y:0lstm_cell_12/Relu:activations:0*
T0*(
_output_shapes
:??????????2
lstm_cell_12/mul_1?
lstm_cell_12/add_1AddV2lstm_cell_12/mul:z:0lstm_cell_12/mul_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell_12/add_1?
lstm_cell_12/Sigmoid_2Sigmoidlstm_cell_12/split:output:3*
T0*(
_output_shapes
:??????????2
lstm_cell_12/Sigmoid_2}
lstm_cell_12/Relu_1Relulstm_cell_12/add_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell_12/Relu_1?
lstm_cell_12/mul_2Mullstm_cell_12/Sigmoid_2:y:0!lstm_cell_12/Relu_1:activations:0*
T0*(
_output_shapes
:??????????2
lstm_cell_12/mul_2?
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_12_matmul_readvariableop_resource-lstm_cell_12_matmul_1_readvariableop_resource,lstm_cell_12_biasadd_readvariableop_resource*
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
while_body_26067561*
condR
while_cond_26067560*M
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
NoOpNoOp$^lstm_cell_12/BiasAdd/ReadVariableOp#^lstm_cell_12/MatMul/ReadVariableOp%^lstm_cell_12/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????????????]: : : 2J
#lstm_cell_12/BiasAdd/ReadVariableOp#lstm_cell_12/BiasAdd/ReadVariableOp2H
"lstm_cell_12/MatMul/ReadVariableOp"lstm_cell_12/MatMul/ReadVariableOp2L
$lstm_cell_12/MatMul_1/ReadVariableOp$lstm_cell_12/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :??????????????????]
"
_user_specified_name
inputs/0
?
?
while_cond_26065279
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_26065279___redundant_placeholder06
2while_while_cond_26065279___redundant_placeholder16
2while_while_cond_26065279___redundant_placeholder26
2while_while_cond_26065279___redundant_placeholder3
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
?
*__inference_lstm_13_layer_call_fn_26068795
inputs_0
unknown:
??
	unknown_0:
??
	unknown_1:	?
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
E__inference_lstm_13_layer_call_and_return_conditional_losses_260655592
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
':???????????????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
5
_output_shapes#
!:???????????????????
"
_user_specified_name
inputs/0
?&
?
while_body_26064860
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_00
while_lstm_cell_12_26064884_0:	]?1
while_lstm_cell_12_26064886_0:
??,
while_lstm_cell_12_26064888_0:	?
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor.
while_lstm_cell_12_26064884:	]?/
while_lstm_cell_12_26064886:
??*
while_lstm_cell_12_26064888:	???*while/lstm_cell_12/StatefulPartitionedCall?
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
*while/lstm_cell_12/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_12_26064884_0while_lstm_cell_12_26064886_0while_lstm_cell_12_26064888_0*
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
J__inference_lstm_cell_12_layer_call_and_return_conditional_losses_260647822,
*while/lstm_cell_12/StatefulPartitionedCall?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder3while/lstm_cell_12/StatefulPartitionedCall:output:0*
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
while/Identity_4Identity3while/lstm_cell_12/StatefulPartitionedCall:output:1^while/NoOp*
T0*(
_output_shapes
:??????????2
while/Identity_4?
while/Identity_5Identity3while/lstm_cell_12/StatefulPartitionedCall:output:2^while/NoOp*
T0*(
_output_shapes
:??????????2
while/Identity_5?

while/NoOpNoOp+^while/lstm_cell_12/StatefulPartitionedCall*"
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
while_lstm_cell_12_26064884while_lstm_cell_12_26064884_0"<
while_lstm_cell_12_26064886while_lstm_cell_12_26064886_0"<
while_lstm_cell_12_26064888while_lstm_cell_12_26064888_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :??????????:??????????: : : : : 2X
*while/lstm_cell_12/StatefulPartitionedCall*while/lstm_cell_12/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
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
?
J__inference_lstm_cell_13_layer_call_and_return_conditional_losses_26069046

inputs
states_0
states_12
matmul_readvariableop_resource:
??4
 matmul_1_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity

identity_1

identity_2??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?MatMul_1/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul_1/ReadVariableOp|
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
add?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
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
B:??????????:??????????:??????????: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:P L
(
_output_shapes
:??????????
 
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
?\
?
E__inference_lstm_13_layer_call_and_return_conditional_losses_26066412

inputs?
+lstm_cell_13_matmul_readvariableop_resource:
??A
-lstm_cell_13_matmul_1_readvariableop_resource:
??;
,lstm_cell_13_biasadd_readvariableop_resource:	?
identity??#lstm_cell_13/BiasAdd/ReadVariableOp?"lstm_cell_13/MatMul/ReadVariableOp?$lstm_cell_13/MatMul_1/ReadVariableOp?whileD
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
:??????????2
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
valueB"?????   27
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
:??????????*
shrink_axis_mask2
strided_slice_2?
"lstm_cell_13/MatMul/ReadVariableOpReadVariableOp+lstm_cell_13_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02$
"lstm_cell_13/MatMul/ReadVariableOp?
lstm_cell_13/MatMulMatMulstrided_slice_2:output:0*lstm_cell_13/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_13/MatMul?
$lstm_cell_13/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_13_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02&
$lstm_cell_13/MatMul_1/ReadVariableOp?
lstm_cell_13/MatMul_1MatMulzeros:output:0,lstm_cell_13/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_13/MatMul_1?
lstm_cell_13/addAddV2lstm_cell_13/MatMul:product:0lstm_cell_13/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell_13/add?
#lstm_cell_13/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_13_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02%
#lstm_cell_13/BiasAdd/ReadVariableOp?
lstm_cell_13/BiasAddBiasAddlstm_cell_13/add:z:0+lstm_cell_13/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_13/BiasAdd~
lstm_cell_13/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_13/split/split_dim?
lstm_cell_13/splitSplit%lstm_cell_13/split/split_dim:output:0lstm_cell_13/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split2
lstm_cell_13/split?
lstm_cell_13/SigmoidSigmoidlstm_cell_13/split:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_13/Sigmoid?
lstm_cell_13/Sigmoid_1Sigmoidlstm_cell_13/split:output:1*
T0*(
_output_shapes
:??????????2
lstm_cell_13/Sigmoid_1?
lstm_cell_13/mulMullstm_cell_13/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_13/mul~
lstm_cell_13/ReluRelulstm_cell_13/split:output:2*
T0*(
_output_shapes
:??????????2
lstm_cell_13/Relu?
lstm_cell_13/mul_1Mullstm_cell_13/Sigmoid:y:0lstm_cell_13/Relu:activations:0*
T0*(
_output_shapes
:??????????2
lstm_cell_13/mul_1?
lstm_cell_13/add_1AddV2lstm_cell_13/mul:z:0lstm_cell_13/mul_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell_13/add_1?
lstm_cell_13/Sigmoid_2Sigmoidlstm_cell_13/split:output:3*
T0*(
_output_shapes
:??????????2
lstm_cell_13/Sigmoid_2}
lstm_cell_13/Relu_1Relulstm_cell_13/add_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell_13/Relu_1?
lstm_cell_13/mul_2Mullstm_cell_13/Sigmoid_2:y:0!lstm_cell_13/Relu_1:activations:0*
T0*(
_output_shapes
:??????????2
lstm_cell_13/mul_2?
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_13_matmul_readvariableop_resource-lstm_cell_13_matmul_1_readvariableop_resource,lstm_cell_13_biasadd_readvariableop_resource*
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
while_body_26066328*
condR
while_cond_26066327*M
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
NoOpNoOp$^lstm_cell_13/BiasAdd/ReadVariableOp#^lstm_cell_13/MatMul/ReadVariableOp%^lstm_cell_13/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:??????????: : : 2J
#lstm_cell_13/BiasAdd/ReadVariableOp#lstm_cell_13/BiasAdd/ReadVariableOp2H
"lstm_cell_13/MatMul/ReadVariableOp"lstm_cell_13/MatMul/ReadVariableOp2L
$lstm_cell_13/MatMul_1/ReadVariableOp$lstm_cell_13/MatMul_1/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?

?
lstm_12_while_cond_26067177,
(lstm_12_while_lstm_12_while_loop_counter2
.lstm_12_while_lstm_12_while_maximum_iterations
lstm_12_while_placeholder
lstm_12_while_placeholder_1
lstm_12_while_placeholder_2
lstm_12_while_placeholder_3.
*lstm_12_while_less_lstm_12_strided_slice_1F
Blstm_12_while_lstm_12_while_cond_26067177___redundant_placeholder0F
Blstm_12_while_lstm_12_while_cond_26067177___redundant_placeholder1F
Blstm_12_while_lstm_12_while_cond_26067177___redundant_placeholder2F
Blstm_12_while_lstm_12_while_cond_26067177___redundant_placeholder3
lstm_12_while_identity
?
lstm_12/while/LessLesslstm_12_while_placeholder*lstm_12_while_less_lstm_12_strided_slice_1*
T0*
_output_shapes
: 2
lstm_12/while/Lessu
lstm_12/while/IdentityIdentitylstm_12/while/Less:z:0*
T0
*
_output_shapes
: 2
lstm_12/while/Identity"9
lstm_12_while_identitylstm_12/while/Identity:output:0*(
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
while_cond_26068537
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_26068537___redundant_placeholder06
2while_while_cond_26068537___redundant_placeholder16
2while_while_cond_26068537___redundant_placeholder26
2while_while_cond_26068537___redundant_placeholder3
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
?
*__inference_dense_6_layer_call_fn_26068884

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
E__inference_dense_6_layer_call_and_return_conditional_losses_260661892
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
?
?
J__inference_sequential_6_layer_call_and_return_conditional_losses_26066196

inputs#
lstm_12_26065979:	]?$
lstm_12_26065981:
??
lstm_12_26065983:	?$
lstm_13_26066144:
??$
lstm_13_26066146:
??
lstm_13_26066148:	?#
dense_6_26066190:	?
dense_6_26066192:
identity??dense_6/StatefulPartitionedCall?lstm_12/StatefulPartitionedCall?lstm_13/StatefulPartitionedCall?
lstm_12/StatefulPartitionedCallStatefulPartitionedCallinputslstm_12_26065979lstm_12_26065981lstm_12_26065983*
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
E__inference_lstm_12_layer_call_and_return_conditional_losses_260659782!
lstm_12/StatefulPartitionedCall?
dropout_12/PartitionedCallPartitionedCall(lstm_12/StatefulPartitionedCall:output:0*
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
H__inference_dropout_12_layer_call_and_return_conditional_losses_260659912
dropout_12/PartitionedCall?
lstm_13/StatefulPartitionedCallStatefulPartitionedCall#dropout_12/PartitionedCall:output:0lstm_13_26066144lstm_13_26066146lstm_13_26066148*
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
E__inference_lstm_13_layer_call_and_return_conditional_losses_260661432!
lstm_13/StatefulPartitionedCall?
dropout_13/PartitionedCallPartitionedCall(lstm_13/StatefulPartitionedCall:output:0*
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
H__inference_dropout_13_layer_call_and_return_conditional_losses_260661562
dropout_13/PartitionedCall?
dense_6/StatefulPartitionedCallStatefulPartitionedCall#dropout_13/PartitionedCall:output:0dense_6_26066190dense_6_26066192*
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
E__inference_dense_6_layer_call_and_return_conditional_losses_260661892!
dense_6/StatefulPartitionedCall?
IdentityIdentity(dense_6/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????2

Identity?
NoOpNoOp ^dense_6/StatefulPartitionedCall ^lstm_12/StatefulPartitionedCall ^lstm_13/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????]: : : : : : : : 2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2B
lstm_12/StatefulPartitionedCalllstm_12/StatefulPartitionedCall2B
lstm_13/StatefulPartitionedCalllstm_13/StatefulPartitionedCall:S O
+
_output_shapes
:?????????]
 
_user_specified_nameinputs
?F
?
E__inference_lstm_12_layer_call_and_return_conditional_losses_26064929

inputs(
lstm_cell_12_26064847:	]?)
lstm_cell_12_26064849:
??$
lstm_cell_12_26064851:	?
identity??$lstm_cell_12/StatefulPartitionedCall?whileD
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
$lstm_cell_12/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_12_26064847lstm_cell_12_26064849lstm_cell_12_26064851*
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
J__inference_lstm_cell_12_layer_call_and_return_conditional_losses_260647822&
$lstm_cell_12/StatefulPartitionedCall?
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_12_26064847lstm_cell_12_26064849lstm_cell_12_26064851*
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
while_body_26064860*
condR
while_cond_26064859*M
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
NoOpNoOp%^lstm_cell_12/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:??????????????????]: : : 2L
$lstm_cell_12/StatefulPartitionedCall$lstm_cell_12/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :??????????????????]
 
_user_specified_nameinputs
?
?
/__inference_lstm_cell_13_layer_call_fn_26069063

inputs
states_0
states_1
unknown:
??
	unknown_0:
??
	unknown_1:	?
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
J__inference_lstm_cell_13_layer_call_and_return_conditional_losses_260652662
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
B:??????????:??????????:??????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
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
while_cond_26067862
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_26067862___redundant_placeholder06
2while_while_cond_26067862___redundant_placeholder16
2while_while_cond_26067862___redundant_placeholder26
2while_while_cond_26067862___redundant_placeholder3
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
H__inference_dropout_13_layer_call_and_return_conditional_losses_26066245

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
ƒ
?
$__inference__traced_restore_26069311
file_prefix2
assignvariableop_dense_6_kernel:	?-
assignvariableop_1_dense_6_bias:&
assignvariableop_2_adam_iter:	 (
assignvariableop_3_adam_beta_1: (
assignvariableop_4_adam_beta_2: '
assignvariableop_5_adam_decay: /
%assignvariableop_6_adam_learning_rate: A
.assignvariableop_7_lstm_12_lstm_cell_12_kernel:	]?L
8assignvariableop_8_lstm_12_lstm_cell_12_recurrent_kernel:
??;
,assignvariableop_9_lstm_12_lstm_cell_12_bias:	?C
/assignvariableop_10_lstm_13_lstm_cell_13_kernel:
??M
9assignvariableop_11_lstm_13_lstm_cell_13_recurrent_kernel:
??<
-assignvariableop_12_lstm_13_lstm_cell_13_bias:	?#
assignvariableop_13_total: #
assignvariableop_14_count: %
assignvariableop_15_total_1: %
assignvariableop_16_count_1: <
)assignvariableop_17_adam_dense_6_kernel_m:	?5
'assignvariableop_18_adam_dense_6_bias_m:I
6assignvariableop_19_adam_lstm_12_lstm_cell_12_kernel_m:	]?T
@assignvariableop_20_adam_lstm_12_lstm_cell_12_recurrent_kernel_m:
??C
4assignvariableop_21_adam_lstm_12_lstm_cell_12_bias_m:	?J
6assignvariableop_22_adam_lstm_13_lstm_cell_13_kernel_m:
??T
@assignvariableop_23_adam_lstm_13_lstm_cell_13_recurrent_kernel_m:
??C
4assignvariableop_24_adam_lstm_13_lstm_cell_13_bias_m:	?<
)assignvariableop_25_adam_dense_6_kernel_v:	?5
'assignvariableop_26_adam_dense_6_bias_v:I
6assignvariableop_27_adam_lstm_12_lstm_cell_12_kernel_v:	]?T
@assignvariableop_28_adam_lstm_12_lstm_cell_12_recurrent_kernel_v:
??C
4assignvariableop_29_adam_lstm_12_lstm_cell_12_bias_v:	?J
6assignvariableop_30_adam_lstm_13_lstm_cell_13_kernel_v:
??T
@assignvariableop_31_adam_lstm_13_lstm_cell_13_recurrent_kernel_v:
??C
4assignvariableop_32_adam_lstm_13_lstm_cell_13_bias_v:	?
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
AssignVariableOpAssignVariableOpassignvariableop_dense_6_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_6_biasIdentity_1:output:0"/device:CPU:0*
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
AssignVariableOp_7AssignVariableOp.assignvariableop_7_lstm_12_lstm_cell_12_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp8assignvariableop_8_lstm_12_lstm_cell_12_recurrent_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp,assignvariableop_9_lstm_12_lstm_cell_12_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp/assignvariableop_10_lstm_13_lstm_cell_13_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOp9assignvariableop_11_lstm_13_lstm_cell_13_recurrent_kernelIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOp-assignvariableop_12_lstm_13_lstm_cell_13_biasIdentity_12:output:0"/device:CPU:0*
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
AssignVariableOp_17AssignVariableOp)assignvariableop_17_adam_dense_6_kernel_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOp'assignvariableop_18_adam_dense_6_bias_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOp6assignvariableop_19_adam_lstm_12_lstm_cell_12_kernel_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOp@assignvariableop_20_adam_lstm_12_lstm_cell_12_recurrent_kernel_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOp4assignvariableop_21_adam_lstm_12_lstm_cell_12_bias_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOp6assignvariableop_22_adam_lstm_13_lstm_cell_13_kernel_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOp@assignvariableop_23_adam_lstm_13_lstm_cell_13_recurrent_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOp4assignvariableop_24_adam_lstm_13_lstm_cell_13_bias_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOp)assignvariableop_25_adam_dense_6_kernel_vIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOp'assignvariableop_26_adam_dense_6_bias_vIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOp6assignvariableop_27_adam_lstm_12_lstm_cell_12_kernel_vIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOp@assignvariableop_28_adam_lstm_12_lstm_cell_12_recurrent_kernel_vIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOp4assignvariableop_29_adam_lstm_12_lstm_cell_12_bias_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOp6assignvariableop_30_adam_lstm_13_lstm_cell_13_kernel_vIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31?
AssignVariableOp_31AssignVariableOp@assignvariableop_31_adam_lstm_13_lstm_cell_13_recurrent_kernel_vIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32?
AssignVariableOp_32AssignVariableOp4assignvariableop_32_adam_lstm_13_lstm_cell_13_bias_vIdentity_32:output:0"/device:CPU:0*
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
while_cond_26068386
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_26068386___redundant_placeholder06
2while_while_cond_26068386___redundant_placeholder16
2while_while_cond_26068386___redundant_placeholder26
2while_while_cond_26068386___redundant_placeholder3
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
?
I
-__inference_dropout_12_layer_call_fn_26068164

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
H__inference_dropout_12_layer_call_and_return_conditional_losses_260659912
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
?
?
/__inference_lstm_cell_12_layer_call_fn_26068982

inputs
states_0
states_1
unknown:	]?
	unknown_0:
??
	unknown_1:	?
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
J__inference_lstm_cell_12_layer_call_and_return_conditional_losses_260647822
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
_construction_contextkEagerRuntime*T
_input_shapesC
A:?????????]:??????????:??????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????]
 
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
?F
?
E__inference_lstm_13_layer_call_and_return_conditional_losses_26065349

inputs)
lstm_cell_13_26065267:
??)
lstm_cell_13_26065269:
??$
lstm_cell_13_26065271:	?
identity??$lstm_cell_13/StatefulPartitionedCall?whileD
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
!:???????????????????2
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
valueB"?????   27
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
:??????????*
shrink_axis_mask2
strided_slice_2?
$lstm_cell_13/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_13_26065267lstm_cell_13_26065269lstm_cell_13_26065271*
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
J__inference_lstm_cell_13_layer_call_and_return_conditional_losses_260652662&
$lstm_cell_13/StatefulPartitionedCall?
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_13_26065267lstm_cell_13_26065269lstm_cell_13_26065271*
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
while_body_26065280*
condR
while_cond_26065279*M
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
NoOpNoOp%^lstm_cell_13/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':???????????????????: : : 2L
$lstm_cell_13/StatefulPartitionedCall$lstm_cell_13/StatefulPartitionedCall2
whilewhile:] Y
5
_output_shapes#
!:???????????????????
 
_user_specified_nameinputs
?
?
*__inference_lstm_12_layer_call_fn_26068120
inputs_0
unknown:	]?
	unknown_0:
??
	unknown_1:	?
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
E__inference_lstm_12_layer_call_and_return_conditional_losses_260649292
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
?
f
H__inference_dropout_12_layer_call_and_return_conditional_losses_26065991

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
?\
?
E__inference_lstm_13_layer_call_and_return_conditional_losses_26068622

inputs?
+lstm_cell_13_matmul_readvariableop_resource:
??A
-lstm_cell_13_matmul_1_readvariableop_resource:
??;
,lstm_cell_13_biasadd_readvariableop_resource:	?
identity??#lstm_cell_13/BiasAdd/ReadVariableOp?"lstm_cell_13/MatMul/ReadVariableOp?$lstm_cell_13/MatMul_1/ReadVariableOp?whileD
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
:??????????2
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
valueB"?????   27
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
:??????????*
shrink_axis_mask2
strided_slice_2?
"lstm_cell_13/MatMul/ReadVariableOpReadVariableOp+lstm_cell_13_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02$
"lstm_cell_13/MatMul/ReadVariableOp?
lstm_cell_13/MatMulMatMulstrided_slice_2:output:0*lstm_cell_13/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_13/MatMul?
$lstm_cell_13/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_13_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02&
$lstm_cell_13/MatMul_1/ReadVariableOp?
lstm_cell_13/MatMul_1MatMulzeros:output:0,lstm_cell_13/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_13/MatMul_1?
lstm_cell_13/addAddV2lstm_cell_13/MatMul:product:0lstm_cell_13/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell_13/add?
#lstm_cell_13/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_13_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02%
#lstm_cell_13/BiasAdd/ReadVariableOp?
lstm_cell_13/BiasAddBiasAddlstm_cell_13/add:z:0+lstm_cell_13/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_13/BiasAdd~
lstm_cell_13/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_13/split/split_dim?
lstm_cell_13/splitSplit%lstm_cell_13/split/split_dim:output:0lstm_cell_13/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split2
lstm_cell_13/split?
lstm_cell_13/SigmoidSigmoidlstm_cell_13/split:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_13/Sigmoid?
lstm_cell_13/Sigmoid_1Sigmoidlstm_cell_13/split:output:1*
T0*(
_output_shapes
:??????????2
lstm_cell_13/Sigmoid_1?
lstm_cell_13/mulMullstm_cell_13/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_13/mul~
lstm_cell_13/ReluRelulstm_cell_13/split:output:2*
T0*(
_output_shapes
:??????????2
lstm_cell_13/Relu?
lstm_cell_13/mul_1Mullstm_cell_13/Sigmoid:y:0lstm_cell_13/Relu:activations:0*
T0*(
_output_shapes
:??????????2
lstm_cell_13/mul_1?
lstm_cell_13/add_1AddV2lstm_cell_13/mul:z:0lstm_cell_13/mul_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell_13/add_1?
lstm_cell_13/Sigmoid_2Sigmoidlstm_cell_13/split:output:3*
T0*(
_output_shapes
:??????????2
lstm_cell_13/Sigmoid_2}
lstm_cell_13/Relu_1Relulstm_cell_13/add_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell_13/Relu_1?
lstm_cell_13/mul_2Mullstm_cell_13/Sigmoid_2:y:0!lstm_cell_13/Relu_1:activations:0*
T0*(
_output_shapes
:??????????2
lstm_cell_13/mul_2?
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_13_matmul_readvariableop_resource-lstm_cell_13_matmul_1_readvariableop_resource,lstm_cell_13_biasadd_readvariableop_resource*
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
while_body_26068538*
condR
while_cond_26068537*M
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
NoOpNoOp$^lstm_cell_13/BiasAdd/ReadVariableOp#^lstm_cell_13/MatMul/ReadVariableOp%^lstm_cell_13/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:??????????: : : 2J
#lstm_cell_13/BiasAdd/ReadVariableOp#lstm_cell_13/BiasAdd/ReadVariableOp2H
"lstm_cell_13/MatMul/ReadVariableOp"lstm_cell_13/MatMul/ReadVariableOp2L
$lstm_cell_13/MatMul_1/ReadVariableOp$lstm_cell_13/MatMul_1/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
??
?
while_body_26066059
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
3while_lstm_cell_13_matmul_readvariableop_resource_0:
??I
5while_lstm_cell_13_matmul_1_readvariableop_resource_0:
??C
4while_lstm_cell_13_biasadd_readvariableop_resource_0:	?
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
1while_lstm_cell_13_matmul_readvariableop_resource:
??G
3while_lstm_cell_13_matmul_1_readvariableop_resource:
??A
2while_lstm_cell_13_biasadd_readvariableop_resource:	???)while/lstm_cell_13/BiasAdd/ReadVariableOp?(while/lstm_cell_13/MatMul/ReadVariableOp?*while/lstm_cell_13/MatMul_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:??????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
(while/lstm_cell_13/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_13_matmul_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02*
(while/lstm_cell_13/MatMul/ReadVariableOp?
while/lstm_cell_13/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_13/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_13/MatMul?
*while/lstm_cell_13/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_13_matmul_1_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02,
*while/lstm_cell_13/MatMul_1/ReadVariableOp?
while/lstm_cell_13/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_13/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_13/MatMul_1?
while/lstm_cell_13/addAddV2#while/lstm_cell_13/MatMul:product:0%while/lstm_cell_13/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_13/add?
)while/lstm_cell_13/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_13_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype02+
)while/lstm_cell_13/BiasAdd/ReadVariableOp?
while/lstm_cell_13/BiasAddBiasAddwhile/lstm_cell_13/add:z:01while/lstm_cell_13/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_13/BiasAdd?
"while/lstm_cell_13/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_13/split/split_dim?
while/lstm_cell_13/splitSplit+while/lstm_cell_13/split/split_dim:output:0#while/lstm_cell_13/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split2
while/lstm_cell_13/split?
while/lstm_cell_13/SigmoidSigmoid!while/lstm_cell_13/split:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_13/Sigmoid?
while/lstm_cell_13/Sigmoid_1Sigmoid!while/lstm_cell_13/split:output:1*
T0*(
_output_shapes
:??????????2
while/lstm_cell_13/Sigmoid_1?
while/lstm_cell_13/mulMul while/lstm_cell_13/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:??????????2
while/lstm_cell_13/mul?
while/lstm_cell_13/ReluRelu!while/lstm_cell_13/split:output:2*
T0*(
_output_shapes
:??????????2
while/lstm_cell_13/Relu?
while/lstm_cell_13/mul_1Mulwhile/lstm_cell_13/Sigmoid:y:0%while/lstm_cell_13/Relu:activations:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_13/mul_1?
while/lstm_cell_13/add_1AddV2while/lstm_cell_13/mul:z:0while/lstm_cell_13/mul_1:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_13/add_1?
while/lstm_cell_13/Sigmoid_2Sigmoid!while/lstm_cell_13/split:output:3*
T0*(
_output_shapes
:??????????2
while/lstm_cell_13/Sigmoid_2?
while/lstm_cell_13/Relu_1Reluwhile/lstm_cell_13/add_1:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_13/Relu_1?
while/lstm_cell_13/mul_2Mul while/lstm_cell_13/Sigmoid_2:y:0'while/lstm_cell_13/Relu_1:activations:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_13/mul_2?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_13/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_13/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:??????????2
while/Identity_4?
while/Identity_5Identitywhile/lstm_cell_13/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:??????????2
while/Identity_5?

while/NoOpNoOp*^while/lstm_cell_13/BiasAdd/ReadVariableOp)^while/lstm_cell_13/MatMul/ReadVariableOp+^while/lstm_cell_13/MatMul_1/ReadVariableOp*"
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
2while_lstm_cell_13_biasadd_readvariableop_resource4while_lstm_cell_13_biasadd_readvariableop_resource_0"l
3while_lstm_cell_13_matmul_1_readvariableop_resource5while_lstm_cell_13_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_13_matmul_readvariableop_resource3while_lstm_cell_13_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :??????????:??????????: : : : : 2V
)while/lstm_cell_13/BiasAdd/ReadVariableOp)while/lstm_cell_13/BiasAdd/ReadVariableOp2T
(while/lstm_cell_13/MatMul/ReadVariableOp(while/lstm_cell_13/MatMul/ReadVariableOp2X
*while/lstm_cell_13/MatMul_1/ReadVariableOp*while/lstm_cell_13/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
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
??
?
J__inference_sequential_6_layer_call_and_return_conditional_losses_26067111

inputsF
3lstm_12_lstm_cell_12_matmul_readvariableop_resource:	]?I
5lstm_12_lstm_cell_12_matmul_1_readvariableop_resource:
??C
4lstm_12_lstm_cell_12_biasadd_readvariableop_resource:	?G
3lstm_13_lstm_cell_13_matmul_readvariableop_resource:
??I
5lstm_13_lstm_cell_13_matmul_1_readvariableop_resource:
??C
4lstm_13_lstm_cell_13_biasadd_readvariableop_resource:	?<
)dense_6_tensordot_readvariableop_resource:	?5
'dense_6_biasadd_readvariableop_resource:
identity??dense_6/BiasAdd/ReadVariableOp? dense_6/Tensordot/ReadVariableOp?+lstm_12/lstm_cell_12/BiasAdd/ReadVariableOp?*lstm_12/lstm_cell_12/MatMul/ReadVariableOp?,lstm_12/lstm_cell_12/MatMul_1/ReadVariableOp?lstm_12/while?+lstm_13/lstm_cell_13/BiasAdd/ReadVariableOp?*lstm_13/lstm_cell_13/MatMul/ReadVariableOp?,lstm_13/lstm_cell_13/MatMul_1/ReadVariableOp?lstm_13/whileT
lstm_12/ShapeShapeinputs*
T0*
_output_shapes
:2
lstm_12/Shape?
lstm_12/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_12/strided_slice/stack?
lstm_12/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_12/strided_slice/stack_1?
lstm_12/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_12/strided_slice/stack_2?
lstm_12/strided_sliceStridedSlicelstm_12/Shape:output:0$lstm_12/strided_slice/stack:output:0&lstm_12/strided_slice/stack_1:output:0&lstm_12/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_12/strided_slicem
lstm_12/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
lstm_12/zeros/mul/y?
lstm_12/zeros/mulMullstm_12/strided_slice:output:0lstm_12/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_12/zeros/mulo
lstm_12/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
lstm_12/zeros/Less/y?
lstm_12/zeros/LessLesslstm_12/zeros/mul:z:0lstm_12/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_12/zeros/Lesss
lstm_12/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
lstm_12/zeros/packed/1?
lstm_12/zeros/packedPacklstm_12/strided_slice:output:0lstm_12/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_12/zeros/packedo
lstm_12/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_12/zeros/Const?
lstm_12/zerosFilllstm_12/zeros/packed:output:0lstm_12/zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
lstm_12/zerosq
lstm_12/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
lstm_12/zeros_1/mul/y?
lstm_12/zeros_1/mulMullstm_12/strided_slice:output:0lstm_12/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_12/zeros_1/muls
lstm_12/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
lstm_12/zeros_1/Less/y?
lstm_12/zeros_1/LessLesslstm_12/zeros_1/mul:z:0lstm_12/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_12/zeros_1/Lessw
lstm_12/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
lstm_12/zeros_1/packed/1?
lstm_12/zeros_1/packedPacklstm_12/strided_slice:output:0!lstm_12/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_12/zeros_1/packeds
lstm_12/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_12/zeros_1/Const?
lstm_12/zeros_1Filllstm_12/zeros_1/packed:output:0lstm_12/zeros_1/Const:output:0*
T0*(
_output_shapes
:??????????2
lstm_12/zeros_1?
lstm_12/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_12/transpose/perm?
lstm_12/transpose	Transposeinputslstm_12/transpose/perm:output:0*
T0*+
_output_shapes
:?????????]2
lstm_12/transposeg
lstm_12/Shape_1Shapelstm_12/transpose:y:0*
T0*
_output_shapes
:2
lstm_12/Shape_1?
lstm_12/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_12/strided_slice_1/stack?
lstm_12/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_12/strided_slice_1/stack_1?
lstm_12/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_12/strided_slice_1/stack_2?
lstm_12/strided_slice_1StridedSlicelstm_12/Shape_1:output:0&lstm_12/strided_slice_1/stack:output:0(lstm_12/strided_slice_1/stack_1:output:0(lstm_12/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_12/strided_slice_1?
#lstm_12/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2%
#lstm_12/TensorArrayV2/element_shape?
lstm_12/TensorArrayV2TensorListReserve,lstm_12/TensorArrayV2/element_shape:output:0 lstm_12/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_12/TensorArrayV2?
=lstm_12/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????]   2?
=lstm_12/TensorArrayUnstack/TensorListFromTensor/element_shape?
/lstm_12/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_12/transpose:y:0Flstm_12/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type021
/lstm_12/TensorArrayUnstack/TensorListFromTensor?
lstm_12/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_12/strided_slice_2/stack?
lstm_12/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_12/strided_slice_2/stack_1?
lstm_12/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_12/strided_slice_2/stack_2?
lstm_12/strided_slice_2StridedSlicelstm_12/transpose:y:0&lstm_12/strided_slice_2/stack:output:0(lstm_12/strided_slice_2/stack_1:output:0(lstm_12/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????]*
shrink_axis_mask2
lstm_12/strided_slice_2?
*lstm_12/lstm_cell_12/MatMul/ReadVariableOpReadVariableOp3lstm_12_lstm_cell_12_matmul_readvariableop_resource*
_output_shapes
:	]?*
dtype02,
*lstm_12/lstm_cell_12/MatMul/ReadVariableOp?
lstm_12/lstm_cell_12/MatMulMatMul lstm_12/strided_slice_2:output:02lstm_12/lstm_cell_12/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_12/lstm_cell_12/MatMul?
,lstm_12/lstm_cell_12/MatMul_1/ReadVariableOpReadVariableOp5lstm_12_lstm_cell_12_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02.
,lstm_12/lstm_cell_12/MatMul_1/ReadVariableOp?
lstm_12/lstm_cell_12/MatMul_1MatMullstm_12/zeros:output:04lstm_12/lstm_cell_12/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_12/lstm_cell_12/MatMul_1?
lstm_12/lstm_cell_12/addAddV2%lstm_12/lstm_cell_12/MatMul:product:0'lstm_12/lstm_cell_12/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
lstm_12/lstm_cell_12/add?
+lstm_12/lstm_cell_12/BiasAdd/ReadVariableOpReadVariableOp4lstm_12_lstm_cell_12_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02-
+lstm_12/lstm_cell_12/BiasAdd/ReadVariableOp?
lstm_12/lstm_cell_12/BiasAddBiasAddlstm_12/lstm_cell_12/add:z:03lstm_12/lstm_cell_12/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_12/lstm_cell_12/BiasAdd?
$lstm_12/lstm_cell_12/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2&
$lstm_12/lstm_cell_12/split/split_dim?
lstm_12/lstm_cell_12/splitSplit-lstm_12/lstm_cell_12/split/split_dim:output:0%lstm_12/lstm_cell_12/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split2
lstm_12/lstm_cell_12/split?
lstm_12/lstm_cell_12/SigmoidSigmoid#lstm_12/lstm_cell_12/split:output:0*
T0*(
_output_shapes
:??????????2
lstm_12/lstm_cell_12/Sigmoid?
lstm_12/lstm_cell_12/Sigmoid_1Sigmoid#lstm_12/lstm_cell_12/split:output:1*
T0*(
_output_shapes
:??????????2 
lstm_12/lstm_cell_12/Sigmoid_1?
lstm_12/lstm_cell_12/mulMul"lstm_12/lstm_cell_12/Sigmoid_1:y:0lstm_12/zeros_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_12/lstm_cell_12/mul?
lstm_12/lstm_cell_12/ReluRelu#lstm_12/lstm_cell_12/split:output:2*
T0*(
_output_shapes
:??????????2
lstm_12/lstm_cell_12/Relu?
lstm_12/lstm_cell_12/mul_1Mul lstm_12/lstm_cell_12/Sigmoid:y:0'lstm_12/lstm_cell_12/Relu:activations:0*
T0*(
_output_shapes
:??????????2
lstm_12/lstm_cell_12/mul_1?
lstm_12/lstm_cell_12/add_1AddV2lstm_12/lstm_cell_12/mul:z:0lstm_12/lstm_cell_12/mul_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_12/lstm_cell_12/add_1?
lstm_12/lstm_cell_12/Sigmoid_2Sigmoid#lstm_12/lstm_cell_12/split:output:3*
T0*(
_output_shapes
:??????????2 
lstm_12/lstm_cell_12/Sigmoid_2?
lstm_12/lstm_cell_12/Relu_1Relulstm_12/lstm_cell_12/add_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_12/lstm_cell_12/Relu_1?
lstm_12/lstm_cell_12/mul_2Mul"lstm_12/lstm_cell_12/Sigmoid_2:y:0)lstm_12/lstm_cell_12/Relu_1:activations:0*
T0*(
_output_shapes
:??????????2
lstm_12/lstm_cell_12/mul_2?
%lstm_12/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2'
%lstm_12/TensorArrayV2_1/element_shape?
lstm_12/TensorArrayV2_1TensorListReserve.lstm_12/TensorArrayV2_1/element_shape:output:0 lstm_12/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_12/TensorArrayV2_1^
lstm_12/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_12/time?
 lstm_12/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2"
 lstm_12/while/maximum_iterationsz
lstm_12/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_12/while/loop_counter?
lstm_12/whileWhile#lstm_12/while/loop_counter:output:0)lstm_12/while/maximum_iterations:output:0lstm_12/time:output:0 lstm_12/TensorArrayV2_1:handle:0lstm_12/zeros:output:0lstm_12/zeros_1:output:0 lstm_12/strided_slice_1:output:0?lstm_12/TensorArrayUnstack/TensorListFromTensor:output_handle:03lstm_12_lstm_cell_12_matmul_readvariableop_resource5lstm_12_lstm_cell_12_matmul_1_readvariableop_resource4lstm_12_lstm_cell_12_biasadd_readvariableop_resource*
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
lstm_12_while_body_26066851*'
condR
lstm_12_while_cond_26066850*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations 2
lstm_12/while?
8lstm_12/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2:
8lstm_12/TensorArrayV2Stack/TensorListStack/element_shape?
*lstm_12/TensorArrayV2Stack/TensorListStackTensorListStacklstm_12/while:output:3Alstm_12/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:??????????*
element_dtype02,
*lstm_12/TensorArrayV2Stack/TensorListStack?
lstm_12/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
lstm_12/strided_slice_3/stack?
lstm_12/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2!
lstm_12/strided_slice_3/stack_1?
lstm_12/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_12/strided_slice_3/stack_2?
lstm_12/strided_slice_3StridedSlice3lstm_12/TensorArrayV2Stack/TensorListStack:tensor:0&lstm_12/strided_slice_3/stack:output:0(lstm_12/strided_slice_3/stack_1:output:0(lstm_12/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
lstm_12/strided_slice_3?
lstm_12/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_12/transpose_1/perm?
lstm_12/transpose_1	Transpose3lstm_12/TensorArrayV2Stack/TensorListStack:tensor:0!lstm_12/transpose_1/perm:output:0*
T0*,
_output_shapes
:??????????2
lstm_12/transpose_1v
lstm_12/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_12/runtime?
dropout_12/IdentityIdentitylstm_12/transpose_1:y:0*
T0*,
_output_shapes
:??????????2
dropout_12/Identityj
lstm_13/ShapeShapedropout_12/Identity:output:0*
T0*
_output_shapes
:2
lstm_13/Shape?
lstm_13/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_13/strided_slice/stack?
lstm_13/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_13/strided_slice/stack_1?
lstm_13/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_13/strided_slice/stack_2?
lstm_13/strided_sliceStridedSlicelstm_13/Shape:output:0$lstm_13/strided_slice/stack:output:0&lstm_13/strided_slice/stack_1:output:0&lstm_13/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_13/strided_slicem
lstm_13/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
lstm_13/zeros/mul/y?
lstm_13/zeros/mulMullstm_13/strided_slice:output:0lstm_13/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_13/zeros/mulo
lstm_13/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
lstm_13/zeros/Less/y?
lstm_13/zeros/LessLesslstm_13/zeros/mul:z:0lstm_13/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_13/zeros/Lesss
lstm_13/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
lstm_13/zeros/packed/1?
lstm_13/zeros/packedPacklstm_13/strided_slice:output:0lstm_13/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_13/zeros/packedo
lstm_13/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_13/zeros/Const?
lstm_13/zerosFilllstm_13/zeros/packed:output:0lstm_13/zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
lstm_13/zerosq
lstm_13/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2
lstm_13/zeros_1/mul/y?
lstm_13/zeros_1/mulMullstm_13/strided_slice:output:0lstm_13/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_13/zeros_1/muls
lstm_13/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
lstm_13/zeros_1/Less/y?
lstm_13/zeros_1/LessLesslstm_13/zeros_1/mul:z:0lstm_13/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_13/zeros_1/Lessw
lstm_13/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2
lstm_13/zeros_1/packed/1?
lstm_13/zeros_1/packedPacklstm_13/strided_slice:output:0!lstm_13/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_13/zeros_1/packeds
lstm_13/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_13/zeros_1/Const?
lstm_13/zeros_1Filllstm_13/zeros_1/packed:output:0lstm_13/zeros_1/Const:output:0*
T0*(
_output_shapes
:??????????2
lstm_13/zeros_1?
lstm_13/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_13/transpose/perm?
lstm_13/transpose	Transposedropout_12/Identity:output:0lstm_13/transpose/perm:output:0*
T0*,
_output_shapes
:??????????2
lstm_13/transposeg
lstm_13/Shape_1Shapelstm_13/transpose:y:0*
T0*
_output_shapes
:2
lstm_13/Shape_1?
lstm_13/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_13/strided_slice_1/stack?
lstm_13/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_13/strided_slice_1/stack_1?
lstm_13/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_13/strided_slice_1/stack_2?
lstm_13/strided_slice_1StridedSlicelstm_13/Shape_1:output:0&lstm_13/strided_slice_1/stack:output:0(lstm_13/strided_slice_1/stack_1:output:0(lstm_13/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_13/strided_slice_1?
#lstm_13/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2%
#lstm_13/TensorArrayV2/element_shape?
lstm_13/TensorArrayV2TensorListReserve,lstm_13/TensorArrayV2/element_shape:output:0 lstm_13/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_13/TensorArrayV2?
=lstm_13/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2?
=lstm_13/TensorArrayUnstack/TensorListFromTensor/element_shape?
/lstm_13/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_13/transpose:y:0Flstm_13/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type021
/lstm_13/TensorArrayUnstack/TensorListFromTensor?
lstm_13/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_13/strided_slice_2/stack?
lstm_13/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_13/strided_slice_2/stack_1?
lstm_13/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_13/strided_slice_2/stack_2?
lstm_13/strided_slice_2StridedSlicelstm_13/transpose:y:0&lstm_13/strided_slice_2/stack:output:0(lstm_13/strided_slice_2/stack_1:output:0(lstm_13/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
lstm_13/strided_slice_2?
*lstm_13/lstm_cell_13/MatMul/ReadVariableOpReadVariableOp3lstm_13_lstm_cell_13_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02,
*lstm_13/lstm_cell_13/MatMul/ReadVariableOp?
lstm_13/lstm_cell_13/MatMulMatMul lstm_13/strided_slice_2:output:02lstm_13/lstm_cell_13/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_13/lstm_cell_13/MatMul?
,lstm_13/lstm_cell_13/MatMul_1/ReadVariableOpReadVariableOp5lstm_13_lstm_cell_13_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02.
,lstm_13/lstm_cell_13/MatMul_1/ReadVariableOp?
lstm_13/lstm_cell_13/MatMul_1MatMullstm_13/zeros:output:04lstm_13/lstm_cell_13/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_13/lstm_cell_13/MatMul_1?
lstm_13/lstm_cell_13/addAddV2%lstm_13/lstm_cell_13/MatMul:product:0'lstm_13/lstm_cell_13/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
lstm_13/lstm_cell_13/add?
+lstm_13/lstm_cell_13/BiasAdd/ReadVariableOpReadVariableOp4lstm_13_lstm_cell_13_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02-
+lstm_13/lstm_cell_13/BiasAdd/ReadVariableOp?
lstm_13/lstm_cell_13/BiasAddBiasAddlstm_13/lstm_cell_13/add:z:03lstm_13/lstm_cell_13/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_13/lstm_cell_13/BiasAdd?
$lstm_13/lstm_cell_13/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2&
$lstm_13/lstm_cell_13/split/split_dim?
lstm_13/lstm_cell_13/splitSplit-lstm_13/lstm_cell_13/split/split_dim:output:0%lstm_13/lstm_cell_13/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split2
lstm_13/lstm_cell_13/split?
lstm_13/lstm_cell_13/SigmoidSigmoid#lstm_13/lstm_cell_13/split:output:0*
T0*(
_output_shapes
:??????????2
lstm_13/lstm_cell_13/Sigmoid?
lstm_13/lstm_cell_13/Sigmoid_1Sigmoid#lstm_13/lstm_cell_13/split:output:1*
T0*(
_output_shapes
:??????????2 
lstm_13/lstm_cell_13/Sigmoid_1?
lstm_13/lstm_cell_13/mulMul"lstm_13/lstm_cell_13/Sigmoid_1:y:0lstm_13/zeros_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_13/lstm_cell_13/mul?
lstm_13/lstm_cell_13/ReluRelu#lstm_13/lstm_cell_13/split:output:2*
T0*(
_output_shapes
:??????????2
lstm_13/lstm_cell_13/Relu?
lstm_13/lstm_cell_13/mul_1Mul lstm_13/lstm_cell_13/Sigmoid:y:0'lstm_13/lstm_cell_13/Relu:activations:0*
T0*(
_output_shapes
:??????????2
lstm_13/lstm_cell_13/mul_1?
lstm_13/lstm_cell_13/add_1AddV2lstm_13/lstm_cell_13/mul:z:0lstm_13/lstm_cell_13/mul_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_13/lstm_cell_13/add_1?
lstm_13/lstm_cell_13/Sigmoid_2Sigmoid#lstm_13/lstm_cell_13/split:output:3*
T0*(
_output_shapes
:??????????2 
lstm_13/lstm_cell_13/Sigmoid_2?
lstm_13/lstm_cell_13/Relu_1Relulstm_13/lstm_cell_13/add_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_13/lstm_cell_13/Relu_1?
lstm_13/lstm_cell_13/mul_2Mul"lstm_13/lstm_cell_13/Sigmoid_2:y:0)lstm_13/lstm_cell_13/Relu_1:activations:0*
T0*(
_output_shapes
:??????????2
lstm_13/lstm_cell_13/mul_2?
%lstm_13/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2'
%lstm_13/TensorArrayV2_1/element_shape?
lstm_13/TensorArrayV2_1TensorListReserve.lstm_13/TensorArrayV2_1/element_shape:output:0 lstm_13/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_13/TensorArrayV2_1^
lstm_13/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_13/time?
 lstm_13/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2"
 lstm_13/while/maximum_iterationsz
lstm_13/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_13/while/loop_counter?
lstm_13/whileWhile#lstm_13/while/loop_counter:output:0)lstm_13/while/maximum_iterations:output:0lstm_13/time:output:0 lstm_13/TensorArrayV2_1:handle:0lstm_13/zeros:output:0lstm_13/zeros_1:output:0 lstm_13/strided_slice_1:output:0?lstm_13/TensorArrayUnstack/TensorListFromTensor:output_handle:03lstm_13_lstm_cell_13_matmul_readvariableop_resource5lstm_13_lstm_cell_13_matmul_1_readvariableop_resource4lstm_13_lstm_cell_13_biasadd_readvariableop_resource*
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
lstm_13_while_body_26066999*'
condR
lstm_13_while_cond_26066998*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations 2
lstm_13/while?
8lstm_13/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2:
8lstm_13/TensorArrayV2Stack/TensorListStack/element_shape?
*lstm_13/TensorArrayV2Stack/TensorListStackTensorListStacklstm_13/while:output:3Alstm_13/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:??????????*
element_dtype02,
*lstm_13/TensorArrayV2Stack/TensorListStack?
lstm_13/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
lstm_13/strided_slice_3/stack?
lstm_13/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2!
lstm_13/strided_slice_3/stack_1?
lstm_13/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_13/strided_slice_3/stack_2?
lstm_13/strided_slice_3StridedSlice3lstm_13/TensorArrayV2Stack/TensorListStack:tensor:0&lstm_13/strided_slice_3/stack:output:0(lstm_13/strided_slice_3/stack_1:output:0(lstm_13/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
lstm_13/strided_slice_3?
lstm_13/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_13/transpose_1/perm?
lstm_13/transpose_1	Transpose3lstm_13/TensorArrayV2Stack/TensorListStack:tensor:0!lstm_13/transpose_1/perm:output:0*
T0*,
_output_shapes
:??????????2
lstm_13/transpose_1v
lstm_13/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_13/runtime?
dropout_13/IdentityIdentitylstm_13/transpose_1:y:0*
T0*,
_output_shapes
:??????????2
dropout_13/Identity?
 dense_6/Tensordot/ReadVariableOpReadVariableOp)dense_6_tensordot_readvariableop_resource*
_output_shapes
:	?*
dtype02"
 dense_6/Tensordot/ReadVariableOpz
dense_6/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_6/Tensordot/axes?
dense_6/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_6/Tensordot/free~
dense_6/Tensordot/ShapeShapedropout_13/Identity:output:0*
T0*
_output_shapes
:2
dense_6/Tensordot/Shape?
dense_6/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_6/Tensordot/GatherV2/axis?
dense_6/Tensordot/GatherV2GatherV2 dense_6/Tensordot/Shape:output:0dense_6/Tensordot/free:output:0(dense_6/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_6/Tensordot/GatherV2?
!dense_6/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_6/Tensordot/GatherV2_1/axis?
dense_6/Tensordot/GatherV2_1GatherV2 dense_6/Tensordot/Shape:output:0dense_6/Tensordot/axes:output:0*dense_6/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_6/Tensordot/GatherV2_1|
dense_6/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_6/Tensordot/Const?
dense_6/Tensordot/ProdProd#dense_6/Tensordot/GatherV2:output:0 dense_6/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_6/Tensordot/Prod?
dense_6/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_6/Tensordot/Const_1?
dense_6/Tensordot/Prod_1Prod%dense_6/Tensordot/GatherV2_1:output:0"dense_6/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_6/Tensordot/Prod_1?
dense_6/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense_6/Tensordot/concat/axis?
dense_6/Tensordot/concatConcatV2dense_6/Tensordot/free:output:0dense_6/Tensordot/axes:output:0&dense_6/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_6/Tensordot/concat?
dense_6/Tensordot/stackPackdense_6/Tensordot/Prod:output:0!dense_6/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_6/Tensordot/stack?
dense_6/Tensordot/transpose	Transposedropout_13/Identity:output:0!dense_6/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????2
dense_6/Tensordot/transpose?
dense_6/Tensordot/ReshapeReshapedense_6/Tensordot/transpose:y:0 dense_6/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense_6/Tensordot/Reshape?
dense_6/Tensordot/MatMulMatMul"dense_6/Tensordot/Reshape:output:0(dense_6/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_6/Tensordot/MatMul?
dense_6/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
dense_6/Tensordot/Const_2?
dense_6/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_6/Tensordot/concat_1/axis?
dense_6/Tensordot/concat_1ConcatV2#dense_6/Tensordot/GatherV2:output:0"dense_6/Tensordot/Const_2:output:0(dense_6/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_6/Tensordot/concat_1?
dense_6/TensordotReshape"dense_6/Tensordot/MatMul:product:0#dense_6/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????2
dense_6/Tensordot?
dense_6/BiasAdd/ReadVariableOpReadVariableOp'dense_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_6/BiasAdd/ReadVariableOp?
dense_6/BiasAddBiasAdddense_6/Tensordot:output:0&dense_6/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????2
dense_6/BiasAdd}
dense_6/SoftmaxSoftmaxdense_6/BiasAdd:output:0*
T0*+
_output_shapes
:?????????2
dense_6/Softmaxx
IdentityIdentitydense_6/Softmax:softmax:0^NoOp*
T0*+
_output_shapes
:?????????2

Identity?
NoOpNoOp^dense_6/BiasAdd/ReadVariableOp!^dense_6/Tensordot/ReadVariableOp,^lstm_12/lstm_cell_12/BiasAdd/ReadVariableOp+^lstm_12/lstm_cell_12/MatMul/ReadVariableOp-^lstm_12/lstm_cell_12/MatMul_1/ReadVariableOp^lstm_12/while,^lstm_13/lstm_cell_13/BiasAdd/ReadVariableOp+^lstm_13/lstm_cell_13/MatMul/ReadVariableOp-^lstm_13/lstm_cell_13/MatMul_1/ReadVariableOp^lstm_13/while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????]: : : : : : : : 2@
dense_6/BiasAdd/ReadVariableOpdense_6/BiasAdd/ReadVariableOp2D
 dense_6/Tensordot/ReadVariableOp dense_6/Tensordot/ReadVariableOp2Z
+lstm_12/lstm_cell_12/BiasAdd/ReadVariableOp+lstm_12/lstm_cell_12/BiasAdd/ReadVariableOp2X
*lstm_12/lstm_cell_12/MatMul/ReadVariableOp*lstm_12/lstm_cell_12/MatMul/ReadVariableOp2\
,lstm_12/lstm_cell_12/MatMul_1/ReadVariableOp,lstm_12/lstm_cell_12/MatMul_1/ReadVariableOp2
lstm_12/whilelstm_12/while2Z
+lstm_13/lstm_cell_13/BiasAdd/ReadVariableOp+lstm_13/lstm_cell_13/BiasAdd/ReadVariableOp2X
*lstm_13/lstm_cell_13/MatMul/ReadVariableOp*lstm_13/lstm_cell_13/MatMul/ReadVariableOp2\
,lstm_13/lstm_cell_13/MatMul_1/ReadVariableOp,lstm_13/lstm_cell_13/MatMul_1/ReadVariableOp2
lstm_13/whilelstm_13/while:S O
+
_output_shapes
:?????????]
 
_user_specified_nameinputs
?
?
while_cond_26066523
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_26066523___redundant_placeholder06
2while_while_cond_26066523___redundant_placeholder16
2while_while_cond_26066523___redundant_placeholder26
2while_while_cond_26066523___redundant_placeholder3
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
??
?
while_body_26065894
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_12_matmul_readvariableop_resource_0:	]?I
5while_lstm_cell_12_matmul_1_readvariableop_resource_0:
??C
4while_lstm_cell_12_biasadd_readvariableop_resource_0:	?
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_12_matmul_readvariableop_resource:	]?G
3while_lstm_cell_12_matmul_1_readvariableop_resource:
??A
2while_lstm_cell_12_biasadd_readvariableop_resource:	???)while/lstm_cell_12/BiasAdd/ReadVariableOp?(while/lstm_cell_12/MatMul/ReadVariableOp?*while/lstm_cell_12/MatMul_1/ReadVariableOp?
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
(while/lstm_cell_12/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_12_matmul_readvariableop_resource_0*
_output_shapes
:	]?*
dtype02*
(while/lstm_cell_12/MatMul/ReadVariableOp?
while/lstm_cell_12/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_12/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_12/MatMul?
*while/lstm_cell_12/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_12_matmul_1_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02,
*while/lstm_cell_12/MatMul_1/ReadVariableOp?
while/lstm_cell_12/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_12/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_12/MatMul_1?
while/lstm_cell_12/addAddV2#while/lstm_cell_12/MatMul:product:0%while/lstm_cell_12/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_12/add?
)while/lstm_cell_12/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_12_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype02+
)while/lstm_cell_12/BiasAdd/ReadVariableOp?
while/lstm_cell_12/BiasAddBiasAddwhile/lstm_cell_12/add:z:01while/lstm_cell_12/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_12/BiasAdd?
"while/lstm_cell_12/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_12/split/split_dim?
while/lstm_cell_12/splitSplit+while/lstm_cell_12/split/split_dim:output:0#while/lstm_cell_12/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split2
while/lstm_cell_12/split?
while/lstm_cell_12/SigmoidSigmoid!while/lstm_cell_12/split:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_12/Sigmoid?
while/lstm_cell_12/Sigmoid_1Sigmoid!while/lstm_cell_12/split:output:1*
T0*(
_output_shapes
:??????????2
while/lstm_cell_12/Sigmoid_1?
while/lstm_cell_12/mulMul while/lstm_cell_12/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:??????????2
while/lstm_cell_12/mul?
while/lstm_cell_12/ReluRelu!while/lstm_cell_12/split:output:2*
T0*(
_output_shapes
:??????????2
while/lstm_cell_12/Relu?
while/lstm_cell_12/mul_1Mulwhile/lstm_cell_12/Sigmoid:y:0%while/lstm_cell_12/Relu:activations:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_12/mul_1?
while/lstm_cell_12/add_1AddV2while/lstm_cell_12/mul:z:0while/lstm_cell_12/mul_1:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_12/add_1?
while/lstm_cell_12/Sigmoid_2Sigmoid!while/lstm_cell_12/split:output:3*
T0*(
_output_shapes
:??????????2
while/lstm_cell_12/Sigmoid_2?
while/lstm_cell_12/Relu_1Reluwhile/lstm_cell_12/add_1:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_12/Relu_1?
while/lstm_cell_12/mul_2Mul while/lstm_cell_12/Sigmoid_2:y:0'while/lstm_cell_12/Relu_1:activations:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_12/mul_2?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_12/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_12/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:??????????2
while/Identity_4?
while/Identity_5Identitywhile/lstm_cell_12/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:??????????2
while/Identity_5?

while/NoOpNoOp*^while/lstm_cell_12/BiasAdd/ReadVariableOp)^while/lstm_cell_12/MatMul/ReadVariableOp+^while/lstm_cell_12/MatMul_1/ReadVariableOp*"
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
2while_lstm_cell_12_biasadd_readvariableop_resource4while_lstm_cell_12_biasadd_readvariableop_resource_0"l
3while_lstm_cell_12_matmul_1_readvariableop_resource5while_lstm_cell_12_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_12_matmul_readvariableop_resource3while_lstm_cell_12_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :??????????:??????????: : : : : 2V
)while/lstm_cell_12/BiasAdd/ReadVariableOp)while/lstm_cell_12/BiasAdd/ReadVariableOp2T
(while/lstm_cell_12/MatMul/ReadVariableOp(while/lstm_cell_12/MatMul/ReadVariableOp2X
*while/lstm_cell_12/MatMul_1/ReadVariableOp*while/lstm_cell_12/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
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
*__inference_lstm_12_layer_call_fn_26068142

inputs
unknown:	]?
	unknown_0:
??
	unknown_1:	?
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
E__inference_lstm_12_layer_call_and_return_conditional_losses_260666082
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
_construction_contextkEagerRuntime*0
_input_shapes
:?????????]: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????]
 
_user_specified_nameinputs
?

?
/__inference_sequential_6_layer_call_fn_26066705
lstm_12_input
unknown:	]?
	unknown_0:
??
	unknown_1:	?
	unknown_2:
??
	unknown_3:
??
	unknown_4:	?
	unknown_5:	?
	unknown_6:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalllstm_12_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
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
J__inference_sequential_6_layer_call_and_return_conditional_losses_260666652
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
_user_specified_namelstm_12_input
?\
?
E__inference_lstm_13_layer_call_and_return_conditional_losses_26068471
inputs_0?
+lstm_cell_13_matmul_readvariableop_resource:
??A
-lstm_cell_13_matmul_1_readvariableop_resource:
??;
,lstm_cell_13_biasadd_readvariableop_resource:	?
identity??#lstm_cell_13/BiasAdd/ReadVariableOp?"lstm_cell_13/MatMul/ReadVariableOp?$lstm_cell_13/MatMul_1/ReadVariableOp?whileF
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
!:???????????????????2
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
valueB"?????   27
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
:??????????*
shrink_axis_mask2
strided_slice_2?
"lstm_cell_13/MatMul/ReadVariableOpReadVariableOp+lstm_cell_13_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02$
"lstm_cell_13/MatMul/ReadVariableOp?
lstm_cell_13/MatMulMatMulstrided_slice_2:output:0*lstm_cell_13/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_13/MatMul?
$lstm_cell_13/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_13_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02&
$lstm_cell_13/MatMul_1/ReadVariableOp?
lstm_cell_13/MatMul_1MatMulzeros:output:0,lstm_cell_13/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_13/MatMul_1?
lstm_cell_13/addAddV2lstm_cell_13/MatMul:product:0lstm_cell_13/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell_13/add?
#lstm_cell_13/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_13_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02%
#lstm_cell_13/BiasAdd/ReadVariableOp?
lstm_cell_13/BiasAddBiasAddlstm_cell_13/add:z:0+lstm_cell_13/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_13/BiasAdd~
lstm_cell_13/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_13/split/split_dim?
lstm_cell_13/splitSplit%lstm_cell_13/split/split_dim:output:0lstm_cell_13/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split2
lstm_cell_13/split?
lstm_cell_13/SigmoidSigmoidlstm_cell_13/split:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_13/Sigmoid?
lstm_cell_13/Sigmoid_1Sigmoidlstm_cell_13/split:output:1*
T0*(
_output_shapes
:??????????2
lstm_cell_13/Sigmoid_1?
lstm_cell_13/mulMullstm_cell_13/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_13/mul~
lstm_cell_13/ReluRelulstm_cell_13/split:output:2*
T0*(
_output_shapes
:??????????2
lstm_cell_13/Relu?
lstm_cell_13/mul_1Mullstm_cell_13/Sigmoid:y:0lstm_cell_13/Relu:activations:0*
T0*(
_output_shapes
:??????????2
lstm_cell_13/mul_1?
lstm_cell_13/add_1AddV2lstm_cell_13/mul:z:0lstm_cell_13/mul_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell_13/add_1?
lstm_cell_13/Sigmoid_2Sigmoidlstm_cell_13/split:output:3*
T0*(
_output_shapes
:??????????2
lstm_cell_13/Sigmoid_2}
lstm_cell_13/Relu_1Relulstm_cell_13/add_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell_13/Relu_1?
lstm_cell_13/mul_2Mullstm_cell_13/Sigmoid_2:y:0!lstm_cell_13/Relu_1:activations:0*
T0*(
_output_shapes
:??????????2
lstm_cell_13/mul_2?
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_13_matmul_readvariableop_resource-lstm_cell_13_matmul_1_readvariableop_resource,lstm_cell_13_biasadd_readvariableop_resource*
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
while_body_26068387*
condR
while_cond_26068386*M
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
NoOpNoOp$^lstm_cell_13/BiasAdd/ReadVariableOp#^lstm_cell_13/MatMul/ReadVariableOp%^lstm_cell_13/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':???????????????????: : : 2J
#lstm_cell_13/BiasAdd/ReadVariableOp#lstm_cell_13/BiasAdd/ReadVariableOp2H
"lstm_cell_13/MatMul/ReadVariableOp"lstm_cell_13/MatMul/ReadVariableOp2L
$lstm_cell_13/MatMul_1/ReadVariableOp$lstm_cell_13/MatMul_1/ReadVariableOp2
whilewhile:_ [
5
_output_shapes#
!:???????????????????
"
_user_specified_name
inputs/0
?
?
*__inference_lstm_13_layer_call_fn_26068806

inputs
unknown:
??
	unknown_0:
??
	unknown_1:	?
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
E__inference_lstm_13_layer_call_and_return_conditional_losses_260661432
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
:??????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
while_cond_26066058
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_26066058___redundant_placeholder06
2while_while_cond_26066058___redundant_placeholder16
2while_while_cond_26066058___redundant_placeholder26
2while_while_cond_26066058___redundant_placeholder3
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
?
f
-__inference_dropout_12_layer_call_fn_26068169

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
H__inference_dropout_12_layer_call_and_return_conditional_losses_260664412
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
?&
?
while_body_26065490
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_01
while_lstm_cell_13_26065514_0:
??1
while_lstm_cell_13_26065516_0:
??,
while_lstm_cell_13_26065518_0:	?
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor/
while_lstm_cell_13_26065514:
??/
while_lstm_cell_13_26065516:
??*
while_lstm_cell_13_26065518:	???*while/lstm_cell_13/StatefulPartitionedCall?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:??????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
*while/lstm_cell_13/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_13_26065514_0while_lstm_cell_13_26065516_0while_lstm_cell_13_26065518_0*
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
J__inference_lstm_cell_13_layer_call_and_return_conditional_losses_260654122,
*while/lstm_cell_13/StatefulPartitionedCall?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder3while/lstm_cell_13/StatefulPartitionedCall:output:0*
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
while/Identity_4Identity3while/lstm_cell_13/StatefulPartitionedCall:output:1^while/NoOp*
T0*(
_output_shapes
:??????????2
while/Identity_4?
while/Identity_5Identity3while/lstm_cell_13/StatefulPartitionedCall:output:2^while/NoOp*
T0*(
_output_shapes
:??????????2
while/Identity_5?

while/NoOpNoOp+^while/lstm_cell_13/StatefulPartitionedCall*"
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
while_lstm_cell_13_26065514while_lstm_cell_13_26065514_0"<
while_lstm_cell_13_26065516while_lstm_cell_13_26065516_0"<
while_lstm_cell_13_26065518while_lstm_cell_13_26065518_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :??????????:??????????: : : : : 2X
*while/lstm_cell_13/StatefulPartitionedCall*while/lstm_cell_13/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
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
?
(sequential_6_lstm_12_while_cond_26064300F
Bsequential_6_lstm_12_while_sequential_6_lstm_12_while_loop_counterL
Hsequential_6_lstm_12_while_sequential_6_lstm_12_while_maximum_iterations*
&sequential_6_lstm_12_while_placeholder,
(sequential_6_lstm_12_while_placeholder_1,
(sequential_6_lstm_12_while_placeholder_2,
(sequential_6_lstm_12_while_placeholder_3H
Dsequential_6_lstm_12_while_less_sequential_6_lstm_12_strided_slice_1`
\sequential_6_lstm_12_while_sequential_6_lstm_12_while_cond_26064300___redundant_placeholder0`
\sequential_6_lstm_12_while_sequential_6_lstm_12_while_cond_26064300___redundant_placeholder1`
\sequential_6_lstm_12_while_sequential_6_lstm_12_while_cond_26064300___redundant_placeholder2`
\sequential_6_lstm_12_while_sequential_6_lstm_12_while_cond_26064300___redundant_placeholder3'
#sequential_6_lstm_12_while_identity
?
sequential_6/lstm_12/while/LessLess&sequential_6_lstm_12_while_placeholderDsequential_6_lstm_12_while_less_sequential_6_lstm_12_strided_slice_1*
T0*
_output_shapes
: 2!
sequential_6/lstm_12/while/Less?
#sequential_6/lstm_12/while/IdentityIdentity#sequential_6/lstm_12/while/Less:z:0*
T0
*
_output_shapes
: 2%
#sequential_6/lstm_12/while/Identity"S
#sequential_6_lstm_12_while_identity,sequential_6/lstm_12/while/Identity:output:0*(
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
?J
?

lstm_13_while_body_26066999,
(lstm_13_while_lstm_13_while_loop_counter2
.lstm_13_while_lstm_13_while_maximum_iterations
lstm_13_while_placeholder
lstm_13_while_placeholder_1
lstm_13_while_placeholder_2
lstm_13_while_placeholder_3+
'lstm_13_while_lstm_13_strided_slice_1_0g
clstm_13_while_tensorarrayv2read_tensorlistgetitem_lstm_13_tensorarrayunstack_tensorlistfromtensor_0O
;lstm_13_while_lstm_cell_13_matmul_readvariableop_resource_0:
??Q
=lstm_13_while_lstm_cell_13_matmul_1_readvariableop_resource_0:
??K
<lstm_13_while_lstm_cell_13_biasadd_readvariableop_resource_0:	?
lstm_13_while_identity
lstm_13_while_identity_1
lstm_13_while_identity_2
lstm_13_while_identity_3
lstm_13_while_identity_4
lstm_13_while_identity_5)
%lstm_13_while_lstm_13_strided_slice_1e
alstm_13_while_tensorarrayv2read_tensorlistgetitem_lstm_13_tensorarrayunstack_tensorlistfromtensorM
9lstm_13_while_lstm_cell_13_matmul_readvariableop_resource:
??O
;lstm_13_while_lstm_cell_13_matmul_1_readvariableop_resource:
??I
:lstm_13_while_lstm_cell_13_biasadd_readvariableop_resource:	???1lstm_13/while/lstm_cell_13/BiasAdd/ReadVariableOp?0lstm_13/while/lstm_cell_13/MatMul/ReadVariableOp?2lstm_13/while/lstm_cell_13/MatMul_1/ReadVariableOp?
?lstm_13/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2A
?lstm_13/while/TensorArrayV2Read/TensorListGetItem/element_shape?
1lstm_13/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemclstm_13_while_tensorarrayv2read_tensorlistgetitem_lstm_13_tensorarrayunstack_tensorlistfromtensor_0lstm_13_while_placeholderHlstm_13/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:??????????*
element_dtype023
1lstm_13/while/TensorArrayV2Read/TensorListGetItem?
0lstm_13/while/lstm_cell_13/MatMul/ReadVariableOpReadVariableOp;lstm_13_while_lstm_cell_13_matmul_readvariableop_resource_0* 
_output_shapes
:
??*
dtype022
0lstm_13/while/lstm_cell_13/MatMul/ReadVariableOp?
!lstm_13/while/lstm_cell_13/MatMulMatMul8lstm_13/while/TensorArrayV2Read/TensorListGetItem:item:08lstm_13/while/lstm_cell_13/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2#
!lstm_13/while/lstm_cell_13/MatMul?
2lstm_13/while/lstm_cell_13/MatMul_1/ReadVariableOpReadVariableOp=lstm_13_while_lstm_cell_13_matmul_1_readvariableop_resource_0* 
_output_shapes
:
??*
dtype024
2lstm_13/while/lstm_cell_13/MatMul_1/ReadVariableOp?
#lstm_13/while/lstm_cell_13/MatMul_1MatMullstm_13_while_placeholder_2:lstm_13/while/lstm_cell_13/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2%
#lstm_13/while/lstm_cell_13/MatMul_1?
lstm_13/while/lstm_cell_13/addAddV2+lstm_13/while/lstm_cell_13/MatMul:product:0-lstm_13/while/lstm_cell_13/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2 
lstm_13/while/lstm_cell_13/add?
1lstm_13/while/lstm_cell_13/BiasAdd/ReadVariableOpReadVariableOp<lstm_13_while_lstm_cell_13_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype023
1lstm_13/while/lstm_cell_13/BiasAdd/ReadVariableOp?
"lstm_13/while/lstm_cell_13/BiasAddBiasAdd"lstm_13/while/lstm_cell_13/add:z:09lstm_13/while/lstm_cell_13/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2$
"lstm_13/while/lstm_cell_13/BiasAdd?
*lstm_13/while/lstm_cell_13/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2,
*lstm_13/while/lstm_cell_13/split/split_dim?
 lstm_13/while/lstm_cell_13/splitSplit3lstm_13/while/lstm_cell_13/split/split_dim:output:0+lstm_13/while/lstm_cell_13/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split2"
 lstm_13/while/lstm_cell_13/split?
"lstm_13/while/lstm_cell_13/SigmoidSigmoid)lstm_13/while/lstm_cell_13/split:output:0*
T0*(
_output_shapes
:??????????2$
"lstm_13/while/lstm_cell_13/Sigmoid?
$lstm_13/while/lstm_cell_13/Sigmoid_1Sigmoid)lstm_13/while/lstm_cell_13/split:output:1*
T0*(
_output_shapes
:??????????2&
$lstm_13/while/lstm_cell_13/Sigmoid_1?
lstm_13/while/lstm_cell_13/mulMul(lstm_13/while/lstm_cell_13/Sigmoid_1:y:0lstm_13_while_placeholder_3*
T0*(
_output_shapes
:??????????2 
lstm_13/while/lstm_cell_13/mul?
lstm_13/while/lstm_cell_13/ReluRelu)lstm_13/while/lstm_cell_13/split:output:2*
T0*(
_output_shapes
:??????????2!
lstm_13/while/lstm_cell_13/Relu?
 lstm_13/while/lstm_cell_13/mul_1Mul&lstm_13/while/lstm_cell_13/Sigmoid:y:0-lstm_13/while/lstm_cell_13/Relu:activations:0*
T0*(
_output_shapes
:??????????2"
 lstm_13/while/lstm_cell_13/mul_1?
 lstm_13/while/lstm_cell_13/add_1AddV2"lstm_13/while/lstm_cell_13/mul:z:0$lstm_13/while/lstm_cell_13/mul_1:z:0*
T0*(
_output_shapes
:??????????2"
 lstm_13/while/lstm_cell_13/add_1?
$lstm_13/while/lstm_cell_13/Sigmoid_2Sigmoid)lstm_13/while/lstm_cell_13/split:output:3*
T0*(
_output_shapes
:??????????2&
$lstm_13/while/lstm_cell_13/Sigmoid_2?
!lstm_13/while/lstm_cell_13/Relu_1Relu$lstm_13/while/lstm_cell_13/add_1:z:0*
T0*(
_output_shapes
:??????????2#
!lstm_13/while/lstm_cell_13/Relu_1?
 lstm_13/while/lstm_cell_13/mul_2Mul(lstm_13/while/lstm_cell_13/Sigmoid_2:y:0/lstm_13/while/lstm_cell_13/Relu_1:activations:0*
T0*(
_output_shapes
:??????????2"
 lstm_13/while/lstm_cell_13/mul_2?
2lstm_13/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_13_while_placeholder_1lstm_13_while_placeholder$lstm_13/while/lstm_cell_13/mul_2:z:0*
_output_shapes
: *
element_dtype024
2lstm_13/while/TensorArrayV2Write/TensorListSetIteml
lstm_13/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_13/while/add/y?
lstm_13/while/addAddV2lstm_13_while_placeholderlstm_13/while/add/y:output:0*
T0*
_output_shapes
: 2
lstm_13/while/addp
lstm_13/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_13/while/add_1/y?
lstm_13/while/add_1AddV2(lstm_13_while_lstm_13_while_loop_counterlstm_13/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lstm_13/while/add_1?
lstm_13/while/IdentityIdentitylstm_13/while/add_1:z:0^lstm_13/while/NoOp*
T0*
_output_shapes
: 2
lstm_13/while/Identity?
lstm_13/while/Identity_1Identity.lstm_13_while_lstm_13_while_maximum_iterations^lstm_13/while/NoOp*
T0*
_output_shapes
: 2
lstm_13/while/Identity_1?
lstm_13/while/Identity_2Identitylstm_13/while/add:z:0^lstm_13/while/NoOp*
T0*
_output_shapes
: 2
lstm_13/while/Identity_2?
lstm_13/while/Identity_3IdentityBlstm_13/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_13/while/NoOp*
T0*
_output_shapes
: 2
lstm_13/while/Identity_3?
lstm_13/while/Identity_4Identity$lstm_13/while/lstm_cell_13/mul_2:z:0^lstm_13/while/NoOp*
T0*(
_output_shapes
:??????????2
lstm_13/while/Identity_4?
lstm_13/while/Identity_5Identity$lstm_13/while/lstm_cell_13/add_1:z:0^lstm_13/while/NoOp*
T0*(
_output_shapes
:??????????2
lstm_13/while/Identity_5?
lstm_13/while/NoOpNoOp2^lstm_13/while/lstm_cell_13/BiasAdd/ReadVariableOp1^lstm_13/while/lstm_cell_13/MatMul/ReadVariableOp3^lstm_13/while/lstm_cell_13/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
lstm_13/while/NoOp"9
lstm_13_while_identitylstm_13/while/Identity:output:0"=
lstm_13_while_identity_1!lstm_13/while/Identity_1:output:0"=
lstm_13_while_identity_2!lstm_13/while/Identity_2:output:0"=
lstm_13_while_identity_3!lstm_13/while/Identity_3:output:0"=
lstm_13_while_identity_4!lstm_13/while/Identity_4:output:0"=
lstm_13_while_identity_5!lstm_13/while/Identity_5:output:0"P
%lstm_13_while_lstm_13_strided_slice_1'lstm_13_while_lstm_13_strided_slice_1_0"z
:lstm_13_while_lstm_cell_13_biasadd_readvariableop_resource<lstm_13_while_lstm_cell_13_biasadd_readvariableop_resource_0"|
;lstm_13_while_lstm_cell_13_matmul_1_readvariableop_resource=lstm_13_while_lstm_cell_13_matmul_1_readvariableop_resource_0"x
9lstm_13_while_lstm_cell_13_matmul_readvariableop_resource;lstm_13_while_lstm_cell_13_matmul_readvariableop_resource_0"?
alstm_13_while_tensorarrayv2read_tensorlistgetitem_lstm_13_tensorarrayunstack_tensorlistfromtensorclstm_13_while_tensorarrayv2read_tensorlistgetitem_lstm_13_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :??????????:??????????: : : : : 2f
1lstm_13/while/lstm_cell_13/BiasAdd/ReadVariableOp1lstm_13/while/lstm_cell_13/BiasAdd/ReadVariableOp2d
0lstm_13/while/lstm_cell_13/MatMul/ReadVariableOp0lstm_13/while/lstm_cell_13/MatMul/ReadVariableOp2h
2lstm_13/while/lstm_cell_13/MatMul_1/ReadVariableOp2lstm_13/while/lstm_cell_13/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
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

?
/__inference_sequential_6_layer_call_fn_26066215
lstm_12_input
unknown:	]?
	unknown_0:
??
	unknown_1:	?
	unknown_2:
??
	unknown_3:
??
	unknown_4:	?
	unknown_5:	?
	unknown_6:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalllstm_12_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
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
J__inference_sequential_6_layer_call_and_return_conditional_losses_260661962
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
_user_specified_namelstm_12_input
?\
?
E__inference_lstm_12_layer_call_and_return_conditional_losses_26065978

inputs>
+lstm_cell_12_matmul_readvariableop_resource:	]?A
-lstm_cell_12_matmul_1_readvariableop_resource:
??;
,lstm_cell_12_biasadd_readvariableop_resource:	?
identity??#lstm_cell_12/BiasAdd/ReadVariableOp?"lstm_cell_12/MatMul/ReadVariableOp?$lstm_cell_12/MatMul_1/ReadVariableOp?whileD
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
"lstm_cell_12/MatMul/ReadVariableOpReadVariableOp+lstm_cell_12_matmul_readvariableop_resource*
_output_shapes
:	]?*
dtype02$
"lstm_cell_12/MatMul/ReadVariableOp?
lstm_cell_12/MatMulMatMulstrided_slice_2:output:0*lstm_cell_12/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_12/MatMul?
$lstm_cell_12/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_12_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02&
$lstm_cell_12/MatMul_1/ReadVariableOp?
lstm_cell_12/MatMul_1MatMulzeros:output:0,lstm_cell_12/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_12/MatMul_1?
lstm_cell_12/addAddV2lstm_cell_12/MatMul:product:0lstm_cell_12/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell_12/add?
#lstm_cell_12/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_12_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02%
#lstm_cell_12/BiasAdd/ReadVariableOp?
lstm_cell_12/BiasAddBiasAddlstm_cell_12/add:z:0+lstm_cell_12/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_12/BiasAdd~
lstm_cell_12/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_12/split/split_dim?
lstm_cell_12/splitSplit%lstm_cell_12/split/split_dim:output:0lstm_cell_12/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split2
lstm_cell_12/split?
lstm_cell_12/SigmoidSigmoidlstm_cell_12/split:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_12/Sigmoid?
lstm_cell_12/Sigmoid_1Sigmoidlstm_cell_12/split:output:1*
T0*(
_output_shapes
:??????????2
lstm_cell_12/Sigmoid_1?
lstm_cell_12/mulMullstm_cell_12/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_12/mul~
lstm_cell_12/ReluRelulstm_cell_12/split:output:2*
T0*(
_output_shapes
:??????????2
lstm_cell_12/Relu?
lstm_cell_12/mul_1Mullstm_cell_12/Sigmoid:y:0lstm_cell_12/Relu:activations:0*
T0*(
_output_shapes
:??????????2
lstm_cell_12/mul_1?
lstm_cell_12/add_1AddV2lstm_cell_12/mul:z:0lstm_cell_12/mul_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell_12/add_1?
lstm_cell_12/Sigmoid_2Sigmoidlstm_cell_12/split:output:3*
T0*(
_output_shapes
:??????????2
lstm_cell_12/Sigmoid_2}
lstm_cell_12/Relu_1Relulstm_cell_12/add_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell_12/Relu_1?
lstm_cell_12/mul_2Mullstm_cell_12/Sigmoid_2:y:0!lstm_cell_12/Relu_1:activations:0*
T0*(
_output_shapes
:??????????2
lstm_cell_12/mul_2?
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_12_matmul_readvariableop_resource-lstm_cell_12_matmul_1_readvariableop_resource,lstm_cell_12_biasadd_readvariableop_resource*
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
while_body_26065894*
condR
while_cond_26065893*M
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
NoOpNoOp$^lstm_cell_12/BiasAdd/ReadVariableOp#^lstm_cell_12/MatMul/ReadVariableOp%^lstm_cell_12/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????]: : : 2J
#lstm_cell_12/BiasAdd/ReadVariableOp#lstm_cell_12/BiasAdd/ReadVariableOp2H
"lstm_cell_12/MatMul/ReadVariableOp"lstm_cell_12/MatMul/ReadVariableOp2L
$lstm_cell_12/MatMul_1/ReadVariableOp$lstm_cell_12/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:?????????]
 
_user_specified_nameinputs
?J
?

lstm_12_while_body_26067178,
(lstm_12_while_lstm_12_while_loop_counter2
.lstm_12_while_lstm_12_while_maximum_iterations
lstm_12_while_placeholder
lstm_12_while_placeholder_1
lstm_12_while_placeholder_2
lstm_12_while_placeholder_3+
'lstm_12_while_lstm_12_strided_slice_1_0g
clstm_12_while_tensorarrayv2read_tensorlistgetitem_lstm_12_tensorarrayunstack_tensorlistfromtensor_0N
;lstm_12_while_lstm_cell_12_matmul_readvariableop_resource_0:	]?Q
=lstm_12_while_lstm_cell_12_matmul_1_readvariableop_resource_0:
??K
<lstm_12_while_lstm_cell_12_biasadd_readvariableop_resource_0:	?
lstm_12_while_identity
lstm_12_while_identity_1
lstm_12_while_identity_2
lstm_12_while_identity_3
lstm_12_while_identity_4
lstm_12_while_identity_5)
%lstm_12_while_lstm_12_strided_slice_1e
alstm_12_while_tensorarrayv2read_tensorlistgetitem_lstm_12_tensorarrayunstack_tensorlistfromtensorL
9lstm_12_while_lstm_cell_12_matmul_readvariableop_resource:	]?O
;lstm_12_while_lstm_cell_12_matmul_1_readvariableop_resource:
??I
:lstm_12_while_lstm_cell_12_biasadd_readvariableop_resource:	???1lstm_12/while/lstm_cell_12/BiasAdd/ReadVariableOp?0lstm_12/while/lstm_cell_12/MatMul/ReadVariableOp?2lstm_12/while/lstm_cell_12/MatMul_1/ReadVariableOp?
?lstm_12/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????]   2A
?lstm_12/while/TensorArrayV2Read/TensorListGetItem/element_shape?
1lstm_12/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemclstm_12_while_tensorarrayv2read_tensorlistgetitem_lstm_12_tensorarrayunstack_tensorlistfromtensor_0lstm_12_while_placeholderHlstm_12/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????]*
element_dtype023
1lstm_12/while/TensorArrayV2Read/TensorListGetItem?
0lstm_12/while/lstm_cell_12/MatMul/ReadVariableOpReadVariableOp;lstm_12_while_lstm_cell_12_matmul_readvariableop_resource_0*
_output_shapes
:	]?*
dtype022
0lstm_12/while/lstm_cell_12/MatMul/ReadVariableOp?
!lstm_12/while/lstm_cell_12/MatMulMatMul8lstm_12/while/TensorArrayV2Read/TensorListGetItem:item:08lstm_12/while/lstm_cell_12/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2#
!lstm_12/while/lstm_cell_12/MatMul?
2lstm_12/while/lstm_cell_12/MatMul_1/ReadVariableOpReadVariableOp=lstm_12_while_lstm_cell_12_matmul_1_readvariableop_resource_0* 
_output_shapes
:
??*
dtype024
2lstm_12/while/lstm_cell_12/MatMul_1/ReadVariableOp?
#lstm_12/while/lstm_cell_12/MatMul_1MatMullstm_12_while_placeholder_2:lstm_12/while/lstm_cell_12/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2%
#lstm_12/while/lstm_cell_12/MatMul_1?
lstm_12/while/lstm_cell_12/addAddV2+lstm_12/while/lstm_cell_12/MatMul:product:0-lstm_12/while/lstm_cell_12/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2 
lstm_12/while/lstm_cell_12/add?
1lstm_12/while/lstm_cell_12/BiasAdd/ReadVariableOpReadVariableOp<lstm_12_while_lstm_cell_12_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype023
1lstm_12/while/lstm_cell_12/BiasAdd/ReadVariableOp?
"lstm_12/while/lstm_cell_12/BiasAddBiasAdd"lstm_12/while/lstm_cell_12/add:z:09lstm_12/while/lstm_cell_12/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2$
"lstm_12/while/lstm_cell_12/BiasAdd?
*lstm_12/while/lstm_cell_12/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2,
*lstm_12/while/lstm_cell_12/split/split_dim?
 lstm_12/while/lstm_cell_12/splitSplit3lstm_12/while/lstm_cell_12/split/split_dim:output:0+lstm_12/while/lstm_cell_12/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split2"
 lstm_12/while/lstm_cell_12/split?
"lstm_12/while/lstm_cell_12/SigmoidSigmoid)lstm_12/while/lstm_cell_12/split:output:0*
T0*(
_output_shapes
:??????????2$
"lstm_12/while/lstm_cell_12/Sigmoid?
$lstm_12/while/lstm_cell_12/Sigmoid_1Sigmoid)lstm_12/while/lstm_cell_12/split:output:1*
T0*(
_output_shapes
:??????????2&
$lstm_12/while/lstm_cell_12/Sigmoid_1?
lstm_12/while/lstm_cell_12/mulMul(lstm_12/while/lstm_cell_12/Sigmoid_1:y:0lstm_12_while_placeholder_3*
T0*(
_output_shapes
:??????????2 
lstm_12/while/lstm_cell_12/mul?
lstm_12/while/lstm_cell_12/ReluRelu)lstm_12/while/lstm_cell_12/split:output:2*
T0*(
_output_shapes
:??????????2!
lstm_12/while/lstm_cell_12/Relu?
 lstm_12/while/lstm_cell_12/mul_1Mul&lstm_12/while/lstm_cell_12/Sigmoid:y:0-lstm_12/while/lstm_cell_12/Relu:activations:0*
T0*(
_output_shapes
:??????????2"
 lstm_12/while/lstm_cell_12/mul_1?
 lstm_12/while/lstm_cell_12/add_1AddV2"lstm_12/while/lstm_cell_12/mul:z:0$lstm_12/while/lstm_cell_12/mul_1:z:0*
T0*(
_output_shapes
:??????????2"
 lstm_12/while/lstm_cell_12/add_1?
$lstm_12/while/lstm_cell_12/Sigmoid_2Sigmoid)lstm_12/while/lstm_cell_12/split:output:3*
T0*(
_output_shapes
:??????????2&
$lstm_12/while/lstm_cell_12/Sigmoid_2?
!lstm_12/while/lstm_cell_12/Relu_1Relu$lstm_12/while/lstm_cell_12/add_1:z:0*
T0*(
_output_shapes
:??????????2#
!lstm_12/while/lstm_cell_12/Relu_1?
 lstm_12/while/lstm_cell_12/mul_2Mul(lstm_12/while/lstm_cell_12/Sigmoid_2:y:0/lstm_12/while/lstm_cell_12/Relu_1:activations:0*
T0*(
_output_shapes
:??????????2"
 lstm_12/while/lstm_cell_12/mul_2?
2lstm_12/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_12_while_placeholder_1lstm_12_while_placeholder$lstm_12/while/lstm_cell_12/mul_2:z:0*
_output_shapes
: *
element_dtype024
2lstm_12/while/TensorArrayV2Write/TensorListSetIteml
lstm_12/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_12/while/add/y?
lstm_12/while/addAddV2lstm_12_while_placeholderlstm_12/while/add/y:output:0*
T0*
_output_shapes
: 2
lstm_12/while/addp
lstm_12/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_12/while/add_1/y?
lstm_12/while/add_1AddV2(lstm_12_while_lstm_12_while_loop_counterlstm_12/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lstm_12/while/add_1?
lstm_12/while/IdentityIdentitylstm_12/while/add_1:z:0^lstm_12/while/NoOp*
T0*
_output_shapes
: 2
lstm_12/while/Identity?
lstm_12/while/Identity_1Identity.lstm_12_while_lstm_12_while_maximum_iterations^lstm_12/while/NoOp*
T0*
_output_shapes
: 2
lstm_12/while/Identity_1?
lstm_12/while/Identity_2Identitylstm_12/while/add:z:0^lstm_12/while/NoOp*
T0*
_output_shapes
: 2
lstm_12/while/Identity_2?
lstm_12/while/Identity_3IdentityBlstm_12/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_12/while/NoOp*
T0*
_output_shapes
: 2
lstm_12/while/Identity_3?
lstm_12/while/Identity_4Identity$lstm_12/while/lstm_cell_12/mul_2:z:0^lstm_12/while/NoOp*
T0*(
_output_shapes
:??????????2
lstm_12/while/Identity_4?
lstm_12/while/Identity_5Identity$lstm_12/while/lstm_cell_12/add_1:z:0^lstm_12/while/NoOp*
T0*(
_output_shapes
:??????????2
lstm_12/while/Identity_5?
lstm_12/while/NoOpNoOp2^lstm_12/while/lstm_cell_12/BiasAdd/ReadVariableOp1^lstm_12/while/lstm_cell_12/MatMul/ReadVariableOp3^lstm_12/while/lstm_cell_12/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
lstm_12/while/NoOp"9
lstm_12_while_identitylstm_12/while/Identity:output:0"=
lstm_12_while_identity_1!lstm_12/while/Identity_1:output:0"=
lstm_12_while_identity_2!lstm_12/while/Identity_2:output:0"=
lstm_12_while_identity_3!lstm_12/while/Identity_3:output:0"=
lstm_12_while_identity_4!lstm_12/while/Identity_4:output:0"=
lstm_12_while_identity_5!lstm_12/while/Identity_5:output:0"P
%lstm_12_while_lstm_12_strided_slice_1'lstm_12_while_lstm_12_strided_slice_1_0"z
:lstm_12_while_lstm_cell_12_biasadd_readvariableop_resource<lstm_12_while_lstm_cell_12_biasadd_readvariableop_resource_0"|
;lstm_12_while_lstm_cell_12_matmul_1_readvariableop_resource=lstm_12_while_lstm_cell_12_matmul_1_readvariableop_resource_0"x
9lstm_12_while_lstm_cell_12_matmul_readvariableop_resource;lstm_12_while_lstm_cell_12_matmul_readvariableop_resource_0"?
alstm_12_while_tensorarrayv2read_tensorlistgetitem_lstm_12_tensorarrayunstack_tensorlistfromtensorclstm_12_while_tensorarrayv2read_tensorlistgetitem_lstm_12_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :??????????:??????????: : : : : 2f
1lstm_12/while/lstm_cell_12/BiasAdd/ReadVariableOp1lstm_12/while/lstm_cell_12/BiasAdd/ReadVariableOp2d
0lstm_12/while/lstm_cell_12/MatMul/ReadVariableOp0lstm_12/while/lstm_cell_12/MatMul/ReadVariableOp2h
2lstm_12/while/lstm_cell_12/MatMul_1/ReadVariableOp2lstm_12/while/lstm_cell_12/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
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
?
f
-__inference_dropout_13_layer_call_fn_26068844

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
H__inference_dropout_13_layer_call_and_return_conditional_losses_260662452
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
?J
?

lstm_12_while_body_26066851,
(lstm_12_while_lstm_12_while_loop_counter2
.lstm_12_while_lstm_12_while_maximum_iterations
lstm_12_while_placeholder
lstm_12_while_placeholder_1
lstm_12_while_placeholder_2
lstm_12_while_placeholder_3+
'lstm_12_while_lstm_12_strided_slice_1_0g
clstm_12_while_tensorarrayv2read_tensorlistgetitem_lstm_12_tensorarrayunstack_tensorlistfromtensor_0N
;lstm_12_while_lstm_cell_12_matmul_readvariableop_resource_0:	]?Q
=lstm_12_while_lstm_cell_12_matmul_1_readvariableop_resource_0:
??K
<lstm_12_while_lstm_cell_12_biasadd_readvariableop_resource_0:	?
lstm_12_while_identity
lstm_12_while_identity_1
lstm_12_while_identity_2
lstm_12_while_identity_3
lstm_12_while_identity_4
lstm_12_while_identity_5)
%lstm_12_while_lstm_12_strided_slice_1e
alstm_12_while_tensorarrayv2read_tensorlistgetitem_lstm_12_tensorarrayunstack_tensorlistfromtensorL
9lstm_12_while_lstm_cell_12_matmul_readvariableop_resource:	]?O
;lstm_12_while_lstm_cell_12_matmul_1_readvariableop_resource:
??I
:lstm_12_while_lstm_cell_12_biasadd_readvariableop_resource:	???1lstm_12/while/lstm_cell_12/BiasAdd/ReadVariableOp?0lstm_12/while/lstm_cell_12/MatMul/ReadVariableOp?2lstm_12/while/lstm_cell_12/MatMul_1/ReadVariableOp?
?lstm_12/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????]   2A
?lstm_12/while/TensorArrayV2Read/TensorListGetItem/element_shape?
1lstm_12/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemclstm_12_while_tensorarrayv2read_tensorlistgetitem_lstm_12_tensorarrayunstack_tensorlistfromtensor_0lstm_12_while_placeholderHlstm_12/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????]*
element_dtype023
1lstm_12/while/TensorArrayV2Read/TensorListGetItem?
0lstm_12/while/lstm_cell_12/MatMul/ReadVariableOpReadVariableOp;lstm_12_while_lstm_cell_12_matmul_readvariableop_resource_0*
_output_shapes
:	]?*
dtype022
0lstm_12/while/lstm_cell_12/MatMul/ReadVariableOp?
!lstm_12/while/lstm_cell_12/MatMulMatMul8lstm_12/while/TensorArrayV2Read/TensorListGetItem:item:08lstm_12/while/lstm_cell_12/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2#
!lstm_12/while/lstm_cell_12/MatMul?
2lstm_12/while/lstm_cell_12/MatMul_1/ReadVariableOpReadVariableOp=lstm_12_while_lstm_cell_12_matmul_1_readvariableop_resource_0* 
_output_shapes
:
??*
dtype024
2lstm_12/while/lstm_cell_12/MatMul_1/ReadVariableOp?
#lstm_12/while/lstm_cell_12/MatMul_1MatMullstm_12_while_placeholder_2:lstm_12/while/lstm_cell_12/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2%
#lstm_12/while/lstm_cell_12/MatMul_1?
lstm_12/while/lstm_cell_12/addAddV2+lstm_12/while/lstm_cell_12/MatMul:product:0-lstm_12/while/lstm_cell_12/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2 
lstm_12/while/lstm_cell_12/add?
1lstm_12/while/lstm_cell_12/BiasAdd/ReadVariableOpReadVariableOp<lstm_12_while_lstm_cell_12_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype023
1lstm_12/while/lstm_cell_12/BiasAdd/ReadVariableOp?
"lstm_12/while/lstm_cell_12/BiasAddBiasAdd"lstm_12/while/lstm_cell_12/add:z:09lstm_12/while/lstm_cell_12/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2$
"lstm_12/while/lstm_cell_12/BiasAdd?
*lstm_12/while/lstm_cell_12/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2,
*lstm_12/while/lstm_cell_12/split/split_dim?
 lstm_12/while/lstm_cell_12/splitSplit3lstm_12/while/lstm_cell_12/split/split_dim:output:0+lstm_12/while/lstm_cell_12/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split2"
 lstm_12/while/lstm_cell_12/split?
"lstm_12/while/lstm_cell_12/SigmoidSigmoid)lstm_12/while/lstm_cell_12/split:output:0*
T0*(
_output_shapes
:??????????2$
"lstm_12/while/lstm_cell_12/Sigmoid?
$lstm_12/while/lstm_cell_12/Sigmoid_1Sigmoid)lstm_12/while/lstm_cell_12/split:output:1*
T0*(
_output_shapes
:??????????2&
$lstm_12/while/lstm_cell_12/Sigmoid_1?
lstm_12/while/lstm_cell_12/mulMul(lstm_12/while/lstm_cell_12/Sigmoid_1:y:0lstm_12_while_placeholder_3*
T0*(
_output_shapes
:??????????2 
lstm_12/while/lstm_cell_12/mul?
lstm_12/while/lstm_cell_12/ReluRelu)lstm_12/while/lstm_cell_12/split:output:2*
T0*(
_output_shapes
:??????????2!
lstm_12/while/lstm_cell_12/Relu?
 lstm_12/while/lstm_cell_12/mul_1Mul&lstm_12/while/lstm_cell_12/Sigmoid:y:0-lstm_12/while/lstm_cell_12/Relu:activations:0*
T0*(
_output_shapes
:??????????2"
 lstm_12/while/lstm_cell_12/mul_1?
 lstm_12/while/lstm_cell_12/add_1AddV2"lstm_12/while/lstm_cell_12/mul:z:0$lstm_12/while/lstm_cell_12/mul_1:z:0*
T0*(
_output_shapes
:??????????2"
 lstm_12/while/lstm_cell_12/add_1?
$lstm_12/while/lstm_cell_12/Sigmoid_2Sigmoid)lstm_12/while/lstm_cell_12/split:output:3*
T0*(
_output_shapes
:??????????2&
$lstm_12/while/lstm_cell_12/Sigmoid_2?
!lstm_12/while/lstm_cell_12/Relu_1Relu$lstm_12/while/lstm_cell_12/add_1:z:0*
T0*(
_output_shapes
:??????????2#
!lstm_12/while/lstm_cell_12/Relu_1?
 lstm_12/while/lstm_cell_12/mul_2Mul(lstm_12/while/lstm_cell_12/Sigmoid_2:y:0/lstm_12/while/lstm_cell_12/Relu_1:activations:0*
T0*(
_output_shapes
:??????????2"
 lstm_12/while/lstm_cell_12/mul_2?
2lstm_12/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_12_while_placeholder_1lstm_12_while_placeholder$lstm_12/while/lstm_cell_12/mul_2:z:0*
_output_shapes
: *
element_dtype024
2lstm_12/while/TensorArrayV2Write/TensorListSetIteml
lstm_12/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_12/while/add/y?
lstm_12/while/addAddV2lstm_12_while_placeholderlstm_12/while/add/y:output:0*
T0*
_output_shapes
: 2
lstm_12/while/addp
lstm_12/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_12/while/add_1/y?
lstm_12/while/add_1AddV2(lstm_12_while_lstm_12_while_loop_counterlstm_12/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lstm_12/while/add_1?
lstm_12/while/IdentityIdentitylstm_12/while/add_1:z:0^lstm_12/while/NoOp*
T0*
_output_shapes
: 2
lstm_12/while/Identity?
lstm_12/while/Identity_1Identity.lstm_12_while_lstm_12_while_maximum_iterations^lstm_12/while/NoOp*
T0*
_output_shapes
: 2
lstm_12/while/Identity_1?
lstm_12/while/Identity_2Identitylstm_12/while/add:z:0^lstm_12/while/NoOp*
T0*
_output_shapes
: 2
lstm_12/while/Identity_2?
lstm_12/while/Identity_3IdentityBlstm_12/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_12/while/NoOp*
T0*
_output_shapes
: 2
lstm_12/while/Identity_3?
lstm_12/while/Identity_4Identity$lstm_12/while/lstm_cell_12/mul_2:z:0^lstm_12/while/NoOp*
T0*(
_output_shapes
:??????????2
lstm_12/while/Identity_4?
lstm_12/while/Identity_5Identity$lstm_12/while/lstm_cell_12/add_1:z:0^lstm_12/while/NoOp*
T0*(
_output_shapes
:??????????2
lstm_12/while/Identity_5?
lstm_12/while/NoOpNoOp2^lstm_12/while/lstm_cell_12/BiasAdd/ReadVariableOp1^lstm_12/while/lstm_cell_12/MatMul/ReadVariableOp3^lstm_12/while/lstm_cell_12/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
lstm_12/while/NoOp"9
lstm_12_while_identitylstm_12/while/Identity:output:0"=
lstm_12_while_identity_1!lstm_12/while/Identity_1:output:0"=
lstm_12_while_identity_2!lstm_12/while/Identity_2:output:0"=
lstm_12_while_identity_3!lstm_12/while/Identity_3:output:0"=
lstm_12_while_identity_4!lstm_12/while/Identity_4:output:0"=
lstm_12_while_identity_5!lstm_12/while/Identity_5:output:0"P
%lstm_12_while_lstm_12_strided_slice_1'lstm_12_while_lstm_12_strided_slice_1_0"z
:lstm_12_while_lstm_cell_12_biasadd_readvariableop_resource<lstm_12_while_lstm_cell_12_biasadd_readvariableop_resource_0"|
;lstm_12_while_lstm_cell_12_matmul_1_readvariableop_resource=lstm_12_while_lstm_cell_12_matmul_1_readvariableop_resource_0"x
9lstm_12_while_lstm_cell_12_matmul_readvariableop_resource;lstm_12_while_lstm_cell_12_matmul_readvariableop_resource_0"?
alstm_12_while_tensorarrayv2read_tensorlistgetitem_lstm_12_tensorarrayunstack_tensorlistfromtensorclstm_12_while_tensorarrayv2read_tensorlistgetitem_lstm_12_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :??????????:??????????: : : : : 2f
1lstm_12/while/lstm_cell_12/BiasAdd/ReadVariableOp1lstm_12/while/lstm_cell_12/BiasAdd/ReadVariableOp2d
0lstm_12/while/lstm_cell_12/MatMul/ReadVariableOp0lstm_12/while/lstm_cell_12/MatMul/ReadVariableOp2h
2lstm_12/while/lstm_cell_12/MatMul_1/ReadVariableOp2lstm_12/while/lstm_cell_12/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
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
E__inference_lstm_13_layer_call_and_return_conditional_losses_26066143

inputs?
+lstm_cell_13_matmul_readvariableop_resource:
??A
-lstm_cell_13_matmul_1_readvariableop_resource:
??;
,lstm_cell_13_biasadd_readvariableop_resource:	?
identity??#lstm_cell_13/BiasAdd/ReadVariableOp?"lstm_cell_13/MatMul/ReadVariableOp?$lstm_cell_13/MatMul_1/ReadVariableOp?whileD
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
:??????????2
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
valueB"?????   27
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
:??????????*
shrink_axis_mask2
strided_slice_2?
"lstm_cell_13/MatMul/ReadVariableOpReadVariableOp+lstm_cell_13_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02$
"lstm_cell_13/MatMul/ReadVariableOp?
lstm_cell_13/MatMulMatMulstrided_slice_2:output:0*lstm_cell_13/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_13/MatMul?
$lstm_cell_13/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_13_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02&
$lstm_cell_13/MatMul_1/ReadVariableOp?
lstm_cell_13/MatMul_1MatMulzeros:output:0,lstm_cell_13/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_13/MatMul_1?
lstm_cell_13/addAddV2lstm_cell_13/MatMul:product:0lstm_cell_13/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell_13/add?
#lstm_cell_13/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_13_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02%
#lstm_cell_13/BiasAdd/ReadVariableOp?
lstm_cell_13/BiasAddBiasAddlstm_cell_13/add:z:0+lstm_cell_13/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_13/BiasAdd~
lstm_cell_13/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_13/split/split_dim?
lstm_cell_13/splitSplit%lstm_cell_13/split/split_dim:output:0lstm_cell_13/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split2
lstm_cell_13/split?
lstm_cell_13/SigmoidSigmoidlstm_cell_13/split:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_13/Sigmoid?
lstm_cell_13/Sigmoid_1Sigmoidlstm_cell_13/split:output:1*
T0*(
_output_shapes
:??????????2
lstm_cell_13/Sigmoid_1?
lstm_cell_13/mulMullstm_cell_13/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_13/mul~
lstm_cell_13/ReluRelulstm_cell_13/split:output:2*
T0*(
_output_shapes
:??????????2
lstm_cell_13/Relu?
lstm_cell_13/mul_1Mullstm_cell_13/Sigmoid:y:0lstm_cell_13/Relu:activations:0*
T0*(
_output_shapes
:??????????2
lstm_cell_13/mul_1?
lstm_cell_13/add_1AddV2lstm_cell_13/mul:z:0lstm_cell_13/mul_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell_13/add_1?
lstm_cell_13/Sigmoid_2Sigmoidlstm_cell_13/split:output:3*
T0*(
_output_shapes
:??????????2
lstm_cell_13/Sigmoid_2}
lstm_cell_13/Relu_1Relulstm_cell_13/add_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell_13/Relu_1?
lstm_cell_13/mul_2Mullstm_cell_13/Sigmoid_2:y:0!lstm_cell_13/Relu_1:activations:0*
T0*(
_output_shapes
:??????????2
lstm_cell_13/mul_2?
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_13_matmul_readvariableop_resource-lstm_cell_13_matmul_1_readvariableop_resource,lstm_cell_13_biasadd_readvariableop_resource*
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
while_body_26066059*
condR
while_cond_26066058*M
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
NoOpNoOp$^lstm_cell_13/BiasAdd/ReadVariableOp#^lstm_cell_13/MatMul/ReadVariableOp%^lstm_cell_13/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:??????????: : : 2J
#lstm_cell_13/BiasAdd/ReadVariableOp#lstm_cell_13/BiasAdd/ReadVariableOp2H
"lstm_cell_13/MatMul/ReadVariableOp"lstm_cell_13/MatMul/ReadVariableOp2L
$lstm_cell_13/MatMul_1/ReadVariableOp$lstm_cell_13/MatMul_1/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
??
?
while_body_26067863
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_12_matmul_readvariableop_resource_0:	]?I
5while_lstm_cell_12_matmul_1_readvariableop_resource_0:
??C
4while_lstm_cell_12_biasadd_readvariableop_resource_0:	?
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_12_matmul_readvariableop_resource:	]?G
3while_lstm_cell_12_matmul_1_readvariableop_resource:
??A
2while_lstm_cell_12_biasadd_readvariableop_resource:	???)while/lstm_cell_12/BiasAdd/ReadVariableOp?(while/lstm_cell_12/MatMul/ReadVariableOp?*while/lstm_cell_12/MatMul_1/ReadVariableOp?
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
(while/lstm_cell_12/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_12_matmul_readvariableop_resource_0*
_output_shapes
:	]?*
dtype02*
(while/lstm_cell_12/MatMul/ReadVariableOp?
while/lstm_cell_12/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_12/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_12/MatMul?
*while/lstm_cell_12/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_12_matmul_1_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02,
*while/lstm_cell_12/MatMul_1/ReadVariableOp?
while/lstm_cell_12/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_12/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_12/MatMul_1?
while/lstm_cell_12/addAddV2#while/lstm_cell_12/MatMul:product:0%while/lstm_cell_12/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_12/add?
)while/lstm_cell_12/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_12_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype02+
)while/lstm_cell_12/BiasAdd/ReadVariableOp?
while/lstm_cell_12/BiasAddBiasAddwhile/lstm_cell_12/add:z:01while/lstm_cell_12/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_12/BiasAdd?
"while/lstm_cell_12/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_12/split/split_dim?
while/lstm_cell_12/splitSplit+while/lstm_cell_12/split/split_dim:output:0#while/lstm_cell_12/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split2
while/lstm_cell_12/split?
while/lstm_cell_12/SigmoidSigmoid!while/lstm_cell_12/split:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_12/Sigmoid?
while/lstm_cell_12/Sigmoid_1Sigmoid!while/lstm_cell_12/split:output:1*
T0*(
_output_shapes
:??????????2
while/lstm_cell_12/Sigmoid_1?
while/lstm_cell_12/mulMul while/lstm_cell_12/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:??????????2
while/lstm_cell_12/mul?
while/lstm_cell_12/ReluRelu!while/lstm_cell_12/split:output:2*
T0*(
_output_shapes
:??????????2
while/lstm_cell_12/Relu?
while/lstm_cell_12/mul_1Mulwhile/lstm_cell_12/Sigmoid:y:0%while/lstm_cell_12/Relu:activations:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_12/mul_1?
while/lstm_cell_12/add_1AddV2while/lstm_cell_12/mul:z:0while/lstm_cell_12/mul_1:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_12/add_1?
while/lstm_cell_12/Sigmoid_2Sigmoid!while/lstm_cell_12/split:output:3*
T0*(
_output_shapes
:??????????2
while/lstm_cell_12/Sigmoid_2?
while/lstm_cell_12/Relu_1Reluwhile/lstm_cell_12/add_1:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_12/Relu_1?
while/lstm_cell_12/mul_2Mul while/lstm_cell_12/Sigmoid_2:y:0'while/lstm_cell_12/Relu_1:activations:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_12/mul_2?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_12/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_12/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:??????????2
while/Identity_4?
while/Identity_5Identitywhile/lstm_cell_12/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:??????????2
while/Identity_5?

while/NoOpNoOp*^while/lstm_cell_12/BiasAdd/ReadVariableOp)^while/lstm_cell_12/MatMul/ReadVariableOp+^while/lstm_cell_12/MatMul_1/ReadVariableOp*"
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
2while_lstm_cell_12_biasadd_readvariableop_resource4while_lstm_cell_12_biasadd_readvariableop_resource_0"l
3while_lstm_cell_12_matmul_1_readvariableop_resource5while_lstm_cell_12_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_12_matmul_readvariableop_resource3while_lstm_cell_12_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :??????????:??????????: : : : : 2V
)while/lstm_cell_12/BiasAdd/ReadVariableOp)while/lstm_cell_12/BiasAdd/ReadVariableOp2T
(while/lstm_cell_12/MatMul/ReadVariableOp(while/lstm_cell_12/MatMul/ReadVariableOp2X
*while/lstm_cell_12/MatMul_1/ReadVariableOp*while/lstm_cell_12/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
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
-__inference_dropout_13_layer_call_fn_26068839

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
H__inference_dropout_13_layer_call_and_return_conditional_losses_260661562
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
?
?
while_cond_26065893
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_26065893___redundant_placeholder06
2while_while_cond_26065893___redundant_placeholder16
2while_while_cond_26065893___redundant_placeholder26
2while_while_cond_26065893___redundant_placeholder3
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
while_cond_26066327
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_26066327___redundant_placeholder06
2while_while_cond_26066327___redundant_placeholder16
2while_while_cond_26066327___redundant_placeholder26
2while_while_cond_26066327___redundant_placeholder3
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
?!
?
E__inference_dense_6_layer_call_and_return_conditional_losses_26066189

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
g
H__inference_dropout_12_layer_call_and_return_conditional_losses_26068159

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
 *  ?>2
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
??
?
while_body_26066524
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_12_matmul_readvariableop_resource_0:	]?I
5while_lstm_cell_12_matmul_1_readvariableop_resource_0:
??C
4while_lstm_cell_12_biasadd_readvariableop_resource_0:	?
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_12_matmul_readvariableop_resource:	]?G
3while_lstm_cell_12_matmul_1_readvariableop_resource:
??A
2while_lstm_cell_12_biasadd_readvariableop_resource:	???)while/lstm_cell_12/BiasAdd/ReadVariableOp?(while/lstm_cell_12/MatMul/ReadVariableOp?*while/lstm_cell_12/MatMul_1/ReadVariableOp?
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
(while/lstm_cell_12/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_12_matmul_readvariableop_resource_0*
_output_shapes
:	]?*
dtype02*
(while/lstm_cell_12/MatMul/ReadVariableOp?
while/lstm_cell_12/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_12/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_12/MatMul?
*while/lstm_cell_12/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_12_matmul_1_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02,
*while/lstm_cell_12/MatMul_1/ReadVariableOp?
while/lstm_cell_12/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_12/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_12/MatMul_1?
while/lstm_cell_12/addAddV2#while/lstm_cell_12/MatMul:product:0%while/lstm_cell_12/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_12/add?
)while/lstm_cell_12/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_12_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype02+
)while/lstm_cell_12/BiasAdd/ReadVariableOp?
while/lstm_cell_12/BiasAddBiasAddwhile/lstm_cell_12/add:z:01while/lstm_cell_12/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_12/BiasAdd?
"while/lstm_cell_12/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_12/split/split_dim?
while/lstm_cell_12/splitSplit+while/lstm_cell_12/split/split_dim:output:0#while/lstm_cell_12/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split2
while/lstm_cell_12/split?
while/lstm_cell_12/SigmoidSigmoid!while/lstm_cell_12/split:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_12/Sigmoid?
while/lstm_cell_12/Sigmoid_1Sigmoid!while/lstm_cell_12/split:output:1*
T0*(
_output_shapes
:??????????2
while/lstm_cell_12/Sigmoid_1?
while/lstm_cell_12/mulMul while/lstm_cell_12/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:??????????2
while/lstm_cell_12/mul?
while/lstm_cell_12/ReluRelu!while/lstm_cell_12/split:output:2*
T0*(
_output_shapes
:??????????2
while/lstm_cell_12/Relu?
while/lstm_cell_12/mul_1Mulwhile/lstm_cell_12/Sigmoid:y:0%while/lstm_cell_12/Relu:activations:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_12/mul_1?
while/lstm_cell_12/add_1AddV2while/lstm_cell_12/mul:z:0while/lstm_cell_12/mul_1:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_12/add_1?
while/lstm_cell_12/Sigmoid_2Sigmoid!while/lstm_cell_12/split:output:3*
T0*(
_output_shapes
:??????????2
while/lstm_cell_12/Sigmoid_2?
while/lstm_cell_12/Relu_1Reluwhile/lstm_cell_12/add_1:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_12/Relu_1?
while/lstm_cell_12/mul_2Mul while/lstm_cell_12/Sigmoid_2:y:0'while/lstm_cell_12/Relu_1:activations:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_12/mul_2?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_12/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_12/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:??????????2
while/Identity_4?
while/Identity_5Identitywhile/lstm_cell_12/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:??????????2
while/Identity_5?

while/NoOpNoOp*^while/lstm_cell_12/BiasAdd/ReadVariableOp)^while/lstm_cell_12/MatMul/ReadVariableOp+^while/lstm_cell_12/MatMul_1/ReadVariableOp*"
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
2while_lstm_cell_12_biasadd_readvariableop_resource4while_lstm_cell_12_biasadd_readvariableop_resource_0"l
3while_lstm_cell_12_matmul_1_readvariableop_resource5while_lstm_cell_12_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_12_matmul_readvariableop_resource3while_lstm_cell_12_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :??????????:??????????: : : : : 2V
)while/lstm_cell_12/BiasAdd/ReadVariableOp)while/lstm_cell_12/BiasAdd/ReadVariableOp2T
(while/lstm_cell_12/MatMul/ReadVariableOp(while/lstm_cell_12/MatMul/ReadVariableOp2X
*while/lstm_cell_12/MatMul_1/ReadVariableOp*while/lstm_cell_12/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
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
?L
?
!__inference__traced_save_26069202
file_prefix-
)savev2_dense_6_kernel_read_readvariableop+
'savev2_dense_6_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop:
6savev2_lstm_12_lstm_cell_12_kernel_read_readvariableopD
@savev2_lstm_12_lstm_cell_12_recurrent_kernel_read_readvariableop8
4savev2_lstm_12_lstm_cell_12_bias_read_readvariableop:
6savev2_lstm_13_lstm_cell_13_kernel_read_readvariableopD
@savev2_lstm_13_lstm_cell_13_recurrent_kernel_read_readvariableop8
4savev2_lstm_13_lstm_cell_13_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop4
0savev2_adam_dense_6_kernel_m_read_readvariableop2
.savev2_adam_dense_6_bias_m_read_readvariableopA
=savev2_adam_lstm_12_lstm_cell_12_kernel_m_read_readvariableopK
Gsavev2_adam_lstm_12_lstm_cell_12_recurrent_kernel_m_read_readvariableop?
;savev2_adam_lstm_12_lstm_cell_12_bias_m_read_readvariableopA
=savev2_adam_lstm_13_lstm_cell_13_kernel_m_read_readvariableopK
Gsavev2_adam_lstm_13_lstm_cell_13_recurrent_kernel_m_read_readvariableop?
;savev2_adam_lstm_13_lstm_cell_13_bias_m_read_readvariableop4
0savev2_adam_dense_6_kernel_v_read_readvariableop2
.savev2_adam_dense_6_bias_v_read_readvariableopA
=savev2_adam_lstm_12_lstm_cell_12_kernel_v_read_readvariableopK
Gsavev2_adam_lstm_12_lstm_cell_12_recurrent_kernel_v_read_readvariableop?
;savev2_adam_lstm_12_lstm_cell_12_bias_v_read_readvariableopA
=savev2_adam_lstm_13_lstm_cell_13_kernel_v_read_readvariableopK
Gsavev2_adam_lstm_13_lstm_cell_13_recurrent_kernel_v_read_readvariableop?
;savev2_adam_lstm_13_lstm_cell_13_bias_v_read_readvariableop
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
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0)savev2_dense_6_kernel_read_readvariableop'savev2_dense_6_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop6savev2_lstm_12_lstm_cell_12_kernel_read_readvariableop@savev2_lstm_12_lstm_cell_12_recurrent_kernel_read_readvariableop4savev2_lstm_12_lstm_cell_12_bias_read_readvariableop6savev2_lstm_13_lstm_cell_13_kernel_read_readvariableop@savev2_lstm_13_lstm_cell_13_recurrent_kernel_read_readvariableop4savev2_lstm_13_lstm_cell_13_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop0savev2_adam_dense_6_kernel_m_read_readvariableop.savev2_adam_dense_6_bias_m_read_readvariableop=savev2_adam_lstm_12_lstm_cell_12_kernel_m_read_readvariableopGsavev2_adam_lstm_12_lstm_cell_12_recurrent_kernel_m_read_readvariableop;savev2_adam_lstm_12_lstm_cell_12_bias_m_read_readvariableop=savev2_adam_lstm_13_lstm_cell_13_kernel_m_read_readvariableopGsavev2_adam_lstm_13_lstm_cell_13_recurrent_kernel_m_read_readvariableop;savev2_adam_lstm_13_lstm_cell_13_bias_m_read_readvariableop0savev2_adam_dense_6_kernel_v_read_readvariableop.savev2_adam_dense_6_bias_v_read_readvariableop=savev2_adam_lstm_12_lstm_cell_12_kernel_v_read_readvariableopGsavev2_adam_lstm_12_lstm_cell_12_recurrent_kernel_v_read_readvariableop;savev2_adam_lstm_12_lstm_cell_12_bias_v_read_readvariableop=savev2_adam_lstm_13_lstm_cell_13_kernel_v_read_readvariableopGsavev2_adam_lstm_13_lstm_cell_13_recurrent_kernel_v_read_readvariableop;savev2_adam_lstm_13_lstm_cell_13_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
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
?: :	?:: : : : : :	]?:
??:?:
??:
??:?: : : : :	?::	]?:
??:?:
??:
??:?:	?::	]?:
??:?:
??:
??:?: 2(
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
:	]?:&	"
 
_output_shapes
:
??:!


_output_shapes	
:?:&"
 
_output_shapes
:
??:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:
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
:	]?:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:&"
 
_output_shapes
:
??:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:%!

_output_shapes
:	?: 

_output_shapes
::%!

_output_shapes
:	]?:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:&"
 
_output_shapes
:
??:& "
 
_output_shapes
:
??:!!

_output_shapes	
:?:"

_output_shapes
: 
?
f
H__inference_dropout_13_layer_call_and_return_conditional_losses_26066156

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
?
?
while_cond_26065489
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_26065489___redundant_placeholder06
2while_while_cond_26065489___redundant_placeholder16
2while_while_cond_26065489___redundant_placeholder26
2while_while_cond_26065489___redundant_placeholder3
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
while_cond_26068235
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_26068235___redundant_placeholder06
2while_while_cond_26068235___redundant_placeholder16
2while_while_cond_26068235___redundant_placeholder26
2while_while_cond_26068235___redundant_placeholder3
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
??
?
while_body_26068387
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
3while_lstm_cell_13_matmul_readvariableop_resource_0:
??I
5while_lstm_cell_13_matmul_1_readvariableop_resource_0:
??C
4while_lstm_cell_13_biasadd_readvariableop_resource_0:	?
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
1while_lstm_cell_13_matmul_readvariableop_resource:
??G
3while_lstm_cell_13_matmul_1_readvariableop_resource:
??A
2while_lstm_cell_13_biasadd_readvariableop_resource:	???)while/lstm_cell_13/BiasAdd/ReadVariableOp?(while/lstm_cell_13/MatMul/ReadVariableOp?*while/lstm_cell_13/MatMul_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:??????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
(while/lstm_cell_13/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_13_matmul_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02*
(while/lstm_cell_13/MatMul/ReadVariableOp?
while/lstm_cell_13/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_13/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_13/MatMul?
*while/lstm_cell_13/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_13_matmul_1_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02,
*while/lstm_cell_13/MatMul_1/ReadVariableOp?
while/lstm_cell_13/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_13/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_13/MatMul_1?
while/lstm_cell_13/addAddV2#while/lstm_cell_13/MatMul:product:0%while/lstm_cell_13/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_13/add?
)while/lstm_cell_13/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_13_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype02+
)while/lstm_cell_13/BiasAdd/ReadVariableOp?
while/lstm_cell_13/BiasAddBiasAddwhile/lstm_cell_13/add:z:01while/lstm_cell_13/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_13/BiasAdd?
"while/lstm_cell_13/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_13/split/split_dim?
while/lstm_cell_13/splitSplit+while/lstm_cell_13/split/split_dim:output:0#while/lstm_cell_13/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split2
while/lstm_cell_13/split?
while/lstm_cell_13/SigmoidSigmoid!while/lstm_cell_13/split:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_13/Sigmoid?
while/lstm_cell_13/Sigmoid_1Sigmoid!while/lstm_cell_13/split:output:1*
T0*(
_output_shapes
:??????????2
while/lstm_cell_13/Sigmoid_1?
while/lstm_cell_13/mulMul while/lstm_cell_13/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:??????????2
while/lstm_cell_13/mul?
while/lstm_cell_13/ReluRelu!while/lstm_cell_13/split:output:2*
T0*(
_output_shapes
:??????????2
while/lstm_cell_13/Relu?
while/lstm_cell_13/mul_1Mulwhile/lstm_cell_13/Sigmoid:y:0%while/lstm_cell_13/Relu:activations:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_13/mul_1?
while/lstm_cell_13/add_1AddV2while/lstm_cell_13/mul:z:0while/lstm_cell_13/mul_1:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_13/add_1?
while/lstm_cell_13/Sigmoid_2Sigmoid!while/lstm_cell_13/split:output:3*
T0*(
_output_shapes
:??????????2
while/lstm_cell_13/Sigmoid_2?
while/lstm_cell_13/Relu_1Reluwhile/lstm_cell_13/add_1:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_13/Relu_1?
while/lstm_cell_13/mul_2Mul while/lstm_cell_13/Sigmoid_2:y:0'while/lstm_cell_13/Relu_1:activations:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_13/mul_2?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_13/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_13/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:??????????2
while/Identity_4?
while/Identity_5Identitywhile/lstm_cell_13/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:??????????2
while/Identity_5?

while/NoOpNoOp*^while/lstm_cell_13/BiasAdd/ReadVariableOp)^while/lstm_cell_13/MatMul/ReadVariableOp+^while/lstm_cell_13/MatMul_1/ReadVariableOp*"
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
2while_lstm_cell_13_biasadd_readvariableop_resource4while_lstm_cell_13_biasadd_readvariableop_resource_0"l
3while_lstm_cell_13_matmul_1_readvariableop_resource5while_lstm_cell_13_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_13_matmul_readvariableop_resource3while_lstm_cell_13_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :??????????:??????????: : : : : 2V
)while/lstm_cell_13/BiasAdd/ReadVariableOp)while/lstm_cell_13/BiasAdd/ReadVariableOp2T
(while/lstm_cell_13/MatMul/ReadVariableOp(while/lstm_cell_13/MatMul/ReadVariableOp2X
*while/lstm_cell_13/MatMul_1/ReadVariableOp*while/lstm_cell_13/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
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
?&
?
while_body_26064650
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_00
while_lstm_cell_12_26064674_0:	]?1
while_lstm_cell_12_26064676_0:
??,
while_lstm_cell_12_26064678_0:	?
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor.
while_lstm_cell_12_26064674:	]?/
while_lstm_cell_12_26064676:
??*
while_lstm_cell_12_26064678:	???*while/lstm_cell_12/StatefulPartitionedCall?
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
*while/lstm_cell_12/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_12_26064674_0while_lstm_cell_12_26064676_0while_lstm_cell_12_26064678_0*
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
J__inference_lstm_cell_12_layer_call_and_return_conditional_losses_260646362,
*while/lstm_cell_12/StatefulPartitionedCall?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder3while/lstm_cell_12/StatefulPartitionedCall:output:0*
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
while/Identity_4Identity3while/lstm_cell_12/StatefulPartitionedCall:output:1^while/NoOp*
T0*(
_output_shapes
:??????????2
while/Identity_4?
while/Identity_5Identity3while/lstm_cell_12/StatefulPartitionedCall:output:2^while/NoOp*
T0*(
_output_shapes
:??????????2
while/Identity_5?

while/NoOpNoOp+^while/lstm_cell_12/StatefulPartitionedCall*"
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
while_lstm_cell_12_26064674while_lstm_cell_12_26064674_0"<
while_lstm_cell_12_26064676while_lstm_cell_12_26064676_0"<
while_lstm_cell_12_26064678while_lstm_cell_12_26064678_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :??????????:??????????: : : : : 2X
*while/lstm_cell_12/StatefulPartitionedCall*while/lstm_cell_12/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
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
?
/__inference_lstm_cell_13_layer_call_fn_26069080

inputs
states_0
states_1
unknown:
??
	unknown_0:
??
	unknown_1:	?
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
J__inference_lstm_cell_13_layer_call_and_return_conditional_losses_260654122
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
B:??????????:??????????:??????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
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
?
g
H__inference_dropout_12_layer_call_and_return_conditional_losses_26066441

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
 *  ?>2
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
?
g
H__inference_dropout_13_layer_call_and_return_conditional_losses_26068834

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
?
?
*__inference_lstm_13_layer_call_fn_26068817

inputs
unknown:
??
	unknown_0:
??
	unknown_1:	?
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
E__inference_lstm_13_layer_call_and_return_conditional_losses_260664122
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
:??????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
while_cond_26067711
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_26067711___redundant_placeholder06
2while_while_cond_26067711___redundant_placeholder16
2while_while_cond_26067711___redundant_placeholder26
2while_while_cond_26067711___redundant_placeholder3
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
?]
?
(sequential_6_lstm_12_while_body_26064301F
Bsequential_6_lstm_12_while_sequential_6_lstm_12_while_loop_counterL
Hsequential_6_lstm_12_while_sequential_6_lstm_12_while_maximum_iterations*
&sequential_6_lstm_12_while_placeholder,
(sequential_6_lstm_12_while_placeholder_1,
(sequential_6_lstm_12_while_placeholder_2,
(sequential_6_lstm_12_while_placeholder_3E
Asequential_6_lstm_12_while_sequential_6_lstm_12_strided_slice_1_0?
}sequential_6_lstm_12_while_tensorarrayv2read_tensorlistgetitem_sequential_6_lstm_12_tensorarrayunstack_tensorlistfromtensor_0[
Hsequential_6_lstm_12_while_lstm_cell_12_matmul_readvariableop_resource_0:	]?^
Jsequential_6_lstm_12_while_lstm_cell_12_matmul_1_readvariableop_resource_0:
??X
Isequential_6_lstm_12_while_lstm_cell_12_biasadd_readvariableop_resource_0:	?'
#sequential_6_lstm_12_while_identity)
%sequential_6_lstm_12_while_identity_1)
%sequential_6_lstm_12_while_identity_2)
%sequential_6_lstm_12_while_identity_3)
%sequential_6_lstm_12_while_identity_4)
%sequential_6_lstm_12_while_identity_5C
?sequential_6_lstm_12_while_sequential_6_lstm_12_strided_slice_1
{sequential_6_lstm_12_while_tensorarrayv2read_tensorlistgetitem_sequential_6_lstm_12_tensorarrayunstack_tensorlistfromtensorY
Fsequential_6_lstm_12_while_lstm_cell_12_matmul_readvariableop_resource:	]?\
Hsequential_6_lstm_12_while_lstm_cell_12_matmul_1_readvariableop_resource:
??V
Gsequential_6_lstm_12_while_lstm_cell_12_biasadd_readvariableop_resource:	???>sequential_6/lstm_12/while/lstm_cell_12/BiasAdd/ReadVariableOp?=sequential_6/lstm_12/while/lstm_cell_12/MatMul/ReadVariableOp??sequential_6/lstm_12/while/lstm_cell_12/MatMul_1/ReadVariableOp?
Lsequential_6/lstm_12/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????]   2N
Lsequential_6/lstm_12/while/TensorArrayV2Read/TensorListGetItem/element_shape?
>sequential_6/lstm_12/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem}sequential_6_lstm_12_while_tensorarrayv2read_tensorlistgetitem_sequential_6_lstm_12_tensorarrayunstack_tensorlistfromtensor_0&sequential_6_lstm_12_while_placeholderUsequential_6/lstm_12/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????]*
element_dtype02@
>sequential_6/lstm_12/while/TensorArrayV2Read/TensorListGetItem?
=sequential_6/lstm_12/while/lstm_cell_12/MatMul/ReadVariableOpReadVariableOpHsequential_6_lstm_12_while_lstm_cell_12_matmul_readvariableop_resource_0*
_output_shapes
:	]?*
dtype02?
=sequential_6/lstm_12/while/lstm_cell_12/MatMul/ReadVariableOp?
.sequential_6/lstm_12/while/lstm_cell_12/MatMulMatMulEsequential_6/lstm_12/while/TensorArrayV2Read/TensorListGetItem:item:0Esequential_6/lstm_12/while/lstm_cell_12/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????20
.sequential_6/lstm_12/while/lstm_cell_12/MatMul?
?sequential_6/lstm_12/while/lstm_cell_12/MatMul_1/ReadVariableOpReadVariableOpJsequential_6_lstm_12_while_lstm_cell_12_matmul_1_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02A
?sequential_6/lstm_12/while/lstm_cell_12/MatMul_1/ReadVariableOp?
0sequential_6/lstm_12/while/lstm_cell_12/MatMul_1MatMul(sequential_6_lstm_12_while_placeholder_2Gsequential_6/lstm_12/while/lstm_cell_12/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????22
0sequential_6/lstm_12/while/lstm_cell_12/MatMul_1?
+sequential_6/lstm_12/while/lstm_cell_12/addAddV28sequential_6/lstm_12/while/lstm_cell_12/MatMul:product:0:sequential_6/lstm_12/while/lstm_cell_12/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2-
+sequential_6/lstm_12/while/lstm_cell_12/add?
>sequential_6/lstm_12/while/lstm_cell_12/BiasAdd/ReadVariableOpReadVariableOpIsequential_6_lstm_12_while_lstm_cell_12_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype02@
>sequential_6/lstm_12/while/lstm_cell_12/BiasAdd/ReadVariableOp?
/sequential_6/lstm_12/while/lstm_cell_12/BiasAddBiasAdd/sequential_6/lstm_12/while/lstm_cell_12/add:z:0Fsequential_6/lstm_12/while/lstm_cell_12/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????21
/sequential_6/lstm_12/while/lstm_cell_12/BiasAdd?
7sequential_6/lstm_12/while/lstm_cell_12/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :29
7sequential_6/lstm_12/while/lstm_cell_12/split/split_dim?
-sequential_6/lstm_12/while/lstm_cell_12/splitSplit@sequential_6/lstm_12/while/lstm_cell_12/split/split_dim:output:08sequential_6/lstm_12/while/lstm_cell_12/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split2/
-sequential_6/lstm_12/while/lstm_cell_12/split?
/sequential_6/lstm_12/while/lstm_cell_12/SigmoidSigmoid6sequential_6/lstm_12/while/lstm_cell_12/split:output:0*
T0*(
_output_shapes
:??????????21
/sequential_6/lstm_12/while/lstm_cell_12/Sigmoid?
1sequential_6/lstm_12/while/lstm_cell_12/Sigmoid_1Sigmoid6sequential_6/lstm_12/while/lstm_cell_12/split:output:1*
T0*(
_output_shapes
:??????????23
1sequential_6/lstm_12/while/lstm_cell_12/Sigmoid_1?
+sequential_6/lstm_12/while/lstm_cell_12/mulMul5sequential_6/lstm_12/while/lstm_cell_12/Sigmoid_1:y:0(sequential_6_lstm_12_while_placeholder_3*
T0*(
_output_shapes
:??????????2-
+sequential_6/lstm_12/while/lstm_cell_12/mul?
,sequential_6/lstm_12/while/lstm_cell_12/ReluRelu6sequential_6/lstm_12/while/lstm_cell_12/split:output:2*
T0*(
_output_shapes
:??????????2.
,sequential_6/lstm_12/while/lstm_cell_12/Relu?
-sequential_6/lstm_12/while/lstm_cell_12/mul_1Mul3sequential_6/lstm_12/while/lstm_cell_12/Sigmoid:y:0:sequential_6/lstm_12/while/lstm_cell_12/Relu:activations:0*
T0*(
_output_shapes
:??????????2/
-sequential_6/lstm_12/while/lstm_cell_12/mul_1?
-sequential_6/lstm_12/while/lstm_cell_12/add_1AddV2/sequential_6/lstm_12/while/lstm_cell_12/mul:z:01sequential_6/lstm_12/while/lstm_cell_12/mul_1:z:0*
T0*(
_output_shapes
:??????????2/
-sequential_6/lstm_12/while/lstm_cell_12/add_1?
1sequential_6/lstm_12/while/lstm_cell_12/Sigmoid_2Sigmoid6sequential_6/lstm_12/while/lstm_cell_12/split:output:3*
T0*(
_output_shapes
:??????????23
1sequential_6/lstm_12/while/lstm_cell_12/Sigmoid_2?
.sequential_6/lstm_12/while/lstm_cell_12/Relu_1Relu1sequential_6/lstm_12/while/lstm_cell_12/add_1:z:0*
T0*(
_output_shapes
:??????????20
.sequential_6/lstm_12/while/lstm_cell_12/Relu_1?
-sequential_6/lstm_12/while/lstm_cell_12/mul_2Mul5sequential_6/lstm_12/while/lstm_cell_12/Sigmoid_2:y:0<sequential_6/lstm_12/while/lstm_cell_12/Relu_1:activations:0*
T0*(
_output_shapes
:??????????2/
-sequential_6/lstm_12/while/lstm_cell_12/mul_2?
?sequential_6/lstm_12/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem(sequential_6_lstm_12_while_placeholder_1&sequential_6_lstm_12_while_placeholder1sequential_6/lstm_12/while/lstm_cell_12/mul_2:z:0*
_output_shapes
: *
element_dtype02A
?sequential_6/lstm_12/while/TensorArrayV2Write/TensorListSetItem?
 sequential_6/lstm_12/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2"
 sequential_6/lstm_12/while/add/y?
sequential_6/lstm_12/while/addAddV2&sequential_6_lstm_12_while_placeholder)sequential_6/lstm_12/while/add/y:output:0*
T0*
_output_shapes
: 2 
sequential_6/lstm_12/while/add?
"sequential_6/lstm_12/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2$
"sequential_6/lstm_12/while/add_1/y?
 sequential_6/lstm_12/while/add_1AddV2Bsequential_6_lstm_12_while_sequential_6_lstm_12_while_loop_counter+sequential_6/lstm_12/while/add_1/y:output:0*
T0*
_output_shapes
: 2"
 sequential_6/lstm_12/while/add_1?
#sequential_6/lstm_12/while/IdentityIdentity$sequential_6/lstm_12/while/add_1:z:0 ^sequential_6/lstm_12/while/NoOp*
T0*
_output_shapes
: 2%
#sequential_6/lstm_12/while/Identity?
%sequential_6/lstm_12/while/Identity_1IdentityHsequential_6_lstm_12_while_sequential_6_lstm_12_while_maximum_iterations ^sequential_6/lstm_12/while/NoOp*
T0*
_output_shapes
: 2'
%sequential_6/lstm_12/while/Identity_1?
%sequential_6/lstm_12/while/Identity_2Identity"sequential_6/lstm_12/while/add:z:0 ^sequential_6/lstm_12/while/NoOp*
T0*
_output_shapes
: 2'
%sequential_6/lstm_12/while/Identity_2?
%sequential_6/lstm_12/while/Identity_3IdentityOsequential_6/lstm_12/while/TensorArrayV2Write/TensorListSetItem:output_handle:0 ^sequential_6/lstm_12/while/NoOp*
T0*
_output_shapes
: 2'
%sequential_6/lstm_12/while/Identity_3?
%sequential_6/lstm_12/while/Identity_4Identity1sequential_6/lstm_12/while/lstm_cell_12/mul_2:z:0 ^sequential_6/lstm_12/while/NoOp*
T0*(
_output_shapes
:??????????2'
%sequential_6/lstm_12/while/Identity_4?
%sequential_6/lstm_12/while/Identity_5Identity1sequential_6/lstm_12/while/lstm_cell_12/add_1:z:0 ^sequential_6/lstm_12/while/NoOp*
T0*(
_output_shapes
:??????????2'
%sequential_6/lstm_12/while/Identity_5?
sequential_6/lstm_12/while/NoOpNoOp?^sequential_6/lstm_12/while/lstm_cell_12/BiasAdd/ReadVariableOp>^sequential_6/lstm_12/while/lstm_cell_12/MatMul/ReadVariableOp@^sequential_6/lstm_12/while/lstm_cell_12/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2!
sequential_6/lstm_12/while/NoOp"S
#sequential_6_lstm_12_while_identity,sequential_6/lstm_12/while/Identity:output:0"W
%sequential_6_lstm_12_while_identity_1.sequential_6/lstm_12/while/Identity_1:output:0"W
%sequential_6_lstm_12_while_identity_2.sequential_6/lstm_12/while/Identity_2:output:0"W
%sequential_6_lstm_12_while_identity_3.sequential_6/lstm_12/while/Identity_3:output:0"W
%sequential_6_lstm_12_while_identity_4.sequential_6/lstm_12/while/Identity_4:output:0"W
%sequential_6_lstm_12_while_identity_5.sequential_6/lstm_12/while/Identity_5:output:0"?
Gsequential_6_lstm_12_while_lstm_cell_12_biasadd_readvariableop_resourceIsequential_6_lstm_12_while_lstm_cell_12_biasadd_readvariableop_resource_0"?
Hsequential_6_lstm_12_while_lstm_cell_12_matmul_1_readvariableop_resourceJsequential_6_lstm_12_while_lstm_cell_12_matmul_1_readvariableop_resource_0"?
Fsequential_6_lstm_12_while_lstm_cell_12_matmul_readvariableop_resourceHsequential_6_lstm_12_while_lstm_cell_12_matmul_readvariableop_resource_0"?
?sequential_6_lstm_12_while_sequential_6_lstm_12_strided_slice_1Asequential_6_lstm_12_while_sequential_6_lstm_12_strided_slice_1_0"?
{sequential_6_lstm_12_while_tensorarrayv2read_tensorlistgetitem_sequential_6_lstm_12_tensorarrayunstack_tensorlistfromtensor}sequential_6_lstm_12_while_tensorarrayv2read_tensorlistgetitem_sequential_6_lstm_12_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :??????????:??????????: : : : : 2?
>sequential_6/lstm_12/while/lstm_cell_12/BiasAdd/ReadVariableOp>sequential_6/lstm_12/while/lstm_cell_12/BiasAdd/ReadVariableOp2~
=sequential_6/lstm_12/while/lstm_cell_12/MatMul/ReadVariableOp=sequential_6/lstm_12/while/lstm_cell_12/MatMul/ReadVariableOp2?
?sequential_6/lstm_12/while/lstm_cell_12/MatMul_1/ReadVariableOp?sequential_6/lstm_12/while/lstm_cell_12/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
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
while_body_26067561
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_12_matmul_readvariableop_resource_0:	]?I
5while_lstm_cell_12_matmul_1_readvariableop_resource_0:
??C
4while_lstm_cell_12_biasadd_readvariableop_resource_0:	?
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_12_matmul_readvariableop_resource:	]?G
3while_lstm_cell_12_matmul_1_readvariableop_resource:
??A
2while_lstm_cell_12_biasadd_readvariableop_resource:	???)while/lstm_cell_12/BiasAdd/ReadVariableOp?(while/lstm_cell_12/MatMul/ReadVariableOp?*while/lstm_cell_12/MatMul_1/ReadVariableOp?
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
(while/lstm_cell_12/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_12_matmul_readvariableop_resource_0*
_output_shapes
:	]?*
dtype02*
(while/lstm_cell_12/MatMul/ReadVariableOp?
while/lstm_cell_12/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_12/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_12/MatMul?
*while/lstm_cell_12/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_12_matmul_1_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02,
*while/lstm_cell_12/MatMul_1/ReadVariableOp?
while/lstm_cell_12/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_12/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_12/MatMul_1?
while/lstm_cell_12/addAddV2#while/lstm_cell_12/MatMul:product:0%while/lstm_cell_12/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_12/add?
)while/lstm_cell_12/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_12_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype02+
)while/lstm_cell_12/BiasAdd/ReadVariableOp?
while/lstm_cell_12/BiasAddBiasAddwhile/lstm_cell_12/add:z:01while/lstm_cell_12/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_12/BiasAdd?
"while/lstm_cell_12/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_12/split/split_dim?
while/lstm_cell_12/splitSplit+while/lstm_cell_12/split/split_dim:output:0#while/lstm_cell_12/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split2
while/lstm_cell_12/split?
while/lstm_cell_12/SigmoidSigmoid!while/lstm_cell_12/split:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_12/Sigmoid?
while/lstm_cell_12/Sigmoid_1Sigmoid!while/lstm_cell_12/split:output:1*
T0*(
_output_shapes
:??????????2
while/lstm_cell_12/Sigmoid_1?
while/lstm_cell_12/mulMul while/lstm_cell_12/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:??????????2
while/lstm_cell_12/mul?
while/lstm_cell_12/ReluRelu!while/lstm_cell_12/split:output:2*
T0*(
_output_shapes
:??????????2
while/lstm_cell_12/Relu?
while/lstm_cell_12/mul_1Mulwhile/lstm_cell_12/Sigmoid:y:0%while/lstm_cell_12/Relu:activations:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_12/mul_1?
while/lstm_cell_12/add_1AddV2while/lstm_cell_12/mul:z:0while/lstm_cell_12/mul_1:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_12/add_1?
while/lstm_cell_12/Sigmoid_2Sigmoid!while/lstm_cell_12/split:output:3*
T0*(
_output_shapes
:??????????2
while/lstm_cell_12/Sigmoid_2?
while/lstm_cell_12/Relu_1Reluwhile/lstm_cell_12/add_1:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_12/Relu_1?
while/lstm_cell_12/mul_2Mul while/lstm_cell_12/Sigmoid_2:y:0'while/lstm_cell_12/Relu_1:activations:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_12/mul_2?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_12/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_12/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:??????????2
while/Identity_4?
while/Identity_5Identitywhile/lstm_cell_12/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:??????????2
while/Identity_5?

while/NoOpNoOp*^while/lstm_cell_12/BiasAdd/ReadVariableOp)^while/lstm_cell_12/MatMul/ReadVariableOp+^while/lstm_cell_12/MatMul_1/ReadVariableOp*"
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
2while_lstm_cell_12_biasadd_readvariableop_resource4while_lstm_cell_12_biasadd_readvariableop_resource_0"l
3while_lstm_cell_12_matmul_1_readvariableop_resource5while_lstm_cell_12_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_12_matmul_readvariableop_resource3while_lstm_cell_12_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :??????????:??????????: : : : : 2V
)while/lstm_cell_12/BiasAdd/ReadVariableOp)while/lstm_cell_12/BiasAdd/ReadVariableOp2T
(while/lstm_cell_12/MatMul/ReadVariableOp(while/lstm_cell_12/MatMul/ReadVariableOp2X
*while/lstm_cell_12/MatMul_1/ReadVariableOp*while/lstm_cell_12/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
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
?
/__inference_lstm_cell_12_layer_call_fn_26068965

inputs
states_0
states_1
unknown:	]?
	unknown_0:
??
	unknown_1:	?
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
J__inference_lstm_cell_12_layer_call_and_return_conditional_losses_260646362
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
_construction_contextkEagerRuntime*T
_input_shapesC
A:?????????]:??????????:??????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????]
 
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
??
?	
#__inference__wrapped_model_26064561
lstm_12_inputS
@sequential_6_lstm_12_lstm_cell_12_matmul_readvariableop_resource:	]?V
Bsequential_6_lstm_12_lstm_cell_12_matmul_1_readvariableop_resource:
??P
Asequential_6_lstm_12_lstm_cell_12_biasadd_readvariableop_resource:	?T
@sequential_6_lstm_13_lstm_cell_13_matmul_readvariableop_resource:
??V
Bsequential_6_lstm_13_lstm_cell_13_matmul_1_readvariableop_resource:
??P
Asequential_6_lstm_13_lstm_cell_13_biasadd_readvariableop_resource:	?I
6sequential_6_dense_6_tensordot_readvariableop_resource:	?B
4sequential_6_dense_6_biasadd_readvariableop_resource:
identity??+sequential_6/dense_6/BiasAdd/ReadVariableOp?-sequential_6/dense_6/Tensordot/ReadVariableOp?8sequential_6/lstm_12/lstm_cell_12/BiasAdd/ReadVariableOp?7sequential_6/lstm_12/lstm_cell_12/MatMul/ReadVariableOp?9sequential_6/lstm_12/lstm_cell_12/MatMul_1/ReadVariableOp?sequential_6/lstm_12/while?8sequential_6/lstm_13/lstm_cell_13/BiasAdd/ReadVariableOp?7sequential_6/lstm_13/lstm_cell_13/MatMul/ReadVariableOp?9sequential_6/lstm_13/lstm_cell_13/MatMul_1/ReadVariableOp?sequential_6/lstm_13/whileu
sequential_6/lstm_12/ShapeShapelstm_12_input*
T0*
_output_shapes
:2
sequential_6/lstm_12/Shape?
(sequential_6/lstm_12/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(sequential_6/lstm_12/strided_slice/stack?
*sequential_6/lstm_12/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*sequential_6/lstm_12/strided_slice/stack_1?
*sequential_6/lstm_12/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*sequential_6/lstm_12/strided_slice/stack_2?
"sequential_6/lstm_12/strided_sliceStridedSlice#sequential_6/lstm_12/Shape:output:01sequential_6/lstm_12/strided_slice/stack:output:03sequential_6/lstm_12/strided_slice/stack_1:output:03sequential_6/lstm_12/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"sequential_6/lstm_12/strided_slice?
 sequential_6/lstm_12/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2"
 sequential_6/lstm_12/zeros/mul/y?
sequential_6/lstm_12/zeros/mulMul+sequential_6/lstm_12/strided_slice:output:0)sequential_6/lstm_12/zeros/mul/y:output:0*
T0*
_output_shapes
: 2 
sequential_6/lstm_12/zeros/mul?
!sequential_6/lstm_12/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2#
!sequential_6/lstm_12/zeros/Less/y?
sequential_6/lstm_12/zeros/LessLess"sequential_6/lstm_12/zeros/mul:z:0*sequential_6/lstm_12/zeros/Less/y:output:0*
T0*
_output_shapes
: 2!
sequential_6/lstm_12/zeros/Less?
#sequential_6/lstm_12/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2%
#sequential_6/lstm_12/zeros/packed/1?
!sequential_6/lstm_12/zeros/packedPack+sequential_6/lstm_12/strided_slice:output:0,sequential_6/lstm_12/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2#
!sequential_6/lstm_12/zeros/packed?
 sequential_6/lstm_12/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 sequential_6/lstm_12/zeros/Const?
sequential_6/lstm_12/zerosFill*sequential_6/lstm_12/zeros/packed:output:0)sequential_6/lstm_12/zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
sequential_6/lstm_12/zeros?
"sequential_6/lstm_12/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2$
"sequential_6/lstm_12/zeros_1/mul/y?
 sequential_6/lstm_12/zeros_1/mulMul+sequential_6/lstm_12/strided_slice:output:0+sequential_6/lstm_12/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2"
 sequential_6/lstm_12/zeros_1/mul?
#sequential_6/lstm_12/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2%
#sequential_6/lstm_12/zeros_1/Less/y?
!sequential_6/lstm_12/zeros_1/LessLess$sequential_6/lstm_12/zeros_1/mul:z:0,sequential_6/lstm_12/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2#
!sequential_6/lstm_12/zeros_1/Less?
%sequential_6/lstm_12/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2'
%sequential_6/lstm_12/zeros_1/packed/1?
#sequential_6/lstm_12/zeros_1/packedPack+sequential_6/lstm_12/strided_slice:output:0.sequential_6/lstm_12/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2%
#sequential_6/lstm_12/zeros_1/packed?
"sequential_6/lstm_12/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"sequential_6/lstm_12/zeros_1/Const?
sequential_6/lstm_12/zeros_1Fill,sequential_6/lstm_12/zeros_1/packed:output:0+sequential_6/lstm_12/zeros_1/Const:output:0*
T0*(
_output_shapes
:??????????2
sequential_6/lstm_12/zeros_1?
#sequential_6/lstm_12/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2%
#sequential_6/lstm_12/transpose/perm?
sequential_6/lstm_12/transpose	Transposelstm_12_input,sequential_6/lstm_12/transpose/perm:output:0*
T0*+
_output_shapes
:?????????]2 
sequential_6/lstm_12/transpose?
sequential_6/lstm_12/Shape_1Shape"sequential_6/lstm_12/transpose:y:0*
T0*
_output_shapes
:2
sequential_6/lstm_12/Shape_1?
*sequential_6/lstm_12/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2,
*sequential_6/lstm_12/strided_slice_1/stack?
,sequential_6/lstm_12/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_6/lstm_12/strided_slice_1/stack_1?
,sequential_6/lstm_12/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_6/lstm_12/strided_slice_1/stack_2?
$sequential_6/lstm_12/strided_slice_1StridedSlice%sequential_6/lstm_12/Shape_1:output:03sequential_6/lstm_12/strided_slice_1/stack:output:05sequential_6/lstm_12/strided_slice_1/stack_1:output:05sequential_6/lstm_12/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2&
$sequential_6/lstm_12/strided_slice_1?
0sequential_6/lstm_12/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????22
0sequential_6/lstm_12/TensorArrayV2/element_shape?
"sequential_6/lstm_12/TensorArrayV2TensorListReserve9sequential_6/lstm_12/TensorArrayV2/element_shape:output:0-sequential_6/lstm_12/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02$
"sequential_6/lstm_12/TensorArrayV2?
Jsequential_6/lstm_12/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????]   2L
Jsequential_6/lstm_12/TensorArrayUnstack/TensorListFromTensor/element_shape?
<sequential_6/lstm_12/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor"sequential_6/lstm_12/transpose:y:0Ssequential_6/lstm_12/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02>
<sequential_6/lstm_12/TensorArrayUnstack/TensorListFromTensor?
*sequential_6/lstm_12/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2,
*sequential_6/lstm_12/strided_slice_2/stack?
,sequential_6/lstm_12/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_6/lstm_12/strided_slice_2/stack_1?
,sequential_6/lstm_12/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_6/lstm_12/strided_slice_2/stack_2?
$sequential_6/lstm_12/strided_slice_2StridedSlice"sequential_6/lstm_12/transpose:y:03sequential_6/lstm_12/strided_slice_2/stack:output:05sequential_6/lstm_12/strided_slice_2/stack_1:output:05sequential_6/lstm_12/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????]*
shrink_axis_mask2&
$sequential_6/lstm_12/strided_slice_2?
7sequential_6/lstm_12/lstm_cell_12/MatMul/ReadVariableOpReadVariableOp@sequential_6_lstm_12_lstm_cell_12_matmul_readvariableop_resource*
_output_shapes
:	]?*
dtype029
7sequential_6/lstm_12/lstm_cell_12/MatMul/ReadVariableOp?
(sequential_6/lstm_12/lstm_cell_12/MatMulMatMul-sequential_6/lstm_12/strided_slice_2:output:0?sequential_6/lstm_12/lstm_cell_12/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2*
(sequential_6/lstm_12/lstm_cell_12/MatMul?
9sequential_6/lstm_12/lstm_cell_12/MatMul_1/ReadVariableOpReadVariableOpBsequential_6_lstm_12_lstm_cell_12_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02;
9sequential_6/lstm_12/lstm_cell_12/MatMul_1/ReadVariableOp?
*sequential_6/lstm_12/lstm_cell_12/MatMul_1MatMul#sequential_6/lstm_12/zeros:output:0Asequential_6/lstm_12/lstm_cell_12/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2,
*sequential_6/lstm_12/lstm_cell_12/MatMul_1?
%sequential_6/lstm_12/lstm_cell_12/addAddV22sequential_6/lstm_12/lstm_cell_12/MatMul:product:04sequential_6/lstm_12/lstm_cell_12/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2'
%sequential_6/lstm_12/lstm_cell_12/add?
8sequential_6/lstm_12/lstm_cell_12/BiasAdd/ReadVariableOpReadVariableOpAsequential_6_lstm_12_lstm_cell_12_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02:
8sequential_6/lstm_12/lstm_cell_12/BiasAdd/ReadVariableOp?
)sequential_6/lstm_12/lstm_cell_12/BiasAddBiasAdd)sequential_6/lstm_12/lstm_cell_12/add:z:0@sequential_6/lstm_12/lstm_cell_12/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2+
)sequential_6/lstm_12/lstm_cell_12/BiasAdd?
1sequential_6/lstm_12/lstm_cell_12/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :23
1sequential_6/lstm_12/lstm_cell_12/split/split_dim?
'sequential_6/lstm_12/lstm_cell_12/splitSplit:sequential_6/lstm_12/lstm_cell_12/split/split_dim:output:02sequential_6/lstm_12/lstm_cell_12/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split2)
'sequential_6/lstm_12/lstm_cell_12/split?
)sequential_6/lstm_12/lstm_cell_12/SigmoidSigmoid0sequential_6/lstm_12/lstm_cell_12/split:output:0*
T0*(
_output_shapes
:??????????2+
)sequential_6/lstm_12/lstm_cell_12/Sigmoid?
+sequential_6/lstm_12/lstm_cell_12/Sigmoid_1Sigmoid0sequential_6/lstm_12/lstm_cell_12/split:output:1*
T0*(
_output_shapes
:??????????2-
+sequential_6/lstm_12/lstm_cell_12/Sigmoid_1?
%sequential_6/lstm_12/lstm_cell_12/mulMul/sequential_6/lstm_12/lstm_cell_12/Sigmoid_1:y:0%sequential_6/lstm_12/zeros_1:output:0*
T0*(
_output_shapes
:??????????2'
%sequential_6/lstm_12/lstm_cell_12/mul?
&sequential_6/lstm_12/lstm_cell_12/ReluRelu0sequential_6/lstm_12/lstm_cell_12/split:output:2*
T0*(
_output_shapes
:??????????2(
&sequential_6/lstm_12/lstm_cell_12/Relu?
'sequential_6/lstm_12/lstm_cell_12/mul_1Mul-sequential_6/lstm_12/lstm_cell_12/Sigmoid:y:04sequential_6/lstm_12/lstm_cell_12/Relu:activations:0*
T0*(
_output_shapes
:??????????2)
'sequential_6/lstm_12/lstm_cell_12/mul_1?
'sequential_6/lstm_12/lstm_cell_12/add_1AddV2)sequential_6/lstm_12/lstm_cell_12/mul:z:0+sequential_6/lstm_12/lstm_cell_12/mul_1:z:0*
T0*(
_output_shapes
:??????????2)
'sequential_6/lstm_12/lstm_cell_12/add_1?
+sequential_6/lstm_12/lstm_cell_12/Sigmoid_2Sigmoid0sequential_6/lstm_12/lstm_cell_12/split:output:3*
T0*(
_output_shapes
:??????????2-
+sequential_6/lstm_12/lstm_cell_12/Sigmoid_2?
(sequential_6/lstm_12/lstm_cell_12/Relu_1Relu+sequential_6/lstm_12/lstm_cell_12/add_1:z:0*
T0*(
_output_shapes
:??????????2*
(sequential_6/lstm_12/lstm_cell_12/Relu_1?
'sequential_6/lstm_12/lstm_cell_12/mul_2Mul/sequential_6/lstm_12/lstm_cell_12/Sigmoid_2:y:06sequential_6/lstm_12/lstm_cell_12/Relu_1:activations:0*
T0*(
_output_shapes
:??????????2)
'sequential_6/lstm_12/lstm_cell_12/mul_2?
2sequential_6/lstm_12/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   24
2sequential_6/lstm_12/TensorArrayV2_1/element_shape?
$sequential_6/lstm_12/TensorArrayV2_1TensorListReserve;sequential_6/lstm_12/TensorArrayV2_1/element_shape:output:0-sequential_6/lstm_12/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02&
$sequential_6/lstm_12/TensorArrayV2_1x
sequential_6/lstm_12/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
sequential_6/lstm_12/time?
-sequential_6/lstm_12/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2/
-sequential_6/lstm_12/while/maximum_iterations?
'sequential_6/lstm_12/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2)
'sequential_6/lstm_12/while/loop_counter?
sequential_6/lstm_12/whileWhile0sequential_6/lstm_12/while/loop_counter:output:06sequential_6/lstm_12/while/maximum_iterations:output:0"sequential_6/lstm_12/time:output:0-sequential_6/lstm_12/TensorArrayV2_1:handle:0#sequential_6/lstm_12/zeros:output:0%sequential_6/lstm_12/zeros_1:output:0-sequential_6/lstm_12/strided_slice_1:output:0Lsequential_6/lstm_12/TensorArrayUnstack/TensorListFromTensor:output_handle:0@sequential_6_lstm_12_lstm_cell_12_matmul_readvariableop_resourceBsequential_6_lstm_12_lstm_cell_12_matmul_1_readvariableop_resourceAsequential_6_lstm_12_lstm_cell_12_biasadd_readvariableop_resource*
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
(sequential_6_lstm_12_while_body_26064301*4
cond,R*
(sequential_6_lstm_12_while_cond_26064300*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations 2
sequential_6/lstm_12/while?
Esequential_6/lstm_12/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2G
Esequential_6/lstm_12/TensorArrayV2Stack/TensorListStack/element_shape?
7sequential_6/lstm_12/TensorArrayV2Stack/TensorListStackTensorListStack#sequential_6/lstm_12/while:output:3Nsequential_6/lstm_12/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:??????????*
element_dtype029
7sequential_6/lstm_12/TensorArrayV2Stack/TensorListStack?
*sequential_6/lstm_12/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2,
*sequential_6/lstm_12/strided_slice_3/stack?
,sequential_6/lstm_12/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2.
,sequential_6/lstm_12/strided_slice_3/stack_1?
,sequential_6/lstm_12/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_6/lstm_12/strided_slice_3/stack_2?
$sequential_6/lstm_12/strided_slice_3StridedSlice@sequential_6/lstm_12/TensorArrayV2Stack/TensorListStack:tensor:03sequential_6/lstm_12/strided_slice_3/stack:output:05sequential_6/lstm_12/strided_slice_3/stack_1:output:05sequential_6/lstm_12/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2&
$sequential_6/lstm_12/strided_slice_3?
%sequential_6/lstm_12/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2'
%sequential_6/lstm_12/transpose_1/perm?
 sequential_6/lstm_12/transpose_1	Transpose@sequential_6/lstm_12/TensorArrayV2Stack/TensorListStack:tensor:0.sequential_6/lstm_12/transpose_1/perm:output:0*
T0*,
_output_shapes
:??????????2"
 sequential_6/lstm_12/transpose_1?
sequential_6/lstm_12/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
sequential_6/lstm_12/runtime?
 sequential_6/dropout_12/IdentityIdentity$sequential_6/lstm_12/transpose_1:y:0*
T0*,
_output_shapes
:??????????2"
 sequential_6/dropout_12/Identity?
sequential_6/lstm_13/ShapeShape)sequential_6/dropout_12/Identity:output:0*
T0*
_output_shapes
:2
sequential_6/lstm_13/Shape?
(sequential_6/lstm_13/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(sequential_6/lstm_13/strided_slice/stack?
*sequential_6/lstm_13/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*sequential_6/lstm_13/strided_slice/stack_1?
*sequential_6/lstm_13/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*sequential_6/lstm_13/strided_slice/stack_2?
"sequential_6/lstm_13/strided_sliceStridedSlice#sequential_6/lstm_13/Shape:output:01sequential_6/lstm_13/strided_slice/stack:output:03sequential_6/lstm_13/strided_slice/stack_1:output:03sequential_6/lstm_13/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"sequential_6/lstm_13/strided_slice?
 sequential_6/lstm_13/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2"
 sequential_6/lstm_13/zeros/mul/y?
sequential_6/lstm_13/zeros/mulMul+sequential_6/lstm_13/strided_slice:output:0)sequential_6/lstm_13/zeros/mul/y:output:0*
T0*
_output_shapes
: 2 
sequential_6/lstm_13/zeros/mul?
!sequential_6/lstm_13/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2#
!sequential_6/lstm_13/zeros/Less/y?
sequential_6/lstm_13/zeros/LessLess"sequential_6/lstm_13/zeros/mul:z:0*sequential_6/lstm_13/zeros/Less/y:output:0*
T0*
_output_shapes
: 2!
sequential_6/lstm_13/zeros/Less?
#sequential_6/lstm_13/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2%
#sequential_6/lstm_13/zeros/packed/1?
!sequential_6/lstm_13/zeros/packedPack+sequential_6/lstm_13/strided_slice:output:0,sequential_6/lstm_13/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2#
!sequential_6/lstm_13/zeros/packed?
 sequential_6/lstm_13/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 sequential_6/lstm_13/zeros/Const?
sequential_6/lstm_13/zerosFill*sequential_6/lstm_13/zeros/packed:output:0)sequential_6/lstm_13/zeros/Const:output:0*
T0*(
_output_shapes
:??????????2
sequential_6/lstm_13/zeros?
"sequential_6/lstm_13/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :?2$
"sequential_6/lstm_13/zeros_1/mul/y?
 sequential_6/lstm_13/zeros_1/mulMul+sequential_6/lstm_13/strided_slice:output:0+sequential_6/lstm_13/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2"
 sequential_6/lstm_13/zeros_1/mul?
#sequential_6/lstm_13/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2%
#sequential_6/lstm_13/zeros_1/Less/y?
!sequential_6/lstm_13/zeros_1/LessLess$sequential_6/lstm_13/zeros_1/mul:z:0,sequential_6/lstm_13/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2#
!sequential_6/lstm_13/zeros_1/Less?
%sequential_6/lstm_13/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :?2'
%sequential_6/lstm_13/zeros_1/packed/1?
#sequential_6/lstm_13/zeros_1/packedPack+sequential_6/lstm_13/strided_slice:output:0.sequential_6/lstm_13/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2%
#sequential_6/lstm_13/zeros_1/packed?
"sequential_6/lstm_13/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"sequential_6/lstm_13/zeros_1/Const?
sequential_6/lstm_13/zeros_1Fill,sequential_6/lstm_13/zeros_1/packed:output:0+sequential_6/lstm_13/zeros_1/Const:output:0*
T0*(
_output_shapes
:??????????2
sequential_6/lstm_13/zeros_1?
#sequential_6/lstm_13/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2%
#sequential_6/lstm_13/transpose/perm?
sequential_6/lstm_13/transpose	Transpose)sequential_6/dropout_12/Identity:output:0,sequential_6/lstm_13/transpose/perm:output:0*
T0*,
_output_shapes
:??????????2 
sequential_6/lstm_13/transpose?
sequential_6/lstm_13/Shape_1Shape"sequential_6/lstm_13/transpose:y:0*
T0*
_output_shapes
:2
sequential_6/lstm_13/Shape_1?
*sequential_6/lstm_13/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2,
*sequential_6/lstm_13/strided_slice_1/stack?
,sequential_6/lstm_13/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_6/lstm_13/strided_slice_1/stack_1?
,sequential_6/lstm_13/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_6/lstm_13/strided_slice_1/stack_2?
$sequential_6/lstm_13/strided_slice_1StridedSlice%sequential_6/lstm_13/Shape_1:output:03sequential_6/lstm_13/strided_slice_1/stack:output:05sequential_6/lstm_13/strided_slice_1/stack_1:output:05sequential_6/lstm_13/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2&
$sequential_6/lstm_13/strided_slice_1?
0sequential_6/lstm_13/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????22
0sequential_6/lstm_13/TensorArrayV2/element_shape?
"sequential_6/lstm_13/TensorArrayV2TensorListReserve9sequential_6/lstm_13/TensorArrayV2/element_shape:output:0-sequential_6/lstm_13/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02$
"sequential_6/lstm_13/TensorArrayV2?
Jsequential_6/lstm_13/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2L
Jsequential_6/lstm_13/TensorArrayUnstack/TensorListFromTensor/element_shape?
<sequential_6/lstm_13/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor"sequential_6/lstm_13/transpose:y:0Ssequential_6/lstm_13/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02>
<sequential_6/lstm_13/TensorArrayUnstack/TensorListFromTensor?
*sequential_6/lstm_13/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2,
*sequential_6/lstm_13/strided_slice_2/stack?
,sequential_6/lstm_13/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_6/lstm_13/strided_slice_2/stack_1?
,sequential_6/lstm_13/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_6/lstm_13/strided_slice_2/stack_2?
$sequential_6/lstm_13/strided_slice_2StridedSlice"sequential_6/lstm_13/transpose:y:03sequential_6/lstm_13/strided_slice_2/stack:output:05sequential_6/lstm_13/strided_slice_2/stack_1:output:05sequential_6/lstm_13/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2&
$sequential_6/lstm_13/strided_slice_2?
7sequential_6/lstm_13/lstm_cell_13/MatMul/ReadVariableOpReadVariableOp@sequential_6_lstm_13_lstm_cell_13_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype029
7sequential_6/lstm_13/lstm_cell_13/MatMul/ReadVariableOp?
(sequential_6/lstm_13/lstm_cell_13/MatMulMatMul-sequential_6/lstm_13/strided_slice_2:output:0?sequential_6/lstm_13/lstm_cell_13/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2*
(sequential_6/lstm_13/lstm_cell_13/MatMul?
9sequential_6/lstm_13/lstm_cell_13/MatMul_1/ReadVariableOpReadVariableOpBsequential_6_lstm_13_lstm_cell_13_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02;
9sequential_6/lstm_13/lstm_cell_13/MatMul_1/ReadVariableOp?
*sequential_6/lstm_13/lstm_cell_13/MatMul_1MatMul#sequential_6/lstm_13/zeros:output:0Asequential_6/lstm_13/lstm_cell_13/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2,
*sequential_6/lstm_13/lstm_cell_13/MatMul_1?
%sequential_6/lstm_13/lstm_cell_13/addAddV22sequential_6/lstm_13/lstm_cell_13/MatMul:product:04sequential_6/lstm_13/lstm_cell_13/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2'
%sequential_6/lstm_13/lstm_cell_13/add?
8sequential_6/lstm_13/lstm_cell_13/BiasAdd/ReadVariableOpReadVariableOpAsequential_6_lstm_13_lstm_cell_13_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02:
8sequential_6/lstm_13/lstm_cell_13/BiasAdd/ReadVariableOp?
)sequential_6/lstm_13/lstm_cell_13/BiasAddBiasAdd)sequential_6/lstm_13/lstm_cell_13/add:z:0@sequential_6/lstm_13/lstm_cell_13/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2+
)sequential_6/lstm_13/lstm_cell_13/BiasAdd?
1sequential_6/lstm_13/lstm_cell_13/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :23
1sequential_6/lstm_13/lstm_cell_13/split/split_dim?
'sequential_6/lstm_13/lstm_cell_13/splitSplit:sequential_6/lstm_13/lstm_cell_13/split/split_dim:output:02sequential_6/lstm_13/lstm_cell_13/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split2)
'sequential_6/lstm_13/lstm_cell_13/split?
)sequential_6/lstm_13/lstm_cell_13/SigmoidSigmoid0sequential_6/lstm_13/lstm_cell_13/split:output:0*
T0*(
_output_shapes
:??????????2+
)sequential_6/lstm_13/lstm_cell_13/Sigmoid?
+sequential_6/lstm_13/lstm_cell_13/Sigmoid_1Sigmoid0sequential_6/lstm_13/lstm_cell_13/split:output:1*
T0*(
_output_shapes
:??????????2-
+sequential_6/lstm_13/lstm_cell_13/Sigmoid_1?
%sequential_6/lstm_13/lstm_cell_13/mulMul/sequential_6/lstm_13/lstm_cell_13/Sigmoid_1:y:0%sequential_6/lstm_13/zeros_1:output:0*
T0*(
_output_shapes
:??????????2'
%sequential_6/lstm_13/lstm_cell_13/mul?
&sequential_6/lstm_13/lstm_cell_13/ReluRelu0sequential_6/lstm_13/lstm_cell_13/split:output:2*
T0*(
_output_shapes
:??????????2(
&sequential_6/lstm_13/lstm_cell_13/Relu?
'sequential_6/lstm_13/lstm_cell_13/mul_1Mul-sequential_6/lstm_13/lstm_cell_13/Sigmoid:y:04sequential_6/lstm_13/lstm_cell_13/Relu:activations:0*
T0*(
_output_shapes
:??????????2)
'sequential_6/lstm_13/lstm_cell_13/mul_1?
'sequential_6/lstm_13/lstm_cell_13/add_1AddV2)sequential_6/lstm_13/lstm_cell_13/mul:z:0+sequential_6/lstm_13/lstm_cell_13/mul_1:z:0*
T0*(
_output_shapes
:??????????2)
'sequential_6/lstm_13/lstm_cell_13/add_1?
+sequential_6/lstm_13/lstm_cell_13/Sigmoid_2Sigmoid0sequential_6/lstm_13/lstm_cell_13/split:output:3*
T0*(
_output_shapes
:??????????2-
+sequential_6/lstm_13/lstm_cell_13/Sigmoid_2?
(sequential_6/lstm_13/lstm_cell_13/Relu_1Relu+sequential_6/lstm_13/lstm_cell_13/add_1:z:0*
T0*(
_output_shapes
:??????????2*
(sequential_6/lstm_13/lstm_cell_13/Relu_1?
'sequential_6/lstm_13/lstm_cell_13/mul_2Mul/sequential_6/lstm_13/lstm_cell_13/Sigmoid_2:y:06sequential_6/lstm_13/lstm_cell_13/Relu_1:activations:0*
T0*(
_output_shapes
:??????????2)
'sequential_6/lstm_13/lstm_cell_13/mul_2?
2sequential_6/lstm_13/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   24
2sequential_6/lstm_13/TensorArrayV2_1/element_shape?
$sequential_6/lstm_13/TensorArrayV2_1TensorListReserve;sequential_6/lstm_13/TensorArrayV2_1/element_shape:output:0-sequential_6/lstm_13/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02&
$sequential_6/lstm_13/TensorArrayV2_1x
sequential_6/lstm_13/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
sequential_6/lstm_13/time?
-sequential_6/lstm_13/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2/
-sequential_6/lstm_13/while/maximum_iterations?
'sequential_6/lstm_13/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2)
'sequential_6/lstm_13/while/loop_counter?
sequential_6/lstm_13/whileWhile0sequential_6/lstm_13/while/loop_counter:output:06sequential_6/lstm_13/while/maximum_iterations:output:0"sequential_6/lstm_13/time:output:0-sequential_6/lstm_13/TensorArrayV2_1:handle:0#sequential_6/lstm_13/zeros:output:0%sequential_6/lstm_13/zeros_1:output:0-sequential_6/lstm_13/strided_slice_1:output:0Lsequential_6/lstm_13/TensorArrayUnstack/TensorListFromTensor:output_handle:0@sequential_6_lstm_13_lstm_cell_13_matmul_readvariableop_resourceBsequential_6_lstm_13_lstm_cell_13_matmul_1_readvariableop_resourceAsequential_6_lstm_13_lstm_cell_13_biasadd_readvariableop_resource*
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
(sequential_6_lstm_13_while_body_26064449*4
cond,R*
(sequential_6_lstm_13_while_cond_26064448*M
output_shapes<
:: : : : :??????????:??????????: : : : : *
parallel_iterations 2
sequential_6/lstm_13/while?
Esequential_6/lstm_13/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   2G
Esequential_6/lstm_13/TensorArrayV2Stack/TensorListStack/element_shape?
7sequential_6/lstm_13/TensorArrayV2Stack/TensorListStackTensorListStack#sequential_6/lstm_13/while:output:3Nsequential_6/lstm_13/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:??????????*
element_dtype029
7sequential_6/lstm_13/TensorArrayV2Stack/TensorListStack?
*sequential_6/lstm_13/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2,
*sequential_6/lstm_13/strided_slice_3/stack?
,sequential_6/lstm_13/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2.
,sequential_6/lstm_13/strided_slice_3/stack_1?
,sequential_6/lstm_13/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_6/lstm_13/strided_slice_3/stack_2?
$sequential_6/lstm_13/strided_slice_3StridedSlice@sequential_6/lstm_13/TensorArrayV2Stack/TensorListStack:tensor:03sequential_6/lstm_13/strided_slice_3/stack:output:05sequential_6/lstm_13/strided_slice_3/stack_1:output:05sequential_6/lstm_13/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2&
$sequential_6/lstm_13/strided_slice_3?
%sequential_6/lstm_13/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2'
%sequential_6/lstm_13/transpose_1/perm?
 sequential_6/lstm_13/transpose_1	Transpose@sequential_6/lstm_13/TensorArrayV2Stack/TensorListStack:tensor:0.sequential_6/lstm_13/transpose_1/perm:output:0*
T0*,
_output_shapes
:??????????2"
 sequential_6/lstm_13/transpose_1?
sequential_6/lstm_13/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
sequential_6/lstm_13/runtime?
 sequential_6/dropout_13/IdentityIdentity$sequential_6/lstm_13/transpose_1:y:0*
T0*,
_output_shapes
:??????????2"
 sequential_6/dropout_13/Identity?
-sequential_6/dense_6/Tensordot/ReadVariableOpReadVariableOp6sequential_6_dense_6_tensordot_readvariableop_resource*
_output_shapes
:	?*
dtype02/
-sequential_6/dense_6/Tensordot/ReadVariableOp?
#sequential_6/dense_6/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2%
#sequential_6/dense_6/Tensordot/axes?
#sequential_6/dense_6/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2%
#sequential_6/dense_6/Tensordot/free?
$sequential_6/dense_6/Tensordot/ShapeShape)sequential_6/dropout_13/Identity:output:0*
T0*
_output_shapes
:2&
$sequential_6/dense_6/Tensordot/Shape?
,sequential_6/dense_6/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,sequential_6/dense_6/Tensordot/GatherV2/axis?
'sequential_6/dense_6/Tensordot/GatherV2GatherV2-sequential_6/dense_6/Tensordot/Shape:output:0,sequential_6/dense_6/Tensordot/free:output:05sequential_6/dense_6/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2)
'sequential_6/dense_6/Tensordot/GatherV2?
.sequential_6/dense_6/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.sequential_6/dense_6/Tensordot/GatherV2_1/axis?
)sequential_6/dense_6/Tensordot/GatherV2_1GatherV2-sequential_6/dense_6/Tensordot/Shape:output:0,sequential_6/dense_6/Tensordot/axes:output:07sequential_6/dense_6/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2+
)sequential_6/dense_6/Tensordot/GatherV2_1?
$sequential_6/dense_6/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2&
$sequential_6/dense_6/Tensordot/Const?
#sequential_6/dense_6/Tensordot/ProdProd0sequential_6/dense_6/Tensordot/GatherV2:output:0-sequential_6/dense_6/Tensordot/Const:output:0*
T0*
_output_shapes
: 2%
#sequential_6/dense_6/Tensordot/Prod?
&sequential_6/dense_6/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2(
&sequential_6/dense_6/Tensordot/Const_1?
%sequential_6/dense_6/Tensordot/Prod_1Prod2sequential_6/dense_6/Tensordot/GatherV2_1:output:0/sequential_6/dense_6/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2'
%sequential_6/dense_6/Tensordot/Prod_1?
*sequential_6/dense_6/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2,
*sequential_6/dense_6/Tensordot/concat/axis?
%sequential_6/dense_6/Tensordot/concatConcatV2,sequential_6/dense_6/Tensordot/free:output:0,sequential_6/dense_6/Tensordot/axes:output:03sequential_6/dense_6/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2'
%sequential_6/dense_6/Tensordot/concat?
$sequential_6/dense_6/Tensordot/stackPack,sequential_6/dense_6/Tensordot/Prod:output:0.sequential_6/dense_6/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2&
$sequential_6/dense_6/Tensordot/stack?
(sequential_6/dense_6/Tensordot/transpose	Transpose)sequential_6/dropout_13/Identity:output:0.sequential_6/dense_6/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????2*
(sequential_6/dense_6/Tensordot/transpose?
&sequential_6/dense_6/Tensordot/ReshapeReshape,sequential_6/dense_6/Tensordot/transpose:y:0-sequential_6/dense_6/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2(
&sequential_6/dense_6/Tensordot/Reshape?
%sequential_6/dense_6/Tensordot/MatMulMatMul/sequential_6/dense_6/Tensordot/Reshape:output:05sequential_6/dense_6/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2'
%sequential_6/dense_6/Tensordot/MatMul?
&sequential_6/dense_6/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&sequential_6/dense_6/Tensordot/Const_2?
,sequential_6/dense_6/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,sequential_6/dense_6/Tensordot/concat_1/axis?
'sequential_6/dense_6/Tensordot/concat_1ConcatV20sequential_6/dense_6/Tensordot/GatherV2:output:0/sequential_6/dense_6/Tensordot/Const_2:output:05sequential_6/dense_6/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2)
'sequential_6/dense_6/Tensordot/concat_1?
sequential_6/dense_6/TensordotReshape/sequential_6/dense_6/Tensordot/MatMul:product:00sequential_6/dense_6/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????2 
sequential_6/dense_6/Tensordot?
+sequential_6/dense_6/BiasAdd/ReadVariableOpReadVariableOp4sequential_6_dense_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02-
+sequential_6/dense_6/BiasAdd/ReadVariableOp?
sequential_6/dense_6/BiasAddBiasAdd'sequential_6/dense_6/Tensordot:output:03sequential_6/dense_6/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????2
sequential_6/dense_6/BiasAdd?
sequential_6/dense_6/SoftmaxSoftmax%sequential_6/dense_6/BiasAdd:output:0*
T0*+
_output_shapes
:?????????2
sequential_6/dense_6/Softmax?
IdentityIdentity&sequential_6/dense_6/Softmax:softmax:0^NoOp*
T0*+
_output_shapes
:?????????2

Identity?
NoOpNoOp,^sequential_6/dense_6/BiasAdd/ReadVariableOp.^sequential_6/dense_6/Tensordot/ReadVariableOp9^sequential_6/lstm_12/lstm_cell_12/BiasAdd/ReadVariableOp8^sequential_6/lstm_12/lstm_cell_12/MatMul/ReadVariableOp:^sequential_6/lstm_12/lstm_cell_12/MatMul_1/ReadVariableOp^sequential_6/lstm_12/while9^sequential_6/lstm_13/lstm_cell_13/BiasAdd/ReadVariableOp8^sequential_6/lstm_13/lstm_cell_13/MatMul/ReadVariableOp:^sequential_6/lstm_13/lstm_cell_13/MatMul_1/ReadVariableOp^sequential_6/lstm_13/while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????]: : : : : : : : 2Z
+sequential_6/dense_6/BiasAdd/ReadVariableOp+sequential_6/dense_6/BiasAdd/ReadVariableOp2^
-sequential_6/dense_6/Tensordot/ReadVariableOp-sequential_6/dense_6/Tensordot/ReadVariableOp2t
8sequential_6/lstm_12/lstm_cell_12/BiasAdd/ReadVariableOp8sequential_6/lstm_12/lstm_cell_12/BiasAdd/ReadVariableOp2r
7sequential_6/lstm_12/lstm_cell_12/MatMul/ReadVariableOp7sequential_6/lstm_12/lstm_cell_12/MatMul/ReadVariableOp2v
9sequential_6/lstm_12/lstm_cell_12/MatMul_1/ReadVariableOp9sequential_6/lstm_12/lstm_cell_12/MatMul_1/ReadVariableOp28
sequential_6/lstm_12/whilesequential_6/lstm_12/while2t
8sequential_6/lstm_13/lstm_cell_13/BiasAdd/ReadVariableOp8sequential_6/lstm_13/lstm_cell_13/BiasAdd/ReadVariableOp2r
7sequential_6/lstm_13/lstm_cell_13/MatMul/ReadVariableOp7sequential_6/lstm_13/lstm_cell_13/MatMul/ReadVariableOp2v
9sequential_6/lstm_13/lstm_cell_13/MatMul_1/ReadVariableOp9sequential_6/lstm_13/lstm_cell_13/MatMul_1/ReadVariableOp28
sequential_6/lstm_13/whilesequential_6/lstm_13/while:Z V
+
_output_shapes
:?????????]
'
_user_specified_namelstm_12_input
??
?
while_body_26068538
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
3while_lstm_cell_13_matmul_readvariableop_resource_0:
??I
5while_lstm_cell_13_matmul_1_readvariableop_resource_0:
??C
4while_lstm_cell_13_biasadd_readvariableop_resource_0:	?
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
1while_lstm_cell_13_matmul_readvariableop_resource:
??G
3while_lstm_cell_13_matmul_1_readvariableop_resource:
??A
2while_lstm_cell_13_biasadd_readvariableop_resource:	???)while/lstm_cell_13/BiasAdd/ReadVariableOp?(while/lstm_cell_13/MatMul/ReadVariableOp?*while/lstm_cell_13/MatMul_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:??????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
(while/lstm_cell_13/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_13_matmul_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02*
(while/lstm_cell_13/MatMul/ReadVariableOp?
while/lstm_cell_13/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_13/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_13/MatMul?
*while/lstm_cell_13/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_13_matmul_1_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02,
*while/lstm_cell_13/MatMul_1/ReadVariableOp?
while/lstm_cell_13/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_13/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_13/MatMul_1?
while/lstm_cell_13/addAddV2#while/lstm_cell_13/MatMul:product:0%while/lstm_cell_13/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_13/add?
)while/lstm_cell_13/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_13_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype02+
)while/lstm_cell_13/BiasAdd/ReadVariableOp?
while/lstm_cell_13/BiasAddBiasAddwhile/lstm_cell_13/add:z:01while/lstm_cell_13/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_13/BiasAdd?
"while/lstm_cell_13/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_13/split/split_dim?
while/lstm_cell_13/splitSplit+while/lstm_cell_13/split/split_dim:output:0#while/lstm_cell_13/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split2
while/lstm_cell_13/split?
while/lstm_cell_13/SigmoidSigmoid!while/lstm_cell_13/split:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_13/Sigmoid?
while/lstm_cell_13/Sigmoid_1Sigmoid!while/lstm_cell_13/split:output:1*
T0*(
_output_shapes
:??????????2
while/lstm_cell_13/Sigmoid_1?
while/lstm_cell_13/mulMul while/lstm_cell_13/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:??????????2
while/lstm_cell_13/mul?
while/lstm_cell_13/ReluRelu!while/lstm_cell_13/split:output:2*
T0*(
_output_shapes
:??????????2
while/lstm_cell_13/Relu?
while/lstm_cell_13/mul_1Mulwhile/lstm_cell_13/Sigmoid:y:0%while/lstm_cell_13/Relu:activations:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_13/mul_1?
while/lstm_cell_13/add_1AddV2while/lstm_cell_13/mul:z:0while/lstm_cell_13/mul_1:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_13/add_1?
while/lstm_cell_13/Sigmoid_2Sigmoid!while/lstm_cell_13/split:output:3*
T0*(
_output_shapes
:??????????2
while/lstm_cell_13/Sigmoid_2?
while/lstm_cell_13/Relu_1Reluwhile/lstm_cell_13/add_1:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_13/Relu_1?
while/lstm_cell_13/mul_2Mul while/lstm_cell_13/Sigmoid_2:y:0'while/lstm_cell_13/Relu_1:activations:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_13/mul_2?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_13/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_13/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:??????????2
while/Identity_4?
while/Identity_5Identitywhile/lstm_cell_13/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:??????????2
while/Identity_5?

while/NoOpNoOp*^while/lstm_cell_13/BiasAdd/ReadVariableOp)^while/lstm_cell_13/MatMul/ReadVariableOp+^while/lstm_cell_13/MatMul_1/ReadVariableOp*"
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
2while_lstm_cell_13_biasadd_readvariableop_resource4while_lstm_cell_13_biasadd_readvariableop_resource_0"l
3while_lstm_cell_13_matmul_1_readvariableop_resource5while_lstm_cell_13_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_13_matmul_readvariableop_resource3while_lstm_cell_13_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :??????????:??????????: : : : : 2V
)while/lstm_cell_13/BiasAdd/ReadVariableOp)while/lstm_cell_13/BiasAdd/ReadVariableOp2T
(while/lstm_cell_13/MatMul/ReadVariableOp(while/lstm_cell_13/MatMul/ReadVariableOp2X
*while/lstm_cell_13/MatMul_1/ReadVariableOp*while/lstm_cell_13/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
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
?
J__inference_lstm_cell_12_layer_call_and_return_conditional_losses_26068916

inputs
states_0
states_11
matmul_readvariableop_resource:	]?4
 matmul_1_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity

identity_1

identity_2??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?MatMul_1/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	]?*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul_1/ReadVariableOp|
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
add?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
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
_construction_contextkEagerRuntime*T
_input_shapesC
A:?????????]:??????????:??????????: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:?????????]
 
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
while_cond_26068688
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_26068688___redundant_placeholder06
2while_while_cond_26068688___redundant_placeholder16
2while_while_cond_26068688___redundant_placeholder26
2while_while_cond_26068688___redundant_placeholder3
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
?F
?
E__inference_lstm_13_layer_call_and_return_conditional_losses_26065559

inputs)
lstm_cell_13_26065477:
??)
lstm_cell_13_26065479:
??$
lstm_cell_13_26065481:	?
identity??$lstm_cell_13/StatefulPartitionedCall?whileD
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
!:???????????????????2
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
valueB"?????   27
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
:??????????*
shrink_axis_mask2
strided_slice_2?
$lstm_cell_13/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_13_26065477lstm_cell_13_26065479lstm_cell_13_26065481*
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
J__inference_lstm_cell_13_layer_call_and_return_conditional_losses_260654122&
$lstm_cell_13/StatefulPartitionedCall?
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_13_26065477lstm_cell_13_26065479lstm_cell_13_26065481*
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
while_body_26065490*
condR
while_cond_26065489*M
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
NoOpNoOp%^lstm_cell_13/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':???????????????????: : : 2L
$lstm_cell_13/StatefulPartitionedCall$lstm_cell_13/StatefulPartitionedCall2
whilewhile:] Y
5
_output_shapes#
!:???????????????????
 
_user_specified_nameinputs
?
?
J__inference_lstm_cell_13_layer_call_and_return_conditional_losses_26065266

inputs

states
states_12
matmul_readvariableop_resource:
??4
 matmul_1_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity

identity_1

identity_2??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?MatMul_1/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
add?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
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
B:??????????:??????????:??????????: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:PL
(
_output_shapes
:??????????
 
_user_specified_namestates:PL
(
_output_shapes
:??????????
 
_user_specified_namestates
?
?
while_cond_26064859
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_26064859___redundant_placeholder06
2while_while_cond_26064859___redundant_placeholder16
2while_while_cond_26064859___redundant_placeholder26
2while_while_cond_26064859___redundant_placeholder3
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
E__inference_lstm_13_layer_call_and_return_conditional_losses_26068320
inputs_0?
+lstm_cell_13_matmul_readvariableop_resource:
??A
-lstm_cell_13_matmul_1_readvariableop_resource:
??;
,lstm_cell_13_biasadd_readvariableop_resource:	?
identity??#lstm_cell_13/BiasAdd/ReadVariableOp?"lstm_cell_13/MatMul/ReadVariableOp?$lstm_cell_13/MatMul_1/ReadVariableOp?whileF
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
!:???????????????????2
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
valueB"?????   27
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
:??????????*
shrink_axis_mask2
strided_slice_2?
"lstm_cell_13/MatMul/ReadVariableOpReadVariableOp+lstm_cell_13_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02$
"lstm_cell_13/MatMul/ReadVariableOp?
lstm_cell_13/MatMulMatMulstrided_slice_2:output:0*lstm_cell_13/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_13/MatMul?
$lstm_cell_13/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_13_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02&
$lstm_cell_13/MatMul_1/ReadVariableOp?
lstm_cell_13/MatMul_1MatMulzeros:output:0,lstm_cell_13/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_13/MatMul_1?
lstm_cell_13/addAddV2lstm_cell_13/MatMul:product:0lstm_cell_13/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell_13/add?
#lstm_cell_13/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_13_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02%
#lstm_cell_13/BiasAdd/ReadVariableOp?
lstm_cell_13/BiasAddBiasAddlstm_cell_13/add:z:0+lstm_cell_13/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_13/BiasAdd~
lstm_cell_13/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_13/split/split_dim?
lstm_cell_13/splitSplit%lstm_cell_13/split/split_dim:output:0lstm_cell_13/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split2
lstm_cell_13/split?
lstm_cell_13/SigmoidSigmoidlstm_cell_13/split:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_13/Sigmoid?
lstm_cell_13/Sigmoid_1Sigmoidlstm_cell_13/split:output:1*
T0*(
_output_shapes
:??????????2
lstm_cell_13/Sigmoid_1?
lstm_cell_13/mulMullstm_cell_13/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_13/mul~
lstm_cell_13/ReluRelulstm_cell_13/split:output:2*
T0*(
_output_shapes
:??????????2
lstm_cell_13/Relu?
lstm_cell_13/mul_1Mullstm_cell_13/Sigmoid:y:0lstm_cell_13/Relu:activations:0*
T0*(
_output_shapes
:??????????2
lstm_cell_13/mul_1?
lstm_cell_13/add_1AddV2lstm_cell_13/mul:z:0lstm_cell_13/mul_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell_13/add_1?
lstm_cell_13/Sigmoid_2Sigmoidlstm_cell_13/split:output:3*
T0*(
_output_shapes
:??????????2
lstm_cell_13/Sigmoid_2}
lstm_cell_13/Relu_1Relulstm_cell_13/add_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell_13/Relu_1?
lstm_cell_13/mul_2Mullstm_cell_13/Sigmoid_2:y:0!lstm_cell_13/Relu_1:activations:0*
T0*(
_output_shapes
:??????????2
lstm_cell_13/mul_2?
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_13_matmul_readvariableop_resource-lstm_cell_13_matmul_1_readvariableop_resource,lstm_cell_13_biasadd_readvariableop_resource*
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
while_body_26068236*
condR
while_cond_26068235*M
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
NoOpNoOp$^lstm_cell_13/BiasAdd/ReadVariableOp#^lstm_cell_13/MatMul/ReadVariableOp%^lstm_cell_13/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':???????????????????: : : 2J
#lstm_cell_13/BiasAdd/ReadVariableOp#lstm_cell_13/BiasAdd/ReadVariableOp2H
"lstm_cell_13/MatMul/ReadVariableOp"lstm_cell_13/MatMul/ReadVariableOp2L
$lstm_cell_13/MatMul_1/ReadVariableOp$lstm_cell_13/MatMul_1/ReadVariableOp2
whilewhile:_ [
5
_output_shapes#
!:???????????????????
"
_user_specified_name
inputs/0
?	
?
&__inference_signature_wrapper_26066784
lstm_12_input
unknown:	]?
	unknown_0:
??
	unknown_1:	?
	unknown_2:
??
	unknown_3:
??
	unknown_4:	?
	unknown_5:	?
	unknown_6:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalllstm_12_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
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
#__inference__wrapped_model_260645612
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
_user_specified_namelstm_12_input
?&
?
while_body_26065280
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_01
while_lstm_cell_13_26065304_0:
??1
while_lstm_cell_13_26065306_0:
??,
while_lstm_cell_13_26065308_0:	?
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor/
while_lstm_cell_13_26065304:
??/
while_lstm_cell_13_26065306:
??*
while_lstm_cell_13_26065308:	???*while/lstm_cell_13/StatefulPartitionedCall?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"?????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:??????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
*while/lstm_cell_13/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_13_26065304_0while_lstm_cell_13_26065306_0while_lstm_cell_13_26065308_0*
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
J__inference_lstm_cell_13_layer_call_and_return_conditional_losses_260652662,
*while/lstm_cell_13/StatefulPartitionedCall?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder3while/lstm_cell_13/StatefulPartitionedCall:output:0*
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
while/Identity_4Identity3while/lstm_cell_13/StatefulPartitionedCall:output:1^while/NoOp*
T0*(
_output_shapes
:??????????2
while/Identity_4?
while/Identity_5Identity3while/lstm_cell_13/StatefulPartitionedCall:output:2^while/NoOp*
T0*(
_output_shapes
:??????????2
while/Identity_5?

while/NoOpNoOp+^while/lstm_cell_13/StatefulPartitionedCall*"
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
while_lstm_cell_13_26065304while_lstm_cell_13_26065304_0"<
while_lstm_cell_13_26065306while_lstm_cell_13_26065306_0"<
while_lstm_cell_13_26065308while_lstm_cell_13_26065308_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :??????????:??????????: : : : : 2X
*while/lstm_cell_13/StatefulPartitionedCall*while/lstm_cell_13/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
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
lstm_12_while_cond_26066850,
(lstm_12_while_lstm_12_while_loop_counter2
.lstm_12_while_lstm_12_while_maximum_iterations
lstm_12_while_placeholder
lstm_12_while_placeholder_1
lstm_12_while_placeholder_2
lstm_12_while_placeholder_3.
*lstm_12_while_less_lstm_12_strided_slice_1F
Blstm_12_while_lstm_12_while_cond_26066850___redundant_placeholder0F
Blstm_12_while_lstm_12_while_cond_26066850___redundant_placeholder1F
Blstm_12_while_lstm_12_while_cond_26066850___redundant_placeholder2F
Blstm_12_while_lstm_12_while_cond_26066850___redundant_placeholder3
lstm_12_while_identity
?
lstm_12/while/LessLesslstm_12_while_placeholder*lstm_12_while_less_lstm_12_strided_slice_1*
T0*
_output_shapes
: 2
lstm_12/while/Lessu
lstm_12/while/IdentityIdentitylstm_12/while/Less:z:0*
T0
*
_output_shapes
: 2
lstm_12/while/Identity"9
lstm_12_while_identitylstm_12/while/Identity:output:0*(
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
?
J__inference_lstm_cell_13_layer_call_and_return_conditional_losses_26065412

inputs

states
states_12
matmul_readvariableop_resource:
??4
 matmul_1_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity

identity_1

identity_2??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?MatMul_1/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
add?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
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
B:??????????:??????????:??????????: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:PL
(
_output_shapes
:??????????
 
_user_specified_namestates:PL
(
_output_shapes
:??????????
 
_user_specified_namestates
?\
?
E__inference_lstm_13_layer_call_and_return_conditional_losses_26068773

inputs?
+lstm_cell_13_matmul_readvariableop_resource:
??A
-lstm_cell_13_matmul_1_readvariableop_resource:
??;
,lstm_cell_13_biasadd_readvariableop_resource:	?
identity??#lstm_cell_13/BiasAdd/ReadVariableOp?"lstm_cell_13/MatMul/ReadVariableOp?$lstm_cell_13/MatMul_1/ReadVariableOp?whileD
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
:??????????2
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
valueB"?????   27
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
:??????????*
shrink_axis_mask2
strided_slice_2?
"lstm_cell_13/MatMul/ReadVariableOpReadVariableOp+lstm_cell_13_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02$
"lstm_cell_13/MatMul/ReadVariableOp?
lstm_cell_13/MatMulMatMulstrided_slice_2:output:0*lstm_cell_13/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_13/MatMul?
$lstm_cell_13/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_13_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02&
$lstm_cell_13/MatMul_1/ReadVariableOp?
lstm_cell_13/MatMul_1MatMulzeros:output:0,lstm_cell_13/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_13/MatMul_1?
lstm_cell_13/addAddV2lstm_cell_13/MatMul:product:0lstm_cell_13/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell_13/add?
#lstm_cell_13/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_13_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02%
#lstm_cell_13/BiasAdd/ReadVariableOp?
lstm_cell_13/BiasAddBiasAddlstm_cell_13/add:z:0+lstm_cell_13/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_13/BiasAdd~
lstm_cell_13/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_13/split/split_dim?
lstm_cell_13/splitSplit%lstm_cell_13/split/split_dim:output:0lstm_cell_13/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split2
lstm_cell_13/split?
lstm_cell_13/SigmoidSigmoidlstm_cell_13/split:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_13/Sigmoid?
lstm_cell_13/Sigmoid_1Sigmoidlstm_cell_13/split:output:1*
T0*(
_output_shapes
:??????????2
lstm_cell_13/Sigmoid_1?
lstm_cell_13/mulMullstm_cell_13/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_13/mul~
lstm_cell_13/ReluRelulstm_cell_13/split:output:2*
T0*(
_output_shapes
:??????????2
lstm_cell_13/Relu?
lstm_cell_13/mul_1Mullstm_cell_13/Sigmoid:y:0lstm_cell_13/Relu:activations:0*
T0*(
_output_shapes
:??????????2
lstm_cell_13/mul_1?
lstm_cell_13/add_1AddV2lstm_cell_13/mul:z:0lstm_cell_13/mul_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell_13/add_1?
lstm_cell_13/Sigmoid_2Sigmoidlstm_cell_13/split:output:3*
T0*(
_output_shapes
:??????????2
lstm_cell_13/Sigmoid_2}
lstm_cell_13/Relu_1Relulstm_cell_13/add_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell_13/Relu_1?
lstm_cell_13/mul_2Mullstm_cell_13/Sigmoid_2:y:0!lstm_cell_13/Relu_1:activations:0*
T0*(
_output_shapes
:??????????2
lstm_cell_13/mul_2?
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_13_matmul_readvariableop_resource-lstm_cell_13_matmul_1_readvariableop_resource,lstm_cell_13_biasadd_readvariableop_resource*
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
while_body_26068689*
condR
while_cond_26068688*M
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
NoOpNoOp$^lstm_cell_13/BiasAdd/ReadVariableOp#^lstm_cell_13/MatMul/ReadVariableOp%^lstm_cell_13/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:??????????: : : 2J
#lstm_cell_13/BiasAdd/ReadVariableOp#lstm_cell_13/BiasAdd/ReadVariableOp2H
"lstm_cell_13/MatMul/ReadVariableOp"lstm_cell_13/MatMul/ReadVariableOp2L
$lstm_cell_13/MatMul_1/ReadVariableOp$lstm_cell_13/MatMul_1/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
J__inference_sequential_6_layer_call_and_return_conditional_losses_26066755
lstm_12_input#
lstm_12_26066733:	]?$
lstm_12_26066735:
??
lstm_12_26066737:	?$
lstm_13_26066741:
??$
lstm_13_26066743:
??
lstm_13_26066745:	?#
dense_6_26066749:	?
dense_6_26066751:
identity??dense_6/StatefulPartitionedCall?"dropout_12/StatefulPartitionedCall?"dropout_13/StatefulPartitionedCall?lstm_12/StatefulPartitionedCall?lstm_13/StatefulPartitionedCall?
lstm_12/StatefulPartitionedCallStatefulPartitionedCalllstm_12_inputlstm_12_26066733lstm_12_26066735lstm_12_26066737*
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
E__inference_lstm_12_layer_call_and_return_conditional_losses_260666082!
lstm_12/StatefulPartitionedCall?
"dropout_12/StatefulPartitionedCallStatefulPartitionedCall(lstm_12/StatefulPartitionedCall:output:0*
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
H__inference_dropout_12_layer_call_and_return_conditional_losses_260664412$
"dropout_12/StatefulPartitionedCall?
lstm_13/StatefulPartitionedCallStatefulPartitionedCall+dropout_12/StatefulPartitionedCall:output:0lstm_13_26066741lstm_13_26066743lstm_13_26066745*
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
E__inference_lstm_13_layer_call_and_return_conditional_losses_260664122!
lstm_13/StatefulPartitionedCall?
"dropout_13/StatefulPartitionedCallStatefulPartitionedCall(lstm_13/StatefulPartitionedCall:output:0#^dropout_12/StatefulPartitionedCall*
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
H__inference_dropout_13_layer_call_and_return_conditional_losses_260662452$
"dropout_13/StatefulPartitionedCall?
dense_6/StatefulPartitionedCallStatefulPartitionedCall+dropout_13/StatefulPartitionedCall:output:0dense_6_26066749dense_6_26066751*
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
E__inference_dense_6_layer_call_and_return_conditional_losses_260661892!
dense_6/StatefulPartitionedCall?
IdentityIdentity(dense_6/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????2

Identity?
NoOpNoOp ^dense_6/StatefulPartitionedCall#^dropout_12/StatefulPartitionedCall#^dropout_13/StatefulPartitionedCall ^lstm_12/StatefulPartitionedCall ^lstm_13/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????]: : : : : : : : 2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2H
"dropout_12/StatefulPartitionedCall"dropout_12/StatefulPartitionedCall2H
"dropout_13/StatefulPartitionedCall"dropout_13/StatefulPartitionedCall2B
lstm_12/StatefulPartitionedCalllstm_12/StatefulPartitionedCall2B
lstm_13/StatefulPartitionedCalllstm_13/StatefulPartitionedCall:Z V
+
_output_shapes
:?????????]
'
_user_specified_namelstm_12_input
??
?
while_body_26067712
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_12_matmul_readvariableop_resource_0:	]?I
5while_lstm_cell_12_matmul_1_readvariableop_resource_0:
??C
4while_lstm_cell_12_biasadd_readvariableop_resource_0:	?
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_12_matmul_readvariableop_resource:	]?G
3while_lstm_cell_12_matmul_1_readvariableop_resource:
??A
2while_lstm_cell_12_biasadd_readvariableop_resource:	???)while/lstm_cell_12/BiasAdd/ReadVariableOp?(while/lstm_cell_12/MatMul/ReadVariableOp?*while/lstm_cell_12/MatMul_1/ReadVariableOp?
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
(while/lstm_cell_12/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_12_matmul_readvariableop_resource_0*
_output_shapes
:	]?*
dtype02*
(while/lstm_cell_12/MatMul/ReadVariableOp?
while/lstm_cell_12/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_12/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_12/MatMul?
*while/lstm_cell_12/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_12_matmul_1_readvariableop_resource_0* 
_output_shapes
:
??*
dtype02,
*while/lstm_cell_12/MatMul_1/ReadVariableOp?
while/lstm_cell_12/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_12/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_12/MatMul_1?
while/lstm_cell_12/addAddV2#while/lstm_cell_12/MatMul:product:0%while/lstm_cell_12/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_12/add?
)while/lstm_cell_12/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_12_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype02+
)while/lstm_cell_12/BiasAdd/ReadVariableOp?
while/lstm_cell_12/BiasAddBiasAddwhile/lstm_cell_12/add:z:01while/lstm_cell_12/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_12/BiasAdd?
"while/lstm_cell_12/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_12/split/split_dim?
while/lstm_cell_12/splitSplit+while/lstm_cell_12/split/split_dim:output:0#while/lstm_cell_12/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split2
while/lstm_cell_12/split?
while/lstm_cell_12/SigmoidSigmoid!while/lstm_cell_12/split:output:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_12/Sigmoid?
while/lstm_cell_12/Sigmoid_1Sigmoid!while/lstm_cell_12/split:output:1*
T0*(
_output_shapes
:??????????2
while/lstm_cell_12/Sigmoid_1?
while/lstm_cell_12/mulMul while/lstm_cell_12/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:??????????2
while/lstm_cell_12/mul?
while/lstm_cell_12/ReluRelu!while/lstm_cell_12/split:output:2*
T0*(
_output_shapes
:??????????2
while/lstm_cell_12/Relu?
while/lstm_cell_12/mul_1Mulwhile/lstm_cell_12/Sigmoid:y:0%while/lstm_cell_12/Relu:activations:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_12/mul_1?
while/lstm_cell_12/add_1AddV2while/lstm_cell_12/mul:z:0while/lstm_cell_12/mul_1:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_12/add_1?
while/lstm_cell_12/Sigmoid_2Sigmoid!while/lstm_cell_12/split:output:3*
T0*(
_output_shapes
:??????????2
while/lstm_cell_12/Sigmoid_2?
while/lstm_cell_12/Relu_1Reluwhile/lstm_cell_12/add_1:z:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_12/Relu_1?
while/lstm_cell_12/mul_2Mul while/lstm_cell_12/Sigmoid_2:y:0'while/lstm_cell_12/Relu_1:activations:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_12/mul_2?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_12/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_12/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:??????????2
while/Identity_4?
while/Identity_5Identitywhile/lstm_cell_12/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:??????????2
while/Identity_5?

while/NoOpNoOp*^while/lstm_cell_12/BiasAdd/ReadVariableOp)^while/lstm_cell_12/MatMul/ReadVariableOp+^while/lstm_cell_12/MatMul_1/ReadVariableOp*"
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
2while_lstm_cell_12_biasadd_readvariableop_resource4while_lstm_cell_12_biasadd_readvariableop_resource_0"l
3while_lstm_cell_12_matmul_1_readvariableop_resource5while_lstm_cell_12_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_12_matmul_readvariableop_resource3while_lstm_cell_12_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :??????????:??????????: : : : : 2V
)while/lstm_cell_12/BiasAdd/ReadVariableOp)while/lstm_cell_12/BiasAdd/ReadVariableOp2T
(while/lstm_cell_12/MatMul/ReadVariableOp(while/lstm_cell_12/MatMul/ReadVariableOp2X
*while/lstm_cell_12/MatMul_1/ReadVariableOp*while/lstm_cell_12/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
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
?
J__inference_lstm_cell_12_layer_call_and_return_conditional_losses_26064636

inputs

states
states_11
matmul_readvariableop_resource:	]?4
 matmul_1_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity

identity_1

identity_2??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?MatMul_1/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	]?*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
add?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
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
_construction_contextkEagerRuntime*T
_input_shapesC
A:?????????]:??????????:??????????: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:?????????]
 
_user_specified_nameinputs:PL
(
_output_shapes
:??????????
 
_user_specified_namestates:PL
(
_output_shapes
:??????????
 
_user_specified_namestates
?
?
while_cond_26064649
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_26064649___redundant_placeholder06
2while_while_cond_26064649___redundant_placeholder16
2while_while_cond_26064649___redundant_placeholder26
2while_while_cond_26064649___redundant_placeholder3
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
?
*__inference_lstm_12_layer_call_fn_26068131

inputs
unknown:	]?
	unknown_0:
??
	unknown_1:	?
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
E__inference_lstm_12_layer_call_and_return_conditional_losses_260659782
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
J__inference_lstm_cell_12_layer_call_and_return_conditional_losses_26068948

inputs
states_0
states_11
matmul_readvariableop_resource:	]?4
 matmul_1_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity

identity_1

identity_2??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?MatMul_1/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	]?*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul_1/ReadVariableOp|
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
add?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
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
_construction_contextkEagerRuntime*T
_input_shapesC
A:?????????]:??????????:??????????: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:?????????]
 
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
?\
?
E__inference_lstm_12_layer_call_and_return_conditional_losses_26067947

inputs>
+lstm_cell_12_matmul_readvariableop_resource:	]?A
-lstm_cell_12_matmul_1_readvariableop_resource:
??;
,lstm_cell_12_biasadd_readvariableop_resource:	?
identity??#lstm_cell_12/BiasAdd/ReadVariableOp?"lstm_cell_12/MatMul/ReadVariableOp?$lstm_cell_12/MatMul_1/ReadVariableOp?whileD
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
"lstm_cell_12/MatMul/ReadVariableOpReadVariableOp+lstm_cell_12_matmul_readvariableop_resource*
_output_shapes
:	]?*
dtype02$
"lstm_cell_12/MatMul/ReadVariableOp?
lstm_cell_12/MatMulMatMulstrided_slice_2:output:0*lstm_cell_12/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_12/MatMul?
$lstm_cell_12/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_12_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype02&
$lstm_cell_12/MatMul_1/ReadVariableOp?
lstm_cell_12/MatMul_1MatMulzeros:output:0,lstm_cell_12/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_12/MatMul_1?
lstm_cell_12/addAddV2lstm_cell_12/MatMul:product:0lstm_cell_12/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell_12/add?
#lstm_cell_12/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_12_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02%
#lstm_cell_12/BiasAdd/ReadVariableOp?
lstm_cell_12/BiasAddBiasAddlstm_cell_12/add:z:0+lstm_cell_12/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_12/BiasAdd~
lstm_cell_12/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_12/split/split_dim?
lstm_cell_12/splitSplit%lstm_cell_12/split/split_dim:output:0lstm_cell_12/BiasAdd:output:0*
T0*d
_output_shapesR
P:??????????:??????????:??????????:??????????*
	num_split2
lstm_cell_12/split?
lstm_cell_12/SigmoidSigmoidlstm_cell_12/split:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_12/Sigmoid?
lstm_cell_12/Sigmoid_1Sigmoidlstm_cell_12/split:output:1*
T0*(
_output_shapes
:??????????2
lstm_cell_12/Sigmoid_1?
lstm_cell_12/mulMullstm_cell_12/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:??????????2
lstm_cell_12/mul~
lstm_cell_12/ReluRelulstm_cell_12/split:output:2*
T0*(
_output_shapes
:??????????2
lstm_cell_12/Relu?
lstm_cell_12/mul_1Mullstm_cell_12/Sigmoid:y:0lstm_cell_12/Relu:activations:0*
T0*(
_output_shapes
:??????????2
lstm_cell_12/mul_1?
lstm_cell_12/add_1AddV2lstm_cell_12/mul:z:0lstm_cell_12/mul_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell_12/add_1?
lstm_cell_12/Sigmoid_2Sigmoidlstm_cell_12/split:output:3*
T0*(
_output_shapes
:??????????2
lstm_cell_12/Sigmoid_2}
lstm_cell_12/Relu_1Relulstm_cell_12/add_1:z:0*
T0*(
_output_shapes
:??????????2
lstm_cell_12/Relu_1?
lstm_cell_12/mul_2Mullstm_cell_12/Sigmoid_2:y:0!lstm_cell_12/Relu_1:activations:0*
T0*(
_output_shapes
:??????????2
lstm_cell_12/mul_2?
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_12_matmul_readvariableop_resource-lstm_cell_12_matmul_1_readvariableop_resource,lstm_cell_12_biasadd_readvariableop_resource*
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
while_body_26067863*
condR
while_cond_26067862*M
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
NoOpNoOp$^lstm_cell_12/BiasAdd/ReadVariableOp#^lstm_cell_12/MatMul/ReadVariableOp%^lstm_cell_12/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????]: : : 2J
#lstm_cell_12/BiasAdd/ReadVariableOp#lstm_cell_12/BiasAdd/ReadVariableOp2H
"lstm_cell_12/MatMul/ReadVariableOp"lstm_cell_12/MatMul/ReadVariableOp2L
$lstm_cell_12/MatMul_1/ReadVariableOp$lstm_cell_12/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:?????????]
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
K
lstm_12_input:
serving_default_lstm_12_input:0?????????]?
dense_64
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

1layers
trainable_variables
regularization_losses
		variables
2layer_metrics
3non_trainable_variables
4metrics
5layer_regularization_losses
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

;layers
trainable_variables
regularization_losses
	variables

<states
=layer_metrics
>non_trainable_variables
?metrics
@layer_regularization_losses
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

Alayers
trainable_variables
regularization_losses
	variables
Blayer_metrics
Cnon_trainable_variables
Dmetrics
Elayer_regularization_losses
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

Klayers
trainable_variables
regularization_losses
	variables

Lstates
Mlayer_metrics
Nnon_trainable_variables
Ometrics
Player_regularization_losses
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

Qlayers
trainable_variables
regularization_losses
	variables
Rlayer_metrics
Snon_trainable_variables
Tmetrics
Ulayer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
!:	?2dense_6/kernel
:2dense_6/bias
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

Vlayers
"trainable_variables
#regularization_losses
$	variables
Wlayer_metrics
Xnon_trainable_variables
Ymetrics
Zlayer_regularization_losses
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
.:,	]?2lstm_12/lstm_cell_12/kernel
9:7
??2%lstm_12/lstm_cell_12/recurrent_kernel
(:&?2lstm_12/lstm_cell_12/bias
/:-
??2lstm_13/lstm_cell_13/kernel
9:7
??2%lstm_13/lstm_cell_13/recurrent_kernel
(:&?2lstm_13/lstm_cell_13/bias
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
?

]layers
7trainable_variables
8regularization_losses
9	variables
^layer_metrics
_non_trainable_variables
`metrics
alayer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
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

blayers
Gtrainable_variables
Hregularization_losses
I	variables
clayer_metrics
dnon_trainable_variables
emetrics
flayer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
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
&:$	?2Adam/dense_6/kernel/m
:2Adam/dense_6/bias/m
3:1	]?2"Adam/lstm_12/lstm_cell_12/kernel/m
>:<
??2,Adam/lstm_12/lstm_cell_12/recurrent_kernel/m
-:+?2 Adam/lstm_12/lstm_cell_12/bias/m
4:2
??2"Adam/lstm_13/lstm_cell_13/kernel/m
>:<
??2,Adam/lstm_13/lstm_cell_13/recurrent_kernel/m
-:+?2 Adam/lstm_13/lstm_cell_13/bias/m
&:$	?2Adam/dense_6/kernel/v
:2Adam/dense_6/bias/v
3:1	]?2"Adam/lstm_12/lstm_cell_12/kernel/v
>:<
??2,Adam/lstm_12/lstm_cell_12/recurrent_kernel/v
-:+?2 Adam/lstm_12/lstm_cell_12/bias/v
4:2
??2"Adam/lstm_13/lstm_cell_13/kernel/v
>:<
??2,Adam/lstm_13/lstm_cell_13/recurrent_kernel/v
-:+?2 Adam/lstm_13/lstm_cell_13/bias/v
?B?
#__inference__wrapped_model_26064561lstm_12_input"?
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
J__inference_sequential_6_layer_call_and_return_conditional_losses_26067111
J__inference_sequential_6_layer_call_and_return_conditional_losses_26067452
J__inference_sequential_6_layer_call_and_return_conditional_losses_26066730
J__inference_sequential_6_layer_call_and_return_conditional_losses_26066755?
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
/__inference_sequential_6_layer_call_fn_26066215
/__inference_sequential_6_layer_call_fn_26067473
/__inference_sequential_6_layer_call_fn_26067494
/__inference_sequential_6_layer_call_fn_26066705?
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
E__inference_lstm_12_layer_call_and_return_conditional_losses_26067645
E__inference_lstm_12_layer_call_and_return_conditional_losses_26067796
E__inference_lstm_12_layer_call_and_return_conditional_losses_26067947
E__inference_lstm_12_layer_call_and_return_conditional_losses_26068098?
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
*__inference_lstm_12_layer_call_fn_26068109
*__inference_lstm_12_layer_call_fn_26068120
*__inference_lstm_12_layer_call_fn_26068131
*__inference_lstm_12_layer_call_fn_26068142?
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
H__inference_dropout_12_layer_call_and_return_conditional_losses_26068147
H__inference_dropout_12_layer_call_and_return_conditional_losses_26068159?
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
-__inference_dropout_12_layer_call_fn_26068164
-__inference_dropout_12_layer_call_fn_26068169?
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
E__inference_lstm_13_layer_call_and_return_conditional_losses_26068320
E__inference_lstm_13_layer_call_and_return_conditional_losses_26068471
E__inference_lstm_13_layer_call_and_return_conditional_losses_26068622
E__inference_lstm_13_layer_call_and_return_conditional_losses_26068773?
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
*__inference_lstm_13_layer_call_fn_26068784
*__inference_lstm_13_layer_call_fn_26068795
*__inference_lstm_13_layer_call_fn_26068806
*__inference_lstm_13_layer_call_fn_26068817?
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
H__inference_dropout_13_layer_call_and_return_conditional_losses_26068822
H__inference_dropout_13_layer_call_and_return_conditional_losses_26068834?
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
-__inference_dropout_13_layer_call_fn_26068839
-__inference_dropout_13_layer_call_fn_26068844?
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
E__inference_dense_6_layer_call_and_return_conditional_losses_26068875?
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
*__inference_dense_6_layer_call_fn_26068884?
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
&__inference_signature_wrapper_26066784lstm_12_input"?
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
J__inference_lstm_cell_12_layer_call_and_return_conditional_losses_26068916
J__inference_lstm_cell_12_layer_call_and_return_conditional_losses_26068948?
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
/__inference_lstm_cell_12_layer_call_fn_26068965
/__inference_lstm_cell_12_layer_call_fn_26068982?
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
J__inference_lstm_cell_13_layer_call_and_return_conditional_losses_26069014
J__inference_lstm_cell_13_layer_call_and_return_conditional_losses_26069046?
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
/__inference_lstm_cell_13_layer_call_fn_26069063
/__inference_lstm_cell_13_layer_call_fn_26069080?
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
#__inference__wrapped_model_26064561}+,-./0 !:?7
0?-
+?(
lstm_12_input?????????]
? "5?2
0
dense_6%?"
dense_6??????????
E__inference_dense_6_layer_call_and_return_conditional_losses_26068875e !4?1
*?'
%?"
inputs??????????
? ")?&
?
0?????????
? ?
*__inference_dense_6_layer_call_fn_26068884X !4?1
*?'
%?"
inputs??????????
? "???????????
H__inference_dropout_12_layer_call_and_return_conditional_losses_26068147f8?5
.?+
%?"
inputs??????????
p 
? "*?'
 ?
0??????????
? ?
H__inference_dropout_12_layer_call_and_return_conditional_losses_26068159f8?5
.?+
%?"
inputs??????????
p
? "*?'
 ?
0??????????
? ?
-__inference_dropout_12_layer_call_fn_26068164Y8?5
.?+
%?"
inputs??????????
p 
? "????????????
-__inference_dropout_12_layer_call_fn_26068169Y8?5
.?+
%?"
inputs??????????
p
? "????????????
H__inference_dropout_13_layer_call_and_return_conditional_losses_26068822f8?5
.?+
%?"
inputs??????????
p 
? "*?'
 ?
0??????????
? ?
H__inference_dropout_13_layer_call_and_return_conditional_losses_26068834f8?5
.?+
%?"
inputs??????????
p
? "*?'
 ?
0??????????
? ?
-__inference_dropout_13_layer_call_fn_26068839Y8?5
.?+
%?"
inputs??????????
p 
? "????????????
-__inference_dropout_13_layer_call_fn_26068844Y8?5
.?+
%?"
inputs??????????
p
? "????????????
E__inference_lstm_12_layer_call_and_return_conditional_losses_26067645?+,-O?L
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
0???????????????????
? ?
E__inference_lstm_12_layer_call_and_return_conditional_losses_26067796?+,-O?L
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
0???????????????????
? ?
E__inference_lstm_12_layer_call_and_return_conditional_losses_26067947r+,-??<
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
0??????????
? ?
E__inference_lstm_12_layer_call_and_return_conditional_losses_26068098r+,-??<
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
0??????????
? ?
*__inference_lstm_12_layer_call_fn_26068109~+,-O?L
E?B
4?1
/?,
inputs/0??????????????????]

 
p 

 
? "&?#????????????????????
*__inference_lstm_12_layer_call_fn_26068120~+,-O?L
E?B
4?1
/?,
inputs/0??????????????????]

 
p

 
? "&?#????????????????????
*__inference_lstm_12_layer_call_fn_26068131e+,-??<
5?2
$?!
inputs?????????]

 
p 

 
? "????????????
*__inference_lstm_12_layer_call_fn_26068142e+,-??<
5?2
$?!
inputs?????????]

 
p

 
? "????????????
E__inference_lstm_13_layer_call_and_return_conditional_losses_26068320?./0P?M
F?C
5?2
0?-
inputs/0???????????????????

 
p 

 
? "3?0
)?&
0???????????????????
? ?
E__inference_lstm_13_layer_call_and_return_conditional_losses_26068471?./0P?M
F?C
5?2
0?-
inputs/0???????????????????

 
p

 
? "3?0
)?&
0???????????????????
? ?
E__inference_lstm_13_layer_call_and_return_conditional_losses_26068622s./0@?=
6?3
%?"
inputs??????????

 
p 

 
? "*?'
 ?
0??????????
? ?
E__inference_lstm_13_layer_call_and_return_conditional_losses_26068773s./0@?=
6?3
%?"
inputs??????????

 
p

 
? "*?'
 ?
0??????????
? ?
*__inference_lstm_13_layer_call_fn_26068784./0P?M
F?C
5?2
0?-
inputs/0???????????????????

 
p 

 
? "&?#????????????????????
*__inference_lstm_13_layer_call_fn_26068795./0P?M
F?C
5?2
0?-
inputs/0???????????????????

 
p

 
? "&?#????????????????????
*__inference_lstm_13_layer_call_fn_26068806f./0@?=
6?3
%?"
inputs??????????

 
p 

 
? "????????????
*__inference_lstm_13_layer_call_fn_26068817f./0@?=
6?3
%?"
inputs??????????

 
p

 
? "????????????
J__inference_lstm_cell_12_layer_call_and_return_conditional_losses_26068916?+,-??
x?u
 ?
inputs?????????]
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
J__inference_lstm_cell_12_layer_call_and_return_conditional_losses_26068948?+,-??
x?u
 ?
inputs?????????]
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
/__inference_lstm_cell_12_layer_call_fn_26068965?+,-??
x?u
 ?
inputs?????????]
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
/__inference_lstm_cell_12_layer_call_fn_26068982?+,-??
x?u
 ?
inputs?????????]
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
1/1???????????
J__inference_lstm_cell_13_layer_call_and_return_conditional_losses_26069014?./0???
y?v
!?
inputs??????????
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
J__inference_lstm_cell_13_layer_call_and_return_conditional_losses_26069046?./0???
y?v
!?
inputs??????????
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
/__inference_lstm_cell_13_layer_call_fn_26069063?./0???
y?v
!?
inputs??????????
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
/__inference_lstm_cell_13_layer_call_fn_26069080?./0???
y?v
!?
inputs??????????
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
J__inference_sequential_6_layer_call_and_return_conditional_losses_26066730y+,-./0 !B??
8?5
+?(
lstm_12_input?????????]
p 

 
? ")?&
?
0?????????
? ?
J__inference_sequential_6_layer_call_and_return_conditional_losses_26066755y+,-./0 !B??
8?5
+?(
lstm_12_input?????????]
p

 
? ")?&
?
0?????????
? ?
J__inference_sequential_6_layer_call_and_return_conditional_losses_26067111r+,-./0 !;?8
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
J__inference_sequential_6_layer_call_and_return_conditional_losses_26067452r+,-./0 !;?8
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
/__inference_sequential_6_layer_call_fn_26066215l+,-./0 !B??
8?5
+?(
lstm_12_input?????????]
p 

 
? "???????????
/__inference_sequential_6_layer_call_fn_26066705l+,-./0 !B??
8?5
+?(
lstm_12_input?????????]
p

 
? "???????????
/__inference_sequential_6_layer_call_fn_26067473e+,-./0 !;?8
1?.
$?!
inputs?????????]
p 

 
? "???????????
/__inference_sequential_6_layer_call_fn_26067494e+,-./0 !;?8
1?.
$?!
inputs?????????]
p

 
? "???????????
&__inference_signature_wrapper_26066784?+,-./0 !K?H
? 
A?>
<
lstm_12_input+?(
lstm_12_input?????????]"5?2
0
dense_6%?"
dense_6?????????